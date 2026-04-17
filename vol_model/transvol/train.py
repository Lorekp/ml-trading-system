#!/usr/bin/env python
"""TransVol V2 training loop — ratio-form QLIKE, loss-balanced MTL, grad accumulation.

Usage:
    PYTHONPATH=. python vol_model/transvol/train.py
    PYTHONPATH=. python vol_model/transvol/train.py --epochs 150 --patience 20
"""

from __future__ import annotations

import argparse
import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vol_model.transvol.config import TransVolConfig
from vol_model.transvol.model import TransVolEncoder, qlike_ratio_loss, huber_jump_loss
from vol_model.transvol.dataset import build_transvol_datasets

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("transvol.train")


def get_cosine_schedule_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(warmup_steps, 1)
        progress = (step - warmup_steps) / max(total_steps - warmup_steps, 1)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def _targets_from_batch(batch, device):
    return {
        "continuous_var": batch["continuous_var"].to(device),
        "jump_var": batch["jump_var"].to(device),
    }


@torch.no_grad()
def evaluate(model: TransVolEncoder, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_q, total_j, n = 0.0, 0.0, 0

    for batch in loader:
        x = batch["features"].to(device)
        t = batch["time_features"].to(device)
        targets = _targets_from_batch(batch, device)

        with torch.amp.autocast("cuda"):
            outputs = model(x, t)
            losses = model.compute_loss(outputs, targets)

        bs = x.shape[0]
        total_q += losses["qlike"].item() * bs
        total_j += losses["huber_jump"].item() * bs
        n += bs

    return {"qlike": total_q / n, "mse_jump": total_j / n}


def train_transvol(config: TransVolConfig) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    # Build datasets
    logger.info("Building datasets...")
    train_ds, val_ds, test_ds, normalizer, _ = build_transvol_datasets(config)
    logger.info("Train: %d, Val: %d, Test: %d", len(train_ds), len(val_ds), len(test_ds))

    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(val_ds, batch_size=config.batch_size * 2,
                            shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=config.batch_size * 2,
                             shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = TransVolEncoder(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("Model: %d parameters (%.2f MB)", n_params, n_params * 4 / 1e6)

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    steps_per_epoch = len(train_loader) // config.grad_accum_steps
    total_steps = steps_per_epoch * config.max_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)
    scaler = torch.amp.GradScaler()

    # Checkpointing
    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_qlike = float("inf")
    best_path = ckpt_dir / "best_model.pt"
    normalizer.save(ckpt_dir / "normalizer.npz")
    no_improve = 0

    logger.info("Training for up to %d epochs (patience=%d, grad_accum=%d, weights=[%.1f,%.1f])...",
                config.max_epochs, config.patience, config.grad_accum_steps,
                config.loss_weight_qlike, config.loss_weight_jump)

    for epoch in range(config.max_epochs):
        model.train()
        epoch_q, epoch_j, n_batches = 0.0, 0.0, 0
        t0 = time.time()

        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            x = batch["features"].to(device)
            t = batch["time_features"].to(device)
            targets = _targets_from_batch(batch, device)

            with torch.amp.autocast("cuda"):
                outputs = model(x, t)
                losses = model.compute_loss(outputs, targets)
                loss = losses["total"] / config.grad_accum_steps

            scaler.scale(loss).backward()

            epoch_q += losses["qlike"].item()
            epoch_j += losses["huber_jump"].item()
            n_batches += 1

            # Gradient accumulation step
            if (step + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        train_time = time.time() - t0
        avg_q = epoch_q / max(n_batches, 1)
        avg_j = epoch_j / max(n_batches, 1)

        # Validate
        val_metrics = evaluate(model, val_loader, device)
        lr_now = scheduler.get_last_lr()[0]

        logger.info(
            "Epoch %3d/%d  train q=%.4f j=%.6f  val q=%.4f j=%.6f  lr=%.1e  %.0fs",
            epoch + 1, config.max_epochs,
            avg_q, avg_j, val_metrics["qlike"], val_metrics["mse_jump"],
            lr_now, train_time,
        )

        # Early stopping on val QLIKE
        if val_metrics["qlike"] < best_val_qlike:
            best_val_qlike = val_metrics["qlike"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch + 1,
                "val_qlike": best_val_qlike,
                "val_mse_jump": val_metrics["mse_jump"],
            }, best_path)
            no_improve = 0
            logger.info("  -> New best (val_qlike=%.4f), saved", best_val_qlike)
        else:
            no_improve += 1
            if no_improve >= config.patience:
                logger.info("  Early stopping after %d epochs", config.patience)
                break

    # Test evaluation
    logger.info("Loading best checkpoint (val_qlike=%.4f)...", best_val_qlike)
    ckpt = torch.load(best_path, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(model, test_loader, device)
    logger.info("Test: qlike=%.4f, mse_jump=%.6f", test_metrics["qlike"], test_metrics["mse_jump"])

    return best_path


def main():
    parser = argparse.ArgumentParser(description="Train TransVol V2")
    parser.add_argument("--epochs", type=int, default=150)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patience", type=int, default=20)
    args = parser.parse_args()

    config = TransVolConfig(
        max_epochs=args.epochs, lr=args.lr,
        batch_size=args.batch_size, patience=args.patience,
    )
    best = train_transvol(config)
    print(f"\nBest checkpoint: {best}")


if __name__ == "__main__":
    main()
