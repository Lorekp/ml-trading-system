"""PatchTST training driver — adapts TransVol's loop for the PatchTST encoder.

The training pipeline is IDENTICAL to TransVol's (same sliding-window dataset,
Time2Vec, AdamW + cosine schedule, grad accumulation + early stopping on val
QLIKE). Only the model class differs. Per-month training is achieved by the
caller setting ``PatchTSTConfig.train_end`` and ``val_end`` before building
datasets.

Usage (from signal_research per-month loop -- not standalone CLI for
production):
    config = PatchTSTConfig(train_end=..., val_end=..., checkpoint_dir=...)
    best_path = train_patchtst(config)
    # Later: load best_path, call extract_embeddings(...)

For a standalone full-data training run, the CLI main() at the bottom trains
with the default PatchTSTConfig.train_end/val_end.
"""

from __future__ import annotations

import logging
import math
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from vol_model.patchtst.config import PatchTSTConfig
from vol_model.patchtst.encoder import PatchTSTEncoder
# Reuse TransVol's sliding-window dataset + normalizer verbatim; they are
# model-agnostic (they just produce (seq_len, n_features) tensors).
from vol_model.transvol.dataset import (
    TransVolDataset as PatchTSTDataset,
    build_transvol_datasets as _build_transvol_datasets,
)

logger = logging.getLogger(__name__)


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
def evaluate(model: PatchTSTEncoder, loader: DataLoader, device: torch.device) -> dict:
    model.eval()
    total_q, total_j, n = 0.0, 0.0, 0
    for batch in loader:
        x = batch["features"].to(device)
        t = batch["time_features"].to(device)
        targets = _targets_from_batch(batch, device)
        use_amp = device.type == "cuda"
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs = model(x, t)
            losses = model.compute_loss(outputs, targets)
        bs = x.shape[0]
        total_q += losses["qlike"].item() * bs
        total_j += losses["huber_jump"].item() * bs
        n += bs
    return {"qlike": total_q / n, "mse_jump": total_j / n}


def _build_patchtst_datasets(config: PatchTSTConfig):
    """Thin wrapper over TransVol's dataset builder.

    TransVol's builder takes a ``config`` with a ``.train_end`` / ``.val_end``
    field; PatchTSTConfig has the same fields, so it works verbatim. Returns
    (train_ds, val_ds, test_ds, normalizer, full_index).
    """
    return _build_transvol_datasets(config)


def train_patchtst(config: PatchTSTConfig) -> Path:
    """Train a PatchTST encoder with early stopping on val QLIKE.

    Returns the path to the best-checkpoint file
    ``<config.checkpoint_dir>/best_model.pt`` (same schema as TransVol so
    downstream extraction code can switch on the config type).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    torch.manual_seed(config.seed)
    np.random.seed(config.seed)

    logger.info("Building datasets (train_end=%s, val_end=%s)...",
                config.train_end, config.val_end)
    train_ds, val_ds, test_ds, normalizer, _ = _build_patchtst_datasets(config)
    logger.info("Train: %d, Val: %d, Test: %d", len(train_ds), len(val_ds), len(test_ds))

    if len(train_ds) < 1000 or len(val_ds) < 100:
        raise RuntimeError(
            f"Insufficient data for PatchTST training: "
            f"train={len(train_ds)}, val={len(val_ds)}"
        )

    # Persistent workers off for Windows stability + short per-month training.
    num_workers = 0 if __import__("platform").system() == "Windows" else 4
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device.type == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size * 2, shuffle=False,
        num_workers=num_workers, pin_memory=(device.type == "cuda"),
    )

    model = PatchTSTEncoder(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info("PatchTST: %d parameters (%.2f MB)", n_params, n_params * 4 / 1e6)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay,
    )
    steps_per_epoch = max(len(train_loader) // config.grad_accum_steps, 1)
    total_steps = steps_per_epoch * config.max_epochs
    scheduler = get_cosine_schedule_with_warmup(optimizer, config.warmup_steps, total_steps)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    ckpt_dir = Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_val_qlike = float("inf")
    best_path = ckpt_dir / "best_model.pt"
    normalizer.save(ckpt_dir / "normalizer.npz")
    no_improve = 0

    logger.info(
        "Training up to %d epochs (patience=%d, grad_accum=%d, weights=[%.1f,%.1f])...",
        config.max_epochs, config.patience, config.grad_accum_steps,
        config.loss_weight_qlike, config.loss_weight_jump,
    )

    for epoch in range(config.max_epochs):
        model.train()
        epoch_q, epoch_j, n_batches = 0.0, 0.0, 0
        t0 = time.time()
        optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(train_loader):
            x = batch["features"].to(device)
            t = batch["time_features"].to(device)
            targets = _targets_from_batch(batch, device)

            use_amp = device.type == "cuda"
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs = model(x, t)
                losses = model.compute_loss(outputs, targets)
                loss = losses["total"] / config.grad_accum_steps

            scaler.scale(loss).backward()
            epoch_q += losses["qlike"].item()
            epoch_j += losses["huber_jump"].item()
            n_batches += 1

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
        val_metrics = evaluate(model, val_loader, device)
        lr_now = scheduler.get_last_lr()[0]

        logger.info(
            "Epoch %3d/%d train q=%.4f j=%.6f val q=%.4f j=%.6f lr=%.1e %.0fs",
            epoch + 1, config.max_epochs,
            avg_q, avg_j, val_metrics["qlike"], val_metrics["mse_jump"],
            lr_now, train_time,
        )

        if val_metrics["qlike"] < best_val_qlike:
            best_val_qlike = val_metrics["qlike"]
            torch.save({
                "model_state_dict": model.state_dict(),
                "config": config,
                "epoch": epoch + 1,
                "val_qlike": best_val_qlike,
                "val_mse_jump": val_metrics["mse_jump"],
                "normalizer_mean": normalizer.mean,
                "normalizer_std": normalizer.std,
            }, best_path)
            no_improve = 0
            logger.info("  -> New best (val_qlike=%.4f), saved", best_val_qlike)
        else:
            no_improve += 1
            if no_improve >= config.patience:
                logger.info("  Early stopping after %d epochs without improvement", config.patience)
                break

    return best_path
