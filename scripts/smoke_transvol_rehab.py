"""GPU smoke test for the R6+R1+R2 TransVol rehab — ONE walk-forward month.

Trains a single rehab month end-to-end on CUDA with explicit per-epoch loss
logging, then asserts:
  * device == cuda (prints GPU name + peak memory),
  * no NaN/inf in the loss or the extracted embeddings,
  * val QLIKE decreases from epoch 1 to the best epoch (loss decreases),
  * a checkpoint is produced with all rehab keys persisted,
  * embeddings extract from that checkpoint with the right shape.

Capped at --max-epochs (default 30) so the smoke is fast; this is NOT the
production depth (150). Use it to calibrate the per-month wall-time, then
multiply by 51 for the full-run estimate.

Usage:
    PYTHONPATH=. python scripts/smoke_transvol_rehab.py --month 2024-06 --max-epochs 30
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from copy import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import json
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vol_model.config import VolModelConfig
from vol_model.labels import load_labels
from vol_model.pipeline import build_dataset
from vol_model.transvol.config import TransVolConfig
from vol_model.transvol.model import TransVolEncoder
from scripts.shap_audit_transvol_optuna import (
    _make_per_month_datasets, _select_transvol_features,
)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2024-06")
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    assert device.type == "cuda", "SMOKE REQUIRES CUDA — none available"
    print(f"Device: {device}  ({torch.cuda.get_device_name(0)})")
    torch.cuda.reset_peak_memory_stats()

    best_path = (REPO_ROOT / "vol_model" / "output"
                 / "shap_audit_transvol_optuna_best_config.json")
    optuna_best = json.loads(best_path.read_text())

    cfg = VolModelConfig(asset="BTCUSDT")
    X, _y, _ = build_dataset(cfg, clip=False)
    labels = load_labels(cfg, clip=False)
    common = X.index.intersection(labels.index)
    X = X.loc[common]
    cont = labels.loc[common, "continuous_var"]
    jump = labels.loc[common, "jump_var"]
    X_tv = _select_transvol_features(X)

    base = TransVolConfig()
    t0 = time.time()
    train_ds, val_ds, normalizer, prep = _make_per_month_datasets(
        X_tv, cont, jump, pd.Timestamp(args.month, tz="UTC"), base.seq_len,
        rehab=True,
    )
    print(f"Rehab prep: winsor p{prep['winsor_lo_q']}/p{prep['winsor_hi_q']}, "
          f"cont_log_bias={prep['cont_log_bias']:.4f}, target_log={prep['target_log']}")
    print(f"train seqs={len(train_ds)}  val seqs={len(val_ds)}")

    tcfg = replace(
        base, lr=optuna_best["lr"], dropout=optuna_best["dropout"],
        weight_decay=optuna_best["weight_decay"], n_features=X_tv.shape[1],
        target_log=True, cont_log_bias=prep["cont_log_bias"],
    )

    train_loader = DataLoader(train_ds, batch_size=tcfg.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=tcfg.batch_size * 2, shuffle=False,
                            num_workers=0, pin_memory=True)

    model = TransVolEncoder(tcfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=tcfg.lr,
                            weight_decay=tcfg.weight_decay)
    steps = max(len(train_loader) // tcfg.grad_accum_steps, 1) * args.max_epochs
    warmup = min(tcfg.warmup_steps, max(steps // 10, 1))

    def lr_lambda(s):
        if s < warmup:
            return s / max(warmup, 1)
        p = (s - warmup) / max(steps - warmup, 1)
        return max(0.01, 0.5 * (1.0 + math.cos(math.pi * p)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler()

    val_curve = []
    best_val = float("inf")
    best_state = None
    no_improve = 0
    nan_seen = False

    for epoch in range(args.max_epochs):
        model.train()
        opt.zero_grad(set_to_none=True)
        ep_train = 0.0
        ep_n = 0
        for step, batch in enumerate(train_loader):
            x = batch["features"].to(device)
            t = batch["time_features"].to(device)
            tgt = {"continuous_var": batch["continuous_var"].to(device),
                   "jump_var": batch["jump_var"].to(device)}
            if "continuous_var_raw_for_metric" in batch:
                tgt["continuous_var_raw_for_metric"] = \
                    batch["continuous_var_raw_for_metric"].to(device)
            with torch.amp.autocast("cuda"):
                out = model(x, t)
                losses = model.compute_loss(out, tgt)
                loss = losses["total"] / tcfg.grad_accum_steps
            if not torch.isfinite(loss):
                nan_seen = True
            scaler.scale(loss).backward()
            if (step + 1) % tcfg.grad_accum_steps == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), tcfg.grad_clip)
                scaler.step(opt)
                scaler.update()
                sched.step()
                opt.zero_grad(set_to_none=True)
            ep_train += float(losses["total"]) * x.shape[0]
            ep_n += x.shape[0]

        model.eval()
        vq, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(device)
                t = batch["time_features"].to(device)
                tgt = {"continuous_var": batch["continuous_var"].to(device),
                       "jump_var": batch["jump_var"].to(device)}
                if "continuous_var_raw_for_metric" in batch:
                    tgt["continuous_var_raw_for_metric"] = \
                        batch["continuous_var_raw_for_metric"].to(device)
                with torch.amp.autocast("cuda"):
                    out = model(x, t)
                    losses = model.compute_loss(out, tgt)
                vq += losses["qlike"].item() * x.shape[0]
                vn += x.shape[0]
        val_q = vq / max(vn, 1)
        val_curve.append(val_q)
        if epoch < 3 or (epoch + 1) % 5 == 0:
            print(f"  epoch {epoch+1:3d}/{args.max_epochs}  "
                  f"train_total={ep_train/max(ep_n,1):.5f}  val_qlike={val_q:.5f}")
        if val_q < best_val:
            best_val = val_q
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= args.patience:
                print(f"  early stop @ epoch {epoch+1}")
                break

    model.load_state_dict(best_state)
    elapsed = time.time() - t0
    peak_mem = torch.cuda.max_memory_allocated() / 1e9

    # Persist a rehab checkpoint and extract embeddings.
    ckpt_dir = REPO_ROOT / "vol_model" / "output" / "models" / "expanding_hybrid_clean_rehab"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    out_path = ckpt_dir / f"{args.month}_transvol.pt"
    torch.save({
        "model_state_dict": model.state_dict(), "config": tcfg,
        "normalizer_mean": normalizer.mean, "normalizer_std": normalizer.std,
        "val_qlike": best_val, "rehab": True, "rehab_recipe": "R6+R1+R2",
        "winsor_lo": prep["winsor_lo"], "winsor_hi": prep["winsor_hi"],
        "winsor_lo_q": prep["winsor_lo_q"], "winsor_hi_q": prep["winsor_hi_q"],
        "time_normalizer_mean": prep["time_normalizer_mean"],
        "time_normalizer_std": prep["time_normalizer_std"],
        "target_log": True, "cont_log_bias": prep["cont_log_bias"],
    }, str(out_path))

    from vol_model.run_walkforward_hybrid import _extract_compact_embeddings
    test_end = pd.Timestamp(args.month, tz="UTC") + pd.DateOffset(months=1)
    emb = _extract_compact_embeddings(out_path, X_tv, labels, test_end)
    emb_finite = bool(np.isfinite(emb.values).all())

    print("\n=== SMOKE SUMMARY ===")
    print(f"  device:            {device}  peak_mem={peak_mem:.2f} GB")
    print(f"  epochs run:        {len(val_curve)}  wall={elapsed:.0f}s "
          f"({elapsed/60:.1f} min) @ max_epochs={args.max_epochs}")
    print(f"  val_qlike[0]:      {val_curve[0]:.5f}")
    print(f"  val_qlike[best]:   {best_val:.5f}")
    print(f"  loss DECREASED:    {best_val < val_curve[0]}")
    print(f"  NaN/inf in train:  {nan_seen} (expect False)")
    print(f"  checkpoint:        {out_path.name} ({out_path.stat().st_size} bytes)")
    print(f"  embeddings:        shape={emb.shape}  all_finite={emb_finite}")
    ok = (best_val < val_curve[0]) and (not nan_seen) and emb_finite \
        and out_path.exists()
    print(f"\n=== SMOKE {'PASS' if ok else 'FAIL'} ===")
    # Per-epoch wall to extrapolate the 150-epoch / 51-month production run.
    per_epoch = elapsed / max(len(val_curve), 1)
    print(f"  ~{per_epoch:.1f}s/epoch -> a 150-epoch month ~= "
          f"{per_epoch*150/60:.1f} min; 51 months ~= {per_epoch*150*51/3600:.1f} h "
          f"(upper bound; early-stop @ patience=20 typically halves this)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
