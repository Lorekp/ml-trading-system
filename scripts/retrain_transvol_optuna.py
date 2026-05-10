"""Walk-forward TransVol retrain at Optuna-best hyperparameters.

Cycle-10 follow-up to the TransVol audit (2026-05-09). The audit
showed Optuna-discovered hyperparameters (lr=7.5e-5, dropout=0.198,
weight_decay=0.0012) win in 7/8 audit months vs production defaults
(lr=3e-5, dropout=0.15, weight_decay=0.01) with -1.97% avg QLIKE
improvement, regime-stable (NOT overfit to the 2024-08 sweep month).

This script retrains all 51 monthly TransVol checkpoints
(2022-02 through 2026-04) at the Optuna-best params using full
production training depth (max_epochs=150, patience=20). Checkpoints
written to ``vol_model/output/models/expanding_hybrid/{YYYY-MM}_transvol.pt``,
overwriting existing checkpoints.

Pre-existing checkpoints are backed up to
``backups/2026-05-09_pre_transvol_optuna_retrain/expanding_hybrid/``
before any overwrite.

Usage::

    PYTHONPATH=. python scripts/retrain_transvol_optuna.py
    PYTHONPATH=. python scripts/retrain_transvol_optuna.py --start 2025-01 --end 2026-04
    PYTHONPATH=. python scripts/retrain_transvol_optuna.py --force-retrain
"""
from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
import time
from copy import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402
import torch  # noqa: E402

from vol_model.config import VolModelConfig  # noqa: E402
from vol_model.labels import load_labels  # noqa: E402
from vol_model.pipeline import build_dataset  # noqa: E402
from vol_model.transvol.config import TransVolConfig  # noqa: E402

from scripts.shap_audit_transvol_optuna import (  # noqa: E402
    _make_per_month_datasets,
    _select_transvol_features,
    _train_transvol,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("transvol_retrain")


CKPT_DIR = REPO_ROOT / "vol_model" / "output" / "models" / "expanding_hybrid"
BACKUP_DIR = (
    REPO_ROOT / "backups" / "2026-05-09_pre_transvol_optuna_retrain"
    / "expanding_hybrid"
)


def _backup_existing_checkpoints():
    """Copy all existing _transvol.pt checkpoints to BACKUP_DIR."""
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    existing = sorted(CKPT_DIR.glob("*_transvol.pt"))
    n_skipped = 0
    n_copied = 0
    for src in existing:
        dst = BACKUP_DIR / src.name
        if dst.exists():
            n_skipped += 1
            continue
        shutil.copy2(src, dst)
        n_copied += 1
    logger.info(
        "Backup: %d copied, %d already existed in %s",
        n_copied, n_skipped, BACKUP_DIR,
    )


def _train_one_month(
    X_tv: pd.DataFrame, cont: pd.Series, jump: pd.Series, month: str,
    optuna_best: dict, device: torch.device, max_epochs: int,
    patience: int,
) -> tuple[float, float]:
    base = TransVolConfig()
    train_ds, val_ds, normalizer = _make_per_month_datasets(
        X_tv, cont, jump, pd.Timestamp(month, tz="UTC"), base.seq_len,
    )
    cfg = replace(
        base,
        lr=optuna_best["lr"],
        dropout=optuna_best["dropout"],
        weight_decay=optuna_best["weight_decay"],
        n_features=X_tv.shape[1],
    )
    t0 = time.time()
    model, val_q = _train_transvol(
        cfg, train_ds, val_ds, device,
        max_epochs=max_epochs, patience=patience,
    )
    elapsed = time.time() - t0

    # Save checkpoint in production format. Match the existing schema:
    #   { model_state_dict, config, normalizer_mean, normalizer_std,
    #     epoch, val_qlike }
    out_path = CKPT_DIR / f"{month}_transvol.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "normalizer_mean": normalizer.mean,
        "normalizer_std": normalizer.std,
        "epoch": -1,  # not tracked separately; max_epochs is in config
        "val_qlike": val_q,
        "optuna_best_params": optuna_best,  # provenance
    }, str(out_path))
    return val_q, elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description="Walk-forward TransVol retrain")
    parser.add_argument("--start", default="2022-02",
                        help="First month (YYYY-MM)")
    parser.add_argument("--end", default="2026-04",
                        help="Last month (YYYY-MM, inclusive)")
    parser.add_argument(
        "--max-epochs", type=int, default=150,
        help="Max epochs per month (production default: 150)",
    )
    parser.add_argument(
        "--patience", type=int, default=20,
        help="Early stopping patience (production default: 20)",
    )
    parser.add_argument(
        "--force-retrain", action="store_true",
        help="Overwrite existing per-month checkpoints",
    )
    parser.add_argument(
        "--no-backup", action="store_true",
        help="Skip backing up existing checkpoints (use only after confirmation)",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    # Load Optuna-best params.
    best_path = (
        REPO_ROOT / "vol_model" / "output"
        / "shap_audit_transvol_optuna_best_config.json"
    )
    if not best_path.exists():
        raise FileNotFoundError(
            "Run scripts/shap_audit_transvol_optuna.py first."
        )
    optuna_best = json.loads(best_path.read_text())
    logger.info("Optuna-best params: %s", optuna_best)

    # Build dataset once.
    config = VolModelConfig(asset="BTCUSDT")
    logger.info("Building dataset (clip=False)...")
    X, _y, _ = build_dataset(config, clip=False)
    labels = load_labels(config, clip=False)
    common = X.index.intersection(labels.index)
    X = X.loc[common]
    cont = labels.loc[common, "continuous_var"]
    jump = labels.loc[common, "jump_var"]
    X_tv = _select_transvol_features(X)
    logger.info("TransVol input: %d rows, %d features", len(X_tv), X_tv.shape[1])

    # Backup existing checkpoints.
    if not args.no_backup:
        _backup_existing_checkpoints()
    else:
        logger.warning("--no-backup specified; skipping checkpoint backup")

    # Generate month list.
    start = pd.Timestamp(args.start + "-01", tz="UTC")
    end = pd.Timestamp(args.end + "-01", tz="UTC")
    months = pd.date_range(start, end, freq="MS")
    months_str = [m.strftime("%Y-%m") for m in months]
    logger.info("Months to train: %d (%s..%s)",
                len(months_str), months_str[0], months_str[-1])

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    t_global = time.time()
    summaries = []

    for i, month in enumerate(months_str):
        ckpt_path = CKPT_DIR / f"{month}_transvol.pt"
        if ckpt_path.exists() and not args.force_retrain:
            logger.info("[%d/%d] %s: cached, skipping", i + 1, len(months_str), month)
            continue
        if ckpt_path.exists() and args.force_retrain:
            logger.warning(
                "[%d/%d] %s: --force-retrain in effect; overwriting %s",
                i + 1, len(months_str), month, ckpt_path.name,
            )

        try:
            val_q, elapsed = _train_one_month(
                X_tv, cont, jump, month, optuna_best, device,
                max_epochs=args.max_epochs, patience=args.patience,
            )
        except (ValueError, RuntimeError) as e:
            logger.error("[%d/%d] %s SKIPPED: %s", i + 1, len(months_str), month, e)
            continue

        summaries.append({"month": month, "val_qlike": val_q, "elapsed_s": elapsed})
        elapsed_global = time.time() - t_global
        logger.info(
            "[%d/%d] %s: val_qlike=%.4f  %.0fs  (cumulative %.1fh)",
            i + 1, len(months_str), month, val_q, elapsed,
            elapsed_global / 3600,
        )

    # Summary.
    df = pd.DataFrame(summaries)
    out = REPO_ROOT / "vol_model" / "output" / "transvol_optuna_retrain_summary.csv"
    df.to_csv(out, index=False)
    total_h = (time.time() - t_global) / 3600
    logger.info("Wrote %s", out)
    logger.info("Total: %d months in %.1f hours", len(summaries), total_h)
    if summaries:
        avg_qlike = df["val_qlike"].mean()
        logger.info("Avg val_qlike: %.4f", avg_qlike)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
