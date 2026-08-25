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
    patience: int, ckpt_dir: Path, rehab: bool = False,
) -> tuple[float, float]:
    base = TransVolConfig()
    cfg = replace(
        base,
        lr=optuna_best["lr"],
        dropout=optuna_best["dropout"],
        weight_decay=optuna_best["weight_decay"],
        n_features=X_tv.shape[1],
    )

    if rehab:
        # R1+R2+R6 clean build. _make_per_month_datasets returns the extra
        # train-only-fit stats (winsor limits, time-normalizer, log-bias).
        train_ds, val_ds, normalizer, prep = _make_per_month_datasets(
            X_tv, cont, jump, pd.Timestamp(month, tz="UTC"), base.seq_len,
            rehab=True,
        )
        # R2: tell the model to use the log-target head with a calibrated bias.
        cfg = replace(
            cfg, target_log=True, cont_log_bias=prep["cont_log_bias"],
        )
    else:
        train_ds, val_ds, normalizer = _make_per_month_datasets(
            X_tv, cont, jump, pd.Timestamp(month, tz="UTC"), base.seq_len,
        )
        prep = None

    t0 = time.time()
    model, val_q = _train_transvol(
        cfg, train_ds, val_ds, device,
        max_epochs=max_epochs, patience=patience,
    )
    elapsed = time.time() - t0

    # Save checkpoint. Production schema:
    #   { model_state_dict, config, normalizer_mean, normalizer_std,
    #     epoch, val_qlike }. Rehab adds the train-only-fit winsor limits,
    #   the persisted time-normalizer (R6), and the log-target flag so
    #   extraction reproduces the exact same prep with NO re-fit on test data.
    out_path = ckpt_dir / f"{month}_transvol.pt"
    payload = {
        "model_state_dict": model.state_dict(),
        "config": cfg,
        "normalizer_mean": normalizer.mean,
        "normalizer_std": normalizer.std,
        "epoch": -1,  # not tracked separately; max_epochs is in config
        "val_qlike": val_q,
        "optuna_best_params": optuna_best,  # provenance
    }
    if rehab:
        payload.update({
            "rehab": True,
            "rehab_recipe": "R6+R1+R2",
            # R1: train-fit winsor clip bounds (applied BEFORE z-score).
            "winsor_lo": prep["winsor_lo"],
            "winsor_hi": prep["winsor_hi"],
            "winsor_lo_q": prep["winsor_lo_q"],
            "winsor_hi_q": prep["winsor_hi_q"],
            # R6: persisted train-only time-normalizer (closes A2-M1 leak).
            "time_normalizer_mean": prep["time_normalizer_mean"],
            "time_normalizer_std": prep["time_normalizer_std"],
            # R2: log-target head provenance (cfg already carries the flag).
            "target_log": True,
            "cont_log_bias": prep["cont_log_bias"],
        })
    torch.save(payload, str(out_path))
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
    parser.add_argument(
        "--rehab", action="store_true",
        help="Apply the pre-committed R6+R1+R2 clean build (winsorize before "
             "z-score, persisted train-only time-normalizer, log-target). "
             "Implies --out-tag _clean_rehab unless overridden, so production "
             "checkpoints in expanding_hybrid/ are NEVER touched.",
    )
    parser.add_argument(
        "--out-tag", type=str, default="",
        help="Suffix for the checkpoint dir: writes to "
             "expanding_hybrid{out_tag}/ instead of the production dir. "
             "Defaults to '_clean_rehab' when --rehab is set; production "
             "(no tag) is unchanged.",
    )
    args = parser.parse_args()

    # Resolve the checkpoint output dir. Rehab writes to a tagged sibling dir
    # so production stays intact unless the rehab WINS the gate.
    out_tag = args.out_tag
    if args.rehab and not out_tag:
        out_tag = "_clean_rehab"
    ckpt_dir = CKPT_DIR if not out_tag else CKPT_DIR.with_name(CKPT_DIR.name + out_tag)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)
    logger.info("Rehab mode: %s | checkpoint dir: %s", args.rehab, ckpt_dir)
    if device.type == "cuda":
        logger.info(
            "GPU: %s (%.1f GB)", torch.cuda.get_device_name(0),
            torch.cuda.get_device_properties(0).total_memory / 1e9,
        )

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

    # Backup existing checkpoints — ONLY when writing into the production dir.
    # A tagged/rehab dir is fresh, so there is nothing to overwrite or back up.
    if ckpt_dir == CKPT_DIR and not args.no_backup:
        _backup_existing_checkpoints()
    elif ckpt_dir != CKPT_DIR:
        logger.info("Tagged dir %s: production checkpoints untouched, no backup needed",
                    ckpt_dir.name)
    else:
        logger.warning("--no-backup specified; skipping checkpoint backup")

    # Generate month list.
    start = pd.Timestamp(args.start + "-01", tz="UTC")
    end = pd.Timestamp(args.end + "-01", tz="UTC")
    months = pd.date_range(start, end, freq="MS")
    months_str = [m.strftime("%Y-%m") for m in months]
    logger.info("Months to train: %d (%s..%s)",
                len(months_str), months_str[0], months_str[-1])

    ckpt_dir.mkdir(parents=True, exist_ok=True)
    t_global = time.time()
    summaries = []

    for i, month in enumerate(months_str):
        ckpt_path = ckpt_dir / f"{month}_transvol.pt"
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
                ckpt_dir=ckpt_dir, rehab=args.rehab,
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

    # Summary (tagged so the rehab run never clobbers the production summary).
    df = pd.DataFrame(summaries)
    summary_name = f"transvol_optuna_retrain_summary{out_tag}.csv"
    out = REPO_ROOT / "vol_model" / "output" / summary_name
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
