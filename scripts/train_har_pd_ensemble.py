"""Train the K=10 deep-ensemble HAR-PD and freeze per-bar uncertainty.

Historical Phase A artifact generator.

Builds the historical HAR-PD feature matrix via ``vol_model.pipeline.
build_dataset`` and trains K independent LightGBM members under purged
K-fold CV. Per-member OOF predictions form an ``(K, n_bars)`` matrix;
per-bar variance across members is the epistemic-uncertainty estimate.
Aleatoric component computed via the existing ``vol_model.conformal.hqr``
module on the calibration window 2021-04-01 → 2023-09-30.

Output: ``vol_model/output/BTC_v2_har_pd_ensemble_var.parquet`` with
columns ``(sigma_epi, sigma_alea, sigma_total, mean, pi_lo, pi_hi,
y_true)`` indexed by bar timestamp. Frozen — refused-on-overwrite.

Run::

    # Smoke test (~5 min): K=2, n_estimators=100
    PYTHONPATH=. python scripts/train_har_pd_ensemble.py --smoke

    # Full run (~60-90 min): K=10, n_estimators=1000
    PYTHONPATH=. python scripts/train_har_pd_ensemble.py
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vol_model.config import VolModelConfig
from vol_model.pipeline import build_dataset
from vol_model.uncertainty import (
    CAL_WINDOW,
    EnsembleParams,
    combine_with_hqr,
    freeze_uncertainty,
    predict_ensemble_var,
    train_har_pd_ensemble,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_har_pd_ensemble")

DEFAULT_OUT = Path("vol_model/output/BTC_v2_har_pd_ensemble_var.parquet")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing artifact (defeat safety)")
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke mode: K=2, n_estimators=100, 1k bars subset")
    parser.add_argument("--K", type=int, default=10,
                        help="Ensemble size (default 10; ignored in --smoke)")
    parser.add_argument("--n-estimators", type=int, default=1000)
    args = parser.parse_args()

    if args.smoke:
        logger.info("SMOKE MODE: K=2, n_estimators=100, full dataset")
        K = 2
        n_estimators = 100
        out_path = args.out.with_name(args.out.stem + "_smoke" + args.out.suffix)
    else:
        K = args.K
        n_estimators = args.n_estimators
        out_path = args.out

    if out_path.exists() and not args.force:
        raise FileExistsError(
            f"{out_path} already exists. Use --force to overwrite "
            "(only after a NEW plan file authorises regeneration)."
        )

    logger.info("Building HAR-PD feature dataset...")
    t0 = time.time()
    cfg = VolModelConfig(
        asset="BTCUSDT",
        freq_s=900,
        n_estimators=n_estimators,
        early_stopping_rounds=50,
        n_splits=5,
    )
    X, y, t1 = build_dataset(cfg)
    logger.info("Dataset built in %.1fs: %d rows, %d features",
                time.time() - t0, len(X), len(X.columns))
    logger.info("Span: %s -> %s", X.index.min(), X.index.max())

    # Training
    params = EnsembleParams(K=K)
    logger.info("Training %d-member ensemble (n_estimators=%d, n_splits=%d)...",
                K, n_estimators, cfg.n_splits)
    logger.info("Seeds: %s", list(params.seeds))
    t0 = time.time()
    ensemble = train_har_pd_ensemble(
        X, y, t1, cfg,
        params=params,
        max_feature_lookback_bars=cfg.bars_per_24h * 7,
    )
    train_secs = time.time() - t0
    logger.info("Ensemble trained in %.1fs (%.1f min)",
                train_secs, train_secs / 60)

    # Per-bar mean + variance
    var_dict = predict_ensemble_var(ensemble["oof_matrix"])
    logger.info("Ensemble variance: mean=%.6f, p10=%.6f, p50=%.6f, p90=%.6f",
                float(np.nanmean(var_dict["var"])),
                float(np.nanpercentile(var_dict["var"], 10)),
                float(np.nanpercentile(var_dict["var"], 50)),
                float(np.nanpercentile(var_dict["var"], 90)))

    # Combine with HQR for aleatoric + calibrated PI
    logger.info("Fitting HQR aleatoric component (cal_window=%s -> %s)...",
                CAL_WINDOW[0].date(), CAL_WINDOW[1].date())
    combined = combine_with_hqr(
        timestamps=X.index,
        ensemble_mean=var_dict["mean"],
        ensemble_std=var_dict["std"],
        y_true=y.values,
        cal_window=CAL_WINDOW,
    )
    logger.info("HQR Vovk q_conformal = %.4f (cal_n_obs=%d)",
                combined["q_conformal"], combined["cal_n_obs"])

    # Build output dataframe
    out_df = pd.DataFrame({
        "ensemble_mean": var_dict["mean"],
        "sigma_epi": var_dict["std"],
        "sigma_alea": combined["sigma_alea"],
        "sigma_total": combined["sigma_total"],
        "spread": combined["spread"],
        "pi_lo": combined["pi_lo"],
        "pi_hi": combined["pi_hi"],
        "y_true": y.values,
    }, index=X.index)
    out_df.index.name = "timestamp"

    # Empirical coverage check on eval window
    eval_mask = out_df.index >= CAL_WINDOW[1]
    if eval_mask.sum() > 0:
        eval_slice = out_df[eval_mask]
        in_pi = ((eval_slice["y_true"] >= eval_slice["pi_lo"])
                 & (eval_slice["y_true"] <= eval_slice["pi_hi"]))
        cov = float(in_pi.mean())
        logger.info("Eval-window 90%% PI empirical coverage: %.4f "
                    "(n=%d, target [0.85, 0.95])",
                    cov, len(eval_slice))

    # Per-fold metrics summary
    logger.info("Per-member fold-mean QLIKE:")
    for i, fm in enumerate(ensemble["fold_metrics_by_member"]):
        qlikes = [f["qlike"] for f in fm]
        logger.info("  member %d (seed=%d): mean QLIKE=%.4f",
                    i, ensemble["seeds_used"][i], float(np.mean(qlikes)))

    # Freeze
    metadata = {
        "K": K,
        "seeds_used": ensemble["seeds_used"],
        "feature_names": ensemble["feature_names"],
        "n_features": len(ensemble["feature_names"]),
        "n_bars": int(len(out_df)),
        "cal_window_start": CAL_WINDOW[0].isoformat(),
        "cal_window_end": CAL_WINDOW[1].isoformat(),
        "q_conformal": combined["q_conformal"],
        "cal_n_obs": combined["cal_n_obs"],
        "alpha": combined["alpha"],
        "n_estimators_per_member": n_estimators,
        "n_splits_purged_cv": cfg.n_splits,
        "training_time_seconds": train_secs,
    }
    freeze_uncertainty(out_df, out_path, metadata=metadata, force=args.force)

    print()
    print("=" * 70)
    print(" K=%d ensemble + HQR uncertainty frozen" % K)
    print("=" * 70)
    print(f"  Output: {out_path}")
    print(f"  K members: {K}")
    print(f"  Bars: {len(out_df):,}")
    print(f"  q_conformal (Vovk): {combined['q_conformal']:.4f}")
    print(f"  Mean sigma_epi: {float(out_df['sigma_epi'].mean()):.6f}")
    print(f"  Mean sigma_alea: {float(out_df['sigma_alea'].mean()):.6f}")
    print(f"  Mean sigma_total: {float(out_df['sigma_total'].mean()):.6f}")
    print(f"  Training time: {train_secs:.1f}s ({train_secs/60:.1f} min)")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
