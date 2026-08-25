"""Train BASELINE / TREATMENT / PLACEBO shortfall classifiers walk-forward.

Historical Cycle 4 Phase A.1 driver.

Loads the invalidated historical V2 trade log, its historical aligner
predictions, and the walk-forward
σ²_epi parquet (Phase A.0 output), trains three LightGBM classifiers
(identical hyperparameters; only the feature set differs), and saves
the pooled-OOS predictions to a frozen parquet for Phase B gate
evaluation.

Pre-committed paths (locked):
- Trade log: ``vol_arb/output/BTC_v2_trade_log.parquet``
- Predictions: ``vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet``
- Walk-forward σ²_epi: ``vol_model/output/BTC_v2_har_pd_ensemble_var_walkforward.parquet``
  (PRIMARY input — must come from Phase A.0; the K-fold parquet at
  ``BTC_v2_har_pd_ensemble_var.parquet`` MUST NOT be used.)

Output: ``vol_arb/output/BTC_v2_shortfall_oos_predictions.parquet`` with
columns ``(p_baseline, p_treatment, p_placebo, y_short)`` indexed by
fired-trade timestamps. Refused-on-overwrite.

Run::

    PYTHONPATH=. python scripts/run_shortfall_classifier.py
"""
from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.shortfall_head import (
    SHORTFALL_FEATURES_TREATMENT,
    SHORTFALL_TEST_YEARS,
    SHORTFALL_THRESHOLD_USD,
    build_shortfall_features,
    compute_shortfall_target,
    per_year_auc_summary,
    train_walkforward_shortfall_classifier,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_shortfall_classifier")


# ─── Pre-committed paths ────────────────────────────────────────────────

DEFAULT_TRADE_LOG = Path("vol_arb/output/BTC_v2_trade_log.parquet")
DEFAULT_PREDICTIONS = Path(
    "vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet"
)
DEFAULT_WALKFORWARD_UNCERTAINTY = Path(
    "vol_model/output/BTC_v2_har_pd_ensemble_var_walkforward.parquet"
)
DEFAULT_OUT = Path("vol_arb/output/BTC_v2_shortfall_oos_predictions.parquet")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cycle 4 Phase A.1: train BASELINE / TREATMENT / PLACEBO "
                    "shortfall classifiers walk-forward; freeze pooled-OOS "
                    "predictions for Phase B gate evaluation.",
    )
    parser.add_argument("--trade-log", type=Path, default=DEFAULT_TRADE_LOG)
    parser.add_argument("--predictions", type=Path, default=DEFAULT_PREDICTIONS)
    parser.add_argument("--walkforward-uncertainty", type=Path,
                        default=DEFAULT_WALKFORWARD_UNCERTAINTY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing OOS predictions parquet")
    parser.add_argument("--threshold-usd", type=float,
                        default=SHORTFALL_THRESHOLD_USD,
                        help=f"Shortfall threshold in USD (default {SHORTFALL_THRESHOLD_USD})")
    parser.add_argument("--horizon-minutes", type=int, default=15,
                        help="Bar horizon in minutes (default 15 for the invalidated historical V2 reference)")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        raise FileExistsError(
            f"{args.out} already exists. Use --force to overwrite "
            "(only after a NEW plan file authorises regeneration)."
        )

    # Critical guard: refuse to run on the Cycle 3 K-fold parquet
    if "walkforward" not in str(args.walkforward_uncertainty):
        raise ValueError(
            f"walkforward_uncertainty path {args.walkforward_uncertainty} "
            "must contain 'walkforward' in the filename. Cycle 4 REQUIRES "
            "the walk-forward σ²_epi parquet (Phase A.0 output); the "
            "Cycle 3 K-fold parquet is leakage-suspect for this gate."
        )

    # Load inputs
    logger.info("Loading trade log: %s", args.trade_log)
    trade_log = pd.read_parquet(args.trade_log)
    logger.info(
        "Trade log: %d rows, span %s -> %s",
        len(trade_log), trade_log.index.min(), trade_log.index.max(),
    )

    logger.info("Loading predictions: %s", args.predictions)
    predictions = pd.read_parquet(args.predictions)
    logger.info(
        "Predictions: %d rows, columns=%s",
        len(predictions), list(predictions.columns),
    )

    logger.info("Loading walk-forward σ²_epi: %s", args.walkforward_uncertainty)
    wfu = pd.read_parquet(args.walkforward_uncertainty)
    if "sigma_epi" not in wfu.columns:
        raise KeyError(
            f"walk-forward parquet missing 'sigma_epi' column; got "
            f"{list(wfu.columns)}"
        )
    logger.info(
        "Walk-forward σ²_epi: %d rows, %d non-NaN",
        len(wfu), int(wfu["sigma_epi"].notna().sum()),
    )

    # Build features + target
    feats = build_shortfall_features(
        trade_log, predictions, wfu, horizon_minutes=args.horizon_minutes,
    )
    target = compute_shortfall_target(
        trade_log, threshold_usd=args.threshold_usd,
    )
    n_pos = int(target.sum())
    logger.info(
        "Shortfall target (X=$%.0f): %d fired trades, %d positives (%.2f%%)",
        args.threshold_usd, len(target), n_pos,
        100.0 * n_pos / max(len(target), 1),
    )
    logger.info(
        "Per-year fired-trade counts: %s",
        target.groupby(target.index.year).size().to_dict(),
    )
    logger.info(
        "Per-year positive-label counts: %s",
        target.groupby(target.index.year).sum().to_dict(),
    )

    # Train all three classifiers
    classifiers: dict[str, pd.Series] = {}
    for tag in ("baseline", "treatment", "placebo"):
        logger.info("=" * 70)
        logger.info("Training feature_set=%r", tag)
        logger.info("=" * 70)
        t0 = time.time()
        scores = train_walkforward_shortfall_classifier(
            feats, target, feature_set=tag,
            test_years=SHORTFALL_TEST_YEARS,
            min_train_trades=500,
            # FAIRNESS PARITY: enforce that BASELINE / TREATMENT / PLACEBO
            # see EXACTLY the same training rows. Without this, BASELINE
            # would train on year 2022 (where DFL features are non-NaN
            # but σ²_epi is NaN by Phase A.0 walk-forward design),
            # while TREATMENT/PLACEBO drop year 2022 → unfair sample-size
            # confound. Passing the 11-feature treatment column list
            # forces all three classifiers to require σ²_epi non-NaN.
            fairness_columns=SHORTFALL_FEATURES_TREATMENT,
        )
        logger.info(
            "%s done in %.1fs", tag, time.time() - t0,
        )
        # Per-year + pooled AUC
        auc_summary = per_year_auc_summary(target, scores,
                                            test_years=SHORTFALL_TEST_YEARS)
        for k, v in auc_summary.items():
            logger.info("  %s: AUC %s = %.4f", tag, k, v)
        classifiers[tag] = scores

    # Save pooled-OOS predictions
    out_df = pd.DataFrame({
        "p_baseline": classifiers["baseline"],
        "p_treatment": classifiers["treatment"],
        "p_placebo": classifiers["placebo"],
        "y_short": target.astype(int),
    })
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(args.out)
    logger.info("Wrote %d rows to %s", len(out_df), args.out)

    print()
    print("=" * 70)
    print(" Cycle 4 Phase A.1: pooled-OOS predictions frozen")
    print("=" * 70)
    print(f"  Output: {args.out}")
    print(f"  Threshold X = ${args.threshold_usd:.0f}")
    print(f"  Test years: {SHORTFALL_TEST_YEARS}")
    print(f"  Pooled fired trades: {len(out_df):,}")
    print(f"  Positives: {n_pos} ({100.0 * n_pos / max(len(target), 1):.2f}%)")
    print()
    print("  Pooled-OOS AUC per classifier:")
    for tag in ("baseline", "treatment", "placebo"):
        out = per_year_auc_summary(target, classifiers[tag],
                                    test_years=SHORTFALL_TEST_YEARS)
        print(f"    {tag:9s}: pooled = {out['pooled']:.4f}, "
              f"per-year = {[(yr, round(out[f'year_{yr}'], 4)) for yr in SHORTFALL_TEST_YEARS]}")
    print("=" * 70)
    print(" Next: PYTHONPATH=. python scripts/evaluate_shortfall_auc_gate.py")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
