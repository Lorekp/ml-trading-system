"""Apply walk-forward prediction aligner to production 50-month OOS preds.

Reads ``vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions.parquet``,
fits per-month calibration on trailing OOS residuals, applies correction,
writes corrected predictions + per-month diagnostic CSV.

Usage
-----
::

    # Scalar bias (QLIKE-optimal mean+variance shift) with 3-month
    # calibration window — the "default" recommendation.
    python -m vol_model.run_walkforward_aligner

    # Mincer-Zarnowitz 2-param QLIKE-optimal.
    python -m vol_model.run_walkforward_aligner --method mz

    # Longer calibration window (more stable, less regime-adaptive).
    python -m vol_model.run_walkforward_aligner --cal-months 6

Outputs
-------
- ``BTCUSDT_pruned_svi_har_pd_predictions_aligned_{method}.parquet``:
  predictions with the corrected ``y_pred`` (original preserved in
  ``y_pred_raw``).
- ``walkforward_aligner_summary_{method}.csv``: per-month diagnostics
  (α, β, n_cal, n_test, QLIKE before/after, improvement %).
- stdout summary with aggregate QLIKE improvement + gate decision.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from vol_model.model import qlike_loss
from vol_model.prediction_aligner import (
    apply_calibrations,
    fit_monthly_calibrations,
    summarize_calibrations,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("walkforward_aligner")

PREDS_PATH = Path("vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions.parquet")
OUTPUT_DIR = Path("vol_model/output")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--method", choices=["bias", "mz"], default="bias",
                        help="Calibration method (default: bias)")
    parser.add_argument("--cal-months", type=int, default=3,
                        help="Trailing calibration window in months (default 3)")
    parser.add_argument("--min-rows", type=int, default=500,
                        help="Minimum valid cal rows; below → pass-through")
    parser.add_argument("--preds-path", type=str, default=str(PREDS_PATH),
                        help="Input predictions parquet path")
    parser.add_argument("--output-dir", type=str, default=str(OUTPUT_DIR),
                        help="Directory for output artifacts")
    args = parser.parse_args(argv)

    preds_path = Path(args.preds_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions parquet not found: {preds_path}")

    logger.info("Reading predictions: %s", preds_path)
    preds = pd.read_parquet(preds_path)
    logger.info("  %d rows, columns=%s, index %s → %s",
                len(preds), list(preds.columns),
                preds.index.min(), preds.index.max())

    if "y_pred" not in preds.columns or "y_true" not in preds.columns:
        raise KeyError(
            f"Predictions must have y_pred + y_true; got {list(preds.columns)}"
        )

    # Fit per-month calibrations on trailing OOS residuals.
    logger.info(
        "Fitting per-month %s calibration (cal_months=%d, min_rows=%d) ...",
        args.method, args.cal_months, args.min_rows,
    )
    cals = fit_monthly_calibrations(
        preds,
        cal_months=args.cal_months,
        method=args.method,
        min_rows=args.min_rows,
    )
    logger.info("  Fit %d monthly calibrations.", len(cals))

    # Apply
    corrected = apply_calibrations(preds, cals)

    # Summary
    summary = summarize_calibrations(cals, corrected)

    summary_path = output_dir / f"walkforward_aligner_summary_{args.method}.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Saved per-month summary: %s", summary_path)

    # Aggregate metrics (on months that had enough cal data)
    calibrated = summary.loc[summary["method"] != "none"].copy()
    if len(calibrated) == 0:
        logger.error("No months had enough calibration data; aborting.")
        return 1

    avg_improvement = float(calibrated["qlike_improvement_pct"].mean())
    n_improved = int((calibrated["qlike_improvement_pct"] < 0).sum())
    n_worsened = int((calibrated["qlike_improvement_pct"] > 0).sum())
    n_total = len(calibrated)

    # Aggregate QLIKE across ALL rows that fall in calibrated months
    # (row-weighted, not month-weighted — more honest for PnL impact).
    calibrated_months = set(calibrated["month"])
    row_months = preds.index.strftime("%Y-%m").to_numpy()
    aligned_mask = np.isin(row_months, list(calibrated_months))
    y_true = preds.loc[aligned_mask, "y_true"].to_numpy()
    y_pred_raw = preds.loc[aligned_mask, "y_pred"].to_numpy()
    y_pred_cal = corrected.loc[aligned_mask, "y_pred_calibrated"].to_numpy()
    mask_finite = np.isfinite(y_true) & np.isfinite(y_pred_raw) & np.isfinite(y_pred_cal)

    qlike_raw = float(qlike_loss(y_true[mask_finite], y_pred_raw[mask_finite]))
    qlike_cal = float(qlike_loss(y_true[mask_finite], y_pred_cal[mask_finite]))
    qlike_rel_delta_pct = (qlike_cal / qlike_raw - 1.0) * 100.0

    # Distribution of beta / alpha for interpretability
    beta_stats = {
        "mean": float(calibrated["beta"].mean()),
        "median": float(calibrated["beta"].median()),
        "p10": float(calibrated["beta"].quantile(0.10)),
        "p90": float(calibrated["beta"].quantile(0.90)),
    }
    alpha_stats = {
        "mean": float(calibrated["alpha"].mean()),
        "median": float(calibrated["alpha"].median()),
        "p10": float(calibrated["alpha"].quantile(0.10)),
        "p90": float(calibrated["alpha"].quantile(0.90)),
    }

    # Write corrected predictions. Rename columns so downstream consumers
    # can adopt with predictions_path swap (keeps y_true, makes y_pred
    # the corrected forecast, preserves original as y_pred_raw).
    out = corrected.rename(
        columns={"y_pred": "y_pred_raw", "y_pred_calibrated": "y_pred"}
    )[["y_pred", "y_pred_raw", "y_true"]]
    out_path = output_dir / f"BTCUSDT_pruned_svi_har_pd_predictions_aligned_{args.method}.parquet"
    out.to_parquet(out_path)
    logger.info("Saved corrected predictions: %s (%d rows)", out_path, len(out))

    # Report
    logger.info("=" * 70)
    logger.info("WALK-FORWARD ALIGNER — REPORT")
    logger.info("=" * 70)
    logger.info("Method: %s,  cal_months: %d,  min_rows: %d",
                args.method, args.cal_months, args.min_rows)
    logger.info("Calibrated months: %d / %d (%d pass-through)",
                n_total, len(cals), len(cals) - n_total)
    logger.info("Per-month QLIKE improvement:")
    logger.info("  avg:             %+.3f%%", avg_improvement)
    logger.info("  improved:        %d / %d", n_improved, n_total)
    logger.info("  worsened:        %d / %d", n_worsened, n_total)
    logger.info("  unchanged:       %d / %d", n_total - n_improved - n_worsened, n_total)
    logger.info("Row-weighted aggregate QLIKE (calibrated months only):")
    logger.info("  raw:             %.4f", qlike_raw)
    logger.info("  calibrated:      %.4f", qlike_cal)
    logger.info("  rel. delta:      %+.3f%%", qlike_rel_delta_pct)
    logger.info("β distribution:  mean=%+.4f  median=%+.4f  p10=%+.4f  p90=%+.4f",
                beta_stats["mean"], beta_stats["median"],
                beta_stats["p10"], beta_stats["p90"])
    if args.method == "mz":
        logger.info("α distribution:  mean=%.4f  median=%.4f  p10=%.4f  p90=%.4f",
                    alpha_stats["mean"], alpha_stats["median"],
                    alpha_stats["p10"], alpha_stats["p90"])
    logger.info("")
    logger.info("Interpretation:")
    if beta_stats["median"] < -0.01:
        logger.info("  β̃ < 0 → the raw model SYSTEMATICALLY OVER-predicts log-vol.")
    elif beta_stats["median"] > 0.01:
        logger.info("  β̃ > 0 → the raw model SYSTEMATICALLY UNDER-predicts log-vol.")
    else:
        logger.info("  |β̃| ≈ 0 → raw model is well-centered; little to correct.")
    if qlike_rel_delta_pct < -1.0:
        logger.info("  QLIKE improved by >1%% in this historical diagnostic.")
    elif qlike_rel_delta_pct < -0.1:
        logger.info("  QLIKE improved marginally in this historical diagnostic.")
    else:
        logger.info("  QLIKE did not meaningfully improve — aligner unhelpful here.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
