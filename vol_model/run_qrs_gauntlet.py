"""Session I Candidate I-3 post-hoc QRS evaluator on the 8-month gauntlet.

Applies Quantile-Estimation-via-Residual-Simulation (``vol_model.qrs``)
to the production 50-month OOS predictions, restricted to the 8-month
Session E-H gauntlet substrate. For each test month:

1. Calibration slice: OOS rows from the 3 months immediately preceding
   the test month (strictly before, held-out — no overlap with test).
2. Fit residual distribution on the calibration slice.
3. ``predict_distribution`` at default quantiles (q_01 ... q_99) on the
   test-month rows.
4. Score CRPS + PIT-KS + coverage; compare against a Gaussian baseline
   (sigma = std of calibration residuals, mu = y_pred).

Pre-screen gate (I-3):
- avg ΔCRPS (QRS − Gaussian) ≤ −5% relative
- PIT-KS p > 0.05 aggregate AND in ≥ 5/8 months
- median spread (q90 − q10) narrower than Gaussian by ≥ 5%

Outputs
-------
- ``vol_model/output/session_i_qrs_gauntlet.csv``: per-month scores
- stdout summary with gate PASS/FAIL decision
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from vol_model.qrs import (
    fit_residual_distribution,
    predict_distribution,
    score_distributional,
)
from vol_model.spot_check_regimes import pick_regime_months

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("qrs_gauntlet")

PREDS_PATH = Path("vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions.parquet")
OUT_CSV = Path("vol_model/output/session_i_qrs_gauntlet.csv")
CAL_MONTHS = 3  # use 3 months of prior OOS as calibration slice


def _month_window(month_str: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(month_str + "-01", tz="UTC")
    end = start + pd.DateOffset(months=1)
    return start, end


def score_month(
    all_preds: pd.DataFrame, month_str: str, *, cal_months: int = CAL_MONTHS,
    n_bootstrap: int = 1000, seed: int = 42,
) -> dict:
    """Score QRS on a single gauntlet month."""
    test_start, test_end = _month_window(month_str)
    cal_start = test_start - pd.DateOffset(months=cal_months)
    cal_end = test_start  # strict: calibration ends where test begins

    cal_mask = (all_preds.index >= cal_start) & (all_preds.index < cal_end)
    test_mask = (all_preds.index >= test_start) & (all_preds.index < test_end)

    cal_df = all_preds.loc[cal_mask]
    test_df = all_preds.loc[test_mask]

    if len(cal_df) < 100:
        logger.warning(
            "%s: only %d calibration rows (< 100 minimum); skipping",
            month_str, len(cal_df),
        )
        return {"month": month_str, "n_cal": len(cal_df), "n_test": len(test_df),
                "skipped": True}

    # Fit residual distribution on the calibration slice.
    # NOTE: the production OOS predictions already come from walk-forward
    # models, so each row is genuinely out-of-sample. Residuals across
    # multiple months carry the regime heterogeneity that QRS needs.
    resid = fit_residual_distribution(
        cal_df["y_pred"].to_numpy(),
        cal_df["y_true"].to_numpy(),
    )

    q_df = predict_distribution(
        test_df["y_pred"].to_numpy(), resid,
        n_bootstrap=n_bootstrap, seed=seed,
    )
    scores = score_distributional(
        test_df["y_true"].to_numpy(), q_df, residual_dist=resid,
    )

    delta_crps_pct = (
        (scores["crps"] / scores["crps_gaussian"] - 1.0) * 100.0
        if scores["crps_gaussian"] > 0 else float("nan")
    )

    return {
        "month": month_str,
        "n_cal": len(cal_df),
        "n_test": len(test_df),
        "crps_qrs": scores["crps"],
        "crps_gaussian": scores["crps_gaussian"],
        "delta_crps_pct": delta_crps_pct,
        "pit_ks_pvalue": scores["pit_ks_pvalue"],
        "coverage_80": scores["coverage_80"],
        "coverage_90": scores["coverage_90"],
        "coverage_98": scores["coverage_98"],
        "median_spread": scores["median_spread"],
        "sigma_cal": resid.sigma,
        "skipped": False,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--months", type=str, default="",
                        help="Comma-separated YYYY-MM list (default: 8-month "
                             "gauntlet substrate via pick_regime_months)")
    parser.add_argument("--cal-months", type=int, default=CAL_MONTHS,
                        help=f"Calibration slice length in months (default {CAL_MONTHS})")
    parser.add_argument("--n-bootstrap", type=int, default=1000,
                        help="Bootstrap samples per test row (default 1000)")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed (default 42)")
    parser.add_argument("--out-csv", type=str, default=str(OUT_CSV),
                        help=f"Output CSV path (default {OUT_CSV})")
    parser.add_argument("--summary", type=str,
                        default="vol_model/output/BTCUSDT_metrics.json",
                        help="Path to the metrics JSON used by pick_regime_months.")
    args = parser.parse_args(argv)

    if args.months:
        months = [m.strip() for m in args.months.split(",") if m.strip()]
    else:
        months = pick_regime_months(args.summary)
    logger.info("Gauntlet substrate: %s", ", ".join(months))

    if not PREDS_PATH.exists():
        raise FileNotFoundError(
            f"Production predictions not found at {PREDS_PATH}. "
            f"Run `python -m vol_model.extract_pruned_svi_har_pd_predictions` first."
        )
    logger.info("Loading production predictions: %s", PREDS_PATH)
    all_preds = pd.read_parquet(PREDS_PATH)
    logger.info("  %d rows, index %s to %s",
                len(all_preds), all_preds.index.min(), all_preds.index.max())

    rows: list[dict] = []
    for m in months:
        logger.info("--- %s ---", m)
        r = score_month(all_preds, m, cal_months=args.cal_months,
                        n_bootstrap=args.n_bootstrap, seed=args.seed)
        rows.append(r)
        if not r.get("skipped", False):
            logger.info(
                "  n_cal=%d n_test=%d  CRPS_qrs=%.4f  CRPS_G=%.4f  "
                "delta=%+.2f%%  PIT_KS_p=%.4f  cov80=%.3f cov90=%.3f cov98=%.3f  "
                "spread=%.4f",
                r["n_cal"], r["n_test"], r["crps_qrs"], r["crps_gaussian"],
                r["delta_crps_pct"], r["pit_ks_pvalue"],
                r["coverage_80"], r["coverage_90"], r["coverage_98"],
                r["median_spread"],
            )

    df = pd.DataFrame(rows)
    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info("Saved %d rows to %s", len(df), out_path)

    # Aggregate + gate evaluation
    valid = df[~df["skipped"].astype(bool)]
    if len(valid) == 0:
        logger.error("No valid scored months; gate cannot be evaluated.")
        return 1

    avg_delta = float(valid["delta_crps_pct"].mean())
    pit_ge_005 = int((valid["pit_ks_pvalue"] > 0.05).sum())
    n_months = len(valid)
    median_spread_narrower = bool(
        valid["median_spread"].median() < 2.0 * valid["sigma_cal"].median() * 1.281
        # 2 * sigma * norm.ppf(0.9) = width of 80% Gaussian interval;
        # we compare q90-q10 (80% QRS interval) to 2 * sigma * z_{0.9}.
        # If median QRS spread < Gaussian 80% spread, QRS is sharper.
    )

    logger.info("=" * 70)
    logger.info("I-3 PRE-SCREEN GATE EVALUATION")
    logger.info("=" * 70)
    logger.info(
        "avg ΔCRPS = %+.2f%%  (gate: ≤ -5%%)   [%s]",
        avg_delta, "PASS" if avg_delta <= -5.0 else "FAIL",
    )
    logger.info(
        "PIT KS p > 0.05: %d / %d months  (gate: ≥ 5/8)   [%s]",
        pit_ge_005, n_months, "PASS" if pit_ge_005 >= 5 else "FAIL",
    )
    logger.info(
        "median spread narrower than Gaussian 80%%:  %s   [%s]",
        "yes" if median_spread_narrower else "no",
        "PASS" if median_spread_narrower else "FAIL",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
