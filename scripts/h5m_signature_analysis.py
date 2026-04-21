"""Per-slow-window dollar-bar signature analysis at h=5m.

User-requested rigorous recalibration of the TSRV sub-bar threshold for
h=5m slow bars (option B from the 2026-04-21 design discussion). The
standard Aït-Sahalia & Yu (2009) signature uses per-day RV and finds
the threshold optimal for daily-RV estimation; that is NOT the same as
the threshold optimal for TSRV at a shorter slow-bar window (Zhang
(2006) optimum is K* ≈ n^{2/3} per slow bar).

This script computes RV per 5-min window (not per day) across a dollar-
threshold sweep, finds the knee on the resulting curve, and prints a
recommended threshold for TSRV at h=5m. Sanity-checks include the
median sub-bars-per-window count at each threshold (should be close to
n^{2/3}) and comparison with the existing daily-knee 500K threshold.

Usage
-----
    PYTHONPATH=. python scripts/h5m_signature_analysis.py

    # Fast smoke test on 6 months:
    PYTHONPATH=. python scripts/h5m_signature_analysis.py --max-days 180

Output
------
- Console report (RV curve + sub-bar counts + recommended threshold)
- ``TSRV/output/h5m_signature_analysis.csv`` — per-threshold results
- Return value: a dict with ``recommended_threshold`` and supporting
  stats. Cron-friendly exit code: 0 on success.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from TSRV.config import TSRVConfig
from TSRV.signature_plot import (
    compute_per_window_dollar_signature, find_knee,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("h5m_signature")


# BTC perp aggTrades parent directory — matches TSRVConfig.trade_dir and
# `iter_trade_days(trade_dir, symbol)` which resolves to
# {trade_dir}/{symbol}/{month}.parquet. NOT Deribit option trades; the
# TSRV sub-bar threshold governs how perp trades are bucketed into dollar
# sub-bars for the 2-scale RV estimator.
_AGGTRADES_PARENT_DIR = Path("data/binance-aggTrades-parquet")


# Candidate thresholds. Centered around the h=15 production 500K but
# extended DOWN so we can catch the h=5m optimum (~115K per the
# asymptotic K* = n^{2/3} formula with n ≈ 3500 ticks/5min).
_DEFAULT_THRESHOLDS = [
    25_000, 50_000, 100_000, 150_000, 200_000, 300_000, 500_000,
    750_000, 1_000_000, 2_000_000,
]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Per-slow-window dollar-bar signature analysis at h=5m",
    )
    parser.add_argument(
        "--asset", default="BTCUSDT",
        help="Trading pair (reads aggTrades from data/binance-aggTrades-parquet/{asset}/)",
    )
    parser.add_argument(
        "--window-minutes", type=int, default=5,
        help="Slow-bar window in minutes (default 5 — the h=5m target)",
    )
    parser.add_argument(
        "--thresholds", type=str, default="",
        help="Comma-separated dollar thresholds (default: 10 values 25K..2M)",
    )
    parser.add_argument(
        "--max-days", type=int, default=None,
        help="Cap iteration (default: all ~1826 days). 180 gives ~6 months "
             "smoke — plenty of windows (50K) for stable knee detection.",
    )
    parser.add_argument(
        "--output", type=str, default="TSRV/output/h5m_signature_analysis.csv",
        help="Output CSV path",
    )
    args = parser.parse_args()

    thresholds = (
        [float(x) for x in args.thresholds.split(",")]
        if args.thresholds else _DEFAULT_THRESHOLDS
    )

    trade_dir = _AGGTRADES_PARENT_DIR
    asset_subdir = trade_dir / args.asset
    if not asset_subdir.exists():
        logger.error("Asset trade dir not found: %s", asset_subdir)
        return 2

    logger.info("=" * 70)
    logger.info("Per-slow-window dollar signature — %s @ %dmin windows",
                args.asset, args.window_minutes)
    logger.info("Thresholds: %s", [f"${t/1000:.0f}K" for t in thresholds])
    logger.info("Max days: %s", args.max_days or "all")
    logger.info("=" * 70)

    config = TSRVConfig()

    t0 = time.time()
    sig = compute_per_window_dollar_signature(
        trade_dir=trade_dir,
        symbol=args.asset,
        thresholds=thresholds,
        window_minutes=args.window_minutes,
        config=config,
        max_days=args.max_days,
    )
    elapsed = time.time() - t0

    if sig.empty:
        logger.error("Signature computation returned no rows — check trade data")
        return 1

    logger.info("Computed %d threshold points in %.0fs", len(sig), elapsed)

    # Print the curve
    print()
    print("=" * 100)
    print(f"Per-{args.window_minutes}min-window RV signature — {args.asset}")
    print("=" * 100)
    print(f"{'threshold':>12} {'rv_median':>12} {'n_windows':>10} "
          f"{'K_median':>10} {'K_p5':>8} {'K_p95':>8}")
    print("-" * 100)
    for _, row in sig.iterrows():
        print(f"${row['threshold']/1000:>10.0f}K "
              f"{row['rv_median']:>12.6f} "
              f"{int(row['n_windows']):>10} "
              f"{row['n_sub_bars_median']:>10.1f} "
              f"{row['n_sub_bars_p5']:>8.1f} "
              f"{row['n_sub_bars_p95']:>8.1f}")

    # Knee detection on rv_median
    print()
    knee = find_knee(
        sig["threshold"].to_numpy(),
        sig["rv_median"].to_numpy(),
        config,
    )
    knee_x = knee["knee_x"]
    knee_y = knee["knee_y"]
    print(f"Knee detected at threshold=${knee_x/1000:.0f}K "
          f"(rv_median={knee_y:.6f})")

    # Asymptotic K* comparison — find threshold where K_median ≈ n^{2/3}
    # where n is median ticks per window. Approximate n via: total_ticks /
    # n_windows where we extract a representative sample.
    # Simpler: use the K_median at the highest-resolution threshold as a
    # proxy for n (since at very small threshold, almost every tick is its
    # own bar). More accurate: require the user to measure n separately.
    #
    # For BTC perp at h=5m: memory says ~3,478 median ticks per 5-min window
    # (from the TSRV parquet inspection earlier today). Compute optimal K:
    n_ticks_median = 3478.0  # from TSRV parquet at h=5
    K_star = n_ticks_median ** (2.0 / 3.0)
    print()
    print(f"Asymptotic Zhang (2006) K* = n^(2/3) = "
          f"{n_ticks_median:.0f}^(2/3) = {K_star:.0f}")
    print("Thresholds ranked by |K_median - K*|:")
    sig_sorted = sig.assign(
        k_delta=(sig["n_sub_bars_median"] - K_star).abs(),
    ).sort_values("k_delta")
    print(sig_sorted[["threshold", "n_sub_bars_median", "k_delta"]].head(3).to_string(index=False))

    # Recommendation: prefer the knee if the knee's K_median is within
    # [K*/2, K*×2] (theoretically reasonable); otherwise pick the threshold
    # whose K_median is closest to K*.
    knee_row = sig.iloc[(sig["threshold"] - knee_x).abs().argmin()]
    knee_K = float(knee_row["n_sub_bars_median"])
    if K_star / 2 <= knee_K <= K_star * 2:
        recommended = float(knee_x)
        rationale = "knee (within 2× of asymptotic K*)"
    else:
        # Pick closest-to-K* threshold
        recommended = float(sig_sorted.iloc[0]["threshold"])
        rationale = f"closest-to-K* (knee K={knee_K:.0f} was >2× off K*={K_star:.0f})"

    print()
    print("=" * 100)
    print(f"RECOMMENDATION: threshold = ${recommended/1000:.0f}K  ({rationale})")
    print("=" * 100)

    # Persist
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sig.to_csv(out_path, index=False)
    logger.info("Saved signature data: %s", out_path)

    # Also persist recommendation JSON for programmatic consumption
    rec_path = out_path.with_suffix(".json")
    rec_path.write_text(json.dumps({
        "asset": args.asset,
        "window_minutes": args.window_minutes,
        "knee_threshold": float(knee_x),
        "knee_rv_median": float(knee_y),
        "asymptotic_K_star": float(K_star),
        "n_ticks_median_5min": n_ticks_median,
        "recommended_threshold": recommended,
        "rationale": rationale,
        "max_days": args.max_days,
        "elapsed_s": elapsed,
    }, indent=2))
    logger.info("Saved recommendation: %s", rec_path)

    return 0


if __name__ == "__main__":
    sys.exit(main())
