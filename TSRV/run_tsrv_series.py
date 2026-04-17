#!/usr/bin/env python
"""Compute TSRV time series for any asset and timeframe.

1. Calibrates optimal dollar sub-bar threshold (signature plot)
2. Computes TSRV for every window across the full dataset
3. Saves to parquet + summary plot

Usage:
    PYTHONPATH="." python TSRV/run_tsrv_series.py
    PYTHONPATH="." python TSRV/run_tsrv_series.py --asset ETHUSDT --tf 3600
    PYTHONPATH="." python TSRV/run_tsrv_series.py --asset BTCUSDT --threshold 250000
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from TSRV.config import TSRVConfig
from TSRV.tsrv_series import calibrate_sub_bar_threshold, compute_tsrv_series
from TSRV.signature_plot import plot_signature

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TSRV")


def main():
    parser = argparse.ArgumentParser(description="TSRV Time Series")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--tf", type=int, default=900, help="Slow timeframe in seconds")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Skip calibration, use this dollar sub-bar threshold")
    parser.add_argument("--max-cal-windows", type=int, default=500,
                        help="Max windows for calibration")
    parser.add_argument("--no-plot", action="store_true")
    args = parser.parse_args()

    config = TSRVConfig(asset=args.asset, slow_timeframe_s=args.tf)
    config.plot = not args.no_plot
    out = Path(config.output_dir)
    tf_min = args.tf // 60
    periods_per_year = 525960 / tf_min

    t0 = time.time()
    print("=" * 80)
    print(f"  TSRV TIME SERIES -- {args.asset}, {tf_min}m bars")
    print(f"  Periods/year: {periods_per_year:.0f}")
    print("=" * 80)

    # ── 1. Calibrate threshold ──
    if args.threshold:
        threshold = args.threshold
        print(f"\n[1/2] Using provided threshold: ${threshold:,.0f}")
    else:
        print(f"\n[1/2] Calibrating sub-bar threshold ({args.max_cal_windows} windows)...")
        cal = calibrate_sub_bar_threshold(
            config.trade_dir, args.asset, config.sub_bar_thresholds,
            args.tf, config, max_windows=args.max_cal_windows)

        threshold = cal["threshold"]
        sig_df = cal["signature_df"]
        knee = cal["knee_result"]

        if len(sig_df) > 0:
            print(f"\n  Sub-bar signature (within {tf_min}m windows):")
            for _, row in sig_df.iterrows():
                ann = float(np.sqrt(max(row["rv_mean"] * periods_per_year, 0)))
                print(f"    ${row['threshold']:>12,.0f}  RV_slow_ann={ann:>6.1%}")

        print(f"\n  >>> Optimal sub-bar threshold: ${threshold:,.0f}")

        if config.plot and len(sig_df) >= 3:
            plot_signature(
                sig_df, "threshold", knee,
                f"{args.asset} Sub-bar Signature ({tf_min}m windows)",
                out / f"{args.asset}_subbar_sig_{tf_min}m.png",
                x_label="Dollar sub-bar threshold (USD)")

    # ── 2. Compute full series ──
    out_path = out / f"{args.asset}_tsrv_{tf_min}m.parquet"
    print(f"\n[2/2] Computing TSRV series (threshold=${threshold:,.0f})...")
    series = compute_tsrv_series(
        config.trade_dir, args.asset, threshold, args.tf, config,
        output_path=out_path, save_every=6)

    if len(series) == 0:
        logger.error("No data produced")
        return

    # Stats
    positive_pct = (series["tsrv"] > 0).mean() * 100
    mean_vol = series["vol_ann"].mean()
    median_vol = series["vol_ann"].median()
    mean_sub_bars = series["n_sub_bars"].mean()

    print(f"\n  Series: {len(series):,} windows")
    print(f"  Date range: {series.index[0]} to {series.index[-1]}")
    print(f"  TSRV > 0: {positive_pct:.1f}%")
    print(f"  Mean ann. vol: {mean_vol:.1%}")
    print(f"  Median ann. vol: {median_vol:.1%}")
    print(f"  Mean sub-bars/window: {mean_sub_bars:.1f}")
    print(f"  Mean ticks/window: {series['n_ticks'].mean():,.0f}")

    # Jump decomposition stats
    if "jump_fraction" in series.columns:
        jf = series["jump_fraction"]
        jump_windows = (jf > 0.15).sum()
        print(f"  Jump fraction: mean={jf.mean():.3f}, median={jf.median():.3f}")
        print(f"  Windows with jump_fraction > 15%: {jump_windows} ({100*jump_windows/len(series):.1f}%)")

    # Final save
    series.to_parquet(out_path)
    print(f"\n  Saved: {out_path}")

    # Plot sample
    if config.plot and len(series) > 100:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Vol time series
        ax = axes[0]
        ax.plot(series.index, series["vol_ann"], linewidth=0.5, color="steelblue")
        ax.set_ylabel("Annualized Volatility")
        ax.set_title(f"{args.asset} TSRV ({tf_min}m) -- threshold=${threshold:,.0f}")
        ax.grid(True, alpha=0.3)

        # Sub-bars per window
        ax = axes[1]
        ax.plot(series.index, series["n_sub_bars"], linewidth=0.5, color="coral")
        ax.set_ylabel("Sub-bars per window")
        ax.set_xlabel("Date")
        ax.grid(True, alpha=0.3)

        fig.tight_layout()
        plot_path = out / f"{args.asset}_tsrv_{tf_min}m.png"
        fig.savefig(plot_path, dpi=150)
        plt.close(fig)
        print(f"  Plot: {plot_path}")

    elapsed = time.time() - t0
    print(f"\n  Runtime: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
