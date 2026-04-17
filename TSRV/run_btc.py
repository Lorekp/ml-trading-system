#!/usr/bin/env python
"""Run volatility signature analysis on BTC tick data.

Computes signature plots (RV vs sampling frequency) for:
1. Time-based bars (1s to 600s)
2. Tick bars (100 to 25,000 ticks)
3. Dollar bars (quote-flow proxy)

Finds the optimal sampling frequency (knee) for each.

Usage:
    cd ml-trading-system
    python TSRV/run_btc.py
    python TSRV/run_btc.py --dates 2023-06-01:2023-06-30
    python TSRV/run_btc.py --n-dates 30
"""

import argparse
import logging
import time
from pathlib import Path

import numpy as np

from TSRV.config import TSRVConfig
from TSRV.data_loader import available_dates
from TSRV.signature_plot import (
    compute_time_signature,
    compute_tick_signature,
    compute_dollar_signature,
    find_knee,
    plot_signature,
)
from TSRV.realized_vol import annualize_rv

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("TSRV")


def main():
    parser = argparse.ArgumentParser(description="BTC Volatility Signature Plot")
    parser.add_argument("--dates", type=str, default=None,
                        help="Date range: YYYY-MM-DD:YYYY-MM-DD")
    parser.add_argument("--n-dates", type=int, default=None,
                        help="Use N evenly-spaced dates (faster)")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-dollar", action="store_true",
                        help="Skip dollar bar signature (slowest)")
    args = parser.parse_args()

    config = TSRVConfig()
    config.plot = not args.no_plot
    out = Path(config.output_dir)

    # Resolve dates
    all_dates = available_dates(config.tick_dir, config.asset)
    if not all_dates:
        logger.error("No tick data found at %s/%s", config.tick_dir, config.asset)
        return

    if args.dates:
        start, end = args.dates.split(":")
        dates = [d for d in all_dates if start <= d <= end]
    elif args.n_dates:
        idx = np.linspace(0, len(all_dates) - 1, args.n_dates, dtype=int)
        dates = [all_dates[i] for i in idx]
    else:
        dates = all_dates

    t0 = time.time()
    print("=" * 80)
    print(f"  VOLATILITY SIGNATURE PLOT — {config.asset}")
    print(f"  {len(dates)} dates: {dates[0]} to {dates[-1]}")
    print(f"  Time intervals: {config.time_intervals}")
    print(f"  Tick thresholds: {config.tick_thresholds}")
    print("=" * 80)

    # ── 1. Time-based signature ──
    print("\n[1/3] Computing time-based signature...")
    time_sig = compute_time_signature(
        config.tick_dir, config.asset, dates, config.time_intervals, config)

    if len(time_sig) < 3:
        logger.error("Not enough data for time signature")
        return

    time_knee = find_knee(
        time_sig["interval"].values, time_sig["rv_mean"].values, config)

    tsrv_ref = time_sig.attrs.get("tsrv_mean")
    noise_ref = time_sig.attrs.get("noise_var_mean")

    print(f"\n  Time-based signature:")
    for _, row in time_sig.iterrows():
        rv_vol = annualize_rv(row["rv_mean"])
        print(f"    {row['interval']:>6.0f}s  RV_ann={rv_vol:>6.1%}  "
              f"(std={annualize_rv(row['rv_std']):>6.1%})")

    print(f"\n  >>> Optimal time interval: {time_knee['knee_x']}s")
    if tsrv_ref is not None and tsrv_ref > 0:
        print(f"  >>> TSRV benchmark (all-tick): {annualize_rv(tsrv_ref):.1%}")
    if noise_ref is not None:
        print(f"  >>> Noise variance estimate: {noise_ref:.2e}")

    if config.plot:
        plot_signature(time_sig, "interval", time_knee,
                       f"{config.asset} Volatility Signature (Time Bars)",
                       out / "sig_time.png",
                       x_label="Sampling interval (seconds)",
                       tsrv_ref=tsrv_ref)

    # ── 2. Tick bar signature ──
    print("\n[2/3] Computing tick bar signature...")
    tick_sig = compute_tick_signature(
        config.tick_dir, config.asset, dates, config.tick_thresholds, config)

    if len(tick_sig) >= 3:
        # For tick bars, RV typically increases with larger bars (fewer bars = noisier)
        # so direction may differ. Auto-detect.
        y = tick_sig["rv_mean"].values
        if y[0] > y[-1]:
            tick_config = config.model_copy()
            tick_config.knee_direction = "decreasing"
        else:
            tick_config = config.model_copy()
            tick_config.knee_direction = "increasing"
            tick_config.knee_curve = "concave"

        tick_knee = find_knee(
            tick_sig["threshold"].values, tick_sig["rv_mean"].values, tick_config)

        print(f"\n  Tick bar signature:")
        for _, row in tick_sig.iterrows():
            rv_vol = annualize_rv(row["rv_mean"])
            print(f"    {row['threshold']:>8,.0f} ticks  RV_ann={rv_vol:>6.1%}")

        print(f"\n  >>> Optimal tick threshold: {tick_knee['knee_x']:,.0f} ticks")

        if config.plot:
            plot_signature(tick_sig, "threshold", tick_knee,
                           f"{config.asset} Volatility Signature (Tick Bars)",
                           out / "sig_tick.png",
                           x_label="Ticks per bar",
                           tsrv_ref=tsrv_ref)
    else:
        print("  Not enough data for tick signature")
        tick_knee = {"knee_x": None}

    # ── 3. Dollar bar signature (real trade volume) ──
    if not args.no_dollar:
        print("\n[3/3] Computing dollar bar signature (aggTrades)...")
        dollar_sig = compute_dollar_signature(
            config.trade_dir, config.asset, config.dollar_thresholds, config,
            max_days=len(dates))

        if len(dollar_sig) >= 3:
            dollar_knee = find_knee(
                dollar_sig["threshold"].values, dollar_sig["rv_mean"].values, config)

            print(f"\n  Dollar bar signature:")
            for _, row in dollar_sig.iterrows():
                rv_vol = annualize_rv(row["rv_mean"])
                print(f"    ${row['threshold']:>14,.0f}  RV_ann={rv_vol:>6.1%}")

            print(f"\n  >>> Optimal dollar threshold: ${dollar_knee['knee_x']:,.0f}")

            if config.plot:
                plot_signature(dollar_sig, "threshold", dollar_knee,
                               f"{config.asset} Volatility Signature (Dollar Bars)",
                               out / "sig_dollar.png",
                               x_label="Dollar flow per bar (USD)",
                               tsrv_ref=tsrv_ref)
        else:
            print("  Not enough data for dollar signature")
            dollar_knee = {"knee_x": None}
    else:
        print("\n[3/3] Skipped dollar bar signature (--no-dollar)")
        dollar_knee = {"knee_x": None}

    # ── Summary ──
    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"  RESULTS SUMMARY — {config.asset}")
    print(f"{'=' * 80}")
    print(f"  Dates analyzed:        {len(dates)}")
    print(f"  Optimal time interval: {time_knee['knee_x']}s")
    print(f"  Optimal tick bars:     {tick_knee.get('knee_x')}")
    print(f"  Optimal dollar bars:   {dollar_knee.get('knee_x')}")
    if tsrv_ref and tsrv_ref > 0:
        print(f"  TSRV (daily avg):      {annualize_rv(tsrv_ref):.1%} ann.")
    print(f"  Runtime:               {elapsed:.0f}s")

    # Save numeric results
    results_path = out / "results.csv"
    time_sig.to_csv(results_path, index=False)
    print(f"\n  Results saved to {out}/")


if __name__ == "__main__":
    main()
