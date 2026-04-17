#!/usr/bin/env python
"""Volatility signature analysis + dollar-bar threshold calibration per asset.

Replaces the BTC-specific ``run_btc.py`` wrapper with an asset-agnostic
entry point. Computes time / tick / dollar signature plots, locates the
knee on each curve, and prints an update-ready snippet for the per-asset
calibration table in ``vol_model/config.py``.

For every new trading symbol (e.g. ETHUSDT, SOLUSDT), run this once on
recent data to pick the dollar-bar threshold before running feature
engineering downstream. The dollar-bar knee is what the pipeline doc
calls out on line 150: "$500,000 per sub-bar (calibrated via signature
plot — knee detection on RV vs threshold curve)".

Usage:
    PYTHONPATH=. python TSRV/run_signature_plot.py --asset BTCUSDT
    PYTHONPATH=. python TSRV/run_signature_plot.py --asset ETHUSDT --dates 2024-01-01:2024-06-30
    PYTHONPATH=. python TSRV/run_signature_plot.py --asset ETHUSDT --n-dates 60 --no-plot
"""

from __future__ import annotations

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
    parser = argparse.ArgumentParser(
        description="Volatility signature plot + dollar-bar knee calibration",
    )
    parser.add_argument("--asset", default="BTCUSDT",
                        help="Binance-perp symbol to calibrate (e.g. BTCUSDT, ETHUSDT)")
    parser.add_argument("--dates", type=str, default=None,
                        help="Date range YYYY-MM-DD:YYYY-MM-DD (default: all available)")
    parser.add_argument("--n-dates", type=int, default=None,
                        help="Use N evenly-spaced dates (faster)")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-dollar", action="store_true",
                        help="Skip dollar bar signature (slowest)")
    args = parser.parse_args()

    config = TSRVConfig(asset=args.asset)
    config.plot = not args.no_plot
    out = Path(f"TSRV/output/signature_{args.asset}")
    out.mkdir(parents=True, exist_ok=True)

    all_dates = available_dates(config.tick_dir, config.asset)
    if not all_dates:
        logger.error("No tick data at %s/%s", config.tick_dir, config.asset)
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
    print(f"  VOLATILITY SIGNATURE + DOLLAR-BAR CALIBRATION — {config.asset}")
    print(f"  {len(dates)} dates: {dates[0]} to {dates[-1]}")
    print("=" * 80)

    # ── 1. Time-based signature (for TSRV noise estimate) ──
    print("\n[1/3] Computing time-based signature...")
    time_sig = compute_time_signature(
        config.tick_dir, config.asset, dates, config.time_intervals, config,
    )
    if len(time_sig) < 3:
        logger.error("Not enough data for time signature")
        return

    time_knee = find_knee(time_sig["interval"].values, time_sig["rv_mean"].values, config)
    tsrv_ref = time_sig.attrs.get("tsrv_mean")
    noise_ref = time_sig.attrs.get("noise_var_mean")

    print(f"\n  >>> Optimal time interval: {time_knee['knee_x']}s")
    if tsrv_ref:
        print(f"  >>> TSRV benchmark: {annualize_rv(tsrv_ref):.1%} ann.")
    if noise_ref:
        print(f"  >>> Noise variance: {noise_ref:.2e}")

    if config.plot:
        plot_signature(time_sig, "interval", time_knee,
                       f"{config.asset} Volatility Signature (Time Bars)",
                       out / "sig_time.png",
                       x_label="Sampling interval (seconds)", tsrv_ref=tsrv_ref)

    # ── 2. Tick bar signature ──
    print("\n[2/3] Computing tick bar signature...")
    tick_sig = compute_tick_signature(
        config.tick_dir, config.asset, dates, config.tick_thresholds, config,
    )

    tick_knee = {"knee_x": None}
    if len(tick_sig) >= 3:
        y = tick_sig["rv_mean"].values
        tick_cfg = config.model_copy()
        if y[0] > y[-1]:
            tick_cfg.knee_direction = "decreasing"
        else:
            tick_cfg.knee_direction = "increasing"
            tick_cfg.knee_curve = "concave"
        tick_knee = find_knee(tick_sig["threshold"].values, tick_sig["rv_mean"].values, tick_cfg)
        print(f"  >>> Optimal tick threshold: {tick_knee['knee_x']:,.0f}")
        if config.plot:
            plot_signature(tick_sig, "threshold", tick_knee,
                           f"{config.asset} Volatility Signature (Tick Bars)",
                           out / "sig_tick.png",
                           x_label="Ticks per bar", tsrv_ref=tsrv_ref)

    # ── 3. Dollar bar signature — THE CALIBRATION TARGET ──
    dollar_knee = {"knee_x": None}
    if not args.no_dollar:
        print("\n[3/3] Computing dollar bar signature (real aggTrades volume)...")
        dollar_sig = compute_dollar_signature(
            config.trade_dir, config.asset, config.dollar_thresholds, config,
            max_days=len(dates),
        )
        if len(dollar_sig) >= 3:
            dollar_knee = find_knee(
                dollar_sig["threshold"].values, dollar_sig["rv_mean"].values, config,
            )
            print(f"\n  Dollar bar signature (USD notional per bar):")
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
            dollar_sig.to_csv(out / "dollar_signature.csv", index=False)
    else:
        print("\n[3/3] Skipped dollar bar signature (--no-dollar)")

    # ── Summary + calibration snippet ──
    elapsed = time.time() - t0
    print(f"\n{'=' * 80}")
    print(f"  CALIBRATION SUMMARY — {config.asset}")
    print(f"{'=' * 80}")
    print(f"  Dates analyzed:        {len(dates)}")
    print(f"  Optimal time interval: {time_knee['knee_x']}s")
    print(f"  Optimal tick bars:     {tick_knee.get('knee_x')}")
    print(f"  Optimal dollar bars:   ${dollar_knee.get('knee_x'):,.0f}" if dollar_knee.get('knee_x') else "  Optimal dollar bars:   <skipped>")
    if tsrv_ref:
        print(f"  TSRV (daily avg):      {annualize_rv(tsrv_ref):.1%} ann.")
    print(f"  Runtime:               {elapsed:.0f}s")
    print(f"  Plots saved to:        {out}/")

    # Emit an update-ready Python snippet for vol_model/config.py
    if dollar_knee.get("knee_x"):
        dbt = int(round(dollar_knee["knee_x"] / 10_000) * 10_000)  # round to 10K
        vpin_bucket = dbt * 2  # match BTC's 2x ratio
        print("\n  Suggested vol_model/config.py _PER_ASSET_CALIBRATION entry:")
        print(f'    "{args.asset}": {{')
        print(f'        "tick_size": <set from exchange spec>,')
        print(f'        "dollar_bar_threshold": {dbt:_},')
        print(f'        "vpin_bucket_size": {vpin_bucket:_},')
        print(f"    }},")

    time_sig.to_csv(out / "time_signature.csv", index=False)
    tick_sig.to_csv(out / "tick_signature.csv", index=False)


if __name__ == "__main__":
    main()
