"""Test improved asset selection: multi-factor + diversification + data quality."""

import sys, os
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.asset_selection import (
    compute_asset_score,
    build_monthly_asset_schedule,
    report_asset_schedule,
)

warnings.filterwarnings("ignore")
DATA_DIR = Path("data/binance-1m")


def main():
    t0 = time.time()
    print("=" * 80)
    print("IMPROVED ASSET SELECTION TEST")
    print("  Multi-factor: volume (50%) + spread (30%) + stability (20%)")
    print("  Diversification: max correlation 0.85")
    print("  Data quality: ADF stationarity, volume consistency, min history")
    print("  Buffer zone: enter top-35, exit below top-45")
    print("=" * 80)

    # Load all assets
    asset_scores = {}
    asset_ohlcv = {}
    daily_returns = {}

    print("\nLoading assets...", flush=True)
    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir() or not d.name.endswith("USDT"):
            continue
        symbol = d.name
        try:
            ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
            ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
            if len(ohlcv_15m) < 5000:
                continue

            scores = compute_asset_score(ohlcv_15m)
            asset_scores[symbol] = scores
            asset_ohlcv[symbol] = ohlcv_15m

            # Daily returns for correlation
            close = ohlcv_15m["close"]
            dr = close.resample("1D").last().pct_change().dropna()
            daily_returns[symbol] = dr

            vol_latest = scores["volume"].dropna().iloc[-1] if len(scores["volume"].dropna()) > 0 else 0
            spread_latest = scores["spread"].dropna().iloc[-1] if len(scores["spread"].dropna()) > 0 else 0
            stab_latest = scores["stability"].dropna().iloc[-1] if len(scores["stability"].dropna()) > 0 else 0
            print(f"  {symbol:<16s} vol=${vol_latest:>14,.0f}  spread={spread_latest:.5f}  stab={stab_latest:.3f}")
        except Exception as e:
            pass

    print(f"\n  Loaded {len(asset_scores)} assets in {time.time() - t0:.0f}s")

    # Build schedule
    print(f"\n{'='*80}")
    print("MONTHLY SCHEDULE (top 30, with buffer + diversification)")
    print(f"{'='*80}")

    schedule = build_monthly_asset_schedule(
        asset_scores,
        start_month="2024-01",
        top_n=30,
        min_history_days=90,
        asset_ohlcv=asset_ohlcv,
        daily_returns=daily_returns,
        max_corr=0.85,
    )

    report_asset_schedule(schedule)

    # Turnover stats
    print(f"\n{'='*80}")
    print("TURNOVER ANALYSIS")
    print(f"{'='*80}")

    total_added = 0
    total_removed = 0
    for month, info in sorted(schedule.items()):
        added = info.get("added", [])
        removed = info.get("removed", [])
        total_added += len(added)
        total_removed += len(removed)

    n_months = len(schedule)
    print(f"  Total months:      {n_months}")
    print(f"  Total additions:   {total_added} ({total_added / n_months:.1f}/month)")
    print(f"  Total removals:    {total_removed} ({total_removed / n_months:.1f}/month)")

    # Asset frequency
    print(f"\n{'='*80}")
    print("ASSET INCLUSION FREQUENCY")
    print(f"{'='*80}")

    asset_freq = {}
    for info in schedule.values():
        for a in info["assets"]:
            asset_freq[a] = asset_freq.get(a, 0) + 1

    print(f"  {'Asset':<16s} {'Months':>6s} {'%':>5s}")
    print("  " + "-" * 30)
    for asset, count in sorted(asset_freq.items(), key=lambda x: -x[1]):
        pct = count / n_months * 100
        print(f"  {asset:<16s} {count:>6d} {pct:>4.0f}%")

    # Rebalance weights example
    print(f"\n{'='*80}")
    print("REBALANCE WEIGHTS (sample months)")
    print(f"{'='*80}")

    for month, info in sorted(schedule.items()):
        half_weight = [a for a, w in info["weights"].items() if w < 1.0]
        if half_weight:
            print(f"  {month}: half-weight entries: {', '.join(half_weight)}")

    print(f"\n  Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
