"""Test dynamic asset selection on historical data.

Computes rolling 30-day median dollar volume for all 30 assets,
builds monthly schedule, reports changes and data gaps.
"""

import sys, os
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.asset_selection import (
    compute_monthly_dollar_volume,
    build_monthly_asset_schedule,
    report_asset_schedule,
)

warnings.filterwarnings("ignore")
DATA_DIR = Path("data/binance-1m")

ASSETS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT",
    "ZEC": "ZECUSDT", "BNX": "BNXUSDT", "BNB": "BNBUSDT", "ENJ": "ENJUSDT",
    "TAO": "TAOUSDT", "ADA": "ADAUSDT", "ANKR": "ANKRUSDT", "SUI": "SUIUSDT",
    "AVAX": "AVAXUSDT", "LINK": "LINKUSDT", "NEAR": "NEARUSDT", "FIL": "FILUSDT",
    "DOT": "DOTUSDT", "SXP": "SXPUSDT", "ENA": "ENAUSDT", "BCH": "BCHUSDT",
    "LTC": "LTCUSDT", "UNI": "UNIUSDT", "WLD": "WLDUSDT", "TRX": "TRXUSDT",
    "FTM": "FTMUSDT", "AAVE": "AAVEUSDT", "FET": "FETUSDT", "APT": "APTUSDT",
    "HBAR": "HBARUSDT", "XLM": "XLMUSDT",
}

# Assets we know should be liquid but aren't in our data
# (for the "missing data" report)
KNOWN_LIQUID = {
    "DOGE": "DOGEUSDT", "ARB": "ARBUSDT", "OP": "OPUSDT",
    "MATIC": "MATICUSDT", "PEPE": "PEPEUSDT",
}


def main():
    t0 = time.time()
    print("=" * 80)
    print("DYNAMIC ASSET SELECTION — 30-day median dollar volume")
    print("=" * 80)

    # Load and compute dollar volumes
    asset_volumes = {}
    print("\nComputing dollar volumes...", flush=True)
    for asset, symbol in sorted(ASSETS.items()):
        try:
            ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
            ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
            vol = compute_monthly_dollar_volume(ohlcv_15m)
            asset_volumes[asset] = vol
            latest = vol.dropna().iloc[-1] if len(vol.dropna()) > 0 else 0
            start = vol.dropna().index[0] if len(vol.dropna()) > 0 else "N/A"
            print(f"  {asset:<8s} latest 30d median: ${latest:>15,.0f}  data from {str(start)[:10]}")
        except Exception as e:
            print(f"  {asset:<8s} FAILED: {e}")

    # Build schedule
    print(f"\n{'='*80}")
    print("MONTHLY SCHEDULE (top 20 by 30d median dollar volume)")
    print(f"{'='*80}")

    schedule = build_monthly_asset_schedule(
        asset_volumes,
        start_month="2024-01",
        top_n=20,
        min_history_days=90,
    )

    report_asset_schedule(schedule, ASSETS)

    # Detailed month-by-month table
    print(f"\n{'='*80}")
    print("DETAILED MONTHLY SELECTION")
    print(f"{'='*80}")

    for month, assets in sorted(schedule.items()):
        print(f"\n  {month}: {', '.join(assets)}")

    # Show what percentage of months each asset appears
    print(f"\n{'='*80}")
    print("ASSET INCLUSION FREQUENCY")
    print(f"{'='*80}")
    n_months = len(schedule)
    asset_freq = {}
    for assets in schedule.values():
        for a in assets:
            asset_freq[a] = asset_freq.get(a, 0) + 1

    print(f"  {'Asset':<8s} {'Months':>6s} {'%':>5s} {'Status':>10s}")
    print("  " + "-" * 35)
    for asset, count in sorted(asset_freq.items(), key=lambda x: -x[1]):
        pct = count / n_months * 100
        status = "ALWAYS" if count == n_months else ("MOST" if pct > 80 else ("SOME" if pct > 50 else "RARE"))
        print(f"  {asset:<8s} {count:>6d} {pct:>4.0f}% {status:>10s}")

    # Assets never selected
    never = set(ASSETS.keys()) - set(asset_freq.keys())
    if never:
        print(f"\n  Never selected: {', '.join(sorted(never))}")

    # Missing high-volume assets
    print(f"\n{'='*80}")
    print("MISSING HIGH-VOLUME ASSETS (need to download)")
    print(f"{'='*80}")
    for asset, symbol in sorted(KNOWN_LIQUID.items()):
        print(f"  {asset:<8s} {symbol} — not in current data, likely top-20 by volume")

    print(f"\n  Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
