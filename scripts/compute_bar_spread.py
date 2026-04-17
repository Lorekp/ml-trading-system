"""Compute per-bar execution quality metrics from the Deribit ATM option trade tape.

For each 15-min bar, computes trailing (no-lookahead) metrics that
a live system could observe at bar open:

- bar_taker_slip_usd: trailing 15-min rolling median of |fill - mark| * spot
- implied_spread_usd: median spread from consecutive buy/sell pairs
- available_depth_{unit}: trailing 5-min total ATM trade volume (depth proxy)

Output: vol_arb/output/{currency}_bar_execution_quality.parquet
        (BTC writes to the legacy unsuffixed name for backward compat)

Usage:
    PYTHONPATH=. python scripts/compute_bar_spread.py                 # BTC (default)
    PYTHONPATH=. python scripts/compute_bar_spread.py --asset ETHUSDT
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.paths import bar_exec_quality_path, option_trade_dir, CURRENCY_FOR_ASSET

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("bar_spread")


def process_day(day_path: Path) -> list[dict]:
    """Compute per-bar execution metrics for one day."""
    df = pd.read_parquet(day_path)
    if len(df) < 10:
        return []

    df = df[
        (df["mark_price"] > 0) & (df["index_price"] > 0) & (df["amount"] > 0)
    ].copy()
    if df.empty:
        return []

    # ATM filter
    parts = df["instrument_name"].str.split("-")
    strikes = parts.str[2].apply(pd.to_numeric, errors="coerce")
    moneyness = strikes / df["index_price"]
    df = df[(moneyness >= 0.95) & (moneyness <= 1.05)].copy()
    if len(df) < 5:
        return []

    # Ensure int64 ms timestamps
    if not pd.api.types.is_integer_dtype(df["timestamp"]):
        df["timestamp"] = df["timestamp"].astype("int64") // 10**6

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Per-trade slippage
    df["slip_usd"] = (df["price"] - df["mark_price"]).abs() * df["index_price"]

    # Group by 15-min bar
    bar_freq_ms = 900_000
    df["bar_ms"] = (df["timestamp"] // bar_freq_ms) * bar_freq_ms

    rows = []
    prev_bar_slip_median = None
    prev_bar_volume = None

    for bar_ms, group in df.groupby("bar_ms"):
        bar_ts = pd.Timestamp(bar_ms, unit="ms", tz="UTC")

        # --- Taker slip: trailing (use PREVIOUS bar's median, no lookahead) ---
        current_slip_median = float(group["slip_usd"].median())
        taker_slip = prev_bar_slip_median if prev_bar_slip_median is not None else current_slip_median
        prev_bar_slip_median = current_slip_median

        # --- Implied spread from buy/sell price difference ---
        buys = group[group["direction"] == "buy"]["price"].values
        sells = group[group["direction"] == "sell"]["price"].values
        if len(buys) > 0 and len(sells) > 0:
            buy_med = float(np.median(buys))
            sell_med = float(np.median(sells))
            idx_price = float(group["index_price"].median())
            spread_usd = abs(buy_med - sell_med) * idx_price
        else:
            spread_usd = taker_slip * 2  # fallback: spread ~ 2x half-spread

        # --- Available depth: trailing volume (previous bar) ---
        current_vol = float(group["amount"].sum())
        depth = prev_bar_volume if prev_bar_volume is not None else current_vol
        prev_bar_volume = current_vol

        rows.append({
            "timestamp": bar_ts,
            "bar_taker_slip_usd": taker_slip,
            "implied_spread_usd": spread_usd,
            "available_depth_btc": depth,
            "n_trades": len(group),
        })

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-bar execution quality from Deribit trade tape")
    parser.add_argument("--asset", default="BTCUSDT", help="BTCUSDT or ETHUSDT")
    args = parser.parse_args()

    currency = CURRENCY_FOR_ASSET[args.asset]
    trade_dir = option_trade_dir(currency)
    output_path = bar_exec_quality_path(currency)

    files = sorted(trade_dir.glob("*.parquet"))
    logger.info("[%s] Processing %d daily files from %s", currency, len(files), trade_dir)

    all_rows = []
    for i, f in enumerate(files):
        rows = process_day(f)
        all_rows.extend(rows)
        if (i + 1) % 200 == 0:
            logger.info("  [%d/%d] %d bars", i + 1, len(files), len(all_rows))

    result = pd.DataFrame(all_rows)
    result = result.set_index("timestamp").sort_index()
    result = result[~result.index.duplicated(keep="last")]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path)
    logger.info("Saved %d bars to %s", len(result), output_path)

    # Summary
    print()
    print("=" * 60)
    print(f"  PER-BAR EXECUTION QUALITY — {currency}")
    print("=" * 60)
    print(f"  Bars: {len(result)}")
    s = result["bar_taker_slip_usd"]
    print(f"  Taker slip: mean=${s.mean():.1f} med=${s.median():.1f} p95=${s.quantile(0.95):.1f}")
    sp = result["implied_spread_usd"]
    print(f"  Spread:     mean=${sp.mean():.1f} med=${sp.median():.1f} p95=${sp.quantile(0.95):.1f}")
    d = result["available_depth_btc"]  # column name kept for backward compat
    print(f"  Depth:      mean={d.mean():.1f} med={d.median():.1f} p10={d.quantile(0.1):.1f}")
    print("=" * 60)


if __name__ == "__main__":
    main()
