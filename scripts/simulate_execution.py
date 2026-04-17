"""Trade-by-trade maker fill simulation against the real Deribit tape.

For each 15-min bar in the vol_arb index, loads the actual ATM option
trades and simulates three execution approaches:

  A: Taker entry + maker exit (post exit limit at bar open, 15m window)
  B: MkrTkr60 entry (60s limit then taker) + maker exit (15m window)
  C: Split-fill entry (70% taker + 30% limit 60s) + maker exit

Output: vol_arb/output/execution_sim.parquet with per-bar maker fill
rates and cost savings for each approach.

Usage:
    PYTHONPATH=. python scripts/simulate_execution.py
    PYTHONPATH=. python scripts/simulate_execution.py --size 0.5
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("exec_sim")

TRADE_DIR = Path("data/deribit/trades/BTC_option")
OUTPUT_PATH = Path("vol_arb/output/execution_sim.parquet")


def simulate_maker_fill(
    trades: pd.DataFrame,
    side: str,
    limit_price: float,
    target_size: float,
    window_end_ms: int,
) -> tuple[float, float]:
    """Walk the trade tape and simulate a maker limit order fill.

    Parameters
    ----------
    trades : DataFrame with columns [timestamp, price, amount, direction]
        Sorted by timestamp (ms). Only ATM trades for relevant instruments.
    side : 'buy' or 'sell'
        Our limit order side.
    limit_price : float
        Price we post at (BTC-denominated option price).
    target_size : float
        BTC we want filled.
    window_end_ms : int
        Timestamp (ms) beyond which we stop looking.

    Returns
    -------
    (filled_amount, avg_fill_price)
    """
    filled = 0.0
    value = 0.0
    ts = trades["timestamp"].values
    prices = trades["price"].values
    amounts = trades["amount"].values
    dirs = trades["direction"].values

    for j in range(len(trades)):
        if ts[j] > window_end_ms:
            break
        # Our BUY limit fills when someone aggressively SELLS at/below our price
        if side == "buy" and dirs[j] == "sell" and prices[j] <= limit_price:
            take = min(amounts[j], target_size - filled)
            filled += take
            value += take * prices[j]
        # Our SELL limit fills when someone aggressively BUYS at/above our price
        elif side == "sell" and dirs[j] == "buy" and prices[j] >= limit_price:
            take = min(amounts[j], target_size - filled)
            filled += take
            value += take * prices[j]
        if filled >= target_size - 1e-10:
            break

    avg_price = value / filled if filled > 0 else 0.0
    return filled, avg_price


def process_day(
    day_path: Path,
    bar_size_btc: float,
    bar_freq_ms: int = 900_000,
    entry_window_ms: int = 60_000,
) -> list[dict]:
    """Process one day of trades, simulate execution for each 15-min bar."""
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

    # Ensure timestamp is int64 ms (some parquets store as datetime)
    if not pd.api.types.is_integer_dtype(df["timestamp"]):
        df["timestamp"] = df["timestamp"].astype("int64") // 10**6

    df = df.sort_values("timestamp").reset_index(drop=True)

    # Group by 15-min bar
    df["bar_ms"] = (df["timestamp"] // bar_freq_ms) * bar_freq_ms
    results = []

    for bar_ms, group in df.groupby("bar_ms"):
        if len(group) < 2:
            continue

        bar_ts = pd.Timestamp(bar_ms, unit="ms", tz="UTC")
        mark = group["mark_price"].iloc[0]
        idx_price = group["index_price"].iloc[0]
        taker_slip_per_btc = (group["price"] - group["mark_price"]).abs().median() * idx_price

        entry_end_ms = bar_ms + entry_window_ms
        exit_end_ms = bar_ms + bar_freq_ms

        # For a SHORT straddle (dominant signal):
        # Entry = SELL call + SELL put (our limit: sell at mark)
        # Exit = BUY call + BUY put (our limit: buy at mark)
        # For simplicity, use the ATM pool (both calls and puts trade near mark)

        entry_trades = group[group["timestamp"] <= entry_end_ms]
        all_trades = group  # full 15-min window

        # --- Approach A: taker entry + maker exit ---
        exit_filled, exit_price = simulate_maker_fill(
            all_trades, "buy", mark, bar_size_btc, exit_end_ms,
        )
        a_exit_rate = min(exit_filled / bar_size_btc, 1.0)

        # --- Approach B: MkrTkr60 entry + maker exit ---
        entry_filled, entry_price = simulate_maker_fill(
            entry_trades, "sell", mark, bar_size_btc, entry_end_ms,
        )
        b_entry_rate = min(entry_filled / bar_size_btc, 1.0)
        b_exit_rate = a_exit_rate  # same exit logic

        # --- Approach C: split-fill entry + maker exit ---
        maker_portion = bar_size_btc * 0.3
        c_entry_filled, _ = simulate_maker_fill(
            entry_trades, "sell", mark, maker_portion, entry_end_ms,
        )
        c_entry_maker_rate = min(c_entry_filled / maker_portion, 1.0) if maker_portion > 0 else 0.0

        # Cost savings (per BTC, for the straddle = 2 legs)
        # Each maker fill saves ~taker_slip_per_btc on that leg
        a_saved = a_exit_rate * taker_slip_per_btc * 2  # 2 exit legs
        b_saved = (b_entry_rate * 2 + b_exit_rate * 2) * taker_slip_per_btc  # 4 legs
        c_saved = (c_entry_maker_rate * 0.3 * 2 + a_exit_rate * 2) * taker_slip_per_btc

        results.append({
            "timestamp": bar_ts,
            "taker_slip_per_btc": taker_slip_per_btc,
            "a_exit_maker_rate": a_exit_rate,
            "a_cost_saved_per_btc": a_saved,
            "b_entry_maker_rate": b_entry_rate,
            "b_exit_maker_rate": b_exit_rate,
            "b_cost_saved_per_btc": b_saved,
            "c_entry_maker_rate": c_entry_maker_rate,
            "c_exit_maker_rate": a_exit_rate,
            "c_cost_saved_per_btc": c_saved,
            "n_atm_trades": len(group),
        })

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate execution approaches")
    parser.add_argument("--size", type=float, default=0.5,
                        help="Position size per leg in BTC (default 0.5)")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    files = sorted(TRADE_DIR.glob("*.parquet"))
    logger.info("Processing %d days at size=%.2f BTC...", len(files), args.size)

    all_rows = []
    for i, f in enumerate(files):
        rows = process_day(f, bar_size_btc=args.size)
        all_rows.extend(rows)
        if (i + 1) % 100 == 0:
            logger.info("  [%d/%d] %d bars so far", i + 1, len(files), len(all_rows))

    result = pd.DataFrame(all_rows)
    result = result.set_index("timestamp").sort_index()
    result = result[~result.index.duplicated(keep="last")]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_parquet(output_path)
    logger.info("Saved %d bars to %s", len(result), output_path)

    # Summary
    print()
    print("=" * 70)
    print(f"  EXECUTION SIMULATION RESULTS (size={args.size} BTC/leg)")
    print("=" * 70)
    print(f"  Total bars: {len(result)}")
    print()
    print(f"  {'Approach':<25} {'Maker Rate':>12} {'Saved/BTC':>12}")
    print("-" * 70)
    print(f"  {'A: maker exit':<25} {result['a_exit_maker_rate'].mean():>11.1%} ${result['a_cost_saved_per_btc'].mean():>11.2f}")
    print(f"  {'B: mkrtkr60 entry+exit':<25} entry {result['b_entry_maker_rate'].mean():.1%} / exit {result['b_exit_maker_rate'].mean():.1%} ${result['b_cost_saved_per_btc'].mean():>11.2f}")
    print(f"  {'C: split + maker exit':<25} entry30% {result['c_entry_maker_rate'].mean():.1%} / exit {result['c_exit_maker_rate'].mean():.1%} ${result['c_cost_saved_per_btc'].mean():>11.2f}")
    print(f"\n  Avg taker cost: ${result['taker_slip_per_btc'].mean():.2f}/BTC")
    print("=" * 70)


if __name__ == "__main__":
    main()
