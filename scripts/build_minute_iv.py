#!/usr/bin/env python
"""Build 1-minute ATM IV timeseries from raw Deribit option trade ticks.

Processes daily parquet files in 30-day chunks, filters to ATM options,
and aggregates to 1-minute buckets with median IV and exit spread.

Output: vol_arb/output/minute_atm_iv.parquet
  Index: timestamp (1-min, END-of-minute, UTC)
  Columns: median_iv, exit_spread_usd, n_trades

Usage:
    PYTHONPATH=. python scripts/build_minute_iv.py
"""

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.options_loader import load_option_trades, filter_atm

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger(__name__)

TRADE_DIR = Path("data/deribit/trades/BTC_option")
OUTPUT = Path("vol_arb/output/minute_atm_iv.parquet")
CHUNK_DAYS = 30
FFILL_LIMIT = 60  # forward-fill up to 60 minutes


def process_chunk(start: str, end: str) -> pd.DataFrame:
    """Load a date range, filter ATM, aggregate to 1-minute buckets."""
    try:
        trades = load_option_trades(TRADE_DIR, start_date=start, end_date=end)
    except FileNotFoundError:
        return pd.DataFrame()

    atm = filter_atm(trades)
    if len(atm) == 0:
        return pd.DataFrame()

    # END-of-minute bucket: trade at 00:00:30 → bucket 00:01:00
    atm = atm.copy()
    atm["minute"] = atm["timestamp"].dt.ceil("1min")

    # Per-minute aggregation
    grouped = atm.groupby("minute")

    # Median IV
    median_iv = grouped["iv"].median()

    # Exit spread: |price - mark_price| * index_price gives per-contract USD cost
    atm["slip_usd"] = (atm["price"] - atm["mark_price"]).abs() * atm["index_price"]
    exit_spread = atm.groupby("minute")["slip_usd"].median()

    # Trade count
    n_trades = grouped.size()

    result = pd.DataFrame({
        "median_iv": median_iv,
        "exit_spread_usd": exit_spread,
        "n_trades": n_trades,
    })
    return result


def main():
    t0 = time.time()

    # Get all available dates
    files = sorted(TRADE_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {TRADE_DIR}")

    dates = [f.stem for f in files]
    first_date, last_date = dates[0], dates[-1]
    logger.info("Date range: %s to %s (%d files)", first_date, last_date, len(files))

    # Process in 30-day chunks
    chunks = []
    date_range = pd.date_range(first_date, last_date, freq="30D")
    # Add the last date to ensure we cover everything
    boundaries = [d.strftime("%Y-%m-%d") for d in date_range] + [last_date]

    for i in range(len(boundaries) - 1):
        start = boundaries[i]
        end = boundaries[i + 1]
        t1 = time.time()
        logger.info("[%d/%d] Processing %s to %s...", i + 1, len(boundaries) - 1, start, end)
        chunk = process_chunk(start, end)
        if len(chunk) > 0:
            chunks.append(chunk)
            logger.info("  → %d minute-buckets (%.1fs)", len(chunk), time.time() - t1)

    if not chunks:
        raise RuntimeError("No data produced")

    # Concatenate and deduplicate (overlapping chunk boundaries)
    df = pd.concat(chunks)
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    logger.info("Raw minute buckets: %d", len(df))

    # Create complete minute grid and forward-fill
    full_idx = pd.date_range(df.index.min(), df.index.max(), freq="1min", tz="UTC")
    df = df.reindex(full_idx)

    # Track which rows are real vs filled
    real_mask = df["n_trades"].notna()

    df["median_iv"] = df["median_iv"].ffill(limit=FFILL_LIMIT)
    df["exit_spread_usd"] = df["exit_spread_usd"].ffill(limit=FFILL_LIMIT)
    df["n_trades"] = df["n_trades"].fillna(0).astype(int)

    # Drop rows that couldn't be forward-filled (>60 min gap)
    df = df.dropna(subset=["median_iv"])

    # Summary stats
    n_real = int(real_mask.sum())
    n_total = len(df)
    elapsed = time.time() - t0
    logger.info("Done in %.0fs", elapsed)
    logger.info("Output: %d rows (%d real, %d filled, %.0f%% coverage)",
                n_total, n_real, n_total - n_real, 100 * n_real / n_total)
    logger.info("Date range: %s to %s", df.index.min(), df.index.max())
    logger.info("median_iv:       mean=%.1f, median=%.1f",
                df["median_iv"].mean(), df["median_iv"].median())
    logger.info("exit_spread_usd: mean=$%.1f, median=$%.1f",
                df["exit_spread_usd"].mean(), df["exit_spread_usd"].median())

    df.to_parquet(OUTPUT)
    logger.info("Saved to %s (%.1f MB)", OUTPUT, OUTPUT.stat().st_size / 1e6)


if __name__ == "__main__":
    main()
