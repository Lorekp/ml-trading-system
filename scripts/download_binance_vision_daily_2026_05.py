#!/usr/bin/env python
"""One-shot daily Binance Vision aggTrades downloader for May 1-12 2026.

Stitches them into data/binance-aggTrades-parquet/BTCUSDT/2026-05.parquet
for the TSRV / vol_arb pipeline to consume.

This is a temporary helper used by the cycle-20-25 audit campaign to
bring the offline pipeline up to the live engine's bar window.
"""
from __future__ import annotations

import io
import logging
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import polars as pl
import requests
from requests.adapters import HTTPAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("dl")

AGG_DAILY_BASE = "https://data.binance.vision/data/futures/um/daily/aggTrades"
SYMBOL = "BTCUSDT"
OUT_PARQUET = Path(f"data/binance-aggTrades-parquet/{SYMBOL}/2026-05.parquet")
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

COLS = [
    "agg_trade_id", "price", "quantity", "first_trade_id",
    "last_trade_id", "transact_time", "is_buyer_maker",
]


def session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=3))
    return s


def fetch_day(s: requests.Session, day: str) -> pl.DataFrame | None:
    url = f"{AGG_DAILY_BASE}/{SYMBOL}/{SYMBOL}-aggTrades-{day}.zip"
    logger.info("  GET %s", url)
    r = s.get(url, timeout=60)
    if r.status_code != 200:
        logger.warning("  %s: HTTP %d", day, r.status_code)
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as f:
            # Check for header
            first = f.readline().decode().strip()
            has_header = first.startswith("agg")
        with zf.open(csv_name) as f:
            df = pl.read_csv(
                f,
                has_header=has_header,
                new_columns=COLS if not has_header else None,
                schema_overrides={
                    "agg_trade_id": pl.Int64,
                    "price": pl.Float64,
                    "quantity": pl.Float64,
                    "first_trade_id": pl.Int64,
                    "last_trade_id": pl.Int64,
                    "transact_time": pl.Int64,
                    "is_buyer_maker": pl.Boolean,
                },
            )
    logger.info("  %s: %d trades", day, len(df))
    return df


def main():
    s = session()
    days = []
    start = datetime(2026, 5, 1, tzinfo=timezone.utc)
    end = datetime(2026, 5, 13, tzinfo=timezone.utc)
    cur = start
    while cur < end:
        days.append(cur.strftime("%Y-%m-%d"))
        cur += timedelta(days=1)

    frames: list[pl.DataFrame] = []
    for d in days:
        df = fetch_day(s, d)
        if df is not None and len(df):
            frames.append(df)

    if not frames:
        logger.error("No data downloaded")
        return 2

    combined = pl.concat(frames, how="vertical_relaxed")
    combined = combined.sort("transact_time")
    logger.info("Total: %d rows, %d days", len(combined), len(frames))
    combined.write_parquet(OUT_PARQUET, compression="zstd")
    logger.info("Wrote %s (%d MB)", OUT_PARQUET, OUT_PARQUET.stat().st_size // (1024 * 1024))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
