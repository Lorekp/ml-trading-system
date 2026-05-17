#!/usr/bin/env python
"""One-shot: extend TSRV aggTrades parquet to cover May 12 (full) + May 13.

Existing file ``data/binance-aggTrades-parquet/BTCUSDT/2026-05.parquet``
covers May 1 → May 12 ~18:00. The cycle-26/27/28 parity check needs
coverage through May 13 12:00.

Strategy:
1. Read existing parquet.
2. Strip rows from 2026-05-12 onwards (those will be replaced).
3. Download May 12 daily ZIP from Binance Vision (full day available).
4. Use REST aggTrades for May 13 00:00 → present.
5. Concat + write back.
"""
from __future__ import annotations

import io
import logging
import sys
import time
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extend-tsrv")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data/binance-aggTrades-parquet/BTCUSDT/2026-05.parquet"
SYMBOL = "BTCUSDT"

COLS = [
    "agg_trade_id", "price", "quantity", "first_trade_id",
    "last_trade_id", "transact_time", "is_buyer_maker",
]


def fetch_vision_day(s: requests.Session, day: str) -> pl.DataFrame | None:
    url = f"https://data.binance.vision/data/futures/um/daily/aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-{day}.zip"
    logger.info("  GET %s", url)
    r = s.get(url, timeout=60)
    if r.status_code != 200:
        logger.warning("  %s: HTTP %d", day, r.status_code)
        return None
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as f:
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
    logger.info("  %s: %d trades (Vision)", day, len(df))
    return df


def fetch_rest_window(start_ms: int, end_ms: int) -> pl.DataFrame:
    """Page through fapi /aggTrades using fromId; ~24h window."""
    s = requests.Session()
    base = "https://fapi.binance.com/fapi/v1/aggTrades"

    rows: list[dict] = []
    # Bootstrap with start_ms
    last_id = None
    last_ts = start_ms - 1
    # 2400 req/min → ≥25ms between requests. Use 50ms = 1200 req/min safe margin.
    delay = 0.05
    while True:
        if last_id is None:
            params = {"symbol": SYMBOL, "startTime": last_ts + 1, "limit": 1000}
        else:
            params = {"symbol": SYMBOL, "fromId": last_id + 1, "limit": 1000}
        for attempt in range(10):
            time.sleep(delay)
            r = s.get(base, params=params, timeout=30)
            if r.status_code == 200:
                break
            if r.status_code == 429:
                # Rate-limited; sleep longer + bump delay
                logger.warning("429 rate-limited, sleeping 30s (attempt %d)", attempt)
                time.sleep(30 + attempt * 10)
                delay = min(delay * 1.5, 0.5)
                continue
            time.sleep(2 + attempt)
        else:
            raise RuntimeError(f"REST failed: {r.status_code} {r.text[:200]}")
        batch = r.json()
        if not batch:
            break
        for t in batch:
            ts = int(t["T"])
            if ts >= end_ms:
                break
            rows.append({
                "agg_trade_id": int(t["a"]),
                "price": float(t["p"]),
                "quantity": float(t["q"]),
                "first_trade_id": int(t["f"]),
                "last_trade_id": int(t["l"]),
                "transact_time": ts,
                "is_buyer_maker": bool(t["m"]),
            })
        last_id = int(batch[-1]["a"])
        last_ts = int(batch[-1]["T"])
        if last_ts >= end_ms:
            break
        if len(rows) % 50_000 < 1000:
            logger.info("  REST: %d trades so far (latest ts %s)", len(rows),
                        datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc))
    logger.info("  REST: total %d trades", len(rows))
    return pl.DataFrame(rows)


def main() -> int:
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(pool_connections=4, pool_maxsize=4, max_retries=3))

    logger.info("Loading existing parquet: %s", OUT)
    existing = pl.read_parquet(OUT)
    logger.info("  rows=%d, max=%s", len(existing),
                datetime.fromtimestamp(existing["transact_time"].max() / 1000, tz=timezone.utc))

    # Strip rows from 2026-05-12 onwards.
    cutoff_ms = int(datetime(2026, 5, 12, tzinfo=timezone.utc).timestamp() * 1000)
    existing_trimmed = existing.filter(pl.col("transact_time") < cutoff_ms)
    logger.info("After trim (<2026-05-12): %d rows", len(existing_trimmed))

    # Fetch May 12 from Vision.
    may12 = fetch_vision_day(s, "2026-05-12")
    if may12 is None:
        logger.error("Failed to fetch May 12 from Vision")
        return 2

    # Fetch May 13 from REST (up to now).
    may13_start_ms = int(datetime(2026, 5, 13, tzinfo=timezone.utc).timestamp() * 1000)
    now_ms = int(time.time() * 1000)
    may13 = fetch_rest_window(may13_start_ms, now_ms)

    # Align all three frames to the existing parquet's column set.
    target_cols = existing_trimmed.columns
    logger.info("target_cols (from existing): %s", target_cols)
    logger.info("may12 cols: %s", may12.columns)
    logger.info("may13 cols: %s", may13.columns)

    def align(df: pl.DataFrame) -> pl.DataFrame:
        # Add missing cols derived from price*quantity if asked
        if "dollar_volume" in target_cols and "dollar_volume" not in df.columns:
            df = df.with_columns(
                (pl.col("price") * pl.col("quantity")).alias("dollar_volume")
            )
        # Drop extras + reorder to match target
        keep = [c for c in target_cols if c in df.columns]
        missing = [c for c in target_cols if c not in df.columns]
        if missing:
            logger.warning("Missing cols in incoming frame: %s — will fill nulls", missing)
            for c in missing:
                df = df.with_columns(pl.lit(None).alias(c))
        return df.select(target_cols)

    may12 = align(may12)
    may13 = align(may13)
    combined = pl.concat([existing_trimmed, may12, may13], how="vertical_relaxed")
    combined = combined.sort("transact_time")
    logger.info("Combined: %d rows", len(combined))
    logger.info("  min=%s",
                datetime.fromtimestamp(combined["transact_time"].min() / 1000, tz=timezone.utc))
    logger.info("  max=%s",
                datetime.fromtimestamp(combined["transact_time"].max() / 1000, tz=timezone.utc))

    combined.write_parquet(OUT, compression="zstd")
    logger.info("Wrote %s (%.1f MB)", OUT, OUT.stat().st_size / (1024 * 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
