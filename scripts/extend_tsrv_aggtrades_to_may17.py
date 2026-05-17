#!/usr/bin/env python
"""One-shot: extend TSRV aggTrades parquet to cover May 13 → May 17 2026.

Existing ``data/binance-aggTrades-parquet/BTCUSDT/2026-05.parquet``
covers May 1 → May 13 ~17:10. The cycle-36 parity check needs
coverage through May 17 ~17:30 (the live bar-trace window).

Strategy:
1. Read existing parquet, strip rows from 2026-05-13 onwards.
2. Download May 13/14/15/16 daily ZIPs from Binance Vision (full days).
3. REST aggTrades for May 17 00:00 → now.
4. Concat + write back (drop the stray __index_level_0__ column).
"""
from __future__ import annotations

import io
import logging
import sys
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("extend-tsrv-may17")

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "data/binance-aggTrades-parquet/BTCUSDT/2026-05.parquet"
SYMBOL = "BTCUSDT"
COLS = ["agg_trade_id", "price", "quantity", "first_trade_id",
        "last_trade_id", "transact_time", "is_buyer_maker"]
VISION_DAYS = ["2026-05-13", "2026-05-14", "2026-05-15", "2026-05-16"]
REST_DAY_START = datetime(2026, 5, 17, tzinfo=timezone.utc)
STRIP_CUTOFF = datetime(2026, 5, 13, tzinfo=timezone.utc)


def fetch_vision_day(s: requests.Session, day: str) -> pl.DataFrame | None:
    url = (f"https://data.binance.vision/data/futures/um/daily/aggTrades/"
           f"{SYMBOL}/{SYMBOL}-aggTrades-{day}.zip")
    content: bytes | None = None
    for attempt in range(6):
        try:
            logger.info("  GET %s (attempt %d)", url, attempt + 1)
            r = s.get(url, timeout=180, stream=False)
            if r.status_code != 200:
                logger.warning("  %s: HTTP %d", day, r.status_code)
                return None
            content = r.content
            # Validate the zip is complete before accepting.
            zipfile.ZipFile(io.BytesIO(content)).namelist()
            break
        except (requests.exceptions.RequestException,
                zipfile.BadZipFile) as exc:
            logger.warning("  %s: download failed (%s) — retrying in %ds",
                            day, type(exc).__name__, 5 * (attempt + 1))
            time.sleep(5 * (attempt + 1))
            content = None
    if content is None:
        logger.error("  %s: all retries exhausted", day)
        return None
    with zipfile.ZipFile(io.BytesIO(content)) as zf:
        csv_name = next(n for n in zf.namelist() if n.endswith(".csv"))
        with zf.open(csv_name) as f:
            first = f.readline().decode().strip()
            has_header = first.startswith("agg")
        with zf.open(csv_name) as f:
            df = pl.read_csv(
                f, has_header=has_header,
                new_columns=COLS if not has_header else None,
                schema_overrides={
                    "agg_trade_id": pl.Int64, "price": pl.Float64,
                    "quantity": pl.Float64, "first_trade_id": pl.Int64,
                    "last_trade_id": pl.Int64, "transact_time": pl.Int64,
                    "is_buyer_maker": pl.Boolean,
                },
            )
    logger.info("  %s: %d trades (Vision)", day, len(df))
    return df


def fetch_rest_window(start_ms: int, end_ms: int) -> pl.DataFrame:
    s = requests.Session()
    base = "https://fapi.binance.com/fapi/v1/aggTrades"
    rows: list[dict] = []
    last_id = None
    last_ts = start_ms - 1
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
        stop = False
        for t in batch:
            ts = int(t["T"])
            if ts >= end_ms:
                stop = True
                break
            rows.append({
                "agg_trade_id": int(t["a"]), "price": float(t["p"]),
                "quantity": float(t["q"]), "first_trade_id": int(t["f"]),
                "last_trade_id": int(t["l"]), "transact_time": ts,
                "is_buyer_maker": bool(t["m"]),
            })
        if stop:
            break
        last_id = int(batch[-1]["a"])
        last_ts = int(batch[-1]["T"])
        if last_ts >= end_ms:
            break
        if len(rows) % 100_000 < 1000:
            logger.info("  REST: %d trades (latest %s)", len(rows),
                        datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc))
    logger.info("  REST: total %d trades", len(rows))
    return pl.DataFrame(rows) if rows else pl.DataFrame(schema={c: (
        pl.Int64 if c.endswith("id") or c == "transact_time"
        else pl.Boolean if c == "is_buyer_maker" else pl.Float64) for c in COLS})


def main() -> int:
    s = requests.Session()
    s.mount("https://", requests.adapters.HTTPAdapter(
        pool_connections=4, pool_maxsize=4, max_retries=3))

    logger.info("Loading existing parquet: %s", OUT)
    existing = pl.read_parquet(OUT)
    if "__index_level_0__" in existing.columns:
        existing = existing.drop("__index_level_0__")
    logger.info("  rows=%d, max=%s", len(existing),
                datetime.fromtimestamp(existing["transact_time"].max() / 1000,
                                       tz=timezone.utc))

    cutoff_ms = int(STRIP_CUTOFF.timestamp() * 1000)
    existing_trimmed = existing.filter(pl.col("transact_time") < cutoff_ms)
    logger.info("After trim (<%s): %d rows", STRIP_CUTOFF.date(), len(existing_trimmed))

    frames = [existing_trimmed]
    target_cols = existing_trimmed.columns

    def align(df: pl.DataFrame) -> pl.DataFrame:
        if "dollar_volume" in target_cols and "dollar_volume" not in df.columns:
            df = df.with_columns(
                (pl.col("price") * pl.col("quantity")).alias("dollar_volume"))
        for c in target_cols:
            if c not in df.columns:
                df = df.with_columns(pl.lit(None).alias(c))
        return df.select(target_cols)

    for day in VISION_DAYS:
        d = fetch_vision_day(s, day)
        if d is None:
            logger.error("Vision day %s unavailable — aborting", day)
            return 2
        frames.append(align(d))

    rest_start_ms = int(REST_DAY_START.timestamp() * 1000)
    now_ms = int(time.time() * 1000)
    rest = fetch_rest_window(rest_start_ms, now_ms)
    if len(rest):
        frames.append(align(rest))

    combined = pl.concat(frames, how="vertical_relaxed").sort("transact_time")
    combined = combined.unique(subset=["agg_trade_id"], keep="first").sort("transact_time")
    logger.info("Combined: %d rows  [%s -> %s]", len(combined),
                datetime.fromtimestamp(combined["transact_time"].min() / 1000, tz=timezone.utc),
                datetime.fromtimestamp(combined["transact_time"].max() / 1000, tz=timezone.utc))
    combined.write_parquet(OUT, compression="zstd")
    logger.info("Wrote %s (%.1f MB)", OUT, OUT.stat().st_size / (1024 * 1024))
    return 0


if __name__ == "__main__":
    sys.exit(main())
