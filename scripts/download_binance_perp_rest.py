#!/usr/bin/env python
"""Download Binance Futures perp aggTrades via REST API (when Vision daily zip
isn't yet published, e.g. for the current day).

Endpoint: https://fapi.binance.com/fapi/v1/aggTrades
Pagination: fromId (last trade's "a" + 1)
"""
from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime, timezone
from pathlib import Path

import polars as pl
import requests

BASE_URL = "https://fapi.binance.com/fapi/v1/aggTrades"
LIMIT = 1000  # max per request
SLEEP = 0.10  # seconds between requests (10 req/s)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("binance-perp-rest")


def fetch_range(symbol: str, start_ms: int, end_ms: int, session: requests.Session) -> list[dict]:
    """Fetch all aggTrades in [start_ms, end_ms]. Uses startTime/endTime first,
    then fromId pagination if the response is full."""
    out: list[dict] = []
    cur_start = start_ms
    last_id = None
    n_requests = 0

    while True:
        if last_id is None:
            params = {"symbol": symbol, "startTime": cur_start, "endTime": end_ms, "limit": LIMIT}
        else:
            params = {"symbol": symbol, "fromId": last_id + 1, "limit": LIMIT}

        for attempt in range(5):
            try:
                r = session.get(BASE_URL, params=params, timeout=30)
                if r.status_code == 429:
                    time.sleep(2 ** attempt)
                    continue
                r.raise_for_status()
                data = r.json()
                break
            except Exception as e:
                if attempt == 4:
                    raise
                time.sleep(2 ** attempt)
        n_requests += 1
        if not data:
            break

        # Filter to <= end_ms (fromId pagination can overshoot)
        data_in_range = [t for t in data if t["T"] <= end_ms]
        out.extend(data_in_range)

        if len(data) < LIMIT or data[-1]["T"] >= end_ms:
            break
        last_id = data[-1]["a"]
        if n_requests % 50 == 0:
            logger.info("  fetched %d trades in %d requests (last_ts=%s)",
                        len(out), n_requests,
                        datetime.fromtimestamp(data[-1]["T"] / 1000, tz=timezone.utc).strftime("%H:%M:%S"))
        time.sleep(SLEEP)

    logger.info("  done: %d trades in %d requests", len(out), n_requests)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", default="BTCUSDT")
    parser.add_argument("--date", required=True, help="YYYY-MM-DD")
    parser.add_argument("--end-time", default=None, help="Optional HH:MM cutoff (UTC) within the day")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    day = datetime.fromisoformat(args.date).replace(tzinfo=timezone.utc)
    start_ms = int(day.timestamp() * 1000)
    if args.end_time:
        h, m = map(int, args.end_time.split(":"))
        end_dt = day.replace(hour=h, minute=m, second=0, microsecond=0)
        end_ms = int(end_dt.timestamp() * 1000) - 1
    else:
        end_ms = int((day.timestamp() + 86400) * 1000) - 1

    out_path = Path(args.out) if args.out else Path(f"/c/tmp/{args.symbol}_perp_{args.date}.parquet")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Fetching %s perp aggTrades %s [%s -> %s]",
                args.symbol, args.date,
                datetime.fromtimestamp(start_ms / 1000, tz=timezone.utc),
                datetime.fromtimestamp(end_ms / 1000, tz=timezone.utc))
    s = requests.Session()
    t0 = time.time()
    trades = fetch_range(args.symbol, start_ms, end_ms, s)
    logger.info("Total: %d trades in %.0fs", len(trades), time.time() - t0)

    if not trades:
        logger.error("No trades fetched")
        return 1

    # Convert to Vision-compatible schema
    # Vision daily ZIP cols: agg_trade_id, price, quantity, first_trade_id,
    #                       last_trade_id, transact_time, is_buyer_maker
    # REST fields: a, p, q, f, l, T, m
    df = pl.DataFrame({
        "agg_trade_id": [t["a"] for t in trades],
        "price": [float(t["p"]) for t in trades],
        "quantity": [float(t["q"]) for t in trades],
        "first_trade_id": [t["f"] for t in trades],
        "last_trade_id": [t["l"] for t in trades],
        "transact_time": [t["T"] for t in trades],
        "is_buyer_maker": [t["m"] for t in trades],
    })
    df = df.sort("transact_time")
    df.write_parquet(out_path, compression="zstd")
    logger.info("Wrote %s (%d MB)", out_path, out_path.stat().st_size // (1024 * 1024))


if __name__ == "__main__":
    main()
