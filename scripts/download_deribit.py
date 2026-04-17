#!/usr/bin/env python
"""Download Deribit historical options + futures trades.

Options trades include IV directly — no need to compute from prices.
Downloads BTC and ETH, May 2023 - Mar 2024, in daily chunks.

Usage:
    python scripts/download_deribit.py
    python scripts/download_deribit.py --currency BTC --kind option
"""

import argparse
import json
import time
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("deribit")

BASE_URL = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
DATA_DIR = Path("data/deribit-trades")
SESSION = requests.Session()
SESSION.headers.update({"Accept": "application/json"})


def download_day(currency: str, kind: str, date: datetime) -> int:
    """Download all trades for one currency/kind/day. Returns trade count."""
    date_str = date.strftime("%Y-%m-%d")
    outdir = DATA_DIR / f"{currency}_{kind}"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{date_str}.json"

    if outfile.exists() and outfile.stat().st_size > 10:
        return -1  # already done

    start_ms = int(date.timestamp() * 1000)
    end_ms = int((date + timedelta(days=1)).timestamp() * 1000)

    all_trades = []
    has_more = True
    start_seq = None
    retries = 0

    while has_more:
        params = {
            "currency": currency,
            "kind": kind,
            "start_timestamp": start_ms,
            "end_timestamp": end_ms,
            "count": 1000,
            "sorting": "asc",
        }
        if start_seq is not None:
            params["start_seq"] = start_seq

        try:
            resp = SESSION.get(BASE_URL, params=params, timeout=30)
            data = resp.json()
        except Exception as e:
            retries += 1
            if retries > 5:
                logger.warning("%s %s %s: gave up after 5 retries: %s",
                               currency, kind, date_str, e)
                return len(all_trades)
            time.sleep(2 ** retries)
            continue

        if "error" in data:
            msg = data["error"].get("message", "")
            if "rate" in msg.lower():
                time.sleep(5)
                continue
            logger.warning("%s %s %s: API error: %s", currency, kind, date_str, msg)
            break

        result = data.get("result", {})
        trades = result.get("trades", [])
        has_more = result.get("has_more", False)

        if not trades:
            break

        all_trades.extend(trades)
        start_seq = trades[-1].get("trade_seq", trades[-1].get("trade_id"))
        retries = 0

        # Respect rate limits
        time.sleep(0.05)

    if all_trades:
        with open(outfile, "w") as f:
            json.dump(all_trades, f)

    return len(all_trades)


def download_range(currency: str, kind: str, start: datetime, end: datetime):
    """Download all days in range for one currency/kind."""
    current = start
    total_trades = 0
    days = 0

    while current < end:
        n = download_day(currency, kind, current)
        if n > 0:
            total_trades += n
        elif n == -1:
            pass  # skipped (already exists)
        days += 1

        if days % 30 == 0:
            logger.info("  %s %s: %d days, %d trades", currency, kind, days, total_trades)

        current += timedelta(days=1)

    logger.info("  %s %s: DONE — %d days, %d trades", currency, kind, days, total_trades)
    return total_trades


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--currency", nargs="+", default=["BTC", "ETH"])
    parser.add_argument("--kind", nargs="+", default=["option", "future"])
    parser.add_argument("--start", default="2023-05-01")
    parser.add_argument("--end", default="2024-04-01")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    print("=" * 70)
    print(f"  Deribit Historical Trades Download")
    print(f"  Currencies: {args.currency}")
    print(f"  Kinds: {args.kind}")
    print(f"  Period: {args.start} to {args.end}")
    print(f"  Workers: {args.workers}")
    print("=" * 70)

    tasks = [(c, k) for c in args.currency for k in args.kind]

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(download_range, c, k, start, end): (c, k)
            for c, k in tasks
        }
        for fut in as_completed(futures):
            c, k = futures[fut]
            try:
                n = fut.result()
                logger.info("Finished %s %s: %d trades", c, k, n)
            except Exception as e:
                logger.error("Failed %s %s: %s", c, k, e)

    # Summary
    total_files = sum(1 for _ in DATA_DIR.rglob("*.json") if _.stat().st_size > 10)
    print(f"\nDone: {total_files} files")


if __name__ == "__main__":
    main()
