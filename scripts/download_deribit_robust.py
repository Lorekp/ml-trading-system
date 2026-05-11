#!/usr/bin/env python3
"""Robust Deribit historical trades downloader.

Designed to run unattended overnight on a remote server.
Handles: rate limits (exponential backoff), corrupt JSON, network errors,
partial downloads (resume-safe), and disk flush after every file.

Downloads BTC + ETH options and futures trades (May 2023 - Mar 2024).
Options trades include implied volatility (iv field).

Usage:
    python3 download_deribit_robust.py
    python3 download_deribit_robust.py --data-dir /hdd/deribit-trades
    nohup python3 download_deribit_robust.py --data-dir /hdd/deribit-trades > deribit.log 2>&1 &
"""

import argparse
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path

import requests

# ── Config ──

CURRENCIES = ["BTC", "ETH"]
KINDS = ["option", "future"]
START = datetime(2023, 5, 1, tzinfo=timezone.utc)
END = datetime(2024, 4, 1, tzinfo=timezone.utc)
BASE_URL = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
PAGE_SIZE = 1000
BASE_DELAY = 0.2        # seconds between API calls (5 req/s)
MAX_RETRIES = 10         # per page
BACKOFF_BASE = 2.0       # exponential backoff base
MAX_BACKOFF = 120.0      # max wait between retries

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("deribit")


def fetch_page(session, currency, kind, start_ms, end_ms, start_seq=None):
    """Fetch one page of trades. Returns (trades, has_more, next_seq) or raises."""
    params = {
        "currency": currency,
        "kind": kind,
        "start_timestamp": start_ms,
        "end_timestamp": end_ms,
        "count": PAGE_SIZE,
        "sorting": "asc",
    }
    if start_seq is not None:
        params["start_seq"] = start_seq

    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(BASE_URL, params=params, timeout=60)

            if resp.status_code == 429 or resp.status_code == 503:
                wait = min(BACKOFF_BASE ** (attempt + 2), MAX_BACKOFF)
                logger.warning("HTTP %d — backing off %.0fs (attempt %d/%d)",
                               resp.status_code, wait, attempt + 1, MAX_RETRIES)
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

        except requests.exceptions.Timeout:
            wait = min(BACKOFF_BASE ** (attempt + 1), MAX_BACKOFF)
            logger.warning("Timeout — retrying in %.0fs (attempt %d/%d)",
                           wait, attempt + 1, MAX_RETRIES)
            time.sleep(wait)
            continue

        except requests.exceptions.ConnectionError as e:
            wait = min(BACKOFF_BASE ** (attempt + 2), MAX_BACKOFF)
            logger.warning("Connection error: %s — retrying in %.0fs", e, wait)
            time.sleep(wait)
            continue

        except json.JSONDecodeError:
            wait = min(BACKOFF_BASE ** (attempt + 1), MAX_BACKOFF)
            logger.warning("Corrupt JSON response — retrying in %.0fs", wait)
            time.sleep(wait)
            continue

        # Check API-level errors
        if "error" in data:
            msg = data["error"].get("message", str(data["error"]))
            code = data["error"].get("code", 0)

            if "too_many_requests" in msg.lower() or code == 10028:
                wait = min(BACKOFF_BASE ** (attempt + 2), MAX_BACKOFF)
                logger.warning("Rate limited — backing off %.0fs (attempt %d/%d)",
                               wait, attempt + 1, MAX_RETRIES)
                time.sleep(wait)
                continue

            # Non-retryable API error
            raise RuntimeError(f"Deribit API error: {msg}")

        result = data.get("result", {})
        trades = result.get("trades", [])
        has_more = result.get("has_more", False)

        next_seq = None
        if trades and has_more:
            next_seq = trades[-1].get("trade_seq", trades[-1].get("trade_id"))

        return trades, has_more, next_seq

    raise RuntimeError(f"Exhausted {MAX_RETRIES} retries")


def download_day(session, currency, kind, date, data_dir):
    """Download all trades for one day. Resume-safe (skips if file exists).

    Streams trades to disk incrementally to avoid OOM on heavy days
    (BTC options can have 600K+ trades/day = ~400MB in memory).
    Writes to tmp file, renames on success.
    Returns trade count, or -1 if skipped.
    """
    date_str = date.strftime("%Y-%m-%d")
    outdir = data_dir / f"{currency}_{kind}"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{date_str}.json"

    # Skip if already downloaded and valid
    if outfile.exists():
        try:
            size = outfile.stat().st_size
            if size > 10:
                with open(outfile, "r") as f:
                    c = f.read(1)
                    if c == "[":
                        return -1  # valid, skip
            outfile.unlink()
        except Exception:
            outfile.unlink(missing_ok=True)

    start_ms = int(date.timestamp() * 1000)
    end_ms = int((date + timedelta(days=1)).timestamp() * 1000)

    tmpfile = outfile.with_suffix(".tmp")
    has_more = True
    start_seq = None
    pages = 0
    total_trades = 0

    try:
        with open(tmpfile, "w") as f:
            f.write("[")
            first = True

            while has_more:
                try:
                    trades, has_more, next_seq = fetch_page(
                        session, currency, kind, start_ms, end_ms, start_seq)
                except RuntimeError as e:
                    logger.error("%s %s %s page %d: %s",
                                 currency, kind, date_str, pages, e)
                    break

                if not trades:
                    break

                for t in trades:
                    if not first:
                        f.write(",")
                    json.dump(t, f, separators=(",", ":"))
                    first = False

                total_trades += len(trades)
                start_seq = next_seq
                pages += 1

                # Flush every 50 pages (~50K trades) to limit memory
                if pages % 50 == 0:
                    f.flush()

                time.sleep(BASE_DELAY)

            f.write("]")
            f.flush()
            os.fsync(f.fileno())

    except Exception as e:
        logger.error("Write failed %s: %s", tmpfile, e)
        tmpfile.unlink(missing_ok=True)
        return 0

    if total_trades > 0:
        tmpfile.rename(outfile)
    else:
        tmpfile.unlink(missing_ok=True)

    return total_trades


def download_stream(currency, kind, data_dir):
    """Download one full stream (currency × kind) sequentially."""
    session = requests.Session()
    session.headers.update({
        "Accept": "application/json",
        "User-Agent": "deribit-historical-downloader/1.0",
    })

    current = START
    total_trades = 0
    total_days = 0
    skipped = 0

    logger.info("START %s %s: %s to %s", currency, kind,
                START.strftime("%Y-%m-%d"), END.strftime("%Y-%m-%d"))

    while current < END:
        try:
            n = download_day(session, currency, kind, current, data_dir)
        except Exception as e:
            logger.error("UNEXPECTED %s %s %s: %s\n%s",
                         currency, kind, current.strftime("%Y-%m-%d"),
                         e, traceback.format_exc())
            # Don't crash the whole stream — skip this day
            n = 0
            time.sleep(10)

        if n == -1:
            skipped += 1
        elif n > 0:
            total_trades += n

        total_days += 1
        current += timedelta(days=1)

        # Progress every 7 days
        if total_days % 7 == 0:
            logger.info("  %s %s: day %d, %d trades, %d skipped",
                         currency, kind, total_days, total_trades, skipped)

    logger.info("DONE %s %s: %d days, %d trades, %d skipped",
                currency, kind, total_days, total_trades, skipped)
    return total_trades


def main():
    parser = argparse.ArgumentParser(description="Robust Deribit historical trades downloader")
    parser.add_argument("--data-dir", default="/hdd/deribit-trades",
                        help="Output directory (default: /hdd/deribit-trades)")
    parser.add_argument("--start", default=None,
                        help="ISO start date (default: hardcoded START)")
    parser.add_argument("--end", default=None,
                        help="ISO end date EXCLUSIVE (default: hardcoded END)")
    parser.add_argument("--currencies", default=None,
                        help="Comma-separated subset of currencies (default: all)")
    parser.add_argument("--kinds", default=None,
                        help="Comma-separated subset of kinds (default: option,future)")
    args = parser.parse_args()

    # Allow CLI overrides of the module-level constants. We mutate the
    # globals so download_stream picks them up without further refactor.
    global START, END, CURRENCIES, KINDS
    if args.start:
        START = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    if args.end:
        END = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    if args.currencies:
        CURRENCIES = [c.strip() for c in args.currencies.split(",") if c.strip()]
    if args.kinds:
        KINDS = [k.strip() for k in args.kinds.split(",") if k.strip()]

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 70)
    logger.info("Deribit Historical Trades — Robust Downloader")
    logger.info("Output: %s", data_dir)
    logger.info("Currencies: %s", CURRENCIES)
    logger.info("Kinds: %s", KINDS)
    logger.info("Period: %s to %s", START.strftime("%Y-%m-%d"), END.strftime("%Y-%m-%d"))
    logger.info("Rate: %.1f req/s base, backoff up to %.0fs", 1 / BASE_DELAY, MAX_BACKOFF)
    logger.info("=" * 70)

    # Run streams SEQUENTIALLY to avoid rate limit issues
    # Each stream: ~335 days × ~10-600 pages/day = hours
    # Sequential = ~2x slower but zero rate limit problems
    grand_total = 0
    for currency in CURRENCIES:
        for kind in KINDS:
            try:
                n = download_stream(currency, kind, data_dir)
                grand_total += n
            except KeyboardInterrupt:
                logger.info("Interrupted by user")
                sys.exit(0)
            except Exception as e:
                logger.error("Stream %s %s FAILED: %s", currency, kind, e)

    # Summary
    total_files = sum(1 for _ in data_dir.rglob("*.json") if _.stat().st_size > 10)
    total_size = sum(f.stat().st_size for f in data_dir.rglob("*.json"))
    logger.info("=" * 70)
    logger.info("ALL DONE: %d trades, %d files, %.1f GB",
                grand_total, total_files, total_size / 1e9)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
