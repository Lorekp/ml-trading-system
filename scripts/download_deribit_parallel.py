#!/usr/bin/env python
"""Parallel Deribit historical trades downloader for the parity-extension cycle.

Optimizations vs ``download_deribit_robust.py``:
  - PAGE_SIZE = 10_000 (Deribit max for this endpoint) → 10× fewer API calls.
  - Per-day parallelism via ThreadPoolExecutor → 7 days in flight at once.
  - Streams trades directly to parquet (Polars) → no 500MB intermediate
    JSON file; ~30x smaller on-disk footprint.
  - Each worker has its own requests.Session so rate-limit budgets are
    per-connection.
  - Adaptive backoff on 429 responses; otherwise minimal delay.

Usage:
    PYTHONPATH=. python scripts/download_deribit_parallel.py \\
        --data-dir data/deribit/trades \\
        --start 2026-05-06 --end 2026-05-13 \\
        --currencies BTC --kinds option,future \\
        --workers 7
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import requests
from requests.adapters import HTTPAdapter

BASE_URL = "https://history.deribit.com/api/v2/public/get_last_trades_by_currency_and_time"
PAGE_SIZE = 10_000          # Deribit max
DEFAULT_DELAY = 0.05        # 20 req/s baseline (per worker)
MAX_RETRIES = 8
BACKOFF_BASE = 2.0
MAX_BACKOFF = 60.0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("deribit-parallel")


def make_session() -> requests.Session:
    s = requests.Session()
    s.mount("https://", HTTPAdapter(pool_connections=4, pool_maxsize=4))
    s.headers.update({
        "Accept": "application/json",
        "User-Agent": "deribit-parallel-downloader/1.0",
    })
    return s


def fetch_page(session, currency, kind, start_ms, end_ms, start_seq=None):
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
            if resp.status_code == 429:
                wait = min(MAX_BACKOFF, BACKOFF_BASE * (2 ** attempt))
                time.sleep(wait)
                continue
            if resp.status_code != 200:
                wait = min(MAX_BACKOFF, BACKOFF_BASE * (2 ** attempt))
                time.sleep(wait)
                continue
            data = resp.json()
            if "error" in data:
                msg = data.get("error", {}).get("message", "")
                if "rate" in msg.lower():
                    time.sleep(BACKOFF_BASE * (2 ** attempt))
                    continue
                return [], False, None
            result = data.get("result", {})
            return (
                result.get("trades", []),
                result.get("has_more", False),
                None,
            )
        except (requests.RequestException, ValueError) as e:
            wait = min(MAX_BACKOFF, BACKOFF_BASE * (2 ** attempt))
            time.sleep(wait)
    raise RuntimeError(f"exhausted retries for {currency}/{kind} {start_ms}-{end_ms}")


def download_day(
    currency: str,
    kind: str,
    date: datetime,
    data_dir: Path,
    delay: float = DEFAULT_DELAY,
) -> tuple[str, str, int]:
    """Download one day → write parquet directly. Returns (date_str, status, n_trades)."""
    date_str = date.strftime("%Y-%m-%d")
    outdir = data_dir / f"{currency}_{kind}"
    outdir.mkdir(parents=True, exist_ok=True)
    outfile = outdir / f"{date_str}.parquet"

    if outfile.exists() and outfile.stat().st_size > 1024:
        return (date_str, "SKIP", -1)

    start_ms = int(date.timestamp() * 1000)
    end_ms = int((date + timedelta(days=1)).timestamp() * 1000)

    session = make_session()
    all_trades: list[dict] = []
    has_more = True
    # Use start_timestamp pagination: when has_more, advance start_ms to last
    # trade's timestamp + 1ms. (start_seq is per-instrument so doesn't work
    # across multi-instrument days.)
    last_seq = None
    pages = 0
    cur_start = start_ms

    while has_more:
        trades, has_more, _ = fetch_page(
            session, currency, kind, cur_start, end_ms,
        )
        if not trades:
            break
        all_trades.extend(trades)
        pages += 1
        # Advance window: next page starts at last trade's timestamp + 1ms.
        last_ts = max(t.get("timestamp", 0) for t in trades)
        if last_ts <= cur_start:
            # Defensive: no progress → break to avoid infinite loop.
            break
        cur_start = last_ts + 1
        time.sleep(delay)
        if pages % 20 == 0:
            logger.info("  %s %s %s: %d pages, %d trades", currency, kind, date_str, pages, len(all_trades))

    if not all_trades:
        return (date_str, "EMPTY", 0)

    # De-dup by trade_id (safety belt — page boundaries can occasionally overlap)
    seen = set()
    deduped = []
    for t in all_trades:
        tid = t.get("trade_id")
        if tid not in seen:
            seen.add(tid)
            deduped.append(t)

    # Write parquet directly via Polars
    df = pl.DataFrame(deduped)
    df = df.sort("timestamp")
    df.write_parquet(outfile, compression="zstd")
    return (date_str, "OK", len(deduped))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/deribit/trades")
    parser.add_argument("--start", required=True, help="ISO date e.g. 2026-05-06")
    parser.add_argument("--end", required=True, help="ISO date EXCLUSIVE e.g. 2026-05-13")
    parser.add_argument("--currencies", default="BTC")
    parser.add_argument("--kinds", default="option,future")
    parser.add_argument("--workers", type=int, default=7)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    args = parser.parse_args()

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
    currencies = [c.strip() for c in args.currencies.split(",") if c.strip()]
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Build the full task matrix
    tasks: list[tuple[str, str, datetime]] = []
    current = start
    while current < end:
        for currency in currencies:
            for kind in kinds:
                tasks.append((currency, kind, current))
        current += timedelta(days=1)

    logger.info("=" * 70)
    logger.info("Parallel Deribit downloader")
    logger.info("  Currencies: %s", currencies)
    logger.info("  Kinds: %s", kinds)
    logger.info("  Period: %s -> %s (%d days)", args.start, args.end, (end - start).days)
    logger.info("  Tasks: %d (currency x kind x day)", len(tasks))
    logger.info("  Workers: %d", args.workers)
    logger.info("  PAGE_SIZE: %d", PAGE_SIZE)
    logger.info("  delay/req: %.3fs (per worker)", args.delay)
    logger.info("=" * 70)

    t0 = time.time()
    grand_total = 0
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {
            ex.submit(download_day, c, k, d, data_dir, args.delay): (c, k, d)
            for (c, k, d) in tasks
        }
        for fut in as_completed(futs):
            c, k, d = futs[fut]
            try:
                date_str, status, n = fut.result()
            except Exception as e:
                logger.error("%s %s %s: %s", c, k, d.strftime("%Y-%m-%d"), e)
                continue
            elapsed = time.time() - t0
            logger.info(
                "[%6.0fs] %s %s %s: %s (%d trades)",
                elapsed, c, k, date_str, status, n if n >= 0 else 0,
            )
            if n > 0:
                grand_total += n

    elapsed = time.time() - t0
    logger.info("=" * 70)
    logger.info("DONE in %.0fs (%.1f min): %d trades total", elapsed, elapsed / 60, grand_total)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
