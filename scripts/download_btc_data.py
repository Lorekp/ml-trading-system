#!/usr/bin/env python
"""Download aggTrades from Binance Vision for multiple symbols.

Resume-safe (skips existing files). Converts ZIPs to Parquet inline via Polars.
Handles both header/no-header CSV formats.

Usage:
    python scripts/download_btc_data.py                          # default 10 symbols
    python scripts/download_btc_data.py --symbols BTCUSDT,ETHUSDT  # custom
"""

import argparse
import io
import logging
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import requests
from requests.adapters import HTTPAdapter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("download")

DEFAULT_SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "BNBUSDT", "XRPUSDT",
    "DOGEUSDT", "AVAXUSDT", "LINKUSDT", "ADAUSDT", "ARBUSDT",
]
START = "2021-01"
END = "2026-03"
AGG_BASE = "https://data.binance.vision/data/futures/um/monthly/aggTrades"
AGG_PARQUET_BASE = Path("data/binance-aggTrades-parquet")
AGG_ZIP_BASE = Path("data/binance-aggTrades")

AGG_COL_NAMES = [
    "agg_trade_id", "price", "quantity", "first_trade_id",
    "last_trade_id", "transact_time", "is_buyer_maker",
]


def get_session():
    s = requests.Session()
    s.mount("https://", HTTPAdapter(pool_connections=10, pool_maxsize=10, max_retries=3))
    return s


def month_range(start: str, end: str) -> list[str]:
    months = []
    y, m = int(start[:4]), int(start[5:7])
    ey, em = int(end[:4]), int(end[5:7])
    while (y, m) <= (ey, em):
        months.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return months


def download_and_convert(symbol: str, month: str) -> tuple[str, str, int]:
    """Download + convert one month of aggTrades. Returns (symbol, month, size_bytes)."""
    import polars as pl

    out_dir = AGG_PARQUET_BASE / symbol
    zip_dir = AGG_ZIP_BASE / symbol
    out_path = out_dir / f"{month}.parquet"
    if out_path.exists() and out_path.stat().st_size > 100:
        return symbol, month, -1

    url = f"{AGG_BASE}/{symbol}/{symbol}-aggTrades-{month}.zip"
    zip_path = zip_dir / f"{symbol}-aggTrades-{month}.zip"
    session = get_session()

    for attempt in range(3):
        try:
            zip_dir.mkdir(parents=True, exist_ok=True)
            tmp_zip = zip_path.with_suffix(".tmp")
            with session.get(url, timeout=600, stream=True) as resp:
                if resp.status_code in (404, 451):
                    return symbol, month, 0
                resp.raise_for_status()
                with open(tmp_zip, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=8 * 1024 * 1024):
                        f.write(chunk)
            tmp_zip.replace(zip_path)

            zf = zipfile.ZipFile(zip_path)
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                return symbol, month, 0

            csv_data = zf.read(csv_names[0])
            zf.close()

            first_line = csv_data[:200].split(b"\n")[0].decode("utf-8", errors="ignore")
            has_header = "agg_trade_id" in first_line

            dtypes = {
                "agg_trade_id": pl.Int64, "price": pl.Float64,
                "quantity": pl.Float64, "first_trade_id": pl.Int64,
                "last_trade_id": pl.Int64, "transact_time": pl.Int64,
                "is_buyer_maker": pl.Boolean,
            }

            if has_header:
                df = pl.read_csv(io.BytesIO(csv_data), has_header=True, schema_overrides=dtypes)
            else:
                df = pl.read_csv(io.BytesIO(csv_data), has_header=False,
                                 new_columns=AGG_COL_NAMES, schema_overrides=dtypes)
            del csv_data

            df = df.with_columns((pl.col("price") * pl.col("quantity")).alias("dollar_volume"))

            out_dir.mkdir(parents=True, exist_ok=True)
            tmp = out_path.with_suffix(".tmp")
            df.write_parquet(tmp, compression="zstd", compression_level=3)
            tmp.replace(out_path)
            del df

            return symbol, month, int(out_path.stat().st_size)

        except (requests.exceptions.ConnectionError, requests.exceptions.Timeout):
            time.sleep(5 * (attempt + 1))
        except zipfile.BadZipFile:
            time.sleep(2)
        except Exception as e:
            logger.warning("  %s %s attempt %d: %s", symbol, month, attempt + 1, e)
            time.sleep(5)

    return symbol, month, 0


def main():
    parser = argparse.ArgumentParser(description="Download aggTrades from Binance Vision")
    parser.add_argument("--symbols", default=",".join(DEFAULT_SYMBOLS),
                        help="Comma-separated symbols")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    symbols = [s.strip() for s in args.symbols.split(",")]
    months = month_range(START, END)

    # Build work queue
    work = []
    cached = 0
    for sym in symbols:
        out_dir = AGG_PARQUET_BASE / sym
        for m in months:
            pq = out_dir / f"{m}.parquet"
            if pq.exists() and pq.stat().st_size > 100:
                cached += 1
            else:
                work.append((sym, m))

    print("=" * 60)
    print(f"  AGGTRADES DOWNLOAD: {len(symbols)} symbols, {START} to {END}")
    print(f"  Symbols: {', '.join(symbols)}")
    print(f"  Total: {len(symbols)*len(months)}, cached: {cached}, to download: {len(work)}")
    print(f"  Workers: {args.workers}")
    print("=" * 60)

    if not work:
        print("  All cached!")
        return

    t0 = time.time()
    n_new = 0
    n_404 = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(download_and_convert, s, m): (s, m) for s, m in work}
        done = 0
        for fut in as_completed(futs):
            sym, month = futs[fut]
            done += 1
            try:
                _, _, size = fut.result()
            except Exception as e:
                logger.warning("  %s %s: %s", sym, month, e)
                size = 0

            if size > 0:
                n_new += 1
                elapsed = time.time() - t0
                rate = done / elapsed * 60 if elapsed > 0 else 0
                eta = (len(work) - done) / rate if rate > 0 else 0
                if done % 5 == 0 or done == len(work):
                    print(f"  [{done}/{len(work)}] {sym} {month}: "
                          f"{size/1e6:.0f}MB  ({rate:.1f}/min, ETA {eta:.0f}min)",
                          flush=True)
            elif size == 0:
                n_404 += 1

    elapsed = time.time() - t0
    print(f"\n  Done: {n_new} downloaded, {n_404} not available ({elapsed:.0f}s)")
    for sym in symbols:
        n = len(list((AGG_PARQUET_BASE / sym).glob("*.parquet")))
        print(f"    {sym}: {n} months")


if __name__ == "__main__":
    main()
