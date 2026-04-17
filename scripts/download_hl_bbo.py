#!/usr/bin/env python
"""Download BTC BBO from Hyperliquid S3 archive → bookTicker parquet format.

Extracts best bid/ask from L2 book snapshots. Output is format-identical to
Binance bookTicker: same columns, dtypes, and directory structure. Skips dates
that already have data (Binance or prior HL downloads).

Source: s3://hyperliquid-archive/market_data/{YYYYMMDD}/{HOUR}/l2Book/BTC.lz4
Output: data/bookTicker_parquet/BTCUSDT/{YYYY-MM-DD}.parquet

Usage:
    python scripts/download_hl_bbo.py                  # default 8 workers
    python scripts/download_hl_bbo.py --workers 16     # more parallelism
"""

import argparse
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
import lz4.frame
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("hl_bbo")

BUCKET = "hyperliquid-archive"
COIN = "BTC"
OUTPUT_DIR = Path("data/bookTicker_parquet/BTCUSDT")


def get_archive_dates(s3) -> list[str]:
    """List all YYYYMMDD dates in the archive."""
    paginator = s3.get_paginator("list_objects_v2")
    dates = []
    for page in paginator.paginate(
        Bucket=BUCKET, Prefix="market_data/", Delimiter="/",
        RequestPayer="requester",
    ):
        for p in page.get("CommonPrefixes", []):
            d = p["Prefix"].split("/")[1]
            if d.isdigit() and len(d) == 8:
                dates.append(d)
    dates.sort()
    return dates


def parse_hour(s3, date_str: str, hour: int) -> list[tuple]:
    """Download + parse one hour of L2 snapshots → list of BBO tuples."""
    key = f"market_data/{date_str}/{hour}/l2Book/{COIN}.lz4"
    try:
        resp = s3.get_object(Bucket=BUCKET, Key=key, RequestPayer="requester")
        raw = lz4.frame.decompress(resp["Body"].read()).decode("utf-8")
    except Exception:
        return []

    rows = []
    for line in raw.strip().split("\n"):
        if not line:
            continue
        try:
            rec = json.loads(line)
            d = rec["raw"]["data"]
            bids, asks = d["levels"][0], d["levels"][1]
            if not bids or not asks:
                continue
            rows.append((
                float(bids[0]["px"]),
                float(bids[0]["sz"]),
                float(asks[0]["px"]),
                float(asks[0]["sz"]),
                int(d["time"]),
            ))
        except (KeyError, IndexError, ValueError, json.JSONDecodeError):
            continue
    return rows


def download_day(date_str: str) -> tuple[str, int]:
    """Download all 24 hours for one day, save as bookTicker parquet.

    Returns (date_str, n_rows). n_rows=-1 if cached, 0 if empty/failed.
    """
    iso = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
    out_path = OUTPUT_DIR / f"{iso}.parquet"
    if out_path.exists() and out_path.stat().st_size > 100:
        return date_str, -1

    # Each thread gets its own S3 client
    s3 = boto3.client("s3")

    all_rows = []
    for hour in range(24):
        all_rows.extend(parse_hour(s3, date_str, hour))

    if not all_rows:
        return date_str, 0

    df = pd.DataFrame(all_rows, columns=[
        "best_bid_price", "best_bid_qty", "best_ask_price",
        "best_ask_qty", "transaction_time",
    ])
    df = df.sort_values("transaction_time").reset_index(drop=True)

    # Clean: drop crossed book, duplicates
    df = df[df["best_bid_price"] < df["best_ask_price"]]
    df = df.drop_duplicates(subset="transaction_time", keep="last")

    if df.empty:
        return date_str, 0

    # Match Binance bookTicker dtypes exactly
    df["best_bid_price"] = df["best_bid_price"].astype(np.float32)
    df["best_bid_qty"] = df["best_bid_qty"].astype(np.float32)
    df["best_ask_price"] = df["best_ask_price"].astype(np.float32)
    df["best_ask_qty"] = df["best_ask_qty"].astype(np.float32)
    df["transaction_time"] = df["transaction_time"].astype(np.int64)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(out_path)

    return date_str, len(df)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()

    s3 = boto3.client("s3")

    print("=" * 60)
    print("  HYPERLIQUID BTC BBO -> bookTicker parquet")
    print("=" * 60)

    print("  Scanning archive...")
    hl_dates = get_archive_dates(s3)
    print(f"  Archive: {hl_dates[0]} to {hl_dates[-1]} ({len(hl_dates)} days)")

    existing = set(f.stem for f in OUTPUT_DIR.glob("*.parquet")) if OUTPUT_DIR.exists() else set()
    to_do = [d for d in hl_dates if f"{d[:4]}-{d[4:6]}-{d[6:8]}" not in existing]

    print(f"  Existing: {len(existing)} files, to download: {len(to_do)}")
    print(f"  Workers: {args.workers}")
    print("=" * 60)

    if not to_do:
        print("  All cached!")
        return

    t0 = time.time()
    n_new = 0
    n_empty = 0

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(download_day, d): d for d in to_do}
        done = 0
        for fut in as_completed(futs):
            done += 1
            try:
                _, n_rows = fut.result()
            except Exception:
                n_rows = 0

            if n_rows > 0:
                n_new += 1
            elif n_rows == 0:
                n_empty += 1

            if done % 50 == 0 or done == len(to_do):
                elapsed = time.time() - t0
                rate = done / elapsed * 60 if elapsed > 0 else 0
                eta = (len(to_do) - done) / rate if rate > 0 else 0
                print(f"  {done:,}/{len(to_do):,} ({n_new:,} new, {n_empty} empty, "
                      f"{rate:.0f}/min, ETA {eta:.0f}min)", flush=True)

    elapsed = time.time() - t0
    total = len(list(OUTPUT_DIR.glob("*.parquet")))
    print(f"\n  Done: {n_new} days, {n_empty} empty ({elapsed:.0f}s)")
    print(f"  Total bookTicker files: {total}")

    # Spot-check: compare an HL day vs Binance day
    print("\n  Validation...")
    for sample_date in ["2023-06-01", "2024-06-01", "2025-01-01"]:
        p = OUTPUT_DIR / f"{sample_date}.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            spread = (df["best_ask_price"] - df["best_bid_price"]).median()
            print(f"    {sample_date}: {len(df):>8,d} rows, "
                  f"bid_mean={df['best_bid_price'].mean():>10.2f}, "
                  f"spread_median={spread:.2f}, "
                  f"dtypes OK={dict(df.dtypes) == {'best_bid_price': np.dtype('float32'), 'best_bid_qty': np.dtype('float32'), 'best_ask_price': np.dtype('float32'), 'best_ask_qty': np.dtype('float32'), 'transaction_time': np.dtype('int64')}}")


if __name__ == "__main__":
    main()
