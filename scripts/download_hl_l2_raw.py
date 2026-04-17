#!/usr/bin/env python
"""Download raw L2 order book snapshots from Hyperliquid S3 archive.

Saves full 10-level depth LZ4 files to data/hl-bbo-raw/{COIN}/{YYYYMMDD}/{HOUR}/l2Book/{COIN}.lz4
for use by the OFI pipeline.

S3 bucket: hyperliquid-archive (requester-pays)
Path:      market_data/{YYYYMMDD}/{HOUR}/l2Book/{COIN}.lz4

Usage:
    PYTHONPATH=. python scripts/download_hl_l2_raw.py --coins BTC
    PYTHONPATH=. python scripts/download_hl_l2_raw.py --coins BTC,ETH,SOL
"""

import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

import boto3
from botocore.config import Config as BotoConfig

BUCKET = "hyperliquid-archive"
RAW_DIR = Path("data/hl-bbo-raw")
REGION = "eu-west-1"


def list_available_dates(s3_client) -> list[str]:
    """List all available dates in the S3 archive."""
    paginator = s3_client.get_paginator("list_objects_v2")
    dates = set()
    for page in paginator.paginate(
        Bucket=BUCKET, Prefix="market_data/", Delimiter="/",
        RequestPayer="requester",
    ):
        for p in page.get("CommonPrefixes", []):
            # Prefix looks like "market_data/20230520/"
            date_str = p["Prefix"].split("/")[1]
            if len(date_str) == 8 and date_str.isdigit():
                dates.add(date_str)
    return sorted(dates)


def get_local_dates(coin: str) -> set[str]:
    """Check which dates already have at least 1 hour file locally."""
    coin_dir = RAW_DIR / coin
    if not coin_dir.exists():
        return set()
    existing = set()
    for d in coin_dir.iterdir():
        if d.is_dir() and len(d.name) == 8 and d.name.isdigit():
            # Check if any hour directory has the LZ4 file
            for h in d.iterdir():
                if h.is_dir():
                    lz4_file = h / "l2Book" / f"{coin}.lz4"
                    if lz4_file.exists():
                        existing.add(d.name)
                        break
    return existing


def download_date(s3_client, coin: str, date_str: str) -> int:
    """Download all 24 hour files for one date. Returns count of files saved."""
    saved = 0
    for hour in range(24):
        key = f"market_data/{date_str}/{hour}/l2Book/{coin}.lz4"
        local_dir = RAW_DIR / coin / date_str / str(hour) / "l2Book"
        local_path = local_dir / f"{coin}.lz4"

        if local_path.exists():
            saved += 1
            continue

        try:
            resp = s3_client.get_object(
                Bucket=BUCKET, Key=key, RequestPayer="requester"
            )
            data = resp["Body"].read()
            if len(data) < 10:
                continue

            local_dir.mkdir(parents=True, exist_ok=True)
            local_path.write_bytes(data)
            saved += 1
        except s3_client.exceptions.NoSuchKey:
            continue
        except Exception:
            continue
    return saved


def main():
    parser = argparse.ArgumentParser(description="Download HL L2 raw LZ4 files")
    parser.add_argument("--coins", type=str, default="BTC",
                        help="Comma-separated coins (default: BTC)")
    parser.add_argument("--workers", type=int, default=20,
                        help="Parallel date workers (default: 20)")
    parser.add_argument("--start", type=str, default="",
                        help="Start date YYYYMMDD (default: all)")
    parser.add_argument("--end", type=str, default="",
                        help="End date YYYYMMDD (default: all)")
    args = parser.parse_args()

    coins = [c.strip() for c in args.coins.split(",")]

    # boto3 client with retry config
    boto_cfg = BotoConfig(
        region_name=REGION,
        retries={"max_attempts": 3, "mode": "adaptive"},
        max_pool_connections=args.workers + 5,
    )
    s3 = boto3.client("s3", config=boto_cfg)

    print("=" * 70)
    print("  HL L2 RAW DOWNLOAD")
    print("=" * 70)

    # List all available dates
    print("  Listing S3 dates...", flush=True)
    all_dates = list_available_dates(s3)
    print(f"  Available: {len(all_dates)} dates ({all_dates[0]} to {all_dates[-1]})")

    # Filter by start/end
    if args.start:
        all_dates = [d for d in all_dates if d >= args.start]
    if args.end:
        all_dates = [d for d in all_dates if d <= args.end]

    for coin in coins:
        print(f"\n  --- {coin} ---")
        local_dates = get_local_dates(coin)
        needed = [d for d in all_dates if d not in local_dates]
        print(f"  Local: {len(local_dates)}, Need: {len(needed)}", flush=True)

        if not needed:
            print(f"  {coin}: up to date")
            continue

        t0 = time.time()
        n_done = 0
        n_ok = 0
        total_files = 0

        # Each thread gets its own s3 client for thread safety
        def _worker(date_str):
            thread_s3 = boto3.client("s3", config=boto_cfg)
            return download_date(thread_s3, coin, date_str)

        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futs = {pool.submit(_worker, d): d for d in needed}
            for fut in as_completed(futs):
                n_done += 1
                try:
                    count = fut.result()
                    if count > 0:
                        n_ok += 1
                        total_files += count
                except Exception:
                    pass

                if n_done % 50 == 0 or n_done == len(needed):
                    elapsed = time.time() - t0
                    rate = n_done / elapsed * 60 if elapsed > 0 else 0
                    eta = (len(needed) - n_done) / (n_done / elapsed) if n_done > 0 else 0
                    print(f"    {n_done}/{len(needed)} dates "
                          f"({n_ok} ok, {total_files} files, "
                          f"{rate:.0f} dates/min, ETA {eta:.0f}s)",
                          flush=True)

        elapsed = time.time() - t0
        print(f"  {coin}: {n_ok} dates, {total_files} files in {elapsed:.0f}s")

    # Summary
    print(f"\n  --- Summary ---")
    for coin in coins:
        coin_dir = RAW_DIR / coin
        if coin_dir.exists():
            n_dates = sum(1 for d in coin_dir.iterdir()
                         if d.is_dir() and len(d.name) == 8)
            print(f"  {coin}: {n_dates} dates")

    print("=" * 70)


if __name__ == "__main__":
    main()
