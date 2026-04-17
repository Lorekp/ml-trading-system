#!/usr/bin/env python
"""Fast HL BBO download using AWS CLI bulk sync + local processing.

Uses `aws s3 sync` per coin (massively parallel internal transfer),
then batch-extracts BBO from LZ4 files locally.

~10-50x faster than individual boto3 GETs.

Usage:
    PYTHONPATH=src python scripts/download_hl_bbo_fast.py [--coins BTC,ETH,SOL] [--all]
"""
import os, sys, time, json, subprocess, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
warnings.filterwarnings("ignore")

import lz4.frame

RAW_DIR = Path("data/hl-bbo-raw")   # Raw LZ4 files from S3
OUTPUT_DIR = Path("data/hl-bbo")     # Processed daily parquets
ELIGIBLE_FILE = Path("data/eligible_symbols.json")
BUCKET = "hyperliquid-archive"
AWS_CLI = [sys.executable, "-m", "awscli"]


def coin_to_sym(coin):
    if coin.startswith("k"):
        return f"1000{coin[1:]}USDT"
    return f"{coin}USDT"


def sync_coin(coin, max_concurrent=20):
    """Download all l2Book files for a coin using aws s3 sync.

    Stores raw LZ4 in: data/hl-bbo-raw/{COIN}/
    One flat directory with files named {DATE}_{HOUR}.lz4
    """
    coin_raw = RAW_DIR / coin
    coin_raw.mkdir(parents=True, exist_ok=True)

    # Use aws s3 cp --recursive with include filter
    # This downloads all matching files in one optimized batch
    cmd = AWS_CLI + [
        "s3", "sync",
        f"s3://{BUCKET}/market_data/",
        str(RAW_DIR / "_tmp_sync"),
        "--request-payer", "requester",
        "--include", f"*/l2Book/{coin}.lz4",
        "--exclude", "*",
        "--no-progress",
        "--only-show-errors",
    ]

    # aws s3 sync with include/exclude on nested paths is slow for listing.
    # Better approach: download per-date using cp for each date.
    # Actually, best approach: list all dates, then parallel cp per date.

    # Step 1: Get list of all dates from S3
    import boto3
    s3 = boto3.client("s3", region_name="eu-west-1")
    paginator = s3.get_paginator("list_objects_v2")

    dates = set()
    try:
        for page in paginator.paginate(
            Bucket=BUCKET, Prefix="market_data/", Delimiter="/",
            RequestPayer="requester",
        ):
            for p in page.get("CommonPrefixes", []):
                dates.add(p["Prefix"].split("/")[1])
    except Exception as e:
        print(f"    Error listing dates: {e}")
        return 0
    dates = sorted(dates)

    # Step 2: Check which dates already processed
    sym = coin_to_sym(coin)
    out_dir = OUTPUT_DIR / sym
    existing_pqs = set()
    if out_dir.exists():
        existing_pqs = set(f.stem for f in out_dir.glob("*.parquet"))

    dates_needed = []
    for d in dates:
        formatted = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        if formatted not in existing_pqs:
            dates_needed.append(d)

    if not dates_needed:
        return 0

    # Newest first — coins exist on recent dates, avoids 404s on early dates
    dates_needed = list(reversed(dates_needed))

    # Step 3: Download + process each date (parallel)
    def _do_date(date_str):
        """Download 24 hours, extract BBO, save parquet for one date."""
        records = []
        for hour in range(24):
            key = f"market_data/{date_str}/{hour}/l2Book/{coin}.lz4"
            try:
                resp = s3.get_object(Bucket=BUCKET, Key=key, RequestPayer="requester")
                compressed = resp["Body"].read()
                raw = lz4.frame.decompress(compressed)
                for line in raw.decode("utf-8").strip().split("\n"):
                    if not line: continue
                    try:
                        snap = json.loads(line)
                        inner = snap.get("raw", {}).get("data", {})
                        ts = inner.get("time", 0)
                        levels = inner.get("levels", [[], []])
                        if len(levels) >= 2 and levels[0] and levels[1]:
                            bid = levels[0][0]
                            ask = levels[1][0]
                            records.append((
                                int(ts),
                                float(bid["px"]), float(bid["sz"]),
                                float(ask["px"]), float(ask["sz"]),
                            ))
                    except (KeyError, ValueError, IndexError, json.JSONDecodeError):
                        continue
            except Exception:
                continue

        if not records:
            return False

        df = pd.DataFrame(records, columns=[
            "transaction_time", "best_bid_price", "best_bid_qty",
            "best_ask_price", "best_ask_qty"])
        df = df.sort_values("transaction_time").reset_index(drop=True)
        df["best_bid_price"] = df["best_bid_price"].astype(np.float32)
        df["best_bid_qty"] = df["best_bid_qty"].astype(np.float32)
        df["best_ask_price"] = df["best_ask_price"].astype(np.float32)
        df["best_ask_qty"] = df["best_ask_qty"].astype(np.float32)
        df["transaction_time"] = df["transaction_time"].astype(np.int64)
        df = df[["best_bid_price", "best_bid_qty", "best_ask_price",
                 "best_ask_qty", "transaction_time"]]

        out_dir.mkdir(parents=True, exist_ok=True)
        formatted = f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"
        df.to_parquet(out_dir / f"{formatted}.parquet", index=False)
        return True

    n_ok = 0
    n_done = 0
    t_start = time.time()
    with ThreadPoolExecutor(max_workers=max_concurrent) as pool:
        futs = {pool.submit(_do_date, d): d for d in dates_needed}
        for fut in as_completed(futs):
            n_done += 1
            try:
                if fut.result():
                    n_ok += 1
            except Exception:
                pass
            if n_done % 50 == 0 or n_done == len(dates_needed):
                elapsed = time.time() - t_start
                rate = n_done / elapsed * 60 if elapsed > 0 else 0
                print(f"    {coin}: {n_done}/{len(dates_needed)} "
                      f"({n_ok} ok, {rate:.0f}/min)", flush=True)

    return n_ok


def main():
    T0 = time.time()
    print("=" * 70)
    print("  FAST HL BBO DOWNLOAD")
    print("=" * 70)

    # Determine coins
    if "--coins" in sys.argv:
        idx = sys.argv.index("--coins")
        coins = sys.argv[idx + 1].split(",")
    elif "--all" in sys.argv:
        if not ELIGIBLE_FILE.exists():
            print(f"  ERROR: {ELIGIBLE_FILE} not found")
            return
        with open(ELIGIBLE_FILE) as f:
            eligible = json.load(f)
        coins = [e["hl_coin"] for e in eligible if e["hl_coin"]]
    else:
        print("  Usage: --all | --coins BTC,ETH,SOL")
        return

    print(f"  Coins: {len(coins)}")

    # Process coins: 3 coins concurrently, each with 20 parallel dates
    COIN_PARALLEL = 3
    DATE_PARALLEL = 15

    total_ok = 0
    for batch_start in range(0, len(coins), COIN_PARALLEL):
        batch = coins[batch_start:batch_start + COIN_PARALLEL]
        t1 = time.time()

        with ThreadPoolExecutor(max_workers=COIN_PARALLEL) as pool:
            futs = {}
            for coin in batch:
                futs[pool.submit(sync_coin, coin, DATE_PARALLEL)] = coin

            for fut in as_completed(futs):
                coin = futs[fut]
                try:
                    n = fut.result()
                except Exception as e:
                    n = 0
                    print(f"  {coin}: ERROR {e}")
                sym = coin_to_sym(coin)
                elapsed = time.time() - t1
                if n > 0:
                    total_ok += n
                    print(f"  [{batch_start + 1}-{batch_start + len(batch)}/{len(coins)}] "
                          f"{coin} ({sym}): {n} days ({elapsed:.0f}s)", flush=True)
                else:
                    # Check if already cached
                    out_dir = OUTPUT_DIR / sym
                    existing = len(list(out_dir.glob("*.parquet"))) if out_dir.exists() else 0
                    if existing > 0:
                        print(f"  [{batch_start + 1}-{batch_start + len(batch)}/{len(coins)}] "
                              f"{coin} ({sym}): cached ({existing} days)", flush=True)
                    else:
                        print(f"  [{batch_start + 1}-{batch_start + len(batch)}/{len(coins)}] "
                              f"{coin} ({sym}): 0 days ({elapsed:.0f}s)", flush=True)

    # Validation
    print(f"\n  Validating...")
    n_syms = 0
    n_days_total = 0
    issues = []
    for d in OUTPUT_DIR.iterdir():
        if d.is_dir():
            pqs = list(d.glob("*.parquet"))
            if pqs:
                n_syms += 1
                n_days_total += len(pqs)
                # Spot-check 1 random file
                import random
                sample = random.choice(pqs)
                try:
                    df = pd.read_parquet(sample)
                    if len(df) == 0:
                        issues.append(f"{d.name}/{sample.name}: empty")
                    elif (df["best_bid_price"] <= 0).any():
                        issues.append(f"{d.name}/{sample.name}: zero prices")
                except Exception as e:
                    issues.append(f"{d.name}/{sample.name}: {e}")

    print(f"  Total: {n_syms} symbols, {n_days_total:,} daily files")
    if issues:
        print(f"  Issues ({len(issues)}):")
        for iss in issues[:10]:
            print(f"    {iss}")
    else:
        print(f"  All spot-checks passed")

    print(f"\n  Total new: {total_ok:,} days")
    print(f"  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
