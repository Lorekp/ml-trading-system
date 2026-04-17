"""Batch-cache HL asset_ctxs — parallel, memory-safe, disk-cached.

12-core parallel: splits 1009 LZ4 files into chunks, each worker
extracts wanted coins from its chunk. Main process merges and saves
to parquet. Skips coins already cached.

~20x faster than sequential per-coin approach.
"""

import sys, os, time, csv, io
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)

import lz4.frame
import pandas as pd
import numpy as np

HL_DIR = Path("data/hl-asset-ctxs/hl-asset-ctxs")
CACHE_DIR = Path("data/hl-asset-ctxs/cache_15m")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

COINS_WANTED = [
    "SOL", "BTC", "LINK", "XRP", "ETH", "AVAX", "DOGE", "SUI",
    "APT", "FIL", "INJ", "NEAR", "ARB", "OP", "WLD", "FET",
    "TIA", "SEI", "JUP", "PENDLE",
]

NUMERIC_COLS = [
    "funding", "open_interest", "premium",
    "oracle_px", "mark_px", "mid_px",
    "impact_bid_px", "impact_ask_px",
]

KEEP_COLS = ["time"] + NUMERIC_COLS


def process_chunk(file_paths: list[str], coins_set: set[str]) -> dict[str, list[list]]:
    """Worker: process a chunk of LZ4 files, return {coin: [[row], ...]}."""
    result: dict[str, list] = {c: [] for c in coins_set}
    for fpath in file_paths:
        try:
            with open(fpath, "rb") as fh:
                raw = lz4.frame.decompress(fh.read()).decode("utf-8")
            reader = csv.DictReader(io.StringIO(raw))
            for row in reader:
                coin = row.get("coin")
                if coin in coins_set:
                    result[coin].append([row.get(c, "") for c in KEEP_COLS])
        except Exception:
            continue
    return result


def main():
    t0 = time.time()
    files = sorted(str(f) for f in HL_DIR.iterdir() if f.name.endswith(".lz4"))

    # Skip already cached coins
    coins_todo = []
    for coin in COINS_WANTED:
        cache_path = CACHE_DIR / f"{coin}.parquet"
        if cache_path.exists() and cache_path.stat().st_size > 1000:
            print(f"  {coin}: CACHED ({cache_path.stat().st_size/1e6:.1f}MB)")
        else:
            coins_todo.append(coin)

    if not coins_todo:
        print("All coins cached.")
        return

    coins_set = set(coins_todo)
    n_workers = min(10, os.cpu_count() or 4)

    # Split files into chunks for parallel processing
    chunk_size = max(1, len(files) // n_workers)
    chunks = [files[i:i + chunk_size] for i in range(0, len(files), chunk_size)]

    print(f"\n{len(files)} files, {len(coins_todo)} coins, {n_workers} workers, "
          f"{len(chunks)} chunks of ~{chunk_size} files")

    # Parallel decompress + extract
    all_rows: dict[str, list] = {c: [] for c in coins_todo}
    done = 0

    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(process_chunk, chunk, coins_set): i
            for i, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            chunk_result = future.result()
            for coin, rows in chunk_result.items():
                all_rows[coin].extend(rows)
            done += 1
            print(f"  chunk {done}/{len(chunks)} done ({time.time()-t0:.0f}s)", flush=True)

    print(f"\nAll chunks done in {time.time()-t0:.0f}s. Converting to parquet...")

    # Convert to parquet per coin
    for coin in coins_todo:
        rows = all_rows[coin]
        if not rows:
            print(f"  {coin}: no data")
            continue

        df = pd.DataFrame(rows, columns=KEEP_COLS)
        df["time"] = pd.to_datetime(df["time"], utc=True)
        df = df.set_index("time").sort_index()
        df = df[~df.index.duplicated(keep="first")]
        for col in NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        cache_path = CACHE_DIR / f"{coin}.parquet"
        df.to_parquet(cache_path)
        sz = cache_path.stat().st_size / 1e6
        print(f"  {coin}: {len(df):,} rows -> {sz:.1f}MB")

        # Free memory
        del all_rows[coin]

    print(f"\nAll done in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
