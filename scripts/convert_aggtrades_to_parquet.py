#!/usr/bin/env python
"""Convert Binance aggTrades ZIPs to Parquet using Polars.

Processes all assets in parallel. Each monthly ZIP → one Parquet file per month.
Memory-efficient: streams ZIP → CSV → Polars → Parquet, one file at a time.

Input:  data/binance-aggTrades/{SYMBOL}/{SYMBOL}-aggTrades-{YYYY-MM}.zip
Output: data/binance-aggTrades-parquet/{SYMBOL}/{YYYY-MM}.parquet

Usage:
    python scripts/convert_aggtrades_to_parquet.py
    python scripts/convert_aggtrades_to_parquet.py --workers 8
"""

import argparse
import io
import logging
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import polars as pl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("convert")

INPUT_DIR = Path("data/binance-aggTrades")
OUTPUT_DIR = Path("data/binance-aggTrades-parquet")

# Binance aggTrades CSV columns (no header in the CSV)
DTYPES = {
    "agg_trade_id": pl.Int64,
    "price": pl.Float64,
    "quantity": pl.Float64,
    "first_trade_id": pl.Int64,
    "last_trade_id": pl.Int64,
    "transact_time": pl.Int64,
    "is_buyer_maker": pl.Boolean,
}


def convert_one(zip_path: Path) -> tuple[str, int, float]:
    """Convert one ZIP to Parquet. Returns (filename, n_rows, size_mb)."""
    symbol = zip_path.parent.name
    # Extract YYYY-MM from filename like BTCUSDT-aggTrades-2023-06.zip
    parts = zip_path.stem.split("-aggTrades-")
    if len(parts) != 2:
        return zip_path.name, 0, 0.0
    month_str = parts[1]  # "2023-06"

    out_dir = OUTPUT_DIR / symbol
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{month_str}.parquet"

    if out_path.exists() and out_path.stat().st_size > 100:
        return zip_path.name, -1, 0.0  # skip

    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                return zip_path.name, 0, 0.0

            csv_data = zf.read(csv_names[0])

        df = pl.read_csv(
            io.BytesIO(csv_data),
            has_header=True,
            schema_overrides=DTYPES,
        )

        # Add dollar_volume column for dollar bars
        df = df.with_columns(
            (pl.col("price") * pl.col("quantity")).alias("dollar_volume")
        )

        # Write parquet with compression
        df.write_parquet(out_path, compression="zstd", compression_level=3)

        size_mb = out_path.stat().st_size / 1e6
        return zip_path.name, len(df), size_mb

    except zipfile.BadZipFile:
        logger.warning("Corrupt ZIP: %s", zip_path)
        return zip_path.name, 0, 0.0
    except Exception as e:
        logger.warning("Failed %s: %s", zip_path.name, e)
        return zip_path.name, 0, 0.0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=6,
                        help="Parallel workers (default: 6)")
    args = parser.parse_args()

    zips = sorted(INPUT_DIR.rglob("*.zip"))
    if not zips:
        logger.error("No ZIP files found in %s", INPUT_DIR)
        return

    # Filter out already converted
    to_convert = []
    skipped = 0
    for z in zips:
        symbol = z.parent.name
        parts = z.stem.split("-aggTrades-")
        if len(parts) == 2:
            month_str = parts[1]
            out_path = OUTPUT_DIR / symbol / f"{month_str}.parquet"
            if out_path.exists() and out_path.stat().st_size > 100:
                skipped += 1
                continue
        to_convert.append(z)

    print(f"{'=' * 70}")
    print(f"  Binance aggTrades -> Parquet (Polars + zstd)")
    print(f"  Total ZIPs: {len(zips)}")
    print(f"  Already converted: {skipped}")
    print(f"  To convert: {len(to_convert)}")
    print(f"  Workers: {args.workers}")
    print(f"{'=' * 70}")

    if not to_convert:
        print("Nothing to do!")
        return

    t0 = time.time()
    done = 0
    total_rows = 0
    total_size = 0.0

    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(convert_one, z): z for z in to_convert}

        for fut in as_completed(futures):
            name, n_rows, size_mb = fut.result()
            done += 1

            if n_rows > 0:
                total_rows += n_rows
                total_size += size_mb
                if done % 20 == 0 or done == len(to_convert):
                    elapsed = time.time() - t0
                    rate = done / elapsed * 60
                    logger.info("[%d/%d] %s: %d rows (%.0f/min, %.1fGB total)",
                                done, len(to_convert), name, n_rows,
                                rate, total_size / 1000)
            elif n_rows == 0:
                pass  # failed or empty, already logged

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"  Done in {elapsed:.0f}s")
    print(f"  Converted: {done} files")
    print(f"  Total rows: {total_rows:,}")
    print(f"  Parquet size: {total_size / 1000:.1f} GB")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
