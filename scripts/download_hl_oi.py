#!/usr/bin/env python
"""Download Hyperliquid asset_ctxs (OI, funding, premium) from S3.

Requester-pays bucket: s3://hyperliquid-archive/asset_ctxs/
Files: YYYYMMDD.csv.lz4 (~1-1.5MB each)
Date range available: 2023-05-20 to ~present

Usage:
    python scripts/download_hl_oi.py
    python scripts/download_hl_oi.py --start 2023-05-20 --end 2024-04-01
"""

import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path

import boto3

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

BUCKET = "hyperliquid-archive"
PREFIX = "asset_ctxs/"


def download_hl_asset_ctxs(
    start_date: str = "2023-05-20",
    end_date: str = "2024-04-01",
    output_dir: str = "data/hl-asset-ctxs/hl-asset-ctxs",
) -> int:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3")

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")

    downloaded = 0
    skipped = 0
    failed = 0
    current = start

    while current < end:
        date_str = current.strftime("%Y%m%d")
        key = f"{PREFIX}{date_str}.csv.lz4"
        local_path = out / f"{date_str}.csv.lz4"

        if local_path.exists():
            skipped += 1
            current += timedelta(days=1)
            continue

        try:
            s3.download_file(
                BUCKET, key, str(local_path),
                ExtraArgs={"RequestPayer": "requester"},
            )
            downloaded += 1
            if downloaded % 30 == 0:
                logger.info("  Downloaded %d files (up to %s)", downloaded, date_str)
        except Exception as e:
            if "404" in str(e) or "NoSuchKey" in str(e):
                pass  # Date not in archive (expected for May 1-19)
            else:
                logger.warning("  Failed %s: %s", date_str, e)
                failed += 1

        current += timedelta(days=1)

    logger.info("Done: %d downloaded, %d skipped (exist), %d failed", downloaded, skipped, failed)
    return downloaded


def main():
    parser = argparse.ArgumentParser(description="Download HL asset_ctxs from S3")
    parser.add_argument("--start", default="2023-05-20")
    parser.add_argument("--end", default="2024-04-01")
    parser.add_argument("--output-dir", default="data/hl-asset-ctxs/hl-asset-ctxs")
    args = parser.parse_args()

    download_hl_asset_ctxs(args.start, args.end, args.output_dir)


if __name__ == "__main__":
    main()
