"""Download missing Binance futures 1m kline data.

Finds all gaps in existing data and downloads them from Binance's
public data archive.
"""

import os
import sys
import time
import gzip
import zipfile
import io
from datetime import datetime
from pathlib import Path

import requests

DATA_DIR = Path("data/binance-1m")
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"

SYMBOLS = [
    "BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "DOGEUSDT", "HYPEUSDT",
    "BNBUSDT", "1000PEPEUSDT", "ZECUSDT", "SUIUSDT", "ADAUSDT",
    "AVAXUSDT", "TAOUSDT", "LINKUSDT", "BCHUSDT", "NEARUSDT", "FILUSDT",
    "DOTUSDT", "LTCUSDT", "ENAUSDT", "AAVEUSDT", "WLDUSDT", "UNIUSDT",
    "TRXUSDT", "FTMUSDT", "FETUSDT", "APTUSDT", "HBARUSDT", "XLMUSDT",
    "SXPUSDT", "BNXUSDT", "ENJUSDT", "ANKRUSDT",
]


def get_existing_months(symbol):
    sym_dir = DATA_DIR / symbol
    if not sym_dir.exists():
        return set()
    months = set()
    for f in sym_dir.glob("*.csv"):
        parts = f.stem.split("-")
        if len(parts) >= 4:
            months.add(f"{parts[2]}-{parts[3]}")
    return months


def get_expected_months(symbol):
    """Generate expected months from first existing to 2026-02."""
    existing = get_existing_months(symbol)
    if not existing:
        return []
    first = min(existing)
    months = []
    for year in range(int(first[:4]), 2027):
        for month in range(1, 13):
            m = f"{year}-{month:02d}"
            if m >= first and m <= "2026-02":
                months.append(m)
    return months


def download_month(symbol, year_month):
    """Download one month of 1m kline data from Binance."""
    year, month = year_month.split("-")
    url = f"{BASE_URL}/{symbol}/1m/{symbol}-1m-{year}-{month}.zip"

    sym_dir = DATA_DIR / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)
    out_path = sym_dir / f"{symbol}-1m-{year}-{month}.csv"

    if out_path.exists():
        return True, "exists"

    try:
        r = requests.get(url, timeout=30)
        if r.status_code == 404:
            return False, "not_found"
        r.raise_for_status()

        # Extract CSV from zip
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            names = zf.namelist()
            csv_name = [n for n in names if n.endswith(".csv")]
            if not csv_name:
                return False, "no_csv_in_zip"
            data = zf.read(csv_name[0])
            out_path.write_bytes(data)
            return True, "downloaded"
    except Exception as e:
        return False, str(e)[:50]


def main():
    total_missing = 0
    total_downloaded = 0
    total_failed = 0

    for symbol in SYMBOLS:
        existing = get_existing_months(symbol)
        expected = get_expected_months(symbol)
        if not expected:
            # New symbol — try from 2020-01
            expected = [f"{y}-{m:02d}" for y in range(2020, 2027) for m in range(1, 13)
                        if f"{y}-{m:02d}" <= "2026-02"]

        missing = [m for m in expected if m not in existing]
        if not missing:
            continue

        print(f"{symbol}: {len(missing)} missing months", flush=True)
        for ym in missing:
            ok, status = download_month(symbol, ym)
            if ok:
                total_downloaded += 1
                print(f"  {ym}: {status}")
            else:
                total_failed += 1
                if status != "not_found":
                    print(f"  {ym}: FAILED ({status})")
            total_missing += 1
            time.sleep(0.2)  # rate limit

    print(f"\nDone: {total_downloaded} downloaded, {total_failed} failed, {total_missing} total gaps")


if __name__ == "__main__":
    main()
