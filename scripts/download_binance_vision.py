#!/usr/bin/env python
"""Download 1m OHLCV from Binance Vision (bulk zips, no API rate limits).

URL: https://data.binance.vision/data/futures/um/monthly/klines/{SYM}/1m/{SYM}-1m-{YYYY}-{MM}.zip

Much faster than API: one HTTP GET per month, pre-packaged CSV.

Usage:
    PYTHONPATH=src python scripts/download_binance_vision.py --all
    PYTHONPATH=src python scripts/download_binance_vision.py --symbols BTCUSDT,ETHUSDT
"""
import os, sys, time, json, io, zipfile, warnings
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
warnings.filterwarnings("ignore")

DATA_DIR = Path("data/binance-1m")
ELIGIBLE_FILE = Path("data/eligible_symbols.json")
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"
MAX_WORKERS = 10


def download_month(symbol, year, month):
    """Download one month of 1m data from Binance Vision.

    Returns (n_rows, error_str or None).
    """
    ym = f"{year}-{month:02d}"
    filename = f"{symbol}-1m-{ym}"
    out_dir = DATA_DIR / symbol
    csv_path = out_dir / f"{filename}.csv"

    if csv_path.exists() and csv_path.stat().st_size > 100:
        return -1, None  # already exists

    url = f"{BASE_URL}/{symbol}/1m/{filename}.zip"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return 0, None  # not available yet
        resp.raise_for_status()

        # Extract CSV from zip
        with zipfile.ZipFile(io.BytesIO(resp.content)) as zf:
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                return 0, "no CSV in zip"

            csv_data = zf.read(csv_names[0]).decode("utf-8")

        # Parse and validate
        lines = csv_data.strip().split("\n")
        # Skip header if present
        if lines and not lines[0][0].isdigit():
            lines = lines[1:]

        if not lines:
            return 0, "empty CSV"

        # Save — only first 6 columns (open_time, O, H, L, C, V)
        out_dir.mkdir(parents=True, exist_ok=True)
        with open(csv_path, "w") as f:
            for line in lines:
                parts = line.split(",")
                if len(parts) >= 6:
                    f.write(",".join(parts[:6]) + "\n")

        return len(lines), None

    except requests.exceptions.HTTPError as e:
        return 0, f"HTTP {resp.status_code}"
    except Exception as e:
        return 0, str(e)[:80]


def get_months_range(listing_date_str):
    """Generate (year, month) tuples from listing date to now."""
    if not listing_date_str:
        listing_date_str = "2024-01-01"
    start = datetime.strptime(listing_date_str[:7], "%Y-%m")
    end = datetime.now().replace(day=1)
    months = []
    current = start
    while current < end:
        months.append((current.year, current.month))
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)
    return months


def download_symbol(symbol, listing_date_str):
    """Download all months for one symbol. Returns (new, existing, errors)."""
    months = get_months_range(listing_date_str)
    new, existing, errors = 0, 0, 0
    consecutive_404 = 0

    for year, month in months:
        n_rows, err = download_month(symbol, year, month)
        if n_rows == -1:
            existing += 1
            consecutive_404 = 0
        elif n_rows > 0:
            new += 1
            consecutive_404 = 0
        elif err:
            errors += 1
            consecutive_404 = 0
        else:
            consecutive_404 += 1
            # Stop if 3 consecutive months missing (we've passed the available data)
            if consecutive_404 >= 3 and new + existing > 0:
                break

    return new, existing, errors


def validate_symbol(symbol):
    """Validate downloaded data. Returns (ok, issues)."""
    sym_dir = DATA_DIR / symbol
    if not sym_dir.exists():
        return False, ["no directory"]
    csvs = sorted(sym_dir.glob(f"{symbol}-1m-*.csv"))
    if not csvs:
        return False, ["no CSV files"]

    issues = []
    for csv_path in csvs:
        if csv_path.stat().st_size == 0:
            issues.append(f"{csv_path.name}: empty")
            continue
        try:
            df = pd.read_csv(csv_path, header=None, nrows=3, usecols=[0, 1, 2, 3, 4, 5])
            if len(df.columns) < 6:
                issues.append(f"{csv_path.name}: {len(df.columns)} cols")
        except Exception as e:
            issues.append(f"{csv_path.name}: {e}")

    return len(issues) == 0, issues


def main():
    T0 = time.time()
    print("=" * 70)
    print("  DOWNLOAD 1M OHLCV FROM BINANCE VISION")
    print("=" * 70)

    # Get symbols to download
    if "--symbols" in sys.argv:
        idx = sys.argv.index("--symbols")
        symbols_list = [(s, None) for s in sys.argv[idx + 1].split(",")]
    elif "--all" in sys.argv:
        if not ELIGIBLE_FILE.exists():
            print(f"  ERROR: {ELIGIBLE_FILE} not found")
            return
        with open(ELIGIBLE_FILE) as f:
            eligible = json.load(f)
        symbols_list = [(e["binance_symbol"], e.get("listing_date")) for e in eligible]
    else:
        print("  Usage: --all | --symbols BTCUSDT,ETHUSDT")
        return

    print(f"  Symbols: {len(symbols_list)}")

    # Download in parallel (per-symbol)
    t1 = time.time()
    total_new = 0
    total_existing = 0
    total_errors = 0

    def _do_symbol(args):
        sym, listing = args
        return sym, download_symbol(sym, listing)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(_do_symbol, s): s for s in symbols_list}
        done = 0
        for fut in as_completed(futs):
            sym, (new, existing, errors) = fut.result()
            done += 1
            total_new += new
            total_existing += existing
            total_errors += errors

            if new > 0 or done % 20 == 0:
                elapsed = time.time() - t1
                rate = done / elapsed * 60 if elapsed > 0 else 0
                print(f"  [{done}/{len(symbols_list)}] {sym}: +{new} new, "
                      f"{existing} cached, {errors} err "
                      f"({rate:.0f} sym/min)", flush=True)

    elapsed = time.time() - t1
    print(f"\n  Download complete: {total_new} new months, "
          f"{total_existing} cached, {total_errors} errors ({elapsed:.0f}s)")

    # Validate
    print(f"\n  Validating all symbols...")
    n_ok, n_fail = 0, 0
    for sym, _ in symbols_list:
        ok, issues = validate_symbol(sym)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            if issues:
                print(f"    FAIL {sym}: {'; '.join(issues[:2])}")

    print(f"  Validation: {n_ok} OK, {n_fail} FAIL")
    print(f"  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
