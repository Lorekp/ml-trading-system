#!/usr/bin/env python
"""Download 1m OHLCV from Binance for eligible symbols not yet local.

Reads eligible_symbols.json, downloads missing data, validates downloads.

Usage:
    PYTHONPATH=src python scripts/download_eligible_1m.py [--dry-run] [--validate-only]
"""
import os, sys, time, json, warnings
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from binance.client import Client

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
warnings.filterwarnings("ignore")

DRY_RUN = "--dry-run" in sys.argv
VALIDATE_ONLY = "--validate-only" in sys.argv
DATA_DIR = Path("data/binance-1m")
ELIGIBLE_FILE = Path("data/eligible_symbols.json")
client = Client()


def download_symbol(symbol, listing_date_str):
    """Download all available monthly 1m data for a symbol. Returns (months_downloaded, errors)."""
    sym_dir = DATA_DIR / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)

    existing = set(f.stem for f in sym_dir.glob(f"{symbol}-1m-*.csv"))

    listing = datetime.strptime(listing_date_str, "%Y-%m-%d")
    start = datetime(listing.year, listing.month, 1)
    end = datetime.now().replace(day=1)

    months_downloaded = 0
    errors = []
    current = start
    while current < end:
        year_month = current.strftime("%Y-%m")
        filename = f"{symbol}-1m-{year_month}"
        if filename in existing:
            current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
            continue

        ms_start = int(current.timestamp() * 1000)
        next_month = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
        ms_end = int(next_month.timestamp() * 1000) - 1

        try:
            klines = client.futures_klines(
                symbol=symbol, interval="1m",
                startTime=ms_start, endTime=ms_end, limit=1500)

            if not klines:
                current = next_month
                continue

            all_klines = list(klines)
            while len(klines) == 1500:
                last_ts = klines[-1][0]
                klines = client.futures_klines(
                    symbol=symbol, interval="1m",
                    startTime=last_ts + 1, endTime=ms_end, limit=1500)
                if klines:
                    all_klines.extend(klines)
                else:
                    break

            if all_klines:
                filepath = sym_dir / f"{filename}.csv"
                with open(filepath, "w") as f:
                    for k in all_klines:
                        f.write(",".join(str(x) for x in k[:6]) + "\n")
                months_downloaded += 1

            time.sleep(0.3)  # rate limit
        except Exception as e:
            err_str = str(e)
            errors.append(f"{year_month}: {e}")
            if "Too many requests" in err_str or "-1003" in err_str:
                time.sleep(30)  # back off on rate limit
            else:
                time.sleep(2)

        current = next_month

    return months_downloaded, errors


def validate_symbol(symbol):
    """Validate downloaded data for a symbol. Returns (ok, issues)."""
    sym_dir = DATA_DIR / symbol
    if not sym_dir.exists():
        return False, ["directory missing"]

    csvs = sorted(sym_dir.glob(f"{symbol}-1m-*.csv"))
    if not csvs:
        return False, ["no CSV files"]

    issues = []
    total_rows = 0
    prev_month = None

    for csv_path in csvs:
        # Check file not empty
        size = csv_path.stat().st_size
        if size == 0:
            issues.append(f"{csv_path.name}: empty file")
            continue

        # Check parseable
        try:
            df = pd.read_csv(csv_path, header=None, nrows=5)
            if len(df.columns) < 6:
                issues.append(f"{csv_path.name}: only {len(df.columns)} columns (need 6)")
                continue
        except Exception as e:
            issues.append(f"{csv_path.name}: parse error: {e}")
            continue

        # Check row count (1m bars: ~43200 per 30-day month, allow range)
        try:
            df_full = pd.read_csv(csv_path, header=None, usecols=[0])
            n_rows = len(df_full)
            total_rows += n_rows
            if n_rows < 1000:
                issues.append(f"{csv_path.name}: too few rows ({n_rows})")
        except Exception:
            pass

        # Check month continuity
        parts = csv_path.stem.split("-")
        if len(parts) >= 4:
            month_str = f"{parts[-2]}-{parts[-1]}"
            if prev_month:
                prev_dt = datetime.strptime(prev_month, "%Y-%m")
                curr_dt = datetime.strptime(month_str, "%Y-%m")
                gap_months = (curr_dt.year - prev_dt.year) * 12 + curr_dt.month - prev_dt.month
                if gap_months > 1:
                    issues.append(f"gap: {prev_month} -> {month_str}")
            prev_month = month_str

    # Check reasonable total
    n_months = len(csvs)
    if total_rows > 0 and n_months > 0:
        avg_per_month = total_rows / n_months
        if avg_per_month < 10000:
            issues.append(f"low avg rows/month: {avg_per_month:.0f}")

    ok = len(issues) == 0
    return ok, issues


def main():
    T0 = time.time()
    print("=" * 70)
    print("  DOWNLOAD ELIGIBLE 1M OHLCV")
    print("=" * 70)

    if not ELIGIBLE_FILE.exists():
        print(f"  ERROR: {ELIGIBLE_FILE} not found. Run discover_eligible_symbols.py first.")
        return

    with open(ELIGIBLE_FILE) as f:
        eligible = json.load(f)

    need_download = [e for e in eligible if not e["has_local_1m"]]
    have_local = [e for e in eligible if e["has_local_1m"]]

    print(f"  Total eligible: {len(eligible)}")
    print(f"  Already local:  {len(have_local)}")
    print(f"  Need download:  {len(need_download)}")

    if VALIDATE_ONLY:
        print("\n  VALIDATION ONLY MODE")
        all_syms = [e["binance_symbol"] for e in eligible]
    else:
        all_syms = []

    # Download missing
    if not VALIDATE_ONLY and not DRY_RUN:
        print(f"\n  Downloading {len(need_download)} symbols...")
        for i, entry in enumerate(need_download, 1):
            sym = entry["binance_symbol"]
            listing = entry["listing_date"]
            if not listing:
                continue
            t1 = time.time()
            n_months, errors = download_symbol(sym, listing)
            elapsed = time.time() - t1
            err_str = f" ({len(errors)} errors)" if errors else ""
            print(f"    [{i}/{len(need_download)}] {sym}: {n_months} months ({elapsed:.0f}s){err_str}",
                  flush=True)
            if errors:
                for e in errors[:3]:
                    print(f"      ! {e}")
            all_syms.append(sym)

    elif DRY_RUN:
        print("\n  DRY RUN — would download:")
        for entry in need_download:
            print(f"    {entry['binance_symbol']} (listed {entry['listing_date']})")
        all_syms = [e["binance_symbol"] for e in eligible if e["has_local_1m"]]

    # Validate all data
    print(f"\n  Validating {len(all_syms) if all_syms else 'all local'} symbols...")
    if not all_syms:
        all_syms = [e["binance_symbol"] for e in eligible if e["has_local_1m"]]
        all_syms += [e["binance_symbol"] for e in need_download]

    n_ok, n_fail = 0, 0
    for sym in sorted(set(all_syms)):
        ok, issues = validate_symbol(sym)
        if ok:
            n_ok += 1
        else:
            n_fail += 1
            if issues:
                print(f"    FAIL {sym}: {'; '.join(issues[:3])}")

    print(f"\n  Validation: {n_ok} OK, {n_fail} FAIL")
    print(f"  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
