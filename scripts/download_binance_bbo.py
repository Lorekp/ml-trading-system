#!/usr/bin/env python
"""Download BBO/bookTicker data from Binance Vision (free, bulk zips).

URL: https://data.binance.vision/data/futures/um/daily/bookTicker/{SYM}/{SYM}-bookTicker-{DATE}.zip

Each zip contains one CSV with columns:
  update_id, best_bid_price, best_bid_qty, best_ask_price, best_ask_qty, transaction_time, event_time

Saves as daily parquets matching existing bookTicker_parquet format.

Usage:
    PYTHONPATH=src python scripts/download_binance_bbo.py --all
    PYTHONPATH=src python scripts/download_binance_bbo.py --symbols BTCUSDT,ETHUSDT
    PYTHONPATH=src python scripts/download_binance_bbo.py --all --start 2023-05-01 --end 2025-03-01
"""
import os, sys, time, json, io, zipfile, warnings
import numpy as np
import pandas as pd
import requests
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
warnings.filterwarnings("ignore")

OUTPUT_DIR = Path("data/bookTicker_parquet")
ELIGIBLE_FILE = Path("data/eligible_symbols.json")
BASE_URL = "https://data.binance.vision/data/futures/um/daily/bookTicker"
MAX_WORKERS = 30
ERR_LOG = Path("data/bbo_download_errors.log")

# Thread-local sessions for connection pooling
import threading
_local = threading.local()

def _get_session():
    if not hasattr(_local, 'session'):
        _local.session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(
            pool_connections=10, pool_maxsize=10, max_retries=3)
        _local.session.mount('https://', adapter)
    return _local.session


def download_day(symbol, date_str):
    """Download one day of bookTicker from Binance Vision, save as parquet.

    Returns: (n_rows, error_or_None)
    """
    out_dir = OUTPUT_DIR / symbol
    pq_path = out_dir / f"{date_str}.parquet"

    if pq_path.exists() and pq_path.stat().st_size > 100:
        return -1, None  # already exists

    url = f"{BASE_URL}/{symbol}/{symbol}-bookTicker-{date_str}.zip"
    session = _get_session()
    last_err = None

    for attempt in range(3):
        try:
            resp = session.get(url, timeout=60)
            if resp.status_code == 404:
                return 0, None
            if resp.status_code == 451:  # geo-blocked
                return 0, None
            resp.raise_for_status()

            zf = zipfile.ZipFile(io.BytesIO(resp.content))
            csv_names = [n for n in zf.namelist() if n.endswith(".csv")]
            if not csv_names:
                return 0, f"no CSV in zip ({zf.namelist()})"

            with zf.open(csv_names[0]) as f:
                df = pd.read_csv(f)

            if len(df) == 0:
                return 0, "empty CSV"

            keep_cols = ["best_bid_price", "best_bid_qty", "best_ask_price",
                         "best_ask_qty", "transaction_time"]
            for c in keep_cols:
                if c not in df.columns:
                    return 0, f"missing column: {c} (has: {list(df.columns)[:5]})"

            df = df[keep_cols]
            df["best_bid_price"] = df["best_bid_price"].astype(np.float32)
            df["best_bid_qty"] = df["best_bid_qty"].astype(np.float32)
            df["best_ask_price"] = df["best_ask_price"].astype(np.float32)
            df["best_ask_qty"] = df["best_ask_qty"].astype(np.float32)
            df["transaction_time"] = df["transaction_time"].astype(np.int64)

            out_dir.mkdir(parents=True, exist_ok=True)
            df.to_parquet(pq_path, index=False)
            return len(df), None

        except requests.exceptions.ConnectionError as e:
            last_err = f"ConnectionError: {e}"
            time.sleep(2 * (attempt + 1))
        except requests.exceptions.Timeout as e:
            last_err = f"Timeout: {e}"
            time.sleep(2 * (attempt + 1))
        except requests.exceptions.HTTPError as e:
            last_err = f"HTTP {resp.status_code}: {resp.reason}"
            if resp.status_code >= 500:
                time.sleep(2 * (attempt + 1))
            else:
                break  # client error, don't retry
        except zipfile.BadZipFile as e:
            last_err = f"BadZip: {len(resp.content)} bytes"
            time.sleep(1)
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:100]}"
            break

    return 0, last_err


def get_date_range(start_str, end_str):
    """Generate date strings between start and end."""
    start = datetime.strptime(start_str, "%Y-%m-%d")
    end = datetime.strptime(end_str, "%Y-%m-%d")
    dates = []
    current = start
    while current <= end:
        dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


def main():
    T0 = time.time()
    print("=" * 70)
    print("  DOWNLOAD BBO FROM BINANCE VISION")
    print("=" * 70)

    # Parse args
    start_date = "2023-01-01"
    end_date = "2026-03-01"
    if "--start" in sys.argv:
        start_date = sys.argv[sys.argv.index("--start") + 1]
    if "--end" in sys.argv:
        end_date = sys.argv[sys.argv.index("--end") + 1]

    if "--symbols" in sys.argv:
        idx = sys.argv.index("--symbols")
        symbols = sys.argv[idx + 1].split(",")
    elif "--all" in sys.argv:
        if not ELIGIBLE_FILE.exists():
            print(f"  ERROR: {ELIGIBLE_FILE} not found")
            return
        with open(ELIGIBLE_FILE) as f:
            eligible = json.load(f)
        symbols = [e["binance_symbol"] for e in eligible]
    else:
        print("  Usage: --all | --symbols BTCUSDT,ETHUSDT")
        return

    dates = get_date_range(start_date, end_date)
    print(f"  Symbols: {len(symbols)}")
    print(f"  Date range: {start_date} to {end_date} ({len(dates)} days)")
    print(f"  Max work items: {len(symbols) * len(dates):,}")
    print(f"  Workers: {MAX_WORKERS}")

    # Build work queue — newest first, skip existing
    work = []
    for sym in symbols:
        out_dir = OUTPUT_DIR / sym
        existing = set()
        if out_dir.exists():
            existing = set(f.stem for f in out_dir.glob("*.parquet"))
        for d in reversed(dates):
            if d not in existing:
                work.append((sym, d))

    print(f"  After skipping cached: {len(work):,} to download\n")

    if not work:
        print("  Everything cached!")
        return

    # Download
    t1 = time.time()
    n_new = 0
    n_fail = 0
    n_404 = 0
    sym_counts = {}
    error_details = {}  # {error_type: count}

    # Clear error log
    ERR_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(ERR_LOG, "w") as f:
        f.write(f"BBO download errors - {datetime.now().isoformat()}\n\n")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs = {pool.submit(download_day, sym, d): (sym, d) for sym, d in work}
        done = 0
        for fut in as_completed(futs):
            sym, d = futs[fut]
            done += 1
            try:
                n_rows, err = fut.result()
            except Exception as e:
                n_rows, err = 0, f"Future exception: {type(e).__name__}: {str(e)[:100]}"

            if n_rows > 0:
                n_new += 1
                sym_counts[sym] = sym_counts.get(sym, 0) + 1
            elif n_rows == 0 and err is None:
                n_404 += 1
            elif err:
                n_fail += 1
                # Categorize error
                err_type = err.split(":")[0] if ":" in err else err[:30]
                error_details[err_type] = error_details.get(err_type, 0) + 1
                # Log to file
                with open(ERR_LOG, "a") as f:
                    f.write(f"{sym} {d}: {err}\n")

            if done % 200 == 0 or done == len(work):
                elapsed = time.time() - t1
                rate = done / elapsed * 60 if elapsed > 0 else 0
                eta = (len(work) - done) / rate if rate > 0 else 0
                err_summary = ", ".join(f"{k}:{v}" for k, v in sorted(
                    error_details.items(), key=lambda x: -x[1])[:3]) if error_details else ""
                print(f"  {done:,}/{len(work):,} ({n_new:,} new, {n_404:,} n/a, "
                      f"{n_fail} err, {len(sym_counts)} syms, "
                      f"{rate:.0f}/min, ETA {eta:.0f}min)"
                      f"{' [' + err_summary + ']' if err_summary else ''}",
                      flush=True)

    elapsed = time.time() - t1
    print(f"\n  Done: {n_new:,} new days, {n_404:,} not available, "
          f"{n_fail} failed ({elapsed:.0f}s)")
    if error_details:
        print(f"\n  Error breakdown:")
        for err_type, count in sorted(error_details.items(), key=lambda x: -x[1]):
            print(f"    {err_type}: {count}")
        print(f"  Full error log: {ERR_LOG}")

    # Validation summary
    print(f"\n  Validation...")
    total_syms = 0
    total_days = 0
    issues = []
    for d in OUTPUT_DIR.iterdir():
        if d.is_dir():
            pqs = list(d.glob("*.parquet"))
            if pqs:
                total_syms += 1
                total_days += len(pqs)
                # Spot check
                import random
                sample = random.choice(pqs)
                try:
                    df = pd.read_parquet(sample)
                    if len(df) == 0:
                        issues.append(f"{d.name}: empty")
                    elif (df["best_bid_price"] <= 0).any():
                        issues.append(f"{d.name}: zero prices")
                except Exception as e:
                    issues.append(f"{d.name}: {e}")

    print(f"  Total: {total_syms} symbols, {total_days:,} daily parquets")
    if issues:
        for i in issues[:5]:
            print(f"    ! {i}")
    else:
        print(f"  All spot-checks OK")
    print(f"  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
