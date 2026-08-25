"""Download 1m Binance futures data for top 100 liquid crypto assets.

Uses Binance data.vision (free) for historical monthly klines.
Downloads last 24 months of 1m data per symbol.
"""
import sys
sys.path.insert(0, "src")

import os
import json
import time
import zipfile
import requests
from pathlib import Path
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed

DATA_DIR = Path("data/binance-1m")
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "https://data.binance.vision/data/futures/um/monthly/klines"


def get_top_symbols(n=100):
    """Get top N Binance USDT-M futures by 24h volume."""
    resp = requests.get("https://fapi.binance.com/fapi/v1/ticker/24hr", timeout=15)
    tickers = resp.json()
    usdt = [t for t in tickers if t["symbol"].endswith("USDT")]
    usdt.sort(key=lambda x: float(x["quoteVolume"]), reverse=True)

    exclude = {"USDCUSDT", "BUSDUSDT", "TUSDUSDT", "FDUSDUSDT", "DAIUSDT",
               "XAUUSDT", "XAGUSDT", "PAXGUSDT", "TSLAUSDT", "USDTUSDT"}

    filtered = []
    for t in usdt:
        sym = t["symbol"]
        if sym in exclude:
            continue
        if float(t["lastPrice"]) < 0.00001:
            continue
        filtered.append(sym)
        if len(filtered) >= n:
            break
    return filtered


def get_existing_symbols():
    """Check which symbols already have data."""
    existing = set()
    if DATA_DIR.exists():
        for d in DATA_DIR.iterdir():
            if d.is_dir():
                csvs = list(d.glob("*-1m-*.csv"))
                if len(csvs) >= 6:  # at least 6 months
                    existing.add(d.name)
    return existing


def download_month(symbol, year, month):
    """Download one month of 1m klines from Binance data.vision."""
    month_str = f"{year}-{month:02d}"
    filename = f"{symbol}-1m-{month_str}.zip"
    url = f"{BASE_URL}/{symbol}/1m/{filename}"

    symbol_dir = DATA_DIR / symbol
    symbol_dir.mkdir(exist_ok=True)

    csv_path = symbol_dir / f"{symbol}-1m-{month_str}.csv"
    if csv_path.exists() and csv_path.stat().st_size > 1000:
        return "exists"

    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code == 404:
            return "not_found"
        if resp.status_code != 200:
            return f"error_{resp.status_code}"

        zip_path = symbol_dir / filename
        with open(zip_path, "wb") as f:
            f.write(resp.content)

        # Extract
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(symbol_dir)
        zip_path.unlink()

        return "ok"
    except Exception as e:
        return f"error: {e}"


def download_symbol(symbol, months=24):
    """Download last N months of data for a symbol."""
    now = datetime.utcnow()
    results = []
    for i in range(months):
        dt = now - timedelta(days=30 * i)
        result = download_month(symbol, dt.year, dt.month)
        results.append((f"{dt.year}-{dt.month:02d}", result))
        if result == "not_found" and i > 6:
            break  # symbol didn't exist before this
        time.sleep(0.1)  # rate limit
    return results


def main():
    print("=" * 70)
    print("DOWNLOADING TOP 100 CRYPTO ASSETS (1m Binance Futures)")
    print("=" * 70)
    print(flush=True)

    symbols = get_top_symbols(100)
    existing = get_existing_symbols()

    have = [s for s in symbols if s in existing]
    need = [s for s in symbols if s not in existing]

    print(f"Top 100 symbols identified")
    already = ", ".join(have[:8])
    print(f"Already have: {len(have)} ({already}...)")
    print(f"Need to download: {len(need)}")
    print(flush=True)

    if not need:
        print("Nothing to download!")
        return

    # Download with thread pool (10 concurrent)
    print(f"\nDownloading {len(need)} symbols (24 months each, 10 threads)...")
    print(flush=True)

    completed = 0
    failed = []

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {pool.submit(download_symbol, sym, 24): sym for sym in need}

        for future in as_completed(futures):
            sym = futures[future]
            try:
                results = future.result()
                ok_count = sum(1 for _, r in results if r in ("ok", "exists"))
                completed += 1
                print(f"  [{completed}/{len(need)}] {sym}: {ok_count} months downloaded", flush=True)
                if ok_count < 6:
                    failed.append((sym, ok_count))
            except Exception as e:
                completed += 1
                print(f"  [{completed}/{len(need)}] {sym}: FAILED ({e})", flush=True)
                failed.append((sym, 0))

    print(f"\nDone! {completed - len(failed)} symbols downloaded successfully.")
    if failed:
        print(f"Failed/insufficient: {len(failed)}")
        for sym, months in failed:
            print(f"  {sym}: only {months} months")

    # Final inventory
    all_existing = get_existing_symbols()
    coverage = sum(1 for s in symbols if s in all_existing)
    print(f"\nFinal coverage: {coverage}/{len(symbols)} top symbols have data")

    # Save symbol list
    with open("data/top100_symbols.json", "w") as f:
        json.dump({
            "all_100": symbols,
            "have_data": sorted(all_existing & set(symbols)),
            "missing": sorted(set(symbols) - all_existing),
        }, f, indent=2)


if __name__ == "__main__":
    main()
