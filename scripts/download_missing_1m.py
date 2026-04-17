#!/usr/bin/env python
"""Download missing Binance 1m futures data for top symbols by volume.

Uses Binance API to get monthly kline CSVs.

Usage:
    PYTHONPATH=src python scripts/download_missing_1m.py
"""
import os, sys, time, warnings
from pathlib import Path
from datetime import datetime, timedelta
from binance.client import Client

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
warnings.filterwarnings("ignore")

DATA_DIR = Path("data/binance-1m")
client = Client()


def get_top_symbols(n=100):
    """Get top N USDT futures by 24h volume."""
    tickers = client.futures_ticker()
    usdt = [(t["symbol"], float(t["quoteVolume"])) for t in tickers if t["symbol"].endswith("USDT")]
    usdt.sort(key=lambda x: -x[1])
    return [sym for sym, _ in usdt[:n]]


def get_symbol_listing_date(symbol):
    """Get approximate listing date from earliest available kline."""
    try:
        klines = client.futures_klines(symbol=symbol, interval="1M", limit=1,
                                        startTime=int(datetime(2019, 1, 1).timestamp() * 1000))
        if klines:
            return datetime.fromtimestamp(klines[0][0] / 1000)
    except:
        pass
    return datetime(2024, 1, 1)  # default


def download_symbol(symbol):
    """Download all available monthly 1m data for a symbol."""
    sym_dir = DATA_DIR / symbol
    sym_dir.mkdir(parents=True, exist_ok=True)

    # Check existing files
    existing = set(f.stem for f in sym_dir.glob(f"{symbol}-1m-*.csv"))

    # Get listing date
    listing = get_symbol_listing_date(symbol)
    start = datetime(listing.year, listing.month, 1)
    end = datetime.now().replace(day=1)  # start of current month

    months_downloaded = 0
    current = start
    while current < end:
        year_month = current.strftime("%Y-%m")
        filename = f"{symbol}-1m-{year_month}"
        if filename in existing:
            current = (current.replace(day=28) + timedelta(days=4)).replace(day=1)
            continue

        # Download this month's klines
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

            # Paginate (1500 limit per request)
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
                        # Format: open_time, open, high, low, close, volume, ...
                        f.write(",".join(str(x) for x in k[:6]) + "\n")
                months_downloaded += 1

            time.sleep(0.1)  # rate limit
        except Exception as e:
            print(f"    Error {symbol} {year_month}: {e}")
            time.sleep(1)

        current = next_month

    return months_downloaded


def main():
    T0 = time.time()
    print("=" * 70)
    print("  DOWNLOAD MISSING 1M FUTURES DATA")
    print("=" * 70)

    top_symbols = get_top_symbols(100)
    local = set(d.name for d in DATA_DIR.iterdir() if d.is_dir())
    missing = [s for s in top_symbols if s not in local]

    # Also check which local symbols need updates (missing recent months)
    need_update = []
    for sym in local:
        sym_dir = DATA_DIR / sym
        csvs = sorted(sym_dir.glob(f"{sym}-1m-*.csv"))
        if csvs:
            last = csvs[-1].stem.split("-")[-2:]  # ['2026', '02']
            last_month = f"{last[0]}-{last[1]}"
            if last_month < "2026-03":
                need_update.append(sym)

    print(f"\n  Top 100 by volume: {len(top_symbols)}")
    print(f"  Already have: {len(top_symbols) - len(missing)}")
    print(f"  Need download: {len(missing)}")
    print(f"  Need update: {len(need_update)}")

    # Download missing
    if missing:
        print(f"\n  Downloading {len(missing)} new symbols...")
        for i, sym in enumerate(missing, 1):
            t1 = time.time()
            n = download_symbol(sym)
            print(f"    [{i}/{len(missing)}] {sym}: {n} months ({time.time()-t1:.0f}s)", flush=True)

    # Update existing
    if need_update:
        print(f"\n  Updating {len(need_update)} existing symbols...")
        for i, sym in enumerate(need_update, 1):
            t1 = time.time()
            n = download_symbol(sym)
            if n > 0:
                print(f"    [{i}/{len(need_update)}] {sym}: +{n} months ({time.time()-t1:.0f}s)", flush=True)

    # Summary
    total_now = len([d for d in DATA_DIR.iterdir() if d.is_dir()])
    print(f"\n  Total symbols now: {total_now}")
    print(f"  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
