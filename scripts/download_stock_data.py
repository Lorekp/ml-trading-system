#!/usr/bin/env python
"""Download stock OHLCV data from Yahoo Finance.

Downloads 1h (max ~2 years) and 1d (max available) data for 30 liquid US stocks.
Saves as parquet files in data/stocks/{interval}/{SYMBOL}.parquet

Format matches what resample_ohlcv() produces:
  - DatetimeIndex (UTC, tz-naive)
  - Columns: open, high, low, close, volume

Usage:
    python scripts/download_stock_data.py
"""
import os, sys, time
import pandas as pd
import yfinance as yf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

STOCKS = {
    # Tech (10)
    "AAPL": "Apple", "MSFT": "Microsoft", "GOOGL": "Google", "AMZN": "Amazon",
    "NVDA": "Nvidia", "META": "Meta", "TSLA": "Tesla", "AMD": "AMD",
    "CRM": "Salesforce", "NFLX": "Netflix",
    # Finance (4)
    "JPM": "JPMorgan", "BAC": "BankOfAmerica", "GS": "GoldmanSachs", "C": "Citigroup",
    # Healthcare (4)
    "JNJ": "J&J", "PFE": "Pfizer", "UNH": "UnitedHealth", "ABBV": "AbbVie",
    # Energy (2)
    "XOM": "Exxon", "CVX": "Chevron",
    # Consumer (4)
    "WMT": "Walmart", "COST": "Costco", "NKE": "Nike", "MCD": "McDonalds",
    # Industrial (3)
    "CAT": "Caterpillar", "BA": "Boeing", "GE": "GE",
    # ETFs (3)
    "SPY": "SP500ETF", "QQQ": "NasdaqETF", "IWM": "Russell2000ETF",
}

INTERVALS = [
    ("1h", "max"),   # ~2 years of hourly
    ("1d", "max"),   # decades of daily
]

OUT_DIR = Path("data/stocks")


def download_one(sym, interval, period):
    """Download one symbol at one interval. Returns (sym, interval, df) or (sym, interval, None)."""
    try:
        ticker = yf.Ticker(sym)
        df = ticker.history(period=period, interval=interval, auto_adjust=True)
        if df is None or len(df) == 0:
            return sym, interval, None

        # Normalize to match system format: tz-naive UTC, lowercase columns
        if df.index.tz is not None:
            df.index = df.index.tz_convert("UTC").tz_localize(None)
        df.index.name = "timestamp"
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.columns = ["open", "high", "low", "close", "volume"]
        df = df.dropna(subset=["close"])
        df = df[df["volume"] > 0]
        return sym, interval, df
    except Exception as e:
        return sym, interval, None


def main():
    T0 = time.time()
    print("=" * 70)
    print("  STOCK DATA DOWNLOAD")
    print(f"  {len(STOCKS)} symbols, intervals: {[i[0] for i in INTERVALS]}")
    print("=" * 70)

    for interval, period in INTERVALS:
        out_dir = OUT_DIR / interval
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"\n  Downloading {interval} data (period={period})...")
        results = {}

        # Use threads (I/O bound)
        with ThreadPoolExecutor(max_workers=5) as pool:
            futs = {pool.submit(download_one, sym, interval, period): sym
                    for sym in STOCKS}
            for f in as_completed(futs):
                sym, intv, df = f.result()
                if df is not None and len(df) > 0:
                    out_path = out_dir / f"{sym}.parquet"
                    df.to_parquet(out_path)
                    results[sym] = len(df)
                else:
                    results[sym] = 0

        # Summary
        ok = {s: n for s, n in results.items() if n > 0}
        fail = {s: n for s, n in results.items() if n == 0}
        total_bars = sum(ok.values())

        print(f"\n  {interval} results: {len(ok)}/{len(STOCKS)} symbols downloaded")
        print(f"  Total bars: {total_bars:,}")

        if ok:
            # Show range for first successful symbol
            sample_sym = list(ok.keys())[0]
            sample_df = pd.read_parquet(out_dir / f"{sample_sym}.parquet")
            print(f"  Date range ({sample_sym}): {sample_df.index[0]} to {sample_df.index[-1]}")
            print(f"  Bars per symbol: min={min(ok.values()):,}, max={max(ok.values()):,}, "
                  f"avg={total_bars // len(ok):,}")

        if fail:
            print(f"  Failed: {', '.join(fail.keys())}")

        # Per-symbol summary
        print(f"\n  {'Symbol':<8} {'Bars':>8} {'Start':>12} {'End':>12}")
        print("  " + "-" * 44)
        for sym in sorted(ok.keys()):
            df = pd.read_parquet(out_dir / f"{sym}.parquet")
            print(f"  {sym:<8} {len(df):>8,} {str(df.index[0].date()):>12} {str(df.index[-1].date()):>12}")

    # Disk usage
    total_mb = sum(f.stat().st_size for f in OUT_DIR.rglob("*.parquet")) / 1e6
    print(f"\n  Total disk usage: {total_mb:.1f} MB")
    print(f"  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
