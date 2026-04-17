#!/usr/bin/env python
"""Discover eligible symbols for dynamic asset selection experiment.

Walk-forward simulation: for each month, determine which symbols ANY of
the 12 dynamic configs would select. Take the union → eligible set.
Map to Hyperliquid names for BBO download.

Usage:
    PYTHONPATH=src python scripts/discover_eligible_symbols.py [--dry-run]
"""
import os, sys, time, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from binance.client import Client

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

import requests
from signal_engine.asset_selection import select_top_n_for_month

DRY_RUN = "--dry-run" in sys.argv
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
MIN_AGE_MONTHS = 14
OUTPUT = Path("data/eligible_symbols.json")


def get_all_usdt_futures():
    """Get ALL active USDT perpetual futures from Binance with listing dates."""
    client = Client()
    info = client.futures_exchange_info()
    symbols = []
    for s in info["symbols"]:
        if (s["symbol"].endswith("USDT") and
            s["contractType"] == "PERPETUAL" and
            s["status"] == "TRADING"):
            symbols.append(s["symbol"])
    print(f"  Active USDT perpetuals: {len(symbols)}")
    return symbols, client


def get_listing_dates(symbols, client):
    """Get listing date for each symbol via earliest kline."""
    print(f"  Getting listing dates for {len(symbols)} symbols...")
    listing = {}
    for i, sym in enumerate(symbols):
        try:
            klines = client.futures_klines(
                symbol=sym, interval="1M", limit=1,
                startTime=int(datetime(2019, 1, 1).timestamp() * 1000))
            if klines:
                listing[sym] = datetime.fromtimestamp(klines[0][0] / 1000)
            else:
                listing[sym] = datetime(2025, 1, 1)
        except Exception:
            listing[sym] = datetime(2025, 1, 1)
        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(symbols)}", flush=True)
            time.sleep(0.5)
        time.sleep(0.05)
    return listing


def get_daily_volumes(symbols, client, listing_dates):
    """Get daily dollar volume for each symbol.

    Uses local 1m data if available (more accurate), otherwise Binance API daily klines.
    """
    print(f"  Computing daily volumes for {len(symbols)} symbols...")
    volumes = {}

    # First: try local data
    local_syms = set()
    for d in DATA_DIRS:
        if d.exists():
            for sd in d.iterdir():
                if sd.is_dir() and sd.name.endswith("USDT"):
                    local_syms.add(sd.name)

    for sym in symbols:
        if sym in local_syms:
            # Use local 1m data
            try:
                from signal_engine.data import load_binance_1m, resample_ohlcv
                ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
                daily = resample_ohlcv(ohlcv_1m, "1D")
                dv = (daily["close"] * daily["volume"]).dropna()
                if dv.index.tz is not None:
                    dv.index = dv.index.tz_localize(None)
                volumes[sym] = dv
            except Exception:
                pass

    # Then: API for remaining
    api_needed = [s for s in symbols if s not in volumes]
    print(f"    Local: {len(volumes)}, API needed: {len(api_needed)}")

    for i, sym in enumerate(api_needed):
        try:
            listing = listing_dates.get(sym, datetime(2024, 1, 1))
            start_ms = int(listing.timestamp() * 1000)
            klines = client.futures_klines(
                symbol=sym, interval="1d", startTime=start_ms, limit=1500)
            if klines:
                dates = [datetime.fromtimestamp(k[0] / 1000) for k in klines]
                # quote_volume is index 7
                qv = [float(k[7]) for k in klines]
                s = pd.Series(qv, index=pd.DatetimeIndex(dates))
                volumes[sym] = s
        except Exception:
            pass
        if (i + 1) % 50 == 0:
            print(f"    API: {i+1}/{len(api_needed)}", flush=True)
            time.sleep(0.5)
        time.sleep(0.05)

    return volumes


def compute_rolling_volumes(daily_volumes):
    """Compute 14d, 30d, 60d rolling median dollar volumes per symbol."""
    vol_14d, vol_30d, vol_60d = {}, {}, {}
    for sym, dv in daily_volumes.items():
        name = sym.replace("USDT", "")
        vol_14d[name] = dv.rolling(14, min_periods=10).median()
        vol_30d[name] = dv.rolling(30, min_periods=20).median()
        vol_60d[name] = dv.rolling(60, min_periods=40).median()
    return vol_14d, vol_30d, vol_60d


def walk_forward_selection(vol_14d, vol_30d, vol_60d, first_dates, months):
    """Run walk-forward selection across all months and volume configs.

    Returns set of symbol names (without USDT) that would be selected
    by ANY config in ANY month.
    """
    eligible = set()
    n_months = len(months)

    for i, month in enumerate(months):
        ms = pd.Timestamp(month.start_time)
        month_eligible = set()

        # Config 1: top-50 by 30d volume (covers Dyn-30 baseline, C1 weekly, C2 bimonth, C3 buffer)
        try:
            sel = select_top_n_for_month(vol_30d, first_dates, ms, top_n=50, min_history_days=180)
            month_eligible.update(sel)
        except Exception:
            pass

        # Config 2: top-50 by 14d volume (D1 variant)
        try:
            sel = select_top_n_for_month(vol_14d, first_dates, ms, top_n=50, min_history_days=180)
            month_eligible.update(sel)
        except Exception:
            pass

        # Config 3: top-50 by 60d volume (D2 variant)
        try:
            sel = select_top_n_for_month(vol_60d, first_dates, ms, top_n=50, min_history_days=180)
            month_eligible.update(sel)
        except Exception:
            pass

        eligible.update(month_eligible)

        if (i + 1) % 12 == 0 or i == n_months - 1:
            print(f"    Month {i+1}/{n_months}: {len(month_eligible)} this month, {len(eligible)} total eligible")

    return eligible


def get_hl_mapping():
    """Get Hyperliquid perpetual names and build Binance→HL mapping."""
    try:
        r = requests.post("https://api.hyperliquid.xyz/info",
                          json={"type": "meta"}, timeout=10)
        meta = r.json()
        hl_coins = {u["name"] for u in meta["universe"]}
    except Exception as e:
        print(f"  WARNING: HL API failed: {e}")
        hl_coins = set()

    # Build mapping: Binance symbol → HL coin name
    # HL uses plain names (BTC, ETH) and k-prefixed for 1000x (kPEPE, kSHIB, kBONK)
    mapping = {}
    for hl_coin in hl_coins:
        # kPEPE → 1000PEPEUSDT
        if hl_coin.startswith("k"):
            binance = f"1000{hl_coin[1:]}USDT"
        else:
            binance = f"{hl_coin}USDT"
        mapping[binance] = hl_coin

    return mapping, hl_coins


def main():
    T0 = time.time()
    print("=" * 70)
    print("  DISCOVER ELIGIBLE SYMBOLS")
    print("=" * 70)

    # Step 1: Get all USDT futures
    symbols, client = get_all_usdt_futures()

    # Step 2: Get listing dates
    listing_dates = get_listing_dates(symbols, client)

    # Filter by age >= 14 months
    cutoff = datetime.now() - timedelta(days=MIN_AGE_MONTHS * 30)
    aged = [s for s in symbols if listing_dates.get(s, datetime.now()) <= cutoff]
    print(f"\n  >= {MIN_AGE_MONTHS} months old: {len(aged)} / {len(symbols)}")

    # Step 3: Get daily volumes
    daily_volumes = get_daily_volumes(aged, client, listing_dates)
    print(f"  Have volume data for: {len(daily_volumes)} symbols")

    # Step 4: Compute rolling volumes
    vol_14d, vol_30d, vol_60d = compute_rolling_volumes(daily_volumes)

    # Build first_dates dict (name → Timestamp)
    first_dates = {}
    for sym, dt in listing_dates.items():
        name = sym.replace("USDT", "")
        first_dates[name] = pd.Timestamp(dt)

    # Step 5: Walk-forward selection
    print("\n  Walk-forward selection simulation...")
    months = pd.period_range("2020-07", "2026-03", freq="M")
    eligible_names = walk_forward_selection(vol_14d, vol_30d, vol_60d, first_dates, months)

    print(f"\n  Eligible symbols (union of all configs): {len(eligible_names)}")

    # Step 6: Map to HL
    hl_mapping, hl_coins = get_hl_mapping()
    print(f"  HL perpetuals available: {len(hl_coins)}")

    # Build output
    results = []
    on_hl = 0
    for name in sorted(eligible_names):
        sym = f"{name}USDT"
        hl_coin = hl_mapping.get(sym)
        if hl_coin:
            on_hl += 1
        listing = listing_dates.get(sym)
        results.append({
            "binance_symbol": sym,
            "name": name,
            "hl_coin": hl_coin,
            "listing_date": listing.strftime("%Y-%m-%d") if listing else None,
            "has_local_1m": any((d / sym).exists() for d in DATA_DIRS if d.exists()),
        })

    # Also check which we already have locally
    have_local = sum(1 for r in results if r["has_local_1m"])
    need_download = sum(1 for r in results if not r["has_local_1m"])

    print(f"\n  Summary:")
    print(f"    Total eligible: {len(results)}")
    print(f"    On Hyperliquid: {on_hl}")
    print(f"    Have local 1m:  {have_local}")
    print(f"    Need download:  {need_download}")

    if DRY_RUN:
        print("\n  DRY RUN — not saving. Eligible symbols:")
        for r in results:
            hl_str = r["hl_coin"] or "N/A"
            local_str = "LOCAL" if r["has_local_1m"] else "NEED"
            print(f"    {r['binance_symbol']:<20} HL={hl_str:<10} {local_str} listed={r['listing_date']}")
    else:
        OUTPUT.parent.mkdir(parents=True, exist_ok=True)
        with open(OUTPUT, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\n  Saved to {OUTPUT}")

    print(f"  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
