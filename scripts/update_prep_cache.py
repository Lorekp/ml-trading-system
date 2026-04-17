#!/usr/bin/env python
"""Update v2_prep_cache.pkl with new month's data.

Run after monthly retrain (download_and_train.py --retrain) to extend the
prep cache so the live asset selector stays in sync with the backtest.

Updates:
  - hold_sigs: extend with new signals from cached parquets
  - asset_volumes: extend 30d rolling median dollar volume
  - vol_series: extend 20d rolling std of daily returns

Usage:
    PYTHONPATH=src python scripts/update_prep_cache.py
    PYTHONPATH=src python scripts/update_prep_cache.py --rebuild  # full rebuild
"""

import os
import sys
import time
import pickle
import warnings
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
warnings.filterwarnings("ignore")

from signal_engine.features import load_binance_1m
from signal_engine.data_loading import resample_ohlcv
from signal_engine.asset_selection import compute_monthly_dollar_volume

CACHE_DIR = Path("experiments/.cache")
CACHE_PATH = CACHE_DIR / "v2_prep_cache.pkl"
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
NEEDED_TFS = [("15m", "15min"), ("1h", "1h"), ("3h", "3h"), ("4h", "4h"), ("24h", "1D")]
FULL5_HOLDS = {"15m": 2, "1h": 4, "3h": 16, "4h": 6, "24h": 12}
REBUILD = "--rebuild" in sys.argv


def _discover_symbols() -> dict[str, str]:
    """Find all USDT-margined assets in data directories."""
    symbols = {}
    for d in DATA_DIRS:
        if d.exists():
            for sd in sorted(d.iterdir()):
                if sd.is_dir() and sd.name.endswith("USDT"):
                    name = sd.name.replace("USDT", "")
                    if name not in symbols:
                        symbols[name] = sd.name
    return symbols


def _compute_vol_one(args):
    """Compute 20d rolling std of daily returns for one asset."""
    name, sym = args
    warnings.filterwarnings("ignore")
    try:
        o = resample_ohlcv(load_binance_1m(sym, DATA_DIRS), "1D")["close"]
        if o.index.tz is not None:
            o.index = o.index.tz_localize(None)
        return name, o.pct_change().rolling(20).std()
    except Exception:
        return name, None


def _compute_volume_one(args):
    """Compute 30d rolling median dollar volume for one asset."""
    name, sym = args
    warnings.filterwarnings("ignore")
    try:
        ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        if len(ohlcv_15m) < 100:
            return name, None
        vol30 = compute_monthly_dollar_volume(ohlcv_15m)
        if vol30.index.tz is not None:
            vol30.index = vol30.index.tz_localize(None)
        return name, vol30
    except Exception:
        return name, None


def main():
    t0 = time.time()
    all_symbols = _discover_symbols()
    print(f"Discovered {len(all_symbols)} symbols in data directories")

    # Load existing cache
    if CACHE_PATH.exists() and not REBUILD:
        with open(CACHE_PATH, "rb") as f:
            cached = pickle.load(f)
        hold_sigs = cached.get("hold_sigs", {})
        asset_volumes = cached.get("asset_volumes", {})
        vol_series = cached.get("vol_series", {})
        # Preserve other fields
        asset_volumes_14d = cached.get("asset_volumes_14d", {})
        asset_volumes_60d = cached.get("asset_volumes_60d", {})
        asset_first_dates = cached.get("asset_first_dates", {})
        asset_scores = cached.get("asset_scores", {})
        daily_returns = cached.get("daily_returns", {})
        print(f"Loaded existing cache: {len(hold_sigs)} assets with sigs, "
              f"{len(asset_volumes)} with volume")
    else:
        hold_sigs = {}
        asset_volumes = {}
        asset_volumes_14d = {}
        asset_volumes_60d = {}
        asset_first_dates = {}
        asset_scores = {}
        daily_returns = {}
        vol_series = {}
        print("Starting fresh (rebuild mode)" if REBUILD else "No existing cache found")

    # ── 1. Update hold_sigs from signal parquets ──
    print("\nPhase 1: Updating hold_sigs from signal parquets...")
    signal_cache = {}
    for label, resample in NEEDED_TFS:
        cache_path = CACHE_DIR / f"sigs_{label}_fwd4.parquet"
        if cache_path.exists():
            sigs = pd.read_parquet(cache_path)
            if len(sigs) > 0:
                signal_cache[label] = sigs
                print(f"  {label}: {len(sigs):,} signals")

    if signal_cache:
        # Rebuild hold_sigs from parquets (authoritative)
        new_hold_sigs = defaultdict(dict)
        for tf_label in signal_cache:
            df = signal_cache[tf_label]
            hold = FULL5_HOLDS[tf_label]
            for name in df["asset"].unique():
                s = df[df["asset"] == name].reset_index(drop=True)
                s["hold_bars"] = hold
                new_hold_sigs[name][tf_label] = s

        # Merge: new data takes precedence
        for name in new_hold_sigs:
            if name not in hold_sigs:
                hold_sigs[name] = {}
            for tf, df in new_hold_sigs[name].items():
                hold_sigs[name][tf] = df

        print(f"  Updated: {len(hold_sigs)} assets with signals")
    else:
        print("  No signal parquets found, hold_sigs unchanged")

    # ── 2. Update asset_volumes (30d rolling median) ──
    print("\nPhase 2: Updating asset volumes...")
    vol_pairs = [(n, s) for n, s in all_symbols.items()]
    updated_vol = 0
    with ProcessPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_compute_volume_one, p): p[0] for p in vol_pairs}
        for f in as_completed(futures):
            name, vol = f.result()
            if vol is not None:
                asset_volumes[name] = vol
                updated_vol += 1
    print(f"  Updated: {updated_vol} assets ({len(asset_volumes)} total)")

    # ── 3. Update vol_series (20d rolling std) ──
    print("\nPhase 3: Updating vol series...")
    vol_pairs = [(n, s) for n, s in all_symbols.items()]
    updated_vs = 0
    with ProcessPoolExecutor(max_workers=8) as pool:
        for name, vs in pool.map(_compute_vol_one, vol_pairs):
            if vs is not None:
                vol_series[name] = vs
                updated_vs += 1
    print(f"  Updated: {updated_vs} assets ({len(vol_series)} total)")

    # ── 4. Report cache date range ──
    btc_vol = vol_series.get("BTC")
    if btc_vol is not None:
        print(f"\nBTC vol_series: {btc_vol.index.min().date()} -> {btc_vol.index.max().date()}")
    btc_volume = asset_volumes.get("BTC")
    if btc_volume is not None:
        print(f"BTC asset_volume: {btc_volume.index.min().date()} -> {btc_volume.index.max().date()}")

    # ── 5. Save updated cache ──
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    # Backup existing
    if CACHE_PATH.exists():
        backup = CACHE_PATH.with_suffix(".pkl.bak")
        import shutil
        shutil.copy2(CACHE_PATH, backup)
        print(f"\nBacked up existing cache to {backup}")

    with open(CACHE_PATH, "wb") as f:
        pickle.dump({
            "asset_volumes": asset_volumes,
            "asset_volumes_14d": asset_volumes_14d,
            "asset_volumes_60d": asset_volumes_60d,
            "asset_first_dates": asset_first_dates,
            "asset_scores": asset_scores,
            "daily_returns": daily_returns,
            "hold_sigs": dict(hold_sigs),
            "vol_series": vol_series,
        }, f)

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s. Cache saved to {CACHE_PATH}")
    print(f"  hold_sigs: {len(hold_sigs)} assets")
    print(f"  asset_volumes: {len(asset_volumes)} assets")
    print(f"  vol_series: {len(vol_series)} assets")


if __name__ == "__main__":
    main()
