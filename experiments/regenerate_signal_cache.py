#!/usr/bin/env python
"""Regenerate signal cache for ALL available symbols (not just fixed 30).

Generates fwd=4 signals for all symbols at each TF and saves to
experiments/.cache/sigs_{tf}_fwd4.parquet.

Supports --symbols-file to restrict to eligible symbols only,
and --tf to regenerate specific timeframes.

Usage:
    PYTHONPATH=src python experiments/regenerate_signal_cache.py
    PYTHONPATH=src python experiments/regenerate_signal_cache.py --symbols-file data/eligible_symbols.json
    PYTHONPATH=src python experiments/regenerate_signal_cache.py --tf 15m,1h
"""
import sys, os, time, json, warnings, gc
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.model import WalkForwardModel
from signal_engine.filters import apply_regime_filter
from signal_engine.config import *

DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
CACHE_DIR = Path("experiments/.cache")
OHLCV_FEATURES = [
    "ret_1","ret_3","ret_7","ret_14","bar_range","vol_zscore","rsi_14",
    "stoch_rsi","macd_hist","bb_pct_b","atr_14","dist_swing_high","dist_swing_low"]
ALL_TFS = [("15m","15min"),("1h","1h"),("3h","3h"),("4h","4h"),("24h","1D")]

# Parse --tf argument
tf_filter = None
if "--tf" in sys.argv:
    idx = sys.argv.index("--tf")
    tf_filter = set(sys.argv[idx + 1].split(","))
TFS = [(label, resample) for label, resample in ALL_TFS
       if tf_filter is None or label in tf_filter]

# Discover symbols
ALL_SYMBOLS = {}
if "--symbols-file" in sys.argv:
    idx = sys.argv.index("--symbols-file")
    sf = Path(sys.argv[idx + 1])
    if sf.exists():
        with open(sf) as f:
            eligible = json.load(f)
        for e in eligible:
            sym = e["binance_symbol"]
            name = e["name"]
            # Check at least one data dir has this symbol
            for d in DATA_DIRS:
                if (d / sym).exists():
                    ALL_SYMBOLS[name] = sym
                    break
    print(f"  Loaded {len(ALL_SYMBOLS)} symbols from {sf}")
else:
    for d in DATA_DIRS:
        if d.exists():
            for sd in sorted(d.iterdir()):
                if sd.is_dir() and sd.name.endswith("USDT"):
                    name = sd.name.replace("USDT", "")
                    if name not in ALL_SYMBOLS:
                        ALL_SYMBOLS[name] = sd.name


def _gen_one(args):
    name, sym, tf_resample = args
    warnings.filterwarnings("ignore")
    config = SystemConfig(
        model=ModelConfig(C=0.10, penalty="elasticnet", solver="saga", max_iter=1000,
                          seed=42, fwd_bars=4, signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=list(OHLCV_FEATURES), winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS], timeframe=tf_resample, regime_timeframe="4h"),
        assets=ALL_SYMBOLS)
    try:
        ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
    except FileNotFoundError:
        return name, None
    ohlcv_tf = resample_ohlcv(ohlcv_1m, tf_resample)
    ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
    feat = compute_features(ohlcv_tf, config.features)
    wf = WalkForwardModel(config, force_monthly_retrain=True)
    sigs = wf.generate_signals_batch(ohlcv_tf, feat, name)
    if len(sigs) == 0: return name, None
    sigs = apply_regime_filter(sigs, ohlcv_4h, config.regime_filter)
    if len(sigs) == 0: return name, None
    return name, sigs


def main():
    T0 = time.time()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Regenerating signal cache for {len(ALL_SYMBOLS)} symbols across {len(TFS)} TFs")
    print(f"Output: {CACHE_DIR}/sigs_{{tf}}_fwd4.parquet\n")

    for tf_label, tf_resample in TFS:
        t1 = time.time()
        cache_path = CACHE_DIR / f"sigs_{tf_label}_fwd4.parquet"

        tasks = [(name, sym, tf_resample) for name, sym in ALL_SYMBOLS.items()]
        all_sigs = []
        n_ok = 0

        with ProcessPoolExecutor(max_workers=8) as pool:
            futs = {pool.submit(_gen_one, t): t[0] for t in tasks}
            done = 0
            for f in as_completed(futs):
                name, sigs = f.result()
                done += 1
                if sigs is not None and len(sigs) > 0:
                    sigs["strategy"] = tf_label
                    sigs["hold_bars"] = 4
                    if "confidence" in sigs.columns:
                        sigs = sigs[sigs["confidence"] >= 0.08].reset_index(drop=True)
                    if len(sigs) > 0:
                        all_sigs.append(sigs)
                        n_ok += 1
                if done % 20 == 0:
                    elapsed = time.time() - t1
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (len(tasks) - done) / rate / 60 if rate > 0 else 0
                    print(f"    {done}/{len(tasks)} done ({n_ok} ok, "
                          f"{elapsed:.0f}s, ETA {eta:.0f}min)", flush=True)

        if all_sigs:
            merged = pd.concat(all_sigs, ignore_index=True)
            merged.to_parquet(cache_path)
            print(f"  {tf_label}: {n_ok}/{len(ALL_SYMBOLS)} assets, "
                  f"{len(merged):,} signals, saved ({time.time()-t1:.0f}s)")
        else:
            print(f"  {tf_label}: no signals")

        gc.collect()

    print(f"\nDone in {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
