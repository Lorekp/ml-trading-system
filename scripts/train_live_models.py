#!/usr/bin/env python
"""Train per-asset per-TF walk-forward models for live paper trading.

Faithful to circuit_breaker_v2_experiment.py:91-113 (_gen_one):
- Same SystemConfig (C=0.10, elasticnet, asset_logistic, 13 features)
- Same WalkForwardModel with force_monthly_retrain=True
- Same regime filter (4H EMA)
- Saves the LATEST _cached_model + _cached_scaler as joblib artifacts

Output:
    data/live/models/{tf}/{asset}_model.joblib
    data/live/models/{tf}/{asset}_scaler.joblib
    data/live/models/{tf}/metadata.json

Usage:
    PYTHONPATH=src python scripts/train_live_models.py [--smoke]
"""

import sys, os, time, json, warnings
import joblib
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
warnings.filterwarnings("ignore")

from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.model import WalkForwardModel
from signal_engine.filters import apply_regime_filter
from signal_engine.config import (
    SystemConfig, ModelConfig, FeatureConfig, RegimeFilterConfig,
    ConfidenceFilterConfig, SignalConfig, DataConfig,
)

SMOKE = "--smoke" in sys.argv
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
OUTPUT_DIR = Path("data/live/models")

# Features — exact match to circuit_breaker_v2_experiment.py:46-48
OHLCV_FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low",
]

# TFs — same as NEEDED_TFS
TFS = [("15m", "15min"), ("1h", "1h"), ("3h", "3h"), ("4h", "4h"), ("24h", "1D")]

# S8 core + likely expansion assets
CORE_ASSETS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "XRP": "XRPUSDT",
    "ADA": "ADAUSDT", "LINK": "LINKUSDT", "LTC": "LTCUSDT",
    "BNB": "BNBUSDT", "BCH": "BCHUSDT", "ETC": "ETCUSDT",
}

EXPANSION_ASSETS = {
    "SOL": "SOLUSDT", "AVAX": "AVAXUSDT", "DOT": "DOTUSDT",
    "NEAR": "NEARUSDT", "UNI": "UNIUSDT", "AAVE": "AAVEUSDT",
    "HBAR": "HBARUSDT", "TRX": "TRXUSDT", "SUI": "SUIUSDT",
    "APT": "APTUSDT", "ENJ": "ENJUSDT", "SXP": "SXPUSDT",
    "ZEC": "ZECUSDT", "XLM": "XLMUSDT", "FET": "FETUSDT",
    "WLD": "WLDUSDT", "ANKR": "ANKRUSDT", "TAO": "TAOUSDT",
    "ENA": "ENAUSDT", "FTM": "FTMUSDT", "FIL": "FILUSDT",
}

ALL_ASSETS = {**CORE_ASSETS, **EXPANSION_ASSETS}
ELIGIBLE_FILE = Path("data/eligible_symbols.json")


def load_eligible_assets() -> dict[str, str]:
    """Load all 190 eligible assets from discovery file."""
    if not ELIGIBLE_FILE.exists():
        raise FileNotFoundError(f"{ELIGIBLE_FILE} not found — run discover_eligible_symbols.py first")
    import json
    with open(ELIGIBLE_FILE) as f:
        eligible = json.load(f)
    return {e["name"]: e["binance_symbol"] for e in eligible}


def _train_one(args):
    """Train one (asset, TF) pair. Returns (asset, tf, n_signals, error)."""
    name, sym, tf_label, tf_resample = args
    warnings.filterwarnings("ignore")

    out_dir = OUTPUT_DIR / tf_label
    model_path = out_dir / f"{name}_model.joblib"
    scaler_path = out_dir / f"{name}_scaler.joblib"

    # Skip if already trained (allow --force to override)
    if model_path.exists() and "--force" not in sys.argv:
        return name, tf_label, -1, None

    try:
        # Exact same config as circuit_breaker_v2:94-103
        config = SystemConfig(
            model=ModelConfig(
                C=0.10, penalty="elasticnet", solver="saga", max_iter=1000,
                seed=42, fwd_bars=4, signal_model_type="asset_logistic",
            ),
            features=FeatureConfig(
                feature_list=list(OHLCV_FEATURES), winsor_lo=0.01, winsor_hi=0.99,
                vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20,
            ),
            regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
            confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
            signal=SignalConfig(step_size=0.1),
            data=DataConfig(
                data_dirs=[str(d) for d in DATA_DIRS],
                timeframe=tf_resample, regime_timeframe="4h",
            ),
        )

        ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
        ohlcv_tf = resample_ohlcv(ohlcv_1m, tf_resample)
        feat = compute_features(ohlcv_tf, config.features)

        try:
            wf = WalkForwardModel(config, force_monthly_retrain=True)
        except TypeError:
            wf = WalkForwardModel(config)
        sigs = wf.generate_signals_batch(ohlcv_tf, feat, name)

        if len(sigs) == 0:
            return name, tf_label, 0, "no signals"

        # The model and scaler are now cached from the last training window
        model = wf._cached_model
        scaler = wf._cached_scaler

        if model is None or scaler is None:
            return name, tf_label, 0, "no model after training"

        # Save artifacts
        out_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(model, model_path)
        joblib.dump(scaler, scaler_path)

        return name, tf_label, len(sigs), None

    except FileNotFoundError:
        return name, tf_label, 0, "no data"
    except Exception as e:
        return name, tf_label, 0, f"{type(e).__name__}: {str(e)[:80]}"


def _parse_asset_filter():
    """Parse --assets flag: comma-separated asset names (e.g. --assets SOL,AVAX)."""
    for i, arg in enumerate(sys.argv):
        if arg == "--assets" and i + 1 < len(sys.argv):
            names = [n.strip() for n in sys.argv[i + 1].split(",") if n.strip()]
            # Map to symbol: try ALL_ASSETS first, then eligible, then assume {NAME}USDT
            result = {}
            all_eligible = {}
            if ELIGIBLE_FILE.exists():
                try:
                    all_eligible = load_eligible_assets()
                except Exception:
                    pass
            lookup = {**ALL_ASSETS, **all_eligible}
            for name in names:
                if name in lookup:
                    result[name] = lookup[name]
                else:
                    result[name] = f"{name}USDT"
            return result
    return None


def main():
    T0 = time.time()
    asset_filter = _parse_asset_filter()
    if asset_filter is not None:
        assets = asset_filter
    elif SMOKE:
        assets = {"BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT"}
    elif "--all" in sys.argv:
        assets = load_eligible_assets()
    else:
        assets = ALL_ASSETS

    print("=" * 70)
    print("  TRAIN LIVE MODELS (per-asset per-TF)")
    print("=" * 70)
    print(f"  Assets: {len(assets)}")
    print(f"  Timeframes: {len(TFS)}")
    print(f"  Total model pairs: {len(assets) * len(TFS)}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Mode: {'SMOKE' if SMOKE else 'FULL'}")
    print()

    # Build work queue
    work = []
    for name, sym in sorted(assets.items()):
        for tf_label, tf_resample in TFS:
            work.append((name, sym, tf_label, tf_resample))

    # Train in parallel
    n_new = 0
    n_skip = 0
    n_fail = 0
    errors = []

    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_train_one, w): w for w in work}
        done = 0
        for fut in as_completed(futs):
            done += 1
            name, tf_label, n_sigs, err = fut.result()

            if n_sigs == -1:
                n_skip += 1
            elif err:
                n_fail += 1
                errors.append(f"  {name}/{tf_label}: {err}")
            elif n_sigs > 0:
                n_new += 1

            if done % 20 == 0 or done == len(work):
                elapsed = time.time() - T0
                print(f"  {done}/{len(work)} ({n_new} new, {n_skip} cached, "
                      f"{n_fail} fail, {elapsed:.0f}s)", flush=True)

    # Save metadata
    metadata = {
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_assets": len(assets),
        "n_tfs": len(TFS),
        "n_models": n_new + n_skip,
        "features": OHLCV_FEATURES,
        "model_config": {
            "C": 0.10, "penalty": "elasticnet", "solver": "saga",
            "fwd_bars": 4, "signal_model_type": "asset_logistic",
        },
        "assets": list(assets.keys()),
        "timeframes": [tf for tf, _ in TFS],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    # Summary
    print(f"\n  Results:")
    print(f"    New models: {n_new}")
    print(f"    Cached (skipped): {n_skip}")
    print(f"    Failed: {n_fail}")
    if errors:
        print(f"\n  Errors:")
        for e in errors[:10]:
            print(e)
    print(f"\n  Runtime: {time.time() - T0:.0f}s")

    # Verify
    total = 0
    for tf_label, _ in TFS:
        tf_dir = OUTPUT_DIR / tf_label
        if tf_dir.exists():
            models = list(tf_dir.glob("*_model.joblib"))
            total += len(models)
            print(f"    {tf_label}: {len(models)} models")
    print(f"    Total: {total} model files")


if __name__ == "__main__":
    main()
