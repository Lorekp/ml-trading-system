#!/usr/bin/env python
"""Deep-dive into paper trader vs backtest divergences.

For each trade where direction matched but confidence filter diverged,
compare the exact OHLCV bars, features, and model inputs side-by-side.

Usage:
    PYTHONPATH="src;." python experiments/validate_paper_divergence.py
"""

import sys, os, warnings, asyncio
import numpy as np
import pandas as pd
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.config import FeatureConfig
from live.signal_generator import SklearnBackend
from live.bybit_client import BybitClient

MODEL_DIR = Path("data/live/models")
DATA_DIRS = [Path("data/binance-1m")]
CONF_THRESHOLD = 0.08

OHLCV_FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low",
]

# Cases where paper trader passed filter but backtest didn't
DIVERGENCES = [
    # (timestamp, asset, tf_label, tf_resample, paper_side, bt_prob, bt_conf)
    ("2026-04-06 23:30", "DOGE", "15m", "15min", "Buy",  0.5552, 0.0552),
    ("2026-04-06 23:30", "ETC",  "15m", "15min", "Buy",  0.5509, 0.0509),
    ("2026-04-06 23:30", "XRP",  "15m", "15min", "Buy",  0.5532, 0.0532),
    ("2026-04-07 00:00", "ADA",  "24h", "1D",    "Sell", 0.4289, 0.0711),
    ("2026-04-07 00:00", "SOL",  "24h", "1D",    "Sell", 0.4849, 0.0151),
    ("2026-04-07 00:00", "1000PEPE","24h","1D",   "Sell", 0.4789, 0.0211),
    ("2026-04-07 00:00", "TRU",  "15m", "15min", "Buy",  0.5647, 0.0647),
    ("2026-04-07 00:00", "SUI",  "4h",  "4h",    "Sell", 0.4365, 0.0635),
]


async def fetch_bybit_bars(client, symbol, interval, limit=200):
    """Fetch bars from Bybit, return as DataFrame matching Binance format."""
    bars = await client.fetch_klines(symbol, interval, limit)
    if not bars:
        return None
    df = pd.DataFrame(bars)
    df.index = pd.to_datetime(df["timestamp"])
    df = df[["open", "high", "low", "close", "volume"]]
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df


def run_model(backend, asset, tf_label, ohlcv, bar_time, feature_config, label=""):
    """Run the full signal pipeline on given OHLCV data up to bar_time.

    Returns dict with all intermediate values.
    """
    ohlcv_slice = ohlcv.loc[:bar_time]
    if len(ohlcv_slice) < 30:
        return {"error": f"only {len(ohlcv_slice)} bars"}

    try:
        feat = compute_features(ohlcv_slice, feature_config)
    except Exception as e:
        return {"error": f"features failed: {e}"}

    if len(feat) == 0:
        return {"error": "empty features"}

    last_row = feat.iloc[[-1]]
    feature_cols = [c for c in feature_config.feature_list if c in last_row.columns]
    row_values = last_row[feature_cols]

    if row_values.isna().any().any():
        return {"error": "NaN in features"}

    prob = backend.predict(asset, tf_label, row_values.values)
    if prob is None:
        return {"error": "predict returned None"}

    direction = 1.0 if prob > 0.5 else -1.0
    confidence = abs(prob - 0.5)

    return {
        "n_bars": len(ohlcv_slice),
        "last_bar_time": ohlcv_slice.index[-1],
        "last_close": float(ohlcv_slice["close"].iloc[-1]),
        "last_5_close": ohlcv_slice["close"].iloc[-5:].values.tolist(),
        "prob": prob,
        "confidence": confidence,
        "direction": direction,
        "passes_filter": confidence >= CONF_THRESHOLD,
        "features": {c: float(row_values[c].iloc[0]) for c in feature_cols},
    }


async def main():
    print("=" * 100)
    print("  DEEP DIVE: Paper Trader vs Backtest Feature Divergence")
    print("=" * 100)

    backend = SklearnBackend()
    n_models = backend.load_models(MODEL_DIR)
    print(f"  Models loaded: {n_models}")

    feature_config = FeatureConfig(
        feature_list=OHLCV_FEATURES, winsor_lo=0.01, winsor_hi=0.99,
        vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20,
    )

    # Connect to Bybit for live-equivalent data
    client = BybitClient(base_url="https://api.bybit.com", demo=False)

    for ts_str, asset, tf_label, tf_resample, paper_side, bt_prob, bt_conf in DIVERGENCES:
        bar_time = pd.Timestamp(ts_str)
        sym = f"{asset}USDT"

        print(f"\n{'='*100}")
        print(f"  {asset}/{tf_label} @ {ts_str}  (paper: {paper_side}, BT prob={bt_prob:.4f} conf={bt_conf:.4f})")
        print(f"{'='*100}")

        # ── Binance data (backtest source) ──
        try:
            raw_1m = load_binance_1m(sym, DATA_DIRS)
            if raw_1m.index.tz is not None:
                raw_1m.index = raw_1m.index.tz_localize(None)
            binance_ohlcv = resample_ohlcv(raw_1m, tf_resample)
        except Exception as e:
            print(f"  Binance: FAILED to load ({e})")
            continue

        bt_result = run_model(backend, asset, tf_label, binance_ohlcv, bar_time, feature_config, "BT")

        # ── Bybit data (paper trader source) ──
        bybit_ohlcv = await fetch_bybit_bars(client, sym, tf_label, 200)
        if bybit_ohlcv is not None:
            live_result = run_model(backend, asset, tf_label, bybit_ohlcv, bar_time, feature_config, "LIVE")
        else:
            live_result = {"error": "no Bybit data"}

        # ── Compare ──
        if "error" in bt_result:
            print(f"  BT error: {bt_result['error']}")
            continue
        if "error" in live_result:
            print(f"  Live error: {live_result['error']}")
            print(f"  BT only: prob={bt_result['prob']:.6f} conf={bt_result['confidence']:.6f} pass={bt_result['passes_filter']}")
            continue

        # Last bar comparison
        print(f"\n  Last bar timestamp:")
        print(f"    Binance: {bt_result['last_bar_time']}  close={bt_result['last_close']:.6f}")
        print(f"    Bybit:   {live_result['last_bar_time']}  close={live_result['last_close']:.6f}")
        close_diff_bps = abs(bt_result['last_close'] - live_result['last_close']) / bt_result['last_close'] * 10000
        print(f"    Close diff: {close_diff_bps:.1f} bps")

        # Last 5 closes
        print(f"\n  Last 5 closes:")
        print(f"    Binance: {[f'{c:.6f}' for c in bt_result['last_5_close']]}")
        print(f"    Bybit:   {[f'{c:.6f}' for c in live_result['last_5_close']]}")

        # Model output
        print(f"\n  Model output:")
        print(f"    {'':>20s} {'Binance (BT)':>14s} {'Bybit (Live)':>14s} {'Diff':>10s}")
        print(f"    {'Probability':>20s} {bt_result['prob']:>14.6f} {live_result['prob']:>14.6f} {live_result['prob']-bt_result['prob']:>+10.6f}")
        print(f"    {'Confidence':>20s} {bt_result['confidence']:>14.6f} {live_result['confidence']:>14.6f} {live_result['confidence']-bt_result['confidence']:>+10.6f}")
        print(f"    {'Passes 0.08':>20s} {'YES' if bt_result['passes_filter'] else 'NO':>14s} {'YES' if live_result['passes_filter'] else 'NO':>14s}")

        # Feature comparison
        bt_feat = bt_result["features"]
        live_feat = live_result["features"]
        common = sorted(set(bt_feat.keys()) & set(live_feat.keys()))

        print(f"\n  Feature comparison ({len(common)} features):")
        print(f"    {'Feature':>20s} {'Binance':>12s} {'Bybit':>12s} {'Diff':>12s} {'Rel%':>8s}")
        for f in common:
            bv = bt_feat[f]
            lv = live_feat[f]
            diff = lv - bv
            rel = abs(diff / bv * 100) if abs(bv) > 1e-10 else 0
            flag = " <<<" if rel > 5 else ""
            print(f"    {f:>20s} {bv:>12.6f} {lv:>12.6f} {diff:>+12.6f} {rel:>7.1f}%{flag}")

        # Scaled feature comparison (what the model actually sees)
        print(f"\n  Scaled model input (after StandardScaler):")
        model_key = (tf_label, asset)
        if model_key in backend._models:
            _, scaler = backend._models[model_key]
            bt_scaled = scaler.transform(pd.DataFrame([bt_feat])[common].values)
            live_scaled = scaler.transform(pd.DataFrame([live_feat])[common].values)
            print(f"    {'Feature':>20s} {'BT scaled':>12s} {'Live scaled':>12s} {'Diff':>12s}")
            for i, f in enumerate(common):
                diff = live_scaled[0][i] - bt_scaled[0][i]
                flag = " <<<" if abs(diff) > 0.1 else ""
                print(f"    {f:>20s} {bt_scaled[0][i]:>12.4f} {live_scaled[0][i]:>12.4f} {diff:>+12.4f}{flag}")

    await client.close()
    print(f"\n{'='*100}")
    print("  Done")
    print(f"{'='*100}")


if __name__ == "__main__":
    asyncio.run(main())
