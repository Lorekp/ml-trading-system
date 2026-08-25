#!/usr/bin/env python
"""Compare signal quality: Binance vs Bybit live data.

Fetches the same bars from both exchanges right now, computes features,
runs models, and compares probabilities/confidences against each other
and against the backtest (Binance historical).

Usage:
    PYTHONPATH="src;." python experiments/compare_exchange_signals.py
"""

import sys, os, warnings, asyncio
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

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

ASSETS = ["BTC", "ETH", "XRP", "SOL", "DOGE", "ADA", "ETC", "LTC",
          "LINK", "BNB", "SUI", "TRU", "1000PEPE"]
TFS = [("15m", "15min"), ("1h", "1h"), ("4h", "4h")]


async def fetch_binance_klines(symbol, interval, limit=200):
    """Fetch klines from Binance futures (public, no auth)."""
    import aiohttp
    interval_map = {"15m": "15m", "1h": "1h", "4h": "4h", "24h": "1d"}
    bi = interval_map.get(interval, interval)
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval={bi}&limit={limit}"
    async with aiohttp.ClientSession() as session:
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            data = await resp.json()
    if not isinstance(data, list):
        return None
    rows = []
    for k in data:
        rows.append({
            "timestamp": pd.Timestamp(k[0], unit="ms", tz="UTC").tz_localize(None),
            "open": float(k[1]), "high": float(k[2]),
            "low": float(k[3]), "close": float(k[4]),
            "volume": float(k[5]),
        })
    df = pd.DataFrame(rows)
    df.index = df["timestamp"]
    return df[["open", "high", "low", "close", "volume"]]


async def fetch_bybit_klines(client, symbol, interval, limit=200):
    bars = await client.fetch_klines(symbol, interval, limit)
    if not bars:
        return None
    df = pd.DataFrame(bars)
    df.index = pd.to_datetime(df["timestamp"])
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["open", "high", "low", "close", "volume"]]


def run_signal(backend, asset, tf_label, ohlcv, feature_config):
    """Run model on OHLCV, return (prob, conf, direction, features_dict) or None."""
    if ohlcv is None or len(ohlcv) < 30:
        return None
    if not backend.has_model(asset, tf_label):
        return None

    try:
        feat = compute_features(ohlcv, feature_config)
    except Exception:
        return None
    if len(feat) == 0:
        return None

    last_row = feat.iloc[[-1]]
    feature_cols = [c for c in feature_config.feature_list if c in last_row.columns]
    if len(feature_cols) < len(feature_config.feature_list) * 0.8:
        return None
    row_values = last_row[feature_cols]
    if row_values.isna().any().any():
        return None

    prob = backend.predict(asset, tf_label, row_values.values)
    if prob is None:
        return None

    features = {c: float(row_values[c].iloc[0]) for c in feature_cols}
    return prob, abs(prob - 0.5), 1.0 if prob > 0.5 else -1.0, features


async def main():
    print("=" * 110)
    print("  EXCHANGE COMPARISON: Binance Live vs Bybit Live vs Binance Historical")
    print("=" * 110)

    backend = SklearnBackend()
    backend.load_models(MODEL_DIR)

    feature_config = FeatureConfig(
        feature_list=OHLCV_FEATURES, winsor_lo=0.01, winsor_hi=0.99,
        vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20,
    )

    bybit = BybitClient(base_url="https://api.bybit.com", demo=False)

    # Also load Binance historical for comparison
    hist_cache = {}
    for asset in ASSETS:
        sym = f"{asset}USDT"
        try:
            raw = load_binance_1m(sym, DATA_DIRS)
            if raw.index.tz is not None:
                raw.index = raw.index.tz_localize(None)
            for tf_label, tf_resample in TFS:
                hist_cache[(asset, tf_label)] = resample_ohlcv(raw, tf_resample)
        except Exception:
            pass

    results = []

    for tf_label, tf_resample in TFS:
        print(f"\n  {'='*106}")
        print(f"  TF: {tf_label}")
        print(f"  {'Asset':>10s} | {'Binance Live':>30s} | {'Bybit Live':>30s} | {'Binance Hist':>30s} | {'vol_z diff':>10s}")
        print(f"  {'':>10s} | {'prob':>8s} {'conf':>8s} {'dir':>5s} {'pass':>5s} | {'prob':>8s} {'conf':>8s} {'dir':>5s} {'pass':>5s} | {'prob':>8s} {'conf':>8s} {'dir':>5s} {'pass':>5s} | {'BN-BY':>10s}")
        print(f"  {'-'*106}")

        for asset in ASSETS:
            sym = f"{asset}USDT"

            # Fetch from both exchanges
            try:
                bn_live = await fetch_binance_klines(sym, tf_label, 200)
            except Exception:
                bn_live = None
            await asyncio.sleep(0.15)

            try:
                by_live = await fetch_bybit_klines(bybit, sym, tf_label, 200)
            except Exception:
                by_live = None
            await asyncio.sleep(0.15)

            bn_hist = hist_cache.get((asset, tf_label))

            # Run models
            r_bn = run_signal(backend, asset, tf_label, bn_live, feature_config)
            r_by = run_signal(backend, asset, tf_label, by_live, feature_config)
            r_hist = run_signal(backend, asset, tf_label, bn_hist, feature_config)

            def fmt(r):
                if r is None:
                    return f"{'---':>8s} {'---':>8s} {'---':>5s} {'---':>5s}"
                prob, conf, d, _ = r
                side = "L" if d > 0 else "S"
                p = "YES" if conf >= CONF_THRESHOLD else "no"
                return f"{prob:>8.4f} {conf:>8.4f} {side:>5s} {p:>5s}"

            # vol_zscore diff between Binance live and Bybit live
            vz_diff = ""
            if r_bn and r_by:
                vz_bn = r_bn[3].get("vol_zscore", 0)
                vz_by = r_by[3].get("vol_zscore", 0)
                vz_diff = f"{vz_bn - vz_by:>+10.3f}"

            print(f"  {asset:>10s} | {fmt(r_bn)} | {fmt(r_by)} | {fmt(r_hist)} | {vz_diff}")

            if r_bn and r_by:
                results.append({
                    "asset": asset, "tf": tf_label,
                    "bn_prob": r_bn[0], "bn_conf": r_bn[1],
                    "by_prob": r_by[0], "by_conf": r_by[1],
                    "bn_pass": r_bn[1] >= CONF_THRESHOLD,
                    "by_pass": r_by[1] >= CONF_THRESHOLD,
                    "dir_match": (r_bn[2] == r_by[2]),
                    "vol_z_bn": r_bn[3].get("vol_zscore", 0),
                    "vol_z_by": r_by[3].get("vol_zscore", 0),
                    "hist_prob": r_hist[0] if r_hist else None,
                })

    await bybit.close()

    # Summary stats
    df = pd.DataFrame(results)
    if len(df) == 0:
        print("\n  No results to compare")
        return

    print(f"\n{'='*110}")
    print(f"  SUMMARY ({len(df)} asset/TF pairs compared)")
    print(f"{'='*110}")

    print(f"\n  Direction agreement: {df['dir_match'].sum()}/{len(df)} ({df['dir_match'].mean()*100:.0f}%)")

    prob_diff = (df["bn_prob"] - df["by_prob"]).abs()
    conf_diff = (df["bn_conf"] - df["by_conf"]).abs()
    print(f"  Probability diff:   mean={prob_diff.mean():.4f}  median={prob_diff.median():.4f}  max={prob_diff.max():.4f}")
    print(f"  Confidence diff:    mean={conf_diff.mean():.4f}  median={conf_diff.median():.4f}  max={conf_diff.max():.4f}")

    # Filter agreement
    bn_passes = df["bn_pass"].sum()
    by_passes = df["by_pass"].sum()
    both_pass = (df["bn_pass"] & df["by_pass"]).sum()
    bn_only = (df["bn_pass"] & ~df["by_pass"]).sum()
    by_only = (~df["bn_pass"] & df["by_pass"]).sum()
    print(f"\n  Filter passes:  Binance={bn_passes}  Bybit={by_passes}  Both={both_pass}  BN-only={bn_only}  BY-only={by_only}")

    # vol_zscore comparison
    vz_diff = (df["vol_z_bn"] - df["vol_z_by"]).abs()
    print(f"\n  vol_zscore diff:    mean={vz_diff.mean():.3f}  median={vz_diff.median():.3f}  max={vz_diff.max():.3f}")

    # Binance live vs Binance historical
    hist_mask = df["hist_prob"].notna()
    if hist_mask.sum() > 0:
        hist_diff = (df.loc[hist_mask, "bn_prob"] - df.loc[hist_mask, "hist_prob"]).abs()
        print(f"\n  Binance Live vs Hist: mean={hist_diff.mean():.4f}  median={hist_diff.median():.4f}  max={hist_diff.max():.4f}")
        bybit_hist_diff = (df.loc[hist_mask, "by_prob"] - df.loc[hist_mask, "hist_prob"]).abs()
        print(f"  Bybit  Live vs Hist: mean={bybit_hist_diff.mean():.4f}  median={bybit_hist_diff.median():.4f}  max={bybit_hist_diff.max():.4f}")

    print(f"\n{'='*110}")
    if prob_diff.mean() < 0.005:
        print("  VERDICT: Exchanges produce nearly identical signals. Either works.")
    elif bn_passes > by_passes:
        print("  VERDICT: Binance data produces more filter-passing signals (closer to training data).")
    else:
        print("  VERDICT: Bybit data produces more filter-passing signals.")
    print(f"{'='*110}")


if __name__ == "__main__":
    asyncio.run(main())
