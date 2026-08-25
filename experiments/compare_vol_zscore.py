#!/usr/bin/env python
"""Compare vol_zscore between Binance (backtest) and Bybit (live) at each bar.

Shows exactly what the paper trader saw vs what the backtest computes.

Usage:
    PYTHONPATH="src;." python experiments/compare_vol_zscore.py
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

OHLCV_FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low",
]

ASSETS = ["BTC", "ETH", "XRP", "DOGE", "ETC", "TRU", "SOL", "RED"]
TFS = [("15m", "15min"), ("1h", "1h")]


async def fetch_bybit(client, symbol, interval, limit=200):
    bars = await client.fetch_klines(symbol, interval, limit)
    if not bars:
        return None
    df = pd.DataFrame(bars)
    df.index = pd.to_datetime(df["timestamp"])
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    return df[["open", "high", "low", "close", "volume"]]


def get_features(ohlcv, fc):
    if ohlcv is None or len(ohlcv) < 30:
        return None
    try:
        feat = compute_features(ohlcv, fc)
        if len(feat) == 0:
            return None
        return feat.iloc[-1]
    except Exception:
        return None


async def main():
    print("=" * 120)
    print("  vol_zscore COMPARISON: Binance (backtest) vs Bybit (paper trader)")
    print("=" * 120)

    fc = FeatureConfig(
        feature_list=OHLCV_FEATURES, winsor_lo=0.01, winsor_hi=0.99,
        vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20,
    )

    backend = SklearnBackend()
    backend.load_models(MODEL_DIR)

    client = BybitClient(base_url="https://api.bybit.com", demo=False)

    print(f"\n  {'Asset':>10s} {'TF':>4s} | {'BN vol_z':>10s} {'BY vol_z':>10s} {'Diff':>10s} |"
          f" {'BN volume':>12s} {'BY volume':>12s} {'Vol Ratio':>10s} |"
          f" {'BN prob':>8s} {'BY prob':>8s} {'P diff':>8s} |"
          f" {'BN conf':>8s} {'BY conf':>8s} {'BN pass':>7s} {'BY pass':>7s}")
    print(f"  {'-'*140}")

    rows = []
    for tf_label, tf_resample in TFS:
        for asset in ASSETS:
            sym = f"{asset}USDT"

            # Binance
            try:
                raw = load_binance_1m(sym, DATA_DIRS)
                if raw.index.tz is not None:
                    raw.index = raw.index.tz_localize(None)
                bn_ohlcv = resample_ohlcv(raw, tf_resample)
            except Exception:
                continue

            # Bybit
            by_ohlcv = await fetch_bybit(client, sym, tf_label, 200)
            await asyncio.sleep(0.15)

            bn_feat = get_features(bn_ohlcv, fc)
            by_feat = get_features(by_ohlcv, fc)

            if bn_feat is None or by_feat is None:
                continue

            bn_vz = bn_feat.get("vol_zscore", np.nan)
            by_vz = by_feat.get("vol_zscore", np.nan)

            # Get raw volume from last bar
            bn_vol = bn_ohlcv["volume"].iloc[-1] if len(bn_ohlcv) > 0 else 0
            by_vol = by_ohlcv["volume"].iloc[-1] if by_ohlcv is not None and len(by_ohlcv) > 0 else 0
            vol_ratio = by_vol / bn_vol if bn_vol > 0 else 0

            # Run model on both
            bn_prob = by_prob = None
            if backend.has_model(asset, tf_label):
                for src, feat_row, label in [("bn", bn_feat, "BN"), ("by", by_feat, "BY")]:
                    cols = [c for c in fc.feature_list if c in feat_row.index]
                    vals = feat_row[cols].values.reshape(1, -1)
                    if not np.isnan(vals).any():
                        p = backend.predict(asset, tf_label, vals)
                        if label == "BN":
                            bn_prob = p
                        else:
                            by_prob = p

            vz_diff = by_vz - bn_vz if np.isfinite(bn_vz) and np.isfinite(by_vz) else np.nan
            p_diff = (by_prob - bn_prob) if bn_prob is not None and by_prob is not None else np.nan
            bn_conf = abs(bn_prob - 0.5) if bn_prob is not None else np.nan
            by_conf = abs(by_prob - 0.5) if by_prob is not None else np.nan

            bn_pass = "YES" if bn_conf >= 0.08 else "no" if np.isfinite(bn_conf) else "---"
            by_pass = "YES" if by_conf >= 0.08 else "no" if np.isfinite(by_conf) else "---"

            print(f"  {asset:>10s} {tf_label:>4s} | {bn_vz:>+10.3f} {by_vz:>+10.3f} {vz_diff:>+10.3f} |"
                  f" {bn_vol:>12.1f} {by_vol:>12.1f} {vol_ratio:>10.2f}x |"
                  f" {bn_prob:>8.4f} {by_prob:>8.4f} {p_diff:>+8.4f} |"
                  f" {bn_conf:>8.4f} {by_conf:>8.4f} {bn_pass:>7s} {by_pass:>7s}")

            rows.append({
                "asset": asset, "tf": tf_label,
                "bn_vz": bn_vz, "by_vz": by_vz, "vz_diff": vz_diff,
                "bn_vol": bn_vol, "by_vol": by_vol, "vol_ratio": vol_ratio,
                "bn_prob": bn_prob, "by_prob": by_prob, "p_diff": p_diff,
                "bn_conf": bn_conf, "by_conf": by_conf,
            })

    await client.close()

    df = pd.DataFrame(rows)
    vz = df["vz_diff"].dropna()
    vol_r = df["vol_ratio"]
    pd_col = df["p_diff"].dropna().abs()

    print(f"\n{'='*120}")
    print(f"  SUMMARY ({len(df)} comparisons)")
    print(f"{'='*120}")
    print(f"\n  vol_zscore difference (Bybit - Binance):")
    print(f"    Mean:   {vz.mean():+.3f}")
    print(f"    Median: {vz.median():+.3f}")
    print(f"    Std:    {vz.std():.3f}")
    print(f"    |diff| > 0.5:  {(vz.abs() > 0.5).sum()}/{len(vz)} ({(vz.abs() > 0.5).mean()*100:.0f}%)")
    print(f"    |diff| > 1.0:  {(vz.abs() > 1.0).sum()}/{len(vz)} ({(vz.abs() > 1.0).mean()*100:.0f}%)")

    print(f"\n  Raw volume ratio (Bybit / Binance):")
    print(f"    Mean:   {vol_r.mean():.2f}x")
    print(f"    Median: {vol_r.median():.2f}x")
    print(f"    Min:    {vol_r.min():.2f}x")
    print(f"    Max:    {vol_r.max():.2f}x")

    print(f"\n  Probability impact:")
    print(f"    Mean |prob diff|: {pd_col.mean():.4f}")
    print(f"    Max  |prob diff|: {pd_col.max():.4f}")

    # Filter disagreements
    bn_pass = df["bn_conf"] >= 0.08
    by_pass = df["by_conf"] >= 0.08
    agree = (bn_pass == by_pass).sum()
    print(f"\n  Filter agreement: {agree}/{len(df)} ({agree/len(df)*100:.0f}%)")
    print(f"    BN passes, BY doesn't: {(bn_pass & ~by_pass).sum()}")
    print(f"    BY passes, BN doesn't: {(~bn_pass & by_pass).sum()}")
    print(f"    Both pass:             {(bn_pass & by_pass).sum()}")

    # Correlation between vol_zscore diff and prob diff
    valid = df[["vz_diff", "p_diff"]].dropna()
    if len(valid) > 5:
        corr = valid["vz_diff"].corr(valid["p_diff"])
        print(f"\n  Correlation(vol_zscore_diff, prob_diff): {corr:.3f}")

    print(f"\n{'='*120}")


if __name__ == "__main__":
    asyncio.run(main())
