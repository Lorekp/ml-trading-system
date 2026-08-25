#!/usr/bin/env python
"""L2 Feature Validation — Phase 2.

Validates L2 features independently before integrating with the signal model.
Tests: correlation with forward returns, alpha decay, cross-asset consistency.
"""

import sys, os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from signal_engine.l2_data import load_book_depth, list_available_dates
from signal_engine.l2_features import aggregate_book_depth_to_15m
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.config import FeatureConfig

DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
FEAT_CFG = FeatureConfig(
    feature_list=["ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
                   "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
                   "dist_swing_high", "dist_swing_low"],
    winsor_lo=0.01, winsor_hi=0.99, vol_filter_quantile=0.50,
    vol_zscore_window=30, swing_lookback=20,
)


def compute_depth_features_for_symbol(symbol, max_dates=None):
    """Compute all bookDepth-based L2 features for a symbol."""
    dates = list_available_dates(symbol, "bookDepth")
    if not dates:
        return pd.DataFrame()
    if max_dates:
        dates = dates[:max_dates]

    all_bars = []
    for date in dates:
        try:
            raw = load_book_depth(symbol, date)
            bars = aggregate_book_depth_to_15m(raw)
            all_bars.append(bars)
        except Exception:
            pass

    if not all_bars:
        return pd.DataFrame()
    return pd.concat(all_bars).sort_index()


def main():
    T0 = time.time()
    print("=" * 80)
    print("  L2 FEATURE VALIDATION — Phase 2")
    print("=" * 80)

    # Test on BTC first, then cross-asset
    symbols = ["BTCUSDT", "ETHUSDT", "SOLUSDT", "XRPUSDT", "BNBUSDT"]

    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"  {symbol}")
        print(f"{'='*60}")

        # Load L2 features (bookDepth, ~3s/day)
        t0 = time.time()
        l2 = compute_depth_features_for_symbol(symbol, max_dates=120)  # 4 months
        if l2.empty:
            print(f"  No bookDepth data")
            continue
        print(f"  bookDepth: {len(l2)} bars ({l2.index[0].date()} to {l2.index[-1].date()}) in {time.time()-t0:.0f}s")

        # Load OHLCV + compute forward returns
        try:
            ohlcv_1m = load_binance_1m(symbol.replace("USDT", "") + "USDT", DATA_DIRS)
        except FileNotFoundError:
            try:
                ohlcv_1m = load_binance_1m(symbol, DATA_DIRS)
            except FileNotFoundError:
                print(f"  No OHLCV data")
                continue

        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        ohlcv_features = compute_features(ohlcv_15m, FEAT_CFG)

        # Forward returns at different horizons
        close = ohlcv_15m["close"]
        fwd_1 = close.pct_change(1).shift(-1)
        fwd_4 = close.pct_change(4).shift(-4)

        # Both should be UTC — join directly
        merged = ohlcv_features.join(l2, how="inner")
        merged["fwd_ret_1"] = fwd_1.reindex(merged.index)
        merged["fwd_ret_4"] = fwd_4.reindex(merged.index)
        merged = merged.dropna(subset=["fwd_ret_1", "fwd_ret_4"])

        print(f"  Merged: {len(merged)} bars with both OHLCV + L2 + fwd returns")

        # ============================================================
        # 2a. Feature-return correlations
        # ============================================================
        print(f"\n  --- Rank correlations with 1-bar forward return ---")
        l2_cols = [c for c in merged.columns if c.startswith("l2_")]
        ohlcv_cols = [c for c in FEAT_CFG.feature_list if c in merged.columns]

        all_features = l2_cols + ohlcv_cols
        for feat in sorted(all_features, key=lambda f: abs(merged[f].corr(merged["fwd_ret_1"], method="spearman")), reverse=True):
            r1 = merged[feat].corr(merged["fwd_ret_1"], method="spearman")
            r4 = merged[feat].corr(merged["fwd_ret_4"], method="spearman")
            prefix = "L2" if feat.startswith("l2_") else "  "
            if np.isfinite(r1):
                print(f"  {prefix} {feat:<30} r1={r1:+.4f}  r4={r4:+.4f}")

        # ============================================================
        # 2c. Alpha decay for OFI
        # ============================================================
        print(f"\n  --- Alpha decay (depth_imbalance vs forward returns at different horizons) ---")
        if "l2_depth_imbalance_1pct" in merged.columns:
            close_aligned = close.reindex(merged.index)
            for horizon in [1, 2, 3, 4, 6, 8]:
                fwd = close_aligned.pct_change(horizon).shift(-horizon)
                r = merged["l2_depth_imbalance_1pct"].corr(fwd, method="spearman")
                print(f"    horizon={horizon} bars ({horizon*15}m): r={r:+.4f}")

    # ============================================================
    # 2d. Cross-asset summary
    # ============================================================
    print(f"\n{'='*80}")
    print("  CROSS-ASSET SUMMARY (depth_imbalance_1pct vs fwd_ret_4)")
    print(f"{'='*80}")
    # Already printed per-symbol above

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
