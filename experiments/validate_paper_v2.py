#!/usr/bin/env python
"""Validate paper trader: regenerate signals from fresh Binance April data.

Uses the SAME models as the homeserver, SAME Binance 1m data (April 2026),
runs the full signal pipeline at every 15m bar, and compares trade-by-trade.

Usage:
    PYTHONPATH="src;." python experiments/validate_paper_v2.py
"""

import sys, os, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.config import FeatureConfig
from live.signal_generator import SklearnBackend

try:
    from afml.bet_sizing import get_signal, discretize_signal
except ImportError:
    def get_signal(prob, num_classes=2): return 2.0 * prob - 1.0
    def discretize_signal(sig, step_size=0.1): return round(sig / step_size) * step_size

MODEL_DIR = Path("data/live/models")
DATA_DIRS = [Path("data/binance-1m")]
CONF_THRESHOLD = 0.08
REGIME_EMA_SPAN = 20

OHLCV_FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low",
]

TF_CONFIG = [
    ("15m", "15min", 15,  2),
    ("1h",  "1h",    60,  4),
    ("3h",  "3h",    180, 16),
    ("4h",  "4h",    240, 6),
    ("24h", "1D",    1440, 12),
]

# All assets that traded on the paper trader
ASSETS = [
    "DOGE", "ETC", "HYPE", "JOE", "PAXG", "RED", "XRP",
]

# Binance paper trader open trades — paper_binance_fresh.db (POST-FIX)
# Apr 8 11:45 to 15:15 UTC — after fetch_tfs |= {"4h"} fix deployed
PAPER_OPENS = [
    ("2026-04-08 11:45", "JOE",   "Buy",  "15m"),
    ("2026-04-08 12:00", "JOE",   "Buy",  "1h"),
    ("2026-04-08 12:00", "HYPE",  "Buy",  "3h"),
    ("2026-04-08 12:30", "JOE",   "Buy",  "15m"),
    ("2026-04-08 12:45", "JOE",   "Buy",  "15m"),
    ("2026-04-08 13:00", "JOE",   "Buy",  "15m"),
    ("2026-04-08 13:00", "JOE",   "Buy",  "1h"),
    ("2026-04-08 13:00", "RED",   "Buy",  "1h"),
    ("2026-04-08 13:00", "PAXG",  "Buy",  "1h"),
    ("2026-04-08 13:30", "JOE",   "Buy",  "15m"),
    ("2026-04-08 13:45", "JOE",   "Buy",  "15m"),
    ("2026-04-08 14:00", "JOE",   "Buy",  "1h"),
    ("2026-04-08 14:15", "JOE",   "Buy",  "15m"),
    ("2026-04-08 14:45", "XRP",   "Buy",  "15m"),
    ("2026-04-08 15:00", "JOE",   "Buy",  "1h"),
    ("2026-04-08 15:00", "ETC",   "Buy",  "15m"),
    ("2026-04-08 15:00", "XRP",   "Buy",  "15m"),
    ("2026-04-08 15:00", "DOGE",  "Buy",  "15m"),
    ("2026-04-08 15:15", "JOE",   "Buy",  "15m"),
]


def check_regime(direction, ohlcv_4h, ema_span=20):
    if len(ohlcv_4h) < ema_span + 5:
        return True, 0.0
    close = ohlcv_4h["close"]
    ema = close.ewm(span=ema_span, adjust=False).mean()
    regime_raw = np.where(close > ema, 1.0, -1.0)
    regime_shifted = pd.Series(regime_raw, index=ohlcv_4h.index).shift(1)
    latest = regime_shifted.iloc[-1]
    if pd.isna(latest):
        return True, 0.0
    if direction > 0 and latest < 0:
        return False, float(latest)
    if direction < 0 and latest > 0:
        return False, float(latest)
    return True, float(latest)


def main():
    print("=" * 110)
    print("  PAPER TRADER VALIDATION v2 — Fresh April Binance Data + Homeserver Models")
    print("=" * 110)

    backend = SklearnBackend()
    n_models = backend.load_models(MODEL_DIR)
    print(f"  Models: {n_models}")

    fc = FeatureConfig(
        feature_list=OHLCV_FEATURES, winsor_lo=0.01, winsor_hi=0.99,
        vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20,
    )

    # Load + resample all data
    print("  Loading data...")
    ohlcv = {}  # (asset, tf_label) -> DataFrame
    ohlcv_4h = {}  # asset -> DataFrame

    for asset in ASSETS:
        sym = f"{asset}USDT"
        try:
            raw = load_binance_1m(sym, DATA_DIRS)
            if raw.index.tz is not None:
                raw.index = raw.index.tz_localize(None)
        except Exception as e:
            print(f"    {asset}: no data ({e})")
            continue

        for tf_label, tf_resample, _, _ in TF_CONFIG:
            try:
                ohlcv[(asset, tf_label)] = resample_ohlcv(raw, tf_resample)
            except Exception:
                pass
        try:
            ohlcv_4h[asset] = resample_ohlcv(raw, "4h")
        except Exception:
            pass

    print(f"    {len(ohlcv)} asset/TF pairs loaded\n")

    # Build paper trade lookup: (timestamp, asset, tf) -> side
    paper_lookup = {}
    for ts, asset, side, tf in PAPER_OPENS:
        paper_lookup[(ts, asset, tf)] = side

    # Get all unique bar times
    bar_times = sorted(set(t[0] for t in PAPER_OPENS))

    # Run signal generation at each bar
    matches = 0
    dir_matches = 0
    mismatches = 0
    bt_only = 0
    paper_only = 0
    total_paper = len(PAPER_OPENS)

    all_rows = []

    for ts_str in bar_times:
        bar_time = pd.Timestamp(ts_str)
        hour, minute = bar_time.hour, bar_time.minute
        total_min = hour * 60 + minute

        active_tfs = []
        for tf_label, _, tf_min, hold in TF_CONFIG:
            if total_min % tf_min == 0:
                active_tfs.append((tf_label, hold))

        # Paper trades at this bar
        paper_at_bar = {(a, tf): s for (t, a, tf), s in paper_lookup.items() if t == ts_str}

        # BT signals at this bar
        bt_at_bar = {}
        for tf_label, hold in active_tfs:
            for asset in ASSETS:
                df = ohlcv.get((asset, tf_label))
                if df is None or len(df) < 30:
                    continue
                if not backend.has_model(asset, tf_label):
                    continue

                df_slice = df[df.index < bar_time]
                if len(df_slice) < 30:
                    continue

                try:
                    feat = compute_features(df_slice, fc)
                except Exception:
                    continue
                if len(feat) == 0:
                    continue

                last = feat.iloc[[-1]]
                cols = [c for c in fc.feature_list if c in last.columns]
                if len(cols) < len(fc.feature_list) * 0.8:
                    continue
                vals = last[cols]
                if vals.isna().any().any():
                    continue

                prob = backend.predict(asset, tf_label, vals.values)
                if prob is None:
                    continue

                direction = 1.0 if prob > 0.5 else -1.0
                conf = abs(prob - 0.5)

                # Confidence filter
                if conf < CONF_THRESHOLD:
                    continue

                # Regime filter
                df_4h = ohlcv_4h.get(asset)
                if df_4h is not None:
                    df_4h_slice = df_4h[df_4h.index < bar_time]
                    passes, regime_val = check_regime(direction, df_4h_slice)
                    if not passes:
                        continue

                side = "Buy" if direction > 0 else "Sell"
                bt_at_bar[(asset, tf_label)] = (side, prob, conf)

        # Compare
        all_keys = set(paper_at_bar.keys()) | set(bt_at_bar.keys())

        for (asset, tf) in sorted(all_keys):
            p_side = paper_at_bar.get((asset, tf))
            bt_info = bt_at_bar.get((asset, tf))

            if p_side and bt_info:
                bt_side, prob, conf = bt_info
                if p_side == bt_side:
                    matches += 1
                    status = "MATCH"
                else:
                    mismatches += 1
                    status = "MISMATCH"
                all_rows.append((ts_str, asset, tf, p_side, bt_side, f"{prob:.4f}", f"{conf:.4f}", status))
            elif p_side and not bt_info:
                paper_only += 1
                all_rows.append((ts_str, asset, tf, p_side, "---", "---", "---", "PAPER_ONLY"))
            elif bt_info and not p_side:
                bt_side, prob, conf = bt_info
                bt_only += 1
                all_rows.append((ts_str, asset, tf, "---", bt_side, f"{prob:.4f}", f"{conf:.4f}", "BT_ONLY"))

    # Print results
    print(f"  {'Time':>16s} {'Asset':>10s} {'TF':>4s} {'Paper':>6s} {'BT':>6s} {'Prob':>8s} {'Conf':>8s} {'Status':>12s}")
    print(f"  {'-'*80}")
    for row in all_rows:
        print(f"  {row[0]:>16s} {row[1]:>10s} {row[2]:>4s} {row[3]:>6s} {row[4]:>6s} {row[5]:>8s} {row[6]:>8s} {row[7]:>12s}")

    print(f"\n{'='*110}")
    print(f"  SUMMARY")
    print(f"{'='*110}")
    print(f"  Paper trader trades:    {total_paper}")
    print(f"  Full match (both agree):        {matches:>4d} ({matches/total_paper*100:.0f}%)")
    print(f"  Direction mismatch:             {mismatches:>4d} ({mismatches/total_paper*100:.0f}%)")
    print(f"  Paper-only (BT filtered out):   {paper_only:>4d} ({paper_only/total_paper*100:.0f}%)")
    print(f"  BT-only (paper didn't take):    {bt_only:>4d}")
    print(f"{'='*110}")


if __name__ == "__main__":
    main()
