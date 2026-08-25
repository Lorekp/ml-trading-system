#!/usr/bin/env python
"""Validate paper trader decisions against local backtest environment.

Loads the SAME models the paper trader uses, runs signal generation on
the same data/timeframes, and compares trade-by-trade.

Usage:
    PYTHONPATH="src;." python experiments/validate_paper_trader.py
"""

import sys, os, warnings
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
from live.signal_generator import SklearnBackend, Signal

# ── Config: match paper trader exactly ──
MODEL_DIR = Path("data/live/models")
DATA_DIRS = [Path("data/binance-1m")]
CONF_THRESHOLD = 0.08
REGIME_EMA_SPAN = 20

OHLCV_FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low",
]

TFS = [
    ("15m", "15min", 2),
    ("1h",  "1h",    4),
    ("3h",  "3h",   16),
    ("4h",  "4h",    6),
    ("24h", "1D",   12),
]

ASSETS = [
    "1000PEPE", "ADA", "BTC", "BNB", "BCH", "DOGE", "ETC", "ETH",
    "HYPE", "LINK", "LTC", "RED", "SIREN", "SOL", "STO", "SUI",
    "TRU", "XPL", "XRP",
]

# Paper trader trades (from DB dump)
PAPER_TRADES = [
    # (timestamp_str, asset, side, tf, price)
    ("2026-04-06 23:00", "XRP",  "Buy", "15m", 1.3176),
    ("2026-04-06 23:00", "ETC",  "Buy", "15m", 8.4232),
    ("2026-04-06 23:00", "ETH",  "Buy", "15m", 2100.21),
    ("2026-04-06 23:00", "LTC",  "Buy", "15m", 53.4357),
    ("2026-04-06 23:15", "ETC",  "Buy", "15m", 8.3742),
    ("2026-04-06 23:15", "XRP",  "Buy", "15m", 1.3144),
    ("2026-04-06 23:15", "TRU",  "Buy", "15m", 0.00887),
    ("2026-04-06 23:15", "DOGE", "Buy", "15m", 0.0901),
    ("2026-04-06 23:15", "ETH",  "Buy", "15m", 2094.41),
    ("2026-04-06 23:15", "LTC",  "Buy", "15m", 53.3357),
    ("2026-04-06 23:30", "TRU",  "Buy", "15m", 0.00830),
    ("2026-04-06 23:30", "DOGE", "Buy", "15m", 0.0901),
    ("2026-04-06 23:30", "ETC",  "Buy", "15m", 8.3902),
    ("2026-04-06 23:30", "XRP",  "Buy", "15m", 1.3161),
    ("2026-04-06 23:45", "TRU",  "Buy", "15m", 0.00838),
    ("2026-04-07 00:00", "STO",  "Sell","3h",  0.1505),
    ("2026-04-07 00:00", "TRU",  "Buy", "1h",  0.00835),
    ("2026-04-07 00:00", "ADA",  "Sell","24h", 0.2455),
    ("2026-04-07 00:00", "SUI",  "Sell","24h", 0.8726),
    ("2026-04-07 00:00", "ETC",  "Sell","24h", 8.4338),
    ("2026-04-07 00:00", "SOL",  "Sell","24h", 79.999),
    ("2026-04-07 00:00", "1000PEPE","Sell","24h",0.00332),
    ("2026-04-07 00:00", "TRU",  "Buy", "15m", 0.00833),
    ("2026-04-07 00:00", "SUI",  "Sell","4h",  0.8718),
]


def check_regime(direction, ohlcv_4h, ema_span=20):
    """4H EMA regime filter — same as signal_generator._check_regime."""
    if len(ohlcv_4h) < ema_span + 5:
        return True
    close = ohlcv_4h["close"]
    ema = close.ewm(span=ema_span, adjust=False).mean()
    regime_raw = np.where(close > ema, 1.0, -1.0)
    regime_shifted = pd.Series(regime_raw, index=ohlcv_4h.index).shift(1)
    latest = regime_shifted.iloc[-1]
    if pd.isna(latest):
        return True
    if direction > 0 and latest < 0:
        return False
    if direction < 0 and latest > 0:
        return False
    return True


try:
    from afml.bet_sizing import discretize_signal, get_signal
except ImportError:
    def get_signal(prob, num_classes=2):
        return 2.0 * prob - 1.0
    def discretize_signal(sig, step_size=0.1):
        return round(sig / step_size) * step_size


def main():
    print("=" * 90)
    print("  PAPER TRADER VALIDATION — Trade-by-Trade Comparison")
    print("=" * 90)

    # Load models (same as paper trader)
    backend = SklearnBackend()
    n_models = backend.load_models(MODEL_DIR)
    print(f"  Models loaded: {n_models}")

    feature_config = FeatureConfig(
        feature_list=OHLCV_FEATURES, winsor_lo=0.01, winsor_hi=0.99,
        vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20,
    )

    # Load and resample data for each asset/TF
    print("  Loading data...")
    ohlcv_cache = {}  # (asset, tf_label) -> resampled DataFrame
    ohlcv_4h_cache = {}  # asset -> 4h DataFrame

    for asset in ASSETS:
        sym = f"{asset}USDT"
        try:
            raw_1m = load_binance_1m(sym, DATA_DIRS)
            if raw_1m.index.tz is not None:
                raw_1m.index = raw_1m.index.tz_localize(None)
        except Exception as e:
            print(f"    {asset}: no data ({e})")
            continue

        for tf_label, tf_resample, hold in TFS:
            try:
                resampled = resample_ohlcv(raw_1m, tf_resample)
                ohlcv_cache[(asset, tf_label)] = resampled
            except Exception:
                pass

        # 4H for regime filter
        try:
            ohlcv_4h_cache[asset] = resample_ohlcv(raw_1m, "4h")
        except Exception:
            pass

    print(f"    Loaded {len(ohlcv_cache)} asset/TF pairs, {len(ohlcv_4h_cache)} 4H series")

    # Generate signals at each paper trade timestamp
    print("\n  Generating signals at paper trade timestamps...\n")

    # Group paper trades by timestamp
    trade_times = sorted(set(t[0] for t in PAPER_TRADES))

    all_results = []

    for ts_str in trade_times:
        bar_time = pd.Timestamp(ts_str)
        paper_at_time = [(a, s, tf, p) for t, a, s, tf, p in PAPER_TRADES if t == ts_str]

        # Determine which TFs fire at this bar time
        hour, minute = bar_time.hour, bar_time.minute
        total_min = hour * 60 + minute
        active_tfs = set()
        for tf_label, _, _ in TFS:
            tf_min = {"15m": 15, "1h": 60, "3h": 180, "4h": 240, "24h": 1440}[tf_label]
            if total_min % tf_min == 0:
                active_tfs.add(tf_label)

        # Generate backtest signals for all assets at this bar
        bt_signals = {}  # (asset, tf) -> (direction, prob, conf, filtered_reason)

        for asset in ASSETS:
            for tf_label, tf_resample, hold in TFS:
                if tf_label not in active_tfs:
                    continue

                key = (asset, tf_label)
                ohlcv = ohlcv_cache.get(key)
                if ohlcv is None or len(ohlcv) < 30:
                    continue

                if not backend.has_model(asset, tf_label):
                    continue

                # Slice to bar_time (no lookahead)
                ohlcv_slice = ohlcv.loc[:bar_time]
                if len(ohlcv_slice) < 30:
                    continue

                # Compute features on slice
                try:
                    feat = compute_features(ohlcv_slice, feature_config)
                except Exception:
                    continue
                if len(feat) == 0:
                    continue

                last_row = feat.iloc[[-1]]
                feature_cols = [c for c in feature_config.feature_list if c in last_row.columns]
                if len(feature_cols) < len(feature_config.feature_list) * 0.8:
                    continue
                row_values = last_row[feature_cols]
                if row_values.isna().any().any():
                    continue

                prob = backend.predict(asset, tf_label, row_values.values)
                if prob is None:
                    continue

                direction = 1.0 if prob > 0.5 else -1.0
                confidence = abs(prob - 0.5)

                # Confidence filter
                if confidence < CONF_THRESHOLD:
                    bt_signals[key] = (direction, prob, confidence, f"FILTERED conf={confidence:.4f}")
                    continue

                # Regime filter
                ohlcv_4h = ohlcv_4h_cache.get(asset)
                if ohlcv_4h is not None:
                    ohlcv_4h_slice = ohlcv_4h.loc[:bar_time]
                    if not check_regime(direction, ohlcv_4h_slice, REGIME_EMA_SPAN):
                        bt_signals[key] = (direction, prob, confidence, "REGIME_BLOCK")
                        continue

                bt_signals[key] = (direction, prob, confidence, "PASS")

        # Compare
        print(f"  Bar {ts_str}  (TFs: {sorted(active_tfs)})")
        print(f"  {'Asset':>10s} {'TF':>4s} {'Paper':>6s} {'BT Dir':>7s} {'Prob':>6s} {'Conf':>6s} {'BT Status':>14s} {'Match':>6s}")
        print(f"  {'-'*65}")

        for asset, side, tf, price in paper_at_time:
            paper_dir = "Buy" if side == "Buy" else "Sell"
            key = (asset, tf)
            if key in bt_signals:
                bt_dir, prob, conf, status = bt_signals[key]
                bt_side = "Buy" if bt_dir > 0 else "Sell"
                match = "YES" if (bt_side == paper_dir and status == "PASS") else "DIR" if bt_side == paper_dir else "NO"
                print(f"  {asset:>10s} {tf:>4s} {paper_dir:>6s} {bt_side:>7s} {prob:>6.4f} {conf:>6.4f} {status:>14s} {match:>6s}")
                all_results.append({"ts": ts_str, "asset": asset, "tf": tf, "paper": paper_dir,
                                    "bt": bt_side, "prob": prob, "conf": conf, "status": status, "match": match})
            else:
                print(f"  {asset:>10s} {tf:>4s} {paper_dir:>6s}     ---   (no BT signal generated)")
                all_results.append({"ts": ts_str, "asset": asset, "tf": tf, "paper": paper_dir,
                                    "bt": "---", "prob": 0, "conf": 0, "status": "MISSING", "match": "???"})
        print()

    # Summary
    df = pd.DataFrame(all_results)
    n_total = len(df)
    n_match = (df["match"] == "YES").sum()
    n_dir_match = df["match"].isin(["YES", "DIR"]).sum()
    n_missing = (df["match"] == "???").sum()

    print("=" * 90)
    print(f"  SUMMARY: {n_total} paper trades compared")
    print(f"    Full match (direction + passes filter):  {n_match}/{n_total} ({n_match/n_total*100:.0f}%)")
    print(f"    Direction match (may be filtered in BT): {n_dir_match}/{n_total} ({n_dir_match/n_total*100:.0f}%)")
    print(f"    Missing (no BT signal):                  {n_missing}/{n_total}")
    print("=" * 90)


if __name__ == "__main__":
    main()
