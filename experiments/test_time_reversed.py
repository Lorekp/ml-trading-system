"""Proper time-reversed backtest.

Reverses the raw OHLCV data, then re-computes ALL features from scratch
on the reversed series. This gives genuinely causal features in the
reversed timeline, making it a valid diagnostic.

If the signal is truly predictive (causal), it should NOT work on
time-reversed data because the causal structure is destroyed.
If it still works, something structural is driving the result.
"""

import sys
sys.path.insert(0, "src")

import logging
import os
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from signal_engine.config import load_config
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.streaming_backtest import StreamingBacktester
from signal_engine.strategies import AlwaysLongStrategy

# Import the strategy from validate script
sys.path.insert(0, "experiments")
from validate_streaming_v2 import ReportAlgorithmStrategy, run_test, fmt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING, format="%(message)s", stream=sys.stdout)

LOOKBACK_MONTHS = 8
FWD_BARS = 4


def reverse_ohlcv(ohlcv):
    """Reverse OHLCV preserving bar structure.

    For each bar, open/close swap (the reversed 'open' is the original 'close'
    and vice versa), and high/low stay as they are (range is symmetric).
    The index is kept ascending but the price data is time-reversed.
    """
    rev = pd.DataFrame(index=ohlcv.index)
    # Reverse the price data — swap open/close since time direction flips
    rev["open"] = ohlcv["close"].values[::-1]
    rev["close"] = ohlcv["open"].values[::-1]
    rev["high"] = ohlcv["high"].values[::-1]
    rev["low"] = ohlcv["low"].values[::-1]
    rev["volume"] = ohlcv["volume"].values[::-1]

    # Ensure high >= max(open, close) and low <= min(open, close)
    rev["high"] = rev[["high", "open", "close"]].max(axis=1)
    rev["low"] = rev[["low", "open", "close"]].min(axis=1)
    return rev


def main():
    t0 = time.time()
    cfg = load_config(Path("configs/default.yaml"))
    data_dirs = [Path(p) for p in cfg.data.data_dirs] or [Path("data/binance-1m")]

    print("Loading BTC 1m data...")
    ohlcv_1m = load_binance_1m("BTCUSDT", data_dirs)

    # ============================================================
    # FORWARD (reference)
    # ============================================================
    print("\n=== FORWARD (reference) ===")
    ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")

    base_features = compute_features(ohlcv_15m, cfg.features)
    regime_parts = []
    for rtf in cfg.close_model.regime_timeframes:
        rtf_ohlcv = resample_ohlcv(ohlcv_1m, rtf)
        rtf_feat = compute_features(rtf_ohlcv, cfg.features)
        aligned = rtf_feat.reindex(ohlcv_15m.index, method="ffill")
        aligned = aligned.add_prefix(f"signal_regime_{rtf}_")
        regime_parts.append(aligned)
    all_features_fwd = pd.concat([base_features] + regime_parts, axis=1)

    close_fwd = ohlcv_15m["close"].to_numpy(dtype=np.float64)
    vol_fwd = ohlcv_15m["close"].pct_change().ewm(span=50, min_periods=20).std().to_numpy(dtype=np.float64)
    timestamps = ohlcv_15m.index

    start_ts = pd.Timestamp("2022-01-01", tz="UTC")
    if timestamps.tz is None:
        start_ts = start_ts.tz_localize(None)
    start_bar = int(timestamps.searchsorted(start_ts))

    print(f"  {all_features_fwd.shape[1]} features, {len(ohlcv_15m)} bars, start_bar={start_bar}")

    engine = StreamingBacktester(
        fwd_bars=FWD_BARS, embargo_bars=FWD_BARS + 1, slippage=0.0001,
        retrain_mode="monthly", lookback_months=LOOKBACK_MONTHS,
        confidence_mode="none", confidence_percentile=0.0,
    )

    r_fwd = run_test("Forward (no conf filter)", engine,
                     ReportAlgorithmStrategy(), all_features_fwd, close_fwd, vol_fwd,
                     timestamps, start_bar)
    print(fmt(r_fwd))

    # ============================================================
    # REVERSED — proper: reverse raw OHLCV, re-compute features
    # ============================================================
    print("\n=== REVERSED (re-computed features from reversed OHLCV) ===")

    # Reverse 1m OHLCV
    ohlcv_1m_rev = reverse_ohlcv(ohlcv_1m)

    # Resample reversed 1m to 15m
    ohlcv_15m_rev = resample_ohlcv(ohlcv_1m_rev, "15min")

    # Compute features from reversed data
    base_features_rev = compute_features(ohlcv_15m_rev, cfg.features)
    regime_parts_rev = []
    for rtf in cfg.close_model.regime_timeframes:
        rtf_ohlcv = resample_ohlcv(ohlcv_1m_rev, rtf)
        rtf_feat = compute_features(rtf_ohlcv, cfg.features)
        aligned = rtf_feat.reindex(ohlcv_15m_rev.index, method="ffill")
        aligned = aligned.add_prefix(f"signal_regime_{rtf}_")
        regime_parts_rev.append(aligned)
    all_features_rev = pd.concat([base_features_rev] + regime_parts_rev, axis=1)

    close_rev = ohlcv_15m_rev["close"].to_numpy(dtype=np.float64)
    vol_rev = ohlcv_15m_rev["close"].pct_change().ewm(span=50, min_periods=20).std().to_numpy(dtype=np.float64)

    print(f"  {all_features_rev.shape[1]} features, {len(ohlcv_15m_rev)} bars")

    r_rev = run_test("Reversed (re-computed features)", engine,
                     ReportAlgorithmStrategy(), all_features_rev, close_rev, vol_rev,
                     timestamps, start_bar)
    print(fmt(r_rev))

    # ============================================================
    # REVERSED — naive (original flawed method for comparison)
    # ============================================================
    print("\n=== REVERSED NAIVE (reversed pre-computed features — flawed method) ===")
    feat_rev_naive = all_features_fwd.iloc[::-1].reset_index(drop=True)
    feat_rev_naive.index = all_features_fwd.index
    close_rev_naive = close_fwd[::-1].copy()
    vol_rev_naive = vol_fwd[::-1].copy()

    r_rev_naive = run_test("Reversed NAIVE (pre-computed)", engine,
                           ReportAlgorithmStrategy(), feat_rev_naive, close_rev_naive,
                           vol_rev_naive, timestamps, start_bar)
    print(fmt(r_rev_naive))

    # ============================================================
    # BASELINES on reversed data
    # ============================================================
    print("\n=== BASELINES ON REVERSED DATA ===")
    engine_simple = StreamingBacktester(fwd_bars=FWD_BARS, embargo_bars=FWD_BARS+1, slippage=0.0001)
    r_al = run_test("Reversed Always-Long", engine_simple, AlwaysLongStrategy(),
                    all_features_rev, close_rev, vol_rev, timestamps, start_bar)
    print(fmt(r_al))

    # ============================================================
    # SUMMARY
    # ============================================================
    print(f"\n{'='*80}")
    print("TIME-REVERSED TEST RESULTS")
    print(f"{'='*80}")
    print(f"  {'Test':<45s} {'Sharpe':>8s} {'Win%':>7s} {'bps':>7s} {'Trades':>8s}")
    print("  " + "-" * 73)
    for r in [r_fwd, r_rev, r_rev_naive, r_al]:
        print(f"  {r['name']:<45s} {r['sharpe']:>8.2f} {r['win_rate']*100:>6.1f}% "
              f"{r['mean_bps']:>6.1f} {r['n_trades']:>8d}")

    print(f"\n  INTERPRETATION:")
    if r_rev["sharpe"] < 3.0 and r_fwd["sharpe"] > 15.0:
        print(f"  Signal is CAUSAL. Forward={r_fwd['sharpe']:.1f}, Reversed={r_rev['sharpe']:.1f}")
        print(f"  (Naive reversed={r_rev_naive['sharpe']:.1f} was inflated by feature/label overlap)")
    elif r_rev["sharpe"] > 10.0:
        print(f"  WARNING: Signal works backwards too. Forward={r_fwd['sharpe']:.1f}, Reversed={r_rev['sharpe']:.1f}")
        print(f"  This suggests a structural property, not genuine forecasting.")
    else:
        print(f"  Mixed result. Forward={r_fwd['sharpe']:.1f}, Reversed={r_rev['sharpe']:.1f}")

    print(f"\n  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
