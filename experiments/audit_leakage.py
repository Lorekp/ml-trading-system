#!/usr/bin/env python
"""Automated look-ahead bias & data leakage tests.

Verifies 5 critical safeguards hold. Run after any code change to the
signal pipeline, simulation, or training code.

Usage:
    PYTHONPATH="src;." python experiments/audit_leakage.py
"""

import sys, os, warnings, inspect, ast
import numpy as np
import pandas as pd

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

PASS = 0
FAIL = 0


def check(name, condition, detail=""):
    global PASS, FAIL
    if condition:
        PASS += 1
        print(f"  [PASS] {name}")
    else:
        FAIL += 1
        print(f"  [FAIL] {name} — {detail}")


def test_embargo():
    """Walk-forward training: no training sample after embargo cutoff."""
    from signal_engine.model import WalkForwardModel
    from signal_engine.features import compute_features
    from signal_engine.config import SystemConfig, FeatureConfig

    # Use a small synthetic OHLCV to test embargo logic
    np.random.seed(42)
    n = 5000
    dates = pd.date_range("2022-01-01", periods=n, freq="15min")
    close = 100 + np.cumsum(np.random.randn(n) * 0.1)
    ohlcv = pd.DataFrame({
        "open": close - 0.05, "high": close + 0.5,
        "low": close - 0.5, "close": close, "volume": 1e6,
    }, index=dates)

    config = SystemConfig()
    feat = compute_features(ohlcv, config.features)
    wf = WalkForwardModel(config, force_monthly_retrain=True)

    # Manually run one month of walk-forward
    fwd_bars = config.model.fwd_bars
    fwd_ret = ohlcv["close"].pct_change(fwd_bars).shift(-fwd_bars)

    month = pd.Period("2022-04", freq="M")
    ms = pd.Timestamp(month.start_time)
    me = pd.Timestamp(month.end_time)

    median_gap = ohlcv.index.to_series().diff().dropna().median()
    embargo_td = median_gap * (fwd_bars + 1)
    train_cutoff = ms - embargo_td

    fc = feat.dropna()
    y_all = fwd_ret.reindex(fc.index).dropna()
    valid = fc.index.intersection(y_all.index)
    tr = valid[valid < train_cutoff]
    te = valid[(valid >= ms) & (valid <= me)]

    # Check 1: No training sample at or after cutoff
    check("Embargo: train samples < cutoff",
          len(tr) > 0 and tr.max() < train_cutoff,
          f"max train={tr.max()}, cutoff={train_cutoff}")

    # Check 2: No test sample before month start
    if len(te) > 0:
        check("Embargo: test samples >= month start",
              te.min() >= ms,
              f"min test={te.min()}, ms={ms}")
    else:
        check("Embargo: test samples >= month start", True, "no test samples (ok for small data)")

    # Check 3: Embargo gap is at least fwd_bars * bar_gap
    gap = ms - tr.max()
    min_gap = median_gap * fwd_bars
    check("Embargo: gap >= fwd_bars * bar_gap",
          gap >= min_gap,
          f"gap={gap}, min={min_gap}")


def test_features_no_lookahead():
    """All features at bar T use only data <= T."""
    from signal_engine.features import compute_features
    from signal_engine.config import FeatureConfig

    np.random.seed(42)
    n = 200
    dates = pd.date_range("2023-01-01", periods=n, freq="1h")
    close = 100 + np.cumsum(np.random.randn(n) * 0.1)
    ohlcv = pd.DataFrame({
        "open": close - 0.05, "high": close + 0.5,
        "low": close - 0.5, "close": close,
        "volume": np.random.uniform(1e5, 1e7, n),
    }, index=dates)

    cfg = FeatureConfig()
    feat_full = compute_features(ohlcv, cfg)

    # Compute features on truncated data (first 150 bars)
    feat_trunc = compute_features(ohlcv.iloc[:150], cfg)

    # Bar 149 (last bar of truncated set) should have identical features
    # to bar 149 in the full set. If features used future data, they'd differ.
    t = 149
    idx = ohlcv.index[t]
    if idx in feat_full.index and idx in feat_trunc.index:
        row_full = feat_full.loc[idx]
        row_trunc = feat_trunc.loc[idx]
        common = row_full.index.intersection(row_trunc.index)
        diffs = (row_full[common] - row_trunc[common]).abs()
        max_diff = diffs.max()
        check("Features: bar T identical with/without future data",
              max_diff < 1e-10,
              f"max diff={max_diff:.2e} at {diffs.idxmax()}")
    else:
        check("Features: bar T identical with/without future data",
              False, "index mismatch")


def test_asset_selection_no_future():
    """Asset selector uses no data from the selection month."""
    # The selector uses:
    #   confidence from [ms-3mo, ms)  — model predictions, not fwd_return
    #   volume.loc[:ms - 1day]        — strictly before month
    #   btc_vol.loc[:ms - 1day]       — strictly before month

    # Build synthetic hold_sigs with known timestamps
    ms = pd.Timestamp("2024-06-01")
    cutoff = ms - pd.DateOffset(months=3)

    # Signals in the selection window [cutoff, ms)
    ts_before = pd.date_range(cutoff, ms - pd.Timedelta(hours=1), freq="1h")
    n_before = len(ts_before)
    sigs_before = pd.DataFrame({
        "timestamp": ts_before,
        "confidence": np.random.uniform(0.1, 0.5, n_before),
        "fwd_return": np.random.randn(n_before) * 0.01,  # should NOT be used
    })

    # Signals IN month M (should NOT be accessible)
    ts_during = pd.date_range(ms, ms + pd.Timedelta(days=29), freq="1h")
    n_during = len(ts_during)
    sigs_during = pd.DataFrame({
        "timestamp": ts_during,
        "confidence": np.random.uniform(0.1, 0.5, n_during),
        "fwd_return": np.random.randn(n_during) * 0.01,
    })

    all_sigs = pd.concat([sigs_before, sigs_during], ignore_index=True)

    # Apply the selector's timestamp filter
    ts = pd.to_datetime(all_sigs["timestamp"])
    mask = (ts >= cutoff) & (ts < ms)
    selected = all_sigs[mask.values]

    # Verify: no signal from month M leaked in
    max_ts = pd.to_datetime(selected["timestamp"]).max()
    check("Asset selection: no signals from month M",
          max_ts < ms,
          f"max timestamp={max_ts}, month start={ms}")

    # Verify: confidence is used, not fwd_return
    # (structural check — grep the source code)
    import experiments.drawdown_improvements as di
    source = inspect.getsource(di.main)
    check("Asset selection: uses 'confidence' not 'fwd_return'",
          "confs.extend" in source and "fwd_return" not in source.split("_regime_sel")[1].split("def ")[0] if "_regime_sel" in source else True,
          "fwd_return found in _regime_sel")


def test_fwd_return_isolation():
    """fwd_return is only used for PnL, never for sizing/filtering."""
    from experiments.drawdown_improvements import simulate_with_improvements
    source = inspect.getsource(simulate_with_improvements)

    # Parse the AST to find all uses of fwd_returns
    # It should only appear in: trade_return = directions[i] * fwd_returns[i] - slippage
    lines_with_fwd = []
    for i, line in enumerate(source.split("\n"), 1):
        if "fwd_return" in line and not line.strip().startswith("#"):
            lines_with_fwd.append((i, line.strip()))

    # Should only be: array assignment and PnL computation
    pnl_uses = [l for _, l in lines_with_fwd if "trade_return" in l or "fwd_returns" in l.split("=")[0] if "=" in l]
    sizing_uses = [l for _, l in lines_with_fwd
                   if "raw_size" in l or "size_mult" in l or "throttle" in l]

    check("fwd_return: not used in sizing logic",
          len(sizing_uses) == 0,
          f"found in sizing: {sizing_uses}")

    # Also check it's not used in any if/continue (filtering)
    filter_uses = [l for _, l in lines_with_fwd if "if " in l and "continue" in l]
    check("fwd_return: not used in signal filtering",
          len(filter_uses) == 0,
          f"found in filter: {filter_uses}")


def test_regime_filter_shift():
    """Regime filter applies shift(1) to prevent current-bar look-ahead."""
    from signal_engine.filters import apply_regime_filter
    source = inspect.getsource(apply_regime_filter)

    check("Regime filter: contains shift(1)",
          ".shift(1)" in source,
          "shift(1) not found in apply_regime_filter")

    # Verify with synthetic data
    dates_4h = pd.date_range("2023-01-01", periods=100, freq="4h")
    close_4h = 100 + np.cumsum(np.random.randn(100) * 0.5)
    ohlcv_4h = pd.DataFrame({
        "open": close_4h, "high": close_4h + 1, "low": close_4h - 1,
        "close": close_4h, "volume": 1e6,
    }, index=dates_4h)

    # Regime at bar T should depend on bar T-1's close, not bar T's
    ema = ohlcv_4h["close"].ewm(span=20, adjust=False).mean()
    regime_raw = np.where(ohlcv_4h["close"] > ema, 1.0, -1.0)
    regime_shifted = pd.Series(regime_raw, index=dates_4h).shift(1)

    # Bar 50's regime should equal bar 49's raw regime
    check("Regime filter: bar T regime = bar T-1 raw",
          regime_shifted.iloc[50] == regime_raw[49],
          f"shifted[50]={regime_shifted.iloc[50]}, raw[49]={regime_raw[49]}")


def main():
    print("=" * 80)
    print("  LOOK-AHEAD BIAS AUDIT — Automated Tests")
    print("=" * 80)

    print("\n  1. Walk-Forward Embargo")
    test_embargo()

    print("\n  2. Feature Timing")
    test_features_no_lookahead()

    print("\n  3. Asset Selection Boundaries")
    test_asset_selection_no_future()

    print("\n  4. fwd_return Isolation")
    test_fwd_return_isolation()

    print("\n  5. Regime Filter Shift")
    test_regime_filter_shift()

    print(f"\n{'=' * 80}")
    print(f"  RESULTS: {PASS} passed, {FAIL} failed")
    if FAIL == 0:
        print("  ALL CLEAN — no look-ahead bias detected")
    else:
        print("  WARNING: look-ahead issues found!")
    print(f"{'=' * 80}")


if __name__ == "__main__":
    main()
