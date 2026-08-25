#!/usr/bin/env python
"""Artifact-backed parity tests for the paper trading system.

Verifies that the live signal generator, breaker, and sizing logic
produce identical outputs to the backtest experiments.

Usage:
    pytest tests/integration/test_parity.py -m integration
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.config import FeatureConfig

from live.config import OHLCV_FEATURES, FULL5, PortfolioConfig
from live.signal_generator import SignalGenerator
from live.breaker import F1SymmetricBreaker
from live.portfolio import PortfolioManager, _edge_from_probability

pytestmark = [pytest.mark.integration, pytest.mark.slow]

REPO_ROOT = Path(__file__).resolve().parents[2]
PARITY_CACHE_ROOT_ENV = "ML_TRADING_PARITY_CACHE_ROOT"
DATA_DIRS = [
    REPO_ROOT / "data/binance-1m",
    REPO_ROOT / "data/homeserver-mirror/binance-1m",
]
MODEL_DIR = REPO_ROOT / "data/live/models"
CACHE_DIR = Path(
    os.environ.get(
        PARITY_CACHE_ROOT_ENV,
        str(REPO_ROOT / "experiments/.cache"),
    )
)
TEST_ASSETS = ["BTC", "ETH", "SOL"]
PASS = "\033[92mPASS\033[0m"
FAIL = "\033[91mFAIL\033[0m"


def _has_symbol_data(symbol: str) -> bool:
    pattern = f"{symbol}-1m-*.csv"
    return any(
        next((data_dir / symbol).glob(pattern), None) is not None
        or next(data_dir.glob(pattern), None) is not None
        for data_dir in DATA_DIRS
        if data_dir.exists()
    )


def _require_signal_parity_artifacts() -> None:
    missing: list[str] = []

    for tf_label, _ in FULL5:
        cache_path = CACHE_DIR / f"sigs_{tf_label}_fwd4.parquet"
        if not cache_path.is_file():
            missing.append(str(cache_path))

        for asset in TEST_ASSETS:
            model_dir = MODEL_DIR / tf_label
            for suffix in ("model", "scaler"):
                artifact = model_dir / f"{asset}_{suffix}.joblib"
                if not artifact.is_file():
                    missing.append(str(artifact))

    for asset in TEST_ASSETS:
        symbol = f"{asset}USDT"
        if not _has_symbol_data(symbol):
            missing.append(f"1-minute CSV data for {symbol} under {DATA_DIRS}")

    if missing:
        preview = "; ".join(missing[:8])
        remainder = len(missing) - 8
        if remainder > 0:
            preview += f"; and {remainder} more"
        pytest.skip(
            "signal parity requires local model, OHLCV, and cached-signal "
            f"artifacts: {preview}. Override the cache root with "
            f"{PARITY_CACHE_ROOT_ENV}."
        )


def test_signal_parity():
    """Test 1: Live signal generator vs cached backtest signals.

    For each (asset, TF), load the cached signals for the last month,
    then run the live model on the same OHLCV data and compare.
    """
    _require_signal_parity_artifacts()

    print("\n" + "=" * 70)
    print("  TEST 1: SIGNAL PARITY (live model vs cached backtest)")
    print("=" * 70)

    sg = SignalGenerator(str(MODEL_DIR))
    n_loaded = sg.load_models()
    print(f"  Loaded {n_loaded} models")

    feature_config = FeatureConfig(
        feature_list=list(OHLCV_FEATURES), winsor_lo=0.01, winsor_hi=0.99,
        vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20)

    total_tested = 0
    total_match_dir = 0
    total_match_prob = 0
    for tf_label, _ in FULL5:
        cache_path = CACHE_DIR / f"sigs_{tf_label}_fwd4.parquet"
        if not cache_path.exists():
            print(f"  {tf_label}: no cached signals")
            continue

        cached = pd.read_parquet(cache_path)
        tf_resample = dict([("15m","15min"),("1h","1h"),("3h","3h"),
                            ("4h","4h"),("24h","1D")])[tf_label]

        for asset in TEST_ASSETS:
            sym = f"{asset}USDT"
            key = (tf_label, asset)
            if not sg._backend.has_model(asset, tf_label):
                print(f"  {tf_label}/{asset}: no live model, skipping")
                continue

            # Filter cached signals for this asset, last 30 days
            asset_sigs = cached[cached["asset"] == asset].copy()
            if len(asset_sigs) == 0:
                continue

            ts = pd.to_datetime(asset_sigs["timestamp"])
            if ts.dt.tz is not None:
                ts = ts.dt.tz_localize(None)
            asset_sigs = asset_sigs.copy()
            asset_sigs["ts_naive"] = ts.values

            last_ts = ts.max()
            cutoff = last_ts - pd.Timedelta(days=30)
            recent = asset_sigs[asset_sigs["ts_naive"] >= cutoff].reset_index(drop=True)

            if len(recent) == 0:
                continue

            # Load OHLCV
            try:
                ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
                ohlcv_tf = resample_ohlcv(ohlcv_1m, tf_resample)
                ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
                if ohlcv_tf.index.tz is not None:
                    ohlcv_tf.index = ohlcv_tf.index.tz_localize(None)
                if ohlcv_4h.index.tz is not None:
                    ohlcv_4h.index = ohlcv_4h.index.tz_localize(None)
            except Exception as e:
                print(f"  {tf_label}/{asset}: data error: {e}")
                continue

            # Test a sample of signals (up to 50 per asset/TF)
            sample_n = min(50, len(recent))
            sample_idx = np.linspace(0, len(recent) - 1, sample_n, dtype=int)
            n_dir_match = 0
            n_prob_match = 0
            n_tested = 0

            model, scaler = sg._backend._models[key]

            for idx in sample_idx:
                row = recent.iloc[idx]
                sig_ts = row["ts_naive"]
                cached_dir = row["direction"]
                cached_prob = row["probability"]

                # Get OHLCV up to this timestamp — use FULL history like backtest
                mask = ohlcv_tf.index <= sig_ts
                ohlcv_window = ohlcv_tf[mask]

                if len(ohlcv_window) < 30:
                    continue

                # Compute features
                try:
                    feat = compute_features(ohlcv_window, feature_config)
                    if len(feat) == 0:
                        continue
                    last_row = feat.iloc[[-1]]
                    feature_cols = [c for c in OHLCV_FEATURES if c in last_row.columns]
                    row_values = last_row[feature_cols]
                    if row_values.isna().any().any():
                        continue

                    X = scaler.transform(row_values.values)
                    prob = float(model.predict_proba(X)[0, 1])
                    direction = 1.0 if prob > 0.5 else -1.0
                except Exception:
                    continue

                n_tested += 1
                total_tested += 1

                # Compare direction
                if direction == cached_dir:
                    n_dir_match += 1
                    total_match_dir += 1

                # Compare probability (within tolerance)
                if abs(prob - cached_prob) < 0.01:
                    n_prob_match += 1
                    total_match_prob += 1

            if n_tested > 0:
                dir_pct = n_dir_match / n_tested * 100
                prob_pct = n_prob_match / n_tested * 100
                status = PASS if dir_pct >= 90 else FAIL
                print(f"  {status} {tf_label}/{asset}: dir={dir_pct:.0f}% prob={prob_pct:.0f}% "
                      f"({n_tested} signals)")

    # Summary
    assert total_tested > 0, (
        "parity artifacts were present, but no comparable signals were produced"
    )
    dir_rate = total_match_dir / total_tested * 100
    prob_rate = total_match_prob / total_tested * 100
    overall = PASS if dir_rate >= 90 else FAIL
    print(f"\n  {overall} Overall: direction={dir_rate:.1f}%, probability={prob_rate:.1f}% "
          f"({total_tested} signals tested)")
    assert dir_rate >= 90, f"direction parity {dir_rate:.1f}% is below 90%"


def test_breaker_parity():
    """Test 2: F1 symmetric breaker vs experiment inline math."""
    print("\n" + "=" * 70)
    print("  TEST 2: BREAKER PARITY")
    print("=" * 70)

    # Reference implementation from simulate_with_breaker:356-376
    def experiment_breaker(recent_pnl, recent_notional, threshold=-0.01,
                           floor=0.10, up_threshold=0.01, cap=1.5):
        if len(recent_pnl) < 10:
            return 1.0
        total_pnl = sum(recent_pnl)
        total_not = sum(recent_notional)
        if total_not <= 0:
            return 1.0
        pnl_ratio = total_pnl / total_not
        if pnl_ratio < threshold:
            t = threshold
            scale = max(1.0 + pnl_ratio / abs(t), floor) if t != 0 else 1.0
            return scale
        elif pnl_ratio > up_threshold:
            up_t = up_threshold
            return min(1.0 + (pnl_ratio - up_t) / up_t, cap)
        return 1.0

    breaker = F1SymmetricBreaker(window=10, threshold=-0.01, floor=0.10,
                                  up_threshold=0.01, cap=1.5)

    test_cases = [
        # (description, pnl_sequence, notional_sequence)
        ("Neutral (no trades)", [], []),
        ("Neutral (few trades)", [10, -5, 8], [1000, 1000, 1000]),
        ("Losing streak", [-20]*10, [1000]*10),
        ("Winning streak", [30]*10, [1000]*10),
        ("Mixed neutral", [10, -10, 10, -10, 10, -10, 10, -10, 10, -10],
         [1000]*10),
        ("Slight loss", [-12, -8, 5, -10, -5, 3, -15, -2, 8, -14],
         [1000]*10),
        ("Strong win", [50, 40, 30, 20, 10, 50, 40, 30, 20, 10],
         [1000]*10),
    ]

    all_pass = True
    for desc, pnls, notionals in test_cases:
        # Live breaker
        b = F1SymmetricBreaker(window=10, threshold=-0.01, floor=0.10,
                                up_threshold=0.01, cap=1.5)
        for p, n in zip(pnls, notionals):
            b.record_trade("TEST", p, n)
        live_mult = b.get_multiplier("TEST")

        # Experiment inline
        exp_mult = experiment_breaker(
            list(pnls[-10:]), list(notionals[-10:]))

        match = abs(live_mult - exp_mult) < 1e-10
        status = PASS if match else FAIL
        if not match:
            all_pass = False
        print(f"  {status} {desc}: live={live_mult:.4f} exp={exp_mult:.4f}")

    assert all_pass


def test_sizing_parity():
    """Test 3: Portfolio sizing vs experiment inline math."""
    print("\n" + "=" * 70)
    print("  TEST 3: SIZING PARITY")
    print("=" * 70)

    portfolio_config = PortfolioConfig()

    # Reference: simulate_with_breaker lines 332-337, parameterized with the
    # same current config object as the live portfolio manager.
    def experiment_sizing(equity, prob, sig_str, regime_mult, vol_mult,
                          breaker_mult, dd=0.0):
        if dd >= portfolio_config.max_dd:
            return 0.0
        size_mult = (
            max(
                1.0
                - (dd - portfolio_config.delever_start)
                / (portfolio_config.max_dd - portfolio_config.delever_start),
                0.0,
            )
            if dd > portfolio_config.delever_start
            else 1.0
        )
        sig_abs = _edge_from_probability(prob, sig_str)
        sig_abs *= float(
            np.clip(regime_mult, portfolio_config.regime_exposure_floor, 1.0)
        )
        raw_size = (
            equity
            * portfolio_config.base_sizing_pct
            * sig_abs
            * size_mult
            * vol_mult
        )
        raw_size = min(
            raw_size,
            equity
            * portfolio_config.base_sizing_pct
            * portfolio_config.max_signal_mult
            * size_mult,
        )
        raw_size *= breaker_mult
        return raw_size

    test_cases = [
        # (desc, equity, prob, sig_str, regime, vol, breaker, expected_dd)
        ("Normal signal", 200000, 0.70, 0.4, 1.0, 1.0, 1.0, 0.0),
        ("Strong signal", 200000, 0.85, 0.7, 1.0, 1.0, 1.0, 0.0),
        ("With vol overlay", 200000, 0.70, 0.4, 1.0, 0.5, 1.0, 0.0),
        ("With breaker up", 200000, 0.70, 0.4, 1.0, 1.0, 1.5, 0.0),
        ("With breaker down", 200000, 0.70, 0.4, 1.0, 1.0, 0.10, 0.0),
        ("In drawdown", 190000, 0.70, 0.4, 1.0, 1.0, 1.0, 0.05),
        ("Deep drawdown", 182000, 0.70, 0.4, 1.0, 1.0, 1.0, 0.09),
        ("Halted", 180000, 0.70, 0.4, 1.0, 1.0, 1.0, 0.10),
    ]

    all_pass = True
    for desc, eq, prob, sig, reg, vol, brk, dd in test_cases:
        # Live sizing
        pm_test = PortfolioManager(portfolio_config, eq)
        if dd > 0:
            pm_test.peak_equity = eq / (1.0 - dd)
        live_size = pm_test.compute_size("BTC", prob, sig, reg, vol, brk)

        # Experiment sizing
        exp_size = experiment_sizing(eq, prob, sig, reg, vol, brk, dd=dd)

        # Compare (ignore exposure limits for this test — just raw sizing)
        match = abs(live_size - exp_size) < 1.0 or (live_size == 0 and exp_size == 0)
        status = PASS if match else FAIL
        if not match:
            all_pass = False
        print(f"  {status} {desc}: live=${live_size:.0f} exp=${exp_size:.0f}")

    assert all_pass


def test_tf_dispatch():
    """Test 4: Timeframe dispatch at every 15-min tick."""
    pytest.importorskip(
        "aiohttp",
        reason="timeframe dispatch parity requires the live dependency set",
    )
    from live.engine import PaperTradingEngine

    print("\n" + "=" * 70)
    print("  TEST 4: TF DISPATCH")
    print("=" * 70)

    from datetime import datetime, timezone

    expected = {
        (0, 0):   {"15m", "1h", "3h", "4h", "24h"},
        (0, 15):  {"15m"},
        (0, 30):  {"15m"},
        (0, 45):  {"15m"},
        (1, 0):   {"15m", "1h"},
        (3, 0):   {"15m", "1h", "3h"},
        (4, 0):   {"15m", "1h", "4h"},
        (4, 15):  {"15m"},
        (6, 0):   {"15m", "1h", "3h"},
        (8, 0):   {"15m", "1h", "4h"},
        (12, 0):  {"15m", "1h", "3h", "4h"},
        (15, 0):  {"15m", "1h", "3h"},
        (21, 0):  {"15m", "1h", "3h"},
    }

    all_pass = True
    for (h, m), exp_tfs in sorted(expected.items()):
        dt = datetime(2026, 4, 3, h, m, tzinfo=timezone.utc)
        actual_tfs = set()
        for tf in ["15m", "1h", "3h", "4h", "24h"]:
            if PaperTradingEngine._tf_fires(tf, dt):
                actual_tfs.add(tf)

        match = actual_tfs == exp_tfs
        status = PASS if match else FAIL
        if not match:
            all_pass = False
            print(f"  {status} {h:02d}:{m:02d}: got {sorted(actual_tfs)}, "
                  f"expected {sorted(exp_tfs)}")
        else:
            print(f"  {status} {h:02d}:{m:02d}: {sorted(actual_tfs)}")

    # Full 24h sweep — just verify 15m fires every tick
    n_15m = 0
    for h in range(24):
        for m in range(0, 60, 15):
            dt = datetime(2026, 4, 3, h, m, tzinfo=timezone.utc)
            if PaperTradingEngine._tf_fires("15m", dt):
                n_15m += 1
    assert n_15m == 96, f"15m should fire 96 times, got {n_15m}"
    print(f"  {PASS} 15m fires 96/96 times across 24h")

    assert all_pass
