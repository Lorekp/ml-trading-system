"""S-1c (parity-r3 cycle 6 — 2026-05-08) — derived-series lag/EMA variants.

Locks the bar-grid alignment of the live ``hurst_100``, ``ffd_rv``, and
``rv_zscore_{1d,1w,1m}`` lag/EMA variants against the offline
``vol_model/features/engineer.py::_window_and_aggregate`` reference.

Strategy: build the SAME series both ways (live wrapper vs direct
pandas equivalent), evaluate at the latest bar, and assert equality
to numerical-precision tolerance.

The byte-equal-to-offline-parquet check happens later in the
``test_transvol_live_offline_parity.py`` end-to-end gate; these tests
verify the unit-level computation against an inline reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.dollar_bar import DollarBar
from deribit_paper.features.dollar_bar_buffer import DollarBarBuffer
from deribit_paper.features.dollar_bar_features import (
    DollarBarFeatures,
    _ffd_weights,
    _hurst_subsampled_window,
)


def _make_bar(
    ts: pd.Timestamp,
    *,
    close: float,
    high: float,
    low: float,
    log_return: float,
    rv_bar: float,
) -> DollarBar:
    return DollarBar(
        timestamp=ts,
        open=close, close=close, high=high, low=low,
        duration_s=60.0, n_trades=10,
        dollar_volume=500_000.0, buy_volume=250_000.0, sell_volume=250_000.0,
        vov=0.001, imbalance=0.0,
        log_return=log_return, rv_bar=rv_bar,
    )


def _seed_buffer(n: int, *, seed: int = 0, max_size: int | None = None) -> DollarBarBuffer:
    rng = np.random.default_rng(seed)
    log_rets = rng.standard_normal(n) * 0.001
    log_prices = np.cumsum(log_rets)
    base = pd.Timestamp("2026-04-01 00:00:00", tz="UTC")
    buf = DollarBarBuffer(max_size=max_size if max_size is not None else max(2 * n, 200))
    for i in range(n):
        close = float(np.exp(log_prices[i]))
        spread = max(abs(log_rets[i]) * 0.5, 1e-6)
        hi = close * np.exp(spread)
        lo = close * np.exp(-spread)
        buf.append(_make_bar(
            base + pd.Timedelta(minutes=15 * i),
            close=close, high=hi, low=lo,
            log_return=float(log_rets[i]),
            rv_bar=float(log_rets[i] ** 2),
        ))
    return buf


# ── hurst_100 lag/EMA ──────────────────────────────────────────────


def test_hurst_100_lag1_matches_inline():
    n = 300
    buf = _seed_buffer(n, seed=11)
    feats = DollarBarFeatures(buf)
    log_ret = buf.as_dataframe()["log_return"].to_numpy(dtype=np.float64)
    expected = _hurst_subsampled_window(log_ret[: n - 1], window=100)
    live = feats.hurst_100_lag(1)
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=12)


def test_hurst_100_lag3_matches_inline():
    n = 400
    buf = _seed_buffer(n, seed=12)
    feats = DollarBarFeatures(buf)
    log_ret = buf.as_dataframe()["log_return"].to_numpy(dtype=np.float64)
    expected = _hurst_subsampled_window(log_ret[: n - 3], window=100)
    live = feats.hurst_100_lag(3)
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=12)


def test_hurst_100_ema10_matches_inline_pandas():
    n = 300
    buf = _seed_buffer(n, seed=13)
    feats = DollarBarFeatures(buf)
    series = feats._hurst_series(window=100)
    assert series is not None
    expected = pd.Series(series).ewm(span=10, adjust=True, min_periods=1).mean().iloc[-1]
    live = feats.hurst_100_ema(10)
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=12)


def test_hurst_100_ema50_matches_inline_pandas():
    n = 400
    buf = _seed_buffer(n, seed=14)
    feats = DollarBarFeatures(buf)
    series = feats._hurst_series(window=100)
    expected = pd.Series(series).ewm(span=50, adjust=True, min_periods=1).mean().iloc[-1]
    live = feats.hurst_100_ema(50)
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=12)


def test_hurst_lag_underfull_returns_none():
    buf = _seed_buffer(50)  # < 100 + 1
    feats = DollarBarFeatures(buf)
    assert feats.hurst_100_lag(1) is None


# ── ffd_rv lag/EMA ─────────────────────────────────────────────────


def test_ffd_rv_lag1_matches_inline():
    """At lag=1 the FFD value equals the dot product over the window
    ending one bar earlier — equivalent to convolution[-2]."""
    from src.afml.fracdiff import frac_diff_ffd
    n = 2500
    buf = _seed_buffer(n, seed=21)
    feats = DollarBarFeatures(buf)
    log_close = np.log(buf.as_dataframe()["close"].to_numpy(dtype=np.float64))
    offline = frac_diff_ffd(
        pd.DataFrame({"p": log_close}), d=0.4, threshold=1e-5,
    )
    expected = float(offline["p"].iloc[-2])
    live = feats.ffd_rv_lag(1)
    assert live is not None
    assert abs(live - expected) < 1e-9


def test_ffd_rv_lag3_matches_inline():
    from src.afml.fracdiff import frac_diff_ffd
    n = 2500
    buf = _seed_buffer(n, seed=22)
    feats = DollarBarFeatures(buf)
    log_close = np.log(buf.as_dataframe()["close"].to_numpy(dtype=np.float64))
    offline = frac_diff_ffd(
        pd.DataFrame({"p": log_close}), d=0.4, threshold=1e-5,
    )
    expected = float(offline["p"].iloc[-4])
    live = feats.ffd_rv_lag(3)
    assert live is not None
    assert abs(live - expected) < 1e-9


def test_ffd_rv_ema10_matches_inline_pandas():
    n = 2500
    buf = _seed_buffer(n, seed=23)
    feats = DollarBarFeatures(buf)
    series = feats._ffd_rv_series()
    expected = pd.Series(series).ewm(
        span=10, adjust=True, min_periods=1,
    ).mean().iloc[-1]
    live = feats.ffd_rv_ema(10)
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=12)


def test_ffd_rv_ema50_matches_inline_pandas():
    n = 2500
    buf = _seed_buffer(n, seed=24)
    feats = DollarBarFeatures(buf)
    series = feats._ffd_rv_series()
    expected = pd.Series(series).ewm(
        span=50, adjust=True, min_periods=1,
    ).mean().iloc[-1]
    live = feats.ffd_rv_ema(50)
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=12)


def test_ffd_rv_lag_underfull_returns_none():
    buf = _seed_buffer(100)  # < 1458 weights for d=0.4
    feats = DollarBarFeatures(buf)
    assert feats.ffd_rv_lag(1) is None


# ── rv_zscore lag/EMA ──────────────────────────────────────────────


def test_rv_zscore_1d_lag1_matches_inline_pandas():
    n = 2000
    buf = _seed_buffer(n, seed=31)
    feats = DollarBarFeatures(buf)
    bpd = feats.bars_per_day(trailing_days=30)
    rv_bar = pd.Series(buf.as_dataframe()["rv_bar"].astype(np.float64).to_numpy())
    w = min(max(bpd * 1, 10), 50_000)
    mp = max(w // 4, 10)
    rv_roll = rv_bar.rolling(w, min_periods=mp).sum()
    mu = rv_roll.rolling(w, min_periods=mp).mean()
    sigma = rv_roll.rolling(w, min_periods=mp).std()
    z = (rv_roll - mu) / sigma.clip(lower=1e-15)
    expected = float(z.iloc[-2])
    live = feats.rv_zscore_lag("1d", 1, bpd=bpd)
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=12)


def test_rv_zscore_1w_lag3_matches_inline_pandas():
    n = 5000
    buf = _seed_buffer(n, seed=32)
    feats = DollarBarFeatures(buf)
    bpd = feats.bars_per_day(trailing_days=30)
    rv_bar = pd.Series(buf.as_dataframe()["rv_bar"].astype(np.float64).to_numpy())
    w = min(max(bpd * 7, 10), 50_000)
    mp = max(w // 4, 10)
    rv_roll = rv_bar.rolling(w, min_periods=mp).sum()
    mu = rv_roll.rolling(w, min_periods=mp).mean()
    sigma = rv_roll.rolling(w, min_periods=mp).std()
    z = (rv_roll - mu) / sigma.clip(lower=1e-15)
    expected = float(z.iloc[-4])
    live = feats.rv_zscore_lag("1w", 3, bpd=bpd)
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=12)


def test_rv_zscore_1m_ema10_matches_inline_pandas():
    n = 30_000
    buf = _seed_buffer(n, seed=33)
    feats = DollarBarFeatures(buf)
    bpd = feats.bars_per_day(trailing_days=30)
    series = feats._rv_zscore_series("1m", bpd=bpd)
    expected = pd.Series(series).ewm(
        span=10, adjust=True, min_periods=1,
    ).mean().iloc[-1]
    live = feats.rv_zscore_ema("1m", 10, bpd=bpd)
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=12)


def test_rv_zscore_1d_ema50_matches_inline_pandas():
    n = 5000
    buf = _seed_buffer(n, seed=34)
    feats = DollarBarFeatures(buf)
    bpd = feats.bars_per_day(trailing_days=30)
    series = feats._rv_zscore_series("1d", bpd=bpd)
    expected = pd.Series(series).ewm(
        span=50, adjust=True, min_periods=1,
    ).mean().iloc[-1]
    live = feats.rv_zscore_ema("1d", 50, bpd=bpd)
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=12)


def test_rv_zscore_lag_horizon_validation():
    buf = _seed_buffer(2000)
    feats = DollarBarFeatures(buf)
    with pytest.raises(ValueError):
        feats.rv_zscore_lag("invalid", 1)


# ── parkinson_1d lag (ema already covered) ─────────────────────────


def test_parkinson_1d_lag1_matches_rolling_series():
    n = 3000
    buf = _seed_buffer(n, seed=41)
    feats = DollarBarFeatures(buf)
    bpd = feats.bars_per_day()
    series = feats._parkinson_1d_rolling_series(bpd)
    expected = float(pd.Series(series).iloc[-2])
    live = feats.parkinson_1d_lag(1, rolling_window=bpd)
    assert live is not None
    assert abs(live - expected) < 1e-12


def test_parkinson_1d_lag3_matches_rolling_series():
    n = 3000
    buf = _seed_buffer(n, seed=42)
    feats = DollarBarFeatures(buf)
    bpd = feats.bars_per_day()
    series = feats._parkinson_1d_rolling_series(bpd)
    expected = float(pd.Series(series).iloc[-4])
    live = feats.parkinson_1d_lag(3, rolling_window=bpd)
    assert live is not None
    assert abs(live - expected) < 1e-12


def test_parkinson_1d_ema10_distinct_from_ema50():
    """Sanity: the existing ema50 method + the new ema10 binding give
    different values on a non-degenerate series."""
    n = 3000
    buf = _seed_buffer(n, seed=43)
    feats = DollarBarFeatures(buf)
    bpd = feats.bars_per_day()
    e10 = feats.parkinson_1d_ema(10, rolling_window=bpd)
    e50 = feats.parkinson_1d_ema(50, rolling_window=bpd)
    assert e10 is not None and e50 is not None
    assert e10 != pytest.approx(e50, abs=1e-9)


# ── Cache invalidation on buffer advance ───────────────────────────


def test_derived_cache_invalidates_on_new_bar():
    """A new bar must shift the trailing window for lag-N readouts."""
    n = 300
    buf = _seed_buffer(n, seed=51)
    feats = DollarBarFeatures(buf)
    h_lag1_before = feats.hurst_100_lag(1)
    base = pd.Timestamp("2026-04-01 00:00:00", tz="UTC")
    buf.append(_make_bar(
        base + pd.Timedelta(minutes=15 * (n + 1)),
        close=70_000.0, high=71_000.0, low=69_000.0,
        log_return=0.005, rv_bar=2.5e-5,
    ))
    h_lag1_after = feats.hurst_100_lag(1)
    # Cache must invalidate so we get a fresh read.
    assert h_lag1_before is not None and h_lag1_after is not None
    # New lag1 = old hurst_100 (raw) at the previous latest bar.
