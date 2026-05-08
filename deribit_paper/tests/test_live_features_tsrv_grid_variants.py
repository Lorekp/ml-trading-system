"""S-1c (parity-r3 cycle 6 — 2026-05-08) — TSRV-grid lag/EMA variants.

Locks the bar-grid alignment of the live ``tsrv_mean_1d``,
``tsrv_mean_1w``, ``tsrv_ffd``, ``up_jump_frac``, ``bpv_tsrv_ratio``,
``down_jump_frac``, and ``noise_level`` lag/EMA variants against an
inline pandas reference computed on the same bar-grid series.

The byte-equal-to-offline-parquet check happens later in the
``test_transvol_live_offline_parity.py`` end-to-end gate; these tests
verify the unit-level computation against an inline reference.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.tsrv_features import (
    TSRVFeatures,
    _BARS_PER_DAY,
    _BARS_PER_WEEK,
)
from deribit_paper.features.tsrv_history import TSRVHistoryBuffer
from deribit_paper.features.tsrv_live import BarTSRV


def _make_bar_tsrv(
    *,
    vol_ann: float,
    up_jump: float = 0.05,
    down_jump: float = 0.04,
    bpv: float = 0.045,
    rv_slow: float = 0.05,
    rv_fast: float = 0.04,
    n_ticks: int = 1000,
) -> BarTSRV:
    return BarTSRV(
        tsrv=vol_ann ** 2 * 1e-4, rv_fast=rv_fast, rv_slow=rv_slow,
        bpv=bpv, medrv=0.045, continuous_var=0.04, jump_var=0.005,
        n_ticks=n_ticks, n_sub_bars=10, n_dash=20.0,
        rs_down=0.02, rs_up=0.02,
        jump_fraction=up_jump + down_jump,
        down_jump_fraction=down_jump,
        up_jump_fraction=up_jump,
        vol_ann=vol_ann, vol_ann_continuous=vol_ann * 0.99,
    )


def _populate(n: int, *, seed: int = 0) -> TSRVHistoryBuffer:
    rng = np.random.default_rng(seed)
    buf = TSRVHistoryBuffer(max_size=max(n + 50, 1000))
    base = pd.Timestamp("2026-04-01", tz="UTC")
    for i in range(n):
        ts = base + pd.Timedelta(minutes=15 * i)
        # Random vol around 0.5; gentle up_jump / down_jump drift
        buf.append_bar(ts, _make_bar_tsrv(
            vol_ann=float(0.4 + 0.2 * rng.random()),
            up_jump=float(0.01 + 0.04 * rng.random()),
            down_jump=float(0.01 + 0.04 * rng.random()),
            bpv=float(0.03 + 0.04 * rng.random()),
            rv_slow=float(0.04 + 0.04 * rng.random()),
            rv_fast=float(0.02 + 0.06 * rng.random()),
            n_ticks=int(500 + 1500 * rng.random()),
        ))
    return buf


# ── tsrv_mean_1d lag1 / ema10 ──────────────────────────────────────


def test_tsrv_mean_1d_lag1_matches_inline():
    buf = _populate(150, seed=11)
    feats = TSRVFeatures(buf)
    ln = np.log(buf.as_series("vol_ann").clip(lower=1e-10))
    expected = ln.rolling(_BARS_PER_DAY, min_periods=24).mean().iloc[-2]
    live = feats.tsrv_mean_1d_lag(1)
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=12)


def test_tsrv_mean_1d_ema10_matches_inline():
    buf = _populate(200, seed=12)
    feats = TSRVFeatures(buf)
    series = feats._tsrv_mean_1d_series()
    expected = series.ewm(span=10, adjust=True, min_periods=1).mean().iloc[-1]
    live = feats.tsrv_mean_1d_ema(10)
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=12)


# ── tsrv_mean_1w lag/ema ───────────────────────────────────────────


def test_tsrv_mean_1w_lag1_matches_inline():
    buf = _populate(800, seed=21)
    feats = TSRVFeatures(buf)
    ln = np.log(buf.as_series("vol_ann").clip(lower=1e-10))
    expected = ln.rolling(_BARS_PER_WEEK, min_periods=168).mean().iloc[-2]
    live = feats.tsrv_mean_1w_lag(1)
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=12)


def test_tsrv_mean_1w_lag3_matches_inline():
    buf = _populate(800, seed=22)
    feats = TSRVFeatures(buf)
    ln = np.log(buf.as_series("vol_ann").clip(lower=1e-10))
    expected = ln.rolling(_BARS_PER_WEEK, min_periods=168).mean().iloc[-4]
    live = feats.tsrv_mean_1w_lag(3)
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=12)


def test_tsrv_mean_1w_ema50_matches_inline():
    buf = _populate(800, seed=23)
    feats = TSRVFeatures(buf)
    series = feats._tsrv_mean_1w_series()
    expected = series.ewm(span=50, adjust=True, min_periods=1).mean().iloc[-1]
    live = feats.tsrv_mean_1w_ema(50)
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=12)


# ── tsrv_ffd lag/ema ───────────────────────────────────────────────


def test_tsrv_ffd_lag1_matches_inline():
    # tsrv_ffd needs >= 1458 bars for d=0.4 (FFD weight count)
    buf = _populate(1600, seed=31)
    feats = TSRVFeatures(buf)
    series = feats._tsrv_ffd_series()
    assert series is not None
    expected = float(series.iloc[-2])
    live = feats.tsrv_ffd_lag(1)
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=12)


def test_tsrv_ffd_ema10_matches_inline():
    buf = _populate(1600, seed=32)
    feats = TSRVFeatures(buf)
    series = feats._tsrv_ffd_series()
    expected = series.ewm(span=10, adjust=True, min_periods=1).mean().iloc[-1]
    live = feats.tsrv_ffd_ema(10)
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=12)


# ── up_jump_frac raw + lag + ema10 ──────────────────────────────────


def test_up_jump_frac_raw_returns_latest():
    buf = _populate(50, seed=41)
    feats = TSRVFeatures(buf)
    expected = float(buf.latest().up_jump_fraction)
    live = feats.up_jump_frac()
    assert live == pytest.approx(expected, abs=1e-15)


def test_up_jump_frac_lag1_matches_inline():
    buf = _populate(100, seed=42)
    feats = TSRVFeatures(buf)
    ser = buf.as_series("up_jump_fraction")
    expected = float(ser.iloc[-2])
    live = feats.up_jump_frac_lag(1)
    assert live == pytest.approx(expected, abs=1e-15)


def test_up_jump_frac_ema10_distinct_from_ema50():
    buf = _populate(100, seed=43)
    feats = TSRVFeatures(buf)
    e10 = feats.up_jump_frac_ema(10)
    e50 = feats.up_jump_frac_ema(50)
    assert e10 is not None and e50 is not None
    assert e10 != pytest.approx(e50, abs=1e-9)


# ── bpv_tsrv_ratio raw + lag + ema10 ───────────────────────────────


def test_bpv_tsrv_ratio_raw_matches_inline():
    buf = _populate(50, seed=51)
    feats = TSRVFeatures(buf)
    bpv = buf.as_series("bpv")
    rv_slow = buf.as_series("rv_slow")
    expected = float((bpv / rv_slow.clip(lower=1e-20)).iloc[-1])
    live = feats.bpv_tsrv_ratio()
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=15)


def test_bpv_tsrv_ratio_lag1_matches_inline():
    buf = _populate(100, seed=52)
    feats = TSRVFeatures(buf)
    bpv = buf.as_series("bpv")
    rv_slow = buf.as_series("rv_slow")
    expected = float((bpv / rv_slow.clip(lower=1e-20)).iloc[-2])
    live = feats.bpv_tsrv_ratio_lag(1)
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=15)


def test_bpv_tsrv_ratio_ema10_distinct_from_ema50():
    buf = _populate(100, seed=53)
    feats = TSRVFeatures(buf)
    e10 = feats.bpv_tsrv_ratio_ema(10)
    e50 = feats.bpv_tsrv_ratio_ema(50)
    assert e10 is not None and e50 is not None
    assert e10 != pytest.approx(e50, abs=1e-9)


# ── down_jump_frac raw + lag + ema ─────────────────────────────────


def test_down_jump_frac_raw_returns_latest():
    buf = _populate(50, seed=61)
    feats = TSRVFeatures(buf)
    expected = float(buf.latest().down_jump_fraction)
    live = feats.down_jump_frac()
    assert live == pytest.approx(expected, abs=1e-15)


def test_down_jump_frac_lag3_matches_inline():
    buf = _populate(100, seed=62)
    feats = TSRVFeatures(buf)
    ser = buf.as_series("down_jump_fraction")
    expected = float(ser.iloc[-4])
    live = feats.down_jump_frac_lag(3)
    assert live == pytest.approx(expected, abs=1e-15)


def test_down_jump_frac_ema50_matches_inline():
    buf = _populate(100, seed=63)
    feats = TSRVFeatures(buf)
    ser = buf.as_series("down_jump_fraction")
    expected = ser.ewm(span=50, adjust=True, min_periods=1).mean().iloc[-1]
    live = feats.down_jump_frac_ema(50)
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=12)


# ── noise_level raw + lag + ema ────────────────────────────────────


def test_noise_level_raw_matches_offline_formula():
    buf = _populate(50, seed=71)
    feats = TSRVFeatures(buf)
    last = buf.latest()
    expected = last.rv_fast / (2.0 * max(last.n_ticks, 1))
    live = feats.noise_level()
    assert live == pytest.approx(expected, abs=1e-15)


def test_noise_level_lag1_matches_inline():
    buf = _populate(100, seed=72)
    feats = TSRVFeatures(buf)
    rv_fast = buf.as_series("rv_fast")
    n_ticks = buf.as_series("n_ticks")
    ratio = rv_fast / (2.0 * n_ticks.clip(lower=1.0))
    expected = float(ratio.iloc[-2])
    live = feats.noise_level_lag(1)
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=15)


def test_noise_level_ema10_matches_inline():
    buf = _populate(100, seed=73)
    feats = TSRVFeatures(buf)
    series = feats._noise_level_series()
    expected = series.ewm(span=10, adjust=True, min_periods=1).mean().iloc[-1]
    live = feats.noise_level_ema(10)
    assert live is not None
    np.testing.assert_almost_equal(live, expected, decimal=12)


def test_noise_level_returns_none_on_empty_buffer():
    buf = TSRVHistoryBuffer(max_size=10)
    feats = TSRVFeatures(buf)
    assert feats.noise_level() is None
    assert feats.noise_level_lag(1) is None
    assert feats.noise_level_ema(10) is None
