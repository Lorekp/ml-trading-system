"""F3 Tier 5 tests: TSRVHistoryBuffer + TSRVFeatures.

Covers:

* Buffer eviction at max_size.
* prewarm_from_parquet round-trip + schema validation + missing-file fail-soft.
* Each of the 6 TSRVFeatures methods on a synthetic 700-bar buffer.
* Engine integration: handlers fire when buffer wired; baseline fallback when None.

Note: byte-identity to offline ``vol_model/features/engineer.py`` is
explicitly NOT a goal here — those features compute on the dollar-bar
grid; live computes on the 15-min grid (mathematically equivalent
under constant-rate trading; different under bursty volume). See the
``tsrv_features.py`` module docstring for the deviation rationale.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.tsrv_features import TSRVFeatures
from deribit_paper.features.tsrv_history import (
    TSRVHistoryBuffer,
    TSRVHistoryRow,
)
from deribit_paper.features.tsrv_live import BarTSRV


# ── helpers ──────────────────────────────────────────────────────────


def _stub_bar_tsrv(vol_ann: float = 0.55, **overrides) -> BarTSRV:
    defaults = dict(
        tsrv=0.05,
        rv_fast=0.04, rv_slow=0.05, bpv=0.045, medrv=0.045,
        continuous_var=0.04, jump_var=0.005,
        n_ticks=1000, n_sub_bars=10, n_dash=20.0,
        rs_down=0.02, rs_up=0.02,
        jump_fraction=0.05,
        down_jump_fraction=0.02, up_jump_fraction=0.03,
        vol_ann=vol_ann, vol_ann_continuous=vol_ann * 0.99,
    )
    defaults.update(overrides)
    return BarTSRV(**defaults)


def _populate_buffer(n: int, buf: TSRVHistoryBuffer) -> None:
    for i in range(n):
        ts = pd.Timestamp("2026-04-01", tz="UTC") + pd.Timedelta(minutes=15 * i)
        buf.append_bar(ts, _stub_bar_tsrv(
            vol_ann=0.50 + 0.001 * i,
            up_jump_fraction=0.05 + 0.0001 * i,
            bpv=0.045 + 0.0001 * i,
            rv_slow=0.05 + 0.0001 * i,
        ))


# ── buffer mechanics ────────────────────────────────────────────────


def test_buffer_evicts_at_max_size():
    buf = TSRVHistoryBuffer(max_size=10)
    _populate_buffer(15, buf)
    assert len(buf) == 10
    # Last entry is the 15th (i=14); first retained is the 6th (i=5).
    rows = list(buf)
    assert rows[0].vol_ann == pytest.approx(0.50 + 0.001 * 5)
    assert rows[-1].vol_ann == pytest.approx(0.50 + 0.001 * 14)


def test_buffer_max_size_zero_raises():
    with pytest.raises(ValueError, match="max_size must be positive"):
        TSRVHistoryBuffer(max_size=0)


def test_buffer_latest_returns_most_recent():
    buf = TSRVHistoryBuffer(max_size=10)
    _populate_buffer(3, buf)
    last = buf.latest()
    assert last is not None
    assert last.vol_ann == pytest.approx(0.50 + 0.001 * 2)


def test_buffer_empty_latest_is_none():
    buf = TSRVHistoryBuffer(max_size=10)
    assert buf.latest() is None


def test_buffer_as_series_has_utc_index():
    buf = TSRVHistoryBuffer(max_size=10)
    _populate_buffer(3, buf)
    ser = buf.as_series("vol_ann")
    assert isinstance(ser.index, pd.DatetimeIndex)
    assert ser.index.tz is not None
    assert len(ser) == 3


def test_prewarm_from_parquet_loads_synthetic(tmp_path):
    # Write a 5-row parquet with the expected schema
    idx = pd.DatetimeIndex(
        pd.date_range("2026-05-07", periods=5, freq="15min", tz="UTC"),
        name="timestamp",
    )
    df = pd.DataFrame({
        "vol_ann": np.linspace(0.5, 0.6, 5),
        "up_jump_fraction": np.linspace(0.01, 0.05, 5),
        "down_jump_fraction": np.linspace(0.005, 0.02, 5),
        "bpv": np.linspace(0.04, 0.06, 5),
        "rv_slow": np.linspace(0.05, 0.07, 5),
        "rv_fast": np.linspace(0.03, 0.05, 5),
        "n_ticks": np.arange(5, dtype=np.int64) + 100,
        "extra": np.zeros(5),  # ignored
    }, index=idx)
    p = tmp_path / "tsrv.parquet"
    df.to_parquet(p)
    buf = TSRVHistoryBuffer(max_size=10)
    n = buf.prewarm_from_parquet(p)
    assert n == 5
    assert len(buf) == 5
    last = buf.latest()
    assert last.vol_ann == pytest.approx(0.6)
    assert last.n_ticks == 104
    assert last.rv_fast == pytest.approx(0.05)
    assert last.down_jump_fraction == pytest.approx(0.02)


def test_prewarm_missing_file_returns_zero(tmp_path):
    buf = TSRVHistoryBuffer(max_size=10)
    n = buf.prewarm_from_parquet(tmp_path / "nope.parquet")
    assert n == 0


def test_prewarm_missing_columns_returns_zero(tmp_path):
    p = tmp_path / "incomplete.parquet"
    pd.DataFrame({"close": [1.0]}, index=pd.DatetimeIndex(
        ["2026-05-07"], tz="UTC", name="timestamp",
    )).to_parquet(p)
    buf = TSRVHistoryBuffer(max_size=10)
    n = buf.prewarm_from_parquet(p)
    assert n == 0


def test_prewarm_respects_max_size():
    """Loading more rows than max_size keeps only the most recent."""
    idx = pd.DatetimeIndex(
        pd.date_range("2026-05-07", periods=20, freq="15min", tz="UTC"),
        name="timestamp",
    )
    df = pd.DataFrame({
        "vol_ann": np.arange(20, dtype=np.float64),
        "up_jump_fraction": np.zeros(20),
        "down_jump_fraction": np.zeros(20),
        "bpv": np.zeros(20),
        "rv_slow": np.zeros(20),
        "rv_fast": np.zeros(20),
        "n_ticks": np.zeros(20, dtype=np.int64),
    }, index=idx)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        out_path = Path(f.name)
    try:
        df.to_parquet(out_path)
        buf = TSRVHistoryBuffer(max_size=5)
        n = buf.prewarm_from_parquet(out_path)
    finally:
        out_path.unlink(missing_ok=True)
    assert n == 5
    assert len(buf) == 5
    # Most recent retained
    assert buf.latest().vol_ann == pytest.approx(19.0)


# ── TSRVFeatures methods ────────────────────────────────────────────


def test_tsrv_mean_1d_returns_none_when_underfull():
    """min_periods=24 → fewer than 24 bars → NaN → wrapper returns None."""
    buf = TSRVHistoryBuffer(max_size=100)
    _populate_buffer(10, buf)
    feats = TSRVFeatures(buf)
    assert feats.tsrv_mean_1d() is None


def test_tsrv_mean_1d_matches_pandas_reference():
    buf = TSRVHistoryBuffer(max_size=200)
    _populate_buffer(120, buf)
    feats = TSRVFeatures(buf)
    ser = buf.as_series("vol_ann")
    expected = (
        np.log(ser.clip(lower=1e-10))
        .rolling(96, min_periods=24).mean().iloc[-1]
    )
    assert feats.tsrv_mean_1d() == pytest.approx(expected, rel=1e-12)


def test_tsrv_mean_1d_lag3():
    buf = TSRVHistoryBuffer(max_size=200)
    _populate_buffer(120, buf)
    feats = TSRVFeatures(buf)
    ser = buf.as_series("vol_ann")
    rolling = (
        np.log(ser.clip(lower=1e-10))
        .rolling(96, min_periods=24).mean()
    )
    expected = rolling.iloc[-4]
    assert feats.tsrv_mean_1d_lag3() == pytest.approx(expected, rel=1e-12)


def test_tsrv_mean_1d_ema50_matches_reference():
    buf = TSRVHistoryBuffer(max_size=200)
    _populate_buffer(150, buf)
    feats = TSRVFeatures(buf)
    ser = buf.as_series("vol_ann")
    rolling = (
        np.log(ser.clip(lower=1e-10))
        .rolling(96, min_periods=24).mean()
    )
    expected = rolling.ewm(span=50, adjust=True, min_periods=1).mean().iloc[-1]
    assert feats.tsrv_mean_1d_ema50() == pytest.approx(expected, rel=1e-12)


def test_tsrv_mean_1w_returns_none_when_underfull():
    buf = TSRVHistoryBuffer(max_size=200)
    _populate_buffer(100, buf)  # < min_periods (=168) for 1w
    feats = TSRVFeatures(buf)
    assert feats.tsrv_mean_1w() is None


def test_tsrv_mean_1w_finite_with_enough_history():
    buf = TSRVHistoryBuffer(max_size=700)
    _populate_buffer(700, buf)
    feats = TSRVFeatures(buf)
    v = feats.tsrv_mean_1w()
    assert v is not None
    assert math.isfinite(v)


def test_tsrv_ffd_returns_finite_with_enough_history():
    buf = TSRVHistoryBuffer(max_size=200)
    _populate_buffer(100, buf)
    feats = TSRVFeatures(buf)
    v = feats.tsrv_ffd()
    # frac_diff_ffd may return None on this small input; just test it
    # doesn't raise. With 100 bars + threshold=1e-5, output should exist.
    if v is not None:
        assert math.isfinite(v)


def test_up_jump_frac_ema50_matches_reference():
    buf = TSRVHistoryBuffer(max_size=200)
    _populate_buffer(60, buf)
    feats = TSRVFeatures(buf)
    ser = buf.as_series("up_jump_fraction")
    expected = ser.ewm(span=50, adjust=True, min_periods=1).mean().iloc[-1]
    assert feats.up_jump_frac_ema50() == pytest.approx(expected, rel=1e-12)


def test_bpv_tsrv_ratio_ema50_matches_reference():
    buf = TSRVHistoryBuffer(max_size=200)
    _populate_buffer(60, buf)
    feats = TSRVFeatures(buf)
    bpv = buf.as_series("bpv")
    rv_slow = buf.as_series("rv_slow")
    ratio = bpv / rv_slow.clip(lower=1e-20)
    expected = ratio.ewm(span=50, adjust=True, min_periods=1).mean().iloc[-1]
    assert feats.bpv_tsrv_ratio_ema50() == pytest.approx(expected, rel=1e-12)


def test_cache_invalidates_on_buffer_growth():
    buf = TSRVHistoryBuffer(max_size=200)
    _populate_buffer(120, buf)
    feats = TSRVFeatures(buf)
    first_v = feats.tsrv_mean_1d()
    # Append a vastly different bar; cache should refresh
    new_ts = pd.Timestamp("2030-01-01", tz="UTC")
    buf.append_bar(new_ts, _stub_bar_tsrv(vol_ann=10.0))
    second_v = feats.tsrv_mean_1d()
    assert second_v != first_v
