"""Unit tests for ``vol_arb/sidecars.py``.

R2 in plan ``C:\\Users\\laure\\.claude\\plans\\live-engine-h15-alignment-2026-04-26\\``.

Contract checks:

1. ``load_kelly_mult`` returns ``None`` for missing path; correctly
   reindexes + clips when the parquet exists.
2. ``load_bar_exec_slip`` forward-fills past values into target index
   (leakage-safe walk-forward semantics).
3. ``load_aligner_betas`` auto-detects JSON vs parquet format.
4. ``lookup_beta_for_month`` falls back to most-recent-prior month when
   exact lookup misses.
5. All loaders are pure: calling twice yields identical outputs.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vol_arb.sidecars import (
    KELLY_CAP,
    KELLY_FALLBACK,
    KELLY_FLOOR,
    SLIP_FALLBACK_USD,
    load_aligner_betas,
    load_bar_exec_slip,
    load_kelly_mult,
    load_kelly_mult_series,
    lookup_beta_for_month,
    lookup_kelly_mult_at,
)


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture
def target_index() -> pd.DatetimeIndex:
    """Eight 15-minute UTC bars, 2026-01-01 00:00 → 01:45."""
    return pd.date_range("2026-01-01 00:00", periods=8, freq="15min", tz="UTC")


@pytest.fixture
def kelly_parquet(tmp_path: Path, target_index: pd.DatetimeIndex) -> Path:
    """Synthetic Kelly sidecar with known multiplier pattern."""
    # First 4 bars are 1.0 (no boost), next 3 are 1.5, last is 3.0 (will clip).
    df = pd.DataFrame(
        {"size_mult": [1.0, 1.0, 1.0, 1.0, 1.5, 1.5, 1.5, 3.0]},
        index=target_index,
    )
    out = tmp_path / "kelly_test.parquet"
    df.to_parquet(out)
    return out


@pytest.fixture
def slip_parquet(tmp_path: Path) -> Path:
    """Synthetic bar_exec_quality with sparse coverage (every other bar)."""
    sparse_idx = pd.date_range(
        "2026-01-01 00:00", periods=4, freq="30min", tz="UTC"
    )
    df = pd.DataFrame(
        {
            "bar_taker_slip_usd": [10.0, 20.0, 15.0, 25.0],
            "implied_spread_usd": [5.0, 10.0, 7.0, 12.0],
        },
        index=sparse_idx,
    )
    out = tmp_path / "exec_test.parquet"
    df.to_parquet(out)
    return out


# ── load_kelly_mult ───────────────────────────────────────────────────────

def test_kelly_returns_none_for_missing_path(target_index):
    assert load_kelly_mult("/no/such/path.parquet", target_index) is None
    assert load_kelly_mult(None, target_index) is None


def test_kelly_loads_and_clips(kelly_parquet, target_index):
    arr = load_kelly_mult(kelly_parquet, target_index)
    assert arr is not None
    assert arr.shape == (8,)
    # First 4 bars: pass-through 1.0.
    np.testing.assert_array_equal(arr[:4], [1.0, 1.0, 1.0, 1.0])
    # Next 3 bars: pass-through 1.5.
    np.testing.assert_array_equal(arr[4:7], [1.5, 1.5, 1.5])
    # Last bar: 3.0 in source clips to KELLY_CAP=2.0.
    assert arr[7] == KELLY_CAP


def test_kelly_nan_falls_back_to_floor(tmp_path, target_index):
    """Bars present in target_index but missing from sidecar -> NaN -> floor."""
    half_idx = target_index[:4]
    df = pd.DataFrame({"size_mult": [1.5] * 4}, index=half_idx)
    p = tmp_path / "k.parquet"
    df.to_parquet(p)

    arr = load_kelly_mult(p, target_index)
    np.testing.assert_array_equal(arr[:4], [1.5, 1.5, 1.5, 1.5])
    # Bars 4-7 not in sidecar -> NaN -> KELLY_FALLBACK (1.0).
    np.testing.assert_array_equal(arr[4:], [KELLY_FALLBACK] * 4)


def test_kelly_idempotent(kelly_parquet, target_index):
    a = load_kelly_mult(kelly_parquet, target_index)
    b = load_kelly_mult(kelly_parquet, target_index)
    np.testing.assert_array_equal(a, b)


# ── load_bar_exec_slip ────────────────────────────────────────────────────

def test_slip_returns_none_for_missing_path(target_index):
    assert load_bar_exec_slip("/no/such/path.parquet", target_index) is None
    assert load_bar_exec_slip(None, target_index) is None


def test_slip_forward_fills(slip_parquet, target_index):
    """Sparse sidecar covers 00:00, 00:30, 01:00, 01:30; 8 target bars at 15min.

    Forward-fill semantics: each bar uses the most recent known value at or
    before its timestamp.
    """
    arr = load_bar_exec_slip(slip_parquet, target_index)
    assert arr is not None
    assert arr.shape == (8,)
    # 00:00 -> 10.0 (exact)
    # 00:15 -> 10.0 (ffill from 00:00)
    # 00:30 -> 20.0 (exact)
    # 00:45 -> 20.0 (ffill from 00:30)
    # 01:00 -> 15.0 (exact)
    # 01:15 -> 15.0 (ffill)
    # 01:30 -> 25.0 (exact)
    # 01:45 -> 25.0 (ffill)
    expected = [10.0, 10.0, 20.0, 20.0, 15.0, 15.0, 25.0, 25.0]
    np.testing.assert_array_equal(arr, expected)


def test_slip_pre_coverage_uses_fallback(tmp_path):
    """Bars before sidecar's first row -> fillna(SLIP_FALLBACK_USD)."""
    later_idx = pd.date_range(
        "2026-01-01 02:00", periods=2, freq="15min", tz="UTC"
    )
    df = pd.DataFrame({"bar_taker_slip_usd": [42.0, 99.0]}, index=later_idx)
    p = tmp_path / "s.parquet"
    df.to_parquet(p)

    target = pd.date_range(
        "2026-01-01 01:00", periods=6, freq="15min", tz="UTC"
    )
    arr = load_bar_exec_slip(p, target)
    # First 4 bars predate sidecar -> fallback.
    np.testing.assert_array_equal(arr[:4], [SLIP_FALLBACK_USD] * 4)
    # Last 2 bars in coverage.
    np.testing.assert_array_equal(arr[4:], [42.0, 99.0])


def test_slip_missing_column_returns_none(tmp_path, target_index):
    df = pd.DataFrame({"other_col": [1.0]}, index=target_index[:1])
    p = tmp_path / "no_col.parquet"
    df.to_parquet(p)
    assert load_bar_exec_slip(p, target_index) is None


# ── load_aligner_betas ────────────────────────────────────────────────────

def test_betas_returns_empty_for_missing_path():
    assert load_aligner_betas(None) == {}
    assert load_aligner_betas("/no/such/path.json") == {}


def test_betas_loads_json(tmp_path):
    p = tmp_path / "betas.json"
    p.write_text(json.dumps({"2026-01": 0.05, "2026-02": -0.01}))
    out = load_aligner_betas(p)
    assert out == {"2026-01": 0.05, "2026-02": -0.01}


def test_betas_computes_from_parquet(tmp_path):
    """y_pred - y_pred_raw averaged per calendar month."""
    idx = pd.date_range("2026-01-01", periods=4, freq="15D", tz="UTC")
    # 2026-01 has rows 0-1 (Jan 1, Jan 16) → diff +0.05, +0.05 → mean 0.05.
    # 2026-02-01 had Jan-end rollover; row 2 = Jan 31 still in 2026-01.
    # Actually Jan 1 + 15d = Jan 16, +30d = Jan 31, +45d = Feb 15.
    # So 2026-01 has 3 rows, 2026-02 has 1.
    df = pd.DataFrame(
        {
            "y_pred_raw": [-2.0, -2.0, -2.0, -2.0],
            "y_pred":     [-1.95, -1.95, -1.95, -2.10],  # bias +0.05 in Jan, -0.10 in Feb.
        },
        index=idx,
    )
    p = tmp_path / "preds.parquet"
    df.to_parquet(p)

    out = load_aligner_betas(p)
    assert set(out.keys()) == {"2026-01", "2026-02"}
    assert out["2026-01"] == pytest.approx(0.05)
    assert out["2026-02"] == pytest.approx(-0.10)


def test_betas_parquet_missing_columns(tmp_path):
    idx = pd.date_range("2026-01-01", periods=2, freq="15min", tz="UTC")
    df = pd.DataFrame({"only_one_col": [1.0, 2.0]}, index=idx)
    p = tmp_path / "bad.parquet"
    df.to_parquet(p)
    assert load_aligner_betas(p) == {}


# ── lookup_beta_for_month ────────────────────────────────────────────────

def test_lookup_exact_match():
    betas = {"2026-01": 0.05, "2026-02": -0.01, "2026-03": 0.03}
    ts = pd.Timestamp("2026-02-15 12:00", tz="UTC")
    assert lookup_beta_for_month(betas, ts) == -0.01


def test_lookup_falls_back_to_most_recent_prior():
    """Live month not in sidecar — should use March (the largest prior)."""
    betas = {"2026-01": 0.05, "2026-02": -0.01, "2026-03": 0.03}
    ts = pd.Timestamp("2026-04-15 12:00", tz="UTC")  # April not in sidecar.
    assert lookup_beta_for_month(betas, ts) == 0.03


def test_lookup_falls_back_to_zero_when_pre_coverage():
    """Live month is before the sidecar's earliest entry — return 0.0."""
    betas = {"2026-03": 0.03}
    ts = pd.Timestamp("2026-01-15", tz="UTC")
    assert lookup_beta_for_month(betas, ts, fallback_zero_warn=False) == 0.0


def test_lookup_empty_dict_returns_zero():
    ts = pd.Timestamp("2026-01-15", tz="UTC")
    assert lookup_beta_for_month({}, ts, fallback_zero_warn=False) == 0.0


# ── load_kelly_mult_series + lookup_kelly_mult_at ────────────────────────

def test_kelly_mult_series_returns_none_for_missing(target_index):
    assert load_kelly_mult_series(None) is None
    assert load_kelly_mult_series("/no/such/path.parquet") is None


def test_kelly_mult_series_loads(kelly_parquet):
    ser = load_kelly_mult_series(kelly_parquet)
    assert ser is not None
    assert ser.shape == (8,)
    assert ser.iloc[0] == 1.0
    assert ser.iloc[7] == 3.0  # raw value (clipping happens in lookup_kelly_mult_at)


def test_lookup_kelly_at_exact_match(kelly_parquet, target_index):
    ser = load_kelly_mult_series(kelly_parquet)
    # Bar 4: source value 1.5 → in band, no clip.
    assert lookup_kelly_mult_at(ser, target_index[4]) == 1.5
    # Bar 7: source value 3.0 → clipped to KELLY_CAP=2.0.
    assert lookup_kelly_mult_at(ser, target_index[7]) == KELLY_CAP


def test_lookup_kelly_at_missing_timestamp_returns_fallback(kelly_parquet):
    ser = load_kelly_mult_series(kelly_parquet)
    # A timestamp not in the sidecar's index → fallback (1.0).
    miss = pd.Timestamp("2027-01-01 00:00", tz="UTC")
    assert lookup_kelly_mult_at(ser, miss) == KELLY_FALLBACK


def test_lookup_kelly_at_none_series_returns_fallback():
    ts = pd.Timestamp("2026-01-15", tz="UTC")
    assert lookup_kelly_mult_at(None, ts) == KELLY_FALLBACK
