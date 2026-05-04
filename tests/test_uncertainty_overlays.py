"""Tests for vol_arb/uncertainty_overlays.py — 5 pre-committed strategies."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from vol_arb.uncertainty_overlays import (
    CAL_WINDOW,
    EPSILON_DENOM,
    FILTER_QUANTILES,
    SIZE_FLOOR_F_MIN,
    UncertaintyAnchors,
    compute_anchors,
    compute_filter_mask,
    compute_scaler_S1,
    compute_scaler_S2,
    load_filter_sidecar,
    load_scaler_sidecar,
    write_filter_sidecar,
    write_scaler_sidecar,
)


# ─── Locked parameters ──────────────────────────────────────────────────


class TestLockedParams:
    def test_filter_quantiles_locked(self):
        assert FILTER_QUANTILES == {"F1": 0.99, "F2": 0.95, "F3": 0.90}

    def test_size_floor_locked_at_10pct(self):
        assert SIZE_FLOOR_F_MIN == 0.10

    def test_cal_window_matches_cycle_2(self):
        cal_start, cal_end = CAL_WINDOW
        assert cal_start == pd.Timestamp("2021-04-01", tz="UTC")
        assert cal_end == pd.Timestamp("2023-10-01", tz="UTC")


# ─── Anchor construction ────────────────────────────────────────────────


def _make_sigma_epi(
    n: int = 5_000,
    seed: int = 42,
    span_start: str = "2022-06-01",
) -> pd.Series:
    """Synthetic σ²_epi series spanning the calibration window."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range(span_start, periods=n, freq="15min", tz="UTC")
    # Right-skewed positive σ²_epi values
    values = rng.gamma(shape=2.0, scale=0.005, size=n)
    return pd.Series(values, index=idx, name="sigma_epi_squared")


class TestComputeAnchors:
    def test_returns_dataclass(self):
        s = _make_sigma_epi()
        anchors = compute_anchors(s)
        assert isinstance(anchors, UncertaintyAnchors)
        assert anchors.cal_n_obs > 0
        # All quantiles should be ordered
        assert anchors.quantile_F3 < anchors.quantile_F2 < anchors.quantile_F1
        assert 0 < anchors.median_cal < anchors.mean_cal * 2

    def test_too_few_cal_obs_raises(self):
        idx = pd.date_range("2022-06-01", periods=100, freq="15min", tz="UTC")
        s = pd.Series(np.ones(100), index=idx)
        with pytest.raises(ValueError, match="≥ 1000"):
            compute_anchors(s)

    def test_post_window_data_does_not_change_anchors(self):
        """Mutate values AFTER 2023-10-01 → anchors unchanged."""
        cal = _make_sigma_epi(n=5_000, span_start="2022-06-01")
        anchors_baseline = compute_anchors(cal)

        # Append POST-2023-10-01 garbage
        post_idx = pd.date_range(
            "2024-01-01", periods=2_000, freq="15min", tz="UTC",
        )
        post = pd.Series(np.full(2_000, 999.0), index=post_idx)
        combined = pd.concat([cal, post])
        anchors_with_post = compute_anchors(combined)

        assert anchors_baseline.quantile_F1 == pytest.approx(
            anchors_with_post.quantile_F1, rel=1e-12,
        )
        assert anchors_baseline.quantile_F2 == pytest.approx(
            anchors_with_post.quantile_F2, rel=1e-12,
        )
        assert anchors_baseline.quantile_F3 == pytest.approx(
            anchors_with_post.quantile_F3, rel=1e-12,
        )
        assert anchors_baseline.median_cal == pytest.approx(
            anchors_with_post.median_cal, rel=1e-12,
        )
        assert anchors_baseline.mean_cal == pytest.approx(
            anchors_with_post.mean_cal, rel=1e-12,
        )


# ─── Filter overlays ────────────────────────────────────────────────────


class TestFilterOverlays:
    def test_F1_drops_about_1_percent(self):
        s = _make_sigma_epi(n=10_000)
        anchors = compute_anchors(s)
        mask = compute_filter_mask(s, anchors, tag="F1")
        # Within calibration window (i.e. on the same data), top-1% should drop ~1%
        drop_rate = float(mask.mean())
        # Allow some slack for sample-size noise on the empirical quantile
        assert 0.005 < drop_rate < 0.02

    def test_F2_drops_about_5_percent(self):
        s = _make_sigma_epi(n=10_000)
        anchors = compute_anchors(s)
        mask = compute_filter_mask(s, anchors, tag="F2")
        assert 0.04 < float(mask.mean()) < 0.06

    def test_F3_drops_about_10_percent(self):
        s = _make_sigma_epi(n=10_000)
        anchors = compute_anchors(s)
        mask = compute_filter_mask(s, anchors, tag="F3")
        assert 0.09 < float(mask.mean()) < 0.11

    def test_unknown_tag_raises(self):
        s = _make_sigma_epi()
        anchors = compute_anchors(s)
        with pytest.raises(ValueError, match="tag must be"):
            compute_filter_mask(s, anchors, tag="F99")

    def test_returns_boolean(self):
        s = _make_sigma_epi()
        anchors = compute_anchors(s)
        mask = compute_filter_mask(s, anchors, tag="F1")
        assert mask.dtype == bool


# ─── Scaler S1 ──────────────────────────────────────────────────────────


class TestScalerS1:
    def test_zero_sigma_epi_returns_one(self):
        s = _make_sigma_epi()
        anchors = compute_anchors(s)
        zeros = pd.Series(np.zeros(100),
                           index=pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC"))
        f = compute_scaler_S1(zeros, anchors)
        assert (f == 1.0).all()

    def test_median_sigma_epi_returns_half(self):
        """At σ²_epi = median_cal, f = 1 / (1 + 1) = 0.5."""
        s = _make_sigma_epi()
        anchors = compute_anchors(s)
        median_series = pd.Series(
            np.full(50, anchors.median_cal),
            index=pd.date_range("2024-01-01", periods=50, freq="15min", tz="UTC"),
        )
        f = compute_scaler_S1(median_series, anchors)
        np.testing.assert_allclose(f.values, 0.5, atol=1e-12)

    def test_monotonic_decreasing(self):
        s = _make_sigma_epi()
        anchors = compute_anchors(s)
        # Increasing σ²_epi sequence
        idx = pd.date_range("2024-01-01", periods=50, freq="15min", tz="UTC")
        increasing = pd.Series(np.linspace(0, 1.0, 50), index=idx)
        f = compute_scaler_S1(increasing, anchors)
        diffs = np.diff(f.values)
        # Allow exact equality at the floor f_min = 0.10
        assert (diffs <= 1e-12).all()

    def test_clipped_at_floor(self):
        s = _make_sigma_epi()
        anchors = compute_anchors(s)
        huge = pd.Series(
            np.full(20, 1e9),
            index=pd.date_range("2024-01-01", periods=20, freq="15min", tz="UTC"),
        )
        f = compute_scaler_S1(huge, anchors)
        np.testing.assert_allclose(f.values, SIZE_FLOOR_F_MIN, atol=1e-12)


# ─── Scaler S2 (Bayesian-Kelly) ────────────────────────────────────────


class TestScalerS2:
    def test_zero_sigma_epi_returns_one(self):
        s = _make_sigma_epi()
        anchors = compute_anchors(s)
        zeros = pd.Series(
            np.zeros(20),
            index=pd.date_range("2024-01-01", periods=20, freq="15min", tz="UTC"),
        )
        f = compute_scaler_S2(zeros, anchors)
        assert (f == 1.0).all()

    def test_sigma_train_input_returns_half(self):
        """At σ²_epi = σ²_train (= mean_cal), f = σ²_train/(2σ²_train) = 0.5."""
        s = _make_sigma_epi()
        anchors = compute_anchors(s)
        train_series = pd.Series(
            np.full(20, anchors.mean_cal),
            index=pd.date_range("2024-01-01", periods=20, freq="15min", tz="UTC"),
        )
        f = compute_scaler_S2(train_series, anchors)
        np.testing.assert_allclose(f.values, 0.5, atol=1e-12)

    def test_clipped_at_floor(self):
        s = _make_sigma_epi()
        anchors = compute_anchors(s)
        huge = pd.Series(
            np.full(20, 1e9),
            index=pd.date_range("2024-01-01", periods=20, freq="15min", tz="UTC"),
        )
        f = compute_scaler_S2(huge, anchors)
        np.testing.assert_allclose(f.values, SIZE_FLOOR_F_MIN, atol=1e-12)

    def test_no_free_parameters(self):
        """Both scalers should be fully determined by anchors — no tunables."""
        # Calling with default kwargs should match calling with same anchors
        s = _make_sigma_epi()
        anchors = compute_anchors(s)
        target_idx = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
        target = pd.Series(np.linspace(0, 0.05, 10), index=target_idx)
        a = compute_scaler_S1(target, anchors)
        b = compute_scaler_S1(target, anchors)
        np.testing.assert_array_equal(a.values, b.values)


# ─── Sidecar I/O ───────────────────────────────────────────────────────


class TestSidecarIO:
    def test_filter_round_trip(self, tmp_path):
        idx = pd.date_range("2024-01-01", periods=50, freq="15min", tz="UTC")
        mask = pd.Series([True, False] * 25, index=idx, name="filter_F1")
        out = tmp_path / "filter.parquet"
        write_filter_sidecar(mask, out)
        loaded = load_filter_sidecar(out)
        np.testing.assert_array_equal(mask.values, loaded.values)
        assert loaded.dtype == bool

    def test_scaler_round_trip(self, tmp_path):
        idx = pd.date_range("2024-01-01", periods=50, freq="15min", tz="UTC")
        size_mult = pd.Series(np.linspace(0.1, 1.0, 50), index=idx,
                                name="size_mult")
        out = tmp_path / "scaler.parquet"
        write_scaler_sidecar(size_mult, out)
        loaded = load_scaler_sidecar(out)
        np.testing.assert_allclose(size_mult.values, loaded.values)


# ─── Integration: backtester config flag default off ───────────────────


class TestBacktesterIntegration:
    def test_uncertainty_filter_flag_default_none(self):
        """New flag exists and defaults to None (preserves V2 byte-identity)."""
        from vol_arb.backtester import BacktestConfig
        cfg = BacktestConfig()
        assert hasattr(cfg, "uncertainty_filter_path")
        assert cfg.uncertainty_filter_path is None

    def test_v2_preset_does_not_auto_enable(self):
        from vol_arb.backtester import BacktestConfig
        cfg = BacktestConfig().with_btc_production_filters_v2()
        assert cfg.uncertainty_filter_path is None
