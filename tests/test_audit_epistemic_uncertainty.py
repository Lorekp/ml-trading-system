"""Tests for scripts/forensic/audit_epistemic_uncertainty.py.

Synthetic-fixture probe sanity + pre-gate threshold lock.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.forensic.audit_epistemic_uncertainty import (
    CAL_WINDOW,
    EVAL_WINDOW_START,
    HQR_COVERAGE_HI,
    HQR_COVERAGE_LO,
    SIGNAL_QUALITY_PREGATE,
    probe_hqr_coverage,
    probe_orthogonality_to_y_pred,
    probe_regime_uncertainty,
    probe_signal_quality_pnl,
    probe_signal_quality_spearman,
)


# ─── Pre-committed thresholds locked ────────────────────────────────────


class TestLockedThresholds:
    def test_signal_quality_pregate_locked(self):
        assert SIGNAL_QUALITY_PREGATE == 0.20

    def test_hqr_coverage_band_locked(self):
        assert HQR_COVERAGE_LO == 0.85
        assert HQR_COVERAGE_HI == 0.95

    def test_cal_window_locked(self):
        cal_start, cal_end = CAL_WINDOW
        assert cal_start == pd.Timestamp("2021-04-01", tz="UTC")
        assert cal_end == pd.Timestamp("2023-10-01", tz="UTC")

    def test_eval_window_start_locked(self):
        assert EVAL_WINDOW_START == pd.Timestamp("2023-10-01", tz="UTC")


# ─── Probe sanity on synthetic fixtures ─────────────────────────────────


def _synthetic_uncertainty_frame(
    n: int = 5_000, seed: int = 42, signal_strength: float = 0.5,
) -> pd.DataFrame:
    """Build a synthetic uncertainty DataFrame.

    σ²_epi correlates with |error| at the strength specified — used to
    test the signal-quality probe.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-06-01", periods=n, freq="15min", tz="UTC")
    # Heteroscedastic: error magnitude scales with sigma_epi
    sigma_epi = np.abs(rng.normal(0.05, 0.02, n))
    abs_err = signal_strength * sigma_epi + (1 - signal_strength) * np.abs(
        rng.normal(0, 0.02, n)
    )
    sign = rng.choice([-1.0, 1.0], n)
    err = sign * abs_err
    y_pred = rng.normal(0, 0.1, n)
    y_true = y_pred + err
    pi_lo = y_pred - 2.0 * sigma_epi
    pi_hi = y_pred + 2.0 * sigma_epi
    df = pd.DataFrame({
        "ensemble_mean": y_pred,
        "sigma_epi": sigma_epi,
        "sigma_alea": np.full(n, 0.01),
        "sigma_total": np.sqrt(sigma_epi ** 2 + 0.01 ** 2),
        "spread": pi_hi - pi_lo,
        "pi_lo": pi_lo,
        "pi_hi": pi_hi,
        "y_true": y_true,
    }, index=idx)
    return df


class TestProbeSignalQuality:
    def test_strong_correlation_passes_pregate(self):
        df = _synthetic_uncertainty_frame(n=5_000, signal_strength=0.95)
        sigma_epi_sq = (df["sigma_epi"] ** 2).rename("sigma_epi_sq")
        result = probe_signal_quality_spearman(
            sigma_epi_sq, df["y_true"], df["ensemble_mean"],
        )
        # Synthetic data spans cal window, so no eval data — pre-gate is NaN
        # Use a fixture where some data falls in eval window
        df2 = df.copy()
        df2.index = pd.date_range("2024-01-01", periods=len(df2), freq="15min", tz="UTC")
        sigma_epi_sq2 = (df2["sigma_epi"] ** 2).rename("sigma_epi_sq")
        result2 = probe_signal_quality_spearman(
            sigma_epi_sq2, df2["y_true"], df2["ensemble_mean"],
        )
        assert result2["eval"].metric > 0.5  # strong signal
        assert result2["eval"].pass_fail == "PASS"

    def test_weak_correlation_fails_pregate(self):
        df = _synthetic_uncertainty_frame(n=5_000, signal_strength=0.0)
        # All of the data in eval window
        df.index = pd.date_range("2024-01-01", periods=len(df), freq="15min", tz="UTC")
        sigma_epi_sq = (df["sigma_epi"] ** 2).rename("sigma_epi_sq")
        result = probe_signal_quality_spearman(
            sigma_epi_sq, df["y_true"], df["ensemble_mean"],
        )
        assert abs(result["eval"].metric) < 0.10
        assert result["eval"].pass_fail == "FAIL"


class TestProbeHqrCoverage:
    def test_coverage_in_band_passes(self):
        # Build a dataset where ~90% of y_true falls inside [pi_lo, pi_hi]
        rng = np.random.default_rng(11)
        n = 5_000
        # Place all data in eval window
        idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        y_pred = rng.normal(0, 0.1, n)
        sigma = np.full(n, 0.05)
        # Use 1.645 × sigma = 90% PI assuming normal residual
        y_true = y_pred + rng.normal(0, sigma, n)
        df = pd.DataFrame({
            "ensemble_mean": y_pred,
            "sigma_epi": sigma,
            "sigma_alea": np.full(n, 0.01),
            "sigma_total": np.sqrt(sigma ** 2 + 0.01 ** 2),
            "spread": 2 * 1.645 * sigma,
            "pi_lo": y_pred - 1.645 * sigma,
            "pi_hi": y_pred + 1.645 * sigma,
            "y_true": y_true,
        }, index=idx)
        result = probe_hqr_coverage(df)
        assert HQR_COVERAGE_LO <= result.metric <= HQR_COVERAGE_HI
        assert result.pass_fail == "PASS"

    def test_coverage_too_low_fails(self):
        rng = np.random.default_rng(13)
        n = 3_000
        idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
        y_pred = rng.normal(0, 0.1, n)
        sigma = np.full(n, 0.05)
        y_true = y_pred + rng.normal(0, sigma * 5, n)  # huge errors → narrow PI fails
        df = pd.DataFrame({
            "ensemble_mean": y_pred,
            "pi_lo": y_pred - 0.5 * sigma,
            "pi_hi": y_pred + 0.5 * sigma,
            "y_true": y_true,
            "sigma_epi": sigma, "sigma_alea": sigma,
            "sigma_total": sigma, "spread": sigma,
        }, index=idx)
        result = probe_hqr_coverage(df)
        assert result.metric < HQR_COVERAGE_LO
        assert result.pass_fail == "FAIL"


class TestProbeOrthogonality:
    def test_runs_on_synthetic_fixture(self):
        df = _synthetic_uncertainty_frame(n=3_000)
        # Move into eval window
        df.index = pd.date_range("2024-01-01", periods=len(df), freq="15min", tz="UTC")
        sigma_epi_sq = (df["sigma_epi"] ** 2).rename("sigma_epi_sq")
        result = probe_orthogonality_to_y_pred(
            sigma_epi_sq, df["ensemble_mean"], df["y_true"],
        )
        # No gate; just verify it doesn't blow up
        assert isinstance(result.metric, float)
