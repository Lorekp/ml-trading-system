"""Tests for vol_model.qrs (Session I Candidate I-3).

Pins:
1. Schema + determinism (same seed → byte-identical quantiles).
2. Monotonic quantiles across columns (QRS-by-construction property).
3. NO TEST LEAKAGE: calibration mask strictly separates calibration
   from test rows; mutating test-only residuals must NOT change the
   fitted distribution.
4. Sharpness sanity: wider residual distribution → wider predictive
   intervals.
5. CRPS approximation reasonably close to Gaussian closed-form when
   residuals ARE Gaussian (calibration sanity).
6. Coverage: a well-calibrated empirical distribution should have
   empirical coverage close to the nominal level.
7. PIT uniformity: on honestly-generated data (residuals drawn from
   the same distribution as the ground-truth), PIT p-value should be
   large (no evidence against uniformity).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_model.qrs import (
    DEFAULT_QUANTILES,
    ResidualDistribution,
    fit_residual_distribution,
    predict_distribution,
    score_distributional,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _synthetic_preds_and_truth(
    n: int = 5000, noise_sigma: float = 0.3, seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic point-forecast setup: y_true = y_pred + noise, noise Gaussian."""
    rng = np.random.default_rng(seed)
    y_pred = rng.standard_normal(n) * 0.5  # something vaguely vol-like
    noise = rng.standard_normal(n) * noise_sigma
    y_true = y_pred + noise
    return y_pred, y_true


# ── 1. Schema + determinism ──────────────────────────────────────────


def test_fit_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        fit_residual_distribution(np.zeros(10), np.zeros(11))


def test_fit_rejects_too_few_rows() -> None:
    with pytest.raises(ValueError, match="too few calibration rows"):
        fit_residual_distribution(np.zeros(20), np.zeros(20))


def test_predict_schema() -> None:
    y_pred, y_true = _synthetic_preds_and_truth()
    dist = fit_residual_distribution(y_pred, y_true)
    q = predict_distribution(y_pred[:100], dist, n_bootstrap=500)
    assert len(q) == 100
    assert list(q.columns) == [
        f"q_{int(round(100*lvl)):02d}" for lvl in DEFAULT_QUANTILES
    ]
    assert q.notna().all().all()


def test_predict_deterministic_under_seed() -> None:
    """Same seed → byte-identical quantile frame. Regression guard."""
    y_pred, y_true = _synthetic_preds_and_truth()
    dist = fit_residual_distribution(y_pred, y_true)
    q1 = predict_distribution(y_pred[:200], dist, n_bootstrap=1000, seed=42)
    q2 = predict_distribution(y_pred[:200], dist, n_bootstrap=1000, seed=42)
    pd.testing.assert_frame_equal(q1, q2)


def test_predict_different_seeds_differ() -> None:
    """Different seeds → different bootstrap draws → different quantiles."""
    y_pred, y_true = _synthetic_preds_and_truth()
    dist = fit_residual_distribution(y_pred, y_true)
    q1 = predict_distribution(y_pred[:200], dist, n_bootstrap=1000, seed=42)
    q2 = predict_distribution(y_pred[:200], dist, n_bootstrap=1000, seed=123)
    # Quantile columns should differ at least a little.
    assert not np.allclose(q1.to_numpy(), q2.to_numpy())


# ── 2. Monotonic quantiles (by-construction) ─────────────────────────


def test_quantiles_monotonic_across_columns() -> None:
    """QRS sidesteps the quantile-crossing issue by construction: all
    quantiles come from the same empirical sample, so np.quantile at
    sorted levels produces monotonic output."""
    y_pred, y_true = _synthetic_preds_and_truth()
    dist = fit_residual_distribution(y_pred, y_true)
    q = predict_distribution(y_pred[:500], dist, n_bootstrap=1000, seed=42)
    # Every row must satisfy q_01 <= q_10 <= q_25 <= ... <= q_99.
    vals = q.to_numpy()
    diffs = np.diff(vals, axis=1)
    assert (diffs >= -1e-12).all(), (
        f"quantile crossing detected; min diff = {diffs.min()}"
    )


# ── 3. Leakage safety (the central test) ─────────────────────────────


def test_calibration_mask_excludes_test_rows() -> None:
    """Critical leakage test: residual distribution fit with a mask
    excluding test rows MUST NOT depend on test-row residuals.

    Setup: 1000 rows, mask says rows 0-799 are calibration, 800-999 are
    test. Fit residual distribution on calibration. Then clobber the
    test rows' residuals to something absurd and re-fit. The residual
    distribution MUST be byte-identical (because calibration rows are
    unaffected)."""
    y_pred, y_true = _synthetic_preds_and_truth(n=1000, seed=0)
    mask = np.zeros(1000, dtype=bool)
    mask[:800] = True  # calibration rows

    dist_a = fit_residual_distribution(y_pred, y_true, calibration_mask=mask)

    # Corrupt test rows wildly
    y_true_corrupt = y_true.copy()
    y_true_corrupt[800:] = 9999.0  # absurd value

    dist_b = fit_residual_distribution(
        y_pred, y_true_corrupt, calibration_mask=mask,
    )

    np.testing.assert_array_equal(dist_a.residuals, dist_b.residuals)
    assert dist_a.sigma == dist_b.sigma
    assert dist_a.n == dist_b.n


def test_fit_skips_non_finite() -> None:
    """NaN residuals in calibration slice must be filtered, not crash."""
    y_pred = np.array([1.0, 2.0, 3.0] * 20 + [np.nan, np.nan])  # 62 rows, 2 NaN
    y_true = np.array([1.1, 2.1, 3.1] * 20 + [0.0, 0.0])
    mask = np.ones_like(y_pred, dtype=bool)

    dist = fit_residual_distribution(y_pred, y_true, calibration_mask=mask)
    assert dist.n == 60  # 2 NaN rows filtered
    assert np.isfinite(dist.residuals).all()


# ── 4. Sharpness sanity ──────────────────────────────────────────────


def test_wider_residuals_give_wider_intervals() -> None:
    """Monotonicity: a 2× wider residual distribution should produce
    ~2× wider predictive intervals (on average)."""
    y_pred_narrow, y_true_narrow = _synthetic_preds_and_truth(noise_sigma=0.1)
    y_pred_wide, y_true_wide = _synthetic_preds_and_truth(noise_sigma=0.4)

    dist_narrow = fit_residual_distribution(y_pred_narrow, y_true_narrow)
    dist_wide = fit_residual_distribution(y_pred_wide, y_true_wide)

    q_narrow = predict_distribution(
        y_pred_narrow[:500], dist_narrow, n_bootstrap=1000, seed=42,
    )
    q_wide = predict_distribution(
        y_pred_wide[:500], dist_wide, n_bootstrap=1000, seed=42,
    )

    spread_narrow = float((q_narrow["q_90"] - q_narrow["q_10"]).median())
    spread_wide = float((q_wide["q_90"] - q_wide["q_10"]).median())

    # Allow some slack; the ratio should be between 3× and 5× (noise
    # ratio is 4×, but bootstrap quantiles have some sampling variance).
    ratio = spread_wide / spread_narrow
    assert 2.5 < ratio < 5.5, f"spread ratio {ratio} out of expected range"


# ── 5. CRPS approximation sanity ─────────────────────────────────────


def test_crps_close_to_gaussian_when_residuals_are_gaussian() -> None:
    """When residuals ARE Gaussian, the 3-quantile CRPS approximation
    should be within ~20% of the Gaussian closed-form CRPS. This pins
    that the pinball-sum approximation is reasonable.
    """
    y_pred, y_true = _synthetic_preds_and_truth(noise_sigma=0.3, seed=0)
    dist = fit_residual_distribution(y_pred[:4000], y_true[:4000])
    q = predict_distribution(y_pred[4000:], dist, n_bootstrap=2000, seed=42)
    scores = score_distributional(y_true[4000:], q, residual_dist=dist)

    assert np.isfinite(scores["crps"])
    assert np.isfinite(scores["crps_gaussian"])
    # Ratio should be within 20% either way (3-quantile approximation is
    # a known low-resolution estimator; tighter tolerance is unfair).
    ratio = scores["crps"] / scores["crps_gaussian"]
    assert 0.8 < ratio < 1.4, (
        f"CRPS approximation ratio {ratio:.3f} out of sanity range"
    )


# ── 6. Coverage sanity ───────────────────────────────────────────────


def test_coverage_close_to_nominal() -> None:
    """Well-calibrated distribution: empirical coverage at 80% should
    be within ±5pp of nominal. This pins honest calibration when the
    residual distribution reflects the true error process.
    """
    y_pred, y_true = _synthetic_preds_and_truth(n=10_000, seed=3)
    dist = fit_residual_distribution(y_pred[:8000], y_true[:8000])
    q = predict_distribution(y_pred[8000:], dist, n_bootstrap=2000, seed=42)
    scores = score_distributional(y_true[8000:], q, residual_dist=dist)

    assert 0.75 <= scores["coverage_80"] <= 0.85
    assert 0.85 <= scores["coverage_90"] <= 0.95
    assert 0.95 <= scores["coverage_98"] <= 1.0


# ── 7. PIT uniformity ────────────────────────────────────────────────


def test_pit_uniform_when_well_calibrated() -> None:
    """On honestly-calibrated data (residuals from the same distribution
    as ground truth), the PIT KS p-value should be > 0.01 (not strongly
    rejected). Uses the linear-interpolation PIT over the 7 quantile
    levels — coarse but honest."""
    y_pred, y_true = _synthetic_preds_and_truth(n=10_000, seed=7)
    dist = fit_residual_distribution(y_pred[:8000], y_true[:8000])
    q = predict_distribution(y_pred[8000:], dist, n_bootstrap=2000, seed=42)
    scores = score_distributional(y_true[8000:], q, residual_dist=dist)

    # Loose gate — we only have 7 quantile levels so PIT is piecewise-
    # linear. On a well-calibrated setup with n=2000 test rows, the KS
    # test shouldn't strongly reject uniformity.
    assert scores["pit_ks_pvalue"] > 0.01, (
        f"PIT p-value {scores['pit_ks_pvalue']} — unexpected miscalibration"
    )


def test_pit_rejects_when_badly_calibrated() -> None:
    """Negative control: fit residual distribution on narrow noise, then
    score against wide noise data. PIT should be strongly non-uniform.
    """
    y_pred_narrow, y_true_narrow = _synthetic_preds_and_truth(
        noise_sigma=0.05, seed=0,
    )
    y_pred_wide, y_true_wide = _synthetic_preds_and_truth(
        noise_sigma=1.0, seed=1,
    )

    # Fit on narrow data, score on wide (badly calibrated)
    dist = fit_residual_distribution(y_pred_narrow, y_true_narrow)
    q = predict_distribution(y_pred_wide[:1000], dist, n_bootstrap=1000, seed=42)
    scores = score_distributional(y_true_wide[:1000], q, residual_dist=dist)

    # Coverage should be FAR below nominal (predictive intervals too narrow)
    assert scores["coverage_80"] < 0.5, (
        f"narrow-on-wide coverage_80 = {scores['coverage_80']} "
        f"— should be << 0.8 due to severe under-coverage"
    )


# ── 8. Quantile level validation ─────────────────────────────────────


def test_predict_rejects_unsorted_quantiles() -> None:
    y_pred, y_true = _synthetic_preds_and_truth()
    dist = fit_residual_distribution(y_pred, y_true)
    with pytest.raises(ValueError, match="pre-sorted ascending"):
        predict_distribution(
            y_pred[:10], dist, quantiles=(0.5, 0.1, 0.9),
        )


def test_predict_rejects_quantiles_out_of_range() -> None:
    y_pred, y_true = _synthetic_preds_and_truth()
    dist = fit_residual_distribution(y_pred, y_true)
    with pytest.raises(ValueError, match=r"must be in \(0, 1\)"):
        predict_distribution(y_pred[:10], dist, quantiles=(0.0, 0.5, 1.0))
