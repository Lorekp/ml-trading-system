"""Unit tests for HQR conformal wrapper.

Property-based tests on synthetic heteroscedastic data:
1. Spread correlates with true generating sigma on a heteroscedastic dataset.
2. Coverage is approximately 1 - alpha on held-out data from the same DGP.
3. Causality: predict_hqr raises if test timestamps overlap calibration.
4. Reproducibility: same seed → identical sigma model and q_conformal.
5. Degenerate input rejection: too-small cal fold raises cleanly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_model.conformal.hqr import (
    HQRFit, evaluate_hqr, fit_hqr, predict_hqr, crps_gaussian_baseline,
)


RNG_SEED = 20260417


def _heteroscedastic_dgp(
    n: int, n_features: int = 4, seed: int = RNG_SEED,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray]:
    """Synthetic DGP where noise scale is a non-trivial function of features.

    y = b · X + ε,  ε ~ N(0, σ(X)^2)  with  σ(X) = 0.1 + 0.4 · |X[:, 0]|

    Returns (X, y, y_hat, true_sigma) where y_hat is a noisy linear estimator
    (intentionally suboptimal so residuals are non-zero).
    """
    rng = np.random.default_rng(seed)
    # Timestamp-indexed so HQR's causality guards have something to check
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    X = pd.DataFrame(
        rng.normal(0, 1, size=(n, n_features)),
        columns=[f"x{i}" for i in range(n_features)],
        index=idx,
    )
    beta = np.array([0.5, -0.3, 0.2, 0.0][:n_features])
    y_mean = X.values @ beta
    true_sigma = 0.1 + 0.4 * np.abs(X["x0"].values)
    y = pd.Series(y_mean + rng.normal(0, 1, n) * true_sigma, index=idx)
    # "Point forecast" = y_mean with small added noise (simulates imperfect model)
    y_hat = y_mean + rng.normal(0, 0.05, n)
    return X, y, y_hat, true_sigma


def test_fit_hqr_returns_expected_artifacts():
    """fit_hqr produces HQRFit dataclass with non-trivial sigma model + q."""
    X, y, y_hat, _ = _heteroscedastic_dgp(n=2000)
    fit = fit_hqr(X, y, y_hat, alpha=0.1, num_boost_round=100)
    assert isinstance(fit, HQRFit)
    assert fit.q_conformal > 0
    assert 0 < fit.n_sigma_train <= 1000
    assert 0 < fit.n_conformal <= 1000
    assert fit.n_sigma_train + fit.n_conformal == len(X)
    assert fit.alpha == 0.1


def test_hqr_spread_correlates_with_true_sigma():
    """On heteroscedastic DGP, σ̂(x) should track the generating σ(x).

    Translating to the session's KPI: Spearman(spread, |err|) > 0.30.
    """
    X_cal, y_cal, y_hat_cal, _ = _heteroscedastic_dgp(n=4000, seed=RNG_SEED)
    X_test, y_test, y_hat_test, true_sigma_test = _heteroscedastic_dgp(
        n=2000, seed=RNG_SEED + 1,
    )
    # Give test a later timestamp range to satisfy the causality check
    X_test.index = X_cal.index[-1] + pd.Timedelta(minutes=15) + (
        X_test.index - X_test.index[0]
    )

    fit = fit_hqr(X_cal, y_cal, y_hat_cal, alpha=0.1, num_boost_round=200)
    result = predict_hqr(X_test, y_hat_test, fit)
    metrics = evaluate_hqr(y_test.values, y_hat_test, result, alpha=0.1)
    # HQR should clear 0.30 spread↔|err| Spearman on a dataset where
    # σ is a truly non-constant function of features
    assert metrics["spearman_spread_err"] >= 0.30, (
        f"Spearman(spread, |err|) = {metrics['spearman_spread_err']:.3f} < 0.30 "
        f"on synthetic heteroscedastic DGP — HQR width estimator not tracking."
    )


def test_hqr_coverage_near_nominal():
    """90% PI coverage should land near 0.90 on held-out data from same DGP."""
    X_cal, y_cal, y_hat_cal, _ = _heteroscedastic_dgp(n=4000, seed=RNG_SEED)
    X_test, y_test, y_hat_test, _ = _heteroscedastic_dgp(
        n=3000, seed=RNG_SEED + 2,
    )
    X_test.index = X_cal.index[-1] + pd.Timedelta(minutes=15) + (
        X_test.index - X_test.index[0]
    )

    fit = fit_hqr(X_cal, y_cal, y_hat_cal, alpha=0.1, num_boost_round=200)
    result = predict_hqr(X_test, y_hat_test, fit)
    coverage = evaluate_hqr(y_test.values, y_hat_test, result, alpha=0.1)["coverage"]
    # Allow ±5pp tolerance — finite-sample + synthetic-DGP noise
    assert 0.85 <= coverage <= 0.95, (
        f"coverage {coverage:.3f} outside [0.85, 0.95] for nominal α=0.1"
    )


def test_hqr_monotone_pi_lo_le_pi_hi():
    """PI bounds must satisfy pi_lo ≤ ŷ ≤ pi_hi always."""
    X_cal, y_cal, y_hat_cal, _ = _heteroscedastic_dgp(n=2000)
    X_test = X_cal.iloc[-300:].copy()
    # Shift test later to satisfy causality
    cal_for_fit = X_cal.iloc[:-300]
    y_cal_for_fit = y_cal.iloc[:-300]
    y_hat_cal_for_fit = y_hat_cal[:-300]
    y_hat_test = y_hat_cal[-300:]
    X_test.index = X_test.index + pd.Timedelta(hours=10)

    fit = fit_hqr(cal_for_fit, y_cal_for_fit, y_hat_cal_for_fit, num_boost_round=100)
    result = predict_hqr(X_test, y_hat_test, fit)

    assert np.all(result["pi_lo"] <= y_hat_test + 1e-9), "pi_lo > y_hat somewhere"
    assert np.all(result["pi_hi"] >= y_hat_test - 1e-9), "pi_hi < y_hat somewhere"
    assert np.all(result["spread"] >= 0), "negative spread"


def test_hqr_causality_violation_raises():
    """predict_hqr must raise if test timestamps overlap calibration window."""
    X, y, y_hat, _ = _heteroscedastic_dgp(n=2000)
    fit = fit_hqr(X, y, y_hat, alpha=0.1, num_boost_round=50)
    # Test window OVERLAPS calibration — last cal row was at fit.cal2_end_ts;
    # create a test window that starts before that
    X_test = X.iloc[-500:].copy()  # all inside cal
    y_hat_test = y_hat[-500:]
    with pytest.raises(ValueError, match="causality violation"):
        predict_hqr(X_test, y_hat_test, fit)


def test_hqr_rejects_too_small_cal_fold():
    """fit_hqr raises when cal fold has <100 rows per split."""
    X, y, y_hat, _ = _heteroscedastic_dgp(n=50)
    with pytest.raises(ValueError, match="calibration fold too small"):
        fit_hqr(X, y, y_hat)


def test_crps_baseline_finite_and_positive():
    """crps_gaussian_baseline returns a finite positive number."""
    rng = np.random.default_rng(RNG_SEED)
    y = rng.normal(0, 1, 500)
    y_hat = rng.normal(0, 0.1, 500)
    crps = crps_gaussian_baseline(y, y_hat, sigma_const=1.0)
    assert np.isfinite(crps)
    assert crps > 0


def test_hqr_reproducibility():
    """Same data + same seed → identical HQRFit artifacts."""
    X, y, y_hat, _ = _heteroscedastic_dgp(n=2000, seed=RNG_SEED)
    fit1 = fit_hqr(X, y, y_hat, alpha=0.1, num_boost_round=100)
    fit2 = fit_hqr(X, y, y_hat, alpha=0.1, num_boost_round=100)
    assert fit1.q_conformal == fit2.q_conformal
    # Predictions on the same inputs should also match
    X_test = X.iloc[:100].copy()
    X_test.index = X.index[-1] + pd.Timedelta(minutes=15) + (
        X_test.index - X_test.index[0]
    )
    y_hat_test = y_hat[:100]
    r1 = predict_hqr(X_test, y_hat_test, fit1)
    r2 = predict_hqr(X_test, y_hat_test, fit2)
    np.testing.assert_array_equal(r1["spread"], r2["spread"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
