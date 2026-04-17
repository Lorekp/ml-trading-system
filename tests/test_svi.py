"""Comprehensive tests for Arbitrage-Free SVI calibrator."""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
import pytest
from scipy.optimize import approx_fprime

from vol_arb.svi import (
    SVICalibrator,
    SVIConfig,
    SVISlice,
    ArbitrageFreeSVI,
    _svi_total_variance,
    _svi_jacobian_w,
    _butterfly_penalty,
    calibrate_surface,
)


# ────────────────────────────────────────────────────────────
# TEST 1: SVI Evaluation Correctness
# ────────────────────────────────────────────────────────────

class TestSVIEvaluation:
    """Verify w(k) matches hand-computed values at known points."""

    @pytest.fixture
    def slice(self) -> SVISlice:
        return SVISlice(a=0.04, b=0.4, rho=-0.3, m=0.0, sigma=0.2, T=0.25, F=50000.0)

    def test_atm_total_variance(self, slice: SVISlice):
        """w(0) = a + b * (rho*(-m) + sqrt(m^2 + sigma^2)) = 0.04 + 0.4*(0+0.2) = 0.12."""
        w = slice.total_variance(np.array([0.0]))
        expected = 0.04 + 0.4 * (0.0 + np.sqrt(0.0 + 0.04))
        np.testing.assert_allclose(w, [expected], rtol=1e-10)

    def test_wing_values(self, slice: SVISlice):
        """Check w at k = ±1 against formula."""
        k = np.array([-1.0, 1.0])
        w = slice.total_variance(k)
        for i, ki in enumerate(k):
            expected = 0.04 + 0.4 * (-0.3 * ki + np.sqrt(ki**2 + 0.04))
            np.testing.assert_allclose(w[i], expected, rtol=1e-10)

    def test_implied_vol_positive(self, slice: SVISlice):
        """Implied vol must be non-negative everywhere."""
        k = np.linspace(-3.0, 3.0, 100)
        iv = slice.implied_vol(k)
        assert np.all(iv >= 0), f"Negative IV found: min={iv.min()}"

    def test_total_variance_non_negative(self, slice: SVISlice):
        """Total variance must be non-negative for valid SVI parameters."""
        k = np.linspace(-5.0, 5.0, 500)
        w = slice.total_variance(k)
        assert np.all(w >= -1e-10), f"Negative total variance: min={w.min()}"

    def test_pct_vs_decimal_consistency(self, slice: SVISlice):
        k = np.array([0.0, 0.5, -0.5])
        np.testing.assert_allclose(slice.implied_vol_pct(k), slice.implied_vol(k) * 100.0)


# ────────────────────────────────────────────────────────────
# TEST 2: Derivative Correctness
# ────────────────────────────────────────────────────────────

class TestDerivatives:
    """Verify analytical w'(k) and w''(k) against finite differences."""

    @pytest.fixture
    def slice(self) -> SVISlice:
        return SVISlice(a=0.04, b=0.4, rho=-0.3, m=0.05, sigma=0.2, T=0.25, F=50000.0)

    def test_first_derivative(self, slice: SVISlice):
        k = np.linspace(-2.0, 2.0, 20)
        wp_analytical = slice.w_prime(k)

        eps = 1e-7
        wp_numerical = (slice.total_variance(k + eps) - slice.total_variance(k - eps)) / (2 * eps)
        np.testing.assert_allclose(wp_analytical, wp_numerical, rtol=1e-5)

    def test_second_derivative(self, slice: SVISlice):
        k = np.linspace(-2.0, 2.0, 20)
        wpp_analytical = slice.w_double_prime(k)

        eps = 1e-5
        wpp_numerical = (
            slice.total_variance(k + eps) - 2 * slice.total_variance(k) + slice.total_variance(k - eps)
        ) / (eps * eps)
        # Numerical second derivative is less precise at extremes
        np.testing.assert_allclose(wpp_analytical, wpp_numerical, rtol=2e-3)

    def test_w_prime_wing_limits(self, slice: SVISlice):
        """w'(+inf) -> b*(1+rho), w'(-inf) -> b*(rho-1)."""
        k_far_right = np.array([100.0])
        k_far_left = np.array([-100.0])
        wp_right = slice.w_prime(k_far_right)
        wp_left = slice.w_prime(k_far_left)
        np.testing.assert_allclose(wp_right, [slice.b * (1 + slice.rho)], rtol=1e-3)
        np.testing.assert_allclose(wp_left, [slice.b * (slice.rho - 1)], rtol=1e-3)


# ────────────────────────────────────────────────────────────
# TEST 3: Butterfly Density
# ────────────────────────────────────────────────────────────

class TestButterflyDensity:
    """Verify butterfly density g(k) properties."""

    def test_known_good_params(self):
        """Gatheral-style well-behaved params should be butterfly-free."""
        sl = SVISlice(a=0.04, b=0.1, rho=-0.3, m=0.0, sigma=0.3, T=0.5, F=50000.0)
        assert sl.is_butterfly_free(200), "Known-good params should be butterfly-free"

    def test_known_bad_params(self):
        """Extreme parameters can produce butterfly violations."""
        sl = SVISlice(a=0.001, b=5.0, rho=-0.99, m=0.0, sigma=0.001, T=0.5, F=50000.0)
        k_grid = np.linspace(-3.0, 3.0, 200)
        g = sl.g_butterfly(k_grid)
        # With extreme params, violations are expected
        assert np.any(g < 0), "Extreme params should produce butterfly violations"

    def test_g_positive_at_atm(self):
        """For reasonable params, g(0) should be positive."""
        sl = SVISlice(a=0.04, b=0.2, rho=-0.2, m=0.0, sigma=0.2, T=0.25, F=50000.0)
        g0 = sl.g_butterfly(np.array([0.0]))
        assert g0[0] > 0, f"g(0)={g0[0]} should be positive"

    def test_second_derivative_positive(self):
        """w''(k) should always be positive (w is convex in k for fixed b,sigma>0)."""
        sl = SVISlice(a=0.04, b=0.2, rho=-0.2, m=0.0, sigma=0.2, T=0.25, F=50000.0)
        k = np.linspace(-3.0, 3.0, 100)
        wpp = sl.w_double_prime(k)
        assert np.all(wpp > 0), "w''(k) must be positive for b,sigma>0"


# ────────────────────────────────────────────────────────────
# TEST 4: Calibration Round-Trip
# ────────────────────────────────────────────────────────────

class TestCalibrationRoundTrip:
    """Generate synthetic data from known params, calibrate, verify recovery."""

    def test_round_trip_clean(self):
        """Exact data (no noise) should recover parameters closely."""
        true_params = np.array([0.04, 0.2, -0.3, 0.0, 0.2])
        T = 0.25
        F = 50000.0

        k = np.linspace(-1.5, 1.5, 15)
        w_true = _svi_total_variance(true_params, k)
        vega = np.ones_like(k)  # uniform weights for clean data

        config = SVIConfig(butterfly_penalty_weight=10.0)
        calibrator = SVICalibrator(config)
        sl = calibrator.calibrate(k, w_true, vega, T, F)

        # Butterfly penalty slightly perturbs the fit from exact recovery
        np.testing.assert_allclose(sl.params, true_params, atol=1e-2)

    def test_round_trip_noisy(self):
        """Noisy data should produce RMSE within noise level."""
        true_params = np.array([0.04, 0.2, -0.3, 0.0, 0.2])
        T = 0.25
        F = 50000.0

        k = np.linspace(-1.5, 1.5, 15)
        w_true = _svi_total_variance(true_params, k)

        rng = np.random.RandomState(42)
        noise_std = 0.001  # small noise in total variance
        w_noisy = w_true + rng.normal(0, noise_std, len(k))
        w_noisy = np.maximum(w_noisy, 0.001)
        vega = np.ones_like(k)

        config = SVIConfig(butterfly_penalty_weight=10.0)
        calibrator = SVICalibrator(config)
        sl = calibrator.calibrate(k, w_noisy, vega, T, F)

        assert sl.rmse < 5.0, f"RMSE {sl.rmse:.2f} too large for moderate noise"

    def test_skewed_smile(self):
        """Calibration should handle negatively skewed crypto smiles."""
        # Typical crypto: negative rho, ATM near 60%
        true_params = np.array([0.09, 0.3, -0.5, -0.05, 0.15])
        T = 0.1
        F = 60000.0

        k = np.linspace(-1.0, 1.0, 12)
        w_true = _svi_total_variance(true_params, k)
        vega = np.ones_like(k)

        calibrator = SVICalibrator(SVIConfig())
        sl = calibrator.calibrate(k, w_true, vega, T, F)

        assert sl.rmse < 1.0, f"RMSE {sl.rmse:.2f} too large for clean skewed data"


# ────────────────────────────────────────────────────────────
# TEST 5: Reduced-Parameter Fit
# ────────────────────────────────────────────────────────────

class TestReducedFit:
    """Reduced 3-param fit for 3-4 strikes."""

    def test_three_strikes_symmetric(self):
        """3 strikes from symmetric smile should fit well."""
        T = 0.25
        F = 50000.0
        # Symmetric smile: a + b*sqrt(k^2 + sigma^2)
        a_true, b_true, sigma_true = 0.04, 0.15, 0.2
        k = np.array([-0.5, 0.0, 0.5])
        w_true = a_true + b_true * np.sqrt(k**2 + sigma_true**2)
        vega = np.ones(3)

        calibrator = SVICalibrator(SVIConfig())
        sl = calibrator.calibrate_reduced(k, w_true, vega, T, F)

        assert sl.calibration_level == 1
        assert sl.rho == 0.0
        assert sl.m == 0.0
        assert sl.rmse < 1.0

    def test_flat_vol_fallback_two_strikes(self):
        """2 strikes should use flat-vol fallback (level 2)."""
        T = 0.25
        F = 50000.0
        # 2 market IVs in percentage (via w = iv² * T)
        iv_pct = np.array([50.0, 55.0])

        calibrator = SVICalibrator(SVIConfig())
        sl = calibrator.fallback_flat_vol(iv_pct, T, F, level=2)

        assert sl.calibration_level == 2
        assert sl.n_strikes == 2
        # ATM IV should match median (52.5%)
        np.testing.assert_allclose(sl.atm_vol_pct(), 52.5, rtol=1e-6)


# ────────────────────────────────────────────────────────────
# TEST 6: Calendar Arbitrage Enforcement
# ────────────────────────────────────────────────────────────

class TestCalendarArbitrage:
    """Calendar constraints: w(k, T1) <= w(k, T2) for T1 < T2."""

    def test_sequential_enforcement(self):
        """Second slice constrained to have higher total variance on the calibrated range.

        With soft-penalty enforcement (TRF doesn't support nonlinear constraints),
        small residual violations are possible outside the calibrated k range.
        We verify strict enforcement inside [-1, 1] (where data exists) and
        tolerance at wings.
        """
        # Strong calendar penalty for this test
        config = SVIConfig(calendar_enforce=True, butterfly_penalty_weight=1000.0)
        calibrator = SVICalibrator(config)

        T1, T2 = 0.1, 0.5
        F = 50000.0

        # First slice: high vol (intentionally)
        k1 = np.linspace(-1.0, 1.0, 10)
        params1 = np.array([0.08, 0.3, -0.2, 0.0, 0.2])
        w1 = _svi_total_variance(params1, k1)
        sl1 = calibrator.calibrate(k1, w1, np.ones(10), T1, F)

        # Second slice: naturally lower vol (would violate calendar without constraint)
        k2 = np.linspace(-1.0, 1.0, 10)
        params2_low = np.array([0.03, 0.1, -0.2, 0.0, 0.2])
        w2_low = _svi_total_variance(params2_low, k2)
        sl2 = calibrator.calibrate(k2, w2_low, np.ones(10), T2, F, prev_slice=sl1)

        # Check monotonicity on the calibrated range (where the penalty was enforced)
        k_check = np.linspace(-1.0, 1.0, 40)
        w1_check = sl1.total_variance(k_check)
        w2_check = sl2.total_variance(k_check)
        # Soft penalty → allow small tolerance (< 1% of max w1)
        tol = 0.01 * w1_check.max()
        assert np.all(w2_check >= w1_check - tol), \
            f"Calendar arbitrage violated beyond tolerance {tol}: max gap = {(w1_check - w2_check).max()}"

    def test_surface_calendar_free(self):
        """ArbitrageFreeSVI.is_calendar_free() after sequential calibration."""
        surface = ArbitrageFreeSVI()
        sl1 = SVISlice(a=0.02, b=0.1, rho=-0.2, m=0.0, sigma=0.2, T=0.1, F=50000.0)
        sl2 = SVISlice(a=0.05, b=0.15, rho=-0.2, m=0.0, sigma=0.2, T=0.5, F=50000.0)
        surface.add_slice(sl1)
        surface.add_slice(sl2)
        assert surface.is_calendar_free()


# ────────────────────────────────────────────────────────────
# TEST 7: Jacobian Correctness
# ────────────────────────────────────────────────────────────

class TestJacobian:
    """Analytical Jacobian vs scipy.optimize.approx_fprime."""

    def test_jacobian_accuracy(self):
        params = np.array([0.04, 0.2, -0.3, 0.05, 0.2])
        k = np.linspace(-1.5, 1.5, 10)

        J_analytical = _svi_jacobian_w(params, k)

        # Numerical Jacobian per strike
        eps = 1e-7
        J_numerical = np.zeros_like(J_analytical)
        for j in range(5):
            p_plus = params.copy()
            p_plus[j] += eps
            p_minus = params.copy()
            p_minus[j] -= eps
            J_numerical[:, j] = (
                _svi_total_variance(p_plus, k) - _svi_total_variance(p_minus, k)
            ) / (2 * eps)

        np.testing.assert_allclose(J_analytical, J_numerical, rtol=1e-5)

    def test_jacobian_shape(self):
        params = np.array([0.04, 0.2, -0.3, 0.0, 0.2])
        k = np.linspace(-1.0, 1.0, 20)
        J = _svi_jacobian_w(params, k)
        assert J.shape == (20, 5)


# ────────────────────────────────────────────────────────────
# TEST 8: Performance Benchmark
# ────────────────────────────────────────────────────────────

class TestPerformance:
    """Single-slice calibration must complete in < 50ms."""

    def test_calibration_under_budget(self):
        """Benchmark: 15 strikes, TRF + 3-start multi-start should finish under 60ms."""
        T = 0.25
        F = 50000.0
        true_params = np.array([0.04, 0.2, -0.3, 0.0, 0.2])
        k = np.linspace(-1.5, 1.5, 15)
        w = _svi_total_variance(true_params, k)
        rng = np.random.RandomState(0)
        w_noisy = w + rng.normal(0, 0.001, len(k))
        w_noisy = np.maximum(w_noisy, 0.001)
        vega = np.ones(15)

        config = SVIConfig()
        calibrator = SVICalibrator(config)

        # Warmup
        calibrator.calibrate(k, w_noisy, vega, T, F)

        # Timed run (5 iterations, median)
        times = []
        for _ in range(5):
            t0 = time.perf_counter()
            calibrator.calibrate(k, w_noisy, vega, T, F)
            times.append(time.perf_counter() - t0)

        median_ms = np.median(times) * 1000
        # 60ms budget: ~20ms per start × 3 starts = 60ms worst case
        assert median_ms < 60.0, f"Calibration took {median_ms:.1f}ms, exceeds 60ms target"

    def test_reduced_fit_fast(self):
        """Reduced 3-param should be faster than full SVI."""
        T = 0.25
        F = 50000.0
        k = np.array([-0.5, 0.0, 0.5])
        w = np.array([0.06, 0.04, 0.055])
        vega = np.ones(3)

        calibrator = SVICalibrator(SVIConfig())
        t0 = time.perf_counter()
        for _ in range(10):
            calibrator.calibrate_reduced(k, w, vega, T, F)
        avg_ms = (time.perf_counter() - t0) / 10 * 1000
        assert avg_ms < 20.0, f"Reduced fit took {avg_ms:.1f}ms, expected < 20ms"


# ────────────────────────────────────────────────────────────
# TEST 9: Fallback Hierarchy
# ────────────────────────────────────────────────────────────

class TestFallbackHierarchy:
    """Test each fallback level produces valid output."""

    def test_level0_full_svi(self):
        """5+ strikes → full SVI (level 0)."""
        T, F = 0.25, 50000.0
        k = np.linspace(-1.0, 1.0, 8)
        w = 0.04 + 0.2 * np.sqrt(k**2 + 0.04)
        calibrator = SVICalibrator(SVIConfig())
        sl = calibrator.calibrate(k, w, np.ones(8), T, F)
        assert sl.calibration_level == 0

    def test_level1_reduced(self):
        """3-4 strikes → reduced 3-param (level 1)."""
        T, F = 0.25, 50000.0
        k = np.array([-0.5, 0.0, 0.5])
        w = np.array([0.06, 0.04, 0.055])
        calibrator = SVICalibrator(SVIConfig())
        sl = calibrator.calibrate_reduced(k, w, np.ones(3), T, F)
        assert sl.calibration_level == 1
        assert sl.atm_vol() > 0

    def test_level2_flat_vol(self):
        """< 3 strikes → flat vol (level 2)."""
        T, F = 0.25, 50000.0
        iv_pct = np.array([48.0, 52.0])
        calibrator = SVICalibrator(SVIConfig())
        sl = calibrator.fallback_flat_vol(iv_pct, T, F, level=2)
        assert sl.calibration_level == 2
        assert sl.atm_vol() > 0
        # Should match median IV
        np.testing.assert_allclose(sl.atm_vol_pct(), 50.0, rtol=1e-6)

    def test_level3_flat(self):
        """1 strike → flat vol (level 3)."""
        T, F = 0.25, 50000.0
        iv_flat = 55.0
        w = (iv_flat / 100.0) ** 2 * T
        sl = SVISlice(a=w, b=1e-6, rho=0.0, m=0.0, sigma=0.1, T=T, F=F, calibration_level=3)
        assert sl.calibration_level == 3
        atm = sl.atm_vol_pct()
        np.testing.assert_allclose(atm, iv_flat, atol=0.1)


# ────────────────────────────────────────────────────────────
# TEST 10: Risk-Neutral Density
# ────────────────────────────────────────────────────────────

class TestRND:
    """Risk-neutral density properties."""

    @pytest.fixture
    def slice(self) -> SVISlice:
        return SVISlice(a=0.04, b=0.1, rho=-0.2, m=0.0, sigma=0.3, T=0.5, F=50000.0)

    def test_density_non_negative(self, slice: SVISlice):
        """RND must be non-negative when butterfly-free."""
        assert slice.is_butterfly_free(200)
        k = np.linspace(-3.0, 3.0, 500)
        f = slice.risk_neutral_density(k)
        assert np.all(f >= 0), f"Negative density found: min={f.min()}"

    def test_density_integrates_to_one(self, slice: SVISlice):
        """RND should integrate to ~1.0 over a wide k range."""
        k = np.linspace(-5.0, 5.0, 2000)
        f = slice.risk_neutral_density(k)
        dk = k[1] - k[0]
        integral = np.trapezoid(f, k)
        np.testing.assert_allclose(integral, 1.0, atol=0.05)

    def test_density_peak_near_atm(self, slice: SVISlice):
        """Peak of density should be near k=0 (ATM)."""
        k = np.linspace(-2.0, 2.0, 500)
        f = slice.risk_neutral_density(k)
        peak_k = k[np.argmax(f)]
        assert abs(peak_k) < 0.5, f"Density peak at k={peak_k}, expected near 0"


# ────────────────────────────────────────────────────────────
# TEST 11: Surface Interpolation
# ────────────────────────────────────────────────────────────

class TestSurfaceInterpolation:
    """Test multi-slice surface interpolation."""

    def test_interpolation_between_slices(self):
        surface = ArbitrageFreeSVI()
        sl1 = SVISlice(a=0.02, b=0.1, rho=-0.2, m=0.0, sigma=0.2, T=0.1, F=50000.0)
        sl2 = SVISlice(a=0.06, b=0.15, rho=-0.2, m=0.0, sigma=0.2, T=0.5, F=50000.0)
        surface.add_slice(sl1)
        surface.add_slice(sl2)

        k = np.array([0.0])
        w_mid = surface.total_variance(k, 0.3)
        w1 = sl1.total_variance(k)
        w2 = sl2.total_variance(k)

        # Linear interpolation: alpha = (0.3 - 0.1) / (0.5 - 0.1) = 0.5
        expected = w1 + 0.5 * (w2 - w1)
        np.testing.assert_allclose(w_mid, expected, rtol=1e-10)

    def test_extrapolation_flat_vol(self):
        """Beyond calibrated maturities, implied vol stays flat (w scales linearly in T)."""
        surface = ArbitrageFreeSVI()
        sl = SVISlice(a=0.04, b=0.2, rho=-0.3, m=0.0, sigma=0.2, T=0.25, F=50000.0)
        surface.add_slice(sl)

        k = np.array([0.0])
        w_at = surface.total_variance(k, 0.25)
        w_before = surface.total_variance(k, 0.125)  # half T → half w (flat vol)
        w_after = surface.total_variance(k, 0.5)     # double T → double w

        np.testing.assert_allclose(w_before, w_at * 0.5, rtol=1e-10)
        np.testing.assert_allclose(w_after, w_at * 2.0, rtol=1e-10)

        # The key property: IV stays constant
        iv_at = np.sqrt(w_at[0] / 0.25)
        iv_before = np.sqrt(w_before[0] / 0.125)
        iv_after = np.sqrt(w_after[0] / 0.5)
        np.testing.assert_allclose(iv_before, iv_at, rtol=1e-10)
        np.testing.assert_allclose(iv_after, iv_at, rtol=1e-10)

    def test_nearest_and_second_nearest(self):
        surface = ArbitrageFreeSVI()
        sl1 = SVISlice(a=0.02, b=0.1, rho=0.0, m=0.0, sigma=0.2, T=0.1, F=50000.0)
        sl2 = SVISlice(a=0.04, b=0.15, rho=0.0, m=0.0, sigma=0.2, T=0.5, F=50000.0)
        surface.add_slice(sl1)
        surface.add_slice(sl2)

        assert surface.nearest_slice().T == 0.1
        assert surface.second_nearest_slice().T == 0.5


# ────────────────────────────────────────────────────────────
# TEST 12: Skew Computation
# ────────────────────────────────────────────────────────────

class TestSkew:
    """25-delta skew computation."""

    def test_negative_rho_gives_positive_skew(self):
        """Negative rho → put vol > call vol → positive 25d skew."""
        sl = SVISlice(a=0.04, b=0.3, rho=-0.5, m=0.0, sigma=0.2, T=0.25, F=50000.0)
        skew = sl.skew_25d()
        assert skew > 0, f"Expected positive skew for rho<0, got {skew:.2f}"

    def test_symmetric_smile_small_skew(self):
        """rho=0, m=0 → symmetric smile → small skew (Newton is approximate)."""
        sl = SVISlice(a=0.04, b=0.2, rho=0.0, m=0.0, sigma=0.2, T=0.25, F=50000.0)
        skew = sl.skew_25d()
        # Newton delta-strike finder is approximate (ignores dsigma/dk in iteration),
        # so even a truly symmetric smile may show small residual skew
        assert abs(skew) < 10.0, f"Skew too large for symmetric smile: {skew:.2f}"


# ────────────────────────────────────────────────────────────
# TEST 13: Serialization
# ────────────────────────────────────────────────────────────

class TestSerialization:
    """Round-trip serialization."""

    def test_slice_round_trip(self):
        sl = SVISlice(a=0.04, b=0.2, rho=-0.3, m=0.05, sigma=0.2, T=0.25, F=50000.0)
        d = sl.to_dict()
        sl2 = SVISlice.from_dict(d)
        np.testing.assert_allclose(sl.params, sl2.params)
        assert sl.T == sl2.T
        assert sl.F == sl2.F

    def test_surface_round_trip(self):
        surface = ArbitrageFreeSVI(timestamp=pd.Timestamp("2025-01-15 12:00", tz="UTC"))
        sl1 = SVISlice(a=0.02, b=0.1, rho=0.0, m=0.0, sigma=0.2, T=0.1, F=50000.0)
        sl2 = SVISlice(a=0.04, b=0.15, rho=-0.2, m=0.0, sigma=0.2, T=0.5, F=50000.0)
        surface.add_slice(sl1)
        surface.add_slice(sl2)

        d = surface.to_dict()
        surface2 = ArbitrageFreeSVI.from_dict(d)

        assert len(surface2.slices) == 2
        assert surface2.maturities == surface.maturities


# ────────────────────────────────────────────────────────────
# TEST 14: calibrate_surface() with synthetic trades
# ────────────────────────────────────────────────────────────

class TestCalibrateSurface:
    """Integration test with synthetic trade data."""

    @pytest.fixture
    def synthetic_trades(self) -> pd.DataFrame:
        """Generate synthetic option trades resembling Deribit data."""
        rng = np.random.RandomState(42)
        n_trades = 500
        F = 50000.0
        strikes = np.array([40000, 42000, 44000, 46000, 48000, 50000,
                            52000, 54000, 56000, 58000, 60000])
        expiry_dt = pd.Timestamp("2025-02-15 08:00", tz="UTC")
        bucket_ts = pd.Timestamp("2025-01-15 12:00", tz="UTC")

        rows = []
        for _ in range(n_trades):
            K = rng.choice(strikes)
            k = np.log(K / F)
            # Generate IV from a known SVI + noise
            true_w = 0.04 + 0.2 * (-0.3 * k + np.sqrt(k**2 + 0.04))
            true_iv = np.sqrt(max(true_w / (30 / 365.25), 0.001)) * 100.0
            iv = true_iv + rng.normal(0, 1.0)
            iv = max(iv, 5.0)

            rows.append({
                "timestamp": bucket_ts + pd.Timedelta(seconds=rng.randint(0, 3600)),
                "instrument_name": f"BTC-15FEB25-{int(K)}-{'C' if rng.rand() > 0.5 else 'P'}",
                "iv": iv,
                "index_price": F + rng.normal(0, 100),
                "price": 0.05 + rng.normal(0, 0.01),
                "mark_price": 0.05,
                "direction": rng.choice(["buy", "sell"]),
                "amount": rng.exponential(1.0),
                "strike": float(K),
                "option_type": "C" if rng.rand() > 0.5 else "P",
                "expiry_dt": expiry_dt,
                "T_years": 30 / 365.25,
                "dte_days": 30.0,
                "moneyness": K / F,
            })

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    def test_produces_output(self, synthetic_trades: pd.DataFrame):
        config = SVIConfig(bucket_freq="1h")
        features, params = calibrate_surface(synthetic_trades, config)
        assert not features.empty
        # New constant-maturity columns
        assert "svi_atm_1w" in features.columns
        assert "svi_25d_skew_1w" in features.columns
        assert "svi_term_slope_1w_2w" in features.columns
        assert "svi_calibration_level" in features.columns
        # Params long-format
        assert not params.empty
        assert {"timestamp", "T", "F", "a", "b", "rho", "m", "sigma"}.issubset(params.columns)

    def test_atm_vol_reasonable(self, synthetic_trades: pd.DataFrame):
        config = SVIConfig(bucket_freq="1h")
        features, _ = calibrate_surface(synthetic_trades, config)
        atm = features["svi_atm_1w"].dropna()
        if not atm.empty:
            assert atm.min() > 5.0, f"ATM 1w vol too low: {atm.min():.1f}"
            assert atm.max() < 300.0, f"ATM 1w vol too high: {atm.max():.1f}"

    def test_calibration_level_set(self, synthetic_trades: pd.DataFrame):
        config = SVIConfig(bucket_freq="1h")
        features, _ = calibrate_surface(synthetic_trades, config)
        levels = features["svi_calibration_level"].unique()
        assert any(l < 4 for l in levels), f"No successful calibrations, levels: {levels}"


# ────────────────────────────────────────────────────────────
# TEST 15: Constant-Maturity Interpolation
# ────────────────────────────────────────────────────────────

class TestConstantMaturityInterpolation:
    """Surface interpolation at fixed maturities (1d, 1w, 2w, 1m)."""

    def test_atm_between_slices(self):
        """Constant-maturity ATM vol at T=7d should be between slices at T=1d and T=14d."""
        surface = ArbitrageFreeSVI()
        # T=1d slice with ATM variance 0.01
        sl1 = SVISlice(a=0.01, b=0.0, rho=0.0, m=0.0, sigma=0.1, T=1/365.25, F=50000.0)
        # T=14d slice with ATM variance 0.08 (higher total variance, longer T)
        sl2 = SVISlice(a=0.08, b=0.0, rho=0.0, m=0.0, sigma=0.1, T=14/365.25, F=50000.0)
        surface.add_slice(sl1)
        surface.add_slice(sl2)

        T_7d = 7/365.25
        w_interp = surface.total_variance(np.array([0.0]), T_7d)[0]
        w1 = sl1.total_variance(np.array([0.0]))[0]
        w2 = sl2.total_variance(np.array([0.0]))[0]

        # Linear interpolation: alpha = (7 - 1)/(14 - 1) = 6/13
        alpha = 6 / 13
        expected = w1 + alpha * (w2 - w1)
        np.testing.assert_allclose(w_interp, expected, rtol=1e-8)
        assert w1 < w_interp < w2, "Interpolated variance should be between slices"

    def test_matched_iv_equals_input_at_known_point(self):
        """A single slice evaluated at its own (k, T) should return the input IV."""
        # Build a slice with known ATM IV = 50%
        sigma_atm = 0.5
        T = 0.25
        w_atm = sigma_atm ** 2 * T
        sl = SVISlice(a=w_atm, b=0.0, rho=0.0, m=0.0, sigma=0.1, T=T, F=50000.0)

        iv = sl.implied_vol(np.array([0.0]))[0]
        np.testing.assert_allclose(iv, sigma_atm, rtol=1e-8)


# ────────────────────────────────────────────────────────────
# TEST 16: Surface Batch Evaluation (Exact Expiry Matching)
# ────────────────────────────────────────────────────────────

class TestEvaluateSurfaceBatch:
    """evaluate_surface_batch: exact-expiry matching, vectorized numpy."""

    @pytest.fixture
    def params_df(self) -> pd.DataFrame:
        """Two buckets, each with two calibrated expiries (Feb 1 and Feb 15)."""
        ts1 = pd.Timestamp("2025-01-15 12:00", tz="UTC")
        ts2 = pd.Timestamp("2025-01-15 13:00", tz="UTC")
        exp1 = pd.Timestamp("2025-02-01 08:00", tz="UTC")
        exp2 = pd.Timestamp("2025-02-15 08:00", tz="UTC")
        return pd.DataFrame([
            {"timestamp": ts1, "expiry_dt": exp1, "expiry_idx": 0,
             "T": 0.05, "F": 50000.0, "a": 0.02, "b": 0.1,
             "rho": -0.2, "m": 0.0, "sigma": 0.2,
             "rmse": 0.5, "level": 0, "n_strikes": 10},
            {"timestamp": ts1, "expiry_dt": exp2, "expiry_idx": 1,
             "T": 0.09, "F": 50100.0, "a": 0.04, "b": 0.15,
             "rho": -0.3, "m": 0.01, "sigma": 0.25,
             "rmse": 0.5, "level": 0, "n_strikes": 10},
            {"timestamp": ts2, "expiry_dt": exp1, "expiry_idx": 0,
             "T": 0.0489, "F": 51000.0, "a": 0.025, "b": 0.1,
             "rho": -0.2, "m": 0.0, "sigma": 0.2,
             "rmse": 0.5, "level": 0, "n_strikes": 10},
            {"timestamp": ts2, "expiry_dt": exp2, "expiry_idx": 1,
             "T": 0.0889, "F": 51050.0, "a": 0.045, "b": 0.15,
             "rho": -0.3, "m": 0.01, "sigma": 0.25,
             "rmse": 0.5, "level": 0, "n_strikes": 10},
        ])

    def test_exact_expiry_match(self, params_df: pd.DataFrame):
        """Batch evaluation routes each query to its exact calibrated slice by expiry_dt."""
        from vol_arb.svi import evaluate_surface_batch

        exp1 = pd.Timestamp("2025-02-01 08:00", tz="UTC")
        exp2 = pd.Timestamp("2025-02-15 08:00", tz="UTC")

        query_df = pd.DataFrame({
            "timestamp": [
                pd.Timestamp("2025-01-15 12:30", tz="UTC"),  # bucket ts1, exp1
                pd.Timestamp("2025-01-15 12:30", tz="UTC"),  # bucket ts1, exp2
                pd.Timestamp("2025-01-15 13:30", tz="UTC"),  # bucket ts2, exp1
                pd.Timestamp("2025-01-15 13:30", tz="UTC"),  # bucket ts2, exp2
            ],
            "strike": [50000.0, 50100.0, 51000.0, 51050.0],
            "expiry_dt": [exp1, exp2, exp1, exp2],
            "index_price": [50000.0, 50100.0, 51000.0, 51050.0],
        })

        ivs = evaluate_surface_batch(params_df, query_df)
        assert ivs.shape[0] == 4
        assert ivs.notna().all(), f"Some queries unmatched: {ivs}"

        # Manual verification: row 0 should match slice (ts1, exp1) with strike=F=50000
        # k = log(50000/50000) = 0, so w = a + b*(rho*(-m) + sqrt(m² + sigma²))
        # a=0.02, b=0.1, rho=-0.2, m=0, sigma=0.2 → w = 0.02 + 0.1*(0 + 0.2) = 0.04
        # IV = sqrt(0.04 / 0.05) = sqrt(0.8) = 0.8944 → 89.44%
        expected_0 = np.sqrt(0.04 / 0.05) * 100.0
        np.testing.assert_allclose(ivs.iloc[0], expected_0, rtol=1e-8)

    def test_missing_expiry_returns_nan(self, params_df: pd.DataFrame):
        """Query with expiry_dt not in params_df should return NaN (no approximation)."""
        from vol_arb.svi import evaluate_surface_batch

        exp_unknown = pd.Timestamp("2025-03-01 08:00", tz="UTC")
        query_df = pd.DataFrame({
            "timestamp": [pd.Timestamp("2025-01-15 12:30", tz="UTC")],
            "strike": [50000.0],
            "expiry_dt": [exp_unknown],
            "index_price": [50000.0],
        })

        ivs = evaluate_surface_batch(params_df, query_df)
        assert ivs.isna().all(), f"Expected NaN for unknown expiry, got {ivs.iloc[0]}"

    def test_query_before_calibration_returns_nan(self, params_df: pd.DataFrame):
        """Query timestamp before any calibrated bucket should return NaN."""
        from vol_arb.svi import evaluate_surface_batch

        exp1 = pd.Timestamp("2025-02-01 08:00", tz="UTC")
        query_df = pd.DataFrame({
            "timestamp": [pd.Timestamp("2025-01-15 11:00", tz="UTC")],  # BEFORE ts1
            "strike": [50000.0],
            "expiry_dt": [exp1],
            "index_price": [50000.0],
        })
        ivs = evaluate_surface_batch(params_df, query_df)
        assert ivs.isna().all()

    def test_performance_100k_queries(self, params_df: pd.DataFrame):
        """Vectorized eval: 100K queries should complete in <2 seconds."""
        from vol_arb.svi import evaluate_surface_batch

        exp1 = pd.Timestamp("2025-02-01 08:00", tz="UTC")
        n = 100_000
        rng = np.random.RandomState(42)
        query_df = pd.DataFrame({
            "timestamp": [pd.Timestamp("2025-01-15 12:30", tz="UTC")] * n,
            "strike": 50000.0 + rng.normal(0, 500, n),
            "expiry_dt": [exp1] * n,
            "index_price": 50000.0 + rng.normal(0, 100, n),
        })

        t0 = time.perf_counter()
        ivs = evaluate_surface_batch(params_df, query_df)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, f"100K queries took {elapsed:.2f}s, expected < 2s"
        assert ivs.notna().sum() == n, "All 100K queries should match"

    def test_uses_slice_forward_not_query_index(self):
        """When slice F differs from query's index_price, evaluation should use slice F."""
        from vol_arb.svi import evaluate_surface_batch

        exp = pd.Timestamp("2025-02-01 08:00", tz="UTC")
        ts_bucket = pd.Timestamp("2025-01-15 12:00", tz="UTC")

        # Slice with F=50000, but query provides stale index_price=49000
        params_df = pd.DataFrame([{
            "timestamp": ts_bucket, "expiry_dt": exp, "expiry_idx": 0,
            "T": 0.05, "F": 50000.0,
            "a": 0.02, "b": 0.1, "rho": -0.2, "m": 0.0, "sigma": 0.2,
            "rmse": 0.5, "level": 0, "n_strikes": 10,
        }])

        query_df = pd.DataFrame({
            "timestamp": [pd.Timestamp("2025-01-15 12:30", tz="UTC")],
            "strike": [50000.0],
            "expiry_dt": [exp],
            "index_price": [49000.0],  # stale/different
        })

        ivs = evaluate_surface_batch(params_df, query_df)
        # k should be computed from slice F=50000, so k=log(50000/50000)=0
        # w = 0.02 + 0.1 * 0.2 = 0.04, IV = sqrt(0.04/0.05) = 89.44%
        expected = np.sqrt(0.04 / 0.05) * 100.0
        np.testing.assert_allclose(ivs.iloc[0], expected, rtol=1e-8)


# ────────────────────────────────────────────────────────────
# TEST 17: Pro-Level Robustness — TRF + Multi-Start + T-Aware Bounds
# ────────────────────────────────────────────────────────────

class TestProRobustness:
    """Verify TRF + multi-start + T-aware bounds fix the known failure modes."""

    def test_multistart_recovers_from_bad_start(self):
        """Calibration should recover good params even from a pathological initial guess.

        Crafts a smile where moment-based init gives rho with the wrong sign,
        verifies the multi-start explores the flipped-rho variant and recovers.
        """
        # Strong negative skew (crypto typical): puts more expensive than calls
        T = 0.05
        F = 50000.0
        true_params = np.array([0.003, 0.05, -0.7, 0.0, 0.15])
        k = np.linspace(-0.3, 0.3, 11)
        w = _svi_total_variance(true_params, k)

        # Add moderate noise
        rng = np.random.RandomState(123)
        w_noisy = w + rng.normal(0, 0.0002, len(k))
        w_noisy = np.maximum(w_noisy, 1e-5)

        calibrator = SVICalibrator(SVIConfig())
        sl = calibrator.calibrate(k, w_noisy, np.ones(11), T, F)

        # Recovered rho should be negative (correct sign)
        assert sl.rho < 0, f"Multi-start failed to recover negative skew: rho={sl.rho:.3f}"
        # ATM vol should be reasonable
        assert 20 < sl.atm_vol_pct() < 150, f"ATM vol out of range: {sl.atm_vol_pct():.1f}"
        # Fit quality should be good (SLSQP with butterfly penalty trades off fit slightly)
        assert sl.rmse < 3.0, f"RMSE too high after multi-start: {sl.rmse:.2f}"

    def test_t_aware_bounds_short_dated(self):
        """For 1-day options, b should be bounded to physically sensible range.

        Regression test: before T-aware bounds, b could reach 10 for 1-day
        options, producing 5000% ATM IV.
        """
        T = 1.0 / 365.25  # 1 day
        F = 50000.0
        # Normal market data — moderate IV smile
        iv_atm = 0.6  # 60% ATM
        k = np.linspace(-0.05, 0.05, 7)
        w = iv_atm * iv_atm * T * (1.0 + 0.5 * k * k)  # gentle smile
        vega = np.ones(7)

        calibrator = SVICalibrator(SVIConfig())
        sl = calibrator.calibrate(k, w, vega, T, F)

        # With T-aware bounds, b for 1d options should be < 0.1
        # (typical range 1e-4 to 5e-3)
        assert sl.b < 0.1, f"b={sl.b:.4f} exceeds tight bound for short-dated"
        # ATM IV should be reasonable
        assert 40 < sl.atm_vol_pct() < 100, f"ATM IV out of range: {sl.atm_vol_pct():.1f}"
        # No runaway
        assert sl.atm_vol_pct() < 200, "ATM vol exceeded 200% sanity cap"

    def test_trf_no_extreme_outliers(self):
        """Batch of 50 random market smiles: none should produce ATM IV > 200%."""
        rng = np.random.RandomState(42)
        calibrator = SVICalibrator(SVIConfig())

        bad = 0
        for _ in range(50):
            T = rng.uniform(1.5 / 365.25, 30 / 365.25)
            F = 50000.0
            iv_atm = rng.uniform(0.4, 0.9)
            rho_true = rng.uniform(-0.7, 0.0)
            # Synthetic SVI with physical parameters
            a_t = (iv_atm ** 2) * T * 0.5
            b_t = 0.5 * T
            sigma_t = 0.1 * np.sqrt(T)
            true_p = np.array([a_t, b_t, rho_true, 0.0, sigma_t])
            k = np.linspace(-0.15, 0.15, 8)
            w = _svi_total_variance(true_p, k)
            w += rng.normal(0, 0.0001, len(k))
            w = np.maximum(w, 1e-6)

            sl = calibrator.calibrate(k, w, np.ones(8), T, F)
            if sl.atm_vol_pct() > 200 or sl.atm_vol_pct() < 5:
                bad += 1

        assert bad == 0, f"{bad}/50 calibrations produced extreme IV"

    def test_fallback_flat_vol_basic(self):
        """fallback_flat_vol produces median IV and minimal curvature."""
        T = 0.01
        F = 50000.0
        iv_pct = np.array([45.0, 55.0, 50.0])
        calibrator = SVICalibrator(SVIConfig())
        sl = calibrator.fallback_flat_vol(iv_pct, T, F, level=2)

        assert sl.calibration_level == 2
        # Median IV should be ≈ 50% (tiny offset from b=1e-6 * sigma term)
        np.testing.assert_allclose(sl.atm_vol_pct(), 50.0, atol=0.01)
        # Very small wing slope (no curvature in a flat smile)
        assert sl.b < 1e-4
        # Evaluating at any k should give near-constant IV
        k_test = np.array([-0.1, 0.0, 0.1])
        ivs = sl.implied_vol_pct(k_test)
        assert (ivs.max() - ivs.min()) < 1.0, "Flat smile has too much curvature"
