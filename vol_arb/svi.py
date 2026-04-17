"""Arbitrage-Free SVI Implied Volatility Surface Calibrator.

Implements the Raw SVI parameterization (Gatheral 2004) with:
    - Vega-weighted SLSQP optimization with analytical Jacobian
    - Butterfly arbitrage prevention via density function g(k) >= 0
    - Calendar arbitrage prevention via monotone total variance in T
    - 6-level fallback hierarchy for sparse crypto option data
    - Risk-Neutral Density extraction via Breeden-Litzenberger
    - Put-call parity forward price extraction

References:
    Gatheral, J. (2004). A parsimonious arbitrage-free implied volatility parameterization.
    Gatheral, J. & Jacquier, A. (2014). Arbitrage-free SVI volatility surfaces.
    Patton, A. (2011). Volatility forecast comparison using imperfect proxies.

Conventions:
    IV: percentage (e.g. 52.3) in DataFrames, decimal (0.523) internally.
    Total variance: w = sigma^2 * T (dimensionless).
    Log-moneyness: k = log(K / F) where F is the forward price.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from pydantic import BaseModel
from scipy.optimize import least_squares, minimize
from scipy.stats import norm

from vol_arb.greeks import compute_greeks

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# CONFIGURATION
# ────────────────────────────────────────────────────────────

class SVIConfig(BaseModel):
    """Configuration for SVI surface calibration."""

    # Aggregation
    bucket_freq: str = "1h"
    min_strikes: int = 5
    fallback_min_strikes: int = 3

    # IV aggregation within bucket
    outlier_zscore: float = 3.0

    # Optimizer
    max_iter: int = 200
    ftol: float = 1e-8

    # Arbitrage enforcement
    butterfly_penalty_weight: float = 100.0
    butterfly_grid_n: int = 20
    butterfly_verify_n: int = 200
    calendar_enforce: bool = True

    # Forward price
    use_pcp_forward: bool = True

    # Risk-neutral density
    compute_rnd: bool = False

    # Caching
    cache_params: bool = True
    cache_path: str = "vol_arb/output/svi_params_cache.parquet"

    # DTE range
    dte_min_days: float = 0.5
    dte_max_days: float = 90.0


# ────────────────────────────────────────────────────────────
# SVI SLICE — single maturity parameterization
# ────────────────────────────────────────────────────────────

@dataclass
class SVISlice:
    """Single-maturity SVI parameterization and evaluation.

    Raw SVI: w(k) = a + b * (rho * (k - m) + sqrt((k - m)^2 + sigma^2))

    Parameters
    ----------
    a : overall variance level
    b : slope angle (b > 0)
    rho : rotation / skew (-1 < rho < 1)
    m : horizontal translation
    sigma : ATM curvature (sigma > 0)
    T : time to expiry in years
    F : forward price
    """

    a: float
    b: float
    rho: float
    m: float
    sigma: float
    T: float
    F: float
    calibration_level: int = 0
    rmse: float = 0.0
    n_strikes: int = 0

    @property
    def params(self) -> np.ndarray:
        return np.array([self.a, self.b, self.rho, self.m, self.sigma])

    def total_variance(self, k: np.ndarray) -> np.ndarray:
        """w(k) = a + b * (rho * (k-m) + sqrt((k-m)^2 + sigma^2))."""
        k = np.asarray(k, dtype=np.float64)
        km = k - self.m
        return self.a + self.b * (
            self.rho * km + np.sqrt(km * km + self.sigma * self.sigma)
        )

    def implied_vol(self, k: np.ndarray) -> np.ndarray:
        """Black-Scholes implied vol (decimal) from total variance."""
        w = self.total_variance(k)
        w = np.maximum(w, 0.0)
        return np.sqrt(w / self.T)

    def implied_vol_pct(self, k: np.ndarray) -> np.ndarray:
        """Implied vol in percentage (e.g. 52.3)."""
        return self.implied_vol(k) * 100.0

    def w_prime(self, k: np.ndarray) -> np.ndarray:
        """First derivative w'(k) = b * (rho + (k-m) / sqrt((k-m)^2 + sigma^2))."""
        k = np.asarray(k, dtype=np.float64)
        km = k - self.m
        return self.b * (self.rho + km / np.sqrt(km * km + self.sigma * self.sigma))

    def w_double_prime(self, k: np.ndarray) -> np.ndarray:
        """Second derivative w''(k) = b * sigma^2 / ((k-m)^2 + sigma^2)^(3/2)."""
        k = np.asarray(k, dtype=np.float64)
        km = k - self.m
        denom = (km * km + self.sigma * self.sigma) ** 1.5
        return self.b * self.sigma * self.sigma / denom

    def g_butterfly(self, k: np.ndarray) -> np.ndarray:
        """Butterfly density function g(k). Must be >= 0 for no butterfly arbitrage.

        g(k) = (1 - k*w'/(2*w))^2 - w'^2/4 * (1/w + 1/4) + w''/2
        """
        k = np.asarray(k, dtype=np.float64)
        w = self.total_variance(k)
        wp = self.w_prime(k)
        wpp = self.w_double_prime(k)

        # Protect against w near zero
        w_safe = np.maximum(w, 1e-12)

        term1 = (1.0 - k * wp / (2.0 * w_safe)) ** 2
        term2 = wp * wp / 4.0 * (1.0 / w_safe + 0.25)
        term3 = wpp / 2.0

        return term1 - term2 + term3

    def is_butterfly_free(self, n_points: int = 200) -> bool:
        """Check g(k) >= 0 on a dense grid."""
        k_grid = np.linspace(-3.0, 3.0, n_points)
        g = self.g_butterfly(k_grid)
        return bool(np.all(g >= -1e-8))

    def atm_vol(self) -> float:
        """ATM implied vol (decimal) at k=0."""
        return float(self.implied_vol(np.array([0.0]))[0])

    def atm_vol_pct(self) -> float:
        """ATM implied vol in percentage."""
        return self.atm_vol() * 100.0

    def skew_25d(self) -> float:
        """25-delta put vol minus 25-delta call vol (percentage points).

        Approximates 25-delta strikes via k ≈ ±0.3 (roughly 25-delta for
        typical crypto vols). Uses Newton refinement if needed.
        """
        # For 25-delta: k_put ≈ -sigma_atm * sqrt(T) * 0.67, k_call ≈ +sigma_atm * sqrt(T) * 0.67
        # Approximate with fixed offsets, refined by BS delta inversion
        sigma_atm = self.atm_vol()
        if sigma_atm <= 0 or self.T <= 0:
            return 0.0

        sqrt_t = np.sqrt(self.T)
        # Initial guess for 25-delta log-moneyness
        k_25d = sigma_atm * sqrt_t * 0.6745  # norm.ppf(0.75) ≈ 0.6745

        # Refine: find k where delta = 0.25 (call) and delta = -0.25 (put)
        k_call = self._find_delta_strike(0.25, k_25d, is_call=True)
        k_put = self._find_delta_strike(-0.25, -k_25d, is_call=False)

        iv_put = self.implied_vol_pct(np.array([k_put]))[0]
        iv_call = self.implied_vol_pct(np.array([k_call]))[0]
        return iv_put - iv_call

    def _find_delta_strike(
        self, target_delta: float, k_init: float, is_call: bool, tol: float = 1e-4, max_iter: int = 10,
    ) -> float:
        """Newton's method to find k where BS delta = target_delta."""
        k = k_init
        for _ in range(max_iter):
            sigma_k = float(self.implied_vol(np.array([k]))[0])
            if sigma_k <= 0:
                return k
            sqrt_t = np.sqrt(self.T)
            d1 = (-k + 0.5 * sigma_k ** 2 * self.T) / (sigma_k * sqrt_t)
            delta = norm.cdf(d1) if is_call else norm.cdf(d1) - 1.0
            # d(delta)/dk via chain rule
            pdf_d1 = norm.pdf(d1)
            dd1_dk = -1.0 / (sigma_k * sqrt_t)
            d_delta_dk = pdf_d1 * dd1_dk
            if abs(d_delta_dk) < 1e-12:
                break
            k = k - (delta - target_delta) / d_delta_dk
            if abs(delta - target_delta) < tol:
                break
        return k

    def risk_neutral_density(self, k: np.ndarray) -> np.ndarray:
        """Risk-neutral density via Breeden-Litzenberger on SVI.

        f(k) = g(k) / sqrt(2*pi*w) * exp(-d_minus^2 / 2)
        where d_minus = -k/sqrt(w) - sqrt(w)/2
        """
        k = np.asarray(k, dtype=np.float64)
        w = self.total_variance(k)
        w_safe = np.maximum(w, 1e-12)
        g = self.g_butterfly(k)

        sqrt_w = np.sqrt(w_safe)
        d_minus = -k / sqrt_w - sqrt_w / 2.0

        density = g / np.sqrt(2.0 * np.pi * w_safe) * np.exp(-0.5 * d_minus * d_minus)
        return np.maximum(density, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "a": self.a, "b": self.b, "rho": self.rho, "m": self.m,
            "sigma": self.sigma, "T": self.T, "F": self.F,
            "calibration_level": self.calibration_level,
            "rmse": self.rmse, "n_strikes": self.n_strikes,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SVISlice:
        return cls(**d)


# ────────────────────────────────────────────────────────────
# SVI CALIBRATOR — single-slice optimizer
# ────────────────────────────────────────────────────────────

def _svi_total_variance(params: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Evaluate raw SVI total variance (standalone function for optimizer)."""
    a, b, rho, m, sigma = params
    km = k - m
    return a + b * (rho * km + np.sqrt(km * km + sigma * sigma))


def _svi_jacobian_w(params: np.ndarray, k: np.ndarray) -> np.ndarray:
    """Jacobian dw/d(params) — shape (N, 5) for N strikes.

    dw/da = 1
    dw/db = rho*(k-m) + sqrt((k-m)^2 + sigma^2)
    dw/drho = b*(k-m)
    dw/dm = b*(-rho - (k-m)/sqrt((k-m)^2 + sigma^2))
    dw/dsigma = b*sigma/sqrt((k-m)^2 + sigma^2)
    """
    a, b, rho, m, sigma = params
    km = k - m
    sqrt_term = np.sqrt(km * km + sigma * sigma)
    N = len(k)

    J = np.empty((N, 5), dtype=np.float64)
    J[:, 0] = 1.0                                          # dw/da
    J[:, 1] = rho * km + sqrt_term                         # dw/db
    J[:, 2] = b * km                                       # dw/drho
    J[:, 3] = b * (-rho - km / sqrt_term)                  # dw/dm
    J[:, 4] = b * sigma / sqrt_term                        # dw/dsigma
    return J


def _butterfly_penalty(params: np.ndarray, k_grid: np.ndarray) -> float:
    """Sum of squared butterfly violations on a grid."""
    a, b, rho, m, sigma = params
    km = k_grid - m
    sq = km * km + sigma * sigma
    sqrt_sq = np.sqrt(sq)

    w = a + b * (rho * km + sqrt_sq)
    wp = b * (rho + km / sqrt_sq)
    wpp = b * sigma * sigma / sq ** 1.5

    w_safe = np.maximum(w, 1e-12)
    g = (1.0 - k_grid * wp / (2.0 * w_safe)) ** 2 - wp * wp / 4.0 * (1.0 / w_safe + 0.25) + wpp / 2.0

    violations = np.minimum(g, 0.0)
    return float(np.sum(violations * violations))


def _butterfly_penalty_grad(params: np.ndarray, k_grid: np.ndarray) -> np.ndarray:
    """Gradient of butterfly penalty (numerical, only evaluated when violations exist)."""
    eps = 1e-7
    grad = np.zeros(5)
    base = _butterfly_penalty(params, k_grid)
    if base < 1e-15:
        return grad
    for i in range(5):
        p_plus = params.copy()
        p_plus[i] += eps
        grad[i] = (_butterfly_penalty(p_plus, k_grid) - base) / eps
    return grad


class SVICalibrator:
    """Fits market IV data to SVI parameters via SLSQP."""

    def __init__(self, config: SVIConfig):
        self.config = config

    def _initial_params(
        self, k: np.ndarray, w_market: np.ndarray, T: float,
    ) -> np.ndarray:
        """Moment-based initial parameter estimation."""
        # ATM total variance
        atm_idx = np.argmin(np.abs(k))
        w_atm = max(w_market[atm_idx], 1e-8)

        # Wing slope
        if len(k) >= 3:
            sorted_idx = np.argsort(k)
            k_sorted = k[sorted_idx]
            w_sorted = w_market[sorted_idx]
            dk = k_sorted[-1] - k_sorted[0]
            dw = w_sorted[-1] - w_sorted[0]
            b_init = max(abs(dw / max(dk, 0.01)), 0.01)
        else:
            b_init = max(w_atm / 2.0, 0.01)

        # Skew direction
        if len(k) >= 2:
            left = w_market[k < 0].mean() if np.any(k < 0) else w_atm
            right = w_market[k > 0].mean() if np.any(k > 0) else w_atm
            skew_sign = -1.0 if left > right else 1.0
            rho_init = skew_sign * 0.3
        else:
            rho_init = -0.3  # typical crypto: negative skew

        # Translation: smile trough
        m_init = k[np.argmin(w_market)]

        # Curvature
        k_range = max(k.max() - k.min(), 0.01)
        sigma_init = max(0.1 * k_range, 0.01)

        # Level: ensure a + b*sigma*sqrt(1-rho^2) >= 0
        a_init = w_atm - b_init * np.sqrt(m_init * m_init + sigma_init * sigma_init)

        return np.array([a_init, b_init, rho_init, m_init, sigma_init])

    @staticmethod
    def _t_aware_bounds(T: float) -> tuple[np.ndarray, np.ndarray]:
        """Physically-motivated parameter bounds scaled by time-to-expiry.

        `b` (wing slope in total-variance space) scales linearly with T.
        `sigma` (smile width at ATM) scales with sqrt(T). `a` is bounded
        so that ATM vol lies in [5%, 300%].
        """
        T_eff = max(T, 1.0 / 365.25)

        # Wing slope: generous upper bound (5x typical) to avoid clipping
        b_max = 5.0 * T_eff + 1e-4
        # Smile curvature: generous (3 sigma of forward range)
        sigma_max = 3.0 * np.sqrt(T_eff) + 1e-4

        # ATM variance range corresponds to IV in [5%, 300%]
        atm_var_max = (3.0 ** 2) * T_eff
        # Lower bound on `a` must accommodate worst-case b*sigma contribution
        a_lower = -(b_max * sigma_max + 1.0)
        a_upper = atm_var_max + 1.0

        lower = np.array([a_lower, 1e-8, -0.999, -3.0, 1e-6])
        upper = np.array([a_upper, b_max, 0.999, 3.0, sigma_max])
        return lower, upper

    @staticmethod
    def _fit_residuals_and_jac(
        k: np.ndarray,
        w_market: np.ndarray,
        sqrt_vw: np.ndarray,
    ):
        """Return (residuals, jacobian) closures for pure vega-weighted fit.

        Both are fully analytical and vectorized. The Jacobian is
        ``sqrt_vw[:, None] * dw/dparams``, computed via the closed-form
        SVI parameter partial derivatives. No finite differences, no
        butterfly/calendar penalties in the hot path — those are verified
        and refined post-hoc in a second stage (rarely needed in practice).
        """
        def residuals(params: np.ndarray) -> np.ndarray:
            w_svi = _svi_total_variance(params, k)
            return sqrt_vw * (w_svi - w_market)

        def jacobian(params: np.ndarray) -> np.ndarray:
            # Analytical: dw/dparams is (N, 5), scale rows by sqrt_vw
            return sqrt_vw[:, None] * _svi_jacobian_w(params, k)

        return residuals, jacobian

    @staticmethod
    def _fit_residuals_and_jac_with_penalty(
        k: np.ndarray,
        w_market: np.ndarray,
        sqrt_vw: np.ndarray,
        k_bf: np.ndarray,
        penalty_sqrt: float,
        cal_grid: np.ndarray | None = None,
        w_prev: np.ndarray | None = None,
        cal_penalty_sqrt: float = 0.0,
    ):
        """Stage-2 residuals (with butterfly + optional calendar penalties).

        Only used when stage 1 produces a slice that violates arbitrage-freedom.
        The Jacobian uses analytical fit terms + analytical butterfly/calendar
        gradients (computed via chain rule through w, w', w'').
        """
        n_fit = len(k)
        n_bf = len(k_bf)
        n_cal = len(cal_grid) if cal_grid is not None else 0

        def residuals(params: np.ndarray) -> np.ndarray:
            w_svi = _svi_total_variance(params, k)
            r_fit = sqrt_vw * (w_svi - w_market)

            a, b, rho, m, sigma = params
            km_bf = k_bf - m
            sq_bf = km_bf * km_bf + sigma * sigma
            sqrt_bf = np.sqrt(sq_bf)
            w_bf = a + b * (rho * km_bf + sqrt_bf)
            wp_bf = b * (rho + km_bf / sqrt_bf)
            wpp_bf = b * sigma * sigma / (sq_bf ** 1.5)
            w_safe = np.maximum(w_bf, 1e-12)
            g = (1.0 - k_bf * wp_bf / (2.0 * w_safe)) ** 2 - (wp_bf * wp_bf / 4.0) * (1.0 / w_safe + 0.25) + wpp_bf / 2.0
            r_bf = penalty_sqrt * np.minimum(g, 0.0)

            if n_cal:
                w_svi_cal = _svi_total_variance(params, cal_grid)
                gap = w_prev - w_svi_cal
                r_cal = cal_penalty_sqrt * np.maximum(gap, 0.0)
                return np.concatenate([r_fit, r_bf, r_cal])
            return np.concatenate([r_fit, r_bf])

        def jacobian(params: np.ndarray) -> np.ndarray:
            # Fit Jacobian (analytical)
            J_fit = sqrt_vw[:, None] * _svi_jacobian_w(params, k)

            # Numerical butterfly Jacobian — only hit in stage-2 fallback,
            # so the perf cost is amortized over rare cases.
            eps = 1e-6
            r0 = residuals(params)
            r0_bf = r0[n_fit:n_fit + n_bf]
            J_bf = np.zeros((n_bf, 5))
            for j in range(5):
                p_plus = params.copy()
                p_plus[j] += eps
                r_plus = residuals(p_plus)
                J_bf[:, j] = (r_plus[n_fit:n_fit + n_bf] - r0_bf) / eps

            if n_cal:
                r0_cal = r0[n_fit + n_bf:]
                J_cal = np.zeros((n_cal, 5))
                for j in range(5):
                    p_plus = params.copy()
                    p_plus[j] += eps
                    r_plus = residuals(p_plus)
                    J_cal[:, j] = (r_plus[n_fit + n_bf:] - r0_cal) / eps
                return np.vstack([J_fit, J_bf, J_cal])
            return np.vstack([J_fit, J_bf])

        return residuals, jacobian

    def calibrate(
        self,
        k: np.ndarray,
        w_market: np.ndarray,
        vega_weights: np.ndarray,
        T: float,
        F: float,
        prev_slice: SVISlice | None = None,
    ) -> SVISlice:
        """Calibrate SVI via SLSQP with analytical gradient + multi-start.

        Uses ``scipy.optimize.minimize(method='SLSQP')`` on a scalar objective
        (vega-weighted SSE + butterfly penalty), with the analytical gradient
        for the fit term and finite differences for the butterfly penalty.
        The butterfly penalty in the objective acts both as an arbitrage
        constraint AND a regularizer — removing it leads to overfitting on
        noisy data, so we keep it throughout optimization.

        Robustness additions over the original SLSQP calibrator:
        - **T-aware bounds**: `b` and `sigma` bounds scale with T, preventing
          runaway parameters for short-dated options
        - **Multi-start (3)**: primary + skew-flip + curvature-widen; keeps
          best fit by in-sample RMSE. Safeguards against local-min traps.
        - **Non-negative variance constraint**: a + b*σ*√(1-ρ²) ≥ 0

        Calendar arbitrage (if prev_slice given) is enforced via SLSQP
        inequality constraints on a 20-point grid in log-moneyness.
        """
        N = len(k)
        vw2 = vega_weights * vega_weights
        vw2 = vw2 / max(float(vw2.sum()), 1e-12)

        k_bf_grid = np.linspace(
            min(k.min(), -2.0), max(k.max(), 2.0), self.config.butterfly_grid_n,
        )
        penalty_w = self.config.butterfly_penalty_weight

        cal_grid = None
        w_prev = None
        if prev_slice is not None:
            cal_grid = np.linspace(-2.0, 2.0, 20)
            w_prev = prev_slice.total_variance(cal_grid)

        def objective(params: np.ndarray) -> float:
            w_model = _svi_total_variance(params, k)
            residual = w_model - w_market
            fit_loss = float(np.dot(vw2, residual * residual))
            bf_loss = penalty_w * _butterfly_penalty(params, k_bf_grid)
            return fit_loss + bf_loss

        def gradient(params: np.ndarray) -> np.ndarray:
            w_model = _svi_total_variance(params, k)
            residual = w_model - w_market
            J = _svi_jacobian_w(params, k)
            fit_grad = 2.0 * (vw2[:, None] * residual[:, None] * J).sum(axis=0)
            bf_grad = penalty_w * _butterfly_penalty_grad(params, k_bf_grid)
            return fit_grad + bf_grad

        # T-aware bounds (tight for short-dated, generous for long-dated)
        lb, ub = self._t_aware_bounds(T)
        bounds = list(zip(lb.tolist(), ub.tolist()))

        constraints: list[dict] = [
            # Non-negative variance: a + b*sigma*sqrt(1-rho^2) >= 0
            {
                "type": "ineq",
                "fun": lambda p: p[0] + p[1] * p[4] * np.sqrt(1.0 - p[2] * p[2]),
            },
        ]
        if prev_slice is not None and cal_grid is not None and w_prev is not None:
            for j in range(len(cal_grid)):
                kj, w_prev_j = float(cal_grid[j]), float(w_prev[j])
                constraints.append({
                    "type": "ineq",
                    "fun": lambda p, _k=kj, _w=w_prev_j: (
                        _svi_total_variance(p, np.array([_k]))[0] - _w
                    ),
                })

        # Multi-start: 3 perturbed initial points
        x0_primary = np.clip(self._initial_params(k, w_market, T), lb, ub)
        rho_init = x0_primary[2]
        starts = [
            x0_primary,
            np.clip(x0_primary * np.array([1.0, 1.5, 1.0, 1.0, 1.3]), lb, ub),
            np.clip(
                np.array([x0_primary[0], x0_primary[1], -rho_init, x0_primary[3], x0_primary[4]]),
                lb, ub,
            ),
        ]

        iv_market_pct = np.sqrt(np.maximum(w_market, 0.0) / max(T, 1e-10)) * 100.0

        best_params: np.ndarray | None = None
        best_obj = np.inf

        for x0 in starts:
            try:
                result = minimize(
                    objective, x0,
                    method="SLSQP",
                    jac=gradient,
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": self.config.max_iter, "ftol": self.config.ftol},
                )
                obj_val = float(result.fun)
                if np.isfinite(obj_val) and obj_val < best_obj:
                    best_obj = obj_val
                    best_params = result.x
            except (ValueError, np.linalg.LinAlgError):
                continue

        if best_params is None:
            iv_median = float(np.median(iv_market_pct)) if len(iv_market_pct) else 50.0
            w_flat = (iv_median / 100.0) ** 2 * T
            return SVISlice(
                a=w_flat, b=1e-6, rho=0.0, m=0.0, sigma=0.1,
                T=T, F=F, calibration_level=2, rmse=0.0, n_strikes=N,
            )

        # Compute RMSE in vol points for the winning fit
        w_fit = _svi_total_variance(best_params, k)
        iv_fit_pct = np.sqrt(np.maximum(w_fit, 0.0) / max(T, 1e-10)) * 100.0
        rmse = float(np.sqrt(np.mean((iv_fit_pct - iv_market_pct) ** 2)))

        sl = SVISlice(
            a=float(best_params[0]), b=float(best_params[1]),
            rho=float(best_params[2]), m=float(best_params[3]),
            sigma=float(best_params[4]),
            T=T, F=F, calibration_level=0, rmse=rmse, n_strikes=N,
        )

        # Verify butterfly; if violated, retry once with stronger penalty
        if not sl.is_butterfly_free(self.config.butterfly_verify_n):
            penalty_w_strong = penalty_w * 10.0

            def obj_strong(params: np.ndarray) -> float:
                w_m = _svi_total_variance(params, k)
                r = w_m - w_market
                return (
                    float(np.dot(vw2, r * r))
                    + penalty_w_strong * _butterfly_penalty(params, k_bf_grid)
                )

            try:
                result2 = minimize(
                    obj_strong, best_params,
                    method="SLSQP",
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": self.config.max_iter, "ftol": self.config.ftol},
                )
                w_fit2 = _svi_total_variance(result2.x, k)
                iv_fit2 = np.sqrt(np.maximum(w_fit2, 0.0) / max(T, 1e-10)) * 100.0
                rmse2 = float(np.sqrt(np.mean((iv_fit2 - iv_market_pct) ** 2)))
                sl = SVISlice(
                    a=float(result2.x[0]), b=float(result2.x[1]),
                    rho=float(result2.x[2]), m=float(result2.x[3]),
                    sigma=float(result2.x[4]),
                    T=T, F=F, calibration_level=0, rmse=rmse2, n_strikes=N,
                )
            except (ValueError, np.linalg.LinAlgError):
                pass

        return sl

    def calibrate_reduced(
        self,
        k: np.ndarray,
        w_market: np.ndarray,
        vega_weights: np.ndarray,
        T: float,
        F: float,
    ) -> SVISlice:
        """Reduced 3-parameter symmetric SVI (rho=0, m=0) for 3-4 strikes.

        w(k) = a + b * sqrt(k^2 + sigma^2)
        """
        vw2 = vega_weights * vega_weights
        vw2 = vw2 / max(vw2.sum(), 1e-12)

        # Initial params
        atm_idx = np.argmin(np.abs(k))
        w_atm = max(w_market[atm_idx], 1e-8)
        k_range = max(k.max() - k.min(), 0.01)

        sigma_init = max(0.1 * k_range, 0.01)
        b_init = max(w_atm / sigma_init, 0.01)
        a_init = w_atm - b_init * sigma_init

        def objective(p: np.ndarray) -> float:
            a, b, sigma = p
            w_model = a + b * np.sqrt(k * k + sigma * sigma)
            residual = w_model - w_market
            return float(np.dot(vw2, residual * residual))

        result = minimize(
            objective,
            x0=np.array([a_init, b_init, sigma_init]),
            method="SLSQP",
            bounds=[(None, None), (1e-6, 10.0), (1e-6, 5.0)],
            constraints=[
                {"type": "ineq", "fun": lambda p: p[0] + p[1] * p[2]},  # w >= 0
            ],
            options={"maxiter": 100, "ftol": 1e-8},
        )

        a, b, sigma = result.x
        w_fit = a + b * np.sqrt(k * k + sigma * sigma)
        iv_fit = np.sqrt(np.maximum(w_fit, 0.0) / T) * 100.0
        iv_market = np.sqrt(np.maximum(w_market, 0.0) / T) * 100.0
        rmse = float(np.sqrt(np.mean((iv_fit - iv_market) ** 2)))

        return SVISlice(
            a=a, b=b, rho=0.0, m=0.0, sigma=sigma,
            T=T, F=F, calibration_level=1, rmse=rmse, n_strikes=len(k),
        )

    def fallback_flat_vol(
        self, iv_market_pct: np.ndarray, T: float, F: float, level: int = 2,
    ) -> SVISlice:
        """Fallback: flat smile at the median market IV.

        Used when there are too few strikes for a meaningful curvature fit
        (< 3 strikes) or when no optimization converged. Produces an SVI slice
        with near-zero wing slope (b ≈ 0) and ATM variance matching the median
        market IV.
        """
        iv_median_pct = float(np.median(iv_market_pct)) if len(iv_market_pct) else 50.0
        w_flat = (iv_median_pct / 100.0) ** 2 * max(T, 1e-10)
        rmse = float(np.std(iv_market_pct)) if len(iv_market_pct) > 1 else 0.0
        return SVISlice(
            a=w_flat, b=1e-6, rho=0.0, m=0.0, sigma=0.1,
            T=T, F=F, calibration_level=level, rmse=rmse, n_strikes=len(iv_market_pct),
        )


# ────────────────────────────────────────────────────────────
# ARBITRAGE-FREE SVI SURFACE — multi-slice
# ────────────────────────────────────────────────────────────

@dataclass
class ArbitrageFreeSVI:
    """Multi-slice SVI surface with calendar arbitrage enforcement.

    Holds calibrated SVISlice objects keyed by T (time to expiry).
    Calendar arbitrage is enforced by constraining each slice so that
    total variance is monotonically increasing in T.
    """

    slices: dict[float, SVISlice] = field(default_factory=dict)
    timestamp: pd.Timestamp | None = None

    def add_slice(self, sl: SVISlice) -> None:
        self.slices[sl.T] = sl

    @property
    def maturities(self) -> list[float]:
        return sorted(self.slices.keys())

    def nearest_slice(self, T: float | None = None) -> SVISlice | None:
        """Return slice with nearest maturity to T. If T is None, return shortest."""
        if not self.slices:
            return None
        if T is None:
            return self.slices[min(self.slices)]
        mats = np.array(list(self.slices.keys()))
        idx = np.argmin(np.abs(mats - T))
        return self.slices[mats[idx]]

    def second_nearest_slice(self) -> SVISlice | None:
        """Return the second-shortest maturity slice."""
        mats = self.maturities
        if len(mats) < 2:
            return None
        return self.slices[mats[1]]

    def total_variance(self, k: np.ndarray, T: float) -> np.ndarray:
        """Interpolated total variance at arbitrary T.

        Linear interpolation in total variance between bracketing slices.
        Beyond the calibrated range, extrapolates under the flat-vol assumption
        (w scales linearly with T), which is the arbitrage-free choice for
        a well-calibrated surface.
        """
        k = np.asarray(k, dtype=np.float64)
        mats = self.maturities
        if not mats:
            return np.full_like(k, np.nan)

        if T <= 0:
            return np.zeros_like(k)

        if T <= mats[0]:
            # Flat-vol extrapolation: same sigma(k) as shortest slice,
            # so w(k, T) = w(k, T_1) * T / T_1
            return self.slices[mats[0]].total_variance(k) * (T / mats[0])
        if T >= mats[-1]:
            # Flat-vol extrapolation on the long end
            return self.slices[mats[-1]].total_variance(k) * (T / mats[-1])

        # Find bracketing slices
        for i in range(len(mats) - 1):
            if mats[i] <= T <= mats[i + 1]:
                w1 = self.slices[mats[i]].total_variance(k)
                w2 = self.slices[mats[i + 1]].total_variance(k)
                alpha = (T - mats[i]) / (mats[i + 1] - mats[i])
                return w1 + alpha * (w2 - w1)

        return self.slices[mats[-1]].total_variance(k) * (T / mats[-1])

    def implied_vol(self, k: np.ndarray, T: float) -> np.ndarray:
        """Interpolated implied vol (decimal) at arbitrary (k, T)."""
        w = self.total_variance(k, T)
        return np.sqrt(np.maximum(w, 0.0) / max(T, 1e-8))

    def atm_vol(self, T: float | None = None) -> float:
        """ATM vol (decimal) at maturity T."""
        if T is not None:
            return float(self.implied_vol(np.array([0.0]), T)[0])
        sl = self.nearest_slice()
        return sl.atm_vol() if sl else 0.0

    def atm_vol_pct(self, T: float | None = None) -> float:
        return self.atm_vol(T) * 100.0

    def is_calendar_free(self, n_k: int = 50) -> bool:
        """Verify total variance is monotone increasing in T."""
        mats = self.maturities
        if len(mats) < 2:
            return True
        k_grid = np.linspace(-2.0, 2.0, n_k)
        for i in range(len(mats) - 1):
            w1 = self.slices[mats[i]].total_variance(k_grid)
            w2 = self.slices[mats[i + 1]].total_variance(k_grid)
            if np.any(w2 < w1 - 1e-8):
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": str(self.timestamp) if self.timestamp else None,
            "slices": {str(T): sl.to_dict() for T, sl in self.slices.items()},
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ArbitrageFreeSVI:
        ts = pd.Timestamp(d["timestamp"]) if d.get("timestamp") else None
        slices = {float(T): SVISlice.from_dict(sl) for T, sl in d["slices"].items()}
        return cls(slices=slices, timestamp=ts)


# ────────────────────────────────────────────────────────────
# DATA AGGREGATION HELPERS
# ────────────────────────────────────────────────────────────

def _extract_forward_pcp(
    group: pd.DataFrame, T: float, r: float = 0.0,
) -> float | None:
    """Extract implied forward from put-call parity at matching strikes.

    For Deribit inverse options (BTC-settled), mark_price is in BTC:
        C_usd - P_usd = (F - K) * exp(-rT)
        S * (C_btc - P_btc) = (F - K) * exp(-rT)

    With r=0:  F = K + S * (C_btc - P_btc)

    Uses near-ATM pairs only (|moneyness - 1| < 0.10) for stability,
    weighted by inverse distance to ATM to reduce wing noise.
    """
    calls = group[group["option_type"] == "C"]
    puts = group[group["option_type"] == "P"]
    if calls.empty or puts.empty:
        return None

    # Pivot to per-strike rows with call/put mark prices
    c_agg = calls.groupby("strike").agg(
        c_mark=("mark_price_median", "first"),
        c_S=("index_price", "first"),
    )
    p_agg = puts.groupby("strike").agg(
        p_mark=("mark_price_median", "first"),
        p_S=("index_price", "first"),
    )
    paired = c_agg.join(p_agg, how="inner")
    if paired.empty:
        return None

    K_arr = paired.index.values
    S_arr = (paired["c_S"].values + paired["p_S"].values) / 2.0
    moneyness = K_arr / S_arr

    # Filter to near-ATM (|moneyness - 1| < 0.10)
    atm_mask = np.abs(moneyness - 1.0) < 0.10
    if not np.any(atm_mask):
        # Relax to 0.20 if nothing near ATM
        atm_mask = np.abs(moneyness - 1.0) < 0.20
    if not np.any(atm_mask):
        return None

    K_atm = K_arr[atm_mask]
    S_atm = S_arr[atm_mask]
    C_btc = paired["c_mark"].values[atm_mask]
    P_btc = paired["p_mark"].values[atm_mask]

    # F = K + S * (C_btc - P_btc)  [PCP with r=0, Deribit inverse]
    disc = np.exp(-r * T) if r != 0 else 1.0
    F_per_strike = K_atm + S_atm * (C_btc - P_btc) * disc

    # Weight by inverse distance to ATM (ATM pairs are most reliable)
    mon_atm = K_atm / S_atm
    weights = 1.0 / (np.abs(mon_atm - 1.0) + 0.001)
    weights /= weights.sum()

    F = float(np.dot(weights, F_per_strike))

    # Sanity check: forward shouldn't deviate from spot by more than 5%
    S_median = float(np.median(S_atm))
    if abs(F / S_median - 1.0) > 0.05:
        return None

    return F


def _aggregate_bucket(
    bucket_trades: pd.DataFrame, config: SVIConfig,
) -> tuple[pd.DataFrame, float]:
    """Aggregate trades within a (bucket, expiry) group.

    Returns (aggregated DataFrame with columns [strike, k, iv_median, w, vega_weight], forward).
    """
    # Median IV per (strike, option_type)
    agg = (
        bucket_trades
        .groupby(["strike", "option_type"])
        .agg(
            iv_median=("iv", "median"),
            mark_price_median=("mark_price", "median"),
            n_trades=("iv", "count"),
            index_price=("index_price", "median"),
            T_years=("T_years", "median"),
        )
        .reset_index()
    )

    if agg.empty:
        return agg, 0.0

    # Forward price: prefer PCP-implied forward, fallback to index_price
    S_fallback = float(agg["index_price"].median())
    T = float(agg["T_years"].median())

    F_pcp = None
    if config.use_pcp_forward:
        F_pcp = _extract_forward_pcp(agg, T)
    F = F_pcp if F_pcp is not None else S_fallback

    # Collapse call/put to single IV per strike (use call-equivalent: average)
    strike_agg = (
        agg.groupby("strike")
        .agg(iv_median=("iv_median", "median"), n_trades=("n_trades", "sum"))
        .reset_index()
    )

    # Log-moneyness
    strike_agg["k"] = np.log(strike_agg["strike"].values / F)

    # Total variance
    iv_dec = strike_agg["iv_median"].values / 100.0
    strike_agg["w"] = iv_dec * iv_dec * T

    # Outlier removal: drop strikes beyond z-score threshold from median IV
    iv_median_all = strike_agg["iv_median"].median()
    iv_std = strike_agg["iv_median"].std()
    if iv_std > 0 and len(strike_agg) > 3:
        zscore = (strike_agg["iv_median"] - iv_median_all).abs() / iv_std
        strike_agg = strike_agg[zscore <= config.outlier_zscore].copy()

    # Compute BS vega weights
    if len(strike_agg) > 0:
        from vol_arb.greeks import compute_greeks as cg
        sigma_dec = strike_agg["iv_median"].values / 100.0
        S_arr = np.full(len(strike_agg), F)
        K_arr = strike_agg["strike"].values
        T_arr = np.full(len(strike_agg), T)
        is_call = strike_agg["k"].values >= 0  # call for OTM calls, put for OTM puts

        greeks = cg(S_arr, K_arr, T_arr, sigma_dec, is_call, r=0.0)
        strike_agg["vega_weight"] = np.abs(greeks["vega"])
        # Replace NaN vega with small value
        strike_agg["vega_weight"] = strike_agg["vega_weight"].fillna(0.01)
    else:
        strike_agg["vega_weight"] = 1.0

    return strike_agg, F


# ────────────────────────────────────────────────────────────
# VECTORIZED PRE-AGGREGATION
# ────────────────────────────────────────────────────────────

def _preaggregate_all(
    trades: pd.DataFrame, config: SVIConfig,
) -> dict[tuple, dict[str, np.ndarray]]:
    """Vectorized pre-aggregation of ALL trades into per-(bucket, expiry) numpy arrays.

    Does in one pass what _aggregate_bucket() does per-group:
    median IV per strike, call/put collapse, outlier removal, log-moneyness,
    total variance, and BS vega weights.

    Returns dict keyed by (bucket_ts, expiry_key) -> {k, w, vega, T, F, iv_median}.
    """
    # Step 1: Median IV per (bucket, expiry, strike, option_type).
    # Include expiry_dt (actual Timestamp, not just string key) for exact matching downstream.
    agg = (
        trades.groupby(["bucket", "expiry_key", "strike", "option_type"], observed=True)
        .agg(
            iv_median=("iv", "median"),
            n_trades=("iv", "count"),
            index_price=("index_price", "median"),
            T_years=("T_years", "median"),
            expiry_dt=("expiry_dt", "first"),
        )
        .reset_index()
    )

    # Step 2: Collapse call/put to single IV per strike
    strike_agg = (
        agg.groupby(["bucket", "expiry_key", "strike"], observed=True)
        .agg(
            iv_median=("iv_median", "median"),
            n_trades=("n_trades", "sum"),
            index_price=("index_price", "median"),
            T_years=("T_years", "median"),
            expiry_dt=("expiry_dt", "first"),
        )
        .reset_index()
    )

    # Step 3: Forward price (index_price median per bucket+expiry)
    F_series = strike_agg.groupby(["bucket", "expiry_key"], observed=True)["index_price"].transform("median")
    strike_agg["F"] = F_series.values
    T_series = strike_agg.groupby(["bucket", "expiry_key"], observed=True)["T_years"].transform("median")
    strike_agg["T"] = T_series.values

    # Step 4: Log-moneyness and total variance (vectorized)
    strike_agg["k"] = np.log(strike_agg["strike"].values / strike_agg["F"].values)
    iv_dec = strike_agg["iv_median"].values / 100.0
    strike_agg["w"] = iv_dec * iv_dec * strike_agg["T"].values

    # Step 5: Outlier z-score per (bucket, expiry) group
    grp_stats = strike_agg.groupby(["bucket", "expiry_key"], observed=True)["iv_median"].transform
    iv_med = grp_stats("median")
    iv_std = grp_stats("std").fillna(0)
    grp_count = strike_agg.groupby(["bucket", "expiry_key"], observed=True)["iv_median"].transform("count")
    zscore = ((strike_agg["iv_median"] - iv_med).abs() / iv_std.clip(lower=1e-8))
    # Only filter groups with > 3 strikes
    keep = (zscore <= config.outlier_zscore) | (grp_count <= 3) | (iv_std == 0)
    strike_agg = strike_agg[keep].copy()

    # Step 6: Vectorized BS vega for ALL strikes at once
    vega_arr = np.full(len(strike_agg), 0.01)
    valid = (strike_agg["T"].values > 0) & (iv_dec[keep.values] > 0) if len(strike_agg) > 0 else np.array([], dtype=bool)
    if len(strike_agg) > 0:
        from vol_arb.greeks import compute_greeks as cg
        # Recompute iv_dec after outlier removal
        iv_dec_clean = strike_agg["iv_median"].values / 100.0
        greeks = cg(
            strike_agg["F"].values, strike_agg["strike"].values,
            strike_agg["T"].values, iv_dec_clean,
            strike_agg["k"].values >= 0, r=0.0,
        )
        vega_arr = np.abs(greeks["vega"])
        vega_arr = np.where(np.isnan(vega_arr), 0.01, np.maximum(vega_arr, 0.01))
    strike_agg["vega_weight"] = vega_arr

    # Step 7: Partition into per-(bucket, expiry) numpy arrays
    result: dict[tuple, dict] = {}
    for (bucket_ts, exp_key), grp in strike_agg.groupby(["bucket", "expiry_key"], observed=True):
        result[(bucket_ts, exp_key)] = {
            "k": grp["k"].values.astype(np.float64),
            "w": grp["w"].values.astype(np.float64),
            "vega": grp["vega_weight"].values.astype(np.float64),
            "iv_median": grp["iv_median"].values.astype(np.float64),
            "T": float(grp["T"].iloc[0]),
            "F": float(grp["F"].iloc[0]),
            "n_strikes": len(grp),
            "expiry_dt": grp["expiry_dt"].iloc[0],  # pd.Timestamp for exact matching
        }

    return result


# ────────────────────────────────────────────────────────────
# PER-BUCKET CALIBRATION (top-level for pickling by ProcessPoolExecutor)
# ────────────────────────────────────────────────────────────

# Fixed maturities (in years) for constant-maturity feature extraction
_CM_MATURITIES = {
    "1d": 1.0 / 365.25,
    "1w": 7.0 / 365.25,
    "2w": 14.0 / 365.25,
    "1m": 30.0 / 365.25,
}


def _constant_maturity_features(
    surface: ArbitrageFreeSVI, bucket_ts,
) -> dict:
    """Extract constant-maturity features from a calibrated surface.

    Samples ATM vol, 25d skew, and term structure at fixed maturities (1d, 1w, 2w, 1m).
    These are comparable across time, unlike nearest-expiry features.
    """
    features: dict = {"timestamp": bucket_ts}

    for label, T in _CM_MATURITIES.items():
        w = surface.total_variance(np.array([0.0]), T)[0]
        iv_dec = np.sqrt(max(w, 0.0) / T) if T > 0 else 0.0
        features[f"svi_atm_{label}"] = iv_dec * 100.0

    # 25-delta skew at 1w and 1m — use approximate k offsets
    # For 25-delta: k ≈ ±sigma_atm * sqrt(T) * 0.6745
    for label in ["1w", "1m"]:
        T = _CM_MATURITIES[label]
        atm_iv = features[f"svi_atm_{label}"] / 100.0
        if atm_iv <= 0 or T <= 0:
            features[f"svi_25d_skew_{label}"] = np.nan
            continue
        k_25d = atm_iv * np.sqrt(T) * 0.6745
        iv_put = np.sqrt(max(surface.total_variance(np.array([-k_25d]), T)[0], 0.0) / T) * 100.0
        iv_call = np.sqrt(max(surface.total_variance(np.array([k_25d]), T)[0], 0.0) / T) * 100.0
        features[f"svi_25d_skew_{label}"] = iv_put - iv_call

    # Term structure slopes (vol points)
    features["svi_term_slope_1w_2w"] = features["svi_atm_2w"] - features["svi_atm_1w"]
    features["svi_term_slope_1w_1m"] = features["svi_atm_1m"] - features["svi_atm_1w"]

    # Nearest-expiry fit quality (for coverage/quality filters)
    near = surface.nearest_slice()
    if near is not None:
        features["svi_calibration_level"] = near.calibration_level
        features["svi_fit_rmse"] = near.rmse
        features["svi_n_strikes"] = near.n_strikes
        features["svi_butterfly_free"] = near.is_butterfly_free(50)
        features["svi_n_expiries"] = len(surface.slices)
    else:
        features["svi_calibration_level"] = 4
        features["svi_fit_rmse"] = np.nan
        features["svi_n_strikes"] = 0
        features["svi_butterfly_free"] = True
        features["svi_n_expiries"] = 0

    # ── SVI-ATM instantaneous spot vol (2026-04-17 Tier 2 feature) ──
    # Fan/Wang/Ye 2025 (arxiv:2604.02743) document ~6% QLIKE reduction on
    # SPX from options-implied spot-vol. We use the nearest-calibrated-slice
    # ATM IV directly (no T→0 extrapolation — safer than interpolating into
    # sparse sub-1d tenors which tend to use fallback level 1-3). Quality
    # gate: require level < 3, n_strikes >= 3, butterfly-free; else NaN so
    # LightGBM's native NaN handling takes over.
    if near is not None and (
        near.calibration_level < 3
        and near.n_strikes >= 3
        and near.is_butterfly_free(50)
    ):
        features["svi_atm_spot"] = near.atm_vol_pct()
    else:
        features["svi_atm_spot"] = np.nan

    return features


def _empty_features(bucket_ts) -> dict:
    """Empty constant-maturity feature row (for buckets with no calibration)."""
    f = {"timestamp": bucket_ts}
    for label in _CM_MATURITIES:
        f[f"svi_atm_{label}"] = np.nan
    for label in ["1w", "1m"]:
        f[f"svi_25d_skew_{label}"] = np.nan
    f["svi_term_slope_1w_2w"] = np.nan
    f["svi_term_slope_1w_1m"] = np.nan
    f["svi_calibration_level"] = 4
    f["svi_fit_rmse"] = np.nan
    f["svi_n_strikes"] = 0
    f["svi_butterfly_free"] = True
    f["svi_n_expiries"] = 0
    f["svi_atm_spot"] = np.nan
    return f


def _calibrate_one_bucket(
    args: tuple,
) -> tuple[dict, list[dict]]:
    """Calibrate all expiry slices for a single time bucket.

    Returns (features_dict, params_list) where:
        features_dict: constant-maturity features for this bucket (wide format)
        params_list: list of per-expiry surface param dicts (long format)

    Accepts pre-aggregated numpy arrays (from _preaggregate_all).
    Top-level function so ProcessPoolExecutor can pickle it.
    """
    bucket_ts, expiry_arrays, config = args
    calibrator = SVICalibrator(config)

    # Each entry: (T, SVISlice, expiry_dt). Carrying expiry_dt enables exact-match
    # evaluation downstream (no T interpolation needed).
    expiry_slices: list[tuple[float, SVISlice, Any]] = []
    prev_slice: SVISlice | None = None

    for exp_key in sorted(expiry_arrays.keys()):
        d = expiry_arrays[exp_key]
        T, F = d["T"], d["F"]
        if T <= 0 or F <= 0:
            continue

        n_unique = d["n_strikes"]
        k, w, vw = d["k"], d["w"], d["vega"]
        expiry_dt = d["expiry_dt"]

        # Calibration dispatch:
        #   Level 0: 5+ strikes → full 5-param SVI (TRF + 3-start multi-start)
        #   Level 1: 3-4 strikes → reduced 3-param symmetric SVI
        #   Level 2: < 3 strikes → flat vol from median IV
        if n_unique >= config.min_strikes:
            sl = calibrator.calibrate(
                k, w, vw, T, F,
                prev_slice=prev_slice if config.calendar_enforce else None,
            )
        elif n_unique >= config.fallback_min_strikes:
            sl = calibrator.calibrate_reduced(k, w, vw, T, F)
        else:
            sl = calibrator.fallback_flat_vol(d["iv_median"], T, F, level=2)

        # Belt-and-suspenders post-fit sanity check: downgrade to flat vol
        # for pathological fits (ATM IV out of [5%, 300%] or RMSE > 20 vp).
        atm_iv_pct = sl.atm_vol_pct()
        if not (5.0 <= atm_iv_pct <= 300.0) or sl.rmse > 20.0 or not np.isfinite(atm_iv_pct):
            sl = calibrator.fallback_flat_vol(d["iv_median"], T, F, level=2)

        expiry_slices.append((T, sl, expiry_dt))
        if config.calendar_enforce and sl.calibration_level == 0:
            prev_slice = sl

    if not expiry_slices:
        return _empty_features(bucket_ts), []

    # Build surface for constant-maturity feature extraction
    surface = ArbitrageFreeSVI(timestamp=bucket_ts)
    for T, sl, _exp_dt in expiry_slices:
        surface.add_slice(sl)

    features = _constant_maturity_features(surface, bucket_ts)

    # Long-format params (one row per expiry slice) — carries expiry_dt for exact matching
    params_list = []
    for expiry_idx, (T, sl, exp_dt) in enumerate(sorted(expiry_slices, key=lambda x: x[0])):
        params_list.append({
            "timestamp": bucket_ts,
            "expiry_dt": exp_dt,
            "expiry_idx": expiry_idx,
            "T": sl.T,
            "F": sl.F,
            "a": sl.a,
            "b": sl.b,
            "rho": sl.rho,
            "m": sl.m,
            "sigma": sl.sigma,
            "rmse": sl.rmse,
            "level": sl.calibration_level,
            "n_strikes": sl.n_strikes,
        })

    return features, params_list


# ────────────────────────────────────────────────────────────
# TOP-LEVEL CALIBRATION FUNCTION
# ────────────────────────────────────────────────────────────

def slice_from_params_row(row: dict | pd.Series) -> SVISlice:
    """Reconstruct an SVISlice from a params row (dict or Series)."""
    return SVISlice(
        a=float(row["a"]), b=float(row["b"]),
        rho=float(row["rho"]), m=float(row["m"]),
        sigma=float(row["sigma"]),
        T=float(row["T"]), F=float(row["F"]),
        calibration_level=int(row.get("level", 0)),
        rmse=float(row.get("rmse", 0.0)),
        n_strikes=int(row.get("n_strikes", 0)),
    )


def surface_from_params_group(params_group: pd.DataFrame) -> ArbitrageFreeSVI:
    """Reconstruct ArbitrageFreeSVI from a params DataFrame (rows for one bucket)."""
    if params_group.empty:
        return ArbitrageFreeSVI()
    ts = params_group["timestamp"].iloc[0] if "timestamp" in params_group.columns else None
    surface = ArbitrageFreeSVI(timestamp=ts)
    for _, row in params_group.iterrows():
        surface.add_slice(slice_from_params_row(row))
    return surface


def evaluate_surface_batch(
    params_df: pd.DataFrame,
    query_df: pd.DataFrame,
) -> pd.Series:
    """Evaluate the SVI surface at each query row's (strike, expiry_dt) — exact matching.

    Fully vectorized: one ``merge_asof(by=expiry_key)`` + one numpy SVI formula.
    For each query, finds the most recent SVI slice calibrated at or before the
    query's timestamp **for that specific expiry date** (no T interpolation).

    This is the pro-grade approach used by real options desks: per-expiry smile
    calibration + exact-match evaluation. Removes all interpolation and
    extrapolation approximation error.

    Parameters
    ----------
    params_df : long-format surface params with columns
        [timestamp, expiry_dt, T, F, a, b, rho, m, sigma] (and optional extras)
    query_df : DataFrame with columns [timestamp, strike, expiry_dt, index_price]
        Each row is a traded instrument (e.g., composite straddle).

    Returns
    -------
    pd.Series of implied vol (percentage) aligned to query_df.index.
    NaN where no slice exists for the query's (timestamp, expiry_dt).
    """
    if params_df.empty or query_df.empty:
        return pd.Series(np.nan, index=query_df.index, name="svi_iv_matched")

    required_q_cols = {"timestamp", "strike", "expiry_dt"}
    if not required_q_cols.issubset(query_df.columns):
        raise ValueError(f"query_df must have columns {required_q_cols}")
    if "expiry_dt" not in params_df.columns:
        raise ValueError("params_df must have 'expiry_dt' column (regenerate SVI batch)")

    # Expiry key for merge_asof `by=` — use YYYY-MM-DD (matches Deribit expiry semantics)
    q = query_df.copy()
    q["_q_idx"] = np.arange(len(q))
    q["_exp_key"] = pd.to_datetime(q["expiry_dt"], utc=True).dt.strftime("%Y-%m-%d")
    q = q.sort_values("timestamp").reset_index(drop=True)

    p = params_df[["timestamp", "expiry_dt", "T", "F", "a", "b", "rho", "m", "sigma"]].copy()
    p["_exp_key"] = pd.to_datetime(p["expiry_dt"], utc=True).dt.strftime("%Y-%m-%d")
    p = p.drop(columns=["expiry_dt"]).sort_values("timestamp").reset_index(drop=True)

    # Normalize timestamp dtype for pandas 3.0 compatibility (avoids
    # "incompatible merge keys datetime64[us, UTC]" when one side was
    # created by a different Python/pandas version).
    _ts_dtype = "datetime64[us, UTC]"
    q["timestamp"] = q["timestamp"].astype(_ts_dtype)
    p["timestamp"] = p["timestamp"].astype(_ts_dtype)

    # merge_asof with by='_exp_key': for each query, find most recent slice
    # calibrated at or before query.timestamp FOR THAT SPECIFIC EXPIRY DATE.
    merged = pd.merge_asof(
        q, p,
        on="timestamp",
        by="_exp_key",
        direction="backward",
        suffixes=("", "_slice"),
    )

    # Vectorized SVI formula — use SLICE's forward (more accurate than query's index_price).
    # Fall back to query's index_price where slice F is unavailable (edge cases).
    F_slice = merged["F"].values
    has_slice = ~np.isnan(F_slice)

    if "index_price" in merged.columns:
        F_fallback = merged["index_price"].values
    else:
        F_fallback = F_slice
    F_use = np.where(has_slice, F_slice, F_fallback)

    strike = merged["strike"].values
    a = merged["a"].values
    b = merged["b"].values
    rho = merged["rho"].values
    m = merged["m"].values
    sigma = merged["sigma"].values
    T_slice = merged["T"].values

    # Log-moneyness at the SLICE's forward
    with np.errstate(invalid="ignore", divide="ignore"):
        k = np.log(strike / F_use)

    km = k - m
    sqrt_term = np.sqrt(km * km + sigma * sigma)
    w = a + b * (rho * km + sqrt_term)
    # Convert to IV using the slice's own T (IV is time-stationary over short gaps)
    with np.errstate(invalid="ignore", divide="ignore"):
        iv_dec = np.sqrt(np.maximum(w, 0.0) / np.maximum(T_slice, 1e-10))
    iv_pct = iv_dec * 100.0

    # Invalidate rows that didn't match any slice for their expiry
    iv_pct = np.where(has_slice, iv_pct, np.nan)

    # Reorder back to original query_df index
    out_sorted = pd.Series(iv_pct, index=merged["_q_idx"].values)
    out = out_sorted.reindex(np.arange(len(query_df))).values
    return pd.Series(out, index=query_df.index, name="svi_iv_matched")


def calibrate_surface(
    trades: pd.DataFrame,
    config: SVIConfig | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibrate SVI surface per time bucket from option trades.

    Parameters
    ----------
    trades : DataFrame from load_option_trades() — all strikes and expiries.
             Must have: timestamp, strike, option_type, iv, index_price, T_years,
                        dte_days, mark_price.
    config : SVIConfig (defaults used if None)

    Returns
    -------
    DataFrame with one row per bucket, columns:
        timestamp, svi_atm_vol, svi_atm_vol_T2, svi_25d_skew, svi_25d_skew_T2,
        svi_term_slope, svi_fit_rmse, svi_n_strikes, svi_calibration_level,
        svi_butterfly_free, svi_forward
    """
    if config is None:
        config = SVIConfig()

    calibrator = SVICalibrator(config)

    # Filter by DTE range
    trades = trades[
        (trades["dte_days"] >= config.dte_min_days)
        & (trades["dte_days"] <= config.dte_max_days)
    ].copy()

    if trades.empty:
        logger.warning("No trades in DTE range [%.1f, %.1f] days", config.dte_min_days, config.dte_max_days)
        return pd.DataFrame(), pd.DataFrame()

    # Bucket timestamps
    trades["bucket"] = trades["timestamp"].dt.floor(config.bucket_freq)

    # Create expiry key for grouping
    trades["expiry_key"] = trades["expiry_dt"].dt.strftime("%Y-%m-%d")

    logger.info("SVI calibration: %d trades, DTE [%.0f, %.0f] days",
                len(trades), config.dte_min_days, config.dte_max_days)

    # Vectorized pre-aggregation: ONE pass for all (bucket, expiry, strike) groups
    t_preagg = time.perf_counter()
    pre_agg = _preaggregate_all(trades, config)
    logger.info("Pre-aggregation: %d (bucket, expiry) groups in %.1fs",
                len(pre_agg), time.perf_counter() - t_preagg)

    # Reorganize by bucket: {bucket_ts: {exp_key: arrays}}
    bucket_map: dict[Any, dict[str, dict]] = {}
    for (bucket_ts, exp_key), arrays in pre_agg.items():
        bucket_map.setdefault(bucket_ts, {})[exp_key] = arrays
    buckets = sorted(bucket_map.keys())
    logger.info("SVI calibration: %d buckets", len(buckets))

    # Build args for parallel calibration
    from concurrent.futures import ProcessPoolExecutor, as_completed
    import os

    n_workers = int(os.environ.get("SVI_MAX_WORKERS", min(os.cpu_count() or 4, 8)))
    bucket_args = [(bucket_ts, bucket_map[bucket_ts], config) for bucket_ts in buckets]

    t_start = time.perf_counter()

    feat_rows: list[dict] = []
    param_rows: list[dict] = []

    if len(buckets) > 100:
        logger.info("Using %d parallel workers", n_workers)
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_calibrate_one_bucket, args): args[0] for args in bucket_args}
            for future in as_completed(futures):
                features, params = future.result()
                feat_rows.append(features)
                param_rows.extend(params)
    else:
        for args in bucket_args:
            features, params = _calibrate_one_bucket(args)
            feat_rows.append(features)
            param_rows.extend(params)

    # Sort features by timestamp
    feat_rows.sort(key=lambda r: r["timestamp"])

    n_level: dict[int, int] = {}
    for r in feat_rows:
        lvl = r["svi_calibration_level"]
        n_level[lvl] = n_level.get(lvl, 0) + 1

    elapsed = time.perf_counter() - t_start
    logger.info(
        "SVI calibration complete: %d buckets in %.1fs (%.1f ms/bucket)",
        len(feat_rows), elapsed, 1000 * elapsed / max(len(feat_rows), 1),
    )
    for lvl, cnt in sorted(n_level.items()):
        if cnt > 0:
            logger.info("  Level %d: %d buckets (%.1f%%)", lvl, cnt, 100 * cnt / max(len(feat_rows), 1))

    if not feat_rows:
        return pd.DataFrame(), pd.DataFrame()

    features_df = pd.DataFrame(feat_rows)
    features_df["timestamp"] = pd.to_datetime(features_df["timestamp"], utc=True)

    # LOOKAHEAD FIX: bucket floored to start of hour uses trades from the ENTIRE hour.
    # Shift timestamp to END of bucket so merge_asof(backward) only assigns this SVI
    # to bars AFTER the bucket closes — no future trade contamination.
    features_df["timestamp"] = features_df["timestamp"] + pd.Timedelta(config.bucket_freq)
    features_df = features_df.sort_values("timestamp").reset_index(drop=True)

    # Surface params (long-format). Includes `expiry_dt` (absolute Timestamp) for
    # exact-expiry matching in evaluate_surface_batch — no T_years interpolation.
    params_cols = ["timestamp", "expiry_dt", "expiry_idx", "T", "F",
                   "a", "b", "rho", "m", "sigma", "rmse", "level", "n_strikes"]
    params_df = pd.DataFrame(param_rows) if param_rows else pd.DataFrame(columns=params_cols)
    if not params_df.empty:
        params_df["timestamp"] = pd.to_datetime(params_df["timestamp"], utc=True)
        # Shift bucket timestamp to end of window (lookahead-safe).
        # expiry_dt is NOT shifted — it's an absolute calendar date.
        params_df["timestamp"] = params_df["timestamp"] + pd.Timedelta(config.bucket_freq)
        params_df["expiry_dt"] = pd.to_datetime(params_df["expiry_dt"], utc=True)
        params_df = params_df[params_cols]
        params_df = params_df.sort_values(["timestamp", "expiry_idx"]).reset_index(drop=True)

    return features_df, params_df
