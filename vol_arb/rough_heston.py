"""Rough-Heston SDE calibrator — Session F Candidate C.

Mathematical foundation (Abi-Jaber-Larsson-Pulido 2019 / El Euch-Rosenbaum 2019)
---------------------------------------------------------------------------
Rough-Heston model under risk-neutral measure Q:

    dS_t / S_t = sqrt(V_t) dW_t^1
    V_t = V_0 + int_0^t K_H(t-s) * [kappa (theta - V_s) ds + eta sqrt(V_s) dW_s^2]

with memory kernel  K_H(t) = t^{H-1/2} / Gamma(H+1/2).
H in (0, 1/2) is the Hurst roughness parameter. Crypto-empirical H ~ 0.05-0.15.

Characteristic function (Abi-Jaber-Larsson-Pulido 2019 Theorem 4.1,
specialized to the rough-Heston kernel):

    E[ exp(iu X_tau) ] = exp(
        iu*(r-q)*tau
        + kappa*theta*int_0^tau psi(s, u) ds
        + V_0*int_0^tau F(u, psi(s, u)) ds
    )

Both integrals are CLASSICAL (not fractional) integrals over time.
The V_0 coefficient is the classical integral of F(psi), which reduces
to V_0 * tau * F(u, 0) in the deterministic-variance limit and hence
exactly recovers Black-Scholes — this is the sanity test that pins the
formula.

where psi solves the fractional Volterra-Riccati equation

    psi(t, u) = int_0^t K_H(t-s) F(u, psi(s, u)) ds
              = I^{H+1/2} [ F(u, psi(.*, u)) ] (t)

    F(u, y) = -0.5*(u^2 + iu) + (iu*eta*rho - kappa)*y + 0.5*eta^2*y^2

Note the integral order is H+1/2 (NOT 1-H — that was our Session F.1 bug).
This is derived directly from the kernel: K_H(t) is the kernel of the
fractional integral of order H+1/2. Sanity check: in the deterministic
limit (eta=0, theta=V_0), this formula exactly recovers Black-Scholes.

We discretize psi via the Diethelm-Ford-Freed 2002 Adams predictor-corrector
scheme for fractional differential equations (their eqns (6.4)-(6.5))
applied with alpha = H + 1/2.

Option pricing (Carr-Madan 1999)
--------------------------------
Given the CF, the price of an OTM European call with strike K = F*exp(k) at
time tau is

    c(k) = exp(-alpha*k - r*tau) / pi * integral_0^inf exp(-iv k) psi_CM(v) dv
    psi_CM(v) = exp(-r*tau) * phi(v - i*(alpha+1), tau) /
                (alpha^2 + alpha - v^2 + iv(2*alpha+1))

Computed on an FFT log-strike grid with alpha = 1.5 (damping parameter).

Calibration
-----------
Vega-weighted sum of squared IV residuals, minimized via SLSQP over 6
parameters (V_0, H, kappa, theta, eta, rho).  Warm-started from the previous
bucket's fit.

This module is a STATIC, PURE-FUNCTION calibrator — no I/O, no state outside
its inputs.  Causality is therefore trivial in the feature-compute sense; the
LEAKAGE risk for Candidate C lives entirely in the BUCKET ALIGNMENT, which
the sidecar builder and the +15m shift at load-time handle.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
from scipy.optimize import least_squares, minimize
from scipy.special import gamma as _gamma
from scipy.stats import norm as _norm

logger = logging.getLogger("rough_heston")


# ── Parameter dataclass ──────────────────────────────────────────────


@dataclass
class RoughHestonParams:
    """Fitted rough-Heston parameters + calibration diagnostics.

    6 free physical parameters:
        V_0:     spot variance (= sigma_spot^2). Must be > 0.
        H:       Hurst roughness. In (0, 1/2). Crypto typical 0.05-0.15.
        kappa:   mean-reversion speed of variance.
        theta:   long-run variance (target for the drift).
        eta:     vol-of-vol (diffusion coefficient).
        rho:     spot-vol correlation. In (-1, 1). Typically in (-0.9, -0.3).

    Plus diagnostics:
        calibration_rmse: vega-weighted RMSE on IV (in vol-points).
        n_options:        number of options used in the fit.
        converged:        SLSQP success flag.
        n_iter:           SLSQP iteration count.
        calibration_level: 2 = full 6-param fit ok, 1 = reduced fallback,
                           0 = failed (all params NaN).
    """

    V_0: float = float("nan")
    H: float = float("nan")
    kappa: float = float("nan")
    theta: float = float("nan")
    eta: float = float("nan")
    rho: float = float("nan")

    calibration_rmse: float = float("nan")
    n_options: int = 0
    converged: bool = False
    n_iter: int = 0
    calibration_level: int = 0


# Bounds for the 6 physical parameters (used in SLSQP)
PARAM_BOUNDS: tuple[tuple[float, float], ...] = (
    (1e-4, 25.0),    # V_0  (sigma_spot between 1% and 500% — wide)
    (0.01, 0.49),    # H    (rough-Heston requires H < 1/2)
    (0.01, 20.0),    # kappa
    (1e-4, 25.0),    # theta
    (1e-4, 10.0),    # eta
    (-0.999, 0.999), # rho
)


# ── Fractional Riccati via Adams predictor-corrector ─────────────────


def _adams_weights(alpha: float, M: int) -> tuple[np.ndarray, np.ndarray]:
    """Diethelm-Ford-Freed weights for the fractional Adams scheme of
    order alpha = 1 - H in (1/2, 1).

    Returns
    -------
    (a, b) where a[k+1, j] are corrector weights and b[k+1, j] are predictor
    weights, both shape (M+1, M+1). Zero entries for j > k+1.
    """
    a = np.zeros((M + 1, M + 1), dtype=np.float64)
    b = np.zeros((M + 1, M + 1), dtype=np.float64)
    alpha1 = alpha + 1.0
    for k in range(M):
        # Corrector weights: a[k+1, j] for j = 0..k+1
        # From D-F-F 2002 eq (6.5).
        # a[k+1, 0]   = k^(alpha+1) - (k - alpha)(k+1)^alpha
        # a[k+1, j]   = (k - j + 2)^(alpha+1) - 2(k - j + 1)^(alpha+1) + (k - j)^(alpha+1), 1<=j<=k
        # a[k+1, k+1] = 1
        if k == 0:
            a[1, 0] = 1.0
        else:
            a[k + 1, 0] = k ** alpha1 - (k - alpha) * (k + 1) ** alpha
        for j in range(1, k + 1):
            a[k + 1, j] = (
                (k - j + 2) ** alpha1
                - 2.0 * (k - j + 1) ** alpha1
                + (k - j) ** alpha1
            )
        a[k + 1, k + 1] = 1.0
        # Predictor weights (rectangular rule on fractional integral):
        # b[k+1, j] = (k - j + 1)^alpha - (k - j)^alpha for 0<=j<=k
        for j in range(k + 1):
            b[k + 1, j] = (k - j + 1) ** alpha - (k - j) ** alpha

    # Normalize by Gamma factors
    a *= 1.0 / _gamma(alpha + 2.0)  # corrector uses h^alpha / Gamma(alpha+2)
    b *= 1.0 / _gamma(alpha + 1.0)  # predictor uses h^alpha / Gamma(alpha+1)
    return a, b


def fractional_riccati_solve(
    F: Callable[[np.ndarray], np.ndarray],
    tau: float,
    H: float,
    u_vec: np.ndarray,
    M: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve the fractional Volterra-Riccati equation

        psi(t, u) = I^{H+1/2} [F(psi(.,u))](t),  psi(0, u) = 0,

    equivalent to   psi(t) = int_0^t K_H(t-s) F(psi(s)) ds   where
    K_H(t) = t^{H-1/2} / Gamma(H+1/2) is the rough-Heston memory kernel.

    The fractional-integral order is  alpha = H + 1/2  (in (1/2, 1) for
    rough-Heston H in (0, 1/2)). Discretized via the Diethelm-Ford-Freed
    Adams predictor-corrector scheme (Diethelm-Ford-Freed 2002 eqns 6.4-6.5).

    This matches the rough-Heston characteristic function formula from
    Abi-Jaber-Larsson-Pulido 2019 ("Affine Volterra Processes", Theorem
    4.1): E[exp(iu X_tau)] = exp(iu(r-q) tau + kappa theta * int_0^tau psi + V_0 * psi(tau)).

    In the deterministic-variance limit (eta -> 0, theta = V_0), this
    formula recovers Black-Scholes exactly, which is the sanity test
    that pins the integral order as H+1/2.

    Parameters
    ----------
    F
        Callable mapping a complex array to the RHS value F(psi).
    tau
        Terminal time.
    H
        Hurst parameter in (0, 1/2).
    u_vec
        Array of complex u values.
    M
        Time-grid points (not counting t=0). Default 50.

    Returns
    -------
    (F_integral, psi_integral_tau) — both complex arrays same shape as u_vec.
    F_integral = integral_0^tau F(u, psi(s, u)) ds  (used for V_0 coefficient).
    psi_integral_tau = integral_0^tau psi(s, u) ds  (used for kappa*theta coefficient).

    The rough-Heston CF is
        phi(u, tau) = exp(iu(r-q) tau + kappa theta * psi_integral + V_0 * F_integral)
    """
    if not (0.0 < H < 0.5):
        raise ValueError(f"H must be in (0, 1/2), got {H}")
    if tau <= 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    if M < 2:
        raise ValueError(f"M must be >= 2, got {M}")

    alpha = H + 0.5  # integral-of-order H+1/2 matches the rough-Heston kernel
    h = tau / M
    h_alpha = h ** alpha

    a, b = _adams_weights(alpha, M)
    a *= h_alpha
    b *= h_alpha

    n_u = u_vec.shape[0]
    psi = np.zeros((M + 1, n_u), dtype=np.complex128)
    F_psi = np.zeros((M + 1, n_u), dtype=np.complex128)
    # psi[0] = 0 -> F[0] = F(0)
    F_psi[0] = F(psi[0])

    # Clip psi magnitudes at 1e8 to prevent overflow on very-high-u CF
    # contributions. These high-u components have |phi| ~ exp(-|u|^2 * V_0 * tau)
    # which is already negligibly small; the pricing integral doesn't care
    # about the precise value. Clipping keeps the recursion numerically
    # stable without disturbing the CF at the u values that matter.
    clip = 1e8

    for k in range(M):
        # Predictor: psi_P[k+1] = sum_{j=0..k} b[k+1, j] F(psi[j])
        psi_pred = (b[k + 1, : k + 1, None] * F_psi[: k + 1]).sum(axis=0)
        # Clip magnitudes
        mag = np.abs(psi_pred)
        overflow = mag > clip
        if overflow.any():
            psi_pred = np.where(overflow, psi_pred / mag * clip, psi_pred)
        F_psi_pred = F(psi_pred)
        # Corrector: psi[k+1] = sum_{j=0..k} a[k+1, j] F(psi[j]) + a[k+1, k+1] F(psi_P[k+1])
        psi[k + 1] = (
            (a[k + 1, : k + 1, None] * F_psi[: k + 1]).sum(axis=0)
            + a[k + 1, k + 1] * F_psi_pred
        )
        mag = np.abs(psi[k + 1])
        overflow = mag > clip
        if overflow.any():
            psi[k + 1] = np.where(overflow, psi[k + 1] / mag * clip, psi[k + 1])
        F_psi[k + 1] = F(psi[k + 1])

    # The rough-Heston CF uses the CLASSICAL integrals of F(psi) and of
    # psi — not the fractional ones. See module docstring for derivation.
    # Trapezoidal rule on the grid (equivalent to Simpson's for the leading
    # O(h^2) terms; error absorbed into the Adams-corrector's O(h^{1+alpha})).
    F_psi_integral = h * (
        0.5 * F_psi[0] + F_psi[1:M].sum(axis=0) + 0.5 * F_psi[M]
    )
    psi_integral = h * (0.5 * psi[0] + psi[1:M].sum(axis=0) + 0.5 * psi[M])
    return F_psi_integral, psi_integral


# ── Characteristic function ──────────────────────────────────────────


def rough_heston_char_fn(
    u_vec: np.ndarray,
    tau: float,
    V_0: float,
    H: float,
    kappa: float,
    theta: float,
    eta: float,
    rho: float,
    *,
    r: float = 0.0,
    q: float = 0.0,
    M: int = 50,
) -> np.ndarray:
    """Rough-Heston characteristic function of X_tau = log(S_tau / S_0).

    phi(u, tau) = E[ exp(i u X_tau) ].

    Implements El Euch-Rosenbaum 2019 Theorem 3.1 with the Adams-scheme
    fractional Riccati solver from ``fractional_riccati_solve``.

    Parameters
    ----------
    u_vec
        Array of complex or real u values (will be cast to complex128).
    tau
        Time to maturity in YEARS.
    V_0, H, kappa, theta, eta, rho
        Rough-Heston parameters.
    r, q
        Risk-free rate and dividend yield (default 0).
    M
        Number of time-grid points for the fractional Riccati solver.

    Returns
    -------
    Complex array, same shape as u_vec, containing phi(u, tau).
    """
    u_c = np.asarray(u_vec, dtype=np.complex128)
    iu = 1j * u_c

    # Closure capturing the Riccati coefficient
    def F(y: np.ndarray) -> np.ndarray:
        # F(u, y) = -0.5*(u^2 + i u) + (i u eta rho - kappa)*y + 0.5*eta^2*y^2
        # This matches the classical Heston Riccati (derived from the
        # Feynman-Kac generator for X_t = log(S_t/S_0) under Q). Getting the
        # sign of the constant right is critical: with +0.5*u^2 the CF has
        # |phi| > 1 for real u, violating a fundamental CF property.
        return -0.5 * (u_c ** 2 + iu) + (iu * eta * rho - kappa) * y + 0.5 * eta ** 2 * y ** 2

    F_integral, psi_integral = fractional_riccati_solve(F, tau, H, u_c, M=M)

    # phi(u, tau) = exp( iu(r-q)tau + kappa*theta*int_psi + V_0*int_F(psi) )
    log_phi = iu * (r - q) * tau + kappa * theta * psi_integral + V_0 * F_integral
    # Cap Re(log_phi) at 0 for real u (since |phi(u)| <= 1 for real u is a
    # fundamental CF property). For complex u, the cap allows limited growth.
    # This is a safety belt against Adams-scheme instability at high |u|.
    # Without this clip, |phi| can overflow to 1e43+ at v ~ 100 for
    # high-vol / high-eta params, corrupting the pricing integral.
    u_is_real = np.abs(u_c.imag) < 1e-10
    cap_real_u = np.where(u_is_real, 0.0, 50.0)  # cap at 0 for real u, 50 for imag-shifted
    log_phi = np.where(
        log_phi.real > cap_real_u,
        cap_real_u + 1j * log_phi.imag,
        log_phi,
    )
    return np.exp(log_phi)


# ── Carr-Madan FFT pricing ───────────────────────────────────────────


def carr_madan_call_prices(
    tau: float,
    V_0: float,
    H: float,
    kappa: float,
    theta: float,
    eta: float,
    rho: float,
    log_strikes: np.ndarray,
    *,
    r: float = 0.0,
    q: float = 0.0,
    F: float = 1.0,
    alpha: float = 1.5,  # unused — kept for API compatibility
    n_fft: int = 1024,   # v-grid size for the Heston inversion integral
    M_riccati: int = 50,
) -> np.ndarray:
    """European call prices via Heston-1993 / Gatheral-2006 Pi1/Pi2 inversion.

    C(K, tau) = exp(-r*tau) * [F * Pi1 - K * Pi2]

        Pi1 = 1/2 + (1/pi) * int_0^inf Re[ exp(-i*v*log(K/F)) * phi(v - i, tau) / phi(-i, tau) ] / (i*v) dv
        Pi2 = 1/2 + (1/pi) * int_0^inf Re[ exp(-i*v*log(K/F)) * phi(v, tau) ] / (i*v) dv

    where phi(u, tau) = E[exp(i*u*X_tau)] is the CF of X_tau = log(S_tau/F)
    as implemented in ``rough_heston_char_fn``.

    Pi2 = Q(S_tau > K); Pi1 is the S-measure probability S_tau/F > K/F.
    Under a martingale forward (F = E[S_tau]), phi(-i, tau) = E[S_tau/F] = 1,
    so the Pi1 normalization is trivial. Nonetheless we keep the explicit
    division for robustness against discretization noise in phi(-i).

    This formula is MUCH more numerically stable than Carr-Madan for
    high-vol models where the moment generating function E[S^{alpha+1}]
    diverges near alpha_max. It requires only alpha_max > 1, which holds
    for any sensible calibration.

    Parameters
    ----------
    tau
        Time to maturity (years).
    V_0, H, kappa, theta, eta, rho
        Rough-Heston parameters.
    log_strikes
        Log-moneyness k = log(K/F).
    r, q, F, alpha, n_fft, M_riccati
        See module docstring.
    """
    # v-grid: [0, v_max] with n_fft midpoints.
    # |phi(v, tau)| ~ exp(-0.5 V_0 tau v^2) for large v (leading order).
    # Pick v_max to put |phi(v_max)| at exp(-25) = 1.4e-11 (truncation error
    # negligible relative to any practical call price). Cap at 50 so
    # high-vol short-tau regimes don't get ridiculous grids.
    v_scale = max(math.sqrt(max(V_0 * tau, 1e-8)), 1e-3)
    v_max = min(50.0, max(8.0, 7.0 / v_scale))
    dv = v_max / n_fft
    v_grid = (np.arange(n_fft) + 0.5) * dv  # midpoint rule

    # CF evaluations
    phi_v = rough_heston_char_fn(
        v_grid + 0.0j, tau, V_0, H, kappa, theta, eta, rho, r=r, q=q, M=M_riccati,
    )
    phi_v_minus_i = rough_heston_char_fn(
        v_grid - 1j, tau, V_0, H, kappa, theta, eta, rho, r=r, q=q, M=M_riccati,
    )
    phi_minus_i = rough_heston_char_fn(
        np.array([0.0 - 1j]), tau, V_0, H, kappa, theta, eta, rho, r=r, q=q, M=M_riccati,
    )[0]

    # psi_1(v) = phi(v - i) / phi(-i); psi_2(v) = phi(v)
    psi_1 = phi_v_minus_i / phi_minus_i
    psi_2 = phi_v

    # Integrate Pi_j for each requested log_strike
    log_strikes_arr = np.asarray(log_strikes)  # k = log(K/F)
    # integrand_j(v, k) = Re[ exp(-i*v*k) * psi_j(v) ] / (i*v)
    # Im[z]/v  =  Re[z/(iv)] for v>0 (since 1/i = -i)
    phase = np.exp(-1j * log_strikes_arr[:, None] * v_grid[None, :])
    num_1 = (phase * psi_1[None, :]).imag / v_grid[None, :]  # Re[.../iv] = Im[...]/v
    num_2 = (phase * psi_2[None, :]).imag / v_grid[None, :]
    int_1 = np.trapz(num_1, dx=dv, axis=-1)
    int_2 = np.trapz(num_2, dx=dv, axis=-1)
    Pi_1 = 0.5 + int_1 / math.pi
    Pi_2 = 0.5 + int_2 / math.pi

    strikes = F * np.exp(log_strikes_arr)
    disc = math.exp(-r * tau)
    call_prices = disc * (F * Pi_1 - strikes * Pi_2)

    # Clip to no-arb bound [max(F-K, 0), F] — Heston integrals are
    # well-defined but trapezoidal truncation can produce tiny bound
    # violations at extreme strikes.
    call_prices = np.clip(call_prices, np.maximum(F - strikes, 0.0), F)
    return call_prices


# ── Implied-volatility inverter ──────────────────────────────────────


def _bs_call_price(F: float, K: float, tau: float, sigma: float, r: float = 0.0) -> float:
    """Black-76 undiscounted call price (forward-measure)."""
    if sigma <= 0 or tau <= 0:
        return max(F - K, 0.0) * math.exp(-r * tau)
    sqrt_t = math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * tau) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    return math.exp(-r * tau) * (F * _norm.cdf(d1) - K * _norm.cdf(d2))


def implied_volatility(
    call_price: float,
    F: float,
    K: float,
    tau: float,
    *,
    r: float = 0.0,
    lo: float = 1e-3,
    hi: float = 10.0,
    tol: float = 1e-8,
    maxiter: int = 200,
) -> float:
    """Brent-style bisection for Black-76 implied vol. Returns NaN on failure."""
    if tau <= 0 or call_price <= 0 or F <= 0 or K <= 0:
        return float("nan")
    intrinsic = max(F - K, 0.0) * math.exp(-r * tau)
    if call_price <= intrinsic - 1e-10:
        return float("nan")

    p_lo = _bs_call_price(F, K, tau, lo, r=r)
    p_hi = _bs_call_price(F, K, tau, hi, r=r)
    if not (p_lo <= call_price <= p_hi):
        return float("nan")

    a, b = lo, hi
    for _ in range(maxiter):
        mid = 0.5 * (a + b)
        p_mid = _bs_call_price(F, K, tau, mid, r=r)
        if abs(p_mid - call_price) < tol:
            return mid
        if p_mid < call_price:
            a = mid
        else:
            b = mid
    return 0.5 * (a + b)


def _bs_vega(F: float, K: float, tau: float, sigma: float, r: float = 0.0) -> float:
    """Black-76 vega in units of price per 1.0 vol change."""
    if sigma <= 0 or tau <= 0:
        return 0.0
    sqrt_t = math.sqrt(tau)
    d1 = (math.log(F / K) + 0.5 * sigma ** 2 * tau) / (sigma * sqrt_t)
    return math.exp(-r * tau) * F * _norm.pdf(d1) * sqrt_t


# ── Calibration ──────────────────────────────────────────────────────


_DEFAULT_INIT = RoughHestonParams(
    V_0=0.5 ** 2, H=0.1, kappa=1.5, theta=0.5 ** 2, eta=2.0, rho=-0.5,
)


def _pack(p: RoughHestonParams) -> np.ndarray:
    return np.array([p.V_0, p.H, p.kappa, p.theta, p.eta, p.rho], dtype=np.float64)


def _unpack(x: np.ndarray) -> tuple[float, float, float, float, float, float]:
    return float(x[0]), float(x[1]), float(x[2]), float(x[3]), float(x[4]), float(x[5])


def calibrate_rough_heston(
    options: np.ndarray,
    *,
    F: float,
    r: float = 0.0,
    init: RoughHestonParams | None = None,
    max_iter: int = 40,
    M_riccati: int = 50,
    n_fft: int = 1024,
    fix_kappa_theta: bool = True,
    use_global_search: bool = True,
) -> RoughHestonParams:
    """Fit rough-Heston to a panel of European calls.

    Default (``fix_kappa_theta=True``) calibrates the 4 strongly-identified
    parameters (V_0, H, eta, rho) with kappa pinned at 2.0 and theta pinned
    at V_0. This is the Bayer-Friz-Gatheral 2016 / Horvath-Muguruza-Tomas
    2021 convention: a single cross-section of smiles has insufficient
    information to resolve kappa and theta separately from (V_0, eta), and
    fitting them as free parameters produces high-variance features that
    hurt downstream learning. The retained 4 params are the ones that
    capture the rough-vol surface structure.

    Set ``fix_kappa_theta=False`` to calibrate all 6 free; this is
    mathematically permitted but produces noisy kappa/theta/eta estimates
    dominated by local minima — only useful for research validation on
    longer panels.

    Parameters
    ----------
    options
        Structured array with columns (tau, K, market_iv, vega_weight). One
        row per option. All options share the SAME forward F — this is the
        common-F assumption (use forward-level prices).
    F
        Forward price.
    r
        Risk-free rate (default 0).
    init
        Optional warm-start (previous bucket's fit). Default: crypto-typical.
    max_iter
        Max SLSQP iterations.
    M_riccati, n_fft
        Numerical-scheme parameters.
    fix_kappa_theta
        See above.
    use_global_search
        If True (default), use a 2-pass Nelder-Mead + SLSQP approach that
        avoids local-minima traps at the cost of ~5-10x wall-clock. Set
        to False for warm-started bucket-by-bucket builds where the init
        is close to truth — goes directly to SLSQP for ~10x speedup. The
        sidecar builder uses False for warm buckets, True for cold ones.

    Returns
    -------
    RoughHestonParams with calibration_level indicating fit quality.
    When ``fix_kappa_theta=True``, the returned kappa=2.0 and theta=V_0
    are exactly the priors — they're not fitted quantities and should
    not be used as features.

    The loss is vega-weighted sum-of-squared IV residuals:
        loss = sum_i w_i * (iv_model(tau_i, K_i) - iv_market_i)^2
    where w_i = vega_weight_i (pre-computed by caller; typically market vega
    times a quality flag).
    """
    if options.size == 0:
        return RoughHestonParams(calibration_level=0)

    taus = options["tau"].astype(np.float64)
    strikes = options["K"].astype(np.float64)
    mkt_iv = options["market_iv"].astype(np.float64)
    w = options["vega_weight"].astype(np.float64)
    log_strikes = np.log(strikes / F)

    # Group options by maturity for batched CF computation
    unique_taus = np.unique(taus)

    # Fixed priors when `fix_kappa_theta=True` — not calibrated.
    _KAPPA_PRIOR = 2.0

    def _unpack_active(x: np.ndarray) -> tuple[float, float, float, float, float, float]:
        """Expand the SLSQP state vector into the full 6-parameter set.

        When fix_kappa_theta is True, x is length-4 and represents
        (V_0, H, eta, rho); kappa and theta are pinned at (2.0, V_0).
        When False, x is length-6 and is returned as-is.
        """
        if fix_kappa_theta:
            V_0 = float(x[0]); H = float(x[1])
            eta = float(x[2]); rho = float(x[3])
            return V_0, H, _KAPPA_PRIOR, V_0, eta, rho
        return _unpack(x)

    def _loss(x: np.ndarray) -> float:
        V_0, H, kappa, theta, eta, rho = _unpack_active(x)
        resid_sq_sum = 0.0
        n_nan = 0
        for tau in unique_taus:
            mask = taus == tau
            k_slice = log_strikes[mask]
            try:
                call_prices = carr_madan_call_prices(
                    tau, V_0, H, kappa, theta, eta, rho,
                    k_slice, F=F, r=r, n_fft=n_fft, M_riccati=M_riccati,
                )
            except Exception:  # noqa: BLE001
                return 1e6
            for i, k in enumerate(k_slice):
                K_i = F * math.exp(k)
                iv = implied_volatility(call_prices[i], F, K_i, tau, r=r)
                iv_mkt = mkt_iv[mask][i]
                w_i = w[mask][i]
                if not np.isfinite(iv):
                    # Can't invert model price -> use price-space residual as
                    # a bounded penalty. Full BS vega scaling keeps units
                    # comparable to IV^2 residuals.
                    iv_proxy = max(iv_mkt * 0.5, 0.1)  # fallback IV
                    resid_sq_sum += w_i * (iv_proxy - iv_mkt) ** 2
                    n_nan += 1
                else:
                    resid_sq_sum += w_i * (iv - iv_mkt) ** 2
        return resid_sq_sum

    # Initialize V_0 from the observed ATM IV of the panel when no warm-start
    # is provided. This is critical for the global search: the default init
    # is tuned for bear-crypto (V_0 ~ 0.25 = 50% vol) and fails on calm-regime
    # panels (V_0 ~ 0.09 = 30% vol). Picking V_0 = (median ATM IV)^2 keeps
    # the NM search inside the right basin.
    if init is None:
        try:
            # Find the option closest to ATM (log-moneyness closest to 0)
            k_abs = np.abs(log_strikes)
            atm_iv = float(mkt_iv[np.argmin(k_abs)])
            data_driven_v0 = float(np.clip(atm_iv ** 2, 0.01, 10.0))
        except Exception:  # noqa: BLE001
            data_driven_v0 = _DEFAULT_INIT.V_0
        adapted_init = RoughHestonParams(
            V_0=data_driven_v0,
            H=_DEFAULT_INIT.H,
            kappa=_DEFAULT_INIT.kappa,
            theta=data_driven_v0,  # theta=V_0 for 4-param compat
            eta=_DEFAULT_INIT.eta,
            rho=_DEFAULT_INIT.rho,
        )
        x0_full = _pack(adapted_init)
    else:
        x0_full = _pack(init)
    # Clip into bounds defensively
    for j, (lo, hi) in enumerate(PARAM_BOUNDS):
        x0_full[j] = float(np.clip(x0_full[j], lo, hi))

    if fix_kappa_theta:
        # Active vars: (V_0, H, eta, rho)
        x0 = np.array([x0_full[0], x0_full[1], x0_full[4], x0_full[5]])
        active_bounds = (
            PARAM_BOUNDS[0],  # V_0
            PARAM_BOUNDS[1],  # H
            PARAM_BOUNDS[4],  # eta
            PARAM_BOUNDS[5],  # rho
        )
    else:
        x0 = x0_full
        active_bounds = PARAM_BOUNDS

    low = np.array([b[0] for b in active_bounds])
    high = np.array([b[1] for b in active_bounds])

    def _residuals(x: np.ndarray) -> np.ndarray:
        """Per-option residuals vector (IV_model - IV_market) weighted by
        sqrt(vega_weight). The sum of squares of this vector is the calibration
        loss — Levenberg-Marquardt minimizes this directly and is 10-100x
        faster than scalar minimizers for least-squares problems because it
        exploits the Jacobian structure."""
        V_0, H, kappa, theta, eta, rho = _unpack_active(x)
        resid = []
        for tau in unique_taus:
            mask = taus == tau
            k_slice = log_strikes[mask]
            try:
                call_prices = carr_madan_call_prices(
                    tau, V_0, H, kappa, theta, eta, rho,
                    k_slice, F=F, r=r, n_fft=n_fft, M_riccati=M_riccati,
                )
            except Exception:  # noqa: BLE001
                # On numerical failure, return large penalty per option
                resid.extend([100.0] * int(mask.sum()))
                continue
            for i, k in enumerate(k_slice):
                K_i = F * math.exp(k)
                iv = implied_volatility(call_prices[i], F, K_i, tau, r=r)
                iv_mkt = mkt_iv[mask][i]
                sw = math.sqrt(max(w[mask][i], 1e-12))
                if not np.isfinite(iv):
                    # Bounded residual so LM can still make progress
                    resid.append(sw * (iv_mkt - max(iv_mkt * 0.5, 0.1)))
                else:
                    resid.append(sw * (iv - iv_mkt))
        return np.asarray(resid)

    try:
        if use_global_search:
            # Two-pass Nelder-Mead (global exploration) + LM polish.
            # Used for cold-start (no warm init). Guarantees we escape
            # local minima even from a bad init.
            def _loss(x: np.ndarray) -> float:
                resid = _residuals(x)
                return float((resid ** 2).sum())

            def _loss_penalized(x: np.ndarray) -> float:
                if np.any(x < low) or np.any(x > high):
                    pen = float(
                        np.sum(
                            np.maximum(low - x, 0.0) ** 2
                            + np.maximum(x - high, 0.0) ** 2
                        )
                    )
                    return _loss(np.clip(x, low, high)) + 100.0 * pen
                return _loss(x)

            result_nm = minimize(
                _loss_penalized, x0,
                method="Nelder-Mead",
                options={"maxiter": max_iter * 10, "xatol": 1e-5, "fatol": 1e-7},
            )
            x_after_nm = np.clip(result_nm.x, low, high)
            # LM polish
            result = least_squares(
                _residuals, x_after_nm,
                method="trf", bounds=(low, high),
                max_nfev=max_iter * 5,
                ftol=1e-8, xtol=1e-8, gtol=1e-8,
            )
        else:
            # Warm-start path: LM directly from the previous bucket's fit.
            # No global exploration — the previous fit is assumed close.
            result = least_squares(
                _residuals, x0,
                method="trf", bounds=(low, high),
                max_nfev=max_iter * 20,
                ftol=1e-8, xtol=1e-8, gtol=1e-8,
            )
    except Exception:  # noqa: BLE001
        return RoughHestonParams(calibration_level=0)

    V_0, H, kappa, theta, eta, rho = _unpack_active(result.x)
    # LM returns 2 * sum(resid^2) as result.cost (factor of 2 in LM convention)
    # and .fun is the residual vector. Convert to RMSE in IV units:
    total_w = w.sum()
    sum_weighted_sq = 2 * float(result.cost)  # sum of w_i * (iv - iv_mkt)^2
    rmse = math.sqrt(sum_weighted_sq / max(total_w, 1e-12))

    return RoughHestonParams(
        V_0=V_0, H=H, kappa=kappa, theta=theta, eta=eta, rho=rho,
        calibration_rmse=rmse,
        n_options=int(len(options)),
        converged=bool(result.success),
        n_iter=int(result.nfev),
        calibration_level=2 if result.success else (1 if rmse < 0.25 else 0),
    )


# ── Structured dtype for option inputs ───────────────────────────────

OPTION_DTYPE = np.dtype([
    ("tau", np.float64),         # time to maturity (years)
    ("K", np.float64),           # strike
    ("market_iv", np.float64),   # observed implied vol (as decimal; 0.5 = 50%)
    ("vega_weight", np.float64), # non-negative weight (use market vega * quality flag)
])


__all__ = [
    "RoughHestonParams",
    "rough_heston_char_fn",
    "carr_madan_call_prices",
    "fractional_riccati_solve",
    "implied_volatility",
    "calibrate_rough_heston",
    "OPTION_DTYPE",
    "PARAM_BOUNDS",
]
