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
from typing import Callable, Literal

import numpy as np
# NumPy 2.0 renamed ``np.trapz`` -> ``np.trapezoid``; alias so both names work.
_trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz")
from scipy.optimize import differential_evolution, least_squares, minimize
from scipy.special import erf as _erf
from scipy.special import gamma as _gamma
from scipy.stats import norm as _norm

# Numba is used to JIT-compile the fractional Riccati Adams inner loop
# (dominant cost of the calibrator, 66% of profile time as a pure-Python loop).
# Optional dependency: if missing, the same function runs in pure-Python mode
# at ~5x slower — the sidecar build takes correspondingly longer but the math
# is bit-identical.
try:
    from numba import njit as _njit
    from numba import prange as _prange
    _NUMBA_AVAILABLE = True
except ImportError:  # pragma: no cover
    _NUMBA_AVAILABLE = False

    def _njit(*args, **kwargs):  # type: ignore[no-redef]
        """Fallback no-op decorator when numba is unavailable."""
        if len(args) == 1 and callable(args[0]):
            return args[0]
        return lambda fn: fn

    _prange = range  # type: ignore[assignment]

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
    """Diethelm-Ford-Freed weights for the fractional Adams scheme.

    Formulas (Diethelm-Ford-Freed 2002 eqns 6.4-6.5), with r = k + 1:

        a[r, 0]      = (r-1)^(alpha+1) - (r-1-alpha) * r^alpha,   r >= 2
        a[1, 0]      = 1
        a[r, j]      = (r-j+1)^(alpha+1) - 2(r-j)^(alpha+1) + (r-j-1)^(alpha+1),
                       for 1 <= j <= r-1
        a[r, r]      = 1
        b[r, j]      = (r-j)^alpha - (r-j-1)^alpha,   for 0 <= j <= r-1

    All other entries zero. Normalized by ``1 / Gamma(alpha+2)`` (corrector)
    and ``1 / Gamma(alpha+1)`` (predictor). Returns ``(a, b)`` of shape
    ``(M+1, M+1)``.

    Vectorized: the original Python double-loop cost ~6 s per DE run at
    M=50 (single-threaded Python arithmetic). The vectorized form
    computes the whole matrix in a handful of numpy ops.
    """
    alpha1 = alpha + 1.0
    # Row index r ∈ [0, M] (we fill rows 1..M), col j ∈ [0, M].
    r_idx = np.arange(M + 1, dtype=np.float64)
    j_idx = np.arange(M + 1, dtype=np.float64)
    R = r_idx[:, None]  # (M+1, 1)
    J = j_idx[None, :]  # (1, M+1)
    diff = R - J        # (M+1, M+1)  = r - j

    # Interior corrector (1 <= j <= r-1): uses three consecutive powers.
    # Safe powers: where the mask is False we never look at the value, but
    # negative bases would still raise. Clamp inputs to ≥ 0.
    d_p1 = np.maximum(diff + 1.0, 0.0)
    d_0  = np.maximum(diff,       0.0)
    d_m1 = np.maximum(diff - 1.0, 0.0)
    a_interior = d_p1 ** alpha1 - 2.0 * d_0 ** alpha1 + d_m1 ** alpha1

    a = np.zeros((M + 1, M + 1), dtype=np.float64)
    interior_mask = (J >= 1) & (J <= R - 1)
    a[interior_mask] = a_interior[interior_mask]

    # Boundary column j=0, r >= 2: a[r,0] = (r-1)^alpha1 - (r-1-alpha) * r^alpha
    r_ge2 = r_idx[2:]
    a[2:, 0] = (r_ge2 - 1.0) ** alpha1 - (r_ge2 - 1.0 - alpha) * r_ge2 ** alpha
    a[1, 0] = 1.0

    # Diagonal j=r, r >= 1
    np.fill_diagonal(a[1:, 1:], 1.0)

    # Predictor: b[r, j] = (r-j)^alpha - (r-j-1)^alpha for 0 <= j <= r-1
    b = np.zeros((M + 1, M + 1), dtype=np.float64)
    b_mask = (J <= R - 1)
    b_vals = d_0 ** alpha - d_m1 ** alpha
    b[b_mask] = b_vals[b_mask]

    a *= 1.0 / _gamma(alpha + 2.0)
    b *= 1.0 / _gamma(alpha + 1.0)
    return a, b


@_njit(cache=True, fastmath=False)
def _adams_kloop_jit(
    a: np.ndarray,          # shape (M+1, M+1), real, pre-scaled by h_alpha
    b: np.ndarray,          # shape (M+1, M+1), real, pre-scaled by h_alpha
    c0: np.ndarray,         # shape (n_u,), complex128
    c1: np.ndarray,         # shape (n_u,), complex128
    c2: complex,            # scalar quadratic coefficient
    M: int,
    u_shape: int,
    h: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Numba-JIT'd Adams predictor-corrector (serial, SIMD-vectorized).

    Implements the Diethelm-Ford-Freed fractional Adams scheme for the
    Riccati equation psi = I^{H+1/2}[F(psi)], F(y) = c0 + c1*y + c2*y^2.
    F(y) is INLINED — no Python callback, no closure capture.

    **Vectorization strategy** (Session G.4d): keeps LLVM's complex128 SIMD
    autovec on the u-dimension (each ``range(u_shape)`` loop is
    data-parallel) but **does NOT use ``prange``** — at n_u=2049 and
    M=50, multi-threading the inner u-loop slows things down by 30-50%
    because DE/LM call this function thousands of times in quick
    succession and thread-pool launch/sync overhead dominates the per-
    call work. The outer bucket-level parallelism (ProcessPoolExecutor
    over months in the sidecar builder) already saturates the CPU at a
    coarser granularity without the thread-thrash penalty.

    Fused predictor+corrector+F-update so each inner pass over u does
    ALL the work per (k, i) — reduces memory traffic (each psi / F_psi
    element is written to once per k).

    Mathematically bit-identical to the closure-based path within 1e-12
    relative (regression-guarded by
    ``tests/test_rough_heston.py::test_jit_fractional_riccati_matches_closure_version``).
    """
    psi = np.zeros((M + 1, u_shape), dtype=np.complex128)
    F_psi = np.zeros((M + 1, u_shape), dtype=np.complex128)
    psi_pred = np.empty(u_shape, dtype=np.complex128)
    F_psi_pred = np.empty(u_shape, dtype=np.complex128)

    # psi[0] = 0 -> F(0) = c0
    for i in range(u_shape):
        F_psi[0, i] = c0[i]

    clip_sq = 1.0e16  # = (1e8)^2

    for k in range(M):
        # Predictor sum: j outer, i inner — cache-friendly because F_psi[j, :]
        # is accessed contiguously per j. (Flipping the loop order makes
        # F_psi strided per-fixed-i and is ~2x slower.)
        for i in range(u_shape):
            psi_pred[i] = 0.0 + 0.0j
        for j in range(k + 1):
            bkj = b[k + 1, j]
            for i in range(u_shape):
                psi_pred[i] = psi_pred[i] + bkj * F_psi[j, i]

        # Clip on squared magnitude (avoids sqrt + any() branch).
        for i in range(u_shape):
            re = psi_pred[i].real
            im = psi_pred[i].imag
            mag_sq = re * re + im * im
            if mag_sq > clip_sq:
                scale = math.sqrt(clip_sq / mag_sq)
                psi_pred[i] = complex(re * scale, im * scale)

        # F(psi_pred) inlined.
        for i in range(u_shape):
            y = psi_pred[i]
            F_psi_pred[i] = c0[i] + c1[i] * y + c2 * y * y

        # Corrector: akk*F_psi_pred + sum_j a[k+1, j] * F_psi[j]
        akk = a[k + 1, k + 1]
        for i in range(u_shape):
            psi[k + 1, i] = akk * F_psi_pred[i]
        for j in range(k + 1):
            akj = a[k + 1, j]
            for i in range(u_shape):
                psi[k + 1, i] = psi[k + 1, i] + akj * F_psi[j, i]

        # Clip psi[k+1] and compute F(psi[k+1]) for next iter.
        for i in range(u_shape):
            re = psi[k + 1, i].real
            im = psi[k + 1, i].imag
            mag_sq = re * re + im * im
            if mag_sq > clip_sq:
                scale = math.sqrt(clip_sq / mag_sq)
                psi[k + 1, i] = complex(re * scale, im * scale)
            y = psi[k + 1, i]
            F_psi[k + 1, i] = c0[i] + c1[i] * y + c2 * y * y

    # Trapezoidal integrals.
    F_psi_integral = np.empty(u_shape, dtype=np.complex128)
    psi_integral = np.empty(u_shape, dtype=np.complex128)
    for i in range(u_shape):
        f_acc = 0.5 * F_psi[0, i] + 0.5 * F_psi[M, i]
        p_acc = 0.5 * psi[0, i] + 0.5 * psi[M, i]
        for k in range(1, M):
            f_acc = f_acc + F_psi[k, i]
            p_acc = p_acc + psi[k, i]
        F_psi_integral[i] = h * f_acc
        psi_integral[i] = h * p_acc
    return F_psi_integral, psi_integral


def _fractional_riccati_solve_inlined(
    c0: np.ndarray,
    c1: np.ndarray,
    c2: float,
    tau: float,
    H: float,
    u_shape: int,
    M: int = 50,
) -> tuple[np.ndarray, np.ndarray]:
    """Hot-path wrapper around ``_adams_kloop_jit``.

    Builds the Adams weight matrices in pure Python (one-shot, cheap)
    then dispatches the k-loop to the JIT'd core. Pulls the JIT path when
    numba is available; otherwise falls back to the same code compiled
    as pure Python (still benefits from buffer reuse + F inlining, but
    runs ~5-10x slower than the JIT version).

    Mathematically byte-identical to ``fractional_riccati_solve`` when
    called with the standard Riccati F closure. Numerical drift vs the
    closure version is < 1e-14 relative per
    ``tests/test_rough_heston.py::test_jit_matches_pure_python``.
    """
    alpha = H + 0.5
    h = tau / M
    h_alpha = h ** alpha

    a, b = _adams_weights(alpha, M)
    a = (a * h_alpha).astype(np.float64)
    b = (b * h_alpha).astype(np.float64)

    return _adams_kloop_jit(a, b, c0, c1, complex(c2), M, u_shape, h)


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
        # Broadcast + sum(axis=0) over the history rows. matmul was tried
        # here (cleaner notation) but is ~4x slower for the small (k+1) ×
        # n_u shapes the Adams scheme works at — BLAS overhead dominates
        # when the inner dimension is < 50.
        psi_pred = (b[k + 1, : k + 1, None] * F_psi[: k + 1]).sum(axis=0)
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
    u_c = np.ascontiguousarray(np.asarray(u_vec, dtype=np.complex128))
    iu = 1j * u_c

    # Precompute the Riccati coefficients c0 (constant), c1 (linear in y),
    # c2 (quadratic in y) once per CF call. The Adams scheme's inner loop
    # inlines F(y) = c0 + c1*y + c2*y^2 — this avoids the closure-dispatch
    # overhead that dominated Session G.2's Adams loop profile.
    c0 = -0.5 * (u_c ** 2 + iu)
    c1 = iu * eta * rho - kappa
    c2 = 0.5 * eta ** 2  # scalar — same across all u

    F_integral, psi_integral = _fractional_riccati_solve_inlined(
        np.ascontiguousarray(c0),
        np.ascontiguousarray(c1),
        float(c2),
        tau, H, u_c.shape[0], M=M,
    )

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

    # Pool the three CF evaluations into a single Adams-scheme solve: the
    # fractional Riccati integration dominates run-time (27% of a DE
    # calibration, per profiling). Stacking u-values cuts scheme overhead
    # ~3x while keeping per-u arithmetic identical. Layout:
    #   u_all[0:n_fft]        = v_grid  (phi evaluation for Pi_2)
    #   u_all[n_fft:2*n_fft]  = v_grid - i  (phi shifted for Pi_1)
    #   u_all[-1]             = -i  (phi(-i) normalization)
    u_all = np.empty(2 * n_fft + 1, dtype=np.complex128)
    u_all[:n_fft] = v_grid
    u_all[n_fft:2 * n_fft] = v_grid - 1j
    u_all[-1] = -1j
    phi_all = rough_heston_char_fn(
        u_all, tau, V_0, H, kappa, theta, eta, rho, r=r, q=q, M=M_riccati,
    )
    phi_v = phi_all[:n_fft]
    phi_v_minus_i = phi_all[n_fft:2 * n_fft]
    phi_minus_i = phi_all[-1]

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
    int_1 = _trapz(num_1, dx=dv, axis=-1)
    int_2 = _trapz(num_2, dx=dv, axis=-1)
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


# ── Vectorized BS + IV inverter (Session G) ──────────────────────────
#
# The scalar ``_bs_call_price`` / ``_bs_vega`` / ``implied_volatility`` above
# are kept for isolated callers (tests, diagnostics). The hot path below
# uses pure-numpy batched variants that skip scipy.stats.norm overhead —
# profiling Session G DE showed 1.6 M scalar norm.cdf calls per calibration
# dominating wall-clock at 38 % of total time. The batched paths below
# drop that to one vectorized CDF via the error-function identity
# Phi(x) = 0.5 * (1 + erf(x / sqrt(2))).


_SQRT2 = math.sqrt(2.0)
_INV_SQRT_2PI = 1.0 / math.sqrt(2.0 * math.pi)


def _norm_cdf_vec(x: np.ndarray) -> np.ndarray:
    """Vectorized standard-normal CDF via the erf identity
    Phi(x) = 0.5 * (1 + erf(x / sqrt(2))). Skips scipy.stats.norm.cdf's
    argument-validation overhead (1.6M calls per Session F calibration).
    """
    return 0.5 * (1.0 + _erf(x / _SQRT2))


def _bs_call_price_vec(
    F: float,
    K: np.ndarray,
    tau: np.ndarray,
    sigma: np.ndarray,
    r: float = 0.0,
) -> np.ndarray:
    """Vectorized Black-76 call price. F is scalar; K, tau, sigma are arrays.

    Returns an array of the same shape as the broadcast of ``K``, ``tau``,
    ``sigma``. Out-of-domain entries (tau<=0 or sigma<=0) collapse to the
    intrinsic value.
    """
    K_arr = np.asarray(K, dtype=np.float64)
    tau_arr = np.asarray(tau, dtype=np.float64)
    sig_arr = np.asarray(sigma, dtype=np.float64)
    disc = np.exp(-r * tau_arr)
    intrinsic = np.maximum(F - K_arr, 0.0) * disc
    # Mask valid inputs; compute BS on them, fall back to intrinsic elsewhere
    valid = (tau_arr > 0) & (sig_arr > 0) & (K_arr > 0)
    out = np.where(valid, 0.0, intrinsic)
    if not np.any(valid):
        return out
    sqrt_t = np.where(valid, np.sqrt(np.maximum(tau_arr, 1e-300)), 1.0)
    log_fk = np.where(valid, np.log(np.maximum(F / np.maximum(K_arr, 1e-300), 1e-300)), 0.0)
    d1 = np.where(
        valid,
        (log_fk + 0.5 * sig_arr ** 2 * tau_arr) / np.maximum(sig_arr * sqrt_t, 1e-300),
        0.0,
    )
    d2 = d1 - sig_arr * sqrt_t
    price = disc * (F * _norm_cdf_vec(d1) - K_arr * _norm_cdf_vec(d2))
    return np.where(valid, price, intrinsic)


def _bs_vega_vec(
    F: float,
    K: np.ndarray,
    tau: np.ndarray,
    sigma: np.ndarray,
    r: float = 0.0,
) -> np.ndarray:
    """Vectorized Black-76 vega (price per 1.0 vol change)."""
    K_arr = np.asarray(K, dtype=np.float64)
    tau_arr = np.asarray(tau, dtype=np.float64)
    sig_arr = np.asarray(sigma, dtype=np.float64)
    valid = (tau_arr > 0) & (sig_arr > 0) & (K_arr > 0)
    if not np.any(valid):
        return np.zeros_like(K_arr, dtype=np.float64)
    sqrt_t = np.where(valid, np.sqrt(np.maximum(tau_arr, 1e-300)), 1.0)
    log_fk = np.where(valid, np.log(np.maximum(F / np.maximum(K_arr, 1e-300), 1e-300)), 0.0)
    d1 = np.where(
        valid,
        (log_fk + 0.5 * sig_arr ** 2 * tau_arr) / np.maximum(sig_arr * sqrt_t, 1e-300),
        0.0,
    )
    pdf_d1 = _INV_SQRT_2PI * np.exp(-0.5 * d1 ** 2)
    vega = np.exp(-r * tau_arr) * F * pdf_d1 * sqrt_t
    return np.where(valid, vega, 0.0)


def implied_volatility_vec(
    prices: np.ndarray,
    F: float,
    K: np.ndarray,
    tau: np.ndarray,
    *,
    r: float = 0.0,
    lo: float = 1e-3,
    hi: float = 10.0,
    tol: float = 1e-8,
    maxiter: int = 100,
) -> np.ndarray:
    """Vectorized Brent-bisection Black-76 IV inversion.

    One bisection step is applied to every option in parallel. Returns
    NaN where the input is out-of-domain (negative price, intrinsic-
    violating, tau<=0) — matching the scalar ``implied_volatility``
    failure semantics.
    """
    prices = np.asarray(prices, dtype=np.float64)
    K_arr = np.asarray(K, dtype=np.float64)
    tau_arr = np.asarray(tau, dtype=np.float64)
    intrinsic = np.maximum(F - K_arr, 0.0) * np.exp(-r * tau_arr)
    invalid = (
        (tau_arr <= 0)
        | (prices <= 0)
        | (K_arr <= 0)
        | (prices < intrinsic - 1e-10)
    )
    p_lo = _bs_call_price_vec(F, K_arr, tau_arr, np.full_like(K_arr, lo), r=r)
    p_hi = _bs_call_price_vec(F, K_arr, tau_arr, np.full_like(K_arr, hi), r=r)
    # Prices outside [p_lo, p_hi] are unreachable in BS — mark NaN.
    oob = (prices < p_lo) | (prices > p_hi)
    dead = invalid | oob

    a = np.full_like(K_arr, lo)
    b = np.full_like(K_arr, hi)
    for _ in range(maxiter):
        mid = 0.5 * (a + b)
        p_mid = _bs_call_price_vec(F, K_arr, tau_arr, mid, r=r)
        diff = p_mid - prices
        a = np.where(diff < -tol, mid, a)
        b = np.where(diff > tol, mid, b)
        if np.all(np.abs(diff) < tol):
            break
    iv = 0.5 * (a + b)
    return np.where(dead, np.nan, iv)


# ── Calibration ──────────────────────────────────────────────────────


_DEFAULT_INIT = RoughHestonParams(
    V_0=0.5 ** 2, H=0.1, kappa=1.5, theta=0.5 ** 2, eta=2.0, rho=-0.5,
)

_KAPPA_PRIOR: float = 2.0  # Bayer-Friz-Gatheral 2016 convention.


class _DELossFn:
    """Picklable callable for scipy DE multiprocess workers.

    The per-bucket residual closure captures market data (prices, vegas,
    τ groupings) that cannot be re-derived on each call. A process pool
    worker receives this state once via pickling at pool init, then
    evaluates the scalar loss on each population member locally.

    The sum-of-squared-residuals is the first-order-Taylor approximation
    to the vega-weighted IV-RMSE (identical optimum, O((Δiv)^2) error
    away from it). See the ``calibrate_rough_heston`` docstring.
    """

    __slots__ = (
        "F", "r", "tau_groups", "market_prices", "resid_coef",
        "fallback_sq_sum", "n_fft", "M", "fix_kappa_theta",
    )

    def __init__(
        self,
        F: float,
        r: float,
        tau_groups: list[tuple[float, np.ndarray, np.ndarray]],
        market_prices: np.ndarray,
        resid_coef: np.ndarray,
        fallback_resid: np.ndarray,
        n_fft: int,
        M: int,
        fix_kappa_theta: bool,
    ) -> None:
        self.F = F
        self.r = r
        self.tau_groups = tau_groups
        self.market_prices = market_prices
        self.resid_coef = resid_coef
        # Store only the scalar fallback loss — the full residual vector
        # isn't needed for DE's scalar-loss interface.
        self.fallback_sq_sum = float(np.sum(fallback_resid ** 2))
        self.n_fft = n_fft
        self.M = M
        self.fix_kappa_theta = fix_kappa_theta

    def __call__(self, x: np.ndarray) -> float:
        if self.fix_kappa_theta:
            V_0 = float(x[0]); H = float(x[1])
            eta = float(x[2]); rho = float(x[3])
            kappa, theta = _KAPPA_PRIOR, V_0
        else:
            V_0, H, kappa, theta, eta, rho = (float(xi) for xi in x)

        model_prices = np.empty_like(self.market_prices)
        for tau, idx, k_slice in self.tau_groups:
            try:
                model_prices[idx] = carr_madan_call_prices(
                    tau, V_0, H, kappa, theta, eta, rho,
                    k_slice, F=self.F, r=self.r,
                    n_fft=self.n_fft, M_riccati=self.M,
                )
            except Exception:  # noqa: BLE001
                return self.fallback_sq_sum
        resid = self.resid_coef * (model_prices - self.market_prices)
        return float(np.dot(resid, resid))


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
    global_method: Literal["nm", "de"] = "de",
    de_popsize: int = 15,
    de_maxiter: int = 30,
    de_seed: int = 42,
    de_n_fft: int = 512,
    de_M_riccati: int = 50,
    de_workers: int = 1,
    warm_rescue_threshold: float = 0.10,
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
        If True (default), use a population / multi-start global search
        followed by an LM polish. Set to False for warm-started bucket-by-
        bucket builds where the init is close to truth — goes directly to
        LM for ~10x speedup. The sidecar builder uses False for warm
        buckets, True for cold ones.
    global_method
        Which global optimizer to use when ``use_global_search=True``.

        - ``"de"`` (default, Session G upgrade): ``scipy.optimize.differential_
          evolution`` with a ``popsize * len(bounds)``-member population.
          Population-based: robust against local minima at the cost of ~1-2x
          wall-clock of the old Nelder-Mead path on cold panels, but higher
          success rate on the real-data non-convex loss surface. The Session F
          failure mode (NM trapped at 14-23vp RMSE from cold-start) motivates
          making this the default.
        - ``"nm"`` (Session F legacy): single-pass Nelder-Mead with quadratic
          penalty on bound violations. Faster on "easy" panels where the data-
          driven V_0 init is already in the right basin, but susceptible to
          local minima on short-expiry skew + kurtosis combinations.
    de_popsize, de_maxiter, de_seed
        DE hyperparameters. Total objective evaluations are bounded by
        ``de_popsize * len(bounds) * de_maxiter`` (≈ 1800 for the default
        4-param setup). Seed is fixed so calibration is deterministic.
    de_n_fft, de_M_riccati
        Pricing-grid resolution used during the DE global search. Defaults
        to a coarser ``n_fft=512`` (vs the ``n_fft=1024`` used by the LM
        polish): DE only needs to locate the correct basin of attraction,
        and at ``n_fft=512`` each objective evaluation is ~2.7x faster
        while the discretization noise (~2.7 vol-points peak, << DE's own
        basin-width) is dwarfed by the optimizer's per-generation progress.
        LM polish always runs at the full ``n_fft``/``M_riccati`` supplied
        to the calibrator, so final parameter estimates are unaffected.
    de_workers
        Parallelism for DE's per-generation population evaluation. Default
        1 (serial) preserves single-bucket determinism and avoids oversub-
        scription when the sidecar builder is the outer parallelism layer.
        Pass ``>1`` (e.g. ``os.cpu_count()``) for standalone diagnostics
        where the calibrator is the sole CPU consumer; on a 12-core
        machine this gives ~4× speedup on a 47-option real panel
        (133 s ↔ 576 s). Internally scipy spawns a multiprocessing pool and
        ships the picklable ``_DELossFn`` callable to each worker once
        at pool init; per-call overhead is just pickling a length-``n_params``
        x-vector. Requires ``updating="deferred"`` (scipy contract when
        ``workers != 1``), which changes DE's per-generation update rule
        slightly but does not change convergence semantics meaningfully
        at this population size. Seed stays fixed, so runs at a given
        ``de_workers=k`` remain deterministic.
    warm_rescue_threshold
        If ``use_global_search=False`` (warm-start path) and the LM fit
        produces RMSE > this threshold (default 10 vol-points), re-run
        calibration as a cold DE search. Prevents level-0 emissions when
        warm-start LM diverges from a bad init. The 10vp default is
        calibrated against the observed Session G DE+LM RMSE floor of
        ~2-6vp on real Deribit panels: anything above 10vp is a clear
        divergence signal. Raising to 20-30vp caused the sidecar build's
        warm-start chain to break pathologically on crypto regime
        transitions (warm LM lands at 15-25vp, sidecar flags as bad and
        retriggers cold DE, 14-hour-total build time).

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
    if global_method not in ("nm", "de"):
        raise ValueError(
            f"global_method must be 'nm' or 'de'; got {global_method!r}"
        )

    if options.size == 0:
        return RoughHestonParams(calibration_level=0)

    taus = options["tau"].astype(np.float64)
    strikes = options["K"].astype(np.float64)
    mkt_iv = options["market_iv"].astype(np.float64)
    w = options["vega_weight"].astype(np.float64)
    log_strikes = np.log(strikes / F)

    # Group options by maturity for batched CF computation
    unique_taus = np.unique(taus)

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

    # ── Precompute per-calibration fixed quantities (Session G) ────────
    #
    # The per-option loss can be written as a first-order Taylor expansion
    # around the market IV:
    #
    #   w_i * (iv_model_i - iv_mkt_i)^2
    #     ≈ (w_i / vega_mkt_i^2) * (price_model_i - price_mkt_i)^2
    #     = resid_coef_i^2 * (price_model_i - price_mkt_i)^2
    #
    # where resid_coef_i = sqrt(w_i) / vega_mkt_i. The approximation is
    # exact to O((Δiv)^2) and in practice indistinguishable from a
    # full-inversion IV-RMSE loss at the converged optimum. The huge
    # benefit: we never invert implied-vol from model prices inside the
    # optimizer loop, which was 38% of Session F DE wall-clock. Market
    # prices + vegas are fixed across all param evals, so we cache them
    # once here.
    strikes_arr = np.exp(log_strikes) * F
    market_prices = _bs_call_price_vec(F, strikes_arr, taus, mkt_iv, r=r)
    vega_mkt = _bs_vega_vec(F, strikes_arr, taus, mkt_iv, r=r)
    # Guard against zero-vega (deep OTM / ITM) — floor at 1e-6 to keep the
    # coefficient finite. Such options get negligible weight anyway because
    # w_i = market_vega * quality_taper is tiny when vega is tiny.
    vega_safe = np.maximum(vega_mkt, 1e-6)
    resid_coef = np.sqrt(np.maximum(w, 1e-12)) / vega_safe  # (n_options,)

    # Pre-group option indices by unique tau so _residuals just loops
    # over taus and pricing is strike-vectorized inside carr_madan.
    tau_groups: list[tuple[float, np.ndarray, np.ndarray]] = []
    for tau in unique_taus:
        idx = np.nonzero(taus == tau)[0]
        tau_groups.append((float(tau), idx, log_strikes[idx]))

    # Catastrophic-failure residual: if pricing raises, return per-option
    # residuals equal to the current market price — bounded, finite,
    # and quickly learned-away by LM.
    fallback_resid = resid_coef * market_prices

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

    def _make_residuals(grid_n_fft: int, grid_M: int):
        """Factory for a residual closure parameterized by pricing grid.

        Returns a pure function x → residuals vector. Used to give DE a
        coarser (faster) grid than the LM polish without duplicating the
        per-bucket caches (`market_prices`, `resid_coef`, `tau_groups`).
        """
        def _residuals(x: np.ndarray) -> np.ndarray:
            """Vectorized per-option residuals, first-order equivalent to
            the vega-weighted IV residuals under Black-76.

            For each unique tau we price the strike slice in one FFT-free
            Heston Pi1/Pi2 call, then the full residual vector is a single
            numpy expression
                resid = resid_coef * (price_model - price_market)
            Sum-of-squares equals the IV-space weighted RMSE to O((Δiv)^2).

            On catastrophic pricing failure (e.g. Riccati blow-up at
            extreme params), the bad tau group falls back to a bounded
            residual equal to resid_coef * market_price — LM sees finite
            residuals and steers away.
            """
            V_0, H, kappa, theta, eta, rho = _unpack_active(x)
            model_prices = np.empty_like(market_prices)
            for tau, idx, k_slice in tau_groups:
                try:
                    model_prices[idx] = carr_madan_call_prices(
                        tau, V_0, H, kappa, theta, eta, rho,
                        k_slice, F=F, r=r,
                        n_fft=grid_n_fft, M_riccati=grid_M,
                    )
                except Exception:  # noqa: BLE001
                    return fallback_resid.copy()
            return resid_coef * (model_prices - market_prices)
        return _residuals

    # Two closures: DE uses the coarse grid for speed, LM polish uses the
    # full grid for accuracy. Both share the same market-data caches.
    _residuals = _make_residuals(n_fft, M_riccati)
    _residuals_de = _make_residuals(de_n_fft, de_M_riccati)

    # Scalar loss variants — DE uses the coarse-grid residuals, NM uses
    # the full-grid residuals (NM has fewer evaluations so the speedup
    # is less important than maintaining the Session F baseline on that
    # path, which remains available via ``global_method="nm"``).
    def _loss_de(x: np.ndarray) -> float:
        resid = _residuals_de(x)
        return float((resid ** 2).sum())

    def _loss(x: np.ndarray) -> float:
        resid = _residuals(x)
        return float((resid ** 2).sum())

    def _run_global(x_start: np.ndarray) -> np.ndarray:
        """Run the configured global optimizer; return clipped x at optimum.

        Dispatches on ``global_method``. Both branches return a point inside
        the bounds box; the subsequent LM polish refines further.
        """
        if global_method == "nm":
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
                _loss_penalized, x_start,
                method="Nelder-Mead",
                options={"maxiter": max_iter * 10, "xatol": 1e-5, "fatol": 1e-7},
            )
            return np.clip(result_nm.x, low, high)
        elif global_method == "de":
            # Population-based global search via differential evolution.
            # Population size = de_popsize * len(bounds), so 15 * 4 = 60
            # members by default for 4-param mode. Sobol init gives better
            # space-fill than the scipy default LHS. polish=False because
            # we do our own LM residual-polish afterwards (exploits Jacobian
            # structure, which scalar L-BFGS-B polish can't).
            de_bounds = list(zip(low.tolist(), high.tolist()))
            n_workers = max(1, int(de_workers))
            de_kwargs = dict(
                strategy="best1bin",
                popsize=de_popsize,
                maxiter=de_maxiter,
                tol=0.01,
                mutation=(0.5, 1.0),
                recombination=0.7,
                init="sobol",
                polish=False,
                seed=de_seed,
            )
            if n_workers == 1:
                # Sequential path — preserves Session G.2 behavior exactly
                # for single-bucket warm-start sidecar calibration.
                result_de = differential_evolution(
                    _loss_de, de_bounds,
                    workers=1,
                    updating="immediate",
                    **de_kwargs,
                )
            else:
                # Multiprocess parallel population evaluation. A
                # ``_DELossFn`` instance carries all market-data caches
                # as pickled state to each worker at pool init, so each
                # per-generation call only pays the cost of pickling a
                # length-4 x vector. scipy's own Pool handles lifecycle.
                parallel_loss = _DELossFn(
                    F=F, r=r,
                    tau_groups=tau_groups,
                    market_prices=market_prices,
                    resid_coef=resid_coef,
                    fallback_resid=fallback_resid,
                    n_fft=de_n_fft, M=de_M_riccati,
                    fix_kappa_theta=fix_kappa_theta,
                )
                result_de = differential_evolution(
                    parallel_loss, de_bounds,
                    workers=n_workers,
                    updating="deferred",
                    **de_kwargs,
                )
            return np.clip(result_de.x, low, high)
        else:
            raise ValueError(f"unknown global_method: {global_method!r}")

    try:
        if use_global_search:
            # Global search followed by LM polish. Escape local minima,
            # then refine on the residual Jacobian.
            x_after_global = _run_global(x0)
            result = least_squares(
                _residuals, x_after_global,
                method="trf", bounds=(low, high),
                max_nfev=max_iter * 5,
                ftol=1e-8, xtol=1e-8, gtol=1e-8,
            )
        else:
            # Warm-start path: LM directly from the previous bucket's fit
            # at FULL resolution (n_fft=1024, tol=1e-8). Session G.4a-c
            # originally coarsened this path to n_fft=512 + tol=1e-5 for
            # speed, but that violated the "at-truth LM stays at truth
            # to < 1 vp" semantic (`test_calibration_at_truth_reproduces_
            # truth_4p`). With the Session G.4d Numba-JIT on the Adams
            # loop, full-resolution LM already runs at ~0.7 s/warm —
            # fast enough to not need the discretization compromise.
            # If warm-path RMSE exceeds ``warm_rescue_threshold``, we fall
            # back to cold DE (below) at the same resolution.
            result = least_squares(
                _residuals, x0,
                method="trf", bounds=(low, high),
                max_nfev=max_iter * 20,
                ftol=1e-8, xtol=1e-8, gtol=1e-8,
            )
            total_w_check = w.sum()
            rmse_check = math.sqrt(
                2 * float(result.cost) / max(total_w_check, 1e-12)
            )
            if rmse_check > warm_rescue_threshold or not np.isfinite(rmse_check):
                # Rescue path: discard the bad warm fit, start cold with DE.
                logger.debug(
                    "warm-start LM RMSE %.4f > %.4f threshold — rescuing with cold DE",
                    rmse_check, warm_rescue_threshold,
                )
                x_rescue = _run_global(x0)
                result = least_squares(
                    _residuals, x_rescue,
                    method="trf", bounds=(low, high),
                    max_nfev=max_iter * 5,
                    ftol=1e-8, xtol=1e-8, gtol=1e-8,
                )
    except Exception:  # noqa: BLE001
        return RoughHestonParams(calibration_level=0)

    V_0, H, kappa, theta, eta, rho = _unpack_active(result.x)
    # Compute the final RMSE in true IV space (not the first-order proxy
    # used inside the optimizer loop). Batched IV inversion runs once
    # over all options via Brent bisection — cheap, ~30 iterations on
    # each option in parallel.
    final_model_prices = np.empty_like(market_prices)
    final_ok = True
    for tau, idx, k_slice in tau_groups:
        try:
            final_model_prices[idx] = carr_madan_call_prices(
                tau, V_0, H, kappa, theta, eta, rho,
                k_slice, F=F, r=r, n_fft=n_fft, M_riccati=M_riccati,
            )
        except Exception:  # noqa: BLE001
            final_ok = False
            break
    if final_ok:
        model_iv = implied_volatility_vec(
            final_model_prices, F, strikes_arr, taus, r=r,
        )
        iv_resid = model_iv - mkt_iv
        # Where IV inversion failed (NaN), fall back to the price-proxy
        # residual so RMSE is still finite.
        iv_resid = np.where(
            np.isfinite(iv_resid),
            iv_resid,
            resid_coef * (final_model_prices - market_prices) / np.sqrt(np.maximum(w, 1e-12)),
        )
        total_w = w.sum()
        sum_weighted_sq = float(np.sum(w * iv_resid ** 2))
        rmse = math.sqrt(sum_weighted_sq / max(total_w, 1e-12))
    else:
        # Pricing blew up at final params — emit large RMSE so caller
        # downgrades to calibration_level=0.
        rmse = 1.0

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
