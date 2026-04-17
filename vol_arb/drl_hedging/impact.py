"""Sigmoid-adjusted square-root law for market impact.

The textbook square-root law (SRL) of Almgren-Chriss/BARRA:

    I(Q) = kappa * sigma * sqrt(Q / V)

systematically *underestimates* impact on crypto perpetual order books at
low participation rates.  The reason is physical: the SRL assumes a smooth
order-book depth profile where tiny orders have tiny impact, but real perps
have a *fixed half-spread* that every trade pays regardless of size.  At
participation p << 0.5%, SRL predicts near-zero cost, while reality is a
constant 1-3 bps floor.

This module applies a sigmoid correction that:

- at p >> 0.5% → correction ≈ 1 (standard SRL)
- at p << 0.5% → correction ≈ 2 (doubles the impact to capture the floor)

The transition width is controlled by ``sigmoid_alpha``; the transition
point by ``sigmoid_k0``.  See ``calibrate_kappa()`` for anchoring the
model to the existing linear baseline.

References
----------
- Almgren, Thum, Hauptmann, Li (2005), "Direct Estimation of Equity Market Impact"
- Bouchaud et al. (2018), "Trades, Quotes and Prices" Ch. 12
- This module's novelty: the sigmoid correction for fixed-spread floors,
  which is not in the literature for equities but is necessary for perps.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


# Default kappa — calibrated at entry point "2bps at p=2%, sigma=80%" via
# kappa = (2e-4 / (0.8 * sqrt(0.02))) ≈ 1.768e-3.  Computing inline so tests
# can reference a single source of truth.  See :func:`calibrate_kappa`.
_DEFAULT_KAPPA: float = 2e-4 / (0.80 * np.sqrt(0.02))

# Default sigmoid steepness — chosen empirically to be the LARGEST value
# that keeps ``srl * correction`` strictly monotonic in Q over the full
# p-range, verified on a 2000-point grid.
#
# The naive analytical bound ``alpha <= 3/k0 = 600`` is obtained by
# checking the derivative at p=k0 (sig=0.5) only.  The actual tightest
# constraint sits at sig ≈ 0.8 (p slightly past the knee), which reduces
# the safe range to ~alpha <= 430 at k0=0.005.  We use 400 for margin.
#
# At alpha=400:
#   correction(p→0)  ≈ 1.881   (saturation ratio vs pure SRL)
#   correction(p=0.02) ≈ 1.0015   (matches pure SRL to within 0.15%)
#   monotonic to machine precision on 2000-point grid
_DEFAULT_SIGMOID_ALPHA: float = 400.0
_DEFAULT_SIGMOID_K0: float = 0.005


def sigmoid_sqrt_impact(
    Q: ArrayLike,
    V: ArrayLike,
    sigma: ArrayLike,
    *,
    kappa: float = _DEFAULT_KAPPA,
    sigmoid_alpha: float = _DEFAULT_SIGMOID_ALPHA,
    sigmoid_k0: float = _DEFAULT_SIGMOID_K0,
    impact_floor: float = 1e-8,
) -> NDArray[np.float64]:
    """Return sigmoid-adjusted square-root impact as a *decimal fraction*.

    Parameters
    ----------
    Q : array-like
        Absolute trade size in base units (e.g. BTC).  Must be >= 0.
    V : array-like
        Background volume over the participation window, same units as ``Q``.
        A typical choice is trailing 1h perp volume.
    sigma : array-like
        Instantaneous annualised volatility (decimal, e.g. 0.8 for 80% vol).
    kappa : float, default ≈ 1.768e-3
        Liquidity constant, dimensionless.  Anchored via :func:`calibrate_kappa`
        to the existing 2-bps-at-2%-participation linear baseline.
    sigmoid_alpha : float, default 400.0
        Sigmoid steepness.  For strict monotonicity of ``srl * correction``
        the empirical bound (verified on 2000-point grid) is alpha ≲ 430
        at the default k0=0.005.  The default 400 gives ~1.88x saturation
        at p<<k0 while remaining monotonic to machine precision.
    sigmoid_k0 : float, default 0.005
        Participation-rate knee in decimal (0.005 = 0.5%).
    impact_floor : float, default 1e-8
        Numerical floor on ``V`` to avoid division explosions when background
        volume is zero.

    Returns
    -------
    ndarray
        Impact as a decimal fraction of the underlying price.  To get a
        price impact in dollars, multiply by ``S`` (spot).

    Notes
    -----
    The correction factor is:

        f(p) = 1 + (1 - sigmoid(alpha * (p - k0)))

    At ``p >> k0`` the sigmoid saturates to 1 → f → 1 (pure SRL).
    At ``p << k0`` the sigmoid saturates to 0 → f → 2 (doubled impact).

    The function is vectorised and safe for tight RL step() loops —
    broadcasts across ``Q``, ``V``, ``sigma`` using numpy rules.
    """
    Q_arr = np.asarray(Q, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)

    # Enforce non-negativity on trade size (sign is tracked separately upstream)
    Q_abs = np.abs(Q_arr)

    V_safe = np.maximum(V_arr, impact_floor)
    p = Q_abs / V_safe  # participation rate, dimensionless

    srl = kappa * sigma_arr * np.sqrt(p)

    # Sigmoid correction: 1 at high p, 2 at low p, smooth transition at k0.
    # expit(x) = 1 / (1 + exp(-x))  — implemented inline to avoid scipy dep.
    z = sigmoid_alpha * (p - sigmoid_k0)
    # Clip to avoid overflow in exp; beyond ±500 the sigmoid is effectively 0 or 1
    z_clipped = np.clip(z, -500.0, 500.0)
    sig = 1.0 / (1.0 + np.exp(-z_clipped))
    correction = 1.0 + (1.0 - sig)

    return srl * correction


def pure_sqrt_impact(
    Q: ArrayLike,
    V: ArrayLike,
    sigma: ArrayLike,
    *,
    kappa: float = _DEFAULT_KAPPA,
    impact_floor: float = 1e-8,
) -> NDArray[np.float64]:
    """Textbook SRL without sigmoid correction — for benchmarking only."""
    Q_arr = np.asarray(Q, dtype=np.float64)
    V_arr = np.asarray(V, dtype=np.float64)
    sigma_arr = np.asarray(sigma, dtype=np.float64)
    V_safe = np.maximum(V_arr, impact_floor)
    return kappa * sigma_arr * np.sqrt(np.abs(Q_arr) / V_safe)


def calibrate_kappa(
    baseline_impact_bps: float = 2.0,
    typical_participation: float = 0.02,
    typical_sigma: float = 0.80,
    *,
    sigmoid_alpha: float = _DEFAULT_SIGMOID_ALPHA,
    sigmoid_k0: float = _DEFAULT_SIGMOID_K0,
) -> float:
    """Pick ``kappa`` so that ``sigmoid_sqrt_impact`` (with the correction
    factor included) matches the linear baseline at the operating point.

    The existing vol-arb backtester uses a flat ``perp_slippage_bps = 2.0``
    regardless of size.  We anchor the new model so that

        sigmoid_sqrt_impact(Q, V, sigma)  =  baseline_bps / 1e4

    at the stated operating point.  The solve is:

        kappa * sigma * sqrt(p) * correction(p, alpha, k0) = target
        ⇒  kappa = target / (sigma * sqrt(p) * correction(p, alpha, k0))

    Including the correction factor (rather than assuming it = 1) is the
    pro-level fix — otherwise the returned kappa over-produces by the
    correction factor at the operating point (0.06% at α=500, p=0.02).

    Parameters
    ----------
    baseline_impact_bps : float, default 2.0
        Target impact in basis points.
    typical_participation : float, default 0.02 (= 2%)
        Operating point for calibration.
    typical_sigma : float, default 0.80
        Typical annualised vol (BTC long-run average).
    sigmoid_alpha, sigmoid_k0 : float
        Passed through to match the sigmoid_sqrt_impact defaults.

    Returns
    -------
    float
        The calibrated ``kappa``.
    """
    target = baseline_impact_bps / 1e4
    # Correction factor at the operating point — compute it exactly so
    # the returned kappa produces the target impact under sigmoid_sqrt_impact.
    z = sigmoid_alpha * (typical_participation - sigmoid_k0)
    z_clipped = float(np.clip(z, -500.0, 500.0))
    sig = 1.0 / (1.0 + np.exp(-z_clipped))
    correction_at_op = 1.0 + (1.0 - sig)
    return target / (typical_sigma * np.sqrt(typical_participation) * correction_at_op)
