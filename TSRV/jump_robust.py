"""Jump-robust volatility estimators.

Bipower Variation (Barndorff-Nielsen & Shephard 2004, 2006)
MedRV (Andersen, Dobrev, Schaumburg 2012)
Jump/continuous decomposition
"""

from __future__ import annotations

import numpy as np


def bipower_variation(log_prices: np.ndarray) -> float:
    """Bipower Variation (BNS 2004).

    BPV = mu_1^{-2} * (n/(n-1)) * sum_{i=2}^{n} |r_{i-1}| * |r_i|

    Converges to integrated variance even in the presence of finite-activity
    jumps, because the product of adjacent absolute returns is insensitive to
    rare large jumps (a jump contaminates one return, not two).

    Args:
        log_prices: Array of log(price) values.

    Returns:
        Bipower variation estimate.
    """
    n = len(log_prices) - 1
    if n < 3:
        return 0.0
    returns = np.diff(log_prices)
    mu1_inv_sq = np.pi / 2  # (sqrt(2/pi))^{-2}
    bpv = mu1_inv_sq * np.sum(np.abs(returns[1:]) * np.abs(returns[:-1]))
    return float(bpv * n / (n - 1))


def median_rv(log_prices: np.ndarray) -> float:
    """Median Realized Variance (Andersen, Dobrev, Schaumburg 2012).

    MedRV = c * (M/(M-2)) * sum_{i=2}^{M-1} median(|r_{i-1}|, |r_i|, |r_{i+1}|)^2

    More robust than BPV to clustered jumps: a single jump contaminates
    two adjacent absolute returns in BPV, but the median discards it.

    Args:
        log_prices: Array of log(price) values.

    Returns:
        MedRV estimate.
    """
    n = len(log_prices) - 1
    if n < 4:
        return 0.0
    abs_returns = np.abs(np.diff(log_prices))
    M = len(abs_returns)
    c = np.pi / (6.0 - 4.0 * np.sqrt(3.0) + np.pi)  # ~2.610

    # Vectorized median of triplets
    triplets = np.stack([abs_returns[:-2], abs_returns[1:-1], abs_returns[2:]], axis=1)
    medians = np.median(triplets, axis=1)
    med_sum = float(np.sum(medians ** 2))

    return c * (M / (M - 2)) * med_sum


def realized_semivariance(log_prices: np.ndarray) -> tuple[float, float]:
    """Signed realized semivariance (Barndorff-Nielsen, Kinnebrock, Shephard 2010).

    RS^- = sum(r_i^2 * 1(r_i < 0))   — downside variance
    RS^+ = sum(r_i^2 * 1(r_i >= 0))  — upside variance

    RS^- + RS^+ = RV (total realized variance).
    """
    n = len(log_prices) - 1
    if n < 2:
        return 0.0, 0.0
    returns = np.diff(log_prices)
    rs_down = float(np.sum(returns[returns < 0] ** 2))
    rs_up = float(np.sum(returns[returns >= 0] ** 2))
    return rs_down, rs_up


def decompose_jump_continuous(rv: float, bpv: float) -> tuple[float, float]:
    """Decompose total variance into continuous + jump components.

    continuous = min(rv, bpv)
    jump = max(rv - bpv, 0)

    This is the standard max/min decomposition (Andersen, Bollerslev, Diebold 2007).
    """
    continuous = min(rv, bpv)
    jump = max(rv - bpv, 0.0)
    return continuous, jump


def directional_jump_fractions(
    rv_slow: float, bpv: float, rs_down: float, rs_up: float,
) -> tuple[float, float]:
    """Split jump variance into down-jump and up-jump fractions.

    Under symmetric continuous diffusion, BPV/2 is the continuous variance
    contribution on each side. The excess semivariance beyond BPV/2 is
    the directional jump component.

    In crypto: high down_jump_fraction -> liquidation cascades (sustained vol).
               high up_jump_fraction -> sell-wall absorption (quick mean-revert).
    """
    half_bpv = bpv / 2.0
    down_jump = max(rs_down - half_bpv, 0.0)
    up_jump = max(rs_up - half_bpv, 0.0)
    denom = max(rv_slow, 1e-20)
    return down_jump / denom, up_jump / denom


def subgrid_rv_slow(log_prices: np.ndarray, sub_bar_indices: np.ndarray,
                    K: int) -> tuple[float, float]:
    """K-subgrid-averaged RV(slow) (Zhang, Mykland, Ait-Sahalia 2005).

    Averages RV computed on K offset sub-grids of the sub-bar index array
    to reduce variance of the slow-scale estimator.

    Args:
        log_prices: Full log-price array for the window.
        sub_bar_indices: Indices of dollar sub-bar closes.
        K: Number of sub-grids.

    Returns:
        (rv_slow_avg, n_bar_avg) tuple.
    """
    M = len(sub_bar_indices)

    # Fall back to single grid when insufficient sub-bars
    if M < max(2 * K, 4):
        if M < 2:
            return 0.0, 0.0
        sub_ret = np.diff(log_prices[sub_bar_indices])
        return float(np.sum(sub_ret ** 2)), float(M - 1)

    rv_sum = 0.0
    valid_grids = 0
    for k in range(K):
        grid_idx = sub_bar_indices[k::K]
        if len(grid_idx) < 2:
            continue
        sub_ret = np.diff(log_prices[grid_idx])
        rv_sum += float(np.sum(sub_ret ** 2))
        valid_grids += 1

    if valid_grids == 0:
        return 0.0, 0.0

    rv_slow_avg = rv_sum / valid_grids
    n_bar_avg = (M - K + 1) / K
    return rv_slow_avg, n_bar_avg
