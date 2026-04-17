"""Realized variance estimators: naive RV, noise variance, TSRV."""

from __future__ import annotations

import numpy as np
import pandas as pd


def realized_variance(prices: pd.Series | np.ndarray,
                      method: str = "log_return") -> float:
    """Sum of squared returns (realized variance over the sample period).

    Args:
        prices: Price series (at least 2 observations).
        method: "log_return" (default) or "simple_return".
    """
    p = np.asarray(prices, dtype=np.float64)
    p = p[np.isfinite(p) & (p > 0)]
    if len(p) < 2:
        return 0.0

    if method == "log_return":
        r = np.diff(np.log(p))
    else:
        r = np.diff(p) / p[:-1]

    return float(np.sum(r ** 2))


def noise_variance(prices: pd.Series | np.ndarray) -> float:
    """Estimate microstructure noise variance.

    sigma_eps^2 = RV_all / (2n)  (Hansen & Lunde)
    """
    p = np.asarray(prices, dtype=np.float64)
    p = p[np.isfinite(p) & (p > 0)]
    n = len(p) - 1
    if n < 1:
        return 0.0
    rv = realized_variance(p)
    return rv / (2 * n)


def tsrv(prices: pd.Series | np.ndarray, K: int | None = None) -> float:
    """Two-Scale Realized Variance (Zhang, Mykland, Ait-Sahalia 2005).

    TSRV = RV_slow - (n_bar_slow / n_bar_all) * RV_all

    Corrects for microstructure noise by combining a slow-scale (subsampled)
    RV with the all-ticks RV as a noise proxy.

    Args:
        prices: Price series.
        K: Number of subgrids. None → optimal K = ceil(n^{2/3}).

    Returns:
        Bias-corrected realized variance. Can be negative if noise dominates
        (rare for reasonable K).
    """
    p = np.asarray(prices, dtype=np.float64)
    p = p[np.isfinite(p) & (p > 0)]
    n = len(p) - 1
    if n < 2:
        return 0.0

    if K is None:
        K = max(1, int(np.ceil(n ** (2 / 3))))
    K = min(K, n)  # can't have more grids than observations

    # All-ticks RV
    log_p = np.log(p)
    returns_all = np.diff(log_p)
    rv_all = float(np.sum(returns_all ** 2))

    # Subsampled RV: average over K grids
    rv_slow = 0.0
    for k in range(K):
        sub_log = log_p[k::K]
        if len(sub_log) < 2:
            continue
        sub_ret = np.diff(sub_log)
        rv_slow += float(np.sum(sub_ret ** 2))
    rv_slow /= K

    n_bar_slow = (n - K + 1) / K
    n_bar_all = n

    return rv_slow - (n_bar_slow / n_bar_all) * rv_all


def annualize_rv(rv: float, period_days: float = 1.0) -> float:
    """Convert realized variance over `period_days` to annualized volatility."""
    if rv <= 0 or period_days <= 0:
        return 0.0
    return float(np.sqrt(rv * 365.25 / period_days))
