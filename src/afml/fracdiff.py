"""
Fractional Differentiation — AFML Chapter 5.

Implements the Fixed-Width Window Fractionally Differentiated (FFD) method.
Standard returns (d=1) erase all memory from prices. Raw prices (d=0) are
non-stationary. FFD with d in (0,1) preserves memory while achieving stationarity.

Key insight: for most financial series, d ≈ 0.3-0.5 achieves stationarity
while retaining 85-99% correlation with the original price series.

Reference: López de Prado (2018), Ch 5, Snippets 5.3-5.4.

IMPORTANT: threshold determines window width. At threshold=1e-5:
    d=0.2 → window=3382 bars (needs ~3400 data points)
    d=0.4 → window=1458 bars (needs ~1500 data points)
    d=0.6 → window=590 bars
    d=0.8 → window=228 bars
For short series (<2000 bars), use threshold=1e-2 or 1e-3.
For our 5000-bar 4H data: threshold=1e-5 works for d≥0.4.
For 1-minute data (500K+ bars): threshold=1e-5 works for any d.
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller


@lru_cache(maxsize=128)
def _cached_weights_ffd(d: float, threshold: float) -> tuple[float, ...]:
    weights = [1.0]
    k = 1
    while True:
        w_next = -weights[-1] / k * (d - k + 1)
        if abs(w_next) < threshold:
            break
        weights.append(w_next)
        k += 1
    return tuple(weights[::-1])


def get_weights_ffd(d: float, threshold: float = 1e-5) -> np.ndarray:
    """
    Compute FFD filter weights for a given fractional order d.

    The weights follow the iterative formula:
        w_0 = 1
        w_k = -w_{k-1} * (d - k + 1) / k

    Weights are truncated when |w_k| < threshold (fixed-width window).

    Args:
        d: Fractional differentiation order. 0 = no change, 1 = standard returns.
        threshold: Minimum absolute weight to include. Smaller = more memory preserved
                   but wider window. Default 1e-5 per AFML recommendation.

    Returns:
        Array of weights in chronological order (oldest first), shape (window_size, 1).
    """
    return np.array(_cached_weights_ffd(float(d), float(threshold)), dtype=np.float64).reshape(-1, 1)


def frac_diff_ffd(
    series: pd.DataFrame,
    d: float,
    threshold: float = 1e-5,
) -> pd.DataFrame:
    """
    Apply Fixed-Width Window Fractional Differentiation to a price series.

    This is the PREFERRED method over expanding-window (fracDiff) because:
    1. Constant memory usage regardless of series length
    2. No information loss at the start of the series (beyond the window)
    3. Suitable for real-time/streaming application

    Args:
        series: DataFrame with price columns (typically log-prices).
        d: Fractional differentiation order.
        threshold: Weight truncation threshold.

    Returns:
        DataFrame of fractionally differentiated values. First `width` rows
        are dropped (insufficient history for the filter window).
    """
    weights = get_weights_ffd(d, threshold)
    width = len(weights) - 1

    result = {}
    for col in series.columns:
        col_data = series[[col]].ffill().dropna()
        if len(col_data) <= width:
            result[col] = pd.Series(dtype=float)
            continue
        values = col_data.iloc[:, 0].to_numpy(dtype=np.float64, copy=False)
        conv = np.convolve(values, weights.ravel()[::-1], mode="valid")
        result[col] = pd.Series(conv, index=col_data.index[width:], dtype=np.float64)

    return pd.concat(result, axis=1)


def find_min_d(
    series: pd.Series,
    d_range: np.ndarray | None = None,
    threshold: float = 1e-5,
    adf_pvalue: float = 0.05,
    adf_maxlag: int = 1,
) -> dict:
    """
    Find the minimum fractional differentiation order d that achieves stationarity.

    Sweeps d from 0 to 1, applies FFD at each d, and runs the Augmented
    Dickey-Fuller test. Returns the minimum d where ADF p-value < adf_pvalue.

    Args:
        series: Price series (NOT log-prices — we take log internally).
        d_range: Array of d values to test. Default: np.linspace(0, 1, 21).
        threshold: FFD weight truncation threshold.
        adf_pvalue: Significance level for stationarity. Default 0.05.
        adf_maxlag: Maximum lag for ADF test. Default 1.

    Returns:
        Dict with keys:
            'd_star': Minimum d achieving stationarity (None if not found)
            'results': DataFrame with d, adf_stat, p_value, correlation for each d
    """
    if d_range is None:
        d_range = np.linspace(0, 1, 21)

    log_prices = np.log(series).to_frame("price")

    records = []
    for d in d_range:
        if d == 0:
            # d=0 means no transformation — just the log-price itself
            ffd = log_prices.dropna()
        else:
            ffd = frac_diff_ffd(log_prices, d, threshold)

        if len(ffd) < 20:
            # Not enough data after windowing
            records.append({"d": d, "adf_stat": np.nan, "p_value": 1.0, "correlation": np.nan})
            continue

        ffd_values = ffd.iloc[:, 0].dropna()

        # ADF test for stationarity
        try:
            adf_result = adfuller(ffd_values, maxlag=adf_maxlag, regression="c", autolag=None)
            adf_stat = adf_result[0]
            p_value = adf_result[1]
        except Exception:
            adf_stat = np.nan
            p_value = 1.0

        # Correlation with original log-prices (memory retention)
        common_idx = log_prices.index.intersection(ffd.index)
        if len(common_idx) > 10:
            corr = np.corrcoef(
                log_prices.loc[common_idx, "price"].values,
                ffd.loc[common_idx].iloc[:, 0].values,
            )[0, 1]
        else:
            corr = np.nan

        records.append({"d": d, "adf_stat": adf_stat, "p_value": p_value, "correlation": corr})

    results_df = pd.DataFrame(records)

    # Find minimum d where p_value < adf_pvalue
    stationary = results_df[results_df["p_value"] < adf_pvalue]
    d_star = float(stationary["d"].min()) if len(stationary) > 0 else None

    return {"d_star": d_star, "results": results_df}
