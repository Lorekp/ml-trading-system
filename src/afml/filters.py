"""
Event Sampling Filters — AFML Chapter 2.

The CUSUM filter detects structural changes (mean shifts) in a time series.
Rather than labeling every bar (which creates massive overlap), we label
only when the CUSUM filter fires — at moments when something meaningful
has changed in the price process.

This dramatically reduces label redundancy and improves ML training quality.

Reference: López de Prado (2018), Ch 2, Snippet 2.4.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def cusum_filter(
    raw_series: pd.Series,
    threshold: float | pd.Series,
) -> pd.DatetimeIndex:
    """
    CUSUM filter for event-based sampling.

    Fires an event when the cumulative sum of positive or negative changes
    exceeds a threshold. After firing, the accumulator resets to zero.

    This detects mean shifts — moments when the price process has moved
    significantly from a recent baseline.

    Args:
        raw_series: Price series (typically close prices).
        threshold: Fixed threshold (float) or dynamic per-bar threshold (pd.Series).
                   For dynamic: pass rolling volatility for adaptive event rate.
                   Typical: daily_vol.mean() for fixed, daily_vol for dynamic.

    Returns:
        DatetimeIndex of event timestamps where the filter fired.
    """
    events = []
    s_pos = 0.0
    s_neg = 0.0
    diff = raw_series.diff().dropna()

    for t in diff.index:
        # Get threshold at time t (scalar or from Series)
        if isinstance(threshold, pd.Series):
            h = threshold.get(t, None)
            if h is None or np.isnan(h):
                continue
        else:
            h = float(threshold)

        s_pos = max(0.0, s_pos + diff.loc[t])
        s_neg = min(0.0, s_neg + diff.loc[t])

        if s_neg < -h:
            s_neg = 0.0
            events.append(t)
        elif s_pos > h:
            s_pos = 0.0
            events.append(t)

    return pd.DatetimeIndex(events)


def z_score_filter(
    raw_series: pd.Series,
    mean_window: int = 50,
    std_window: int = 50,
    z_threshold: float = 3.0,
) -> pd.DatetimeIndex:
    """
    Z-score based event filter.

    Fires when the z-score of the current value relative to a rolling
    window exceeds a threshold. Alternative to CUSUM for detecting
    statistical anomalies.

    Args:
        raw_series: Price or return series.
        mean_window: Rolling window for mean computation.
        std_window: Rolling window for std computation.
        z_threshold: Number of standard deviations to trigger an event.

    Returns:
        DatetimeIndex of event timestamps.
    """
    rolling_mean = raw_series.rolling(mean_window).mean()
    rolling_std = raw_series.rolling(std_window).std()

    z_scores = (raw_series - rolling_mean) / rolling_std
    z_scores = z_scores.dropna()

    events = z_scores[z_scores.abs() > z_threshold].index
    return pd.DatetimeIndex(events)


def get_daily_vol(
    close: pd.Series,
    span: int = 50,
) -> pd.Series:
    """
    Compute daily volatility using exponentially weighted moving standard deviation.

    This is used as the dynamic threshold for triple-barrier labeling
    and CUSUM filtering.

    For 4H bars: span=50 ≈ 8 calendar days of lookback.
    For 1H bars: span=100 ≈ 4 calendar days.
    For 1m bars: span=1440 ≈ 1 day.

    Args:
        close: Close price series.
        span: EWM span parameter (number of bars).

    Returns:
        Series of rolling volatility estimates (same index as close).
    """
    # Compute returns
    returns = close.pct_change().dropna()
    # Exponentially weighted standard deviation
    vol = returns.ewm(span=span).std()
    return vol
