"""
Shared utilities for vol-based strategies.

All functions operate on Series/DataFrames and return aligned Series.
All signals are inherently lagged (use only past data at each bar).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def vol_regime(
    predicted_vol: pd.Series,
    window: int = 100,
    low_pct: float = 0.30,
    high_pct: float = 0.70,
) -> pd.Series:
    """
    Classify vol regime: low (trending), high (choppy), neutral.

    Returns Series with values: -1 (low vol), 0 (neutral), +1 (high vol).
    """
    rolling_rank = predicted_vol.rolling(window).rank(pct=True)
    regime = pd.Series(0, index=predicted_vol.index, dtype=float)
    regime[rolling_rank < low_pct] = -1.0  # Low vol regime
    regime[rolling_rank > high_pct] = 1.0   # High vol regime
    return regime


def bb_squeeze(
    close: pd.Series,
    window: int = 20,
    n_std: float = 2.0,
    squeeze_pct: float = 0.10,
    lookback: int = 50,
) -> pd.DataFrame:
    """
    Detect Bollinger Band squeeze and breakout.

    Returns DataFrame with:
        bb_upper, bb_lower: band levels
        bb_width: normalized width (width / SMA)
        is_squeeze: True when width is in bottom squeeze_pct percentile
        breakout_up: True when close > bb_upper (and was squeezed recently)
        breakout_down: True when close < bb_lower (and was squeezed recently)
    """
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    width = (upper - lower) / sma

    # Squeeze: width in bottom percentile over lookback
    width_threshold = width.rolling(lookback).quantile(squeeze_pct)
    is_squeeze = width <= width_threshold

    # Recent squeeze (was squeezed in last 5 bars)
    recent_squeeze = is_squeeze.rolling(5).max().fillna(0) > 0

    result = pd.DataFrame(index=close.index)
    result["bb_upper"] = upper
    result["bb_lower"] = lower
    result["bb_width"] = width
    result["is_squeeze"] = is_squeeze
    result["breakout_up"] = (close > upper) & recent_squeeze
    result["breakout_down"] = (close < lower) & recent_squeeze
    return result


def vol_zscore_signal(
    predicted_vol: pd.Series,
    window: int = 50,
) -> pd.Series:
    """
    Z-score of predicted vol vs recent history.

    High z-score (>2) = vol model predicts unusually high vol = big move coming.
    """
    mean = predicted_vol.rolling(window).mean()
    std = predicted_vol.rolling(window).std()
    return ((predicted_vol - mean) / std.replace(0, np.nan)).fillna(0)


def first_significant_bar(
    returns: pd.Series,
    threshold: pd.Series,
    signal_active: pd.Series,
    lookforward: int = 5,
) -> pd.DataFrame:
    """
    After a signal fires, find the first bar where |return| > threshold.

    Returns DataFrame with:
        trigger_bar: True on the first significant bar after signal
        direction: +1 or -1 based on the significant bar's return
    """
    result = pd.DataFrame(index=returns.index)
    result["trigger_bar"] = False
    result["direction"] = 0.0

    active = False
    bars_since_signal = 0

    for i in range(len(returns)):
        if signal_active.iloc[i] and not active:
            active = True
            bars_since_signal = 0

        if active:
            bars_since_signal += 1
            if bars_since_signal > lookforward:
                active = False
                continue

            thresh = threshold.iloc[i] if i < len(threshold) else 0.01
            ret = returns.iloc[i]

            if abs(ret) > thresh:
                result.iloc[i, result.columns.get_loc("trigger_bar")] = True
                result.iloc[i, result.columns.get_loc("direction")] = np.sign(ret)
                active = False

    return result


def momentum_signal(close: pd.Series, lookback: int = 14) -> pd.Series:
    """Simple momentum: sign of trailing N-bar return. Lagged 1 bar."""
    return np.sign(close.pct_change(lookback)).shift(1).fillna(0)
