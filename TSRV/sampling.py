"""Resample tick data into time bars, dollar bars, or tick bars.

Supports both bookTicker (quote) data and aggTrades (trade) data.
Dollar bars from aggTrades use real trade volume — no proxy needed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def resample_time(ticks: pd.DataFrame, interval_s: float,
                  price_col: str = "mid") -> pd.Series:
    """Resample prices at fixed time intervals.

    Uses last-tick-in-bucket, forward-fills gaps.
    Returns Series of prices on a regular time grid.
    """
    prices = ticks[price_col]
    freq = pd.Timedelta(seconds=interval_s)
    resampled = prices.resample(freq).last().ffill().dropna()
    return resampled


def resample_tick_bars(ticks: pd.DataFrame, n_ticks: int,
                       price_col: str = "mid") -> pd.Series:
    """Create tick bars: one bar every n_ticks events.

    Returns Series of bar-close prices.
    """
    prices = ticks[price_col].values
    n = len(prices)
    indices = np.arange(n_ticks - 1, n, n_ticks)
    return pd.Series(prices[indices], index=ticks.index[indices])


def resample_dollar_bars(ticks: pd.DataFrame, threshold: float,
                         price_col: str = "price",
                         dollar_vol_col: str = "dollar_volume") -> pd.Series:
    """Create dollar bars from trade data with real volume.

    A bar closes when cumulative dollar volume since last bar >= threshold.
    Uses pre-computed dollar_volume = price * quantity from aggTrades.

    Returns Series of bar-close prices.
    """
    prices = ticks[price_col].values.astype(np.float64)
    dollar_vol = ticks[dollar_vol_col].values.astype(np.float64)

    bar_indices = _accumulate_bars(dollar_vol, threshold)

    if not bar_indices:
        return pd.Series([prices[-1]], index=[ticks.index[-1]])

    idx = np.array(bar_indices)
    return pd.Series(prices[idx], index=ticks.index[idx])


try:
    from numba import njit

    @njit(cache=True)
    def _accumulate_bars_numba(values, threshold):
        """Numba-jitted dollar bar accumulation — ~50x faster than pure Python."""
        n = len(values)
        # Pre-allocate max possible size
        out = np.empty(n, dtype=np.int64)
        count = 0
        cumulative = 0.0
        for i in range(n):
            cumulative += values[i]
            if cumulative >= threshold:
                out[count] = i
                count += 1
                cumulative = 0.0
        return out[:count]

    def _accumulate_bars(values: np.ndarray, threshold: float) -> list[int]:
        """Accumulate values until threshold, emit bar index. Numba-accelerated."""
        result = _accumulate_bars_numba(values.astype(np.float64), threshold)
        return result.tolist()

except ImportError:
    def _accumulate_bars(values: np.ndarray, threshold: float) -> list[int]:
        """Accumulate values until threshold, emit bar index. Pure Python fallback."""
        bar_indices = []
        cumulative = 0.0
        for i in range(len(values)):
            cumulative += values[i]
            if cumulative >= threshold:
                bar_indices.append(i)
                cumulative = 0.0
        return bar_indices
