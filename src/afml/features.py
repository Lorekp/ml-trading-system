"""
Microstructural + Entropy Features — AFML Chapters 18-19.

Features derived from OHLCV data that capture market microstructure
without requiring tick-level data.

Parkinson volatility: more efficient than close-close vol (uses H/L)
Corwin-Schultz spread: estimates bid-ask spread from H/L prices
Lempel-Ziv entropy: measures predictability/complexity of return series

Reference: López de Prado (2018), Ch 18 (Entropy), Ch 19 (Microstructure).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import pandas as pd


# --- Volatility Features (Ch 19) ---

def parkinson_vol(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Parkinson (1980) volatility estimator using high-low range.

    5x more efficient than close-close volatility because it uses
    the full intrabar price range.

    Formula: σ²_HL = (1/4Tln2) * Σ(ln(H/L))²

    Args:
        high: High price series.
        low: Low price series.
        window: Rolling window size.

    Returns:
        Rolling Parkinson volatility estimate.
    """
    log_hl = np.log(high / low)
    log_hl_sq = log_hl ** 2
    return np.sqrt(log_hl_sq.rolling(window).mean() / (4 * np.log(2)))


def corwin_schultz_spread(
    high: pd.Series,
    low: pd.Series,
    window: int = 20,
) -> pd.Series:
    """
    Corwin-Schultz (2012) bid-ask spread estimator from H/L prices.

    Estimates the effective spread without needing order book data.
    Uses the relationship between single-bar and two-bar high-low ranges.

    Args:
        high: High price series.
        low: Low price series.
        window: Rolling window for smoothing beta.

    Returns:
        Estimated spread as fraction of price (e.g., 0.001 = 0.1%).
    """
    # Beta: average of sum of squared log(H/L) for consecutive pairs
    log_hl_sq = np.log(high / low) ** 2
    beta = log_hl_sq.rolling(2).sum().rolling(window).mean()

    # Gamma: squared log of two-bar high-low range
    h2 = high.rolling(2).max()
    l2 = low.rolling(2).min()
    gamma = np.log(h2 / l2) ** 2

    # Alpha
    denom = 3 - 2 * np.sqrt(2)
    alpha = (np.sqrt(2 * beta) - np.sqrt(beta)) / denom - np.sqrt(gamma / denom)
    alpha = alpha.clip(lower=0)  # Alpha can't be negative

    # Spread
    spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha))
    return spread


# --- Entropy Features (Ch 18) ---

def _discretize_returns(returns: np.ndarray, n_bins: int = 5) -> str:
    """Discretize continuous returns into a string of symbols."""
    quantiles = np.percentile(returns[~np.isnan(returns)], np.linspace(0, 100, n_bins + 1))
    quantiles[0] = -np.inf
    quantiles[-1] = np.inf
    indices = np.digitize(returns, quantiles[1:])
    return "".join(str(int(i)) for i in indices)


@lru_cache(maxsize=200000)
def _lempel_ziv_complexity_bytes(msg: bytes) -> float:
    if len(msg) <= 1:
        return 0.0

    i = 0
    dictionary = set()
    complexity = 0

    while i < len(msg):
        for j in range(i + 1, len(msg) + 1):
            substr = msg[i:j]
            if substr not in dictionary:
                dictionary.add(substr)
                complexity += 1
                i = j
                break
        else:
            break

    return complexity / len(msg)


def lempel_ziv_complexity(msg: str) -> float:
    """
    Lempel-Ziv (1976) complexity measure.

    Counts the number of distinct substrings found by the LZ76 algorithm.
    Higher complexity = more random/unpredictable sequence.
    Lower complexity = more structured/predictable.

    Normalized by message length to allow comparison across different lengths.

    Args:
        msg: String of discrete symbols.

    Returns:
        Normalized complexity measure (0 = constant, ~1 = random).
    """
    if len(msg) <= 1:
        return 0.0
    return _lempel_ziv_complexity_bytes(msg.encode("ascii"))


def rolling_entropy(
    close: pd.Series,
    window: int = 50,
    n_bins: int = 5,
) -> pd.Series:
    """
    Rolling Lempel-Ziv entropy of discretized returns.

    Lower entropy → more predictable (mean-reversion may work).
    Higher entropy → more random (harder to predict).

    Args:
        close: Close price series.
        window: Rolling window for entropy computation.
        n_bins: Number of bins for discretizing returns.

    Returns:
        Series of rolling entropy values.
    """
    returns = close.pct_change().dropna()
    result = pd.Series(index=returns.index, dtype=float)
    if len(returns) <= window:
        return result

    values = returns.to_numpy(dtype=np.float64, copy=False)
    n_out = len(values) - window
    out = np.full(n_out, np.nan, dtype=np.float64)
    q = np.linspace(0, 100, n_bins + 1)[1:-1]
    chunk_size = 50000

    for start in range(0, n_out, chunk_size):
        end = min(start + chunk_size, n_out)
        source = values[start : end + window]
        windows = np.lib.stride_tricks.sliding_window_view(source, window_shape=window)[:-1]
        bounds = np.percentile(windows, q, axis=1).T
        digitized = np.sum(windows[:, :, None] > bounds[:, None, :], axis=2).astype(np.uint8)
        for offset, row in enumerate(digitized):
            out[start + offset] = _lempel_ziv_complexity_bytes(row.tobytes())

    result.iloc[window:] = out
    return result


# --- Additional Features ---

def ohlc_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    OHLC interaction features that capture bar structure.

    Body size, upper/lower wicks, and range encode information about
    intrabar dynamics that close-only features miss entirely.
    """
    o, h, l, c = ohlcv["open"], ohlcv["high"], ohlcv["low"], ohlcv["close"]
    features = pd.DataFrame(index=ohlcv.index)

    # Bar range (total movement)
    bar_range = (h - l) / c
    features["bar_range"] = bar_range

    # Body size (open-close relative to range)
    body = (c - o).abs() / (h - l).replace(0, np.nan)
    features["body_ratio"] = body

    # Upper/lower wick ratio (buying/selling pressure)
    features["upper_wick"] = (h - np.maximum(o, c)) / (h - l).replace(0, np.nan)
    features["lower_wick"] = (np.minimum(o, c) - l) / (h - l).replace(0, np.nan)

    # Direction (bullish/bearish bar)
    features["bar_direction"] = np.sign(c - o)

    return features


def volume_features(
    close: pd.Series,
    volume: pd.Series,
    window: int = 20,
) -> pd.DataFrame:
    """
    Volume-based features capturing market conviction and activity.

    Volume carries information about the strength of price moves.
    High volume + directional move = strong conviction (continuation likely).
    High volume + no move = absorption (reversal likely).
    """
    features = pd.DataFrame(index=close.index)

    # Volume-weighted return (conviction-scaled)
    returns = close.pct_change()
    vol_z = (volume - volume.rolling(window).mean()) / volume.rolling(window).std().replace(0, np.nan)
    features["vol_weighted_ret"] = returns * vol_z

    # Volume ratio (current vs rolling average)
    features["vol_ratio"] = volume / volume.rolling(window).mean()

    # Volume trend (is volume increasing or decreasing?)
    features["vol_trend"] = volume.rolling(window).mean().pct_change(5)

    # Price-volume divergence (price up but volume down = weak rally)
    price_dir = np.sign(returns.rolling(5).sum())
    vol_dir = np.sign(volume.diff().rolling(5).sum())
    features["pv_divergence"] = (price_dir != vol_dir).astype(float)

    # On-balance volume (OBV) derivative
    obv = (np.sign(returns) * volume).cumsum()
    features["obv_slope"] = obv.diff(window) / window

    return features


def rolling_volume_zscore(
    volume: pd.Series,
    window: int = 30,
) -> pd.Series:
    """Volume z-score relative to rolling window."""
    mean = volume.rolling(window).mean()
    std = volume.rolling(window).std()
    return (volume - mean) / std.replace(0, np.nan)


def hour_of_day(index: pd.DatetimeIndex) -> pd.Series:
    """Extract hour-of-day as cyclical feature (sin encoding)."""
    hours = index.hour
    return pd.Series(np.sin(2 * np.pi * hours / 24), index=index, name="hour_sin")


def day_of_week(index: pd.DatetimeIndex) -> pd.Series:
    """Extract day-of-week as cyclical feature (sin encoding)."""
    days = index.dayofweek
    return pd.Series(np.sin(2 * np.pi * days / 7), index=index, name="dow_sin")
