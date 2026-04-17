"""
Technical Indicators — RSI, MACD, Bollinger %B, ATR, Stochastic RSI.

These are NOT used as standalone signals (momentum fails at 47-50%).
They are FEATURES fed into XGBoost alongside derivatives data (funding, OI)
for stacked direction prediction.

All computations are backward-looking only. No look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    RSI < 30 = oversold, RSI > 70 = overbought.
    As an ML feature, captures momentum exhaustion.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)

    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()

    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f"rsi_{period}")


def stochastic_rsi(
    close: pd.Series,
    rsi_period: int = 14,
    stoch_period: int = 14,
) -> pd.Series:
    """
    Stochastic RSI — RSI of RSI. More sensitive to extremes.

    Values 0-100. < 20 = deeply oversold, > 80 = deeply overbought.
    """
    rsi_val = rsi(close, rsi_period)
    rsi_min = rsi_val.rolling(stoch_period).min()
    rsi_max = rsi_val.rolling(stoch_period).max()
    rsi_range = rsi_max - rsi_min
    stoch = ((rsi_val - rsi_min) / rsi_range.replace(0, np.nan) * 100)
    return stoch.rename("stoch_rsi")


def macd(
    close: pd.Series,
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    MACD line, signal line, and histogram.

    Histogram > 0 = bullish momentum increasing.
    Histogram < 0 = bearish momentum increasing.
    Histogram crossing zero = momentum shift.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line

    result = pd.DataFrame(index=close.index)
    result["macd_hist"] = histogram / close * 100  # Normalize by price
    return result


def bollinger_pct_b(
    close: pd.Series,
    window: int = 20,
    n_std: float = 2.0,
) -> pd.Series:
    """
    Bollinger %B — position within Bollinger Bands.

    0 = at lower band, 1 = at upper band.
    < 0 = below lower band (extreme oversold).
    > 1 = above upper band (extreme overbought).
    """
    sma = close.rolling(window).mean()
    std = close.rolling(window).std()
    upper = sma + n_std * std
    lower = sma - n_std * std
    band_width = upper - lower
    pct_b = (close - lower) / band_width.replace(0, np.nan)
    return pct_b.rename("bb_pct_b")


def atr(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Average True Range — volatility measure using H/L/C.

    Different from Parkinson (which only uses H/L).
    ATR accounts for gaps (close-to-high, close-to-low).
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return (tr.ewm(span=period, adjust=False).mean() / close).rename(f"atr_{period}")


def swing_levels(
    high: pd.Series,
    low: pd.Series,
    lookback: int = 20,
    n_levels: int = 3,
) -> pd.DataFrame:
    """
    Detect swing high/low levels (support/resistance).

    A swing high: bar where high is the highest in lookback window.
    A swing low: bar where low is the lowest in lookback window.

    Returns distance to nearest swing high and low as features.
    BACKWARD-LOOKING ONLY — no look-ahead bias.
    """
    result = pd.DataFrame(index=high.index)

    # Rolling max high and min low
    rolling_high = high.rolling(lookback).max()
    rolling_low = low.rolling(lookback).min()

    # Is this bar a swing high/low?
    is_swing_high = (high == rolling_high)
    is_swing_low = (low == rolling_low)

    # Track most recent swing levels
    swing_highs = high.where(is_swing_high).ffill()
    swing_lows = low.where(is_swing_low).ffill()

    # Distance to nearest swing (normalized by current price)
    close = (high + low) / 2  # mid price approximation
    result["dist_swing_high"] = (swing_highs - close) / close
    result["dist_swing_low"] = (close - swing_lows) / close

    # Combined: distance to nearest level (min of both)
    result["dist_nearest_level"] = pd.concat(
        [result["dist_swing_high"].abs(), result["dist_swing_low"].abs()],
        axis=1
    ).min(axis=1)

    return result


def multi_tf_momentum(
    close_fine: pd.Series,
    close_coarse: pd.Series,
    fine_lookback: int = 14,
    coarse_lookback: int = 14,
) -> pd.Series:
    """
    Multi-timeframe momentum confirmation.

    +1 when both fine and coarse timeframes agree on direction.
    -1 when they disagree.
     0 when either is flat.

    Alignment = stronger trend signal.
    Disagreement = potential reversal.
    """
    fine_mom = np.sign(close_fine.pct_change(fine_lookback))
    coarse_aligned = close_coarse.reindex(close_fine.index, method="ffill")
    coarse_mom = np.sign(coarse_aligned.pct_change(coarse_lookback))

    # +1 = both agree, -1 = disagree, 0 = either flat
    agreement = fine_mom * coarse_mom
    return agreement.shift(1).fillna(0).rename("mtf_momentum")  # Lag 1 bar
