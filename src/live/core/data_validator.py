"""OHLCV data validation — catch corrupt data before it reaches the model.

Validates:
- Required columns present
- OHLC relationships (high >= low, etc.)
- Non-negative volume
- No NaN/inf in price columns
- Monotonic timestamps
- Staleness detection (latest bar too old)
- Gap detection (missing bars)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Interval string → expected bar duration
_INTERVAL_MINUTES = {
    "1m": 1, "5m": 5, "15m": 15, "15min": 15, "30m": 30,
    "1h": 60, "3h": 180, "4h": 240,
    "1d": 1440, "1D": 1440, "24h": 1440,
}

REQUIRED_COLS = {"open", "high", "low", "close", "volume"}


def validate_ohlcv(
    df: pd.DataFrame,
    symbol: str,
    interval: str,
    max_staleness_factor: float = 3.0,
    max_gap_factor: float = 3.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Validate and clean OHLCV data.

    Returns:
        (cleaned_df, warnings) — cleaned_df may be shorter than input.
        Empty DataFrame + warnings if data is unusable.
    """
    warnings = []

    if df is None or len(df) == 0:
        return pd.DataFrame(), [f"{symbol}/{interval}: empty DataFrame"]

    # 1. Required columns
    missing_cols = REQUIRED_COLS - set(df.columns)
    if missing_cols:
        return pd.DataFrame(), [f"{symbol}/{interval}: missing columns {missing_cols}"]

    original_len = len(df)

    # 2. OHLC relationships: high >= max(open, close, low), low <= min(open, close, high)
    bad_hl = df["high"] < df["low"]
    bad_ho = df["high"] < df["open"]
    bad_hc = df["high"] < df["close"]
    bad_lo = df["low"] > df["open"]
    bad_lc = df["low"] > df["close"]
    bad_bars = bad_hl | bad_ho | bad_hc | bad_lo | bad_lc
    if bad_bars.any():
        n_bad = int(bad_bars.sum())
        warnings.append(f"{symbol}/{interval}: {n_bad} bars with invalid OHLC relationships")
        df = df[~bad_bars]

    # 3. Non-negative volume
    neg_vol = df["volume"] < 0
    if neg_vol.any():
        warnings.append(f"{symbol}/{interval}: {int(neg_vol.sum())} bars with negative volume")
        df = df[~neg_vol]

    # 4. NaN/inf in price columns
    price_cols = ["open", "high", "low", "close"]
    has_nan = df[price_cols].isna().any(axis=1)
    has_inf = np.isinf(df[price_cols].values).any(axis=1)
    bad_values = has_nan | has_inf
    if bad_values.any():
        warnings.append(f"{symbol}/{interval}: {int(bad_values.sum())} bars with NaN/inf prices")
        df = df[~bad_values]

    # 5. Zero prices (likely corrupt)
    zero_price = (df[price_cols] <= 0).any(axis=1)
    if zero_price.any():
        warnings.append(f"{symbol}/{interval}: {int(zero_price.sum())} bars with zero/negative prices")
        df = df[~zero_price]

    if len(df) == 0:
        return pd.DataFrame(), warnings + [f"{symbol}/{interval}: all bars removed by validation"]

    # 6. Timestamp ordering
    if not df.index.is_monotonic_increasing:
        warnings.append(f"{symbol}/{interval}: non-monotonic timestamps, re-sorting")
        df = df.sort_index()
        # Remove duplicates
        df = df[~df.index.duplicated(keep='last')]

    # 7. Staleness check
    interval_minutes = _INTERVAL_MINUTES.get(interval)
    if interval_minutes:
        now = datetime.now(tz=timezone.utc)
        latest_bar = df.index[-1]
        if hasattr(latest_bar, 'tz') and latest_bar.tz is None:
            latest_bar = latest_bar.tz_localize('UTC')
        expected_age = timedelta(minutes=interval_minutes * max_staleness_factor)
        actual_age = now - latest_bar
        if actual_age > expected_age:
            warnings.append(
                f"{symbol}/{interval}: stale data — latest bar is {actual_age} old "
                f"(expected < {expected_age})")

    # 8. Gap detection
    if interval_minutes and len(df) > 1:
        expected_gap = timedelta(minutes=interval_minutes)
        actual_gaps = df.index.to_series().diff().dropna()
        large_gaps = actual_gaps[actual_gaps > expected_gap * max_gap_factor]
        if len(large_gaps) > 0:
            warnings.append(
                f"{symbol}/{interval}: {len(large_gaps)} gaps > {max_gap_factor}x expected "
                f"(largest: {large_gaps.max()})")

    # Report if we dropped bars
    n_dropped = original_len - len(df)
    if n_dropped > 0:
        logger.debug("%s/%s: dropped %d/%d bars", symbol, interval, n_dropped, original_len)

    return df, warnings


def validate_ticker(
    ticker: dict[str, float],
    symbol: str,
) -> tuple[dict[str, float], list[str]]:
    """Validate ticker data. Returns (ticker, warnings)."""
    warnings = []

    bid = ticker.get("bid", 0)
    ask = ticker.get("ask", 0)
    last = ticker.get("last_price", 0)

    if last <= 0:
        warnings.append(f"{symbol}: ticker last_price is {last}")
    if bid <= 0:
        warnings.append(f"{symbol}: ticker bid is {bid}")
    if ask <= 0:
        warnings.append(f"{symbol}: ticker ask is {ask}")
    if bid > 0 and ask > 0 and bid > ask:
        warnings.append(f"{symbol}: crossed spread bid={bid} > ask={ask}")
    if bid > 0 and ask > 0:
        spread_bps = (ask - bid) / ((bid + ask) / 2) * 10000
        if spread_bps > 50:
            warnings.append(f"{symbol}: wide spread {spread_bps:.0f} bps")

    return ticker, warnings
