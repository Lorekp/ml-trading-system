"""Signal filters: confidence percentile/absolute and 4H regime.

Extracted from experiments/multi_approach_comparison.py (lines 742-829)
and experiments/expanded_portfolio_test.py (lines 672-755).
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from signal_engine.config import ConfidenceFilterConfig, RegimeFilterConfig

logger = logging.getLogger(__name__)


def apply_confidence_filter(
    signals_df: pd.DataFrame,
    config: ConfidenceFilterConfig,
) -> pd.DataFrame:
    """Apply confidence filter to raw signals.

    In *percentile* mode the filter keeps signals whose ``|prob - 0.5|``
    is at or above the *threshold*-th percentile of all confidences.

    In *absolute* mode the filter keeps signals whose probability exceeds
    *threshold* or is below ``1 - threshold``.

    Args:
        signals_df: Raw signals with ``probability`` and ``confidence``
            columns.
        config: Filter mode and threshold.

    Returns:
        Filtered signals DataFrame (index reset).
    """
    if signals_df is None or len(signals_df) == 0:
        return pd.DataFrame()

    df = signals_df.copy()

    if config.mode == "percentile":
        conf = df["confidence"].values
        if len(conf) > 5:
            conf_threshold = np.percentile(conf, config.threshold)
            mask = conf >= conf_threshold
        else:
            mask = np.ones(len(conf), dtype=bool)
        return df[mask].reset_index(drop=True)

    elif config.mode == "absolute":
        # Filter on confidence = |prob - 0.5|.  Threshold is in the same
        # units (e.g., 0.04 means only keep signals where model is at
        # least 0.04 away from the coin-flip probability of 0.5).
        conf = df["confidence"].values
        mask = conf >= config.threshold
        return df[mask].reset_index(drop=True)

    else:
        raise ValueError(f"Unknown filter mode: {config.mode}")


def apply_regime_filter(
    signals_df: pd.DataFrame,
    ohlcv_4h: pd.DataFrame,
    config: RegimeFilterConfig,
) -> pd.DataFrame:
    """Apply 4H regime filter: suppress counter-trend signals.

    Computes a 4H EMA(*ema_span*) regime indicator.  For each signal the
    direction must agree with the regime:

    * 4H close > EMA -> bullish -> allow long, suppress short
    * 4H close < EMA -> bearish -> allow short, suppress long

    The 4H regime is forward-filled to signal timestamps.

    Args:
        signals_df: Signals with ``timestamp`` and ``direction`` columns.
        ohlcv_4h: 4-hour OHLCV DataFrame.
        config: Regime filter configuration (EMA span, etc.).

    Returns:
        Filtered signals DataFrame (index reset).
    """
    if signals_df is None or len(signals_df) == 0:
        return pd.DataFrame()

    df = signals_df.copy()

    # Compute 4H EMA regime.
    # CRITICAL: shift(1) so each bar's regime is only available AFTER the
    # bar closes.  A 4H bar labelled 04:00 covers 04:00-07:45 — its close
    # isn't known until 07:45.  Without the shift, any 15m signal between
    # 04:00-07:45 would see the future close price (look-ahead bias).
    ema = ohlcv_4h["close"].ewm(span=config.ema_span, adjust=False).mean()
    regime_raw = np.where(ohlcv_4h["close"] > ema, 1.0, -1.0)
    regime_4h = pd.Series(regime_raw, index=ohlcv_4h.index, name="regime").shift(1)

    # Align to signal timestamps via forward-fill
    signal_times = pd.to_datetime(df["timestamp"])

    # Handle timezone compatibility
    if regime_4h.index.tz is not None and signal_times.dt.tz is None:
        signal_times = signal_times.dt.tz_localize("UTC")
    elif regime_4h.index.tz is None and signal_times.dt.tz is not None:
        signal_times = signal_times.dt.tz_localize(None)

    # Create a combined index for reindexing
    regime_aligned = regime_4h.reindex(
        regime_4h.index.union(signal_times.unique())
    ).ffill()
    signal_regimes = regime_aligned.reindex(signal_times).values

    # Filter: direction must agree with regime
    # regime=1 (bullish) -> allow long (direction=1), suppress short
    # regime=-1 (bearish) -> allow short (direction=-1), suppress long
    directions = df["direction"].values
    mask = (directions * signal_regimes) > 0  # same sign = agree

    n_before = len(df)
    result = df[mask].reset_index(drop=True)
    logger.info(
        "Regime filter: %d -> %d signals (%.1f%% kept)",
        n_before,
        len(result),
        100.0 * len(result) / max(n_before, 1),
    )
    return result
