"""Derivatives features — funding rate and open interest.

Funding rate: 8-hourly from Binance, forward-filled to 15m grid.
Open interest: 15m from Binance Futures API (if available).

Forward-fill uses ONLY past values (method='ffill'). No future data.
Reuses existing feature functions from src/afml/derivatives.py.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from vol_model.config import VolModelConfig

logger = logging.getLogger(__name__)


def _load_funding_15m(
    funding_dir: str,
    symbol: str,
    target_index: pd.DatetimeIndex,
    window_short: int,
    window_long: int,
) -> pd.DataFrame:
    """Load 8h funding rate and forward-fill to 15m grid.

    Forward-fill only — each 15m bar sees the last known funding rate.
    No future data is used.
    """
    funding_file = Path(funding_dir) / f"{symbol}_funding.parquet"
    if not funding_file.exists():
        logger.warning("Funding rate file not found: %s", funding_file)
        return pd.DataFrame(index=target_index)

    raw = pd.read_parquet(funding_file)
    funding_rate = raw["fundingRate"]

    # Forward-fill to 15m grid — only uses past values
    funding_15m = funding_rate.reindex(
        funding_rate.index.union(target_index)
    ).sort_index().ffill().reindex(target_index)

    # Compute features using existing src/afml/derivatives.py logic
    features = pd.DataFrame(index=target_index)
    features["funding"] = funding_15m

    # Z-score (short window) — when std=0 (constant funding), z-score=0
    mean_s = funding_15m.rolling(window_short, min_periods=1).mean()
    std_s = funding_15m.rolling(window_short, min_periods=1).std().clip(lower=1e-15)
    features["funding_zscore"] = (funding_15m - mean_s) / std_s

    # Z-score (long window)
    mean_l = funding_15m.rolling(window_long, min_periods=1).mean()
    std_l = funding_15m.rolling(window_long, min_periods=1).std().clip(lower=1e-15)
    features["funding_zscore_long"] = (funding_15m - mean_l) / std_l

    # Rate of change
    features["funding_change"] = funding_15m.diff(5)

    # Cumulative funding (carry cost over last 50 bars = 12.5 hours)
    features["funding_cumulative"] = funding_15m.rolling(50, min_periods=1).sum()

    return features


def _load_oi_15m(
    oi_dir: str,
    symbol: str,
    target_index: pd.DatetimeIndex,
    close_prices: pd.Series | None = None,
    hl_cache_dir: str = "data/hl-asset-ctxs/cache_15m",
) -> pd.DataFrame:
    """Load 15m OI data and compute features.

    Tries sources in order:
    1. Binance OI (data/oi/{SYMBOL}_oi_15m.parquet)
    2. HL asset_ctxs cache (data/hl-asset-ctxs/cache_15m/{COIN}.parquet)

    Returns empty DataFrame if no OI data is available.
    """
    oi = None

    # Source 1: Binance OI
    oi_file = Path(oi_dir) / f"{symbol}_oi_15m.parquet"
    if oi_file.exists():
        raw = pd.read_parquet(oi_file)
        if "sumOpenInterestValue" in raw.columns:
            oi = pd.to_numeric(raw["sumOpenInterestValue"], errors="coerce")
        elif "sumOpenInterest" in raw.columns:
            oi = pd.to_numeric(raw["sumOpenInterest"], errors="coerce")
        if oi is not None:
            logger.info("  OI source: Binance (%d rows)", len(oi))

    # Source 2: HL asset_ctxs (1-min, needs resampling)
    if oi is None:
        coin = symbol.replace("USDT", "")
        hl_file = Path(hl_cache_dir) / f"{coin}.parquet"
        if hl_file.exists():
            hl = pd.read_parquet(hl_file)
            if "open_interest" in hl.columns and "mark_px" in hl.columns:
                # OI in coin units -> USD: open_interest * mark_px
                oi_usd = hl["open_interest"] * hl["mark_px"]
                # Resample 1-min to 15-min (last value per window)
                oi = oi_usd.resample("15min").last().dropna()
                logger.info("  OI source: HL asset_ctxs (%d rows after 15m resample)", len(oi))

    if oi is None:
        logger.info("  OI data not found — skipping OI features")
        return pd.DataFrame(index=target_index)

    # Align to target index (forward-fill)
    oi_aligned = oi.reindex(
        oi.index.union(target_index)
    ).sort_index().ffill().reindex(target_index)

    features = pd.DataFrame(index=target_index)

    # OI rate of change
    oi_pct = oi_aligned.pct_change(5)
    features["oi_change_5"] = oi_pct

    # OI z-score
    oi_mean = oi_aligned.rolling(50, min_periods=1).mean()
    oi_std = oi_aligned.rolling(50, min_periods=1).std()
    features["oi_zscore"] = (oi_aligned - oi_mean) / oi_std.replace(0, np.nan)

    # OI-price divergence (only if close prices available)
    if close_prices is not None:
        price_dir = np.sign(close_prices.pct_change(5))
        oi_dir = np.sign(oi_pct)
        features["oi_price_divergence"] = (price_dir != oi_dir).astype(float)
        features["oi_price_interaction"] = oi_pct * close_prices.pct_change(5)

    # OI acceleration (second derivative)
    features["oi_acceleration"] = oi_pct.diff(5)

    return features


def compute_derivatives_features(
    config: VolModelConfig,
    close_prices: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute all derivatives features (funding + OI).

    Args:
        config: model configuration
        close_prices: optional 15m close prices for OI-price features.
            If None, OI-price interaction features will be NaN.

    Returns DataFrame indexed by window start time (UTC).
    """
    # Build target 15m index from TSRV output
    tsrv = pd.read_parquet(config.get_tsrv_path())
    target_index = tsrv.index

    # Funding rate features (always available)
    funding = _load_funding_15m(
        config.funding_dir, config.asset, target_index,
        config.funding_window_short, config.funding_window_long)

    # OI features (optional — gracefully degrades if not available)
    oi = _load_oi_15m(config.oi_dir, config.asset, target_index, close_prices)

    # Merge
    features = pd.concat([funding, oi], axis=1)

    logger.info("  Derivatives features: %d rows, %d columns (%s)",
                len(features), len(features.columns), list(features.columns))

    return features
