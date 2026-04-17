"""
Derivatives Features — Funding Rate, Open Interest, Premium.

These are the UNCORRELATED features that OHLCV-only models miss.
Funding rate and OI come from the derivatives market structure,
not from price patterns. This orthogonality is what makes stacking work.

Data source: Hyperliquid S3 asset_ctxs (1-minute resolution, May 2023+).

Reference: Ackerer et al. (2025) proves funding creates mean-reversion.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def funding_features(
    funding_rate: pd.Series,
    window_short: int = 50,
    window_long: int = 200,
) -> pd.DataFrame:
    """
    Compute funding rate features.

    Funding rate is the cost of holding a perpetual position.
    Extreme positive = longs paying shorts (crowded longs, bearish signal).
    Extreme negative = shorts paying longs (crowded shorts, bullish signal).

    Ackerer (2025): f_t = E[x_{t+theta}] where theta ~ Geometric(kappa).
    Extreme funding → faster mean-reversion.
    """
    features = pd.DataFrame(index=funding_rate.index)

    # Raw funding (already a feature — it's the cost of being long)
    features["funding"] = funding_rate

    # Z-score: how extreme is current funding vs recent history
    mean = funding_rate.rolling(window_short).mean()
    std = funding_rate.rolling(window_short).std()
    features["funding_zscore"] = ((funding_rate - mean) / std.replace(0, np.nan))

    # Long-term z-score (slower regime detection)
    mean_long = funding_rate.rolling(window_long).mean()
    std_long = funding_rate.rolling(window_long).std()
    features["funding_zscore_long"] = ((funding_rate - mean_long) / std_long.replace(0, np.nan))

    # Rate of change: is funding accelerating?
    features["funding_change"] = funding_rate.diff(5)

    # Cumulative funding (cost of holding for N bars)
    features["funding_cumulative"] = funding_rate.rolling(50).sum()

    return features


def oi_features(
    open_interest: pd.Series,
    close: pd.Series,
    window: int = 50,
) -> pd.DataFrame:
    """
    Compute open interest features.

    OI rising + price rising = new money entering (continuation).
    OI rising + price falling = new shorts entering (bearish).
    OI falling + price rising = short covering (weak rally).
    OI falling + price falling = long liquidation (capitulation).
    """
    features = pd.DataFrame(index=open_interest.index)

    # OI rate of change (normalized)
    oi_pct = open_interest.pct_change(5)
    features["oi_change_5"] = oi_pct

    # OI z-score
    oi_mean = open_interest.rolling(window).mean()
    oi_std = open_interest.rolling(window).std()
    features["oi_zscore"] = (open_interest - oi_mean) / oi_std.replace(0, np.nan)

    # OI-price divergence (the key signal)
    price_dir = np.sign(close.pct_change(5))
    oi_dir = np.sign(oi_pct)
    features["oi_price_divergence"] = (price_dir != oi_dir).astype(float)

    # OI-price interaction: signed product
    features["oi_price_interaction"] = oi_pct * close.pct_change(5)

    # OI acceleration (second derivative)
    features["oi_acceleration"] = oi_pct.diff(5)

    return features


def premium_features(
    mark_price: pd.Series,
    oracle_price: pd.Series,
    window: int = 50,
) -> pd.DataFrame:
    """
    Compute basis/premium features.

    Premium = (mark - oracle) / oracle.
    Positive premium = futures trading above spot (longs dominant).
    Negative premium = futures below spot (shorts dominant).

    Ackerer (2025): premium mean-reverts proportional to funding intensity.
    """
    features = pd.DataFrame(index=mark_price.index)

    basis = (mark_price - oracle_price) / oracle_price.replace(0, np.nan)
    features["premium"] = basis

    # Premium z-score
    mean = basis.rolling(window).mean()
    std = basis.rolling(window).std()
    features["premium_zscore"] = (basis - mean) / std.replace(0, np.nan)

    # Premium direction (persistent positive or negative)
    features["premium_sign_streak"] = (
        np.sign(basis)
        .groupby((np.sign(basis) != np.sign(basis).shift()).cumsum())
        .cumcount()
        + 1
    ) * np.sign(basis)

    return features


def impact_features(
    impact_bid: pd.Series,
    impact_ask: pd.Series,
    mid_price: pd.Series,
) -> pd.DataFrame:
    """
    Compute bid-ask impact features from HL asset_ctxs.

    Impact prices show where a standard-sized trade would fill.
    Wide impact spread = thin liquidity = volatile.
    Skewed impact = order book imbalance.
    """
    features = pd.DataFrame(index=mid_price.index)

    # Impact spread (normalized)
    spread = (impact_ask - impact_bid) / mid_price.replace(0, np.nan)
    features["impact_spread"] = spread

    # Impact skew: (mid - bid_impact) vs (ask_impact - mid)
    # Positive skew = more selling pressure (bid side thinner)
    bid_depth = mid_price - impact_bid
    ask_depth = impact_ask - mid_price
    total_depth = bid_depth + ask_depth
    features["impact_skew"] = (
        (ask_depth - bid_depth) / total_depth.replace(0, np.nan)
    )

    return features


def load_asset_ctxs(
    filepath: str,
    coin: str = "BTC",
) -> pd.DataFrame:
    """
    Load and parse a single HL asset_ctxs LZ4 file.

    Returns DataFrame with all fields for the specified coin,
    indexed by timestamp.
    """
    import lz4.frame
    import csv
    import io

    with open(filepath, "rb") as f:
        data = lz4.frame.decompress(f.read()).decode("utf-8")

    reader = csv.DictReader(io.StringIO(data))
    rows = [r for r in reader if r["coin"] == coin]

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time")

    # Convert numeric columns
    numeric_cols = [
        "funding", "open_interest", "prev_day_px", "day_ntl_vlm",
        "premium", "oracle_px", "mark_px", "mid_px",
        "impact_bid_px", "impact_ask_px",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df
