"""Feature computation for the 14-feature universal signal model.

Extracted from experiments/multi_approach_comparison.py (lines 351-395).
Delegates indicator math to ``afml.technical`` — no duplication.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

from afml.technical import (
    atr,
    bollinger_pct_b,
    macd,
    rsi,
    stochastic_rsi,
    swing_levels,
)
from afml.features import (
    parkinson_vol,
    corwin_schultz_spread,
    rolling_entropy,
)
from afml.fracdiff import frac_diff_ffd

from signal_engine.config import FeatureConfig

logger = logging.getLogger(__name__)


def rolling_sadf(
    close: pd.Series,
    min_window: int = 50,
    step: int = 10,
) -> pd.Series:
    """Rolling SADF approximation using expanding-backward ADF windows."""
    log_p = np.log(close.replace(0, np.nan).to_numpy(dtype=np.float64))
    n = len(log_p)
    sadf = np.full(n, np.nan, dtype=np.float32)
    if n < min_window:
        return pd.Series(sadf, index=close.index, dtype=np.float32)

    for t in range(min_window, n):
        max_stat = -np.inf
        start_lo = max(0, t - min_window * 3)
        start_hi = t - min_window + 1
        for w_start in range(start_lo, start_hi, max(step, 1)):
            window = log_p[w_start : t + 1]
            if len(window) < min_window or not np.isfinite(window).all():
                continue
            try:
                stat = adfuller(window, maxlag=1, regression="c", autolag=None)[0]
            except Exception:
                continue
            if stat > max_stat:
                max_stat = stat
        if max_stat > -np.inf:
            sadf[t] = np.float32(max_stat)
    return pd.Series(sadf, index=close.index, dtype=np.float32)


def compute_features(ohlcv: pd.DataFrame, config: FeatureConfig) -> pd.DataFrame:
    """Compute all 14 universal features from OHLCV data.

    No asset-specific calibration.  No Hyperliquid derivatives.  Pure OHLCV.

    Args:
        ohlcv: DataFrame with columns ``open, high, low, close, volume`` and a
            ``DatetimeIndex``.
        config: Feature configuration (lookback windows, etc.).

    Returns:
        DataFrame with exactly ``len(config.feature_list)`` float32 columns,
        sharing the same index as *ohlcv*.  Early rows will contain NaN due to
        lookback requirements.
    """
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    volume = ohlcv["volume"]

    f = pd.DataFrame(index=ohlcv.index)

    # Returns
    for lb in [1, 3, 7, 14]:
        f[f"ret_{lb}"] = close.pct_change(lb)

    # Bar range
    f["bar_range"] = (high - low) / close.replace(0, np.nan)

    # Volume z-score (handle zero/constant volume — common in forex HistData)
    vol_window = config.vol_zscore_window
    vol_mean = volume.rolling(vol_window, min_periods=10).mean()
    vol_std = volume.rolling(vol_window, min_periods=10).std()
    # NEW (B-024): always compute z-score, then fillna(0.0) for any NaN rows.
    # The previous 'if vol_std.max() > 0' check short-circuited the WHOLE
    # series to 0 even when only some windows had std=0; for assets with first
    # 30 bars constant volume (std=0), late-variable windows got suppressed.
    f["vol_zscore"] = (volume - vol_mean) / vol_std.replace(0, np.nan)
    f["vol_zscore"] = f["vol_zscore"].fillna(0.0)

    # Technical indicators
    f["rsi_14"] = rsi(close, 14)
    f["stoch_rsi"] = stochastic_rsi(close, 14, 14)
    f["macd_hist"] = macd(close)["macd_hist"]
    f["bb_pct_b"] = bollinger_pct_b(close, 20)
    f["atr_14"] = atr(high, low, close, 14)

    # Swing levels
    sw = swing_levels(high, low, lookback=config.swing_lookback)
    f["dist_swing_high"] = sw["dist_swing_high"]
    f["dist_swing_low"] = sw["dist_swing_low"]
    f["dist_nearest_level"] = sw["dist_nearest_level"]

    # --- Optional features (only computed if in feature_list) ---
    if "parkinson_vol_20" in config.feature_list:
        f["parkinson_vol_20"] = parkinson_vol(high, low, window=20)

    if "cs_spread" in config.feature_list:
        f["cs_spread"] = corwin_schultz_spread(high, low, window=20)

    if "entropy" in config.feature_list:
        f["entropy"] = rolling_entropy(close, window=50)

    if "ffd_close" in config.feature_list or "ffd_ret_1" in config.feature_list:
        try:
            log_close = np.log(close.replace(0, np.nan)).to_frame("log_close")
            ffd = frac_diff_ffd(
                log_close,
                d=config.fracdiff_d,
                threshold=config.fracdiff_threshold,
            )
            ffd_series = ffd.iloc[:, 0].reindex(f.index)
            if "ffd_close" in config.feature_list:
                f["ffd_close"] = ffd_series
            if "ffd_ret_1" in config.feature_list:
                f["ffd_ret_1"] = ffd_series.diff()
        except Exception:
            if "ffd_close" in config.feature_list:
                f["ffd_close"] = np.nan
            if "ffd_ret_1" in config.feature_list:
                f["ffd_ret_1"] = np.nan

    if "sadf" in config.feature_list:
        try:
            f["sadf"] = rolling_sadf(
                close,
                min_window=max(int(config.sadf_min_window), 20),
                step=max(int(config.sadf_step), 1),
            )
        except Exception:
            f["sadf"] = np.nan

    # Downcast to float32 where precision doesn't matter
    for col in f.columns:
        f[col] = f[col].astype(np.float32)

    # Return only features specified in config
    out = f[[c for c in config.feature_list if c in f.columns]]
    logger.debug("Computed %d features on %d bars", len(out.columns), len(out))
    return out


# --------------------------------------------------------------------------
# Derivatives features (optional — requires HL asset_ctxs data)
# --------------------------------------------------------------------------

DERIVATIVES_FEATURE_NAMES = [
    "funding_zscore", "funding_zscore_long", "funding_cumulative",
    "oi_change_5", "oi_zscore", "oi_price_divergence",
    "premium_zscore", "impact_skew",
]


def compute_derivatives_features(
    deriv_15m: pd.DataFrame,
    close: pd.Series,
    feature_list: list[str] | None = None,
) -> pd.DataFrame:
    """Compute derivatives features from HL asset_ctxs data.

    All features are strictly backward-looking (rolling windows or
    point-in-time snapshots).  Gracefully handles missing columns —
    returns only features that can be computed from available data.

    Args:
        deriv_15m: Derivatives DataFrame resampled to OHLCV frequency.
            Expected columns: funding, open_interest, mark_px, oracle_px,
            mid_px, impact_bid_px, impact_ask_px.
        close: Close price series aligned to deriv_15m index.
        feature_list: Which derivatives features to compute.
            Defaults to all 8.

    Returns:
        DataFrame with float32 columns, same index as *close*.
    """
    from afml.derivatives import (
        funding_features,
        oi_features,
        premium_features,
        impact_features,
    )

    if feature_list is None:
        feature_list = DERIVATIVES_FEATURE_NAMES

    feat = pd.DataFrame(index=close.index)

    # Funding features
    funding_cols = {"funding_zscore", "funding_zscore_long", "funding_cumulative"}
    if funding_cols & set(feature_list) and "funding" in deriv_15m.columns:
        ff = funding_features(deriv_15m["funding"], window_short=50, window_long=200)
        for col in funding_cols & set(feature_list):
            if col in ff.columns:
                feat[col] = ff[col].reindex(close.index)

    # OI features
    oi_cols = {"oi_change_5", "oi_zscore", "oi_price_divergence"}
    if oi_cols & set(feature_list) and "open_interest" in deriv_15m.columns:
        of = oi_features(deriv_15m["open_interest"], close, window=50)
        for col in oi_cols & set(feature_list):
            if col in of.columns:
                feat[col] = of[col].reindex(close.index)

    # Premium features
    if "premium_zscore" in feature_list:
        if "mark_px" in deriv_15m.columns and "oracle_px" in deriv_15m.columns:
            pf = premium_features(
                deriv_15m["mark_px"], deriv_15m["oracle_px"], window=50,
            )
            if "premium_zscore" in pf.columns:
                feat["premium_zscore"] = pf["premium_zscore"].reindex(close.index)

    # Impact features
    if "impact_skew" in feature_list:
        if "impact_bid_px" in deriv_15m.columns and "impact_ask_px" in deriv_15m.columns:
            mid = deriv_15m.get(
                "mid_px",
                (deriv_15m["impact_bid_px"] + deriv_15m["impact_ask_px"]) / 2,
            )
            imf = impact_features(
                deriv_15m["impact_bid_px"], deriv_15m["impact_ask_px"], mid,
            )
            if "impact_skew" in imf.columns:
                feat["impact_skew"] = imf["impact_skew"].reindex(close.index)

    # Downcast to float32
    for col in feat.columns:
        feat[col] = feat[col].astype(np.float32)

    computed = [c for c in feature_list if c in feat.columns]
    logger.debug("Computed %d/%d derivatives features", len(computed), len(feature_list))
    return feat[computed] if computed else pd.DataFrame(index=close.index)
