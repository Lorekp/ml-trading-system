"""Windowing / contextualization — lags and EMAs for any feature set.

For every raw feature f, produces:
    f (t0): current value (identity)
    f_lag1: f.shift(1) — 1 bar ago
    f_lag3: f.shift(3) — 3 bars ago (45 min at 15m)
    f_ema10: ewm(span=10) — short-term smoothed
    f_ema50: ewm(span=50) — long-term smoothed

All operations are strictly backward-looking (shift and EMA use only past values).
"""

from __future__ import annotations

import pandas as pd

from vol_model.config import VolModelConfig


def apply_windowing(
    features: pd.DataFrame,
    lags: list[int] | None = None,
    ema_spans: list[int] | None = None,
) -> pd.DataFrame:
    """Apply windowing (lags + EMAs) to all columns in features.

    Args:
        features: DataFrame with raw features (one column per feature).
        lags: list of lag periods (default [1, 3]).
        ema_spans: list of EMA spans (default [10, 50]).

    Returns:
        DataFrame with original + lagged + EMA columns.
        Column naming: {original}_lag{N}, {original}_ema{N}
    """
    if lags is None:
        lags = [1, 3]
    if ema_spans is None:
        ema_spans = [10, 50]

    result_parts = [features]  # Include original columns as t(0)

    for lag in lags:
        lagged = features.shift(lag)
        lagged.columns = [f"{c}_lag{lag}" for c in features.columns]
        result_parts.append(lagged)

    for span in ema_spans:
        ema = features.ewm(span=span, min_periods=1).mean()
        ema.columns = [f"{c}_ema{span}" for c in features.columns]
        result_parts.append(ema)

    return pd.concat(result_parts, axis=1)
