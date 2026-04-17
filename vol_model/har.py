"""Causal HAR (Corsi 2009) forecaster for residual-target training.

HAR decomposes log-volatility into a cascade of rolling-mean horizons
(daily, weekly, monthly) plus intercept. It's the gold-standard baseline
for realized-vol forecasting. We use it here to re-parameterize the ML
target:

    y_residual(t) = y_true(t) - HAR_pred(t)

The ML model (pruned_svi_v1 architecture) then predicts only the
microstructure/event residual; reconstruction at inference is
``y_pred(t) = model_pred(t) + HAR_pred(t)``.

**Leakage guarantee** (enforced by unit tests):
    HAR_pred(t) uses ONLY y values with timestamp strictly less than t.

The module is intentionally lightweight -- a thin, explicit wrapper over
expanding-window OLS -- so the leakage argument can be read off the
source in a few lines without wrapping around an LGBM dataset.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# HAR horizons are parameterized in BARS (not minutes) to stay
# freq-agnostic. Defaults below target 15-minute bars:
#     1 bar  = 15 min
#     4 bars = 1 hour
#     96 bars = 1 day
#     672 bars = 1 week
# The production pipeline's 15-minute timeframe is hardcoded in practice,
# but we keep the horizons explicit so a future 5-minute or 1-hour refit
# stays ergonomic.
HAR_DEFAULT_HORIZONS_BARS: tuple[int, ...] = (1, 4, 96, 672)


@dataclass
class HARFit:
    """Container for a fitted HAR forecast series."""

    predictions: pd.Series   # indexed identically to the input y; NaN where history insufficient
    horizons: tuple[int, ...]
    warmup_bars: int         # first N rows have NaN predictions; kept explicit for caller
    min_train_bars: int      # min training window before any OOS pred is made


def _rolling_mean_shifted(y: pd.Series, window: int) -> pd.Series:
    """Rolling mean of last `window` values STRICTLY BEFORE the current bar.

    This is the canonical causal rolling mean for vol forecasting:
    at bar t we use y[t-window : t-1] (inclusive-exclusive), NOT y[t].

    Implementation: rolling(window).mean().shift(1). The shift by one bar
    is the explicit leakage guard -- the current-bar value never enters
    the rolling mean.
    """
    return y.rolling(window=window, min_periods=1).mean().shift(1)


def fit_causal_har(
    y: pd.Series,
    horizons: tuple[int, ...] = HAR_DEFAULT_HORIZONS_BARS,
    min_train_bars: int | None = None,
) -> HARFit:
    """Fit an expanding-window HAR with strict causality.

    At every row t:
      1. Feature vector is the rolling mean of y over each horizon, shifted
         by one bar so the current value never enters.
      2. OLS coefficients are refit on y[warmup : t-1] (expanding).
      3. Prediction is features(t) @ coefficients.

    Parameters
    ----------
    y
        Log-volatility series indexed by timestamp (DatetimeIndex OK).
    horizons
        Tuple of rolling-mean window lengths in bars.
    min_train_bars
        Minimum number of observations before any OOS prediction is issued.
        Defaults to max(max(horizons), 5 * max(horizons) // 1) for stability.

    Returns
    -------
    HARFit
        Predictions indexed identically to y. The first
        ``warmup_bars`` rows are NaN because the rolling-mean features
        themselves require a warmup window (equal to max(horizons)).
    """
    if not isinstance(y, pd.Series):
        raise TypeError("y must be a pandas Series")
    if y.empty:
        raise ValueError("y is empty")

    max_h = int(max(horizons))
    warmup = max_h  # rolling+shift(1) => first `max_h` rows are NaN
    if min_train_bars is None:
        # Need at least enough rows to fit the 4-variable OLS stably.
        min_train_bars = max(warmup + 50, 5 * len(horizons))

    n = len(y)
    predictions = pd.Series(np.nan, index=y.index)
    if n < min_train_bars + 1:
        return HARFit(predictions, tuple(horizons), warmup, min_train_bars)

    # Build the causal feature matrix once (all rolling means are shifted).
    feature_arrays = [
        _rolling_mean_shifted(y, h).to_numpy()
        for h in horizons
    ]
    X_all = np.column_stack(feature_arrays)  # (n, k)
    y_arr = y.to_numpy()

    # Expanding-window OOS loop. We start issuing predictions at
    # t = min_train_bars so the first fit has enough data to be stable.
    for t in range(min_train_bars, n):
        # Training rows: every row strictly before t that has a valid
        # feature vector (i.e., past warmup).
        if t <= warmup:
            continue
        X_train = X_all[warmup:t]
        y_train = y_arr[warmup:t]
        # Drop NaN guards
        if np.any(np.isnan(X_train)) or np.any(np.isnan(y_train)):
            # The rolling/shift guarantees warmup rows are NaN; after that
            # they should be clean. Defensive skip.
            continue
        X_aug = np.column_stack([np.ones(len(X_train)), X_train])
        try:
            beta, *_ = np.linalg.lstsq(X_aug, y_train, rcond=None)
        except np.linalg.LinAlgError:
            continue

        x_t = X_all[t]
        if np.any(np.isnan(x_t)):
            continue
        predictions.iloc[t] = beta[0] + x_t @ beta[1:]

    return HARFit(predictions, tuple(horizons), warmup, min_train_bars)


def build_har_residual_target(
    y: pd.Series,
    horizons: tuple[int, ...] = HAR_DEFAULT_HORIZONS_BARS,
) -> tuple[pd.Series, pd.Series]:
    """Convenience: build (residual_target, har_pred) aligned to y.

    Returns
    -------
    residual : pd.Series
        ``y - har_pred``, with NaN on rows where HAR is warming up.
    har_pred : pd.Series
        Causal HAR forecast aligned to y.
    """
    fit = fit_causal_har(y, horizons=horizons)
    return y - fit.predictions, fit.predictions


def reconstruct_from_residual(
    residual_pred: pd.Series | np.ndarray,
    har_pred: pd.Series | np.ndarray,
) -> np.ndarray:
    """Inverse of build_har_residual_target.

    Given ML-predicted residuals and HAR predictions, reconstruct the
    full log-vol prediction via simple addition.
    """
    r = np.asarray(residual_pred, dtype=float)
    h = np.asarray(har_pred, dtype=float)
    if r.shape != h.shape:
        raise ValueError(f"shape mismatch: residual={r.shape} har={h.shape}")
    return r + h
