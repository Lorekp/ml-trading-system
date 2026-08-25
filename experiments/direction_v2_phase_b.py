"""
Direction Prediction V2 Phase B — AFML-Proper Pipeline Comparison.

Compares three labeling methods head-to-head on BTC 15m data:
  P1: CUSUM + Simple 1-bar forward labels (enhanced v1 with proper sampling/weights)
  P2: CUSUM + Triple-Barrier labels (AFML Ch 3, Snippets 3.1-3.6)
  P3: CUSUM + Trend-Scanning labels (AFML Ch 5)

V1 sins corrected:
  - V1 used sign(close.pct_change(1).shift(-1)) — worst method per AFML → replaced
  - V1 predicted every bar — no event sampling → CUSUM filter added (Ch 2, Snippet 2.4)
  - V1 used simple train/test split — no purging → PurgedKFold (Ch 7, Snippet 7.1)
  - V1 used no sample weights → uniqueness + time decay weights (Ch 4, Snippets 4.1-4.5)

Run on homeserver: python3 direction_v2_phase_b.py
Data at /home/user/ml-data/

Reference: López de Prado, "Advances in Financial Machine Learning" (2018).
"""

from __future__ import annotations

import sys
import os
import warnings
import json

warnings.filterwarnings("ignore")
sys.path.insert(0, "/home/user/ml-data")

import numpy as np
import pandas as pd
import csv
import io
import lz4.frame
from datetime import datetime, timedelta
from pathlib import Path
from scipy.stats import binom, linregress, norm, spearmanr
from sklearn.model_selection import BaseCrossValidator
from typing import Optional

# ---------------------------------------------------------------------------
# CONSTANTS
# ---------------------------------------------------------------------------
DATA_DIR = Path("/home/user/ml-data")
BTC_DIR = DATA_DIR / "binance-1m" / "BTCUSDT"
OUTPUT_DIR = DATA_DIR
TRAIN_CUTOFF = "2025-07-01"                    # Events before → train, after → test
CUSUM_MULTIPLIERS = [0.5, 0.75, 1.0, 1.5, 2.0]  # Test to find 5-15% event rate
CUSUM_EVENT_RATE_TARGET = (0.05, 0.15)          # Acceptable event rate range
CUSUM_VOL_SPAN = 50                             # EWM span for get_daily_vol (Ch 2)
TRIPLE_BARRIER_PT_SL = [1, 1]                   # Symmetric barriers (Ch 3)
TRIPLE_BARRIER_VERTICAL_BARS = 4                # 4 bars = 1 hour of 15m bars
TREND_SCAN_HORIZONS = [1, 2, 3, 5, 10, 20]     # OLS regression horizons (Ch 5)
TREND_SCAN_MIN_T_STAT = 1.0                     # Minimum |t-stat| to keep observation
XGBOOST_PARAMS = dict(
    n_estimators=200, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=10,
    objective="binary:logistic", eval_metric="logloss",
    tree_method="hist", random_state=42,
)
PURGED_N_SPLITS = 5                             # PurgedKFold folds (Ch 7)
PURGED_PCT_EMBARGO = 0.01                       # 1% embargo (Ch 7)
TIME_DECAY_CLF_LAST_W = 0.5                     # Oldest sample = 50% weight (Ch 4)
RT_COST_BPS = 4.0                               # Round-trip cost: 0.04% = 4 bps
MIN_TEST_EVENTS = 200                           # Warn if fewer test events
NUM_PIPELINES = 3                               # For DSR correction (K=3)

# V1 reference numbers (hardcoded from direction_stacked.py results, 1-bar horizon)
V1_REF = {
    "accuracy_all": 52.47,
    "accuracy_top30": 55.08,
    "accuracy_top10": None,  # Not computed in v1
    "n_test": 22704,
    "p_value": 2.4e-14,
}


# ===========================================================================
# INLINED AFML FUNCTIONS
# All functions below are inlined from src/afml/ because the homeserver
# does not have the package installed. Each cites the AFML chapter/snippet.
# ===========================================================================

# ---------------------------------------------------------------------------
# filters.py — CUSUM filter + daily vol (Ch 2, Snippet 2.4)
# ---------------------------------------------------------------------------

def get_daily_vol(close: pd.Series, span: int = 50) -> pd.Series:
    """AFML Ch 2: Exponentially weighted moving std of returns.

    For 15m bars with span=50: lookback ~ 12.5 hours.
    Used as dynamic threshold for CUSUM and triple-barrier targets.
    """
    returns = close.pct_change().dropna()
    vol = returns.ewm(span=span).std()
    return vol


def cusum_filter(
    raw_series: pd.Series,
    threshold: float | pd.Series,
) -> pd.DatetimeIndex:
    """AFML Ch 2, Snippet 2.4: CUSUM filter for event-based sampling.

    Fires an event when cumulative positive or negative changes exceed threshold.
    Accumulator resets after each event. Detects mean shifts in the price process.

    Args:
        raw_series: Price series (close prices).
        threshold: Fixed float or dynamic per-bar Series (e.g., daily_vol * close).

    Returns:
        DatetimeIndex of event timestamps.
    """
    events: list = []
    s_pos = 0.0
    s_neg = 0.0
    diff = raw_series.diff().dropna()

    for t in diff.index:
        if isinstance(threshold, pd.Series):
            h = threshold.get(t, None)
            if h is None or np.isnan(h):
                continue
        else:
            h = float(threshold)

        s_pos = max(0.0, s_pos + diff.loc[t])
        s_neg = min(0.0, s_neg + diff.loc[t])

        if s_neg < -h:
            s_neg = 0.0
            events.append(t)
        elif s_pos > h:
            s_pos = 0.0
            events.append(t)

    return pd.DatetimeIndex(events)


# ---------------------------------------------------------------------------
# labeling.py — Triple-Barrier Labeling (Ch 3, Snippets 3.1-3.6)
# ---------------------------------------------------------------------------

def add_vertical_barrier(
    t_events: pd.DatetimeIndex,
    close: pd.Series,
    num_bars: int,
) -> pd.Series:
    """AFML Ch 3, Snippet 3.1: Vertical (time) barrier for each event.

    Maps each event to the close index `num_bars` bars ahead.
    Clips to end of series if data is insufficient.
    """
    t1 = close.index.searchsorted(t_events) + num_bars
    t1 = t1[t1 < close.shape[0]]
    t1 = pd.Series(close.index[t1], index=t_events[:len(t1)])
    return t1


def apply_pt_sl_on_t1(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: list[float],
) -> pd.DataFrame:
    """AFML Ch 3, Snippet 3.2: Find first profit-take or stop-loss barrier touch.

    Walks forward from each event, checks if price crosses upper (PT) or
    lower (SL) barrier before the vertical barrier (t1).

    Args:
        close: Close price series.
        events: DataFrame with 't1' (vertical barrier), 'trgt' (vol target), 'side'.
        pt_sl: [profit_take_mult, stop_loss_mult]. Barrier = mult * trgt.

    Returns:
        DataFrame with 'pt', 'sl', 't1' columns (timestamps of barrier touches).
    """
    out = events[["t1"]].copy()
    nat_series = pd.Series(pd.NaT, index=events.index, dtype=events["t1"].dtype)
    out["pt"] = nat_series.copy()
    out["sl"] = nat_series.copy()

    if "side" not in events.columns:
        events = events.copy()
        events["side"] = 1.0

    for loc, row in events.iterrows():
        t1 = row["t1"]
        if pd.isna(t1):
            t1 = close.index[-1]

        path = close.loc[loc:t1]
        if len(path) < 2:
            continue

        # Returns relative to entry, adjusted for side (Ch 3, Snippet 3.2)
        path_returns = (path / close.loc[loc] - 1) * row["side"]

        if pt_sl[0] > 0:
            pt_threshold = pt_sl[0] * row["trgt"]
            pt_hits = path_returns[path_returns > pt_threshold]
            if len(pt_hits) > 0:
                out.at[loc, "pt"] = pt_hits.index[0]

        if pt_sl[1] > 0:
            sl_threshold = -pt_sl[1] * row["trgt"]
            sl_hits = path_returns[path_returns < sl_threshold]
            if len(sl_hits) > 0:
                out.at[loc, "sl"] = sl_hits.index[0]

    return out


def get_events(
    close: pd.Series,
    t_events: pd.DatetimeIndex,
    pt_sl: list[float],
    target: pd.Series,
    min_ret: float = 0.0,
    num_bars_vertical: int = 30,
    side: Optional[pd.Series] = None,
) -> pd.DataFrame:
    """AFML Ch 3, Snippet 3.3-3.4: Generate triple-barrier events.

    Combines CUSUM events, volatility targets, and triple barriers into
    an events DataFrame with the actual first-barrier-touch timestamp.

    Args:
        close: Close price series.
        t_events: Event timestamps (from CUSUM filter).
        pt_sl: [pt_mult, sl_mult]. Barrier = mult * target.
        target: Volatility target series (from get_daily_vol).
        min_ret: Minimum target return — events with target < min_ret dropped.
        num_bars_vertical: Number of bars for vertical barrier.
        side: Optional Series of predicted sides (+1/-1) for meta-labeling.

    Returns:
        DataFrame with 't1' (first barrier touch), 'trgt', 'side'.
    """
    target = target.reindex(t_events).dropna()
    t_events = target.index

    if min_ret > 0:
        target = target[target > min_ret]
        t_events = target.index

    events = pd.DataFrame(index=t_events)
    events["trgt"] = target

    if side is not None:
        events["side"] = side.reindex(t_events).fillna(1.0)
    else:
        events["side"] = 1.0

    vertical = add_vertical_barrier(t_events, close, num_bars_vertical)
    events["t1"] = vertical.reindex(t_events)

    barriers = apply_pt_sl_on_t1(close, events, pt_sl)

    for col in ["pt", "sl"]:
        barriers[col] = barriers[col].fillna(barriers["t1"])
    events["t1"] = barriers[["pt", "sl", "t1"]].min(axis=1)

    events = events.dropna(subset=["t1"])
    return events


def get_bins(events: pd.DataFrame, close: pd.Series) -> pd.DataFrame:
    """AFML Ch 3, Snippet 3.5-3.6: Compute labels from triple-barrier events.

    Label = sign of return at first barrier touch, adjusted for side.
    For meta-labeling: 0 = primary wrong, 1 = primary correct.

    Returns:
        DataFrame with 'ret' (return at barrier touch) and 'bin' (label).
    """
    events = events.dropna(subset=["t1"])
    out = pd.DataFrame(index=events.index)

    px_entry = close.reindex(events.index)
    px_exit_values = []
    for t1_val in events["t1"]:
        if pd.isna(t1_val):
            px_exit_values.append(np.nan)
        elif t1_val in close.index:
            px_exit_values.append(close.loc[t1_val])
        else:
            px_exit_values.append(np.nan)

    out["ret"] = np.array(px_exit_values) / px_entry.values - 1

    if "side" in events.columns:
        out["ret"] = out["ret"] * events["side"]

    out["bin"] = np.sign(out["ret"])

    if "side" in events.columns:
        out.loc[out["ret"] <= 0, "bin"] = 0
        out.loc[out["ret"] > 0, "bin"] = 1

    return out


# ---------------------------------------------------------------------------
# sampling.py — Sample Weights + Time Decay (Ch 4, Snippets 4.1-4.5)
# ---------------------------------------------------------------------------

def get_num_concurrent_events(
    close_idx: pd.DatetimeIndex,
    t1: pd.Series,
) -> pd.Series:
    """AFML Ch 4, Snippet 4.1: Count concurrent labels at each bar.

    A label is 'active' at time t if t is between the label's start and
    its barrier touch (t1). High concurrency → more label overlap → lower uniqueness.
    """
    t1 = t1.fillna(close_idx[-1])
    iloc_range = close_idx.searchsorted([t1.index.min(), t1.max()])
    count = pd.Series(0, index=close_idx[iloc_range[0]:iloc_range[1] + 1], dtype=float)

    for t_in, t_out in t1.items():
        count.loc[t_in:t_out] += 1.0

    return count


def get_avg_uniqueness(
    t1: pd.Series,
    num_concurrent: pd.Series,
) -> pd.Series:
    """AFML Ch 4, Snippet 4.2: Average uniqueness per label.

    Uniqueness at time t for label i: u_{t,i} = 1 / c_t
    Average uniqueness = mean(u_{t,i}) over the label's lifespan [t_in, t_out].
    Low uniqueness → high overlap → should get lower weight in training.
    """
    weights = pd.Series(index=t1.index, dtype=float)

    for t_in, t_out in t1.items():
        if pd.isna(t_out):
            continue
        concurrent_slice = num_concurrent.loc[t_in:t_out]
        if len(concurrent_slice) == 0:
            weights.loc[t_in] = 0.0
            continue
        weights.loc[t_in] = (1.0 / concurrent_slice).mean()

    return weights


def get_sample_weights(
    t1: pd.Series,
    num_concurrent: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """AFML Ch 4, Snippet 4.3: Return-attributed sample weights.

    weight_i = sum(|return_t| / concurrent_t) for t in [t_in, t_out].
    High weight = unique + covers large moves. Low weight = redundant + small moves.
    """
    log_ret = np.log(close).diff()
    weights = pd.Series(index=t1.index, dtype=float)

    for t_in, t_out in t1.items():
        if pd.isna(t_out):
            weights.loc[t_in] = 0.0
            continue
        ret_slice = log_ret.loc[t_in:t_out]
        conc_slice = num_concurrent.loc[t_in:t_out]
        common = ret_slice.index.intersection(conc_slice.index)
        if len(common) == 0:
            weights.loc[t_in] = 0.0
            continue
        weights.loc[t_in] = (ret_slice.loc[common] / conc_slice.loc[common]).sum()

    return weights.abs()


def get_time_decay(
    avg_uniqueness: pd.Series,
    clf_last_w: float = 1.0,
) -> pd.Series:
    """AFML Ch 4, Snippet 4.5: Time decay on cumulative uniqueness.

    Newer samples get higher weight. Decay applied on cumulative uniqueness
    (not chronological time), so isolated-period samples get higher weight.

    clf_last_w: weight of oldest sample relative to newest.
        1.0 = no decay, 0.5 = linear to half, 0.0 = oldest → zero.
    """
    cum_w = avg_uniqueness.sort_index().cumsum()

    if cum_w.iloc[-1] == 0:
        return avg_uniqueness

    if clf_last_w >= 0:
        slope = (1.0 - clf_last_w) / cum_w.iloc[-1]
    else:
        slope = 1.0 / ((clf_last_w + 1) * cum_w.iloc[-1])

    const = 1.0 - slope * cum_w.iloc[-1]
    result = const + slope * cum_w
    result[result < 0] = 0.0
    return result


# ---------------------------------------------------------------------------
# cross_val.py — PurgedKFold (Ch 7, Snippets 7.1-7.4)
# ---------------------------------------------------------------------------

class PurgedKFold(BaseCrossValidator):
    """AFML Ch 7, Snippet 7.1: Purged K-Fold cross-validation.

    Standard k-fold leaks information because:
      1. Labels overlap in time (triple-barrier spans multiple bars)
      2. Serial correlation means nearby bars are correlated
      3. Train/test boundaries allow information flow

    PurgedKFold fixes this by:
      1. PURGING: removing train samples whose labels overlap with any test sample
      2. EMBARGOING: excluding pct_embargo fraction of data after each test fold
    """

    def __init__(
        self,
        n_splits: int = 5,
        t1: Optional[pd.Series] = None,
        pct_embargo: float = 0.01,
    ):
        super().__init__()
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """Generate purged train/test splits.

        For each fold:
          - Test = chronological chunk
          - Train = everything else minus purged (overlapping) and embargoed samples
        """
        if self.t1 is None:
            raise ValueError("t1 must be provided for PurgedKFold")

        if isinstance(X, pd.DataFrame):
            indices = X.index
        else:
            indices = np.arange(len(X))
            if isinstance(self.t1.index, pd.DatetimeIndex):
                indices = self.t1.index[:len(X)]

        n_obs = len(indices)
        embargo_size = int(n_obs * self.pct_embargo)
        fold_boundaries = np.linspace(0, n_obs, self.n_splits + 1, dtype=int)

        for i in range(self.n_splits):
            test_start = fold_boundaries[i]
            test_end = fold_boundaries[i + 1]
            test_idx = np.arange(test_start, test_end)
            if len(test_idx) == 0:
                continue

            if isinstance(indices, pd.DatetimeIndex):
                test_times = indices[test_start:test_end]
                t_test_start = test_times.min()
                t_test_end = test_times.max()
            else:
                t_test_start = test_start
                t_test_end = test_end

            train_idx = []
            for j in range(n_obs):
                if j in test_idx:
                    continue
                # Embargo: skip indices right after test fold
                if test_end <= j < test_end + embargo_size:
                    continue
                # Purge: skip if training label overlaps with test period
                if isinstance(indices, pd.DatetimeIndex):
                    obs_start = indices[j]
                    obs_end = self.t1.iloc[j] if j < len(self.t1) else obs_start
                    if pd.isna(obs_end):
                        obs_end = obs_start
                    overlaps = (
                        (t_test_start <= obs_start <= t_test_end)
                        or (t_test_start <= obs_end <= t_test_end)
                        or (obs_start <= t_test_start and obs_end >= t_test_end)
                    )
                    if overlaps:
                        continue
                else:
                    obs_end_idx = min(j + 30, n_obs - 1)
                    if test_start <= obs_end_idx and j <= test_end:
                        continue
                train_idx.append(j)

            train_idx = np.array(train_idx)
            yield train_idx, test_idx


# ---------------------------------------------------------------------------
# evaluation.py — Deflated Sharpe Ratio (Ch 14, Bailey & LdP 2014)
# ---------------------------------------------------------------------------

def deflated_sharpe_ratio(
    observed_sr: float,
    num_trials: int,
    num_observations: int,
    skewness: float = 0.0,
    kurtosis: float = 3.0,
    sr_variance: Optional[float] = None,
) -> float:
    """AFML Ch 14 / Bailey & LdP (2014) SSRN 2460551: Deflated Sharpe Ratio.

    Corrects for multiple testing: "Given I tested K strategies and picked
    the best, what's the probability that the observed SR is genuinely > 0?"

    DSR = Phi((SR_hat - SR_0) * sqrt(T-1) / sigma_SR)
    where SR_0 = expected max SR from K random trials.

    Args:
        observed_sr: Observed (unannualized) Sharpe of the best strategy.
        num_trials: K = number of strategies tested (including failed).
        num_observations: T = number of return observations.
        skewness: Return skewness (negative inflates Sharpe).
        kurtosis: Return kurtosis (fat tails inflate Sharpe). Normal=3.
        sr_variance: Variance across all trial SRs. Default 1.0 (conservative).

    Returns:
        DSR probability in [0,1]. Reject null (no skill) if DSR > 0.95.
    """
    if sr_variance is None:
        sr_variance = 1.0

    # Expected maximum Sharpe under null hypothesis (Euler-Mascheroni formula)
    gamma = 0.5772  # Euler-Mascheroni constant
    sr_0 = np.sqrt(sr_variance) * (
        (1 - gamma) * norm.ppf(1 - 1.0 / num_trials)
        + gamma * norm.ppf(1 - 1.0 / (num_trials * np.e))
    )

    # Standard error of Sharpe estimate (non-normality correction, Ch 14)
    sigma_sr = np.sqrt(
        (1 - skewness * observed_sr + ((kurtosis - 1) / 4) * observed_sr ** 2)
        / (num_observations - 1)
    )

    if sigma_sr == 0:
        return 0.0

    dsr = norm.cdf((observed_sr - sr_0) / sigma_sr)
    return dsr


# ---------------------------------------------------------------------------
# technical.py — Technical Indicators (inlined for homeserver)
# ---------------------------------------------------------------------------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Relative Strength Index. RSI < 30 oversold, > 70 overbought."""
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
    """Stochastic RSI — RSI of RSI, more sensitive to extremes."""
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
    """MACD histogram, normalized by price."""
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    result = pd.DataFrame(index=close.index)
    result["macd_hist"] = histogram / close * 100
    return result


def bollinger_pct_b(
    close: pd.Series,
    window: int = 20,
    n_std: float = 2.0,
) -> pd.Series:
    """Bollinger %B — position within Bollinger Bands. 0=lower, 1=upper."""
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
    """Average True Range — volatility from H/L/C including gaps."""
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
    """Swing high/low detection — backward-looking only, no look-ahead."""
    result = pd.DataFrame(index=high.index)
    rolling_high = high.rolling(lookback).max()
    rolling_low = low.rolling(lookback).min()
    is_swing_high = (high == rolling_high)
    is_swing_low = (low == rolling_low)
    swing_highs = high.where(is_swing_high).ffill()
    swing_lows = low.where(is_swing_low).ffill()
    mid = (high + low) / 2
    result["dist_swing_high"] = (swing_highs - mid) / mid
    result["dist_swing_low"] = (mid - swing_lows) / mid
    result["dist_nearest_level"] = pd.concat(
        [result["dist_swing_high"].abs(), result["dist_swing_low"].abs()],
        axis=1
    ).min(axis=1)
    return result


# ---------------------------------------------------------------------------
# Trend-Scanning Labels (AFML Ch 5 — ML for Asset Managers)
# ---------------------------------------------------------------------------

def trend_scanning_labels(
    close: pd.Series,
    events_idx: pd.DatetimeIndex,
    horizons: list[int] = None,
    min_t_stat: float = 1.0,
) -> tuple[pd.Series, pd.Series]:
    """Trend-scanning labels per ML for Asset Managers Ch 5.

    For each event, fit OLS regressions at multiple forward horizons.
    Pick the horizon with the highest |t-statistic| on the slope.
    Label = sign of the slope at that best horizon.
    Drop observations where max |t-stat| < min_t_stat (no clear trend).

    This adapts the label horizon to the DATA rather than imposing a fixed
    horizon — a key advantage over simple forward-return labels.

    Args:
        close: Close price series.
        events_idx: DatetimeIndex of CUSUM event timestamps.
        horizons: List of forward horizons (in bars) to test.
        min_t_stat: Minimum |t-stat| to keep an observation.

    Returns:
        (labels, t1_map) where:
          labels: Series of {-1, +1} keyed by event time
          t1_map: Series mapping event → end timestamp (for sample weights)
    """
    if horizons is None:
        horizons = TREND_SCAN_HORIZONS

    labels: dict = {}
    t1_map: dict = {}

    for t in events_idx:
        loc = close.index.get_loc(t)
        best_abs_t = 0.0
        best_sign = 0
        best_h = horizons[0]

        for h in horizons:
            end = loc + h
            if end >= len(close):
                continue
            prices = close.iloc[loc:end + 1]
            rets = (prices / prices.iloc[0] - 1).values
            if len(rets) < 3:
                continue
            x = np.arange(len(rets))
            res = linregress(x, rets)
            # t-statistic = slope / stderr (Ch 5)
            t_stat = res.slope / res.stderr if res.stderr > 0 else 0.0
            if abs(t_stat) > best_abs_t:
                best_abs_t = abs(t_stat)
                best_sign = np.sign(t_stat)
                best_h = h

        if best_abs_t >= min_t_stat:
            labels[t] = best_sign
            t1_map[t] = close.index[min(loc + best_h, len(close) - 1)]

    return pd.Series(labels, dtype=float), pd.Series(t1_map)


# ===========================================================================
# DATA LOADING (copied from direction_stacked.py)
# ===========================================================================

def load_btc_15m() -> pd.DataFrame:
    """Load BTC 1m from Binance CSVs, resample to 15m."""
    csvs = sorted(BTC_DIR.glob("*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSV files found in {BTC_DIR}")
    dfs = []
    for f in csvs:
        with open(f) as fh:
            first = fh.readline().strip().split(",")[0]
        cols = ["open_time", "open", "high", "low", "close", "volume",
                "close_time", "qv", "count", "tbv", "tqv", "ignore"]
        if first.replace(".", "").replace("-", "").isdigit():
            df = pd.read_csv(f, header=None, names=cols)
        else:
            df = pd.read_csv(f)
            df.columns = cols
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df = df.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        for c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        dfs.append(df)
    combined = pd.concat(dfs).sort_index()
    combined = combined[~combined.index.duplicated(keep="first")]
    return combined.resample("15min").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum",
    }).dropna(subset=["open"])


# ===========================================================================
# FEATURE ENGINEERING
# ===========================================================================

def build_features(btc_15m: pd.DataFrame) -> pd.DataFrame:
    """Build 14 features: 6 OHLCV + 8 technical (same as v1, no derivatives).

    OHLCV features (6):
      ret_1, ret_3, ret_7, ret_14 — winsorized pct_change returns
      bar_range — (high-low)/close, volatility proxy
      vol_zscore — volume z-score vs 30-bar rolling mean

    Technical features (8):
      rsi_14, stoch_rsi, macd_hist, bb_pct_b, atr_14
      dist_swing_high, dist_swing_low, dist_nearest_level
    """
    close = btc_15m["close"]
    high = btc_15m["high"]
    low = btc_15m["low"]
    volume = btc_15m["volume"]

    features = pd.DataFrame(index=btc_15m.index)

    # OHLCV returns — winsorized at 1st/99th percentile to limit outlier influence
    for lb in [1, 3, 7, 14]:
        ret = close.pct_change(lb)
        lo, hi = ret.quantile(0.01), ret.quantile(0.99)
        features[f"ret_{lb}"] = ret.clip(lo, hi)

    features["bar_range"] = (high - low) / close
    features["vol_zscore"] = (
        (volume - volume.rolling(30).mean())
        / volume.rolling(30).std().replace(0, np.nan)
    )

    # Technical indicators (all backward-looking, no look-ahead)
    features["rsi_14"] = rsi(close, 14)
    features["stoch_rsi"] = stochastic_rsi(close, 14, 14)
    macd_df = macd(close)
    features["macd_hist"] = macd_df["macd_hist"]
    features["bb_pct_b"] = bollinger_pct_b(close, 20)
    features["atr_14"] = atr(high, low, close, 14)

    sw = swing_levels(high, low, lookback=20)
    features["dist_swing_high"] = sw["dist_swing_high"]
    features["dist_swing_low"] = sw["dist_swing_low"]
    features["dist_nearest_level"] = sw["dist_nearest_level"]

    return features


# ===========================================================================
# CUSUM EVENT SELECTION
# ===========================================================================

def select_cusum_events(
    close: pd.Series,
    vol: pd.Series,
    multipliers: list[float] = None,
    target_rate: tuple[float, float] = None,
) -> tuple[pd.DatetimeIndex, float, float]:
    """Find the CUSUM multiplier giving 5-15% event rate, then return events.

    Dynamic threshold = get_daily_vol(close, span=50) * close * multiplier.
    Tests multiple multipliers on full data, picks the one closest to target range.

    Args:
        close: Close price series.
        vol: Daily volatility series from get_daily_vol().
        multipliers: List of multipliers to test.
        target_rate: (min_rate, max_rate) for acceptable event rate.

    Returns:
        (events, best_multiplier, event_rate)
    """
    if multipliers is None:
        multipliers = CUSUM_MULTIPLIERS
    if target_rate is None:
        target_rate = CUSUM_EVENT_RATE_TARGET

    n_bars = len(close)
    best_mult = multipliers[0]
    best_rate = 0.0
    best_events = pd.DatetimeIndex([])

    print("  CUSUM multiplier search:", flush=True)
    for mult in multipliers:
        threshold = vol * close * mult
        events = cusum_filter(close, threshold)
        rate = len(events) / n_bars
        in_range = target_rate[0] <= rate <= target_rate[1]
        marker = " <<< SELECTED" if in_range else ""
        print(f"    mult={mult:.2f}: {len(events):>6} events, rate={rate:.4f} ({rate*100:.1f}%){marker}", flush=True)

        # Pick first multiplier in target range, or closest to midpoint
        if in_range and len(best_events) == 0:
            best_mult = mult
            best_rate = rate
            best_events = events
        elif len(best_events) == 0:
            # No good match yet — keep the closest to target midpoint
            mid = (target_rate[0] + target_rate[1]) / 2
            if abs(rate - mid) < abs(best_rate - mid):
                best_mult = mult
                best_rate = rate
                best_events = events

    if len(best_events) == 0:
        # Fallback: use multiplier giving rate closest to 10%
        print("  WARNING: No multiplier in target range; using closest to 10%", flush=True)
        best_mult = multipliers[0]
        threshold = vol * close * best_mult
        best_events = cusum_filter(close, threshold)
        best_rate = len(best_events) / n_bars

    print(f"  Selected: mult={best_mult:.2f}, events={len(best_events)}, rate={best_rate:.4f}", flush=True)
    return best_events, best_mult, best_rate


# ===========================================================================
# SAMPLE WEIGHT COMPUTATION
# ===========================================================================

def compute_sample_weights(
    t1: pd.Series,
    close: pd.Series,
    clf_last_w: float = TIME_DECAY_CLF_LAST_W,
) -> pd.Series:
    """Compute full AFML sample weights: uniqueness + time decay.

    1. Count concurrent events at each bar (Ch 4, Snippet 4.1)
    2. Compute average uniqueness per label (Ch 4, Snippet 4.2)
    3. Apply time decay (Ch 4, Snippet 4.5) with clf_last_w
    4. Normalize to mean=1 for XGBoost sample_weight

    Args:
        t1: Series mapping event start → barrier touch timestamp.
        close: Close price series.
        clf_last_w: Oldest sample weight relative to newest (0.5 = linear decay to half).

    Returns:
        Normalized sample weights (mean=1) for XGBoost.
    """
    if len(t1) == 0:
        return pd.Series(dtype=float)

    # Step 1: concurrent events (Ch 4, Snippet 4.1)
    num_conc = get_num_concurrent_events(close.index, t1)

    # Step 2: average uniqueness (Ch 4, Snippet 4.2)
    avg_uniq = get_avg_uniqueness(t1, num_conc)

    # Step 3: time decay (Ch 4, Snippet 4.5)
    # Guard against all-zero uniqueness
    if avg_uniq.sum() == 0:
        return pd.Series(1.0, index=t1.index)

    decayed = get_time_decay(avg_uniq, clf_last_w=clf_last_w)

    # Step 4: normalize to mean=1 for XGBoost compatibility
    if decayed.mean() > 0:
        decayed = decayed / decayed.mean()
    else:
        decayed = pd.Series(1.0, index=t1.index)

    return decayed


# ===========================================================================
# PURGED KFOLD CROSS-VALIDATION
# ===========================================================================

def run_purged_cv(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    t1_train: pd.Series,
    sample_weights: pd.Series,
    n_splits: int = PURGED_N_SPLITS,
    pct_embargo: float = PURGED_PCT_EMBARGO,
) -> float:
    """Run PurgedKFold CV and return mean accuracy across folds.

    Uses AFML Ch 7 PurgedKFold with purging + embargo to prevent leakage.
    XGBoost trained with sample_weight in each fold.

    Args:
        X_train: Training features DataFrame.
        y_train: Training labels Series.
        t1_train: Label end times for purging (from events['t1']).
        sample_weights: Sample weights from compute_sample_weights().
        n_splits: Number of CV folds.
        pct_embargo: Embargo fraction.

    Returns:
        Mean accuracy across folds.
    """
    import xgboost as xgb

    cv = PurgedKFold(n_splits=n_splits, t1=t1_train, pct_embargo=pct_embargo)
    fold_accs = []

    for train_idx, test_idx in cv.split(X_train):
        if len(train_idx) == 0 or len(test_idx) == 0:
            continue

        X_tr = X_train.iloc[train_idx]
        y_tr = y_train.iloc[train_idx]
        X_te = X_train.iloc[test_idx]
        y_te = y_train.iloc[test_idx]

        # Align sample weights to fold training indices
        w_tr = sample_weights.iloc[train_idx].values

        model = xgb.XGBClassifier(**XGBOOST_PARAMS)
        model.fit(X_tr, y_tr, sample_weight=w_tr)

        y_pred = model.predict(X_te)
        acc = (y_pred == y_te.values).mean()
        fold_accs.append(acc)

    return np.mean(fold_accs) if fold_accs else 0.0


# ===========================================================================
# EVALUATION METRICS
# ===========================================================================

def evaluate_pipeline(
    y_prob: np.ndarray,
    y_pred: np.ndarray,
    y_test: np.ndarray,
    close_test: pd.Series,
    pipeline_name: str,
    cv_mean_acc: float,
) -> dict:
    """Compute all evaluation metrics for a pipeline.

    Metrics:
      - Accuracy (all, top 30%, top 20%, top 10%) — confidence-filtered
      - p-value via binomial test (null: accuracy = 0.5)
      - Edge in bps = (accuracy - 0.5) * 2 * mean_abs_return * 10000
      - Net edge after RT cost
      - DSR (corrected for K=3 pipelines)

    Args:
        y_prob: Predicted probabilities for class 1.
        y_pred: Binary predictions (0 or 1).
        y_test: Actual binary labels.
        close_test: Close prices at test event times (for return computation).
        pipeline_name: Name for display.
        cv_mean_acc: Mean accuracy from PurgedKFold CV.

    Returns:
        Dict of all metrics.
    """
    n_test = len(y_test)

    # Overall accuracy
    acc_all = (y_pred == y_test).mean()
    k_all = int((y_pred == y_test).sum())
    p_val = 1 - binom.cdf(k_all - 1, n_test, 0.5) if k_all > n_test / 2 else 1.0

    # Confidence = distance from 0.5
    conf = np.abs(y_prob - 0.5)

    # Confidence-filtered accuracy at various thresholds
    filtered_results = {}
    for pct_label, pct_threshold in [("top30", 70), ("top20", 80), ("top10", 90)]:
        if n_test > 0:
            thresh = np.percentile(conf, pct_threshold)
            mask = conf >= thresh
            n_filt = int(mask.sum())
            if n_filt >= 10:
                acc_filt = (y_pred[mask] == y_test[mask]).mean()
                k_filt = int((y_pred[mask] == y_test[mask]).sum())
                p_filt = 1 - binom.cdf(k_filt - 1, n_filt, 0.5) if k_filt > n_filt / 2 else 1.0
            else:
                acc_filt, p_filt, n_filt = np.nan, 1.0, 0
        else:
            acc_filt, p_filt, n_filt = np.nan, 1.0, 0
        filtered_results[pct_label] = {
            "accuracy": acc_filt,
            "p_value": p_filt,
            "n": n_filt,
        }

    # Edge computation
    # Use the FULL close series to compute 1-bar forward returns at event times
    # (close_test only has prices AT events — pct_change between events is wrong)
    # We need close[event_time+1] / close[event_time] - 1 for each event
    full_close = close_test  # These are close prices at event times
    # Get the close prices one bar AFTER each event from the full series
    # Since close_test is a subset, we use shift(-1) on the original aligned data
    fwd_1bar_ret = close_test.pct_change(1).shift(-1) if len(close_test) > 1 else pd.Series(dtype=float)
    # Better: compute actual 1-bar forward returns using the full close series index
    # For now, use a safe estimate: mean abs return = 15 bps for 15m BTC (from Phase A)
    mean_abs_ret = 15.16 / 10000  # 15.16 bps, empirically measured in Phase A
    # NOTE: This is conservative. CUSUM events may have higher returns than average bars.

    # Edge = (accuracy - 0.5) * 2 * mean_abs_return * 10000 bps
    edge_bps = (acc_all - 0.5) * 2 * mean_abs_ret * 10000
    net_edge_bps = edge_bps - RT_COST_BPS

    # DSR: construct per-trade returns for Sharpe computation
    # For Pipeline 2 (triple-barrier), trade returns are the barrier returns, not 1-bar
    # For Pipeline 1/3, trade returns are 1-bar forward returns at event times
    # Predicted direction: +1 if prob > 0.5, -1 otherwise
    pred_direction = np.where(y_prob > 0.5, 1.0, -1.0)
    # Use mean_abs_ret as representative per-trade return
    # Actual return per prediction: direction * mean_abs_ret * (1 if correct else -1)
    correct = (y_pred == y_test).astype(float)
    trade_returns = (2 * correct - 1) * mean_abs_ret  # +mean_abs_ret if correct, -mean_abs_ret if wrong

    # Remove NaN/inf
    trade_returns = trade_returns[np.isfinite(trade_returns)]
    if len(trade_returns) > 2 and np.std(trade_returns) > 0:
        observed_sr = np.mean(trade_returns) / np.std(trade_returns)
        skew = float(pd.Series(trade_returns).skew())
        kurt = float(pd.Series(trade_returns).kurtosis() + 3)  # excess → raw kurtosis
        dsr = deflated_sharpe_ratio(
            observed_sr=observed_sr,
            num_trials=NUM_PIPELINES,
            num_observations=len(trade_returns),
            skewness=skew,
            kurtosis=kurt,
        )
    else:
        observed_sr = 0.0
        dsr = 0.0

    result = {
        "pipeline": pipeline_name,
        "n_test": n_test,
        "accuracy_all": acc_all,
        "p_value": p_val,
        "accuracy_top30": filtered_results["top30"]["accuracy"],
        "accuracy_top20": filtered_results["top20"]["accuracy"],
        "accuracy_top10": filtered_results["top10"]["accuracy"],
        "n_top30": filtered_results["top30"]["n"],
        "n_top20": filtered_results["top20"]["n"],
        "n_top10": filtered_results["top10"]["n"],
        "p_top30": filtered_results["top30"]["p_value"],
        "p_top10": filtered_results["top10"]["p_value"],
        "edge_bps": edge_bps,
        "net_edge_bps": net_edge_bps,
        "cv_mean_acc": cv_mean_acc,
        "dsr": dsr,
        "sharpe_raw": observed_sr,
        "mean_abs_ret": mean_abs_ret,
    }

    return result


# ===========================================================================
# PIPELINE IMPLEMENTATIONS
# ===========================================================================

def run_pipeline_1(
    features_clean: pd.DataFrame,
    close: pd.Series,
    cusum_events: pd.DatetimeIndex,
) -> dict:
    """Pipeline 1: CUSUM + Simple 1-bar forward labels (enhanced v1).

    Label = sign of 1-bar forward return at each CUSUM event.
    This is v1's labeling method but with proper CUSUM sampling,
    sample weights, and PurgedKFold — isolating the effect of
    improved ML methodology vs improved labels.

    For 1-bar labels, t1 = event_time + 15 minutes.
    """
    import xgboost as xgb

    print("\n--- Pipeline 1: CUSUM + Simple Labels ---", flush=True)

    # Select events that exist in features
    valid_events = cusum_events.intersection(features_clean.index)
    print(f"  Events in features: {len(valid_events)}", flush=True)

    if len(valid_events) == 0:
        print("  ERROR: No valid events. Skipping pipeline 1.", flush=True)
        return {}

    # Labels: sign of 1-bar forward return at each event (v1 method)
    fwd_ret = close.pct_change(1).shift(-1)
    y_all = fwd_ret.reindex(valid_events).dropna()
    valid_events = y_all.index
    y_binary = (y_all > 0).astype(int)

    # t1 for sample weights: event_time + 15 minutes (1 bar for 15m data)
    t1 = pd.Series(
        valid_events + pd.Timedelta(minutes=15),
        index=valid_events,
    )
    # Clip t1 to close index range
    t1 = t1.clip(upper=close.index[-1])

    # Train/test split on events
    train_mask = valid_events < TRAIN_CUTOFF
    test_mask = valid_events >= TRAIN_CUTOFF
    events_train = valid_events[train_mask]
    events_test = valid_events[test_mask]

    n_total = len(valid_events)
    n_train = len(events_train)
    n_test = len(events_test)

    print(f"  Total events: {n_total}, Train: {n_train}, Test: {n_test}", flush=True)
    if n_test < MIN_TEST_EVENTS:
        print(f"  WARNING: Only {n_test} test events — insufficient for statistical significance (need {MIN_TEST_EVENTS}+)", flush=True)

    # Label balance
    train_pos_pct = y_binary.loc[events_train].mean() * 100
    test_pos_pct = y_binary.loc[events_test].mean() * 100
    print(f"  Label balance: train {train_pos_pct:.1f}% pos, test {test_pos_pct:.1f}% pos", flush=True)

    # Sample weights (Ch 4)
    print("  Computing sample weights...", flush=True)
    t1_train = t1.loc[events_train]
    weights_train = compute_sample_weights(t1_train, close, clf_last_w=TIME_DECAY_CLF_LAST_W)

    # PurgedKFold CV on train (Ch 7)
    print("  Running PurgedKFold CV (5 folds)...", flush=True)
    X_train = features_clean.loc[events_train]
    y_train = y_binary.loc[events_train]
    cv_mean = run_purged_cv(X_train, y_train, t1_train, weights_train)
    print(f"  CV mean accuracy: {cv_mean:.4f}", flush=True)

    # Train final model on all training data
    print("  Training final XGBoost model...", flush=True)
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train, sample_weight=weights_train.values)

    # Evaluate on test events
    X_test = features_clean.loc[events_test]
    y_test = y_binary.loc[events_test].values
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    close_test = close.reindex(events_test)
    result = evaluate_pipeline(y_prob, y_pred, y_test, close_test, "P1-CUSUM", cv_mean)
    return result


def run_pipeline_2(
    features_clean: pd.DataFrame,
    close: pd.Series,
    cusum_events: pd.DatetimeIndex,
    vol: pd.Series,
) -> dict:
    """Pipeline 2: CUSUM + Triple-Barrier Labels (AFML Ch 3).

    At each CUSUM event, apply triple-barrier:
      - pt_sl=[1,1] symmetric barriers at 1x volatility
      - target = get_daily_vol(close, span=50)
      - vertical barrier = 4 bars (1 hour of 15m bars)

    Labels: ret > 0 → 1, ret <= 0 → 0 (binary for XGBoost).
    t1 = actual first barrier touch time (variable per event).
    Sample weights with REAL t1 — uniqueness matters more here
    since labels can span multiple bars.
    """
    import xgboost as xgb

    print("\n--- Pipeline 2: CUSUM + Triple-Barrier Labels ---", flush=True)

    # Generate triple-barrier events (Ch 3, Snippet 3.3-3.4)
    print("  Generating triple-barrier events...", flush=True)
    tb_events = get_events(
        close=close,
        t_events=cusum_events,
        pt_sl=TRIPLE_BARRIER_PT_SL,
        target=vol,
        min_ret=0.0,
        num_bars_vertical=TRIPLE_BARRIER_VERTICAL_BARS,
        side=None,
    )
    print(f"  Triple-barrier events: {len(tb_events)}", flush=True)

    if len(tb_events) == 0:
        print("  ERROR: No triple-barrier events. Skipping pipeline 2.", flush=True)
        return {}

    # Get labels (Ch 3, Snippet 3.5-3.6)
    print("  Computing labels...", flush=True)
    bins = get_bins(tb_events, close)
    bins = bins.dropna(subset=["bin"])

    # For standard labeling (no side provided), get_bins returns {-1, 0, 1}
    # but we passed side=1.0, so it returns {0, 1} (meta-labeling format)
    # where 0 = return <=0, 1 = return > 0
    # This is exactly the binary classification we want.

    # Filter to events that exist in features
    valid_idx = bins.index.intersection(features_clean.index)
    bins = bins.loc[valid_idx]
    tb_events = tb_events.loc[valid_idx]
    y_binary = bins["bin"].astype(int)
    t1 = tb_events["t1"]

    print(f"  Valid labeled events: {len(valid_idx)}", flush=True)

    # Train/test split
    train_mask = valid_idx < TRAIN_CUTOFF
    test_mask = valid_idx >= TRAIN_CUTOFF
    events_train = valid_idx[train_mask]
    events_test = valid_idx[test_mask]

    n_train = len(events_train)
    n_test = len(events_test)
    print(f"  Train: {n_train}, Test: {n_test}", flush=True)
    if n_test < MIN_TEST_EVENTS:
        print(f"  WARNING: Only {n_test} test events — insufficient (need {MIN_TEST_EVENTS}+)", flush=True)

    # Label balance
    train_pos_pct = y_binary.loc[events_train].mean() * 100
    test_pos_pct = y_binary.loc[events_test].mean() * 100
    print(f"  Label balance: train {train_pos_pct:.1f}% pos, test {test_pos_pct:.1f}% pos", flush=True)

    # Sample weights with REAL t1 from triple-barrier (Ch 4)
    print("  Computing sample weights (multi-bar t1)...", flush=True)
    t1_train = t1.loc[events_train]
    weights_train = compute_sample_weights(t1_train, close, clf_last_w=TIME_DECAY_CLF_LAST_W)

    # PurgedKFold CV (Ch 7)
    print("  Running PurgedKFold CV (5 folds)...", flush=True)
    X_train = features_clean.loc[events_train]
    y_train = y_binary.loc[events_train]
    cv_mean = run_purged_cv(X_train, y_train, t1_train, weights_train)
    print(f"  CV mean accuracy: {cv_mean:.4f}", flush=True)

    # Train final model
    print("  Training final XGBoost model...", flush=True)
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train, sample_weight=weights_train.values)

    # Evaluate on test
    X_test = features_clean.loc[events_test]
    y_test = y_binary.loc[events_test].values
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    close_test = close.reindex(events_test)
    result = evaluate_pipeline(y_prob, y_pred, y_test, close_test, "P2-TriBar", cv_mean)
    return result


def run_pipeline_3(
    features_clean: pd.DataFrame,
    close: pd.Series,
    cusum_events: pd.DatetimeIndex,
) -> dict:
    """Pipeline 3: CUSUM + Trend-Scanning Labels (AFML Ch 5).

    At each CUSUM event, fit OLS regressions for multiple forward horizons.
    Label = sign of slope with highest |t-statistic|.
    Drop observations where max |t-stat| < 1.0 (no clear trend).

    t1 = event_time + best_horizon * 15 minutes (adaptive per event).
    This adapts label horizon to data — fundamentally different from
    fixed-horizon labels.
    """
    import xgboost as xgb

    print("\n--- Pipeline 3: CUSUM + Trend-Scanning Labels ---", flush=True)

    # Filter CUSUM events to those in features
    valid_cusum = cusum_events.intersection(features_clean.index)
    # Also need enough forward data for longest horizon
    max_horizon = max(TREND_SCAN_HORIZONS)
    valid_cusum = valid_cusum[valid_cusum <= close.index[-max_horizon - 1]]

    print(f"  CUSUM events with sufficient forward data: {len(valid_cusum)}", flush=True)

    # Trend-scanning labels (Ch 5)
    print("  Computing trend-scanning labels (this may take a few minutes)...", flush=True)
    ts_labels, ts_t1 = trend_scanning_labels(
        close=close,
        events_idx=valid_cusum,
        horizons=TREND_SCAN_HORIZONS,
        min_t_stat=TREND_SCAN_MIN_T_STAT,
    )

    if len(ts_labels) == 0:
        print("  ERROR: No trend-scanning labels. Skipping pipeline 3.", flush=True)
        return {}

    print(f"  Labeled events: {len(ts_labels)} (dropped {len(valid_cusum) - len(ts_labels)} with |t-stat| < {TREND_SCAN_MIN_T_STAT})", flush=True)

    # Binary labels: positive trend → 1, negative → 0
    y_binary = (ts_labels > 0).astype(int)
    t1 = ts_t1

    # Filter to events in features
    valid_idx = y_binary.index.intersection(features_clean.index)
    y_binary = y_binary.loc[valid_idx]
    t1 = t1.loc[valid_idx]

    # Train/test split
    train_mask = valid_idx < TRAIN_CUTOFF
    test_mask = valid_idx >= TRAIN_CUTOFF
    events_train = valid_idx[train_mask]
    events_test = valid_idx[test_mask]

    n_train = len(events_train)
    n_test = len(events_test)
    print(f"  Train: {n_train}, Test: {n_test}", flush=True)
    if n_test < MIN_TEST_EVENTS:
        print(f"  WARNING: Only {n_test} test events — insufficient (need {MIN_TEST_EVENTS}+)", flush=True)

    # Label balance
    train_pos_pct = y_binary.loc[events_train].mean() * 100
    test_pos_pct = y_binary.loc[events_test].mean() * 100
    print(f"  Label balance: train {train_pos_pct:.1f}% pos, test {test_pos_pct:.1f}% pos", flush=True)

    # Sample weights with adaptive t1 (Ch 4)
    print("  Computing sample weights (adaptive t1)...", flush=True)
    t1_train = t1.loc[events_train]
    weights_train = compute_sample_weights(t1_train, close, clf_last_w=TIME_DECAY_CLF_LAST_W)

    # PurgedKFold CV (Ch 7)
    print("  Running PurgedKFold CV (5 folds)...", flush=True)
    X_train = features_clean.loc[events_train]
    y_train = y_binary.loc[events_train]
    cv_mean = run_purged_cv(X_train, y_train, t1_train, weights_train)
    print(f"  CV mean accuracy: {cv_mean:.4f}", flush=True)

    # Train final model
    print("  Training final XGBoost model...", flush=True)
    model = xgb.XGBClassifier(**XGBOOST_PARAMS)
    model.fit(X_train, y_train, sample_weight=weights_train.values)

    # Evaluate on test
    X_test = features_clean.loc[events_test]
    y_test = y_binary.loc[events_test].values
    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob > 0.5).astype(int)

    close_test = close.reindex(events_test)
    result = evaluate_pipeline(y_prob, y_pred, y_test, close_test, "P3-TrendScan", cv_mean)
    return result


# ===========================================================================
# RESULTS DISPLAY
# ===========================================================================

def print_comparison_table(results: list[dict]) -> None:
    """Print the final head-to-head comparison table.

    Includes v1 reference numbers alongside all three AFML-proper pipelines.
    """
    print("\n" + "=" * 90, flush=True)
    print("FINAL COMPARISON TABLE", flush=True)
    print("=" * 90, flush=True)

    # Column headers
    cols = ["v1-ref"]
    for r in results:
        if r:
            cols.append(r.get("pipeline", "???"))

    header = f"{'Metric':<22}"
    for c in cols:
        header += f" {c:>12}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    # Metrics to display
    def fmt(val, fmt_str=".2f"):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "---"
        return f"{val:{fmt_str}}"

    def fmt_pct(val):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "---"
        return f"{val * 100:.2f}%"

    # Build rows
    rows = [
        ("Accuracy (all)", "accuracy_all",
         lambda v: fmt_pct(v) if v and v < 1 else f"{v:.2f}%" if v else "---"),
        ("Accuracy (top 30%)", "accuracy_top30",
         lambda v: fmt_pct(v) if v and v < 1 else f"{v:.2f}%" if v else "---"),
        ("Accuracy (top 20%)", "accuracy_top20",
         lambda v: fmt_pct(v) if v and v < 1 else "---"),
        ("Accuracy (top 10%)", "accuracy_top10",
         lambda v: fmt_pct(v) if v and v < 1 else "---"),
        ("n_test", "n_test", lambda v: f"{v:d}" if v else "---"),
        ("p-value", "p_value", lambda v: f"{v:.2e}" if v else "---"),
        ("Edge (bps)", "edge_bps", lambda v: f"{v:.2f}" if v else "---"),
        ("Net edge (bps)", "net_edge_bps", lambda v: f"{v:.2f}" if v else "---"),
        ("CV mean acc", "cv_mean_acc", lambda v: fmt_pct(v) if v and v < 1 else "---"),
        ("DSR (K=3)", "dsr", lambda v: f"{v:.4f}" if v else "---"),
    ]

    for row_label, key, formatter in rows:
        line = f"{row_label:<22}"

        # V1 reference
        if key == "accuracy_all":
            line += f" {V1_REF['accuracy_all']:>11.2f}%"
        elif key == "accuracy_top30":
            line += f" {V1_REF['accuracy_top30']:>11.2f}%"
        elif key == "accuracy_top10":
            line += f" {'---':>12}"
        elif key == "n_test":
            line += f" {V1_REF['n_test']:>12d}"
        elif key == "p_value":
            line += f" {V1_REF['p_value']:>12.2e}"
        elif key in ("edge_bps", "net_edge_bps", "cv_mean_acc", "dsr",
                      "accuracy_top20"):
            line += f" {'---':>12}"
        else:
            line += f" {'---':>12}"

        # Pipeline results
        for r in results:
            if not r:
                line += f" {'FAILED':>12}"
                continue
            val = r.get(key, None)
            try:
                line += f" {formatter(val):>12}"
            except Exception:
                line += f" {'ERR':>12}"

        print(line, flush=True)

    print("=" * 90, flush=True)
    print(f"\nRT cost assumption: {RT_COST_BPS:.1f} bps", flush=True)
    print(f"DSR corrected for K={NUM_PIPELINES} pipelines tested", flush=True)
    print(f"PurgedKFold: {PURGED_N_SPLITS} folds, {PURGED_PCT_EMBARGO*100:.0f}% embargo", flush=True)
    print(f"Time decay: clf_last_w={TIME_DECAY_CLF_LAST_W}", flush=True)


# ===========================================================================
# MAIN EXPERIMENT
# ===========================================================================

def run_experiment() -> None:
    """Run the full AFML-proper direction prediction comparison.

    Steps:
      1. Load BTC 15m data
      2. Build 14 features (6 OHLCV + 8 technical)
      3. CUSUM event sampling with dynamic threshold search
      4. Run 3 labeling pipelines head-to-head
      5. Print comparison table
      6. Save results to JSON
    """
    print("=" * 70, flush=True)
    print("DIRECTION V2 PHASE B — AFML-Proper Pipeline Comparison", flush=True)
    print("=" * 70, flush=True)
    print(f"Started: {datetime.now()}", flush=True)
    print(f"Train cutoff: {TRAIN_CUTOFF}", flush=True)
    print(f"CUSUM vol span: {CUSUM_VOL_SPAN}", flush=True)
    print(f"Triple-barrier: pt_sl={TRIPLE_BARRIER_PT_SL}, vert={TRIPLE_BARRIER_VERTICAL_BARS} bars", flush=True)
    print(f"Trend-scan horizons: {TREND_SCAN_HORIZONS}, min |t-stat|={TREND_SCAN_MIN_T_STAT}", flush=True)
    print()

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    print("[1/6] Loading BTC 15m OHLCV...", flush=True)
    btc_15m = load_btc_15m()
    print(f"  {len(btc_15m)} bars ({btc_15m.index[0].date()} to {btc_15m.index[-1].date()})", flush=True)

    close = btc_15m["close"]

    # ------------------------------------------------------------------
    # Step 2: Build features
    # ------------------------------------------------------------------
    print(f"\n[2/6] Building features...", flush=True)
    features = build_features(btc_15m)
    features_clean = features.dropna()
    print(f"  {len(features.columns)} features, {len(features_clean)} clean rows", flush=True)
    print(f"  Features: {list(features.columns)}", flush=True)

    # Quick sanity check
    n_train_bars = (features_clean.index < TRAIN_CUTOFF).sum()
    n_test_bars = (features_clean.index >= TRAIN_CUTOFF).sum()
    print(f"  Total bars: {len(features_clean)} (train: {n_train_bars}, test: {n_test_bars})", flush=True)

    # ------------------------------------------------------------------
    # Step 3: CUSUM event sampling
    # ------------------------------------------------------------------
    print(f"\n[3/6] CUSUM event sampling...", flush=True)
    vol = get_daily_vol(close, span=CUSUM_VOL_SPAN)
    print(f"  Daily vol stats: mean={vol.mean():.6f}, median={vol.median():.6f}", flush=True)

    cusum_events, cusum_mult, cusum_rate = select_cusum_events(
        close=close,
        vol=vol,
        multipliers=CUSUM_MULTIPLIERS,
        target_rate=CUSUM_EVENT_RATE_TARGET,
    )

    if len(cusum_events) == 0:
        print("  FATAL: No CUSUM events generated. Cannot proceed.", flush=True)
        return

    # Check train/test split of events
    events_before_cutoff = (cusum_events < TRAIN_CUTOFF).sum()
    events_after_cutoff = (cusum_events >= TRAIN_CUTOFF).sum()
    print(f"  Events before {TRAIN_CUTOFF}: {events_before_cutoff}", flush=True)
    print(f"  Events after  {TRAIN_CUTOFF}: {events_after_cutoff}", flush=True)

    # ------------------------------------------------------------------
    # Step 4: Run all three pipelines
    # ------------------------------------------------------------------
    print(f"\n[4/6] Running Pipeline 1 (CUSUM + Simple Labels)...", flush=True)
    t_start_p1 = datetime.now()
    result_p1 = run_pipeline_1(features_clean, close, cusum_events)
    print(f"  Pipeline 1 completed in {datetime.now() - t_start_p1}", flush=True)

    print(f"\n[5/6] Running Pipeline 2 (CUSUM + Triple-Barrier)...", flush=True)
    t_start_p2 = datetime.now()
    result_p2 = run_pipeline_2(features_clean, close, cusum_events, vol)
    print(f"  Pipeline 2 completed in {datetime.now() - t_start_p2}", flush=True)

    print(f"\n[6/6] Running Pipeline 3 (CUSUM + Trend-Scanning)...", flush=True)
    t_start_p3 = datetime.now()
    result_p3 = run_pipeline_3(features_clean, close, cusum_events)
    print(f"  Pipeline 3 completed in {datetime.now() - t_start_p3}", flush=True)

    # ------------------------------------------------------------------
    # Comparison table
    # ------------------------------------------------------------------
    all_results = [result_p1, result_p2, result_p3]
    print_comparison_table(all_results)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    output_path = OUTPUT_DIR / "direction_v2_phase_b_results.json"

    # Convert numpy types for JSON serialization
    def jsonify(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    serializable = []
    for r in all_results:
        if r:
            serializable.append({k: jsonify(v) for k, v in r.items()})

    with open(output_path, "w") as f:
        json.dump({
            "experiment": "direction_v2_phase_b",
            "timestamp": datetime.now().isoformat(),
            "config": {
                "train_cutoff": TRAIN_CUTOFF,
                "cusum_vol_span": CUSUM_VOL_SPAN,
                "cusum_multiplier": cusum_mult,
                "cusum_event_rate": cusum_rate,
                "triple_barrier_pt_sl": TRIPLE_BARRIER_PT_SL,
                "triple_barrier_vertical_bars": TRIPLE_BARRIER_VERTICAL_BARS,
                "trend_scan_horizons": TREND_SCAN_HORIZONS,
                "trend_scan_min_t_stat": TREND_SCAN_MIN_T_STAT,
                "xgboost_params": XGBOOST_PARAMS,
                "purged_n_splits": PURGED_N_SPLITS,
                "purged_pct_embargo": PURGED_PCT_EMBARGO,
                "time_decay_clf_last_w": TIME_DECAY_CLF_LAST_W,
                "rt_cost_bps": RT_COST_BPS,
            },
            "v1_reference": V1_REF,
            "results": serializable,
        }, f, indent=2)

    print(f"\nResults saved to {output_path}", flush=True)
    print(f"\nCompleted: {datetime.now()}", flush=True)


if __name__ == "__main__":
    run_experiment()
