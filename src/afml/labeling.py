"""
Triple-Barrier Labeling + Meta-Labeling — AFML Chapter 3.

The triple-barrier method labels each trade entry by which of three barriers
is touched first:
    +1: Upper barrier (profit target) hit first
    -1: Lower barrier (stop loss) hit first
     0: Vertical barrier (time expiry) hit first → label = sign of return

Meta-labeling decouples SIDE from SIZE:
    Primary model predicts direction (long/short)
    Meta-labeler predicts whether to take the trade {0=skip, 1=take}
    Bet size = predicted probability of label=1

Reference: López de Prado (2018), Ch 3, Snippets 3.1-3.6.

CRITICAL NOTES:
- Barriers are SYMMETRIC by default (pt_sl=[1,1]). Asymmetric barriers
  (e.g., [2,1]) bias toward profit-taking at 2× vol while stopping at 1× vol.
- The vertical barrier should be wide enough to allow the trade thesis to play out.
  For 4H bars: 10-30 bars (2-5 days). For 1m bars: 240-1440 bars (4h-1day).
- Dynamic thresholds (target=volatility series) are ESSENTIAL. Fixed thresholds
  generate mostly 0-labels during calm periods and mostly ±1 during volatile ones.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_vertical_barrier(
    t_events: pd.DatetimeIndex,
    close: pd.Series,
    num_bars: int,
) -> pd.Series:
    """
    Add vertical (time) barriers to events.

    For each event timestamp, find the bar index `num_bars` ahead.
    If the series ends before num_bars, use the last available bar.

    Args:
        t_events: Event timestamps (e.g., from CUSUM filter).
        close: Close price series (used for index alignment).
        num_bars: Number of bars for the vertical barrier.

    Returns:
        Series mapping each event to its vertical barrier timestamp.
    """
    # Find integer positions of events in the close index
    t1 = close.index.searchsorted(t_events) + num_bars
    # Clip to series length
    t1 = t1[t1 < close.shape[0]]
    # Map back to timestamps
    t1 = pd.Series(close.index[t1], index=t_events[: len(t1)])
    return t1


def apply_pt_sl_on_t1(
    close: pd.Series,
    events: pd.DataFrame,
    pt_sl: list[float],
) -> pd.DataFrame:
    """
    Find the first barrier touch (profit-take, stop-loss, or vertical).

    For each event, walks forward through the price path and checks
    whether price hits the upper barrier (profit-take) or lower barrier
    (stop-loss) before the vertical barrier (time expiry).

    Args:
        close: Close price series.
        events: DataFrame with columns:
            - 't1': vertical barrier timestamp
            - 'trgt': volatility target (threshold for barriers)
            - 'side': +1 for long, -1 for short (optional, default +1)
        pt_sl: [profit_take_multiplier, stop_loss_multiplier].
               Barrier = pt_sl[i] * trgt. Set to 0 to disable a barrier.

    Returns:
        DataFrame with columns 'pt' (profit-take time), 'sl' (stop-loss time),
        't1' (vertical barrier time). NaT means that barrier was not touched.
    """
    out = events[["t1"]].copy()
    # Initialize as same dtype as t1 to avoid tz-naive/aware mismatch
    nat_series = pd.Series(pd.NaT, index=events.index, dtype=events["t1"].dtype)
    out["pt"] = nat_series.copy()
    out["sl"] = nat_series.copy()

    # Handle optional side column
    if "side" not in events.columns:
        events = events.copy()
        events["side"] = 1.0

    for loc, row in events.iterrows():
        t1 = row["t1"]
        if pd.isna(t1):
            t1 = close.index[-1]

        # Price path from entry to vertical barrier
        path = close.loc[loc:t1]
        if len(path) < 2:
            continue

        # Returns relative to entry, adjusted for side
        path_returns = (path / close.loc[loc] - 1) * row["side"]

        # Check profit-take barrier
        if pt_sl[0] > 0:
            pt_threshold = pt_sl[0] * row["trgt"]
            pt_hits = path_returns[path_returns > pt_threshold]
            if len(pt_hits) > 0:
                out.at[loc, "pt"] = pt_hits.index[0]

        # Check stop-loss barrier
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
    side: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Main entry point: generate triple-barrier events.

    Combines event timestamps, volatility targets, vertical barriers,
    and profit-take/stop-loss barriers into a single events DataFrame.

    Args:
        close: Close price series.
        t_events: Event timestamps (from CUSUM filter).
        pt_sl: [profit_take_mult, stop_loss_mult]. Barrier = mult * target.
        target: Volatility target series (from get_daily_vol).
        min_ret: Minimum target return. Events with target < min_ret are dropped.
        num_bars_vertical: Number of bars for vertical barrier.
        side: Optional Series of predicted sides (+1/-1). For meta-labeling.
              If None, both directions are considered (standard labeling).

    Returns:
        DataFrame with columns:
            - 't1': first barrier touch timestamp
            - 'trgt': volatility target at event time
            - 'side': trade direction (if side was provided)
    """
    # Filter events to those in the target index
    target = target.reindex(t_events).dropna()
    t_events = target.index

    # Apply minimum return filter
    if min_ret > 0:
        target = target[target > min_ret]
        t_events = target.index

    # Build events DataFrame
    events = pd.DataFrame(index=t_events)
    events["trgt"] = target

    # Add side if provided (for meta-labeling)
    if side is not None:
        events["side"] = side.reindex(t_events).fillna(1.0)
    else:
        events["side"] = 1.0  # Default: long direction

    # Add vertical barrier
    vertical = add_vertical_barrier(t_events, close, num_bars_vertical)
    events["t1"] = vertical.reindex(t_events)

    # Apply profit-take and stop-loss barriers
    barriers = apply_pt_sl_on_t1(close, events, pt_sl)

    # For each event, find the FIRST barrier touched
    # Compare pt, sl, and t1 timestamps — earliest wins
    for col in ["pt", "sl"]:
        barriers[col] = barriers[col].fillna(barriers["t1"])
    events["t1"] = barriers[["pt", "sl", "t1"]].min(axis=1)

    # Drop events where vertical barrier couldn't be set (end of data)
    events = events.dropna(subset=["t1"])

    return events


def get_bins(
    events: pd.DataFrame,
    close: pd.Series,
) -> pd.DataFrame:
    """
    Compute labels from triple-barrier events.

    For standard labeling: label = sign of return at first barrier touch.
    For meta-labeling (when 'side' column exists): label = 0 (primary wrong)
    or 1 (primary correct).

    Args:
        events: DataFrame from get_events() with 't1' and optionally 'side'.
        close: Close price series.

    Returns:
        DataFrame with columns:
            - 'ret': return from entry to first barrier touch
            - 'bin': label (-1, 0, +1 for standard; 0, 1 for meta-labeling)
    """
    events = events.dropna(subset=["t1"])

    # Compute returns
    out = pd.DataFrame(index=events.index)

    # Get close prices at entry and exit
    px_entry = close.reindex(events.index)

    # Use .loc for exit prices to preserve timezone info
    # (.values strips tz, causing reindex to return NaN)
    px_exit_values = []
    for t1_val in events["t1"]:
        if pd.isna(t1_val):
            px_exit_values.append(np.nan)
        elif t1_val in close.index:
            px_exit_values.append(close.loc[t1_val])
        else:
            px_exit_values.append(np.nan)

    out["ret"] = np.array(px_exit_values) / px_entry.values - 1

    # Apply side if meta-labeling
    if "side" in events.columns:
        out["ret"] = out["ret"] * events["side"]

    # Standard labeling: bin = sign(ret)
    out["bin"] = np.sign(out["ret"])

    # Meta-labeling: convert to {0, 1}
    if "side" in events.columns:
        out.loc[out["ret"] <= 0, "bin"] = 0  # Primary model was wrong → skip
        out.loc[out["ret"] > 0, "bin"] = 1   # Primary model was right → take

    return out


def drop_rare_labels(
    events: pd.DataFrame,
    min_pct: float = 0.05,
) -> pd.DataFrame:
    """
    Drop labels that appear less than min_pct of the time.

    Prevents training on extremely rare classes that would cause
    the model to be biased.

    Args:
        events: DataFrame with 'bin' column.
        min_pct: Minimum fraction for a label to be kept.

    Returns:
        Filtered DataFrame.
    """
    while True:
        counts = events["bin"].value_counts(normalize=True)
        if counts.min() > min_pct or len(counts) < 3:
            break
        rarest = counts.idxmin()
        events = events[events["bin"] != rarest]
    return events
