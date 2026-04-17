"""
Sample Weights + Sequential Bootstrap — AFML Chapter 4.

Financial labels overlap in time: a triple-barrier label spanning t to t+30
shares returns with labels starting at t+1, t+2, etc. This makes samples
non-IID, causing ML models to overfit on redundant information.

This module computes:
1. Number of concurrent labels at each bar
2. Average uniqueness of each label (how much it shares with others)
3. Sample weights based on uniqueness and return attribution
4. Sequential bootstrap (draws samples to maximize uniqueness)
5. Time decay weights (older samples less relevant)

Reference: López de Prado (2018), Ch 4, Snippets 4.1-4.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def get_num_concurrent_events(
    close_idx: pd.DatetimeIndex,
    t1: pd.Series,
) -> pd.Series:
    """
    Count how many labels are active at each bar.

    A label is "active" at time t if t is between the label's start
    and its first barrier touch (t1).

    Args:
        close_idx: Full DatetimeIndex of the close price series.
        t1: Series mapping label start → first barrier touch timestamp.

    Returns:
        Series (indexed by close_idx) of concurrent label counts.
    """
    t1 = t1.fillna(close_idx[-1])

    # Trim to relevant range
    iloc_range = close_idx.searchsorted([t1.index.min(), t1.max()])
    count = pd.Series(0, index=close_idx[iloc_range[0] : iloc_range[1] + 1], dtype=float)

    for t_in, t_out in t1.items():
        count.loc[t_in:t_out] += 1.0

    return count


def get_avg_uniqueness(
    t1: pd.Series,
    num_concurrent: pd.Series,
) -> pd.Series:
    """
    Compute average uniqueness for each label.

    Uniqueness at time t for label i: u_{t,i} = 1 / c_t
    where c_t is the number of concurrent labels at t.
    Average uniqueness: mean of u_{t,i} over the label's lifespan.

    Labels with low uniqueness (high overlap) should get lower weight.

    Args:
        t1: Series mapping label start → barrier touch timestamp.
        num_concurrent: Concurrent event count from get_num_concurrent_events.

    Returns:
        Series of average uniqueness per label (index = label start times).
    """
    weights = pd.Series(index=t1.index, dtype=float)

    for t_in, t_out in t1.items():
        if pd.isna(t_out):
            continue
        concurrent_slice = num_concurrent.loc[t_in:t_out]
        if len(concurrent_slice) == 0:
            weights.loc[t_in] = 0.0
            continue
        # Uniqueness = 1/concurrent, averaged over lifespan
        weights.loc[t_in] = (1.0 / concurrent_slice).mean()

    return weights


def get_sample_weights(
    t1: pd.Series,
    num_concurrent: pd.Series,
    close: pd.Series,
) -> pd.Series:
    """
    Compute return-attributed sample weights.

    Combines uniqueness with absolute return attribution:
    weight_i = sum(|return_t| / concurrent_t) for t in [t_in, t_out]

    This gives high weight to labels that:
    1. Have low overlap (high uniqueness)
    2. Cover large price moves (high absolute return)

    Args:
        t1: Series mapping label start → barrier touch timestamp.
        num_concurrent: Concurrent event count.
        close: Close price series.

    Returns:
        Series of absolute return-attributed weights per label.
    """
    log_ret = np.log(close).diff()
    weights = pd.Series(index=t1.index, dtype=float)

    for t_in, t_out in t1.items():
        if pd.isna(t_out):
            weights.loc[t_in] = 0.0
            continue
        ret_slice = log_ret.loc[t_in:t_out]
        conc_slice = num_concurrent.loc[t_in:t_out]
        # Align indices
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
    """
    Apply time-decay to sample weights.

    Newer samples get higher weight than older ones.
    Decay is applied on cumulative uniqueness (not chronological time),
    so samples in isolated periods get higher decay than clustered ones.

    Args:
        avg_uniqueness: Average uniqueness weights from get_avg_uniqueness.
        clf_last_w: Weight of the oldest sample relative to newest.
                    1.0 = no decay, 0.5 = linear decay to half, 0.0 = oldest → zero.

    Returns:
        Series of time-decayed weights.
    """
    cum_w = avg_uniqueness.sort_index().cumsum()

    if clf_last_w >= 0:
        slope = (1.0 - clf_last_w) / cum_w.iloc[-1]
    else:
        slope = 1.0 / ((clf_last_w + 1) * cum_w.iloc[-1])

    const = 1.0 - slope * cum_w.iloc[-1]
    result = const + slope * cum_w
    result[result < 0] = 0.0
    return result


def get_indicator_matrix(
    t1: pd.Series,
    close_idx: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Build indicator matrix for sequential bootstrap.

    Element (t, i) = 1 if bar t falls within label i's lifespan, 0 otherwise.
    This matrix is used to efficiently compute uniqueness during bootstrap.

    Args:
        t1: Series mapping label start → barrier touch.
        close_idx: Full DatetimeIndex of close prices.

    Returns:
        DataFrame (bars × labels), binary indicator matrix.
    """
    # Trim close_idx to relevant range
    idx_start = close_idx.searchsorted(t1.index.min())
    idx_end = close_idx.searchsorted(t1.max()) + 1
    trimmed_idx = close_idx[idx_start:idx_end]

    ind_matrix = pd.DataFrame(0, index=trimmed_idx, columns=range(len(t1)))

    for i, (t_in, t_out) in enumerate(t1.items()):
        if pd.isna(t_out):
            continue
        ind_matrix.loc[t_in:t_out, i] = 1

    return ind_matrix


def seq_bootstrap(
    ind_matrix: pd.DataFrame,
    sample_length: int | None = None,
) -> list[int]:
    """
    Sequential bootstrap — draws samples to maximize average uniqueness.

    Standard bootstrap on financial data is broken because overlapping
    labels cause correlated draws. Sequential bootstrap fixes this by
    making the probability of drawing a sample inversely proportional
    to its overlap with already-drawn samples.

    Args:
        ind_matrix: Binary indicator matrix from get_indicator_matrix.
        sample_length: Number of samples to draw. Default: number of labels.

    Returns:
        List of column indices (label indices) in draw order.
    """
    if sample_length is None:
        sample_length = ind_matrix.shape[1]

    drawn = []

    for _ in range(sample_length):
        avg_u = pd.Series(dtype=float)

        for col in ind_matrix.columns:
            # Compute average uniqueness if we add this column to drawn set
            subset = ind_matrix[drawn + [col]]
            # Concurrent count for this subset
            concurrent = subset.sum(axis=1)
            concurrent = concurrent[concurrent > 0]
            if len(concurrent) == 0:
                avg_u[col] = 0.0
                continue
            # Uniqueness of the new column
            col_active = ind_matrix[col]
            col_bars = col_active[col_active > 0].index
            col_bars = col_bars.intersection(concurrent.index)
            if len(col_bars) == 0:
                avg_u[col] = 0.0
                continue
            avg_u[col] = (1.0 / concurrent.loc[col_bars]).mean()

        # Draw proportional to uniqueness
        prob = avg_u / avg_u.sum()
        if prob.sum() == 0:
            # All zero — draw uniformly
            choice = np.random.choice(ind_matrix.columns)
        else:
            choice = np.random.choice(avg_u.index, p=prob.values)
        drawn.append(choice)

    return drawn
