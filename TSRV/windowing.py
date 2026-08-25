"""Shared timestamp windowing for realized-volatility pipelines."""

from __future__ import annotations

import numpy as np
import pandas as pd


def timestamp_window_ids(
    index: pd.DatetimeIndex,
    freq: pd.Timedelta,
) -> tuple[np.ndarray, int]:
    """Return integer window IDs and the window width in nanoseconds."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("index must be a pandas DatetimeIndex")

    freq_ns = int(pd.Timedelta(freq).value)
    if freq_ns <= 0:
        raise ValueError("freq must be positive")

    # pandas 3 may preserve a microsecond-backed DatetimeIndex. Normalize the
    # unit before integer conversion because ``Timedelta.value`` is always ns.
    timestamps_ns = index.to_numpy(dtype="datetime64[ns]")
    ts_ns = timestamps_ns.astype(np.int64, copy=False)
    return ts_ns // freq_ns, freq_ns
