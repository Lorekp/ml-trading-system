"""S-1d (parity-r3 cycle 7 — 2026-05-08) — DollarBarBuffer extensions.

Locks the new ``timestamp_at(idx)`` and ``as_index()`` methods used by
the cycle-7 ``TSRVDollarBarFeatures`` ffill-onto-grid path and by the
revised ``time_to_funding_lag*`` handlers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.dollar_bar import DollarBar
from deribit_paper.features.dollar_bar_buffer import DollarBarBuffer


def _make_bar(ts: pd.Timestamp, close: float = 70_000.0) -> DollarBar:
    return DollarBar(
        timestamp=ts,
        open=close, close=close, high=close, low=close,
        duration_s=60.0, n_trades=10,
        dollar_volume=500_000.0, buy_volume=250_000.0, sell_volume=250_000.0,
        vov=0.0, imbalance=0.0,
        log_return=0.0, rv_bar=0.0,
    )


def _populate(n: int, *, max_size: int = 100) -> DollarBarBuffer:
    buf = DollarBarBuffer(max_size=max_size)
    base = pd.Timestamp("2026-04-30 00:00:00", tz="UTC")
    for i in range(n):
        ts = base + pd.Timedelta(seconds=10 * i)
        buf.append(_make_bar(ts))
    return buf


# ── timestamp_at ───────────────────────────────────────────────


def test_timestamp_at_negative_index():
    buf = _populate(5)
    base = pd.Timestamp("2026-04-30 00:00:00", tz="UTC")
    assert buf.timestamp_at(-1) == base + pd.Timedelta(seconds=40)
    assert buf.timestamp_at(-2) == base + pd.Timedelta(seconds=30)
    assert buf.timestamp_at(-5) == base


def test_timestamp_at_positive_index():
    buf = _populate(5)
    base = pd.Timestamp("2026-04-30 00:00:00", tz="UTC")
    assert buf.timestamp_at(0) == base
    assert buf.timestamp_at(4) == base + pd.Timedelta(seconds=40)


def test_timestamp_at_out_of_bounds_returns_none():
    buf = _populate(3)
    assert buf.timestamp_at(10) is None
    assert buf.timestamp_at(-10) is None
    assert buf.timestamp_at(3) is None


def test_timestamp_at_empty_buffer():
    buf = DollarBarBuffer(max_size=10)
    assert buf.timestamp_at(0) is None
    assert buf.timestamp_at(-1) is None


# ── as_index ───────────────────────────────────────────────────


def test_as_index_returns_utc_datetime_index():
    buf = _populate(3)
    idx = buf.as_index()
    assert isinstance(idx, pd.DatetimeIndex)
    assert idx.tz is not None
    assert str(idx.tz) == "UTC"
    assert len(idx) == 3


def test_as_index_empty_buffer_returns_empty_utc_index():
    buf = DollarBarBuffer(max_size=10)
    idx = buf.as_index()
    assert isinstance(idx, pd.DatetimeIndex)
    assert len(idx) == 0
    assert str(idx.tz) == "UTC"


def test_as_index_matches_as_dataframe_index():
    buf = _populate(5)
    df_idx = buf.as_dataframe().index
    fast_idx = buf.as_index()
    pd.testing.assert_index_equal(df_idx, fast_idx)


# ── 60000-bar cap support ──────────────────────────────────────


def test_buffer_supports_60000_max_size():
    """S-1d Phase 1c: the cycle-7 default flips to 60000."""
    buf = DollarBarBuffer(max_size=60000)
    assert buf.max_size == 60000


def test_buffer_evicts_at_60000_bars():
    """A buffer with max_size=100 evicting at 200 appends — sanity
    that the underlying ``deque(maxlen=...)`` honours the cap. We use
    100 to keep the test fast (60000 has the same semantic)."""
    buf = _populate(200, max_size=100)
    assert len(buf) == 100
    base = pd.Timestamp("2026-04-30 00:00:00", tz="UTC")
    assert buf.timestamp_at(0) == base + pd.Timedelta(seconds=10 * 100)
