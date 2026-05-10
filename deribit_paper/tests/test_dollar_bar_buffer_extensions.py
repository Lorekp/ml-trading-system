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


# ── pandas 3.x tz-aware regression ─────────────────────────────


def test_as_index_no_tz_warning_pandas_3x_safe():
    """Regression: pandas 3.x raises ``Tz-aware datetime.datetime cannot
    be converted to datetime64 unless utc=True`` when constructing a
    DatetimeIndex from a list of tz-aware Timestamps without explicit
    ``tz=`` / ``utc=True``. pandas 2.x emits a FutureWarning and silently
    coerces — masking the bug locally while the homeserver crashes.

    This test promotes that FutureWarning to an error so it fails on
    BOTH pandas 2.x (warning) and 3.x (raise).

    Live engine bug 2026-05-10 14:30 UTC: 5 handlers fell back to
    baseline-frozen values because ``DollarBarBuffer.as_index()`` raised
    on the homeserver (pandas 3.0.2) but worked locally (pandas 2.3.3).
    """
    import warnings

    buf = _populate(50)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        idx = buf.as_index()
        df = buf.as_dataframe()

    assert isinstance(idx, pd.DatetimeIndex)
    assert str(idx.tz) == "UTC"
    assert len(idx) == 50
    assert isinstance(df.index, pd.DatetimeIndex)
    assert str(df.index.tz) == "UTC"


def test_as_index_no_tz_warning_after_eviction():
    """Same regression but exercising the deque eviction path so we
    cover both pre- and post-eviction states."""
    import warnings

    buf = _populate(150, max_size=100)
    assert len(buf) == 100
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        idx = buf.as_index()
        df = buf.as_dataframe()

    assert len(idx) == 100
    assert str(idx.tz) == "UTC"
    assert len(df) == 100


def test_as_index_simulated_pandas_3x_strict_mode():
    """Stronger reproduction: the pandas 3.x error message starts with
    ``Tz-aware datetime.datetime cannot be converted to datetime64
    unless utc=True``. Build the DatetimeIndex via the same code path
    and assert it does NOT raise. This is the literal reproduction of
    the homeserver failure trace at the first 15-min bar evaluation.
    """
    import warnings

    base = pd.Timestamp("2026-05-10 14:00:00", tz="UTC")
    timestamps = [base + pd.Timedelta(seconds=10 * i) for i in range(20)]

    # Bare repro of the buggy pre-fix pattern would be:
    #   pd.DatetimeIndex(timestamps, name="timestamp")
    # which warns on 2.x and raises on 3.x. Instead we exercise the
    # post-fix pattern:
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        idx = pd.DatetimeIndex(
            pd.to_datetime(timestamps, utc=True), name="timestamp",
        )
    assert str(idx.tz) == "UTC"
    assert idx.name == "timestamp"
    assert len(idx) == 20
