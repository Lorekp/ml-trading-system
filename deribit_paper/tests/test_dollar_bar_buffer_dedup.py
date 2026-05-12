"""Cycle-26 (2026-05-13) regression tests for DollarBarBuffer same-ms dedup.

Pre-cycle-26 ``DollarBarBuffer.as_index()`` returned timestamps verbatim.
When two dollar bars complete in the same millisecond (high-volume
periods), the returned ``DatetimeIndex`` had duplicates. Downstream
``TSRVDollarBarFeatures._ffill_to_dollar_bars`` then raised
``cannot reindex on an axis with duplicate labels`` and the affected
handlers (tsrv_mean_1d, up_jump_frac and all their lag/ema derivatives)
silently fell back to FROZEN baseline values. 7 of 77 LGBM input
features were stale on every live bar in the May 10-12 window
→ ~10vp ``rv_pred`` drift vs offline backtester.

Cycle-26 fix: apply monotonic nanosecond offsets to same-ms duplicates,
mirroring offline's ``vol_model/features/engineer.py:121-122`` ::

    dup_offsets = bars.groupby(bars.index).cumcount()
    bars.index = bars.index + pd.to_timedelta(dup_offsets, unit="ns")

These tests pin that behavior so a future ``as_index()`` rewrite that
drops the dedup fails CI.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.dollar_bar import DollarBar
from deribit_paper.features.dollar_bar_buffer import DollarBarBuffer


def _make_bar(ts: pd.Timestamp) -> DollarBar:
    """Synthetic bar with all required fields populated to plausible values."""
    return DollarBar(
        timestamp=ts,
        open=60000.0,
        close=60010.0,
        high=60020.0,
        low=59990.0,
        duration_s=10.5,
        n_trades=42,
        dollar_volume=500_000.0,
        buy_volume=260_000.0,
        sell_volume=240_000.0,
        vov=0.0003,
        imbalance=0.04,
        log_return=0.000167,
        rv_bar=2.78e-8,
    )


# ── No-duplicate cases (preserves legacy behavior) ────────────────────


def test_empty_buffer_returns_empty_index() -> None:
    buf = DollarBarBuffer(max_size=10)
    idx = buf.as_index()
    assert len(idx) == 0
    assert idx.tz is not None and str(idx.tz) == "UTC"


def test_single_bar_unchanged() -> None:
    buf = DollarBarBuffer(max_size=10)
    ts = pd.Timestamp("2026-05-12 12:00:00.123456", tz="UTC")
    buf.append(_make_bar(ts))
    idx = buf.as_index()
    assert len(idx) == 1
    assert idx[0] == ts


def test_distinct_ms_bars_unchanged() -> None:
    """When all bars have distinct ms-level timestamps, no offsets are
    applied — index is identical to the input."""
    buf = DollarBarBuffer(max_size=10)
    ts_list = [
        pd.Timestamp("2026-05-12 12:00:00.000", tz="UTC"),
        pd.Timestamp("2026-05-12 12:00:00.500", tz="UTC"),
        pd.Timestamp("2026-05-12 12:00:01.000", tz="UTC"),
        pd.Timestamp("2026-05-12 12:00:02.250", tz="UTC"),
        pd.Timestamp("2026-05-12 12:00:03.750", tz="UTC"),
    ]
    for ts in ts_list:
        buf.append(_make_bar(ts))
    idx = buf.as_index()
    assert len(idx) == 5
    for got, want in zip(idx, ts_list):
        assert got == want, f"distinct-ms case: got {got}, want {want}"


# ── Duplicate cases (the cycle-26 dedup logic) ────────────────────────


def test_two_bars_same_ms_offset_by_1ns() -> None:
    """Cycle-26: two bars completing in the same millisecond receive
    cumcount-based nanosecond offsets (0, 1)."""
    buf = DollarBarBuffer(max_size=10)
    ts = pd.Timestamp("2026-05-12 12:00:00.123", tz="UTC")
    buf.append(_make_bar(ts))
    buf.append(_make_bar(ts))
    idx = buf.as_index()
    assert len(idx) == 2
    assert not idx.has_duplicates, "as_index() must return unique entries"
    # Use Timedelta direct ns access — total_seconds() loses precision at 1ns
    delta_ns = (idx[1] - idx[0]).value  # value is in nanoseconds
    assert delta_ns == 1, (
        f"Expected 1ns offset between duplicates; got {delta_ns}ns"
    )


def test_three_bars_same_ms_offset_monotonic() -> None:
    """Cycle-26: three bars in same ms get offsets 0ns, 1ns, 2ns."""
    buf = DollarBarBuffer(max_size=10)
    ts = pd.Timestamp("2026-05-12 12:00:00.500", tz="UTC")
    for _ in range(3):
        buf.append(_make_bar(ts))
    idx = buf.as_index()
    assert len(idx) == 3
    assert not idx.has_duplicates
    assert (idx[1] - idx[0]) == pd.Timedelta(nanoseconds=1)
    assert (idx[2] - idx[1]) == pd.Timedelta(nanoseconds=1)


def test_mixed_distinct_and_duplicate() -> None:
    """Cycle-26: distinct timestamps stay unchanged; only duplicates
    within a single ms get incremented. Two duplicates at 12:00:00.500
    sandwiched between distinct values at 12:00:00.000 and 12:00:01.000."""
    buf = DollarBarBuffer(max_size=10)
    bars_ts = [
        pd.Timestamp("2026-05-12 12:00:00.000", tz="UTC"),
        pd.Timestamp("2026-05-12 12:00:00.500", tz="UTC"),
        pd.Timestamp("2026-05-12 12:00:00.500", tz="UTC"),
        pd.Timestamp("2026-05-12 12:00:01.000", tz="UTC"),
    ]
    for ts in bars_ts:
        buf.append(_make_bar(ts))
    idx = buf.as_index()
    assert len(idx) == 4
    assert not idx.has_duplicates
    # First and last must equal their inputs (no offset)
    assert idx[0] == bars_ts[0]
    assert idx[3] == bars_ts[3]
    # Two duplicates at index 1 and 2 get 0ns and 1ns offsets
    assert idx[1] == bars_ts[1]
    assert (idx[2] - bars_ts[2]) == pd.Timedelta(nanoseconds=1)


# ── End-to-end probe ──────────────────────────────────────────────────


def test_tsrv_dollar_bar_features_does_not_raise_with_duplicates() -> None:
    """Cycle-26 leakage probe: TSRVDollarBarFeatures.tsrv_mean_1d() and
    .up_jump_frac() previously raised ``cannot reindex on an axis with
    duplicate labels`` when the buffer had same-ms duplicates. Post-fix,
    they must return finite values without raising."""
    from deribit_paper.features.tsrv_dollar_bar_grid import TSRVDollarBarFeatures
    from deribit_paper.features.tsrv_history import TSRVHistoryBuffer, TSRVHistoryRow

    # Buffer with two bars in the same ms — historically the trigger
    buf = DollarBarBuffer(max_size=10)
    ts_dup = pd.Timestamp("2026-05-12 12:00:00.500", tz="UTC")
    buf.append(_make_bar(ts_dup))
    buf.append(_make_bar(ts_dup))
    # Plus a couple of older distinct ms bars to ensure rolling windows
    # have data without warmup edge cases.
    for i in range(5):
        ts = pd.Timestamp("2026-05-12 11:50:00", tz="UTC") + pd.Timedelta(seconds=i)
        buf.append(_make_bar(ts))

    # Minimal TSRVHistoryBuffer with a few rows so as_series() has data.
    tsrv_buf = TSRVHistoryBuffer(max_size=672)
    for i in range(10):
        ts = pd.Timestamp("2026-05-12 10:00:00", tz="UTC") + pd.Timedelta(minutes=15 * i)
        tsrv_buf._rows.append(TSRVHistoryRow(
            timestamp=ts,
            vol_ann=0.30 + i * 0.01,
            up_jump_fraction=0.05 + i * 0.001,
            down_jump_fraction=0.04 + i * 0.001,
            bpv=0.0001 * (i + 1),
            rv_slow=0.0001 * (i + 1),
            rv_fast=0.0002 * (i + 1),
            n_ticks=100 + i,
        ))

    # Constructor signature: (tsrv_buffer, db_buffer)
    feats = TSRVDollarBarFeatures(tsrv_buf, buf)
    # These two MUST NOT raise (cycle-26 bug surfaced exactly here)
    v1 = feats.tsrv_mean_1d()
    v2 = feats.up_jump_frac()
    # Either finite or None (warmup) is acceptable — what's NOT acceptable
    # is an exception. The whole point is no more handler failures.
    assert v1 is None or np.isfinite(v1), f"tsrv_mean_1d returned non-finite: {v1}"
    assert v2 is None or np.isfinite(v2), f"up_jump_frac returned non-finite: {v2}"
