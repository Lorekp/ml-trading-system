"""F3 Tier 6 tests: time_to_funding + time_to_funding_lag1 handlers.

Pure datetime arithmetic — no buffer state. Byte-identity to offline
``vol_model/features/engineer.py:417-419`` is exact (modulo float64
representation of ``hour + minute/60 + second/3600``).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deribit_paper.signal.feature_assembler import (
    _time_to_funding_seconds,
    _h_time_to_funding,
    _h_time_to_funding_lag1,
)


# ── _time_to_funding_seconds: byte-identity to offline formula ──


def _offline_reference(ts: pd.Timestamp) -> float:
    """Replay engineer.py:417-419 exactly."""
    h = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    return float(np.where(
        h < 8, 8.0 - h,
        np.where(h < 16, 16.0 - h, 24.0 - h),
    )) * 3600.0


@pytest.mark.parametrize("hour, minute", [
    (0, 0),     # midnight: 8h until first funding
    (4, 0),     # 4 hours into day
    (7, 59),    # 1 minute before 8am funding
    (8, 0),     # exactly at funding boundary
    (8, 1),     # 1 minute past 8am
    (12, 30),   # 3.5h until 16:00 funding
    (15, 59),
    (16, 0),
    (20, 0),
    (23, 59),
])
def test_time_to_funding_matches_offline_reference(hour, minute):
    ts = pd.Timestamp(f"2026-05-07 {hour:02d}:{minute:02d}", tz="UTC")
    assert _time_to_funding_seconds(ts) == pytest.approx(
        _offline_reference(ts), rel=1e-12,
    )


def test_time_to_funding_at_8am_is_zero_or_eight_hours():
    """Boundary: at exactly 08:00 UTC, h=8.0, so we fall into the second
    branch: 16-8 = 8 hours = 28800 seconds."""
    ts = pd.Timestamp("2026-05-07 08:00:00", tz="UTC")
    assert _time_to_funding_seconds(ts) == pytest.approx(8 * 3600)


def test_time_to_funding_just_before_8am_is_tiny():
    ts = pd.Timestamp("2026-05-07 07:59:59", tz="UTC")
    assert _time_to_funding_seconds(ts) == pytest.approx(1.0, abs=1e-3)


# ── handler dispatch ──


class _Bar:
    def __init__(self, ts):
        self.timestamp = ts


class _Ctx:
    def __init__(self, ts):
        self.bar = _Bar(ts)


def test_h_time_to_funding_uses_bar_timestamp():
    """At 12:00 UTC: 4h until 16:00 funding = 14400s."""
    ctx = _Ctx(pd.Timestamp("2026-05-07 12:00", tz="UTC"))
    assert _h_time_to_funding(ctx) == pytest.approx(4 * 3600)


def test_h_time_to_funding_lag1_uses_15min_earlier_timestamp():
    """At 12:00, lag1 (T-15min = 11:45) should give 4.25h until funding."""
    ctx = _Ctx(pd.Timestamp("2026-05-07 12:00", tz="UTC"))
    expected = _time_to_funding_seconds(
        pd.Timestamp("2026-05-07 11:45", tz="UTC")
    )
    assert _h_time_to_funding_lag1(ctx) == pytest.approx(expected)


def test_h_time_to_funding_lag1_crosses_funding_boundary():
    """At 08:10, lag1 (07:55) was 5min before the 08:00 funding."""
    ctx = _Ctx(pd.Timestamp("2026-05-07 08:10", tz="UTC"))
    # T-1 = 07:55 → 5 minutes until 08:00 funding = 300s.
    assert _h_time_to_funding_lag1(ctx) == pytest.approx(300.0, rel=1e-9)
    # At T = 08:10: in the [8,16) bucket, so 16-8.1666... = 7.8333h
    # = 28200 seconds
    assert _h_time_to_funding(ctx) == pytest.approx(
        (16.0 - (8 + 10 / 60)) * 3600, rel=1e-9,
    )
