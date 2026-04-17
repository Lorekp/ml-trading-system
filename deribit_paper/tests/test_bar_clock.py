"""Tests for BarClock — deterministic via injected now_fn + sleep_fn."""

from __future__ import annotations

import asyncio
from datetime import datetime, timedelta, timezone

import pandas as pd
import pytest

from deribit_paper.market_data.bar_clock import BarClock


# ─── Boundary math ───────────────────────────────────────────────────

def test_next_boundary_aligns_to_15min():
    clock = BarClock(bar_minutes=15)
    cases = [
        ("2026-04-10 12:03:45", "2026-04-10 12:15:00"),
        ("2026-04-10 12:14:59", "2026-04-10 12:15:00"),
        ("2026-04-10 12:15:00", "2026-04-10 12:30:00"),  # exactly on boundary → next
        ("2026-04-10 12:29:01", "2026-04-10 12:30:00"),
        ("2026-04-10 12:45:00", "2026-04-10 13:00:00"),
        ("2026-04-10 23:59:59", "2026-04-11 00:00:00"),
    ]
    for now_str, expected_str in cases:
        now = datetime.fromisoformat(now_str).replace(tzinfo=timezone.utc)
        expected = datetime.fromisoformat(expected_str).replace(tzinfo=timezone.utc)
        got = clock.next_boundary(after=now)
        assert got == expected, f"now={now_str} expected={expected_str} got={got}"


def test_next_boundary_5min_alignment():
    clock = BarClock(bar_minutes=5)
    now = datetime.fromisoformat("2026-04-10 12:07:00").replace(tzinfo=timezone.utc)
    got = clock.next_boundary(after=now)
    assert got == datetime.fromisoformat("2026-04-10 12:10:00").replace(tzinfo=timezone.utc)


def test_invalid_bar_minutes_raises():
    with pytest.raises(ValueError):
        BarClock(bar_minutes=0)
    with pytest.raises(ValueError):
        BarClock(bar_minutes=7)  # doesn't divide 60


# ─── Firing logic (deterministic simulator) ─────────────────────────

class _FakeClock:
    """Deterministic clock: now() and sleep() driven by a virtual time axis."""

    def __init__(self, start: datetime):
        self._now = start

    def now(self) -> datetime:
        return self._now

    async def sleep(self, seconds: float) -> None:
        self._now = self._now + timedelta(seconds=seconds)


@pytest.mark.asyncio
async def test_bar_clock_fires_callbacks_at_boundaries():
    start = datetime.fromisoformat("2026-04-10 12:12:30").replace(tzinfo=timezone.utc)
    fake = _FakeClock(start)
    clock = BarClock(bar_minutes=15, now_fn=fake.now, sleep_fn=fake.sleep)

    ticks: list[pd.Timestamp] = []

    async def on_tick(ts: pd.Timestamp) -> None:
        ticks.append(ts)
        if len(ticks) >= 3:
            clock.stop()

    clock.add_callback(on_tick)
    await clock.run_forever()

    assert len(ticks) == 3
    expected = [
        pd.Timestamp("2026-04-10 12:15:00", tz="UTC"),
        pd.Timestamp("2026-04-10 12:30:00", tz="UTC"),
        pd.Timestamp("2026-04-10 12:45:00", tz="UTC"),
    ]
    assert ticks == expected


@pytest.mark.asyncio
async def test_bar_clock_multiple_callbacks_all_fire():
    start = datetime.fromisoformat("2026-04-10 12:14:00").replace(tzinfo=timezone.utc)
    fake = _FakeClock(start)
    clock = BarClock(bar_minutes=15, now_fn=fake.now, sleep_fn=fake.sleep)

    a: list[pd.Timestamp] = []
    b: list[pd.Timestamp] = []

    async def cb_a(ts):
        a.append(ts)
    async def cb_b(ts):
        b.append(ts)
        if len(b) >= 2:
            clock.stop()

    clock.add_callback(cb_a)
    clock.add_callback(cb_b)
    await clock.run_forever()

    assert len(a) == 2
    assert len(b) == 2
    assert a == b


@pytest.mark.asyncio
async def test_bar_clock_callback_exception_does_not_kill_loop():
    start = datetime.fromisoformat("2026-04-10 12:14:00").replace(tzinfo=timezone.utc)
    fake = _FakeClock(start)
    clock = BarClock(bar_minutes=15, now_fn=fake.now, sleep_fn=fake.sleep)

    fires = [0]

    async def bad_cb(ts):
        fires[0] += 1
        if fires[0] < 2:
            raise RuntimeError("boom")
        if fires[0] >= 3:
            clock.stop()

    clock.add_callback(bad_cb)
    await clock.run_forever()
    assert fires[0] >= 3
