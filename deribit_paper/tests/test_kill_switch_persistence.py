"""Tests for KillSwitch <-> SQLiteStore persistence (H1, 2026-05-06).

Why this exists
---------------

Before H1, the engine's KillSwitch was in-memory only. ``store.open()``
seeded a ``kill_switch_state`` row at engine boot but the engine never
called ``get_kill_switch_state`` to restore it, nor wrote back via
``set_kill_switch_state`` on a trip. State therefore disappeared across a
restart even though the SQLite schema already represented it.

These tests pin the contract added by H1:

1. ``KillSwitch.trip(reason)`` invokes the persist callback with
   ``(True, reason)``.
2. ``KillSwitch.reset(...)`` invokes the persist callback with
   ``(False, None)``.
3. A persist callback that raises does NOT mask the in-memory trip
   (alerting and persistence are best-effort; the trip itself is
   the source of truth).
4. End-to-end against a real ``SQLiteStore``: trip → close → reopen →
   read back → halted=True with the same reason.
5. End-to-end reset clears the persisted state.
"""
from __future__ import annotations

from pathlib import Path

import aiosqlite
import pytest

from deribit_paper.position.equity import EquityTracker
from deribit_paper.position.store import SQLiteStore
from deribit_paper.risk.kill_switch import KillSwitch


def _make_switch_with_persist(persist_calls: list[tuple[bool, str | None]]):
    equity = EquityTracker(initial_equity_usd=100_000.0)

    def _persist(halted: bool, reason: str | None) -> None:
        persist_calls.append((halted, reason))

    ks = KillSwitch(
        equity_tracker=equity,
        max_daily_drawdown_pct=2.0,
        max_consecutive_errors=3,
        network_outage_seconds=120.0,
        persist_callback=_persist,
    )
    return ks, equity


def test_trip_invokes_persist_callback_with_halted_true_and_reason():
    persist_calls: list[tuple[bool, str | None]] = []
    ks, _ = _make_switch_with_persist(persist_calls)

    ks.trip("test trip reason")

    assert ks.is_halted() is True
    assert persist_calls == [(True, "test trip reason")]


def test_reset_invokes_persist_callback_with_halted_false():
    persist_calls: list[tuple[bool, str | None]] = []
    ks, _ = _make_switch_with_persist(persist_calls)

    ks.trip("first trip")
    ks.reset(operator="test")

    assert ks.is_halted() is False
    assert persist_calls == [
        (True, "first trip"),
        (False, None),
    ]


def test_double_trip_does_not_double_persist():
    """``trip()`` returns early when already halted — must not re-fire
    the persist callback so we don't churn SQLite on every gate check."""
    persist_calls: list[tuple[bool, str | None]] = []
    ks, _ = _make_switch_with_persist(persist_calls)

    ks.trip("first")
    ks.trip("second")  # should be a no-op

    assert persist_calls == [(True, "first")]


def test_persist_callback_exception_does_not_mask_trip():
    """If the persist callback raises (DB locked, disk full, etc.), the
    in-memory trip MUST still take effect. Persistence is best-effort;
    halting trades is the source of truth."""

    def _persist_that_raises(halted: bool, reason: str | None) -> None:
        raise RuntimeError("simulated DB write failure")

    ks = KillSwitch(
        equity_tracker=EquityTracker(initial_equity_usd=100_000.0),
        persist_callback=_persist_that_raises,
    )

    # Must not raise — the callback failure is logged and swallowed.
    ks.trip("trip with bad persist")

    assert ks.is_halted() is True
    assert "trip with bad persist" in (ks.state.reason or "")


def test_no_persist_callback_is_silent():
    """KillSwitch without a persist_callback must work exactly like
    pre-H1 (no-op on persistence). Backward compat for unit tests."""
    ks = KillSwitch(
        equity_tracker=EquityTracker(initial_equity_usd=100_000.0),
        # no persist_callback
    )
    ks.trip("plain trip")
    ks.reset()
    assert ks.is_halted() is False  # reset worked


@pytest.mark.asyncio
async def test_end_to_end_trip_persists_to_sqlite_and_survives_reopen(tmp_path):
    """Real SQLiteStore round-trip: a tripped engine that crashes and
    restarts MUST come back halted with the original reason."""
    db_path = tmp_path / "ks_persist.db"

    # ── First lifecycle: trip and let the persist task run to completion ──
    store1 = SQLiteStore(db_path)
    await store1.open()

    persisted_results: list[bool] = []

    async def _persist_async(halted: bool, reason: str | None) -> None:
        await store1.set_kill_switch_state(halted, reason=reason)
        persisted_results.append(halted)

    # Sync wrapper that schedules the async write — matches the engine's
    # pattern in TradingEngine.initialize().
    import asyncio
    persist_tasks: list[asyncio.Task] = []

    def _persist_sync(halted: bool, reason: str | None) -> None:
        persist_tasks.append(
            asyncio.create_task(_persist_async(halted, reason))
        )

    ks = KillSwitch(
        equity_tracker=EquityTracker(initial_equity_usd=100_000.0),
        persist_callback=_persist_sync,
    )
    ks.trip("daily DD 2.5% exceeds 2.0%")

    # Drain the scheduled write
    await asyncio.gather(*persist_tasks)

    halted, reason, halted_at = await store1.get_kill_switch_state()
    assert halted is True
    assert reason == "daily DD 2.5% exceeds 2.0%"
    assert halted_at is not None

    await store1.close()

    # ── Second lifecycle: reopen the same DB; halt state must persist ──
    store2 = SQLiteStore(db_path)
    await store2.open()
    halted2, reason2, halted_at2 = await store2.get_kill_switch_state()
    assert halted2 is True
    assert reason2 == "daily DD 2.5% exceeds 2.0%"
    assert halted_at2 == halted_at  # exact-byte preservation
    await store2.close()


@pytest.mark.asyncio
async def test_end_to_end_reset_clears_sqlite_state(tmp_path):
    """``reset()`` must persist the cleared state so the next restart
    starts clean (the operator-clears-trip workflow)."""
    db_path = tmp_path / "ks_reset.db"

    store = SQLiteStore(db_path)
    await store.open()

    import asyncio
    persist_tasks: list[asyncio.Task] = []

    async def _persist_async(halted: bool, reason: str | None) -> None:
        await store.set_kill_switch_state(halted, reason=reason)

    def _persist_sync(halted: bool, reason: str | None) -> None:
        persist_tasks.append(
            asyncio.create_task(_persist_async(halted, reason))
        )

    ks = KillSwitch(
        equity_tracker=EquityTracker(initial_equity_usd=100_000.0),
        persist_callback=_persist_sync,
    )

    # Trip, drain, verify halted; then reset, drain, verify cleared.
    ks.trip("network outage 200s")
    await asyncio.gather(*persist_tasks)
    persist_tasks.clear()

    halted, _, _ = await store.get_kill_switch_state()
    assert halted is True

    ks.reset(operator="end_to_end_test")
    await asyncio.gather(*persist_tasks)

    halted, reason, halted_at = await store.get_kill_switch_state()
    assert halted is False
    assert reason is None
    assert halted_at is None

    await store.close()
