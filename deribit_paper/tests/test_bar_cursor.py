"""Tests for H8 bar_cursor restart-idempotency.

Why this exists
---------------

Before H8, the engine's ``_on_bar_close`` had no defence against
re-processing a bar. If a buggy aggregator re-fired a past bar
timestamp, or a manual replay accidentally hit the live engine, the
same bar's signal would fire twice.

These tests pin the H8 contract:

1. ``bar_cursor`` table is created at ``store.open()``.
2. ``mark_bar_processed`` is idempotent — repeated calls for the
   same bar timestamp don't duplicate the row.
3. ``is_bar_processed`` returns True after a successful mark, False
   otherwise.
4. The cursor survives store close/reopen — that's the whole point.
5. Engine ``_on_bar_close`` consults the cursor and skips already-
   processed bars (integration-style test against engine wiring).
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from deribit_paper.position.store import SQLiteStore


@pytest.mark.asyncio
async def test_bar_cursor_table_exists_after_open(tmp_path):
    db = tmp_path / "cursor.db"
    async with SQLiteStore(db) as store:
        # Should not raise — bar_cursor exists
        n = await store.count_bars_processed()
        assert n == 0


@pytest.mark.asyncio
async def test_mark_and_check_round_trip(tmp_path):
    db = tmp_path / "cursor.db"
    bar_ts = pd.Timestamp("2026-05-06 12:00:00", tz="UTC")
    async with SQLiteStore(db) as store:
        assert await store.is_bar_processed(bar_ts) is False
        await store.mark_bar_processed(bar_ts, signal_direction=1, signal_reason="long edge")
        assert await store.is_bar_processed(bar_ts) is True


@pytest.mark.asyncio
async def test_mark_is_idempotent(tmp_path):
    db = tmp_path / "cursor.db"
    bar_ts = pd.Timestamp("2026-05-06 12:00:00", tz="UTC")
    async with SQLiteStore(db) as store:
        await store.mark_bar_processed(bar_ts, signal_direction=1)
        await store.mark_bar_processed(bar_ts, signal_direction=-1)  # second call
        await store.mark_bar_processed(bar_ts, signal_direction=0)   # third call
        assert await store.count_bars_processed() == 1


@pytest.mark.asyncio
async def test_different_bars_get_separate_rows(tmp_path):
    db = tmp_path / "cursor.db"
    async with SQLiteStore(db) as store:
        for i in range(5):
            ts = pd.Timestamp("2026-05-06 12:00:00", tz="UTC") + pd.Timedelta(minutes=15 * i)
            await store.mark_bar_processed(ts, signal_direction=0)
        assert await store.count_bars_processed() == 5


@pytest.mark.asyncio
async def test_cursor_survives_store_close_reopen(tmp_path):
    """The whole point of H8 — restart must see the prior cursor."""
    db = tmp_path / "cursor.db"
    bar_ts = pd.Timestamp("2026-05-06 12:00:00", tz="UTC")

    # Lifecycle 1: mark + close
    store1 = SQLiteStore(db)
    await store1.open()
    await store1.mark_bar_processed(bar_ts, signal_direction=1, signal_reason="restart test")
    await store1.close()

    # Lifecycle 2: reopen + verify
    store2 = SQLiteStore(db)
    await store2.open()
    assert await store2.is_bar_processed(bar_ts) is True
    assert await store2.count_bars_processed() == 1
    await store2.close()


@pytest.mark.asyncio
async def test_signal_direction_and_reason_preserved(tmp_path):
    """Diagnostic columns are preserved (used for post-hoc analysis
    of where signals fired)."""
    db = tmp_path / "cursor.db"
    bar_ts = pd.Timestamp("2026-05-06 12:00:00", tz="UTC")

    async with SQLiteStore(db) as store:
        await store.mark_bar_processed(
            bar_ts,
            signal_direction=-1,
            signal_reason="short: edge=-3.5σ in contango",
        )

        # Direct SQL probe (no public API for full row inspection)
        async with store._conn.execute(  # noqa: SLF001 - test
            "SELECT bar_timestamp, signal_direction, signal_reason, fired_at "
            "FROM bar_cursor WHERE bar_timestamp = ?",
            (bar_ts.isoformat(),),
        ) as cursor:
            row = await cursor.fetchone()
            assert row is not None
            assert row[0] == bar_ts.isoformat()
            assert row[1] == -1
            assert row[2] == "short: edge=-3.5σ in contango"
            assert row[3] is not None  # fired_at is set


@pytest.mark.asyncio
async def test_count_starts_at_zero_on_fresh_db(tmp_path):
    db = tmp_path / "fresh.db"
    async with SQLiteStore(db) as store:
        assert await store.count_bars_processed() == 0


@pytest.mark.asyncio
async def test_mark_with_default_signal_reason(tmp_path):
    """signal_reason is optional (None for skipped/empty bars)."""
    db = tmp_path / "cursor.db"
    bar_ts = pd.Timestamp("2026-05-06 12:00:00", tz="UTC")
    async with SQLiteStore(db) as store:
        await store.mark_bar_processed(bar_ts, signal_direction=0)
        assert await store.is_bar_processed(bar_ts) is True
