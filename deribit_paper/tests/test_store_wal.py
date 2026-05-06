"""Tests for H14 SQLite WAL pragmas (2026-05-06).

Why this exists
---------------

Default SQLite journal mode (DELETE) raises ``database is locked`` on
concurrent writes. The live engine has 4+ asyncio tasks hitting the
store concurrently (trade close, kill-switch persist, equity snapshot,
profit-scale persist, bar cursor). WAL mode allows readers to coexist
with writers; busy_timeout makes brief contention block instead of
raising.

Tests pin the H14 contract:

1. ``open()`` sets journal_mode=WAL.
2. ``open()`` sets busy_timeout to 5000ms.
3. ``open()`` sets synchronous=NORMAL (safe given our cursor model).
4. Concurrent writes from multiple tasks don't raise.
"""
from __future__ import annotations

import asyncio

import aiosqlite
import pandas as pd
import pytest

from deribit_paper.position.store import SQLiteStore


@pytest.mark.asyncio
async def test_journal_mode_is_wal_after_open(tmp_path):
    db = tmp_path / "wal.db"
    async with SQLiteStore(db) as store:
        async with store._conn.execute("PRAGMA journal_mode") as cur:  # noqa: SLF001
            row = await cur.fetchone()
            assert row[0].lower() == "wal"


@pytest.mark.asyncio
async def test_busy_timeout_is_set(tmp_path):
    db = tmp_path / "wal.db"
    async with SQLiteStore(db) as store:
        async with store._conn.execute("PRAGMA busy_timeout") as cur:  # noqa: SLF001
            row = await cur.fetchone()
            assert int(row[0]) == 5000


@pytest.mark.asyncio
async def test_synchronous_is_normal(tmp_path):
    db = tmp_path / "wal.db"
    async with SQLiteStore(db) as store:
        async with store._conn.execute("PRAGMA synchronous") as cur:  # noqa: SLF001
            row = await cur.fetchone()
            # 1 = NORMAL, 2 = FULL
            assert int(row[0]) == 1


@pytest.mark.asyncio
async def test_concurrent_writes_do_not_raise(tmp_path):
    """20 concurrent writes from multiple tasks. Pre-H14 this would
    intermittently raise ``database is locked``."""
    db = tmp_path / "concurrent.db"
    async with SQLiteStore(db) as store:
        async def write_kill_switch():
            for i in range(5):
                await store.set_kill_switch_state(
                    halted=(i % 2 == 0), reason=f"r{i}",
                )

        async def write_bar_cursor():
            base = pd.Timestamp("2026-05-06 12:00", tz="UTC")
            for i in range(5):
                await store.mark_bar_processed(
                    base + pd.Timedelta(minutes=15 * i),
                    signal_direction=i % 3 - 1,
                )

        async def write_profit_scale():
            for i in range(5):
                await store.save_profit_scale_state({
                    "gross_per_btc_ema_usd": 1000.0 + i,
                    "gross_count": i,
                    "hedge_per_btc_ema_usd": 20.0 + i,
                })

        async def read_kill_switch():
            for _ in range(5):
                await store.get_kill_switch_state()

        # Run them all concurrently — must not raise
        await asyncio.gather(
            write_kill_switch(),
            write_bar_cursor(),
            write_profit_scale(),
            read_kill_switch(),
        )

        # Sanity: final state is whichever ran last; just verify
        # the DB is still queryable.
        halted, _, _ = await store.get_kill_switch_state()
        assert isinstance(halted, bool)
        assert await store.count_bars_processed() == 5
