"""Tests for live.db — SQLite state persistence."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from live.db import LiveDB
from live.models import Position, PositionStatus, Side, Signal, Trade


@pytest_asyncio.fixture
async def db(tmp_path: Path) -> LiveDB:
    """Create an ephemeral LiveDB for testing."""
    db_path = tmp_path / "test.db"
    db = LiveDB(str(db_path))
    await db.init()
    yield db
    await db.close()


@pytest.mark.asyncio
async def test_create_tables(tmp_path: Path) -> None:
    """Tables are created on init and WAL mode is enabled."""
    db_path = tmp_path / "tables_test.db"
    db = LiveDB(str(db_path))
    await db.init()

    assert db._conn is not None
    # Check WAL mode
    cursor = await db._conn.execute("PRAGMA journal_mode")
    row = await cursor.fetchone()
    assert row[0] == "wal"

    # Check tables exist
    cursor = await db._conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    )
    tables = [row[0] for row in await cursor.fetchall()]
    assert "positions" in tables
    assert "trades" in tables
    assert "signals_log" in tables
    assert "checkpoints" in tables
    assert "daily_equity" in tables

    await db.close()


@pytest.mark.asyncio
async def test_position_lifecycle(db: LiveDB) -> None:
    """Open -> close -> verify position lifecycle."""
    now = datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc)

    # Open position
    pos = Position(
        asset="BTC",
        side=Side.LONG,
        entry_price=50000.0,
        entry_time=now,
        size_usd=200.0,
        signal_strength=0.4,
        confidence=0.12,
        hold_bars=4,
        bars_held=0,
    )
    pos_id = await db.open_position(pos)
    assert pos_id > 0

    # Verify open positions
    open_positions = await db.get_open_positions()
    assert len(open_positions) == 1
    assert open_positions[0].asset == "BTC"
    assert open_positions[0].side == Side.LONG

    # Close position
    exit_time = datetime(2025, 6, 15, 13, 0, tzinfo=timezone.utc)
    await db.close_position(pos_id, 50500.0, exit_time, 2.0)

    # Verify no open positions
    open_positions = await db.get_open_positions()
    assert len(open_positions) == 0


@pytest.mark.asyncio
async def test_trade_log(db: LiveDB) -> None:
    """Log a trade and verify it was stored."""
    trade = Trade(
        id="trade_001",
        timestamp=datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc),
        asset="ETH",
        side=Side.SHORT,
        action="open",
        price=3000.0,
        size_usd=150.0,
        position_id=1,
        reason="signal_strength=0.3",
    )
    trade_id = await db.log_trade(trade)
    assert trade_id == "trade_001"

    # Verify stored
    cursor = await db._conn.execute("SELECT * FROM trades WHERE id = ?", ("trade_001",))
    row = await cursor.fetchone()
    assert row is not None
    assert row[2] == "ETH"  # asset column


@pytest.mark.asyncio
async def test_signal_log(db: LiveDB) -> None:
    """Log a signal and verify storage."""
    signal = Signal(
        timestamp=datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc),
        asset="BTC",
        direction=1.0,
        probability=0.62,
        confidence=0.12,
        signal_strength=0.4,
        regime_aligned=True,
        passed_confidence=True,
    )
    row_id = await db.log_signal(signal, executed=True, reject_reason=None)
    assert row_id > 0

    # Verify
    cursor = await db._conn.execute("SELECT * FROM signals_log WHERE id = ?", (row_id,))
    row = await cursor.fetchone()
    assert row is not None
    assert row[2] == "BTC"  # asset


@pytest.mark.asyncio
async def test_checkpoint_round_trip(db: LiveDB) -> None:
    """Save and load a checkpoint."""
    await db.save_checkpoint("last_bar", "2025-06-15T12:00:00")
    val = await db.load_checkpoint("last_bar")
    assert val == "2025-06-15T12:00:00"

    # Overwrite
    await db.save_checkpoint("last_bar", "2025-06-15T12:15:00")
    val = await db.load_checkpoint("last_bar")
    assert val == "2025-06-15T12:15:00"

    # Non-existent key
    val = await db.load_checkpoint("missing_key")
    assert val is None


@pytest.mark.asyncio
async def test_daily_equity(db: LiveDB) -> None:
    """Save and retrieve daily equity snapshots."""
    await db.save_daily_equity("2025-06-14", 10050.0, 50.0, 5, 12)
    await db.save_daily_equity("2025-06-15", 10120.0, 70.0, 3, 8)

    history = await db.get_equity_history()
    assert len(history) == 2
    assert history[0][0] == "2025-06-14"
    assert history[0][1] == 10050.0
    assert history[1][0] == "2025-06-15"
    assert history[1][2] == 70.0  # pnl

    # Upsert overwrites
    await db.save_daily_equity("2025-06-15", 10100.0, 50.0, 4, 10)
    history = await db.get_equity_history()
    assert len(history) == 2
    assert history[1][1] == 10100.0  # updated equity
