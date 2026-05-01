"""Tests for live.position_tracker — reconciliation between local DB and exchange."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from live.config import LiveConfig
from live.db import LiveDB
from live.exchange import MockExchangeAdapter
from live.models import Position, PositionStatus, Side
from live.position_tracker import PositionTracker


@pytest_asyncio.fixture
async def tracker_setup(
    tmp_path: Path, sample_live_config: LiveConfig
) -> tuple[PositionTracker, LiveDB, MockExchangeAdapter]:
    """Create a PositionTracker with test dependencies."""
    db_path = tmp_path / "tracker_test.db"
    db = LiveDB(str(db_path))
    await db.init()

    exchange = MockExchangeAdapter()
    tracker = PositionTracker(db, exchange, sample_live_config.assets)

    yield tracker, db, exchange
    await db.close()


def _make_position(asset: str = "BTC", size_usd: float = 1000.0) -> Position:
    """Create a test position."""
    return Position(
        asset=asset,
        side=Side.LONG,
        entry_price=50000.0,
        entry_time=datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc),
        size_usd=size_usd,
        signal_strength=0.4,
        confidence=0.12,
        hold_bars=4,
        bars_held=0,
        status=PositionStatus.OPEN,
    )


@pytest.mark.asyncio
async def test_reconcile_match(
    tracker_setup: tuple[PositionTracker, LiveDB, MockExchangeAdapter],
) -> None:
    """No mismatches when local and exchange positions agree."""
    tracker, db, exchange = tracker_setup

    # Add a local position
    pos = _make_position("BTC", size_usd=1000.0)
    await db.open_position(pos)

    # Set matching exchange position
    exchange._mock_positions = {
        "BTC_USDT": {
            "side": "long",
            "size": 0.02,  # 1000 / 50000
            "entry_price": 50000.0,
            "unrealised_pnl": 0.0,
        }
    }

    result = await tracker.reconcile()
    assert result["local_count"] == 1
    assert result["exchange_count"] == 1
    # Small drift is expected due to rounding
    assert result["max_drift_pct"] < 0.05


@pytest.mark.asyncio
async def test_reconcile_mismatch_missing_on_exchange(
    tracker_setup: tuple[PositionTracker, LiveDB, MockExchangeAdapter],
) -> None:
    """Position exists locally but not on exchange -> mismatch."""
    tracker, db, exchange = tracker_setup

    pos = _make_position("BTC")
    await db.open_position(pos)

    # Exchange has no positions
    exchange._mock_positions = {}

    result = await tracker.reconcile()
    assert len(result["mismatches"]) >= 1
    assert result["mismatches"][0]["type"] == "missing_on_exchange"
    assert result["max_drift_pct"] == 1.0


@pytest.mark.asyncio
async def test_reconcile_mismatch_missing_locally(
    tracker_setup: tuple[PositionTracker, LiveDB, MockExchangeAdapter],
) -> None:
    """Position exists on exchange but not locally -> mismatch."""
    tracker, db, exchange = tracker_setup

    # No local positions, but exchange has one
    exchange._mock_positions = {
        "ETH_USDT": {
            "side": "long",
            "size": 1.0,
            "entry_price": 3000.0,
            "unrealised_pnl": 0.0,
        }
    }

    result = await tracker.reconcile()
    assert len(result["mismatches"]) >= 1
    missing_local = [m for m in result["mismatches"] if m["type"] == "missing_locally"]
    assert len(missing_local) == 1
    assert missing_local[0]["exchange_symbol"] == "ETH_USDT"


@pytest.mark.asyncio
async def test_reconcile_empty(
    tracker_setup: tuple[PositionTracker, LiveDB, MockExchangeAdapter],
) -> None:
    """No positions on either side -> clean reconciliation."""
    tracker, db, exchange = tracker_setup
    exchange._mock_positions = {}

    result = await tracker.reconcile()
    assert result["mismatches"] == []
    assert result["local_count"] == 0
    assert result["exchange_count"] == 0


@pytest.mark.asyncio
async def test_reconcile_exchange_unavailable(
    tracker_setup: tuple[PositionTracker, LiveDB, MockExchangeAdapter],
) -> None:
    """Exchange failure returns error result without crashing."""
    tracker, db, exchange = tracker_setup

    # Make get_open_positions raise an exception
    async def _raise(*args, **kwargs):
        raise ConnectionError("exchange down")
    exchange.get_open_positions = _raise

    result = await tracker.reconcile()
    assert "error" in result
    assert "exchange_unavailable" in result["error"]


@pytest.mark.asyncio
async def test_sync_on_startup_no_crash(
    tracker_setup: tuple[PositionTracker, LiveDB, MockExchangeAdapter],
) -> None:
    """sync_on_startup completes without error."""
    tracker, db, exchange = tracker_setup
    exchange._mock_positions = {}

    # Should not raise
    await tracker.sync_on_startup()


@pytest.mark.asyncio
async def test_sync_on_startup_with_mismatches(
    tracker_setup: tuple[PositionTracker, LiveDB, MockExchangeAdapter],
) -> None:
    """sync_on_startup logs warnings when mismatches exist."""
    tracker, db, exchange = tracker_setup

    pos = _make_position("BTC")
    await db.open_position(pos)
    exchange._mock_positions = {}

    # Should log warnings but not crash
    await tracker.sync_on_startup()
