"""Tests for live.executor — live order execution with mock exchange."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from live.config import LiveConfig
from live.db import LiveDB
from live.exchange import MockExchangeAdapter
from live.executor import LiveExecutor
from live.kill_switch import KillSwitch
from live.models import Position, PositionStatus, Side, Signal
from live.position_tracker import PositionTracker
from live.risk_manager import RiskManager


def _make_signal(
    asset: str = "BTC",
    direction: float = 1.0,
    probability: float = 0.62,
    signal_strength: float = 0.4,
) -> Signal:
    """Helper to create a test signal."""
    return Signal(
        timestamp=datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc),
        asset=asset,
        direction=direction,
        probability=probability,
        confidence=abs(probability - 0.5),
        signal_strength=signal_strength,
        regime_aligned=True,
        passed_confidence=True,
    )


@pytest_asyncio.fixture
async def live_executor_setup(
    tmp_path: Path, sample_live_config: LiveConfig
) -> tuple[LiveExecutor, LiveDB, MockExchangeAdapter, RiskManager]:
    """Create a LiveExecutor with mock exchange and temp database."""
    db_path = tmp_path / "live_exec_test.db"
    db = LiveDB(str(db_path))
    await db.init()

    exchange = MockExchangeAdapter()
    exchange._mock_balance = 10000.0

    kill_switch = KillSwitch(str(tmp_path / "KILL"))
    position_tracker = PositionTracker(db, exchange, sample_live_config.assets)
    risk_manager = RiskManager(
        sample_live_config, db, exchange, kill_switch, position_tracker
    )

    executor = LiveExecutor(
        sample_live_config, db, exchange, risk_manager
    )

    yield executor, db, exchange, risk_manager
    await db.close()


@pytest.mark.asyncio
async def test_place_entry_order(
    live_executor_setup: tuple[LiveExecutor, LiveDB, MockExchangeAdapter, RiskManager],
) -> None:
    """A valid signal places a limit order and creates a position."""
    executor, db, exchange, _ = live_executor_setup
    signal = _make_signal()

    trade = await executor.process_signal_with_price(signal, 50000.0)
    assert trade is not None
    assert trade.action == "open"
    assert trade.asset == "BTC"
    assert trade.side == Side.LONG
    assert trade.price > 0

    # Position should exist in DB
    positions = await db.get_open_positions()
    assert len(positions) == 1
    assert positions[0].asset == "BTC"
    assert positions[0].side == Side.LONG


@pytest.mark.asyncio
async def test_short_entry_order(
    live_executor_setup: tuple[LiveExecutor, LiveDB, MockExchangeAdapter, RiskManager],
) -> None:
    """A short signal places a short order."""
    executor, db, exchange, _ = live_executor_setup
    signal = _make_signal(direction=-1.0)

    trade = await executor.process_signal_with_price(signal, 50000.0)
    assert trade is not None
    assert trade.side == Side.SHORT

    positions = await db.get_open_positions()
    assert len(positions) == 1
    assert positions[0].side == Side.SHORT


@pytest.mark.asyncio
async def test_stop_loss_placed_after_fill(
    live_executor_setup: tuple[LiveExecutor, LiveDB, MockExchangeAdapter, RiskManager],
) -> None:
    """After entry fill, a stop-loss order is placed."""
    executor, db, exchange, _ = live_executor_setup
    signal = _make_signal()

    trade = await executor.process_signal_with_price(signal, 50000.0)
    assert trade is not None

    # Mock exchange should have at least 2 orders:
    # 1 entry + 1 stop-loss
    assert hasattr(exchange, "_mock_orders")
    assert len(exchange._mock_orders) >= 2


@pytest.mark.asyncio
async def test_exit_at_hold_bars(
    live_executor_setup: tuple[LiveExecutor, LiveDB, MockExchangeAdapter, RiskManager],
) -> None:
    """Position is closed when bars_held reaches hold_bars."""
    executor, db, exchange, _ = live_executor_setup
    signal = _make_signal()
    await executor.process_signal_with_price(signal, 50000.0)

    # Simulate 4 bars passing (hold_bars=4)
    for i in range(4):
        exits = await executor.check_exits({"BTC": 50100.0})
        if i < 3:
            assert len(exits) == 0
        else:
            assert len(exits) == 1
            assert exits[0].action == "close"

    # Position should be closed now
    positions = await db.get_open_positions()
    assert len(positions) == 0


@pytest.mark.asyncio
async def test_close_all_positions(
    live_executor_setup: tuple[LiveExecutor, LiveDB, MockExchangeAdapter, RiskManager],
) -> None:
    """close_all_positions closes every open position."""
    executor, db, exchange, _ = live_executor_setup

    # Open 2 positions
    for asset in ["BTC", "ETH"]:
        sig = _make_signal(asset=asset, signal_strength=0.3)
        await executor.process_signal_with_price(sig, 1000.0)

    positions = await db.get_open_positions()
    assert len(positions) == 2

    # Emergency close
    exits = await executor.close_all_positions(reason="test_kill")
    assert len(exits) == 2
    for trade in exits:
        assert trade.action == "close"
        assert "emergency_close" in trade.reason

    # All positions should be closed
    positions = await db.get_open_positions()
    assert len(positions) == 0


@pytest.mark.asyncio
async def test_risk_manager_blocks_trade(
    live_executor_setup: tuple[LiveExecutor, LiveDB, MockExchangeAdapter, RiskManager],
) -> None:
    """Risk manager rejection prevents trade execution."""
    executor, db, exchange, risk_manager = live_executor_setup

    # Trigger kill switch to block trading
    kill_path = Path(risk_manager._kill_switch._path)
    kill_path.write_text("test block")

    signal = _make_signal()
    trade = await executor.process_signal_with_price(signal, 50000.0)
    assert trade is None

    # Clean up
    kill_path.unlink()


@pytest.mark.asyncio
async def test_get_equity(
    live_executor_setup: tuple[LiveExecutor, LiveDB, MockExchangeAdapter, RiskManager],
) -> None:
    """get_equity returns exchange balance when available."""
    executor, db, exchange, _ = live_executor_setup
    exchange._mock_balance = 15000.0

    equity = await executor.get_equity()
    assert equity == 15000.0


@pytest.mark.asyncio
async def test_get_equity_fallback(
    live_executor_setup: tuple[LiveExecutor, LiveDB, MockExchangeAdapter, RiskManager],
) -> None:
    """get_equity falls back to paper equity when exchange unavailable."""
    executor, db, exchange, _ = live_executor_setup
    exchange._mock_balance = 0.0  # Simulate unavailable

    equity = await executor.get_equity()
    # Should fall back to paper starting capital
    assert equity == 10000.0


@pytest.mark.asyncio
async def test_size_too_small_skipped(
    live_executor_setup: tuple[LiveExecutor, LiveDB, MockExchangeAdapter, RiskManager],
) -> None:
    """Very weak signals with tiny size are skipped."""
    executor, db, exchange, _ = live_executor_setup
    # Very small signal strength
    signal = _make_signal(signal_strength=0.001)

    trade = await executor.process_signal_with_price(signal, 50000.0)
    assert trade is None
