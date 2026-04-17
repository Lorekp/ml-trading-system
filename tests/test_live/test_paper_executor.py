"""Tests for live.paper_executor — position management and risk limits."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from live.config import LiveConfig
from live.db import LiveDB
from live.models import Position, PositionStatus, Side, Signal
from live.paper_executor import PaperExecutor


@pytest_asyncio.fixture
async def executor_and_db(
    tmp_path: Path, sample_live_config: LiveConfig
) -> tuple[PaperExecutor, LiveDB]:
    """Create an executor with a temp database."""
    db_path = tmp_path / "exec_test.db"
    db = LiveDB(str(db_path))
    await db.init()
    executor = PaperExecutor(sample_live_config, db)
    yield executor, db
    await db.close()


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


@pytest.mark.asyncio
async def test_open_position_on_signal(
    executor_and_db: tuple[PaperExecutor, LiveDB],
) -> None:
    """A valid signal opens a paper position."""
    executor, db = executor_and_db
    signal = _make_signal()

    trade = await executor.process_signal_with_price(signal, 50000.0)
    assert trade is not None
    assert trade.action == "open"
    assert trade.asset == "BTC"
    assert trade.side == Side.LONG

    positions = await db.get_open_positions()
    assert len(positions) == 1
    assert positions[0].asset == "BTC"


@pytest.mark.asyncio
async def test_exit_on_hold_expiry(
    executor_and_db: tuple[PaperExecutor, LiveDB],
) -> None:
    """Position is closed when bars_held reaches hold_bars."""
    executor, db = executor_and_db
    signal = _make_signal()
    await executor.process_signal_with_price(signal, 50000.0)

    # Simulate 4 bars passing (hold_bars=4)
    for i in range(4):
        exits = await executor.check_exits({"BTC": 50100.0})
        if i < 3:
            assert len(exits) == 0
        else:
            # On the 4th call, bars_held=4 >= hold_bars=4
            assert len(exits) == 1
            assert exits[0].action == "close"

    # Position should be closed now
    positions = await db.get_open_positions()
    assert len(positions) == 0


@pytest.mark.asyncio
async def test_slippage_applied(
    executor_and_db: tuple[PaperExecutor, LiveDB],
) -> None:
    """Slippage is applied on both entry and exit."""
    executor, db = executor_and_db
    signal = _make_signal(direction=1.0)  # LONG

    trade = await executor.process_signal_with_price(signal, 50000.0)
    assert trade is not None
    # For a LONG: fill_price = 50000 * (1 + 1/10000) = 50005.0
    assert trade.price == pytest.approx(50005.0, rel=1e-6)


@pytest.mark.asyncio
async def test_max_positions_enforced(
    executor_and_db: tuple[PaperExecutor, LiveDB],
    sample_live_config: LiveConfig,
) -> None:
    """No more than max_positions can be opened."""
    executor, db = executor_and_db

    # Open max_positions (6) across different assets
    assets = ["BTC", "ETH", "XRP", "BNB", "SUI", "LINK"]
    for asset in assets:
        sig = _make_signal(asset=asset, signal_strength=0.3)
        result = await executor.process_signal_with_price(sig, 1000.0)
        assert result is not None, f"Failed to open position for {asset}"

    # 7th should be rejected
    extra = _make_signal(asset="SOL", signal_strength=0.3)
    result = await executor.process_signal_with_price(extra, 100.0)
    assert result is None


@pytest.mark.asyncio
async def test_per_asset_limit_enforced(
    executor_and_db: tuple[PaperExecutor, LiveDB],
) -> None:
    """Per-asset exposure cap prevents over-concentration."""
    executor, db = executor_and_db

    # First trade: should succeed
    sig1 = _make_signal(asset="BTC", signal_strength=2.0)
    trade1 = await executor.process_signal_with_price(sig1, 50000.0)
    assert trade1 is not None

    # Second trade for same asset: per_asset_pct=0.06 * 10000 = $600 cap
    # First trade already took some of that cap
    sig2 = _make_signal(asset="BTC", signal_strength=2.0)
    trade2 = await executor.process_signal_with_price(sig2, 50000.0)
    # Either rejected or capped
    if trade2 is not None:
        total_exposure = await executor.get_asset_exposure("BTC")
        equity = await executor.get_equity()
        assert total_exposure <= equity * 0.06 + 1.0  # small tolerance


@pytest.mark.asyncio
async def test_pnl_computation(
    executor_and_db: tuple[PaperExecutor, LiveDB],
) -> None:
    """PnL is correctly computed for a long position."""
    executor, db = executor_and_db
    signal = _make_signal(direction=1.0, signal_strength=0.4)

    await executor.process_signal_with_price(signal, 50000.0)

    # Price goes up 1%
    exit_price = 50500.0
    # Advance bars to trigger exit
    for _ in range(4):
        exits = await executor.check_exits({"BTC": exit_price})

    assert len(exits) == 1
    # Position was LONG, price went up -> positive PnL
    # PnL ~ size * (exit - entry) / entry
    # Slippage affects both entry and exit
    positions_cursor = await db._conn.execute(
        "SELECT pnl FROM positions WHERE status = 'closed'"
    )
    row = await positions_cursor.fetchone()
    pnl = row[0]
    # Should be positive (price increased for a long position)
    assert pnl > 0
