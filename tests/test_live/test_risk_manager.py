"""Tests for live.risk_manager — circuit breakers and risk checks."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
import pytest_asyncio

from live.config import LiveConfig
from live.db import LiveDB
from live.exchange import MockExchangeAdapter
from live.kill_switch import KillSwitch
from live.models import Signal
from live.position_tracker import PositionTracker
from live.risk_manager import RiskManager


def _make_signal(
    asset: str = "BTC",
    direction: float = 1.0,
    signal_strength: float = 0.4,
) -> Signal:
    """Helper to create a test signal."""
    return Signal(
        timestamp=datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc),
        asset=asset,
        direction=direction,
        probability=0.62,
        confidence=0.12,
        signal_strength=signal_strength,
        regime_aligned=True,
        passed_confidence=True,
    )


@pytest_asyncio.fixture
async def risk_setup(
    tmp_path: Path, sample_live_config: LiveConfig
) -> tuple[RiskManager, LiveDB, MockExchangeAdapter, KillSwitch]:
    """Create a RiskManager with test dependencies."""
    db_path = tmp_path / "risk_test.db"
    db = LiveDB(str(db_path))
    await db.init()

    exchange = MockExchangeAdapter()
    exchange._mock_balance = 10000.0

    kill_switch = KillSwitch(str(tmp_path / "KILL"))
    position_tracker = PositionTracker(db, exchange, sample_live_config.assets)
    risk_manager = RiskManager(
        sample_live_config, db, exchange, kill_switch, position_tracker
    )

    yield risk_manager, db, exchange, kill_switch
    await db.close()


@pytest.mark.asyncio
async def test_all_breakers_pass(
    risk_setup: tuple[RiskManager, LiveDB, MockExchangeAdapter, KillSwitch],
) -> None:
    """Under normal conditions, all circuit breakers pass."""
    rm, db, exchange, kill_switch = risk_setup
    signal = _make_signal()

    ok, reason = await rm.can_trade(signal, 100.0)
    assert ok is True
    assert reason == "ok"


@pytest.mark.asyncio
async def test_kill_switch_triggers(
    risk_setup: tuple[RiskManager, LiveDB, MockExchangeAdapter, KillSwitch],
) -> None:
    """Kill switch file present blocks all trading."""
    rm, db, exchange, kill_switch = risk_setup
    signal = _make_signal()

    kill_switch.trigger("test kill")

    ok, reason = await rm.can_trade(signal, 100.0)
    assert ok is False
    assert "kill_switch" in reason

    # Clean up
    kill_switch.reset()


@pytest.mark.asyncio
async def test_drawdown_halt_triggers(
    risk_setup: tuple[RiskManager, LiveDB, MockExchangeAdapter, KillSwitch],
) -> None:
    """Equity drop >15% from peak triggers drawdown halt."""
    rm, db, exchange, kill_switch = risk_setup
    signal = _make_signal()

    # Record a peak equity of 10000
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    await db.update_peak_equity(today, 10000.0)

    # Simulate equity drop to 8000 (20% drawdown)
    exchange._mock_balance = 8000.0

    ok, reason = await rm.check_drawdown()
    assert ok is False
    assert "drawdown_halt" in reason


@pytest.mark.asyncio
async def test_drawdown_within_limits(
    risk_setup: tuple[RiskManager, LiveDB, MockExchangeAdapter, KillSwitch],
) -> None:
    """Equity drop within limits passes."""
    rm, db, exchange, kill_switch = risk_setup

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    await db.update_peak_equity(today, 10000.0)

    # 5% drop — within 15% limit
    exchange._mock_balance = 9500.0

    ok, reason = await rm.check_drawdown()
    assert ok is True


@pytest.mark.asyncio
async def test_api_error_rate_triggers(
    risk_setup: tuple[RiskManager, LiveDB, MockExchangeAdapter, KillSwitch],
) -> None:
    """5+ API errors in 10 minutes triggers error rate halt."""
    rm, db, exchange, kill_switch = risk_setup

    # Record 5 API errors
    for i in range(5):
        await db.record_api_error("timeout", f"test error {i}")

    ok, reason = await rm.check_api_errors()
    assert ok is False
    assert "api_error_rate" in reason


@pytest.mark.asyncio
async def test_api_error_within_limits(
    risk_setup: tuple[RiskManager, LiveDB, MockExchangeAdapter, KillSwitch],
) -> None:
    """Fewer than 5 API errors passes."""
    rm, db, exchange, kill_switch = risk_setup

    # Record only 3 errors
    for i in range(3):
        await db.record_api_error("timeout", f"test error {i}")

    ok, reason = await rm.check_api_errors()
    assert ok is True


@pytest.mark.asyncio
async def test_position_mismatch_detected(
    risk_setup: tuple[RiskManager, LiveDB, MockExchangeAdapter, KillSwitch],
) -> None:
    """Local vs exchange position divergence triggers mismatch breaker."""
    rm, db, exchange, kill_switch = risk_setup

    # Add a local position that doesn't exist on exchange
    from live.models import Position, PositionStatus, Side
    pos = Position(
        asset="BTC",
        side=Side.LONG,
        entry_price=50000.0,
        entry_time=datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc),
        size_usd=1000.0,
        signal_strength=0.4,
        confidence=0.12,
        hold_bars=4,
        bars_held=0,
        status=PositionStatus.OPEN,
    )
    await db.open_position(pos)

    # Exchange has no positions — mismatch
    exchange._mock_positions = {}

    # Clear cached result to force fresh reconciliation
    rm._last_reconcile_result = None

    ok, reason = await rm.check_position_mismatch()
    assert ok is False
    assert "position_mismatch" in reason


@pytest.mark.asyncio
async def test_max_positions_enforced(
    risk_setup: tuple[RiskManager, LiveDB, MockExchangeAdapter, KillSwitch],
) -> None:
    """Cannot exceed max_positions limit."""
    rm, db, exchange, kill_switch = risk_setup
    from live.models import Position, PositionStatus, Side

    # Fill up to max_positions (6)
    assets = ["BTC", "ETH", "XRP", "BNB", "SUI", "LINK"]
    for asset in assets:
        pos = Position(
            asset=asset,
            side=Side.LONG,
            entry_price=100.0,
            entry_time=datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc),
            size_usd=50.0,
            signal_strength=0.4,
            confidence=0.12,
            hold_bars=4,
            bars_held=0,
            status=PositionStatus.OPEN,
        )
        await db.open_position(pos)

    # Set matching exchange positions so reconciliation doesn't trigger first
    exchange._mock_positions = {
        f"{asset}_USDT": {
            "side": "long",
            "size": 0.5,
            "entry_price": 100.0,
            "unrealised_pnl": 0.0,
        }
        for asset in assets
    }
    # Clear cached reconciliation to force fresh one with matching positions
    rm._last_reconcile_result = None

    signal = _make_signal(asset="SOL")
    ok, reason = await rm.can_trade(signal, 100.0)
    assert ok is False
    assert "max_positions" in reason


@pytest.mark.asyncio
async def test_record_error_and_query(
    risk_setup: tuple[RiskManager, LiveDB, MockExchangeAdapter, KillSwitch],
) -> None:
    """record_error persists and is queryable."""
    rm, db, exchange, kill_switch = risk_setup

    await rm.record_error("timeout", "connection timed out")
    count = await db.get_api_errors_since(10)
    assert count >= 1


@pytest.mark.asyncio
async def test_update_peak_equity(
    risk_setup: tuple[RiskManager, LiveDB, MockExchangeAdapter, KillSwitch],
) -> None:
    """update_peak_equity tracks the highest equity seen."""
    rm, db, exchange, kill_switch = risk_setup

    await rm.update_peak_equity(10000.0)
    await rm.update_peak_equity(11000.0)
    await rm.update_peak_equity(10500.0)  # Should not lower the peak

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    peak = await db.get_peak_equity(today)
    assert peak == 11000.0


@pytest.mark.asyncio
async def test_rolling_sharpe_insufficient_data(
    tmp_path: Path,
    sample_live_config: LiveConfig,
) -> None:
    """Rolling Sharpe check passes when insufficient data (< window days)."""
    from live.db import LiveDB
    from live.exchange import MockExchangeAdapter
    from live.position_tracker import PositionTracker
    from live.kill_switch import KillSwitch
    from live.risk_manager import RiskManager

    db = LiveDB(str(tmp_path / "test.db"))
    await db.init()

    exchange = MockExchangeAdapter({})
    pt = PositionTracker(db, exchange, {})
    ks = KillSwitch(str(tmp_path / "KILL"))
    rm = RiskManager(sample_live_config, db, exchange, pt, ks)

    # No equity history at all -> should pass (not enough data to judge)
    ok, reason = await rm.check_rolling_sharpe()
    assert ok is True


@pytest.mark.asyncio
async def test_rolling_sharpe_halt(
    tmp_path: Path,
    sample_live_config: LiveConfig,
) -> None:
    """Rolling Sharpe check blocks trades when below halt threshold."""
    from live.db import LiveDB
    from live.exchange import MockExchangeAdapter
    from live.position_tracker import PositionTracker
    from live.kill_switch import KillSwitch
    from live.risk_manager import RiskManager

    db = LiveDB(str(tmp_path / "test.db"))
    await db.init()

    # Insert 25 days of LOSING daily equity (rolling Sharpe will be negative)
    for i in range(25):
        date_str = f"2024-01-{i+1:02d}"
        await db.save_daily_equity(date_str, 10000 - i * 50, -50.0, 5, 10)

    exchange = MockExchangeAdapter({})
    pt = PositionTracker(db, exchange, {})
    ks = KillSwitch(str(tmp_path / "KILL"))

    # Set halt threshold to 0.0 — negative Sharpe should trigger
    sample_live_config.circuit_breakers.rolling_sharpe_halt_threshold = 0.0
    sample_live_config.circuit_breakers.rolling_sharpe_window_days = 20
    rm = RiskManager(sample_live_config, db, exchange, pt, ks)

    ok, reason = await rm.check_rolling_sharpe()
    assert ok is False
    assert "rolling_sharpe" in reason
