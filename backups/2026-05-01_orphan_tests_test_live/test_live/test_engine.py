"""Tests for live.engine — crash recovery and data conversion."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd
import pytest
import pytest_asyncio

from live.config import LiveConfig
from live.db import LiveDB
from live.engine import LiveEngine
from live.models import OHLCVBar, Position, PositionStatus, Side


@pytest_asyncio.fixture
async def engine_with_db(
    tmp_path: Path, sample_live_config: LiveConfig
) -> tuple[LiveEngine, LiveDB]:
    """Create an engine and DB for testing recovery logic."""
    db_path = tmp_path / "engine_test.db"
    config = sample_live_config.model_copy()
    config.db_path = str(db_path)

    engine = LiveEngine(config)
    db = LiveDB(str(db_path))
    await db.init()
    engine._db = db

    yield engine, db
    await db.close()


@pytest.mark.asyncio
async def test_crash_recovery_expired(
    engine_with_db: tuple[LiveEngine, LiveDB],
) -> None:
    """Expired positions are closed at entry price on recovery."""
    engine, db = engine_with_db

    # Insert a position that would have expired (entry 2 hours ago, hold_bars=4)
    old_time = datetime.now(timezone.utc) - timedelta(hours=2)
    pos = Position(
        asset="BTC",
        side=Side.LONG,
        entry_price=50000.0,
        entry_time=old_time,
        size_usd=200.0,
        signal_strength=0.4,
        confidence=0.12,
        hold_bars=4,
        bars_held=0,
    )
    await db.open_position(pos)

    # Recover
    await engine.recover_state()

    # Should be closed
    open_positions = await db.get_open_positions()
    assert len(open_positions) == 0


@pytest.mark.asyncio
async def test_crash_recovery_active(
    engine_with_db: tuple[LiveEngine, LiveDB],
) -> None:
    """Recent positions remain open with updated bars_held."""
    engine, db = engine_with_db

    # Insert a position from 20 minutes ago (only 1 bar, hold_bars=4)
    recent_time = datetime.now(timezone.utc) - timedelta(minutes=20)
    pos = Position(
        asset="ETH",
        side=Side.SHORT,
        entry_price=3000.0,
        entry_time=recent_time,
        size_usd=150.0,
        signal_strength=0.3,
        confidence=0.08,
        hold_bars=4,
        bars_held=0,
    )
    await db.open_position(pos)

    # Recover
    await engine.recover_state()

    # Should still be open with updated bars_held
    open_positions = await db.get_open_positions()
    assert len(open_positions) == 1
    assert open_positions[0].asset == "ETH"
    # bars_held should be ~1 (20min / 15min = 1.33 -> int = 1)
    assert open_positions[0].bars_held >= 1


class TestBarsToDf:
    """Test the OHLCVBar to DataFrame conversion."""

    def test_bars_to_df_conversion(self) -> None:
        """Bars are correctly converted to a pandas DataFrame."""
        bars = [
            OHLCVBar(
                timestamp=datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc),
                open=50000.0,
                high=50100.0,
                low=49900.0,
                close=50050.0,
                volume=1000.0,
            ),
            OHLCVBar(
                timestamp=datetime(2025, 6, 15, 12, 15, tzinfo=timezone.utc),
                open=50050.0,
                high=50200.0,
                low=49950.0,
                close=50150.0,
                volume=1200.0,
            ),
        ]

        df = LiveEngine._bars_to_df(bars)

        assert len(df) == 2
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
        assert df.iloc[0]["close"] == 50050.0
        assert df.iloc[1]["volume"] == 1200.0
        assert df.index.tz is not None  # timezone-aware

    def test_bars_to_df_empty(self) -> None:
        """Empty input returns an empty DataFrame with correct columns."""
        df = LiveEngine._bars_to_df([])
        assert len(df) == 0
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]
