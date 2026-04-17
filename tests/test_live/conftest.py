"""Shared fixtures for live trading engine tests."""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import pytest_asyncio

from live.config import (
    CircuitBreakerConfig,
    ExchangeConfig,
    LiveConfig,
    LiveModeConfig,
    PaperConfig,
    RiskConfig,
    SchedulerConfig,
    TelegramConfig,
)
from live.db import LiveDB
from live.exchange import MockExchangeAdapter
from live.models import OHLCVBar, Side, Signal
from signal_engine.config import (
    ConfidenceFilterConfig,
    DataConfig,
    FeatureConfig,
    ModelConfig,
    PortfolioConfig,
    RegimeFilterConfig,
    SignalConfig,
    SystemConfig,
)


@pytest.fixture
def sample_live_config() -> LiveConfig:
    """LiveConfig with test values — no real API keys."""
    signal_config = SystemConfig(
        model=ModelConfig(),
        features=FeatureConfig(),
        portfolio=PortfolioConfig(),
        regime_filter=RegimeFilterConfig(),
        confidence_filter=ConfidenceFilterConfig(mode="absolute", threshold=0.04),
        signal=SignalConfig(),
        data=DataConfig(data_dirs=["data/binance-1m"]),
        assets={"BTC": "BTCUSDT", "ETH": "ETHUSDT"},
    )
    return LiveConfig(
        signal=signal_config,
        exchange=ExchangeConfig(
            name="mock",
            api_key="test_key",
            api_secret="test_secret",
        ),
        telegram=TelegramConfig(
            bot_token="",
            chat_id="",
            enabled=False,
        ),
        paper=PaperConfig(
            slippage_bps=1.0,
            starting_capital=10000.0,
        ),
        scheduler=SchedulerConfig(
            bar_interval_m=15,
            poll_offset_s=5,
        ),
        risk=RiskConfig(
            max_positions=6,
            max_per_asset_pct=0.06,
            max_total_exposure_pct=0.80,
            daily_dd_halt_pct=0.05,
            confidence_threshold=0.04,
            hold_bars=4,
            base_sizing_pct=0.02,
            max_signal_mult=2.0,
        ),
        db_path="data/live/test_state.db",
        log_level="DEBUG",
        assets={"BTC": "BTCUSDT", "ETH": "ETHUSDT"},
    )


@pytest_asyncio.fixture
async def tmp_db(tmp_path: Path) -> LiveDB:
    """Ephemeral LiveDB using a temp directory."""
    db_path = tmp_path / "test.db"
    db = LiveDB(str(db_path))
    await db.init()
    yield db
    await db.close()


def _generate_ohlcv_bars(
    n: int = 200,
    base_price: float = 50000.0,
    start_time: datetime | None = None,
    interval_minutes: int = 15,
) -> list[OHLCVBar]:
    """Generate synthetic OHLCV bars for testing."""
    if start_time is None:
        start_time = datetime(2025, 1, 1, tzinfo=timezone.utc)

    rng = np.random.default_rng(42)
    bars: list[OHLCVBar] = []
    price = base_price

    for i in range(n):
        ret = rng.normal(0, 0.002)
        price *= 1 + ret
        high = price * (1 + abs(rng.normal(0, 0.001)))
        low = price * (1 - abs(rng.normal(0, 0.001)))
        open_price = price * (1 + rng.normal(0, 0.0005))
        volume = abs(rng.normal(1000, 200))

        bars.append(
            OHLCVBar(
                timestamp=start_time + timedelta(minutes=interval_minutes * i),
                open=open_price,
                high=high,
                low=low,
                close=price,
                volume=volume,
            )
        )

    return bars


@pytest.fixture
def mock_exchange() -> MockExchangeAdapter:
    """MockExchangeAdapter with synthetic bars for BTC and ETH."""
    btc_bars = _generate_ohlcv_bars(200, base_price=50000.0)
    eth_bars = _generate_ohlcv_bars(200, base_price=3000.0)
    btc_4h = _generate_ohlcv_bars(50, base_price=50000.0, interval_minutes=240)
    eth_4h = _generate_ohlcv_bars(50, base_price=3000.0, interval_minutes=240)

    fixtures = {
        "BTCUSDT": btc_bars,
        "BTCUSDT_15m": btc_bars,
        "BTCUSDT_4h": btc_4h,
        "ETHUSDT": eth_bars,
        "ETHUSDT_15m": eth_bars,
        "ETHUSDT_4h": eth_4h,
    }
    return MockExchangeAdapter(fixtures)


@pytest.fixture
def sample_signal() -> Signal:
    """Pre-built Signal object for testing."""
    return Signal(
        timestamp=datetime(2025, 6, 15, 12, 0, tzinfo=timezone.utc),
        asset="BTC",
        direction=1.0,
        probability=0.62,
        confidence=0.12,
        signal_strength=0.4,
        regime_aligned=True,
        passed_confidence=True,
    )


@pytest.fixture
def sample_ohlcv_df() -> pd.DataFrame:
    """200-bar DataFrame of realistic BTC 15m data."""
    bars = _generate_ohlcv_bars(200, base_price=50000.0)
    data = [
        {
            "open": b.open,
            "high": b.high,
            "low": b.low,
            "close": b.close,
            "volume": b.volume,
        }
        for b in bars
    ]
    index = pd.DatetimeIndex([b.timestamp for b in bars], tz="UTC")
    return pd.DataFrame(data, index=index)
