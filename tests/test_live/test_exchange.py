"""Tests for live.exchange — mock adapter, symbol/interval mapping."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from live.exchange import (
    MockExchangeAdapter,
    _to_mexc_interval,
    _to_mexc_symbol,
)
from live.models import OHLCVBar


@pytest.mark.asyncio
async def test_mock_exchange_returns_bars(mock_exchange: MockExchangeAdapter) -> None:
    """MockExchangeAdapter returns bars from fixtures."""
    bars = await mock_exchange.fetch_klines("BTCUSDT", "15m", 50)
    assert len(bars) == 50
    assert all(isinstance(b, OHLCVBar) for b in bars)
    assert bars[0].close > 0


def test_symbol_mapping() -> None:
    """BTCUSDT -> BTC_USDT conversion."""
    assert _to_mexc_symbol("BTCUSDT") == "BTC_USDT"
    assert _to_mexc_symbol("ETHUSDT") == "ETH_USDT"
    assert _to_mexc_symbol("XRPUSDT") == "XRP_USDT"
    # Already in MEXC format
    assert _to_mexc_symbol("BTC_USDT") == "BTC_USDT"
    # Non-USDT pair
    assert _to_mexc_symbol("BTCBTC") == "BTCBTC"


def test_interval_mapping() -> None:
    """Standard interval strings map to MEXC API parameters."""
    assert _to_mexc_interval("1m") == "Min1"
    assert _to_mexc_interval("5m") == "Min5"
    assert _to_mexc_interval("15m") == "Min15"
    assert _to_mexc_interval("30m") == "Min30"
    assert _to_mexc_interval("1h") == "Min60"
    assert _to_mexc_interval("4h") == "Hour4"
    assert _to_mexc_interval("8h") == "Hour8"
    assert _to_mexc_interval("1d") == "Day1"

    with pytest.raises(ValueError, match="Unsupported interval"):
        _to_mexc_interval("2h")


@pytest.mark.asyncio
async def test_empty_response_handled() -> None:
    """MockExchangeAdapter returns empty list for unknown symbols."""
    adapter = MockExchangeAdapter(fixtures={})
    bars = await adapter.fetch_klines("UNKNOWN", "15m", 100)
    assert bars == []

    ticker = await adapter.fetch_ticker("UNKNOWN")
    assert ticker["last_price"] == 0.0
