"""Tests for MarketDataSubscriber — parsing + filtering + routing."""

from __future__ import annotations

import pandas as pd
import pytest

from deribit_paper.market_data.buffer import IndexPriceBuffer, TradeBuffer
from deribit_paper.market_data.subscriber import (
    MarketDataSubscriber,
    parse_option_instrument,
)


# ─── parse_option_instrument ─────────────────────────────────────────

def test_parse_option_instrument_call():
    result = parse_option_instrument("BTC-28NOV25-60000-C")
    assert result is not None
    expiry, strike, option_type = result
    assert strike == 60000.0
    assert option_type == "C"
    assert expiry == pd.Timestamp("2025-11-28 08:00:00", tz="UTC")


def test_parse_option_instrument_put():
    result = parse_option_instrument("BTC-1JAN26-50000-P")
    assert result is not None
    assert result[1] == 50000.0
    assert result[2] == "P"


def test_parse_option_instrument_fractional_strike():
    result = parse_option_instrument("BTC-28NOV25-60000.5-C")
    assert result is not None
    assert result[1] == 60000.5


def test_parse_option_instrument_rejects_perp():
    assert parse_option_instrument("BTC-PERPETUAL") is None


def test_parse_option_instrument_rejects_future():
    assert parse_option_instrument("BTC-28NOV25") is None


def test_parse_option_instrument_rejects_garbage():
    assert parse_option_instrument("not_an_instrument") is None


# ─── Subscriber routing (no live WS) ─────────────────────────────────

@pytest.fixture
def subscriber_and_buffers():
    perp = TradeBuffer(name="perp", window_seconds=3600)
    opt = TradeBuffer(name="opt", window_seconds=3600)
    idx = IndexPriceBuffer()
    sub = MarketDataSubscriber(
        ws=None,  # no live WS — we call the handlers directly
        perp_buffer=perp,
        option_buffer=opt,
        index_buffer=idx,
        currency="BTC",
    )
    return sub, perp, opt, idx


@pytest.mark.asyncio
async def test_perp_trade_routing(subscriber_and_buffers):
    sub, perp, _, _ = subscriber_and_buffers
    params = {
        "channel": "trades.BTC-PERPETUAL.raw",
        "data": [
            {
                "trade_id": "t1",
                "timestamp": int(pd.Timestamp("2026-04-10 12:00:00", tz="UTC").timestamp() * 1000),
                "instrument_name": "BTC-PERPETUAL",
                "price": 50000.0,
                "amount": 100.0,
                "direction": "buy",
                "index_price": 50000.5,
                "mark_price": 50000.3,
            },
        ],
    }
    await sub._on_perp_trade_event(params)
    assert len(perp) == 1
    trade = list(perp._trades)[0]
    assert trade.trade_id == "t1"
    assert trade.direction == 1
    assert trade.instrument_name == "BTC-PERPETUAL"
    assert sub.stats["perp_added"] == 1


@pytest.mark.asyncio
async def test_perp_trade_dedup_on_replay(subscriber_and_buffers):
    sub, perp, _, _ = subscriber_and_buffers
    raw = {
        "trade_id": "dup",
        "timestamp": int(pd.Timestamp("2026-04-10 12:00:00", tz="UTC").timestamp() * 1000),
        "instrument_name": "BTC-PERPETUAL",
        "price": 50000.0,
        "amount": 100.0,
        "direction": "sell",
    }
    params = {"channel": "trades.BTC-PERPETUAL.raw", "data": [raw, raw]}
    await sub._on_perp_trade_event(params)
    assert len(perp) == 1
    assert sub.stats["perp_added"] == 1
    assert sub.stats["perp_dup"] == 1


@pytest.mark.asyncio
async def test_option_trade_routing_in_moneyness_range(subscriber_and_buffers):
    sub, _, opt, _ = subscriber_and_buffers
    params = {
        "channel": "trades.option.BTC.raw",
        "data": [
            {
                "trade_id": "o1",
                "timestamp": int(pd.Timestamp("2026-04-10 12:00:00", tz="UTC").timestamp() * 1000),
                "instrument_name": "BTC-12APR26-60000-C",
                "price": 0.05,
                "amount": 1.0,
                "direction": "buy",
                "iv": 75.0,
                "index_price": 60000.0,  # moneyness = 1.0 → accepted
                "mark_price": 0.049,
            },
        ],
    }
    await sub._on_option_trade_event(params)
    assert len(opt) == 1
    assert sub.stats["option_added"] == 1


@pytest.mark.asyncio
async def test_option_trade_filtered_by_moneyness(subscriber_and_buffers):
    sub, _, opt, _ = subscriber_and_buffers
    params = {
        "channel": "trades.option.BTC.raw",
        "data": [
            {
                "trade_id": "o_deep_otm",
                "timestamp": int(pd.Timestamp("2026-04-10 12:00:00", tz="UTC").timestamp() * 1000),
                "instrument_name": "BTC-12APR26-100000-C",  # strike 100k
                "price": 0.01, "amount": 1.0, "direction": "buy",
                "iv": 90.0, "index_price": 60000.0,  # moneyness = 1.67 → filtered
            },
        ],
    }
    await sub._on_option_trade_event(params)
    assert len(opt) == 0
    assert sub.stats["option_filtered_moneyness"] == 1


@pytest.mark.asyncio
async def test_option_trade_filtered_by_dte(subscriber_and_buffers):
    sub, _, opt, _ = subscriber_and_buffers
    params = {
        "channel": "trades.option.BTC.raw",
        "data": [
            {
                "trade_id": "o_far",
                "timestamp": int(pd.Timestamp("2026-04-10 12:00:00", tz="UTC").timestamp() * 1000),
                "instrument_name": "BTC-25DEC26-60000-C",  # 8+ months out
                "price": 0.2, "amount": 1.0, "direction": "buy",
                "iv": 70.0, "index_price": 60000.0,
            },
        ],
    }
    await sub._on_option_trade_event(params)
    assert len(opt) == 0
    assert sub.stats["option_filtered_dte"] == 1


@pytest.mark.asyncio
async def test_option_unparseable_counted(subscriber_and_buffers):
    sub, _, opt, _ = subscriber_and_buffers
    params = {
        "channel": "trades.option.BTC.raw",
        "data": [
            {
                "trade_id": "garbage",
                "timestamp": int(pd.Timestamp("2026-04-10 12:00:00", tz="UTC").timestamp() * 1000),
                "instrument_name": "NOT_AN_OPTION_NAME",
                "price": 0.1, "amount": 1.0, "direction": "buy",
                "iv": 70.0, "index_price": 60000.0,
            },
        ],
    }
    await sub._on_option_trade_event(params)
    assert len(opt) == 0
    assert sub.stats["option_unparseable"] == 1


@pytest.mark.asyncio
async def test_index_event_routing(subscriber_and_buffers):
    sub, _, _, idx = subscriber_and_buffers
    params = {
        "channel": "deribit_price_index.btc_usd",
        "data": {
            "price": 60500.25,
            "timestamp": int(pd.Timestamp("2026-04-10 12:00:00", tz="UTC").timestamp() * 1000),
            "index_name": "btc_usd",
        },
    }
    await sub._on_index_event(params)
    assert len(idx) == 1
    assert idx.latest() == 60500.25
    assert sub.stats["index_updates"] == 1


@pytest.mark.asyncio
async def test_index_event_missing_fields_ignored(subscriber_and_buffers):
    sub, _, _, idx = subscriber_and_buffers
    await sub._on_index_event({"channel": "deribit_price_index.btc_usd", "data": {}})
    assert len(idx) == 0
    assert sub.stats["index_updates"] == 0


# ─── Kill switch WS marking ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_ws_messages_mark_kill_switch():
    """WS event handlers must call kill_switch.mark_ws_message().

    Regression test for the 2026-04-13 incident where mark_ws_message
    lived in the bar-close handler instead of the WS event handlers.
    """
    from unittest.mock import MagicMock

    perp = TradeBuffer(name="perp", window_seconds=3600)
    opt = TradeBuffer(name="opt", window_seconds=3600)
    idx = IndexPriceBuffer()
    ks = MagicMock()

    sub = MarketDataSubscriber(
        ws=None, perp_buffer=perp, option_buffer=opt,
        index_buffer=idx, currency="BTC",
    )
    sub.set_kill_switch(ks)

    # Perp trade event
    await sub._on_perp_trade_event({
        "data": [{
            "trade_id": "t1",
            "timestamp": int(pd.Timestamp("2026-04-10 12:00:00", tz="UTC").timestamp() * 1000),
            "instrument_name": "BTC-PERPETUAL",
            "price": 50000.0, "amount": 1.0, "direction": "buy",
        }],
    })
    assert ks.mark_ws_message.call_count == 1

    # Index event
    await sub._on_index_event({
        "data": {
            "price": 50000.0,
            "timestamp": int(pd.Timestamp("2026-04-10 12:00:00", tz="UTC").timestamp() * 1000),
        },
    })
    assert ks.mark_ws_message.call_count == 2

    # Option event
    await sub._on_option_trade_event({
        "data": [{
            "trade_id": "o1",
            "timestamp": int(pd.Timestamp("2026-04-10 12:00:00", tz="UTC").timestamp() * 1000),
            "instrument_name": "BTC-14APR26-71000-C",
            "price": 0.01, "amount": 1.0, "direction": "buy",
            "iv": 50.0, "index_price": 71000.0, "mark_price": 0.01,
        }],
    })
    assert ks.mark_ws_message.call_count == 3


@pytest.mark.asyncio
async def test_no_kill_switch_marking_without_set():
    """Without set_kill_switch(), event handlers must not crash."""
    perp = TradeBuffer(name="perp", window_seconds=3600)
    opt = TradeBuffer(name="opt", window_seconds=3600)
    idx = IndexPriceBuffer()
    sub = MarketDataSubscriber(
        ws=None, perp_buffer=perp, option_buffer=opt,
        index_buffer=idx, currency="BTC",
    )
    # Should not raise even without kill switch
    await sub._on_perp_trade_event({
        "data": [{
            "trade_id": "t1",
            "timestamp": int(pd.Timestamp("2026-04-10 12:00:00", tz="UTC").timestamp() * 1000),
            "instrument_name": "BTC-PERPETUAL",
            "price": 50000.0, "amount": 1.0, "direction": "buy",
        }],
    })
    assert len(perp) == 1
