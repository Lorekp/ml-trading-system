"""Tests for H7 sanity bounds on incoming market-data fields.

Why this exists
---------------

Before H7, the MarketDataSubscriber's three parsers (perp trades,
option trades, index price) had ZERO sanity bounds on incoming
numeric fields. A corrupt/zero/NaN/inf price or amount went straight
into the SVI calibrator, the strike picker, and the PnL math. Real
Deribit incidents (newly-listed instruments) have published a 0.0
mark price; a stuck or partially-decoded feed could push NaN.

These tests pin the H7 contract:

1. Zero, negative, NaN, inf, and absurdly-large prices are dropped at
   the parser, NOT propagated to the buffer / calibrator / engine.
2. The drop bumps a stats counter (``bad_price_*``) so operators can
   monitor data-quality drift.
3. Drops are logged at WARNING but rate-limited so a stuck corrupt
   feed doesn't spam the log.
4. Legitimate edge values (small option prices, large BTC index
   prices) still flow through.
"""
from __future__ import annotations

import math

import pandas as pd
import pytest

from deribit_paper.market_data.buffer import (
    IndexPriceBuffer,
    TradeBuffer,
)
from deribit_paper.market_data.subscriber import (
    MarketDataSubscriber,
    _is_sane_positive,
    _MAX_INDEX_PRICE,
    _MAX_OPTION_PRICE_BTC,
    _MAX_PERP_PRICE,
    _MAX_TRADE_AMOUNT,
)


def _make_sub() -> MarketDataSubscriber:
    perp = TradeBuffer(name="perp", window_seconds=3600.0)
    opt = TradeBuffer(name="option", window_seconds=3600.0)
    idx = IndexPriceBuffer(window_seconds=3600.0)
    return MarketDataSubscriber(
        ws=None, perp_buffer=perp, option_buffer=opt, index_buffer=idx,
        currency="BTC",
    )


# ── Pure helper tests ────────────────────────────────────────────────


@pytest.mark.parametrize("x,max_val,expected", [
    (1.0, 1e7, True),
    (60_000.0, 1e7, True),
    (0.0, 1e7, False),       # zero rejected
    (-1.0, 1e7, False),      # negative rejected
    (math.nan, 1e7, False),  # NaN rejected
    (math.inf, 1e7, False),  # inf rejected
    (-math.inf, 1e7, False), # -inf rejected
    (1e8, 1e7, False),       # above max rejected
    (1e7, 1e7, False),       # equal to max rejected (strict <)
    ("60000", 1e7, False),   # string rejected
    (None, 1e7, False),      # None rejected
])
def test_is_sane_positive(x, max_val, expected):
    assert _is_sane_positive(x, max_val=max_val) is expected


# ── Perp trade ───────────────────────────────────────────────────────


def _perp_raw(price: float = 60_000.0, amount: float = 100.0) -> dict:
    return {
        "trade_id": "T1",
        "timestamp": 1_700_000_000_000,
        "instrument_name": "BTC-PERPETUAL",
        "price": price,
        "amount": amount,
        "direction": "buy",
    }


@pytest.mark.parametrize("bad_price", [0.0, -1.0, math.nan, math.inf, _MAX_PERP_PRICE + 1])
def test_perp_rejects_bad_price(bad_price):
    sub = _make_sub()
    raw = _perp_raw(price=bad_price)
    trade = sub._parse_perp_trade(raw)
    assert trade is None
    assert sub.stats["bad_price_perp"] == 1


@pytest.mark.parametrize("bad_amount", [0.0, -10.0, math.nan, _MAX_TRADE_AMOUNT + 1])
def test_perp_rejects_bad_amount(bad_amount):
    sub = _make_sub()
    raw = _perp_raw(amount=bad_amount)
    trade = sub._parse_perp_trade(raw)
    assert trade is None
    assert sub.stats["bad_price_perp"] == 1


def test_perp_accepts_legitimate_price_and_amount():
    sub = _make_sub()
    trade = sub._parse_perp_trade(_perp_raw(price=60_000.0, amount=100.0))
    assert trade is not None
    assert trade.price == 60_000.0
    assert trade.amount == 100.0
    assert sub.stats["bad_price_perp"] == 0


# ── Option trade ─────────────────────────────────────────────────────


def _option_raw(
    *, price: float = 0.05, amount: float = 1.0,
    instrument: str = "BTC-28NOV25-60000-C",
) -> dict:
    return {
        "trade_id": "O1",
        "timestamp": 1_700_000_000_000,
        "instrument_name": instrument,
        "price": price,
        "amount": amount,
        "direction": "buy",
        "iv": 60.0,
        "index_price": 60_000.0,
    }


@pytest.mark.parametrize("bad_price", [0.0, -0.01, math.nan, _MAX_OPTION_PRICE_BTC + 1])
def test_option_rejects_bad_price(bad_price):
    sub = _make_sub()
    trade = sub._parse_option_trade(_option_raw(price=bad_price))
    assert trade is None
    assert sub.stats["bad_price_option"] == 1


def test_option_accepts_small_legitimate_price():
    """Deep OTM options can trade at 0.0001 BTC. Must not be rejected."""
    sub = _make_sub()
    trade = sub._parse_option_trade(_option_raw(price=0.0001, amount=0.5))
    assert trade is not None
    assert trade.price == 0.0001
    assert sub.stats["bad_price_option"] == 0


def test_option_unparseable_instrument_does_not_bump_bad_price_counter():
    """Bad instrument names use the existing ``option_unparseable``
    counter, NOT the H7 ``bad_price_option`` counter — they're
    different failure modes."""
    sub = _make_sub()
    trade = sub._parse_option_trade(
        _option_raw(instrument="GIBBERISH"),
    )
    assert trade is None
    assert sub.stats["option_unparseable"] == 1
    assert sub.stats["bad_price_option"] == 0


# ── Index price ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_index_rejects_zero_price():
    sub = _make_sub()
    await sub._on_index_event({
        "data": {"price": 0.0, "timestamp": 1_700_000_000_000},
    })
    assert sub.stats["bad_price_index"] == 1
    assert sub.stats["index_updates"] == 0
    assert sub.index_buffer.latest() is None


@pytest.mark.asyncio
async def test_index_rejects_nan_price():
    sub = _make_sub()
    await sub._on_index_event({
        "data": {"price": math.nan, "timestamp": 1_700_000_000_000},
    })
    assert sub.stats["bad_price_index"] == 1
    assert sub.index_buffer.latest() is None


@pytest.mark.asyncio
async def test_index_rejects_absurd_price():
    sub = _make_sub()
    await sub._on_index_event({
        "data": {"price": 1e20, "timestamp": 1_700_000_000_000},
    })
    assert sub.stats["bad_price_index"] == 1
    assert sub.index_buffer.latest() is None


@pytest.mark.asyncio
async def test_index_accepts_legitimate_high_price():
    """BTC index plausibly hits $200k+ in future runs. Must not be rejected."""
    sub = _make_sub()
    await sub._on_index_event({
        "data": {"price": 200_000.0, "timestamp": 1_700_000_000_000},
    })
    assert sub.stats["index_updates"] == 1
    assert sub.stats["bad_price_index"] == 0
    assert sub.index_buffer.latest() == 200_000.0


# ── Rate-limited logging ─────────────────────────────────────────────


def test_repeated_bad_perp_only_logs_once_within_cooldown(caplog):
    """A stuck corrupt feed must NOT spam the log. The stats counter
    still increments per event so we measure the magnitude."""
    import logging
    sub = _make_sub()
    sub._BAD_PRICE_LOG_COOLDOWN = 60.0  # ensure no expiry mid-test

    with caplog.at_level(logging.WARNING, logger="deribit_paper.market_data.subscriber"):
        for _ in range(50):
            sub._parse_perp_trade(_perp_raw(price=0.0))

    # All 50 bumped the counter, but only 1 WARNING line.
    assert sub.stats["bad_price_perp"] == 50
    h7_warnings = [
        r for r in caplog.records
        if "H7 sanity reject" in r.getMessage() and r.levelname == "WARNING"
    ]
    assert len(h7_warnings) == 1
