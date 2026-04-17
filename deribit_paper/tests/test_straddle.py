"""Tests for StrikePicker — ATM straddle selection from Deribit instruments."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pandas as pd
import pytest

from deribit_paper.features.straddle import (
    StraddleSelection,
    StrikePicker,
    _parse_instrument,
)


def _make_picker(instruments: list[dict]) -> StrikePicker:
    """Build a StrikePicker with a mocked REST client returning canned instruments."""
    rest = AsyncMock()
    rest.get_instruments.return_value = instruments
    return StrikePicker(
        rest=rest,
        currency="BTC",
        dte_min_days=1.0,
        dte_max_days=14.0,
        instrument_refresh_hours=1.0,
    )


def _instrument(name: str) -> dict:
    return {"instrument_name": name, "kind": "option"}


# ─── _parse_instrument ────────────────────────────────────────────────

def test_parse_call_strike_and_expiry():
    result = _parse_instrument("BTC-28NOV25-60000-C")
    assert result is not None
    expiry_dt, strike, option_type = result
    assert strike == 60_000.0
    assert option_type == "C"
    assert expiry_dt == pd.Timestamp("2025-11-28 08:00:00", tz="UTC")


def test_parse_put():
    result = _parse_instrument("BTC-1JAN26-50000-P")
    assert result is not None
    assert result[1] == 50_000.0
    assert result[2] == "P"


def test_parse_rejects_non_option():
    assert _parse_instrument("BTC-PERPETUAL") is None
    assert _parse_instrument("BTC-28NOV25") is None


# ─── pick_atm_straddle ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pick_atm_straddle_happy_path():
    # Build a front expiry with multiple strikes, both call and put
    instruments = [
        _instrument("BTC-12APR26-58000-C"),
        _instrument("BTC-12APR26-58000-P"),
        _instrument("BTC-12APR26-60000-C"),
        _instrument("BTC-12APR26-60000-P"),
        _instrument("BTC-12APR26-62000-C"),
        _instrument("BTC-12APR26-62000-P"),
    ]
    picker = _make_picker(instruments)

    # "Now" is 1 day before expiry → front expiry is selected
    now = pd.Timestamp("2026-04-11 08:00:00", tz="UTC")
    result = await picker.pick_atm_straddle(
        index_price=59_900.0, now=now,
    )

    assert result is not None
    assert result.strike == 60_000.0  # closest to 59_900
    assert result.call_instrument == "BTC-12APR26-60000-C"
    assert result.put_instrument == "BTC-12APR26-60000-P"
    assert result.expiry_dt == pd.Timestamp("2026-04-12 08:00:00", tz="UTC")
    assert 0.9 < result.moneyness < 1.1
    assert 0.9 < result.dte_days < 1.1


@pytest.mark.asyncio
async def test_pick_atm_selects_nearest_strike():
    """With strikes 58k, 60k, 62k and index at 61_000, we want the 60k strike."""
    instruments = [
        _instrument("BTC-12APR26-58000-C"),
        _instrument("BTC-12APR26-58000-P"),
        _instrument("BTC-12APR26-60000-C"),
        _instrument("BTC-12APR26-60000-P"),
        _instrument("BTC-12APR26-62000-C"),
        _instrument("BTC-12APR26-62000-P"),
    ]
    picker = _make_picker(instruments)
    now = pd.Timestamp("2026-04-11 08:00:00", tz="UTC")
    result = await picker.pick_atm_straddle(index_price=61_000.0, now=now)
    # 61k is equidistant between 60k and 62k (actually 1k from each)
    # Our "min by abs distance" should pick the first (60k) or 62k deterministically
    assert result.strike in (60_000.0, 62_000.0)


@pytest.mark.asyncio
async def test_pick_selects_front_expiry():
    """Given multiple expiries in the DTE band, the closest should be chosen."""
    instruments = [
        # Front expiry (3 days out)
        _instrument("BTC-13APR26-60000-C"),
        _instrument("BTC-13APR26-60000-P"),
        # Later expiry (10 days out)
        _instrument("BTC-20APR26-60000-C"),
        _instrument("BTC-20APR26-60000-P"),
    ]
    picker = _make_picker(instruments)
    now = pd.Timestamp("2026-04-10 08:00:00", tz="UTC")
    result = await picker.pick_atm_straddle(index_price=60_000.0, now=now)
    assert result is not None
    assert result.expiry_dt == pd.Timestamp("2026-04-13 08:00:00", tz="UTC")


@pytest.mark.asyncio
async def test_pick_returns_none_when_dte_filter_excludes_all():
    instruments = [
        # 30 days out — outside default 1-14 DTE band
        _instrument("BTC-10MAY26-60000-C"),
        _instrument("BTC-10MAY26-60000-P"),
    ]
    picker = _make_picker(instruments)
    now = pd.Timestamp("2026-04-10 08:00:00", tz="UTC")
    result = await picker.pick_atm_straddle(index_price=60_000.0, now=now)
    assert result is None


@pytest.mark.asyncio
async def test_pick_returns_none_when_no_strikes_have_both_legs():
    instruments = [
        _instrument("BTC-12APR26-60000-C"),
        # Missing the matching put
    ]
    picker = _make_picker(instruments)
    now = pd.Timestamp("2026-04-11 08:00:00", tz="UTC")
    result = await picker.pick_atm_straddle(index_price=60_000.0, now=now)
    assert result is None


@pytest.mark.asyncio
async def test_pick_returns_none_with_empty_instruments():
    picker = _make_picker([])
    now = pd.Timestamp("2026-04-11 08:00:00", tz="UTC")
    result = await picker.pick_atm_straddle(index_price=60_000.0, now=now)
    assert result is None


@pytest.mark.asyncio
async def test_instruments_cache_avoids_redundant_refresh():
    """Two picks within the refresh window should hit the REST endpoint only once."""
    instruments = [
        _instrument("BTC-12APR26-60000-C"),
        _instrument("BTC-12APR26-60000-P"),
    ]
    picker = _make_picker(instruments)
    picker.instrument_refresh_hours = 1.0
    now = pd.Timestamp("2026-04-11 08:00:00", tz="UTC")

    await picker.pick_atm_straddle(index_price=60_000.0, now=now)
    await picker.pick_atm_straddle(index_price=60_100.0, now=now + pd.Timedelta(minutes=30))

    # Only one REST call despite two picks
    assert picker._rest.get_instruments.await_count == 1


@pytest.mark.asyncio
async def test_instruments_cache_refreshed_after_ttl():
    instruments = [
        _instrument("BTC-12APR26-60000-C"),
        _instrument("BTC-12APR26-60000-P"),
    ]
    picker = _make_picker(instruments)
    picker.instrument_refresh_hours = 0.5  # 30 min TTL
    now = pd.Timestamp("2026-04-11 08:00:00", tz="UTC")

    await picker.pick_atm_straddle(index_price=60_000.0, now=now)
    await picker.pick_atm_straddle(
        index_price=60_100.0, now=now + pd.Timedelta(minutes=31),
    )
    # Second call is past the TTL so we expect a fresh REST call
    assert picker._rest.get_instruments.await_count == 2


@pytest.mark.asyncio
async def test_straddle_selection_is_atm_flag():
    # Expiry at 12APR26-08:00; pick "now" 2 days earlier to leave
    # comfortable DTE headroom so small minute offsets stay inside the band.
    instruments = [
        _instrument("BTC-12APR26-60000-C"),
        _instrument("BTC-12APR26-60000-P"),
    ]
    picker = _make_picker(instruments)
    now = pd.Timestamp("2026-04-10 08:00:00", tz="UTC")

    result = await picker.pick_atm_straddle(index_price=60_000.0, now=now)
    assert result is not None
    assert result.is_atm is True

    result_off = await picker.pick_atm_straddle(
        index_price=59_000.0, now=now + pd.Timedelta(minutes=1),
    )
    assert result_off is not None
    assert result_off.is_atm is False
