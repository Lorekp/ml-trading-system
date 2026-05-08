"""O-4 (parity-r3 cycle 2 — 2026-05-08) regression tests.

Deribit's option tick size depends on mark price:
  mark <  0.05 BTC → 0.0001 BTC tick (cheap options, far OTM / weeklies)
  mark >= 0.05 BTC → 0.0005 BTC tick (ATM / near-term)

Pre-fix the engine hardcoded 0.0005 in both ``place_ioc_taker`` and
``_place_tiered_taker``, over-rounding caps on cheap options by 5x —
typically pulling the cap THROUGH the touch and inflating an intended
0.5% slippage cap into a 5% one.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from deribit_paper.execution.order_manager import (
    OrderManager,
    OrderResult,
    _option_tick_for_mark,
)


# ── Helper-level: tick chooser ────────────────────────────────────────


def test_tick_helper_below_breakpoint_returns_fine():
    assert _option_tick_for_mark(0.001) == 0.0001
    assert _option_tick_for_mark(0.0499) == 0.0001


def test_tick_helper_at_breakpoint_returns_coarse():
    """``< 0.05`` strict — exactly 0.05 maps to the coarse tick (matches
    Deribit testnet behaviour observed 2026-05-08)."""
    assert _option_tick_for_mark(0.05) == 0.0005


def test_tick_helper_above_breakpoint_returns_coarse():
    assert _option_tick_for_mark(0.0501) == 0.0005
    assert _option_tick_for_mark(0.5) == 0.0005
    assert _option_tick_for_mark(2.0) == 0.0005


def test_tick_helper_zero_or_negative_falls_back_to_coarse():
    """Defensive: missing mark price (0 / negative / NaN) → pre-O-4
    behaviour (coarse tick)."""
    assert _option_tick_for_mark(0.0) == 0.0005
    assert _option_tick_for_mark(-1.0) == 0.0005
    assert _option_tick_for_mark(float("nan")) == 0.0005


# ── Integration: place_ioc_taker uses the right tick ─────────────────


def _stub_rest(*, mark_price: float, best_ask: float, best_bid: float) -> MagicMock:
    rest = MagicMock()
    rest.get_ticker = AsyncMock(return_value={
        "best_ask_price": best_ask,
        "best_bid_price": best_bid,
        "mark_price": mark_price,
    })
    rest.place_order = AsyncMock(return_value={
        "_stub": True,
        "order": {
            "order_id": "stub_order",
            "order_state": "filled",
            "filled_amount": 1.0,
            "average_price": best_ask,
        },
    })
    rest.cancel_order = AsyncMock(return_value={"order_state": "cancelled"})
    return rest


@pytest.mark.asyncio
async def test_place_ioc_taker_uses_fine_tick_for_cheap_option():
    """Mark = 0.002 BTC (cheap weekly) → cap rounds to 0.0001 grid."""
    # Best ask 0.00203 with 0.5% cap = 0.00204 (cap)
    # Fine tick (0.0001): round(0.00204015 / 0.0001) * 0.0001 = 0.0020
    # Coarse tick (0.0005): round(0.00204015 / 0.0005) * 0.0005 = 0.002
    # The fine grid lets us land at 0.0020 which is BELOW best ask 0.00203
    # (because rounding to grid). Use a cheap mark so the difference
    # between the two ticks is obvious in the placed price.
    rest = _stub_rest(mark_price=0.002, best_ask=0.0023, best_bid=0.0022)
    manager = OrderManager(rest=rest)

    result: OrderResult = await manager.place_ioc_taker(
        instrument_name="BTC-30APR26-30000-C",
        side="buy",
        amount=1.0,
        max_slippage_pct=0.5,
    )

    # Inspect the actual price submitted: must be on the fine 0.0001 grid.
    place_order_call = rest.place_order.call_args
    submitted_price = float(place_order_call.kwargs["price"])
    # The cap = best_ask × 1.005 = 0.0023 × 1.005 = 0.0023115
    # Fine tick: round(0.0023115 / 0.0001) * 0.0001 = 0.0023
    assert submitted_price == pytest.approx(0.0023, abs=1e-9)
    # Reject any value that would only happen on the coarse 0.0005 grid
    assert (submitted_price * 10000) % 5 != 0 or submitted_price in (0.0005, 0.001, 0.0015, 0.002, 0.0025)
    assert result.is_filled or result.is_dry_run


@pytest.mark.asyncio
async def test_place_ioc_taker_uses_coarse_tick_for_atm_option():
    """Mark = 0.10 BTC (ATM near-term) → cap rounds to 0.0005 grid."""
    rest = _stub_rest(mark_price=0.10, best_ask=0.1003, best_bid=0.1002)
    manager = OrderManager(rest=rest)

    await manager.place_ioc_taker(
        instrument_name="BTC-30APR26-60000-C",
        side="buy",
        amount=1.0,
        max_slippage_pct=0.5,
    )

    place_order_call = rest.place_order.call_args
    submitted_price = float(place_order_call.kwargs["price"])
    # cap = 0.1003 × 1.005 = 0.100802 → coarse tick rounds to 0.1010
    assert submitted_price == pytest.approx(0.1010, abs=1e-9)


@pytest.mark.asyncio
async def test_place_ioc_taker_at_boundary_uses_coarse_tick():
    """Exactly at mark = 0.05 BTC the coarse tick kicks in."""
    rest = _stub_rest(mark_price=0.05, best_ask=0.0501, best_bid=0.0499)
    manager = OrderManager(rest=rest)

    await manager.place_ioc_taker(
        instrument_name="BTC-30APR26-60000-C",
        side="buy",
        amount=1.0,
        max_slippage_pct=0.5,
    )

    place_order_call = rest.place_order.call_args
    submitted_price = float(place_order_call.kwargs["price"])
    # cap = 0.0501 × 1.005 = 0.05035 → coarse tick rounds to 0.0505
    assert submitted_price == pytest.approx(0.0505, abs=1e-9)


@pytest.mark.asyncio
async def test_place_ioc_taker_missing_mark_falls_back_to_coarse():
    """No mark_price in the ticker payload → fall back to coarse tick
    (defensive — matches pre-O-4 behaviour)."""
    rest = MagicMock()
    rest.get_ticker = AsyncMock(return_value={
        "best_ask_price": 0.1003,
        "best_bid_price": 0.1002,
        # mark_price intentionally missing
    })
    rest.place_order = AsyncMock(return_value={
        "_stub": True,
        "order": {
            "order_id": "stub_order",
            "order_state": "filled",
            "filled_amount": 1.0,
            "average_price": 0.1003,
        },
    })
    manager = OrderManager(rest=rest)

    await manager.place_ioc_taker(
        instrument_name="BTC-30APR26-60000-C",
        side="buy",
        amount=1.0,
        max_slippage_pct=0.5,
    )

    place_order_call = rest.place_order.call_args
    submitted_price = float(place_order_call.kwargs["price"])
    # Same as the boundary test — coarse tick produces 0.1010
    assert submitted_price == pytest.approx(0.1010, abs=1e-9)
