"""Tests for H3 IOC residual grid-rounding (2026-05-06).

Why this exists
---------------

Before H3, the 3-tier IOC ladder passed
``residual = amount - r1.filled_amount`` straight into tier-2 without
rounding. Floating-point drift across stages or future support for
instruments with non-0.1 grids could submit a non-grid residual,
which Deribit rejects with ``min_size``/increment errors. Three
consecutive rejected attempts trip the kill-switch on a moving market.

These tests pin the H3 contract:

1. A residual that rounds DOWN to an exact grid amount is rounded
   before submission.
2. A residual smaller than the minimum contract size (0.1 BTC) is
   dropped — the ladder exits cleanly at tier-1's result rather than
   shipping a guaranteed-rejection tier-2 call.
3. A grid-aligned residual is passed through unchanged.
4. The same rounding applies to the tier-3 residual.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from deribit_paper.execution.order_manager import OrderManager
from deribit_paper.risk.dry_run import DryRunState


@pytest.fixture(autouse=True)
def _reset_dry_run():
    DryRunState().reset()
    yield
    DryRunState().reset()


def _ticker_btc() -> dict:
    return {"best_ask_price": 0.05, "best_bid_price": 0.0495}


def _stub_fill(*, amount: float, filled_amount: float, price: float) -> dict:
    return {
        "_stub": True,
        "order": {
            "order_id": f"stub_{filled_amount:.4f}",
            "order_state": "filled" if filled_amount >= amount - 1e-9 else "cancelled",
            "instrument_name": "BTC-OPT",
            "direction": "buy",
            "amount": amount,
            "filled_amount": filled_amount,
            "average_price": price if filled_amount > 0 else 0.0,
            "order_type": "limit",
            "creation_timestamp": 0,
        },
        "trades": [],
    }


@pytest.mark.asyncio
async def test_residual_below_min_contract_exits_at_tier1():
    """Tier-1 fills 0.95 of 1.0 → raw residual 0.05 BTC, below min
    contract 0.1 BTC → engine MUST exit at tier-1 instead of submitting
    a 0.05 BTC tier-2 call (which Deribit would reject)."""
    rest = AsyncMock()
    rest.get_ticker = AsyncMock(return_value=_ticker_btc())
    rest.place_order = AsyncMock(side_effect=[
        _stub_fill(amount=1.0, filled_amount=0.95, price=0.0505),
    ])
    manager = OrderManager(rest=rest)

    result = await manager.place_ioc_taker(
        instrument_name="BTC-OPT",
        side="buy", amount=1.0,
        max_slippage_pct=0.5,
        tiered=True,
    )

    # Only tier-1 fired. 0.05 BTC residual was correctly NOT submitted.
    assert rest.place_order.call_count == 1
    assert result.filled_amount == pytest.approx(0.95)


@pytest.mark.asyncio
async def test_residual_with_floating_point_drift_rounds_to_grid():
    """Tier-1 fills 0.4 of 1.0; raw residual = 1.0 - 0.4 = 0.6 (clean
    grid). With FP drift, simulate tier-1 returning 0.40000001 → raw
    residual 0.59999999 → must be rounded to 0.6 before tier-2."""
    rest = AsyncMock()
    rest.get_ticker = AsyncMock(return_value=_ticker_btc())
    rest.place_order = AsyncMock(side_effect=[
        _stub_fill(amount=1.0, filled_amount=0.40000001, price=0.0505),
        _stub_fill(amount=0.6, filled_amount=0.6, price=0.0510),
    ])
    manager = OrderManager(rest=rest)

    await manager.place_ioc_taker(
        instrument_name="BTC-OPT",
        side="buy", amount=1.0,
        max_slippage_pct=0.5,
        tiered=True,
    )

    # Tier-2 was called with rounded amount 0.6, NOT 0.59999999.
    tier2_call_kwargs = rest.place_order.call_args_list[1].kwargs
    assert tier2_call_kwargs["amount"] == pytest.approx(0.6, abs=1e-6)


@pytest.mark.asyncio
async def test_grid_aligned_residual_passed_through():
    """A residual already on the 0.1-BTC grid is unchanged."""
    rest = AsyncMock()
    rest.get_ticker = AsyncMock(return_value=_ticker_btc())
    rest.place_order = AsyncMock(side_effect=[
        _stub_fill(amount=1.0, filled_amount=0.3, price=0.0505),
        _stub_fill(amount=0.7, filled_amount=0.7, price=0.0510),
    ])
    manager = OrderManager(rest=rest)

    await manager.place_ioc_taker(
        instrument_name="BTC-OPT",
        side="buy", amount=1.0,
        max_slippage_pct=0.5,
        tiered=True,
    )

    tier2_call_kwargs = rest.place_order.call_args_list[1].kwargs
    assert tier2_call_kwargs["amount"] == pytest.approx(0.7)


@pytest.mark.asyncio
async def test_tier3_residual_also_rounded_to_grid():
    """Same rounding contract applies to the tier-2 → tier-3 transition."""
    rest = AsyncMock()
    rest.get_ticker = AsyncMock(return_value=_ticker_btc())
    rest.place_order = AsyncMock(side_effect=[
        # tier-1: partial 0.5 of 2.0
        _stub_fill(amount=2.0, filled_amount=0.5, price=0.0505),
        # tier-2: partial 0.7 of 1.5; raw tier-3 residual = 0.79999... drift
        _stub_fill(amount=1.5, filled_amount=0.7000001, price=0.0510),
        # tier-3: receives rounded amount
        _stub_fill(amount=0.8, filled_amount=0.8, price=0.0512),
    ])
    manager = OrderManager(rest=rest)

    await manager.place_ioc_taker(
        instrument_name="BTC-OPT",
        side="buy", amount=2.0,
        max_slippage_pct=0.5,
        tiered=True,
    )

    tier3_call_kwargs = rest.place_order.call_args_list[2].kwargs
    # tier-3 residual = 1.5 - 0.7000001 = 0.7999999 → rounded to 0.8
    assert tier3_call_kwargs["amount"] == pytest.approx(0.8, abs=1e-6)


@pytest.mark.asyncio
async def test_zero_residual_no_tier2_submitted():
    """Tier-1 full fill → residual 0 → exit cleanly. Sanity check that
    H3 didn't break the existing exit path."""
    rest = AsyncMock()
    rest.get_ticker = AsyncMock(return_value=_ticker_btc())
    rest.place_order = AsyncMock(side_effect=[
        _stub_fill(amount=1.0, filled_amount=1.0, price=0.0505),
    ])
    manager = OrderManager(rest=rest)

    result = await manager.place_ioc_taker(
        instrument_name="BTC-OPT",
        side="buy", amount=1.0,
        max_slippage_pct=0.5,
        tiered=True,
    )

    assert rest.place_order.call_count == 1
    assert result.is_filled
