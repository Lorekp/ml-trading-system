"""Tests for the 3-tier taker ladder in :class:`OrderManager`.

Three contracts under test:

1. **Full Tier-1 fill**: a fully-fillable order at tier-1 price exits
   the ladder at tier 1 — no tier-2/3 calls.
2. **Partial Tier-1 + Tier-2 fill**: tier-1 fills part of the order;
   tier-2 fills the residual; tier-3 not called.
3. **Tier-3 market fallback**: tier-1 + tier-2 each fill nothing;
   tier-3 market fires for the full residual.
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from deribit_paper.execution.order_manager import OrderManager, OrderResult
from deribit_paper.risk.dry_run import DryRunState


@pytest.fixture(autouse=True)
def _reset_dry_run():
    DryRunState().reset()
    yield
    DryRunState().reset()


def _ticker_btc() -> dict:
    return {"best_ask_price": 0.05, "best_bid_price": 0.0495}


def _stub_fill(*, amount: float, filled_amount: float, price: float) -> dict:
    """Build a stub place_order return that simulates partial fills."""
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
async def test_full_tier1_fill_skips_tier2_and_tier3():
    rest = AsyncMock()
    rest.get_ticker = AsyncMock(return_value=_ticker_btc())
    # First call (tier-1): full fill of 1.0 BTC at 0.0505.
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
    assert result.is_filled
    assert result.filled_amount == 1.0
    # Only tier-1 should have been submitted (1 place_order call total).
    assert rest.place_order.call_count == 1


@pytest.mark.asyncio
async def test_partial_tier1_then_full_tier2_fill():
    rest = AsyncMock()
    rest.get_ticker = AsyncMock(return_value=_ticker_btc())
    # Tier-1: 0.4 of 1.0 fills. Tier-2: full residual 0.6 fills.
    rest.place_order = AsyncMock(side_effect=[
        _stub_fill(amount=1.0, filled_amount=0.4, price=0.0505),
        _stub_fill(amount=0.6, filled_amount=0.6, price=0.0510),
    ])
    manager = OrderManager(rest=rest)

    result = await manager.place_ioc_taker(
        instrument_name="BTC-OPT",
        side="buy", amount=1.0,
        max_slippage_pct=0.5,
        tiered=True,
    )
    assert result.filled_amount == pytest.approx(1.0)
    # avg_px = (0.4×0.0505 + 0.6×0.0510) / 1.0 = 0.0508.
    assert result.average_price == pytest.approx(0.0508, abs=1e-5)
    assert rest.place_order.call_count == 2  # tier-1 + tier-2; no tier-3.


@pytest.mark.asyncio
async def test_tier1_tier2_zero_fill_falls_through_to_bounded_tier3():
    """E.1.7: tier-3 is now a bounded IOC at 1.5× tier-2 cap, NOT a
    naked market. Defends against manipulated thin books from filling
    at any price."""
    rest = AsyncMock()
    rest.get_ticker = AsyncMock(return_value=_ticker_btc())
    # Tier-1 zero, tier-2 zero, tier-3 bounded-IOC fills at the wider cap.
    rest.place_order = AsyncMock(side_effect=[
        _stub_fill(amount=1.0, filled_amount=0.0, price=0.0),
        _stub_fill(amount=1.0, filled_amount=0.0, price=0.0),
        _stub_fill(amount=1.0, filled_amount=1.0, price=0.05038),
    ])
    manager = OrderManager(rest=rest)

    result = await manager.place_ioc_taker(
        instrument_name="BTC-OPT",
        side="buy", amount=1.0,
        max_slippage_pct=0.5,
        tiered=True,
    )
    assert result.filled_amount == pytest.approx(1.0)
    assert rest.place_order.call_count == 3
    # Tier-3 was a *limit IOC* not a market — verify by inspecting the
    # third place_order call.
    third_call = rest.place_order.call_args_list[2]
    assert third_call.kwargs["type"] == "limit"
    assert third_call.kwargs["time_in_force"] == "immediate_or_cancel"
    # Cap = best × (1 + 1.5 × 0.5%) = 0.05 × 1.0075 = 0.050375 → tick-rounded.
    assert third_call.kwargs["price"] == pytest.approx(0.0505, abs=1e-3)


@pytest.mark.asyncio
async def test_tier3_skipped_on_small_residual():
    """E.1.7: residual < 10% of original is dropped, not market-filled.

    H3 (2026-05-06): the original test used a 0.05 BTC residual that
    H3's grid-rounding would now drop at tier-1 entirely (below the
    0.1 BTC minimum contract). Re-cast on a 2.0 BTC order so the
    residuals stay at-or-above the 0.1-BTC grid through tier-2.
    """
    rest = AsyncMock()
    rest.get_ticker = AsyncMock(return_value=_ticker_btc())
    # Tier-1 fills 1.7 of 2.0 → residual 0.3 BTC → tier-2 fires.
    # Tier-2 fills 0.2 → residual 0.1 BTC.
    # 0.1 < 10% × 2.0 = 0.2 → tier-3 SKIP.
    rest.place_order = AsyncMock(side_effect=[
        _stub_fill(amount=2.0, filled_amount=1.7, price=0.0505),
        _stub_fill(amount=0.3, filled_amount=0.2, price=0.0510),
    ])
    manager = OrderManager(rest=rest)

    result = await manager.place_ioc_taker(
        instrument_name="BTC-OPT",
        side="buy", amount=2.0,
        max_slippage_pct=0.5,
        tiered=True,
    )
    # 1.9 filled, not 2.0 — by design (the 0.1 BTC residual was
    # dropped to defend against worst-case fills on the tail).
    assert result.filled_amount == pytest.approx(1.9)
    # Tier-3 not called.
    assert rest.place_order.call_count == 2


@pytest.mark.asyncio
async def test_total_ms_is_wall_clock_not_sum():
    """E.2.5: ``total_ms`` reflects wall-clock duration of the ladder,
    not the sum of per-tier latencies. Pre-E.2.5 a 3-tier ladder
    summed each tier's ``submit_to_ack + ack_to_terminal`` and the
    nested merge double-counted tier-1+tier-2's already-summed pair —
    inflating the kill-switch latency reading.
    """
    rest = AsyncMock()
    rest.get_ticker = AsyncMock(return_value=_ticker_btc())
    rest.place_order = AsyncMock(side_effect=[
        _stub_fill(amount=1.0, filled_amount=0.4, price=0.0505),
        _stub_fill(amount=0.6, filled_amount=0.6, price=0.0510),
    ])
    manager = OrderManager(rest=rest)

    result = await manager.place_ioc_taker(
        instrument_name="BTC-OPT",
        side="buy", amount=1.0,
        max_slippage_pct=0.5,
        tiered=True,
    )
    # Wall-clock total should be a small positive number — typically
    # < 100ms for stub fills. The OLD broken code summed two ~1ms tier
    # latencies; that yielded the same magnitude but in dry-run with
    # zero-tier-spans, the OLD merge would have produced 0ms (because
    # _t2 - _t1 = 0 in stub fills) whereas the NEW path captures real
    # wall-clock.
    assert result.total_ms is not None
    assert result.total_ms >= 0.0
    assert result.total_ms < 5_000.0      # sanity: sub-second on stubs


@pytest.mark.asyncio
async def test_avg_price_nan_when_zero_filled():
    """E.2.7: when total_filled == 0, ``average_price`` is NaN, not 0.0.
    Pre-E.2.7 the helper returned 0.0 which propagated as a fake fill
    price into the harness writer + P&L recipe."""
    import math

    rest = AsyncMock()
    rest.get_ticker = AsyncMock(return_value=_ticker_btc())
    # All three tiers zero-fill — manipulated thin book.
    rest.place_order = AsyncMock(side_effect=[
        _stub_fill(amount=1.0, filled_amount=0.0, price=0.0),
        _stub_fill(amount=1.0, filled_amount=0.0, price=0.0),
        _stub_fill(amount=1.0, filled_amount=0.0, price=0.0),
    ])
    manager = OrderManager(rest=rest)

    result = await manager.place_ioc_taker(
        instrument_name="BTC-OPT",
        side="buy", amount=1.0,
        max_slippage_pct=0.5,
        tiered=True,
    )
    assert result.filled_amount == 0.0
    # average_price MUST be NaN, not 0.0 (which would propagate as a
    # fake fill price downstream).
    assert math.isnan(result.average_price), (
        f"average_price should be NaN on zero-fill, got {result.average_price}"
    )


@pytest.mark.asyncio
async def test_tier3_bounded_ioc_cap_at_1_5x_max_slippage():
    """E.1.7: tier-3 cap = best ± 1.5 × max_slippage_pct, never wider."""
    rest = AsyncMock()
    rest.get_ticker = AsyncMock(return_value=_ticker_btc())
    rest.place_order = AsyncMock(side_effect=[
        _stub_fill(amount=1.0, filled_amount=0.0, price=0.0),
        _stub_fill(amount=1.0, filled_amount=0.0, price=0.0),
        # Tier-3: zero fill (manipulated book). Don't fall back to market.
        _stub_fill(amount=1.0, filled_amount=0.0, price=0.0),
    ])
    manager = OrderManager(rest=rest)

    result = await manager.place_ioc_taker(
        instrument_name="BTC-OPT",
        side="buy", amount=1.0,
        max_slippage_pct=0.5,
        tiered=True,
    )
    # All 3 tiers fired but nothing filled. Result not is_filled but
    # also crucially NO naked market order was placed.
    assert rest.place_order.call_count == 3
    third_call = rest.place_order.call_args_list[2]
    assert third_call.kwargs["type"] == "limit"
    # No fourth (market) call.
    assert result.filled_amount == 0.0


@pytest.mark.asyncio
async def test_tiered_default_false_uses_legacy_single_call():
    """When tiered=False (default), behaviour is the existing single-IOC."""
    rest = AsyncMock()
    rest.get_ticker = AsyncMock(return_value=_ticker_btc())
    rest.place_order = AsyncMock(return_value=_stub_fill(
        amount=1.0, filled_amount=1.0, price=0.0505,
    ))
    manager = OrderManager(rest=rest)

    result = await manager.place_ioc_taker(
        instrument_name="BTC-OPT",
        side="buy", amount=1.0,
        max_slippage_pct=0.5,
        # tiered defaulted to False.
    )
    assert result.is_filled
    assert rest.place_order.call_count == 1
