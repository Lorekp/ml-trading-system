"""Tests for OrderManager — place/retry/fill-polling/cancel."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from deribit_paper.client.errors import (
    DeribitOrderRejected,
    DeribitRateLimitError,
)
from deribit_paper.execution.order_manager import OrderManager, OrderResult
from deribit_paper.risk.dry_run import DryRunState, set_dry_run


@pytest.fixture(autouse=True)
def _reset_dry_run():
    DryRunState().reset()
    yield
    DryRunState().reset()


def _stub_rest_dry_run_fill():
    """Build a mock REST client that returns a dry-run stub fill."""
    rest = AsyncMock()
    rest.place_order = AsyncMock(return_value={
        "_stub": True,
        "order": {
            "order_id": "stub_abc123",
            "order_state": "filled",
            "instrument_name": "BTC-PERPETUAL",
            "direction": "buy",
            "amount": 1000.0,
            "filled_amount": 1000.0,
            "average_price": 50_000.0,
            "order_type": "market",
            "creation_timestamp": 0,
        },
        "trades": [],
    })
    return rest


@pytest.mark.asyncio
async def test_dry_run_short_circuits_without_polling():
    """In DRY_RUN mode, place_and_wait returns instantly with the stub fill."""
    rest = _stub_rest_dry_run_fill()
    manager = OrderManager(rest=rest)

    result = await manager.place_and_wait(
        instrument_name="BTC-PERPETUAL",
        side="buy",
        amount=1000.0,
        order_type="market",
    )
    assert result.is_dry_run is True
    assert result.is_filled is True
    assert result.order_id == "stub_abc123"
    assert result.filled_amount == 1000.0
    # No polling in dry-run mode
    rest._request.assert_not_called()


@pytest.mark.asyncio
async def test_duplicate_client_order_id_is_rejected():
    """IdempotencyGuard prevents sending the same label twice."""
    rest = _stub_rest_dry_run_fill()
    manager = OrderManager(rest=rest)

    # First send succeeds
    result1 = await manager.place_and_wait(
        instrument_name="BTC-PERPETUAL",
        side="buy",
        amount=1000.0,
        order_type="market",
        label="my_unique_order",
    )
    assert result1.is_filled is True
    # Second send with the same label is rejected locally
    result2 = await manager.place_and_wait(
        instrument_name="BTC-PERPETUAL",
        side="buy",
        amount=1000.0,
        order_type="market",
        label="my_unique_order",
    )
    assert result2.state == "rejected"
    assert "duplicate" in (result2.error or "").lower()
    # Only one real submit attempt
    assert rest.place_order.call_count == 1


@pytest.mark.asyncio
async def test_rate_limit_retries_then_succeeds():
    rest = AsyncMock()
    # First call raises rate limit, second returns success
    rest.place_order = AsyncMock(side_effect=[
        DeribitRateLimitError("too_many", code=10_028),
        {
            "_stub": True,
            "order": {
                "order_id": "stub_retry",
                "order_state": "filled",
                "filled_amount": 100.0,
                "average_price": 50_000.0,
                "instrument_name": "BTC-PERPETUAL",
                "direction": "buy",
                "amount": 100.0,
                "order_type": "market",
                "creation_timestamp": 0,
            },
            "trades": [],
        },
    ])
    manager = OrderManager(
        rest=rest,
        retry_backoff_seconds=0.01,  # fast test
        max_rejection_retries=2,
    )
    result = await manager.place_and_wait(
        instrument_name="BTC-PERPETUAL",
        side="buy",
        amount=100.0,
        order_type="market",
    )
    assert result.is_filled is True
    assert rest.place_order.call_count == 2


@pytest.mark.asyncio
async def test_rate_limit_exhausted_returns_rejected():
    rest = AsyncMock()
    rest.place_order = AsyncMock(side_effect=DeribitRateLimitError("too_many", code=10_028))
    manager = OrderManager(
        rest=rest,
        retry_backoff_seconds=0.01,
        max_rejection_retries=2,
    )
    result = await manager.place_and_wait(
        instrument_name="BTC-PERPETUAL",
        side="buy",
        amount=100.0,
        order_type="market",
    )
    assert result.state == "rejected"
    assert result.is_filled is False
    # max_rejection_retries=2 → 3 total attempts
    assert rest.place_order.call_count == 3


@pytest.mark.asyncio
async def test_non_retryable_rejection_aborts_immediately():
    rest = AsyncMock()
    rest.place_order = AsyncMock(side_effect=DeribitOrderRejected(
        "insufficient_funds", code=10_002, retryable=False,
    ))
    manager = OrderManager(rest=rest)
    result = await manager.place_and_wait(
        instrument_name="BTC-PERPETUAL",
        side="buy",
        amount=100.0,
        order_type="market",
    )
    assert result.state == "rejected"
    assert "insufficient_funds" in (result.error or "")
    assert rest.place_order.call_count == 1  # no retries


@pytest.mark.asyncio
async def test_live_fill_polling_returns_filled_state():
    """With DRY_RUN off, live order is tracked by polling get_order_state."""
    set_dry_run(False, confirmation_token="test")

    rest = AsyncMock()
    # Submit returns an OPEN order (no _stub flag = not dry-run)
    rest.place_order = AsyncMock(return_value={
        "order": {
            "order_id": "real_456",
            "order_state": "open",
            "filled_amount": 0.0,
            "average_price": 0.0,
            "instrument_name": "BTC-PERPETUAL",
            "direction": "buy",
            "amount": 500.0,
        },
        "trades": [],
    })
    # Polling: first call returns open, second returns filled
    rest._request = AsyncMock(side_effect=[
        {"order_state": "open", "filled_amount": 0.0, "average_price": 0.0},
        {"order_state": "filled", "filled_amount": 500.0, "average_price": 50_100.0},
    ])

    manager = OrderManager(rest=rest, fill_poll_seconds=0.01)
    result = await manager.place_and_wait(
        instrument_name="BTC-PERPETUAL",
        side="buy",
        amount=500.0,
        order_type="limit",
        price=50_100.0,
        timeout_seconds=10.0,
    )
    assert result.is_dry_run is False
    assert result.is_filled is True
    assert result.filled_amount == 500.0
    assert result.average_price == 50_100.0


@pytest.mark.asyncio
async def test_live_fill_polling_times_out_and_cancels():
    """If order stays open past timeout, poller cancels + returns cancelled."""
    set_dry_run(False, confirmation_token="test")

    rest = AsyncMock()
    rest.place_order = AsyncMock(return_value={
        "order": {
            "order_id": "real_789",
            "order_state": "open",
            "filled_amount": 0.0,
            "average_price": 0.0,
            "instrument_name": "BTC-30APR26-60000-C",
            "direction": "buy",
            "amount": 0.1,
        },
        "trades": [],
    })
    rest._request = AsyncMock(return_value={
        "order_state": "open", "filled_amount": 0.0, "average_price": 0.0,
    })
    rest.cancel_order = AsyncMock(return_value={"order_state": "cancelled"})

    manager = OrderManager(rest=rest, fill_poll_seconds=0.01)
    result = await manager.place_and_wait(
        instrument_name="BTC-30APR26-60000-C",
        side="buy",
        amount=0.1,
        order_type="limit",
        price=0.05,
        timeout_seconds=0.05,  # fast test
    )
    assert result.state == "cancelled"
    assert result.is_filled is False
    assert "timeout" in (result.error or "").lower()
    rest.cancel_order.assert_called_once()
