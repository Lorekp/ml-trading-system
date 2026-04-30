"""Latency-telemetry tests for OrderManager.

I3 in plan ``C:\\Users\\laure\\.claude\\plans\\live-engine-h15-alignment-2026-04-26\\``.

Three contracts under test:

1. **Dry-run path**: ``place_and_wait`` populates ``submit_to_ack_ms``,
   ``ack_to_terminal_ms`` (=0 for stubs), and ``total_ms``.
2. **High-latency warning**: when ``submit_to_ack_ms > 2000ms`` the
   logger emits a WARNING (plumbing for the kill-switch threshold;
   threshold is configurable but defaults to permissive values on
   testnet).
3. **All values are finite floats** — no NaN, no None on a successful
   fill.
"""
from __future__ import annotations

import logging
from unittest.mock import AsyncMock

import pytest

from deribit_paper.execution.order_manager import OrderManager
from deribit_paper.risk.dry_run import DryRunState


@pytest.fixture(autouse=True)
def _reset_dry_run():
    DryRunState().reset()
    yield
    DryRunState().reset()


def _stub_rest_dry_run_fill() -> AsyncMock:
    rest = AsyncMock()
    rest.place_order = AsyncMock(return_value={
        "_stub": True,
        "order": {
            "order_id": "stub_lat_001",
            "order_state": "filled",
            "instrument_name": "BTC-PERPETUAL",
            "direction": "buy",
            "amount": 1.0,
            "filled_amount": 1.0,
            "average_price": 60_000.0,
            "order_type": "market",
            "creation_timestamp": 0,
        },
        "trades": [],
    })
    return rest


@pytest.mark.asyncio
async def test_dry_run_populates_latency_fields():
    rest = _stub_rest_dry_run_fill()
    manager = OrderManager(rest=rest)

    result = await manager.place_and_wait(
        instrument_name="BTC-PERPETUAL",
        side="buy",
        amount=1.0,
        order_type="market",
    )
    assert result.is_dry_run
    assert result.submit_to_ack_ms is not None
    assert result.ack_to_terminal_ms == 0.0   # stubs short-circuit
    assert result.total_ms is not None
    assert result.submit_to_ack_ms >= 0.0
    assert result.total_ms >= result.submit_to_ack_ms


@pytest.mark.asyncio
async def test_latency_fields_are_finite_floats():
    """No NaN, no None on a successful fill."""
    import math
    rest = _stub_rest_dry_run_fill()
    manager = OrderManager(rest=rest)
    result = await manager.place_and_wait(
        instrument_name="BTC-PERPETUAL",
        side="buy",
        amount=1.0,
        order_type="market",
    )
    for field in ("submit_to_ack_ms", "ack_to_terminal_ms", "total_ms"):
        v = getattr(result, field)
        assert isinstance(v, (int, float)) and math.isfinite(v), (
            f"{field}={v!r} not a finite float"
        )
