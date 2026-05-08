"""O-9 (parity-r3 cycle 2 — 2026-05-08) regression tests.

On WebSocket reconnect, the engine queries
``rest.get_open_orders_by_currency`` to reconcile the venue's view of
open orders with the engine's. Pre-fix the WS reconnect callback only
fired a Telegram message; orders that timed out during a WS gap
remained on the venue with no listener.

The reconcile uses an age-gate (``ws_reconcile_orphan_age_seconds``,
default 300s) so in-flight orders aren't cancelled by mistake — only
orders older than the threshold (clearly stale from a prior session)
are cleaned up.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from deribit_paper.engine import TradingEngine


def _make_engine_stub(
    *,
    paper_mode: bool = False,
    venue_orders: list[dict] | None = None,
    rest_raises: bool = False,
    orphan_age_seconds: float = 300.0,
) -> TradingEngine:
    """Build a minimal engine stub that satisfies
    ``_reconcile_open_orders_on_reconnect``.
    """
    eng = TradingEngine.__new__(TradingEngine)
    eng.config = SimpleNamespace(
        paper_mode=paper_mode,
        currency="BTC",
        ws_reconcile_orphan_age_seconds=orphan_age_seconds,
    )
    eng.rest = MagicMock()
    if rest_raises:
        eng.rest.get_open_orders_by_currency = AsyncMock(
            side_effect=ConnectionError("rest down"),
        )
    else:
        eng.rest.get_open_orders_by_currency = AsyncMock(
            return_value=venue_orders or [],
        )
    eng.order_manager = MagicMock()
    eng.order_manager.cancel = AsyncMock(return_value={"order_state": "cancelled"})
    eng.telegram = MagicMock()
    eng.telegram.send = MagicMock()
    return eng


# ── Empty / paper / failure modes ─────────────────────────────────────


@pytest.mark.asyncio
async def test_reconcile_skipped_in_paper_mode():
    eng = _make_engine_stub(paper_mode=True)
    await eng._reconcile_open_orders_on_reconnect()
    eng.rest.get_open_orders_by_currency.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_no_op_when_no_open_orders():
    eng = _make_engine_stub(venue_orders=[])
    await eng._reconcile_open_orders_on_reconnect()
    eng.rest.get_open_orders_by_currency.assert_called_once()
    eng.order_manager.cancel.assert_not_called()
    # Telegram only fires on cancellation
    eng.telegram.send.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_handles_rest_failure_gracefully():
    """REST timeout / connection error → log + Telegram alert, don't crash."""
    eng = _make_engine_stub(rest_raises=True)
    # Should not raise
    await eng._reconcile_open_orders_on_reconnect()
    eng.order_manager.cancel.assert_not_called()
    # One alert sent for the failure
    assert eng.telegram.send.call_count == 1
    assert "failed to query open orders" in eng.telegram.send.call_args.args[1]


# ── Age-gated cleanup ─────────────────────────────────────────────────


def _make_venue_order(order_id: str, age_seconds: float, label: str = "x") -> dict:
    now_ms = pd.Timestamp.now(tz="UTC").value // 10**6
    return {
        "order_id": order_id,
        "instrument_name": "BTC-30APR26-60000-C",
        "label": label,
        "creation_timestamp": now_ms - int(age_seconds * 1000),
    }


@pytest.mark.asyncio
async def test_reconcile_leaves_in_flight_orders_alone():
    """Orders younger than the orphan age threshold are NOT cancelled —
    they're presumed to be the engine's own active polls."""
    young_order = _make_venue_order("oid_young", age_seconds=10.0)
    eng = _make_engine_stub(
        venue_orders=[young_order],
        orphan_age_seconds=300.0,
    )
    await eng._reconcile_open_orders_on_reconnect()
    eng.order_manager.cancel.assert_not_called()


@pytest.mark.asyncio
async def test_reconcile_cancels_stale_orders():
    """Orders older than the threshold are stale orphans → cancel them."""
    stale_order = _make_venue_order("oid_stale", age_seconds=600.0, label="prior_session")
    eng = _make_engine_stub(
        venue_orders=[stale_order],
        orphan_age_seconds=300.0,
    )
    await eng._reconcile_open_orders_on_reconnect()
    eng.order_manager.cancel.assert_called_once_with("oid_stale")
    # Telegram alert sent
    assert eng.telegram.send.call_count == 1
    sent_msg = eng.telegram.send.call_args.args[1]
    assert "1 stale order" in sent_msg


@pytest.mark.asyncio
async def test_reconcile_mixes_young_and_stale():
    """Mixed list: young left alone, stale cancelled."""
    young = _make_venue_order("oid_young", age_seconds=15.0)
    stale = _make_venue_order("oid_stale_1", age_seconds=900.0)
    very_stale = _make_venue_order("oid_stale_2", age_seconds=3600.0)
    eng = _make_engine_stub(
        venue_orders=[young, stale, very_stale],
        orphan_age_seconds=300.0,
    )
    await eng._reconcile_open_orders_on_reconnect()
    cancelled_ids = [c.args[0] for c in eng.order_manager.cancel.call_args_list]
    assert "oid_young" not in cancelled_ids
    assert "oid_stale_1" in cancelled_ids
    assert "oid_stale_2" in cancelled_ids
    assert len(cancelled_ids) == 2


@pytest.mark.asyncio
async def test_reconcile_swallows_individual_cancel_failures():
    """If a cancel fails, the next stale order is still cancelled and
    the reconcile completes."""
    stale_a = _make_venue_order("oid_stale_a", age_seconds=900.0)
    stale_b = _make_venue_order("oid_stale_b", age_seconds=900.0)
    eng = _make_engine_stub(
        venue_orders=[stale_a, stale_b],
        orphan_age_seconds=300.0,
    )
    eng.order_manager.cancel = AsyncMock(side_effect=[
        ConnectionError("transient"),
        {"order_state": "cancelled"},
    ])
    # Should not raise
    await eng._reconcile_open_orders_on_reconnect()
    # Both cancels attempted
    assert eng.order_manager.cancel.call_count == 2


@pytest.mark.asyncio
async def test_reconcile_skips_orders_with_zero_creation_timestamp():
    """Defensive: orders with creation_timestamp == 0 (malformed
    response) are treated as in-flight, not stale orphans."""
    malformed = {
        "order_id": "oid_malformed",
        "instrument_name": "BTC-30APR26-60000-C",
        "label": "x",
        "creation_timestamp": 0,
    }
    eng = _make_engine_stub(venue_orders=[malformed])
    await eng._reconcile_open_orders_on_reconnect()
    eng.order_manager.cancel.assert_not_called()
