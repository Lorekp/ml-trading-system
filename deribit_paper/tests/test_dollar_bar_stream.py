"""Tests for :class:`DollarBarSpotStream` — REST polling + aggregator drive.

F3 Tier 2 (2026-05-07): mocks the Binance Spot REST client so we don't
hit live Binance, then asserts the stream feeds the aggregator
correctly under the standard contract — pagination, bar production,
staleness, connection-state tracking, and graceful shutdown.

Mocking pattern mirrors ``deribit_paper/tests/test_vpin_rest_polling.py``:
``MagicMock(spec=BinanceRESTClient)`` with ``fetch_agg_trades`` as
``AsyncMock(side_effect=itertools.chain([page1, page2], itertools.repeat([])))``.
"""

from __future__ import annotations

import asyncio
import itertools
import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from deribit_paper.client.binance_rest import BinanceRESTClient
from deribit_paper.features.dollar_bar import DollarBarAggregator
from deribit_paper.features.dollar_bar_stream import DollarBarSpotStream


def _make_trade(
    agg_id: int,
    *,
    price: float = 50_000.0,
    qty: float = 0.001,
    is_buyer_maker: bool = False,
    ts_ms: int | None = None,
) -> dict:
    """Construct a Binance Spot aggTrades response entry."""
    return {
        "a": agg_id,
        "p": str(price),
        "q": str(qty),
        "f": agg_id,
        "l": agg_id,
        "T": int(ts_ms if ts_ms is not None else agg_id * 1000),
        "m": is_buyer_maker,
    }


def _scripted_pages(*pages: list[dict]):
    """Iterator emitting ``pages`` in order, then ``[]`` forever.

    Avoids ``StopAsyncIteration`` when the poll loop polls faster than
    the script provides — the empty pages mimic a quiet venue.
    """
    return itertools.chain(list(pages), itertools.repeat([]))


# ── 1. Basic flow: trades drive the aggregator and bars are emitted ───────


@pytest.mark.asyncio
async def test_basic_flow_trades_emit_bars():
    """Page of large trades → at least one bar is emitted via the
    aggregator's on_bar_close callback, latest_bar reflects it.

    Trades sized so cumulative dollar volume crosses the $1000 test
    threshold within the first page.
    """
    rest_client = MagicMock(spec=BinanceRESTClient)
    rest_client.fetch_agg_trades = AsyncMock(
        side_effect=_scripted_pages([
            _make_trade(i, price=50_000.0, qty=0.005)  # $250 each
            for i in range(1, 8)  # 7 trades * $250 = $1750
        ]),
    )

    captured_bars: list = []
    agg = DollarBarAggregator(
        threshold=1_000.0,
        on_bar_close=captured_bars.append,
    )
    stream = DollarBarSpotStream(
        rest_client, agg,
        symbol="BTCUSDT",
        poll_seconds=0.001,
        max_stale_seconds=60.0,
    )

    async def _stop():
        await asyncio.sleep(0.05)
        stream.stop()

    asyncio.create_task(_stop())
    await stream.run_forever()

    # 7 trades * $250 → 1 boundary crossing ($1000) → 1 bar emitted.
    # Bar 0 has trades [0, 4) (offline-shifted convention from Tier 1).
    assert len(captured_bars) >= 1
    assert stream.n_bars_received == len(captured_bars)
    assert stream.latest_bar is captured_bars[-1]
    assert stream.trades_processed == 7


# ── 2. from_id pagination across two polls ────────────────────────────────


@pytest.mark.asyncio
async def test_from_id_pagination():
    """After page 1 with max(a)=10, second poll passes ``from_id=11``."""
    rest_client = MagicMock(spec=BinanceRESTClient)
    page1 = [_make_trade(i, qty=0.0001) for i in range(1, 11)]
    page2 = [_make_trade(i, qty=0.0001) for i in range(11, 16)]
    rest_client.fetch_agg_trades = AsyncMock(
        side_effect=_scripted_pages(page1, page2),
    )

    agg = DollarBarAggregator(threshold=1e9)  # huge threshold → no bars emit
    stream = DollarBarSpotStream(
        rest_client, agg,
        symbol="BTCUSDT",
        poll_seconds=0.001,
    )

    async def _stop():
        await asyncio.sleep(0.05)
        stream.stop()

    asyncio.create_task(_stop())
    await stream.run_forever()

    calls = rest_client.fetch_agg_trades.await_args_list
    # First call: from_id=None
    assert calls[0].kwargs.get("from_id") is None
    # Second call: from_id = max(page1.a) + 1 = 11
    assert calls[1].kwargs.get("from_id") == 11
    if len(calls) >= 3:
        # Third call: from_id = max(page2.a) + 1 = 16
        assert calls[2].kwargs.get("from_id") == 16
    # Last seen aggTradeId == 15
    assert stream._last_agg_trade_id == 15
    assert stream.trades_processed == 15


# ── 3. latest_bar advances as new bars are emitted ────────────────────────


@pytest.mark.asyncio
async def test_latest_bar_advances():
    """latest_bar updates to the most-recently-closed bar across pages."""
    rest_client = MagicMock(spec=BinanceRESTClient)
    # Page 1: trades that close 1 bar
    # Page 2: trades that close another bar
    page1 = [
        _make_trade(i, price=50_000.0, qty=0.005, ts_ms=i * 1000)
        for i in range(1, 8)  # crosses $1000 threshold once
    ]
    page2 = [
        _make_trade(i, price=50_010.0, qty=0.005, ts_ms=i * 1000)
        for i in range(8, 15)  # crosses again
    ]
    rest_client.fetch_agg_trades = AsyncMock(
        side_effect=_scripted_pages(page1, page2),
    )

    agg = DollarBarAggregator(threshold=1_000.0)
    stream = DollarBarSpotStream(
        rest_client, agg,
        symbol="BTCUSDT",
        poll_seconds=0.001,
    )

    async def _stop():
        await asyncio.sleep(0.08)
        stream.stop()

    asyncio.create_task(_stop())
    await stream.run_forever()

    # Should have emitted at least 2 bars across the 2 pages.
    assert stream.n_bars_received >= 2
    assert stream.latest_bar is not None
    # Latest bar's close should reflect page-2's prices (50_010), not page-1's.
    assert stream.latest_bar.close == 50_010.0


# ── 4. is_stale fires after timeout when no trades arrive ──────────────────


@pytest.mark.asyncio
async def test_is_stale_after_timeout():
    """is_stale flips True once max_stale_seconds elapses with no updates."""
    rest_client = MagicMock(spec=BinanceRESTClient)
    rest_client.fetch_agg_trades = AsyncMock(return_value=[])

    agg = DollarBarAggregator(threshold=1_000.0)
    stream = DollarBarSpotStream(
        rest_client, agg,
        symbol="BTCUSDT",
        poll_seconds=0.005,
        max_stale_seconds=0.05,
    )

    # Initially not stale (within grace).
    assert stream.is_stale is False

    async def _stop():
        await asyncio.sleep(0.10)  # > max_stale_seconds
        stream.stop()

    asyncio.create_task(_stop())
    await stream.run_forever()

    # After running > max_stale_seconds with no updates, is_stale True.
    assert stream.is_stale is True


# ── 5. is_connected flips True after first successful poll ────────────────


@pytest.mark.asyncio
async def test_is_connected_flips_on_first_poll():
    """is_connected is False initially, flips True after first fetch."""
    rest_client = MagicMock(spec=BinanceRESTClient)
    rest_client.fetch_agg_trades = AsyncMock(
        side_effect=_scripted_pages([_make_trade(1, qty=0.0001)]),
    )

    agg = DollarBarAggregator(threshold=1e9)
    stream = DollarBarSpotStream(
        rest_client, agg,
        symbol="BTCUSDT",
        poll_seconds=0.001,
    )
    assert stream.is_connected is False

    snapshots: list[bool] = []

    async def _watcher_then_stop():
        for _ in range(8):
            snapshots.append(stream.is_connected)
            await asyncio.sleep(0.005)
        stream.stop()

    asyncio.create_task(_watcher_then_stop())
    await stream.run_forever()

    # At some point during the run is_connected was True.
    assert any(snapshots), (
        f"is_connected never flipped True; snapshots={snapshots}"
    )


# ── 6. stop() ends run_forever gracefully ──────────────────────────────────


@pytest.mark.asyncio
async def test_stop_ends_run_forever():
    """Calling stop() makes run_forever return without raising."""
    rest_client = MagicMock(spec=BinanceRESTClient)
    rest_client.fetch_agg_trades = AsyncMock(return_value=[])

    agg = DollarBarAggregator(threshold=1_000.0)
    stream = DollarBarSpotStream(
        rest_client, agg,
        symbol="BTCUSDT",
        poll_seconds=0.005,
    )

    started = time.time()

    async def _stop():
        await asyncio.sleep(0.03)
        stream.stop()

    asyncio.create_task(_stop())
    # Should return cleanly, not hang. Wrap in wait_for as a safety belt:
    # if stop() is broken the test should fail fast rather than time out
    # the whole pytest run.
    await asyncio.wait_for(stream.run_forever(), timeout=2.0)
    elapsed = time.time() - started
    # Sanity: returned soon after stop() was signaled (well under 1s).
    assert elapsed < 1.0


# ── 7. Defensive — invalid construction args raise ─────────────────────────


def test_invalid_construction_raises():
    rest_client = MagicMock(spec=BinanceRESTClient)
    agg = DollarBarAggregator(threshold=1.0)
    with pytest.raises(ValueError, match="poll_seconds must be positive"):
        DollarBarSpotStream(
            rest_client, agg, symbol="BTCUSDT", poll_seconds=0.0,
        )
    with pytest.raises(ValueError, match="max_stale_seconds must be positive"):
        DollarBarSpotStream(
            rest_client, agg, symbol="BTCUSDT", max_stale_seconds=0.0,
        )
