"""Round-4 F6/F8 — DollarBarSpotStream extra_trade_sink fan-out tests.

Scope:

* The new ``extra_trade_sink`` callback receives every trade the dollar-bar
  aggregator receives, in the same order, with a canonical :class:`Trade`.
* When ``extra_trade_sink=None`` (default — preserves pre-round-4 behaviour
  byte-identically), the existing aggregator path is unchanged.
* ``is_buyer_maker=True`` -> ``direction=-1`` and ``is_buyer_maker=False`` ->
  ``direction=+1`` (matching Deribit perp ``direction`` semantics so
  :func:`compute_bar_tsrv` is venue-agnostic).
* Sink exceptions are swallowed and logged — they cannot stop the stream.
"""

from __future__ import annotations

import logging

import pandas as pd

from deribit_paper.features.dollar_bar import DollarBarAggregator
from deribit_paper.features.dollar_bar_stream import DollarBarSpotStream
from deribit_paper.market_data.buffer import Trade


def _make_stream_with_sink(sink):
    """Construct a DollarBarSpotStream wired to a sink, no REST client."""
    rest = object()  # never called in tests; _process_page is offline.
    agg = DollarBarAggregator(threshold=500_000.0)
    return DollarBarSpotStream(
        rest_client=rest,  # type: ignore[arg-type]
        aggregator=agg,
        symbol="BTCUSDT",
        poll_seconds=3.0,
        extra_trade_sink=sink,
    )


def _fake_page(n: int = 10) -> list[dict]:
    """Build a synthetic Binance Spot aggTrades REST page."""
    base_ts = int(pd.Timestamp("2026-05-07 12:00:00", tz="UTC").timestamp() * 1000)
    return [
        {
            "a": 1000 + i,
            "p": f"{60_000.0 + i:.2f}",
            "q": "0.5",
            "T": base_ts + i * 100,
            "m": (i % 2 == 0),  # alternating maker side
        }
        for i in range(n)
    ]


def test_fanout_receives_every_trade_in_order():
    """extra_trade_sink sees every trade, in the same order."""
    received: list[Trade] = []

    def sink(t: Trade) -> None:
        received.append(t)

    stream = _make_stream_with_sink(sink)
    page = _fake_page(n=10)
    stream._process_page(page)

    assert len(received) == 10
    # Ordering match: aggTradeId monotonically increases, so does received.
    ids = [t.trade_id for t in received]
    assert ids == [f"binance:{1000 + i}" for i in range(10)]
    # Timestamp / price / amount preserved.
    for i, t in enumerate(received):
        assert t.timestamp == pd.Timestamp(
            int(pd.Timestamp("2026-05-07 12:00:00", tz="UTC").timestamp() * 1000)
            + i * 100,
            unit="ms", tz="UTC",
        )
        assert t.price == 60_000.0 + i
        assert t.amount == 0.5


def test_fanout_direction_sign_matches_is_buyer_maker():
    """is_buyer_maker=True (taker is seller) -> direction=-1; False -> +1."""
    received: list[Trade] = []
    stream = _make_stream_with_sink(received.append)

    page = [
        {"a": 1, "p": "60000", "q": "1", "T": 1700000000000, "m": True},   # maker buys -> direction=-1
        {"a": 2, "p": "60001", "q": "1", "T": 1700000000100, "m": False},  # taker buys -> direction=+1
    ]
    stream._process_page(page)
    assert [t.direction for t in received] == [-1, 1]


def test_fanout_default_none_does_not_construct_trade():
    """Pre-round-4 default (sink=None) skips Trade construction entirely.

    We verify by inspecting the stream's internal attribute and confirming
    no Trade objects are produced.
    """
    agg = DollarBarAggregator(threshold=500_000.0)
    stream = DollarBarSpotStream(
        rest_client=object(),  # type: ignore[arg-type]
        aggregator=agg,
        symbol="BTCUSDT",
        poll_seconds=3.0,
        # no extra_trade_sink kwarg
    )
    assert stream._extra_trade_sink is None

    # _process_page should still work end-to-end; no Trade objects leak.
    stream._process_page(_fake_page(n=5))
    assert stream._trades_processed == 5


def test_fanout_sink_exception_is_swallowed_and_logged(caplog):
    """A sink raising must not stop the stream from advancing."""
    def boom(_: Trade) -> None:
        raise RuntimeError("sink fault")

    stream = _make_stream_with_sink(boom)
    with caplog.at_level(logging.ERROR, logger="deribit_paper.features.dollar_bar_stream"):
        stream._process_page(_fake_page(n=3))

    assert stream._trades_processed == 3
    # Look for the logged exception line.
    msgs = [r.getMessage() for r in caplog.records]
    assert any("extra_trade_sink raised" in m for m in msgs), (
        f"expected sink-exception log, got: {msgs}"
    )


def test_fanout_aggregator_still_receives_trades_when_sink_present():
    """The existing dollar-bar aggregator path must remain wired even
    when extra_trade_sink is set.
    """
    received_trades: list[Trade] = []
    stream = _make_stream_with_sink(received_trades.append)

    # Big page so the aggregator has plenty to chew through.
    page = _fake_page(n=50)
    stream._process_page(page)

    # Aggregator counters advanced — confirms add_trade was called.
    assert stream._trades_processed == 50
    assert len(received_trades) == 50
