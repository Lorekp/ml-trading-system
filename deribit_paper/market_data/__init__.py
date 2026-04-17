"""Live market data ingestion and bar aggregation.

Layered design:

- :mod:`deribit_paper.market_data.buffer` — rolling in-memory buffers for
  perp trades, option trades, and the index price.  Time-based eviction,
  trade-id dedup on replay, O(1) append.
- :mod:`deribit_paper.market_data.subscriber` — wires the Deribit
  WebSocket client to the buffers.  One handler per channel.
- :mod:`deribit_paper.market_data.bar_clock` — async scheduler that fires
  callbacks at wall-clock 15-min boundaries (:00, :15, :30, :45 UTC).
- :mod:`deribit_paper.market_data.aggregator` — on each bar tick, slices
  the buffers and produces a :class:`Bar` snapshot with the option trades
  from ``[T - 15min, T)`` and the perp trades from the same window.

The aggregator's output is the input to the feature pipeline (Phase 3).
"""

from deribit_paper.market_data.aggregator import Bar, BarAggregator
from deribit_paper.market_data.bar_clock import BarClock
from deribit_paper.market_data.buffer import IndexPriceBuffer, TradeBuffer
from deribit_paper.market_data.subscriber import MarketDataSubscriber

__all__ = [
    "Bar",
    "BarAggregator",
    "BarClock",
    "IndexPriceBuffer",
    "TradeBuffer",
    "MarketDataSubscriber",
]
