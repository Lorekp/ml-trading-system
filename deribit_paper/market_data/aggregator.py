"""Bar aggregator: slice buffers on bar close to produce Bar snapshots.

When :class:`BarClock` fires at wall-clock ``T``, the aggregator's
callback is invoked with that timestamp.  It then:

1. Slices the perp trade buffer to the window ``[T - bar_minutes, T)``.
2. Slices the option trade buffer to the same window.
3. Reads the latest index price ``at_or_before(T)``.
4. Evicts older entries from the buffers (keeps memory bounded).
5. Publishes a :class:`Bar` snapshot to all registered subscribers.

The :class:`Bar` object is the input to the feature pipeline (Phase 3)
and matches the column conventions of ``vol_arb/pipeline.py`` so the
downstream batch code can be reused with minimal adaptation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

import pandas as pd

from deribit_paper.market_data.buffer import (
    IndexPriceBuffer,
    Trade,
    TradeBuffer,
)

logger = logging.getLogger(__name__)


BarSubscriber = Callable[["Bar"], Awaitable[None]]


@dataclass
class Bar:
    """Snapshot of one bar's worth of market data.

    ``timestamp`` is the END of the bar's window (matches the backtester's
    END-stamped convention fixed in Stage 14 / Stage 16.8 of the docs).
    The bar covers trades with timestamps in ``[timestamp - bar_minutes, timestamp)``.
    """
    timestamp: pd.Timestamp
    bar_minutes: int
    perp_trades: list[Trade]
    option_trades: list[Trade]
    index_price: float | None   # latest index at or before timestamp

    @property
    def window_start(self) -> pd.Timestamp:
        return self.timestamp - pd.Timedelta(minutes=self.bar_minutes)

    @property
    def n_perp_trades(self) -> int:
        return len(self.perp_trades)

    @property
    def n_option_trades(self) -> int:
        return len(self.option_trades)

    def perp_dataframe(self) -> pd.DataFrame:
        """Return perp trades as a DataFrame for downstream TSRV computation."""
        if not self.perp_trades:
            return pd.DataFrame(columns=["timestamp", "price", "amount", "direction"])
        return pd.DataFrame([
            {
                "timestamp": t.timestamp,
                "price": t.price,
                "amount": t.amount,
                "direction": t.direction,
            }
            for t in self.perp_trades
        ]).sort_values("timestamp").reset_index(drop=True)

    def option_dataframe(self) -> pd.DataFrame:
        """Return option trades as a DataFrame for downstream SVI + straddle builder."""
        if not self.option_trades:
            return pd.DataFrame(columns=[
                "timestamp", "instrument_name", "price", "amount", "direction",
                "strike", "option_type", "expiry_dt", "iv", "index_price", "mark_price",
            ])
        return pd.DataFrame([
            {
                "timestamp": t.timestamp,
                "instrument_name": t.instrument_name,
                "price": t.price,
                "amount": t.amount,
                "direction": t.direction,
                "strike": t.strike,
                "option_type": t.option_type,
                "expiry_dt": t.expiry_dt,
                "iv": t.iv,
                "index_price": t.index_price,
                "mark_price": t.mark_price,
            }
            for t in self.option_trades
        ]).sort_values("timestamp").reset_index(drop=True)


class BarAggregator:
    """Slices buffers on bar close and publishes :class:`Bar` snapshots."""

    def __init__(
        self,
        *,
        bar_minutes: int,
        perp_buffer: TradeBuffer,
        option_buffer: TradeBuffer,
        index_buffer: IndexPriceBuffer,
    ):
        self.bar_minutes = bar_minutes
        self.perp_buffer = perp_buffer
        self.option_buffer = option_buffer
        self.index_buffer = index_buffer
        self._subscribers: list[BarSubscriber] = []
        self.last_bar: Bar | None = None

    def subscribe(self, callback: BarSubscriber) -> None:
        """Register an async callback for each emitted bar."""
        self._subscribers.append(callback)

    async def on_bar_close(self, timestamp: pd.Timestamp) -> None:
        """Handle a bar-close tick from :class:`BarClock`.

        Slices the buffers, constructs a :class:`Bar`, notifies all
        subscribers, then evicts old entries.
        """
        window_start = timestamp - pd.Timedelta(minutes=self.bar_minutes)

        perp_trades = self.perp_buffer.slice_window(window_start, timestamp)
        option_trades = self.option_buffer.slice_window(window_start, timestamp)
        index_price = self.index_buffer.at_or_before(timestamp)

        bar = Bar(
            timestamp=timestamp,
            bar_minutes=self.bar_minutes,
            perp_trades=perp_trades,
            option_trades=option_trades,
            index_price=index_price,
        )
        self.last_bar = bar

        logger.info(
            "Bar %s: %d perp trades, %d option trades, index=%s",
            timestamp, len(perp_trades), len(option_trades),
            f"{index_price:.2f}" if index_price is not None else "None",
        )

        # Notify subscribers (e.g., feature pipeline, metrics sink)
        for cb in self._subscribers:
            try:
                await cb(bar)
            except Exception:  # noqa: BLE001 - log + continue
                logger.exception(
                    "Bar subscriber raised — continuing to next subscriber",
                )

        # Evict stale data now that the bar has been consumed
        self.perp_buffer.evict_older(timestamp)
        self.option_buffer.evict_older(timestamp)
        self.index_buffer.evict_older(timestamp)
