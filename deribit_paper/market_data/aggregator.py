"""Bar aggregator: slice buffers on bar close to produce Bar snapshots.

When :class:`BarClock` fires at wall-clock ``T``, the aggregator's
callback is invoked with that timestamp.  It then:

1. Slices the perp trade buffer to the window ``[T - bar_minutes, T)``.
2. Slices the option trade buffer to the same window.
3. (Round-4 F6/F8) Slices the optional Binance Spot TSRV buffer to the
   same window for live TSRV computation.
4. Reads the latest index price ``at_or_before(T)``.
5. Evicts older entries from the buffers (keeps memory bounded).
6. Publishes a :class:`Bar` snapshot to all registered subscribers.

The :class:`Bar` object is the input to the feature pipeline (Phase 3)
and matches the column conventions of ``vol_arb/pipeline.py`` so the
downstream batch code can be reused with minimal adaptation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
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

    Round-4 F6/F8: ``tsrv_trades`` is the parallel slice from the Binance
    Spot TSRV buffer when the engine wires one. When ``None`` (default,
    pre-round-4 behaviour), :meth:`tsrv_dataframe` falls back to
    :meth:`perp_dataframe` so existing tests and Bar fixtures remain
    valid byte-for-byte.
    """
    timestamp: pd.Timestamp
    bar_minutes: int
    perp_trades: list[Trade]
    option_trades: list[Trade]
    index_price: float | None   # latest index at or before timestamp
    tsrv_trades: list[Trade] | None = field(default=None)

    @property
    def window_start(self) -> pd.Timestamp:
        return self.timestamp - pd.Timedelta(minutes=self.bar_minutes)

    @property
    def n_perp_trades(self) -> int:
        return len(self.perp_trades)

    @property
    def n_option_trades(self) -> int:
        return len(self.option_trades)

    @property
    def n_tsrv_trades(self) -> int:
        """Round-4 F6/F8: count of Binance Spot trades in TSRV slice.

        Returns 0 when ``tsrv_trades`` is ``None`` (pre-round-4 path).
        """
        return len(self.tsrv_trades) if self.tsrv_trades is not None else 0

    def perp_dataframe(self) -> pd.DataFrame:
        """Return perp trades as a DataFrame for downstream computation.

        The sort uses ``kind='mergesort'`` so trades sharing a millisecond
        timestamp keep their insertion order. Round-4 F6/F8 byte-identity
        gates depend on this — Binance Vision parquet rows tied at the
        same ``transact_time`` rely on stable ordering to match the
        offline pipeline. The previous default (quicksort) was unstable
        and could shuffle ties non-deterministically.
        """
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
        ]).sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    def tsrv_dataframe(self) -> pd.DataFrame:
        """Return TSRV-input trades as a DataFrame.

        Round-4 F6/F8 (parity-cycle): when ``tsrv_trades`` is populated
        (engine wired ``spot_trade_buffer`` for Binance Spot live TSRV),
        this returns those rows. When ``tsrv_trades is None`` — the
        pre-round-4 path, also taken by every existing fixture — it
        falls back to :meth:`perp_dataframe` so callers (notably
        :func:`compute_bar_tsrv`) see byte-identical input to today.

        Stable mergesort on ``timestamp`` so byte-identity vs the
        offline pipeline holds even when the Binance feed has multiple
        trades sharing one millisecond.
        """
        if self.tsrv_trades is None:
            return self.perp_dataframe()
        if not self.tsrv_trades:
            return pd.DataFrame(columns=["timestamp", "price", "amount", "direction"])
        return pd.DataFrame([
            {
                "timestamp": t.timestamp,
                "price": t.price,
                "amount": t.amount,
                "direction": t.direction,
            }
            for t in self.tsrv_trades
        ]).sort_values("timestamp", kind="mergesort").reset_index(drop=True)

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
    """Slices buffers on bar close and publishes :class:`Bar` snapshots.

    Round-4 F6/F8: optional ``tsrv_buffer`` provides a parallel
    ``TradeBuffer`` of Binance Spot aggTrades (typically fed by the
    fan-out hook on :class:`DollarBarSpotStream`). When present, the
    same window is sliced into ``Bar.tsrv_trades`` and evicted after
    the bar fires; ``Bar.tsrv_dataframe()`` then returns those rows
    so :func:`compute_bar_tsrv` reads from the venue the offline
    pipeline trains on. Default ``None`` preserves pre-round-4
    behaviour byte-identically.
    """

    def __init__(
        self,
        *,
        bar_minutes: int,
        perp_buffer: TradeBuffer,
        option_buffer: TradeBuffer,
        index_buffer: IndexPriceBuffer,
        tsrv_buffer: TradeBuffer | None = None,
    ):
        self.bar_minutes = bar_minutes
        self.perp_buffer = perp_buffer
        self.option_buffer = option_buffer
        self.index_buffer = index_buffer
        # Round-4 F6/F8: optional Binance Spot trade buffer for live TSRV.
        self.tsrv_buffer = tsrv_buffer
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

        # Round-4 F6/F8: slice the Binance Spot buffer in parallel when wired.
        tsrv_trades: list[Trade] | None = None
        if self.tsrv_buffer is not None:
            tsrv_trades = self.tsrv_buffer.slice_window(window_start, timestamp)

        bar = Bar(
            timestamp=timestamp,
            bar_minutes=self.bar_minutes,
            perp_trades=perp_trades,
            option_trades=option_trades,
            index_price=index_price,
            tsrv_trades=tsrv_trades,
        )
        self.last_bar = bar

        logger.info(
            "Bar %s: %d perp trades, %d option trades, %d spot trades, index=%s",
            timestamp, len(perp_trades), len(option_trades),
            (len(tsrv_trades) if tsrv_trades is not None else 0),
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
        if self.tsrv_buffer is not None:
            self.tsrv_buffer.evict_older(timestamp)
