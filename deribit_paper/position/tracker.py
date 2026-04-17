"""Open-position tracker.

Maintains a dict of currently-open straddle trades keyed by trade_id.
The tracker is updated by the :class:`StraddleFlow` on entry + exit
and queried by the risk layer for the "how many positions are open"
and "what's the total margin utilisation" checks.

Unrealized PnL is computed as a best-effort estimate from the entry
fill prices vs the current mark (approximate — a full mark-to-market
requires Greeks + live SVI which is too expensive to run on every
tick).  The approximation is good enough for:

- Risk limit checks (is this position within bounds?)
- Telegram heartbeat ("3 positions open, unrealized $+450")
- Dashboard display

Exact realized PnL comes from the exit fills and flows to
:class:`EquityTracker` via ``on_trade_closed``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from deribit_paper.execution.straddle_flow import TradeOutcome

logger = logging.getLogger(__name__)


@dataclass
class OpenPosition:
    """State of one open straddle trade (+ delta hedge) in the tracker."""
    trade_id: str
    signal_direction: int       # +1 long, -1 short
    call_instrument: str
    put_instrument: str
    strike: float
    expiry_dt: pd.Timestamp
    n_contracts_btc: float
    entry_call_price: float     # BTC per contract, from fill
    entry_put_price: float
    entry_timestamp: pd.Timestamp
    entry_equity_usd: float     # equity at entry (for sizing sanity)
    hedge_inventory_btc: float = 0.0     # live perp position, updated by hedger
    last_mark_call: float | None = None  # latest mark for unrealized PnL
    last_mark_put: float | None = None
    status: str = "open"

    @property
    def entry_premium_btc(self) -> float:
        """Total premium paid/received at entry (per-contract × N)."""
        return (self.entry_call_price + self.entry_put_price) * self.n_contracts_btc

    def unrealized_pnl_btc(self) -> float | None:
        """Rough unrealized PnL in BTC using the last observed marks.

        Returns ``None`` if no marks have been recorded since entry
        (e.g., right after the position opened).
        """
        if self.last_mark_call is None or self.last_mark_put is None:
            return None
        current_premium = (self.last_mark_call + self.last_mark_put) * self.n_contracts_btc
        if self.signal_direction == -1:  # short straddle — profit when premium shrinks
            return self.entry_premium_btc - current_premium
        else:  # long straddle — profit when premium grows
            return current_premium - self.entry_premium_btc


class PositionTracker:
    """Thread-safe (asyncio-single-threaded) open-positions registry."""

    def __init__(self):
        self._positions: dict[str, OpenPosition] = {}
        self._closed_count = 0

    def __len__(self) -> int:
        return len(self._positions)

    @property
    def n_open(self) -> int:
        return len(self._positions)

    @property
    def n_closed(self) -> int:
        return self._closed_count

    def open_position(
        self,
        *,
        trade_id: str,
        outcome: TradeOutcome,
        current_equity_usd: float,
    ) -> OpenPosition:
        """Register a successfully-opened trade in the tracker.

        Called by :class:`StraddleFlow` immediately after both legs
        have filled (before the holding-period wait).
        """
        if outcome.entry_call is None or outcome.entry_put is None:
            raise ValueError("open_position called without both entry legs filled")
        if outcome.signal.straddle is None:
            raise ValueError("open_position called with no straddle selection")

        pos = OpenPosition(
            trade_id=trade_id,
            signal_direction=outcome.signal.direction,
            call_instrument=outcome.entry_call.instrument_name,
            put_instrument=outcome.entry_put.instrument_name,
            strike=outcome.signal.straddle.strike,
            expiry_dt=pd.Timestamp(outcome.signal.straddle.expiry_dt),
            n_contracts_btc=outcome.n_contracts_btc,
            entry_call_price=outcome.entry_call.average_price,
            entry_put_price=outcome.entry_put.average_price,
            entry_timestamp=outcome.started_at or pd.Timestamp.now(tz="UTC"),
            entry_equity_usd=current_equity_usd,
        )
        self._positions[trade_id] = pos
        logger.info(
            "PositionTracker opened %s: %s strike=%.0f N=%.2fBTC "
            "(equity=%.2f, open=%d)",
            trade_id,
            "SHORT" if pos.signal_direction == -1 else "LONG",
            pos.strike, pos.n_contracts_btc, current_equity_usd,
            len(self._positions),
        )
        return pos

    def close_position(self, trade_id: str) -> OpenPosition | None:
        """Remove a position from the open set.  Returns the removed position."""
        pos = self._positions.pop(trade_id, None)
        if pos is None:
            logger.warning("close_position: trade_id %s not found", trade_id)
            return None
        pos.status = "closed"
        self._closed_count += 1
        logger.info(
            "PositionTracker closed %s (remaining open=%d, total closed=%d)",
            trade_id, len(self._positions), self._closed_count,
        )
        return pos

    def get(self, trade_id: str) -> OpenPosition | None:
        return self._positions.get(trade_id)

    def all_open(self) -> list[OpenPosition]:
        return list(self._positions.values())

    def total_notional_btc(self) -> float:
        """Sum of ``n_contracts_btc`` across all open positions."""
        return sum(p.n_contracts_btc for p in self._positions.values())

    def total_unrealized_pnl_btc(self) -> float:
        """Sum of unrealized PnL across positions with recorded marks.

        Positions without marks contribute 0 (conservative — treats
        them as flat).
        """
        return sum(
            (p.unrealized_pnl_btc() or 0.0) for p in self._positions.values()
        )

    def update_mark(
        self,
        trade_id: str,
        *,
        call_mark: float,
        put_mark: float,
    ) -> None:
        """Record the latest mark prices for a position's two legs."""
        pos = self._positions.get(trade_id)
        if pos is None:
            return
        pos.last_mark_call = call_mark
        pos.last_mark_put = put_mark

    def update_hedge_inventory(self, trade_id: str, inventory_btc: float) -> None:
        """Record the latest perp hedge inventory for a position."""
        pos = self._positions.get(trade_id)
        if pos is None:
            return
        pos.hedge_inventory_btc = inventory_btc
