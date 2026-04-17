"""Paper execution — simulates fills locally without hitting exchange API."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

from live.core.interfaces import ExchangeAdapter, TradeResult, PendingOrder
from live.core.state import StateDB

logger = logging.getLogger(__name__)


class PaperExecution:
    """Simulates fills at current mid price. No exchange orders placed."""

    def __init__(self, exchange: ExchangeAdapter, state: StateDB,
                 slippage_bps: float = 2.0) -> None:
        self._exchange = exchange
        self._state = state
        self._slippage_bps = slippage_bps

    async def execute_entry(
        self, symbol: str, side: str, qty: float,
        strategy_id: str = "",
    ) -> Optional[TradeResult]:
        ticker = await self._exchange.fetch_ticker(symbol)
        bid = ticker.get("bid", 0)
        ask = ticker.get("ask", 0)
        mid = (bid + ask) / 2.0 if bid > 0 and ask > 0 else ticker.get("last_price", 0)

        if mid <= 0:
            logger.warning("Paper: no price for %s, skipping", symbol)
            return None

        # Simulate slippage: buy at ask-side, sell at bid-side
        slip = mid * self._slippage_bps / 10_000
        if side in ("Buy", "open_long", "close_short"):
            fill_price = mid + slip
        else:
            fill_price = mid - slip

        return TradeResult(
            filled=True,
            fill_price=fill_price,
            fill_qty=qty,
            method="paper",
            order_id=f"paper-{datetime.now(tz=timezone.utc).strftime('%H%M%S%f')}",
            decision_mid=mid,
        )

    async def execute_exit(
        self, symbol: str, side: str, qty: float,
    ) -> Optional[TradeResult]:
        return await self.execute_entry(symbol, side, qty)
