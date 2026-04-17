"""Simple market order execution — for strategies that don't need maker fills."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Optional

from live.core.interfaces import ExchangeAdapter, TradeResult, PendingOrder
from live.core.state import StateDB

logger = logging.getLogger(__name__)


class MarketExecution:
    """Simple market order execution. Implements ExecutionAlgo protocol."""

    def __init__(self, exchange: ExchangeAdapter, state: StateDB) -> None:
        self._exchange = exchange
        self._state = state

    async def execute_entry(
        self, symbol: str, side: str, qty: float,
        strategy_id: str = "",
    ) -> Optional[TradeResult]:
        ticker = await self._exchange.fetch_ticker(symbol)
        decision_mid = (ticker.get("bid", 0) + ticker.get("ask", 0)) / 2.0

        pending = PendingOrder(
            symbol=symbol, exchange_order_id="", side=side, qty=qty,
            order_type="market", strategy_id=strategy_id,
            placed_at=datetime.now(tz=timezone.utc),
        )
        pending_id = await self._state.insert_pending_order(pending)

        result = await self._exchange.place_market_order(symbol, side, qty)
        if result["status"] != "submitted":
            await self._state.update_pending_order(pending_id, "cancelled")
            return None

        await asyncio.sleep(1.0)
        status = await self._exchange.get_order_status(symbol, result["order_id"])
        await self._state.update_pending_order(pending_id, "filled")

        return TradeResult(
            filled=status.get("status") == "filled",
            fill_price=status.get("filled_price", decision_mid),
            fill_qty=status.get("filled_qty", qty),
            method="taker", order_id=result["order_id"],
            decision_mid=decision_mid,
        )

    async def execute_exit(
        self, symbol: str, side: str, qty: float,
    ) -> Optional[TradeResult]:
        return await self.execute_entry(symbol, side, qty)
