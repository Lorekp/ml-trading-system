"""MkrTkr60 execution algorithm with pending order tracking.

PostOnly limit at BBO → 60s wait → taker fallback.
Records pending orders in DB for crash recovery.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Optional

from live.core.interfaces import (
    ExchangeAdapter, TradeResult, PendingOrder,
)
from live.core.state import StateDB
from live.core.tca import TCARecord

logger = logging.getLogger(__name__)


class MkrTkr60(object):
    """MkrTkr60 execution with crash-safe pending order tracking.

    Changes from original executor.py:
    - Inserts pending order into DB BEFORE placing on exchange
    - Updates pending order status AFTER fill/cancel
    - On crash, reconciler finds and resolves stale pending orders
    """

    def __init__(
        self,
        exchange: ExchangeAdapter,
        state: StateDB,
        maker_wait_s: int = 60,
        poll_interval_s: float = 5.0,
        max_retries: int = 2,
    ) -> None:
        self._exchange = exchange
        self._state = state
        self._maker_wait_s = maker_wait_s
        self._poll_interval_s = poll_interval_s
        self._max_retries = max_retries
        self._pending: dict[str, str] = {}  # symbol -> order_id (in-memory lock)

    @classmethod
    def from_config(cls, exchange, state, config) -> MkrTkr60:
        return cls(
            exchange=exchange, state=state,
            maker_wait_s=config.maker_wait_s,
            poll_interval_s=config.poll_interval_s,
            max_retries=config.max_retries,
        )

    async def execute_entry(
        self, symbol: str, side: str, qty: float,
        strategy_id: str = "",
    ) -> Optional[TradeResult]:
        """Full MkrTkr60 entry execution with pending order tracking."""
        if symbol in self._pending:
            logger.warning("%s: order already pending, skipping", symbol)
            return None

        # TCA: record decision time
        decision_ts = datetime.now(tz=timezone.utc)

        # 1. Get BBO
        ticker = await self._exchange.fetch_ticker(symbol)
        bid, ask = ticker.get("bid", 0), ticker.get("ask", 0)
        if bid <= 0 or ask <= 0:
            logger.error("%s: invalid BBO bid=%.4f ask=%.4f", symbol, bid, ask)
            return None

        decision_mid = (bid + ask) / 2.0
        spread_bps = (ask - bid) / decision_mid * 10000 if decision_mid > 0 else 0
        limit_price = bid if side == "Buy" else ask

        # 2. Place PostOnly with pending order tracking
        for attempt in range(self._max_retries + 1):
            # Record pending order in DB BEFORE placing
            pending = PendingOrder(
                symbol=symbol, exchange_order_id="",
                side=side, qty=qty, price=limit_price,
                order_type="limit_postonly", strategy_id=strategy_id,
                placed_at=datetime.now(tz=timezone.utc),
            )
            pending_id = await self._state.insert_pending_order(pending)

            result = await self._exchange.place_limit_order(
                symbol, side, qty, limit_price, post_only=True)

            if result["status"] == "submitted":
                order_id = result["order_id"]
                # Update pending with exchange order ID
                await self._state.update_pending_order(pending_id, "pending")
                # Store exchange_order_id
                await self._state._db.execute(
                    "UPDATE pending_orders SET exchange_order_id=? WHERE id=?",
                    (order_id, pending_id))
                await self._state._db.commit()
                self._pending[symbol] = order_id
                break

            # PostOnly rejected
            await self._state.update_pending_order(pending_id, "cancelled")
            ret_code = result.get("retCode", 0)
            if ret_code == 140025 or "PostOnly" in result.get("error", ""):
                logger.info("%s: PostOnly rejected (attempt %d), refreshing BBO",
                            symbol, attempt + 1)
                ticker = await self._exchange.fetch_ticker(symbol)
                bid, ask = ticker.get("bid", 0), ticker.get("ask", 0)
                limit_price = bid if side == "Buy" else ask
                await asyncio.sleep(0.5)
            else:
                logger.warning("%s: order rejected: %s", symbol, result.get("error"))
                break
        else:
            # All retries failed → taker
            logger.info("%s: PostOnly exhausted, taker fallback", symbol)
            return await self._taker_fallback(symbol, side, qty, decision_mid, strategy_id)

        if symbol not in self._pending:
            return await self._taker_fallback(symbol, side, qty, decision_mid, strategy_id)

        order_id = self._pending[symbol]

        try:
            # 3. Monitor for maker_wait_s
            fill = await self._wait_for_fill(symbol, order_id, self._maker_wait_s)

            # 4. Handle outcome
            if fill["status"] == "filled":
                await self._state.update_pending_order(pending_id, "filled")
                return TradeResult(
                    filled=True, fill_price=fill["filled_price"],
                    fill_qty=fill["filled_qty"], method="maker",
                    order_id=order_id, decision_mid=decision_mid,
                )

            elif fill["status"] == "partially_filled" and fill.get("filled_qty", 0) > 0:
                await self._exchange.cancel_order(symbol, order_id)
                await self._state.update_pending_order(pending_id, "filled")
                filled_qty = fill["filled_qty"]
                filled_price = fill["filled_price"]
                unfilled_qty = qty - filled_qty

                if unfilled_qty > 0:
                    taker = await self._taker_fallback(
                        symbol, side, unfilled_qty, decision_mid, strategy_id)
                    if taker and taker.filled:
                        total_qty = filled_qty + taker.fill_qty
                        blended = (filled_price * filled_qty +
                                   taker.fill_price * taker.fill_qty) / total_qty
                        return TradeResult(
                            filled=True, fill_price=blended,
                            fill_qty=total_qty, method="blended",
                            order_id=order_id, decision_mid=decision_mid,
                        )

                return TradeResult(
                    filled=True, fill_price=filled_price,
                    fill_qty=filled_qty, method="maker",
                    order_id=order_id, decision_mid=decision_mid,
                )

            else:
                await self._exchange.cancel_order(symbol, order_id)
                await self._state.update_pending_order(pending_id, "cancelled")
                return await self._taker_fallback(
                    symbol, side, qty, decision_mid, strategy_id)

        finally:
            self._pending.pop(symbol, None)

    async def execute_exit(
        self, symbol: str, side: str, qty: float,
    ) -> Optional[TradeResult]:
        """Exit via market order (no MkrTkr60 for exits)."""
        ticker = await self._exchange.fetch_ticker(symbol)
        decision_mid = (ticker.get("bid", 0) + ticker.get("ask", 0)) / 2.0

        result = await self._exchange.place_market_order(
            symbol, side, qty, reduce_only=True)

        if result["status"] != "submitted":
            logger.error("%s: exit market order failed: %s", symbol, result.get("error"))
            return None

        await asyncio.sleep(1.0)
        status = await self._exchange.get_order_status(symbol, result["order_id"])

        return TradeResult(
            filled=status.get("status") == "filled",
            fill_price=status.get("filled_price", decision_mid),
            fill_qty=status.get("filled_qty", qty),
            method="taker", order_id=result["order_id"],
            decision_mid=decision_mid,
        )

    async def _taker_fallback(
        self, symbol: str, side: str, qty: float,
        decision_mid: float, strategy_id: str = "",
    ) -> Optional[TradeResult]:
        """Market order taker fallback."""
        pending = PendingOrder(
            symbol=symbol, exchange_order_id="",
            side=side, qty=qty, order_type="market",
            strategy_id=strategy_id,
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

    async def record_tca(
        self,
        result: TradeResult,
        symbol: str,
        side: str,
        strategy_id: str,
        decision_ts: datetime,
        submit_ts: datetime,
        spread_bps: float = 0.0,
    ) -> None:
        """Record TCA for a completed execution."""
        now = datetime.now(tz=timezone.utc)
        tca = TCARecord(
            symbol=symbol, side=side, strategy_id=strategy_id,
            decision_ts=decision_ts, submit_ts=submit_ts,
            first_fill_ts=now,  # approximate — real fill time from WS in future
            complete_ts=now,
            decision_mid=result.decision_mid,
            arrival_mid=result.decision_mid,  # approximate
            fill_price=result.fill_price,
            spread_cost_bps=spread_bps / 2,
            method=result.method,
            total_qty=result.fill_qty,
            maker_qty=result.fill_qty if result.method == "maker" else 0,
            taker_qty=result.fill_qty if result.method == "taker" else 0,
        )
        if result.method == "blended":
            tca.maker_qty = result.fill_qty * 0.5  # approximate split
            tca.taker_qty = result.fill_qty * 0.5

        tca.finalize()
        await self._state.insert_tca(tca)

    async def _wait_for_fill(self, symbol: str, order_id: str, timeout: float) -> dict:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            status = await self._exchange.get_order_status(symbol, order_id)
            if status["status"] in ("filled", "cancelled"):
                return status
            if status["status"] == "partially_filled":
                remaining = deadline - time.monotonic()
                if remaining < self._poll_interval_s:
                    return status
            await asyncio.sleep(self._poll_interval_s)
        return await self._exchange.get_order_status(symbol, order_id)
