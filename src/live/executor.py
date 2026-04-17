"""MkrTkr60 execution protocol for Bybit demo.

PostOnly limit order at BBO → 60s wait → taker fallback for unfilled.
Tracks fill method (maker/taker/blended) for TCA.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Optional

from live.bybit_client import BybitClient

logger = logging.getLogger(__name__)


@dataclass
class TradeResult:
    """Result of an MkrTkr60 execution."""
    filled: bool = False
    fill_price: float = 0.0
    fill_qty: float = 0.0
    method: str = ""         # "maker", "taker", "blended"
    order_id: str = ""
    decision_mid: float = 0.0  # TCA benchmark: (bid+ask)/2 at signal time


class MkrTkr60Executor:
    """MkrTkr60 execution: PostOnly limit → 60s wait → taker fallback.

    Prevents double-booking with per-symbol pending lock.
    """

    def __init__(
        self,
        client: BybitClient,
        maker_wait_s: int = 60,
        poll_interval_s: float = 5.0,
        max_retries: int = 2,
    ) -> None:
        self._client = client
        self._maker_wait_s = maker_wait_s
        self._poll_interval_s = poll_interval_s
        self._max_retries = max_retries
        self._pending: dict[str, str] = {}  # symbol -> order_id

    @classmethod
    def from_config(cls, client: BybitClient, config) -> MkrTkr60Executor:
        return cls(
            client=client,
            maker_wait_s=config.maker_wait_s,
            poll_interval_s=config.poll_interval_s,
            max_retries=config.max_retries,
        )

    async def execute(
        self,
        symbol: str,
        side: str,      # "Buy" or "Sell"
        quantity: float,
    ) -> Optional[TradeResult]:
        """Full MkrTkr60 execution protocol.

        1. Get BBO
        2. Place PostOnly limit at best bid (Buy) / best ask (Sell)
        3. Wait up to maker_wait_s for fill
        4. If unfilled: cancel + market order (taker fallback)

        Returns:
            TradeResult or None if execution fails entirely.
        """
        # Pending lock — prevent double-booking same symbol
        if symbol in self._pending:
            logger.warning("%s: order already pending (%s), skipping",
                           symbol, self._pending[symbol])
            return None

        # 1. Get BBO
        ticker = await self._client.fetch_ticker(symbol)
        bid, ask = ticker["bid"], ticker["ask"]
        if bid <= 0 or ask <= 0:
            logger.error("%s: invalid BBO bid=%.4f ask=%.4f", symbol, bid, ask)
            return None

        decision_mid = (bid + ask) / 2.0
        limit_price = bid if side == "Buy" else ask  # maker at BBO

        # 2. Place PostOnly limit (with retry on rejection)
        order_result = None
        for attempt in range(self._max_retries + 1):
            order_result = await self._client.place_limit_order(
                symbol, side, quantity, limit_price, post_only=True)

            if order_result["status"] == "submitted":
                break

            # PostOnly rejected (would cross spread) — retry with fresh BBO
            ret_code = order_result.get("retCode", 0)
            if ret_code == 140025 or "PostOnly" in order_result.get("error", ""):
                logger.info("%s: PostOnly rejected (attempt %d), refreshing BBO",
                            symbol, attempt + 1)
                ticker = await self._client.fetch_ticker(symbol)
                bid, ask = ticker["bid"], ticker["ask"]
                limit_price = bid if side == "Buy" else ask
                await asyncio.sleep(0.5)
            else:
                logger.warning("%s: order rejected: %s", symbol, order_result.get("error"))
                break

        if order_result is None or order_result["status"] != "submitted":
            # All retries failed — go straight to taker
            logger.info("%s: PostOnly exhausted, falling back to taker", symbol)
            return await self._taker_fallback(symbol, side, quantity, decision_mid)

        order_id = order_result["order_id"]
        self._pending[symbol] = order_id

        try:
            # 3. Monitor for maker_wait_s
            fill = await self._wait_for_fill(symbol, order_id, self._maker_wait_s)

            # 4. Handle outcome
            if fill["status"] == "filled":
                logger.info("%s: MAKER fill @ %.4f", symbol, fill["filled_price"])
                return TradeResult(
                    filled=True,
                    fill_price=fill["filled_price"],
                    fill_qty=fill["filled_qty"],
                    method="maker",
                    order_id=order_id,
                    decision_mid=decision_mid,
                )

            elif fill["status"] == "partially_filled" and fill["filled_qty"] > 0:
                # Cancel remainder, taker for unfilled
                await self._client.cancel_order(symbol, order_id)
                filled_qty = fill["filled_qty"]
                filled_price = fill["filled_price"]
                unfilled_qty = quantity - filled_qty

                logger.info("%s: partial fill %.6f/%.6f, taker for remainder",
                            symbol, filled_qty, quantity)

                if unfilled_qty > 0:
                    taker = await self._taker_fallback(
                        symbol, side, unfilled_qty, decision_mid)
                    if taker and taker.filled:
                        # Blended price
                        total_qty = filled_qty + taker.fill_qty
                        blended = (filled_price * filled_qty +
                                   taker.fill_price * taker.fill_qty) / total_qty
                        return TradeResult(
                            filled=True,
                            fill_price=blended,
                            fill_qty=total_qty,
                            method="blended",
                            order_id=order_id,
                            decision_mid=decision_mid,
                        )

                # Return partial fill only
                return TradeResult(
                    filled=True,
                    fill_price=filled_price,
                    fill_qty=filled_qty,
                    method="maker",
                    order_id=order_id,
                    decision_mid=decision_mid,
                )

            else:
                # Not filled at all — cancel and taker
                await self._client.cancel_order(symbol, order_id)
                logger.info("%s: no fill after %ds, taker fallback",
                            symbol, self._maker_wait_s)
                return await self._taker_fallback(
                    symbol, side, quantity, decision_mid)

        finally:
            self._pending.pop(symbol, None)

    async def execute_exit(
        self,
        symbol: str,
        side: str,
        quantity: float,
    ) -> Optional[TradeResult]:
        """Exit execution — market order (no MkrTkr60 for exits)."""
        ticker = await self._client.fetch_ticker(symbol)
        decision_mid = (ticker["bid"] + ticker["ask"]) / 2.0

        result = await self._client.place_market_order(
            symbol, side, quantity, reduce_only=True)

        if result["status"] != "submitted":
            logger.error("%s: exit market order failed: %s", symbol, result.get("error"))
            return None

        # Wait briefly for fill
        await asyncio.sleep(1.0)
        status = await self._client.get_order_status(symbol, result["order_id"])

        return TradeResult(
            filled=status["status"] == "filled",
            fill_price=status.get("filled_price", decision_mid),
            fill_qty=status.get("filled_qty", quantity),
            method="taker",
            order_id=result["order_id"],
            decision_mid=decision_mid,
        )

    async def _taker_fallback(
        self,
        symbol: str,
        side: str,
        quantity: float,
        decision_mid: float,
    ) -> Optional[TradeResult]:
        """Place market order as taker fallback."""
        result = await self._client.place_market_order(symbol, side, quantity)

        if result["status"] != "submitted":
            logger.error("%s: taker fallback failed: %s", symbol, result.get("error"))
            return None

        # Wait briefly for fill confirmation
        await asyncio.sleep(1.0)
        status = await self._client.get_order_status(symbol, result["order_id"])

        return TradeResult(
            filled=status.get("status") == "filled",
            fill_price=status.get("filled_price", decision_mid),
            fill_qty=status.get("filled_qty", quantity),
            method="taker",
            order_id=result["order_id"],
            decision_mid=decision_mid,
        )

    async def _wait_for_fill(
        self, symbol: str, order_id: str, timeout: float
    ) -> dict:
        """Poll for fill status every poll_interval_s. Returns final status."""
        deadline = time.monotonic() + timeout

        while time.monotonic() < deadline:
            status = await self._client.get_order_status(symbol, order_id)

            if status["status"] in ("filled", "cancelled"):
                return status
            if status["status"] == "partially_filled":
                # Check if enough time left for more fills
                remaining = deadline - time.monotonic()
                if remaining < self._poll_interval_s:
                    return status

            await asyncio.sleep(self._poll_interval_s)

        # Final check after timeout
        return await self._client.get_order_status(symbol, order_id)
