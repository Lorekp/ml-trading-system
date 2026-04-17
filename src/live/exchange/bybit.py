"""Bybit exchange adapter — implements both ExchangeAdapter and DataProvider.

Wraps the existing bybit_client.BybitClient with the new interfaces,
adding get_open_orders() for reconciliation.
"""

from __future__ import annotations

import logging
from typing import Optional

import pandas as pd

from live.bybit_client import BybitClient

logger = logging.getLogger(__name__)


class BybitExchange(BybitClient):
    """Bybit exchange adapter implementing ExchangeAdapter + DataProvider protocols.

    Inherits all existing BybitClient functionality and adds:
    - get_open_orders() for startup reconciliation
    - fetch_ohlcv() returning pd.DataFrame (DataProvider interface)
    """

    async def get_open_orders(self) -> list[dict]:
        """Fetch all open/unfilled orders on the account.

        Required for reconciliation: detect orders placed but not filled
        after a crash during MkrTkr60 60s wait.
        """
        params = {
            "category": "linear",
            "settleCoin": "USDT",
        }
        try:
            data = await self._request("GET", "/v5/order/realtime",
                                       params=params, auth=True)
            orders = []
            for o in data.get("result", {}).get("list", []):
                status = o.get("orderStatus", "")
                if status in ("New", "PartiallyFilled", "Untriggered"):
                    orders.append({
                        "order_id": o.get("orderId", ""),
                        "symbol": o.get("symbol", ""),
                        "side": o.get("side", ""),
                        "qty": float(o.get("qty", 0)),
                        "price": float(o.get("price", 0)),
                        "order_type": o.get("orderType", ""),
                        "status": status,
                        "filled_qty": float(o.get("cumExecQty", 0)),
                    })
            return orders
        except Exception as e:
            logger.error("Get open orders: %s", e)
            return []

    async def fetch_ohlcv(
        self, symbol: str, interval: str, limit: int = 200
    ) -> pd.DataFrame:
        """DataProvider interface: fetch OHLCV as DataFrame."""
        bars = await self.fetch_klines(symbol, interval, limit)
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        df.set_index("timestamp", inplace=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
