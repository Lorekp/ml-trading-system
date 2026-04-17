"""Binance Futures data provider for live trading.

Public endpoints only — no auth needed for klines, tickers, instruments.
Implements DataProvider protocol + asset selector methods (fetch_all_tickers,
fetch_klines) so it can serve as both data source and universe discovery.

Uses the same rate limiting pattern as BybitClient.
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)

_INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "3h": "3h", "4h": "4h",
    "1d": "1d", "1D": "1d", "24h": "1d",
}


class BinanceFuturesData:
    """Binance USDT-M Futures data provider. Public endpoints, no auth.

    Implements:
    - DataProvider protocol (fetch_ohlcv, fetch_ticker, close)
    - Asset selector interface (fetch_all_tickers, fetch_klines)
    """

    def __init__(
        self,
        base_url: str = "https://fapi.binance.com",
        rate_limit_per_sec: int = 5,
        timeout_s: float = 10.0,
    ) -> None:
        self._base_url = base_url
        self._semaphore = asyncio.Semaphore(rate_limit_per_sec)
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._session: aiohttp.ClientSession | None = None
        self._last_request_time = 0.0

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    async def _request(self, endpoint: str, params: dict) -> dict | list:
        """Rate-limited GET request with retries."""
        session = self._ensure_session()
        url = f"{self._base_url}{endpoint}"

        for attempt in range(3):
            async with self._semaphore:
                # Minimum spacing
                elapsed = time.monotonic() - self._last_request_time
                if elapsed < 0.2:
                    await asyncio.sleep(0.2 - elapsed)
                self._last_request_time = time.monotonic()

                try:
                    async with session.get(url, params=params) as resp:
                        if resp.status == 429:
                            wait = 2 ** attempt
                            logger.warning("Binance 429, waiting %ds", wait)
                            await asyncio.sleep(wait)
                            continue
                        if resp.status == 418:  # IP ban
                            logger.error("Binance 418 IP ban")
                            await asyncio.sleep(60)
                            continue
                        data = await resp.json()
                        if isinstance(data, dict) and data.get("code"):
                            logger.warning("Binance API error: %s (code %s)",
                                           data.get("msg"), data.get("code"))
                        return data
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning("Binance request failed (attempt %d): %s", attempt + 1, e)
                    await asyncio.sleep(2 ** attempt)

        logger.error("Binance request failed after 3 retries: %s", endpoint)
        return {}

    # ── DataProvider protocol ──

    async def fetch_ohlcv(
        self, symbol: str, interval: str, limit: int = 200
    ) -> pd.DataFrame:
        """Fetch OHLCV bars. Returns DataFrame with OHLCV columns + datetime index.

        Binance doesn't support 3h klines natively. For 3h, fetches 1h bars
        and resamples — matching the backtest pipeline exactly.
        """
        # 3h: fetch 1h and resample (Binance has no native 3h interval)
        if interval in ("3h", "180"):
            df_1h = await self.fetch_ohlcv(symbol, "1h", limit * 3)
            if df_1h.empty:
                return df_1h
            resampled = df_1h.resample("3h").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum",
            }).dropna()
            return resampled.tail(limit)

        bi = _INTERVAL_MAP.get(interval, interval)
        data = await self._request("/fapi/v1/klines", {
            "symbol": symbol, "interval": bi, "limit": min(limit, 1500),
        })

        if not isinstance(data, list) or not data:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])

        rows = []
        for k in data:
            try:
                rows.append({
                    "timestamp": pd.Timestamp(int(k[0]), unit="ms", tz="UTC"),
                    "open": float(k[1]), "high": float(k[2]),
                    "low": float(k[3]), "close": float(k[4]),
                    "volume": float(k[5]),
                })
            except (IndexError, ValueError):
                continue

        df = pd.DataFrame(rows)
        df.index = df["timestamp"]
        df.index.name = None
        return df[["open", "high", "low", "close", "volume"]]

    async def fetch_ticker(self, symbol: str) -> dict[str, float]:
        """Fetch current ticker (24hr stats + book top).

        Returns `quote_volume_24h` as USDT-denominated dollar volume — used by
        CalibratedPaperExecution for slippage participation caps and the
        max_profitable_size guard (both expect dollar volume, matching the
        backtest's `asset_volumes` convention).
        """
        data = await self._request("/fapi/v1/ticker/24hr", {"symbol": symbol})
        if not isinstance(data, dict):
            return {"last_price": 0, "bid": 0, "ask": 0,
                    "volume_24h": 0, "quote_volume_24h": 0}

        return {
            "last_price": float(data.get("lastPrice", 0)),
            "bid": float(data.get("lastPrice", 0)),  # 24hr endpoint doesn't have BBO
            "ask": float(data.get("lastPrice", 0)),
            "volume_24h": float(data.get("volume", 0)),
            "quote_volume_24h": float(data.get("quoteVolume", 0)),
        }

    # ── Asset selector interface ──

    async def fetch_all_tickers(self) -> list[dict]:
        """Fetch tickers for ALL USDT-M perpetuals in one call."""
        data = await self._request("/fapi/v1/ticker/24hr", {})
        if not isinstance(data, list):
            return []

        result = []
        for t in data:
            sym = t.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            result.append({
                "symbol": sym,
                "asset": sym.replace("USDT", ""),
                "last_price": float(t.get("lastPrice", 0)),
                "volume_24h": float(t.get("volume", 0)),
                "turnover_24h": float(t.get("quoteVolume", 0)),
            })
        logger.debug("fetch_all_tickers: %d USDT perps", len(result))
        return result

    async def fetch_klines(
        self, symbol: str, interval: str, limit: int = 200
    ) -> list[dict]:
        """Fetch klines as list of dicts (for asset selector BTC vol computation)."""
        bi = _INTERVAL_MAP.get(interval, interval)
        data = await self._request("/fapi/v1/klines", {
            "symbol": symbol, "interval": bi, "limit": min(limit, 1500),
        })

        if not isinstance(data, list):
            return []

        bars = []
        for k in data:
            try:
                bars.append({
                    "timestamp": datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc),
                    "open": float(k[1]), "high": float(k[2]),
                    "low": float(k[3]), "close": float(k[4]),
                    "volume": float(k[5]),
                })
            except (IndexError, ValueError):
                continue
        return bars

    # ── ExchangeAdapter stubs (paper mode only) ──

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """No-op for paper mode."""
        return True

    async def get_open_positions(self) -> dict[str, dict]:
        """No positions (paper mode)."""
        return {}

    async def get_open_orders(self) -> list[dict]:
        """No orders (paper mode)."""
        return []

    async def get_account_balance(self) -> float:
        return 0.0

    async def get_instrument_info(self, symbol: str) -> dict:
        return {"tick_size": 0.01, "min_qty": 0.001, "qty_step": 0.001}

    async def place_limit_order(self, symbol, side, quantity, price,
                                post_only=True, reduce_only=False) -> dict:
        return {}

    async def place_market_order(self, symbol, side, quantity,
                                 reduce_only=False) -> dict:
        return {}

    async def cancel_order(self, symbol, order_id) -> bool:
        return True

    async def get_order_status(self, symbol, order_id) -> dict:
        return {}

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
