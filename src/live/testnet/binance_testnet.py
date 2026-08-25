"""Binance USDT-M Futures Testnet adapter.

HMAC-signed REST + signed order endpoints against
``https://testnet.binancefuture.com``.  Implements both the DataProvider and
ExchangeAdapter protocols so the production ``TradingEngine`` treats it as
a drop-in replacement for ``BinanceFuturesData``.

Testnet specifics
-----------------
- Test funds are minted by ``https://testnet.binancefuture.com`` (sign-up).
- Symbols are the SAME contract set as production (BTCUSDT, ETHUSDT, ...).
- Fees mirror production VIP tier rates; account snapshot in test mode is
  meaningless (always ~3000 USDT demo funds).  We pull real testnet fills
  for PnL reconciliation; account balance is reported as configured
  ``capital`` to keep the engine's portfolio math consistent.

The adapter is read-only safe — ``place_*_order`` requires ``api_key`` +
``api_secret`` to be set.  If either is empty, the adapter logs a clear
warning and falls back to ``calibrated-paper`` behaviour (so the CLI is
testable without testnet creds).
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import time
import urllib.parse
from datetime import datetime, timezone
from typing import Optional

import aiohttp
import pandas as pd

logger = logging.getLogger(__name__)

# Testnet base URL — production futures is fapi.binance.com.
TESTNET_BASE = "https://testnet.binancefuture.com"

_INTERVAL_MAP = {
    "1m": "1m", "5m": "5m", "15m": "15m", "30m": "30m",
    "1h": "1h", "3h": "3h", "4h": "4h",
    "1d": "1d", "1D": "1d", "24h": "1d",
}


class BinanceFuturesTestnet:
    """HMAC-signed Binance USDT-M Futures Testnet adapter.

    Drop-in for ``BinanceFuturesData``: implements both DataProvider
    (fetch_ohlcv / fetch_ticker / fetch_all_tickers / fetch_klines) and
    ExchangeAdapter (place / cancel / status / positions / orders / balance /
    leverage / instrument info / ticker / validate_connection / close).
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        base_url: str = TESTNET_BASE,
        rate_limit_per_sec: int = 5,
        timeout_s: float = 10.0,
    ) -> None:
        self._api_key = api_key or ""
        self._api_secret = api_secret or ""
        self._base_url = base_url.rstrip("/")
        self._semaphore = asyncio.Semaphore(rate_limit_per_sec)
        self._timeout = aiohttp.ClientTimeout(total=timeout_s)
        self._session: aiohttp.ClientSession | None = None
        self._last_request_time = 0.0
        # recvWindow generous enough for a slow VM clock skew.
        self._recv_window_ms = 5000

        if not self._api_key or not self._api_secret:
            logger.warning(
                "BinanceFuturesTestnet: missing api_key/api_secret — "
                "place_*_order / get_account_balance will return empty; "
                "paper-mode only."
            )

    # ── HTTP plumbing ──

    def _ensure_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)
        return self._session

    def _sign(self, params: dict) -> str:
        """HMAC-SHA256 of the URL-encoded query string."""
        qs = urllib.parse.urlencode(params)
        return hmac.new(
            self._api_secret.encode("utf-8"),
            qs.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    async def _request(
        self,
        method: str,
        endpoint: str,
        params: Optional[dict] = None,
        signed: bool = False,
        max_retries: int = 3,
    ) -> dict | list:
        params = dict(params or {})
        if signed:
            params.setdefault("recvWindow", self._recv_window_ms)
            params["timestamp"] = int(time.time() * 1000)
            params["signature"] = self._sign(params)

        url = f"{self._base_url}{endpoint}"
        session = self._ensure_session()

        for attempt in range(max_retries):
            async with self._semaphore:
                # Minimum spacing — Binance is 1200 req/min = 20/sec but
                # we conservatively clamp to 5/sec.
                elapsed = time.monotonic() - self._last_request_time
                if elapsed < 0.2:
                    await asyncio.sleep(0.2 - elapsed)
                self._last_request_time = time.monotonic()

                try:
                    headers = {"X-MBX-APIKEY": self._api_key} if signed else {}
                    if method == "GET":
                        async with session.get(url, params=params, headers=headers) as r:
                            return await self._handle(r, attempt)
                    elif method == "POST":
                        async with session.post(url, params=params, headers=headers) as r:
                            return await self._handle(r, attempt)
                    elif method == "DELETE":
                        async with session.delete(url, params=params, headers=headers) as r:
                            return await self._handle(r, attempt)
                    else:
                        raise ValueError(f"unsupported method {method}")
                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    logger.warning(
                        "BinanceTestnet %s %s failed (attempt %d): %s",
                        method, endpoint, attempt + 1, e)
                    await asyncio.sleep(2 ** attempt)
        logger.error("BinanceTestnet %s %s failed after %d retries",
                     method, endpoint, max_retries)
        return {}

    async def _handle(self, resp: aiohttp.ClientResponse, attempt: int):
        if resp.status == 429:
            wait = 2 ** attempt
            logger.warning("BinanceTestnet 429, waiting %ds", wait)
            await asyncio.sleep(wait)
            raise aiohttp.ClientError("rate-limited")
        if resp.status == 418:
            logger.error("BinanceTestnet 418 IP ban")
            await asyncio.sleep(60)
            raise aiohttp.ClientError("ip-banned")
        text = await resp.text()
        try:
            data = await resp.json(content_type=None) if text else {}
        except Exception:
            data = {"raw": text}
        if isinstance(data, dict) and data.get("code") and int(data.get("code", 0)) < 0:
            logger.warning("BinanceTestnet API error: %s (code %s)",
                           data.get("msg"), data.get("code"))
        return data if text else {}

    # ── DataProvider ──

    async def fetch_ohlcv(
        self, symbol: str, interval: str, limit: int = 200
    ) -> pd.DataFrame:
        """Mirror BinanceFuturesData.fetch_ohlcv, including 3h resample."""
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
        data = await self._request(
            "GET", "/fapi/v1/klines",
            {"symbol": symbol, "interval": bi, "limit": min(limit, 1500)},
        )

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
        if df.empty:
            return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
        df.index = df["timestamp"]
        df.index.name = None
        return df[["open", "high", "low", "close", "volume"]]

    async def fetch_ticker(self, symbol: str) -> dict[str, float]:
        """24hr ticker + bookTicker for bid/ask (testnet has bookTicker)."""
        data = await self._request(
            "GET", "/fapi/v1/ticker/24hr", {"symbol": symbol})
        if not isinstance(data, dict):
            return {"last_price": 0, "bid": 0, "ask": 0,
                    "volume_24h": 0, "quote_volume_24h": 0}
        last = float(data.get("lastPrice", 0))

        # Best bid/ask from bookTicker — separate signed call.
        try:
            book = await self._request(
                "GET", "/fapi/v1/ticker/bookTicker", {"symbol": symbol})
        except Exception:
            book = {}
        bid = float(book.get("bidPrice", 0)) if isinstance(book, dict) else 0.0
        ask = float(book.get("askPrice", 0)) if isinstance(book, dict) else 0.0
        fallback_source = None
        if bid <= 0 or ask <= 0:
            # NEW (B-003): do NOT silently set bid=ask=last. That masks 100% of
            # the real slippage as 0 bps, hiding Sharpe collapse from the
            # reconciler. Instead use a synthetic half-spread of 1.5 bps around
            # last (matches B-030 worst-case) and tag the ticker so downstream
            # FillRecord carries `fallback_source='bookticker_missing'`.
            half_spread_bps = 1.5
            bid = last * (1.0 - half_spread_bps / 2.0 / 10_000.0)
            ask = last * (1.0 + half_spread_bps / 2.0 / 10_000.0)
            fallback_source = "bookticker_missing"
            logger.warning(
                "fetch_ticker: bookTicker missing for %s; using synthetic "
                "%.1f bps half-spread around last=%.4f",
                symbol, half_spread_bps, last)

        return {
            "last_price": last,
            "bid": bid,
            "ask": ask,
            "volume_24h": float(data.get("volume", 0)),
            "quote_volume_24h": float(data.get("quoteVolume", 0)),
            "fallback_source": fallback_source,
        }

    async def fetch_all_tickers(self) -> list[dict]:
        data = await self._request("GET", "/fapi/v1/ticker/24hr")
        if not isinstance(data, list):
            return []
        out = []
        for t in data:
            sym = t.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            out.append({
                "symbol": sym,
                "asset": sym.replace("USDT", ""),
                "last_price": float(t.get("lastPrice", 0)),
                "volume_24h": float(t.get("volume", 0)),
                "turnover_24h": float(t.get("quoteVolume", 0)),
            })
        return out

    async def fetch_klines(
        self, symbol: str, interval: str, limit: int = 200
    ) -> list[dict]:
        bi = _INTERVAL_MAP.get(interval, interval)
        data = await self._request(
            "GET", "/fapi/v1/klines",
            {"symbol": symbol, "interval": bi, "limit": min(limit, 1500)},
        )
        if not isinstance(data, list):
            return []
        out = []
        for k in data:
            try:
                out.append({
                    "timestamp": datetime.fromtimestamp(int(k[0]) / 1000, tz=timezone.utc),
                    "open": float(k[1]), "high": float(k[2]),
                    "low": float(k[3]), "close": float(k[4]),
                    "volume": float(k[5]),
                })
            except (IndexError, ValueError, TypeError):
                continue
        return out

    # ── ExchangeAdapter: orders ──

    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        post_only: bool = True,
        reduce_only: bool = False,
    ) -> dict:
        if not self._api_key or not self._api_secret:
            logger.warning("place_limit_order: missing testnet creds; stub")
            return {"orderId": "stub", "status": "STUB"}
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "LIMIT",
            "timeInForce": "GTX" if post_only else "GTC",
            "quantity": f"{quantity:.8f}",
            "price": f"{price:.8f}",
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        return await self._request("POST", "/fapi/v1/order", params, signed=True) or {}

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
    ) -> dict:
        if not self._api_key or not self._api_secret:
            logger.warning("place_market_order: missing testnet creds; stub")
            return {"orderId": "stub", "status": "STUB"}
        params = {
            "symbol": symbol,
            "side": side.upper(),
            "type": "MARKET",
            "quantity": f"{quantity:.8f}",
        }
        if reduce_only:
            params["reduceOnly"] = "true"
        return await self._request("POST", "/fapi/v1/order", params, signed=True) or {}

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        if not self._api_key:
            return False
        data = await self._request(
            "DELETE", "/fapi/v1/order",
            {"symbol": symbol, "orderId": order_id}, signed=True)
        return bool(data)

    async def get_order_status(self, symbol: str, order_id: str) -> dict:
        return await self._request(
            "GET", "/fapi/v1/order",
            {"symbol": symbol, "orderId": order_id}, signed=True) or {}

    # ── ExchangeAdapter: account ──

    async def get_open_positions(self) -> dict[str, dict]:
        if not self._api_key:
            return {}
        data = await self._request(
            "GET", "/fapi/v2/positionRisk", signed=True)
        if not isinstance(data, list):
            return {}
        out: dict[str, dict] = {}
        for p in data:
            amt = abs(float(p.get("positionAmt", 0) or 0))
            if amt > 0:
                sym = p.get("symbol", "")
                out[sym] = {
                    "symbol": sym,
                    "asset": sym.replace("USDT", ""),
                    "size": amt,
                    "entry_price": float(p.get("entryPrice", 0) or 0),
                    "mark_price": float(p.get("markPrice", 0) or 0),
                    "unrealized_pnl": float(p.get("unRealizedProfit", 0) or 0),
                    "side": "long" if float(p.get("positionAmt", 0)) > 0 else "short",
                    "leverage": float(p.get("leverage", 1) or 1),
                }
        return out

    async def get_open_orders(self) -> list[dict]:
        if not self._api_key:
            return []
        data = await self._request(
            "GET", "/fapi/v1/openOrders", signed=True)
        if not isinstance(data, list):
            return []
        return [
            {
                "order_id": str(o.get("orderId", "")),
                "symbol": o.get("symbol", ""),
                "side": o.get("side", ""),
                "qty": float(o.get("origQty", 0) or 0),
                "price": float(o.get("price", 0) or 0),
                "order_type": o.get("type", ""),
                "status": o.get("status", ""),
                "filled_qty": float(o.get("executedQty", 0) or 0),
            }
            for o in data
        ]

    async def get_account_balance(self) -> float:
        if not self._api_key:
            return 0.0
        data = await self._request(
            "GET", "/fapi/v2/account", signed=True)
        if not isinstance(data, dict):
            return 0.0
        for a in data.get("assets", []):
            if a.get("asset") == "USDT":
                return float(a.get("walletBalance", 0) or 0)
        return float(data.get("totalWalletBalance", 0) or 0)

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        if not self._api_key:
            return False
        data = await self._request(
            "POST", "/fapi/v1/leverage",
            {"symbol": symbol, "leverage": leverage}, signed=True)
        return bool(data) and "leverage" in (data or {})

    async def get_instrument_info(self, symbol: str) -> dict:
        data = await self._request(
            "GET", "/fapi/v1/exchangeInfo")
        if not isinstance(data, dict):
            return {"tick_size": 0.01, "min_qty": 0.001, "qty_step": 0.001,
                    "pricePrecision": 2, "quantityPrecision": 3}
        for s in data.get("symbols", []):
            if s.get("symbol") == symbol:
                f = {f["filterType"]: f for f in s.get("filters", [])}
                lot = f.get("LOT_SIZE", {})
                pf = f.get("PRICE_FILTER", {})
                return {
                    "tick_size": float(pf.get("tickSize", 0.01) or 0.01),
                    "min_qty": float(lot.get("minQty", 0.001) or 0.001),
                    "qty_step": float(lot.get("stepSize", 0.001) or 0.001),
                    "pricePrecision": int(s.get("pricePrecision", 2) or 2),
                    "quantityPrecision": int(s.get("quantityPrecision", 3) or 3),
                }
        return {"tick_size": 0.01, "min_qty": 0.001, "qty_step": 0.001,
                "pricePrecision": 2, "quantityPrecision": 3}

    async def place_trading_stop(
        self, symbol: str, side: str, stop_loss: float,
    ) -> bool:
        """Binance testnet supports STOP_MARKET via order endpoint."""
        if not self._api_key:
            return False
        params = {
            "symbol": symbol,
            "side": "SELL" if side.lower() == "buy" else "BUY",
            "type": "STOP_MARKET",
            "stopPrice": f"{stop_loss:.8f}",
            "closePosition": "true",
        }
        data = await self._request(
            "POST", "/fapi/v1/order", params, signed=True)
        return bool(data) and "orderId" in (data or {})

    async def validate_connection(self) -> bool:
        """HMAC-signed account probe — true if creds accepted by testnet."""
        if not self._api_key or not self._api_secret:
            logger.error("validate_connection: missing creds")
            return False
        try:
            data = await self._request(
                "GET", "/fapi/v2/account", signed=True, max_retries=1)
            ok = isinstance(data, dict) and ("totalWalletBalance" in data or "assets" in data)
            if ok:
                logger.info("validate_connection: OK (testnet account reachable)")
            else:
                logger.error("validate_connection: bad payload: %s", data)
            return ok
        except Exception as e:
            logger.error("validate_connection: %s", e)
            return False

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
