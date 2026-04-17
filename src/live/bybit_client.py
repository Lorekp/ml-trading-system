"""Bybit V5 REST client for linear USDT perpetuals (demo mode).

Adapted from backups/2026-03-24_execution_engine/live/bybit_adapter.py.
Handles HMAC-SHA256 auth, rate limiting, retries, and demo URL enforcement.

Key additions over backup:
- demo=True forces api-demo.bybit.com
- place_market_order() for MkrTkr60 taker fallback
- get_instrument_info() for tick/lot size
- 3h interval support
"""

from __future__ import annotations

import asyncio
import base64
import hashlib
import hmac
import json
import logging
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlencode

import aiohttp

logger = logging.getLogger(__name__)

# Interval mapping: our format → Bybit format
_INTERVAL_MAP: dict[str, str] = {
    "1m": "1", "5m": "5", "15m": "15", "15min": "15", "30m": "30",
    "1h": "60", "3h": "180", "4h": "240",
    "1d": "D", "1D": "D", "24h": "D",
}

RECV_WINDOW = "5000"

# Siebly broker IDs — different for EU vs global
_BROKER_ID_GLOBAL = "bybitapinode"
_BROKER_ID_EU = "Cg000971"


class BybitClient:
    """Bybit V5 linear perpetuals REST client.

    Designed for demo paper trading. Forces demo URL when demo=True.
    """

    def __init__(
        self,
        api_key: str = "",
        api_secret: str = "",
        demo: bool = True,
        base_url: str = "",
        rate_limit_per_sec: int = 10,
        timeout_s: float = 10.0,
        private_key_path: str = "",
    ) -> None:
        self._api_key = api_key
        self._api_secret = api_secret
        self._private_key = None
        if private_key_path:
            self._private_key = self._load_rsa_key(private_key_path)
            logger.info("RSA signing enabled (self-generated API key)")
        self._demo = demo
        self._timeout_s = timeout_s
        self._session: Optional[aiohttp.ClientSession] = None
        self._semaphore = asyncio.Semaphore(rate_limit_per_sec)
        self._last_request_time: float = 0.0
        self._instrument_cache: dict[str, dict] = {}

        if demo:
            self._base_url = "https://api-demo.bybit.com"
            logger.warning("DEMO MODE: using api-demo.bybit.com — no real money")
        else:
            self._base_url = base_url or "https://api.bybit.com"

        # Auto-detect EU region from base URL
        self._is_eu = ".eu" in self._base_url
        self._broker_id = _BROKER_ID_EU if self._is_eu else _BROKER_ID_GLOBAL
        if self._is_eu:
            logger.info("EU region detected — using broker ID %s", self._broker_id)

    @classmethod
    def from_config(cls, config) -> BybitClient:
        """Create from an ExchangeConfig."""
        return cls(
            api_key=config.api_key,
            api_secret=config.api_secret,
            demo=config.demo,
            base_url=config.base_url,
            rate_limit_per_sec=config.rate_limit_per_sec,
            timeout_s=config.timeout_s,
            private_key_path=getattr(config, "private_key_path", ""),
        )

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._timeout_s)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    @staticmethod
    def _load_rsa_key(path: str):
        """Load RSA private key from PEM file."""
        from cryptography.hazmat.primitives.serialization import load_pem_private_key
        pem = Path(path).read_bytes()
        return load_pem_private_key(pem, password=None)

    def _sign(self, timestamp: str, params_str: str) -> str:
        """Sign request — RSA-SHA256 for self-generated keys, HMAC-SHA256 otherwise."""
        sign_str = f"{timestamp}{self._api_key}{RECV_WINDOW}{params_str}"
        if self._private_key is not None:
            from cryptography.hazmat.primitives import hashes
            from cryptography.hazmat.primitives.asymmetric import padding
            sig = self._private_key.sign(
                sign_str.encode("utf-8"),
                padding.PKCS1v15(),
                hashes.SHA256(),
            )
            return base64.b64encode(sig).decode("utf-8")
        return hmac.new(
            self._api_secret.encode("utf-8"),
            sign_str.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _auth_headers(self, params_str: str = "") -> dict[str, str]:
        timestamp = str(int(time.time() * 1000))
        return {
            "X-BAPI-API-KEY": self._api_key,
            "X-BAPI-TIMESTAMP": timestamp,
            "X-BAPI-SIGN": self._sign(timestamp, params_str),
            "X-BAPI-RECV-WINDOW": RECV_WINDOW,
            "Content-Type": "application/json",
            "x-referer": self._broker_id,
        }

    async def _request(
        self,
        method: str,
        path: str,
        params: Optional[dict] = None,
        body: Optional[dict] = None,
        auth: bool = False,
    ) -> Any:
        """Rate-limited, retried HTTP request."""
        session = await self._get_session()
        url = f"{self._base_url}{path}"
        last_error: Optional[Exception] = None

        for attempt in range(3):
            async with self._semaphore:
                now = time.monotonic()
                elapsed = now - self._last_request_time
                if elapsed < 0.1:
                    await asyncio.sleep(0.1 - elapsed)

                try:
                    headers = {}
                    if auth:
                        if method == "GET" and params:
                            params_str = urlencode(sorted(params.items()))
                        elif method == "POST" and body:
                            params_str = json.dumps(body)
                        else:
                            params_str = ""
                        headers = self._auth_headers(params_str)

                    if method == "GET":
                        async with session.get(url, params=params, headers=headers) as resp:
                            self._last_request_time = time.monotonic()
                            if resp.status == 429:
                                wait = 2 ** attempt
                                logger.warning("Rate limited, waiting %ds", wait)
                                await asyncio.sleep(wait)
                                continue
                            resp.raise_for_status()
                            data = await resp.json()
                    else:
                        async with session.post(url, json=body, headers=headers) as resp:
                            self._last_request_time = time.monotonic()
                            if resp.status == 429:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            resp.raise_for_status()
                            data = await resp.json()

                    ret_code = data.get("retCode", -1)
                    if ret_code != 0:
                        ret_msg = data.get("retMsg", "unknown error")
                        # Don't retry terminal errors
                        no_retry = {
                            110004, 110007,  # insufficient balance
                            110001,          # order does not exist
                            110008,          # order already completed
                            10003, 10004, 10005,  # auth failures
                        }
                        if ret_code in no_retry:
                            logger.error("Bybit terminal error: %s (code %d)", ret_msg, ret_code)
                            return data
                        if ret_code == 10006:  # rate limit
                            logger.warning("Rate limited (10006), backing off 2s")
                            await asyncio.sleep(2)
                            continue
                        if ret_code == 10002:  # clock drift
                            logger.error("Clock drift detected (10002) — check NTP sync")
                        logger.warning("Bybit API error: %s (code %s)", ret_msg, ret_code)
                        if attempt < 2:
                            await asyncio.sleep(1)
                            continue
                        return data

                    return data

                except (aiohttp.ClientError, asyncio.TimeoutError) as e:
                    last_error = e
                    logger.warning("Request failed (attempt %d): %s", attempt + 1, e)
                    await asyncio.sleep(2 ** attempt)

        raise last_error or RuntimeError("Bybit request failed after retries")

    # ── Market Data ──

    async def fetch_klines(
        self, symbol: str, interval: str, limit: int = 200
    ) -> list[dict]:
        """Fetch klines from Bybit V5. Returns list of OHLCV dicts."""
        bybit_interval = _INTERVAL_MAP.get(interval)
        if bybit_interval is None:
            logger.warning("Unsupported interval: %s", interval)
            return []

        params = {
            "category": "linear",
            "symbol": symbol,
            "interval": bybit_interval,
            "limit": str(min(limit, 1000)),
        }

        try:
            data = await self._request("GET", "/v5/market/kline", params=params)
            raw_list = data.get("result", {}).get("list", [])
        except Exception as e:
            logger.error("Klines %s: %s", symbol, e)
            return []

        bars = []
        # Bybit returns newest first — reverse for chronological order
        for candle in reversed(raw_list):
            try:
                bars.append({
                    "timestamp": datetime.fromtimestamp(
                        int(candle[0]) / 1000, tz=timezone.utc),
                    "open": float(candle[1]),
                    "high": float(candle[2]),
                    "low": float(candle[3]),
                    "close": float(candle[4]),
                    "volume": float(candle[5]),
                })
            except (IndexError, ValueError):
                continue

        return bars

    async def fetch_ticker(self, symbol: str) -> dict[str, float]:
        """Fetch current ticker (last, bid, ask)."""
        params = {"category": "linear", "symbol": symbol}
        try:
            data = await self._request("GET", "/v5/market/tickers", params=params)
            tickers = data.get("result", {}).get("list", [])
            if tickers:
                t = tickers[0]
                return {
                    "last_price": float(t.get("lastPrice", 0)),
                    "bid": float(t.get("bid1Price", 0)),
                    "ask": float(t.get("ask1Price", 0)),
                    "volume_24h": float(t.get("volume24h", 0)),
                }
        except Exception as e:
            logger.error("Ticker %s: %s", symbol, e)
        return {"last_price": 0.0, "bid": 0.0, "ask": 0.0, "volume_24h": 0.0}

    async def fetch_all_tickers(self) -> list[dict]:
        """Fetch tickers for ALL USDT linear perpetuals in one call.

        Uses /v5/market/tickers without symbol param → returns full market.
        No auth required (public endpoint).

        Returns list of {symbol, asset, last_price, volume_24h, turnover_24h}.
        """
        params = {"category": "linear"}
        try:
            data = await self._request("GET", "/v5/market/tickers", params=params)
            tickers = data.get("result", {}).get("list", [])
        except Exception as e:
            logger.error("fetch_all_tickers failed: %s", e)
            return []

        result = []
        for t in tickers:
            sym = t.get("symbol", "")
            if not sym.endswith("USDT"):
                continue
            result.append({
                "symbol": sym,
                "asset": sym.replace("USDT", ""),
                "last_price": float(t.get("lastPrice", 0)),
                "volume_24h": float(t.get("volume24h", 0)),
                "turnover_24h": float(t.get("turnover24h", 0)),
            })
        logger.debug("fetch_all_tickers: %d USDT perps", len(result))
        return result

    async def get_instrument_info(self, symbol: str) -> dict:
        """Fetch tick size and lot size for a symbol. Cached."""
        if symbol in self._instrument_cache:
            return self._instrument_cache[symbol]

        params = {"category": "linear", "symbol": symbol}
        try:
            data = await self._request("GET", "/v5/market/instruments-info",
                                       params=params)
            instruments = data.get("result", {}).get("list", [])
            if instruments:
                inst = instruments[0]
                price_filter = inst.get("priceFilter", {})
                lot_filter = inst.get("lotSizeFilter", {})
                info = {
                    "tick_size": float(price_filter.get("tickSize", 0.01)),
                    "min_qty": float(lot_filter.get("minOrderQty", 0.001)),
                    "qty_step": float(lot_filter.get("qtyStep", 0.001)),
                }
                self._instrument_cache[symbol] = info
                return info
        except Exception as e:
            logger.error("Instrument info %s: %s", symbol, e)

        return {"tick_size": 0.01, "min_qty": 0.001, "qty_step": 0.001}

    # ── Order Management ──

    async def place_limit_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        price: float,
        post_only: bool = True,
        reduce_only: bool = False,
    ) -> dict:
        """Place a limit order. Returns {order_id, status, error?}."""
        # Round price and qty to valid increments
        info = await self.get_instrument_info(symbol)
        price = self._round_to_tick(price, info["tick_size"])
        quantity = self._round_to_step(quantity, info["qty_step"])

        if quantity < info["min_qty"]:
            return {"order_id": "", "status": "rejected", "error": "below_min_qty"}

        body = {
            "category": "linear",
            "symbol": symbol,
            "side": side,  # "Buy" or "Sell"
            "orderType": "Limit",
            "qty": str(quantity),
            "price": str(price),
            "timeInForce": "PostOnly" if post_only else "GTC",
            "positionIdx": 0,
            "orderLinkId": f"ml-{uuid.uuid4().hex[:16]}",
        }
        if reduce_only:
            body["reduceOnly"] = True

        try:
            data = await self._request("POST", "/v5/order/create",
                                       body=body, auth=True)
            result = data.get("result", {})
            order_id = result.get("orderId", "")

            if data.get("retCode") == 0 and order_id:
                logger.info("Limit order: %s %s %.6f @ %.4f (PostOnly=%s)",
                            symbol, side, quantity, price, post_only)
                return {"order_id": order_id, "status": "submitted"}

            return {
                "order_id": "",
                "status": "rejected",
                "error": data.get("retMsg", "unknown"),
                "retCode": data.get("retCode"),
            }
        except Exception as e:
            logger.error("Order %s: %s", symbol, e)
            return {"order_id": "", "status": "rejected", "error": str(e)}

    async def place_market_order(
        self,
        symbol: str,
        side: str,
        quantity: float,
        reduce_only: bool = False,
    ) -> dict:
        """Place a market order (taker). Returns {order_id, status, error?}."""
        info = await self.get_instrument_info(symbol)
        quantity = self._round_to_step(quantity, info["qty_step"])

        if quantity < info["min_qty"]:
            return {"order_id": "", "status": "rejected", "error": "below_min_qty"}

        body = {
            "category": "linear",
            "symbol": symbol,
            "side": side,
            "orderType": "Market",
            "qty": str(quantity),
            "positionIdx": 0,
            "orderLinkId": f"ml-{uuid.uuid4().hex[:16]}",
        }
        if reduce_only:
            body["reduceOnly"] = True

        try:
            data = await self._request("POST", "/v5/order/create",
                                       body=body, auth=True)
            result = data.get("result", {})
            order_id = result.get("orderId", "")

            if data.get("retCode") == 0 and order_id:
                logger.info("Market order: %s %s %.6f", symbol, side, quantity)
                return {"order_id": order_id, "status": "submitted"}

            return {
                "order_id": "",
                "status": "rejected",
                "error": data.get("retMsg", "unknown"),
            }
        except Exception as e:
            logger.error("Market order %s: %s", symbol, e)
            return {"order_id": "", "status": "rejected", "error": str(e)}

    async def cancel_order(self, symbol: str, order_id: str) -> bool:
        """Cancel an order by ID."""
        body = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id,
        }
        try:
            data = await self._request("POST", "/v5/order/cancel",
                                       body=body, auth=True)
            success = data.get("retCode") == 0
            if success:
                logger.info("Cancelled order %s on %s", order_id, symbol)
            return success
        except Exception as e:
            logger.error("Cancel %s/%s: %s", symbol, order_id, e)
            return False

    async def get_order_status(self, symbol: str, order_id: str) -> dict:
        """Query order status. Returns {status, filled_qty, filled_price}."""
        params = {
            "category": "linear",
            "symbol": symbol,
            "orderId": order_id,
        }
        try:
            data = await self._request("GET", "/v5/order/realtime",
                                       params=params, auth=True)
            orders = data.get("result", {}).get("list", [])
            if not orders:
                return {"status": "unknown", "filled_qty": 0.0, "filled_price": 0.0}

            order = orders[0]
            status_map = {
                "New": "new",
                "PartiallyFilled": "partially_filled",
                "Filled": "filled",
                "Cancelled": "cancelled",
                "PartiallyFilledCanceled": "partially_filled",
                "Rejected": "cancelled",
                "Deactivated": "cancelled",
            }

            return {
                "status": status_map.get(order.get("orderStatus", ""), "unknown"),
                "filled_qty": float(order.get("cumExecQty", 0)),
                "filled_price": float(order.get("avgPrice", 0)),
            }
        except Exception as e:
            logger.warning("Order status %s: %s", order_id, e)
            return {"status": "unknown", "filled_qty": 0.0, "filled_price": 0.0}

    # ── Account ──

    async def get_account_balance(self) -> float:
        """Fetch USDT wallet equity."""
        params = {"accountType": "UNIFIED", "coin": "USDT"}
        try:
            data = await self._request("GET", "/v5/account/wallet-balance",
                                       params=params, auth=True)
            accounts = data.get("result", {}).get("list", [])
            if accounts:
                return float(accounts[0].get("totalEquity", 0))
            return 0.0
        except Exception as e:
            logger.error("Balance: %s", e)
            return 0.0

    async def get_open_positions(self) -> dict[str, dict]:
        """Fetch all open positions."""
        params = {"category": "linear", "settleCoin": "USDT"}
        try:
            data = await self._request("GET", "/v5/position/list",
                                       params=params, auth=True)
            positions: dict[str, dict] = {}
            for pos in data.get("result", {}).get("list", []):
                size = float(pos.get("size", 0))
                if abs(size) < 1e-10:
                    continue
                symbol = pos.get("symbol", "")
                positions[symbol] = {
                    "side": pos.get("side", "").lower(),
                    "size": size,
                    "entry_price": float(pos.get("avgPrice", 0)),
                    "unrealised_pnl": float(pos.get("unrealisedPnl", 0)),
                    "leverage": pos.get("leverage", "1"),
                }
            return positions
        except Exception as e:
            logger.error("Positions: %s", e)
            return {}

    async def set_leverage(self, symbol: str, leverage: int) -> bool:
        """Set leverage for a symbol."""
        body = {
            "category": "linear",
            "symbol": symbol,
            "buyLeverage": str(leverage),
            "sellLeverage": str(leverage),
        }
        try:
            data = await self._request("POST", "/v5/position/set-leverage",
                                       body=body, auth=True)
            return data.get("retCode") in (0, 110043)  # 110043 = already set
        except Exception as e:
            logger.error("Leverage %s: %s", symbol, e)
            return False

    # ── Risk Management ──

    async def place_trading_stop(
        self, symbol: str, side: str, stop_loss: float,
    ) -> bool:
        """Attach a stop-loss to an existing open position.

        Uses Bybit V5 /v5/position/trading-stop endpoint.
        Full-position mode: market SL on entire position.

        Args:
            symbol: e.g. "BTCUSDT"
            side: Position side ("Buy" for long, "Sell" for short)
            stop_loss: Trigger price for the SL
        """
        info = await self.get_instrument_info(symbol)
        sl_price = self._round_to_tick(stop_loss, info["tick_size"])

        body = {
            "category": "linear",
            "symbol": symbol,
            "stopLoss": str(sl_price),
            "slTriggerBy": "MarkPrice",
            "tpslMode": "Full",
            "positionIdx": 0,
        }
        try:
            data = await self._request("POST", "/v5/position/trading-stop",
                                       body=body, auth=True)
            success = data.get("retCode") == 0
            if success:
                logger.info("SL placed: %s %s @ %.4f", symbol, side, sl_price)
            else:
                logger.warning("SL failed %s: %s", symbol, data.get("retMsg"))
            return success
        except Exception as e:
            logger.error("place_trading_stop %s: %s", symbol, e)
            return False

    # ── Pre-flight Validation ──

    async def validate_connection(self) -> bool:
        """Pre-flight check: verify API key, permissions, and clock sync.

        Returns True if connection is healthy and ready to trade.
        """
        checks_passed = True

        # 1. Clock sync — compare local time to server time
        try:
            data = await self._request("GET", "/v5/market/time")
            server_ms = int(data.get("result", {}).get("timeSecond", "0")) * 1000
            if server_ms == 0:
                server_ms = int(data.get("result", {}).get("timeNano", "0")) // 1_000_000
            local_ms = int(time.time() * 1000)
            drift_ms = abs(local_ms - server_ms)
            if drift_ms > 2000:
                logger.error("Clock drift: %dms — sync with NTP", drift_ms)
                checks_passed = False
            else:
                logger.info("Clock sync OK (drift: %dms)", drift_ms)
        except Exception as e:
            logger.error("Clock check failed: %s", e)
            checks_passed = False

        # 2. API key + balance (verifies auth + trade permissions)
        try:
            balance = await self.get_account_balance()
            if balance >= 0:
                logger.info("Account balance: $%.2f", balance)
            else:
                logger.error("Failed to fetch balance — check API key permissions")
                checks_passed = False
        except Exception as e:
            logger.error("Balance check failed: %s", e)
            checks_passed = False

        # 3. Verify one-way position mode (positionIdx=0)
        try:
            params = {"category": "linear", "symbol": "BTCUSDT"}
            data = await self._request("GET", "/v5/position/list",
                                       params=params, auth=True)
            positions = data.get("result", {}).get("list", [])
            if positions:
                pos_idx = int(positions[0].get("positionIdx", 0))
                if pos_idx != 0:
                    logger.error("Position mode is hedge (idx=%d) — must be one-way (0)", pos_idx)
                    checks_passed = False
            logger.info("Position mode: one-way (OK)")
        except Exception as e:
            logger.warning("Position mode check: %s", e)

        if checks_passed:
            logger.info("Pre-flight validation PASSED")
        else:
            logger.error("Pre-flight validation FAILED — do not trade")

        return checks_passed

    async def close(self) -> None:
        """Close HTTP session."""
        if self._session and not self._session.closed:
            await self._session.close()
        logger.info("Bybit client closed")

    # ── Helpers ──

    @staticmethod
    def _round_to_tick(price: float, tick_size: float) -> float:
        if tick_size <= 0:
            return price
        return round(round(price / tick_size) * tick_size, 10)

    @staticmethod
    def _round_to_step(qty: float, step: float) -> float:
        if step <= 0:
            return qty
        return round(int(qty / step) * step, 10)
