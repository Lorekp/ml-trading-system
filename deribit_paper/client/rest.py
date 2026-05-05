"""Async Deribit REST client.

Thin wrapper around ``httpx.AsyncClient`` with:

- Automatic OAuth2 bearer token injection via :class:`DeribitAuth`
- Rate-limit retry with exponential backoff for code 10028
- JSON-RPC error decoding into the :mod:`deribit_paper.client.errors`
  exception hierarchy
- DRY_RUN honoring on all write paths (place_order, cancel_order, etc.)

Covers only the endpoints we actually need for the paper-trading loop:

- ``public/get_instruments`` — find active option + perp instruments
- ``public/ticker`` — current bid/ask/mark for an instrument
- ``public/get_last_trades_by_instrument`` — historical trade seed at startup
- ``public/get_index_price`` — index price fallback when WS isn't up yet
- ``private/get_account_summary`` — equity, margin, demo BTC balance
- ``private/get_positions`` — reconcile open positions on startup
- ``private/buy`` / ``private/sell`` — place orders (straddle legs + perp hedges)
- ``private/cancel`` — cancel an order
- ``private/get_open_orders_by_currency`` — reconcile pending orders

All JSON-RPC method names match Deribit's API docs exactly.
"""

from __future__ import annotations

import asyncio
import logging
import random
from typing import Any

import httpx

from deribit_paper.client.auth import DeribitAuth
from deribit_paper.client.errors import (
    DeribitAuthError,
    DeribitError,
    DeribitOrderRejected,
    DeribitRateLimitError,
    DeribitRequestError,
)
from deribit_paper.risk.dry_run import get_dry_run

logger = logging.getLogger(__name__)


# Deribit rate-limit error code from the JSON-RPC response
_RATE_LIMIT_CODE = 10028

# HTTP status codes that indicate a transient network/server issue
_TRANSIENT_HTTP_STATUSES = (500, 502, 503, 504)


class DeribitRESTClient:
    """Async REST client for Deribit with auth + retries + DRY_RUN honour.

    Normal usage (inside an async context)::

        auth = DeribitAuth(client_id, client_secret, rest_url)
        rest = DeribitRESTClient(auth, rest_url)
        async with rest:
            info = await rest.get_account_summary(currency="BTC")
            order = await rest.place_order(
                instrument="BTC-PERPETUAL",
                side="buy",
                amount=0.1,
                type="market",
            )
    """

    def __init__(
        self,
        auth: DeribitAuth | None,
        rest_base_url: str,
        max_retries: int = 3,
        retry_base_delay: float = 0.5,
    ):
        # ``auth`` may be None in paper-mode where the engine connects
        # to mainnet for read-only public quotes (ticker, index price)
        # and never invokes a private endpoint. Authenticated requests
        # raise a clear error rather than crashing on AttributeError.
        self._auth = auth
        self._rest_base_url = rest_base_url.rstrip("/")
        self._max_retries = max_retries
        self._retry_base_delay = retry_base_delay
        self._http: httpx.AsyncClient | None = None
        self._closed = False

    async def __aenter__(self) -> "DeribitRESTClient":
        await self._ensure_http()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def _ensure_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self._rest_base_url,
                timeout=httpx.Timeout(15.0, connect=10.0),
                http2=True,
            )
        return self._http

    async def close(self) -> None:
        if self._http is not None:
            await self._http.aclose()
            self._http = None
        self._closed = True

    # ── Public endpoints (no auth required) ──────────────────────────

    async def get_instruments(
        self,
        currency: str = "BTC",
        kind: str = "option",
        expired: bool = False,
    ) -> list[dict[str, Any]]:
        result = await self._request(
            "GET",
            "/public/get_instruments",
            params={"currency": currency, "kind": kind, "expired": str(expired).lower()},
            authenticated=False,
        )
        return result

    async def get_ticker(self, instrument_name: str) -> dict[str, Any]:
        return await self._request(
            "GET",
            "/public/ticker",
            params={"instrument_name": instrument_name},
            authenticated=False,
        )

    async def get_last_trades_by_instrument(
        self,
        instrument_name: str,
        count: int = 1000,
    ) -> dict[str, Any]:
        return await self._request(
            "GET",
            "/public/get_last_trades_by_instrument",
            params={"instrument_name": instrument_name, "count": count},
            authenticated=False,
        )

    async def get_index_price(self, index_name: str = "btc_usd") -> dict[str, Any]:
        return await self._request(
            "GET",
            "/public/get_index_price",
            params={"index_name": index_name},
            authenticated=False,
        )

    # ── Private endpoints (auth required) ────────────────────────────

    async def get_account_summary(
        self, currency: str = "BTC", extended: bool = False,
    ) -> dict[str, Any]:
        return await self._request(
            "GET",
            "/private/get_account_summary",
            params={"currency": currency, "extended": str(extended).lower()},
            authenticated=True,
        )

    async def get_positions(
        self, currency: str = "BTC", kind: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"currency": currency}
        if kind is not None:
            params["kind"] = kind
        return await self._request(
            "GET",
            "/private/get_positions",
            params=params,
            authenticated=True,
        )

    async def get_open_orders_by_currency(
        self, currency: str = "BTC", kind: str | None = None,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {"currency": currency}
        if kind is not None:
            params["kind"] = kind
        return await self._request(
            "GET",
            "/private/get_open_orders_by_currency",
            params=params,
            authenticated=True,
        )

    async def place_order(
        self,
        *,
        instrument_name: str,
        side: str,
        amount: float,
        type: str = "limit",
        price: float | None = None,
        label: str | None = None,
        post_only: bool | None = None,
        reduce_only: bool | None = None,
        time_in_force: str | None = None,
    ) -> dict[str, Any]:
        """Place a buy or sell order.

        Honours the global DRY_RUN flag — in dry-run mode, returns a
        stub response that mimics Deribit's structure so downstream
        code can proceed without touching the exchange.

        Parameters
        ----------
        side : str
            ``"buy"`` or ``"sell"`` — dispatches to ``/private/buy`` or
            ``/private/sell`` respectively.
        """
        if side not in ("buy", "sell"):
            raise ValueError(f"side must be 'buy' or 'sell', got {side!r}")

        params: dict[str, Any] = {
            "instrument_name": instrument_name,
            "amount": amount,
            "type": type,
        }
        if price is not None:
            params["price"] = price
        if label is not None:
            params["label"] = label
        if post_only is not None:
            params["post_only"] = str(post_only).lower()
        if reduce_only is not None:
            params["reduce_only"] = str(reduce_only).lower()
        if time_in_force is not None:
            params["time_in_force"] = time_in_force

        if get_dry_run():
            # In dry-run, fetch real mark price so PnL tracking works.
            # Public endpoint — no auth needed, works even in dry-run.
            fill_price = price
            if fill_price is None:
                try:
                    ticker = await self.get_ticker(instrument_name)
                    fill_price = float(ticker.get("mark_price", 0.0) or 0.0)
                except Exception:
                    fill_price = 0.0
            logger.info(
                "DRY_RUN place_order %s %s %.4f @ %s type=%s label=%s",
                side, instrument_name, amount, fill_price, type, label,
            )
            return self._stub_fill(instrument_name, side, amount, fill_price, type)

        endpoint = f"/private/{side}"
        return await self._request("GET", endpoint, params=params, authenticated=True)

    async def cancel_order(self, order_id: str) -> dict[str, Any]:
        if get_dry_run():
            logger.info("DRY_RUN cancel_order %s", order_id)
            return {"order_id": order_id, "order_state": "cancelled", "_stub": True}

        return await self._request(
            "GET",
            "/private/cancel",
            params={"order_id": order_id},
            authenticated=True,
        )

    # ── Core request pipeline ────────────────────────────────────────

    async def _request(
        self,
        method: str,
        path: str,
        *,
        params: dict[str, Any] | None = None,
        authenticated: bool = False,
    ) -> Any:
        """Perform a single REST request with retries + auth injection."""
        if self._closed:
            raise DeribitRequestError("client is closed")

        http = await self._ensure_http()
        headers: dict[str, str] = {}
        if authenticated:
            if self._auth is None:
                raise DeribitAuthError(
                    f"REST endpoint {path} requires authentication but no "
                    "DeribitAuth instance was provided (paper_mode=True "
                    "with no DERIBIT_PROD_* credentials). The caller "
                    "should gate this call on cfg.paper_mode.",
                )
            token = await self._auth.get_token()
            headers["Authorization"] = f"Bearer {token}"

        attempt = 0
        while True:
            attempt += 1
            try:
                resp = await http.request(
                    method, path, params=params, headers=headers,
                )
            except httpx.HTTPError as exc:
                if attempt < self._max_retries and _is_transient(exc):
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "Deribit transient network error (%s) — retry %d/%d in %.2fs",
                        exc, attempt, self._max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                raise DeribitRequestError(
                    f"HTTP error after {attempt} attempts: {exc}",
                ) from exc

            if resp.status_code in _TRANSIENT_HTTP_STATUSES and attempt < self._max_retries:
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Deribit HTTP %d — retry %d/%d in %.2fs",
                    resp.status_code, attempt, self._max_retries, delay,
                )
                await asyncio.sleep(delay)
                continue

            try:
                body = resp.json()
            except ValueError as exc:
                raise DeribitRequestError(
                    f"Deribit returned non-JSON body (status {resp.status_code}): {resp.text[:200]}",
                ) from exc

            if "error" in body:
                err = body["error"]
                code = err.get("code")
                message = err.get("message", "unknown error")
                if code == _RATE_LIMIT_CODE and attempt < self._max_retries:
                    delay = self._backoff_delay(attempt)
                    logger.warning(
                        "Deribit rate limit — retry %d/%d in %.2fs",
                        attempt, self._max_retries, delay,
                    )
                    await asyncio.sleep(delay)
                    continue
                if code == _RATE_LIMIT_CODE:
                    raise DeribitRateLimitError(message, code=code, data=err)
                if code in (13_009, 13_004, 13_005):  # auth-related codes
                    raise DeribitAuthError(message, code=code, data=err)
                # Order rejections land here too — the caller can pattern-match
                if "order" in path or "buy" in path or "sell" in path:
                    raise DeribitOrderRejected(
                        message, code=code, data=err,
                        retryable=code in (10_028, 10_040),  # too_many, busy
                    )
                raise DeribitRequestError(message, code=code, data=err)

            return body.get("result")

    def _backoff_delay(self, attempt: int) -> float:
        """Exponential backoff with jitter for the retry loop."""
        base = self._retry_base_delay * (2 ** (attempt - 1))
        jitter = base * 0.25 * random.random()
        return base + jitter

    @staticmethod
    def _stub_fill(
        instrument_name: str,
        side: str,
        amount: float,
        price: float | None,
        type: str,
    ) -> dict[str, Any]:
        """Return a fake fill that mimics Deribit's structure for dry-run."""
        import time as _time
        import uuid as _uuid
        fill_price = price if price is not None else 0.0
        return {
            "_stub": True,
            "order": {
                "order_id": f"stub_{_uuid.uuid4().hex[:12]}",
                "order_state": "filled",
                "instrument_name": instrument_name,
                "direction": side,
                "amount": amount,
                "filled_amount": amount,
                "average_price": fill_price,
                "price": fill_price,
                "order_type": type,
                "creation_timestamp": int(_time.time() * 1000),
            },
            "trades": [
                {
                    "trade_id": f"stub_t_{_uuid.uuid4().hex[:12]}",
                    "instrument_name": instrument_name,
                    "direction": side,
                    "amount": amount,
                    "price": fill_price,
                    "timestamp": int(_time.time() * 1000),
                    "fee": 0.0,
                    "fee_currency": "BTC",
                },
            ],
        }


def _is_transient(exc: httpx.HTTPError) -> bool:
    """Heuristic: is this HTTP error worth retrying?

    Connection timeouts, connect errors, and read timeouts are retried;
    permanent errors like ``InvalidURL`` are not.
    """
    return isinstance(
        exc,
        (
            httpx.ConnectError,
            httpx.ConnectTimeout,
            httpx.ReadTimeout,
            httpx.WriteTimeout,
            httpx.PoolTimeout,
            httpx.RemoteProtocolError,
        ),
    )
