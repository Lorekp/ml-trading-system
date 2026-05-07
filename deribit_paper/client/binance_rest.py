"""Generic Binance Futures REST client (public endpoints only).

E (parity 2026-05-07) — built to replace the geo-blocked
``wss://fstream.binance.com/ws/btcusdt@aggTrade`` WebSocket feed with
REST polling of ``https://fapi.binance.com/fapi/v1/aggTrades``. From
the homeserver (Sweden ProtonVPN AS208172) the Futures WS is
geo-blocked (handshake succeeds, zero frames flow) but the REST
endpoint IS reachable.

Independent of ``deribit_paper/client/rest.py`` (Deribit-RPC).
Public endpoints only — no auth, no order submission. Returns raw
dicts; the caller is responsible for parsing.

Reference: https://binance-docs.github.io/apidocs/futures/en/#compressed-aggregate-trades-list
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Binance Futures REST defaults
_DEFAULT_BASE_URL = "https://fapi.binance.com"
_DEFAULT_TIMEOUT = 10.0
_AGG_TRADES_PATH = "/fapi/v1/aggTrades"
# Per Binance docs, max page size = 1000 trades.
_MAX_LIMIT = 1000
# Max attempts on transient 429/5xx before giving up the cycle.
_MAX_RETRIES = 4
# Backoff base (exponential)
_RETRY_BASE_S = 1.0
_MAX_BACKOFF_S = 30.0


class BinanceRESTError(RuntimeError):
    """Raised for non-retryable Binance REST errors (4xx other than 429)."""


class BinanceRESTClient:
    """Async HTTP client for Binance Futures public REST endpoints.

    Wraps ``httpx.AsyncClient`` with retry/backoff for transient errors
    (429, 5xx). Honours the ``Retry-After`` header when present (Binance
    sets it on rate-limit errors).

    The client is single-instance / shared across the engine. Construct
    once, use the same instance for every poll.
    """

    def __init__(
        self,
        *,
        base_url: str = _DEFAULT_BASE_URL,
        timeout_s: float = _DEFAULT_TIMEOUT,
        max_retries: int = _MAX_RETRIES,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        self._client: httpx.AsyncClient | None = None

    async def __aenter__(self) -> "BinanceRESTClient":
        await self.connect()
        return self

    async def __aexit__(self, *_exc_info: Any) -> None:
        await self.close()

    async def connect(self) -> None:
        """Open the underlying HTTP client. Idempotent."""
        if self._client is None:
            self._client = httpx.AsyncClient(
                base_url=self._base_url,
                timeout=self._timeout_s,
                http2=True,
                headers={"User-Agent": "deribit_paper/binance_rest_client"},
            )

    async def close(self) -> None:
        """Close the underlying HTTP client."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def fetch_agg_trades(
        self,
        symbol: str,
        *,
        from_id: int | None = None,
        limit: int = _MAX_LIMIT,
    ) -> list[dict]:
        """Fetch a page of aggregated trades.

        ``from_id`` lets the caller resume from the last-seen aggTradeId
        (set ``from_id = last_seen + 1``). Without ``from_id`` the
        endpoint returns the most recent ``limit`` trades.

        Returns the raw response array (each entry is a dict with
        keys ``a, p, q, f, l, T, m``). Empty list on no new trades.
        """
        if self._client is None:
            raise RuntimeError(
                "BinanceRESTClient not connected — call connect() or use "
                "as an async context manager",
            )
        if not (0 < limit <= _MAX_LIMIT):
            raise ValueError(
                f"limit must be in (0, {_MAX_LIMIT}], got {limit}",
            )

        params: dict[str, Any] = {"symbol": symbol, "limit": int(limit)}
        if from_id is not None:
            params["fromId"] = int(from_id)

        backoff = _RETRY_BASE_S
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = await self._client.get(_AGG_TRADES_PATH, params=params)
            except httpx.HTTPError as exc:
                last_exc = exc
                logger.warning(
                    "BinanceRESTClient: HTTP error on attempt %d/%d: %s — "
                    "retrying in %.1fs",
                    attempt + 1, self._max_retries, exc, backoff,
                )
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2.0, _MAX_BACKOFF_S)
                continue

            if resp.status_code == 200:
                data = resp.json()
                if not isinstance(data, list):
                    raise BinanceRESTError(
                        f"expected list response, got {type(data).__name__}",
                    )
                return data

            if resp.status_code == 429 or resp.status_code >= 500:
                # Honour Retry-After when set (Binance uses it for rate
                # limits + IP bans). Otherwise exponential backoff.
                retry_after = resp.headers.get("Retry-After")
                if retry_after:
                    try:
                        wait = float(retry_after)
                    except ValueError:
                        wait = backoff
                else:
                    wait = backoff
                logger.warning(
                    "BinanceRESTClient: HTTP %d on attempt %d/%d — "
                    "retrying in %.1fs (Retry-After=%s)",
                    resp.status_code, attempt + 1, self._max_retries,
                    wait, retry_after,
                )
                await asyncio.sleep(min(wait, _MAX_BACKOFF_S))
                backoff = min(backoff * 2.0, _MAX_BACKOFF_S)
                continue

            # Non-retryable 4xx
            raise BinanceRESTError(
                f"Binance REST returned HTTP {resp.status_code}: {resp.text[:200]}"
            )

        raise BinanceRESTError(
            f"BinanceRESTClient: exhausted {self._max_retries} retries; "
            f"last exception: {last_exc}"
        )


__all__ = ["BinanceRESTClient", "BinanceRESTError"]
