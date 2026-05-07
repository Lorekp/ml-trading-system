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

# Binance REST defaults — Futures is the legacy default (built for VPIN).
# F3 (2026-05-07): added Spot constants for dollar-bar reconstruction
# from Binance Vision aggTrades. The offline pipeline trains on Binance
# spot dollar bars (vol_model/features/engineer.py:_build_bars_sequential
# loads from data/binance-aggTrades-parquet, which is Spot). Live must
# pull from the SAME source for byte-identity.
_DEFAULT_BASE_URL = "https://fapi.binance.com"
_DEFAULT_TIMEOUT = 10.0
_AGG_TRADES_PATH = "/fapi/v1/aggTrades"          # Futures
_SPOT_BASE_URL = "https://api.binance.com"
_SPOT_AGG_TRADES_PATH = "/api/v3/aggTrades"      # Spot
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
        agg_trades_path: str = _AGG_TRADES_PATH,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._timeout_s = timeout_s
        self._max_retries = max_retries
        # F3 (2026-05-07): path is configurable so the same client class
        # serves both Binance Futures (``/fapi/v1/aggTrades``) and Spot
        # (``/api/v3/aggTrades``). The two endpoints share request schema
        # (symbol + fromId + limit) and response schema (a, p, q, T, m).
        self._agg_trades_path = agg_trades_path
        self._client: httpx.AsyncClient | None = None

    @classmethod
    def for_spot(
        cls,
        *,
        timeout_s: float = _DEFAULT_TIMEOUT,
        max_retries: int = _MAX_RETRIES,
    ) -> "BinanceRESTClient":
        """Construct a client pointed at Binance Spot (api.binance.com).

        Used by the dollar-bar reconstruction (F3 — parity-cycle round 3)
        which mirrors the offline pipeline's spot-aggTrades source so the
        24 dollar-bar-derived features can byte-identity-match the model's
        training distribution.
        """
        return cls(
            base_url=_SPOT_BASE_URL,
            timeout_s=timeout_s,
            max_retries=max_retries,
            agg_trades_path=_SPOT_AGG_TRADES_PATH,
        )

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
        start_time_ms: int | None = None,
        end_time_ms: int | None = None,
    ) -> list[dict]:
        """Fetch a page of aggregated trades.

        ``from_id`` lets the caller resume from the last-seen aggTradeId
        (set ``from_id = last_seen + 1``). Without ``from_id`` the
        endpoint returns the most recent ``limit`` trades.

        ``start_time_ms`` / ``end_time_ms`` (Round-4 F6/F8 — 2026-05-08):
        time-range query for the warm-start sidecar generator. Binance
        spec: the time range cannot exceed 1 hour. Mutually exclusive
        with ``from_id``. When set, Binance returns up to ``limit``
        trades whose ``T`` falls in ``[start_time_ms, end_time_ms]``.

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
        if from_id is not None and (
            start_time_ms is not None or end_time_ms is not None
        ):
            raise ValueError(
                "fetch_agg_trades: from_id is mutually exclusive with "
                "start_time_ms / end_time_ms (Binance spec)"
            )

        params: dict[str, Any] = {"symbol": symbol, "limit": int(limit)}
        if from_id is not None:
            params["fromId"] = int(from_id)
        if start_time_ms is not None:
            params["startTime"] = int(start_time_ms)
        if end_time_ms is not None:
            params["endTime"] = int(end_time_ms)

        backoff = _RETRY_BASE_S
        last_exc: Exception | None = None
        for attempt in range(self._max_retries):
            try:
                resp = await self._client.get(self._agg_trades_path, params=params)
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
