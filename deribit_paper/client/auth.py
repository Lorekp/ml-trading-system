"""OAuth2 client_credentials flow for Deribit.

Deribit exposes an OAuth2 ``client_credentials`` endpoint at
``/public/auth`` that returns a short-lived access token (default 900
seconds) plus a refresh token.  This module wraps the flow as an async
class with automatic token refresh before expiry.

Used by :class:`deribit_paper.client.rest.DeribitRESTClient` and
:class:`deribit_paper.client.ws.DeribitWebSocketClient` — both call
``DeribitAuth.get_token()`` before any authenticated request.  The
token is cached in-process and refreshed proactively with a small
safety margin.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import httpx

from deribit_paper.client.errors import DeribitAuthError

logger = logging.getLogger(__name__)


# Refresh token this many seconds BEFORE it officially expires.  Gives
# us a safety margin so a slow network round-trip doesn't result in a
# 401 mid-request.
_REFRESH_MARGIN_SEC = 60


@dataclass
class _TokenState:
    access_token: str
    refresh_token: str
    expires_at: float  # unix timestamp

    @property
    def seconds_remaining(self) -> float:
        return self.expires_at - time.time()

    @property
    def needs_refresh(self) -> bool:
        return self.seconds_remaining <= _REFRESH_MARGIN_SEC


class DeribitAuth:
    """OAuth2 client_credentials flow for Deribit.

    Thread-unsafe on purpose — this is async.  The class uses an
    ``asyncio.Lock`` to serialise refresh attempts so multiple tasks
    sharing the same auth instance don't trigger concurrent refreshes.
    """

    def __init__(
        self,
        client_id: str,
        client_secret: str,
        rest_base_url: str,
        http_client: httpx.AsyncClient | None = None,
    ):
        if not client_id or not client_secret:
            raise DeribitAuthError(
                "DeribitAuth requires non-empty client_id and client_secret "
                "(set DERIBIT_TESTNET_CLIENT_ID / _SECRET in .env)",
            )
        self.client_id = client_id
        self.client_secret = client_secret
        self.rest_base_url = rest_base_url.rstrip("/")
        self._http = http_client  # if None, a dedicated client is created in get_token
        self._owns_http = http_client is None
        self._token: _TokenState | None = None
        self._lock = asyncio.Lock()

    async def _ensure_http(self) -> httpx.AsyncClient:
        if self._http is None:
            self._http = httpx.AsyncClient(
                base_url=self.rest_base_url,
                timeout=httpx.Timeout(15.0, connect=10.0),
                http2=True,
            )
        return self._http

    async def close(self) -> None:
        """Close the internal HTTP client if we created it."""
        if self._owns_http and self._http is not None:
            await self._http.aclose()
            self._http = None

    async def get_token(self) -> str:
        """Return a valid access token, refreshing proactively if needed.

        Thread-safe within the asyncio event loop via the internal lock.
        """
        async with self._lock:
            if self._token is None or self._token.needs_refresh:
                await self._authenticate()
            assert self._token is not None
            return self._token.access_token

    async def _authenticate(self) -> None:
        """Hit the ``/public/auth`` endpoint and store the new token."""
        http = await self._ensure_http()
        params = {
            "grant_type": "client_credentials",
            "client_id": self.client_id,
            "client_secret": self.client_secret,
        }
        try:
            resp = await http.get("/public/auth", params=params)
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise DeribitAuthError(
                f"OAuth2 client_credentials request failed: {exc}",
            ) from exc

        body = resp.json()
        if "error" in body:
            err = body["error"]
            raise DeribitAuthError(
                f"Deribit auth rejected: {err.get('message', 'unknown')}",
                code=err.get("code"),
                data=err,
            )

        result = body.get("result", {})
        access_token = result.get("access_token")
        refresh_token = result.get("refresh_token")
        expires_in = result.get("expires_in")
        if not access_token or not expires_in:
            raise DeribitAuthError(
                f"Deribit auth response missing fields: {result}",
                data=result,
            )

        self._token = _TokenState(
            access_token=access_token,
            refresh_token=refresh_token or "",
            expires_at=time.time() + float(expires_in),
        )
        logger.info(
            "Deribit auth OK — token expires in %ds", int(expires_in),
        )

    async def __aenter__(self) -> "DeribitAuth":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()
