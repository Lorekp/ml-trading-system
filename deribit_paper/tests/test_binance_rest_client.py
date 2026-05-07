"""E (parity 2026-05-07): unit tests for BinanceRESTClient.

Verifies: pagination via fromId, retry on 429/5xx, Retry-After header
honoured, non-retryable 4xx raises BinanceRESTError.

Mocks ``httpx.AsyncClient`` so we don't hit the live Binance endpoint.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from deribit_paper.client.binance_rest import (
    BinanceRESTClient,
    BinanceRESTError,
)


def _resp(status: int, body: list[dict] | dict | str = "", headers: dict | None = None):
    """Build a mock httpx.Response."""
    m = MagicMock(spec=httpx.Response)
    m.status_code = status
    if isinstance(body, (list, dict)):
        m.json = MagicMock(return_value=body)
        m.text = ""
    else:
        m.json = MagicMock(side_effect=ValueError("not json"))
        m.text = body
    m.headers = headers or {}
    return m


# ── 1. Happy path: 200 OK with array ───────────────────────────────────────


@pytest.mark.asyncio
async def test_fetch_agg_trades_200_ok():
    client = BinanceRESTClient()
    payload = [
        {"a": 1, "p": "65000", "q": "0.1", "f": 1, "l": 1, "T": 1, "m": True},
        {"a": 2, "p": "65001", "q": "0.2", "f": 2, "l": 2, "T": 2, "m": False},
    ]
    fake = AsyncMock()
    fake.get = AsyncMock(return_value=_resp(200, payload))
    fake.aclose = AsyncMock()
    with patch("httpx.AsyncClient", return_value=fake):
        await client.connect()
        result = await client.fetch_agg_trades("BTCUSDT", from_id=1, limit=100)
    assert result == payload
    fake.get.assert_awaited_once()
    args, kwargs = fake.get.call_args
    assert args[0] == "/fapi/v1/aggTrades"
    assert kwargs["params"]["symbol"] == "BTCUSDT"
    assert kwargs["params"]["fromId"] == 1
    assert kwargs["params"]["limit"] == 100


# ── 2. Pagination: from_id correctly forwarded ─────────────────────────────


@pytest.mark.asyncio
async def test_pagination_from_id_increments():
    """Caller can paginate by passing the previous response's max ``a`` + 1.
    The client just forwards ``fromId``; the caller drives pagination.
    """
    client = BinanceRESTClient()
    fake = AsyncMock()
    page1 = [{"a": i, "q": "1", "m": False} for i in range(1, 11)]
    page2 = [{"a": i, "q": "1", "m": False} for i in range(11, 21)]
    fake.get = AsyncMock(side_effect=[_resp(200, page1), _resp(200, page2)])
    fake.aclose = AsyncMock()
    with patch("httpx.AsyncClient", return_value=fake):
        await client.connect()
        r1 = await client.fetch_agg_trades("BTCUSDT")
        last_id = max(t["a"] for t in r1)
        r2 = await client.fetch_agg_trades("BTCUSDT", from_id=last_id + 1)
    assert r1 == page1
    assert r2 == page2
    second_call_kwargs = fake.get.call_args_list[1].kwargs
    assert second_call_kwargs["params"]["fromId"] == 11


# ── 3. 429 Retry-After header honoured ─────────────────────────────────────


@pytest.mark.asyncio
async def test_429_then_200_retry_succeeds():
    """One 429 followed by 200 → returns the 200 body after waiting."""
    client = BinanceRESTClient(max_retries=3)
    fake = AsyncMock()
    fake.get = AsyncMock(side_effect=[
        _resp(429, "rate limit exceeded", headers={"Retry-After": "0"}),
        _resp(200, [{"a": 1, "q": "0.1", "m": False}]),
    ])
    fake.aclose = AsyncMock()
    with patch("httpx.AsyncClient", return_value=fake):
        await client.connect()
        result = await client.fetch_agg_trades("BTCUSDT")
    assert len(result) == 1
    assert fake.get.await_count == 2


# ── 4. 5xx retry sequence ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_5xx_retries_then_succeeds():
    """500 + 502 + 200 → succeeds on third try."""
    client = BinanceRESTClient(max_retries=4)
    fake = AsyncMock()
    fake.get = AsyncMock(side_effect=[
        _resp(500, "internal", headers={"Retry-After": "0"}),
        _resp(502, "bad gateway", headers={"Retry-After": "0"}),
        _resp(200, []),
    ])
    fake.aclose = AsyncMock()
    with patch("httpx.AsyncClient", return_value=fake):
        await client.connect()
        result = await client.fetch_agg_trades("BTCUSDT")
    assert result == []
    assert fake.get.await_count == 3


# ── 5. Non-retryable 4xx raises ────────────────────────────────────────────


@pytest.mark.asyncio
async def test_non_retryable_400_raises():
    client = BinanceRESTClient()
    fake = AsyncMock()
    fake.get = AsyncMock(return_value=_resp(400, "bad symbol"))
    fake.aclose = AsyncMock()
    with patch("httpx.AsyncClient", return_value=fake):
        await client.connect()
        with pytest.raises(BinanceRESTError, match="HTTP 400"):
            await client.fetch_agg_trades("INVALID")
    assert fake.get.await_count == 1


# ── 6. Exhausted retries raise BinanceRESTError ────────────────────────────


@pytest.mark.asyncio
async def test_exhausted_retries_raises():
    """All attempts return 503 → raises after max_retries."""
    client = BinanceRESTClient(max_retries=2)
    fake = AsyncMock()
    fake.get = AsyncMock(return_value=_resp(503, "unavailable", headers={"Retry-After": "0"}))
    fake.aclose = AsyncMock()
    with patch("httpx.AsyncClient", return_value=fake):
        await client.connect()
        with pytest.raises(BinanceRESTError):
            await client.fetch_agg_trades("BTCUSDT")
    assert fake.get.await_count == 2


# ── 7. Limit validation ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_invalid_limit_raises_value_error():
    client = BinanceRESTClient()
    fake = AsyncMock()
    fake.aclose = AsyncMock()
    with patch("httpx.AsyncClient", return_value=fake):
        await client.connect()
        with pytest.raises(ValueError, match="limit"):
            await client.fetch_agg_trades("BTCUSDT", limit=0)
        with pytest.raises(ValueError, match="limit"):
            await client.fetch_agg_trades("BTCUSDT", limit=10_000)


# ── 8. Connect/close lifecycle ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_connect_idempotent_close_safe():
    client = BinanceRESTClient()
    fake = AsyncMock()
    fake.aclose = AsyncMock()
    with patch("httpx.AsyncClient", return_value=fake):
        await client.connect()
        await client.connect()  # idempotent — must not raise
        await client.close()
        await client.close()  # safe to call twice


# ── 9. Async context manager ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_async_context_manager():
    fake = AsyncMock()
    fake.get = AsyncMock(return_value=_resp(200, []))
    fake.aclose = AsyncMock()
    with patch("httpx.AsyncClient", return_value=fake):
        async with BinanceRESTClient() as client:
            await client.fetch_agg_trades("BTCUSDT")
    fake.aclose.assert_awaited_once()
