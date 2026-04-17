"""Tests for the Deribit async client (auth + REST + errors).

These tests use the ``respx`` library or ``httpx`` transports to mock
Deribit HTTP responses so they run without a real network connection.
WebSocket tests that need mocked socket behaviour are in
``test_ws.py`` (not yet written — Phase 1 only covers REST + auth).
"""

from __future__ import annotations

import asyncio
import json
import time

import httpx
import pytest

from deribit_paper.client.auth import DeribitAuth
from deribit_paper.client.errors import (
    DeribitAuthError,
    DeribitOrderRejected,
    DeribitRateLimitError,
    DeribitRequestError,
)
from deribit_paper.client.rest import DeribitRESTClient
from deribit_paper.risk.dry_run import DryRunState, set_dry_run


@pytest.fixture(autouse=True)
def _dry_run_cleanup():
    DryRunState().reset()
    yield
    DryRunState().reset()


class _MockTransport(httpx.AsyncBaseTransport):
    """Minimal mock transport that returns canned responses per path suffix.

    Matches on ``request.url.path.endswith(key)`` so tests can use short
    relative paths like ``/public/auth`` regardless of the base URL prefix.
    """

    def __init__(self, responses: dict[str, dict]):
        self.responses = responses
        self.call_log: list[dict] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.call_log.append({
            "method": request.method,
            "path": path,
            "query": dict(request.url.params),
            "headers": dict(request.headers),
        })
        canned = None
        for suffix, response in self.responses.items():
            if path.endswith(suffix):
                canned = response
                break
        if canned is None:
            return httpx.Response(404, json={"error": {"code": 0, "message": f"no mock for {path}"}})
        status = canned.get("status", 200)
        body = canned.get("body", {})
        return httpx.Response(status, json=body)


# ── Auth tests ────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_auth_rejects_empty_credentials():
    with pytest.raises(DeribitAuthError, match="non-empty"):
        DeribitAuth("", "", "https://test.deribit.com/api/v2")


@pytest.mark.asyncio
async def test_auth_success_caches_token():
    responses = {
        "/public/auth": {
            "body": {
                "result": {
                    "access_token": "tok_abc",
                    "refresh_token": "ref_xyz",
                    "expires_in": 900,
                },
            },
        },
    }
    transport = _MockTransport(responses)
    http = httpx.AsyncClient(
        base_url="https://test.deribit.com/api/v2",
        transport=transport,
    )
    auth = DeribitAuth("cid", "csecret", "https://test.deribit.com/api/v2", http_client=http)

    token1 = await auth.get_token()
    assert token1 == "tok_abc"
    assert len(transport.call_log) == 1

    # Second call should reuse the cached token
    token2 = await auth.get_token()
    assert token2 == "tok_abc"
    assert len(transport.call_log) == 1

    await http.aclose()


@pytest.mark.asyncio
async def test_auth_refreshes_near_expiry():
    responses = {
        "/public/auth": {
            "body": {
                "result": {
                    "access_token": "tok_short",
                    "refresh_token": "",
                    "expires_in": 30,  # 30s - below the 60s refresh margin
                },
            },
        },
    }
    transport = _MockTransport(responses)
    http = httpx.AsyncClient(
        base_url="https://test.deribit.com/api/v2",
        transport=transport,
    )
    auth = DeribitAuth("cid", "csecret", "https://test.deribit.com/api/v2", http_client=http)

    await auth.get_token()
    # Needs refresh immediately because expires_in (30s) < margin (60s)
    await auth.get_token()
    assert len(transport.call_log) == 2

    await http.aclose()


@pytest.mark.asyncio
async def test_auth_error_response_raises():
    responses = {
        "/public/auth": {
            "body": {
                "error": {"code": 13_004, "message": "invalid_credentials"},
            },
        },
    }
    transport = _MockTransport(responses)
    http = httpx.AsyncClient(
        base_url="https://test.deribit.com/api/v2",
        transport=transport,
    )
    auth = DeribitAuth("cid", "csecret", "https://test.deribit.com/api/v2", http_client=http)

    with pytest.raises(DeribitAuthError, match="invalid_credentials"):
        await auth.get_token()

    await http.aclose()


# ── REST client tests ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_rest_get_instruments_parses_result():
    responses = {
        "/public/auth": {
            "body": {"result": {"access_token": "tok", "refresh_token": "", "expires_in": 900}},
        },
        "/public/get_instruments": {
            "body": {
                "result": [
                    {"instrument_name": "BTC-28NOV25-60000-C", "kind": "option"},
                    {"instrument_name": "BTC-28NOV25-60000-P", "kind": "option"},
                ],
            },
        },
    }
    transport = _MockTransport(responses)
    http = httpx.AsyncClient(
        base_url="https://test.deribit.com/api/v2",
        transport=transport,
    )
    auth = DeribitAuth("cid", "csecret", "https://test.deribit.com/api/v2", http_client=http)
    rest = DeribitRESTClient(auth, "https://test.deribit.com/api/v2")
    rest._http = http  # inject the mocked client

    instruments = await rest.get_instruments(currency="BTC", kind="option")
    assert len(instruments) == 2
    assert instruments[0]["instrument_name"].startswith("BTC-")

    await http.aclose()


@pytest.mark.asyncio
async def test_rest_rate_limit_retries():
    call_count = {"auth": 0, "endpoint": 0}

    class _RetryTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            if request.url.path.endswith("/public/auth"):
                call_count["auth"] += 1
                return httpx.Response(200, json={
                    "result": {"access_token": "tok", "refresh_token": "", "expires_in": 900},
                })
            call_count["endpoint"] += 1
            # First 2 calls return rate-limit; third succeeds
            if call_count["endpoint"] < 3:
                return httpx.Response(200, json={
                    "error": {"code": 10028, "message": "too_many_requests"},
                })
            return httpx.Response(200, json={"result": {"index_price": 50000.0}})

    transport = _RetryTransport()
    http = httpx.AsyncClient(
        base_url="https://test.deribit.com/api/v2",
        transport=transport,
    )
    auth = DeribitAuth("cid", "csecret", "https://test.deribit.com/api/v2", http_client=http)
    rest = DeribitRESTClient(
        auth, "https://test.deribit.com/api/v2",
        max_retries=5, retry_base_delay=0.01,
    )
    rest._http = http

    result = await rest.get_index_price()
    assert result["index_price"] == 50000.0
    assert call_count["endpoint"] == 3

    await http.aclose()


@pytest.mark.asyncio
async def test_rest_rate_limit_exhausted_raises():
    class _AlwaysRateLimit(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            if request.url.path.endswith("/public/auth"):
                return httpx.Response(200, json={
                    "result": {"access_token": "tok", "refresh_token": "", "expires_in": 900},
                })
            return httpx.Response(200, json={
                "error": {"code": 10028, "message": "too_many_requests"},
            })

    http = httpx.AsyncClient(
        base_url="https://test.deribit.com/api/v2",
        transport=_AlwaysRateLimit(),
    )
    auth = DeribitAuth("cid", "csecret", "https://test.deribit.com/api/v2", http_client=http)
    rest = DeribitRESTClient(
        auth, "https://test.deribit.com/api/v2",
        max_retries=2, retry_base_delay=0.01,
    )
    rest._http = http

    with pytest.raises(DeribitRateLimitError):
        await rest.get_index_price()

    await http.aclose()


@pytest.mark.asyncio
async def test_rest_place_order_dry_run_returns_stub():
    # No HTTP at all — dry-run should short-circuit before any request
    rest = DeribitRESTClient.__new__(DeribitRESTClient)
    rest._closed = False
    # dry_run defaults True

    result = await rest.place_order(
        instrument_name="BTC-PERPETUAL",
        side="buy",
        amount=0.1,
        type="market",
    )
    assert result["_stub"] is True
    assert result["order"]["order_state"] == "filled"
    assert result["order"]["filled_amount"] == 0.1
    assert result["order"]["direction"] == "buy"


@pytest.mark.asyncio
async def test_rest_place_order_live_hits_api():
    """When DRY_RUN is off, place_order must actually hit the API."""
    set_dry_run(False, confirmation_token="test_token")

    calls: list[str] = []

    class _OrderTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            calls.append(request.url.path)
            if request.url.path.endswith("/public/auth"):
                return httpx.Response(200, json={
                    "result": {"access_token": "tok", "refresh_token": "", "expires_in": 900},
                })
            return httpx.Response(200, json={
                "result": {
                    "order": {"order_id": "real_123", "order_state": "open"},
                    "trades": [],
                },
            })

    http = httpx.AsyncClient(
        base_url="https://test.deribit.com/api/v2",
        transport=_OrderTransport(),
    )
    auth = DeribitAuth("cid", "csecret", "https://test.deribit.com/api/v2", http_client=http)
    rest = DeribitRESTClient(auth, "https://test.deribit.com/api/v2")
    rest._http = http

    try:
        result = await rest.place_order(
            instrument_name="BTC-PERPETUAL",
            side="buy",
            amount=0.1,
            type="limit",
            price=50000.0,
        )
        assert result["order"]["order_id"] == "real_123"
        assert any(c.endswith("/private/buy") for c in calls)
    finally:
        DryRunState().reset()
        await http.aclose()


@pytest.mark.asyncio
async def test_rest_order_rejection_raises():
    set_dry_run(False, confirmation_token="test_token")

    class _RejectTransport(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            if request.url.path.endswith("/public/auth"):
                return httpx.Response(200, json={
                    "result": {"access_token": "tok", "refresh_token": "", "expires_in": 900},
                })
            return httpx.Response(200, json={
                "error": {"code": 10002, "message": "insufficient_funds"},
            })

    http = httpx.AsyncClient(
        base_url="https://test.deribit.com/api/v2",
        transport=_RejectTransport(),
    )
    auth = DeribitAuth("cid", "csecret", "https://test.deribit.com/api/v2", http_client=http)
    rest = DeribitRESTClient(auth, "https://test.deribit.com/api/v2")
    rest._http = http

    try:
        with pytest.raises(DeribitOrderRejected) as exc_info:
            await rest.place_order(
                instrument_name="BTC-PERPETUAL",
                side="buy",
                amount=100.0,  # absurd amount
                type="market",
            )
        assert exc_info.value.retryable is False
        assert "insufficient_funds" in str(exc_info.value)
    finally:
        DryRunState().reset()
        await http.aclose()


@pytest.mark.asyncio
async def test_rest_cancel_order_dry_run():
    rest = DeribitRESTClient.__new__(DeribitRESTClient)
    rest._closed = False
    result = await rest.cancel_order("some_id")
    assert result["_stub"] is True
    assert result["order_state"] == "cancelled"
