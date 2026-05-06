"""Tests for H17 Deribit rate-limit ``data.wait`` hint honoring.

Why this exists
---------------

Deribit's 10028 ``too_many_requests`` error sometimes carries a
``data.wait`` field with milliseconds the client should wait. Pre-H17
the engine ignored this and used generic exponential backoff
(``retry_base_delay=0.5s × 2^attempt``) — too short for matching-
engine bursts during volatile minutes. H17 honors the server hint
when present and longer than the computed backoff.

Tests pin the H17 contract:

1. Hint absent → fall back to computed backoff (existing behaviour).
2. Hint present and longer → use hint.
3. Hint present and shorter → use computed backoff (defensive
   floor; we never sleep LESS than our own policy says).
4. Hint with absurd value (e.g., 60_000_000) is capped at 30s.
"""
from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from deribit_paper.client.auth import DeribitAuth
from deribit_paper.client.rest import DeribitRESTClient


def _build_rest(transport: httpx.AsyncBaseTransport) -> DeribitRESTClient:
    http = httpx.AsyncClient(
        base_url="https://test.deribit.com/api/v2",
        transport=transport,
    )
    auth = DeribitAuth(
        "cid", "csecret", "https://test.deribit.com/api/v2", http_client=http,
    )
    rest = DeribitRESTClient(
        auth, "https://test.deribit.com/api/v2",
        max_retries=3, retry_base_delay=0.01,
    )
    rest._http = http  # noqa: SLF001
    return rest


@pytest.mark.asyncio
async def test_no_hint_uses_computed_backoff():
    """Existing behaviour: no ``data.wait`` field → computed backoff."""
    call_count = {"auth": 0, "endpoint": 0}

    class T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            if request.url.path.endswith("/public/auth"):
                call_count["auth"] += 1
                return httpx.Response(200, json={
                    "result": {"access_token": "t", "refresh_token": "", "expires_in": 900},
                })
            call_count["endpoint"] += 1
            if call_count["endpoint"] < 2:
                return httpx.Response(200, json={
                    "error": {"code": 10028, "message": "too_many_requests"},
                })
            return httpx.Response(200, json={"result": {"index_price": 50000.0}})

    rest = _build_rest(T())
    t0 = time.monotonic()
    await rest.get_index_price()
    elapsed = time.monotonic() - t0
    # First retry uses base 0.01s + jitter — should be well under 1s
    assert elapsed < 1.0


@pytest.mark.asyncio
async def test_hint_longer_than_computed_is_honored():
    """``data.wait=200`` (200ms) > computed backoff (~10ms)."""
    call_count = {"auth": 0, "endpoint": 0}

    class T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            if request.url.path.endswith("/public/auth"):
                call_count["auth"] += 1
                return httpx.Response(200, json={
                    "result": {"access_token": "t", "refresh_token": "", "expires_in": 900},
                })
            call_count["endpoint"] += 1
            if call_count["endpoint"] < 2:
                return httpx.Response(200, json={
                    "error": {
                        "code": 10028,
                        "message": "too_many_requests",
                        "data": {"wait": 200},
                    },
                })
            return httpx.Response(200, json={"result": {"index_price": 50000.0}})

    rest = _build_rest(T())
    t0 = time.monotonic()
    await rest.get_index_price()
    elapsed = time.monotonic() - t0
    # 200ms hint must have been applied (allow some slack for scheduling)
    assert elapsed >= 0.18


@pytest.mark.asyncio
async def test_hint_caps_at_30_seconds():
    """A pathological hint (e.g., 5 minutes) must be capped at 30s.
    We don't actually wait 30s in the test — we verify by inspecting
    the sleep call. Use unittest.mock.patch on asyncio.sleep."""
    from unittest.mock import patch
    call_count = {"auth": 0, "endpoint": 0}

    class T(httpx.AsyncBaseTransport):
        async def handle_async_request(self, request):
            if request.url.path.endswith("/public/auth"):
                call_count["auth"] += 1
                return httpx.Response(200, json={
                    "result": {"access_token": "t", "refresh_token": "", "expires_in": 900},
                })
            call_count["endpoint"] += 1
            if call_count["endpoint"] < 2:
                return httpx.Response(200, json={
                    "error": {
                        "code": 10028,
                        "message": "too_many_requests",
                        "data": {"wait": 5_000_000},  # 5,000s = absurd
                    },
                })
            return httpx.Response(200, json={"result": {"index_price": 50000.0}})

    rest = _build_rest(T())

    sleep_durations = []

    async def fake_sleep(seconds):
        sleep_durations.append(seconds)
        # Don't actually sleep

    with patch.object(asyncio, "sleep", fake_sleep):
        await rest.get_index_price()

    # The retry sleep was capped at 30s
    assert any(d == pytest.approx(30.0, abs=0.01) for d in sleep_durations), (
        f"expected a 30.0s capped sleep, got {sleep_durations}"
    )
