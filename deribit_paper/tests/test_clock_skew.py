"""Tests for H12 clock-skew gate (2026-05-06).

Why this exists
---------------

The bar boundary uses ``BarClock`` which reads the local system clock.
If that clock drifts vs Deribit's server clock (e.g., a homeserver
without NTP sync), every signal fires at a slightly different offset
relative to where the V2 backtest sampled. A 30s skew on a 15min bar
is operationally meaningful — it shifts every signal by 30s.

The check uses ``rest.get_server_time_ms()`` which calls the public
unauthenticated ``/public/get_time`` endpoint. Failure modes:

- Network failure / timeout → log warning, continue (best-effort)
- Skew < 250ms → log INFO (acceptable)
- Skew 250ms <= x < 5000ms → log WARNING (NTP recommended)
- Skew >= 5000ms → raise RuntimeError (refuse to start)
"""
from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from deribit_paper.client.rest import DeribitRESTClient


@pytest.mark.asyncio
async def test_get_server_time_ms_returns_int():
    """Sanity: the helper returns an int from the JSON response."""
    rest = DeribitRESTClient.__new__(DeribitRESTClient)
    rest._request = AsyncMock(return_value=1_700_000_000_000)
    server_ms = await rest.get_server_time_ms()
    assert server_ms == 1_700_000_000_000


@pytest.mark.asyncio
async def test_get_server_time_ms_handles_string_response():
    """If Deribit ever returns the timestamp as a string, our int()
    cast must not raise (defensive against API drift)."""
    rest = DeribitRESTClient.__new__(DeribitRESTClient)
    rest._request = AsyncMock(return_value="1700000000000")
    server_ms = await rest.get_server_time_ms()
    assert server_ms == 1_700_000_000_000
