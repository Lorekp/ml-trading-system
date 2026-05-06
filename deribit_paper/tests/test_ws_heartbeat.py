"""Tests for H11 Deribit set_heartbeat + test_request handling.

Why this exists
---------------

Deribit's ``public/set_heartbeat`` is the canonical liveness primitive
for WS connections. After enabling it with interval=N seconds, the
server pushes a ``test_request`` notification every N seconds; the
client MUST reply with ``public/test`` within the timeout or the
server kills the connection. This is belt-and-suspenders to our
data-stale watchdog (which detects client-side that the server has
stopped pushing data); set_heartbeat lets the SERVER detect that we
are wedged.

Tests pin the H11 contract:

1. ``test_request`` notification triggers a ``public/test`` response.
2. Plain ``heartbeat`` (no ``test_request``) is a no-op.
3. The response is sent without awaiting a reply (fire-and-forget).
4. The counter ``_heartbeat_test_responses_sent`` is bumped per reply.
"""
from __future__ import annotations

import asyncio
import json

import pytest

from deribit_paper.client.ws import DeribitWebSocketClient


class _CapturingWS:
    """Minimal duck-typed WS that records every send() call."""
    def __init__(self):
        self.close_code = None
        self.sent: list[str] = []

    async def send(self, payload: str) -> None:
        self.sent.append(payload)

    async def close(self, code: int = 1000, reason: str = "") -> None:
        self.close_code = code


@pytest.mark.asyncio
async def test_test_request_triggers_public_test_response():
    client = DeribitWebSocketClient(ws_url="wss://test.deribit.com/ws/api/v2")
    fake = _CapturingWS()
    client._ws = fake  # noqa: SLF001

    msg = b'{"jsonrpc":"2.0","method":"heartbeat","params":{"type":"test_request"}}'
    await client._handle_message(msg)

    assert len(fake.sent) == 1
    payload = json.loads(fake.sent[0])
    assert payload["method"] == "public/test"
    assert payload["jsonrpc"] == "2.0"
    assert "id" in payload
    assert payload["params"] == {}
    assert client._heartbeat_test_responses_sent == 1


@pytest.mark.asyncio
async def test_plain_heartbeat_does_not_send_response():
    """``type=heartbeat`` is informational; no response required."""
    client = DeribitWebSocketClient(ws_url="wss://test.deribit.com/ws/api/v2")
    fake = _CapturingWS()
    client._ws = fake

    msg = b'{"jsonrpc":"2.0","method":"heartbeat","params":{"type":"heartbeat"}}'
    await client._handle_message(msg)

    assert fake.sent == []
    assert client._heartbeat_test_responses_sent == 0


@pytest.mark.asyncio
async def test_multiple_test_requests_each_get_response():
    client = DeribitWebSocketClient(ws_url="wss://test.deribit.com/ws/api/v2")
    fake = _CapturingWS()
    client._ws = fake

    msg = b'{"jsonrpc":"2.0","method":"heartbeat","params":{"type":"test_request"}}'
    for _ in range(5):
        await client._handle_message(msg)

    assert len(fake.sent) == 5
    assert client._heartbeat_test_responses_sent == 5
    # Each response must carry a unique id
    ids = [json.loads(p)["id"] for p in fake.sent]
    assert len(set(ids)) == 5


@pytest.mark.asyncio
async def test_test_request_with_no_ws_does_not_raise():
    """Defensive: if the WS dropped between the test_request arriving
    and our send() call, we just log the failure — must not raise out
    of _handle_message."""
    client = DeribitWebSocketClient(ws_url="wss://test.deribit.com/ws/api/v2")
    # _ws is None by default
    msg = b'{"jsonrpc":"2.0","method":"heartbeat","params":{"type":"test_request"}}'
    # Must not raise
    await client._handle_message(msg)


def test_heartbeat_disabled_when_interval_zero():
    """``heartbeat_interval_seconds=0`` disables the post-connect
    set_heartbeat call. The connection still works; just no
    server-side liveness probing."""
    client = DeribitWebSocketClient(
        ws_url="wss://test.deribit.com/ws/api/v2",
        heartbeat_interval_seconds=0,
    )
    assert client._heartbeat_interval_seconds == 0
