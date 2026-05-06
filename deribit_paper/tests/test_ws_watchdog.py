"""Unit tests for the WS data-stale watchdog.

Why this exists
---------------

2026-05-06: the live engine kill-switched 6 consecutive 15-min bars
because the Deribit mainnet WS entered a "zombie" state — TCP-level
ping/pong kept succeeding (so the websockets-library keepalive saw
no failure), but no subscription messages were arriving. The
underlying connection only died after ~2h, at which point the
existing reconnect path fired.

The watchdog added in ``deribit_paper/client/ws.py`` cuts that
detection time from hours to ``data_stale_seconds + check_interval``
by closing the WS proactively when no message has arrived for too
long. These tests pin the watchdog's contract:

1. The watchdog forces a close once the staleness gap exceeds the
   configured threshold.
2. Each received message refreshes the timestamp (no stuck close
   while data is flowing).
3. ``data_stale_seconds=0`` disables the watchdog entirely.
4. Watchdog is cancelled on ``close()``.
"""
from __future__ import annotations

import asyncio

import pytest

from deribit_paper.client.ws import DeribitWebSocketClient


class _FakeWS:
    """Minimal duck-typed WS with the attributes the watchdog touches."""

    def __init__(self):
        self.close_code = None
        self.closed_with_code: int | None = None
        self.closed_with_reason: str | None = None
        self._close_event = asyncio.Event()

    async def close(self, code: int = 1000, reason: str = "") -> None:
        # Only record the FIRST close so we can distinguish a watchdog
        # close from a subsequent ``client.close()`` cleanup.
        if self.closed_with_code is None:
            self.closed_with_code = code
            self.closed_with_reason = reason
            self.close_code = code
            self._close_event.set()


@pytest.mark.asyncio
async def test_watchdog_forces_close_when_data_stale():
    """Watchdog closes the WS once the gap exceeds threshold."""
    client = DeribitWebSocketClient(
        ws_url="wss://test.deribit.com/ws/api/v2",
        data_stale_seconds=0.10,  # 100 ms threshold
    )
    fake = _FakeWS()
    client._ws = fake  # noqa: SLF001 - test wiring
    # Force the staleness clock far into the past so the watchdog's
    # first tick will cross the threshold.
    client._last_message_ts -= 10.0  # noqa: SLF001
    client._start_watchdog_task()  # noqa: SLF001

    # Watchdog check_interval = max(15s, threshold/3) = 15s, so the
    # first iteration would normally take 15s to fire. The watchdog
    # uses ``shutdown.wait`` with timeout, which gives us a chance to
    # observe the close happens within a bounded wait — but only if
    # the floor allows. We adjust the floor by overriding it through
    # the loop indirectly: the watchdog's wait is bounded by the
    # check_interval. For a fast unit test we re-arm with a *much*
    # smaller threshold AND check that the close DID happen by the
    # time the watchdog task completes.
    try:
        await asyncio.wait_for(fake._close_event.wait(), timeout=20.0)
    finally:
        client._shutdown.set()  # noqa: SLF001
        await client.close()

    assert fake.closed_with_code == 1011
    assert fake.closed_with_reason == "data-stale watchdog"


@pytest.mark.asyncio
async def test_watchdog_disabled_when_threshold_is_zero():
    """data_stale_seconds=0 means the watchdog never spawns."""
    client = DeribitWebSocketClient(
        ws_url="wss://test.deribit.com/ws/api/v2",
        data_stale_seconds=0,
    )
    fake = _FakeWS()
    client._ws = fake  # noqa: SLF001
    client._start_watchdog_task()  # noqa: SLF001

    assert client._watchdog_task is None  # noqa: SLF001
    # No close happens even after a long monkey-patched silence.
    client._last_message_ts -= 10_000  # noqa: SLF001
    await asyncio.sleep(0.05)
    assert fake.closed_with_code is None
    await client.close()


@pytest.mark.asyncio
async def test_handle_message_refreshes_timestamp():
    """Every received message refreshes ``_last_message_ts``."""
    client = DeribitWebSocketClient(
        ws_url="wss://test.deribit.com/ws/api/v2",
        data_stale_seconds=0,  # we don't need the watchdog for this
    )
    # Backdate the timestamp.
    client._last_message_ts -= 60.0  # noqa: SLF001
    before = client._last_message_ts  # noqa: SLF001

    # Push a malformed frame (bad JSON) — the timestamp should still
    # refresh because something arrived through the socket.
    await client._handle_message(b"not-json-at-all")  # noqa: SLF001
    after_bad = client._last_message_ts  # noqa: SLF001
    assert after_bad > before

    # Push a well-formed subscription frame.
    await client._handle_message(  # noqa: SLF001
        b'{"jsonrpc":"2.0","method":"subscription","params":{"channel":"x","data":1}}'
    )
    after_good = client._last_message_ts  # noqa: SLF001
    assert after_good >= after_bad

    await client.close()


@pytest.mark.asyncio
async def test_close_cancels_watchdog_task():
    """``close()`` cancels the watchdog so the task doesn't leak."""
    client = DeribitWebSocketClient(
        ws_url="wss://test.deribit.com/ws/api/v2",
        data_stale_seconds=10.0,
    )
    client._ws = _FakeWS()  # noqa: SLF001
    client._start_watchdog_task()  # noqa: SLF001
    task = client._watchdog_task  # noqa: SLF001
    assert task is not None
    assert not task.done()

    await client.close()
    assert client._watchdog_task is None  # noqa: SLF001
    assert task.done()
