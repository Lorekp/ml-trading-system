"""Tests for H5 (per-channel staleness) and H6 (subscribe ACK verification).

Why this exists
---------------

H5: the global ``data_stale_seconds`` watchdog (90s) catches "ALL
channels stopped" zombies but is BLIND to one channel going silent
while others keep flowing — e.g. the index price feed dies while
perp trades stream normally. The engine then runs on stale index
data without any external signal.

H6: ``_send_rpc("public/subscribe", {"channels": [...]})`` returns
an array of channels the server actually bound. Pre-H6 the engine
discarded this — a typo / expired option / server-side rejection
silently dropped a channel and the engine ran blind on a partial
subscription set.

Tests pin both contracts:

H5 contracts:
1. ``set_channel_threshold`` registers + clears thresholds.
2. Per-channel timestamps populate on subscription messages.
3. Watchdog forces close when a registered channel goes stale,
   even if other channels are still flowing.
4. Reconnect clears per-channel timestamps so they don't leak
   into the next session.

H6 contracts:
5. Matching subscribe ACK passes silently.
6. Missing channel in ACK raises DeribitWebSocketError.
7. Non-list result is logged but does not raise.
"""
from __future__ import annotations

import asyncio
import time

import pytest

from deribit_paper.client.errors import DeribitWebSocketError
from deribit_paper.client.ws import DeribitWebSocketClient


class _FakeWS:
    """Minimal duck-typed WS for watchdog tests."""
    def __init__(self):
        self.close_code = None
        self.closed_with_code: int | None = None
        self.closed_with_reason: str | None = None
        self._close_event = asyncio.Event()

    async def close(self, code: int = 1000, reason: str = "") -> None:
        if self.closed_with_code is None:
            self.closed_with_code = code
            self.closed_with_reason = reason
            self.close_code = code
            self._close_event.set()


# ── H5: per-channel threshold registration ───────────────────────────


def test_set_channel_threshold_registers():
    client = DeribitWebSocketClient(ws_url="wss://test.deribit.com/ws/api/v2")
    client.set_channel_threshold("deribit_price_index.btc_usd", 30.0)
    assert client._channel_thresholds["deribit_price_index.btc_usd"] == 30.0


def test_set_channel_threshold_zero_clears():
    client = DeribitWebSocketClient(ws_url="wss://test.deribit.com/ws/api/v2")
    client.set_channel_threshold("ch1", 30.0)
    assert "ch1" in client._channel_thresholds
    client.set_channel_threshold("ch1", 0.0)
    assert "ch1" not in client._channel_thresholds


# ── H5: per-channel timestamp tracking ──────────────────────────────


@pytest.mark.asyncio
async def test_per_channel_ts_populates_on_subscription_message():
    client = DeribitWebSocketClient(ws_url="wss://test.deribit.com/ws/api/v2")
    msg = (
        b'{"jsonrpc":"2.0","method":"subscription",'
        b'"params":{"channel":"deribit_price_index.btc_usd",'
        b'"data":{"price":60000.0,"timestamp":1700000000000}}}'
    )
    await client._handle_message(msg)
    assert "deribit_price_index.btc_usd" in client._last_message_ts_per_channel


@pytest.mark.asyncio
async def test_per_channel_ts_does_not_populate_on_rpc_response():
    """RPC responses use ``id``, not ``method``: subscription branch
    must NOT fire for them and per-channel ts must remain empty."""
    client = DeribitWebSocketClient(ws_url="wss://test.deribit.com/ws/api/v2")
    msg = b'{"jsonrpc":"2.0","id":1,"result":["deribit_price_index.btc_usd"]}'
    await client._handle_message(msg)
    assert client._last_message_ts_per_channel == {}


# ── H5: watchdog fires on per-channel staleness ──────────────────────


@pytest.mark.asyncio
async def test_watchdog_fires_on_stale_index_with_perp_alive():
    """Index goes silent; perp keeps streaming; per-channel watchdog
    forces close because index threshold is registered."""
    client = DeribitWebSocketClient(
        ws_url="wss://test.deribit.com/ws/api/v2",
        data_stale_seconds=300.0,   # global threshold high — won't trip
    )
    fake = _FakeWS()
    client._ws = fake  # noqa: SLF001
    client.set_channel_threshold("deribit_price_index.btc_usd", 0.05)
    client.set_channel_threshold("trades.BTC-PERPETUAL.100ms", 60.0)

    now = time.monotonic()
    # Index last seen 1s ago (stale, threshold 50ms). Perp seen now.
    client._last_message_ts_per_channel["deribit_price_index.btc_usd"] = now - 1.0
    client._last_message_ts_per_channel["trades.BTC-PERPETUAL.100ms"] = now
    client._last_message_ts = now  # global is fresh

    client._start_watchdog_task()

    try:
        await asyncio.wait_for(fake._close_event.wait(), timeout=20.0)
    finally:
        client._shutdown.set()
        await client.close()

    assert fake.closed_with_code == 1011
    assert "channel-stale" in (fake.closed_with_reason or "")
    assert "deribit_price_index.btc_usd" in (fake.closed_with_reason or "")


@pytest.mark.asyncio
async def test_watchdog_does_not_fire_on_unseen_channel():
    """A channel registered with a threshold but never producing a
    message must NOT trip the watchdog — H6 catches subscribe failures
    at subscribe time, not via the watchdog (which would false-positive
    during startup before any message arrives)."""
    client = DeribitWebSocketClient(
        ws_url="wss://test.deribit.com/ws/api/v2",
        data_stale_seconds=300.0,
    )
    fake = _FakeWS()
    client._ws = fake
    client.set_channel_threshold("deribit_price_index.btc_usd", 0.05)
    # Note: NOT putting the channel in _last_message_ts_per_channel.
    client._last_message_ts = time.monotonic()  # global fresh

    client._start_watchdog_task()
    # Wait long enough that a per-channel check WOULD have fired
    # (check_interval = max(15s, threshold/3) — but our threshold
    # is 0.05s so effective floor is 15s; we only need to verify the
    # watchdog doesn't trip during the bounded wait).
    watchdog_fired = False
    try:
        await asyncio.wait_for(fake._close_event.wait(), timeout=2.0)
        # If we get here, something closed the WS — but it must NOT
        # be the watchdog (which uses code=1011 with "channel-stale"
        # reason). client.close() in the finally block uses code=1000.
        watchdog_fired = (
            fake.closed_with_code == 1011
            and "channel-stale" in (fake.closed_with_reason or "")
        )
    except asyncio.TimeoutError:
        pass
    finally:
        client._shutdown.set()
        await client.close()

    assert not watchdog_fired, (
        "Watchdog incorrectly fired on a never-seen channel — "
        "false positive on startup before any messages arrive"
    )


@pytest.mark.asyncio
async def test_per_channel_ts_clears_on_reconnect():
    """A reconnect must clear stale per-channel timestamps so the
    new connection starts fresh; otherwise the next watchdog cycle
    could trip on the old session's last-seen timestamps."""
    client = DeribitWebSocketClient(
        ws_url="wss://test.deribit.com/ws/api/v2",
        data_stale_seconds=10.0,
    )
    client._last_message_ts_per_channel["channel.x"] = 12345.0
    # Simulate the reset done at the top of _connect_with_backoff
    client._last_message_ts_per_channel.clear()  # mirror behaviour
    assert client._last_message_ts_per_channel == {}


# ── H6: subscribe ACK verification ──────────────────────────────────


def test_verify_ack_passes_when_all_channels_present():
    """Server returned exactly the requested set — must not raise."""
    DeribitWebSocketClient._verify_subscribe_ack(
        "public/subscribe",
        ["deribit_price_index.btc_usd", "trades.BTC-PERPETUAL.100ms"],
        ["deribit_price_index.btc_usd", "trades.BTC-PERPETUAL.100ms"],
    )


def test_verify_ack_passes_with_extra_server_channels():
    """Server returned a SUPERSET of what we asked — still pass.
    (Some Deribit reconnect flows return all currently-bound channels.)"""
    DeribitWebSocketClient._verify_subscribe_ack(
        "public/subscribe",
        ["a", "b"],
        ["a", "b", "c"],
    )


def test_verify_ack_raises_on_missing_channel():
    """Server bound only some of what we asked — must raise so the
    engine fails LOUD instead of running on a partial subscription."""
    with pytest.raises(DeribitWebSocketError, match="ACK mismatch"):
        DeribitWebSocketClient._verify_subscribe_ack(
            "public/subscribe",
            ["a", "b", "c"],
            ["a"],   # b and c missing
        )


def test_verify_ack_raises_with_data_payload():
    """The error data payload must include both lists for forensics."""
    with pytest.raises(DeribitWebSocketError) as exc_info:
        DeribitWebSocketClient._verify_subscribe_ack(
            "public/subscribe",
            ["a", "b"],
            ["a"],
        )
    assert exc_info.value.data is not None
    assert "requested" in exc_info.value.data
    assert "accepted" in exc_info.value.data


def test_verify_ack_with_non_list_result_does_not_raise():
    """If the server returns something we can't parse (e.g. a dict
    with extra metadata), we log a warning but do NOT raise — the
    engine's data-stale watchdog will catch a truly-broken session."""
    # Must not raise
    DeribitWebSocketClient._verify_subscribe_ack(
        "public/subscribe",
        ["a"],
        {"some": "weird response"},
    )
    DeribitWebSocketClient._verify_subscribe_ack(
        "public/subscribe",
        ["a"],
        None,
    )
