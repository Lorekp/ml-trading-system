"""Async Deribit WebSocket client with auto-reconnect and subscription state.

Deribit's WebSocket API uses JSON-RPC 2.0 over ``wss://``.  Clients send
``public/subscribe`` or ``private/subscribe`` messages to join channels,
then receive push notifications for each channel event.  On reconnect,
subscriptions must be re-sent.

This module provides:

- Connection lifecycle with auto-reconnect (exponential backoff, max 30s)
- Subscription state tracking so reconnects can restore channels
- Typed callback dispatch (channel name → handler)
- Heartbeat monitoring (Deribit sends ping frames; we reply automatically
  via the ``websockets`` library's built-in keepalive)
- Gap detection: each trade/ticker event carries a sequence number and
  the consumer can request a snapshot to fill gaps after a disconnect

The client is deliberately minimal — it exposes `connect`, `subscribe`,
`dispatch_forever` and that's it.  Higher-level concerns (trade buffering,
bar aggregation, etc.) live in ``deribit_paper.market_data``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from typing import Any, Awaitable, Callable

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from deribit_paper.client.auth import DeribitAuth
from deribit_paper.client.errors import DeribitWebSocketError

logger = logging.getLogger(__name__)


_RECONNECT_INITIAL_DELAY = 1.0
_RECONNECT_MAX_DELAY = 30.0
_RECONNECT_MAX_ATTEMPTS = 0  # 0 = forever

# TCP-level ping/pong keepalive handled by the websockets library.
# Useful when the underlying socket is gone, but does NOT detect the
# "zombie connection" failure mode where the server keeps answering
# pings while no longer pushing subscription data — see watchdog below.
_PING_INTERVAL = 20
_PING_TIMEOUT = 15

# App-level data-stale watchdog. Unlike ping/pong (TCP liveness), this
# proactively forces a reconnect when NO subscription messages have
# arrived for ``_DATA_STALE_SECONDS_DEFAULT``, which catches the
# Deribit-zombie pattern observed 2026-05-06: WS open + pings OK but
# no trades/index updates flowing for 2h+. Threshold sits BELOW the
# kill-switch network-outage trip (default 120s) so the watchdog
# triggers a reconnect BEFORE the engine halts trading.
_DATA_STALE_SECONDS_DEFAULT = 90.0
_WATCHDOG_CHECK_FLOOR_SECONDS = 15.0

# H11 (2026-05-06): Deribit's canonical liveness primitive. After
# ``public/set_heartbeat`` with interval=N, the server pushes a
# ``test_request`` notification every N seconds; the client MUST reply
# with ``public/test`` within the timeout or the server kills the
# connection. This is belt-and-suspenders to the data-stale watchdog —
# it triggers the server SIDE to drop the connection, while our
# watchdog handles the case where the server is technically up but
# silently broken.
_HEARTBEAT_INTERVAL_DEFAULT = 30


ChannelHandler = Callable[[dict[str, Any]], Awaitable[None]]


class DeribitWebSocketClient:
    """Subscribes to Deribit public/private channels and dispatches events.

    Usage::

        async def handle_trade(params):
            print("trade", params["data"])

        client = DeribitWebSocketClient(ws_url="wss://test.deribit.com/ws/api/v2")
        await client.subscribe(["trades.BTC-PERPETUAL.raw"], handle_trade)
        await client.dispatch_forever()
    """

    def __init__(
        self,
        ws_url: str,
        auth: DeribitAuth | None = None,
        on_reconnect: Callable[[], None] | None = None,
        data_stale_seconds: float = _DATA_STALE_SECONDS_DEFAULT,
        heartbeat_interval_seconds: int = _HEARTBEAT_INTERVAL_DEFAULT,
    ):
        self._ws_url = ws_url
        self._auth = auth
        self._on_reconnect = on_reconnect
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._rpc_id = 0
        self._rpc_id_lock = asyncio.Lock()
        # channel_name -> handler
        self._handlers: dict[str, ChannelHandler] = {}
        # cached list of channels to re-subscribe on reconnect
        self._active_channels: set[str] = set()
        # fired to wake waiters during shutdown
        self._shutdown = asyncio.Event()
        self._pending_responses: dict[int, asyncio.Future] = {}
        # Background reader task — started in _connect_with_backoff so
        # RPC responses are consumed even before dispatch_forever() is called
        self._reader_task: asyncio.Task | None = None
        # Data-stale watchdog: tracks ``time.monotonic()`` of last received
        # message of any kind. The watchdog task closes the WS once the
        # gap exceeds ``_data_stale_seconds`` so the existing reconnect
        # logic kicks in. Callers can raise/lower the threshold via the
        # constructor; passing 0 disables the watchdog.
        self._data_stale_seconds = float(data_stale_seconds)
        self._last_message_ts: float = time.monotonic()
        self._watchdog_task: asyncio.Task | None = None
        # H5 (2026-05-06): per-channel staleness tracking. The global
        # ``_last_message_ts`` only catches "ALL channels stopped" — but
        # a Deribit-side bug or thin liquidity could silence ONE channel
        # (e.g. the index price) while others keep flowing, leaving the
        # global watchdog blind. Map: channel -> monotonic ts of last
        # message. Only populated AFTER the first message arrives, so a
        # never-bound channel doesn't false-trip on startup.
        self._last_message_ts_per_channel: dict[str, float] = {}
        # Per-channel max-staleness thresholds (seconds). Caller (engine)
        # populates via ``set_channel_threshold`` for channels it knows
        # are high-frequency. Channels not in this map use only the
        # global watchdog.
        self._channel_thresholds: dict[str, float] = {}
        # Initialised flag for the WS-auth path so a missing ``__init__``
        # write doesn't surface as ``AttributeError`` on first reconnect.
        self._authenticated_ws: bool = False
        # H11: Deribit set_heartbeat interval. 0 disables; default 30s.
        self._heartbeat_interval_seconds = int(heartbeat_interval_seconds)
        # Stats counter for observability — bumped each time we reply
        # to a server-pushed test_request.
        self._heartbeat_test_responses_sent: int = 0

    @property
    def connected(self) -> bool:
        return self._ws is not None and not self._ws.close_code

    async def connect(self) -> None:
        """Open the WS connection (retries until success)."""
        await self._connect_with_backoff()

    async def close(self) -> None:
        self._shutdown.set()
        if self._watchdog_task is not None and not self._watchdog_task.done():
            self._watchdog_task.cancel()
            try:
                await asyncio.wait_for(self._watchdog_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self._watchdog_task = None
        if self._reader_task is not None and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await asyncio.wait_for(self._reader_task, timeout=2.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self._reader_task = None
        if self._ws is not None:
            await self._ws.close()
            self._ws = None

    def set_channel_threshold(self, channel: str, seconds: float) -> None:
        """H5 (2026-05-06): set a per-channel max-staleness threshold.

        When the watchdog is active, a per-channel threshold means the
        WS will be force-reconnected if the named channel goes silent
        for longer than ``seconds`` AFTER it has produced at least one
        message. Channels without a registered threshold are governed
        only by the global ``data_stale_seconds`` watchdog.

        Typical use from engine.initialize:
            ws.set_channel_threshold("deribit_price_index.btc_usd", 30.0)
            ws.set_channel_threshold("trades.BTC-PERPETUAL.100ms", 120.0)
        """
        if seconds <= 0:
            self._channel_thresholds.pop(channel, None)
            return
        self._channel_thresholds[channel] = float(seconds)

    async def subscribe(
        self,
        channels: list[str],
        handler: ChannelHandler,
    ) -> None:
        """Subscribe to one or more channels with a shared handler.

        Multiple calls with different handlers ARE supported — each
        call registers a handler for its own set of channels.  Two
        subscribe calls with overlapping channel names will result in
        the later handler overwriting the earlier one.

        Deribit requires WS authentication even for some public
        channels (notably the ``.raw`` trade feeds).  We authenticate
        the WS session on the first subscribe call and cache the
        "authenticated" state until the next reconnect.

        H6 (2026-05-06): verifies the server actually accepted the
        requested channel set. Deribit returns the array of accepted
        channels in the JSON-RPC ``result``; we diff against requested
        and raise on mismatch so a typo / expired option / server-side
        rejection fails LOUD instead of leaving the engine flying blind.
        """
        for channel in channels:
            self._handlers[channel] = handler
            self._active_channels.add(channel)

        if self._ws is None or self._ws.close_code:
            await self._connect_with_backoff()

        # Always authenticate the WS session if we have credentials.
        # Deribit gates ``.raw`` trade feeds behind auth even though
        # they're "public" channels.
        if self._auth is not None:
            await self._auth_ws_if_needed()

        is_private = any(ch.startswith("user.") for ch in channels)
        method = "private/subscribe" if is_private else "public/subscribe"
        result = await self._send_rpc(method, {"channels": channels})
        self._verify_subscribe_ack(method, channels, result)

    async def _auth_ws_if_needed(self) -> None:
        """Authenticate the current WS session via client_credentials.

        Idempotent within a single connection — once the session is
        authenticated, subsequent calls are no-ops.  A fresh reconnect
        resets the flag.
        """
        if self._auth is None:
            raise DeribitWebSocketError(
                "Cannot authenticate WS: no DeribitAuth instance provided",
            )
        if getattr(self, "_authenticated_ws", False):
            return
        # Deribit's WS public/auth uses the SAME grant_type values as REST.
        # "client_credentials" is the canonical path and reuses the client
        # id/secret we already have in the auth object.
        await self._send_rpc(
            "public/auth",
            {
                "grant_type": "client_credentials",
                "client_id": self._auth.client_id,
                "client_secret": self._auth.client_secret,
            },
        )
        self._authenticated_ws = True

    async def dispatch_forever(self) -> None:
        """Block until ``close()`` is called, reconnecting as needed.

        The actual message reading is done by the background task
        started in :meth:`_connect_with_backoff`.  This method's job
        is to:
        1. Detect when the reader task has exited (connection dropped)
        2. Reconnect + resubscribe + spawn a new reader
        3. Block on the shutdown event otherwise
        """
        while not self._shutdown.is_set():
            # If the reader task has finished, the connection dropped
            if self._reader_task is None or self._reader_task.done():
                if self._ws is None or self._ws.close_code:
                    logger.info("Deribit WS reader stopped — reconnecting")
                    await self._connect_with_backoff()
                    await self._resubscribe_all()
                    if self._on_reconnect is not None:
                        try:
                            self._on_reconnect()
                        except Exception:  # noqa: BLE001
                            logger.exception("on_reconnect callback raised")
                if self._reader_task is None:
                    # shutdown won during reconnect
                    return

            # Wait for either the reader to die or the shutdown signal
            done, _pending = await asyncio.wait(
                {asyncio.create_task(self._shutdown.wait()), self._reader_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if self._shutdown.is_set():
                return

    # ── Internals ────────────────────────────────────────────────────

    async def _connect_with_backoff(self) -> None:
        """Connect to the WS URL, retrying with exponential backoff.

        Spawns a background reader task immediately after the connect
        so RPC responses are consumed even before :meth:`dispatch_forever`
        has been called.

        NOTE: we deliberately do NOT authenticate the WS session here.
        Deribit only requires WS auth for ``user.*`` private channels,
        and we poll private state via REST instead (simpler, more
        robust).  If a caller later subscribes to a private channel,
        :meth:`subscribe` will invoke :meth:`_auth_ws_if_needed` lazily.
        """
        attempt = 0
        while not self._shutdown.is_set():
            attempt += 1
            try:
                self._ws = await websockets.connect(
                    self._ws_url,
                    ping_interval=_PING_INTERVAL,
                    ping_timeout=_PING_TIMEOUT,
                    max_size=2**22,  # 4 MB — Deribit sends big snapshots
                )
                logger.info(
                    "Deribit WS connected to %s (attempt %d)",
                    self._ws_url, attempt,
                )
                # Reset the data-stale tracker so a fresh connection
                # starts with a clean slate; the watchdog will only fire
                # if NEW messages stop arriving after this point.
                self._last_message_ts = time.monotonic()
                # H5: clear per-channel timestamps so the per-channel
                # watchdog doesn't trip on the carry-over from the dead
                # connection. Repopulated as messages arrive on the new
                # session.
                self._last_message_ts_per_channel.clear()
                # Start the background reader so future RPC responses
                # are consumed.
                self._start_reader_task()
                # Start the app-level watchdog (no-op if disabled or
                # already running).
                self._start_watchdog_task()
                self._authenticated_ws = False  # fresh connection
                # H11: enable Deribit-side heartbeat so the server
                # actively probes us with test_request and tears down
                # the connection if we don't respond. Best-effort —
                # failure to enable is logged and ignored, the
                # data-stale watchdog still protects us.
                if self._heartbeat_interval_seconds > 0:
                    try:
                        await self._send_rpc(
                            "public/set_heartbeat",
                            {"interval": self._heartbeat_interval_seconds},
                        )
                        logger.info(
                            "Deribit set_heartbeat enabled: interval=%ds",
                            self._heartbeat_interval_seconds,
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "Failed to enable set_heartbeat — "
                            "data-stale watchdog still active",
                        )
                return
            except (OSError, WebSocketException) as exc:
                if _RECONNECT_MAX_ATTEMPTS and attempt >= _RECONNECT_MAX_ATTEMPTS:
                    raise DeribitWebSocketError(
                        f"exhausted reconnect attempts ({attempt}): {exc}",
                    ) from exc
                delay = self._backoff_delay(attempt)
                logger.warning(
                    "Deribit WS connect failed (%s) — retry %d in %.2fs",
                    exc, attempt, delay,
                )
                try:
                    await asyncio.wait_for(
                        self._shutdown.wait(), timeout=delay,
                    )
                    return  # shutdown during backoff
                except asyncio.TimeoutError:
                    pass  # backoff complete, retry connect

    def _start_reader_task(self) -> None:
        """Spawn the background message reader if not already running."""
        if self._reader_task is not None and not self._reader_task.done():
            return
        self._reader_task = asyncio.create_task(
            self._read_forever(), name="ws_reader",
        )

    def _start_watchdog_task(self) -> None:
        """Spawn the data-stale watchdog if not already running.

        The watchdog is a single coroutine that periodically compares
        ``time.monotonic() - self._last_message_ts`` against
        ``self._data_stale_seconds``. When the gap exceeds the
        threshold it forcibly closes the WS (via ``self._ws.close``),
        which causes ``_read_forever`` to exit and the outer
        ``dispatch_forever`` loop to reconnect + resubscribe.

        Threshold ``0`` disables the watchdog (useful for unit tests
        and integration runs where stalls are expected).
        """
        if self._data_stale_seconds <= 0:
            return
        if self._watchdog_task is not None and not self._watchdog_task.done():
            return
        self._watchdog_task = asyncio.create_task(
            self._watchdog_loop(), name="ws_watchdog",
        )

    async def _watchdog_loop(self) -> None:
        """Force-reconnect when no message has arrived for too long.

        Why this exists: Deribit's WS occasionally enters a "zombie"
        state where the underlying TCP connection stays alive (so the
        websockets-library keepalive ping/pong succeeds) but the
        server stops pushing subscription data. Observed 2026-05-06:
        no trade/index updates for ~2h, six 15-min bars kill-switched
        in a row, only recovered when the underlying TCP finally
        died. This watchdog cuts that detection time from ``hours``
        to ``threshold + check_interval`` (under 2 minutes by default).

        H5 (2026-05-06): also detects PARTIAL silence — a single
        channel going quiet while others keep flowing (the global
        watchdog can't see this case). Per-channel thresholds set via
        ``set_channel_threshold``; a channel-stale event force-closes
        the WS the same way as a global stale.
        """
        # Check cadence: bounded by both the global threshold AND any
        # per-channel threshold (whichever is smallest), so a channel
        # configured with 30s staleness gets checked at least every 10s.
        all_thresholds = [self._data_stale_seconds, *self._channel_thresholds.values()]
        smallest = min(t for t in all_thresholds if t > 0) if all_thresholds else self._data_stale_seconds
        check_interval = max(
            _WATCHDOG_CHECK_FLOOR_SECONDS,
            smallest / 3.0,
        )
        try:
            while not self._shutdown.is_set():
                try:
                    await asyncio.wait_for(
                        self._shutdown.wait(),
                        timeout=check_interval,
                    )
                    return  # shutdown
                except asyncio.TimeoutError:
                    pass

                ws = self._ws
                if ws is None or getattr(ws, "close_code", None):
                    # Connection already gone — reconnect path will
                    # restart us. Exit cleanly.
                    return

                now = time.monotonic()

                # Global staleness check
                stale = now - self._last_message_ts
                if stale > self._data_stale_seconds:
                    logger.warning(
                        "WS data-stale watchdog: %.0fs since last message "
                        "(threshold %.0fs) — forcing reconnect",
                        stale, self._data_stale_seconds,
                    )
                    try:
                        await ws.close(
                            code=1011, reason="data-stale watchdog",
                        )
                    except Exception:  # noqa: BLE001 - best-effort close
                        logger.exception("Watchdog: ws.close raised")
                    return  # reconnect path will respawn us

                # H5: per-channel staleness check. Only fires for
                # channels we've SEEN at least once — never-bound
                # channels are the subscribe-ACK problem (H6 catches
                # those at subscribe time). Once seen, a channel
                # stopping is a strong signal of partial blindness.
                for ch, thresh in list(self._channel_thresholds.items()):
                    last = self._last_message_ts_per_channel.get(ch)
                    if last is None:
                        continue
                    ch_stale = now - last
                    if ch_stale > thresh:
                        logger.warning(
                            "WS per-channel watchdog: channel %s stale "
                            "%.0fs (threshold %.0fs) — forcing reconnect",
                            ch, ch_stale, thresh,
                        )
                        try:
                            await ws.close(
                                code=1011,
                                reason=f"channel-stale {ch}",
                            )
                        except Exception:  # noqa: BLE001
                            logger.exception("Watchdog: ws.close raised")
                        return  # reconnect path will respawn us
        except asyncio.CancelledError:
            pass

    async def _read_forever(self) -> None:
        """Continuously read frames from the current socket + dispatch.

        Exits when the connection closes; the outer
        :meth:`dispatch_forever` loop (or the next subscribe call) will
        reconnect and spawn a fresh reader task.
        """
        if self._ws is None:
            return
        try:
            async for raw in self._ws:
                await self._handle_message(raw)
        except ConnectionClosed as exc:
            logger.warning("Deribit WS reader closed: %s", exc)
        except Exception:  # noqa: BLE001 - log + exit; outer loop will reconnect
            logger.exception("Deribit WS reader raised")

    async def _auth_ws(self) -> None:
        """DEPRECATED — use :meth:`_auth_ws_if_needed` instead.

        Kept for backward-compat in case older callers exist.
        """
        await self._auth_ws_if_needed()

    @staticmethod
    def _verify_subscribe_ack(
        method: str, requested: list[str], result: Any,
    ) -> None:
        """H6: raise on subscribe ACK mismatch.

        Deribit's ``public/subscribe`` and ``private/subscribe`` return
        the JSON array of channels the server bound. If a channel
        was rejected (typo, expired option, server-side migration), it
        is silently dropped from the array. Pre-H6 the result was
        discarded — our engine would happily run with one or more
        channels missing, blind to the gap until the kill-switch
        eventually noticed lower data volume than expected.

        Now: diff requested vs accepted; raise DeribitWebSocketError
        on any missing channels so the caller (engine.initialize or
        the reconnect path) fails LOUD.
        """
        if not isinstance(result, list):
            logger.warning(
                "%s ACK was not a list: %r — skipping verification",
                method, result,
            )
            return
        accepted = {str(c) for c in result}
        missing = [c for c in requested if c not in accepted]
        if missing:
            extra = sorted(accepted - set(requested))
            raise DeribitWebSocketError(
                f"{method} ACK mismatch: server did not bind {missing} "
                f"(requested {len(requested)}, got {len(accepted)}; "
                f"extra={extra})",
                data={"requested": list(requested), "accepted": sorted(accepted)},
            )
        logger.debug(
            "%s ACK OK: %d channel(s) bound", method, len(accepted),
        )

    async def _resubscribe_all(self) -> None:
        """Re-send subscribe for every channel we had before the disconnect.

        Authenticates the fresh WS session first — Deribit gates ``.raw``
        channels behind WS auth even though they are "public".

        H6: verifies ACKs (see ``_verify_subscribe_ack``).
        """
        if not self._active_channels:
            return
        # Authenticate the fresh session before subscribing.
        if self._auth is not None:
            await self._auth_ws_if_needed()
        public = [ch for ch in self._active_channels if not ch.startswith("user.")]
        private = [ch for ch in self._active_channels if ch.startswith("user.")]
        try:
            if public:
                result = await self._send_rpc(
                    "public/subscribe", {"channels": public},
                )
                self._verify_subscribe_ack("public/subscribe", public, result)
            if private:
                result = await self._send_rpc(
                    "private/subscribe", {"channels": private},
                )
                self._verify_subscribe_ack("private/subscribe", private, result)
        except DeribitWebSocketError:
            logger.exception(
                "Resubscribe failed after reconnect — will retry on next cycle",
            )
            raise
        logger.info(
            "Resubscribed to %d public + %d private channels after reconnect",
            len(public), len(private),
        )

    async def _send_rpc(
        self, method: str, params: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a JSON-RPC call and wait for the response."""
        if self._ws is None:
            raise DeribitWebSocketError("WS not connected")

        async with self._rpc_id_lock:
            self._rpc_id += 1
            rpc_id = self._rpc_id

        future: asyncio.Future = asyncio.get_event_loop().create_future()
        self._pending_responses[rpc_id] = future

        payload = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": method,
            "params": params,
        }
        await self._ws.send(json.dumps(payload))

        try:
            result = await asyncio.wait_for(future, timeout=10.0)
        except asyncio.TimeoutError as exc:
            self._pending_responses.pop(rpc_id, None)
            raise DeribitWebSocketError(
                f"RPC timeout waiting for response to {method}",
            ) from exc
        finally:
            self._pending_responses.pop(rpc_id, None)

        return result

    async def _handle_message(self, raw: str | bytes) -> None:
        """Parse a WS frame and dispatch to handlers or pending RPCs."""
        # Stamp BEFORE JSON-parsing so even malformed frames count as
        # "the connection is alive". Anything that reaches us through
        # the socket reader counts; JSON errors only trigger if a
        # genuine frame arrived.
        now = time.monotonic()
        self._last_message_ts = now
        try:
            msg = json.loads(raw)
        except json.JSONDecodeError:
            logger.error("Deribit WS sent non-JSON frame: %r", raw[:200])
            return

        # Response to an RPC call
        if "id" in msg:
            rpc_id = msg["id"]
            future = self._pending_responses.get(rpc_id)
            if future is not None and not future.done():
                if "error" in msg:
                    future.set_exception(
                        DeribitWebSocketError(
                            f"RPC error: {msg['error']}",
                            data=msg.get("error"),
                        ),
                    )
                else:
                    future.set_result(msg.get("result"))
            return

        # Subscription push notification
        if msg.get("method") == "subscription":
            params = msg.get("params", {})
            channel = params.get("channel")
            # H5: per-channel staleness tracking. Stamp BEFORE handler
            # dispatch so a slow / failing handler doesn't backdate the
            # channel's last-seen timestamp.
            if channel:
                self._last_message_ts_per_channel[channel] = now
            if channel in self._handlers:
                try:
                    await self._handlers[channel](params)
                except Exception:  # noqa: BLE001 - log + continue
                    logger.exception(
                        "Handler for channel %s raised — dispatching continues",
                        channel,
                    )
            else:
                logger.debug("Received message for unhandled channel %s", channel)
            return

        # H11: Deribit heartbeat ``test_request`` — server probing us
        # for liveness. Must reply with public/test within the timeout
        # or the server kills the connection. ``heartbeat`` (no
        # ``test_request``) is a no-op informational ping.
        if msg.get("method") == "heartbeat":
            params = msg.get("params", {})
            if params.get("type") == "test_request":
                try:
                    await self._send_test_response()
                    self._heartbeat_test_responses_sent += 1
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "Failed to send public/test response — server may "
                        "drop the connection",
                    )
            # ``type=heartbeat`` is informational: server is alive.
            # No action required.
            return

        logger.debug("Deribit WS unhandled message: %s", msg)

    async def _send_test_response(self) -> None:
        """H11: respond to a server-pushed ``test_request`` with
        ``public/test``. Fire-and-forget JSON-RPC notification; we do
        NOT await a response (the heartbeat exchange is one-shot per
        interval, and ``_send_rpc`` would block until the response
        arrives, which is wasteful for every heartbeat tick).
        """
        if self._ws is None:
            return
        async with self._rpc_id_lock:
            self._rpc_id += 1
            rpc_id = self._rpc_id
        payload = {
            "jsonrpc": "2.0",
            "id": rpc_id,
            "method": "public/test",
            "params": {},
        }
        await self._ws.send(json.dumps(payload))

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        """Exponential backoff with jitter, capped at ``_RECONNECT_MAX_DELAY``."""
        base = min(
            _RECONNECT_INITIAL_DELAY * (2 ** (attempt - 1)),
            _RECONNECT_MAX_DELAY,
        )
        jitter = base * 0.25 * random.random()
        return base + jitter
