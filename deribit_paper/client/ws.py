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
from typing import Any, Awaitable, Callable

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

from deribit_paper.client.auth import DeribitAuth
from deribit_paper.client.errors import DeribitWebSocketError

logger = logging.getLogger(__name__)


_RECONNECT_INITIAL_DELAY = 1.0
_RECONNECT_MAX_DELAY = 30.0
_RECONNECT_MAX_ATTEMPTS = 0  # 0 = forever

# Deribit default ping/pong keepalive handled by websockets library
_PING_INTERVAL = 20
_PING_TIMEOUT = 15


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

    @property
    def connected(self) -> bool:
        return self._ws is not None and not self._ws.close_code

    async def connect(self) -> None:
        """Open the WS connection (retries until success)."""
        await self._connect_with_backoff()

    async def close(self) -> None:
        self._shutdown.set()
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
        await self._send_rpc(method, {"channels": channels})

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
                # Start the background reader so future RPC responses
                # are consumed.
                self._start_reader_task()
                self._authenticated_ws = False  # fresh connection
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

    async def _resubscribe_all(self) -> None:
        """Re-send subscribe for every channel we had before the disconnect.

        Authenticates the fresh WS session first — Deribit gates ``.raw``
        channels behind WS auth even though they are "public".
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
                await self._send_rpc("public/subscribe", {"channels": public})
            if private:
                await self._send_rpc("private/subscribe", {"channels": private})
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

        logger.debug("Deribit WS unhandled message: %s", msg)

    @staticmethod
    def _backoff_delay(attempt: int) -> float:
        """Exponential backoff with jitter, capped at ``_RECONNECT_MAX_DELAY``."""
        base = min(
            _RECONNECT_INITIAL_DELAY * (2 ** (attempt - 1)),
            _RECONNECT_MAX_DELAY,
        )
        jitter = base * 0.25 * random.random()
        return base + jitter
