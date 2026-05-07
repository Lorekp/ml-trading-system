"""Live VPIN computation from Binance BTCUSDT perp aggTrades WebSocket.

Hybrid warm-start architecture:

1. At startup, load serialized ``VPINStreamer`` state (produced by
   ``scripts/recompute_vpin.py``) so the rolling window of 50 bucket
   imbalances already reflects 5 years of history.
2. Connect to ``wss://fstream.binance.com/ws/btcusdt@aggTrade`` and
   process each incoming aggTrade through scalar bucket accumulation.
3. Expose ``latest_vpin`` as a synchronous property for the execution
   layer (read by ``StraddleFlow.begin_trade`` at each 15-min bar).

The Binance aggTrade WebSocket sends one JSON message per aggTrade::

    {"e":"aggTrade","s":"BTCUSDT","p":"65432.10","q":"0.125","m":true,"T":1712345678901}

``m`` is ``is_buyer_maker``: ``true`` means the aggressor was the seller.

VPIN is a slow-moving quantity (50-bucket rolling mean over ~4 hours of
flow at default params).  Brief WebSocket disconnections have negligible
effect; the fallback value (training-distribution mean) produces a
near-zero normalized observation under VecNormalize.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from collections import deque
from pathlib import Path
from statistics import mean as _mean

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

logger = logging.getLogger(__name__)

# Default Binance Futures aggTrade stream URL (public, no auth needed).
# Use _default_binance_ws_url(currency) to build the URL for any asset.
_DEFAULT_WS_URL = "wss://fstream.binance.com/ws/btcusdt@aggTrade"


def _default_binance_ws_url(currency: str = "BTC") -> str:
    """Build Binance Futures aggTrade WebSocket URL for any asset."""
    return f"wss://fstream.binance.com/ws/{currency.lower()}usdt@aggTrade"

# Max reconnect backoff (seconds)
_MAX_BACKOFF_S = 30.0


class _LiveBucketAccumulator:
    """Scalar bucket accumulator for streaming VPIN computation.

    Maintains the same state as ``VPINStreamer`` from
    ``scripts/recompute_vpin.py`` but processes one trade at a time
    (pure scalar arithmetic, no numpy/pandas) for real-time use.
    """

    __slots__ = (
        "V", "N", "_partial_vol", "_partial_buy", "_partial_sell",
        "_recent_imbalances", "_latest_vpin", "_buckets_closed",
    )

    def __init__(
        self,
        bucket_volume_btc: float = 1000.0,
        rolling_window: int = 50,
    ):
        self.V = float(bucket_volume_btc)
        self.N = int(rolling_window)
        self._partial_vol = 0.0
        self._partial_buy = 0.0
        self._partial_sell = 0.0
        self._recent_imbalances: deque[float] = deque(maxlen=self.N)
        self._latest_vpin: float | None = None
        self._buckets_closed = 0

    def load_state(self, state: dict) -> None:
        """Restore carry-forward state from a deserialized JSON dict."""
        self._partial_vol = float(state["partial_vol"])
        self._partial_buy = float(state["partial_buy"])
        self._partial_sell = float(state["partial_sell"])
        self._recent_imbalances = deque(
            (float(x) for x in state["recent_imbalances"]),
            maxlen=self.N,
        )
        if self._recent_imbalances:
            self._latest_vpin = _mean(self._recent_imbalances)

    def process_trade(self, quantity: float, is_seller_aggressor: bool) -> None:
        """Process a single aggTrade.  May close one or more buckets."""
        buy = 0.0 if is_seller_aggressor else quantity
        sell = quantity if is_seller_aggressor else 0.0
        self._partial_vol += quantity
        self._partial_buy += buy
        self._partial_sell += sell

        # Close as many full buckets as this trade completes
        while self._partial_vol >= self.V:
            # Fraction of this trade that belongs to the current bucket
            overflow = self._partial_vol - self.V
            frac = (quantity - overflow) / quantity if quantity > 0 else 1.0
            # But we already added the full trade — adjust the bucket
            # by removing the overflow portion that belongs to the next
            # bucket.
            bucket_buy = self._partial_buy - (buy * (overflow / quantity) if quantity > 0 else 0.0)
            bucket_sell = self._partial_sell - (sell * (overflow / quantity) if quantity > 0 else 0.0)

            imbalance = abs(bucket_buy - bucket_sell) / self.V
            self._recent_imbalances.append(imbalance)
            self._latest_vpin = _mean(self._recent_imbalances)
            self._buckets_closed += 1

            # Carry the overflow into the next bucket
            self._partial_vol = overflow
            self._partial_buy = buy * (overflow / quantity) if quantity > 0 else 0.0
            self._partial_sell = sell * (overflow / quantity) if quantity > 0 else 0.0


class LiveVPINStream:
    """Live VPIN from Binance BTCUSDT perp aggTrades WebSocket.

    Usage::

        stream = LiveVPINStream(
            state_path=Path("vol_arb/output/vpin_streamer_state.json"),
        )
        task = asyncio.create_task(stream.run_forever())
        ...
        vpin = stream.latest_vpin   # 0.189 fallback until first bucket
        ...
        await stream.stop()
    """

    def __init__(
        self,
        *,
        bucket_volume_btc: float = 1000.0,
        rolling_window: int = 50,
        state_path: Path | None = None,
        fallback_vpin: float = 0.189,
        ws_url: str = _DEFAULT_WS_URL,
        stale_seconds: float = 600.0,
    ):
        self._acc = _LiveBucketAccumulator(bucket_volume_btc, rolling_window)
        self._fallback = float(fallback_vpin)
        self._ws_url = ws_url
        self._lock = threading.Lock()
        self._shutdown = asyncio.Event()
        self._connected = False
        self._reconnect_attempts = 0
        self._last_update_ts: float | None = None
        self._trades_processed = 0
        self._stale_seconds = float(stale_seconds)
        # Bug fix (2026-05-07): track stream construction time so
        # ``is_stale`` can transition from "warmup" to "stale" after a
        # reasonable grace period when ``_last_update_ts`` stays None.
        # Without this, a permanently-silent feed (e.g. geo-blocked
        # Binance Futures returning zero messages) reads as "warming up"
        # forever and never trips the operator alert.
        self._stream_start_ts: float = time.time()

        # Warm-start from serialized state
        if state_path is not None:
            p = Path(state_path)
            if p.exists():
                from scripts.recompute_vpin import VPINStreamer
                state = VPINStreamer.load_state(p)
                self._acc.load_state(state)
                logger.info(
                    "LiveVPINStream warm-start: loaded %d imbalances, "
                    "initial VPIN=%.4f, partial_vol=%.1f BTC",
                    len(self._acc._recent_imbalances),
                    self._acc._latest_vpin or self._fallback,
                    self._acc._partial_vol,
                )
            else:
                logger.warning(
                    "LiveVPINStream state_path %s does not exist; "
                    "starting cold (fallback=%.3f)",
                    p, self._fallback,
                )

    @property
    def latest_vpin(self) -> float:
        """Return the most recent VPIN value (thread-safe read).

        Falls back to ``fallback_vpin`` if no buckets have closed since
        startup AND no warm-start state was loaded.
        """
        with self._lock:
            v = self._acc._latest_vpin
        return v if v is not None else self._fallback

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_stale(self) -> bool:
        """True if no trade updates received for > stale_seconds.

        Two failure modes produce stale data:

        1. Stream WAS receiving and went silent (last_update_ts > 0,
           but ``time.time() - last_update_ts`` exceeds ``stale_seconds``).
        2. Stream NEVER received any messages — e.g. the Binance
           Futures aggTrade endpoint is geo-blocked from this host's
           IP. The WS handshake succeeds but no aggTrade frames flow.
           Pre-fix, this case returned False forever ("warmup, not
           stale"); the operator alert never fired and the live
           engine ran on the warm-start VPIN value indefinitely.
           Now: if the stream has been running longer than
           ``stale_seconds`` AND has never received a message, that
           is stale, not warmup.
        """
        now = time.time()
        if self._last_update_ts is None:
            # Pre-fix: returned False unconditionally. Now: stale if we
            # have been running long enough that "warmup" no longer
            # explains the silence.
            return (now - self._stream_start_ts) > self._stale_seconds
        return (now - self._last_update_ts) > self._stale_seconds

    @property
    def buckets_since_start(self) -> int:
        return self._acc._buckets_closed

    @property
    def last_update_ts(self) -> float | None:
        return self._last_update_ts

    async def run_forever(self) -> None:
        """Connect to Binance aggTrade WS with auto-reconnect.

        Runs until ``stop()`` is called.  Reconnects with exponential
        backoff on disconnection, matching the Deribit WS client pattern.
        """
        while not self._shutdown.is_set():
            try:
                async with websockets.connect(
                    self._ws_url,
                    ping_interval=20,
                    ping_timeout=15,
                ) as ws:
                    self._connected = True
                    self._reconnect_attempts = 0
                    logger.info("LiveVPINStream: Binance aggTrade WS connected")
                    async for raw in ws:
                        if self._shutdown.is_set():
                            return
                        self._process_message(raw)
            except (ConnectionClosed, WebSocketException, OSError) as exc:
                self._connected = False
                if self._shutdown.is_set():
                    return
                delay = min(2 ** self._reconnect_attempts, _MAX_BACKOFF_S)
                logger.warning(
                    "LiveVPINStream: WS disconnected (%s) -- retry in %.1fs",
                    exc, delay,
                )
                self._reconnect_attempts += 1
                try:
                    await asyncio.wait_for(
                        self._shutdown.wait(), timeout=delay,
                    )
                    return  # shutdown signaled during backoff
                except asyncio.TimeoutError:
                    pass  # backoff elapsed, reconnect
            except Exception:
                self._connected = False
                if self._shutdown.is_set():
                    return
                logger.exception("LiveVPINStream: unexpected error")
                await asyncio.sleep(5.0)

    async def stop(self) -> None:
        """Signal graceful shutdown."""
        self._shutdown.set()
        self._connected = False
        logger.info(
            "LiveVPINStream stopped: %d trades processed, %d buckets closed, "
            "final VPIN=%.4f",
            self._trades_processed, self._acc._buckets_closed,
            self.latest_vpin,
        )

    def _process_message(self, raw: str) -> None:
        """Parse a Binance aggTrade JSON message and update the accumulator."""
        try:
            msg = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            logger.warning("LiveVPINStream: malformed message, skipping")
            return

        # Binance aggTrade format: {"e":"aggTrade","q":"0.125","m":true,...}
        try:
            quantity = float(msg["q"])
            is_seller_aggressor = bool(msg["m"])
        except (KeyError, ValueError, TypeError):
            return  # silently skip non-aggTrade messages (e.g. pong)

        with self._lock:
            self._acc.process_trade(quantity, is_seller_aggressor)

        self._trades_processed += 1
        self._last_update_ts = time.time()
