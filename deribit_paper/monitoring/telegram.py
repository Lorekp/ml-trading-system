"""Async Telegram bot notifier for live alerts.

Wraps the Telegram Bot API ``sendMessage`` endpoint with rate-limiting
and per-category throttling so the engine doesn't spam the user.

Alert categories:

- ``trade_open`` — one per opened straddle (direction, strike, N)
- ``trade_close`` — one per closed straddle (realized PnL)
- ``kill_switch`` — one per halt event (reason, state)
- ``error`` — throttled: at most one per minute per category
- ``heartbeat`` — hourly status: n_open, equity, daily_pnl
- ``daily_summary`` — once at 00:00 UTC: trade count, win rate, PnL

Messages are queued via :meth:`send` and sent asynchronously in
:meth:`flush_loop` (runs as a background asyncio task).  If Telegram
is unreachable the messages buffer up; the buffer is bounded so we
don't run out of memory during a long network outage.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

import httpx

logger = logging.getLogger(__name__)


_TELEGRAM_API_BASE = "https://api.telegram.org"
_DEFAULT_QUEUE_MAX: int = 500

# Per-category throttle: minimum seconds between alerts of the same
# category.  Most categories have no throttle (send immediately) but
# ``error`` gets 60s to prevent storms.
_CATEGORY_THROTTLE_SECONDS: dict[str, float] = {
    "error": 60.0,
    "heartbeat": 0.0,
    "trade_open": 0.0,
    "trade_close": 0.0,
    "kill_switch": 0.0,
    "daily_summary": 0.0,
}


@dataclass
class _QueuedMessage:
    category: str
    text: str
    enqueued_at: float


class TelegramNotifier:
    """Async Telegram bot wrapper with category throttling + bounded queue."""

    def __init__(
        self,
        *,
        bot_token: str,
        chat_id: str,
        enabled: bool = True,
        queue_max: int = _DEFAULT_QUEUE_MAX,
    ):
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._enabled = enabled and bool(bot_token and chat_id)
        self._queue: deque[_QueuedMessage] = deque(maxlen=queue_max)
        self._last_sent_at: dict[str, float] = {}
        self._client: httpx.AsyncClient | None = None
        self._shutdown = asyncio.Event()
        if not self._enabled:
            logger.warning(
                "TelegramNotifier disabled (bot_token=%s chat_id=%s)",
                bool(bot_token), bool(chat_id),
            )

    @property
    def enabled(self) -> bool:
        return self._enabled

    async def start(self) -> None:
        if not self._enabled:
            return
        self._client = httpx.AsyncClient(
            base_url=f"{_TELEGRAM_API_BASE}/bot{self._bot_token}",
            timeout=httpx.Timeout(15.0, connect=10.0),
        )

    async def stop(self) -> None:
        self._shutdown.set()
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    def send(self, category: str, text: str) -> None:
        """Enqueue a message for the background flush loop.

        Drops throttled messages silently (per-category min spacing).
        """
        if not self._enabled:
            return
        now = time.monotonic()
        throttle = _CATEGORY_THROTTLE_SECONDS.get(category, 0.0)
        if throttle > 0.0:
            last = self._last_sent_at.get(category, 0.0)
            if now - last < throttle:
                return
        self._last_sent_at[category] = now
        self._queue.append(_QueuedMessage(category=category, text=text, enqueued_at=now))

    async def flush_once(self) -> int:
        """Drain the queue once.  Returns the number sent successfully."""
        if not self._enabled or self._client is None:
            return 0
        sent = 0
        while self._queue:
            msg = self._queue.popleft()
            try:
                resp = await self._client.post(
                    "/sendMessage",
                    json={
                        "chat_id": self._chat_id,
                        "text": f"[{msg.category}] {msg.text}",
                        "parse_mode": "HTML",
                    },
                )
                resp.raise_for_status()
                sent += 1
            except httpx.HTTPError as exc:
                logger.warning(
                    "Telegram send failed (%s); re-queueing %d messages",
                    exc, len(self._queue) + 1,
                )
                self._queue.appendleft(msg)
                break
        return sent

    async def flush_loop(self, interval_seconds: float = 2.0) -> None:
        """Background loop that periodically flushes the queue."""
        while not self._shutdown.is_set():
            try:
                await self.flush_once()
            except Exception:  # noqa: BLE001 - never let the loop die
                logger.exception("flush_loop iteration raised — continuing")
            try:
                await asyncio.wait_for(self._shutdown.wait(), timeout=interval_seconds)
            except asyncio.TimeoutError:
                pass  # normal path — loop again

    # ── Convenience methods for typed alerts ────────────────────────

    def trade_open(
        self,
        *,
        trade_id: str,
        direction: int,
        strike: float,
        n_contracts_btc: float,
        expiry_dt: Any,
        iv_used_pct: float,
    ) -> None:
        side = "SHORT" if direction == -1 else "LONG"
        self.send(
            "trade_open",
            f"🎯 <b>{side}</b> straddle {trade_id[:8]}\n"
            f"strike ${strike:,.0f}, N={n_contracts_btc:.2f}BTC\n"
            f"expiry {expiry_dt} IV={iv_used_pct:.2f}%",
        )

    def trade_close(
        self,
        *,
        trade_id: str,
        realized_pnl_usd: float,
        hedge_ticks: int,
    ) -> None:
        emoji = "✅" if realized_pnl_usd >= 0 else "⚠️"
        self.send(
            "trade_close",
            f"{emoji} closed {trade_id[:8]}: "
            f"PnL ${realized_pnl_usd:+,.2f}, {hedge_ticks} hedge ticks",
        )

    def kill_switch(self, reason: str) -> None:
        self.send(
            "kill_switch",
            f"🛑 <b>KILL SWITCH TRIPPED</b>\n{reason}",
        )

    def error(self, message: str) -> None:
        self.send("error", f"❌ {message}")

    def heartbeat(
        self,
        *,
        n_open: int,
        equity_usd: float,
        daily_pnl_pct: float,
        drawdown_pct: float,
        vpin: float | None = None,
        vpin_connected: bool | None = None,
        bars_processed: int = 0,
        svi_age_hours: float | None = None,
        vpin_stale: bool = False,
        consecutive_empty_bars: int = 0,
        kill_switch_halted: bool = False,
        svi_recal_failures: int = 0,
        paused: bool = False,
        draining: bool = False,
    ) -> None:
        parts = [
            f"n_open={n_open} equity=${equity_usd:,.0f} "
            f"daily {daily_pnl_pct:+.2f}% dd {drawdown_pct:+.2f}%",
        ]
        if vpin is not None:
            ws_status = "OK" if vpin_connected else "DISCONNECTED"
            parts.append(f"vpin={vpin:.3f} ws={ws_status}")
        if bars_processed > 0:
            parts.append(f"bars={bars_processed}")
        # R4: Enriched diagnostics
        diag: list[str] = []
        if svi_age_hours is not None:
            diag.append(f"svi_age={svi_age_hours:.1f}h")
        if vpin_stale:
            diag.append("VPIN_STALE")
        if consecutive_empty_bars > 0:
            diag.append(f"empty_bars={consecutive_empty_bars}")
        if kill_switch_halted:
            diag.append("KILL_SWITCH_ON")
        if svi_recal_failures > 0:
            diag.append(f"svi_fails={svi_recal_failures}")
        if paused:
            diag.append("PAUSED")
        if draining:
            diag.append("DRAINING")
        if diag:
            parts.append(" ".join(diag))
        self.send("heartbeat", "\n".join(parts))

    def daily_summary(
        self,
        *,
        date: str,
        n_trades: int,
        win_rate_pct: float,
        realized_pnl_usd: float,
        equity_usd: float,
    ) -> None:
        self.send(
            "daily_summary",
            f"📊 {date}: {n_trades} trades, "
            f"win rate {win_rate_pct:.1f}%, "
            f"PnL ${realized_pnl_usd:+,.2f}, equity ${equity_usd:,.0f}",
        )
