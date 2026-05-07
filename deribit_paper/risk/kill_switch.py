"""Kill switch: automatic halt of new trades under adverse conditions.

Monitors several trip conditions in parallel:

1. **Daily drawdown** — if the current equity is below the 00:00 UTC
   snapshot by more than ``max_daily_drawdown_pct``, halt.
2. **Consecutive errors** — if the last N execution attempts all
   resulted in errors (order rejection, network failure), halt.
3. **Network outage** — if the market-data WebSocket has been silent
   for more than ``network_outage_seconds``, halt.
4. **Manual halt** — the CLI or the engine can call :meth:`trip`
   directly (e.g., when a position blows through its hard margin cap).

When tripped, the engine:
- Stops accepting new signals (existing positions continue until exit)
- Sends a Telegram alert
- Persists the halt state to SQLite so a restart doesn't auto-resume

:meth:`reset` is the manual re-arm after human review.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Callable

import pandas as pd

from deribit_paper.position.equity import EquityTracker

logger = logging.getLogger(__name__)


@dataclass
class KillSwitchState:
    halted: bool
    reason: str | None
    halted_at: pd.Timestamp | None


class KillSwitch:
    """Stateful kill switch checked before every signal execution.

    Usage::

        ks = KillSwitch(
            equity_tracker=equity,
            max_daily_drawdown_pct=2.0,
            max_consecutive_errors=3,
            network_outage_seconds=120,
        )

        # On every bar:
        if ks.is_halted():
            logger.warning("KillSwitch halted: %s", ks.state.reason)
            return

        # After each signal execution:
        ks.on_trade_result(success=outcome.status == "completed")
    """

    def __init__(
        self,
        *,
        equity_tracker: EquityTracker,
        max_daily_drawdown_pct: float = 2.0,
        max_consecutive_errors: int = 3,
        network_outage_seconds: float = 120.0,
        alert_callback: Callable[[str], None] | None = None,
        persist_callback: Callable[[bool, str | None], None] | None = None,
        started_at: pd.Timestamp | None = None,
    ):
        self._equity = equity_tracker
        self.max_daily_dd_pct = max_daily_drawdown_pct
        self.max_consecutive_errors = max_consecutive_errors
        self.network_outage_seconds = network_outage_seconds
        self._alert_callback = alert_callback
        # H1 (2026-05-06): persist_callback writes (halted, reason) to the
        # SQLite store so a tripped kill-switch survives process restart.
        # Called from trip() and reset() in best-effort fashion — failure
        # to persist must not mask the trip itself, so any callback
        # exception is logged and swallowed (mirroring alert_callback).
        # Sync signature; engine wraps an asyncio.create_task internally.
        self._persist_callback = persist_callback
        self._state = KillSwitchState(halted=False, reason=None, halted_at=None)
        self._consecutive_errors = 0
        self._last_ws_message_at: pd.Timestamp | None = None
        # C2: Flag for engine to trigger emergency close on next process_bar
        self.needs_emergency_close: bool = False
        # Bug fix P0-2 (2026-05-07): track construction time so the
        # network-outage check can trip when ``_last_ws_message_at`` is
        # None AND too much time has passed since startup (e.g., Deribit
        # subscribe ACKed but no events ever arrive — a silent
        # geo-block, or all channels rejected silently). Pre-fix the
        # check was skipped entirely when ``_last_ws_message_at is
        # None``, leaving the outage trip dead during cold-start
        # silence.
        self._started_at: pd.Timestamp = (
            started_at if started_at is not None
            else pd.Timestamp.now(tz="UTC")
        )

    @property
    def state(self) -> KillSwitchState:
        return self._state

    def is_halted(self) -> bool:
        return self._state.halted

    def mark_ws_message(self, at: pd.Timestamp | None = None) -> None:
        """Record that a live WS push notification was received.

        Must be called from the market data subscriber's event handlers
        (actual WS data arrival), NOT from bar-clock ticks.  Using
        bar ticks would mask a dead feed — the BarClock fires on
        wall-clock time regardless of whether data is flowing.
        """
        self._last_ws_message_at = at or pd.Timestamp.now(tz="UTC")

    def check_before_signal(self, now: pd.Timestamp | None = None) -> bool:
        """Evaluate all trip conditions.  Returns True if the switch is now halted.

        Should be called before executing any new signal.  If True is
        returned, the engine MUST NOT open a new position.
        """
        if self._state.halted:
            return True

        now = now or pd.Timestamp.now(tz="UTC")

        # 1. Daily drawdown
        if self._equity.dropped_below_threshold(self.max_daily_dd_pct):
            return self.trip(
                f"daily drawdown {self._equity.drawdown_pct:.2f}% "
                f"exceeds threshold -{self.max_daily_dd_pct:.2f}%",
                now=now,
            )

        # 2. Network outage
        # P0-2 fix (2026-05-07): when no WS message has EVER arrived,
        # measure silence from engine startup time instead of skipping
        # the check. Pre-fix path was silent on cold-start scenarios
        # like the Binance Futures geo-block — the engine would run for
        # hours with `_last_ws_message_at is None` and the kill-switch
        # never tripped because the `if not None` gate was False.
        if self._last_ws_message_at is None:
            ref_ts = self._started_at
            silence_kind = "WS messages never arrived since startup"
        else:
            ref_ts = self._last_ws_message_at
            silence_kind = "since last WS message"
        silence = (now - ref_ts).total_seconds()
        if silence > self.network_outage_seconds:
            return self.trip(
                f"network outage: {silence:.1f}s {silence_kind} "
                f"(threshold {self.network_outage_seconds:.0f}s)",
                now=now,
            )

        # 3. Consecutive errors (handled in on_trade_result)
        return False

    def on_trade_result(self, *, success: bool) -> None:
        """Update the consecutive-error counter after a trade execution."""
        if success:
            self._consecutive_errors = 0
        else:
            self._consecutive_errors += 1
            if self._consecutive_errors >= self.max_consecutive_errors:
                self.trip(
                    f"{self._consecutive_errors} consecutive execution errors "
                    f"(threshold {self.max_consecutive_errors})",
                )

    def on_latency_breach(
        self,
        *,
        total_ms: float,
        threshold_ms: float,
        instrument_name: str = "",
    ) -> None:
        """Record a latency breach. Trips the kill switch on consecutive
        breaches above ``max_consecutive_errors`` (we treat latency
        breaches identically to execution errors — both signal venue
        degradation that warrants halting). Single isolated breaches
        WARN but don't trip — the venue can hiccup for a single order
        without indicating systemic failure.

        E.2.8 in plan ``starry-rolling-rabin.md``.
        """
        if not (math.isfinite(total_ms) and total_ms > threshold_ms):
            return
        # Treat as a soft error — feeds into the consecutive-error path.
        self._consecutive_errors += 1
        logger.warning(
            "Latency breach: %s took %.0fms (threshold %.0fms) — "
            "consecutive=%d/%d",
            instrument_name or "order", total_ms, threshold_ms,
            self._consecutive_errors, self.max_consecutive_errors,
        )
        if self._consecutive_errors >= self.max_consecutive_errors:
            self.trip(
                f"{self._consecutive_errors} consecutive latency breaches "
                f"(latest {total_ms:.0f}ms > {threshold_ms:.0f}ms)",
            )

    def trip(
        self,
        reason: str,
        *,
        now: pd.Timestamp | None = None,
    ) -> bool:
        """Manually trip the kill switch.  Returns True (for chaining)."""
        if self._state.halted:
            return True
        self._state = KillSwitchState(
            halted=True,
            reason=reason,
            halted_at=now or pd.Timestamp.now(tz="UTC"),
        )
        self.needs_emergency_close = True
        logger.error("KillSwitch TRIPPED: %s", reason)
        if self._alert_callback is not None:
            try:
                self._alert_callback(reason)
            except Exception:  # noqa: BLE001 - alert failure must not mask the trip
                logger.exception("KillSwitch alert callback raised")
        # H1: best-effort persist; failure must not mask the trip.
        if self._persist_callback is not None:
            try:
                self._persist_callback(True, reason)
            except Exception:  # noqa: BLE001
                logger.exception("KillSwitch persist callback raised on trip")
        return True

    def reset(self, *, operator: str = "manual") -> None:
        """Re-arm the kill switch after human review.

        This is DELIBERATELY manual — there is no auto-recovery.  The
        operator must confirm the trip cause has been addressed before
        resuming trading.
        """
        if not self._state.halted:
            return
        logger.warning(
            "KillSwitch reset by %s — previous halt reason: %s",
            operator, self._state.reason,
        )
        self._state = KillSwitchState(halted=False, reason=None, halted_at=None)
        self._consecutive_errors = 0
        # H1: persist the cleared state so the next restart starts clean.
        if self._persist_callback is not None:
            try:
                self._persist_callback(False, None)
            except Exception:  # noqa: BLE001
                logger.exception("KillSwitch persist callback raised on reset")
