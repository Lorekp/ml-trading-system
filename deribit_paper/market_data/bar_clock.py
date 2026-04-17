"""Async scheduler that fires callbacks at wall-clock bar boundaries.

Bars are aligned to UTC: for 15-minute bars they fire at :00, :15, :30,
:45 of every hour.  The clock sleeps until the next boundary, invokes
the registered callback, then sleeps again.

Design decisions:

- **Align to wall clock, not engine start time** — a bar timestamped
  ``12:15:00`` must always contain data from ``[12:00, 12:15)``
  regardless of when the engine was launched.  This is critical for
  matching the backtester's bar convention and for SVI cache
  consistency (SVI recalibrates at the top of each hour, so the bar
  clock MUST land on the hour precisely).
- **Use local monotonic clock for sleep, wall clock for alignment** —
  compute the target wall-clock time, then ``asyncio.sleep`` for the
  delta.  System clock jumps (NTP adjustment, manual changes) are
  tolerated; the next tick re-aligns.
- **Deterministic in tests** — accept a ``now_fn`` and a ``sleep_fn``
  so tests can drive the clock without real time.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone
from typing import Awaitable, Callable

import pandas as pd

logger = logging.getLogger(__name__)


BarCallback = Callable[[pd.Timestamp], Awaitable[None]]


class BarClock:
    """Async scheduler firing at wall-clock N-minute boundaries.

    Usage::

        clock = BarClock(bar_minutes=15)
        clock.add_callback(my_async_handler)
        asyncio.create_task(clock.run_forever())

    The callback is invoked with the **closing timestamp** of the bar
    that just completed.  For example, at wall-clock 12:15:00 UTC the
    callback fires with ``ts = pd.Timestamp("2026-04-10 12:15:00+00:00")``,
    meaning "trades in [12:00, 12:15) are now final, please aggregate".
    """

    def __init__(
        self,
        bar_minutes: int = 15,
        *,
        now_fn: Callable[[], datetime] | None = None,
        sleep_fn: Callable[[float], Awaitable[None]] | None = None,
    ):
        if bar_minutes <= 0 or 60 % bar_minutes != 0:
            raise ValueError(
                f"bar_minutes must divide 60 (got {bar_minutes})",
            )
        self.bar_minutes = bar_minutes
        self._callbacks: list[BarCallback] = []
        self._now_fn = now_fn or (lambda: datetime.now(timezone.utc))
        self._sleep_fn = sleep_fn or asyncio.sleep
        self._stop = asyncio.Event()

    def add_callback(self, callback: BarCallback) -> None:
        """Register an async callback to fire on each bar tick."""
        self._callbacks.append(callback)

    def stop(self) -> None:
        """Signal the ``run_forever`` loop to exit."""
        self._stop.set()

    def next_boundary(self, after: datetime | None = None) -> datetime:
        """Return the next bar boundary strictly after ``after`` (UTC).

        If ``after`` is exactly on a boundary, returns the next one.
        """
        now = after if after is not None else self._now_fn()
        # Snap down to the current bar's start, then add one bar
        bucket_minute = (now.minute // self.bar_minutes) * self.bar_minutes
        snap = now.replace(minute=bucket_minute, second=0, microsecond=0)
        from datetime import timedelta
        return snap + timedelta(minutes=self.bar_minutes)

    async def run_forever(self) -> None:
        """Run the clock loop until :meth:`stop` is called.

        Each iteration:
          1. Compute the next boundary.
          2. Sleep until it arrives (via the injected ``sleep_fn``).
          3. Invoke every registered callback with the bar-close timestamp.
          4. Repeat.

        The real-time production path uses ``asyncio.sleep``; tests
        inject a deterministic virtual-time sleep via the ``sleep_fn``
        parameter.  Either way, the sleep is uninterruptible within a
        single bar interval — shutdown via :meth:`stop` takes effect
        at the next boundary check.  In production, this bounds
        shutdown latency to one bar (15 min max); in tests, virtual
        sleep is instantaneous so latency is nil.
        """
        logger.info(
            "BarClock started: %d-min bars, %d callback(s)",
            self.bar_minutes, len(self._callbacks),
        )
        while not self._stop.is_set():
            target = self.next_boundary()
            now = self._now_fn()
            delta = (target - now).total_seconds()
            if delta > 0:
                await self._sleep_fn(delta)
            if self._stop.is_set():
                break

            # Fire callbacks with the bar-close timestamp.  ``target`` is
            # already tz-aware (UTC) from ``next_boundary``, so we pass
            # it to pandas without the ``tz=`` kwarg (which rejects
            # already-aware inputs on newer pandas).
            ts = pd.Timestamp(target)
            logger.debug("BarClock tick: %s", ts)
            for cb in self._callbacks:
                try:
                    await cb(ts)
                except Exception:  # noqa: BLE001 - log + continue, don't kill the clock
                    logger.exception(
                        "BarClock callback raised — continuing to next tick",
                    )
