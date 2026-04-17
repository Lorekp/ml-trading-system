"""Token-bucket rate limiter mirroring Deribit's credit system.

Deribit uses: 500 credits/request, 10,000 credits/sec refill, 50,000 pool.
This limiter tracks credits identically so we max throughput without 10028 errors.
"""

from __future__ import annotations

import threading
import time


class TokenBucketLimiter:
    """Thread-safe token bucket that mirrors Deribit's credit-based rate limiter."""

    def __init__(self, pool: int = 50_000, refill_per_sec: int = 10_000,
                 cost: int = 500):
        self.pool = pool
        self.max_pool = pool
        self.refill_per_sec = refill_per_sec
        self.cost = cost
        self._lock = threading.Lock()
        self._last_refill = time.monotonic()
        self._total_acquired = 0
        self._total_waits = 0

    def _refill(self) -> None:
        """Add credits based on elapsed time since last refill."""
        now = time.monotonic()
        elapsed = now - self._last_refill
        if elapsed > 0:
            added = elapsed * self.refill_per_sec
            self.pool = min(self.pool + added, self.max_pool)
            self._last_refill = now

    def acquire(self, cost: int | None = None) -> None:
        """Block until enough credits are available, then deduct."""
        cost = cost or self.cost
        while True:
            with self._lock:
                self._refill()
                if self.pool >= cost:
                    self.pool -= cost
                    self._total_acquired += 1
                    return

            # Not enough credits — sleep for the minimum time needed
            deficit = cost - self.pool
            wait = deficit / self.refill_per_sec + 0.001  # small buffer
            self._total_waits += 1
            time.sleep(wait)

    def drain(self) -> None:
        """Signal a rate limit hit — drain pool, pause, then resume at full rate.

        Per Deribit docs: 50ms restores ~500 credits (10 credits/ms).
        We drain to 0 and let the natural refill handle recovery rather than
        permanently degrading the refill rate.
        """
        with self._lock:
            self.pool = 0
            self._last_refill = time.monotonic()
            self._total_drains = getattr(self, '_total_drains', 0) + 1

    @property
    def stats(self) -> dict:
        with self._lock:
            self._refill()
            return {
                "credits_available": int(self.pool),
                "total_requests": self._total_acquired,
                "total_waits": self._total_waits,
            }
