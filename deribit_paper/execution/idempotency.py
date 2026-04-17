"""Client order ID generation and duplicate-send protection.

Network hiccups + reconnects can cause the same order to be submitted
twice.  Deribit's REST API is idempotent ONLY if the client attaches
a unique ``label`` to each order (Deribit treats label as a free-form
string; we use it to carry our client_order_id).

This module provides:

- :func:`new_client_order_id` — uniqueness via UUID4, always prefixed
  ``paper_`` so testnet fills are easy to spot in Deribit's order log.
- :class:`IdempotencyGuard` — bounded deque of recently-sent IDs.
  Callers check ``was_sent`` before submitting; the guard refuses
  duplicates for a configurable TTL.

The guard is in-memory only.  If the engine restarts mid-trade, we
rely on the SQLite store + Deribit-side reconciliation (Phase 6) to
prevent double-fills; the idempotency guard only protects against
intra-process retries.
"""

from __future__ import annotations

import logging
import time
import uuid
from collections import deque
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


_DEFAULT_TTL_SECONDS: float = 60.0
_DEFAULT_MAX_ENTRIES: int = 1000


def new_client_order_id(prefix: str = "paper") -> str:
    """Generate a unique client order ID with the given prefix."""
    return f"{prefix}_{uuid.uuid4().hex[:16]}"


@dataclass
class IdempotencyGuard:
    """Bounded recent-IDs set with TTL-based eviction.

    Thread-unsafe (asyncio single-threaded).  Check ``was_sent``
    before placing a new order and ``mark_sent`` after a successful
    submit.  Duplicates within the TTL window will be rejected by the
    caller (not by this class — the class just reports the state).
    """
    ttl_seconds: float = _DEFAULT_TTL_SECONDS
    max_entries: int = _DEFAULT_MAX_ENTRIES
    _recent: deque[tuple[str, float]] = field(default_factory=deque)
    _lookup: set[str] = field(default_factory=set)

    def was_sent(self, client_order_id: str) -> bool:
        """Return True if the ID was marked sent within the TTL."""
        self._evict()
        return client_order_id in self._lookup

    def mark_sent(self, client_order_id: str) -> None:
        """Record that we've submitted this client_order_id."""
        self._evict()
        if client_order_id in self._lookup:
            logger.warning(
                "IdempotencyGuard.mark_sent called twice for %s",
                client_order_id,
            )
            return
        now = time.monotonic()
        self._recent.append((client_order_id, now))
        self._lookup.add(client_order_id)
        # Hard cap — evict oldest if we exceed max_entries
        while len(self._recent) > self.max_entries:
            oid, _ = self._recent.popleft()
            self._lookup.discard(oid)

    def _evict(self) -> None:
        """Drop entries older than ttl_seconds."""
        now = time.monotonic()
        cutoff = now - self.ttl_seconds
        while self._recent and self._recent[0][1] < cutoff:
            oid, _ = self._recent.popleft()
            self._lookup.discard(oid)

    def clear(self) -> None:
        self._recent.clear()
        self._lookup.clear()
