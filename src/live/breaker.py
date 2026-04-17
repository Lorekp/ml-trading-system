"""F1 symmetric circuit breaker — per-asset PnL-weighted sizing.

Exact port of simulate_with_breaker():356-376 from circuit_breaker_v2_experiment.py.

Tracks rolling PnL/notional per asset over the last N trades.
- If ratio < threshold (-1%): scale DOWN sizing (floor at 0.10x)
- If ratio > up_threshold (+1%): scale UP sizing (cap at 1.5x)
- Otherwise: 1.0x neutral
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict, deque
from typing import Optional

logger = logging.getLogger(__name__)


class F1SymmetricBreaker:
    """F1 symmetric circuit breaker.

    Same math as circuit_breaker_v2_experiment.py lines 356-376.
    """

    def __init__(
        self,
        window: int = 10,
        threshold: float = -0.01,
        floor: float = 0.10,
        up_threshold: float = 0.01,
        cap: float = 1.5,
    ) -> None:
        self._window = window
        self._threshold = threshold
        self._floor = floor
        self._up_threshold = up_threshold
        self._cap = cap
        self._recent_pnl: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window))
        self._recent_notional: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window))

    @classmethod
    def from_config(cls, config) -> F1SymmetricBreaker:
        return cls(
            window=config.window,
            threshold=config.threshold,
            floor=config.floor,
            up_threshold=config.up_threshold,
            cap=config.cap,
        )

    def get_multiplier(self, asset: str) -> float:
        """Compute sizing multiplier for an asset.

        Returns:
            Float in [floor, cap] range. 1.0 = neutral.
        """
        recent_pnl = self._recent_pnl[asset]
        recent_not = self._recent_notional[asset]

        if len(recent_pnl) < self._window:
            return 1.0

        total_pnl = sum(recent_pnl)
        total_not = sum(recent_not)

        if total_not <= 0:
            return 1.0

        pnl_ratio = total_pnl / total_not

        # Scale DOWN (exact port of lines 364-368)
        if pnl_ratio < self._threshold:
            t = self._threshold
            scale = max(1.0 + pnl_ratio / abs(t), self._floor) if t != 0 else 1.0
            return scale

        # Scale UP (exact port of lines 370-376)
        if pnl_ratio > self._up_threshold:
            up_t = self._up_threshold
            scale = min(1.0 + (pnl_ratio - up_t) / up_t, self._cap)
            return scale

        return 1.0

    def record_trade(self, asset: str, pnl: float, notional: float) -> None:
        """Record a completed trade for breaker tracking."""
        self._recent_pnl[asset].append(pnl)
        self._recent_notional[asset].append(notional)

    def get_state(self) -> dict:
        """Serialize breaker state for persistence."""
        return {
            asset: {
                "pnl": list(self._recent_pnl[asset]),
                "notional": list(self._recent_notional[asset]),
            }
            for asset in self._recent_pnl
        }

    def restore_state(self, state: dict) -> None:
        """Restore breaker state from persistence."""
        for asset, data in state.items():
            for pnl in data.get("pnl", []):
                self._recent_pnl[asset].append(pnl)
            for notional in data.get("notional", []):
                self._recent_notional[asset].append(notional)
        logger.info("Restored breaker state for %d assets", len(state))
