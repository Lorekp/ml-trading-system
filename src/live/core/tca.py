"""Transaction Cost Analysis — per-order execution quality recording.

Inspired by HftBacktest's dual-timestamp architecture and Recorder pattern.
Tracks decision→submit→fill→complete timestamps, costs in bps, and fill quality.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class TCARecord:
    """Per-order execution quality record."""

    # Identity
    timestamp: Optional[datetime] = None
    symbol: str = ""
    side: str = ""
    strategy_id: str = ""

    # Dual timestamps (HftBacktest pattern)
    decision_ts: Optional[datetime] = None   # signal generated
    submit_ts: Optional[datetime] = None     # order sent to exchange
    first_fill_ts: Optional[datetime] = None # first partial/full fill
    complete_ts: Optional[datetime] = None   # execution finished

    # Prices
    decision_mid: float = 0.0    # (bid+ask)/2 at signal time
    arrival_mid: float = 0.0     # (bid+ask)/2 at order submit time
    fill_price: float = 0.0      # actual fill price (blended if partial)

    # Costs (bps) — computed by compute_costs()
    impl_shortfall_bps: float = 0.0  # (fill - decision_mid) / decision_mid × 10000
    arrival_cost_bps: float = 0.0    # (fill - arrival_mid) / arrival_mid × 10000
    spread_cost_bps: float = 0.0     # half-spread at decision time
    timing_cost_bps: float = 0.0     # (arrival_mid - decision_mid) / decision_mid × 10000

    # Fill quality
    method: str = ""             # "maker", "taker", "blended"
    maker_fill_rate: float = 0.0 # fraction filled as maker [0, 1]
    total_qty: float = 0.0
    maker_qty: float = 0.0
    taker_qty: float = 0.0

    # Latency (ms)
    decision_to_submit_ms: float = 0.0
    submit_to_first_fill_ms: float = 0.0
    submit_to_complete_ms: float = 0.0

    def compute_costs(self) -> None:
        """Compute cost metrics from timestamps and prices.

        Call after all price fields are populated.
        Direction-aware: for buys, higher fill = worse;
        for sells, lower fill = worse.
        """
        sign = 1.0 if self.side == "Buy" else -1.0

        if self.decision_mid > 0:
            self.impl_shortfall_bps = (
                sign * (self.fill_price - self.decision_mid)
                / self.decision_mid * 10000
            )
            self.timing_cost_bps = (
                sign * (self.arrival_mid - self.decision_mid)
                / self.decision_mid * 10000
            )

        if self.arrival_mid > 0:
            self.arrival_cost_bps = (
                sign * (self.fill_price - self.arrival_mid)
                / self.arrival_mid * 10000
            )

    def compute_latencies(self) -> None:
        """Compute latency metrics from timestamps."""
        if self.decision_ts and self.submit_ts:
            self.decision_to_submit_ms = (
                (self.submit_ts - self.decision_ts).total_seconds() * 1000
            )
        if self.submit_ts and self.first_fill_ts:
            self.submit_to_first_fill_ms = (
                (self.first_fill_ts - self.submit_ts).total_seconds() * 1000
            )
        if self.submit_ts and self.complete_ts:
            self.submit_to_complete_ms = (
                (self.complete_ts - self.submit_ts).total_seconds() * 1000
            )

    def compute_fill_quality(self) -> None:
        """Compute maker fill rate from qty fields."""
        if self.total_qty > 0:
            self.maker_fill_rate = self.maker_qty / self.total_qty
        if self.method == "maker":
            self.maker_qty = self.total_qty
            self.taker_qty = 0.0
            self.maker_fill_rate = 1.0
        elif self.method == "taker":
            self.maker_qty = 0.0
            self.taker_qty = self.total_qty
            self.maker_fill_rate = 0.0

    def finalize(self) -> None:
        """Compute all derived fields. Call once before persisting."""
        self.timestamp = self.timestamp or datetime.utcnow()
        self.compute_costs()
        self.compute_latencies()
        self.compute_fill_quality()
