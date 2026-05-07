"""Running equity curve + drawdown tracking.

Maintains the engine's notion of "current equity" across trade
closes.  Each realized trade PnL is added via :meth:`on_trade_closed`,
and derived quantities (peak, drawdown, daily return) are exposed
as properties for the risk layer and the telegram heartbeat.

Margin utilisation is computed from the currently-open positions
(supplied by the :class:`PositionTracker`) against the running equity.

Convention: equity is tracked in **USD** (not BTC) so the same tracker
works for options + perps without currency conversion headaches.  The
Deribit testnet account reports equity in BTC but this is converted
at startup via the current index price.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class EquitySnapshot:
    """One point on the equity curve."""
    timestamp: pd.Timestamp
    equity_usd: float
    realized_pnl_usd: float   # cumulative realized PnL since inception
    peak_equity_usd: float    # running peak
    drawdown_pct: float       # (equity - peak) / peak, negative or zero
    n_trades_closed: int


class EquityTracker:
    """Stateful equity curve maintained across the engine's lifetime.

    Usage::

        tracker = EquityTracker(initial_equity_usd=100_000.0)
        tracker.on_trade_closed(pnl_usd=1500.0, timestamp=pd.Timestamp.utcnow())
        print(tracker.current_equity)       # 101500.0
        print(tracker.drawdown_pct)         # 0.0 (new high)

    Thread-safe under asyncio (single-threaded event loop).
    """

    def __init__(
        self,
        *,
        initial_equity_usd: float,
        max_snapshots: int = 10_000,
    ):
        if initial_equity_usd <= 0:
            raise ValueError(f"initial_equity_usd must be positive, got {initial_equity_usd}")
        self._initial = float(initial_equity_usd)
        self._current = float(initial_equity_usd)
        self._peak = float(initial_equity_usd)
        self._realized_pnl = 0.0
        self._n_trades = 0
        self._snapshots: deque[EquitySnapshot] = deque(maxlen=max_snapshots)
        # Seed the curve with the initial point
        self._snapshots.append(EquitySnapshot(
            timestamp=pd.Timestamp.now(tz="UTC"),
            equity_usd=initial_equity_usd,
            realized_pnl_usd=0.0,
            peak_equity_usd=initial_equity_usd,
            drawdown_pct=0.0,
            n_trades_closed=0,
        ))

    @property
    def initial_equity(self) -> float:
        return self._initial

    @property
    def current_equity(self) -> float:
        return self._current

    @property
    def peak_equity(self) -> float:
        return self._peak

    @property
    def realized_pnl(self) -> float:
        return self._realized_pnl

    @property
    def total_return_pct(self) -> float:
        """(current / initial) - 1, as a percentage."""
        return (self._current / self._initial - 1.0) * 100.0

    @property
    def drawdown_pct(self) -> float:
        """Current drawdown from peak, as a negative percentage (or 0.0 at peak)."""
        if self._peak <= 0:
            return 0.0
        return (self._current / self._peak - 1.0) * 100.0

    @property
    def n_trades_closed(self) -> int:
        return self._n_trades

    def on_trade_closed(
        self,
        pnl_usd: float,
        *,
        timestamp: pd.Timestamp | None = None,
    ) -> EquitySnapshot:
        """Apply a closed trade's realized PnL to the running equity.

        Returns the new :class:`EquitySnapshot`.
        """
        self._realized_pnl += pnl_usd
        self._current += pnl_usd
        if self._current > self._peak:
            self._peak = self._current
        self._n_trades += 1

        snapshot = EquitySnapshot(
            timestamp=timestamp or pd.Timestamp.now(tz="UTC"),
            equity_usd=self._current,
            realized_pnl_usd=self._realized_pnl,
            peak_equity_usd=self._peak,
            drawdown_pct=self.drawdown_pct,
            n_trades_closed=self._n_trades,
        )
        self._snapshots.append(snapshot)
        logger.info(
            "EquityTracker trade closed: pnl=%+.2f → equity=%.2f (peak=%.2f, dd=%.2f%%)",
            pnl_usd, self._current, self._peak, self.drawdown_pct,
        )
        return snapshot

    def set_equity(self, new_equity_usd: float, *, reason: str = "reconcile") -> None:
        """Force-set the current equity (used for Deribit reconciliation on startup).

        Records a snapshot but does NOT increment the trade counter or
        the realized PnL total — this path exists specifically for
        non-trade equity updates (deposits, withdrawals, restart
        reconciliation against the exchange's reported balance).
        """
        if new_equity_usd < 0:
            raise ValueError(f"new_equity must be non-negative, got {new_equity_usd}")
        delta = new_equity_usd - self._current
        self._current = float(new_equity_usd)
        if self._current > self._peak:
            self._peak = self._current
        self._snapshots.append(EquitySnapshot(
            timestamp=pd.Timestamp.now(tz="UTC"),
            equity_usd=self._current,
            realized_pnl_usd=self._realized_pnl,
            peak_equity_usd=self._peak,
            drawdown_pct=self.drawdown_pct,
            n_trades_closed=self._n_trades,
        ))
        logger.info(
            "EquityTracker equity set to %.2f (delta %+.2f, reason: %s)",
            self._current, delta, reason,
        )

    def snapshots(self) -> Iterable[EquitySnapshot]:
        """Iterate over the in-memory snapshot buffer (oldest → newest)."""
        return list(self._snapshots)

    def daily_return_pct(self, now: pd.Timestamp | None = None) -> float:
        """Return the PnL since 00:00 UTC today, as a percent of start-of-day equity."""
        now = now or pd.Timestamp.now(tz="UTC")
        start_of_day = pd.Timestamp(now.date(), tz="UTC")
        # Find the snapshot at or immediately before start_of_day
        start_equity = self._initial
        for snap in self._snapshots:
            if snap.timestamp <= start_of_day:
                start_equity = snap.equity_usd
            else:
                break
        if start_equity <= 0:
            return 0.0
        return (self._current / start_equity - 1.0) * 100.0

    def dropped_below_threshold(self, dd_pct_threshold: float) -> bool:
        """Return True if current drawdown is worse than ``-dd_pct_threshold``."""
        return self.drawdown_pct <= -abs(dd_pct_threshold)

    # ── State persistence (S1 — parity-cycle round 2, 2026-05-07) ──

    def state_dict(self) -> dict:
        """Snapshot of fields needed to resume an EquityTracker after restart.

        Pre-S1 every restart re-seeded ``current_equity`` at
        ``cfg.paper_initial_equity_usd`` and ``peak_equity`` at the same,
        silently re-baselining the kill-switch daily-DD reference and
        producing different sizing decisions vs a continuous run.

        ``initial_equity`` is NOT persisted — that's a constructor-time
        choice; resuming with the same constructor + restored state is
        what gives parity. The in-memory snapshot deque is also not
        persisted (DB has the canonical equity_snapshots table).
        """
        return {
            "current_equity_usd": self._current,
            "peak_equity_usd": self._peak,
            "realized_pnl_usd": self._realized_pnl,
            "n_trades_closed": self._n_trades,
        }

    def load_state(self, state: dict) -> None:
        """Restore from a ``state_dict()``. Replaces current/peak/realized
        and trade count; the snapshot deque keeps only the seed point
        until new trades fire.
        """
        self._current = float(state["current_equity_usd"])
        self._peak = float(state["peak_equity_usd"])
        self._realized_pnl = float(state["realized_pnl_usd"])
        self._n_trades = int(state["n_trades_closed"])
        # Rewrite the seed snapshot to reflect the restored state so the
        # `daily_return_pct` "start of day" lookup doesn't return a
        # stale initial equity.
        self._snapshots.clear()
        self._snapshots.append(EquitySnapshot(
            timestamp=pd.Timestamp.now(tz="UTC"),
            equity_usd=self._current,
            realized_pnl_usd=self._realized_pnl,
            peak_equity_usd=self._peak,
            drawdown_pct=self.drawdown_pct,
            n_trades_closed=self._n_trades,
        ))
        logger.info(
            "EquityTracker restored from state: equity=%.2f peak=%.2f "
            "realized_pnl=%+.2f n_trades=%d",
            self._current, self._peak, self._realized_pnl, self._n_trades,
        )
