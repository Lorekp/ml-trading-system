"""Intra-bar perp hedger — executes DRL policy actions as Deribit perp orders.

The backtester uses 10 hedge ticks per 15-minute bar.  This live
counterpart runs an async loop at the same cadence:

- Every ``bar_minutes / n_steps`` seconds (default 90s), fetch the
  current perp spot price
- Call :meth:`LiveDRLHedger.step` to get the target inventory
- Compute the delta vs the current tracked perp position
- If the delta exceeds a tiny dust threshold, submit a market order
  to Deribit for the delta (USD notional, not BTC)
- Update the tracked position from the fill

The hedger tracks its own shadow position in BTC for the delta
calculation.  Reconciliation with Deribit's reported position happens
at trade close + on engine restart (Phase 6 StateDB).

Cancellation: the hedge loop is scheduled via
``asyncio.create_task(hedger.run(...))`` and can be cancelled by the
straddle flow at any time (holding-period expiry, kill switch, error).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from deribit_paper.execution.amounts import perp_amount_usd, side_from_signed
from deribit_paper.execution.order_manager import OrderManager, OrderResult
from deribit_paper.signal.drl_hedger_live import LiveDRLHedger

logger = logging.getLogger(__name__)


# Minimum delta (BTC) that will trigger a hedge order.  Anything
# smaller is considered dust and skipped to avoid paying fees on
# trivial adjustments.
_DUST_THRESHOLD_BTC: float = 0.01


@dataclass
class HedgeTickResult:
    """One hedge tick's worth of diagnostics (for logging + metrics)."""
    tick: int
    spot: float
    target_inventory_btc: float
    current_inventory_btc: float
    delta_btc: float
    usd_notional: float
    side: str | None            # "buy"/"sell" or None if skipped
    order_result: OrderResult | None
    reason: str


@dataclass
class HedgeRunSummary:
    """Aggregate stats for a whole hedge run (one trade's lifetime)."""
    trade_id: str
    n_ticks_executed: int
    n_ticks_skipped_dust: int
    total_usd_traded: float
    final_inventory_btc: float
    per_tick: list[HedgeTickResult] = field(default_factory=list)


class PerpHedger:
    """Runs the intra-bar hedging loop for a single open straddle trade.

    One ``PerpHedger`` instance per active trade.  Owns its own
    shadow-position tracker and tick counter; the :class:`LiveDRLHedger`
    (shared across trades) holds the per-trade policy state.
    """

    def __init__(
        self,
        *,
        order_manager: OrderManager,
        live_drl_hedger: LiveDRLHedger,
        perp_instrument: str = "BTC-PERPETUAL",
        bar_minutes: int = 15,
        n_ticks_per_bar: int = 10,
        dust_threshold_btc: float = _DUST_THRESHOLD_BTC,
        spot_fetcher=None,                  # async callable returning current spot
        clock_sleep_fn=None,                # injectable for tests
    ):
        self._order_manager = order_manager
        self._drl = live_drl_hedger
        self._perp_instrument = perp_instrument
        self._bar_minutes = bar_minutes
        self._n_ticks = n_ticks_per_bar
        self._tick_interval_s = (bar_minutes * 60.0) / n_ticks_per_bar
        self._dust_threshold = dust_threshold_btc
        self._spot_fetcher = spot_fetcher
        self._sleep_fn = clock_sleep_fn or asyncio.sleep
        self._cancelled = asyncio.Event()

    def cancel(self) -> None:
        """Signal the run loop to exit at the next tick boundary."""
        self._cancelled.set()

    async def run(
        self,
        trade_id: str,
        *,
        entry_time: pd.Timestamp,
    ) -> HedgeRunSummary:
        """Run the hedge loop for one trade's holding period.

        Fires N intra-bar ticks spaced ``tick_interval_s`` seconds
        apart.  Each tick:
          1. Fetch current spot via ``spot_fetcher`` (or use index price)
          2. Ask LiveDRLHedger.step() for target inventory
          3. Compute delta vs tracked position
          4. If |delta| > dust threshold, submit a perp market order
          5. Update tracked inventory on fill

        Returns a :class:`HedgeRunSummary` on completion (normal or cancel).
        """
        if self._spot_fetcher is None:
            raise RuntimeError("PerpHedger.run requires a spot_fetcher callable")

        summary = HedgeRunSummary(
            trade_id=trade_id,
            n_ticks_executed=0,
            n_ticks_skipped_dust=0,
            total_usd_traded=0.0,
            final_inventory_btc=0.0,
        )
        tracked_inventory_btc = 0.0

        for tick in range(1, self._n_ticks + 1):
            # Wait until this tick's wall-clock time
            await self._sleep_fn(self._tick_interval_s)
            if self._cancelled.is_set():
                logger.info(
                    "PerpHedger %s cancelled at tick %d/%d",
                    trade_id, tick, self._n_ticks,
                )
                break

            try:
                spot = await self._spot_fetcher()
            except Exception as exc:  # noqa: BLE001 - log + keep going
                logger.exception(
                    "PerpHedger %s spot fetch failed at tick %d: %s",
                    trade_id, tick, exc,
                )
                continue

            try:
                target_inventory = self._drl.step(trade_id, current_spot=spot)
            except Exception as exc:  # noqa: BLE001 - log + keep going
                logger.exception(
                    "PerpHedger %s DRL step failed at tick %d: %s",
                    trade_id, tick, exc,
                )
                continue

            delta_btc = target_inventory - tracked_inventory_btc
            if abs(delta_btc) < self._dust_threshold:
                summary.per_tick.append(HedgeTickResult(
                    tick=tick,
                    spot=spot,
                    target_inventory_btc=target_inventory,
                    current_inventory_btc=tracked_inventory_btc,
                    delta_btc=delta_btc,
                    usd_notional=0.0,
                    side=None,
                    order_result=None,
                    reason=f"dust ({abs(delta_btc):.4f} < {self._dust_threshold:.4f})",
                ))
                summary.n_ticks_skipped_dust += 1
                continue

            usd_notional = perp_amount_usd(delta_btc, spot)
            if usd_notional <= 0.0:
                summary.per_tick.append(HedgeTickResult(
                    tick=tick,
                    spot=spot,
                    target_inventory_btc=target_inventory,
                    current_inventory_btc=tracked_inventory_btc,
                    delta_btc=delta_btc,
                    usd_notional=0.0,
                    side=None,
                    order_result=None,
                    reason="below min perp notional",
                ))
                summary.n_ticks_skipped_dust += 1
                continue

            side = side_from_signed(delta_btc)
            result = await self._order_manager.place_and_wait(
                instrument_name=self._perp_instrument,
                side=side,
                amount=usd_notional,
                order_type="market",
                label=f"hedge_{trade_id[:8]}_t{tick:02d}",
                timeout_seconds=10.0,
            )

            if result.is_filled:
                # Update tracked inventory — for perps, filled_amount is in USD
                # notional, convert back to BTC via the fill price
                fill_price = result.average_price if result.average_price > 0 else spot
                filled_btc = result.filled_amount / fill_price
                if side == "sell":
                    filled_btc = -filled_btc
                tracked_inventory_btc += filled_btc
                summary.total_usd_traded += result.filled_amount
                summary.n_ticks_executed += 1
                reason = "filled"
            else:
                reason = f"not_filled state={result.state} err={result.error}"
                logger.warning(
                    "PerpHedger %s tick %d order not filled: %s",
                    trade_id, tick, reason,
                )

            summary.per_tick.append(HedgeTickResult(
                tick=tick,
                spot=spot,
                target_inventory_btc=target_inventory,
                current_inventory_btc=tracked_inventory_btc,
                delta_btc=delta_btc,
                usd_notional=usd_notional,
                side=side,
                order_result=result,
                reason=reason,
            ))

        summary.final_inventory_btc = tracked_inventory_btc
        return summary
