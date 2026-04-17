"""Top-level straddle trade coordinator.

Given a :class:`Signal` emitted by the predictor and the current
equity, runs the full trade lifecycle:

1. **Size** the position using the backtester formula::

        N = equity * target_margin_util / (im_fraction * spot)
        N = min(N, max_participation * trailing_1h_volume)
        if signal < 0:
            N *= short_sizing_penalty

   Quantised to Deribit's option contract increment (0.1 BTC).

2. **Open** both call and put legs via the order manager.  Waits for
   both fills before proceeding.  If either leg fails to fill, the
   filled leg is closed and the trade is aborted.

3. **Begin** a live DRL hedge trade and **spawn** a :class:`PerpHedger`
   task that runs the intra-bar hedging loop in the background.

4. **Wait** for the holding period (default 15 minutes).

5. **Close** the straddle (sell calls, sell puts; mirror of entry).

6. **Cancel** the hedge loop if still running, flatten any residual
   perp position, and close the DRL trade.

7. Return a :class:`TradeOutcome` with fill details + hedge summary +
   realized PnL (approximate; exact PnL requires Greeks integration
   which the PositionTracker will handle in Phase 6).
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import numpy as np
import pandas as pd

from deribit_paper.execution.amounts import option_amount_btc
from deribit_paper.execution.order_manager import OrderManager, OrderResult
from deribit_paper.execution.perp_hedger import HedgeRunSummary, PerpHedger
from deribit_paper.features.straddle import StraddleSelection
from deribit_paper.signal.drl_hedger_live import LiveDRLHedger
from deribit_paper.signal.predictor import Signal

logger = logging.getLogger(__name__)


@dataclass
class TradeOutcome:
    """Full result of one straddle trade's lifecycle."""
    trade_id: str
    signal: Signal
    entry_call: OrderResult | None
    entry_put: OrderResult | None
    exit_call: OrderResult | None
    exit_put: OrderResult | None
    n_contracts_btc: float
    status: str  # "completed" | "aborted_entry" | "cancelled"
    hedge_summary: HedgeRunSummary | None = None
    error: str | None = None
    started_at: pd.Timestamp | None = None
    ended_at: pd.Timestamp | None = None


class StraddleFlow:
    """Orchestrates entry + hedging + exit for one straddle trade.

    Instances are created per-signal by the engine and discarded
    after completion.  They hold no long-running state — all
    persistent state (equity, positions, kill switch) lives in the
    engine-level services (Phase 6).
    """

    def __init__(
        self,
        *,
        order_manager: OrderManager,
        live_drl_hedger: LiveDRLHedger,
        perp_instrument: str = "BTC-PERPETUAL",
        holding_minutes: int = 15,
        target_margin_utilization: float = 0.05,  # mirrors BacktestConfig default
        im_fraction: float = 0.20,
        max_participation_rate: float = 0.10,
        short_sizing_penalty: float = 0.50,
        max_trade_size_btc: float = 2.0,
        min_trade_size_btc: float = 0.1,
        n_hedge_ticks: int = 10,
        spot_fetcher: Callable[[], Awaitable[float]] | None = None,
        vpin_provider: Callable[[], float] | None = None,
        maker_wait_seconds: float = 60.0,
        baseline_target_dvol: float = 50.0,   # DVOL at which full margin deployed
        sizing_edge_cap: float = 3.0,          # max edge multiplier (composite_v2)
        sizing_dd_floor: float = 0.25,         # min scale at deep drawdown (composite_v2)
        min_trade_floor: bool = False,         # return exchange min instead of 0.0
        on_position_opened: Callable[..., Awaitable[None]] | None = None,
        on_position_closed: Callable[..., Awaitable[None]] | None = None,
    ):
        self._order_manager = order_manager
        self._drl = live_drl_hedger
        self._perp_instrument = perp_instrument
        self._holding_minutes = holding_minutes
        self._target_margin_util = target_margin_utilization
        self._im_fraction = im_fraction
        self._max_participation = max_participation_rate
        self._short_penalty = short_sizing_penalty
        self._max_trade_size = max_trade_size_btc
        self._min_trade_size = min_trade_size_btc
        self._min_trade_floor = min_trade_floor
        self._on_position_opened = on_position_opened
        self._on_position_closed = on_position_closed
        self._n_hedge_ticks = n_hedge_ticks
        self._spot_fetcher = spot_fetcher
        self._vpin_provider = vpin_provider
        self._maker_wait_seconds = maker_wait_seconds
        self._baseline_target_dvol = baseline_target_dvol
        self._sizing_edge_cap = sizing_edge_cap
        self._sizing_dd_floor = sizing_dd_floor

    # ── Sizing ──────────────────────────────────────────────────────

    def size_position(
        self,
        *,
        equity: float,
        signal_direction: int,
        spot: float,
        trailing_1h_volume_btc: float | None = None,
        liq_scale: float = 1.0,
        dvol_pct: float | None = None,
        edge_vp: float = 0.0,
        rolling_median_edge: float = 1.0,
        peak_equity: float = 0.0,
    ) -> float:
        """Compute the BTC notional for a new straddle trade.

        Mirrors backtester's composite_v2 model:

        1. **DVOL-scaled base margin** — `target_margin * (baseline_dvol / dvol)`,
           clipped so we never exceed ``target_margin_utilization``.
        2. **Edge scale** — size up when this bar's edge is above the recent
           median, size down when below.  Bounded to [0.25, sizing_edge_cap].
        3. **Drawdown floor** — scale down proportionally when in drawdown,
           with a floor of ``sizing_dd_floor``.
        4. **Liquidity scale** (liq_scale, caller-supplied) — spread-EMA/current
           spread.  Already implemented by the caller via exec_cost_ema.

        Returns 0.0 if the computed size is below the minimum contract size.
        """
        if equity <= 0 or spot <= 0:
            return 0.0

        # 1. DVOL-scaled dynamic margin
        dvol = dvol_pct if (dvol_pct is not None and dvol_pct > 0) else self._baseline_target_dvol
        dynamic_margin = self._target_margin_util * (self._baseline_target_dvol / dvol)
        # Clip: never exceed the configured target; floor at 0.05 (backtester convention)
        dynamic_margin = max(0.05, min(dynamic_margin, self._target_margin_util))

        n = equity * dynamic_margin / (self._im_fraction * spot)

        # Participation cap (skipped if volume unknown)
        if trailing_1h_volume_btc is not None and trailing_1h_volume_btc > 0:
            cap = trailing_1h_volume_btc * self._max_participation
            n = min(n, cap)

        # Short penalty
        if signal_direction == -1:
            n *= self._short_penalty

        # 2. Edge scale: size up on high-conviction signals, down on weak ones
        abs_edge = abs(edge_vp)
        if rolling_median_edge > 0 and abs_edge > 0:
            edge_scale = float(np.clip(abs_edge / rolling_median_edge, 0.25, self._sizing_edge_cap))
        else:
            edge_scale = 1.0

        # 3. Drawdown floor: scale back proportionally when in drawdown
        if peak_equity > 0 and equity < peak_equity:
            dd = (equity - peak_equity) / peak_equity
            dd_scale = float(np.clip(1.0 + 2.0 * dd, self._sizing_dd_floor, 1.0))
        else:
            dd_scale = 1.0

        # 4. Liquidity scale (supplied by caller from spread EMA vs current spread)
        n *= edge_scale * dd_scale * liq_scale

        # Hard cap
        n = min(n, self._max_trade_size)

        # Quantise to option increment
        n_rounded = option_amount_btc(n)
        if n_rounded < self._min_trade_size:
            # At small equity the formula underproduces.  Return the
            # exchange minimum — the risk check still gates affordability.
            if equity > 0 and self._min_trade_floor:
                return self._min_trade_size
            return 0.0
        return n_rounded

    # ── Full lifecycle ──────────────────────────────────────────────

    async def execute(
        self,
        *,
        signal: Signal,
        equity: float,
        trailing_1h_volume_btc: float | None = None,
        exec_cost_ema: float = 12.0,
        sizing_liq_cap: float = 1.5,
        dvol_pct: float | None = None,
        edge_vp: float = 0.0,
        rolling_median_edge: float = 1.0,
        peak_equity: float = 0.0,
    ) -> TradeOutcome:
        """Run the full signal → entry → hedge → exit lifecycle.

        Safe under DRY_RUN: every order routes through the REST
        client which short-circuits to stub fills, and the hedger's
        order_manager uses the same path.

        ``exec_cost_ema`` is the rolling average execution cost ($/BTC).
        Used with the current spread to compute a liquidity scaling
        factor: trade less when spreads are wide, more when tight.
        """
        started_at = pd.Timestamp.now(tz="UTC")

        if not signal.is_actionable or signal.straddle is None:
            return TradeOutcome(
                trade_id="",
                signal=signal,
                entry_call=None,
                entry_put=None,
                exit_call=None,
                exit_put=None,
                n_contracts_btc=0.0,
                status="aborted_entry",
                error="signal not actionable",
                started_at=started_at,
            )

        straddle = signal.straddle
        spot = signal.straddle.index_price

        # ── Liquidity scaling from current spread ──────────────────
        liq_scale = 1.0
        try:
            if (self._order_manager
                    and hasattr(self._order_manager, '_rest')
                    and self._order_manager._rest is not None):
                ticker = await self._order_manager._rest.get_ticker(
                    straddle.call_instrument,
                )
                if isinstance(ticker, dict):
                    best_ask = float(ticker.get("best_ask_price", 0) or 0)
                    best_bid = float(ticker.get("best_bid_price", 0) or 0)
                    if best_ask > 0 and best_bid > 0:
                        spread_usd = (best_ask - best_bid) * spot
                        current_cost = max(spread_usd, 1.0)
                        if exec_cost_ema > 0:
                            liq_scale = max(0.25, min(
                                exec_cost_ema / current_cost, sizing_liq_cap,
                            ))
                        logger.info(
                            "Spread on %s: $%.1f (EMA: $%.1f, liq_scale: %.2f)",
                            straddle.call_instrument, current_cost,
                            exec_cost_ema, liq_scale,
                        )
        except Exception:
            logger.debug("Could not fetch spread for liq scaling", exc_info=True)

        n_contracts = self.size_position(
            equity=equity,
            signal_direction=signal.direction,
            spot=spot,
            trailing_1h_volume_btc=trailing_1h_volume_btc,
            liq_scale=liq_scale,
            dvol_pct=dvol_pct,
            edge_vp=edge_vp,
            rolling_median_edge=rolling_median_edge,
            peak_equity=peak_equity,
        )
        if n_contracts <= 0.0:
            return TradeOutcome(
                trade_id="",
                signal=signal,
                entry_call=None,
                entry_put=None,
                exit_call=None,
                exit_put=None,
                n_contracts_btc=0.0,
                status="aborted_entry",
                error="position size below minimum contract",
                started_at=started_at,
            )

        # ── 1. OPEN both legs ──────────────────────────────────────
        entry_side = "sell" if signal.direction == -1 else "buy"
        logger.info(
            "StraddleFlow opening %s %s/%s n=%.2fBTC @ strike=%.0f",
            "SHORT" if signal.direction == -1 else "LONG",
            straddle.call_instrument, straddle.put_instrument,
            n_contracts, straddle.strike,
        )

        entry_call, entry_put = await asyncio.gather(
            self._order_manager.place_ioc_taker(
                instrument_name=straddle.call_instrument,
                side=entry_side,
                amount=n_contracts,
                max_slippage_pct=0.5,
                label=f"straddle_entry_C_{started_at.value}",
            ),
            self._order_manager.place_ioc_taker(
                instrument_name=straddle.put_instrument,
                side=entry_side,
                amount=n_contracts,
                max_slippage_pct=0.5,
                label=f"straddle_entry_P_{started_at.value}",
            ),
        )

        if not entry_call.is_filled or not entry_put.is_filled:
            logger.error(
                "StraddleFlow entry failed: call=%s put=%s — aborting + "
                "closing any filled leg",
                entry_call.state, entry_put.state,
            )
            # Close any leg that DID fill to avoid an unhedged single-leg position
            await self._close_if_filled(entry_call, straddle.call_instrument, entry_side, n_contracts)
            await self._close_if_filled(entry_put, straddle.put_instrument, entry_side, n_contracts)
            return TradeOutcome(
                trade_id="",
                signal=signal,
                entry_call=entry_call,
                entry_put=entry_put,
                exit_call=None,
                exit_put=None,
                n_contracts_btc=n_contracts,
                status="aborted_entry",
                error=f"entry leg not filled: call={entry_call.state} put={entry_put.state}",
                started_at=started_at,
                ended_at=pd.Timestamp.now(tz="UTC"),
            )

        # ── 2. BEGIN DRL hedge trade ───────────────────────────────
        tau_entry_years = max(
            (straddle.expiry_dt - started_at).total_seconds() / (365.25 * 24 * 3600),
            1e-6,
        )
        trade_id = self._drl.begin_trade(
            signal=signal.direction,
            S_entry=spot,
            K=straddle.strike,
            vol_ann_entry=float(signal.bar_tsrv.vol_ann_continuous) if signal.bar_tsrv else 0.8,
            iv_entry=float(signal.iv_used_pct) / 100.0 if signal.iv_used_pct else 0.75,
            tau_entry=tau_entry_years,
            V_bg=float(trailing_1h_volume_btc) if trailing_1h_volume_btc else 100.0,
            jump_fraction=float(signal.bar_tsrv.jump_fraction) if signal.bar_tsrv else 0.0,
            vpin=self._vpin_provider() if self._vpin_provider else 0.5,
        )

        # ── 2b. Notify position opened (persist to DB + tracker) ──
        if self._on_position_opened is not None:
            try:
                await self._on_position_opened(
                    trade_id=trade_id,
                    call_instrument=straddle.call_instrument,
                    put_instrument=straddle.put_instrument,
                    signal_direction=signal.direction,
                    n_contracts_btc=n_contracts,
                    strike=straddle.strike,
                    expiry_dt=str(straddle.expiry_dt),
                    opened_at=str(started_at),
                    entry_call_price=entry_call.average_price,
                    entry_put_price=entry_put.average_price,
                )
            except Exception:
                logger.exception("on_position_opened callback failed")

        # ── 3. Spawn hedge loop ────────────────────────────────────
        hedger = PerpHedger(
            order_manager=self._order_manager,
            live_drl_hedger=self._drl,
            perp_instrument=self._perp_instrument,
            bar_minutes=self._holding_minutes,
            n_ticks_per_bar=self._n_hedge_ticks,
            spot_fetcher=self._spot_fetcher,
        )
        hedge_task = asyncio.create_task(
            hedger.run(trade_id, entry_time=started_at),
        )

        # ── 4. Wait holding period ─────────────────────────────────
        holding_seconds = self._holding_minutes * 60.0
        try:
            await asyncio.sleep(holding_seconds)
        except asyncio.CancelledError:
            # M2: On cancellation (shutdown/kill switch), close straddle legs
            # before re-raising to avoid leaving naked positions.
            logger.warning(
                "StraddleFlow.execute cancelled during holding period — "
                "closing straddle legs before propagating cancellation"
            )
            hedger.cancel()
            exit_side = "buy" if signal.direction == -1 else "sell"
            try:
                await asyncio.gather(
                    self._order_manager.place_and_wait(
                        instrument_name=straddle.call_instrument,
                        side=exit_side, amount=n_contracts,
                        order_type="market",
                        label=f"cancel_exit_C_{started_at.value}",
                        timeout_seconds=15.0, reduce_only=True,
                    ),
                    self._order_manager.place_and_wait(
                        instrument_name=straddle.put_instrument,
                        side=exit_side, amount=n_contracts,
                        order_type="market",
                        label=f"cancel_exit_P_{started_at.value}",
                        timeout_seconds=15.0, reduce_only=True,
                    ),
                )
                logger.info("Cancellation exit orders sent for %s", trade_id)
            except Exception as close_exc:
                logger.error(
                    "Failed to close legs on cancellation: %s", close_exc,
                )
            self._drl.close_trade(trade_id)
            if self._on_position_closed is not None:
                try:
                    await self._on_position_closed(trade_id=trade_id)
                except Exception:
                    logger.exception("on_position_closed callback failed during cancel")
            raise

        # Stop hedge loop and collect summary
        hedger.cancel()
        try:
            hedge_summary = await asyncio.wait_for(hedge_task, timeout=30.0)
        except asyncio.TimeoutError:
            logger.warning("Hedge task did not finish within 30s after cancel")
            hedge_summary = None

        # ── 5. CLOSE both legs ─────────────────────────────────────
        exit_side = "buy" if signal.direction == -1 else "sell"
        exit_call, exit_put = await asyncio.gather(
            self._order_manager.place_ioc_taker(
                instrument_name=straddle.call_instrument,
                side=exit_side,
                amount=n_contracts,
                max_slippage_pct=0.5,
                label=f"straddle_exit_C_{started_at.value}",
                reduce_only=True,
            ),
            self._order_manager.place_ioc_taker(
                instrument_name=straddle.put_instrument,
                side=exit_side,
                amount=n_contracts,
                max_slippage_pct=0.5,
                label=f"straddle_exit_P_{started_at.value}",
                reduce_only=True,
            ),
        )

        # ── 6. Close DRL trade state + notify position closed ──────
        self._drl.close_trade(trade_id)

        if self._on_position_closed is not None:
            try:
                await self._on_position_closed(trade_id=trade_id)
            except Exception:
                logger.exception("on_position_closed callback failed")

        ended_at = pd.Timestamp.now(tz="UTC")
        logger.info(
            "StraddleFlow trade %s completed: entry=%s/%s exit=%s/%s hedge_ticks=%d",
            trade_id,
            entry_call.state, entry_put.state,
            exit_call.state, exit_put.state,
            hedge_summary.n_ticks_executed if hedge_summary else 0,
        )

        return TradeOutcome(
            trade_id=trade_id,
            signal=signal,
            entry_call=entry_call,
            entry_put=entry_put,
            exit_call=exit_call,
            exit_put=exit_put,
            n_contracts_btc=n_contracts,
            status="completed",
            hedge_summary=hedge_summary,
            started_at=started_at,
            ended_at=ended_at,
        )

    async def _close_if_filled(
        self,
        order: OrderResult,
        instrument: str,
        entry_side: str,
        n_contracts: float,
        *,
        max_retries: int = 3,
        alert_callback: Callable[[str], None] | None = None,
    ) -> None:
        """If an entry leg was partially/fully filled, close it with retry.

        C5: Retries up to ``max_retries`` times with 2s backoff. If all
        retries fail, logs at CRITICAL and sends alert so the operator
        knows a naked single-leg position exists on the exchange.
        """
        if not order.is_filled or order.filled_amount <= 0.0:
            return
        close_side = "buy" if entry_side == "sell" else "sell"
        logger.warning(
            "Closing unilateral filled leg %s (%.2f BTC) due to other-leg failure",
            instrument, order.filled_amount,
        )
        for attempt in range(1, max_retries + 1):
            try:
                result = await self._order_manager.place_and_wait(
                    instrument_name=instrument,
                    side=close_side,
                    amount=order.filled_amount,
                    order_type="market",
                    label=f"unilateral_close_{instrument}_{attempt}",
                    timeout_seconds=15.0,
                    reduce_only=True,
                )
                if result.is_filled:
                    logger.info(
                        "Unilateral close succeeded on attempt %d: %s %.4f @ %.4f",
                        attempt, instrument, result.filled_amount, result.average_price,
                    )
                    return
                logger.warning(
                    "Unilateral close attempt %d not filled: %s state=%s",
                    attempt, instrument, result.state,
                )
            except Exception as exc:
                logger.error(
                    "Unilateral close attempt %d failed for %s: %s",
                    attempt, instrument, exc,
                )
            if attempt < max_retries:
                await asyncio.sleep(2.0)

        critical_msg = (
            f"CRITICAL: Failed to close unilateral position {instrument} "
            f"({order.filled_amount:.2f} BTC {close_side}) after {max_retries} "
            f"attempts — NAKED SINGLE-LEG POSITION ON EXCHANGE"
        )
        logger.critical(critical_msg)
        if alert_callback is not None:
            alert_callback(critical_msg)
