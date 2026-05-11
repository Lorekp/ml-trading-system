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
import math
import secrets
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable

import numpy as np
import pandas as pd

from deribit_paper.execution.amounts import option_amount_btc
from deribit_paper.execution.order_manager import OrderManager, OrderResult
from deribit_paper.execution.perp_hedger import HedgeRunSummary, PerpHedger
from deribit_paper.execution.profit_scale import bsm_vega_per_vol_pt
from deribit_paper.features.straddle import StraddleSelection
from deribit_paper.signal.drl_hedger_live import LiveDRLHedger
from deribit_paper.signal.predictor import Signal

logger = logging.getLogger(__name__)

# E.3.3 (2026-05-01): single source of truth for the "is this scale
# meaningfully > 1.0 / != 0.0" epsilon used across the sizing chain.
# Previously: 1e-9 in some clamps, 1e-12 in others, 1e-6 in slip helper.
# Unifying at 1e-9 is enough to filter floating-point comparison noise
# without rejecting legitimately-tight sizing values.
_SCALE_EPS: float = 1e-9


@dataclass
class TradeOutcome:
    """Full result of one straddle trade's lifecycle.

    ``maker_adverse_selection_charge_usd`` (E-7, parity-r3 cycle 3 —
    2026-05-08): USD penalty applied to legs that filled as maker.
    Mirrors the offline backtester at ``vol_arb/backtester.py:2256-2259``
    which charges ``vega_s × cfg.maker_adverse_selection_iv`` on
    ``maker_filled_arr[i] == True`` trades. Live now charges
    proportionally based on the per-leg maker fill fraction. Engine's
    PnL accounting deducts this from ``realized_pnl_usd``.
    """
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
    maker_adverse_selection_charge_usd: float = 0.0


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
        delta_hedge_band: float = 0.01,         # E2 (parity-cycle round 2): wired through to PerpHedger
        spot_fetcher: Callable[[], Awaitable[float]] | None = None,
        vpin_provider: Callable[[], float] | None = None,
        maker_wait_seconds: float = 60.0,
        baseline_target_dvol: float = 50.0,   # DVOL at which full margin deployed
        sizing_edge_cap: float = 3.0,          # max edge multiplier (composite_v2)
        sizing_dd_floor: float = 0.25,         # min scale at deep drawdown (composite_v2)
        min_trade_floor: bool = False,         # return exchange min instead of 0.0
        use_tiered_taker: bool = False,        # E.1.6: 3-tier IOC ladder
        cancel_exit_max_slippage_pct: float = 1.0,  # E-8 (2026-05-08): cap for recovery exits
        # E-7 (parity-r3 cycle 3 — 2026-05-08): vol-points penalty charged
        # per BTC of maker-filled leg; mirrors offline `cfg.maker_adverse_selection_iv`.
        maker_adverse_selection_iv: float = 0.5,
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
        self._delta_hedge_band = float(delta_hedge_band)
        self._spot_fetcher = spot_fetcher
        self._vpin_provider = vpin_provider
        self._maker_wait_seconds = maker_wait_seconds
        self._baseline_target_dvol = baseline_target_dvol
        self._sizing_edge_cap = sizing_edge_cap
        self._sizing_dd_floor = sizing_dd_floor
        self._use_tiered_taker = use_tiered_taker
        self._cancel_exit_max_slippage_pct = float(cancel_exit_max_slippage_pct)
        self._maker_adverse_selection_iv = float(maker_adverse_selection_iv)

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
        profit_scale: float = 1.0,
        kelly_mult: float = 1.0,
    ) -> float:
        """Compute the BTC notional for a new straddle trade.

        Mirrors backtester's composite_v2 model with the V2 Kelly-shrinkage
        upside-only multiplier appended after the four base scales:

        1. **DVOL-scaled base margin** — `target_margin * (baseline_dvol / dvol)`,
           clipped so we never exceed ``target_margin_utilization``.
        2. **Edge scale** — size up when this bar's edge is above the recent
           median, size down when below.  Bounded to [0.25, sizing_edge_cap].
        3. **Drawdown floor** — scale down proportionally when in drawdown,
           with a floor of ``sizing_dd_floor``.
        4. **Liquidity scale** (liq_scale, caller-supplied) — spread-EMA/current
           spread.  Already implemented by the caller via exec_cost_ema.
        5. **Profit scale** (profit_scale, caller-supplied) — empirical-Bayes
           shrinkage estimator from
           :class:`deribit_paper.execution.profit_scale.ProfitScaleEstimator`,
           applied AFTER edge/dd/liq and BEFORE Kelly. Mirrors the backtester
           at ``vol_arb/backtester.py:1986-1993``. Defaults to 1.0 (no-op)
           so the ``default`` config preset stays byte-id with prior runs.
        6. **Kelly K1 multiplier** (kelly_mult, caller-supplied) — per-bar
           upside-only multiplier in [1.0, 2.0] sourced from
           ``BTC_v2_kelly_mult_K1.parquet``. Applied AFTER DVOL/edge/dd/liq
           /profit_scale and BEFORE the dynamic BTC cap, mirroring the
           backtester at ``vol_arb/backtester.py:2001-2006``. Shorts always
           pass through with mult=1.0 (caller is responsible for the
           asymmetry).

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

        # D3 (parity 2026-05-07): ``dynamic_btc_cap`` is the
        # PARTICIPATION cap, mirroring V2 backtester at
        # ``vol_arb/backtester.py:1796-1798``:
        #     ``dynamic_btc_cap = clip(trailing_vol * max_participation_rate, 1.0, 50.0)``
        # and ``vol_arb/sizing.py:_base_size`` where every strategy's
        # base size is ``min(margin_envelope, dynamic_btc_cap)`` — i.e.
        # the participation cap is the ceiling, not the margin envelope.
        # Pre-D3 the live engine set this to ``float(n)`` (margin
        # envelope), so subsequent re-clips at profit_scale (L236) and
        # Kelly (L247) used the wrong ceiling, allowing N to drift from
        # the V2 reference. Fall back to the margin envelope when
        # trailing volume is unknown so early bars / venue outages
        # don't allow unbounded sizing.
        if trailing_1h_volume_btc is not None and trailing_1h_volume_btc > 0:
            dynamic_btc_cap = float(np.clip(
                trailing_1h_volume_btc * self._max_participation,
                1.0, 50.0,
            ))
        else:
            dynamic_btc_cap = float(n)
        n = min(n, dynamic_btc_cap)

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
        # D3: V2 backtester `size_composite_v2` returns
        # `min(base * edge*dd*liq, dynamic_btc_cap)` at sizing.py:271.
        # Mirror that re-cap here so profit_scale and Kelly operate on a
        # participation-clipped value (was missing pre-D3).
        n = min(n, dynamic_btc_cap)

        # 5. Profit scale (empirical-Bayes shrinkage, V2 only). Mirrors
        # backtester order: applied AFTER edge/dd/liq, BEFORE Kelly.
        # Caller passes 1.0 when ``use_profit_scaling`` is off, so the
        # default preset is byte-id with the pre-E.1.1 behaviour. Clip
        # defensively to a safe band even though the estimator already
        # bounds its output. Re-clip against dynamic_btc_cap (E.2.6).
        if 0.0 < profit_scale < 1.0 - _SCALE_EPS:
            n *= float(np.clip(profit_scale, 0.05, 1.0))
            n = min(n, dynamic_btc_cap)

        # 6. Kelly K1 multiplier (upside-only, applied AFTER edge/dd/liq
        # /profit_scale). Caller passes 1.0 when the sidecar is missing
        # / signal is short / bar timestamp falls outside sidecar
        # coverage. The backtester short-circuits on ``km > 1.0`` so the
        # size_mult arithmetic is a no-op at floor=1.0; we mirror that
        # exactly. Re-clip against dynamic_btc_cap so Kelly can never
        # breach the bar's margin envelope (E.2.6).
        if kelly_mult > 1.0 + _SCALE_EPS:
            n *= kelly_mult
            n = min(n, dynamic_btc_cap)

        # Hard cap (static — final risk envelope).
        # D10 (parity 2026-05-07): ``_max_trade_size <= 0`` disables the
        # cap. The V2 backtester has no per-trade hard cap (uses
        # ``dynamic_btc_cap`` only); the V2 preset wires through 0.0
        # so the only ceiling is the participation cap. Non-V2 presets
        # keep the legacy 2 BTC default.
        if self._max_trade_size > 0.0:
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

    def _maker_adverse_selection_charge(
        self,
        *,
        entry_call: OrderResult,
        entry_put: OrderResult,
        n_contracts: float,
        spot: float,
        strike: float,
        iv_entry_decimal: float,
        tau_entry_years: float,
    ) -> float:
        """E-7 (parity-r3 cycle 3 — 2026-05-08): per-leg maker adverse-
        selection charge in USD.

        Mirrors offline ``vol_arb/backtester.py:2256-2272`` semantic
        ``maker_adv_sel_penalty = vega_s * cfg.maker_adverse_selection_iv``
        applied per-BTC and scaled by ``N`` — but live charges
        proportionally based on the per-leg maker fill fraction (more
        granular than the backtester's trade-level boolean).

        Returns 0.0 when neither leg filled as maker, when the penalty
        is configured off (``maker_adverse_selection_iv <= 0``), or
        when the BSM vega computation degenerates. The cost is
        ``vega_per_vol_pt × penalty × n_filled_as_maker``.
        """
        if self._maker_adverse_selection_iv <= 0.0:
            return 0.0

        def _maker_filled_btc(result: OrderResult, fallback_n: float) -> float:
            raw = result.raw or {}
            exec_type = str(raw.get("execution_type", "")).lower()
            if exec_type == "maker":
                # Full leg filled at maker; in the rare partial maker
                # finished case, ``maker_filled`` would be < total.
                return float(raw.get("maker_filled", result.filled_amount))
            if exec_type == "partial_maker":
                return float(raw.get("maker_filled", 0.0))
            # ``taker`` / unknown: nothing to charge.
            return 0.0

        n_call_maker = _maker_filled_btc(entry_call, n_contracts)
        n_put_maker = _maker_filled_btc(entry_put, n_contracts)
        total_maker_btc = max(n_call_maker + n_put_maker, 0.0)
        if total_maker_btc <= 0.0:
            return 0.0

        # Per-leg vega in USD per vol-pt (BS vega is direction-symmetric;
        # call vega == put vega at the same K, T, σ).
        leg_vega_usd_per_vol_pt = bsm_vega_per_vol_pt(
            spot=spot,
            strike=strike,
            iv_decimal=iv_entry_decimal,
            tau_years=tau_entry_years,
        )
        if leg_vega_usd_per_vol_pt <= 0.0:
            return 0.0

        # Total USD charge = per-leg vega × penalty × BTC of leg-fills
        # marked as maker. When BOTH legs maker-fill the full size, this
        # equals ``2 × leg_vega × penalty × n_contracts`` — i.e. the
        # composite straddle vega × penalty × N, matching the backtester's
        # ``vega_s × cfg.maker_adverse_selection_iv × N`` exactly.
        return float(
            leg_vega_usd_per_vol_pt
            * self._maker_adverse_selection_iv
            * total_maker_btc
        )

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
        profit_scale: float = 1.0,
        kelly_mult: float = 1.0,
        taker_slippage_pct: float = 0.5,
        vol_ann_entry_lagged: float | None = None,
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

        # ── Liquidity scaling + observed-spread fetch ──────────────
        # One ticker call serves both liq_scale (uses spread/EMA) and
        # the observed-spread override clause in the slippage calc
        # (E.1.9 2026-05-01: pre-E.1.9 the override path was unreachable
        # because engine.py hardcoded ``observed_spread_usd=None``).
        liq_scale = 1.0
        observed_spread_usd: float | None = None
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
                    if best_ask > 0 and best_bid > 0 and spot > 0:
                        # Spread in USD = (ask - bid) × spot when prices
                        # are quoted in BTC (Deribit option convention).
                        spread_usd = (best_ask - best_bid) * spot
                        observed_spread_usd = max(spread_usd, 0.0)
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

        # Recompute the slippage cap using the observed spread, so the
        # 1.5× override clause in ``live_slippage_pct`` becomes reachable.
        # Caller (engine.process_bar) already passed ``taker_slippage_pct``
        # computed from the sidecar alone; we override here only when we
        # have a fresh observed_spread_usd.
        if observed_spread_usd is not None and spot > 0:
            from deribit_paper.execution.slippage_calc import live_slippage_pct
            # Convert ``taker_slippage_pct`` (already a pct) to USD-per-BTC
            # for the helper's "sidecar pct vs observed" comparison.
            sidecar_implied_usd = taker_slippage_pct / 100.0 * spot
            taker_slippage_pct = live_slippage_pct(
                slip_usd_per_btc=sidecar_implied_usd,
                spot=spot,
                observed_spread_usd=observed_spread_usd,
                fallback_pct=taker_slippage_pct,
            )

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
            profit_scale=profit_scale,
            kelly_mult=kelly_mult,
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

        # E.1.13: append random suffix so labels can't be guessed by an
        # adversary observing log timestamps. The timestamp prefix is
        # kept for human-readable forensics.
        entry_nonce = secrets.token_hex(8)
        # E-5 (parity-r3 cycle 2 — 2026-05-08): use maker-then-taker on
        # entry to mirror the V2 backtester's SOR (`backtester.py:2215-2234`),
        # which simulates a maker fill (zero option slippage) when a taker
        # prints through the would-be-maker within 5s. ~30-60% of bars
        # see maker fills offline; pre-fix live always paid full taker
        # slip on every leg, accumulating ~$5-15/trade × 7,400 trades ≈
        # $40-100K of phantom backtest PnL not reproducible live.
        # `place_maker_then_taker` already implements the cancel-and-fall-back
        # logic at `order_manager.py:819-972`, and `cfg.maker_wait_seconds`
        # (default 60s) is already plumbed via `engine.py:861`. Exits stay
        # on `place_ioc_taker` (immediately below) — bar-aligned exit
        # timing makes a 60s maker wait miss the next bar's signal.
        entry_call, entry_put = await asyncio.gather(
            self._order_manager.place_maker_then_taker(
                instrument_name=straddle.call_instrument,
                side=entry_side,
                amount=n_contracts,
                maker_wait_seconds=self._maker_wait_seconds,
                label=f"straddle_entry_C_{started_at.value}_{entry_nonce}",
                reduce_only=False,
                taker_max_slippage_pct=taker_slippage_pct,
                taker_tiered=self._use_tiered_taker,
            ),
            self._order_manager.place_maker_then_taker(
                instrument_name=straddle.put_instrument,
                side=entry_side,
                amount=n_contracts,
                maker_wait_seconds=self._maker_wait_seconds,
                label=f"straddle_entry_P_{started_at.value}_{entry_nonce}",
                reduce_only=False,
                taker_max_slippage_pct=taker_slippage_pct,
                taker_tiered=self._use_tiered_taker,
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

        # ── 1b. E-7: maker adverse-selection charge (parity-r3 cycle 3) ──
        # The V2 backtester at ``vol_arb/backtester.py:2256-2259`` charges
        # ``vega_s × cfg.maker_adverse_selection_iv`` (per-BTC USD, scaled
        # by N at line 2274) on trades whose SOR confirmed a maker fill.
        # Pre-fix the live engine charged nothing on its maker entries
        # (E-5 closed in cycle 2 enabled maker-then-taker but the penalty
        # was never wired in).
        # Live now charges proportionally based on the per-leg maker fill
        # fraction — more honest than the backtester's trade-level
        # boolean. ``execution_type == "maker"`` means full maker;
        # ``"partial_maker"`` reports the partial fill in `raw["maker_filled"]`.
        # ``"taker"`` (or absent) means no maker → zero charge for that leg.
        maker_adverse_selection_charge_usd = self._maker_adverse_selection_charge(
            entry_call=entry_call,
            entry_put=entry_put,
            n_contracts=n_contracts,
            spot=spot,
            strike=straddle.strike,
            iv_entry_decimal=(
                float(signal.iv_used_pct) / 100.0
                if signal.iv_used_pct and signal.iv_used_pct > 0
                else 0.75
            ),
            tau_entry_years=max(
                (straddle.expiry_dt - started_at).total_seconds()
                / (365.25 * 24 * 3600),
                1e-6,
            ),
        )
        if maker_adverse_selection_charge_usd > 0.0:
            logger.info(
                "E-7: maker adverse-selection charge $%.2f "
                "(call=%s put=%s, vol_pts=%.2f)",
                maker_adverse_selection_charge_usd,
                (entry_call.raw or {}).get("execution_type", "taker"),
                (entry_put.raw or {}).get("execution_type", "taker"),
                self._maker_adverse_selection_iv,
            )

        # ── 2. BEGIN DRL hedge trade ───────────────────────────────
        tau_entry_years = max(
            (straddle.expiry_dt - started_at).total_seconds() / (365.25 * 24 * 3600),
            1e-6,
        )
        # D1 (parity-cycle round 2 — 2026-05-07): plumb the SVI slice
        # params populated by the predictor through to the DRL hedger so
        # IV updates per-tick instead of staying frozen at ``iv_entry``
        # for all 10 ticks. ``None`` is the safe degraded mode — the
        # hedger falls back to ``iv_entry`` (matches pre-fix behaviour
        # under uncalibrated SVI / missing-expiry).
        # D-8 (parity-r3 cycle 2 — 2026-05-08): prefer the lagged
        # ``vol_ann`` (raw, prior bar) over the current bar's
        # ``vol_ann_continuous``. The training env at
        # ``env.py:172`` reads ``df["vol_ann"].shift(1)`` — i.e. the
        # PRIOR bar's raw vol_ann — and the offline backtester's
        # ``DRLHedger.simulate`` consumes the same column. Live's
        # pre-fix value used (a) the current bar (one ahead) and (b)
        # the jump-corrected ``continuous`` column (different formula).
        # Fall back to the current bar only at cold start (no prior
        # bar in the buffer yet).
        if vol_ann_entry_lagged is not None and vol_ann_entry_lagged > 0:
            vol_ann_entry_value = float(vol_ann_entry_lagged)
        elif signal.bar_tsrv is not None:
            vol_ann_entry_value = float(signal.bar_tsrv.vol_ann_continuous)
        else:
            vol_ann_entry_value = 0.8
        # D-10 (parity-r3 cycle 3 — 2026-05-08): VPIN stream can drop
        # mid-trade (Binance WebSocket reconnect, sub disconnect) or
        # return non-finite values (degenerate aggregation). Pre-fix the
        # raw provider value flowed straight into ``begin_trade`` →
        # downstream the SAC v4 obs vector took NaN, silently mispricing
        # the hedge ratio.
        #
        # Cycle-22 (2026-05-11): the previous fallback ``0.5`` was a
        # 10σ outlier vs SAC v4's training-time VPIN distribution
        # (mean ≈ 0.189, std ≈ 0.032 on the production parquet). Using
        # 0.5 forces the policy into a regime it was never trained on.
        # The 7-agent audit (A6 H7) flagged this as a silent corruption
        # of every hedge action when the provider misbehaves.
        # Replaced the fallback with ``VPIN_TRAINING_MEAN = 0.189`` —
        # in-distribution neutral. The hard fail-fast (raise on
        # provider=None when DRL is enabled) is enforced at engine
        # startup time (engine.py LiveVPINStream wiring), so by the
        # time control reaches here, ``self._vpin_provider`` is wired
        # in any prod V2 config. Tests can still pass ``None`` to
        # bypass the wiring.
        VPIN_TRAINING_MEAN = 0.189  # SAC v4 training distribution mean
        vpin_value: float = VPIN_TRAINING_MEAN
        if self._vpin_provider is not None:
            try:
                vpin_raw = float(self._vpin_provider())
                if math.isfinite(vpin_raw):
                    vpin_value = vpin_raw
                else:
                    logger.warning(
                        "Cycle-22: VPIN provider returned non-finite "
                        "(%s); falling back to training-mean %.3f",
                        vpin_raw, VPIN_TRAINING_MEAN,
                    )
            except (TypeError, ValueError) as exc:
                logger.warning(
                    "Cycle-22: VPIN provider raised %s; falling back "
                    "to training-mean %.3f",
                    exc, VPIN_TRAINING_MEAN,
                )
        trade_id = self._drl.begin_trade(
            signal=signal.direction,
            S_entry=spot,
            K=straddle.strike,
            vol_ann_entry=vol_ann_entry_value,
            iv_entry=float(signal.iv_used_pct) / 100.0 if signal.iv_used_pct else 0.75,
            tau_entry=tau_entry_years,
            V_bg=float(trailing_1h_volume_btc) if trailing_1h_volume_btc else 100.0,
            jump_fraction=float(signal.bar_tsrv.jump_fraction) if signal.bar_tsrv else 0.0,
            vpin=vpin_value,
            svi_slice_params=signal.svi_slice,
        )

        # ── 2b. Notify position opened (persist to DB + tracker) ──
        # O-8 (parity-r3 cycle 8 — 2026-05-08): record the EFFECTIVE
        # filled straddle size = min(call_filled, put_filled). Pre-
        # cycle-8 we recorded ``n_contracts`` (the REQUESTED amount),
        # so a tier-3 partial fill on either leg silently overstated
        # the position. The straddle is constrained by the smaller
        # leg's fill (you can't be long N-call/M-put as a "straddle"
        # — it's max(M, N) of straddle + abs(N-M) of single-leg
        # residual). For position-tracking we use the straddle size.
        effective_n_contracts = min(
            float(entry_call.filled_amount),
            float(entry_put.filled_amount),
        )
        if effective_n_contracts <= 0.0:
            effective_n_contracts = float(n_contracts)
        if self._on_position_opened is not None:
            try:
                await self._on_position_opened(
                    trade_id=trade_id,
                    call_instrument=straddle.call_instrument,
                    put_instrument=straddle.put_instrument,
                    signal_direction=signal.direction,
                    n_contracts_btc=effective_n_contracts,
                    strike=straddle.strike,
                    expiry_dt=str(straddle.expiry_dt),
                    opened_at=str(started_at),
                    entry_call_price=entry_call.average_price,
                    entry_put_price=entry_put.average_price,
                )
            except Exception:
                logger.exception("on_position_opened callback failed")

        # ── 3. Spawn hedge loop ────────────────────────────────────
        # E2 (parity-cycle round 2 — 2026-05-07): wire ``cfg.delta_hedge_band``
        # through to PerpHedger's dust threshold. Pre-fix the engine
        # accepted ``delta_hedge_band=0.05`` from config but never
        # passed it through, so PerpHedger used a hardcoded 0.01 BTC
        # dust threshold — 5x tighter than configured.
        hedger = PerpHedger(
            order_manager=self._order_manager,
            live_drl_hedger=self._drl,
            perp_instrument=self._perp_instrument,
            bar_minutes=self._holding_minutes,
            n_ticks_per_bar=self._n_hedge_ticks,
            dust_threshold_btc=self._delta_hedge_band,
            spot_fetcher=self._spot_fetcher,
        )
        hedge_task = asyncio.create_task(
            hedger.run(trade_id, entry_time=started_at),
        )

        # ── 4. Wait holding period (D12 parity 2026-05-07) ─────────
        # V2 backtester exits at bar-boundary (loop counter ``i +
        # holding_bars``); pre-D12 the live engine sleeped the
        # wall-clock holding duration from the entry time. With
        # ``holding_minutes == bar_minutes`` (V2 preset 15/15) the two
        # are equivalent within ≤1s of execution latency, but for
        # non-V2 horizons or under entry latency >1s they drift.
        #
        # Fix: compute target exit = ``started_at + holding_minutes``
        # then round UP to the next bar-boundary that is ≥ target.
        # ``bar_minutes == holding_minutes`` here (the StraddleFlow
        # constructor passes ``bar_minutes=self._holding_minutes`` to
        # the PerpHedger above, so for V2 this is bar-aligned).
        # E-9 (parity-r3 cycle 8 — 2026-05-08): anchor the holding-
        # period target to the SIGNAL's bar timestamp, not the
        # post-fill ``started_at``. The latter floats with broker
        # latency (entry-fill latency, IOC tier sleeps, etc.); the
        # offline backtester models the exit at exactly bar T+1's
        # boundary regardless of fill time. Falls back to
        # ``started_at`` when the signal lacks a usable timestamp.
        holding_seconds = self._holding_minutes * 60.0
        anchor = (
            signal.timestamp
            if signal.timestamp is not None
            else started_at
        )
        target_exit = anchor + pd.Timedelta(seconds=holding_seconds)
        # Round UP to the next bar boundary. ``bar_seconds == 0`` (test
        # configurations with holding_minutes=0) skips the alignment.
        bar_seconds = int(self._holding_minutes * 60)
        if bar_seconds > 0:
            epoch_seconds = int(target_exit.timestamp())
            remainder = epoch_seconds % bar_seconds
            if remainder != 0:
                target_exit = target_exit + pd.Timedelta(
                    seconds=(bar_seconds - remainder),
                )
        sleep_seconds = max(
            0.0,
            (target_exit - pd.Timestamp.now(tz="UTC")).total_seconds(),
        )
        try:
            await asyncio.sleep(sleep_seconds)
        except asyncio.CancelledError:
            # M2: On cancellation (shutdown/kill switch), close straddle legs
            # before re-raising to avoid leaving naked positions.
            logger.warning(
                "StraddleFlow.execute cancelled during holding period — "
                "closing straddle legs before propagating cancellation"
            )
            hedger.cancel()
            exit_side = "buy" if signal.direction == -1 else "sell"
            cancel_nonce = secrets.token_hex(8)
            try:
                # E-8 (parity-r3 cycle 2 — 2026-05-08): cancel-exit uses
                # tiered IOC instead of naked market — the slippage cap
                # is enforced even on shutdown so the engine respects
                # the "never naked market" contract from E.1.7.
                await asyncio.gather(
                    self._order_manager.place_ioc_taker(
                        instrument_name=straddle.call_instrument,
                        side=exit_side, amount=n_contracts,
                        max_slippage_pct=self._cancel_exit_max_slippage_pct,
                        tiered=True,
                        label=f"cancel_exit_C_{started_at.value}_{cancel_nonce}",
                        reduce_only=True,
                    ),
                    self._order_manager.place_ioc_taker(
                        instrument_name=straddle.put_instrument,
                        side=exit_side, amount=n_contracts,
                        max_slippage_pct=self._cancel_exit_max_slippage_pct,
                        tiered=True,
                        label=f"cancel_exit_P_{started_at.value}_{cancel_nonce}",
                        reduce_only=True,
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
        exit_nonce = secrets.token_hex(8)
        exit_call, exit_put = await asyncio.gather(
            self._order_manager.place_ioc_taker(
                instrument_name=straddle.call_instrument,
                side=exit_side,
                amount=n_contracts,
                max_slippage_pct=taker_slippage_pct,
                label=f"straddle_exit_C_{started_at.value}_{exit_nonce}",
                reduce_only=True,
                tiered=self._use_tiered_taker,
            ),
            self._order_manager.place_ioc_taker(
                instrument_name=straddle.put_instrument,
                side=exit_side,
                amount=n_contracts,
                max_slippage_pct=taker_slippage_pct,
                label=f"straddle_exit_P_{started_at.value}_{exit_nonce}",
                reduce_only=True,
                tiered=self._use_tiered_taker,
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
            maker_adverse_selection_charge_usd=maker_adverse_selection_charge_usd,
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

        E-8 (parity-r3 cycle 2 — 2026-05-08): close uses tiered IOC
        taker, not naked market. The tier-3 ladder still escalates to
        a wider IOC at ``1.5×max_slippage_pct``; on full miss it logs
        CRITICAL and the per-attempt retry loop here gives the next
        retry a chance at a stabilising book. Tier-3 protects against
        manipulated thin books (E.1.7).
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
                result = await self._order_manager.place_ioc_taker(
                    instrument_name=instrument,
                    side=close_side,
                    amount=order.filled_amount,
                    max_slippage_pct=self._cancel_exit_max_slippage_pct,
                    tiered=True,
                    label=(
                        f"unilateral_close_{instrument}_{attempt}_"
                        f"{secrets.token_hex(8)}"
                    ),
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
