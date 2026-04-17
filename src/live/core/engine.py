"""Strategy-agnostic trading engine.

Only depends on protocols (Strategy, DataProvider, ExecutionAlgo, ExchangeAdapter).
No strategy-specific imports. All strategy logic lives in strategies/.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Optional

import numpy as np
import pandas as pd

from live.core.interfaces import (
    Strategy, DataProvider, ExchangeAdapter, ExecutionAlgo,
    Signal, Position, PortfolioView, TradeResult,
)
from live.core.state import StateDB
from live.core.aggregator import SignalAggregator
from live.core.reconciler import Reconciler
from live.core.monitor import ExecutionMonitor
from live.core.data_validator import validate_ohlcv, validate_ticker
from live.notifier import TelegramNotifier
from live.config import (
    TF_MINUTES, PortfolioConfig, MonitorConfig, ReconcilerConfig,
    DataValidatorConfig,
)

logger = logging.getLogger(__name__)


class TradingEngine:
    """Strategy-agnostic main loop.

    Lifecycle:
        1. initialize() — load strategies, reconcile, restore state
        2. run() — main loop (sleep → on_bar → repeat)
        3. shutdown() — save state, close connections

    The engine never imports strategy-specific code. All strategy
    logic is behind the Strategy protocol.
    """

    def __init__(
        self,
        strategies: list[Strategy],
        data_provider: DataProvider,
        exchange: ExchangeAdapter,
        execution: ExecutionAlgo,
        state: StateDB,
        notifier: TelegramNotifier,
        starting_capital: float = 250_000.0,
        bar_interval_m: int = 15,
        poll_offset_s: int = 5,
        leverage: int = 1,
        max_signals_per_bar: int | None = None,
        max_risk_per_trade: float | None = None,
        sl_cap: float = 0.75,
        portfolio_config: PortfolioConfig | None = None,
        monitor_config: MonitorConfig | None = None,
        reconciler_config: ReconcilerConfig | None = None,
        data_validator_config: DataValidatorConfig | None = None,
        quote_currency: str = "USDT",
        tf_lookbacks: dict[str, int] | None = None,
        paper_mode: bool = False,
    ) -> None:
        self._strategies = strategies
        self._data = data_provider
        self._exchange = exchange
        self._execution = execution
        self._state = state
        self._notifier = notifier
        self._starting_capital = starting_capital
        self._bar_interval_m = bar_interval_m
        self._poll_offset_s = poll_offset_s
        self._leverage = leverage
        self._max_risk_per_trade = max_risk_per_trade
        self._sl_cap = sl_cap
        self._quote_currency = quote_currency
        self._tf_lookbacks = tf_lookbacks or {}
        self._paper_mode = paper_mode

        # Resolve portfolio config eagerly
        pc = portfolio_config or PortfolioConfig()
        self._max_per_asset = pc.max_per_asset
        self._max_exposure = pc.max_exposure
        self._max_dd = pc.max_dd
        self._daily_dd_limit = pc.daily_dd_limit
        self._max_margin_pct = pc.max_margin_pct
        self._max_notional_mult = pc.max_notional_mult

        # HyroTrader compliance state
        self._initial_balance = starting_capital
        self._day_start_equity = starting_capital
        self._daily_dd_halted = False
        self._last_dd_reset_day: int | None = None

        # Resolve data validator config
        dv = data_validator_config or DataValidatorConfig()
        self._min_bars = dv.min_bars
        self._max_staleness_factor = dv.max_staleness_factor
        self._max_gap_factor = dv.max_gap_factor

        # Store reconciler config for use in initialize()
        self._reconciler_config = reconciler_config or ReconcilerConfig()

        self._aggregator = SignalAggregator(
            max_positions_per_strategy=pc.max_positions_per_strategy,
            max_exposure_pct=pc.max_exposure,
            max_per_asset_pct=pc.max_per_asset,
            max_signals_per_bar=max_signals_per_bar,
        )
        self._monitor = ExecutionMonitor(state, notifier, config=monitor_config)

        # Runtime state
        self._equity = starting_capital
        self._peak_equity = starting_capital
        self._open_positions: list[Position] = []
        self._vol_series: dict[str, pd.Series] = {}
        self._vol_cache_times: dict[str, float] = {}
        self._running = False
        self._daily_trades = 0
        self._daily_signals = 0

    async def initialize(self) -> None:
        """Initialize all components and reconcile state."""
        logger.info("Initializing trading engine...")

        # 0. Pre-flight: validate exchange connection (if supported)
        if hasattr(self._exchange, 'validate_connection') and not self._paper_mode:
            ok = await self._exchange.validate_connection()
            if not ok:
                raise RuntimeError("Exchange pre-flight validation failed — aborting")

        # 1. Reconcile: sync DB ↔ exchange
        reconciler = Reconciler(
            self._exchange, self._state,
            stale_order_age=timedelta(minutes=self._reconciler_config.stale_order_age_minutes),
            paper_mode=self._paper_mode,
        )
        report = await reconciler.reconcile()
        if report.warnings:
            for w in report.warnings:
                logger.warning("Reconcile: %s", w)
            await self._notifier.notify_error(
                f"Reconciliation: {len(report.orphaned_positions)} orphaned, "
                f"{len(report.ghost_positions)} ghost, "
                f"{report.stale_orders_cancelled} stale orders cancelled")

        # 2. Compute equity from trades (authoritative, not checkpoint)
        self._equity = await self._state.compute_equity_from_trades(self._starting_capital)
        saved_peak = await self._state.load_checkpoint("peak_equity")
        self._peak_equity = float(saved_peak) if saved_peak else self._equity

        if self._equity > self._peak_equity:
            self._peak_equity = self._equity

        # 3. Restore open positions
        self._open_positions = await self._state.get_open_positions()

        # 3b. Mark-to-market open positions (adjust equity for unrealized PnL)
        unrealized = 0.0
        for pos in self._open_positions:
            try:
                ticker = await self._data.fetch_ticker(pos.symbol)
                price = ticker.get("last_price", pos.entry_price)
                if price > 0 and pos.entry_price > 0:
                    if pos.direction > 0:
                        unrealized += pos.size_usd * (price - pos.entry_price) / pos.entry_price
                    else:
                        unrealized += pos.size_usd * (pos.entry_price - price) / pos.entry_price
            except Exception:
                pass
        if abs(unrealized) > 0.01:
            logger.info("MTM adjustment: $%.2f unrealized PnL on %d positions",
                        unrealized, len(self._open_positions))
            self._equity += unrealized

        # 4. Restore strategy state
        for strategy in self._strategies:
            state = await self._state.load_strategy_state(strategy.name)
            if state:
                strategy.restore_state(state)
                logger.info("Restored state for strategy '%s'", strategy.name)

        # 5. Set leverage for universe
        all_assets = set()
        for strategy in self._strategies:
            for pos in self._open_positions:
                if pos.strategy_id == strategy.name:
                    all_assets.add(pos.symbol)
        for symbol in all_assets:
            await self._exchange.set_leverage(symbol, self._leverage)

        # 5b. Log HyroTrader limits
        logger.info("Engine initialized: equity=$%.0f, %d strategies, %d open positions",
                     self._equity, len(self._strategies), len(self._open_positions))
        logger.info("HyroTrader limits (initial=$%.0f): risk/trade=$%.0f, "
                     "daily_dd=$%.0f, margin=$%.0f, notional=$%.0f, floor=$%.0f",
                     self._initial_balance,
                     self._initial_balance * (self._max_risk_per_trade or 0),
                     self._initial_balance * self._daily_dd_limit,
                     self._initial_balance * self._max_margin_pct,
                     self._initial_balance * self._max_notional_mult,
                     self._initial_balance * (1 - self._max_dd))

        await self._notifier.notify_startup(
            self._equity, len(self._strategies),
            len(self._open_positions), True)

    async def run(self) -> None:
        """Main loop: sleep until next bar, run cycle, repeat."""
        self._running = True
        logger.info("Engine running. Ctrl+C to stop.")

        while self._running:
            try:
                next_bar = self._next_bar_time()
                now = datetime.now(tz=timezone.utc)
                wait_s = (next_bar - now).total_seconds()

                if wait_s > 0:
                    logger.info("Next bar at %s (%.0fs)", next_bar.strftime("%H:%M"), wait_s)
                    await asyncio.sleep(wait_s)

                bar_time = datetime.now(tz=timezone.utc)
                await self._on_bar(bar_time)

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Bar cycle error: %s", e, exc_info=True)
                await self._notifier.notify_error(f"Bar error: {e}")
                await asyncio.sleep(60)

    async def _on_bar(self, bar_time: datetime) -> None:
        """One cycle — strategy-agnostic signal flow."""
        t0 = time.monotonic()

        # 0. Which TFs fired? (union of all strategies' timeframes)
        all_tfs = set()
        for strategy in self._strategies:
            for tf in strategy.timeframes:
                if self._tf_fires(tf, bar_time):
                    all_tfs.add(tf)

        if not all_tfs:
            return

        logger.info("Bar %s — active TFs: %s", bar_time.strftime("%H:%M"), sorted(all_tfs))

        # Refresh open positions from DB
        self._open_positions = await self._state.get_open_positions()

        # HyroTrader: daily DD reset at UTC midnight
        today = bar_time.timetuple().tm_yday
        if self._last_dd_reset_day != today:
            if self._last_dd_reset_day is not None:
                logger.info("Daily DD reset: day_start=$%.0f, equity=$%.0f",
                            self._day_start_equity, self._equity)
            self._day_start_equity = self._equity
            self._daily_dd_halted = False
            self._last_dd_reset_day = today


        # 0b. Strategy pre-bar hook (universe rebalance, model loading, etc.)
        for strategy in self._strategies:
            if hasattr(strategy, 'prepare_bar'):
                try:
                    await strategy.prepare_bar(bar_time, self._data, self._exchange)
                except Exception as e:
                    logger.error("Strategy '%s' prepare_bar error: %s",
                                 strategy.name, e, exc_info=True)

        # 1. Determine universe (all assets any strategy could trade)
        universe = self._get_universe()

        # 2. Fetch + validate market data
        # Always include 4h for regime filter, even when 4h TF doesn't fire
        fetch_tfs = all_tfs | {"4h"}
        market_data: dict[tuple[str, str], pd.DataFrame] = {}
        data_warnings = []
        for tf in fetch_tfs:
            lookback = self._tf_lookbacks.get(tf, 200)
            for asset in universe:
                symbol = f"{asset}{self._quote_currency}"
                try:
                    raw = await self._data.fetch_ohlcv(symbol, tf, lookback)
                    # Drop incomplete current bar — only use fully closed bars.
                    # Cutoff must be per-TF: at 18:00 the current 4h bar started
                    # at 16:00 (covers 16:00-19:59), so 4h cutoff = 16:00, not 18:00.
                    tf_min = TF_MINUTES.get(tf, self._bar_interval_m)
                    total_min = bar_time.hour * 60 + bar_time.minute
                    cutoff_min = (total_min // tf_min) * tf_min
                    bar_cutoff = bar_time.replace(
                        hour=cutoff_min // 60, minute=cutoff_min % 60,
                        second=0, microsecond=0)
                    raw = raw[raw.index < bar_cutoff]
                    validated, warnings = validate_ohlcv(
                        raw, symbol, tf,
                        max_staleness_factor=self._max_staleness_factor,
                        max_gap_factor=self._max_gap_factor,
                    )
                    data_warnings.extend(warnings)
                    if len(validated) >= self._min_bars:
                        market_data[(tf, asset)] = validated
                except Exception as e:
                    data_warnings.append(f"{symbol}/{tf}: fetch error: {e}")

        if data_warnings:
            critical = [w for w in data_warnings if "stale" in w.lower() or "empty" in w.lower()]
            if critical:
                await self._notifier.notify_error(
                    f"Data warnings: {'; '.join(critical[:3])}")

        # 3. Check exits (per-strategy, per-position)
        exits = 0
        for pos in list(self._open_positions):
            # Increment bars_held only when position's TF fires
            if pos.timeframe in all_tfs:
                pos.bars_held += 1
                await self._state.update_bars_held(pos.id, pos.bars_held)

            # In-engine SL monitoring — matches backtest dynamic SL formula.
            # Checks every bar (not just TF bars) since price can breach SL anytime.
            exit_reason = None
            if pos.entry_price > 0 and pos.size_usd > 0:
                sl_pct = min(
                    self._max_risk_per_trade * self._initial_balance / pos.size_usd,
                    self._sl_cap,
                )
                current_price = None
                tf_data = market_data.get((pos.timeframe, pos.asset))
                if tf_data is not None and len(tf_data) > 0:
                    current_price = tf_data["close"].iloc[-1]
                if current_price and current_price > 0:
                    unrealized = pos.direction * (current_price - pos.entry_price) / pos.entry_price
                    if unrealized < -sl_pct:
                        exit_reason = "compliance_sl"
                        logger.warning(
                            "SL_TRIGGER %s/%s: unrealized=%.2f%% < -%.2f%% (sl_pct)",
                            pos.asset, pos.timeframe,
                            unrealized * 100, sl_pct * 100)

            # Strategy-specific exit check
            if not exit_reason:
                strategy = self._get_strategy(pos.strategy_id)
                if strategy:
                    exit_reason = strategy.check_exit(pos, market_data, bar_time)
            else:
                strategy = self._get_strategy(pos.strategy_id)

            # Universal backstop: hold_bars expiry
            if exit_reason or pos.bars_held >= pos.hold_bars:
                await self._close_position(
                    pos, strategy, exit_reason or "hold_expired", bar_time)
                exits += 1

        # 4. Generate signals (per-strategy)
        all_signals: list[tuple[Strategy, Signal]] = []
        for strategy in self._strategies:
            strategy_tfs = set(strategy.timeframes) & all_tfs
            if not strategy_tfs:
                continue
            try:
                signals = strategy.on_bar(bar_time, market_data,
                                          self._portfolio_view(), strategy_tfs)
                for sig in signals:
                    all_signals.append((strategy, sig))
                    self._daily_signals += 1
            except Exception as e:
                logger.error("Strategy '%s' error: %s", strategy.name, e, exc_info=True)
                await self._notifier.notify_error(f"Strategy {strategy.name}: {e}")

        # 5. Aggregate: resolve conflicts, apply budgets
        filtered = self._aggregator.aggregate(all_signals, self._portfolio_view())

        # 6. Execute
        trades = 0
        for strategy, signal in filtered:
            await self._execute_signal(strategy, signal, bar_time)
            trades += 1

        # 6b. HyroTrader swing daily DD check (includes unrealized P&L)
        if self._daily_dd_limit and not self._daily_dd_halted:
            daily_floor = self._day_start_equity - (self._initial_balance * self._daily_dd_limit)
            # Include unrealized P&L on open positions
            unrealized = 0.0
            for pos in self._open_positions:
                try:
                    ticker = await self._data.fetch_ticker(pos.symbol)
                    price = ticker.get("last_price", pos.entry_price)
                    if pos.direction > 0:
                        unrealized += pos.size_usd * (price - pos.entry_price) / pos.entry_price
                    else:
                        unrealized += pos.size_usd * (pos.entry_price - price) / pos.entry_price
                except Exception:
                    pass
            mtm_equity = self._equity + unrealized
            if mtm_equity <= daily_floor:
                self._daily_dd_halted = True
                logger.critical(
                    "DAILY_DD_HALT: mtm_equity=$%.0f <= floor=$%.0f "
                    "(day_start=$%.0f, unrealized=$%.0f, limit=%.0f%%)",
                    mtm_equity, daily_floor, self._day_start_equity,
                    unrealized, self._daily_dd_limit * 100)
                await self._notifier.notify_error(
                    f"DAILY DD HALT: ${mtm_equity:,.0f} <= floor ${daily_floor:,.0f}")

        # 7. Save state
        await self._save_state()

        # 8. Daily summary at midnight UTC
        if bar_time.hour == 0 and bar_time.minute < self._bar_interval_m:
            await self._send_daily_summary(bar_time)
            self._daily_trades = 0
            self._daily_signals = 0

        elapsed = time.monotonic() - t0
        logger.info("Bar done: %d signals, %d trades, %d exits (%.1fs)",
                     len(all_signals), trades, exits, elapsed)

        # Hourly heartbeat via Telegram
        if bar_time.minute == 0:
            await self._notifier.notify_heartbeat(
                self._equity, len(self._open_positions), len(all_signals))

    async def _execute_signal(
        self, strategy: Strategy, signal: Signal, bar_time: datetime
    ) -> None:
        """Size + execute a single signal with atomic state persistence."""
        # Max drawdown halt — matches backtest (drawdown_improvements.py:289)
        if self._drawdown >= self._max_dd:
            logger.warning("MAX_DD HALT: drawdown %.1f%% >= %.1f%%, skipping all signals",
                           self._drawdown * 100, self._max_dd * 100)
            return

        # HyroTrader: absolute floor from initial balance (10%)
        abs_floor = self._initial_balance * (1 - self._max_dd)
        if self._equity <= abs_floor:
            logger.critical("MAX_LOSS HALT: equity $%.0f <= floor $%.0f (initial=$%.0f)",
                            self._equity, abs_floor, self._initial_balance)
            return

        # HyroTrader: swing daily DD halt
        if self._daily_dd_halted:
            logger.debug("DAILY_DD_HALTED: skipping %s", signal.symbol)
            return

        # Update vol series if needed
        await self._ensure_vol_series(signal.asset)

        # Compute vol multiplier
        vol_mult = 1.0
        if hasattr(strategy, 'compute_vol_mult'):
            strategy.set_vol_series(self._vol_series)
            vol_mult = strategy.compute_vol_mult(signal.asset)

        # Strategy-specific sizing
        size_usd = strategy.compute_size(
            signal, self._equity, self._drawdown, vol_mult)

        if size_usd < 1.0:
            return

        # Exposure limits (portfolio-level, from config)
        asset_exp = sum(p.size_usd for p in self._open_positions if p.asset == signal.asset)
        if asset_exp + size_usd > self._equity * self._max_per_asset:
            size_usd = max(self._equity * self._max_per_asset - asset_exp, 0)
        total_exp = sum(p.size_usd for p in self._open_positions)
        if total_exp + size_usd > self._equity * self._max_exposure:
            size_usd = max(self._equity * self._max_exposure - total_exp, 0)

        # HyroTrader: margin limit (25% of initial balance)
        if self._max_margin_pct and self._leverage > 0:
            total_margin = total_exp / self._leverage
            margin_limit = self._initial_balance * self._max_margin_pct
            new_margin = size_usd / self._leverage
            if total_margin + new_margin > margin_limit:
                size_usd = max((margin_limit - total_margin) * self._leverage, 0)

        # HyroTrader: notional limit (2× initial balance)
        if self._max_notional_mult:
            notional_limit = self._initial_balance * self._max_notional_mult
            if total_exp + size_usd > notional_limit:
                size_usd = max(notional_limit - total_exp, 0)

        if size_usd < 1.0:
            return

        # Get current price for qty
        ticker = await self._data.fetch_ticker(signal.symbol)
        _, ticker_warnings = validate_ticker(ticker, signal.symbol)
        if ticker_warnings:
            logger.warning("Ticker warnings: %s", ticker_warnings)

        current_price = ticker.get("last_price", 0)
        if current_price <= 0:
            return

        qty = size_usd / current_price
        side = "Buy" if signal.direction > 0 else "Sell"

        # Cross-exchange price divergence check
        if self._data is not self._exchange:
            try:
                ex_ticker = await self._exchange.fetch_ticker(signal.symbol)
                ex_mid = (ex_ticker.get("bid", 0) + ex_ticker.get("ask", 0)) / 2
                data_mid = (ticker.get("bid", 0) + ticker.get("ask", 0)) / 2
                if ex_mid > 0 and data_mid > 0:
                    div_bps = abs(ex_mid - data_mid) / data_mid * 10_000
                    if div_bps > 20:
                        logger.warning("PRICE_DIVERGENCE %s: data=%.2f exec=%.2f (%.1f bps)",
                                       signal.symbol, data_mid, ex_mid, div_bps)
                    # Use execution exchange price for qty calculation
                    qty = size_usd / ex_mid
            except Exception as e:
                logger.warning("Cross-exchange price check failed for %s: %s", signal.symbol, e)

        # Set leverage
        await self._exchange.set_leverage(signal.symbol, self._leverage)

        # Execute
        result = await self._execution.execute_entry(
            signal.symbol, side, qty, strategy_id=strategy.name)

        if result and result.filled:
            pos = Position(
                asset=signal.asset, symbol=signal.symbol,
                timeframe=signal.timeframe,
                side="long" if signal.direction > 0 else "short",
                direction=signal.direction,
                entry_price=result.fill_price,
                entry_time=bar_time,
                size_usd=size_usd,
                hold_bars=signal.hold_bars,
                fill_method=result.method,
                order_id=result.order_id,
                strategy_id=strategy.name,
            )

            # Atomic: position + trade + checkpoint
            await self._state.atomic_open_position(
                pos,
                {"timestamp": bar_time.isoformat(), "side": side},
                {"equity": str(self._equity)},
            )
            self._open_positions.append(pos)
            self._daily_trades += 1

            # Compliance SL: dynamic widest SL within risk budget
            if self._max_risk_per_trade is not None:
                await self._place_compliance_sl(
                    signal.symbol, side, qty, result.fill_price, size_usd)

            # Record TCA if executor supports it
            if hasattr(self._execution, 'record_tca'):
                await self._execution.record_tca(
                    result, signal.symbol, side, strategy.name,
                    decision_ts=bar_time, submit_ts=bar_time)

            await self._notifier.notify_trade(
                signal.asset, side, size_usd, result.fill_price,
                result.method, signal.timeframe)

            logger.info("TRADE %s %s $%.0f @ %.4f (%s, %s, h=%d)",
                        side, signal.asset, size_usd, result.fill_price,
                        result.method, signal.timeframe, signal.hold_bars)

    async def _close_position(
        self, pos: Position, strategy: Optional[Strategy],
        reason: str, bar_time: datetime,
    ) -> None:
        """Close a position with atomic state persistence."""
        exit_side = "Sell" if pos.direction > 0 else "Buy"
        qty = pos.size_usd / pos.entry_price if pos.entry_price > 0 else 0

        result = await self._execution.execute_exit(pos.symbol, exit_side, qty)

        if result and result.filled:
            exit_price = result.fill_price
        else:
            ticker = await self._data.fetch_ticker(pos.symbol)
            exit_price = ticker.get("last_price", pos.entry_price)

        # Compute PnL
        if pos.direction > 0:
            pnl = pos.size_usd * (exit_price - pos.entry_price) / pos.entry_price
        else:
            pnl = pos.size_usd * (pos.entry_price - exit_price) / pos.entry_price

        self._equity += pnl
        if self._equity > self._peak_equity:
            self._peak_equity = self._equity

        # Atomic: close position + log trade + update checkpoint
        await self._state.atomic_close_position(
            pos.id, exit_price, bar_time, pnl,
            {"timestamp": bar_time.isoformat(), "asset": pos.asset,
             "symbol": pos.symbol, "side": exit_side, "size_usd": pos.size_usd,
             "strategy_id": pos.strategy_id, "timeframe": pos.timeframe},
            {"equity": str(self._equity), "peak_equity": str(self._peak_equity)},
        )

        # Update strategy state
        if strategy:
            strategy.on_trade(pos.asset, pnl, pos.size_usd)

        # Remove from open list
        self._open_positions = [p for p in self._open_positions if p.id != pos.id]

        await self._notifier.notify_exit(
            pos.asset, pos.side, pnl, pos.entry_price, exit_price)

        logger.info("EXIT %s %s PnL=$%.2f (%s) eq=$%.0f dd=%.1f%%",
                     pos.asset, reason, pnl, pos.strategy_id,
                     self._equity, self._drawdown * 100)

    async def _place_compliance_sl(
        self, symbol: str, entry_side: str, qty: float,
        fill_price: float, size_usd: float,
    ) -> None:
        """Place dynamic compliance SL — widest distance within 3% risk rule.

        sl_pct = min(max_risk * initial_balance / position_size, sl_cap)
        Uses INITIAL balance per HyroTrader rules (3% of initial, not current equity).
        """
        sl_pct = min(self._max_risk_per_trade * self._initial_balance / max(size_usd, 1),
                     self._sl_cap)
        if entry_side == "Buy":  # long position, SL below
            sl_price = fill_price * (1 - sl_pct)
            sl_side = "Sell"
        else:  # short position, SL above
            sl_price = fill_price * (1 + sl_pct)
            sl_side = "Buy"

        try:
            if hasattr(self._exchange, 'place_trading_stop'):
                await self._exchange.place_trading_stop(
                    symbol=symbol, side=entry_side, stop_loss=sl_price)
            elif hasattr(self._exchange, 'place_conditional_order'):
                await self._exchange.place_conditional_order(
                    symbol=symbol, side=sl_side, qty=qty,
                    trigger_price=sl_price, reduce_only=True)
            else:
                # In-engine SL monitoring handles this case (see _on_bar exit loop)
                logger.debug("Exchange SL not available for %s — in-engine SL active", symbol)
                return
            logger.info("Compliance SL placed: %s %s @ %.4f (%.0f%% from %.4f)",
                        sl_side, symbol, sl_price, sl_pct * 100, fill_price)
        except Exception as e:
            logger.error("Failed to place compliance SL for %s: %s", symbol, e)
            await self._notifier.notify_error(
                f"COMPLIANCE SL FAILED: {symbol} — place manually!")

    # ── Helpers ──

    def _get_universe(self) -> set[str]:
        """Get union of all tradeable assets from strategies + open positions."""
        universe = set()
        for strategy in self._strategies:
            universe.update(strategy.tradeable_universe)
            for pos in self._open_positions:
                if pos.strategy_id == strategy.name:
                    universe.add(pos.asset)
        if not universe:
            universe = {pos.asset for pos in self._open_positions}
        return universe

    def _get_strategy(self, strategy_id: str) -> Optional[Strategy]:
        for s in self._strategies:
            if s.name == strategy_id:
                return s
        return None

    def _portfolio_view(self) -> PortfolioView:
        """Create read-only portfolio snapshot for strategies."""
        n_by_asset = defaultdict(int)
        exp_by_asset = defaultdict(float)
        for p in self._open_positions:
            n_by_asset[p.asset] += 1
            exp_by_asset[p.asset] += p.size_usd

        return PortfolioView(
            equity=self._equity,
            peak_equity=self._peak_equity,
            drawdown=self._drawdown,
            open_positions=list(self._open_positions),
            n_positions_by_asset=dict(n_by_asset),
            exposure_by_asset=dict(exp_by_asset),
            total_exposure=sum(p.size_usd for p in self._open_positions),
        )

    @property
    def _drawdown(self) -> float:
        if self._peak_equity <= 0:
            return 0.0
        return (self._peak_equity - self._equity) / self._peak_equity

    async def _ensure_vol_series(self, asset: str) -> None:
        now = time.monotonic()
        cached_at = self._vol_cache_times.get(asset, 0)
        if asset in self._vol_series and (now - cached_at) < 86400:  # 24h TTL
            return
        try:
            df = await self._data.fetch_ohlcv(f"{asset}{self._quote_currency}", "1d", 200)
            if len(df) >= 30:
                self._vol_series[asset] = df["close"].pct_change().rolling(20).std()
                self._vol_cache_times[asset] = now
        except Exception:
            pass

    async def _save_state(self) -> None:
        await self._state.save_checkpoint("equity", str(self._equity))
        await self._state.save_checkpoint("peak_equity", str(self._peak_equity))
        for strategy in self._strategies:
            await self._state.save_strategy_state(strategy.name, strategy.get_state())

    async def _send_daily_summary(self, bar_time: datetime) -> None:
        date_str = bar_time.strftime("%Y-%m-%d")
        history = await self._state.get_equity_history()
        prev_equity = history[-1][1] if history else self._starting_capital
        daily_pnl = self._equity - prev_equity

        await self._state.save_daily_equity(
            date_str, self._equity, daily_pnl,
            self._daily_trades, self._daily_signals)

        await self._notifier.notify_daily_summary(
            date_str, self._equity, daily_pnl,
            self._daily_trades, len(self._open_positions),
            self._drawdown * 100)

        # TCA report (checks alerts automatically)
        tca_report = await self._monitor.generate_report(window_hours=24)
        if tca_report:
            logger.info("TCA: fill_rate=%.0f%% shortfall=%.1fbps p95=%.0fms",
                        tca_report.maker_fill_rate * 100,
                        tca_report.avg_shortfall_bps,
                        tca_report.latency_p95_ms)

    async def shutdown(self) -> None:
        self._running = False
        logger.info("Shutting down...")
        await self._save_state()
        await self._data.close()
        await self._exchange.close()
        await self._notifier.close()
        await self._state.close()
        logger.info("Shutdown complete")

    def _next_bar_time(self) -> datetime:
        now = datetime.now(tz=timezone.utc)
        interval = self._bar_interval_m
        minutes = now.minute
        next_min = ((minutes // interval) + 1) * interval
        next_bar = now.replace(second=0, microsecond=0)
        next_bar += timedelta(minutes=next_min - minutes)
        next_bar += timedelta(seconds=self._poll_offset_s)
        return next_bar

    @staticmethod
    def _tf_fires(tf_label: str, bar_time: datetime) -> bool:
        minutes = TF_MINUTES.get(tf_label, 15)
        total_minutes = bar_time.hour * 60 + bar_time.minute
        return total_minutes % minutes == 0
