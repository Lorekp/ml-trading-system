"""Paper trading engine — main orchestrator.

Mirrors the experiment's signal → sizing → execution flow in real-time.
Ticks every 15 minutes, checks which TFs have new bars, generates signals,
sizes positions, and executes via MkrTkr60 on Bybit demo.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import pandas as pd

from live.config import (
    LiveConfig, FULL5, NEEDED_TFS, TF_MINUTES, load_live_config,
)
from live.bybit_client import BybitClient
from live.signal_generator import SignalGenerator
from live.executor import MkrTkr60Executor
from live.breaker import F1SymmetricBreaker
from live.portfolio import PortfolioManager, Position
from live.asset_selector import S8AssetSelector
from live.state import StateDB
from live.notifier import TelegramNotifier

logger = logging.getLogger(__name__)


class PaperTradingEngine:
    """Main paper trading engine.

    Lifecycle:
        1. initialize() — load config, models, restore state
        2. run() — main loop (sleep → on_bar → repeat)
        3. shutdown() — save state, close connections
    """

    def __init__(self, config: LiveConfig) -> None:
        self._config = config
        self._client: Optional[BybitClient] = None
        self._signal_gen: Optional[SignalGenerator] = None
        self._executor: Optional[MkrTkr60Executor] = None
        self._breaker: Optional[F1SymmetricBreaker] = None
        self._portfolio: Optional[PortfolioManager] = None
        self._selector: Optional[S8AssetSelector] = None
        self._state: Optional[StateDB] = None
        self._notifier: Optional[TelegramNotifier] = None
        self._vol_series: dict[str, pd.Series] = {}
        self._running = False
        self._daily_trades = 0
        self._daily_signals = 0

    async def initialize(self) -> None:
        """Initialize all components and restore state."""
        cfg = self._config
        logger.info("Initializing paper trading engine...")

        # 1. Bybit client
        self._client = BybitClient.from_config(cfg.exchange)
        logger.info("Bybit client: %s (demo=%s)",
                     cfg.exchange.base_url, cfg.exchange.demo)

        # 2. State DB
        self._state = StateDB(cfg.db_path)
        await self._state.initialize()

        # 3. Signal generator
        self._signal_gen = SignalGenerator(cfg.model_dir)
        n_models = self._signal_gen.load_models()
        logger.info("Loaded %d models", n_models)

        # 4. F1 symmetric breaker
        self._breaker = F1SymmetricBreaker.from_config(cfg.breaker)
        breaker_state = await self._state.load_breaker_state()
        if breaker_state:
            self._breaker.restore_state(breaker_state)

        # 5. Portfolio manager
        self._portfolio = PortfolioManager(cfg.portfolio, cfg.capital,
                                           cfg.vol_overlay)

        # Restore equity from checkpoint
        saved_equity = await self._state.load_checkpoint("equity")
        if saved_equity:
            self._portfolio.equity = float(saved_equity)
            logger.info("Restored equity: $%.0f", self._portfolio.equity)

        saved_peak = await self._state.load_checkpoint("peak_equity")
        if saved_peak:
            self._portfolio.peak_equity = float(saved_peak)

        # Restore open positions
        open_positions = await self._state.get_open_positions()
        if open_positions:
            self._portfolio.restore_positions(open_positions)

        # 6. MkrTkr60 executor
        self._executor = MkrTkr60Executor.from_config(self._client, cfg.mkrtk60)

        # 7. Asset selector
        self._selector = S8AssetSelector.from_config(cfg.s8)

        # Restore universe from checkpoint
        saved_universe = await self._state.load_checkpoint("universe")
        if saved_universe:
            self._selector.set_universe(json.loads(saved_universe))
        else:
            # Initial universe = core + assets with models
            all_available = set()
            for tf_cfg in cfg.full5:
                all_available.update(self._signal_gen.available_assets(tf_cfg.tf))
            self._selector.set_universe(
                sorted(self._selector._core & all_available) +
                sorted(all_available - self._selector._core)[:cfg.s8.max_total - len(self._selector._core)]
            )

        # 8. Notifier
        self._notifier = TelegramNotifier.from_config(cfg.telegram)

        # 9. Set leverage for all symbols
        for asset in self._selector.universe:
            await self._client.set_leverage(f"{asset}USDT", cfg.leverage)

        logger.info("Engine initialized: %d assets, $%.0f equity, %d open positions",
                     len(self._selector.universe), self._portfolio.equity,
                     len(self._portfolio.open_positions))

        await self._notifier.notify_startup(
            self._portfolio.equity, n_models,
            len(self._selector.universe), cfg.exchange.demo)

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
                await self._notifier.notify_error(str(e))
                await asyncio.sleep(60)

    async def _on_bar(self, bar_time: datetime) -> None:
        """One 15-minute cycle — mirrors experiment signal flow."""
        t0 = time.monotonic()
        cfg = self._config

        # 0. Which TFs fire this bar?
        active_tfs = []
        for tf_cfg in cfg.full5:
            if self._tf_fires(tf_cfg.tf, bar_time):
                active_tfs.append(tf_cfg)

        if not active_tfs:
            return

        tf_labels = [tf.tf for tf in active_tfs]
        logger.info("Bar %s — active TFs: %s", bar_time.strftime("%H:%M"), tf_labels)

        # 1. Monthly rebalance check (S8)
        available = set()
        for tf_cfg in cfg.full5:
            available.update(self._signal_gen.available_assets(tf_cfg.tf))
        await self._selector.maybe_rebalance(bar_time, self._client, sorted(available))
        universe = self._selector.universe

        # 2. Check exits for open positions
        # CRITICAL: only increment bars_held when the position's OWN timeframe fires.
        # A 4h position with hold=6 expires after 6 × 4h bars, not 6 × 15m ticks.
        active_tf_labels = set(tf_labels)
        exits = 0
        for pos in list(self._portfolio.open_positions):
            if pos.timeframe in active_tf_labels:
                pos.bars_held += 1
                await self._state.update_bars_held(pos.id, pos.bars_held)

            if pos.bars_held >= pos.hold_bars:
                await self._close_position(pos, bar_time)
                exits += 1

        # 3. Fetch OHLCV for active TFs × universe
        ohlcv_cache: dict[tuple[str, str], pd.DataFrame] = {}
        for tf_cfg in active_tfs:
            for asset in universe:
                bars = await self._client.fetch_klines(
                    f"{asset}USDT", tf_cfg.resample, tf_cfg.lookback)
                if bars:
                    ohlcv_cache[(tf_cfg.tf, asset)] = self._bars_to_df(bars)

        # 4. Fetch 4h bars for regime filter
        ohlcv_4h: dict[str, pd.DataFrame] = {}
        for asset in universe:
            bars = await self._client.fetch_klines(f"{asset}USDT", "4h", 50)
            if bars:
                ohlcv_4h[asset] = self._bars_to_df(bars)

        # 5. Update vol series (for vol overlay)
        await self._update_vol_series(universe, bar_time)

        # 6. Generate signals
        signals = []
        for tf_cfg in active_tfs:
            for asset in universe:
                ohlcv = ohlcv_cache.get((tf_cfg.tf, asset))
                if ohlcv is None:
                    continue
                sig = self._signal_gen.generate(
                    asset, tf_cfg.tf, ohlcv, ohlcv_4h.get(asset), tf_cfg.hold)
                if sig:
                    signals.append(sig)
                    self._daily_signals += 1

        # 7. Size and execute
        trades = 0
        for sig in signals:
            vol_mult = self._portfolio.compute_vol_mult(sig.asset, self._vol_series)
            breaker_mult = self._breaker.get_multiplier(sig.asset)

            size_usd = self._portfolio.compute_size(
                asset=sig.asset,
                probability=sig.probability,
                signal_strength=sig.signal_strength,
                regime_mult=1.0,  # regime already filtered in signal_gen
                vol_mult=vol_mult,
                breaker_mult=breaker_mult,
            )

            if size_usd < 1.0:
                continue

            # Get current price for qty calculation
            ticker = await self._client.fetch_ticker(f"{sig.asset}USDT")
            current_price = ticker["last_price"]
            if current_price <= 0:
                continue

            qty = size_usd / current_price
            side = "Buy" if sig.direction > 0 else "Sell"

            # MkrTkr60 execution
            result = await self._executor.execute(f"{sig.asset}USDT", side, qty)

            if result and result.filled:
                # Record position
                pos = Position(
                    asset=sig.asset,
                    symbol=f"{sig.asset}USDT",
                    timeframe=sig.timeframe,
                    side="long" if sig.direction > 0 else "short",
                    direction=sig.direction,
                    entry_price=result.fill_price,
                    entry_time=bar_time,
                    size_usd=size_usd,
                    hold_bars=sig.hold_bars,
                    fill_method=result.method,
                    order_id=result.order_id,
                )
                self._portfolio.open_position(pos)
                await self._state.open_position(pos)
                await self._state.log_trade(
                    sig.asset, f"{sig.asset}USDT", side, "open",
                    result.fill_price, size_usd, result.method,
                    timeframe=sig.timeframe, order_id=result.order_id)

                await self._notifier.notify_trade(
                    sig.asset, side, size_usd, result.fill_price,
                    result.method, sig.timeframe)

                trades += 1
                self._daily_trades += 1

        # 8. Save state
        await self._save_state()

        # 9. Daily summary at midnight UTC
        if bar_time.hour == 0 and bar_time.minute < 15:
            await self._send_daily_summary(bar_time)
            self._daily_trades = 0
            self._daily_signals = 0

        elapsed = time.monotonic() - t0
        logger.info("Bar done: %d signals, %d trades, %d exits (%.1fs)",
                     len(signals), trades, exits, elapsed)

    async def _close_position(self, pos: Position, bar_time: datetime) -> None:
        """Close an expired position via market order."""
        exit_side = "Sell" if pos.direction > 0 else "Buy"
        qty = pos.size_usd / pos.entry_price if pos.entry_price > 0 else 0

        result = await self._executor.execute_exit(pos.symbol, exit_side, qty)

        if result and result.filled:
            exit_price = result.fill_price
        else:
            # Fallback: use last ticker price
            ticker = await self._client.fetch_ticker(pos.symbol)
            exit_price = ticker["last_price"]

        pnl = self._portfolio.close_position(pos, exit_price, bar_time)
        await self._state.close_position(pos.id, exit_price, bar_time, pnl)
        await self._state.log_trade(
            pos.asset, pos.symbol, exit_side, "close",
            exit_price, pos.size_usd, "taker", pnl=pnl,
            timeframe=pos.timeframe)

        # Update breaker
        self._breaker.record_trade(pos.asset, pnl, pos.size_usd)

        await self._notifier.notify_exit(
            pos.asset, pos.side, pnl, pos.entry_price, exit_price)

    async def _update_vol_series(self, universe: list[str],
                                 bar_time: datetime) -> None:
        """Populate / refresh the per-asset daily realized-vol series that
        drives the inverse-vol sizing overlay (PortfolioManager.compute_vol_mult).

        Robustness — the previous version silently failed open: it fetched
        each asset once per session, required >=30 daily bars, and swallowed
        every exception. Any asset whose fetch failed or which lacked daily
        history was simply absent from _vol_series, and compute_vol_mult then
        returned 1.0 (full notional). That sized DENT/PIPPIN at full risk and
        caused 97% of the live paper loss.

        This version:
          * refreshes once per UTC day (the 00:00 bar) so vol is not stale;
          * for new listings with <30 daily bars, derives a daily-vol estimate
            from hourly bars (hourly_vol * sqrt(24)) instead of skipping them;
          * logs fetch failures instead of swallowing them, and leaves the
            asset OUT of _vol_series so compute_vol_mult fails safe to the
            clip floor and the next bar retries.
        """
        force = bar_time.hour == 0 and bar_time.minute < 15
        for asset in universe:
            if asset in self._vol_series and not force:
                continue  # already have today's series

            # Primary: 20-day rolling std of daily returns (matches backtest).
            try:
                bars = await self._client.fetch_klines(f"{asset}USDT", "1d", 200)
            except Exception as e:
                logger.warning("vol fetch failed for %s (%s); will retry", asset, e)
                continue

            series: Optional[pd.Series] = None
            if bars and len(bars) >= 30:
                closes = pd.Series([b["close"] for b in bars])
                series = closes.pct_change().rolling(20).std()
            else:
                # New listing — derive daily vol from hourly bars so the
                # asset is still risk-scaled rather than sized at full size.
                try:
                    hbars = await self._client.fetch_klines(
                        f"{asset}USDT", "1h", 720)
                except Exception as e:
                    logger.warning("hourly vol fallback failed for %s (%s)",
                                   asset, e)
                    hbars = None
                if hbars and len(hbars) > 480:
                    hcloses = pd.Series([b["close"] for b in hbars])
                    hvol = hcloses.pct_change().rolling(480).std()
                    series = hvol * float(np.sqrt(24.0))  # hourly -> daily

            last = series.iloc[-1] if series is not None and len(series) else np.nan
            if series is None or not np.isfinite(last):
                logger.warning(
                    "no usable vol series for %s; compute_vol_mult fails "
                    "safe to floor (minimum size)", asset)
                continue
            self._vol_series[asset] = series

    async def _save_state(self) -> None:
        """Persist critical state to DB."""
        await self._state.save_checkpoint("equity", str(self._portfolio.equity))
        await self._state.save_checkpoint("peak_equity", str(self._portfolio.peak_equity))
        await self._state.save_checkpoint(
            "universe", json.dumps(self._selector.universe))
        await self._state.save_breaker_state(self._breaker.get_state())

    async def _send_daily_summary(self, bar_time: datetime) -> None:
        """Send daily summary notification and save equity."""
        date_str = bar_time.strftime("%Y-%m-%d")
        equity = self._portfolio.equity

        # Try to get actual balance from Bybit
        try:
            bybit_balance = await self._client.get_account_balance()
            if bybit_balance > 0:
                logger.info("Bybit balance: $%.0f (local: $%.0f)",
                            bybit_balance, equity)
        except Exception:
            pass

        # Compute daily PnL from equity history
        history = await self._state.get_equity_history()
        if history:
            prev_equity = history[-1][1]
            daily_pnl = equity - prev_equity
        else:
            daily_pnl = equity - self._config.capital

        await self._state.save_daily_equity(
            date_str, equity, daily_pnl,
            self._daily_trades, self._daily_signals)

        await self._notifier.notify_daily_summary(
            date_str, equity, daily_pnl,
            self._daily_trades, len(self._portfolio.open_positions),
            self._portfolio.drawdown * 100)

    async def shutdown(self) -> None:
        """Graceful shutdown: save state, close connections."""
        self._running = False
        logger.info("Shutting down...")

        await self._save_state()

        if self._client:
            await self._client.close()
        if self._notifier:
            await self._notifier.close()
        if self._state:
            await self._state.close()

        logger.info("Shutdown complete")

    # ── Helpers ──

    def _next_bar_time(self) -> datetime:
        """Compute next 15-minute bar boundary + offset."""
        now = datetime.now(tz=timezone.utc)
        interval = self._config.scheduler.bar_interval_m
        offset = self._config.scheduler.poll_offset_s

        # Round up to next bar boundary
        minutes = now.minute
        next_min = ((minutes // interval) + 1) * interval
        if next_min >= 60:
            next_bar = now.replace(minute=0, second=0, microsecond=0)
            next_bar = next_bar.replace(hour=now.hour + next_min // 60)
        else:
            next_bar = now.replace(minute=next_min, second=0, microsecond=0)

        # Add offset
        from datetime import timedelta
        next_bar += timedelta(seconds=offset)

        return next_bar

    @staticmethod
    def _tf_fires(tf_label: str, bar_time: datetime) -> bool:
        """Check if a timeframe has a new bar at this time."""
        minutes = TF_MINUTES.get(tf_label, 15)
        total_minutes = bar_time.hour * 60 + bar_time.minute
        return total_minutes % minutes == 0

    @staticmethod
    def _bars_to_df(bars: list[dict]) -> pd.DataFrame:
        """Convert list of bar dicts to OHLCV DataFrame."""
        if not bars:
            return pd.DataFrame()
        df = pd.DataFrame(bars)
        df.set_index("timestamp", inplace=True)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        return df
