"""Top-level TradingEngine — wires every service into a runnable async loop.

The engine is a single-process async coordinator that assembles:

- Deribit REST + WebSocket clients (Phase 1)
- Market data buffers + bar aggregator + bar clock (Phase 2)
- Live TSRV, SVI, ATM strike picker (Phase 3)
- Model bundle + feature assembler + predictor + LiveDRLHedger (Phase 4)
- Order manager + straddle flow + perp hedger (Phase 5)
- Equity + position tracker + SQLite store + risk + kill switch + telegram + metrics (Phase 6)

Public API::

    engine = TradingEngine(config)
    await engine.initialize()
    await engine.run()          # blocks until ``stop()`` is called
    await engine.shutdown()

For tests, the engine exposes :meth:`process_bar` so a driver can feed
synthetic :class:`Bar` objects through the full pipeline without
needing a live WebSocket.
"""

from __future__ import annotations

import asyncio
import logging
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deribit_paper.client import DeribitAuth, DeribitRESTClient, DeribitWebSocketClient
from deribit_paper.config import PaperConfig
from deribit_paper.execution import OrderManager, StraddleFlow
from deribit_paper.execution.idempotency import IdempotencyGuard
from deribit_paper.features import LiveSVICalibrator, StrikePicker, compute_bar_tsrv
from deribit_paper.market_data import (
    Bar,
    BarAggregator,
    BarClock,
    IndexPriceBuffer,
    MarketDataSubscriber,
    TradeBuffer,
)
from deribit_paper.monitoring import MetricsSink, TelegramNotifier, setup_json_logging
from deribit_paper.monitoring.harness_writer import HarnessWriter
from deribit_paper.position import EquityTracker, PositionTracker, SQLiteStore
from deribit_paper.risk.dry_run import get_dry_run
from deribit_paper.risk.kill_switch import KillSwitch
from deribit_paper.risk.limits import RiskLimits
from deribit_paper.risk.rnd_monitor import RNDConfig, RNDMonitor
from deribit_paper.signal import (
    FeatureAssembler,
    LiveDRLHedger,
    Predictor,
    load_model_bundle,
)

logger = logging.getLogger(__name__)


@dataclass
class EngineStats:
    """Running stats exposed for telegram heartbeat + diagnostics."""
    bars_processed: int = 0
    signals_generated: int = 0
    signals_long: int = 0
    signals_short: int = 0
    signals_flat: int = 0
    signals_rejected_by_risk: int = 0
    signals_executed: int = 0
    trades_completed: int = 0
    trades_aborted: int = 0
    last_bar_timestamp: pd.Timestamp | None = None
    last_signal_reason: str = ""
    started_at: pd.Timestamp | None = None


class TradingEngine:
    """Single-process async coordinator for the paper-trading loop."""

    def __init__(self, config: PaperConfig):
        self.config = config
        self.config.ensure_dirs()

        # Service handles — populated by :meth:`initialize`
        self.auth: DeribitAuth | None = None
        self.rest: DeribitRESTClient | None = None
        self.ws: DeribitWebSocketClient | None = None

        self.perp_buffer: TradeBuffer | None = None
        self.option_buffer: TradeBuffer | None = None
        self.index_buffer: IndexPriceBuffer | None = None
        self.subscriber: MarketDataSubscriber | None = None
        self.bar_clock: BarClock | None = None
        self.aggregator: BarAggregator | None = None

        self.svi_calibrator: LiveSVICalibrator | None = None
        self.strike_picker: StrikePicker | None = None

        self.model_bundle: Any = None
        self.feature_assembler: FeatureAssembler | None = None
        self.predictor: Predictor | None = None
        self.live_drl_hedger: LiveDRLHedger | None = None

        self.order_manager: OrderManager | None = None
        self.straddle_flow: StraddleFlow | None = None
        self.vpin_stream: Any = None  # LiveVPINStream (lazy import)

        self.equity_tracker: EquityTracker | None = None
        self.position_tracker: PositionTracker | None = None
        self.store: SQLiteStore | None = None

        self.risk_limits: RiskLimits | None = None
        self.kill_switch: KillSwitch | None = None
        self.rnd_monitor: RNDMonitor | None = None

        self.telegram: TelegramNotifier | None = None
        self.metrics_sink: MetricsSink | None = None

        self.stats = EngineStats()
        self._stop = asyncio.Event()
        self._background_tasks: list[asyncio.Task] = []

        # Execution cost EMA for liquidity-conditioned sizing
        self._exec_cost_ema: float = 12.0  # empirical median from backtester data
        self._exec_cost_alpha: float = 2.0 / (config.execution_cost_ema_bars + 1)

        # Kelly K1 sidecar (loaded once at initialize). None disables Kelly
        # boost — every per-bar lookup returns 1.0 via lookup_kelly_mult_at.
        self._kelly_mult_series: Any = None  # pd.Series | None

        # bar_execution_quality slip series (V2 wiring, C4 2026-04-26).
        # None falls back to ``cfg.max_slippage_pct`` flat cap (legacy
        # live default 0.5%).
        self._bar_slip_series: Any = None  # pd.Series | None

        # Harness-grade trade-row writer (V2 wiring, C6 2026-04-26).
        # Appends one row per closed trade to
        # vol_arb/output/live_trades_harness.parquet. Used by I4 to
        # diff live PnL vs the V2 backtest replay over the same window.
        self._harness_writer = HarnessWriter(
            "vol_arb/output/live_trades_harness.parquet",
        )

        # Rolling edge history for composite_v2 edge_scale (1 day of 15m bars)
        self._edge_history: deque[float] = deque(maxlen=96)

        # Stale-data guard: consecutive bars with zero trades
        self._consecutive_empty_bars: int = 0

        # SVI recalibration failure tracking (R2)
        self._svi_recal_failures: int = 0

        # Equity reconciliation counter (R3)
        self._bars_since_equity_reconcile: int = 0

        # VPIN staleness alert already sent (C4) — avoids spam
        self._vpin_stale_alerted: bool = False

        # Pause/drain modes
        self._paused: bool = False
        self._draining: bool = False

    # ── Initialisation ─────────────────────────────────────────────

    async def initialize(self) -> None:
        """Construct every service and authenticate with Deribit."""
        cfg = self.config
        setup_json_logging(logs_dir=cfg.logs_dir, level=cfg.log_level)
        logger.info(
            "TradingEngine initialising venue=%s dry_run=%s",
            cfg.venue, get_dry_run(),
        )

        # ── Phase 1: Deribit client ──
        if not cfg.client_id or not cfg.client_secret:
            raise RuntimeError(
                "Deribit credentials missing — set DERIBIT_TESTNET_CLIENT_ID + "
                "_SECRET in .env before running the engine",
            )
        self.auth = DeribitAuth(cfg.client_id, cfg.client_secret, cfg.rest_base_url)
        self.rest = DeribitRESTClient(self.auth, cfg.rest_base_url)
        await self.rest.__aenter__()  # opens the httpx session

        # Sanity-check: authenticate + fetch account
        summary = await self.rest.get_account_summary(currency=cfg.currency)
        account_equity_btc = float(summary.get("equity", 0.0) or 0.0)
        logger.info(
            "Deribit auth OK — account equity %.4f %s",
            account_equity_btc, cfg.currency,
        )

        # ── Phase 2: Market data buffers + WebSocket ──
        self.perp_buffer = TradeBuffer(
            name="perp",
            window_seconds=cfg.perp_buffer_hours * 3600.0,
        )
        self.option_buffer = TradeBuffer(
            name="option",
            window_seconds=cfg.option_buffer_hours * 3600.0,
        )
        self.index_buffer = IndexPriceBuffer(window_seconds=3600.0)

        def _on_ws_reconnect() -> None:
            if self.telegram is not None:
                self.telegram.send(
                    "error",
                    "Deribit WS reconnected \u2014 channels resubscribed",
                )

        self.ws = DeribitWebSocketClient(
            cfg.ws_url, auth=self.auth, on_reconnect=_on_ws_reconnect,
        )
        self.subscriber = MarketDataSubscriber(
            ws=self.ws,
            perp_buffer=self.perp_buffer,
            option_buffer=self.option_buffer,
            index_buffer=self.index_buffer,
            currency=cfg.currency,
            dte_min_days=0.0,
            dte_max_days=21.0,
        )

        self.bar_clock = BarClock(bar_minutes=cfg.bar_minutes)
        self.aggregator = BarAggregator(
            bar_minutes=cfg.bar_minutes,
            perp_buffer=self.perp_buffer,
            option_buffer=self.option_buffer,
            index_buffer=self.index_buffer,
        )
        self.aggregator.subscribe(self._on_bar_close)
        self.bar_clock.add_callback(self.aggregator.on_bar_close)

        # ── Phase 3: Feature pipeline ──
        self.svi_calibrator = LiveSVICalibrator(
            cache_path=Path(cfg.state_dir) / "svi_cache.parquet",
            recalibrate_hours=cfg.svi_recalibrate_hours,
        )
        self.svi_calibrator.load_cache()  # warm from disk if present

        self.strike_picker = StrikePicker(
            rest=self.rest,
            currency=cfg.currency,
            dte_min_days=cfg.dte_min_days,
            dte_max_days=cfg.dte_max_days,
        )

        # ── Phase 4: Models + predictor ──
        self.model_bundle = load_model_bundle(
            lgbm_models_dir=cfg.lgbm_models_dir,
            transvol_models_dir=cfg.transvol_models_dir,
            drl_policy_path=cfg.drl_policy_path,
            drl_vecnormalize_path=cfg.drl_vecnormalize_path,
        )
        # Historical baseline: merged base-features + SVI-features
        # parquets, loaded once at startup. Manifest-driven dispatch
        # (W3.1) validates every feature is producible from either the
        # TransVol slots, a live handler, or the baseline — and raises
        # at construction time if any are missing.
        features_parquet = Path(cfg.features_parquet)
        if not features_parquet.exists():
            raise FileNotFoundError(
                f"Historical features parquet not found at {features_parquet} — "
                "re-run the batch feature engineering pipeline before starting "
                "the live engine",
            )
        svi_features_parquet = Path(cfg.svi_features_parquet)
        if not svi_features_parquet.exists():
            raise FileNotFoundError(
                f"SVI features parquet not found at {svi_features_parquet} — "
                "re-run vol_arb/run_pipeline.py --svi to regenerate the "
                "constant-maturity SVI features the T2.1 models depend on",
            )
        self.feature_assembler = FeatureAssembler(
            features_parquet=features_parquet,
            svi_features_parquet=svi_features_parquet,
            bundle=self.model_bundle,
            svi_calibrator=self.svi_calibrator,
            rolling_history_bars=cfg.bars_per_day * 7,
        )
        # ── V2 sidecar loads (aligner β* + Kelly K1) ──
        # Aligner β*: per-month QLIKE-optimal scalar bias on log-vol
        # predictions (C2 2026-04-26). Empty dict → predictor passes raw
        # LGBM output through. Kelly K1: per-bar upside-only multiplier
        # in [1.0, 2.0] keyed by 15m UTC bar timestamp (C3 2026-04-26).
        # Both are no-op fallbacks when their config paths are empty,
        # so the ``default`` preset stays byte-id with prior live runs.
        from vol_arb.sidecars import (
            load_aligner_betas as _load_betas,
            load_bar_exec_slip_series as _load_slip,
            load_kelly_mult_series as _load_kelly,
        )
        aligner_betas = (
            _load_betas(cfg.aligner_betas_json) if cfg.aligner_betas_json else {}
        )
        if aligner_betas:
            logger.info(
                "Aligner β* loaded: %d months, %s..%s",
                len(aligner_betas),
                min(aligner_betas.keys()), max(aligner_betas.keys()),
            )
        self._kelly_mult_series = (
            _load_kelly(cfg.kelly_mult_path) if cfg.kelly_mult_path else None
        )
        self._bar_slip_series = (
            _load_slip(cfg.bar_exec_quality_path)
            if cfg.bar_exec_quality_path else None
        )
        self.predictor = Predictor(
            feature_assembler=self.feature_assembler,
            svi_calibrator=self.svi_calibrator,
            entry_threshold_vol_pts=cfg.entry_threshold,
            entry_threshold_long_vol_pts=cfg.entry_threshold_long,
            base_spread_iv=cfg.base_spread_iv,
            spread_jump_k=cfg.spread_jump_k,
            max_jump_fraction_for_shorts=cfg.max_jump_fraction_for_shorts,
            skip_long_if_term_slope_gt=cfg.skip_long_if_term_slope_gt,
            aligner_betas=aligner_betas,
            log_every_bar=True,
        )
        self.live_drl_hedger = LiveDRLHedger(
            bundle=self.model_bundle,
            n_steps_per_episode=cfg.intra_bar_hedge_ticks,
            bar_minutes=cfg.bar_minutes,
        )

        # ── Phase 4b: Live VPIN stream ──
        from deribit_paper.features.vpin_live import LiveVPINStream
        if cfg.vpin_state_path:
            self.vpin_stream = LiveVPINStream(
                state_path=Path(cfg.vpin_state_path),
                fallback_vpin=cfg.vpin_fallback,
                ws_url=cfg.binance_ws_url,
            )
            logger.info(
                "LiveVPINStream ready (warm-start=%s, fallback=%.3f)",
                cfg.vpin_state_path, cfg.vpin_fallback,
            )
        else:
            logger.info("LiveVPINStream disabled (vpin_state_path empty)")

        # ── Phase 5: Execution ──
        self.order_manager = OrderManager(
            rest=self.rest,
            idempotency_guard=IdempotencyGuard(ttl_seconds=60.0),
            fill_poll_seconds=1.5,
        )
        self.straddle_flow = StraddleFlow(
            order_manager=self.order_manager,
            live_drl_hedger=self.live_drl_hedger,
            perp_instrument=f"{cfg.currency}-PERPETUAL",
            holding_minutes=cfg.holding_minutes,
            target_margin_utilization=cfg.target_margin_utilization,
            im_fraction=cfg.im_fraction,
            max_participation_rate=cfg.max_participation_rate,
            short_sizing_penalty=cfg.short_sizing_penalty,
            max_trade_size_btc=cfg.max_single_trade_btc,
            n_hedge_ticks=cfg.intra_bar_hedge_ticks,
            spot_fetcher=self._current_spot,
            vpin_provider=(
                (lambda: self.vpin_stream.latest_vpin)
                if self.vpin_stream is not None else None
            ),
            maker_wait_seconds=float(cfg.maker_wait_seconds),
            baseline_target_dvol=cfg.baseline_target_dvol,
            sizing_edge_cap=cfg.sizing_edge_cap,
            sizing_dd_floor=cfg.sizing_dd_floor,
            min_trade_floor=cfg.min_trade_floor,
            on_position_opened=self._on_position_opened,
            on_position_closed=self._on_position_closed,
        )

        # ── Phase 6: State + risk + monitoring ──
        # Equity: seed from Deribit's reported balance, converted to USD
        # via the index price.  On subsequent bars we track via realized PnL.
        index_info = await self.rest.get_index_price(index_name=f"{cfg.currency.lower()}_usd")
        index_price_usd = float(index_info.get("index_price", 0.0) or 0.0)
        initial_equity_usd = (
            account_equity_btc * index_price_usd
            if account_equity_btc > 0 and index_price_usd > 0
            else cfg.initial_equity * 60_000.0  # fallback to 1 BTC × $60k
        )
        self.equity_tracker = EquityTracker(initial_equity_usd=initial_equity_usd)
        self.position_tracker = PositionTracker()
        self.store = SQLiteStore(cfg.sqlite_db)
        await self.store.open()

        self.telegram = TelegramNotifier(
            bot_token=cfg.telegram_bot_token,
            chat_id=cfg.telegram_chat_id,
            enabled=bool(cfg.telegram_bot_token and cfg.telegram_chat_id),
        )
        await self.telegram.start()
        self.metrics_sink = MetricsSink(path=cfg.metrics_parquet, flush_every=20)

        self.risk_limits = RiskLimits(
            position_tracker=self.position_tracker,
            equity_tracker=self.equity_tracker,
            max_open_positions=cfg.max_open_positions,
            max_margin_util_hard=cfg.max_margin_util_hard,
            max_single_trade_btc=cfg.max_single_trade_btc,
            min_single_trade_btc=cfg.min_trade_size_btc if hasattr(cfg, "min_trade_size_btc") else 0.1,
            dte_min_days=cfg.dte_min_days,
            dte_max_days=cfg.dte_max_days,
            im_fraction=cfg.im_fraction,
        )
        self.kill_switch = KillSwitch(
            equity_tracker=self.equity_tracker,
            max_daily_drawdown_pct=cfg.kill_switch_daily_dd_pct,
            max_consecutive_errors=cfg.kill_switch_consecutive_errors,
            network_outage_seconds=cfg.network_outage_seconds,
            alert_callback=lambda reason: self.telegram.kill_switch(reason) if self.telegram else None,
        )
        # Wire kill switch into the subscriber so mark_ws_message is called
        # on actual WS data arrival, not on bar ticks.
        self.subscriber.set_kill_switch(self.kill_switch)

        # RND-based tail-risk monitor (T3.2). Uses defaults — tail_move=5%,
        # 30-day baseline window, halt at 2× rolling median tail prob, size
        # scaling from 1.0 down to 0.25 as tail prob approaches the halt
        # threshold. The monitor is fed at each bar close from the
        # LiveSVICalibrator's cached params, and checked before every
        # signal decision alongside the kill switch.
        self.rnd_monitor = RNDMonitor(config=RNDConfig(bar_minutes=cfg.bar_minutes))

        # ── Startup position reconciliation (C1) ──
        # Query exchange for any open positions left by a previous crash/restart.
        # If found, log + alert + attempt to close orphaned positions.
        await self._reconcile_positions_on_startup()

        # ── Startup self-test: catch broken models/params before bar loop ──
        try:
            # Verify order construction produces valid API params
            from deribit_paper.execution.order_manager import _build_order_params
        except ImportError:
            pass  # _build_order_params may not exist as standalone
        # Verify the signal pipeline can produce a prediction
        if hasattr(self, "predictor") and self.predictor is not None:
            logger.info("Startup self-test: signal pipeline...")
            # (actual prediction requires bar data; model smoke test is in model_loader)
        logger.info("Startup self-test: passed")

        self.stats.started_at = pd.Timestamp.now(tz="UTC")
        logger.info(
            "TradingEngine initialised: equity=$%.2f, model=%s, dry_run=%s",
            initial_equity_usd, self.model_bundle.month_key, get_dry_run(),
        )
        if self.telegram.enabled:
            # Detect cold start vs restart from last_shutdown.json
            start_type = "COLD START"
            try:
                import json as _json
                shutdown_file = Path(cfg.state_dir) / "last_shutdown.json"
                if shutdown_file.exists():
                    last = _json.loads(shutdown_file.read_text())
                    last_ts = pd.Timestamp(last.get("timestamp", ""))
                    age = (pd.Timestamp.now(tz="UTC") - last_ts).total_seconds()
                    if age < 600:  # less than 10 minutes → restart
                        start_type = f"RESTART (down {age:.0f}s)"
                    else:
                        start_type = f"START (last shutdown {age / 3600:.1f}h ago)"
            except Exception:
                pass
            self.telegram.send(
                "heartbeat",
                f"{start_type}: equity ${initial_equity_usd:,.0f}, "
                f"model {self.model_bundle.month_key}, dry_run={get_dry_run()}",
            )



    # ── Background tasks ───────────────────────────────────────────

    async def run(self) -> None:
        """Start the main loop.  Blocks until :meth:`stop` is called."""
        if self.ws is None or self.bar_clock is None:
            raise RuntimeError("run() called before initialize()")

        # Subscribe to market data channels
        await self.subscriber.start()

        # Spawn background tasks
        self._background_tasks = [
            asyncio.create_task(self.ws.dispatch_forever(), name="ws_dispatch"),
            asyncio.create_task(self.bar_clock.run_forever(), name="bar_clock"),
            asyncio.create_task(self.telegram.flush_loop(), name="telegram_flush"),
            asyncio.create_task(self._heartbeat_loop(), name="heartbeat"),
        ]
        if self.vpin_stream is not None:
            self._background_tasks.append(
                asyncio.create_task(self.vpin_stream.run_forever(), name="vpin_ws"),
            )

        logger.info("TradingEngine running — waiting for stop signal")
        await self._stop.wait()
        logger.info("TradingEngine received stop — shutting down")

    async def _heartbeat_loop(self) -> None:
        """Periodic Telegram heartbeat for unattended monitoring.

        Fires every ``config.heartbeat_minutes`` (default 60).  Includes
        equity, drawdown, open positions, VPIN, and bar count so a
        silent engine is distinguishable from a crashed engine.
        """
        interval = self.config.heartbeat_minutes * 60
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
                return  # stop signaled during sleep
            except asyncio.TimeoutError:
                pass  # interval elapsed — send heartbeat
            if self.telegram is None or not self.telegram.enabled:
                continue
            # C4: Check VPIN staleness and alert once
            vpin_stale = (
                self.vpin_stream is not None and self.vpin_stream.is_stale
            )
            if vpin_stale and not self._vpin_stale_alerted:
                self._vpin_stale_alerted = True
                self.telegram.send(
                    "error",
                    "VPIN feed STALE — Binance WS may be disconnected. "
                    "DRL hedger using fallback VPIN. Check connectivity.",
                )
            elif not vpin_stale and self._vpin_stale_alerted:
                self._vpin_stale_alerted = False  # recovered

            # R4: Enriched heartbeat
            svi_age_hours = (
                self.svi_calibrator.stale_age_hours(pd.Timestamp.now(tz="UTC"))
                if self.svi_calibrator is not None else None
            )
            self.telegram.heartbeat(
                n_open=(
                    self.position_tracker.n_open
                    if self.position_tracker else 0
                ),
                equity_usd=(
                    self.equity_tracker.current_equity
                    if self.equity_tracker else 0.0
                ),
                daily_pnl_pct=(
                    self.equity_tracker.total_return_pct
                    if self.equity_tracker else 0.0
                ),
                drawdown_pct=(
                    self.equity_tracker.drawdown_pct
                    if self.equity_tracker else 0.0
                ),
                vpin=(
                    self.vpin_stream.latest_vpin
                    if self.vpin_stream else None
                ),
                vpin_connected=(
                    self.vpin_stream.is_connected
                    if self.vpin_stream else None
                ),
                bars_processed=self.stats.bars_processed,
                svi_age_hours=svi_age_hours,
                vpin_stale=vpin_stale,
                consecutive_empty_bars=self._consecutive_empty_bars,
                kill_switch_halted=self.kill_switch.is_halted() if self.kill_switch else False,
                svi_recal_failures=self._svi_recal_failures,
                paused=self._paused,
                draining=self._draining,
            )

    def stop(self) -> None:
        self._stop.set()
        if self.bar_clock is not None:
            self.bar_clock.stop()

    def pause(self) -> None:
        """Stop new entries but keep engine running.  Existing trades continue."""
        if self._paused:
            return
        self._paused = True
        logger.info("Engine PAUSED — no new trades until resumed")
        if self.telegram is not None:
            self.telegram.send(
                "heartbeat",
                "PAUSED: no new entries. Existing trades continue. "
                "Delete state/pause file to resume.",
            )

    def resume(self) -> None:
        """Resume normal trading after pause."""
        if not self._paused:
            return
        self._paused = False
        logger.info("Engine RESUMED — normal trading restored")
        if self.telegram is not None:
            self.telegram.send("heartbeat", "RESUMED: normal trading restored")

    def drain(self) -> None:
        """Stop new trades, let existing ones finish, then stop the engine."""
        if self._draining:
            return
        self._draining = True
        logger.info("Engine entering DRAIN mode — no new trades, will stop when empty")
        if self.telegram is not None:
            self.telegram.send(
                "heartbeat",
                "DRAIN mode: finishing open trades, no new entries. "
                "Engine will stop when all trades complete.",
            )

    @property
    def is_paused(self) -> bool:
        return self._paused

    @property
    def is_draining(self) -> bool:
        return self._draining

    def _check_sentinel_files(self) -> None:
        """Check for pause/drain sentinel files in the state directory.

        - ``state/pause`` exists → enter pause mode; delete → resume
        - ``state/drain`` exists → enter drain mode; consumed once
        """
        state_dir = Path(self.config.state_dir)
        pause_file = state_dir / "pause"
        drain_file = state_dir / "drain"

        if pause_file.exists():
            if not self._paused:
                self.pause()
        elif self._paused:
            self.resume()

        if drain_file.exists():
            if not self._draining:
                self.drain()
            # Remove drain file so it doesn't trigger again on restart
            try:
                drain_file.unlink()
            except OSError:
                pass

    async def shutdown(self) -> None:
        """Clean teardown of all services."""
        logger.info("TradingEngine shutdown sequence starting")
        self._stop.set()

        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        for task in self._background_tasks:
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self._background_tasks.clear()

        if self.vpin_stream is not None:
            await self.vpin_stream.stop()
        if self.metrics_sink is not None:
            self.metrics_sink.flush()
        if self.telegram is not None:
            await self.telegram.stop()
        if self.ws is not None:
            await self.ws.close()
        if self.rest is not None:
            await self.rest.close()
        if self.auth is not None:
            await self.auth.close()
        if self.store is not None:
            await self.store.close()

        # Write shutdown timestamp for cold-start vs restart detection
        try:
            import json
            shutdown_file = Path(self.config.state_dir) / "last_shutdown.json"
            shutdown_file.parent.mkdir(parents=True, exist_ok=True)
            shutdown_file.write_text(json.dumps({
                "timestamp": pd.Timestamp.now(tz="UTC").isoformat(),
                "bars_processed": self.stats.bars_processed,
                "trades_completed": self.stats.trades_completed,
            }))
        except Exception:
            pass  # non-critical

        logger.info("TradingEngine shutdown complete")

    # ── Bar-close handler (driven by BarClock via BarAggregator) ──

    async def _on_bar_close(self, bar: Bar) -> None:
        """Main per-bar logic — signal → risk → execute → persist."""
        self.stats.bars_processed += 1
        self.stats.last_bar_timestamp = bar.timestamp
        logger.info(
            "Bar %s: perp=%d opt=%d idx=%s",
            bar.timestamp, bar.n_perp_trades, bar.n_option_trades,
            f"{bar.index_price:.2f}" if bar.index_price else "None",
        )

        try:
            await self.process_bar(bar)
        except Exception as exc:  # noqa: BLE001 - log + continue next bar
            logger.exception("process_bar raised: %s", exc)
            if self.telegram is not None:
                self.telegram.error(f"bar {bar.timestamp} failed: {exc}")
            self.kill_switch.on_trade_result(success=False)

    async def process_bar(self, bar: Bar) -> None:
        """Run the signal → risk → execute pipeline for one bar.

        Exposed as a public method so tests can drive the engine with
        synthetic bars without spinning up the WebSocket.
        """
        # 1. Compute fresh TSRV for the bar
        bar_tsrv = compute_bar_tsrv(
            bar.perp_dataframe(),
            bar_seconds=float(self.config.bar_seconds),
        )

        # 1b. Stale-data guard — refuse to signal on empty/stale bars
        if bar.n_perp_trades == 0 and bar.n_option_trades == 0:
            self._consecutive_empty_bars += 1
            logger.warning(
                "Bar %s has no trade data (empty bar %d/%d), skipping signal",
                bar.timestamp, self._consecutive_empty_bars,
                self.config.max_consecutive_empty_bars,
            )
            if (
                self._consecutive_empty_bars >= self.config.max_consecutive_empty_bars
                and not self.kill_switch.is_halted()
            ):
                reason = (
                    f"data stale: {self._consecutive_empty_bars} consecutive empty bars "
                    f"(threshold {self.config.max_consecutive_empty_bars})"
                )
                self.kill_switch.trip(reason)
            return
        else:
            self._consecutive_empty_bars = 0

        # 1c. Low-volume guard — skip signal if insufficient perp data for features
        if bar.n_perp_trades < self.config.min_perp_trades_per_bar:
            logger.warning(
                "Bar %s has only %d perp trades (min %d), skipping signal",
                bar.timestamp, bar.n_perp_trades,
                self.config.min_perp_trades_per_bar,
            )
            return

        # 2. Kill switch gating
        if self.kill_switch.check_before_signal(now=bar.timestamp):
            logger.warning(
                "KillSwitch halted, skipping bar %s: %s",
                bar.timestamp, self.kill_switch.state.reason,
            )
            # C2: Emergency close on first detection of kill switch trip
            if self.kill_switch.needs_emergency_close:
                self.kill_switch.needs_emergency_close = False
                asyncio.create_task(
                    self._emergency_close_all(self.kill_switch.state.reason or "kill switch"),
                    name="emergency_close",
                )
            return

        # 2a. Sentinel file check for pause/drain
        self._check_sentinel_files()

        # 2b. Pause mode — skip signal but keep engine running
        if self._paused:
            logger.debug("Engine paused, skipping signal at %s", bar.timestamp)
            return

        # 2c. Drain mode — skip signal, auto-stop when no open trades
        if self._draining:
            n_open = self.position_tracker.n_open if self.position_tracker else 0
            active_tasks = sum(
                1 for t in self._background_tasks
                if not t.done() and (t.get_name() or "").startswith(("trade_", "resume_"))
            )
            if n_open == 0 and active_tasks == 0:
                logger.info("Drain complete — all trades finished. Stopping.")
                self.stop()
            else:
                logger.debug(
                    "Draining: %d open positions, %d active trade tasks",
                    n_open, active_tasks,
                )
            return

        # 2d. RND tail-risk monitor — update from the current SVI cache,
        # then check the halt rule.  Uses the SAME ``_params_df`` the
        # SVI calibrator computed from the last recalibration (not the
        # incoming bar's trades), so it's always one-recalibration-behind.
        # For a monitor that triggers on regime shifts rather than
        # bar-by-bar noise, that lag is fine.
        if self.rnd_monitor is not None and self.svi_calibrator is not None:
            try:
                index_price_for_rnd = bar.index_price or (
                    self.index_buffer.latest() if self.index_buffer else None
                )
                if index_price_for_rnd is not None:
                    self.rnd_monitor.update(
                        timestamp=bar.timestamp,
                        spot=float(index_price_for_rnd),
                        params_df=self.svi_calibrator._params_df,
                    )
            except Exception:  # noqa: BLE001 - monitor is advisory, never block
                logger.exception("RND monitor update failed — continuing")

            if self.rnd_monitor.should_halt():
                logger.warning(
                    "RND monitor halted, skipping bar %s: %s",
                    bar.timestamp, self.rnd_monitor.halt_reason(),
                )
                return

        # 3. Scheduled SVI recalibration (fire-and-forget background)
        if self.svi_calibrator.needs_refresh(bar.timestamp):
            logger.info("SVI recalibration scheduled at %s", bar.timestamp)
            trades_df = self.option_buffer.to_dataframe() if self.option_buffer else pd.DataFrame()
            if not trades_df.empty:
                # Run in background thread to avoid blocking the bar loop
                asyncio.create_task(
                    self._recalibrate_svi_task(trades_df, bar.timestamp),
                    name="svi_recalibrate",
                )

        # 3b. SVI staleness check (C3): refuse to trade on stale IV
        if self.svi_calibrator is not None and self.svi_calibrator.is_stale(
            bar.timestamp, max_stale_hours=self.config.svi_max_stale_hours,
        ):
            stale_msg = (
                f"SVI stale at {bar.timestamp}: last calibration "
                f"{self.svi_calibrator.stale_age_hours(bar.timestamp):.1f}h ago "
                f"(max {self.config.svi_max_stale_hours}h) — skipping signal"
            )
            logger.warning(stale_msg)
            if self.telegram is not None:
                self.telegram.send("error", stale_msg)
            return

        # 3c. Periodic equity reconciliation (R3)
        self._bars_since_equity_reconcile += 1
        if self._bars_since_equity_reconcile >= self.config.equity_reconcile_bars:
            self._bars_since_equity_reconcile = 0
            asyncio.create_task(
                self._reconcile_equity(),
                name="equity_reconcile",
            )

        # 4. Pick ATM straddle
        index_price = bar.index_price or (
            self.index_buffer.latest() if self.index_buffer else None
        )
        if index_price is None:
            logger.warning("Bar %s has no index price, skipping signal", bar.timestamp)
            return

        try:
            straddle = await self.strike_picker.pick_atm_straddle(
                index_price=index_price, now=bar.timestamp,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Strike picker failed: %s", exc)
            straddle = None

        # 5. Run predictor
        signal = self.predictor.predict(
            bar=bar, bar_tsrv=bar_tsrv, straddle=straddle,
            transvol_embedding=None,  # fresh TransVol is Phase 4B follow-up
        )
        self.stats.signals_generated += 1
        self.stats.last_signal_reason = signal.reason
        if signal.direction == 1:
            self.stats.signals_long += 1
        elif signal.direction == -1:
            self.stats.signals_short += 1
        else:
            self.stats.signals_flat += 1

        # Record per-bar metrics row
        self.metrics_sink.record({
            "timestamp": bar.timestamp,
            "signal": signal.direction,
            "edge_vol_pts": signal.edge_vol_pts,
            "rv_pred_pct": signal.rv_pred_pct,
            "iv_used_pct": signal.iv_used_pct,
            "index_price": index_price,
            "vol_ann": bar_tsrv.vol_ann_continuous,
            "jump_fraction": bar_tsrv.jump_fraction,
            "equity_usd": self.equity_tracker.current_equity,
            "drawdown_pct": self.equity_tracker.drawdown_pct,
        })

        # Update rolling edge history for composite_v2 sizing (all non-NaN edges)
        if math.isfinite(signal.edge_vol_pts):
            self._edge_history.append(abs(signal.edge_vol_pts))
        rolling_median_edge = float(np.median(list(self._edge_history))) if self._edge_history else 1.0

        # DVOL proxy: SVI ATM IV when calibrated, else realized vol from TSRV
        iv = signal.iv_used_pct
        dvol_pct: float | None = (
            float(iv) if (iv is not None and math.isfinite(iv) and iv > 0)
            else (bar_tsrv.vol_ann_continuous * 100.0 if bar_tsrv.vol_ann_continuous else None)
        )

        if signal.direction == 0:
            logger.info("Flat signal at %s: %s", bar.timestamp, signal.reason)
            return

        # 6. Size position + risk check
        trailing_vol = self._trailing_1h_volume_btc()
        # Per-bar Kelly K1 lookup (V2 wiring, C3 2026-04-26). Returns 1.0
        # for shorts (sidecar puts mult=1.0 by construction), bars
        # outside sidecar coverage, and when Kelly is disabled.
        # Per-bar empirical slip USD/BTC (V2 wiring, C4 2026-04-26)
        # converted to a max_slippage_pct cap for the IOC taker. Falls
        # back to the legacy ``cfg.max_slippage_pct`` flat cap when the
        # sidecar is missing or covers different bars.
        from vol_arb.sidecars import lookup_kelly_mult_at, lookup_slip_usd_at
        from deribit_paper.execution.slippage_calc import live_slippage_pct
        kelly_mult = lookup_kelly_mult_at(self._kelly_mult_series, bar.timestamp)
        slip_usd = lookup_slip_usd_at(self._bar_slip_series, bar.timestamp)
        taker_slippage_pct = live_slippage_pct(
            slip_usd_per_btc=slip_usd,
            spot=index_price,
            observed_spread_usd=None,                     # spread unavailable here; override happens at order site
            fallback_pct=self.config.max_slippage_pct,
        )
        n_contracts = self.straddle_flow.size_position(
            equity=self.equity_tracker.current_equity,
            signal_direction=signal.direction,
            spot=index_price,
            trailing_1h_volume_btc=trailing_vol,
            dvol_pct=dvol_pct,
            edge_vp=signal.edge_vol_pts,
            rolling_median_edge=rolling_median_edge,
            peak_equity=self.equity_tracker.peak_equity,
            kelly_mult=kelly_mult,
        )
        # 6b. RND-based size scaling — shrinks position as market prices in
        # more tail risk. Returns 1.0 during warmup (insufficient history)
        # and when tail prob is at or below the rolling baseline.
        if self.rnd_monitor is not None:
            rnd_scale = self.rnd_monitor.size_scale()
            if rnd_scale < 1.0:
                logger.info(
                    "RND size scaling at %s: %.3f (tail prob %.4f vs baseline %.4f)",
                    bar.timestamp, rnd_scale,
                    self.rnd_monitor.latest.tail_prob_total if self.rnd_monitor.latest else 0.0,
                    self.rnd_monitor._baseline_tail_prob() or 0.0,
                )
                n_contracts *= rnd_scale
        risk_check = self.risk_limits.check_signal(
            signal=signal,
            proposed_n_contracts_btc=n_contracts,
            now=bar.timestamp,
        )
        if not risk_check.passed:
            logger.info(
                "Signal rejected by risk at %s: %s",
                bar.timestamp, risk_check.reason,
            )
            self.stats.signals_rejected_by_risk += 1
            return

        # 7. Execute via straddle flow (fire-and-forget background task —
        # the bar loop must not block waiting for the 15-min holding period)
        self.stats.signals_executed += 1
        asyncio.create_task(
            self._execute_and_persist(
                signal, n_contracts,
                dvol_pct=dvol_pct,
                edge_vp=signal.edge_vol_pts,
                rolling_median_edge=rolling_median_edge,
                peak_equity=self.equity_tracker.peak_equity,
                kelly_mult=kelly_mult,
                taker_slippage_pct=taker_slippage_pct,
            ),
            name=f"trade_{bar.timestamp.value}",
        )

    async def _execute_and_persist(
        self,
        signal,
        n_contracts_btc: float,
        *,
        dvol_pct: float | None = None,
        edge_vp: float = 0.0,
        rolling_median_edge: float = 1.0,
        peak_equity: float = 0.0,
        kelly_mult: float = 1.0,
        taker_slippage_pct: float = 0.5,
    ) -> None:
        """Run a straddle trade + update all state after completion."""
        try:
            outcome = await self.straddle_flow.execute(
                signal=signal,
                equity=self.equity_tracker.current_equity,
                exec_cost_ema=self._exec_cost_ema,
                sizing_liq_cap=self.config.sizing_liq_cap,
                dvol_pct=dvol_pct,
                edge_vp=edge_vp,
                rolling_median_edge=rolling_median_edge,
                peak_equity=peak_equity,
                kelly_mult=kelly_mult,
                taker_slippage_pct=taker_slippage_pct,
            )
        except Exception as exc:  # noqa: BLE001 - log + kill-switch
            logger.exception("StraddleFlow.execute failed: %s", exc)
            self.kill_switch.on_trade_result(success=False)
            if self.telegram is not None:
                self.telegram.error(f"trade execution failed: {exc}")
            return

        if outcome.status == "completed":
            self.stats.trades_completed += 1
            # Rough realized PnL in USD — exact MTM comes from position
            # tracker's marks, computed at mark time.  For now: estimate
            # from the entry vs exit fill prices × N × index_price.
            realized_pnl_usd = 0.0
            if (outcome.entry_call and outcome.exit_call
                    and outcome.entry_put and outcome.exit_put):
                premium_in_btc = (
                    outcome.entry_call.average_price + outcome.entry_put.average_price
                )
                premium_out_btc = (
                    outcome.exit_call.average_price + outcome.exit_put.average_price
                )
                pnl_btc = (premium_in_btc - premium_out_btc) * outcome.n_contracts_btc
                if outcome.signal.direction == 1:
                    pnl_btc = -pnl_btc
                index = outcome.signal.straddle.index_price if outcome.signal.straddle else 60_000.0
                realized_pnl_usd = pnl_btc * index

            self.equity_tracker.on_trade_closed(realized_pnl_usd)
            self.kill_switch.on_trade_result(success=True)

            # Update execution cost EMA from actual fill slippage
            if (outcome.entry_call and outcome.entry_call.average_price > 0
                    and outcome.entry_put and outcome.entry_put.average_price > 0):
                index = outcome.signal.straddle.index_price if outcome.signal.straddle else 60_000.0
                # Approximate fill cost: total premium paid × index / N
                total_premium = outcome.entry_call.average_price + outcome.entry_put.average_price
                fill_cost = total_premium * index  # rough $/BTC execution cost proxy
                self._exec_cost_ema += self._exec_cost_alpha * (fill_cost - self._exec_cost_ema)

            if self.telegram is not None:
                self.telegram.trade_close(
                    trade_id=outcome.trade_id,
                    realized_pnl_usd=realized_pnl_usd,
                    hedge_ticks=outcome.hedge_summary.n_ticks_executed
                                if outcome.hedge_summary else 0,
                )

            try:
                await self.store.persist_trade(outcome, realized_pnl_usd=realized_pnl_usd)
                latest_snap = self.equity_tracker.snapshots()[-1]
                await self.store.persist_equity_snapshot(latest_snap)
            except Exception:  # noqa: BLE001 - persistence failure is non-fatal
                logger.exception("Persist trade failed")

            # ── Harness row (C6) — one row per completed trade in the
            # V2 backtest schema, consumed offline by the I4 harness.
            term_slope = (
                self.predictor._compute_term_slope_1w_2w()
                if (self.predictor is not None
                    and hasattr(self.predictor, "_compute_term_slope_1w_2w"))
                else None
            )
            self._harness_writer.consolidate_and_write(
                outcome,
                realized_pnl_usd=realized_pnl_usd,
                kelly_mult=kelly_mult,
                taker_slippage_pct=taker_slippage_pct,
                term_slope_1w_2w=term_slope,
                config_preset=self.config.config_preset,
            )
        else:
            self.stats.trades_aborted += 1
            self.kill_switch.on_trade_result(success=False)
            if self.telegram is not None:
                self.telegram.error(
                    f"trade {outcome.trade_id or 'abort'} {outcome.status}: {outcome.error}",
                )

    async def _recalibrate_svi_task(
        self, option_trades: pd.DataFrame, now: pd.Timestamp,
    ) -> None:
        """Background wrapper around recalibrate_async with error handling."""
        try:
            await self.svi_calibrator.recalibrate_async(option_trades, now=now)
            self._svi_recal_failures = 0
        except Exception as exc:  # noqa: BLE001
            self._svi_recal_failures += 1
            logger.exception(
                "SVI recalibrate failed (%d consecutive): %s",
                self._svi_recal_failures, exc,
            )
            if self.telegram is not None:
                self.telegram.error(
                    f"SVI recalibrate failed ({self._svi_recal_failures}×): {exc}"
                )

    def _trailing_1h_volume_btc(self) -> float | None:
        """Aggregate the last 1 hour of perp trades for participation-rate cap."""
        if self.perp_buffer is None:
            return None
        df = self.perp_buffer.to_dataframe()
        if df.empty:
            return None
        cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(hours=1)
        recent = df[df["timestamp"] >= cutoff]
        if recent.empty:
            return None
        return float(recent["amount"].abs().sum())

    async def _current_spot(self) -> float:
        """Fetch the latest index price for the perp hedger's step callback."""
        if self.index_buffer is not None:
            price = self.index_buffer.latest()
            if price is not None:
                return price
        # Fallback: REST call
        info = await self.rest.get_index_price(
            index_name=f"{self.config.currency.lower()}_usd",
        )
        return float(info.get("index_price", 0.0) or 0.0)

    # ── Position lifecycle callbacks ─────────────────────────────────

    async def _on_position_opened(self, **kwargs: Any) -> None:
        """Called by StraddleFlow after both entry legs fill.

        Updates the in-memory tracker and persists to SQLite so the
        position survives engine crashes.
        """
        from deribit_paper.position.tracker import OpenPosition
        trade_id = kwargs["trade_id"]
        pos = OpenPosition(
            trade_id=trade_id,
            signal_direction=kwargs["signal_direction"],
            call_instrument=kwargs["call_instrument"],
            put_instrument=kwargs["put_instrument"],
            strike=kwargs["strike"],
            expiry_dt=pd.Timestamp(kwargs["expiry_dt"]),
            n_contracts_btc=kwargs["n_contracts_btc"],
            entry_call_price=kwargs["entry_call_price"],
            entry_put_price=kwargs["entry_put_price"],
            entry_timestamp=pd.Timestamp(kwargs["opened_at"]),
            entry_equity_usd=self.equity_tracker.current_equity,
        )
        self.position_tracker._positions[trade_id] = pos
        logger.info(
            "Position opened: %s %s N=%.2f",
            trade_id, "SHORT" if pos.signal_direction == -1 else "LONG",
            pos.n_contracts_btc,
        )
        if self.store is not None:
            await self.store.persist_open_position(**kwargs)

    async def _on_position_closed(self, *, trade_id: str) -> None:
        """Called by StraddleFlow after exit fills (or on cancellation)."""
        self.position_tracker.close_position(trade_id)
        if self.store is not None:
            await self.store.remove_open_position(trade_id)

    # ── C1: Startup position reconciliation ────────────────────────

    async def _reconcile_positions_on_startup(self) -> None:
        """Reconcile DB positions with exchange and resume or close.

        Three-tier strategy:
        1. Load positions from SQLite ``open_positions`` table
        2. Query exchange for actual positions via REST
        3. Match:
           - DB + exchange → **resume** (spawn abbreviated holding period)
           - Exchange only (orphaned) → **close** (market order)
           - DB only (stale) → **clean up** DB record
        """
        if self.rest is None:
            return

        # ── 1. Load DB positions ──
        db_positions: list[dict] = []
        if self.store is not None:
            try:
                db_positions = await self.store.load_open_positions()
            except Exception as exc:
                logger.error("Startup reconciliation: load_open_positions failed: %s", exc)

        db_by_instruments: dict[tuple[str, str], dict] = {}
        for row in db_positions:
            key = (row["call_instrument"], row["put_instrument"])
            db_by_instruments[key] = row

        # ── 2. Query exchange ──
        try:
            exchange_positions = await self.rest.get_positions(
                currency=self.config.currency,
            )
        except Exception as exc:
            logger.error("Startup reconciliation: get_positions failed: %s", exc)
            if self.telegram is not None:
                self.telegram.error(
                    f"STARTUP: get_positions failed ({exc}) — "
                    "cannot verify no orphaned positions exist"
                )
            return

        # Build set of non-zero exchange instruments
        exchange_open = {
            p.get("instrument_name", ""): p
            for p in (exchange_positions or [])
            if abs(float(p.get("size", 0) or 0)) > 0
        }

        if not db_positions and not exchange_open:
            logger.info("Startup reconciliation: no positions anywhere")
            return

        # ── 3a. Resume DB positions confirmed on exchange ──
        resumed = 0
        for (call_inst, put_inst), db_row in list(db_by_instruments.items()):
            call_on_exchange = call_inst in exchange_open
            put_on_exchange = put_inst in exchange_open
            if call_on_exchange and put_on_exchange:
                await self._resume_trade(db_row)
                # Remove from exchange_open so they're not treated as orphans
                exchange_open.pop(call_inst, None)
                exchange_open.pop(put_inst, None)
                resumed += 1
            else:
                # Partial match or DB-only — clean up DB record
                logger.warning(
                    "Startup reconciliation: DB position %s has missing legs "
                    "on exchange (call=%s, put=%s) — cleaning up DB",
                    db_row["trade_id"], call_on_exchange, put_on_exchange,
                )
                if self.store is not None:
                    await self.store.remove_open_position(db_row["trade_id"])

        if resumed > 0:
            msg = f"STARTUP: Resumed {resumed} trade(s) from previous session"
            logger.info(msg)
            if self.telegram is not None:
                self.telegram.send("heartbeat", msg)

        # ── 3b. Close remaining exchange-only orphans ──
        if not exchange_open:
            return

        instruments = list(exchange_open.keys())
        sizes = [float(exchange_open[i].get("size", 0) or 0) for i in instruments]
        msg = (
            f"STARTUP RECONCILIATION: {len(exchange_open)} orphaned position(s) "
            f"found on exchange: {list(zip(instruments, sizes))}"
        )
        logger.warning(msg)
        if self.telegram is not None:
            self.telegram.send("error", msg)

        if not get_dry_run():
            for instrument, pos in exchange_open.items():
                size = float(pos.get("size", 0) or 0)
                if size == 0:
                    continue
                close_side = "sell" if size > 0 else "buy"
                close_amount = abs(size)
                for attempt in range(1, self.config.max_orphan_close_retries + 1):
                    try:
                        logger.warning(
                            "Closing orphaned %s: %s %.4f (attempt %d/%d)",
                            instrument, close_side, close_amount,
                            attempt, self.config.max_orphan_close_retries,
                        )
                        result = await self.order_manager.place_and_wait(
                            instrument_name=instrument,
                            side=close_side,
                            amount=close_amount,
                            order_type="market",
                            label=f"orphan_close_{instrument}_{attempt}",
                            timeout_seconds=30.0,
                            reduce_only=True,
                        )
                        if result.is_filled:
                            logger.info(
                                "Orphan closed: %s %.4f @ %.4f",
                                instrument, result.filled_amount,
                                result.average_price,
                            )
                            break
                        logger.warning(
                            "Orphan close not filled: %s state=%s",
                            instrument, result.state,
                        )
                    except Exception as exc:
                        logger.error(
                            "Orphan close attempt %d failed for %s: %s",
                            attempt, instrument, exc,
                        )
                        if attempt < self.config.max_orphan_close_retries:
                            await asyncio.sleep(2.0)
                else:
                    critical_msg = (
                        f"CRITICAL: Failed to close orphaned position "
                        f"{instrument} after {self.config.max_orphan_close_retries} "
                        f"attempts — MANUAL INTERVENTION REQUIRED"
                    )
                    logger.critical(critical_msg)
                    if self.telegram is not None:
                        self.telegram.send("error", critical_msg)
        else:
            logger.info(
                "Startup reconciliation: dry_run=True, skipping orphan close "
                "(would close %d position(s))", len(exchange_open),
            )

    async def _resume_trade(self, db_row: dict[str, Any]) -> None:
        """Resume a trade that was in-flight when the engine last stopped.

        Registers the position in the in-memory tracker.  If the holding
        period has time remaining, spawns an abbreviated close task.  If
        it already expired, closes the position immediately.
        """
        from deribit_paper.position.tracker import OpenPosition
        trade_id = db_row["trade_id"]
        opened_at = pd.Timestamp(db_row["opened_at"])
        now = pd.Timestamp.now(tz="UTC")
        elapsed = (now - opened_at).total_seconds()
        remaining = self.config.holding_minutes * 60 - elapsed

        # Register in tracker
        pos = OpenPosition(
            trade_id=trade_id,
            signal_direction=db_row["signal_direction"],
            call_instrument=db_row["call_instrument"],
            put_instrument=db_row["put_instrument"],
            strike=db_row["strike"],
            expiry_dt=pd.Timestamp(db_row["expiry_dt"]),
            n_contracts_btc=db_row["n_contracts_btc"],
            entry_call_price=db_row.get("entry_call_price", 0.0),
            entry_put_price=db_row.get("entry_put_price", 0.0),
            entry_timestamp=opened_at,
            entry_equity_usd=self.equity_tracker.current_equity,
        )
        self.position_tracker._positions[trade_id] = pos

        if remaining <= 0:
            logger.info(
                "Resumed trade %s — holding period expired (%.0fs ago), closing now",
                trade_id, -remaining,
            )
            asyncio.create_task(
                self._close_resumed_position(db_row),
                name=f"resume_close_{trade_id}",
            )
        else:
            logger.info(
                "Resumed trade %s — %.0fs remaining in holding period",
                trade_id, remaining,
            )
            asyncio.create_task(
                self._resumed_holding_period(db_row, remaining_seconds=remaining),
                name=f"resume_{trade_id}",
            )

    async def _resumed_holding_period(
        self, db_row: dict[str, Any], remaining_seconds: float,
    ) -> None:
        """Wait out the remaining holding period, then close the position."""
        trade_id = db_row["trade_id"]
        try:
            await asyncio.sleep(remaining_seconds)
        except asyncio.CancelledError:
            logger.warning("Resumed holding period for %s was cancelled", trade_id)
            # Still try to close on cancellation
            await self._close_resumed_position(db_row)
            raise
        await self._close_resumed_position(db_row)

    async def _close_resumed_position(self, db_row: dict[str, Any]) -> None:
        """Close both legs of a resumed straddle position via market orders."""
        trade_id = db_row["trade_id"]
        direction = db_row["signal_direction"]
        n = db_row["n_contracts_btc"]
        exit_side = "buy" if direction == -1 else "sell"

        logger.info("Closing resumed position %s: %s %.2f BTC", trade_id, exit_side, n)
        try:
            await asyncio.gather(
                self.order_manager.place_and_wait(
                    instrument_name=db_row["call_instrument"],
                    side=exit_side, amount=n,
                    order_type="market",
                    label=f"resume_exit_C_{trade_id}",
                    timeout_seconds=30.0, reduce_only=True,
                ),
                self.order_manager.place_and_wait(
                    instrument_name=db_row["put_instrument"],
                    side=exit_side, amount=n,
                    order_type="market",
                    label=f"resume_exit_P_{trade_id}",
                    timeout_seconds=30.0, reduce_only=True,
                ),
            )
            logger.info("Resumed position %s closed successfully", trade_id)
        except Exception as exc:
            logger.error("Failed to close resumed position %s: %s", trade_id, exc)
            if self.telegram is not None:
                self.telegram.error(f"Failed to close resumed position {trade_id}: {exc}")

        # Clean up tracker + DB
        self.position_tracker.close_position(trade_id)
        if self.store is not None:
            await self.store.remove_open_position(trade_id)

    # ── C2: Emergency close all positions ──────────────────────────

    async def _emergency_close_all(self, reason: str) -> None:
        """Close all known positions + flatten perp hedge.

        Called when the kill switch trips.  Best-effort — logs failures
        but does not re-trip the kill switch.
        """
        logger.warning("EMERGENCY CLOSE ALL: %s", reason)
        closed = 0

        # 1. Cancel running trade tasks (holding-period sleeps)
        for task in asyncio.all_tasks():
            if task.get_name().startswith("trade_") and not task.done():
                task.cancel()
                logger.info("Cancelled running trade task: %s", task.get_name())

        # 2. Close option positions known to the tracker
        if self.position_tracker is not None:
            for pos in self.position_tracker.all_open():
                for instrument in (pos.call_instrument, pos.put_instrument):
                    close_side = "buy" if pos.signal_direction == -1 else "sell"
                    try:
                        result = await self.order_manager.place_and_wait(
                            instrument_name=instrument,
                            side=close_side,
                            amount=pos.n_contracts_btc,
                            order_type="market",
                            label=f"emergency_{instrument}",
                            timeout_seconds=15.0,
                            reduce_only=True,
                        )
                        if result.is_filled:
                            closed += 1
                            logger.info("Emergency closed: %s", instrument)
                        else:
                            logger.error(
                                "Emergency close not filled: %s state=%s",
                                instrument, result.state,
                            )
                    except Exception as exc:
                        logger.error(
                            "Emergency close failed for %s: %s", instrument, exc,
                        )

        # 3. Flatten perp hedge
        try:
            perp_positions = await self.rest.get_positions(
                currency=self.config.currency, kind="future",
            )
            for pp in (perp_positions or []):
                size = float(pp.get("size", 0) or 0)
                if abs(size) > 0:
                    instrument = pp.get("instrument_name", "")
                    close_side = "sell" if size > 0 else "buy"
                    result = await self.order_manager.place_and_wait(
                        instrument_name=instrument,
                        side=close_side,
                        amount=abs(size),
                        order_type="market",
                        label=f"emergency_perp_{instrument}",
                        timeout_seconds=15.0,
                        reduce_only=True,
                    )
                    if result.is_filled:
                        closed += 1
                        logger.info("Emergency flattened perp: %s", instrument)
        except Exception as exc:
            logger.error("Emergency perp flatten failed: %s", exc)

        summary = f"Emergency close complete: {closed} fills. Reason: {reason}"
        logger.warning(summary)
        if self.telegram is not None:
            self.telegram.send("error", summary)

    # ── R3: Periodic equity reconciliation ─────────────────────────

    async def _reconcile_equity(self) -> None:
        """Compare internal equity tracker with exchange account summary."""
        if self.rest is None or self.equity_tracker is None:
            return
        try:
            summary = await self.rest.get_account_summary(
                currency=self.config.currency,
            )
            exchange_equity_btc = float(summary.get("equity", 0.0) or 0.0)
            index_info = await self.rest.get_index_price(
                index_name=f"{self.config.currency.lower()}_usd",
            )
            index_price = float(index_info.get("index_price", 0.0) or 0.0)
            if exchange_equity_btc <= 0 or index_price <= 0:
                return
            exchange_equity_usd = exchange_equity_btc * index_price
            internal_equity = self.equity_tracker.current_equity
            if internal_equity <= 0:
                return
            delta_pct = abs(exchange_equity_usd - internal_equity) / internal_equity * 100
            if delta_pct > 15.0:
                msg = (
                    f"EQUITY MISMATCH CRITICAL: exchange=${exchange_equity_usd:,.0f} "
                    f"vs internal=${internal_equity:,.0f} (delta {delta_pct:.1f}%)"
                )
                logger.error(msg)
                if self.telegram is not None:
                    self.telegram.send("error", msg)
                self.kill_switch.trip(msg)
            elif delta_pct > 5.0:
                msg = (
                    f"Equity drift: exchange=${exchange_equity_usd:,.0f} "
                    f"vs internal=${internal_equity:,.0f} (delta {delta_pct:.1f}%)"
                )
                logger.warning(msg)
                if self.telegram is not None:
                    self.telegram.send("error", msg)
            else:
                logger.info(
                    "Equity reconciled: exchange=$%.0f internal=$%.0f (delta %.1f%%)",
                    exchange_equity_usd, internal_equity, delta_pct,
                )
        except Exception as exc:
            logger.error("Equity reconciliation failed: %s", exc)
