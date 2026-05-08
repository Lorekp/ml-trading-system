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
import os
import secrets
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from deribit_paper.client import DeribitAuth, DeribitRESTClient, DeribitWebSocketClient
from deribit_paper.config import PaperConfig
from deribit_paper.execution import OrderManager, StraddleFlow
from deribit_paper.execution.idempotency import IdempotencyGuard
from deribit_paper.execution.pnl_decomp import decompose_gross_pnl
from deribit_paper.execution.profit_scale import (
    ProfitScaleEstimator,
    bsm_vega_per_vol_pt,
)
from deribit_paper.features import LiveSVICalibrator, StrikePicker, compute_bar_tsrv
from deribit_paper.features.dollar_bar import DollarBarAggregator
from deribit_paper.features.dollar_bar_buffer import DollarBarBuffer
from deribit_paper.features.dollar_bar_stream import DollarBarSpotStream
from deribit_paper.features.tsrv_history import TSRVHistoryBuffer
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


def _assert_sidecar_fresh(
    path: str | Path,
    *,
    name: str,
    max_age_days: int,
    allow_stale: bool,
) -> None:
    """H9 (2026-05-06): refuse to start when a sidecar is too old.

    Mirrors the aligner-β* staleness pattern (engine.initialize() lines
    353-393) but generalised to any file-based sidecar by using
    ``os.path.getmtime``. Per-sidecar override flags on ``PaperConfig``
    (e.g., ``allow_stale_kelly``) downgrade the refusal to a WARNING.

    Why this matters: before H9, a 6-month-old `kelly_mult` parquet
    silently flowed into live sizing because no mtime check existed.
    Same for ``bar_execution_quality`` (slippage), the historical
    features parquet, and the SVI features parquet. The aligner had a
    refusal but the four other sidecars did not.

    Parameters
    ----------
    path : str | Path
        File whose mtime is checked. Missing files are NOT a staleness
        failure here — the caller's existence check (or downstream
        load) catches that.
    name : str
        Short label for log + error messages (e.g. ``"kelly K1"``).
    max_age_days : int
        Refusal threshold. The sidecar is allowed up to and including
        ``max_age_days`` days old; refusal triggers strictly above.
    allow_stale : bool
        When True, downgrade refusal to a WARNING. Set via the
        per-sidecar CLI override flag (e.g. ``--allow-stale-kelly``).
    """
    p = Path(path)
    if not p.exists():
        # Existence is enforced elsewhere; just skip silently.
        return
    # S4 (parity-cycle round 2 — 2026-05-07): prefer parquet data-age
    # over file mtime. A ``cp`` deploy of older data refreshes mtime,
    # which masked the staleness pre-S4. ``data_age_seconds`` reads
    # parquet row-group statistics for a recognised time column and
    # falls back to mtime when no time column is present (CSV / pickle
    # sidecars, or parquets without a timestamp column).
    if p.suffix == ".parquet":
        from deribit_paper.utils.sidecar_staleness import data_age_seconds
        age_seconds = data_age_seconds(p)
    else:
        age_seconds = time.time() - os.path.getmtime(p)
    age_days = age_seconds / 86400.0
    if age_days <= max_age_days:
        logger.info(
            "Sidecar %s fresh: %s (age=%.1f days, threshold=%d)",
            name, p, age_days, max_age_days,
        )
        return
    msg = (
        f"Sidecar '{name}' at {p} is stale: age={age_days:.1f} days, "
        f"threshold={max_age_days} days. The live engine is about to "
        f"trade against a sidecar last regenerated more than {max_age_days} "
        f"days ago — its calibration is unlikely to match the current "
        f"market regime."
    )
    if not allow_stale:
        logger.error(msg)
        raise RuntimeError(
            msg + f" Refresh the sidecar OR pass --allow-stale-{name.replace(' ', '-').lower()} "
            f"to override (degrading sidecar fidelity)."
        )
    logger.warning(
        "%s — OVERRIDDEN by --allow-stale flag; engine will proceed "
        "with stale sidecar.",
        msg,
    )


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

        # Fire-and-forget trade tasks tracked separately so shutdown can
        # await their completion (E.1.4 2026-05-01). Without this, an
        # in-flight trade dies on shutdown with positions still open on
        # Deribit but no record locally — the dropped task path.
        self._trade_tasks: list[asyncio.Task] = []

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

        # Profit-scale estimator (V2 wiring, E.1.1 2026-05-01).
        # Empirical-Bayes shrinkage on realised gross PnL per BTC vs a
        # theoretical edge×vega prior. ``None`` when ``use_profit_scaling``
        # is False — engine emits profit_scale=1.0 in that case so the
        # ``default`` preset stays byte-id with the prior live behaviour.
        self._profit_scale_estimator: ProfitScaleEstimator | None = None

        # Harness-grade trade-row writer (V2 wiring, C6 2026-04-26).
        # Appends one row per closed trade to
        # vol_arb/output/live_trades_harness.parquet. Used by I4 to
        # diff live PnL vs the V2 backtest replay over the same window.
        self._harness_writer = HarnessWriter(
            "vol_arb/output/live_trades_harness.parquet",
        )

        # D4 (parity 2026-05-07): rolling-median-edge EMA matching V2
        # backtester at ``vol_arb/backtester.py:1846-1847, 2205-2206``.
        # Pre-D4 the live engine kept a 96-bar median-deque updated on
        # every finite-edge bar; V2 keeps a 500-bar EMA (alpha=2/501)
        # seeded from the median of the FIRST 96 bars and updated ONLY
        # on traded bars (signal != 0 AND N > 0). Mirror exactly:
        #   * ``_edge_warmup`` collects |edge| from every finite-edge
        #     bar until 96 entries are seen.
        #   * ``_edge_ema`` is then seeded with the median and starts
        #     EMA updates after risk-check + size finalisation.
        self._edge_warmup: list[float] = []
        self._edge_ema: float | None = None
        # Class-level constants matching V2 backtester. Kept as instance
        # attrs for testability (a unit test can override the warmup
        # length to exercise the seed transition without 96 bars).
        self._EDGE_WARMUP_N: int = 96
        self._EDGE_EMA_ALPHA: float = 2.0 / 501.0

        # Stale-data guard: consecutive bars with zero trades
        self._consecutive_empty_bars: int = 0

        # SVI recalibration failure tracking (R2)
        self._svi_recal_failures: int = 0
        # 2026-05-07: success-counter for heartbeat. The svi_age field
        # phase-locks at a constant value when heartbeat and recal
        # cadences are both 60 min, so an operator can't distinguish
        # "healthy" from "frozen". A monotonic count makes churn
        # visible: if it stays the same across two consecutive
        # heartbeats, recal is dead.
        self._svi_recal_count: int = 0

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
        # Paper mode (cfg.paper_mode=True) connects to public mainnet
        # WS/REST without credentials; the dry-run safety belt prevents
        # any private endpoint from being called. All other modes
        # require credentials and authenticate on startup.
        if cfg.paper_mode:
            self.auth = None
            logger.info(
                "Paper mode: connecting unauthenticated; no /private/* "
                "endpoints will be called",
            )
        else:
            if not cfg.client_id or not cfg.client_secret:
                raise RuntimeError(
                    "Deribit credentials missing — set DERIBIT_TESTNET_CLIENT_ID + "
                    "_SECRET in .env before running the engine",
                )
            self.auth = DeribitAuth(cfg.client_id, cfg.client_secret, cfg.rest_base_url)
        self.rest = DeribitRESTClient(self.auth, cfg.rest_base_url)
        await self.rest.__aenter__()  # opens the httpx session

        # H12 (2026-05-06): clock-skew gate. The bar boundary uses the
        # local system clock (BarClock); if it drifts vs Deribit's
        # server clock, every signal fires at a slightly different
        # offset relative to where the V2 backtest sampled. Warn at
        # 250ms, refuse at 5s. Best-effort: failure to query
        # /public/get_time logs a warning but does not block startup.
        try:
            server_ms = await self.rest.get_server_time_ms()
            local_ms = int(time.time() * 1000)
            skew_ms = abs(local_ms - server_ms)
            if skew_ms >= 5000:
                msg = (
                    f"H12 clock-skew check: local clock differs from "
                    f"Deribit server by {skew_ms}ms (server={server_ms}, "
                    f"local={local_ms}, threshold=5000ms). The bar-clock "
                    f"would mis-align signals vs the V2 backtest. Run "
                    f"ntpdate / systemd-timesyncd before re-launching."
                )
                logger.error(msg)
                raise RuntimeError(msg)
            elif skew_ms >= 250:
                logger.warning(
                    "H12 clock-skew check: local clock differs from "
                    "Deribit server by %dms (threshold warn 250ms, "
                    "refuse 5000ms). NTP sync recommended.",
                    skew_ms,
                )
            else:
                logger.info(
                    "H12 clock-skew check: local-vs-Deribit %dms (OK)",
                    skew_ms,
                )
        except RuntimeError:
            raise
        except Exception:  # noqa: BLE001
            logger.exception(
                "H12: failed to fetch Deribit server time — skew check "
                "skipped (engine continues, but operator should verify "
                "system clock is NTP-synced)"
            )

        if cfg.paper_mode:
            # No account on Deribit; equity is synthetic.
            account_equity_btc = 0.0
            logger.info(
                "Paper mode: synthetic equity=$%.0f (no Deribit account)",
                cfg.paper_initial_equity_usd,
            )
        else:
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
        # Round-4 F6/F8 (parity-cycle 2026-05-07): Binance Spot trade
        # buffer fed by ``DollarBarSpotStream``'s fan-out hook. Live
        # ``compute_bar_tsrv`` reads from this buffer instead of the
        # Deribit perp buffer so TSRV-derived features match the
        # venue the offline pipeline (Binance Vision spot aggTrades)
        # trains on. Window mirrors ``tsrv_rolling_window_hours``.
        self.spot_trade_buffer = TradeBuffer(
            name="spot",
            window_seconds=float(cfg.spot_trade_buffer_hours) * 3600.0,
        )

        def _on_ws_reconnect() -> None:
            # O-9 (parity-r3 cycle 2 \u2014 2026-05-08): on WS reconnect, fire
            # a fire-and-forget reconciliation against
            # ``rest.get_open_orders_by_currency`` so the engine knows
            # whether anything orphaned during the WS gap. Pre-fix the
            # engine just sent a Telegram message; orders submitted via
            # the maker-then-taker path that timed out during the gap
            # could remain open at the venue with no listener \u2014 and a
            # second WS hiccup + 3 REST timeouts would trip the kill
            # switch on what is really a transient network blip.
            if self.telegram is not None:
                self.telegram.send(
                    "error",
                    "Deribit WS reconnected \u2014 channels resubscribed",
                )
            # Schedule the reconciliation; never block the WS reconnect
            # callback itself (it must return synchronously to the WS
            # client's reconnect path).
            try:
                asyncio.create_task(
                    self._reconcile_open_orders_on_reconnect(),
                    name="reconcile_open_orders_on_ws_reconnect",
                )
            except RuntimeError:
                # No running event loop (e.g., test harness invoking
                # the callback directly). Safe to skip; the next reconnect
                # under a real loop will catch up.
                logger.debug(
                    "O-9: no running loop on _on_ws_reconnect; skipping reconcile",
                )

        self.ws = DeribitWebSocketClient(
            cfg.ws_url,
            auth=self.auth,
            on_reconnect=_on_ws_reconnect,
            data_stale_seconds=float(cfg.ws_data_stale_seconds),
        )
        # Paper mode uses the 100ms aggregated trade channels which work
        # without WS auth; production runs use .raw which Deribit gates
        # behind authentication in some configurations.
        trades_interval = "100ms" if cfg.paper_mode else "raw"
        self.subscriber = MarketDataSubscriber(
            ws=self.ws,
            perp_buffer=self.perp_buffer,
            option_buffer=self.option_buffer,
            index_buffer=self.index_buffer,
            currency=cfg.currency,
            dte_min_days=0.0,
            dte_max_days=21.0,
            trades_interval=trades_interval,
        )

        self.bar_clock = BarClock(bar_minutes=cfg.bar_minutes)
        self.aggregator = BarAggregator(
            bar_minutes=cfg.bar_minutes,
            perp_buffer=self.perp_buffer,
            option_buffer=self.option_buffer,
            index_buffer=self.index_buffer,
            tsrv_buffer=self.spot_trade_buffer,  # Round-4 F6/F8
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
        # H9: features parquet is regenerated when monthly aligner refresh
        # runs. 45 days = 1.5 months gives slack for an aligner refresh
        # being a few days late.
        _assert_sidecar_fresh(
            features_parquet,
            name="features",
            max_age_days=45,
            allow_stale=getattr(cfg, "allow_stale_features", False),
        )
        svi_features_parquet = Path(cfg.svi_features_parquet)
        if not svi_features_parquet.exists():
            raise FileNotFoundError(
                f"SVI features parquet not found at {svi_features_parquet} — "
                "re-run vol_arb/run_pipeline.py --svi to regenerate the "
                "constant-maturity SVI features the T2.1 models depend on",
            )
        _assert_sidecar_fresh(
            svi_features_parquet,
            name="svi-features",
            max_age_days=45,
            allow_stale=getattr(cfg, "allow_stale_svi_features", False),
        )
        # F3 Tier 6 (parity-cycle round 3 — 2026-05-07): live dollar-bar
        # buffer + TSRV-history buffer. Both warm-started from offline
        # sidecars so the 22 dollar-bar / TSRV-grid handlers (Tiers 3+5)
        # produce live values from the first bar instead of falling
        # through to baseline for ~37 min / ~1 week of warmup.
        self.dollar_bar_buffer = DollarBarBuffer(
            max_size=cfg.dollar_bar_history_bars,
        )
        n_dollar_bars = self.dollar_bar_buffer.prewarm_from_parquet(
            cfg.dollar_bar_cache_path,
        )
        logger.info(
            "DollarBarBuffer ready: max_size=%d, prewarmed=%d bars from %s",
            cfg.dollar_bar_history_bars, n_dollar_bars, cfg.dollar_bar_cache_path,
        )
        self.tsrv_history_buffer = TSRVHistoryBuffer(max_size=672)
        _tsrv_history_parquet = Path(
            f"TSRV/output/{cfg.asset_pair}_tsrv_{cfg.bar_minutes}m.parquet"
        )
        n_tsrv_rows = self.tsrv_history_buffer.prewarm_from_parquet(
            _tsrv_history_parquet,
        )
        logger.info(
            "TSRVHistoryBuffer ready: max_size=672, prewarmed=%d rows from %s",
            n_tsrv_rows, _tsrv_history_parquet,
        )

        self.feature_assembler = FeatureAssembler(
            features_parquet=features_parquet,
            svi_features_parquet=svi_features_parquet,
            bundle=self.model_bundle,
            svi_calibrator=self.svi_calibrator,
            dollar_bar_buffer=self.dollar_bar_buffer,
            tsrv_history_buffer=self.tsrv_history_buffer,
            rolling_history_bars=cfg.bars_per_day * 7,
        )
        # C (parity 2026-05-07): pre-warm the HAR-PD circular buffer
        # from the offline TSRV parquet so the first live bar emits
        # non-NaN HAR-PD features. Without this the buffer would need
        # ~24h of live data before HAR-PD features go non-NaN, which
        # would silently disable the HAR-PD model variant on launch.
        # No-op + warning when the parquet is missing; HAR-PD only
        # affects 6 of 50 features and the assembler falls back to
        # baseline if compute_features returns NaN.
        tsrv_parquet = Path(
            f"TSRV/output/{cfg.asset_pair}_tsrv_{cfg.bar_minutes}m.parquet"
        )
        self.feature_assembler.prewarm_har_pd_from_tsrv(tsrv_parquet)
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
            # ── E.2.3: staleness alert + soft refusal ──
            # The aligner sidecar is regenerated monthly. Stale > 1
            # month → WARN (degraded but trading continues). Stale > 2
            # months under V2 preset → refuse to start (the live engine
            # is silently using a β* that hasn't seen recent regime
            # data; better to halt than to ship biased predictions).
            try:
                latest_month_str = max(aligner_betas.keys())
                latest_month = pd.Period(latest_month_str, freq="M")
                now_month = pd.Period(
                    pd.Timestamp.now(tz="UTC"), freq="M",
                )
                gap_months = (now_month - latest_month).n
                if (gap_months >= 2 and cfg.config_preset == "v2"
                        and not getattr(cfg, "allow_stale_aligner", False)):
                    msg = (
                        f"Aligner β* sidecar stale by {gap_months} months "
                        f"(latest={latest_month_str}, now={now_month}). "
                        f"Under V2 preset this is a refusal to start — "
                        f"refresh via scripts/extract_aligner_betas.py + "
                        f"scripts/refresh_live_sidecars.py before "
                        f"launching, OR pass --allow-stale-aligner to "
                        f"override (predictions will be biased)."
                    )
                    logger.error(msg)
                    raise RuntimeError(msg)
                if gap_months >= 2 and getattr(cfg, "allow_stale_aligner", False):
                    logger.warning(
                        "Aligner β* sidecar stale by %d months — "
                        "OVERRIDDEN by --allow-stale-aligner. "
                        "Predictions will fall back to the most-recent-"
                        "prior β* month (%s).",
                        gap_months, latest_month_str,
                    )
                elif gap_months >= 1:
                    logger.warning(
                        "Aligner β* sidecar stale by %d month(s) "
                        "(latest=%s) — predictions remain biased toward "
                        "older regime",
                        gap_months, latest_month_str,
                    )
            except RuntimeError:
                raise
            except Exception:  # noqa: BLE001 - parse failure non-fatal
                logger.exception(
                    "Aligner β* staleness check failed (advisory only)"
                )
        # H9: Kelly K1 sidecar staleness — regenerated alongside the
        # aligner. 45 days for the same reason as features.
        if cfg.kelly_mult_path:
            _assert_sidecar_fresh(
                cfg.kelly_mult_path,
                name="kelly-K1",
                max_age_days=45,
                allow_stale=getattr(cfg, "allow_stale_kelly", False),
            )
        self._kelly_mult_series = (
            _load_kelly(cfg.kelly_mult_path) if cfg.kelly_mult_path else None
        )
        # H9: bar_execution_quality slip sidecar — empirical from the
        # backtester's IOC-ladder fill-quality model. Updated less
        # frequently than aligner because liquidity-conditioned slip is
        # more stable. 90 days = 3 months.
        if cfg.bar_exec_quality_path:
            _assert_sidecar_fresh(
                cfg.bar_exec_quality_path,
                name="bar-exec-quality",
                max_age_days=90,
                allow_stale=getattr(cfg, "allow_stale_slip", False),
            )
        self._bar_slip_series = (
            _load_slip(cfg.bar_exec_quality_path)
            if cfg.bar_exec_quality_path else None
        )
        if cfg.use_profit_scaling:
            self._profit_scale_estimator = ProfitScaleEstimator(
                floor=cfg.profit_scaling_floor,
                cap=1.0,
                edge_discount=0.5,                # backtester V2 default
                tau=200.0,                        # backtester V2 default
                spread_jump_k=cfg.spread_jump_k,
                ioc_cap_multiplier=2.0,           # backtester V2 default
                perp_taker_fee=cfg.perp_taker_fee,
                option_maker_fee=cfg.option_maker_fee,
            )
            logger.info(
                "ProfitScaleEstimator enabled (floor=%.2f, edge_disc=0.5, tau=200)",
                cfg.profit_scaling_floor,
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
            # D2 (parity 2026-05-07): mirror V2 backtester long-hour filter.
            skip_long_hours_utc=cfg.skip_long_hours_utc,
            aligner_betas=aligner_betas,
            # E.1.14: V2 preset configures the aligner sidecar; an empty
            # dict here means the configuration failed silently. Fail
            # fast at startup rather than bypass β* on every bar.
            require_aligner=(cfg.config_preset == "v2"),
            log_every_bar=True,
        )
        self.live_drl_hedger = LiveDRLHedger(
            bundle=self.model_bundle,
            n_steps_per_episode=cfg.intra_bar_hedge_ticks,
            bar_minutes=cfg.bar_minutes,
            # D4 (parity-cycle round 2 — 2026-05-07): align internal
            # inventory updates with the dust filter the PerpHedger
            # applies downstream so policy hedge_inventory_btc doesn't
            # drift on sub-dust ticks. Same threshold both layers use
            # (PerpHedger.dust_threshold_btc=cfg.delta_hedge_band per E2).
            dust_threshold_btc=cfg.delta_hedge_band,
        )

        # ── E.2.2: DRL policy bar-minutes manifest check ──
        # The SAC bundle carries no metadata for the training horizon.
        # We sit a sibling ``<policy>.manifest.json`` next to the .zip
        # with {"bar_minutes": N}. WARN at startup when it doesn't match
        # ``cfg.bar_minutes`` — using a 15m-trained policy on 5m bars
        # produces silently-broken hedges.
        try:
            import json as _json
            manifest_path = Path(cfg.drl_policy_path).with_suffix(".manifest.json")
            if not manifest_path.exists():
                # Fallback for the .zip suffix: try alongside-replacement.
                manifest_path = (
                    Path(cfg.drl_policy_path).parent
                    / (Path(cfg.drl_policy_path).stem + ".manifest.json")
                )
            if manifest_path.exists():
                manifest = _json.loads(manifest_path.read_text(encoding="utf-8"))
                trained_bar_minutes = int(manifest.get("bar_minutes", 0))
                if trained_bar_minutes != cfg.bar_minutes:
                    msg = (
                        f"DRL policy {cfg.drl_policy_path!r} trained at "
                        f"bar_minutes={trained_bar_minutes} but engine is "
                        f"configured for cfg.bar_minutes={cfg.bar_minutes}. "
                        f"Hedges will be silently broken. Re-train or "
                        f"point cfg.drl_policy_path at a horizon-matched "
                        f"bundle."
                    )
                    logger.warning(msg)
                    if self.telegram is not None:
                        self.telegram.send("error", msg)
                else:
                    logger.info(
                        "DRL policy %s manifest confirms bar_minutes=%d",
                        Path(cfg.drl_policy_path).name, trained_bar_minutes,
                    )
            else:
                logger.warning(
                    "DRL policy %s has no manifest.json — assuming "
                    "bar_minutes=%d (cfg). Add %s with "
                    "{\"bar_minutes\": N} to silence this warning.",
                    Path(cfg.drl_policy_path).name, cfg.bar_minutes,
                    manifest_path.name,
                )
        except Exception as exc:  # noqa: BLE001 - manifest is advisory
            logger.warning("DRL manifest check failed (advisory only): %s", exc)

        # ── Phase 4b: Live VPIN stream ──
        # E (parity 2026-05-07): branch on ``cfg.vpin_feed_kind`` so the
        # operator can pick WS (default for un-blocked IPs) or REST
        # polling (default — homeserver Sweden ProtonVPN is geo-blocked
        # from the Futures WS). Both feed kinds use the same
        # _LiveBucketAccumulator + warm-start state + staleness logic.
        from deribit_paper.features.vpin_live import LiveVPINStream
        if cfg.vpin_state_path:
            self.vpin_stream = LiveVPINStream(
                state_path=Path(cfg.vpin_state_path),
                fallback_vpin=cfg.vpin_fallback,
                ws_url=cfg.binance_ws_url,
                feed_kind=cfg.vpin_feed_kind,
                symbol=f"{cfg.currency.upper()}USDT",
                binance_rest_url=cfg.binance_rest_url,
                poll_seconds=cfg.vpin_poll_seconds,
            )
            logger.info(
                "LiveVPINStream ready (kind=%s, warm-start=%s, fallback=%.3f, "
                "poll=%.1fs)",
                cfg.vpin_feed_kind, cfg.vpin_state_path, cfg.vpin_fallback,
                cfg.vpin_poll_seconds,
            )
        else:
            logger.info("LiveVPINStream disabled (vpin_state_path empty)")

        # ── F3 Tier 6: Live dollar-bar spot stream ──
        # Polls Binance spot aggTrades into the DollarBarAggregator,
        # which emits dollar bars into self.dollar_bar_buffer (via
        # on_bar_close). Tier 3+5 handlers consume from the buffer.
        # Runs in parallel with the existing LiveVPINStream (Futures perp)
        # — they feed independent layers; cutover happens once the live
        # spot-VPIN has soaked.
        from deribit_paper.client.binance_rest import BinanceRESTClient
        self._dollar_bar_rest = BinanceRESTClient.for_spot()
        self.dollar_bar_aggregator = DollarBarAggregator(
            threshold=cfg.dollar_bar_threshold_btc,
            on_bar_close=self.dollar_bar_buffer.append,
        )
        self.dollar_bar_stream = DollarBarSpotStream(
            rest_client=self._dollar_bar_rest,
            aggregator=self.dollar_bar_aggregator,
            symbol=f"{cfg.currency.upper()}USDT",
            poll_seconds=cfg.spot_poll_seconds,
            # Round-4 F6/F8: fan out every parsed Binance Spot trade
            # into the spot trade buffer so live TSRV reads from the
            # same venue the offline pipeline trains on.
            extra_trade_sink=self.spot_trade_buffer.append,
        )
        logger.info(
            "DollarBarSpotStream ready (symbol=%sUSDT, threshold=$%.0f, "
            "poll=%.1fs, fanout=spot_trade_buffer)",
            cfg.currency.upper(),
            cfg.dollar_bar_threshold_btc,
            cfg.spot_poll_seconds,
        )

        # Round-4 F6/F8 warm-start: pre-fill spot_trade_buffer from
        # the last ~20 min of Binance Spot aggTrades so the first bar
        # after restart has TSRV input. Best-effort — a missing or
        # stale sidecar simply leaves the buffer empty, in which case
        # ``compute_bar_tsrv`` returns its existing zero-fill (same
        # behaviour as today's first-bar Deribit-perp fallback) and
        # the prewarmed ``TSRVHistoryBuffer`` (F3 Tier 5) keeps
        # serving rolling features.
        if cfg.spot_trade_buffer_warmstart_path:
            try:
                added = self.spot_trade_buffer.prewarm_from_parquet(
                    cfg.spot_trade_buffer_warmstart_path,
                    instrument_name=f"{cfg.currency.upper()}USDT",
                    trade_id_prefix="warmstart",
                    max_age_seconds=(
                        float(cfg.spot_trade_buffer_warmstart_minutes) * 60.0
                    ),
                )
                logger.info(
                    "spot_trade_buffer warm-start: %d trades from %s "
                    "(window=%.1fh, max_age=%.1fmin)",
                    added,
                    cfg.spot_trade_buffer_warmstart_path,
                    float(cfg.spot_trade_buffer_hours),
                    float(cfg.spot_trade_buffer_warmstart_minutes),
                )
            except FileNotFoundError as e:
                logger.warning(
                    "spot_trade_buffer warm-start skipped: %s. "
                    "First bar after restart will have zero-fill TSRV.",
                    e,
                )
            except Exception:  # noqa: BLE001 — warm-start failure must not block start
                logger.exception(
                    "spot_trade_buffer warm-start failed; first bar "
                    "after restart will have zero-fill TSRV.",
                )

        # ── Phase 5: Execution ──
        # E.2.8: latency-breach callback. Wired AFTER kill_switch is
        # constructed below; we capture it via a local closure so this
        # can be passed at OrderManager construction time.
        def _latency_breach(total_ms: float, threshold: float, instrument: str) -> None:
            if self.kill_switch is not None:
                self.kill_switch.on_latency_breach(
                    total_ms=total_ms, threshold_ms=threshold,
                    instrument_name=instrument,
                )

        self.order_manager = OrderManager(
            rest=self.rest,
            idempotency_guard=IdempotencyGuard(ttl_seconds=60.0),
            fill_poll_seconds=1.5,
            latency_threshold_ms=cfg.kill_switch_latency_ms,
            on_latency_breach=_latency_breach,
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
            delta_hedge_band=cfg.delta_hedge_band,  # E2 (parity-cycle round 2)
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
            use_tiered_taker=cfg.use_tiered_taker,
            cancel_exit_max_slippage_pct=cfg.cancel_exit_max_slippage_pct,
            on_position_opened=self._on_position_opened,
            on_position_closed=self._on_position_closed,
        )

        # ── Phase 6: State + risk + monitoring ──
        # Equity: seed from Deribit's reported balance, converted to USD
        # via the index price.  On subsequent bars we track via realized PnL.
        index_info = await self.rest.get_index_price(index_name=f"{cfg.currency.lower()}_usd")
        index_price_usd = float(index_info.get("index_price", 0.0) or 0.0)
        if cfg.paper_mode:
            # Paper run: seed from paper_initial_equity_usd so the
            # per-trade dollar PnL distribution lines up with the V2
            # backtest convention ($100K starting equity).
            initial_equity_usd = cfg.paper_initial_equity_usd
        else:
            initial_equity_usd = (
                account_equity_btc * index_price_usd
                if account_equity_btc > 0 and index_price_usd > 0
                else cfg.initial_equity * 60_000.0  # fallback to 1 BTC × $60k
            )
        self.equity_tracker = EquityTracker(initial_equity_usd=initial_equity_usd)
        self.position_tracker = PositionTracker()
        self.store = SQLiteStore(cfg.sqlite_db)
        await self.store.open()

        # D-11 (parity-r3 cycle 2 — 2026-05-08): wire the DRL hedger's
        # state-persistence callbacks to the SQLite store. Sync→async
        # bridge via ``asyncio.create_task`` so the synchronous
        # ``LiveDRLHedger.step()`` hot path returns immediately while
        # the SQLite write proceeds in the background. ``store`` is
        # opened above; ``live_drl_hedger`` was constructed earlier
        # without persister callbacks (cold-start safe — persistence
        # is disabled until this wiring runs).
        def _drl_state_persister(state_dict: dict[str, Any]) -> None:
            asyncio.create_task(
                self.store.persist_drl_trade_state(state_dict),
                name=f"persist_drl_{state_dict['trade_id']}",
            )

        def _drl_state_remover(trade_id: str) -> None:
            asyncio.create_task(
                self.store.remove_drl_trade_state(trade_id),
                name=f"remove_drl_{trade_id}",
            )

        self.live_drl_hedger._state_persister = _drl_state_persister
        self.live_drl_hedger._state_remover = _drl_state_remover

        # S1+S3 (parity-cycle round 2 — 2026-05-07): restore engine
        # state across restarts. Pre-fix the equity tracker, edge_ema,
        # and exec_cost_ema all reset to constructor defaults at every
        # restart, silently re-baselining the kill-switch daily-DD
        # reference and re-warming the rolling EMAs. Mirror H4 pattern.
        try:
            persisted = await self.store.load_engine_state()
        except Exception:  # noqa: BLE001 — defensive, never block startup
            logger.exception("S1: failed to load engine_state; cold-start defaults")
            persisted = None
        # S5+S8 (parity-cycle round 2 — 2026-05-07): the kill_switch
        # consecutive_errors counter and SVI recal counters are part of
        # engine_state but the kill_switch isn't constructed until below.
        # Stash on self for the post-kill_switch-creation handoff.
        self._persisted_engine_state: dict | None = persisted
        if persisted is not None:
            try:
                self.equity_tracker.load_state({
                    "current_equity_usd": persisted["current_equity_usd"],
                    "peak_equity_usd": persisted["peak_equity_usd"],
                    "realized_pnl_usd": persisted["realized_pnl_usd"],
                    "n_trades_closed": persisted["n_trades_closed"],
                })
                self._exec_cost_ema = float(
                    persisted["exec_cost_ema_usd_per_btc"]
                )
                if persisted.get("edge_ema") is not None:
                    self._edge_ema = float(persisted["edge_ema"])
                    self._edge_warmup = []  # warmup is complete
                else:
                    self._edge_ema = None
                    self._edge_warmup = list(persisted.get("edge_warmup", []))
                # S8: SVI recal counters (early enough — they're plain
                # ints on self).
                self._svi_recal_count = int(
                    persisted.get("svi_recal_count", 0)
                )
                self._svi_recal_failures = int(
                    persisted.get("svi_recal_failures", 0)
                )
                logger.info(
                    "S1+S3+S8: engine_state restored — equity=$%.2f peak=$%.2f "
                    "n_trades=%d edge_ema=%s exec_cost_ema=%.2f "
                    "svi_recal=(n=%d, fail=%d) (last_updated=%s)",
                    persisted["current_equity_usd"],
                    persisted["peak_equity_usd"],
                    persisted["n_trades_closed"],
                    persisted["edge_ema"],
                    persisted["exec_cost_ema_usd_per_btc"],
                    self._svi_recal_count,
                    self._svi_recal_failures,
                    persisted["last_updated"],
                )
            except (KeyError, ValueError, TypeError) as exc:
                logger.exception(
                    "S1: engine_state row malformed (%s); cold-start defaults", exc,
                )

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
            # H15 (2026-05-06): defence-in-depth caps.
            max_trades_per_hour=getattr(cfg, "max_trades_per_hour", 0),
            max_gross_notional_usd=getattr(cfg, "max_gross_notional_usd", 0.0),
        )
        # H1 (2026-05-06): persist kill-switch state across restarts.
        # The store has had get/set_kill_switch_state methods since
        # Phase 6, but engine.initialize() never read or wrote them.
        # An engine that tripped (e.g., daily-DD limit) and restarted
        # would silently come back unhalted, losing the operator's
        # signal-to-stop. Now: trip/reset persist via callback, and
        # initialize() restores prior halted state unless
        # --clear-kill-state was passed.
        self._ks_persist_tasks: set[asyncio.Task] = set()

        def _persist_ks(halted: bool, reason: str | None) -> None:
            # Sync wrapper invoked by KillSwitch.trip / .reset; we must
            # be inside the event loop here (the engine init runs under
            # asyncio.run). Best-effort: if scheduling itself raises,
            # log and swallow — persistence failure must NEVER mask the
            # in-memory trip.
            if self.store is None:
                return
            try:
                task = asyncio.create_task(
                    self.store.set_kill_switch_state(halted, reason=reason),
                    name=f"persist_ks_{int(halted)}",
                )
            except RuntimeError:
                # No running loop — extremely unlikely from engine code,
                # but possible from a unit test that constructs KillSwitch
                # outside an event loop. Drop silently.
                logger.warning(
                    "KillSwitch persist callback fired with no event loop "
                    "(halted=%s reason=%s) — state NOT persisted",
                    halted, reason,
                )
                return
            self._ks_persist_tasks.add(task)
            task.add_done_callback(self._ks_persist_tasks.discard)
            # H15 (2026-05-06): emit a Telegram alert on RESET as well as
            # trip. Pre-H15, only the trip path notified (via
            # ``alert_callback`` -> ``telegram.kill_switch``); the reset
            # was silent. Operators want visibility on both transitions
            # so they can correlate clearings with deploys / restarts.
            if not halted and self.telegram is not None:
                try:
                    self.telegram.send(
                        "info",
                        f"✓ Kill-switch RESET (was: {reason or 'no reason recorded'})",
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "KillSwitch reset Telegram alert failed",
                    )

        self.kill_switch = KillSwitch(
            equity_tracker=self.equity_tracker,
            max_daily_drawdown_pct=cfg.kill_switch_daily_dd_pct,
            max_consecutive_errors=cfg.kill_switch_consecutive_errors,
            network_outage_seconds=cfg.network_outage_seconds,
            alert_callback=lambda reason: self.telegram.kill_switch(reason) if self.telegram else None,
            persist_callback=_persist_ks,
        )

        # S5 (parity-cycle round 2 — 2026-05-07): restore the kill-switch
        # consecutive-error counter from engine_state. Pre-S5 the counter
        # reset to 0 on every restart, so an engine that was 1 error
        # away from tripping got 2 free retries.
        if self._persisted_engine_state is not None:
            ks_errors = int(
                self._persisted_engine_state.get(
                    "kill_switch_consecutive_errors", 0,
                )
            )
            if ks_errors > 0:
                self.kill_switch._consecutive_errors = ks_errors
                logger.info(
                    "S5: kill_switch consecutive_errors restored to %d",
                    ks_errors,
                )

            # R-5 (parity-r3 cycle 3 — 2026-05-08): restore the data-stale
            # consecutive-empty-bars counter so an engine 1 bar from
            # tripping doesn't get a free reset on restart.
            empties = int(
                self._persisted_engine_state.get(
                    "consecutive_empty_bars", 0,
                )
            )
            if empties > 0:
                self._consecutive_empty_bars = empties
                logger.info(
                    "R-5: consecutive_empty_bars restored to %d (cap %d)",
                    empties, self.config.max_consecutive_empty_bars,
                )

            # R-6 (parity-r3 cycle 3 — 2026-05-08): restore the H15
            # per-hour rate limiter's deque. Old engine_state files that
            # predate this key load cleanly with an empty deque.
            try:
                ts_iso_list = self._persisted_engine_state.get(
                    "recent_trade_timestamps", [],
                ) or []
                restored_count = 0
                for ts_str in ts_iso_list:
                    try:
                        ts = pd.Timestamp(ts_str)
                        if ts.tzinfo is None:
                            ts = ts.tz_localize("UTC")
                        self.risk_limits._recent_trade_timestamps.append(ts)
                        restored_count += 1
                    except (ValueError, TypeError):
                        continue
                if restored_count > 0:
                    logger.info(
                        "R-6: _recent_trade_timestamps restored — %d entries",
                        restored_count,
                    )
            except Exception:  # noqa: BLE001 — defensive, never block startup
                logger.exception(
                    "R-6: failed to restore _recent_trade_timestamps; "
                    "starting with empty deque",
                )

        # H1: restore prior halt state (or clear if --clear-kill-state).
        prior_halted, prior_reason, prior_at = await self.store.get_kill_switch_state()
        if prior_halted and getattr(cfg, "clear_kill_state", False):
            logger.warning(
                "Kill-switch was halted at last shutdown (reason=%s, at=%s) "
                "but --clear-kill-state was passed — clearing state and "
                "starting clean",
                prior_reason, prior_at,
            )
            await self.store.set_kill_switch_state(False, reason=None)
            if self.telegram is not None:
                self.telegram.send(
                    "info",
                    f"Kill-switch CLEARED on startup (was: {prior_reason})",
                )
        elif prior_halted:
            restored_msg = (
                f"[restored from {prior_at}] {prior_reason}"
                if prior_reason
                else f"[restored from {prior_at}] (no reason recorded)"
            )
            logger.error(
                "Kill-switch state RESTORED from SQLite: halted=True, "
                "reason=%s — engine will refuse new signals until "
                "operator clears state via --clear-kill-state",
                restored_msg,
            )
            # S6 (parity-cycle round 2 — 2026-05-07): pass ``restored=True``
            # so the kill_switch sets ``halted=True`` from the persisted
            # row but does NOT re-fire ``needs_emergency_close``. The
            # prior session already flattened on the original trip; a
            # second close-all on every restart sends spurious Telegram
            # alerts and risks placing reduce-only orders against zero
            # positions if the ledger reconciliation lags.
            self.kill_switch.trip(restored_msg, restored=True)
        else:
            logger.info("Kill-switch state at startup: clean")

        # H4 (2026-05-06): restore ProfitScaleEstimator state from SQLite.
        # Without this, every restart drops gross_count to 0 → λ=0 →
        # est_gross collapses to est_theo for the first ~200 trades,
        # drifting away from V2-parity. See profit_scale_state schema.
        if self._profit_scale_estimator is not None:
            try:
                ps_state = await self.store.load_profit_scale_state()
            except Exception:  # noqa: BLE001
                logger.exception(
                    "H4: profit_scale_state load failed — starting cold"
                )
                ps_state = None
            if ps_state is not None:
                self._profit_scale_estimator.load_state(ps_state)
                logger.info(
                    "ProfitScaleEstimator state RESTORED from SQLite: "
                    "gross_ema=$%.2f/BTC, gross_count=%d, hedge_ema=$%.2f/BTC, "
                    "last_updated=%s, λ=%.3f",
                    self._profit_scale_estimator.gross_per_btc_ema_usd,
                    self._profit_scale_estimator.gross_count,
                    self._profit_scale_estimator.hedge_per_btc_ema_usd,
                    ps_state.get("last_updated", "?"),
                    self._profit_scale_estimator.shrinkage_lambda,
                )
            else:
                logger.info(
                    "ProfitScaleEstimator state at startup: cold "
                    "(no profit_scale_state row in SQLite)"
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
        #
        # D6 (parity 2026-05-07): V2 backtester has no RND counterpart.
        # Gated behind ``cfg.use_rnd_scaling`` (default False, V2 preset
        # leaves False) so V2-preset live behaviour matches the backtest.
        # Operators wanting tail-risk scaling can opt in via
        # ``cfg.use_rnd_scaling = True``.
        if cfg.use_rnd_scaling:
            self.rnd_monitor = RNDMonitor(
                config=RNDConfig(bar_minutes=cfg.bar_minutes),
            )
        else:
            logger.info(
                "RND tail-risk monitor disabled (cfg.use_rnd_scaling=False) "
                "— matches V2 backtester contract",
            )
            self.rnd_monitor = None

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

        # H5 (2026-05-06): set per-channel staleness thresholds on the
        # WS client. The global ``data_stale_seconds`` (90s default)
        # catches "EVERY channel stopped" zombies; these per-channel
        # thresholds catch PARTIAL silence — index dies while perp
        # keeps flowing. Conservative defaults so legitimate quiet
        # minutes don't false-trip:
        #   - index: 60s (Deribit publishes ~1Hz, 60s is generous)
        #   - perp trades 100ms: 180s (perp can be quiet at off-hours)
        #   - option trades 100ms: NOT enforced — option flow is
        #     genuinely sparse (testnet can be silent for hours)
        ccy = self.config.currency
        ccy_lower = ccy.lower()
        trades_iv = "100ms" if self.config.paper_mode else "raw"
        try:
            self.ws.set_channel_threshold(
                f"deribit_price_index.{ccy_lower}_usd", 60.0,
            )
            self.ws.set_channel_threshold(
                f"trades.{ccy}-PERPETUAL.{trades_iv}", 180.0,
            )
            logger.info(
                "Per-channel staleness thresholds wired: "
                "index=60s, perp_trades=180s",
            )
        except Exception:  # noqa: BLE001 - non-fatal
            logger.exception(
                "Failed to set per-channel staleness thresholds — "
                "global watchdog still active",
            )

        # Spawn background tasks
        self._background_tasks = [
            asyncio.create_task(self.ws.dispatch_forever(), name="ws_dispatch"),
            asyncio.create_task(self.bar_clock.run_forever(), name="bar_clock"),
            asyncio.create_task(self.telegram.flush_loop(), name="telegram_flush"),
            asyncio.create_task(self._heartbeat_loop(), name="heartbeat"),
            # H10 (2026-05-06): bar-advance watchdog. The hourly Telegram
            # heartbeat takes up to 60 minutes to notice a wedged engine.
            # The bar-advance watchdog checks every 30s that
            # ``stats.last_bar_timestamp`` advances at the expected
            # cadence; alerts via Telegram if no bar has closed for
            # 2× the bar interval (default 30 min for 15min bars).
            asyncio.create_task(
                self._bar_advance_watchdog(), name="bar_advance_watchdog",
            ),
            # H15 (2026-05-06): daily P&L summary at 00:00 UTC.
            asyncio.create_task(
                self._daily_summary_loop(), name="daily_summary",
            ),
        ]
        if self.vpin_stream is not None:
            self._background_tasks.append(
                asyncio.create_task(self.vpin_stream.run_forever(), name="vpin_ws"),
            )

        # F3 Tier 6: spawn the live dollar-bar spot stream alongside the
        # VPIN stream. Lazy-connect the REST client (the stream calls
        # connect() internally on first poll if needed).
        if self.dollar_bar_stream is not None:
            await self._dollar_bar_rest.connect()
            self._background_tasks.append(
                asyncio.create_task(
                    self.dollar_bar_stream.run_forever(),
                    name="dollar_bar_spot",
                ),
            )

        logger.info("TradingEngine running — waiting for stop signal")
        await self._stop.wait()
        logger.info("TradingEngine received stop — shutting down")

    async def _daily_summary_loop(self) -> None:
        """H15 (2026-05-06): emit a daily P&L summary at 00:00 UTC.

        ``TelegramNotifier.daily_summary`` was defined in monitoring/
        telegram.py but never called by the engine. Wire it now: at
        00:00 UTC each day, query SQLite for the last 24h of trades and
        send a one-line summary (count, win rate, realized PnL,
        equity).

        Sleeps until next 00:00 UTC; uses ``asyncio.wait_for`` against
        the stop event so the engine can shutdown cleanly even mid-day.
        """
        while not self._stop.is_set():
            now = pd.Timestamp.now(tz="UTC")
            tomorrow_midnight = (
                now.normalize() + pd.Timedelta(days=1)
            )
            sleep_seconds = (tomorrow_midnight - now).total_seconds()
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=sleep_seconds,
                )
                return  # stop signaled during sleep
            except asyncio.TimeoutError:
                pass

            if self.telegram is None or not self.telegram.enabled:
                continue
            if not getattr(self.config, "daily_summary_enabled", True):
                continue

            try:
                # Query last 24h of trades from SQLite
                yesterday = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=1)
                cursor = await self.store._conn.execute(  # noqa: SLF001
                    "SELECT COUNT(*), "
                    "SUM(CASE WHEN realized_pnl_usd > 0 THEN 1 ELSE 0 END), "
                    "COALESCE(SUM(realized_pnl_usd), 0.0) "
                    "FROM trades WHERE ended_at >= ? "
                    "AND status = 'completed'",
                    (yesterday.isoformat(),),
                )
                row = await cursor.fetchone()
                await cursor.close()
                n_trades = int(row[0] or 0)
                wins = int(row[1] or 0)
                realized = float(row[2] or 0.0)
                win_rate = (100.0 * wins / n_trades) if n_trades > 0 else 0.0
                equity = self.equity_tracker.current_equity if self.equity_tracker else 0.0
                date_str = yesterday.strftime("%Y-%m-%d")
                self.telegram.daily_summary(
                    date=date_str,
                    n_trades=n_trades,
                    win_rate_pct=win_rate,
                    realized_pnl_usd=realized,
                    equity_usd=equity,
                )
            except Exception:  # noqa: BLE001
                logger.exception("Daily summary task failed")

    async def _bar_advance_watchdog(self) -> None:
        """H10 (2026-05-06): alert when bars stop advancing.

        The hourly Telegram heartbeat (``_heartbeat_loop``) takes up to
        60 minutes to notice a wedged engine. The bar-advance watchdog
        runs at a 30s cadence and alerts via Telegram if
        ``stats.last_bar_timestamp`` hasn't advanced within 2× the bar
        interval — a 15min bar engine alerts at ~30 minutes of silence.

        Self-rate-limited: alerts at most once per ``alert_cooldown``
        seconds so a stuck engine doesn't spam Telegram.
        """
        check_interval_seconds = 30.0
        bar_interval_seconds = self.config.bar_minutes * 60.0
        # Tolerate up to 2 bars of silence before alerting (gives one
        # bar's worth of slack if the engine is briefly busy).
        max_silence_seconds = 2.0 * bar_interval_seconds
        # Start the engine clock from when the watchdog itself starts —
        # avoids alerting before the FIRST bar even closes.
        watchdog_start = pd.Timestamp.now(tz="UTC")
        last_alert_at: pd.Timestamp | None = None
        alert_cooldown_seconds = bar_interval_seconds  # alert at most once per bar

        while not self._stop.is_set():
            try:
                await asyncio.wait_for(
                    self._stop.wait(), timeout=check_interval_seconds,
                )
                return
            except asyncio.TimeoutError:
                pass

            now = pd.Timestamp.now(tz="UTC")
            last_bar = self.stats.last_bar_timestamp
            # Use start-of-watchdog as the reference until first bar closes
            reference_ts = last_bar if last_bar is not None else watchdog_start
            silence_seconds = (now - reference_ts).total_seconds()

            if silence_seconds <= max_silence_seconds:
                continue

            # Throttle alerts
            if (last_alert_at is not None
                    and (now - last_alert_at).total_seconds()
                    < alert_cooldown_seconds):
                continue
            last_alert_at = now

            msg = (
                f"H10 bar-advance watchdog: no bar close for "
                f"{silence_seconds:.0f}s "
                f"(last_bar={last_bar}, threshold={max_silence_seconds:.0f}s, "
                f"bar_interval={bar_interval_seconds:.0f}s). "
                f"Engine may be wedged."
            )
            logger.error(msg)
            if self.telegram is not None and self.telegram.enabled:
                try:
                    self.telegram.send("error", msg)
                except Exception:  # noqa: BLE001
                    logger.exception("Bar-advance watchdog: Telegram send raised")

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
                svi_recal_count=self._svi_recal_count,
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
        """Clean teardown of all services.

        Trade tasks (open straddles waiting on the holding period) are
        awaited up to ``holding_minutes + 60s`` — long enough for an
        in-flight trade to close cleanly. Background services (WS,
        bar clock, telegram flush, heartbeat, vpin WS) are cancelled
        first so they stop emitting bar/heartbeat events while the
        trade tasks drain.
        """
        logger.info("TradingEngine shutdown sequence starting")
        self._stop.set()

        # ── 1. Cancel + await background services ──
        for task in self._background_tasks:
            if not task.done():
                task.cancel()
        for task in self._background_tasks:
            try:
                await asyncio.wait_for(task, timeout=5.0)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass
        self._background_tasks.clear()

        # ── 2. Drain trade tasks (let open trades finish) ──
        # Reap any that have already finished, then wait on the rest
        # with a generous timeout. On timeout, cancel — the cancellation
        # path inside StraddleFlow.execute closes legs before re-raising
        # so we don't strand naked positions.
        active = [t for t in self._trade_tasks if not t.done()]
        if active:
            drain_timeout = self.config.holding_minutes * 60.0 + 60.0
            logger.info(
                "Shutdown: draining %d in-flight trade tasks (timeout %.0fs)",
                len(active), drain_timeout,
            )
            done, pending = await asyncio.wait(
                active,
                timeout=drain_timeout,
                return_when=asyncio.ALL_COMPLETED,
            )
            for t in pending:
                logger.warning(
                    "Trade task %s did not finish within drain timeout — "
                    "cancelling (StraddleFlow handles leg-close on cancel)",
                    t.get_name(),
                )
                t.cancel()
            for t in pending:
                try:
                    await asyncio.wait_for(t, timeout=30.0)
                except (asyncio.CancelledError, asyncio.TimeoutError):
                    pass
        self._trade_tasks.clear()

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
        # H1: drain in-flight kill-switch persistence tasks BEFORE
        # closing the store — otherwise a task firing right at shutdown
        # could try to write to a closed connection. Bounded wait
        # because each task is a single UPDATE.
        if getattr(self, "_ks_persist_tasks", None):
            pending_persists = [t for t in self._ks_persist_tasks if not t.done()]
            if pending_persists:
                try:
                    await asyncio.wait_for(
                        asyncio.gather(*pending_persists, return_exceptions=True),
                        timeout=5.0,
                    )
                except asyncio.TimeoutError:
                    logger.warning(
                        "Shutdown: %d kill-switch persist task(s) still pending "
                        "after 5s timeout",
                        len(pending_persists),
                    )
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
        """Main per-bar logic — signal → risk → execute → persist.

        H8 (2026-05-06): consults the bar_cursor table at the top of the
        method and refuses to re-process a bar it has already handled.
        This is defence-in-depth against:
        * a buggy aggregator that re-fires a past bar timestamp,
        * an accidental ``replay`` mode running against a live engine,
        * a manual debugger session that re-invokes ``_on_bar_close``
          for an already-processed bar.

        Marks the cursor AFTER ``process_bar`` returns successfully —
        if process_bar itself raises, the bar is NOT marked, and the
        operator can fix the underlying cause and replay safely.
        """
        # H8: refuse to re-process a bar already in the cursor.
        if self.store is not None:
            try:
                already = await self.store.is_bar_processed(bar.timestamp)
            except Exception:  # noqa: BLE001 - cursor lookup non-fatal
                logger.exception(
                    "H8: bar_cursor lookup failed for %s — proceeding "
                    "without idempotency guard", bar.timestamp,
                )
                already = False
            if already:
                logger.warning(
                    "H8: bar %s already in bar_cursor — skipping "
                    "(restart-idempotency guard)", bar.timestamp,
                )
                return

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
            # Do NOT mark the cursor — operator may want to replay.
            return

        # H8: mark the bar as processed now that signal evaluation
        # completed (even if no trade fired — the cursor is bar-level,
        # not trade-level). Best-effort: failure to persist must not
        # break the bar loop. signal_direction=0 here is a placeholder
        # — process_bar updates EngineStats.signals_long/short but
        # doesn't expose this bar's direction back to the caller, so we
        # keep the cursor row diagnostic-poor but idempotency-correct.
        if self.store is not None:
            try:
                await self.store.mark_bar_processed(
                    bar.timestamp,
                    signal_direction=0,
                    signal_reason=self.stats.last_signal_reason or None,
                )
            except Exception:  # noqa: BLE001
                logger.exception(
                    "H8: mark_bar_processed failed for %s", bar.timestamp,
                )

    async def process_bar(self, bar: Bar) -> None:
        """Run the signal → risk → execute pipeline for one bar.

        Exposed as a public method so tests can drive the engine with
        synthetic bars without spinning up the WebSocket.
        """
        # 1. Compute fresh TSRV for the bar.
        # Round-4 F6/F8 (parity-cycle 2026-05-07): switched from
        # ``perp_dataframe()`` (Deribit perp trades) to
        # ``tsrv_dataframe()`` (Binance Spot trades when wired,
        # falling back to perp when not). This closes the venue gap
        # vs the offline pipeline which trains TSRV on Binance Vision
        # spot aggTrades. ``Bar.tsrv_dataframe()`` returns the perp
        # DataFrame if ``Bar.tsrv_trades`` is None (back-compat for
        # tests and any non-spot-wired path).
        bar_tsrv = compute_bar_tsrv(
            bar.tsrv_dataframe(),
            bar_seconds=float(self.config.bar_seconds),
        )
        # NOTE: ``tsrv_history_buffer.append_bar`` for this bar's TSRV is
        # deferred to AFTER ``predictor.predict`` below — see the F-1 fix
        # comment near line ~1925. Pre-fix the append ran here, before
        # the predictor read the buffer, so the 6 TSRV-grid features
        # (tsrv_mean_*, tsrv_ffd, etc.) silently included vol_ann(T) in
        # rolling means while the offline pipeline strict-shifts(1).
        # Strict-past contract: at predict time, ``buf[-1]`` is the
        # most recent COMPLETED bar; current bar is appended once the
        # predictor has read what it needs.

        # 1b. Stale-data guard — refuse to signal on bars where we
        # couldn't trade. E.3.5 (2026-05-01): treat low-volume AND
        # zero-trade bars as untradeable for kill-switch purposes —
        # counter resets only on a fully-tradeable bar. Pre-E.3.5 a
        # venue outputting 1 trade per bar (below low-volume threshold)
        # would silently reset the counter every bar even though no
        # signal could fire.
        is_empty = bar.n_perp_trades == 0 and bar.n_option_trades == 0
        is_low_volume = bar.n_perp_trades < self.config.min_perp_trades_per_bar
        if is_empty or is_low_volume:
            self._consecutive_empty_bars += 1
            kind = "empty" if is_empty else "low-volume"
            logger.warning(
                "Bar %s %s (perp=%d opt=%d, count %d/%d), skipping signal",
                bar.timestamp, kind,
                bar.n_perp_trades, bar.n_option_trades,
                self._consecutive_empty_bars,
                self.config.max_consecutive_empty_bars,
            )
            if (
                self._consecutive_empty_bars >= self.config.max_consecutive_empty_bars
                and not self.kill_switch.is_halted()
            ):
                reason = (
                    f"data stale: {self._consecutive_empty_bars} consecutive "
                    f"untradeable bars (threshold "
                    f"{self.config.max_consecutive_empty_bars})"
                )
                self.kill_switch.trip(reason)
            return

        # Bar is fully tradeable — reset the counter.
        self._consecutive_empty_bars = 0

        # S2 (parity-cycle round 2 — 2026-05-07): drain any in-flight
        # trade-close coroutines briefly so sizing for THIS bar reads
        # the post-close equity instead of pre-close. Pre-S2 the
        # bar-aligned exit (D12) made trade-T's exit fire on bar T+1's
        # open; the equity_tracker.on_trade_closed update was a coroutine
        # racing with bar T+1's process_bar — coroutine ordering decided
        # whether sizing saw stale or fresh equity. The 200ms cap keeps
        # the bar loop responsive on a stuck trade-close path; on
        # timeout we proceed with the stale value (degraded but live)
        # and emit a warning metric so the operator notices.
        if self._trade_tasks:
            pending = [t for t in self._trade_tasks if not t.done()]
            if pending:
                _, still_pending = await asyncio.wait(pending, timeout=0.2)
                if still_pending:
                    logger.warning(
                        "S2: %d trade-close coroutine(s) did not drain within "
                        "200ms; bar %s sizing reads pre-close equity",
                        len(still_pending), bar.timestamp,
                    )

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
        # D-8 (parity-r3 cycle 2 — 2026-05-08): capture vol_ann_lagged
        # BEFORE the F-1 append below so it reflects bar T-1's vol_ann
        # (the value training env + offline backtester consume as the
        # GBM simulator's σ at decision time T). Pre-fix the engine
        # passed ``signal.bar_tsrv.vol_ann_continuous`` (current bar,
        # plus a wrong column — env uses raw ``vol_ann``, not
        # ``vol_ann_continuous``). After the F-1 append the buffer
        # would be at bar T; we'd then need ``iloc[-2]`` for lagged.
        # Capturing here avoids that fragile post-append index dance.
        # ``getattr`` with default keeps test stubs that don't construct
        # the buffer happy (some unit tests build engines via
        # ``TradingEngine.__new__`` to bypass the full async setup).
        vol_ann_lagged: float | None = None
        _hist = getattr(self, "tsrv_history_buffer", None)
        if _hist is not None:
            vol_ann_lagged = _hist.latest_vol_ann_lagged()
        # F-1 fix (round-3 audit, parity-r5 — 2026-05-08):
        # Append this bar's TSRV to the history buffer ONLY AFTER the
        # predictor has read it. Pre-fix this lived right after
        # ``compute_bar_tsrv`` above, so at predict time ``buf[-1] ==
        # current bar`` — the 6 TSRV-grid handlers
        # (``tsrv_mean_1d``, ``tsrv_mean_1d_lag3``, ``tsrv_mean_1w``,
        # ``tsrv_ffd``, ``up_jump_frac_ema50``, ``bpv_tsrv_ratio_ema50``)
        # silently included vol_ann(T) in their rolling means while the
        # offline pipeline strict-shifts(1) before the same rolling
        # (``vol_model/features/engineer.py:251``). The model trained on
        # shift(1) data was being fed shift(0) data live → systematic
        # bias on every prediction. This relocation enforces the
        # strict-past contract HAR-PD already follows
        # (``feature_assembler.py:_har_pd_state.append`` at L815-820).
        try:
            if self.tsrv_history_buffer is not None and bar_tsrv is not None:
                self.tsrv_history_buffer.append_bar(bar.timestamp, bar_tsrv)
        except Exception:  # noqa: BLE001
            logger.exception(
                "F-1 (round-3): tsrv_history_buffer append failed at %s",
                bar.timestamp,
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

        # D4 (parity 2026-05-07): warmup-buffer accumulation + EMA
        # consumption. The post-trade EMA update happens AFTER
        # risk-check passes and the trade actually fires (mirrors V2's
        # ``signal[i] != 0 and N[i] > 0`` gating).
        if math.isfinite(signal.edge_vol_pts):
            abs_edge = abs(signal.edge_vol_pts)
            if self._edge_ema is None:
                self._edge_warmup.append(abs_edge)
                if len(self._edge_warmup) >= self._EDGE_WARMUP_N:
                    self._edge_ema = float(np.median(self._edge_warmup))
                    logger.info(
                        "Edge-EMA warmup complete (%d bars): seeded with "
                        "median=%.3f, alpha=%.4f (500-bar EMA)",
                        len(self._edge_warmup), self._edge_ema,
                        self._EDGE_EMA_ALPHA,
                    )
                    self._edge_warmup.clear()
        # During warmup return V2's <96-bar fallback (35.0); after,
        # return the EMA. Mirrors backtester:1846 init.
        rolling_median_edge = (
            self._edge_ema if self._edge_ema is not None else 35.0
        )

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
        kelly_mult = lookup_kelly_mult_at(
            self._kelly_mult_series, bar.timestamp,
            signal_direction=signal.direction,
        )
        slip_usd = lookup_slip_usd_at(self._bar_slip_series, bar.timestamp)
        taker_slippage_pct = live_slippage_pct(
            slip_usd_per_btc=slip_usd,
            spot=index_price,
            observed_spread_usd=None,                     # spread unavailable here; override happens at order site
            fallback_pct=self.config.max_slippage_pct,
        )
        # First pass: get base notional with profit_scale=1.0 so the
        # estimator can size its slippage estimate on the pre-scale N.
        # Mirrors backtester.py:1854-1872 which calls compute_position_size
        # before profit_scaling.
        n_pre_ps = self.straddle_flow.size_position(
            equity=self.equity_tracker.current_equity,
            signal_direction=signal.direction,
            spot=index_price,
            trailing_1h_volume_btc=trailing_vol,
            dvol_pct=dvol_pct,
            edge_vp=signal.edge_vol_pts,
            rolling_median_edge=rolling_median_edge,
            peak_equity=self.equity_tracker.peak_equity,
            profit_scale=1.0,
            kelly_mult=1.0,                       # exclude Kelly from the estimator's N basis
        )

        # Profit-scale lookup (V2 wiring, E.1.1). ``None`` estimator →
        # 1.0 (no-op); under V2 preset, computes empirical-Bayes
        # shrinkage from realised gross EMA + theoretical edge×vega prior.
        profit_scale = 1.0
        if self._profit_scale_estimator is not None and n_pre_ps > 0.0 and signal.straddle is not None:
            iv_dec = (
                signal.iv_used_pct / 100.0
                if (signal.iv_used_pct is not None
                    and math.isfinite(signal.iv_used_pct)
                    and signal.iv_used_pct > 0.0)
                else 0.6
            )
            tau_years = max(
                (signal.straddle.expiry_dt - bar.timestamp).total_seconds()
                / (365.25 * 24 * 3600),
                1e-6,
            )
            vega_per_pt_usd = bsm_vega_per_vol_pt(
                spot=index_price,
                strike=signal.straddle.strike,
                iv_decimal=iv_dec,
                tau_years=tau_years,
            )
            profit_scale = self._profit_scale_estimator.estimate(
                edge_vp=signal.edge_vol_pts,
                vega_per_vol_pt_usd=vega_per_pt_usd,
                spot=index_price,
                jump_fraction=float(bar_tsrv.jump_fraction or 0.0),
                slip_usd_per_btc=slip_usd,
                n_btc=n_pre_ps,
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
            profit_scale=profit_scale,
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
        # D7 (parity 2026-05-07): once n_contracts is finalised, refresh
        # the IOC-cap pct using the V2 impact formula
        # (``2 * sidecar * sqrt(N) * exp(k * jf)``, capped at
        # ``ioc_cap_multiplier * sidecar``). The earlier call at L1708
        # was a pre-sizing estimate (no N) that would request a static
        # cap; this re-call sizes the cap to the actual trade.
        if n_contracts > 0.0:
            taker_slippage_pct = live_slippage_pct(
                slip_usd_per_btc=slip_usd,
                spot=index_price,
                observed_spread_usd=None,
                fallback_pct=self.config.max_slippage_pct,
                n_btc=n_contracts,
                jump_fraction=float(bar_tsrv.jump_fraction or 0.0),
                spread_jump_k=self.config.spread_jump_k,
                # ioc_cap_multiplier defaults to 2.0 (V2 default).
            )
        risk_check = self.risk_limits.check_signal(
            signal=signal,
            proposed_n_contracts_btc=n_contracts,
            now=bar.timestamp,
            # R-9 (parity-r3 cycle 3 — 2026-05-08): use the live bar's
            # spot, not the snapshot embedded in signal.straddle. Mirrors
            # offline ``vol_arb/sizing.py`` which sizes off the bar.
            bar_index_price=float(bar.index_price),
        )
        if not risk_check.passed:
            logger.info(
                "Signal rejected by risk at %s: %s",
                bar.timestamp, risk_check.reason,
            )
            self.stats.signals_rejected_by_risk += 1
            return

        # 7. Execute via straddle flow (fire-and-forget background task —
        # the bar loop must not block waiting for the 15-min holding period).
        # Tracked in self._trade_tasks so shutdown() can await completion
        # (E.1.4 2026-05-01).
        self.stats.signals_executed += 1
        # H15: feed the rate limiter BEFORE scheduling the trade so a
        # signal that would push us past the cap is rejected on the
        # next bar.
        self.risk_limits.record_trade_fired(at=bar.timestamp)
        # D4 (parity 2026-05-07): mirror V2 backtester at
        # ``backtester.py:2205-2206`` — only update the edge EMA on
        # bars where ``signal != 0 AND N > 0``. The check is
        # implicit here: we passed the risk gate and are about to
        # fire (n_contracts > 0 was already required upstream).
        if (
            self._edge_ema is not None
            and math.isfinite(signal.edge_vol_pts)
        ):
            abs_edge = abs(signal.edge_vol_pts)
            self._edge_ema += self._EDGE_EMA_ALPHA * (abs_edge - self._edge_ema)
        trade_task = asyncio.create_task(
            self._execute_and_persist(
                signal, n_contracts,
                dvol_pct=dvol_pct,
                edge_vp=signal.edge_vol_pts,
                rolling_median_edge=rolling_median_edge,
                peak_equity=self.equity_tracker.peak_equity,
                profit_scale=profit_scale,
                kelly_mult=kelly_mult,
                taker_slippage_pct=taker_slippage_pct,
                vol_ann_lagged=vol_ann_lagged,
            ),
            name=f"trade_{bar.timestamp.value}",
        )
        self._trade_tasks.append(trade_task)
        # Reap completed trade tasks so the list doesn't grow without bound.
        # Safe to do every bar — list mutation is single-threaded under the
        # asyncio loop.
        self._trade_tasks = [t for t in self._trade_tasks if not t.done()]

    async def _execute_and_persist(
        self,
        signal,
        n_contracts_btc: float,
        *,
        dvol_pct: float | None = None,
        edge_vp: float = 0.0,
        rolling_median_edge: float = 1.0,
        peak_equity: float = 0.0,
        profit_scale: float = 1.0,
        kelly_mult: float = 1.0,
        taker_slippage_pct: float = 0.5,
        vol_ann_lagged: float | None = None,
    ) -> None:
        """Run a straddle trade + update all state after completion.

        D-8 (parity-r3 cycle 2 — 2026-05-08): ``vol_ann_lagged`` is the
        prior bar's ``vol_ann`` (raw, not continuous), captured by the
        bar-loop BEFORE the F-1 ``tsrv_history_buffer`` append. The
        offline backtester (``DRLHedger.simulate``) and training env
        (``env.py:172``: ``df["vol_ann"].shift(1)``) both consume the
        lagged value as ``sigma`` in the GBM simulator. Pre-fix the
        engine passed ``signal.bar_tsrv.vol_ann_continuous`` (current
        bar, jump-corrected) which differs in BOTH lag and column.
        """
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
                profit_scale=profit_scale,
                kelly_mult=kelly_mult,
                taker_slippage_pct=taker_slippage_pct,
                vol_ann_entry_lagged=vol_ann_lagged,
            )
        except Exception as exc:  # noqa: BLE001 - log + kill-switch
            logger.exception("StraddleFlow.execute failed: %s", exc)
            self.kill_switch.on_trade_result(success=False)
            if self.telegram is not None:
                self.telegram.error(f"trade execution failed: {exc}")
            return

        if outcome.status == "completed":
            self.stats.trades_completed += 1
            # ── Net realized PnL (E.1.10 + E.1.15 2026-05-01) ──
            # Net = option_premium_pnl + hedge_pnl - option_fees - hedge_fees.
            # Mirrors V2 backtester's ``net_pnl_nc`` recipe so the I4
            # harness compares like-with-like.
            #
            # E.1.15: refuse to compute realized PnL when the trade
            # context is degenerate (no straddle, zero index price, NaN).
            # Pre-E.1.15 the engine fell back to a hardcoded
            # ``index_price = 60_000.0`` — silently masked ETH or
            # other-asset misconfiguration. Now we log ERROR and skip
            # the equity update; the engine's reconcile path catches
            # any residual drift later.
            straddle_ok = (
                outcome.signal.straddle is not None
                and outcome.signal.straddle.index_price > 0
                and math.isfinite(outcome.signal.straddle.index_price)
            )
            legs_ok = (
                outcome.entry_call is not None and outcome.exit_call is not None
                and outcome.entry_put is not None and outcome.exit_put is not None
            )
            if not straddle_ok:
                logger.error(
                    "Trade %s: skipping equity update — degenerate "
                    "straddle context (signal.straddle=%s index_price=%s). "
                    "ETH or other-asset misconfiguration?",
                    outcome.trade_id,
                    outcome.signal.straddle is not None,
                    getattr(outcome.signal.straddle, "index_price", None),
                )
                # Still mark kill-switch success so it doesn't trip on a
                # data-quality issue. Persistence path below skips too.
                self.kill_switch.on_trade_result(success=True)
                return

            index = float(outcome.signal.straddle.index_price)
            option_pnl_usd = 0.0
            option_fees_usd = 0.0
            if legs_ok:
                premium_in_btc = (
                    outcome.entry_call.average_price
                    + outcome.entry_put.average_price
                )
                premium_out_btc = (
                    outcome.exit_call.average_price
                    + outcome.exit_put.average_price
                )
                pnl_btc = (premium_in_btc - premium_out_btc) * outcome.n_contracts_btc
                if outcome.signal.direction == 1:
                    pnl_btc = -pnl_btc
                option_pnl_usd = pnl_btc * index
                # Deribit option fee: 0.03% of underlying per contract
                # (same maker & taker for options), CAPPED at 12.5% of
                # the option's premium (per Deribit schedule). Cap is
                # applied PER FILL against the actual fill premium —
                # using the recorded leg premiums (BTC) is more accurate
                # than the backtester's Brenner-Subrahmanyam approximation.
                _flat = self.config.option_maker_fee
                option_fees_btc = (
                    min(_flat, 0.125 * outcome.entry_call.average_price)
                    + min(_flat, 0.125 * outcome.entry_put.average_price)
                    + min(_flat, 0.125 * outcome.exit_call.average_price)
                    + min(_flat, 0.125 * outcome.exit_put.average_price)
                ) * outcome.n_contracts_btc
                option_fees_usd = option_fees_btc * index
            else:
                logger.warning(
                    "Skipping option PnL: missing leg fills "
                    "(entry_call=%s entry_put=%s exit_call=%s exit_put=%s) "
                    "— hedge PnL still folded in",
                    outcome.entry_call is not None,
                    outcome.entry_put is not None,
                    outcome.exit_call is not None,
                    outcome.exit_put is not None,
                )

            # Hedge PnL (E.1.10): from PerpHedger's running MTM.
            hedge_pnl_usd = 0.0
            hedge_fees_usd = 0.0
            if outcome.hedge_summary is not None:
                hedge_pnl_usd = float(getattr(
                    outcome.hedge_summary, "realized_pnl_usd", 0.0,
                ) or 0.0)
                total_usd_traded = float(getattr(
                    outcome.hedge_summary, "total_usd_traded", 0.0,
                ) or 0.0)
                # E1 (parity-cycle round 2 — 2026-05-07): the V2
                # backtester charges ``cost_rate = perp_fee +
                # perp_slippage_bps/1e4`` (= 7 bps total at the V2
                # default) on every hedge tick (backtester.py:803).
                # Pre-fix the live engine charged only ``perp_taker_fee``
                # (5 bps), silently dropping the 2-bps slippage
                # component. Across ~7,400 lifetime trades and
                # ~5-10 BTC perp turnover per trade, that's ~$200-370K
                # of un-modelled hedge cost, biasing live realized PnL
                # higher than V2 would predict and feeding the
                # profit_scale empirical-Bayes prior an upward bias.
                cost_rate = (
                    self.config.perp_taker_fee
                    + self.config.perp_slippage_bps / 10_000.0
                )
                hedge_fees_usd = cost_rate * total_usd_traded

            realized_pnl_usd = (
                option_pnl_usd + hedge_pnl_usd
                - option_fees_usd - hedge_fees_usd
            )

            self.equity_tracker.on_trade_closed(realized_pnl_usd)
            self.kill_switch.on_trade_result(success=True)

            # Profit-scale EMA update (V2 wiring, E.1.1; E4 parity-cycle round 2 2026-05-07).
            # V2 backtester at ``vol_arb/backtester.py:2303-2305`` feeds the
            # estimator the BS-decomposition GROSS per-BTC:
            #     ``vega_pnl_btc + theta_pnl_btc + delta_pnl_btc``
            # which is PRE-slippage and PRE-adverse-selection. ``ProfitScaleEstimator``
            # SEPARATELY estimates those costs in ``est_costs`` (full_slip +
            # hedge + option_fee), so feeding actual realized fills
            # double-subtracts those costs and biases the EMA below the V2
            # reference, pushing ``profit_scale`` toward its floor over time.
            #
            # Pre-E4 the live engine passed ``option_pnl_usd + hedge_pnl_usd``
            # — the actual fill-difference, which silently includes
            # slippage + adverse-selection. ``decompose_gross_pnl`` now
            # mirrors V2's BS Greeks shape: vega from BS vega × dIV (entry IV
            # → exit IV from the live SVI calibrator), theta from BS theta ×
            # dt_days, delta from realized perp MTM (the live analog of
            # V2's ``simulate(...)`` delta_pnl).
            #
            # Hedge cost (fees-on-perp-volume) is reported separately for
            # the estimator's audit trail; slippage is tracked by
            # exec_cost_ema, not folded in here.
            if self._profit_scale_estimator is not None:
                hedge_cost_usd = 0.0
                if outcome.hedge_summary is not None:
                    total_usd_traded = float(getattr(
                        outcome.hedge_summary, "total_usd_traded", 0.0,
                    ) or 0.0)
                    # E1 (2026-05-07): include perp_slippage_bps in
                    # the cost rate so profit_scale's hedge-cost
                    # input matches V2's ``perp_fee + slip`` formula.
                    _cost_rate = (
                        self.config.perp_taker_fee
                        + self.config.perp_slippage_bps / 10_000.0
                    )
                    hedge_cost_usd = (
                        _cost_rate * total_usd_traded
                    )
                # E4 (2026-05-07): exit IV from the live SVI calibrator at
                # trade-close time. ``NaN`` falls through to the helper
                # which treats it as zero vega contribution (degraded mode).
                exit_iv_decimal = self._fetch_exit_iv_decimal(outcome)
                entry_iv_decimal = (
                    float(outcome.signal.iv_used_pct) / 100.0
                    if (outcome.signal.iv_used_pct is not None
                        and math.isfinite(outcome.signal.iv_used_pct))
                    else float("nan")
                )
                # Tau at entry: years between entry-time and expiry.
                tau_entry_years: float = float("nan")
                if (
                    outcome.signal.straddle is not None
                    and outcome.started_at is not None
                ):
                    tau_entry_years = max(
                        (
                            outcome.signal.straddle.expiry_dt
                            - outcome.started_at
                        ).total_seconds() / (365.25 * 24 * 3600),
                        0.0,
                    )
                holding_dt_days = 0.0
                if outcome.started_at is not None and outcome.ended_at is not None:
                    holding_dt_days = max(
                        (outcome.ended_at - outcome.started_at).total_seconds()
                        / 86_400.0,
                        0.0,
                    )
                if (
                    math.isfinite(entry_iv_decimal)
                    and math.isfinite(tau_entry_years)
                    and tau_entry_years > 0.0
                ):
                    decomp = decompose_gross_pnl(
                        signal_direction=int(outcome.signal.direction),
                        entry_index_price=index,
                        strike=float(outcome.signal.straddle.strike),
                        entry_iv_decimal=entry_iv_decimal,
                        exit_iv_decimal=exit_iv_decimal,
                        entry_tau_years=tau_entry_years,
                        holding_dt_days=holding_dt_days,
                        n_btc=float(outcome.n_contracts_btc),
                        realized_hedge_pnl_usd=hedge_pnl_usd,
                    )
                    gross_pnl_usd = decomp.total_usd
                else:
                    # Degraded path: missing IV / tau → fall back to actual
                    # realized (matches pre-E4 behaviour but logs once).
                    logger.warning(
                        "E4: gross-PnL decomposition skipped (entry_iv=%s "
                        "tau=%s); falling back to realized fill-difference",
                        entry_iv_decimal, tau_entry_years,
                    )
                    gross_pnl_usd = option_pnl_usd + hedge_pnl_usd
                self._profit_scale_estimator.update_realized(
                    gross_pnl_usd=gross_pnl_usd,
                    hedge_cost_usd=hedge_cost_usd,
                    n_btc=outcome.n_contracts_btc,
                )
                # H4 (2026-05-06): persist EMA state after every trade so
                # a restart resumes the estimator at the live calibration
                # rather than collapsing back to lambda=0 / est_theo for
                # ~200 trades. Best-effort — DB write failure must not
                # block the trade close path.
                if self.store is not None:
                    try:
                        await self.store.save_profit_scale_state(
                            self._profit_scale_estimator.state_dict()
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "H4: profit_scale_state persist failed for trade %s",
                            getattr(outcome, "trade_id", "?"),
                        )

            # Update execution cost EMA from actual fill slippage
            if (outcome.entry_call and outcome.entry_call.average_price > 0
                    and outcome.entry_put and outcome.entry_put.average_price > 0
                    and outcome.signal.straddle is not None
                    and outcome.signal.straddle.index_price > 0):
                spot_for_ema = float(outcome.signal.straddle.index_price)
                # Approximate fill cost: total premium paid × spot
                total_premium = (
                    outcome.entry_call.average_price
                    + outcome.entry_put.average_price
                )
                fill_cost = total_premium * spot_for_ema   # $/BTC proxy
                self._exec_cost_ema += self._exec_cost_alpha * (
                    fill_cost - self._exec_cost_ema
                )

            # S1+S3+S5+S8 (parity-cycle round 2 — 2026-05-07): persist engine
            # state so a restart resumes equity / edge_ema / exec_cost_ema /
            # kill_switch consecutive_errors / svi_recal counters at their
            # current values rather than re-baselining at constructor
            # defaults. Best-effort — DB write failure must not block the
            # trade close path. Mirrors H4 pattern.
            if self.store is not None:
                try:
                    await self.store.save_engine_state(
                        self._build_engine_state_dict()
                    )
                except Exception:  # noqa: BLE001
                    logger.exception(
                        "S1+S3+S5+S8: engine_state persist failed for trade %s",
                        getattr(outcome, "trade_id", "?"),
                    )

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
            # ``term_slope_1w_2w`` cached on the Signal at predict time
            # (E.1.8 2026-05-01) so the value reflects signal-fire SVI
            # state, NOT post-hold T+15m state.
            term_slope = getattr(outcome.signal, "term_slope_1w_2w", None)
            await self._harness_writer.consolidate_and_write(
                outcome,
                realized_pnl_usd=realized_pnl_usd,
                kelly_mult=kelly_mult,
                taker_slippage_pct=taker_slippage_pct,
                term_slope_1w_2w=term_slope,
                config_preset=self.config.config_preset,
                profit_scale=profit_scale,
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
            # 2026-05-07: increment success counter so the operator-facing
            # heartbeat can show recal CHURN. Without this, ``svi_age``
            # appears stuck at a constant value when the heartbeat and
            # recal cadences are both 60 min (phase-locked) — an
            # operator can't distinguish "healthy 1h cadence" from
            # "calibrator frozen since startup". A monotonic counter
            # makes the difference unambiguous.
            self._svi_recal_count += 1
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
        # S8 (parity-cycle round 2 — 2026-05-07): persist svi_recal
        # counters here as well as on every trade-close. Recals fire on
        # an hourly cadence even on bars that don't trade, so the
        # per-trade hook isn't sufficient to make these counters
        # restart-resilient between trade gaps.
        if self.store is not None:
            try:
                await self.store.save_engine_state(
                    self._build_engine_state_dict()
                )
            except Exception:  # noqa: BLE001
                logger.debug(
                    "S8: engine_state persist after recal failed",
                    exc_info=True,
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
        """Fetch the latest index price for the perp hedger's step callback.

        P1-4 fix (2026-05-07): require the buffered price to be fresh
        within 60s. Pre-fix this returned the most recent buffered
        sample regardless of age — a silent index channel could feed
        an hours-old price into the perp hedger and produce wrong
        deltas. Now: if the buffer is empty OR the last sample is
        stale, fall back to a REST ``get_index_price`` call (already
        wired below as the fallback).
        """
        if self.index_buffer is not None:
            price = self.index_buffer.latest_fresh(max_age_seconds=60.0)
            if price is not None:
                return price
        # Fallback: REST call (also exercised when the buffer's latest
        # sample is older than 60s — the index channel is silent).
        info = await self.rest.get_index_price(
            index_name=f"{self.config.currency.lower()}_usd",
        )
        return float(info.get("index_price", 0.0) or 0.0)

    def _build_engine_state_dict(self) -> dict:
        """Snapshot the in-memory state covered by ``engine_state``.

        Centralised so per-trade and per-recal save paths stay in lockstep.
        Includes S1+S3 (equity/EMAs), S5 (kill_switch errors), S8 (svi
        recal counters).
        """
        # R-5 (parity-r3 cycle 3 — 2026-05-08): the data-stale kill-switch
        # counter survives restart. Pre-fix it reset to 0 on every boot,
        # so an engine that was 1 bar from tripping got a free reset.
        # R-6 (parity-r3 cycle 3 — 2026-05-08): the H15 per-hour rate
        # limiter's deque survives restart so the cap doesn't silently
        # re-base after a crash. Timestamps serialise as ISO strings
        # (UTC) so `pd.Timestamp(...)` round-trips cleanly.
        recent_trade_timestamps_iso: list[str] = []
        try:
            if self.risk_limits is not None:
                recent_trade_timestamps_iso = [
                    pd.Timestamp(ts).isoformat()
                    for ts in self.risk_limits._recent_trade_timestamps
                ]
        except Exception:  # noqa: BLE001 - best-effort, never block a save
            logger.exception(
                "R-6: failed to serialise _recent_trade_timestamps; "
                "engine_state will record empty deque",
            )
        return {
            "current_equity_usd": self.equity_tracker.current_equity,
            "peak_equity_usd": self.equity_tracker.peak_equity,
            "realized_pnl_usd": self.equity_tracker.realized_pnl,
            "n_trades_closed": self.equity_tracker.n_trades_closed,
            "edge_ema": self._edge_ema,
            "edge_warmup": list(self._edge_warmup),
            "exec_cost_ema_usd_per_btc": self._exec_cost_ema,
            "kill_switch_consecutive_errors": int(
                getattr(self.kill_switch, "_consecutive_errors", 0)
            ),
            "svi_recal_count": int(self._svi_recal_count),
            "svi_recal_failures": int(self._svi_recal_failures),
            "consecutive_empty_bars": int(self._consecutive_empty_bars),
            "recent_trade_timestamps": recent_trade_timestamps_iso,
        }

    def _fetch_exit_iv_decimal(self, outcome) -> float:
        """E4 (2026-05-07): pull the SVI ATM IV at the trade's strike,
        evaluated at trade-close time, and return as decimal (e.g. 0.75).

        Returns ``NaN`` when:
        - the calibrator is uncalibrated (warmup), or
        - the trade has no straddle context, or
        - the SVI surface lookup returns NaN / non-finite.

        ``NaN`` is the safe degraded mode for ``decompose_gross_pnl`` —
        the helper drops the vega leg when dIV is unavailable, leaving
        theta + delta as the gross prior.
        """
        try:
            if (
                self.svi_calibrator is None
                or not self.svi_calibrator.has_params
                or outcome.signal is None
                or outcome.signal.straddle is None
                or outcome.ended_at is None
            ):
                return float("nan")
            straddle = outcome.signal.straddle
            query = pd.DataFrame({
                "timestamp": [outcome.ended_at],
                "strike": [float(straddle.strike)],
                "expiry_dt": [pd.Timestamp(straddle.expiry_dt)],
                "index_price": [float(straddle.index_price)],
            })
            result = self.svi_calibrator.evaluate(query)
            if result is None or result.empty:
                return float("nan")
            iv_pct = float(result.iloc[0])
            if not math.isfinite(iv_pct) or iv_pct <= 0.0:
                return float("nan")
            return iv_pct / 100.0
        except Exception:
            logger.debug("E4 exit-IV lookup failed", exc_info=True)
            return float("nan")

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

    # ── O-9: WS reconnect open-orders reconcile ──────────────────────

    async def _reconcile_open_orders_on_reconnect(self) -> None:
        """O-9 (parity-r3 cycle 2 — 2026-05-08): reconcile venue's open
        orders with the engine after a WS reconnect.

        Strategy: query ``get_open_orders_by_currency`` and surface the
        result. We do NOT cancel orders aggressively — the inline
        polling in ``OrderManager.place_and_wait`` is each call's own
        custodian (it polls REST until terminal regardless of WS state).
        Instead we log + alert so the operator has visibility, and we
        cancel only orders OLDER than ``ws_reconcile_orphan_age_seconds``
        (default 300s = 5 min — well past any reasonable in-flight call's
        timeout) — those are the ones an interrupted prior session
        almost certainly left behind.

        Paper mode: skipped (no exchange-side state to reconcile).
        """
        if self.config.paper_mode:
            return
        if self.rest is None:
            return
        try:
            orders = await self.rest.get_open_orders_by_currency(
                currency=self.config.currency,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "O-9: get_open_orders_by_currency failed on reconnect: %s",
                exc,
            )
            if self.telegram is not None:
                self.telegram.send(
                    "error",
                    f"WS reconnect: failed to query open orders ({exc})",
                )
            return

        n = len(orders or [])
        logger.info(
            "O-9: WS reconnect — venue has %d open order(s)",
            n,
        )
        if n == 0:
            return

        # Age-gate cleanup: orders older than ``orphan_age_seconds`` are
        # almost certainly stale (the engine's inline polls all return
        # to terminal within seconds-to-30s).
        now_ms = pd.Timestamp.now(tz="UTC").value // 10**6
        orphan_age_ms = int(
            float(self.config.ws_reconcile_orphan_age_seconds) * 1000,
        )
        orphans: list[dict[str, Any]] = []
        in_flight: list[dict[str, Any]] = []
        for o in orders:
            creation_ms = int(
                o.get("creation_timestamp", 0) or 0,
            )
            age_ms = max(0, now_ms - creation_ms)
            if creation_ms > 0 and age_ms >= orphan_age_ms:
                orphans.append(o)
            else:
                in_flight.append(o)

        if in_flight:
            logger.info(
                "O-9: %d order(s) appear in-flight (age < %ds); leaving "
                "them to the inline polls",
                len(in_flight),
                int(self.config.ws_reconcile_orphan_age_seconds),
            )

        if not orphans:
            return

        cancelled = 0
        for o in orphans:
            order_id = str(o.get("order_id", ""))
            instrument = str(o.get("instrument_name", ""))
            label = str(o.get("label", "(no-label)"))
            try:
                await self.order_manager.cancel(order_id)
                cancelled += 1
                logger.warning(
                    "O-9: cancelled stale order %s on %s "
                    "(label=%s, age >= %ds)",
                    order_id, instrument, label,
                    int(self.config.ws_reconcile_orphan_age_seconds),
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "O-9: cancel of stale order %s failed: %s",
                    order_id, exc,
                )

        if self.telegram is not None and cancelled > 0:
            self.telegram.send(
                "error",
                f"WS reconnect cleanup: cancelled {cancelled} stale order(s) "
                f"(>= {int(self.config.ws_reconcile_orphan_age_seconds)}s old)",
            )

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
        if self.config.paper_mode:
            # No exchange-side positions exist in paper mode; the engine
            # simulates fills locally. DB-side state still matters (a
            # crash mid-trade leaves an open position record we should
            # resume), but exchange reconciliation is a no-op.
            db_positions: list[dict] = []
            if self.store is not None:
                try:
                    db_positions = await self.store.load_open_positions()
                except Exception as exc:
                    logger.error(
                        "Paper-mode startup: load_open_positions failed: %s",
                        exc,
                    )
            for db_row in db_positions:
                await self._resume_trade(db_row)
            if db_positions:
                logger.info(
                    "Paper-mode startup: resumed %d DB positions (no "
                    "exchange query in paper mode)",
                    len(db_positions),
                )
            else:
                logger.info("Paper-mode startup: no DB positions to resume")
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
                        # E-8 (parity-r3 cycle 2 — 2026-05-08): tiered IOC
                        # instead of naked market on orphan close. Same
                        # rationale as the cancel-exit path in
                        # `straddle_flow.execute`: respect the slippage
                        # cap even during recovery; tier-3 escalates to
                        # ``1.5×`` of the cap before logging CRITICAL.
                        result = await self.order_manager.place_ioc_taker(
                            instrument_name=instrument,
                            side=close_side,
                            amount=close_amount,
                            max_slippage_pct=self.config.cancel_exit_max_slippage_pct,
                            tiered=True,
                            label=(
                                f"orphan_close_{instrument}_{attempt}_"
                                f"{secrets.token_hex(8)}"
                            ),
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

        D-11 (parity-r3 cycle 2 — 2026-05-08): also rehydrates the DRL
        hedger's per-trade state (``hedge_inventory_btc``, frozen SVI
        slice, etc.) from the persisted ``drl_trade_state`` row. Pre-fix
        the hedger had no entry on resume → next perp tick computed
        ``delta_required = target - 0.0`` and over-traded by exactly
        the prior tick's hedge inventory.
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

        # D-11: rehydrate the DRL hedger if a persisted state row exists.
        if self.store is not None and self.live_drl_hedger is not None:
            try:
                drl_states = await self.store.load_drl_trade_states()
            except Exception:  # noqa: BLE001 — defensive, never block resume
                logger.exception(
                    "D-11: load_drl_trade_states failed; %s will be hedged "
                    "from inventory=0 (one-tick over-trade possible)",
                    trade_id,
                )
                drl_states = []
            for state in drl_states:
                if state["trade_id"] == trade_id:
                    try:
                        self.live_drl_hedger.resume_trade_from_dict(state)
                        logger.info(
                            "D-11: resumed DRL state for %s (hedge_inv=%.4f, tick=%d)",
                            trade_id, state["hedge_inventory_btc"], state["tick"],
                        )
                    except Exception:  # noqa: BLE001
                        logger.exception(
                            "D-11: resume_trade_from_dict failed for %s",
                            trade_id,
                        )
                    break
            else:
                logger.warning(
                    "D-11: no persisted DRL state for %s — hedger will start "
                    "from inventory=0 (next tick may over-trade by the prior "
                    "session's last hedge_inventory_btc)",
                    trade_id,
                )

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
        """Close both legs of a resumed straddle position.

        E-8 (parity-r3 cycle 2 — 2026-05-08): uses tiered IOC instead
        of naked market — see ``cancel_exit_max_slippage_pct`` in config.
        """
        trade_id = db_row["trade_id"]
        direction = db_row["signal_direction"]
        n = db_row["n_contracts_btc"]
        exit_side = "buy" if direction == -1 else "sell"

        logger.info("Closing resumed position %s: %s %.2f BTC", trade_id, exit_side, n)
        try:
            await asyncio.gather(
                self.order_manager.place_ioc_taker(
                    instrument_name=db_row["call_instrument"],
                    side=exit_side, amount=n,
                    max_slippage_pct=self.config.cancel_exit_max_slippage_pct,
                    tiered=True,
                    label=f"resume_exit_C_{trade_id}_{secrets.token_hex(8)}",
                    reduce_only=True,
                ),
                self.order_manager.place_ioc_taker(
                    instrument_name=db_row["put_instrument"],
                    side=exit_side, amount=n,
                    max_slippage_pct=self.config.cancel_exit_max_slippage_pct,
                    tiered=True,
                    label=f"resume_exit_P_{trade_id}_{secrets.token_hex(8)}",
                    reduce_only=True,
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
                            label=f"emergency_{instrument}_{secrets.token_hex(8)}",
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

        # 3. Flatten perp hedge (skip in paper mode — no real positions)
        if self.config.paper_mode:
            summary = f"Emergency close complete: {closed} fills. Reason: {reason}"
            logger.warning(summary)
            if self.telegram is not None:
                self.telegram.send("error", summary)
            return
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
                        label=f"emergency_perp_{instrument}_{secrets.token_hex(8)}",
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
        if self.config.paper_mode:
            # Paper mode has no exchange-side equity to reconcile against;
            # internal tracker is the source of truth.
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
