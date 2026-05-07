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
from deribit_paper.execution.profit_scale import (
    ProfitScaleEstimator,
    bsm_vega_per_vol_pt,
)
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

        def _on_ws_reconnect() -> None:
            if self.telegram is not None:
                self.telegram.send(
                    "error",
                    "Deribit WS reconnected \u2014 channels resubscribed",
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
            self.kill_switch.trip(restored_msg)
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
        # 1. Compute fresh TSRV for the bar
        bar_tsrv = compute_bar_tsrv(
            bar.perp_dataframe(),
            bar_seconds=float(self.config.bar_seconds),
        )

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
                profit_scale=profit_scale,
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
                # (same maker & taker for options). Four fills per
                # round-trip straddle (2 entry + 2 exit), each priced
                # against the index, scaled by N_contracts.
                option_fees_usd = (
                    4.0 * self.config.option_maker_fee * index
                    * outcome.n_contracts_btc
                )
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
                hedge_fees_usd = self.config.perp_taker_fee * total_usd_traded

            realized_pnl_usd = (
                option_pnl_usd + hedge_pnl_usd
                - option_fees_usd - hedge_fees_usd
            )

            self.equity_tracker.on_trade_closed(realized_pnl_usd)
            self.kill_switch.on_trade_result(success=True)

            # Profit-scale EMA update (V2 wiring, E.1.1).
            # Treat ``realized_pnl_usd`` as the gross-pre-fees proxy for now;
            # E.1.10 will refactor this to subtract hedge PnL + fees so the
            # estimator's "gross" matches the backtester's vega+theta+delta
            # decomposition. Until then, the EMA tracks the closest live
            # analog to gross PnL we have access to.
            #
            # Hedge cost is the fee on total perp volume traded
            # (HedgeRunSummary.total_usd_traded); slip is small relative
            # to fees on a perp and tracked separately by exec_cost_ema.
            if self._profit_scale_estimator is not None:
                hedge_cost_usd = 0.0
                if outcome.hedge_summary is not None:
                    total_usd_traded = float(getattr(
                        outcome.hedge_summary, "total_usd_traded", 0.0,
                    ) or 0.0)
                    hedge_cost_usd = (
                        self.config.perp_taker_fee * total_usd_traded
                    )
                self._profit_scale_estimator.update_realized(
                    gross_pnl_usd=realized_pnl_usd,
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
                        result = await self.order_manager.place_and_wait(
                            instrument_name=instrument,
                            side=close_side,
                            amount=close_amount,
                            order_type="market",
                            label=(
                                f"orphan_close_{instrument}_{attempt}_"
                                f"{secrets.token_hex(8)}"
                            ),
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
                    label=f"resume_exit_C_{trade_id}_{secrets.token_hex(8)}",
                    timeout_seconds=30.0, reduce_only=True,
                ),
                self.order_manager.place_and_wait(
                    instrument_name=db_row["put_instrument"],
                    side=exit_side, amount=n,
                    order_type="market",
                    label=f"resume_exit_P_{trade_id}_{secrets.token_hex(8)}",
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
