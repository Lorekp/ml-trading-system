"""Configuration for the Deribit paper-trading engine.

Single source of truth for all knobs.  Loads credentials + env-specific
settings from the project ``.env`` file, and mirrors strategy-level
defaults from ``BacktestConfig`` / ``DRLHedgingConfig`` so the paper run
reproduces the backtester's behaviour exactly.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

try:
    from dotenv import load_dotenv
    _DOTENV_AVAILABLE = True
except ImportError:
    _DOTENV_AVAILABLE = False


def _load_env() -> None:
    """Load ``.env`` from the project root if python-dotenv is installed.

    Kept optional so tests that don't need environment variables don't
    have to install python-dotenv.  If ``.env`` is missing or dotenv is
    not installed, we fall back to ``os.environ`` directly.
    """
    if not _DOTENV_AVAILABLE:
        return
    project_root = Path(__file__).resolve().parent.parent
    env_path = project_root / ".env"
    if env_path.exists():
        load_dotenv(dotenv_path=env_path, override=False)


@dataclass
class PaperConfig:
    """All configuration for the paper-trading engine.

    Grouped by concern: venue/credentials, strategy knobs, execution,
    risk, monitoring, paths.  Most strategy knobs mirror
    ``vol_arb.backtester.BacktestConfig`` defaults so a paper run is a
    faithful live replica of the backtest.
    """

    # ── Venue / credentials ─────────────────────────────────────────
    # Deribit testnet by default — flip to ``prod`` only after Phase 7
    # validation passes + passphrase confirmation at the CLI.
    venue: str = "testnet"  # "testnet" | "prod"
    client_id: str = ""
    client_secret: str = ""
    rest_base_url: str = "https://test.deribit.com/api/v2"
    ws_url: str = "wss://test.deribit.com/ws/api/v2"
    # Paper mode: connect to the configured venue (typically prod) for
    # real market data, but never authenticate for private endpoints
    # and never submit orders. Requires no Deribit API credentials.
    # Implies ``dry_run=True`` (enforced by the CLI). Used for live
    # signal-stack validation against real mainnet quotes without any
    # account-side risk.
    paper_mode: bool = False
    # Synthetic starting equity (USD) when ``paper_mode`` is on and no
    # exchange account is being read. Defaults match V2 backtest
    # conventions ($100K) so the per-trade dollar PnL distribution is
    # directly comparable to ``BTC_v2_trade_log.parquet``.
    paper_initial_equity_usd: float = 100_000.0

    # ── Strategy (mirrors BacktestConfig) ──────────────────────────
    currency: str = "BTC"
    holding_minutes: int = 15
    bar_minutes: int = 15
    base_spread_iv: float = 1.5
    spread_jump_k: float = 3.0
    entry_threshold: float = 25.0       # mirrors BacktestConfig: shorts need 25 vp edge
    # NOTE: ``default`` preset uses 30 (legacy live behaviour). The V2
    # production stack drops this to 12 because the contango filter
    # (``skip_long_if_term_slope_gt=0.0``) already removes the bad-regime
    # longs, so surviving longs have structural positive edge — see
    # ``BacktestConfig.with_btc_production_filters_v2`` docstring. Flipped
    # to 12.0 by ``PaperConfig.with_v2_preset()``.
    entry_threshold_long: float = 30.0
    # V2: drop LONG signals when 1w-2w term slope > 0 (IV contango).
    # ``None`` disables the filter (legacy live default). The V2 preset
    # sets this to 0.0 to match the backtest contract.
    skip_long_if_term_slope_gt: float | None = None
    # D2 (parity 2026-05-07): mute LONG (+1) signals at these UTC hours.
    # Mirrors ``BacktestConfig.skip_long_hours_utc``. The V2 production
    # backtest uses ``(14, 15, 16)`` to drop US-cash-open longs that fade
    # 4/5 historical years. Empty tuple disables (default; preserves the
    # legacy live behaviour for non-V2 presets).
    skip_long_hours_utc: tuple[int, ...] = ()
    option_maker_fee: float = 0.0003    # Deribit: 0.03% of underlying (same maker & taker)
    perp_taker_fee: float = 0.0005     # Deribit: 0.05% taker (maker gets -0.01% rebate)
    perp_slippage_bps: float = 2.0
    delta_hedge_band: float = 0.05
    hedge_sim_steps: int = 10
    maker_adverse_selection_iv: float = 0.5
    # DTE filter for which option expiries to consider for the straddle leg
    dte_min_days: int = 1
    dte_max_days: int = 14
    # Moneyness band for ATM straddle strike selection
    moneyness_lo: float = 0.95
    moneyness_hi: float = 1.05

    # ── Position sizing ────────────────────────────────────────────
    initial_equity: float = 1.0              # demo BTC (testnet starts with ~10 BTC)
    # ``default`` preset matches the legacy live conservative posture
    # (5%). The V2 preset bumps this to 0.15 — the 3x leverage difference
    # is the single largest dollar driver of live-vs-backtest divergence
    # per ``01-landscape.md``. Flipped to 0.15 by
    # ``PaperConfig.with_v2_preset()``.
    target_margin_utilization: float = 0.05
    # Conviction-weighted size scaling (mirrors BacktestConfig): scales
    # position by trailing-window profit factor floored at
    # ``profit_scaling_floor``. Off in ``default``, on under V2 preset.
    use_profit_scaling: bool = False
    profit_scaling_floor: float = 0.3
    # Deribit ATM option IM = max(0.15 - OTM/spot, 0.10) + mark_price.
    # For ATM straddles: 0.15 + mark ≈ 0.20.  The backtester uses 0.10
    # (understates real margin); the paper engine uses the realistic value
    # so sizing and risk checks match what the exchange will enforce.
    im_fraction: float = 0.20
    max_participation_rate: float = 0.10
    short_sizing_penalty: float = 0.50
    baseline_target_dvol: float = 50.0
    max_jump_fraction_for_shorts: float = 0.15
    # D6 (parity 2026-05-07): RND tail-risk size scaling has no V2
    # backtester counterpart. Default OFF so V2 preset matches the
    # backtest contract; opt-in for live operators who want the
    # additional defence-in-depth (memory: project_rnd_monitor_validated
    # — historical halt rate ~0%, but the size-scaling path can shrink
    # positions on real tail-risk events).
    use_rnd_scaling: bool = False

    # ── Execution ──────────────────────────────────────────────────
    maker_wait_seconds: int = 60              # wait this long for maker fill before falling back to taker
    intra_bar_hedge_ticks: int = 10           # matches backtester's hedge_sim_steps
    order_retry_max: int = 3                  # retry transient errors this many times
    max_slippage_pct: float = 0.5             # IOC cap above best ask / below best bid
    # E.1.6: when True, all 4 production IOC sites in straddle_flow.py
    # escalate through the 3-tier ladder
    # (touch+1tick → cap → bounded-IOC). Off by default to preserve the
    # legacy ``default`` preset; flipped on by ``with_v2_preset``.
    use_tiered_taker: bool = False
    sizing_liq_cap: float = 1.5              # max upscale in liquid markets (conservative for live)
    sizing_edge_cap: float = 3.0             # mirrors BacktestConfig: max edge multiplier
    sizing_dd_floor: float = 0.25            # mirrors BacktestConfig: min scale at deep drawdown
    execution_cost_ema_bars: int = 200        # bars for rolling execution cost EMA

    # ── Risk ───────────────────────────────────────────────────────
    max_open_positions: int = 5
    max_margin_util_hard: float = 0.20        # hard cap; kill switch trips above this
    max_single_trade_btc: float = 2.0
    kill_switch_daily_dd_pct: float = 2.0     # halt if daily DD > this (% of peak equity)
    kill_switch_consecutive_errors: int = 3
    # E.2.8: per-order latency above this threshold is treated as a soft
    # execution error and feeds the consecutive-error counter. Default
    # disabled (inf) on testnet because testnet latency varies wildly;
    # mainnet operators should set ~3000ms.
    kill_switch_latency_ms: float = float("inf")
    network_outage_seconds: int = 120         # halt if no WS messages for this long
    # WS data-stale watchdog: forcibly close + reconnect the WS if no
    # subscription message arrives in this many seconds. Sits BELOW
    # ``network_outage_seconds`` so the watchdog triggers a recovery
    # BEFORE the kill switch halts trading. ``0`` disables it (rely on
    # TCP-level ping/pong only — NOT recommended on mainnet, where
    # zombie connections have been observed).
    ws_data_stale_seconds: float = 90.0
    max_consecutive_empty_bars: int = 3       # trip kill switch after N bars with zero trades
    min_perp_trades_per_bar: int = 5          # skip signal if bar has fewer perp trades
    max_orphan_close_retries: int = 3         # C1: retry orphaned position closes on startup
    svi_max_stale_hours: float = 4.0          # C3: refuse to trade if SVI older than this
    equity_reconcile_bars: int = 96           # R3: reconcile with exchange every N bars (~1 day)

    # ── DRL hedger ─────────────────────────────────────────────────
    # W5 (2026-04-12): promoted SAC v4 (true bucketed VPIN, 5M steps)
    # after statistically significant A/B improvement over v3.
    # Rollback: change these two paths back to production_sac_v3_final.
    drl_policy_path: str = "vol_arb/drl_hedging/models/production_sac_v4_final.zip"
    drl_vecnormalize_path: str = "vol_arb/drl_hedging/models/production_sac_v4_final_vecnormalize.pkl"
    drl_svi_params_path: str = "vol_arb/output/svi_surface_params_1h.parquet"
    # Algorithm for model loading ("risk_averse_sac" = SAC family, "tqc" = TQC)
    drl_algorithm: str = "risk_averse_sac"

    # ── Live VPIN ─────────────────────────────────────────────────
    # Warm-start state from the offline recompute (5y of bucket history).
    # Set to "" to disable live VPIN (falls back to vpin_fallback constant).
    vpin_state_path: str = "vol_arb/output/vpin_streamer_state.json"
    vpin_fallback: float = 0.189          # training-distribution mean
    binance_ws_url: str = ""              # sentinel -> resolved from currency in __post_init__

    # ── Model inference ────────────────────────────────────────────
    # lgbm_models_dir holds the LightGBM ``.txt`` + ``_features.json``
    # manifests for the walk-forward variant in use.  W3.1 (2026-04-11)
    # switched this to the Stage 18 T2.1 pruned+SVI variant. C
    # (parity 2026-05-07) flips the default to the 50-feature HAR-PD
    # variant (``_pruned_svi_har_pd_v1``) — the model the V2 backtest
    # was trained against and which the aligner β* was calibrated for.
    # The 6 extra features are computed live by the FeatureAssembler
    # (see deribit_paper/features/har_pd_live.py); the byte-identity
    # test ``tests/test_har_pd_live_byte_identity.py`` locks the
    # numerical equivalence vs the offline parquet.
    lgbm_models_dir: str = "vol_model/output/models/expanding_pruned_svi_har_pd_v1"
    # The pruned+SVI variant reuses TransVol checkpoints from the
    # earlier ``expanding_hybrid/`` run — no TransVol re-training was
    # needed because T2.1 only prunes LightGBM-level features.  So the
    # transvol weights live in a different directory than the LightGBM
    # models.  Loader must accept both paths independently.
    transvol_models_dir: str = "vol_model/output/models/expanding_hybrid"
    # Historical feature parquets loaded at engine startup to seed the
    # baseline row the FeatureAssembler uses when it can't compute a
    # feature fresh on the live bar.  Two parquets are merged: the
    # 126-column base features + the 13-column SVI features.
    features_parquet: str = ""              # sentinel -> resolved from currency + bar_minutes
    svi_features_parquet: str = ""          # sentinel -> resolved from bar_minutes

    # ── Config preset metadata (set by CLI / with_v2_preset) ────────
    # Tagged into harness trade rows so the I4 reproduction analysis
    # can attribute drift to a specific preset. "default" preserves
    # the legacy live behaviour; "v2" enables the production parity
    # stack (β*, Kelly K1, realistic_taker, contango filter, …).
    config_preset: str = "default"

    # ── V2 sidecars (predictions, Kelly, slippage) ─────────────────
    # Aligned-bias HAR-PD predictions parquet — used as the parquet
    # source for β* extraction (``scripts/extract_aligner_betas.py``)
    # and as the cross-check against fresh LightGBM inference.
    # Default: empty string disables β* application. The V2 preset
    # points this at the production aligner output.
    aligned_predictions_parquet: str = ""
    # Per-month aligner β* JSON sidecar produced by the extract script.
    # Live engine reads this at startup, picks up the appropriate
    # β* per current calendar month (with most-recent-prior fallback)
    # and adds it to fresh LightGBM predictions.
    aligner_betas_json: str = ""
    # Per-bar Kelly K1 multiplier sidecar (parquet, indexed on the
    # 15-min UTC bar timestamp). Empty string disables Kelly boost.
    kelly_mult_path: str = ""
    # Per-bar empirical taker slip USD/BTC sidecar (parquet). Live
    # engine consults this when sizing IOC limit prices in C4. Empty
    # string falls back to ``max_slippage_pct`` flat cap.
    bar_exec_quality_path: str = ""

    # ── SVI live calibration ───────────────────────────────────────
    svi_recalibrate_hours: int = 1           # run calibrate_surface() every N hours
    svi_rolling_window_hours: int = 2        # use the last N hours of option trades as input
    svi_bucket_freq: str = "1h"

    # ── TSRV live ──────────────────────────────────────────────────
    tsrv_rolling_window_hours: int = 2       # use the last N hours of perp trades

    # ── Market data buffers ────────────────────────────────────────
    perp_buffer_hours: int = 24              # 24h rolling perp trade buffer in memory
    option_buffer_hours: int = 4             # 4h rolling option trade buffer in memory

    # ── Monitoring / Telegram ──────────────────────────────────────
    telegram_bot_token: str = ""
    telegram_chat_id: str = ""
    heartbeat_minutes: int = 60              # Telegram heartbeat cadence
    log_level: str = "INFO"

    # ── Paths ──────────────────────────────────────────────────────
    state_dir: str = "deribit_paper/state"
    logs_dir: str = "deribit_paper/logs"
    metrics_parquet: str = "deribit_paper/state/live_metrics.parquet"
    sqlite_db: str = "deribit_paper/state/paper.db"

    # ── Small-account sizing ─────────────────────────────────────────
    # When True, size_position() returns the exchange minimum (0.1 BTC)
    # instead of 0.0 when the formula underproduces.  The risk check
    # still gates whether the trade is affordable.
    min_trade_floor: bool = False

    # ── Safety ─────────────────────────────────────────────────────
    # This MUST default to True.  The flag is checked by every execution
    # path and by the CLI before allowing any live orders.  The only way
    # to disable it is an explicit ``--live`` flag + passphrase confirm.
    dry_run: bool = True
    # E.2.3 escape hatch (set via CLI ``--allow-stale-aligner``). Defaults
    # off — the V2 staleness check refuses to start at ≥2 months stale.
    # When True, the check downgrades to a WARN and the lookup falls
    # back to the most-recent-prior β* month.
    allow_stale_aligner: bool = False
    # H1 escape hatch (set via CLI ``--clear-kill-state``). When True
    # AND the SQLite ``kill_switch_state`` row is halted=True at engine
    # startup, the engine clears the row to ``halted=False`` BEFORE
    # constructing the in-memory KillSwitch, so the engine starts in a
    # clean state. Without this flag, a tripped kill-switch from a
    # previous run is restored on startup and the engine refuses
    # signals until the operator clears the state explicitly.
    clear_kill_state: bool = False
    # H9 escape hatches (per-sidecar staleness override). Default off —
    # engine refuses to start when a sidecar is older than the per-
    # sidecar max_age_days. Mirrors the aligner pattern.
    allow_stale_kelly: bool = False
    allow_stale_slip: bool = False
    allow_stale_features: bool = False
    allow_stale_svi_features: bool = False

    # H15 caps (Tier 2 hardening, 2026-05-06). Defence-in-depth on top
    # of the existing per-trade and position-count caps.
    # max_trades_per_hour: 0 disables; recommended 6 for V2 (which
    # fires ~4 trades/day on average; 6/hour = 1.5×/15min bar = ~9×
    # natural rate, so it only catches runaway signals not legit bursts).
    max_trades_per_hour: int = 6
    # max_gross_notional_usd: 0 disables; recommended 30k (~30% of the
    # default $100k paper equity, in line with margin_util_hard=0.20
    # but expressed in dollar notional).
    max_gross_notional_usd: float = 30_000.0
    # H15: send a daily P&L summary at 00:00 UTC. The summary helper
    # already exists on TelegramNotifier (line 245); this enables it.
    daily_summary_enabled: bool = True

    def __post_init__(self) -> None:
        """Resolve sentinel defaults from (currency, bar_minutes)."""
        ccy_lower = self.currency.lower()
        asset_pair = f"{self.currency.upper()}USDT"
        tf = f"{self.bar_minutes}m"
        if not self.binance_ws_url:
            self.binance_ws_url = (
                f"wss://fstream.binance.com/ws/{ccy_lower}usdt@aggTrade"
            )
        if not self.features_parquet:
            self.features_parquet = (
                f"vol_model/output/{asset_pair}_features_{tf}.parquet"
            )
        if not self.svi_features_parquet:
            self.svi_features_parquet = (
                f"vol_arb/output/svi_features_{tf}.parquet"
            )

    @property
    def bar_seconds(self) -> int:
        """Bar duration in seconds (e.g. 900 for 15-min bars)."""
        return self.bar_minutes * 60

    @property
    def bars_per_day(self) -> int:
        """Number of bars per calendar day (e.g. 96 for 15-min bars)."""
        return 1440 // self.bar_minutes

    @property
    def asset_pair(self) -> str:
        """Exchange asset pair string (e.g. 'BTCUSDT')."""
        return f"{self.currency.upper()}USDT"

    @classmethod
    def from_env(cls) -> "PaperConfig":
        """Construct a ``PaperConfig`` populated from environment variables.

        Credential fields are loaded from ``.env`` via python-dotenv if
        available, otherwise from ``os.environ`` directly.  Unset
        credentials remain empty strings — the client will fail loudly
        on the first authenticated call rather than silently hanging.
        """
        _load_env()
        cfg = cls()
        cfg.client_id = os.environ.get("DERIBIT_TESTNET_CLIENT_ID", "")
        cfg.client_secret = os.environ.get("DERIBIT_TESTNET_CLIENT_SECRET", "")
        rest_env = os.environ.get("DERIBIT_TESTNET_BASE_URL")
        if rest_env:
            cfg.rest_base_url = rest_env
        ws_env = os.environ.get("DERIBIT_TESTNET_WS_URL")
        if ws_env:
            cfg.ws_url = ws_env
        cfg.telegram_bot_token = os.environ.get("TELEGRAM_BOT_TOKEN", "")
        cfg.telegram_chat_id = os.environ.get("TELEGRAM_CHAT_ID", "")
        # PAPER_DRY_RUN env var can set dry_run=False only if explicitly "false"
        dry_run_env = os.environ.get("PAPER_DRY_RUN", "true").lower()
        cfg.dry_run = dry_run_env != "false"
        return cfg

    @classmethod
    def for_small_account(cls, equity_usd: float = 750.0) -> "PaperConfig":
        """Create a config tuned for small live accounts ($500–$2000).

        Enables ``min_trade_floor`` so the sizing formula returns the
        exchange minimum (0.1 BTC) when equity is too low for the
        composite_v2 formula to produce a full lot.  As equity grows
        past ~$20K the formula naturally takes over and the floor
        becomes non-binding.

        Does NOT hardcode ``max_single_trade_btc`` — that would lock
        the account to 0.1 BTC forever even as it scales.
        """
        cfg = cls.from_env()
        btc_approx = 100_000.0  # overridden from REST at startup
        cfg.max_open_positions = 1
        cfg.kill_switch_daily_dd_pct = 5.0
        cfg.initial_equity = equity_usd / btc_approx
        cfg.min_trade_floor = True
        return cfg

    @classmethod
    def with_v2_preset(cls, base: "PaperConfig | None" = None) -> "PaperConfig":
        """Return a config aligned with ``BacktestConfig.with_btc_production_filters_v2()``.

        Invariants flipped or set:

        * ``entry_threshold`` / ``entry_threshold_long``: V1/V2 thresholds.
        * ``skip_long_if_term_slope_gt``: None → 0.0 (drop longs in IV contango).
        * ``skip_long_hours_utc``: () → (14, 15, 16) — D2 (2026-05-07).
        * ``dte_min_days``: 1 → 5 — D1 (2026-05-07).
        * ``target_margin_utilization``: 0.05 → 0.15 (3x leverage closer to backtest).
        * ``use_profit_scaling``: False → True (conviction-weighted sizing).
        * ``profit_scaling_floor``: stays 0.3 (matches BacktestConfig V2).
        * ``max_single_trade_btc``: 2.0 → 0.0 (= disabled per V2) — D10 (2026-05-07).
        * ``use_rnd_scaling``: stays False (V2 has no RND monitor) — D6 (2026-05-07).
        * ``aligned_predictions_parquet`` → production aligner parquet.
        * ``aligner_betas_json`` → derived per-month β* JSON path.
        * ``kelly_mult_path`` → V2 Kelly K1 sidecar.
        * ``bar_exec_quality_path`` → V2 realistic_taker slippage sidecar.

        ``base`` (optional) lets the caller pre-populate venue / credentials
        from ``PaperConfig.from_env()`` first; the preset only overrides
        strategy fields and leaves env-bound fields alone.

        Single source of truth: this method's expected output is asserted
        against ``BacktestConfig.with_btc_production_filters_v2()`` in
        ``deribit_paper/tests/test_config_v2_parity.py``. Updating the V2
        invariants in either side without updating the cross-check test
        will fail Phase A's regression gate.
        """
        cfg = base if base is not None else cls()
        cfg.config_preset = "v2"
        cfg.entry_threshold = 22.0  # V1 production stack short threshold
        cfg.entry_threshold_long = 12.0
        cfg.skip_long_if_term_slope_gt = 0.0
        # D2 (parity 2026-05-07): mute longs at US cash-open hours. The V2
        # backtest applies this in ``_apply_entry_filters``; the live
        # predictor mirrors the same shape after the contango filter.
        cfg.skip_long_hours_utc = (14, 15, 16)
        # D1 (parity 2026-05-07): live default of 1 lets the engine trade
        # the 1-4 DTE band the V2 backtester deliberately skips
        # (theta-decay-dominated, fee-vs-vega ratio is poor).
        cfg.dte_min_days = 5
        cfg.target_margin_utilization = 0.15
        cfg.use_profit_scaling = True
        cfg.profit_scaling_floor = 0.3
        # D10 (parity 2026-05-07): V2 backtester has no per-trade BTC cap
        # (``BacktestConfig.max_position_btc=0.0``). The live default 2.0
        # was a defence-in-depth knob; under V2 we disable it (``0.0``)
        # so dynamic_btc_cap is the only ceiling. The cap is still
        # honoured under non-V2 presets where the field stays at 2.0.
        cfg.max_single_trade_btc = 0.0
        # D6 (parity 2026-05-07): no RND monitor in V2 backtester. Stays
        # False under the V2 preset; opt-in for non-V2 live deployments
        # via ``cfg.use_rnd_scaling = True``.
        cfg.use_rnd_scaling = False
        cfg.aligned_predictions_parquet = (
            "vol_model/output/"
            "BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet"
        )
        cfg.aligner_betas_json = (
            "vol_model/output/aligner_betas_by_month.json"
        )
        cfg.kelly_mult_path = "vol_arb/output/BTC_v2_kelly_mult_K1.parquet"
        cfg.bar_exec_quality_path = "vol_arb/output/bar_execution_quality.parquet"
        cfg.use_tiered_taker = True   # E.1.6: 3-tier ladder at all 4 IOC sites
        return cfg

    def ensure_dirs(self) -> None:
        """Create state and logs directories if they don't exist."""
        for path in (self.state_dir, self.logs_dir):
            Path(path).mkdir(parents=True, exist_ok=True)

    def switch_to_prod(self, *, paper_mode: bool = False) -> None:
        """Flip the venue URLs to production Deribit.

        This should ONLY be called after Phase 7 validation passes and
        the user has explicitly confirmed via CLI passphrase.  The
        default venue is testnet and stays testnet unless this is
        invoked.  ``DERIBIT_PROD_CLIENT_ID`` / ``_SECRET`` must be set
        separately — we refuse to reuse testnet credentials on prod.

        When ``paper_mode=True``, mainnet URLs are still set but
        credentials are NOT required: the engine connects to public
        WS channels only and the dry-run safety belt prevents any
        order submission. If ``DERIBIT_PROD_*`` env vars happen to be
        set anyway they will be loaded but unused (auth never fires).
        """
        self.venue = "prod"
        self.rest_base_url = "https://www.deribit.com/api/v2"
        self.ws_url = "wss://www.deribit.com/ws/api/v2"
        self.paper_mode = paper_mode
        self.client_id = os.environ.get("DERIBIT_PROD_CLIENT_ID", "")
        self.client_secret = os.environ.get("DERIBIT_PROD_CLIENT_SECRET", "")
        if not paper_mode and (not self.client_id or not self.client_secret):
            raise RuntimeError(
                "switch_to_prod() requires DERIBIT_PROD_CLIENT_ID and "
                "DERIBIT_PROD_CLIENT_SECRET in the environment — refusing "
                "to reuse testnet credentials on production."
            )


DEFAULT_CONFIG = PaperConfig()
