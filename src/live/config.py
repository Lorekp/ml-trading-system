"""Configuration for the Bybit demo paper trading system.

Constants match circuit_breaker_v2_experiment.py and dynamic_asset_v2_experiment.py
exactly. Pydantic models for YAML loading with env var injection.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

# ── Constants from circuit_breaker_v2_experiment.py ──

OHLCV_FEATURES = [
    "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
    "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
    "dist_swing_high", "dist_swing_low",
]

NEEDED_TFS = [
    ("15m", "15min"), ("1h", "1h"), ("3h", "3h"), ("4h", "4h"), ("24h", "1D"),
]

FULL5 = [("15m", 2), ("1h", 4), ("3h", 16), ("4h", 6), ("24h", 12)]

BASELINE_WR = 0.65
SPREAD_BPS = 0.5

# Bybit VIP tiers: (name, equity_min, vol30_min, maker_bps, taker_bps)
VIP_TIERS = [
    ("VIP0", 0, 0, 2.00, 5.50),
    ("VIP1", 100_000, 10_000_000, 1.80, 4.00),
    ("VIP2", 250_000, 25_000_000, 1.60, 3.75),
    ("VIP3", 500_000, 50_000_000, 1.40, 3.50),
    ("VIP4", 1_000_000, 100_000_000, 1.20, 3.20),
    ("VIP5", 2_000_000, 250_000_000, 1.00, 3.20),
    ("Supreme", 5_000_000, 500_000_000, 0.00, 3.00),
]

# S8 core assets — anyone would have traded these in mid-2020
OBVIOUS_CORE = {"BTC", "ETH", "XRP", "ADA", "LINK", "LTC", "BNB", "BCH", "ETC"}

# TF label → minutes (for scheduling)
TF_MINUTES = {
    "15m": 15, "1h": 60, "3h": 180, "4h": 240, "24h": 1440,
}


def get_vip_tier(equity: float, vol_30d: float = 0) -> tuple[str, float, float]:
    """Return (tier_name, maker_bps, taker_bps) for given equity and volume."""
    result = VIP_TIERS[0]
    for t in VIP_TIERS:
        if equity >= t[1] or vol_30d >= t[2]:
            result = t
    return result[0], result[3], result[4]


# ── Pydantic config models ──


class ExchangeConfig(BaseModel):
    name: str = "bybit"
    demo: bool = True
    base_url: str = "https://api-demo.bybit.com"
    ws_private_url: str = "wss://stream-demo.bybit.com/v5/private"
    ws_public_url: str = "wss://stream.bybit.com/v5/public/linear"
    api_key: str = ""
    api_secret: str = ""
    private_key_path: str = ""  # RSA PEM for self-generated API keys
    rate_limit_per_sec: int = 10
    timeout_s: float = 10.0
    quote_currency: str = "USDT"


class TelegramConfig(BaseModel):
    bot_token: str = ""
    chat_id: str = ""
    enabled: bool = True
    notify_on: list[str] = Field(
        default_factory=lambda: ["trade", "error", "daily"]
    )


class TimeframeConfig(BaseModel):
    tf: str             # "15m", "1h", "3h", "4h", "24h"
    hold: int           # hold bars: 2, 4, 16, 6, 12
    resample: str       # pandas resample string: "15min", "1h", "3h", "4h", "1D"
    lookback: int = 200 # kline bars to fetch for features


class MkrTkr60Config(BaseModel):
    maker_wait_s: int = 60
    poll_interval_s: float = 5.0
    max_retries: int = 2


class BreakerConfig(BaseModel):
    type: str = "pnl_ratio_symmetric"
    window: int = 10
    threshold: float = -0.01
    floor: float = 0.10
    up_threshold: float = 0.01
    cap: float = 1.5


class PortfolioConfig(BaseModel):
    max_dd: float = 0.10
    delever_start: float = 0.05
    max_exposure: float = 0.90
    max_per_asset: float = 0.06
    max_per_strategy: float = 0.50
    max_positions_per_strategy: int = 20
    base_sizing_pct: float = 0.060
    max_signal_mult: float = 2.0
    sizing_mode: str = "probability_edge"
    regime_exposure_floor: float = 0.50
    # HyroTrader compliance (prop firm rules)
    daily_dd_limit: float = 0.05        # 5% swing daily DD (of initial balance)
    max_margin_pct: float = 0.25        # 25% max margin (of initial balance)
    max_notional_mult: float = 2.0      # 2× max notional (of initial balance)


class S8Config(BaseModel):
    core: list[str] = Field(
        default_factory=lambda: list(OBVIOUS_CORE)
    )
    max_total: int = 30
    btc_vol_window: int = 20
    rebalance_day: int = 1
    prep_cache_path: str = "experiments/.cache/v2_prep_cache.pkl"


class SchedulerConfig(BaseModel):
    bar_interval_m: int = 15
    poll_offset_s: int = 5


class MonitorConfig(BaseModel):
    expected_maker_fill_rate: float = 0.77
    expected_spread_bps: float = 1.22
    expected_avg_cost_bps: float = 3.9
    fill_rate_drift_threshold: float = 0.10
    slippage_alert_bps: float = 5.0
    latency_p95_alert_ms: float = 65_000


class ReconcilerConfig(BaseModel):
    stale_order_age_minutes: int = 5


class DataValidatorConfig(BaseModel):
    max_staleness_factor: float = 3.0
    max_gap_factor: float = 3.0
    min_bars: int = 30


class VolOverlayConfig(BaseModel):
    target_vol: float = 0.025
    floor: float = 0.3
    cap: float = 1.5


class SignalGeneratorConfig(BaseModel):
    conf_threshold: float = 0.08
    regime_ema_span: int = 20
    winsor_lo: float = 0.01
    winsor_hi: float = 0.99
    vol_filter_quantile: float = 0.50
    vol_zscore_window: int = 30
    swing_lookback: int = 20


class ExecutionConfig(BaseModel):
    """Paper-execution slippage model selection (D6 fix).

    type="flat_2bps": legacy PaperExecution with a fixed 2bps one-way slip.
    type="calibrated": CalibratedPaperExecution using the backtest's
        volume-aware slippage model from signal_engine.slippage.compute_slippage.
    """
    type: str = "flat_2bps"
    calibration_path: str = "data/execution_calibration.json"
    vip_tier_name: str = "VIP0"
    vip_maker_bps: float = 2.0
    vip_taker_bps: float = 5.5
    daily_vol_ttl_s: int = 3600


class DataSourceConfig(BaseModel):
    """Data provider config — separate from exchange when using cross-exchange
    execution (e.g. Binance data + Bybit orders)."""
    name: str = ""  # "" = use exchange as data source (backward compat)
    base_url: str = ""
    rate_limit_per_sec: int = 5
    timeout_s: float = 10.0


class LiveConfig(BaseModel):
    """Top-level config for the paper trading engine."""

    exchange: ExchangeConfig = Field(default_factory=ExchangeConfig)
    data: DataSourceConfig = Field(default_factory=DataSourceConfig)
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)
    capital: float = 200_000.0
    model_dir: str = "data/live/models"

    full5: list[TimeframeConfig] = Field(default_factory=lambda: [
        TimeframeConfig(tf="15m", hold=2, resample="15min", lookback=200),
        TimeframeConfig(tf="1h", hold=4, resample="1h", lookback=100),
        TimeframeConfig(tf="3h", hold=16, resample="3h", lookback=80),
        TimeframeConfig(tf="4h", hold=6, resample="4h", lookback=50),
        TimeframeConfig(tf="24h", hold=12, resample="1D", lookback=30),
    ])

    mkrtk60: MkrTkr60Config = Field(default_factory=MkrTkr60Config)
    breaker: BreakerConfig = Field(default_factory=BreakerConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    s8: S8Config = Field(default_factory=S8Config)
    scheduler: SchedulerConfig = Field(default_factory=SchedulerConfig)

    monitor: MonitorConfig = Field(default_factory=MonitorConfig)
    reconciler: ReconcilerConfig = Field(default_factory=ReconcilerConfig)
    data_validator: DataValidatorConfig = Field(default_factory=DataValidatorConfig)
    vol_overlay: VolOverlayConfig = Field(default_factory=VolOverlayConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)

    db_path: str = "data/live/paper_state.db"
    log_level: str = "INFO"
    leverage: int = 1
    max_signals_per_bar: int | None = None
    max_risk_per_trade: float | None = None  # 0.03 = HyroTrader 3% risk rule
    sl_cap: float = 0.75  # max SL distance for dynamic sizing

    # Per-strategy config (loaded dynamically by strategy loader)
    strategies: dict[str, dict] = Field(default_factory=dict)

    # Additional assets beyond S8 core (populated by asset selector)
    assets: dict[str, str] = Field(default_factory=dict)

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, v: str) -> str:
        return v.upper()


def load_live_config(path: Path) -> LiveConfig:
    """Load config YAML with env var injection for secrets."""
    try:
        from dotenv import load_dotenv
        env_path = Path(path).resolve().parent
        for _ in range(5):
            candidate = env_path / ".env"
            if candidate.exists():
                load_dotenv(candidate)
                break
            env_path = env_path.parent
    except ImportError:
        pass

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")

    with open(path) as fh:
        raw: dict[str, Any] = yaml.safe_load(fh) or {}

    # Inject secrets — look for {EXCHANGE_NAME}_API_KEY / _API_SECRET env vars
    exchange_raw = raw.setdefault("exchange", {})
    ex_name = exchange_raw.get("name", "bybit").upper()
    exchange_raw.setdefault("api_key", os.environ.get(f"{ex_name}_API_KEY", ""))
    exchange_raw.setdefault("api_secret", os.environ.get(f"{ex_name}_API_SECRET", ""))

    telegram_raw = raw.setdefault("telegram", {})
    telegram_raw.setdefault("bot_token", os.environ.get("TELEGRAM_BOT_TOKEN", ""))
    telegram_raw.setdefault("chat_id", os.environ.get("TELEGRAM_CHAT_ID", ""))

    config = LiveConfig(**raw)

    # Force demo URL if demo mode
    if config.exchange.demo:
        config.exchange.base_url = "https://api-demo.bybit.com"
        config.exchange.ws_private_url = "wss://stream-demo.bybit.com/v5/private"

    logger.info("Loaded config from %s (capital: $%,.0f, demo: %s)",
                path, config.capital, config.exchange.demo)
    return config
