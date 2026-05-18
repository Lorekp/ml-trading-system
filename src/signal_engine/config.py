"""Configuration models for the Signal Engine.

Pydantic v2 models matching configs/default.yaml structure.
All parameters are validated at load time — no silent defaults.
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)


class ModelConfig(BaseModel):
    """Logistic regression model hyperparameters."""

    C: float = 0.1
    penalty: str = "l1"
    solver: str = "saga"
    max_iter: int = 1000
    seed: int = 42
    fwd_bars: int = 4
    signal_model_type: str = "regime_xgboost"

    @field_validator("signal_model_type")
    @classmethod
    def _check_signal_model_type(cls, v: str) -> str:
        allowed = {
            "asset_logistic",
            "asset_xgboost",
            "regime_logistic",
            "regime_xgboost",
        }
        if v not in allowed:
            raise ValueError(
                f"signal_model_type must be one of {sorted(allowed)}, got {v}"
            )
        return v


class FeatureConfig(BaseModel):
    """Feature computation parameters."""

    feature_list: list[str] = Field(
        default_factory=lambda: [
            "ret_1", "ret_3", "ret_7", "ret_14", "bar_range", "vol_zscore",
            "rsi_14", "stoch_rsi", "macd_hist", "bb_pct_b", "atr_14",
            "dist_swing_high", "dist_swing_low",
        ]
    )
    winsor_lo: float = 0.01
    winsor_hi: float = 0.99
    vol_filter_quantile: float = 0.50
    vol_zscore_window: int = 30
    swing_lookback: int = 20
    fracdiff_d: float = 0.4
    fracdiff_threshold: float = 1e-3
    feature_research_enabled: bool = False
    sadf_min_window: int = 50
    sadf_step: int = 10
    # Derivatives features (optional — requires HL asset_ctxs data)
    derivatives_data_dir: str | None = None
    derivatives_cache_dir: str | None = None
    derivatives_features: list[str] = Field(default_factory=list)

    @field_validator("winsor_lo", "winsor_hi")
    @classmethod
    def _check_quantile_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Quantile must be in [0, 1], got {v}")
        return v


class PortfolioConfig(BaseModel):
    """Portfolio simulation parameters."""

    starting_capital: float = 10000.0
    max_dd: float = 0.15
    delever_start: float = 0.075
    slippage: float = 0.0001
    max_exposure: float = 0.80
    max_per_asset: float = 0.06
    max_per_strategy: float = 1.0  # 1.0 = no per-strategy limit (backward compatible)
    base_sizing_pct: float = 0.02
    max_signal_mult: float = 2.0
    kelly_sizing: bool = False
    allocation_method: str = "kelly"  # "equal" | "kelly" | "nco"
    nco_clusters: int = 8
    sizing_mode: str = "probability_edge"  # "signal_strength" | "probability_edge"
    use_regime_exposure_multiplier: bool = True
    regime_exposure_floor: float = 0.50
    # Per-asset inverse-vol sizing overlay — risk-parity across the universe.
    # vol_mult = clip(target / asset_daily_vol, floor, cap), fail-safe to floor.
    # Opt-in (False) to keep existing backtests byte-identical; the honest
    # Binance re-backtest enables it. Mirrors the live S8Sym15 overlay.
    per_asset_vol_scaling: bool = False
    per_asset_vol_target: float = 0.025
    per_asset_vol_floor: float = 0.10
    per_asset_vol_cap: float = 2.0

    @field_validator("sizing_mode")
    @classmethod
    def _check_sizing_mode(cls, v: str) -> str:
        allowed = {"signal_strength", "probability_edge"}
        if v not in allowed:
            raise ValueError(f"sizing_mode must be one of {sorted(allowed)}, got {v}")
        return v

    @field_validator("regime_exposure_floor")
    @classmethod
    def _check_regime_exposure_floor(cls, v: float) -> float:
        if not 0.0 < v <= 1.0:
            raise ValueError(f"regime_exposure_floor must be in (0, 1], got {v}")
        return v


class VolOverlayConfig(BaseModel):
    """BTC volatility regime sizing overlay.

    Scales signal_strength by a multiplier that depends on where current
    BTC realized volatility sits in its recent distribution.  Validated
    result: +0.05 Sharpe, +$2/cal with conservative 3-tier.
    """

    enabled: bool = False
    vol_window: int = 20       # days for realized vol
    pct_window: int = 60       # days for percentile ranking
    low_pct: float = 0.20      # below this → low_mult
    high_pct: float = 0.80     # above this → high_mult
    low_mult: float = 0.50     # sizing multiplier in low-vol regime
    mid_mult: float = 1.00     # sizing multiplier in mid-vol regime
    high_mult: float = 1.25    # sizing multiplier in high-vol regime


class RegimeFilterConfig(BaseModel):
    """4H regime filter configuration."""

    enabled: bool = True
    timeframe: str = "4h"
    ema_span: int = 20


class ConfidenceFilterConfig(BaseModel):
    """Confidence filter configuration."""

    mode: str = "percentile"
    threshold: float = 70.0

    @field_validator("mode")
    @classmethod
    def _check_mode(cls, v: str) -> str:
        if v not in ("percentile", "absolute"):
            raise ValueError(f"mode must be 'percentile' or 'absolute', got '{v}'")
        return v


class ExitConfig(BaseModel):
    """Path-dependent exit configuration.

    Supports fixed-horizon, ATR-scaled barriers, percentage-based stops,
    trailing stops, and signal-strength-adaptive hold periods.
    All barrier fields default to 0 (disabled).
    """

    max_hold_bars: int = 4
    # Fixed-percentage stops (for live-executor consistency audit)
    stop_loss_pct: float = 0.0
    take_profit_pct: float = 0.0
    # ATR-scaled barriers (volatility-adaptive)
    stop_loss_atr_mult: float = 0.0
    take_profit_atr_mult: float = 0.0
    # Trailing stop
    trail_atr_mult: float = 0.0
    trail_min_profit_pct: float = 0.005
    # Signal-strength-adaptive hold
    adaptive_hold: bool = False
    adaptive_hold_extra: int = 0  # max extra bars for strongest signal


class CloseModelConfig(BaseModel):
    """ML close-model configuration."""

    enabled: bool = False
    decision_timeframes: list[str] = Field(
        default_factory=lambda: ["5min", "10min", "15min"]
    )
    production_decision_timeframe: str = "15min"
    research_decision_timeframes: list[str] = Field(
        default_factory=lambda: ["5min", "10min", "15min"]
    )
    max_hold_minutes: int = 60
    min_hold_minutes: int = 15
    retrain_frequency_months: int = 1
    lookbacks: list[int] = Field(
        default_factory=lambda: [1, 2, 3, 4, 6, 8, 10, 12]
    )
    model_type: str = "xgboost"
    label_mode: str = "residual_triple_barrier"
    threshold: float = 0.5
    threshold_metric: str = "sharpe"
    threshold_min_precision: float = 0.55
    calibration: str = "isotonic"
    use_afml_optimization: bool = True
    pct_embargo: float = 0.01
    time_decay_last_weight: float = 0.5
    min_signal_strength: float = 0.1
    pt_mult: float = 1.0
    sl_mult: float = 1.0
    min_improvement_bps: float = 2.0
    de_risk_mode: str = "partial_ladder"
    partial_exit_ladder: list[float] = Field(
        default_factory=lambda: [0.50, 0.75, 0.90]
    )
    partial_exit_remaining: list[float] = Field(
        default_factory=lambda: [1.00, 0.50, 0.25, 0.00]
    )
    execution_cost_scenarios: list[float] = Field(
        default_factory=lambda: [1.0, 2.0, 3.0]
    )
    regime_timeframes: list[str] = Field(default_factory=lambda: ["1h", "4h"])
    xgb_max_depth_grid: list[int] = Field(default_factory=lambda: [3, 4, 5])
    xgb_min_child_weight_grid: list[int] = Field(default_factory=lambda: [10, 20, 40])
    xgb_subsample_grid: list[float] = Field(default_factory=lambda: [0.6, 0.8, 1.0])
    xgb_colsample_grid: list[float] = Field(default_factory=lambda: [0.6, 0.8, 1.0])
    xgb_learning_rate_grid: list[float] = Field(default_factory=lambda: [0.03, 0.05, 0.08])
    xgb_max_grid_trials: int = 12
    xgb_early_stopping_rounds: int = 30

    @field_validator("threshold")
    @classmethod
    def _check_threshold(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"threshold must be in [0, 1], got {v}")
        return v

    @field_validator("pct_embargo")
    @classmethod
    def _check_pct_embargo(cls, v: float) -> float:
        if not 0.0 <= v < 1.0:
            raise ValueError(f"pct_embargo must be in [0, 1), got {v}")
        return v

    @field_validator("threshold_min_precision")
    @classmethod
    def _check_threshold_min_precision(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"threshold_min_precision must be in [0, 1], got {v}")
        return v

    @field_validator("model_type")
    @classmethod
    def _check_model_type(cls, v: str) -> str:
        allowed = {"logistic", "xgboost"}
        if v not in allowed:
            raise ValueError(f"model_type must be one of {sorted(allowed)}, got {v}")
        return v

    @field_validator("label_mode")
    @classmethod
    def _check_label_mode(cls, v: str) -> str:
        allowed = {"best_remaining", "residual_triple_barrier"}
        if v not in allowed:
            raise ValueError(f"label_mode must be one of {sorted(allowed)}, got {v}")
        return v

    @field_validator("threshold_metric")
    @classmethod
    def _check_threshold_metric(cls, v: str) -> str:
        allowed = {"sharpe"}
        if v not in allowed:
            raise ValueError(
                f"threshold_metric must be one of {sorted(allowed)}, got {v}"
            )
        return v

    @field_validator("calibration")
    @classmethod
    def _check_calibration(cls, v: str) -> str:
        allowed = {"none", "isotonic"}
        if v not in allowed:
            raise ValueError(f"calibration must be one of {sorted(allowed)}, got {v}")
        return v

    @field_validator("de_risk_mode")
    @classmethod
    def _check_de_risk_mode(cls, v: str) -> str:
        allowed = {"binary_exit", "partial_ladder"}
        if v not in allowed:
            raise ValueError(f"de_risk_mode must be one of {sorted(allowed)}, got {v}")
        return v



class SignalConfig(BaseModel):
    """Signal discretization parameters."""

    step_size: float = 0.1


class DataConfig(BaseModel):
    """Data loading configuration."""

    data_dirs: list[str] = Field(
        default_factory=lambda: [
            "data/binance-1m",
            "data/homeserver-mirror/binance-1m",
        ]
    )
    timeframe: str = "15min"
    regime_timeframe: str = "4h"


class SystemConfig(BaseModel):
    """Top-level system configuration combining all sub-configs."""

    model: ModelConfig = Field(default_factory=ModelConfig)
    features: FeatureConfig = Field(default_factory=FeatureConfig)
    portfolio: PortfolioConfig = Field(default_factory=PortfolioConfig)
    regime_filter: RegimeFilterConfig = Field(default_factory=RegimeFilterConfig)
    vol_overlay: VolOverlayConfig = Field(default_factory=VolOverlayConfig)
    confidence_filter: ConfidenceFilterConfig = Field(default_factory=ConfidenceFilterConfig)
    close_model: CloseModelConfig = Field(default_factory=CloseModelConfig)
    signal: SignalConfig = Field(default_factory=SignalConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    assets: dict[str, str] = Field(default_factory=dict)


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge *override* into *base*, returning a new dict."""
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def load_config(path: Path) -> SystemConfig:
    """Load a YAML configuration file and return a validated SystemConfig.

    Args:
        path: Path to the YAML config file.

    Returns:
        Validated SystemConfig instance.

    Raises:
        FileNotFoundError: If path does not exist.
        pydantic.ValidationError: If the config is invalid.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path, "r") as fh:
        raw = yaml.safe_load(fh) or {}

    logger.info("Loaded config from %s", path)
    return SystemConfig(**raw)


def load_config_with_overrides(
    base_path: Path,
    override_path: Path | None = None,
) -> SystemConfig:
    """Load base config and optionally deep-merge an override config on top.

    Args:
        base_path: Path to the base YAML config.
        override_path: Optional path to an override YAML config. Keys in the
            override replace the corresponding keys in the base; nested dicts
            are merged recursively.

    Returns:
        Validated SystemConfig after merging.
    """
    base_path = Path(base_path)
    if not base_path.exists():
        raise FileNotFoundError(f"Base config not found: {base_path}")

    with open(base_path, "r") as fh:
        base_raw = yaml.safe_load(fh) or {}

    if override_path is not None:
        override_path = Path(override_path)
        if not override_path.exists():
            raise FileNotFoundError(f"Override config not found: {override_path}")
        with open(override_path, "r") as fh:
            override_raw = yaml.safe_load(fh) or {}
        merged = _deep_merge(base_raw, override_raw)
    else:
        merged = base_raw

    logger.info(
        "Loaded config from %s (override=%s)",
        base_path,
        override_path or "none",
    )
    return SystemConfig(**merged)
