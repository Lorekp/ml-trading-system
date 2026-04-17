"""Tests for signal_engine.config."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from signal_engine.config import (
    CloseModelConfig,
    ConfidenceFilterConfig,
    FeatureConfig,
    SystemConfig,
    load_config,
    load_config_with_overrides,
)


def test_load_default_config(tmp_path: Path) -> None:
    """Loading a valid YAML produces a SystemConfig with correct values."""
    yaml_content = textwrap.dedent("""\
        model:
          C: 0.1
          penalty: "l1"
          solver: "saga"
          max_iter: 1000
          seed: 42
          fwd_bars: 4
        features:
          winsor_lo: 0.01
          winsor_hi: 0.99
        portfolio:
          starting_capital: 10000.0
          max_dd: 0.15
        assets:
          BTC: "BTCUSDT"
    """)
    cfg_path = tmp_path / "test.yaml"
    cfg_path.write_text(yaml_content)

    cfg = load_config(cfg_path)

    assert cfg.model.C == 0.1
    assert cfg.model.penalty == "l1"
    assert cfg.features.winsor_lo == 0.01
    assert cfg.portfolio.starting_capital == 10000.0
    assert cfg.assets["BTC"] == "BTCUSDT"


def test_invalid_config_raises(tmp_path: Path) -> None:
    """Invalid quantile value raises ValidationError."""
    yaml_content = textwrap.dedent("""\
        features:
          winsor_lo: -0.5
    """)
    cfg_path = tmp_path / "bad.yaml"
    cfg_path.write_text(yaml_content)

    with pytest.raises(ValidationError):
        load_config(cfg_path)


def test_invalid_confidence_mode() -> None:
    """Unknown filter mode raises ValidationError."""
    with pytest.raises(ValidationError):
        ConfidenceFilterConfig(mode="unknown", threshold=70.0)


def test_override_merging(tmp_path: Path) -> None:
    """Override config merges correctly, replacing only specified keys."""
    base_yaml = textwrap.dedent("""\
        portfolio:
          max_exposure: 0.80
          max_per_asset: 0.06
          starting_capital: 10000.0
        assets:
          BTC: "BTCUSDT"
          ETH: "ETHUSDT"
    """)
    override_yaml = textwrap.dedent("""\
        portfolio:
          max_exposure: 0.60
          max_per_asset: 0.20
        assets:
          BTC: "BTCUSDT"
    """)
    base_path = tmp_path / "base.yaml"
    base_path.write_text(base_yaml)
    override_path = tmp_path / "override.yaml"
    override_path.write_text(override_yaml)

    cfg = load_config_with_overrides(base_path, override_path)

    # Overridden values
    assert cfg.portfolio.max_exposure == 0.60
    assert cfg.portfolio.max_per_asset == 0.20
    # Preserved from base (nested merge keeps non-overridden keys)
    assert cfg.portfolio.starting_capital == 10000.0
    # Assets are a flat dict — deep merge merges both sets of keys
    assert cfg.assets["BTC"] == "BTCUSDT"
    assert cfg.assets["ETH"] == "ETHUSDT"
    assert len(cfg.assets) == 2


def test_missing_config_file() -> None:
    """FileNotFoundError on missing config."""
    with pytest.raises(FileNotFoundError):
        load_config(Path("/nonexistent/config.yaml"))


def test_default_feature_list() -> None:
    """Default FeatureConfig has exactly 13 features."""
    fc = FeatureConfig()
    assert len(fc.feature_list) == 13


def test_systemconfig_defaults() -> None:
    """SystemConfig with no arguments uses all defaults."""
    cfg = SystemConfig()
    assert cfg.model.C == 0.1
    assert cfg.confidence_filter.mode == "percentile"
    assert cfg.close_model.enabled is False
    assert cfg.close_model.max_hold_minutes == 60
    assert cfg.close_model.min_hold_minutes == 15
    assert cfg.close_model.lookbacks == [1, 2, 3, 4, 6, 8, 10, 12]
    assert cfg.close_model.model_type == "xgboost"
    assert cfg.close_model.label_mode == "residual_triple_barrier"
    assert cfg.close_model.production_decision_timeframe == "15min"
    assert cfg.close_model.research_decision_timeframes == ["5min", "10min", "15min"]
    assert cfg.close_model.use_afml_optimization is True
    assert cfg.close_model.pct_embargo == 0.01
    assert cfg.close_model.time_decay_last_weight == 0.5
    assert cfg.close_model.min_signal_strength == 0.1
    assert cfg.close_model.threshold_metric == "sharpe"
    assert cfg.close_model.threshold_min_precision == 0.55
    assert cfg.close_model.calibration == "isotonic"
    assert cfg.close_model.pt_mult == 1.0
    assert cfg.close_model.sl_mult == 1.0
    assert cfg.close_model.min_improvement_bps == 2.0
    assert cfg.signal.step_size == 0.1


def test_close_model_embargo_validation() -> None:
    with pytest.raises(ValidationError):
        CloseModelConfig(pct_embargo=1.0)


def test_close_model_mode_validation() -> None:
    with pytest.raises(ValidationError):
        CloseModelConfig(model_type="svm")
    with pytest.raises(ValidationError):
        CloseModelConfig(label_mode="foo")
