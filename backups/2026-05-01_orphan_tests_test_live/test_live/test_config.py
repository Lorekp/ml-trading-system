"""Tests for live.config — YAML loading, env injection, validation."""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from live.config import LiveConfig, load_live_config


def _write_yaml(tmp_path: Path, data: dict) -> Path:
    """Helper: write a dict as YAML and return the file path."""
    p = tmp_path / "test_config.yaml"
    with open(p, "w") as f:
        yaml.dump(data, f)
    return p


class TestLoadValidConfig:
    """Test loading a well-formed config file."""

    def test_load_valid_config(self, tmp_path: Path) -> None:
        data = {
            "signal": {
                "model": {"C": 0.1},
                "confidence_filter": {"mode": "absolute", "threshold": 0.04},
            },
            "exchange": {"name": "mexc"},
            "paper": {"starting_capital": 5000.0},
            "assets": {"BTC": "BTCUSDT"},
            "db_path": "data/live/test.db",
        }
        path = _write_yaml(tmp_path, data)
        config = load_live_config(path)

        assert config.exchange.name == "mexc"
        assert config.paper.starting_capital == 5000.0
        assert config.assets["BTC"] == "BTCUSDT"
        assert config.signal.confidence_filter.mode == "absolute"


class TestEnvVarInjection:
    """Test that secrets are injected from environment variables."""

    def test_env_var_injection(self, tmp_path: Path) -> None:
        data = {
            "assets": {"BTC": "BTCUSDT"},
        }
        path = _write_yaml(tmp_path, data)

        env_vars = {
            "MEXC_API_KEY": "my_api_key",
            "MEXC_API_SECRET": "my_api_secret",
            "TELEGRAM_BOT_TOKEN": "bot123:abc",
            "TELEGRAM_CHAT_ID": "12345",
        }
        with patch.dict(os.environ, env_vars, clear=False):
            config = load_live_config(path)

        assert config.exchange.api_key == "my_api_key"
        assert config.exchange.api_secret == "my_api_secret"
        assert config.telegram.bot_token == "bot123:abc"
        assert config.telegram.chat_id == "12345"


class TestInvalidConfig:
    """Test that invalid configs raise appropriate errors."""

    def test_invalid_config_raises(self, tmp_path: Path) -> None:
        data = {
            "log_level": "INVALID_LEVEL",
        }
        path = _write_yaml(tmp_path, data)

        with pytest.raises(Exception):
            load_live_config(path)

    def test_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_live_config(Path("nonexistent_config.yaml"))


class TestDefaultValues:
    """Test that defaults are applied when keys are omitted."""

    def test_default_values(self, tmp_path: Path) -> None:
        data = {"assets": {"BTC": "BTCUSDT"}}
        path = _write_yaml(tmp_path, data)
        config = load_live_config(path)

        assert config.paper.slippage_bps == 1.0
        assert config.paper.starting_capital == 10000.0
        assert config.scheduler.bar_interval_m == 15
        assert config.risk.max_positions == 6
        assert config.risk.hold_bars == 4
        assert config.exchange.rate_limit_per_sec == 20
        assert config.log_level == "INFO"
