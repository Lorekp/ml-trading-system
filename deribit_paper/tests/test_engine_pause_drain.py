"""Tests for pause and drain modes in TradingEngine."""

from __future__ import annotations

import asyncio
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

import pandas as pd
import pytest

from deribit_paper.config import PaperConfig
from deribit_paper.engine import TradingEngine
from deribit_paper.market_data import Bar


def _make_engine(tmp_path: Path) -> TradingEngine:
    """Create a minimally-configured engine for unit testing."""
    cfg = PaperConfig()
    cfg.state_dir = str(tmp_path / "state")
    Path(cfg.state_dir).mkdir(parents=True, exist_ok=True)
    engine = TradingEngine(cfg)
    # Stub out services that process_bar needs
    engine.kill_switch = MagicMock()
    engine.kill_switch.check_before_signal.return_value = False
    engine.kill_switch.is_halted.return_value = False
    engine.kill_switch.needs_emergency_close = False
    engine.position_tracker = MagicMock()
    engine.position_tracker.n_open = 0
    engine.equity_tracker = MagicMock()
    engine.equity_tracker.current_equity = 1000.0
    engine.equity_tracker.drawdown_pct = 0.0
    engine.equity_tracker.peak_equity = 1000.0
    engine.equity_tracker.total_return_pct = 0.0
    engine.telegram = MagicMock()
    engine.telegram.enabled = False
    engine._background_tasks = []
    return engine


def _make_bar() -> Bar:
    """Create a minimal bar for testing."""
    return Bar(
        timestamp=pd.Timestamp("2026-04-10 12:00:00", tz="UTC"),
        index_price=100_000.0,
        n_perp_trades=100,
        n_option_trades=10,
        perp_trades=[],
        option_trades=[],
    )


# ── Pause mode ────────────────────────────────────────────────────


def test_pause_sets_flag():
    cfg = PaperConfig()
    engine = TradingEngine(cfg)
    engine.telegram = MagicMock()
    engine.telegram.enabled = False
    assert engine.is_paused is False
    engine.pause()
    assert engine.is_paused is True


def test_resume_clears_flag():
    cfg = PaperConfig()
    engine = TradingEngine(cfg)
    engine.telegram = MagicMock()
    engine.telegram.enabled = False
    engine.pause()
    engine.resume()
    assert engine.is_paused is False


def test_pause_is_idempotent():
    cfg = PaperConfig()
    engine = TradingEngine(cfg)
    engine.telegram = MagicMock()
    engine.telegram.enabled = True
    engine.pause()
    engine.pause()  # second call should be no-op
    assert engine.is_paused is True
    # Only one telegram message sent
    assert engine.telegram.send.call_count == 1


# ── Drain mode ────────────────────────────────────────────────────


def test_drain_sets_flag():
    cfg = PaperConfig()
    engine = TradingEngine(cfg)
    engine.telegram = MagicMock()
    engine.telegram.enabled = False
    assert engine.is_draining is False
    engine.drain()
    assert engine.is_draining is True


def test_drain_is_idempotent():
    cfg = PaperConfig()
    engine = TradingEngine(cfg)
    engine.telegram = MagicMock()
    engine.telegram.enabled = True
    engine.drain()
    engine.drain()
    assert engine.is_draining is True
    assert engine.telegram.send.call_count == 1


# ── Sentinel files ────────────────────────────────────────────────


def test_sentinel_pause_file_activates_pause(tmp_path):
    engine = _make_engine(tmp_path)
    pause_file = Path(engine.config.state_dir) / "pause"
    pause_file.touch()
    engine._check_sentinel_files()
    assert engine.is_paused is True


def test_sentinel_pause_file_removal_resumes(tmp_path):
    engine = _make_engine(tmp_path)
    pause_file = Path(engine.config.state_dir) / "pause"
    pause_file.touch()
    engine._check_sentinel_files()
    assert engine.is_paused is True
    pause_file.unlink()
    engine._check_sentinel_files()
    assert engine.is_paused is False


def test_sentinel_drain_file_activates_drain(tmp_path):
    engine = _make_engine(tmp_path)
    drain_file = Path(engine.config.state_dir) / "drain"
    drain_file.touch()
    engine._check_sentinel_files()
    assert engine.is_draining is True
    # Drain file should be consumed (deleted)
    assert not drain_file.exists()


def test_no_sentinel_files_no_change(tmp_path):
    engine = _make_engine(tmp_path)
    engine._check_sentinel_files()
    assert engine.is_paused is False
    assert engine.is_draining is False
