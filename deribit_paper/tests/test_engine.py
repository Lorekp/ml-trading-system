"""Tests for TradingEngine wiring + process_bar.

The engine is constructed with the real production config and all
services mocked at the boundary (REST client, WS client, model
bundle, order manager).  We drive synthetic :class:`Bar` objects
through :meth:`TradingEngine.process_bar` and verify the full
pipeline runs without errors.

A true end-to-end test against Deribit testnet lives in the
Phase 8 smoke runner — not here.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from deribit_paper.config import PaperConfig
from deribit_paper.engine import EngineStats, TradingEngine
from deribit_paper.features.straddle import StraddleSelection
from deribit_paper.market_data.aggregator import Bar
from deribit_paper.market_data.buffer import Trade
from deribit_paper.signal.model_loader import ModelBundle


def _make_synthetic_bar() -> Bar:
    ts = pd.Timestamp("2026-04-10 12:15:00", tz="UTC")
    # 50 perp trades in the bar
    perp_trades = [
        Trade(
            trade_id=f"p{i}",
            timestamp=ts - pd.Timedelta(seconds=900 - i * 18),
            instrument_name="BTC-PERPETUAL",
            price=50_000.0 + i * 5,
            amount=0.5,
            direction=1 if i % 2 == 0 else -1,
            index_price=50_000.0 + i * 5,
            mark_price=50_000.0 + i * 5,
        )
        for i in range(50)
    ]
    return Bar(
        timestamp=ts,
        bar_minutes=15,
        perp_trades=perp_trades,
        option_trades=[],
        index_price=50_250.0,
    )


@pytest.fixture
def mock_config(tmp_path):
    """PaperConfig pointed at temp dirs, with dummy credentials."""
    cfg = PaperConfig()
    cfg.client_id = "test_id"
    cfg.client_secret = "test_secret"
    cfg.state_dir = str(tmp_path / "state")
    cfg.logs_dir = str(tmp_path / "logs")
    cfg.sqlite_db = str(tmp_path / "paper.db")
    cfg.metrics_parquet = str(tmp_path / "metrics.parquet")
    cfg.telegram_bot_token = ""
    cfg.telegram_chat_id = ""
    cfg.ensure_dirs()
    return cfg


def _stub_bundle() -> ModelBundle:
    """Minimal ModelBundle with stubbed LightGBM + DRL."""
    lgbm = MagicMock()
    lgbm.predict.return_value = np.array([-0.22])  # ln(vol_ann) ~0.8
    drl = MagicMock()
    drl.predict.return_value = (np.array([[0.0]], dtype=np.float32), None)
    return ModelBundle(
        month_key="stub",
        lgbm_model=lgbm,
        lgbm_features=[f"tv_{i}" for i in range(16)] + [
            "n_bars_in_window", "duration_s", "n_trades", "vpin",
            "tsrv_mean_1d", "tsrv_mean_1w", "noise_level", "parkinson_1d",
            "ffd_rv", "tsrv_ffd", "bpv_tsrv_ratio", "imbalance",
            "vov", "up_jump_frac", "time_to_funding",
            # padding to reach 51
            *[f"extra_{i}" for i in range(20)],
        ],
        transvol_model=None,
        transvol_config=None,
        transvol_feat_mean=None,
        transvol_feat_std=None,
        transvol_time_mean=None,
        transvol_time_std=None,
        drl_policy=drl,
        drl_vec_normalize_mean=None,
        drl_vec_normalize_var=None,
        drl_vec_normalize_clip=10.0,
    )


# ─── EngineStats dataclass ───────────────────────────────────────────

def test_engine_stats_defaults():
    stats = EngineStats()
    assert stats.bars_processed == 0
    assert stats.signals_generated == 0
    assert stats.trades_completed == 0
    assert stats.trades_aborted == 0


# ─── Construction (no network) ──────────────────────────────────────

def test_engine_construct_does_not_network(mock_config):
    """Constructing the engine must not touch the network."""
    engine = TradingEngine(mock_config)
    assert engine.config is mock_config
    assert engine.rest is None
    assert engine.ws is None
    assert engine.stats.bars_processed == 0


@pytest.mark.asyncio
async def test_engine_run_requires_initialize(mock_config):
    """Calling run() before initialize() should fail fast."""
    engine = TradingEngine(mock_config)
    with pytest.raises(RuntimeError, match="initialize"):
        await engine.run()


# ─── process_bar with fully-mocked services ─────────────────────────

@pytest.mark.asyncio
async def test_process_bar_flat_signal_no_execution(mock_config):
    """A flat signal should increment stats but never call execute."""
    engine = TradingEngine(mock_config)
    # Manually wire up just what process_bar needs
    from deribit_paper.features.svi_live import LiveSVICalibrator
    from deribit_paper.market_data import IndexPriceBuffer, TradeBuffer
    from deribit_paper.monitoring import MetricsSink, TelegramNotifier
    from deribit_paper.position import EquityTracker, PositionTracker, SQLiteStore
    from deribit_paper.risk.kill_switch import KillSwitch
    from deribit_paper.risk.limits import RiskLimits

    engine.perp_buffer = TradeBuffer(name="perp", window_seconds=3600)
    engine.option_buffer = TradeBuffer(name="opt", window_seconds=3600)
    engine.index_buffer = IndexPriceBuffer()
    engine.svi_calibrator = LiveSVICalibrator(
        cache_path=mock_config.state_dir + "/svi.parquet",
    )
    engine.equity_tracker = EquityTracker(initial_equity_usd=100_000.0)
    engine.position_tracker = PositionTracker()
    engine.telegram = TelegramNotifier(bot_token="", chat_id="")
    engine.metrics_sink = MetricsSink(
        path=mock_config.state_dir + "/metrics.parquet", flush_every=999,
    )
    engine.store = None  # not used when no trade executes

    engine.kill_switch = KillSwitch(
        equity_tracker=engine.equity_tracker,
        max_daily_drawdown_pct=5.0,
    )
    engine.risk_limits = RiskLimits(
        position_tracker=engine.position_tracker,
        equity_tracker=engine.equity_tracker,
    )

    # Stubs for services the predictor would call
    engine.predictor = MagicMock()
    flat_signal = MagicMock()
    flat_signal.direction = 0
    flat_signal.reason = "FLAT — edge below threshold"
    flat_signal.rv_pred_pct = 72.0
    flat_signal.iv_used_pct = 72.3
    flat_signal.edge_vol_pts = -0.3
    flat_signal.is_actionable = False
    flat_signal.straddle = None
    engine.predictor.predict.return_value = flat_signal

    engine.strike_picker = AsyncMock()
    engine.strike_picker.pick_atm_straddle = AsyncMock(return_value=None)

    engine.straddle_flow = MagicMock()
    engine.straddle_flow.size_position = MagicMock(return_value=0.0)
    engine.straddle_flow.execute = AsyncMock()

    bar = _make_synthetic_bar()
    await engine.process_bar(bar)

    # Flat signal → no execution
    engine.straddle_flow.execute.assert_not_called()
    assert engine.stats.signals_generated == 1
    assert engine.stats.signals_flat == 1
    assert engine.stats.signals_executed == 0


@pytest.mark.asyncio
async def test_process_bar_kill_switch_short_circuits(mock_config):
    """If the kill switch is halted, process_bar must skip without calling predictor."""
    engine = TradingEngine(mock_config)
    from deribit_paper.features.svi_live import LiveSVICalibrator
    from deribit_paper.market_data import IndexPriceBuffer, TradeBuffer
    from deribit_paper.monitoring import MetricsSink, TelegramNotifier
    from deribit_paper.position import EquityTracker, PositionTracker
    from deribit_paper.risk.kill_switch import KillSwitch

    engine.perp_buffer = TradeBuffer(name="perp", window_seconds=3600)
    engine.option_buffer = TradeBuffer(name="opt", window_seconds=3600)
    engine.index_buffer = IndexPriceBuffer()
    engine.svi_calibrator = LiveSVICalibrator(cache_path=mock_config.state_dir + "/svi.parquet")
    engine.equity_tracker = EquityTracker(initial_equity_usd=100_000.0)
    engine.position_tracker = PositionTracker()
    engine.telegram = TelegramNotifier(bot_token="", chat_id="")
    engine.metrics_sink = MetricsSink(
        path=mock_config.state_dir + "/metrics.parquet", flush_every=999,
    )

    engine.kill_switch = KillSwitch(equity_tracker=engine.equity_tracker)
    engine.kill_switch.trip("test halt")
    assert engine.kill_switch.is_halted()

    engine.predictor = MagicMock()
    engine.strike_picker = AsyncMock()
    engine.straddle_flow = MagicMock()

    bar = _make_synthetic_bar()
    await engine.process_bar(bar)

    # No predictor call, no execution
    engine.predictor.predict.assert_not_called()
    engine.straddle_flow.execute.assert_not_called()


@pytest.mark.asyncio
async def test_process_bar_rnd_halt_short_circuits(mock_config):
    """If the RND monitor is halted, process_bar must skip without calling predictor."""
    engine = TradingEngine(mock_config)
    from deribit_paper.features.svi_live import LiveSVICalibrator
    from deribit_paper.market_data import IndexPriceBuffer, TradeBuffer
    from deribit_paper.monitoring import MetricsSink, TelegramNotifier
    from deribit_paper.position import EquityTracker, PositionTracker
    from deribit_paper.risk.kill_switch import KillSwitch
    from deribit_paper.risk.rnd_monitor import RNDConfig, RNDMonitor, RNDSnapshot

    engine.perp_buffer = TradeBuffer(name="perp", window_seconds=3600)
    engine.option_buffer = TradeBuffer(name="opt", window_seconds=3600)
    engine.index_buffer = IndexPriceBuffer()
    engine.svi_calibrator = LiveSVICalibrator(cache_path=mock_config.state_dir + "/svi.parquet")
    engine.equity_tracker = EquityTracker(initial_equity_usd=100_000.0)
    engine.position_tracker = PositionTracker()
    engine.telegram = TelegramNotifier(bot_token="", chat_id="")
    engine.metrics_sink = MetricsSink(
        path=mock_config.state_dir + "/metrics.parquet", flush_every=999,
    )
    engine.kill_switch = KillSwitch(equity_tracker=engine.equity_tracker)
    engine.predictor = MagicMock()
    engine.strike_picker = AsyncMock()
    engine.straddle_flow = MagicMock()

    # Wire a halted RND monitor by forcing its internal halt reason directly
    # (bypasses the need for a real SVI surface + tail calculation).
    engine.rnd_monitor = RNDMonitor(config=RNDConfig(baseline_warmup_bars=1))
    engine.rnd_monitor._halt_reason = "forced halt for test"

    bar = _make_synthetic_bar()
    await engine.process_bar(bar)

    # RND halt short-circuits BEFORE the predictor is called
    engine.predictor.predict.assert_not_called()
    engine.straddle_flow.execute.assert_not_called()


# ─── Trailing volume helper ─────────────────────────────────────────


def test_trailing_1h_volume_with_trades(mock_config):
    """_trailing_1h_volume_btc returns the trailing 1h trade volume
    converted from Deribit perp USD-notional to BTC.

    Cycle-24-C1 (2026-05-11, audit A3 C1): Deribit BTC-PERPETUAL
    ``amount`` is reported in USD-equivalent (contracts × $10), so
    the engine divides the sum by current spot to land in BTC. The
    test fixture now feeds USD-notional values to match production
    semantics (pre-fix the test fed BTC values and got summed BTC
    back — that interpretation didn't match Deribit's actual API).
    """
    from deribit_paper.market_data.buffer import Trade, TradeBuffer

    engine = TradingEngine(mock_config)
    engine.perp_buffer = TradeBuffer(name="perp", window_seconds=86400.0)
    # Stub the index_buffer to return the canonical spot used below.
    class _StubIdx:
        def latest(self): return 60_000.0
    engine.index_buffer = _StubIdx()

    now = pd.Timestamp.now(tz="UTC")
    # 10 trades of $30K USD-notional each (= 0.5 BTC at $60K spot)
    for i in range(10):
        engine.perp_buffer.append(Trade(
            trade_id=f"t{i}",
            timestamp=now - pd.Timedelta(minutes=30 - i),
            instrument_name="BTC-PERPETUAL",
            price=60_000.0,
            amount=30_000.0,  # USD-notional per Deribit convention
            direction=1 if i % 2 == 0 else -1,
            index_price=60_000.0,
            mark_price=60_000.0,
        ))

    vol = engine._trailing_1h_volume_btc()
    assert vol is not None
    # 10 trades × $30K = $300K USD → $300K / $60K spot = 5 BTC
    assert vol == pytest.approx(5.0)


def test_trailing_1h_volume_empty(mock_config):
    """_trailing_1h_volume_btc returns None when buffer is empty."""
    engine = TradingEngine(mock_config)
    engine.perp_buffer = None
    assert engine._trailing_1h_volume_btc() is None


# ─── Heartbeat scheduler ────────────────────────────────────────────


def _make_heartbeat_engine(mock_config, vpin_stream=None):
    """Helper: set up an engine with minimal services for heartbeat testing."""
    from deribit_paper.monitoring.telegram import TelegramNotifier
    from deribit_paper.position.equity import EquityTracker
    from deribit_paper.position.tracker import PositionTracker

    engine = TradingEngine(mock_config)
    engine.stats = EngineStats()
    engine.telegram = MagicMock(spec=TelegramNotifier)
    engine.telegram.enabled = True
    engine.equity_tracker = EquityTracker(initial_equity_usd=100_000.0)
    engine.position_tracker = PositionTracker()
    engine.vpin_stream = vpin_stream
    engine._stop = asyncio.Event()
    engine.config.heartbeat_minutes = 1 / 120  # 0.5 seconds
    return engine


async def _wait_for_heartbeat(engine, timeout=5.0):
    """Poll until heartbeat has been called, or timeout."""
    import time
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if engine.telegram.heartbeat.called:
            return
        await asyncio.sleep(0.1)
    raise AssertionError("Heartbeat was not called within timeout")


@pytest.mark.asyncio
async def test_heartbeat_loop_fires(mock_config):
    """Heartbeat loop should fire at least once within the interval."""
    engine = _make_heartbeat_engine(mock_config)
    task = asyncio.create_task(engine._heartbeat_loop())

    await _wait_for_heartbeat(engine)
    engine._stop.set()
    await asyncio.wait_for(task, timeout=2.0)

    call_kwargs = engine.telegram.heartbeat.call_args.kwargs
    assert "n_open" in call_kwargs
    assert "equity_usd" in call_kwargs
    assert "drawdown_pct" in call_kwargs
    assert "bars_processed" in call_kwargs


@pytest.mark.asyncio
async def test_heartbeat_includes_vpin(mock_config):
    """When vpin_stream is present, heartbeat includes VPIN fields."""
    vpin = MagicMock()
    vpin.latest_vpin = 0.195
    vpin.is_connected = True
    engine = _make_heartbeat_engine(mock_config, vpin_stream=vpin)

    task = asyncio.create_task(engine._heartbeat_loop())
    await _wait_for_heartbeat(engine)
    engine._stop.set()
    await asyncio.wait_for(task, timeout=2.0)

    call_kwargs = engine.telegram.heartbeat.call_args.kwargs
    assert call_kwargs["vpin"] == pytest.approx(0.195)
    assert call_kwargs["vpin_connected"] is True


@pytest.mark.asyncio
async def test_heartbeat_stops_on_shutdown(mock_config):
    """Heartbeat loop exits cleanly when _stop is set."""
    engine = TradingEngine(mock_config)
    engine.stats = EngineStats()

    from deribit_paper.monitoring.telegram import TelegramNotifier
    engine.telegram = MagicMock(spec=TelegramNotifier)
    engine.telegram.enabled = True

    from deribit_paper.position.equity import EquityTracker
    engine.equity_tracker = EquityTracker(initial_equity_usd=100_000.0)

    from deribit_paper.position.tracker import PositionTracker
    engine.position_tracker = PositionTracker()

    engine.vpin_stream = None
    engine._stop = asyncio.Event()
    engine.config.heartbeat_minutes = 60  # long interval

    task = asyncio.create_task(engine._heartbeat_loop())
    # Signal stop immediately — loop should exit before interval elapses
    engine._stop.set()
    await asyncio.wait_for(task, timeout=2.0)
    assert task.done()


# ─── Stale-data guard ──────────────────────────────────────────────


def _make_guard_engine(mock_config):
    """Wire up a minimal engine for stale-data guard testing."""
    from deribit_paper.features.svi_live import LiveSVICalibrator
    from deribit_paper.market_data import IndexPriceBuffer, TradeBuffer
    from deribit_paper.monitoring import MetricsSink, TelegramNotifier
    from deribit_paper.position import EquityTracker, PositionTracker
    from deribit_paper.risk.kill_switch import KillSwitch
    from deribit_paper.risk.limits import RiskLimits

    engine = TradingEngine(mock_config)
    engine.perp_buffer = TradeBuffer(name="perp", window_seconds=3600)
    engine.option_buffer = TradeBuffer(name="opt", window_seconds=3600)
    engine.index_buffer = IndexPriceBuffer()
    engine.svi_calibrator = LiveSVICalibrator(
        cache_path=mock_config.state_dir + "/svi.parquet",
    )
    engine.equity_tracker = EquityTracker(initial_equity_usd=100_000.0)
    engine.position_tracker = PositionTracker()
    engine.telegram = TelegramNotifier(bot_token="", chat_id="")
    engine.metrics_sink = MetricsSink(
        path=mock_config.state_dir + "/metrics.parquet", flush_every=999,
    )
    engine.kill_switch = KillSwitch(equity_tracker=engine.equity_tracker)
    engine.risk_limits = RiskLimits(
        position_tracker=engine.position_tracker,
        equity_tracker=engine.equity_tracker,
    )
    engine.predictor = MagicMock()
    engine.strike_picker = AsyncMock()
    engine.straddle_flow = MagicMock()
    engine.rnd_monitor = None
    return engine


@pytest.mark.asyncio
async def test_process_bar_empty_bar_skips_signal(mock_config):
    """A bar with zero trades should skip signal generation entirely."""
    engine = _make_guard_engine(mock_config)

    empty_bar = Bar(
        timestamp=pd.Timestamp("2026-04-10 12:15:00", tz="UTC"),
        bar_minutes=15,
        perp_trades=[],
        option_trades=[],
        index_price=50000.0,
    )
    await engine.process_bar(empty_bar)

    engine.predictor.predict.assert_not_called()
    assert engine._consecutive_empty_bars == 1
    assert not engine.kill_switch.is_halted()


@pytest.mark.asyncio
async def test_consecutive_empty_bars_trips_kill_switch(mock_config):
    """N consecutive empty bars should trip the kill switch."""
    mock_config.max_consecutive_empty_bars = 2
    engine = _make_guard_engine(mock_config)

    ts_base = pd.Timestamp("2026-04-10 12:00:00", tz="UTC")
    for i in range(1, 3):
        empty_bar = Bar(
            timestamp=ts_base + pd.Timedelta(minutes=15 * i),
            bar_minutes=15,
            perp_trades=[],
            option_trades=[],
            index_price=50000.0,
        )
        await engine.process_bar(empty_bar)

    assert engine._consecutive_empty_bars == 2
    assert engine.kill_switch.is_halted()
    assert "data stale" in (engine.kill_switch.state.reason or "")


@pytest.mark.asyncio
async def test_nonempty_bar_resets_empty_counter(mock_config):
    """A bar with trades must reset the consecutive empty counter."""
    mock_config.max_consecutive_empty_bars = 3
    engine = _make_guard_engine(mock_config)

    # Make predictor return flat to short-circuit before sizing
    flat_signal = MagicMock()
    flat_signal.direction = 0
    flat_signal.reason = "FLAT"
    flat_signal.rv_pred_pct = 72.0
    flat_signal.iv_used_pct = 72.3
    flat_signal.edge_vol_pts = -0.3
    flat_signal.is_actionable = False
    flat_signal.straddle = None
    engine.predictor.predict.return_value = flat_signal

    ts = pd.Timestamp("2026-04-10 12:15:00", tz="UTC")

    # One empty bar
    await engine.process_bar(Bar(
        timestamp=ts, bar_minutes=15,
        perp_trades=[], option_trades=[], index_price=50000.0,
    ))
    assert engine._consecutive_empty_bars == 1

    # Then a bar with real data — counter must reset
    await engine.process_bar(_make_synthetic_bar())
    assert engine._consecutive_empty_bars == 0
    assert not engine.kill_switch.is_halted()
