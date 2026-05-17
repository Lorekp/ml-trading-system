"""Tests for ``scripts/replay_parity_harness.py`` — cycle-38 (2026-05-17).

The replay-parity harness drives the live engine's ``process_bar`` with
canonical offline data so the per-bar feature trace can be diffed
against the offline trace. These three tests cover:

  1. ``test_reconstruct_bar_two_bars`` — synthetic 2-bar reconstruction:
     ``reconstruct_bar`` partitions trades into the correct 15-min
     half-open windows and attaches the canonical close-of-bar index.

  2. ``test_process_bar_drive_produces_trace_row`` — driving a
     reconstructed :class:`Bar` through ``TradingEngine.process_bar``
     with a real :class:`BarTraceWriter` enabled emits exactly one
     JSONL trace row.

  3. ``test_window_filtering_correctness`` — the half-open window
     ``[T-15min, T)`` slicing helpers exclude the boundary at ``T`` and
     include the boundary at ``T-15min``, matching the backtester's
     bar convention.

No network: the engine in test 2 is wired manually (same pattern as
``test_engine.py``) so ``initialize()`` — which opens a REST session —
is never called.
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from scripts.replay_parity_harness import (
    ReplaySources,
    _bar_grid,
    _option_trades_for_window,
    _perp_trades_for_window,
    _slice_window,
    _tsrv_trades_for_window,
    reconstruct_bar,
)


# ── Fixtures ────────────────────────────────────────────────────────


def _future_df_two_bars() -> pd.DataFrame:
    """Deribit BTC-PERPETUAL trades spanning two 15-min bars.

    Bar A window [12:00, 12:15): trades at 12:01, 12:07.
    Bar B window [12:15, 12:30): trades at 12:16, 12:29.
    Plus a boundary trade exactly at 12:15 (belongs to bar B's window
    start, not bar A's exclusive end).
    """
    rows = [
        # bar A
        {"trade_id": "f1", "timestamp": "2026-05-14 12:01:00",
         "instrument_name": "BTC-PERPETUAL", "price": 60_000.0,
         "amount": 1000.0, "direction": "buy", "index_price": 60_000.0,
         "mark_price": 60_000.0},
        {"trade_id": "f2", "timestamp": "2026-05-14 12:07:00",
         "instrument_name": "BTC-PERPETUAL", "price": 60_050.0,
         "amount": 2000.0, "direction": "sell", "index_price": 60_050.0,
         "mark_price": 60_050.0},
        # boundary at 12:15 — half-open => belongs to bar B
        {"trade_id": "fB", "timestamp": "2026-05-14 12:15:00",
         "instrument_name": "BTC-PERPETUAL", "price": 60_100.0,
         "amount": 500.0, "direction": "buy", "index_price": 60_100.0,
         "mark_price": 60_100.0},
        # bar B
        {"trade_id": "f3", "timestamp": "2026-05-14 12:16:00",
         "instrument_name": "BTC-PERPETUAL", "price": 60_110.0,
         "amount": 3000.0, "direction": "buy", "index_price": 60_110.0,
         "mark_price": 60_110.0},
        {"trade_id": "f4", "timestamp": "2026-05-14 12:29:00",
         "instrument_name": "BTC-PERPETUAL", "price": 60_120.0,
         "amount": 4000.0, "direction": "sell", "index_price": 60_120.0,
         "mark_price": 60_120.0},
        # a non-perp future instrument — must be excluded
        {"trade_id": "fX", "timestamp": "2026-05-14 12:05:00",
         "instrument_name": "BTC-26JUN26", "price": 61_000.0,
         "amount": 9999.0, "direction": "buy", "index_price": 61_000.0,
         "mark_price": 61_000.0},
    ]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _option_df_two_bars() -> pd.DataFrame:
    """Deribit option trades, one in each of the two 15-min bars."""
    rows = [
        {"trade_id": "o1", "timestamp": "2026-05-14 12:03:00",
         "instrument_name": "BTC-15MAY26-60000-C", "price": 0.05,
         "amount": 1.0, "direction": "buy", "iv": 55.0,
         "index_price": 60_000.0, "mark_price": 0.05},
        {"trade_id": "o2", "timestamp": "2026-05-14 12:20:00",
         "instrument_name": "BTC-15MAY26-60000-P", "price": 0.04,
         "amount": 2.0, "direction": "sell", "iv": 56.0,
         "index_price": 60_100.0, "mark_price": 0.04},
    ]
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    return df


def _binance_df_two_bars() -> pd.DataFrame:
    """Binance USD-M aggTrades spanning the two bars, with a ``_ts`` column."""
    rows = [
        {"agg_trade_id": 1, "price": 60_000.0, "quantity": 0.1,
         "is_buyer_maker": False, "binance_ts": "2026-05-14 12:02:00"},
        {"agg_trade_id": 2, "price": 60_010.0, "quantity": 0.2,
         "is_buyer_maker": True, "binance_ts": "2026-05-14 12:10:00"},
        {"agg_trade_id": 3, "price": 60_110.0, "quantity": 0.3,
         "is_buyer_maker": False, "binance_ts": "2026-05-14 12:18:00"},
    ]
    df = pd.DataFrame(rows)
    df["binance_ts"] = pd.to_datetime(df["binance_ts"], utc=True)
    return df


def _svi_df_two_bars() -> pd.DataFrame:
    """Canonical 15-min-indexed frame carrying the close-of-bar index."""
    idx = pd.DatetimeIndex(
        ["2026-05-14 12:15:00", "2026-05-14 12:30:00"], tz="UTC",
        name="timestamp",
    )
    return pd.DataFrame({"index_price": [60_105.0, 60_125.0]}, index=idx)


def _sources_two_bars() -> ReplaySources:
    fut = _future_df_two_bars()
    opt = _option_df_two_bars()
    day = "2026-05-14"
    return ReplaySources(
        svi_df=_svi_df_two_bars(),
        option_by_day={day: opt},
        future_by_day={day: fut},
        binance_df=_binance_df_two_bars(),
        all_option_df=opt,
    )


# ── Test 1: synthetic 2-bar reconstruction ──────────────────────────


def test_reconstruct_bar_two_bars():
    """``reconstruct_bar`` partitions trades into the correct 15-min windows."""
    sources = _sources_two_bars()

    bar_a = reconstruct_bar(sources, pd.Timestamp("2026-05-14 12:15:00", tz="UTC"))
    bar_b = reconstruct_bar(sources, pd.Timestamp("2026-05-14 12:30:00", tz="UTC"))

    # Bar A window [12:00, 12:15): f1, f2 — the 12:15 boundary trade fB
    # is excluded (half-open), and the non-perp BTC-26JUN26 is excluded.
    a_perp_ids = {t.trade_id for t in bar_a.perp_trades}
    assert a_perp_ids == {"f1", "f2"}
    # Bar B window [12:15, 12:30): boundary fB + f3, f4.
    b_perp_ids = {t.trade_id for t in bar_b.perp_trades}
    assert b_perp_ids == {"fB", "f3", "f4"}

    # Option trades: one per bar.
    assert [t.trade_id for t in bar_a.option_trades] == ["o1"]
    assert [t.trade_id for t in bar_b.option_trades] == ["o2"]
    # Option metadata parsed from the instrument name.
    o1 = bar_a.option_trades[0]
    assert o1.strike == 60_000.0
    assert o1.option_type == "C"
    assert o1.expiry_dt is not None

    # TSRV (Binance) trades: 2 in bar A, 1 in bar B.
    assert {t.trade_id for t in bar_a.tsrv_trades} == {"binance:1", "binance:2"}
    assert {t.trade_id for t in bar_b.tsrv_trades} == {"binance:3"}
    # Direction: is_buyer_maker False -> +1 (aggressor bought); True -> -1.
    by_id = {t.trade_id: t for t in bar_a.tsrv_trades}
    assert by_id["binance:1"].direction == 1
    assert by_id["binance:2"].direction == -1

    # Close-of-bar index pulled from the canonical svi frame.
    assert bar_a.index_price == 60_105.0
    assert bar_b.index_price == 60_125.0
    # Bar shape sanity.
    assert bar_a.bar_minutes == 15
    assert bar_a.timestamp == pd.Timestamp("2026-05-14 12:15:00", tz="UTC")


# ── Test 2: process_bar drive produces a trace row ──────────────────


@pytest.fixture
def mock_config(tmp_path):
    """PaperConfig pointed at temp dirs with a feature_trace_path set."""
    from deribit_paper.config import PaperConfig

    cfg = PaperConfig()
    cfg.client_id = "test_id"
    cfg.client_secret = "test_secret"
    cfg.state_dir = str(tmp_path / "state")
    cfg.logs_dir = str(tmp_path / "logs")
    cfg.sqlite_db = str(tmp_path / "paper.db")
    cfg.metrics_parquet = str(tmp_path / "metrics.parquet")
    cfg.telegram_bot_token = ""
    cfg.telegram_chat_id = ""
    cfg.feature_trace_path = str(tmp_path / "bar_trace_replay.jsonl")
    cfg.ensure_dirs()
    return cfg


@pytest.mark.asyncio
async def test_process_bar_drive_produces_trace_row(mock_config, tmp_path):
    """Driving a reconstructed Bar through ``process_bar`` emits one trace row.

    The engine is wired manually (no ``initialize()`` / no network) with
    a real :class:`BarTraceWriter` pointed at a temp JSONL. The
    predictor is stubbed to return a flat signal; the harness still
    expects the BarTrace (C0/C1) to be flushed exactly once.
    """
    from deribit_paper.diagnostics.bar_trace import BarTraceWriter
    from deribit_paper.engine import TradingEngine
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
    engine.store = None
    engine.kill_switch = KillSwitch(
        equity_tracker=engine.equity_tracker, max_daily_drawdown_pct=5.0,
    )
    engine.risk_limits = RiskLimits(
        position_tracker=engine.position_tracker,
        equity_tracker=engine.equity_tracker,
    )

    # Real BarTraceWriter — the whole point of this test.
    trace_path = tmp_path / "bar_trace_replay.jsonl"
    engine.bar_trace_writer = BarTraceWriter(path=trace_path)
    assert engine.bar_trace_writer.enabled

    # Stub the predictor to a flat signal (mutates the trace's C2-C8 in
    # production; here we only assert C0/C1 were captured + flushed).
    engine.predictor = MagicMock()
    flat = MagicMock()
    flat.direction = 0
    flat.reason = "FLAT"
    flat.rv_pred_pct = 72.0
    flat.iv_used_pct = 72.3
    flat.edge_vol_pts = -0.3
    flat.is_actionable = False
    flat.straddle = None
    engine.predictor.predict.return_value = flat

    engine.strike_picker = AsyncMock()
    engine.strike_picker.pick_atm_straddle = AsyncMock(return_value=None)
    engine.straddle_flow = MagicMock()
    engine.straddle_flow.size_position = MagicMock(return_value=0.0)
    engine.straddle_flow.execute = AsyncMock()

    # Reconstruct a Bar from synthetic canonical sources and drive it.
    sources = _sources_two_bars()
    bar = reconstruct_bar(sources, pd.Timestamp("2026-05-14 12:15:00", tz="UTC"))
    # The synthetic bar has only 2 perp trades; the engine's low-volume
    # guard would skip it. Pad to clear ``min_perp_trades_per_bar``.
    from deribit_paper.market_data.buffer import Trade

    pad = [
        Trade(
            trade_id=f"pad{i}", timestamp=bar.timestamp - pd.Timedelta(seconds=900 - i),
            instrument_name="BTC-PERPETUAL", price=60_000.0 + i,
            amount=1000.0, direction=1 if i % 2 == 0 else -1,
            index_price=60_000.0 + i, mark_price=60_000.0 + i,
        )
        for i in range(max(0, mock_config.min_perp_trades_per_bar))
    ]
    bar.perp_trades = bar.perp_trades + pad

    await engine.process_bar(bar)

    # Exactly one JSONL trace row was written.
    assert trace_path.exists()
    lines = [ln for ln in trace_path.read_text(encoding="utf-8").splitlines() if ln.strip()]
    assert len(lines) == 1
    row = json.loads(lines[0])
    # The row carries the join key + C0 raw inputs from this bar.
    assert pd.Timestamp(row["bar_close_ts"]) == bar.timestamp
    assert row["side"] == "live"
    assert "n_perp_trades" in row
    assert row["n_perp_trades"] == len(bar.perp_trades)


# ── Test 3: window-filtering correctness ────────────────────────────


def test_window_filtering_correctness():
    """The half-open ``[T-15min, T)`` slicing excludes T, includes T-15min."""
    fut = _future_df_two_bars()
    start = pd.Timestamp("2026-05-14 12:00:00", tz="UTC")
    end = pd.Timestamp("2026-05-14 12:15:00", tz="UTC")

    # _slice_window: half-open. The 12:15 boundary row (fB) is excluded.
    sliced = _slice_window(fut, start, end)
    assert set(sliced["trade_id"]) == {"f1", "f2", "fX"}
    # ... and the start boundary IS included.
    start_boundary = _slice_window(
        fut,
        pd.Timestamp("2026-05-14 12:01:00", tz="UTC"),
        pd.Timestamp("2026-05-14 12:08:00", tz="UTC"),
    )
    assert "f1" in set(start_boundary["trade_id"])  # 12:01 == window start

    # _perp_trades_for_window drops the non-perp BTC-26JUN26 row (fX).
    perp = _perp_trades_for_window(fut, start, end)
    assert {t.trade_id for t in perp} == {"f1", "f2"}

    # _option_trades_for_window: only o1 falls in [12:00, 12:15).
    opt = _option_df_two_bars()
    opt_win = _option_trades_for_window(opt, start, end)
    assert {t.trade_id for t in opt_win} == {"o1"}

    # _tsrv_trades_for_window: 2 Binance rows in [12:00, 12:15).
    binance = _binance_df_two_bars()
    tsrv_win = _tsrv_trades_for_window(binance, start, end)
    assert {t.trade_id for t in tsrv_win} == {"binance:1", "binance:2"}
    # The 12:18 row is in the NEXT bar's window only.
    next_win = _tsrv_trades_for_window(
        binance, end, pd.Timestamp("2026-05-14 12:30:00", tz="UTC"),
    )
    assert {t.trade_id for t in next_win} == {"binance:3"}

    # _bar_grid produces canonical 15-min bar-close timestamps and floors
    # off-grid CLI endpoints onto the grid.
    grid = _bar_grid(
        pd.Timestamp("2026-05-14 12:03:00", tz="UTC"),   # off-grid -> ceil 12:15
        pd.Timestamp("2026-05-14 12:52:00", tz="UTC"),   # off-grid -> floor 12:45
    )
    assert grid == [
        pd.Timestamp("2026-05-14 12:15:00", tz="UTC"),
        pd.Timestamp("2026-05-14 12:30:00", tz="UTC"),
        pd.Timestamp("2026-05-14 12:45:00", tz="UTC"),
    ]
    # Empty grid when end precedes start.
    assert _bar_grid(end, start) == []


def test_cycle39b_harness_uses_exchange_order() -> None:
    """Cycle-39b source sentinel: the harness reconstructs tsrv_trades in
    exchange (agg_trade_id) order — the order the live fapi /aggTrades
    REST poller feeds the buffer.

    The replay MUST NOT order Binance trades by:
      * timestamp alone (unstable quicksort -> arbitrary same-ms ties),
      * the offline ``load_trades`` ``sort_index()`` order (which would
        make the parity test circular).
    ``_accumulate_bars`` is order-sensitive, so an unrealistic order
    fabricates a TSRV divergence that the live engine never produces.
    """
    import re
    src = Path("scripts/replay_parity_harness.py").read_text(encoding="utf-8")
    assert re.search(r'sort_values\(\s*"agg_trade_id"', src), (
        "harness must order Binance aggTrades by agg_trade_id (exchange order)"
    )
    # The circular / unrealistic orderings must not drive the load:
    # load_trades must not be imported or called (a comment may still
    # name it to explain WHY it is avoided).
    assert "import load_trades" not in src and "load_trades(" not in src, (
        "harness must NOT use the offline load_trades order (circular)"
    )
    assert 'sort_values("binance_ts")' not in src, (
        "harness must NOT order by timestamp alone (arbitrary tie order)"
    )
