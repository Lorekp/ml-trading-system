"""Tests for SQLiteStore — trades, equity, kill switch state."""

from __future__ import annotations

import pandas as pd
import pytest

from deribit_paper.execution.order_manager import OrderResult
from deribit_paper.execution.straddle_flow import TradeOutcome
from deribit_paper.features.straddle import StraddleSelection
from deribit_paper.features.tsrv_live import BarTSRV
from deribit_paper.position.equity import EquitySnapshot
from deribit_paper.position.store import SQLiteStore
from deribit_paper.signal.predictor import Signal


def _make_outcome() -> TradeOutcome:
    straddle = StraddleSelection(
        call_instrument="BTC-C", put_instrument="BTC-P",
        strike=60_000.0,
        expiry_dt=pd.Timestamp("2026-04-20 08:00:00", tz="UTC"),
        index_price=60_000.0, moneyness=1.0, dte_days=5.0,
    )
    bar_tsrv = BarTSRV(
        tsrv=0.001, rv_fast=0.001, rv_slow=0.001, bpv=0.001, medrv=0.001,
        continuous_var=0.001, jump_var=0.0, jump_fraction=0.05,
        rs_down=0.0005, rs_up=0.0005, down_jump_fraction=0.0, up_jump_fraction=0.0,
        n_ticks=100, n_sub_bars=10, n_dash=9.0,
        vol_ann=0.8, vol_ann_continuous=0.8,
    )
    signal = Signal(
        timestamp=pd.Timestamp("2026-04-10 12:00:00", tz="UTC"),
        direction=-1, edge_vol_pts=-1.5,
        rv_pred_pct=73.0, iv_used_pct=75.0,
        bar_tsrv=bar_tsrv, straddle=straddle, reason="test",
    )
    entry = OrderResult(
        client_order_id="c", order_id="oc",
        instrument_name="BTC-C", side="sell",
        amount_requested=1.0, filled_amount=1.0,
        average_price=0.05, state="filled", is_dry_run=True,
    )
    return TradeOutcome(
        trade_id="tid_test", signal=signal,
        entry_call=entry, entry_put=entry,
        exit_call=entry, exit_put=entry,
        n_contracts_btc=1.0, status="completed",
        started_at=pd.Timestamp("2026-04-10 12:00:00", tz="UTC"),
        ended_at=pd.Timestamp("2026-04-10 12:15:00", tz="UTC"),
    )


@pytest.mark.asyncio
async def test_open_and_close(tmp_path):
    db = tmp_path / "paper.db"
    async with SQLiteStore(db) as store:
        assert db.exists()
        # Tables exist via introspection
        cursor = await store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'",
        )
        names = {row[0] for row in await cursor.fetchall()}
        assert "trades" in names
        assert "equity_snapshots" in names
        assert "kill_switch_state" in names


@pytest.mark.asyncio
async def test_persist_and_load_trade(tmp_path):
    async with SQLiteStore(tmp_path / "paper.db") as store:
        outcome = _make_outcome()
        await store.persist_trade(outcome, realized_pnl_usd=250.0)
        trades = await store.load_recent_trades(limit=10)
        assert len(trades) == 1
        t = trades[0]
        assert t["trade_id"] == "tid_test"
        assert t["status"] == "completed"
        assert t["realized_pnl_usd"] == 250.0
        assert t["signal_direction"] == -1
        assert t["n_contracts_btc"] == 1.0


@pytest.mark.asyncio
async def test_persist_equity_snapshot(tmp_path):
    async with SQLiteStore(tmp_path / "paper.db") as store:
        snap = EquitySnapshot(
            timestamp=pd.Timestamp("2026-04-10 12:00:00", tz="UTC"),
            equity_usd=101_500.0,
            realized_pnl_usd=1_500.0,
            peak_equity_usd=101_500.0,
            drawdown_pct=0.0,
            n_trades_closed=1,
        )
        await store.persist_equity_snapshot(snap)
        loaded = await store.load_latest_equity()
        assert loaded is not None
        assert loaded.equity_usd == 101_500.0
        assert loaded.n_trades_closed == 1


@pytest.mark.asyncio
async def test_kill_switch_state_default_false(tmp_path):
    async with SQLiteStore(tmp_path / "paper.db") as store:
        halted, reason, halted_at = await store.get_kill_switch_state()
        assert halted is False
        assert reason is None


@pytest.mark.asyncio
async def test_kill_switch_state_set_and_get(tmp_path):
    async with SQLiteStore(tmp_path / "paper.db") as store:
        await store.set_kill_switch_state(True, reason="test halt")
        halted, reason, halted_at = await store.get_kill_switch_state()
        assert halted is True
        assert reason == "test halt"
        assert halted_at is not None

        await store.set_kill_switch_state(False)
        halted, reason, halted_at = await store.get_kill_switch_state()
        assert halted is False


@pytest.mark.asyncio
async def test_persist_trade_requires_open(tmp_path):
    store = SQLiteStore(tmp_path / "paper.db")
    with pytest.raises(RuntimeError, match="open"):
        await store.persist_trade(_make_outcome())


# ── M3: Open positions persistence ────────────────────────────────


@pytest.mark.asyncio
async def test_open_positions_table_exists(tmp_path):
    async with SQLiteStore(tmp_path / "paper.db") as store:
        cursor = await store._conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'",
        )
        names = {row[0] for row in await cursor.fetchall()}
        assert "open_positions" in names


@pytest.mark.asyncio
async def test_persist_and_load_open_position(tmp_path):
    async with SQLiteStore(tmp_path / "paper.db") as store:
        await store.persist_open_position(
            trade_id="trade_001",
            call_instrument="BTC-12APR26-60000-C",
            put_instrument="BTC-12APR26-60000-P",
            signal_direction=-1,
            n_contracts_btc=0.1,
            strike=60000.0,
            expiry_dt="2026-04-12T08:00:00+00:00",
            opened_at="2026-04-10T12:00:00+00:00",
            entry_call_price=0.045,
            entry_put_price=0.042,
        )
        positions = await store.load_open_positions()
        assert len(positions) == 1
        p = positions[0]
        assert p["trade_id"] == "trade_001"
        assert p["call_instrument"] == "BTC-12APR26-60000-C"
        assert p["signal_direction"] == -1
        assert p["n_contracts_btc"] == 0.1
        assert p["entry_call_price"] == 0.045
        assert p["entry_put_price"] == 0.042


@pytest.mark.asyncio
async def test_remove_open_position(tmp_path):
    async with SQLiteStore(tmp_path / "paper.db") as store:
        await store.persist_open_position(
            trade_id="trade_002",
            call_instrument="BTC-C",
            put_instrument="BTC-P",
            signal_direction=1,
            n_contracts_btc=0.2,
            strike=65000.0,
            expiry_dt="2026-04-15T08:00:00+00:00",
            opened_at="2026-04-11T10:00:00+00:00",
        )
        await store.remove_open_position("trade_002")
        positions = await store.load_open_positions()
        assert len(positions) == 0
