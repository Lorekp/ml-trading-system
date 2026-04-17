"""Tests for PositionTracker."""

from __future__ import annotations

import pandas as pd
import pytest

from deribit_paper.execution.order_manager import OrderResult
from deribit_paper.execution.straddle_flow import TradeOutcome
from deribit_paper.features.straddle import StraddleSelection
from deribit_paper.features.tsrv_live import BarTSRV
from deribit_paper.position.tracker import OpenPosition, PositionTracker
from deribit_paper.signal.predictor import Signal


def _make_outcome(trade_id: str = "live_abc") -> TradeOutcome:
    straddle = StraddleSelection(
        call_instrument="BTC-12APR26-60000-C",
        put_instrument="BTC-12APR26-60000-P",
        strike=60_000.0,
        expiry_dt=pd.Timestamp("2026-04-12 08:00:00", tz="UTC"),
        index_price=60_000.0,
        moneyness=1.0,
        dte_days=2.0,
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
        direction=-1,
        edge_vol_pts=-1.5,
        rv_pred_pct=73.0,
        iv_used_pct=75.0,
        bar_tsrv=bar_tsrv,
        straddle=straddle,
        reason="test",
    )
    entry_call = OrderResult(
        client_order_id="c", order_id="oc",
        instrument_name="BTC-12APR26-60000-C",
        side="sell", amount_requested=1.0, filled_amount=1.0,
        average_price=0.05, state="filled", is_dry_run=True,
    )
    entry_put = OrderResult(
        client_order_id="p", order_id="op",
        instrument_name="BTC-12APR26-60000-P",
        side="sell", amount_requested=1.0, filled_amount=1.0,
        average_price=0.04, state="filled", is_dry_run=True,
    )
    return TradeOutcome(
        trade_id=trade_id, signal=signal,
        entry_call=entry_call, entry_put=entry_put,
        exit_call=None, exit_put=None,
        n_contracts_btc=1.0,
        status="completed",
        started_at=pd.Timestamp("2026-04-10 12:00:00", tz="UTC"),
    )


def test_initial_state():
    t = PositionTracker()
    assert t.n_open == 0
    assert t.n_closed == 0
    assert t.all_open() == []
    assert t.total_notional_btc() == 0.0
    assert t.total_unrealized_pnl_btc() == 0.0


def test_open_and_close_position():
    t = PositionTracker()
    outcome = _make_outcome("tid1")
    pos = t.open_position(
        trade_id="tid1", outcome=outcome, current_equity_usd=100_000.0,
    )
    assert isinstance(pos, OpenPosition)
    assert t.n_open == 1
    assert t.total_notional_btc() == 1.0

    closed = t.close_position("tid1")
    assert closed is not None
    assert closed.status == "closed"
    assert t.n_open == 0
    assert t.n_closed == 1


def test_unrealized_pnl_without_marks_is_none():
    t = PositionTracker()
    outcome = _make_outcome("tid")
    pos = t.open_position(
        trade_id="tid", outcome=outcome, current_equity_usd=100_000.0,
    )
    assert pos.unrealized_pnl_btc() is None
    # Total ignores positions with no marks (treats as 0)
    assert t.total_unrealized_pnl_btc() == 0.0


def test_unrealized_pnl_short_straddle_profit():
    """Short straddle profits when premium shrinks."""
    t = PositionTracker()
    outcome = _make_outcome("tid")
    pos = t.open_position(
        trade_id="tid", outcome=outcome, current_equity_usd=100_000.0,
    )
    # entry premium per BTC = 0.05 + 0.04 = 0.09
    # mark premium shrinks to 0.06 + 0.02 = 0.08 → short profit = 0.01 BTC
    t.update_mark("tid", call_mark=0.06, put_mark=0.02)
    pnl = pos.unrealized_pnl_btc()
    assert pnl is not None
    assert pnl == pytest.approx(0.01)


def test_unrealized_pnl_short_straddle_loss():
    t = PositionTracker()
    outcome = _make_outcome("tid")
    pos = t.open_position(
        trade_id="tid", outcome=outcome, current_equity_usd=100_000.0,
    )
    # premium inflates to 0.08 + 0.07 = 0.15 → short loss = -0.06 BTC
    t.update_mark("tid", call_mark=0.08, put_mark=0.07)
    pnl = pos.unrealized_pnl_btc()
    assert pnl is not None
    assert pnl == pytest.approx(-0.06)


def test_close_unknown_trade_returns_none():
    t = PositionTracker()
    assert t.close_position("does_not_exist") is None
    assert t.n_closed == 0


def test_update_hedge_inventory():
    t = PositionTracker()
    outcome = _make_outcome("tid")
    t.open_position(trade_id="tid", outcome=outcome, current_equity_usd=100_000.0)
    t.update_hedge_inventory("tid", 0.5)
    pos = t.get("tid")
    assert pos.hedge_inventory_btc == 0.5
