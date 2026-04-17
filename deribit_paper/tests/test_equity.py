"""Tests for EquityTracker — running PnL, peak, drawdown, daily return."""

from __future__ import annotations

import pandas as pd
import pytest

from deribit_paper.position.equity import EquityTracker


def test_initial_state():
    t = EquityTracker(initial_equity_usd=100_000.0)
    assert t.initial_equity == 100_000.0
    assert t.current_equity == 100_000.0
    assert t.peak_equity == 100_000.0
    assert t.realized_pnl == 0.0
    assert t.drawdown_pct == 0.0
    assert t.n_trades_closed == 0
    assert t.total_return_pct == 0.0


def test_initial_equity_must_be_positive():
    with pytest.raises(ValueError, match="positive"):
        EquityTracker(initial_equity_usd=0.0)
    with pytest.raises(ValueError, match="positive"):
        EquityTracker(initial_equity_usd=-100.0)


def test_winning_trade_updates_peak():
    t = EquityTracker(initial_equity_usd=100_000.0)
    snap = t.on_trade_closed(5_000.0)
    assert t.current_equity == 105_000.0
    assert t.peak_equity == 105_000.0
    assert t.realized_pnl == 5_000.0
    assert t.drawdown_pct == 0.0  # at peak
    assert snap.n_trades_closed == 1
    assert t.total_return_pct == pytest.approx(5.0)


def test_losing_trade_creates_drawdown():
    t = EquityTracker(initial_equity_usd=100_000.0)
    t.on_trade_closed(5_000.0)   # equity 105k, peak 105k
    t.on_trade_closed(-3_000.0)  # equity 102k, peak 105k, dd -2.857%
    assert t.peak_equity == 105_000.0
    assert t.current_equity == 102_000.0
    assert t.drawdown_pct == pytest.approx(-2.8571, abs=1e-3)


def test_dropped_below_threshold():
    t = EquityTracker(initial_equity_usd=100_000.0)
    t.on_trade_closed(10_000.0)  # peak 110k
    t.on_trade_closed(-5_000.0)  # -4.545% dd
    assert t.dropped_below_threshold(5.0) is False
    assert t.dropped_below_threshold(4.0) is True


def test_set_equity_non_trade_update():
    t = EquityTracker(initial_equity_usd=100_000.0)
    t.set_equity(98_000.0, reason="reconcile")
    assert t.current_equity == 98_000.0
    assert t.realized_pnl == 0.0  # reconcile does not change realized PnL
    assert t.n_trades_closed == 0


def test_set_equity_rejects_negative():
    t = EquityTracker(initial_equity_usd=100_000.0)
    with pytest.raises(ValueError, match="non-negative"):
        t.set_equity(-10.0)


def test_snapshots_buffer_appends():
    t = EquityTracker(initial_equity_usd=100_000.0)
    snaps_before = list(t.snapshots())
    t.on_trade_closed(1000.0)
    t.on_trade_closed(-500.0)
    snaps_after = list(t.snapshots())
    assert len(snaps_after) == len(snaps_before) + 2


def test_daily_return_pct_basic():
    t = EquityTracker(initial_equity_usd=100_000.0)
    # With no intermediate snapshots, daily return uses initial equity
    t.on_trade_closed(2_000.0)
    daily = t.daily_return_pct()
    assert daily > 0  # positive return today
