"""Tests for RiskLimits pre-trade checks."""

from __future__ import annotations

import pandas as pd
import pytest

from deribit_paper.features.straddle import StraddleSelection
from deribit_paper.features.tsrv_live import BarTSRV
from deribit_paper.position.equity import EquityTracker
from deribit_paper.position.tracker import PositionTracker
from deribit_paper.risk.limits import RiskCheck, RiskLimits
from deribit_paper.signal.predictor import Signal


def _make_signal(
    direction: int = -1,
    strike: float = 60_000.0,
    expiry_days: float = 3.0,
    index_price: float = 60_000.0,
) -> Signal:
    now = pd.Timestamp("2026-04-10 12:00:00", tz="UTC")
    straddle = StraddleSelection(
        call_instrument="BTC-C",
        put_instrument="BTC-P",
        strike=strike,
        expiry_dt=now + pd.Timedelta(days=expiry_days),
        index_price=index_price,
        moneyness=strike / index_price,
        dte_days=expiry_days,
    )
    bar_tsrv = BarTSRV(
        tsrv=0.001, rv_fast=0.001, rv_slow=0.001, bpv=0.001, medrv=0.001,
        continuous_var=0.001, jump_var=0.0, jump_fraction=0.05,
        rs_down=0.0005, rs_up=0.0005, down_jump_fraction=0.0, up_jump_fraction=0.0,
        n_ticks=100, n_sub_bars=10, n_dash=9.0,
        vol_ann=0.8, vol_ann_continuous=0.8,
    )
    return Signal(
        timestamp=now,
        direction=direction,
        edge_vol_pts=-1.5 if direction == -1 else 1.5,
        rv_pred_pct=73.0,
        iv_used_pct=75.0,
        bar_tsrv=bar_tsrv,
        straddle=straddle,
        reason="test",
    )


def _make_limits(
    positions: PositionTracker,
    equity: EquityTracker,
    **overrides,
) -> RiskLimits:
    defaults = dict(
        max_open_positions=5,
        max_margin_util_hard=0.20,
        max_single_trade_btc=2.0,
        min_single_trade_btc=0.1,
        dte_min_days=1.0,
        dte_max_days=14.0,
        min_index_price=1_000.0,
        im_fraction=0.10,
    )
    defaults.update(overrides)
    return RiskLimits(
        position_tracker=positions,
        equity_tracker=equity,
        **defaults,
    )


def test_passes_healthy_signal():
    positions = PositionTracker()
    equity = EquityTracker(initial_equity_usd=100_000.0)
    limits = _make_limits(positions, equity)
    signal = _make_signal()
    check = limits.check_signal(
        signal, proposed_n_contracts_btc=1.0,
        now=pd.Timestamp("2026-04-10 12:00:00", tz="UTC"),
    )
    assert check.passed is True
    assert check.reason == "all checks passed"


def test_rejects_non_actionable():
    positions = PositionTracker()
    equity = EquityTracker(initial_equity_usd=100_000.0)
    limits = _make_limits(positions, equity)
    signal = _make_signal()
    signal.direction = 0
    signal.straddle = None
    check = limits.check_signal(signal, proposed_n_contracts_btc=1.0)
    assert check.passed is False


def test_rejects_too_small_size():
    positions = PositionTracker()
    equity = EquityTracker(initial_equity_usd=100_000.0)
    limits = _make_limits(positions, equity)
    signal = _make_signal()
    check = limits.check_signal(
        signal, proposed_n_contracts_btc=0.05,
        now=pd.Timestamp("2026-04-10 12:00:00", tz="UTC"),
    )
    assert check.passed is False
    assert "< min" in check.reason


def test_rejects_too_large_size():
    positions = PositionTracker()
    equity = EquityTracker(initial_equity_usd=1_000_000.0)
    limits = _make_limits(positions, equity, max_single_trade_btc=1.0)
    signal = _make_signal()
    check = limits.check_signal(
        signal, proposed_n_contracts_btc=1.5,
        now=pd.Timestamp("2026-04-10 12:00:00", tz="UTC"),
    )
    assert check.passed is False
    assert "> max" in check.reason


def test_rejects_dte_too_short():
    positions = PositionTracker()
    equity = EquityTracker(initial_equity_usd=100_000.0)
    limits = _make_limits(positions, equity, dte_min_days=2.0)
    signal = _make_signal(expiry_days=1.0)
    check = limits.check_signal(
        signal, proposed_n_contracts_btc=1.0,
        now=pd.Timestamp("2026-04-10 12:00:00", tz="UTC"),
    )
    assert check.passed is False
    assert "dte" in check.reason


def test_rejects_dte_too_long():
    positions = PositionTracker()
    equity = EquityTracker(initial_equity_usd=100_000.0)
    limits = _make_limits(positions, equity, dte_max_days=7.0)
    signal = _make_signal(expiry_days=14.0)
    check = limits.check_signal(
        signal, proposed_n_contracts_btc=1.0,
        now=pd.Timestamp("2026-04-10 12:00:00", tz="UTC"),
    )
    assert check.passed is False
    assert "dte" in check.reason


def test_rejects_max_open_positions():
    positions = PositionTracker()
    # Pre-populate with fake positions to hit the cap
    from deribit_paper.position.tracker import OpenPosition
    for i in range(5):
        positions._positions[f"t{i}"] = OpenPosition(
            trade_id=f"t{i}", signal_direction=-1,
            call_instrument="c", put_instrument="p",
            strike=60_000.0,
            expiry_dt=pd.Timestamp("2026-04-20", tz="UTC"),
            n_contracts_btc=1.0,
            entry_call_price=0.05, entry_put_price=0.04,
            entry_timestamp=pd.Timestamp("2026-04-10", tz="UTC"),
            entry_equity_usd=100_000.0,
        )
    equity = EquityTracker(initial_equity_usd=1_000_000.0)
    limits = _make_limits(positions, equity, max_open_positions=5)
    signal = _make_signal()
    check = limits.check_signal(
        signal, proposed_n_contracts_btc=1.0,
        now=pd.Timestamp("2026-04-10 12:00:00", tz="UTC"),
    )
    assert check.passed is False
    assert "max_open_positions" in check.reason


def test_rejects_margin_util_hard_cap():
    positions = PositionTracker()
    # 10 BTC open → notional = 600k USD, margin = 60k, equity = 100k → 60% util
    from deribit_paper.position.tracker import OpenPosition
    positions._positions["big"] = OpenPosition(
        trade_id="big", signal_direction=-1,
        call_instrument="c", put_instrument="p",
        strike=60_000.0,
        expiry_dt=pd.Timestamp("2026-04-20", tz="UTC"),
        n_contracts_btc=10.0,
        entry_call_price=0.05, entry_put_price=0.04,
        entry_timestamp=pd.Timestamp("2026-04-10", tz="UTC"),
        entry_equity_usd=100_000.0,
    )
    equity = EquityTracker(initial_equity_usd=100_000.0)
    limits = _make_limits(positions, equity, max_margin_util_hard=0.20)
    signal = _make_signal()
    check = limits.check_signal(
        signal, proposed_n_contracts_btc=1.0,
        now=pd.Timestamp("2026-04-10 12:00:00", tz="UTC"),
    )
    assert check.passed is False
    assert "margin_util" in check.reason


def test_rejects_low_index_price():
    positions = PositionTracker()
    equity = EquityTracker(initial_equity_usd=100_000.0)
    limits = _make_limits(positions, equity, min_index_price=10_000.0)
    signal = _make_signal(index_price=500.0)
    check = limits.check_signal(
        signal, proposed_n_contracts_btc=1.0,
        now=pd.Timestamp("2026-04-10 12:00:00", tz="UTC"),
    )
    assert check.passed is False
    assert "index_price" in check.reason


def test_risk_check_truthy():
    assert bool(RiskCheck(passed=True)) is True
    assert bool(RiskCheck(passed=False)) is False
