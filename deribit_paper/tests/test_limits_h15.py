"""Tests for H15 RiskLimits new caps (max_trades_per_hour + max_gross_notional).

Why this exists
---------------

Before H15, the engine had no defence against a runaway predictor
firing dozens of trades in minutes (only the per-trade ``max_single_
trade_btc`` and the position-count ``max_open_positions``). It also
had no hard cap on TOTAL open USD notional — N small positions each
within per-trade limits could still aggregate above the operator's
intended exposure.

Tests pin the H15 contract:

1. ``max_trades_per_hour=0`` disables the rate cap (legacy behaviour).
2. ``record_trade_fired`` populates the rolling deque.
3. The rate cap blocks the (cap+1)th trade within 1h.
4. Old timestamps fall out of the 1h window.
5. ``max_gross_notional_usd=0`` disables the notional cap.
6. The notional cap blocks a trade that pushes the total over.
"""
from __future__ import annotations

import pandas as pd
import pytest

from deribit_paper.features.straddle import StraddleSelection
from deribit_paper.position.equity import EquityTracker
from deribit_paper.position.tracker import PositionTracker
from deribit_paper.risk.limits import RiskLimits
from deribit_paper.signal.predictor import Signal


def _make_signal(*, direction: int = -1, spot: float = 100_000.0) -> Signal:
    now = pd.Timestamp.now(tz="UTC")
    expiry = now + pd.Timedelta(days=10)
    straddle = StraddleSelection(
        call_instrument="BTC-C", put_instrument="BTC-P",
        strike=spot, expiry_dt=expiry,
        index_price=spot, moneyness=1.0, dte_days=10.0,
    )
    return Signal(
        timestamp=now,
        direction=direction, edge_vol_pts=-1.5,
        rv_pred_pct=73.0, iv_used_pct=75.0,
        bar_tsrv=None, straddle=straddle, reason="test",
    )


def _make_limits(**overrides) -> RiskLimits:
    tracker = PositionTracker()
    equity = EquityTracker(initial_equity_usd=100_000.0)
    defaults = dict(
        position_tracker=tracker,
        equity_tracker=equity,
        max_open_positions=10,
        max_margin_util_hard=1.0,    # generous, we want H15 caps to bind
        max_single_trade_btc=2.0,
        min_single_trade_btc=0.1,
        im_fraction=0.05,
    )
    defaults.update(overrides)
    return RiskLimits(**defaults)


# ── max_trades_per_hour ──────────────────────────────────────────────


def test_zero_max_trades_per_hour_disables_cap():
    """Default 0 = disabled. Many trades pass."""
    limits = _make_limits(max_trades_per_hour=0)
    signal = _make_signal()
    for _ in range(50):
        limits.record_trade_fired()
    check = limits.check_signal(signal, proposed_n_contracts_btc=0.5)
    assert check.passed is True


def test_record_trade_fired_with_zero_cap_does_not_grow_deque():
    """Recording is a no-op when the cap is disabled."""
    limits = _make_limits(max_trades_per_hour=0)
    for _ in range(20):
        limits.record_trade_fired()
    assert len(limits._recent_trade_timestamps) == 0  # noqa: SLF001


def test_record_trade_fired_populates_deque():
    limits = _make_limits(max_trades_per_hour=6)
    for _ in range(3):
        limits.record_trade_fired()
    assert len(limits._recent_trade_timestamps) == 3  # noqa: SLF001


def test_cap_blocks_burst():
    """6 trades per hour cap. The 7th MUST be rejected."""
    limits = _make_limits(max_trades_per_hour=6)
    signal = _make_signal()
    now = pd.Timestamp.now(tz="UTC")
    for i in range(6):
        limits.record_trade_fired(at=now - pd.Timedelta(seconds=i * 10))
    # 6 in last hour → 7th blocked
    check = limits.check_signal(signal, proposed_n_contracts_btc=0.5, now=now)
    assert check.passed is False
    assert "max_trades_per_hour" in check.reason


def test_old_trades_fall_out_of_window():
    """Trades older than 1h don't count toward the cap."""
    limits = _make_limits(max_trades_per_hour=3)
    signal = _make_signal()
    now = pd.Timestamp.now(tz="UTC")
    # 3 trades 90 min ago — outside the 1h window
    for i in range(3):
        limits.record_trade_fired(at=now - pd.Timedelta(minutes=90 + i))
    # 1 fresh trade
    limits.record_trade_fired(at=now - pd.Timedelta(minutes=5))
    check = limits.check_signal(signal, proposed_n_contracts_btc=0.5, now=now)
    assert check.passed is True


def test_cap_at_exact_boundary():
    """Cap of 6: 5 fired = pass, 6 fired = block."""
    limits = _make_limits(max_trades_per_hour=6)
    signal = _make_signal()
    now = pd.Timestamp.now(tz="UTC")
    for _ in range(5):
        limits.record_trade_fired(at=now)
    assert limits.check_signal(signal, proposed_n_contracts_btc=0.5, now=now).passed
    limits.record_trade_fired(at=now)
    assert not limits.check_signal(signal, proposed_n_contracts_btc=0.5, now=now).passed


# ── max_gross_notional_usd ─────────────────────────────────────────


def test_zero_max_gross_notional_disables_cap():
    """0 = disabled (legacy)."""
    limits = _make_limits(max_gross_notional_usd=0.0)
    signal = _make_signal(spot=100_000.0)
    check = limits.check_signal(signal, proposed_n_contracts_btc=2.0)
    # 2 BTC × $100k = $200k notional; should still pass with cap=0
    assert check.passed is True


def test_gross_notional_cap_blocks_oversized_trade():
    """At $100k spot and $30k cap, max trade size is 0.3 BTC.
    A 0.5 BTC trade = $50k notional → rejected."""
    limits = _make_limits(max_gross_notional_usd=30_000.0)
    signal = _make_signal(spot=100_000.0)
    check = limits.check_signal(signal, proposed_n_contracts_btc=0.5)
    assert check.passed is False
    assert "gross_notional" in check.reason


def test_gross_notional_cap_passes_within_budget():
    """0.2 BTC × $100k = $20k notional, under $30k cap."""
    limits = _make_limits(max_gross_notional_usd=30_000.0)
    signal = _make_signal(spot=100_000.0)
    check = limits.check_signal(signal, proposed_n_contracts_btc=0.2)
    assert check.passed is True
