"""Tests for RiskLimits — especially minimum lot exemption for small accounts."""

from __future__ import annotations

import pandas as pd
import pytest

from deribit_paper.position.equity import EquityTracker
from deribit_paper.position.tracker import PositionTracker
from deribit_paper.risk.limits import RiskCheck, RiskLimits
from deribit_paper.signal.predictor import Signal
from deribit_paper.features.straddle import StraddleSelection


def _make_signal(*, direction: int = -1, spot: float = 100_000.0) -> Signal:
    # Timestamps are relative to wall-clock now so RiskLimits.check_signal
    # (which falls back to pd.Timestamp.now(tz="UTC") for the DTE gate when
    # no explicit `now=` is passed) sees a future expiry. Keep the original
    # "signal at T, expiry at T+10d" semantic.
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


def _make_limits(equity_usd: float = 750.0, **overrides) -> RiskLimits:
    tracker = PositionTracker()
    equity = EquityTracker(initial_equity_usd=equity_usd)
    defaults = dict(
        position_tracker=tracker,
        equity_tracker=equity,
        max_open_positions=1,
        max_margin_util_hard=0.20,
        max_single_trade_btc=2.0,
        min_single_trade_btc=0.1,
        im_fraction=0.10,
    )
    defaults.update(overrides)
    return RiskLimits(**defaults)


# ── Minimum lot exemption ────────────────────────────────────────


def test_minimum_lot_passes_at_small_equity():
    """0.1 BTC + 0 open positions passes even at 133% margin util."""
    limits = _make_limits(equity_usd=750.0)
    signal = _make_signal()
    check = limits.check_signal(signal, proposed_n_contracts_btc=0.1)
    assert check.passed is True


def test_larger_than_minimum_blocked_at_small_equity():
    """0.2 BTC at $750 equity is NOT exempt — margin util too high."""
    limits = _make_limits(equity_usd=750.0)
    signal = _make_signal()
    check = limits.check_signal(signal, proposed_n_contracts_btc=0.2)
    assert check.passed is False
    assert "margin_util" in check.reason


def test_minimum_lot_exempt_even_when_position_already_open():
    """R-4 (parity-r3 cycle 3): the floor lot is exempt regardless of
    open-position count. Pre-fix this test asserted the opposite — the
    exemption only fired when ``n_open == 0`` — which silently blocked
    DD-recovery scenarios where the engine restarted with one min-lot
    trade still on the book and could not enter a second min-lot trade.
    Forward-looking semantic: Deribit's 0.1 BTC floor is the smallest
    expressible size, so the operator's only avenue to scale OUT of a
    degraded equity state is to keep entering floor lots. Larger
    proposals still go through the normal margin gate.
    """
    tracker = PositionTracker()
    equity = EquityTracker(initial_equity_usd=750.0)
    limits = RiskLimits(
        position_tracker=tracker,
        equity_tracker=equity,
        max_open_positions=2,  # allow 2 to test margin check, not position cap
        min_single_trade_btc=0.1,
        im_fraction=0.10,
    )
    # Simulate one open position. Same relative-to-now convention as
    # _make_signal so the synthetic state stays internally consistent.
    from deribit_paper.position.tracker import OpenPosition
    now = pd.Timestamp.now(tz="UTC")
    tracker._positions["existing"] = OpenPosition(
        trade_id="existing", signal_direction=-1,
        call_instrument="BTC-C", put_instrument="BTC-P",
        strike=100_000.0,
        expiry_dt=now + pd.Timedelta(days=10),
        n_contracts_btc=0.1,
        entry_call_price=0.05, entry_put_price=0.05,
        entry_timestamp=now,
        entry_equity_usd=750.0,
    )
    signal = _make_signal()
    check = limits.check_signal(signal, proposed_n_contracts_btc=0.1)
    assert check.passed is True
    # Sanity: a NON-floor proposal in the same state IS still gated.
    check_oversize = limits.check_signal(signal, proposed_n_contracts_btc=0.5)
    assert check_oversize.passed is False
    assert "margin_util" in check_oversize.reason


# ── Normal margin check at higher equity ─────────────────────────


def test_normal_margin_check_at_high_equity():
    """At $100K equity, 0.5 BTC passes normal margin check (util ~0.5%)."""
    limits = _make_limits(equity_usd=100_000.0)
    signal = _make_signal()
    check = limits.check_signal(signal, proposed_n_contracts_btc=0.5)
    assert check.passed is True


def test_margin_check_rejects_over_cap_at_high_equity():
    """At $100K equity, huge trade still blocked by margin util hard cap."""
    limits = _make_limits(equity_usd=100_000.0, max_single_trade_btc=50.0)
    signal = _make_signal()
    # 30 BTC × 100K × 0.10 = 300K margin / 100K equity = 300% → blocked
    check = limits.check_signal(signal, proposed_n_contracts_btc=30.0)
    assert check.passed is False
    assert "margin_util" in check.reason


# ── Scaling transition: floor → formula ──────────────────────────


def test_minimum_lot_at_5k_equity_passes():
    """At $5K equity, 0.1 BTC margin util = 20% → exactly at normal cap.
    The minimum lot exemption still applies (n_open=0)."""
    limits = _make_limits(equity_usd=5_000.0)
    signal = _make_signal()
    check = limits.check_signal(signal, proposed_n_contracts_btc=0.1)
    assert check.passed is True


def test_formula_size_at_20k_equity_passes():
    """At $20K equity, formula produces 0.1 BTC, margin util = 5% → passes normally."""
    limits = _make_limits(equity_usd=20_000.0)
    signal = _make_signal()
    check = limits.check_signal(signal, proposed_n_contracts_btc=0.1)
    assert check.passed is True


# ── Realistic Deribit IM (im_fraction=0.20) ─────────────────────


def test_deribit_im_minimum_lot_at_small_equity():
    """With realistic IM (0.20), 0.1 BTC at $750 needs exemption (267% util)."""
    limits = _make_limits(equity_usd=750.0, im_fraction=0.20)
    signal = _make_signal()
    check = limits.check_signal(signal, proposed_n_contracts_btc=0.1)
    assert check.passed is True  # minimum lot exemption


def test_deribit_im_5k_equity_needs_exemption():
    """With realistic IM (0.20), 0.1 BTC at $5K → 40% util → exceeds 20% cap.
    Minimum lot exemption still allows it."""
    limits = _make_limits(equity_usd=5_000.0, im_fraction=0.20)
    signal = _make_signal()
    check = limits.check_signal(signal, proposed_n_contracts_btc=0.1)
    assert check.passed is True  # minimum lot exemption


def test_deribit_im_20k_equity_passes_normally():
    """With realistic IM (0.20), 0.1 BTC at $20K → 10% util → below 20% cap."""
    limits = _make_limits(equity_usd=20_000.0, im_fraction=0.20)
    signal = _make_signal()
    check = limits.check_signal(signal, proposed_n_contracts_btc=0.1)
    assert check.passed is True  # passes normal margin check


def test_deribit_im_larger_trade_at_20k():
    """With realistic IM (0.20), 0.3 BTC at $20K → 30% util → blocked."""
    limits = _make_limits(equity_usd=20_000.0, im_fraction=0.20)
    signal = _make_signal()
    check = limits.check_signal(signal, proposed_n_contracts_btc=0.3)
    assert check.passed is False
    assert "margin_util" in check.reason
