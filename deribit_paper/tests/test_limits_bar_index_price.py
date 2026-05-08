"""R-9 (parity-r3 cycle 3 — 2026-05-08) regression tests.

``RiskLimits.check_signal`` now accepts a ``bar_index_price`` kwarg
that drives the BTC→USD conversion in the margin envelope. This
matches offline ``vol_arb/sizing.py`` which uses the bar's spot, not
the snapshot embedded in ``signal.straddle.index_price`` (which can
lag the bar by tens of seconds during high-volatility ticks).

Pre-fix the gate always used the snapshot — under-/over-stating the
gate by ``spot_drift × notional`` against the operator's mental model.
"""

from __future__ import annotations

import pandas as pd
import pytest

from deribit_paper.features.straddle import StraddleSelection
from deribit_paper.position.equity import EquityTracker
from deribit_paper.position.tracker import PositionTracker
from deribit_paper.risk.limits import RiskLimits
from deribit_paper.signal.predictor import Signal


def _make_signal(*, snapshot_spot: float = 100_000.0) -> Signal:
    now = pd.Timestamp.now(tz="UTC")
    expiry = now + pd.Timedelta(days=10)
    straddle = StraddleSelection(
        call_instrument="BTC-C", put_instrument="BTC-P",
        strike=snapshot_spot, expiry_dt=expiry,
        index_price=snapshot_spot, moneyness=1.0, dte_days=10.0,
    )
    return Signal(
        timestamp=now,
        direction=-1, edge_vol_pts=-1.5,
        rv_pred_pct=73.0, iv_used_pct=75.0,
        bar_tsrv=None, straddle=straddle, reason="test",
    )


def _make_limits(*, equity_usd: float = 100_000.0) -> RiskLimits:
    tracker = PositionTracker()
    equity = EquityTracker(initial_equity_usd=equity_usd)
    return RiskLimits(
        position_tracker=tracker,
        equity_tracker=equity,
        max_open_positions=5,
        max_margin_util_hard=0.20,
        max_single_trade_btc=2.0,
        min_single_trade_btc=0.1,
        im_fraction=0.20,
    )


def test_bar_index_price_overrides_snapshot_when_provided():
    """When `bar_index_price` is provided AND > min_index_price, the
    margin envelope uses it instead of `signal.straddle.index_price`.
    Set the snapshot to 100K and the bar to 200K — the gate should
    compute notional×200K, not notional×100K."""
    limits = _make_limits(equity_usd=100_000.0)
    # 0.5 BTC @ snapshot 100K → margin_used = 0.20 × 0.5 × 100K = $10K
    # margin_util = $10K / $100K = 10% → passes (cap 20%).
    # 0.5 BTC @ bar 200K → margin_used = 0.20 × 0.5 × 200K = $20K
    # margin_util = $20K / $100K = 20% → at cap, NOT > → still passes.
    # 0.6 BTC @ bar 200K → margin_used = 0.20 × 0.6 × 200K = $24K
    # margin_util = 24% → BLOCKED.
    sig = _make_signal(snapshot_spot=100_000.0)
    bar_spot = 200_000.0
    check_passes = limits.check_signal(
        sig, proposed_n_contracts_btc=0.5,
        bar_index_price=bar_spot,
    )
    assert check_passes.passed is True
    check_blocks = limits.check_signal(
        sig, proposed_n_contracts_btc=0.6,
        bar_index_price=bar_spot,
    )
    assert check_blocks.passed is False
    assert "margin_util" in check_blocks.reason


def test_falls_back_to_snapshot_when_bar_index_price_none():
    """No bar_index_price kwarg → snapshot is used (legacy contract)."""
    limits = _make_limits(equity_usd=100_000.0)
    sig = _make_signal(snapshot_spot=100_000.0)
    # 0.5 BTC at 100K snapshot → 10% util → passes.
    check = limits.check_signal(sig, proposed_n_contracts_btc=0.5)
    assert check.passed is True


def test_bar_index_price_below_min_falls_back_to_snapshot():
    """Defensive: if `bar_index_price` is degenerate (below
    `min_index_price`), the gate ignores it and uses the snapshot.
    Avoids a stale/zeroed bar field tricking the gate into an
    artificially small margin envelope."""
    limits = _make_limits(equity_usd=100_000.0)
    sig = _make_signal(snapshot_spot=100_000.0)
    # bar_spot=0.0 is degenerate → snapshot used → 10% util → passes.
    check = limits.check_signal(
        sig, proposed_n_contracts_btc=0.5,
        bar_index_price=0.0,
    )
    assert check.passed is True
    # Sanity: at the same equity + size, with snapshot 100K, the gate
    # is computing margin_used = 0.20 × 0.5 × 100K = $10K → 10% util.
    # If the gate had used bar=0 → notional=0 → 0% util, which would
    # ALSO pass — so verify by setting a snapshot that should block.
    sig_huge = _make_signal(snapshot_spot=2_000_000.0)
    check_blocked = limits.check_signal(
        sig_huge, proposed_n_contracts_btc=0.5,
        bar_index_price=0.0,
    )
    # 0.5 BTC × 2M × 0.20 = $200K margin / $100K equity = 200% → blocked.
    assert check_blocked.passed is False
    assert "margin_util" in check_blocked.reason


def test_index_price_sanity_uses_snapshot_not_bar():
    """The snapshot still drives the index_price sanity check (gate 4).
    A bar that drifts above min_index_price doesn't rescue a degenerate
    snapshot — the snapshot's spot was used to pick the ATM strike, so
    a degenerate snapshot means the trade idea itself is malformed."""
    limits = _make_limits(equity_usd=100_000.0)
    sig = _make_signal(snapshot_spot=500.0)  # below min 1000
    check = limits.check_signal(
        sig, proposed_n_contracts_btc=0.1,
        bar_index_price=100_000.0,  # healthy, but doesn't rescue
    )
    assert check.passed is False
    assert "index_price" in check.reason
