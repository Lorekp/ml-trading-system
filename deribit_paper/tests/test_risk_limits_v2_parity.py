"""D10 (parity 2026-05-07): per-trade size cap behaviour.

Mirrors V2 backtester ``BacktestConfig.max_position_btc=0.0`` (= no
hard cap, dynamic_btc_cap is the only ceiling). Live ``RiskLimits``
treats ``max_single_trade_btc <= 0`` as "skip the cap check" so V2
preset can set ``cfg.max_single_trade_btc=0.0`` for parity.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd

from deribit_paper.features.straddle import StraddleSelection
from deribit_paper.position.equity import EquityTracker
from deribit_paper.position.tracker import PositionTracker
from deribit_paper.risk.limits import RiskLimits
from deribit_paper.signal.predictor import Signal


def _make_signal(strike: float = 100_000.0) -> Signal:
    """Construct a minimal actionable signal."""
    s = MagicMock(spec=Signal)
    s.is_actionable = True
    s.direction = 1
    s.straddle = MagicMock(spec=StraddleSelection)
    s.straddle.strike = strike
    s.straddle.expiry_dt = pd.Timestamp("2026-05-21", tz="UTC")
    s.straddle.index_price = strike
    return s


def _make_limits(*, max_single_trade_btc: float) -> RiskLimits:
    """Construct RiskLimits with paper-default trackers.

    ``max_margin_util_hard`` set absurdly high so the per-trade-size
    cap behaviour can be tested in isolation without margin gating
    interfering. Margin-util semantics are covered separately.
    """
    return RiskLimits(
        position_tracker=PositionTracker(),
        equity_tracker=EquityTracker(initial_equity_usd=100_000.0),
        max_open_positions=5,
        max_margin_util_hard=10.0,  # absurdly high; isolating the size-cap test
        max_single_trade_btc=max_single_trade_btc,
        min_single_trade_btc=0.1,
        dte_min_days=1.0,
        dte_max_days=30.0,
        im_fraction=0.20,
    )


def test_legacy_default_cap_at_2btc_blocks_3btc_trade():
    """Without V2, the 2 BTC default cap blocks oversized trades."""
    limits = _make_limits(max_single_trade_btc=2.0)
    sig = _make_signal()
    check = limits.check_signal(
        sig,
        proposed_n_contracts_btc=3.0,
        now=pd.Timestamp("2026-05-07 12:00", tz="UTC"),
    )
    assert not check.passed
    assert "max" in check.reason


def test_v2_cap_disabled_allows_oversize_trade():
    """D10: ``max_single_trade_btc=0.0`` disables the static cap;
    dynamic_btc_cap is the only ceiling under V2 preset.
    """
    limits = _make_limits(max_single_trade_btc=0.0)
    sig = _make_signal()
    check = limits.check_signal(
        sig,
        proposed_n_contracts_btc=10.0,
        now=pd.Timestamp("2026-05-07 12:00", tz="UTC"),
    )
    assert check.passed, f"V2 should allow >2 BTC trade; got {check.reason!r}"


def test_v2_cap_disabled_still_enforces_min():
    """The min_single_trade_btc gate stays in force even with cap off."""
    limits = _make_limits(max_single_trade_btc=0.0)
    sig = _make_signal()
    check = limits.check_signal(
        sig,
        proposed_n_contracts_btc=0.05,
        now=pd.Timestamp("2026-05-07 12:00", tz="UTC"),
    )
    assert not check.passed
    assert "min" in check.reason
