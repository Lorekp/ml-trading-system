"""Kelly K1 application in :class:`StraddleFlow.size_position`.

C3 in plan ``C:\\Users\\laure\\.claude\\plans\\live-engine-h15-alignment-2026-04-26\\``.

Three contracts under test:

1. **Default kelly_mult=1.0**: behaviour byte-id with the pre-C3 chain.
2. **Boost**: kelly_mult > 1.0 multiplies the post-edge/dd/liq notional
   before the hard cap and before option-amount rounding.
3. **Cap interaction**: the hard cap (max_trade_size_btc) clips a
   Kelly-boosted notional so risk envelope stays bounded.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from deribit_paper.execution.straddle_flow import StraddleFlow


def _make_flow(*, max_trade_size: float = 2.0) -> StraddleFlow:
    """StraddleFlow with mocked downstream services. We only exercise
    size_position so the order_manager and DRL hedger are stubs."""
    return StraddleFlow(
        order_manager=MagicMock(),
        live_drl_hedger=MagicMock(),
        target_margin_utilization=0.15,
        im_fraction=0.20,
        short_sizing_penalty=0.50,
        max_trade_size_btc=max_trade_size,
        min_trade_size_btc=0.1,
        baseline_target_dvol=50.0,
        sizing_edge_cap=3.0,
        sizing_dd_floor=0.25,
    )


@pytest.fixture
def base_args() -> dict:
    """Sizing inputs that yield a baseline notional well below the hard cap."""
    return dict(
        equity=100_000.0,            # USD
        signal_direction=1,           # LONG
        spot=60_000.0,                # BTC index
        trailing_1h_volume_btc=None,  # no participation cap
        liq_scale=1.0,
        dvol_pct=50.0,                # baseline DVOL → dynamic_margin = target_margin
        edge_vp=10.0,
        rolling_median_edge=10.0,     # edge_scale = 1.0
        peak_equity=100_000.0,        # no DD
    )


def test_kelly_mult_default_one_is_passthrough(base_args):
    """kelly_mult=1.0 (default) → notional unchanged vs no-Kelly call."""
    flow = _make_flow()
    n_no_kelly = flow.size_position(**base_args)
    n_default_kelly = flow.size_position(**base_args, kelly_mult=1.0)
    assert n_no_kelly == n_default_kelly


def test_kelly_mult_boost_grows_notional(base_args):
    """kelly_mult=1.5 → notional ~1.5x the baseline (subject to rounding)."""
    flow = _make_flow(max_trade_size=10.0)  # raise cap so boost isn't clipped
    n_baseline = flow.size_position(**base_args)
    n_boosted = flow.size_position(**base_args, kelly_mult=1.5)
    assert n_boosted > n_baseline
    # Ratio should be near 1.5 within rounding tolerance (option_amount_btc
    # quantises to 0.1 BTC increments).
    ratio = n_boosted / n_baseline
    assert 1.4 < ratio < 1.6


def test_kelly_mult_clipped_by_hard_cap(base_args):
    """kelly_mult that would push above max_trade_size_btc clips at cap."""
    flow = _make_flow(max_trade_size=0.3)  # tiny cap
    n_at_cap = flow.size_position(**base_args, kelly_mult=2.0)
    # The pre-Kelly notional may itself exceed 0.3, so just assert ≤ cap.
    assert n_at_cap <= 0.3 + 1e-9


def test_kelly_mult_below_floor_is_passthrough(base_args):
    """Defensive: a malformed sidecar that returns 0.99 should not shrink size.

    The sidecars.lookup_kelly_mult_at clips to [1.0, 2.0] before passing
    here, so this is the engine-level safety belt — even if the caller
    forgot to clip, the size_position application short-circuits at
    kelly_mult <= 1.0 + 1e-12.
    """
    flow = _make_flow(max_trade_size=10.0)
    n_baseline = flow.size_position(**base_args, kelly_mult=1.0)
    n_below = flow.size_position(**base_args, kelly_mult=0.99)
    assert n_below == n_baseline


def test_kelly_mult_short_passes_when_caller_sends_one(base_args):
    """Per the V2 contract, shorts ALWAYS get kelly_mult=1.0 (the sidecar
    enforces this). The engine-level lookup returns 1.0 for shorts, so
    size_position never sees a >1 mult on short signals; mirror that.
    """
    flow = _make_flow(max_trade_size=10.0)
    short_args = {**base_args, "signal_direction": -1}
    n_baseline = flow.size_position(**short_args, kelly_mult=1.0)
    # If a buggy caller forgot the asymmetry and sent 1.5 for a short,
    # the multiplier would still apply at size_position level — we
    # deliberately don't double-guard this in size_position because
    # forcing the asymmetry there would couple two layers; the lookup
    # layer is the single source of truth. Document the behaviour:
    n_buggy = flow.size_position(**short_args, kelly_mult=1.5)
    assert n_buggy > n_baseline  # confirms responsibility lives upstream
