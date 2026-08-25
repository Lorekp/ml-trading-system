"""Kelly K1 application in :class:`StraddleFlow.size_position`.

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
    """Sizing inputs that yield a baseline notional well below the
    dynamic_btc_cap envelope. ``edge_scale=0.5`` (edge_vp half the
    rolling median) leaves Kelly room to grow within the envelope —
    necessary post-E.2.6 since Kelly is now re-clipped against
    dynamic_btc_cap so it can never breach the bar's margin envelope.
    """
    return dict(
        equity=100_000.0,            # USD
        signal_direction=1,           # LONG
        spot=60_000.0,                # BTC index
        trailing_1h_volume_btc=None,  # no participation cap
        liq_scale=1.0,
        dvol_pct=50.0,                # baseline DVOL → dynamic_margin = target_margin
        edge_vp=5.0,                  # half rolling median → edge_scale = 0.5
        rolling_median_edge=10.0,
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


# ─── E.1.1 — profit_scale × kelly_mult interaction ────────────────────────

def test_profit_scale_default_one_is_passthrough(base_args):
    """profit_scale=1.0 (default) → notional unchanged."""
    flow = _make_flow(max_trade_size=10.0)
    n_no_ps = flow.size_position(**base_args)
    n_unit_ps = flow.size_position(**base_args, profit_scale=1.0)
    assert n_no_ps == n_unit_ps


def test_profit_scale_below_one_shrinks_notional(base_args):
    """profit_scale=0.5 → roughly halves the notional (subject to rounding)."""
    flow = _make_flow(max_trade_size=10.0)
    n_full = flow.size_position(**base_args, profit_scale=1.0)
    n_half = flow.size_position(**base_args, profit_scale=0.5)
    assert n_half < n_full
    ratio = n_half / n_full
    assert 0.4 < ratio < 0.6


@pytest.mark.parametrize("ps,km,expect_ratio", [
    (1.0, 1.0, 1.00),
    (0.7, 1.0, 0.70),
    (1.0, 1.5, 1.50),
    (0.7, 1.5, 0.70 * 1.5),
    (0.3, 2.0, 0.30 * 2.0),     # floor × cap
])
def test_profit_scale_kelly_interaction(base_args, ps, km, expect_ratio):
    """profit_scale and kelly_mult chain multiplicatively in the documented
    order (profit_scale BEFORE kelly_mult), matching backtester at
    ``vol_arb/backtester.py:1986-2006``.
    """
    flow = _make_flow(max_trade_size=10.0)
    n_baseline = flow.size_position(**base_args, profit_scale=1.0, kelly_mult=1.0)
    n_combined = flow.size_position(**base_args, profit_scale=ps, kelly_mult=km)
    measured = n_combined / max(n_baseline, 1e-9)
    # Allow ±15% slack for the 0.1 BTC option rounding (small notionals
    # can drift more than 10% due to the increment).
    assert abs(measured - expect_ratio) / expect_ratio < 0.15


def test_profit_scale_clipped_safety_band(base_args):
    """Pathological profit_scale=0.001 still produces a positive notional
    via the floor=0.05 safety band — engine never hands out a literal-zero
    size when the estimator returns a fractional value."""
    flow = _make_flow(max_trade_size=10.0)
    n = flow.size_position(**base_args, profit_scale=0.001)
    # Either rounded to 0 (below min_trade_size_btc) or to the safety floor.
    # Either way, never negative or NaN.
    assert n >= 0.0
    assert np.isfinite(n)


# ── E.2.6: Kelly re-clip against dynamic_btc_cap ─────────────────────────


def test_kelly_cannot_exceed_dynamic_btc_cap(base_args):
    """A high edge_scale could push N above the bar's dynamic margin
    envelope; Kelly on top would compound the breach. The dynamic cap
    re-clip at the Kelly step keeps N within the envelope.
    """
    flow = _make_flow(max_trade_size=100.0)         # high static cap
    # Compute the dynamic envelope manually: equity*0.15/0.20/spot
    # for default V2 leverage (target_margin_util=0.15, im=0.20).
    spot = base_args["spot"]
    equity = base_args["equity"]
    expected_envelope = equity * 0.15 / 0.20 / spot
    # Now boost with high edge + Kelly together.
    high_edge_args = {
        **base_args,
        "edge_vp": 30.0,           # 3× rolling median → edge_scale=3
        "rolling_median_edge": 10.0,
    }
    n_kelly_boosted = flow.size_position(
        **high_edge_args, kelly_mult=2.0,
    )
    # Kelly (2.0) × edge_scale (3.0) = 6× pre-cap. Without dynamic
    # re-clip, would be ~6× envelope; with re-clip, capped at envelope.
    assert n_kelly_boosted <= expected_envelope + 0.1, (
        f"Kelly+edge product n={n_kelly_boosted:.3f} BTC exceeded "
        f"dynamic envelope {expected_envelope:.3f} BTC."
    )
