"""Tests for liquidity-conditioned sizing strategies (S6-S8)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from vol_arb.sizing import (
    compute_position_size,
    size_composite_v2,
    size_cost_edge,
    size_liquidity_conditioned,
)

# Minimal cfg-like object with all required fields
_CFG = SimpleNamespace(
    im_fraction=0.10,
    short_sizing_penalty=0.50,
    sizing_edge_cap=3.0,
    sizing_dd_floor=0.25,
    sizing_risk_fraction=0.005,
    sizing_kelly_fraction=0.25,
    sizing_liq_cap=2.0,
)

_BASE_KW = dict(
    equity=1_000_000.0,
    spot=60_000.0,
    signal=1,
    dynamic_margin=0.05,
    dynamic_btc_cap=50.0,
    cfg=_CFG,
)


# ─── S6: Liquidity-conditioned ─────────────────────────────────

def test_liq_scale_inverse():
    """When slip is 2x EMA, size should be ~0.5x base."""
    base = size_liquidity_conditioned(**_BASE_KW, bar_taker_slip=24.0, rolling_slip_ema=12.0)
    full = size_liquidity_conditioned(**_BASE_KW, bar_taker_slip=12.0, rolling_slip_ema=12.0)
    assert base == pytest.approx(full * 0.5, rel=0.01)


def test_liq_scale_cheap():
    """When slip is 0.5x EMA, size should be 2x base (capped)."""
    cheap = size_liquidity_conditioned(**_BASE_KW, bar_taker_slip=6.0, rolling_slip_ema=12.0)
    full = size_liquidity_conditioned(**_BASE_KW, bar_taker_slip=12.0, rolling_slip_ema=12.0)
    assert cheap == pytest.approx(full * 2.0, rel=0.01)


def test_liq_floor():
    """When slip is 10x EMA, size should be 0.25x base (floor)."""
    expensive = size_liquidity_conditioned(**_BASE_KW, bar_taker_slip=120.0, rolling_slip_ema=12.0)
    full = size_liquidity_conditioned(**_BASE_KW, bar_taker_slip=12.0, rolling_slip_ema=12.0)
    assert expensive == pytest.approx(full * 0.25, rel=0.01)


def test_liq_zero_slip_fallback():
    """Zero or negative slip falls back to base."""
    n = size_liquidity_conditioned(**_BASE_KW, bar_taker_slip=0.0, rolling_slip_ema=12.0)
    base = size_liquidity_conditioned(**_BASE_KW, bar_taker_slip=12.0, rolling_slip_ema=12.0)
    assert n == pytest.approx(base, rel=0.01)


# ─── S7: Cost-edge ─────────────────────────────────────────────

def test_cost_edge_zero_vega():
    """When vega_per_pt is 0, fall back to base."""
    n = size_cost_edge(
        **_BASE_KW,
        edge_vp=50.0, rolling_median_edge=35.0,
        bar_taker_slip=12.0, vega_per_pt=0.0,
    )
    # Should equal base (no scaling applied)
    base = size_cost_edge(
        **_BASE_KW,
        edge_vp=50.0, rolling_median_edge=0.0,
        bar_taker_slip=12.0, vega_per_pt=0.001,
    )
    assert n == pytest.approx(base, rel=0.01)


def test_cost_edge_high_cost_reduces():
    """High execution cost should reduce size vs low cost."""
    low_cost = size_cost_edge(
        **_BASE_KW,
        edge_vp=50.0, rolling_median_edge=35.0,
        bar_taker_slip=5.0, vega_per_pt=0.001,
    )
    high_cost = size_cost_edge(
        **_BASE_KW,
        edge_vp=50.0, rolling_median_edge=35.0,
        bar_taker_slip=50.0, vega_per_pt=0.001,
    )
    assert high_cost < low_cost


# ─── S8: Composite v2 ──────────────────────────────────────────

def test_composite_v2_multiplicative():
    """Composite v2 = edge_scale * dd_scale * liq_scale."""
    # At median edge, no drawdown, median slip: scale = 1.0 * 1.0 * 1.0
    n_flat = size_composite_v2(
        **_BASE_KW,
        edge_vp=35.0, rolling_median_edge=35.0,
        peak_equity=1_000_000.0,
        bar_taker_slip=12.0, rolling_slip_ema=12.0,
    )
    # High edge, in drawdown, expensive slip
    n_mixed = size_composite_v2(
        **_BASE_KW,
        edge_vp=70.0, rolling_median_edge=35.0,  # 2x edge → 2.0 scale
        peak_equity=2_000_000.0,                   # 50% DD → 0.25 scale (floor)
        bar_taker_slip=24.0, rolling_slip_ema=12.0,  # 2x slip → 0.5 scale
    )
    # Expected: n_flat * 2.0 * 0.25 * 0.5 = n_flat * 0.25
    assert n_mixed == pytest.approx(n_flat * 0.25, rel=0.05)


# ─── Dispatch ──────────────────────────────────────────────────

def test_dispatch_liq():
    n = compute_position_size(strategy="liq", **_BASE_KW, bar_taker_slip=12.0, rolling_slip_ema=12.0)
    assert n > 0


def test_dispatch_cost_edge():
    n = compute_position_size(
        strategy="cost_edge", **_BASE_KW,
        edge_vp=50.0, rolling_median_edge=35.0,
        bar_taker_slip=12.0, vega_per_pt=0.001,
    )
    assert n > 0


def test_dispatch_composite_v2():
    n = compute_position_size(
        strategy="composite_v2", **_BASE_KW,
        edge_vp=50.0, rolling_median_edge=35.0,
        peak_equity=1_000_000.0,
        bar_taker_slip=12.0, rolling_slip_ema=12.0,
    )
    assert n > 0


# ─── No cherry-picking ────────────────────────────────────────

@pytest.mark.parametrize("strategy", ["liq", "cost_edge", "composite_v2"])
def test_no_cherry_picking(strategy):
    """All strategies must return > 0 when signal != 0 and equity > 0."""
    # Even with worst-case inputs: 10x slip, drawdown, low edge
    n = compute_position_size(
        strategy=strategy,
        equity=100_000.0, spot=60_000.0, signal=-1,
        dynamic_margin=0.05, dynamic_btc_cap=50.0,
        cfg=_CFG,
        edge_vp=5.0, rolling_median_edge=35.0,
        peak_equity=500_000.0,  # 80% drawdown
        bar_taker_slip=120.0, rolling_slip_ema=12.0,
        vega_per_pt=0.001,
        rolling_win_rate=0.3, rolling_payoff_ratio=0.5,
    )
    assert n > 0, f"Strategy {strategy} returned 0 — cherry-picking detected"
