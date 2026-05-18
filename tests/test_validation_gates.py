"""Tests for the Phase-5 validation gates (2026-05-18).

Synthetic cases pin the pass/fail logic. The real-data cases DEMONSTRATE
the gates would have caught the V2 fake-PnL artifact and the cycle-50
sizing artifact — they assert the gate FAILS on the known-bad inputs.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from validation.gates import (
    pnl_realism_gate,
    selection_bias_gate,
    sizing_sanity_gate,
)

V2_TRADE_LOG = Path("vol_arb/output/BTC_v2_trade_log.parquet")
CYCLE50_TRADES = Path("vol_arb/output/cycle50_iv_carry_trades.parquet")


# ── Gate 1: pnl_realism ────────────────────────────────────────────────
def test_pnl_realism_passes_when_model_matches_fills() -> None:
    rng = np.random.RandomState(0)
    fill = rng.normal(50, 200, 500)
    model = fill + rng.normal(0, 20, 500)  # small noise, same series
    res = pnl_realism_gate(model, fill)
    assert res.passed, res.message


def test_pnl_realism_fails_on_sign_flip() -> None:
    """The V2 signature: model says profit, reality says loss."""
    rng = np.random.RandomState(1)
    fill = rng.normal(-48, 300, 500)        # honest: net loss
    model = rng.normal(+133, 300, 500)      # model: net profit
    res = pnl_realism_gate(model, fill)
    assert not res.passed
    assert "SIGN FLIP" in res.message
    assert res.detail["sign_agree"] is False


def test_pnl_realism_fails_on_magnitude_divergence() -> None:
    rng = np.random.RandomState(2)
    fill = rng.normal(20, 100, 500)
    model = fill * 3.0  # same sign & shape, 3x magnitude
    res = pnl_realism_gate(model, fill)
    assert not res.passed
    assert "magnitude" in res.message


def test_pnl_realism_fails_on_low_correlation() -> None:
    rng = np.random.RandomState(3)
    fill = rng.normal(30, 100, 500)
    model = rng.normal(30, 100, 500)  # same distribution, uncorrelated
    res = pnl_realism_gate(model, fill)
    assert not res.passed
    assert "corr" in res.message


# ── Gate 2: sizing_sanity ──────────────────────────────────────────────
def test_sizing_sanity_passes_when_sign_stable() -> None:
    rng = np.random.RandomState(4)
    edge = rng.normal(2.0, 5.0, 400)          # net edge per unit, +ve mean
    exposure = rng.uniform(20, 60, 400)       # exposure uncorrelated w/ edge
    res = sizing_sanity_gate(edge, exposure)
    assert res.passed, res.message


def test_sizing_sanity_fails_on_sign_flip() -> None:
    """The cycle-50 signature: edge is +ve per-trade, but high-exposure
    trades carry the losses, so fixed-N (exposure-weighted) flips it."""
    rng = np.random.RandomState(5)
    n = 400
    edge = rng.normal(2.5, 6.0, n)            # +ve mean -> constant-exposure +ve
    # make exposure strongly anti-correlated with edge: worst trades biggest
    exposure = 60.0 - 8.0 * edge + rng.normal(0, 3, n)
    exposure = np.clip(exposure, 5.0, None)
    res = sizing_sanity_gate(edge, exposure)
    assert not res.passed
    assert "SIGN FLIP" in res.message


def test_sizing_sanity_rejects_nonpositive_exposure() -> None:
    res = sizing_sanity_gate(np.array([1.0, 2.0]), np.array([10.0, -1.0]))
    assert not res.passed


# ── Gate 3: selection_bias ─────────────────────────────────────────────
def test_selection_bias_passes_when_subset_representative() -> None:
    rng = np.random.RandomState(6)
    universe = rng.normal(1.5, 4.0, 5000)
    traded = rng.choice(universe, 300, replace=False)  # random subset
    res = selection_bias_gate(traded, universe)
    assert res.passed, res.message


def test_selection_bias_fails_when_subset_too_rich() -> None:
    """The cycle-50 signature: sampler over-selects rich bars."""
    rng = np.random.RandomState(7)
    universe = rng.normal(1.5, 4.0, 5000)
    traded = rng.normal(5.0, 4.0, 300)  # subset mean 5.0 vs universe 1.5
    res = selection_bias_gate(traded, universe)
    assert not res.passed
    assert "selection bias" in res.message


# ── Real-data demonstration: the gates catch the known artifacts ───────
@pytest.mark.skipif(not V2_TRADE_LOG.exists(),
                    reason="V2 trade log not present")
def test_pnl_realism_gate_catches_v2_artifact() -> None:
    """DEMONSTRATION: pnl_realism_gate FAILS on V2 — model `vega*d(iv)`
    PnL (+$1.6M) vs IV-frozen honest PnL (net_nc - vega_nc, ~-$587K)."""
    tl = pd.read_parquet(V2_TRADE_LOG)
    fired = tl[tl["gross_pnl"] != 0]
    model = pd.to_numeric(fired["net_pnl_nc"], errors="coerce")
    honest = model - pd.to_numeric(fired["vega_pnl_nc"], errors="coerce")
    res = pnl_realism_gate(model.to_numpy(), honest.to_numpy())
    assert not res.passed, "gate should FAIL on V2's fake PnL"
    assert res.detail["sign_agree"] is False
    # model is strongly positive, honest is strongly negative
    assert res.detail["total_model"] > 0 > res.detail["total_fill_marked"]


@pytest.mark.skipif(not CYCLE50_TRADES.exists(),
                    reason="cycle-50 trade log not present")
def test_sizing_sanity_gate_catches_cycle50_artifact() -> None:
    """DEMONSTRATION: sizing_sanity_gate FAILS on the cycle-50 IV-carry
    trades — net edge (VRP - cost) is +ve per-trade but fixed-N
    (vega-weighted) sizing flips the total negative."""
    tr = pd.read_parquet(CYCLE50_TRADES)
    cost_vp = 2.5
    edge = (pd.to_numeric(tr["vrp_vp"], errors="coerce") - cost_vp).to_numpy()
    vega = pd.to_numeric(tr["vega"], errors="coerce").to_numpy()
    res = sizing_sanity_gate(edge, vega)
    assert not res.passed, "gate should FAIL on cycle-50's sizing artifact"
    assert res.detail["sign_flip"] is True
