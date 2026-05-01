"""Tests for ``deribit_paper/execution/profit_scale.py``.

E.1.1 in plan ``C:\\Users\\laure\\.claude\\plans\\starry-rolling-rabin.md``.

Three contracts under test:

1. **Cold-start prior**: with no realised trades, profit_scale is
   driven by the theoretical edge×vega prior; clipping at floor when
   net is negative.
2. **EMA convergence**: after enough realised gross PnL is folded in,
   the shrinkage λ approaches 1 and the estimator follows the realised
   ratio, not the prior.
3. **Defensive paths**: NaN inputs / zero N / zero spot all return
   floor without raising.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from deribit_paper.execution.profit_scale import (
    ProfitScaleEstimator,
    bsm_vega_per_vol_pt,
)


# ─── BS vega helper ────────────────────────────────────────────────────────

def test_bsm_vega_atm_one_year_60pct_vol_known_value():
    """Sanity: ATM 1y BSM vega per vol-pt at S=K=60k, σ=60% ≈ S × N(0) × √T × 0.01.

    With σ=0.6, T=1, K=S=60k:
        d1 = 0.5 × σ × √T = 0.30
        N(0.30) ≈ 0.3814 (standard normal pdf at 0.30)
        vega = 60000 × 0.3814 × 1 × 0.01 ≈ 228.85 USD/pp
    """
    v = bsm_vega_per_vol_pt(spot=60_000.0, strike=60_000.0,
                             iv_decimal=0.6, tau_years=1.0)
    # Exact: spot × pdf(d1) × sqrt(T) × 0.01.
    d1 = 0.5 * 0.6
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2 * math.pi)
    expected = 60_000.0 * pdf * 0.01
    assert abs(v - expected) < 0.5


def test_bsm_vega_zero_inputs_return_zero():
    assert bsm_vega_per_vol_pt(spot=0, strike=60000, iv_decimal=0.6, tau_years=1) == 0.0
    assert bsm_vega_per_vol_pt(spot=60000, strike=0, iv_decimal=0.6, tau_years=1) == 0.0
    assert bsm_vega_per_vol_pt(spot=60000, strike=60000, iv_decimal=0.0, tau_years=1) == 0.0
    assert bsm_vega_per_vol_pt(spot=60000, strike=60000, iv_decimal=0.6, tau_years=0) == 0.0


# ─── Cold-start estimator ──────────────────────────────────────────────────

def test_cold_start_uses_theoretical_prior():
    """No realised trades → λ = 0 → estimator depends purely on the
    theoretical edge×vega prior vs costs."""
    est = ProfitScaleEstimator()
    assert est.gross_count == 0
    assert est.shrinkage_lambda == 0.0

    # Strong edge with low costs → expect ratio close to cap.
    ps_strong = est.estimate(
        edge_vp=20.0,                # high conviction
        vega_per_vol_pt_usd=200.0,   # ~ATM 1m
        spot=60_000.0,
        jump_fraction=0.0,
        slip_usd_per_btc=12.0,
        n_btc=0.5,
    )
    assert ps_strong > 0.5
    assert ps_strong <= 1.0


def test_cold_start_low_edge_clips_to_floor():
    """Tiny edge can't beat costs → clip to floor."""
    est = ProfitScaleEstimator(floor=0.3)
    ps = est.estimate(
        edge_vp=0.1,                  # vanishing edge
        vega_per_vol_pt_usd=200.0,
        spot=60_000.0,
        jump_fraction=0.0,
        slip_usd_per_btc=12.0,
        n_btc=0.5,
    )
    assert ps == pytest.approx(0.3)


# ─── EMA convergence ──────────────────────────────────────────────────────

def test_ema_converges_with_realised_gross():
    """Folding in many positive trades pulls the gross EMA up toward the
    realised value and increases shrinkage λ → estimator follows
    realised, not theoretical."""
    est = ProfitScaleEstimator(ema_period_gross=50)
    # Synthetic realised: 1000 USD gross / 0.5 BTC per trade = 2000 USD/BTC
    for _ in range(300):
        est.update_realized(gross_pnl_usd=1000.0,
                              hedge_cost_usd=200.0, n_btc=0.5)
    assert est.gross_count == 300
    # 300 / (300 + 200) = 0.6 — shrinkage weight on realised.
    assert 0.55 < est.shrinkage_lambda < 0.65
    # EMA should be close to 2000 USD/BTC after 300 trades w/ period=50.
    assert 1900.0 < est.gross_per_btc_ema_usd < 2100.0


def test_floor_when_estimated_net_is_negative():
    """If realised gross has been net-negative, ratio <= 0 → floor."""
    est = ProfitScaleEstimator(floor=0.3, ema_period_gross=10, ema_period_hedge=10)
    # Realised: gross 100 USD / 0.5 BTC = 200 USD/BTC; costs ~ 2000+ USD/BTC.
    for _ in range(300):
        est.update_realized(gross_pnl_usd=100.0,
                              hedge_cost_usd=600.0, n_btc=0.5)
    # Now estimate — costs (slip + hedge + opt fee) >> gross prior → floor.
    ps = est.estimate(
        edge_vp=5.0, vega_per_vol_pt_usd=200.0,
        spot=60_000.0, jump_fraction=0.0,
        slip_usd_per_btc=50.0, n_btc=0.5,
    )
    assert ps == pytest.approx(0.3)


# ─── Defensive paths ──────────────────────────────────────────────────────

def test_zero_n_returns_floor():
    est = ProfitScaleEstimator()
    assert est.estimate(
        edge_vp=20.0, vega_per_vol_pt_usd=200.0,
        spot=60_000.0, jump_fraction=0.0,
        slip_usd_per_btc=12.0, n_btc=0.0,
    ) == est._floor


def test_zero_spot_returns_floor():
    est = ProfitScaleEstimator()
    assert est.estimate(
        edge_vp=20.0, vega_per_vol_pt_usd=200.0,
        spot=0.0, jump_fraction=0.0,
        slip_usd_per_btc=12.0, n_btc=0.5,
    ) == est._floor


def test_nan_edge_returns_floor():
    est = ProfitScaleEstimator()
    assert est.estimate(
        edge_vp=float("nan"), vega_per_vol_pt_usd=200.0,
        spot=60_000.0, jump_fraction=0.0,
        slip_usd_per_btc=12.0, n_btc=0.5,
    ) == est._floor


def test_update_with_zero_n_is_noop():
    est = ProfitScaleEstimator()
    est.update_realized(gross_pnl_usd=1000.0, hedge_cost_usd=100.0, n_btc=0.0)
    assert est.gross_count == 0
    assert est.gross_per_btc_ema_usd == 0.0
    assert est.hedge_per_btc_ema_usd == 0.0
