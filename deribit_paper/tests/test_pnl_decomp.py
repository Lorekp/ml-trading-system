"""E4 (parity-cycle round 2 — 2026-05-07): BS-decomposition gross_pnl helper.

Verifies that ``decompose_gross_pnl`` produces V2-shaped vega + theta +
delta breakdown so ``ProfitScaleEstimator.update_realized`` is fed the
same per-trade prior the V2 backtester computes (and not actual fills,
which double-subtract slippage with ``est_costs``).

Mirrors V2 ``vol_arb/backtester.py:1555-1559, 2303-2305``:

    vega_pnl_btc  = signal * vega_s     * dIV
    theta_pnl_btc = signal * theta_s    * dt_days
    delta_pnl_btc = simulate(...)

Ground-truth tests target the BS Greek formulas + the multiplicative
shape (signal-direction sign-flip, dIV scaling, n_btc scaling).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from deribit_paper.execution.pnl_decomp import (
    GrossPnLDecomp,
    bsm_theta_per_day,
    decompose_gross_pnl,
)
from deribit_paper.execution.profit_scale import bsm_vega_per_vol_pt


# ── bsm_theta_per_day ────────────────────────────────────────────────


def test_theta_at_atm_is_negative_for_long_option():
    """ATM theta is negative — long option decays."""
    t = bsm_theta_per_day(spot=50_000, strike=50_000, iv_decimal=0.75, tau_years=0.04)
    assert t < 0.0


def test_theta_zero_on_degenerate_inputs():
    assert bsm_theta_per_day(spot=0.0, strike=50_000, iv_decimal=0.75, tau_years=0.04) == 0.0
    assert bsm_theta_per_day(spot=50_000, strike=0.0, iv_decimal=0.75, tau_years=0.04) == 0.0
    assert bsm_theta_per_day(spot=50_000, strike=50_000, iv_decimal=0.0, tau_years=0.04) == 0.0
    assert bsm_theta_per_day(spot=50_000, strike=50_000, iv_decimal=0.75, tau_years=0.0) == 0.0


def test_theta_scales_with_sqrt_inverse_tau():
    """Shorter time-to-expiry → faster decay (more negative theta)."""
    t_long = bsm_theta_per_day(spot=50_000, strike=50_000, iv_decimal=0.75, tau_years=0.10)
    t_short = bsm_theta_per_day(spot=50_000, strike=50_000, iv_decimal=0.75, tau_years=0.01)
    assert t_short < t_long  # both negative; t_short more negative


# ── decompose_gross_pnl: signal direction + sizing ───────────────────


def test_decomp_zero_size_returns_zero():
    out = decompose_gross_pnl(
        signal_direction=1,
        entry_index_price=50_000,
        strike=50_000,
        entry_iv_decimal=0.75,
        exit_iv_decimal=0.80,
        entry_tau_years=0.04,
        holding_dt_days=0.0104,  # 15 min in days
        n_btc=0.0,
        realized_hedge_pnl_usd=10.0,
    )
    assert out.vega_pnl_usd == 0.0
    assert out.theta_pnl_usd == 0.0
    assert out.delta_pnl_usd == 0.0


def test_decomp_zero_signal_returns_zero():
    out = decompose_gross_pnl(
        signal_direction=0,
        entry_index_price=50_000,
        strike=50_000,
        entry_iv_decimal=0.75,
        exit_iv_decimal=0.80,
        entry_tau_years=0.04,
        holding_dt_days=0.0104,
        n_btc=1.0,
        realized_hedge_pnl_usd=10.0,
    )
    assert out.total_usd == 0.0


def test_decomp_short_straddle_iv_up_loses_on_vega():
    """Short straddle + IV up → vega leg negative (loses)."""
    out = decompose_gross_pnl(
        signal_direction=-1,
        entry_index_price=50_000,
        strike=50_000,
        entry_iv_decimal=0.75,
        exit_iv_decimal=0.80,  # +5 vol pts
        entry_tau_years=0.04,
        holding_dt_days=0.0104,
        n_btc=1.0,
        realized_hedge_pnl_usd=0.0,
    )
    assert out.vega_pnl_usd < 0.0


def test_decomp_long_straddle_iv_up_gains_on_vega():
    """Long straddle + IV up → vega leg positive."""
    out = decompose_gross_pnl(
        signal_direction=1,
        entry_index_price=50_000,
        strike=50_000,
        entry_iv_decimal=0.75,
        exit_iv_decimal=0.80,
        entry_tau_years=0.04,
        holding_dt_days=0.0104,
        n_btc=1.0,
        realized_hedge_pnl_usd=0.0,
    )
    assert out.vega_pnl_usd > 0.0


def test_decomp_long_straddle_theta_negative():
    """Long straddle decays — theta leg should be negative."""
    out = decompose_gross_pnl(
        signal_direction=1,
        entry_index_price=50_000,
        strike=50_000,
        entry_iv_decimal=0.75,
        exit_iv_decimal=0.75,
        entry_tau_years=0.04,
        holding_dt_days=0.0104,
        n_btc=1.0,
        realized_hedge_pnl_usd=0.0,
    )
    assert out.theta_pnl_usd < 0.0


def test_decomp_short_straddle_theta_positive():
    """Short straddle benefits from decay — theta leg positive."""
    out = decompose_gross_pnl(
        signal_direction=-1,
        entry_index_price=50_000,
        strike=50_000,
        entry_iv_decimal=0.75,
        exit_iv_decimal=0.75,
        entry_tau_years=0.04,
        holding_dt_days=0.0104,
        n_btc=1.0,
        realized_hedge_pnl_usd=0.0,
    )
    assert out.theta_pnl_usd > 0.0


def test_decomp_delta_pnl_passes_through():
    """Delta leg = realized hedge PnL, unchanged by decomposition."""
    out = decompose_gross_pnl(
        signal_direction=1,
        entry_index_price=50_000,
        strike=50_000,
        entry_iv_decimal=0.75,
        exit_iv_decimal=0.75,
        entry_tau_years=0.04,
        holding_dt_days=0.0,
        n_btc=1.0,
        realized_hedge_pnl_usd=123.45,
    )
    assert out.delta_pnl_usd == pytest.approx(123.45)


def test_decomp_total_usd_is_sum():
    out = GrossPnLDecomp(vega_pnl_usd=10.0, theta_pnl_usd=-2.0, delta_pnl_usd=3.0)
    assert out.total_usd == pytest.approx(11.0)


def test_decomp_vega_scales_linearly_with_n_btc():
    out_1 = decompose_gross_pnl(
        signal_direction=1, entry_index_price=50_000, strike=50_000,
        entry_iv_decimal=0.75, exit_iv_decimal=0.80, entry_tau_years=0.04,
        holding_dt_days=0.01, n_btc=1.0, realized_hedge_pnl_usd=0.0,
    )
    out_5 = decompose_gross_pnl(
        signal_direction=1, entry_index_price=50_000, strike=50_000,
        entry_iv_decimal=0.75, exit_iv_decimal=0.80, entry_tau_years=0.04,
        holding_dt_days=0.01, n_btc=5.0, realized_hedge_pnl_usd=0.0,
    )
    assert out_5.vega_pnl_usd == pytest.approx(5.0 * out_1.vega_pnl_usd)
    assert out_5.theta_pnl_usd == pytest.approx(5.0 * out_1.theta_pnl_usd)


def test_decomp_nonfinite_exit_iv_drops_vega_leg():
    """NaN exit IV → vega leg is zero, theta + delta still computed."""
    out = decompose_gross_pnl(
        signal_direction=1,
        entry_index_price=50_000, strike=50_000,
        entry_iv_decimal=0.75,
        exit_iv_decimal=float("nan"),
        entry_tau_years=0.04,
        holding_dt_days=0.01,
        n_btc=1.0,
        realized_hedge_pnl_usd=42.0,
    )
    assert out.vega_pnl_usd == 0.0
    assert out.theta_pnl_usd != 0.0
    assert out.delta_pnl_usd == pytest.approx(42.0)


def test_decomp_vega_matches_bs_greek_at_atm():
    """Vega leg = signal * 2 * vega_per_pt * dIV_pts * n_btc — verify the formula."""
    spot = 50_000.0
    K = 50_000.0
    iv_entry = 0.75
    iv_exit = 0.80
    tau = 0.04
    n_btc = 2.5
    expected_vega_per_pt = bsm_vega_per_vol_pt(
        spot=spot, strike=K, iv_decimal=iv_entry, tau_years=tau,
    )
    expected_d_iv_pts = (iv_exit - iv_entry) * 100.0
    expected_vega_pnl = 1 * 2.0 * expected_vega_per_pt * expected_d_iv_pts * n_btc
    out = decompose_gross_pnl(
        signal_direction=1, entry_index_price=spot, strike=K,
        entry_iv_decimal=iv_entry, exit_iv_decimal=iv_exit,
        entry_tau_years=tau, holding_dt_days=0.0, n_btc=n_btc,
        realized_hedge_pnl_usd=0.0,
    )
    assert out.vega_pnl_usd == pytest.approx(expected_vega_pnl, rel=1e-9)


def test_decomp_nonfinite_hedge_pnl_treated_as_zero():
    out = decompose_gross_pnl(
        signal_direction=1, entry_index_price=50_000, strike=50_000,
        entry_iv_decimal=0.75, exit_iv_decimal=0.75, entry_tau_years=0.04,
        holding_dt_days=0.01, n_btc=1.0,
        realized_hedge_pnl_usd=float("inf"),
    )
    assert out.delta_pnl_usd == 0.0
