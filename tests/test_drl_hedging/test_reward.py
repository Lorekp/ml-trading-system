"""Unit tests for the accounting-PnL reward and BSM pricing helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from vol_arb.drl_hedging.reward import (
    accounting_pnl_reward,
    bsm_price_from_iv,
    residual_delta_reward,
    straddle_delta_from_iv,
    straddle_gamma_from_iv,
    straddle_price_from_iv,
)


# ─── BSM pricing ─────────────────────────────────────────────────────────

def test_call_matches_known_atm_value():
    """ATM call at T=0.25, sigma=0.20: closed-form check.

    For S=K=100, r=0, T=0.25, sigma=0.20:
    d1 = 0.5 * 0.20 * sqrt(0.25) / (0.20 * sqrt(0.25)) = 0.05
    C = 100 * N(0.05) - 100 * N(-0.05) ≈ 3.988
    """
    price = bsm_price_from_iv(S=100.0, K=100.0, T=0.25, sigma=0.20, is_call=True)
    assert 3.9 < price < 4.1


def test_put_call_parity_at_the_money():
    S, K, T, sigma = 100.0, 100.0, 0.5, 0.5
    call = bsm_price_from_iv(S=S, K=K, T=T, sigma=sigma, is_call=True)
    put  = bsm_price_from_iv(S=S, K=K, T=T, sigma=sigma, is_call=False)
    # r=0: C - P = S - K
    assert abs((call - put) - (S - K)) < 1e-10


def test_intrinsic_at_expiry():
    assert bsm_price_from_iv(S=100, K=90, T=0.0, sigma=0.5, is_call=True) == 10.0
    assert bsm_price_from_iv(S=100, K=90, T=0.0, sigma=0.5, is_call=False) == 0.0
    assert bsm_price_from_iv(S=80, K=90, T=0.0, sigma=0.5, is_call=False) == 10.0


def test_zero_vol_returns_intrinsic():
    assert bsm_price_from_iv(S=100, K=110, T=1.0, sigma=0.0, is_call=True) == 0.0


def test_straddle_price_equals_call_plus_put():
    S, K, T, sigma = 50_000, 50_000, 0.02, 0.8
    straddle = straddle_price_from_iv(S=S, K=K, T=T, sigma=sigma)
    call = bsm_price_from_iv(S=S, K=K, T=T, sigma=sigma, is_call=True)
    put  = bsm_price_from_iv(S=S, K=K, T=T, sigma=sigma, is_call=False)
    assert abs(straddle - (call + put)) < 1e-10


def test_atm_straddle_delta_near_zero():
    """Straddle delta at S=K, r=0 should be approximately zero."""
    d = straddle_delta_from_iv(S=100, K=100, T=0.25, sigma=0.20)
    assert abs(d) < 0.05


def test_itm_call_delta_bounds():
    d = straddle_delta_from_iv(S=200, K=100, T=0.25, sigma=0.20)
    assert 0.5 < d <= 1.0


def test_gamma_positive_atm():
    g = straddle_gamma_from_iv(S=100, K=100, T=0.25, sigma=0.20)
    assert g > 0


# ─── Reward ──────────────────────────────────────────────────────────────

def test_accounting_reward_components_sum_correctly():
    R = accounting_pnl_reward(
        V_t=-100.0,
        V_t_minus_1=-105.0,
        H_t_minus_1=0.5,
        P_t=50_100.0,
        P_t_minus_1=50_000.0,
        cost=1.0,
        reward_scale=1.0,   # unscaled for test arithmetic
    )
    # option pnl: (-100) - (-105) = 5
    # hedge pnl: 0.5 * (50100 - 50000) = 50
    # cost: 1
    # R = 5 + 50 - 1 = 54
    assert abs(R - 54.0) < 1e-12


def test_accounting_reward_no_cost_no_hedge_equals_option_pnl():
    R = accounting_pnl_reward(
        V_t=50.0, V_t_minus_1=40.0,
        H_t_minus_1=0.0, P_t=100.0, P_t_minus_1=100.0,
        cost=0.0, reward_scale=1.0,
    )
    assert R == 10.0


def test_reward_variance_penalty_reduces_large_moves():
    """With lambda_reward_var > 0, larger |R| should be penalised more."""
    # Base reward = 100 (big swing)
    big = accounting_pnl_reward(
        V_t=100.0, V_t_minus_1=0.0,
        H_t_minus_1=0.0, P_t=0.0, P_t_minus_1=0.0,
        cost=0.0, reward_scale=1.0, lambda_reward_var=0.01,
    )
    # 100 - 0.01 * 100^2 = 100 - 100 = 0
    assert abs(big - 0.0) < 1e-10

    small = accounting_pnl_reward(
        V_t=10.0, V_t_minus_1=0.0,
        H_t_minus_1=0.0, P_t=0.0, P_t_minus_1=0.0,
        cost=0.0, reward_scale=1.0, lambda_reward_var=0.01,
    )
    # 10 - 0.01 * 10^2 = 10 - 1 = 9
    assert abs(small - 9.0) < 1e-10


def test_accounting_reward_scales():
    R_unscaled = accounting_pnl_reward(
        V_t=0.0, V_t_minus_1=-100.0, H_t_minus_1=0.0,
        P_t=1.0, P_t_minus_1=1.0, cost=0.0, reward_scale=1.0,
    )
    R_scaled = accounting_pnl_reward(
        V_t=0.0, V_t_minus_1=-100.0, H_t_minus_1=0.0,
        P_t=1.0, P_t_minus_1=1.0, cost=0.0, reward_scale=100.0,
    )
    assert abs(R_unscaled - 100.0) < 1e-10
    assert abs(R_scaled - 1.0) < 1e-10


# ─── Residual-delta reward (Deep Hedging objective) ─────────────────────

def test_residual_reward_perfect_hedge_pays_only_cost():
    """residual = 0 → R = -cost / scale"""
    R = residual_delta_reward(
        delta_portfolio_t=-0.3,
        H_t_minus_1=0.3,     # exactly offsets
        P_t=50_100.0,
        P_t_minus_1=50_000.0,
        cost=5.0,
        reward_scale=1.0,
    )
    assert abs(R - (-5.0)) < 1e-10


def test_residual_reward_unhedged_exposure_drives_pnl():
    """residual = 0.5, dS = 100 → R = 50 - cost."""
    R = residual_delta_reward(
        delta_portfolio_t=0.5,   # long 0.5 delta exposure
        H_t_minus_1=0.0,          # no hedge
        P_t=50_100.0,
        P_t_minus_1=50_000.0,
        cost=0.0,
        reward_scale=1.0,
    )
    assert abs(R - 50.0) < 1e-10


def test_residual_reward_uncontrollable_pnl_not_in_reward():
    """Unlike accounting_pnl_reward, the residual reward ignores
    V_t - V_{t-1}.  Two trades with identical residual exposure and
    identical cost should give identical rewards regardless of option
    value drift.
    """
    R1 = residual_delta_reward(
        delta_portfolio_t=-0.1, H_t_minus_1=0.1,  # residual = 0
        P_t=50_100.0, P_t_minus_1=50_000.0,
        cost=2.0, reward_scale=1.0,
    )
    R2 = residual_delta_reward(
        delta_portfolio_t=-0.1, H_t_minus_1=0.1,  # residual = 0
        P_t=50_100.0, P_t_minus_1=50_000.0,
        cost=2.0, reward_scale=1.0,
    )
    assert R1 == R2 == -2.0  # only cost matters when hedged


def test_residual_reward_scale_halves_reward():
    R_raw = residual_delta_reward(
        delta_portfolio_t=0.5, H_t_minus_1=0.0,
        P_t=50_100.0, P_t_minus_1=50_000.0,
        cost=0.0, reward_scale=1.0,
    )
    R_scaled = residual_delta_reward(
        delta_portfolio_t=0.5, H_t_minus_1=0.0,
        P_t=50_100.0, P_t_minus_1=50_000.0,
        cost=0.0, reward_scale=100.0,
    )
    assert abs(R_raw - 50.0) < 1e-10
    assert abs(R_scaled - 0.5) < 1e-10


def test_reward_returns_python_float():
    R = residual_delta_reward(
        delta_portfolio_t=0.0, H_t_minus_1=0.0,
        P_t=1.0, P_t_minus_1=1.0, cost=0.0, reward_scale=1.0,
    )
    assert isinstance(R, float)
