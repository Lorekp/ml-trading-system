"""Step rewards for the DRL hedging environment.

Two reward modes are supported:

1. ``"residual"`` (DEFAULT, RECOMMENDED) — the classic Deep Hedging
   objective from Buehler et al. (2019):

       R_t = residual * (P_t - P_{t-1}) - C(A_t)

   where ``residual = portfolio_delta + hedge_inventory`` is the total
   unhedged delta exposure.  This form **removes the uncontrollable
   option P&L** (theta, gamma, option mark drift) that the agent cannot
   influence in the short term, isolating the hedging signal.  In the
   limit of perfect hedging (``residual = 0``) the reward is just
   ``-cost``.  The agent's learning signal is no longer drowned out by
   noise it cannot control.

2. ``"accounting"`` — the textbook mark-to-market formulation:

       R_t = (V_t - V_{t-1}) + H_{t-1} * (P_t - P_{t-1}) - C(A_t)

   Included for benchmarking and because it's the "natural" P&L
   definition.  NOT used for training because the option P&L component
   (V_t - V_{t-1}) dominates the learning gradient with noise that is
   orthogonal to the agent's actions.

The residual form is the standard in the Deep Hedging literature because
it matches the stochastic optimal control formulation: the agent
minimises ``E[residual² * sigma² * dt + cost]`` across the holding
period, which is exactly what SAC learns when given the residual reward.

Reward scaling
--------------
A ``reward_scale`` factor divides both reward forms.  At default 100,
typical straddle P&L per tick of $10-$1000 maps to O(0.1-10) rewards,
keeping squared rewards in the O(1-100) range.  This is critical for
the :class:`RiskAverseSAC` dual critic — without it, ``critic_sq_loss``
runs in the 10⁶-10⁸ range and the variance penalty drowns the actor
update.

A companion :func:`bsm_price_from_iv` wraps Black-Scholes for the
SVI-interpolated IV path (used when ``reward_mode == "accounting"``).
"""

from __future__ import annotations

from math import erf, exp, log, sqrt

import numpy as np

from vol_arb.drl_hedging.config import LAMBDA_REWARD_VAR


# ─────────────────────────────────────────────────────────────────────────
# Reward
# ─────────────────────────────────────────────────────────────────────────

def residual_delta_reward(
    *,
    delta_portfolio_t: float,
    H_t_minus_1: float,
    P_t: float,
    P_t_minus_1: float,
    cost: float,
    reward_scale: float = 100.0,
    lambda_reward_var: float = LAMBDA_REWARD_VAR,
) -> float:
    """Residual-delta reward — the Deep Hedging objective.

    R = residual * (P_t - P_{t-1}) - cost
    residual = delta_portfolio_t + H_{t-1}

    This is the standard formulation in the Deep Hedging literature.
    It isolates the hedging signal by removing uncontrollable option
    P&L components (theta, gamma drift, option mark changes).

    Parameters
    ----------
    delta_portfolio_t : float
        Current portfolio delta (sign-aware: positive for long, negative
        for short straddle).  At ATM this is ≈ 0.
    H_t_minus_1 : float
        Hedge inventory carried into tick ``t`` (BTC on perps, signed).
    P_t, P_t_minus_1 : float
        Underlying price before and after the tick.
    cost : float
        Round-trip execution cost of the chosen action (USD).
    reward_scale : float, default 100.0
        Divides the raw reward to bring it to O(1) for training stability.
        See module docstring for why this matters for RiskAverseSAC.
    lambda_reward_var : float, default 0.0
        Optional squared-reward penalty — leave at 0 to use the dual
        critic in :class:`RiskAverseSAC` instead.

    Returns
    -------
    float
        The scaled step reward.  In perfect-hedging regime
        (``residual = 0``) this equals ``-cost / reward_scale``.
    """
    residual = delta_portfolio_t + H_t_minus_1
    raw_reward = residual * (P_t - P_t_minus_1) - cost

    if lambda_reward_var > 0.0:
        raw_reward = raw_reward - lambda_reward_var * raw_reward * raw_reward

    return float(raw_reward / reward_scale)


def accounting_pnl_reward(
    *,
    V_t: float,
    V_t_minus_1: float,
    H_t_minus_1: float,
    P_t: float,
    P_t_minus_1: float,
    cost: float,
    reward_scale: float = 100.0,
    lambda_reward_var: float = LAMBDA_REWARD_VAR,
) -> float:
    """Accounting-PnL reward (benchmark only, NOT recommended for training).

    R = (V_t - V_{t-1}) + H_{t-1}*(P_t - P_{t-1}) - cost

    Included for comparison with :func:`residual_delta_reward`.  Dominated
    by uncontrollable option P&L in practice, which drowns the hedging
    signal during SAC training.
    """
    option_pnl = V_t - V_t_minus_1
    hedge_pnl  = H_t_minus_1 * (P_t - P_t_minus_1)
    step_pnl   = option_pnl + hedge_pnl - cost

    if lambda_reward_var > 0.0:
        step_pnl = step_pnl - lambda_reward_var * step_pnl * step_pnl

    return float(step_pnl / reward_scale)


# ─────────────────────────────────────────────────────────────────────────
# Black-Scholes pricing (for repricing the option at each tick)
# ─────────────────────────────────────────────────────────────────────────

def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erf — scalar, no scipy dep."""
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def bsm_price_from_iv(
    *,
    S: float,
    K: float,
    T: float,
    sigma: float,
    is_call: bool,
    r: float = 0.0,
) -> float:
    """Return the Black-Scholes price given an implied vol.

    Matches the convention used by ``vol_arb.greeks.compute_greeks``:
    zero rates (crypto), ``sigma`` as a decimal (e.g. 0.80 = 80%).

    For T <= 0 or sigma <= 0 we return the intrinsic value.
    """
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        intrinsic = max(0.0, S - K) if is_call else max(0.0, K - S)
        return float(intrinsic)

    sqrt_T = sqrt(T)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    if is_call:
        return S * _norm_cdf(d1) - K * exp(-r * T) * _norm_cdf(d2)
    return K * exp(-r * T) * _norm_cdf(-d2) - S * _norm_cdf(-d1)


def straddle_price_from_iv(
    *,
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
) -> float:
    """ATM straddle = call(K) + put(K).  Convenience wrapper."""
    call = bsm_price_from_iv(S=S, K=K, T=T, sigma=sigma, is_call=True, r=r)
    put  = bsm_price_from_iv(S=S, K=K, T=T, sigma=sigma, is_call=False, r=r)
    return call + put


def straddle_delta_from_iv(
    *,
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
) -> float:
    """ATM straddle delta = delta_call + delta_put = N(d1) - N(-d1) = 2N(d1) - 1.

    Needed by the env to compute ``delta_theoretical`` at each tick without
    bouncing out to ``vol_arb.greeks.compute_greeks`` (which is vectorised
    for DataFrames and adds overhead for scalar calls).
    """
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        if S > K:
            return 1.0  # deep ITM call + deep OTM put
        if S < K:
            return -1.0
        return 0.0
    sqrt_T = sqrt(T)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    return 2.0 * _norm_cdf(d1) - 1.0


def straddle_gamma_from_iv(
    *,
    S: float,
    K: float,
    T: float,
    sigma: float,
    r: float = 0.0,
) -> float:
    """ATM straddle gamma = 2 * gamma_call (call and put have same gamma)."""
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return 0.0
    sqrt_T = sqrt(T)
    d1 = (log(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    pdf_d1 = exp(-0.5 * d1 * d1) / sqrt(2.0 * np.pi)
    return 2.0 * pdf_d1 / (S * sigma * sqrt_T)
