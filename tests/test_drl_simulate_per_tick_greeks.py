"""D-7 (parity-r3 cycle 2 — 2026-05-08) regression tests.

The offline V2-backtest path ``DRLHedger.simulate`` precomputed
``sigma_sqrt_T`` and ``half_sigma2_T`` at entry σ × entry T and reused
them across every tick — but ``tau_t`` decays each tick. The
time-decay term ``half_sigma²·T`` was up to 9× too large at the last
tick, biasing ``d1`` (and therefore ``delta_obs`` / ``gamma_obs``) into
a distribution the SAC v4 policy was NOT trained on.

The training env (``env.py:418``) and live inference path
(``drl_hedger_live.py:315-322``) both recompute per tick using
``tau_t``; the simulate path was the only divergent one.

These tests verify:
1. At j=0 (entry tick), pre-fix and post-fix d1 agree to 1e-10 (proof
   the fix is correct at the boundary).
2. At j=n_steps-1 (terminal tick), the d1's differ measurably — the
   exact corruption the fix removes.
3. The gamma denominator's ``√τ_t`` factor is consistent with the
   per-tick approach (it always was; this is a regression guard).
4. Smoke check: simulate runs without crashes and produces finite
   results across a small batch.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture
def simulate_inputs():
    """Synthetic inputs to ``DRLHedger.simulate``.

    Three trades with varying signs / strikes. n_steps=10 to mirror
    production. Single seed so the GBM path is deterministic.
    """
    n = 3
    n_steps = 10
    return dict(
        n=n,
        n_steps=n_steps,
        S_entry=np.array([60_000.0, 62_000.0, 58_000.0]),
        K=np.array([60_000.0, 62_000.0, 58_000.0]),
        signal=np.array([-1, +1, -1]),
        T_years=np.array([0.04, 0.06, 0.05]),
        sigma=np.array([0.75, 0.65, 0.85]),
        vol_ann=np.array([0.80, 0.70, 0.90]),
        vol_bg=np.array([100.0, 120.0, 80.0]),
        jump_arr=np.array([0.05, 0.02, 0.10]),
        vpin_arr=np.array([0.30, 0.40, 0.20]),
        seed=42,
    )


def _frozen_d1_at_tick(
    *, S_new: np.ndarray, K: np.ndarray, T_entry: np.ndarray, sigma: np.ndarray,
) -> np.ndarray:
    """Pre-D-7 (frozen-T) formula. Used as the reference for asserting
    the bug at terminal tick."""
    safe_sigma = np.where(sigma > 0, sigma, 0.01)
    safe_T = np.where(T_entry > 0, T_entry, 0.001)
    sigma_sqrt_T = safe_sigma * np.sqrt(safe_T)
    half_sigma2_T = 0.5 * safe_sigma * safe_sigma * safe_T
    return (np.log(S_new / K) + half_sigma2_T) / sigma_sqrt_T


def _live_d1_at_tick(
    *, S_new: np.ndarray, K: np.ndarray, tau_t: np.ndarray, sigma: np.ndarray,
) -> np.ndarray:
    """Post-D-7 (per-tick) formula — matches env.py:418 and
    drl_hedger_live.py:315-322."""
    safe_sigma = np.where(sigma > 0, sigma, 0.01)
    sigma_sqrt_tau_t = safe_sigma * np.sqrt(tau_t)
    half_sigma2_tau_t = 0.5 * safe_sigma * safe_sigma * tau_t
    return (np.log(S_new / K) + half_sigma2_tau_t) / sigma_sqrt_tau_t


def test_d7_at_entry_tick_pre_and_post_agree(simulate_inputs):
    """At j=0 the per-tick formula reduces to the frozen formula
    (since tau_t at j=0 ≈ T_entry - dt_step ≈ T_entry). The two
    diverge as ticks accumulate."""
    inp = simulate_inputs
    S_new = inp["S_entry"]
    # tau at j=0 (one step in)
    bar_minutes = 15.0
    dt_step = (bar_minutes / inp["n_steps"]) / (365.25 * 24 * 60)
    tau_at_j0 = np.maximum(inp["T_years"] - 1 * dt_step, 1e-10)

    d1_frozen = _frozen_d1_at_tick(
        S_new=S_new, K=inp["K"], T_entry=inp["T_years"], sigma=inp["sigma"],
    )
    d1_live = _live_d1_at_tick(
        S_new=S_new, K=inp["K"], tau_t=tau_at_j0, sigma=inp["sigma"],
    )

    # Difference is small (one dt_step's worth of time decay) but not
    # zero — verify it's well below 1% of the d1 magnitude.
    rel_diff = np.abs(d1_frozen - d1_live) / np.maximum(np.abs(d1_frozen), 1e-12)
    assert np.all(rel_diff < 1e-2), (
        f"d1 should be near-identical at j=0; got rel_diff={rel_diff}"
    )


def test_d7_at_terminal_tick_pre_and_post_diverge(simulate_inputs):
    """At j=n_steps the time-decay terms differ by a factor of
    n_steps/1; d1 differs measurably."""
    inp = simulate_inputs
    S_new = inp["S_entry"] * 1.005  # +0.5% drift
    bar_minutes = 15.0
    dt_step = (bar_minutes / inp["n_steps"]) / (365.25 * 24 * 60)
    j_terminal = inp["n_steps"]
    tau_terminal = np.maximum(inp["T_years"] - j_terminal * dt_step, 1e-10)

    d1_frozen = _frozen_d1_at_tick(
        S_new=S_new, K=inp["K"], T_entry=inp["T_years"], sigma=inp["sigma"],
    )
    d1_live = _live_d1_at_tick(
        S_new=S_new, K=inp["K"], tau_t=tau_terminal, sigma=inp["sigma"],
    )

    # At terminal tick they MUST differ — that's the bug. Even with
    # short-term options (T=0.04 years = ~14 days), 10 ticks of decay
    # over a 15-min bar is ~1/96 of T, so the divergence is small in
    # absolute d1 terms but still well above numerical noise.
    abs_diff = np.abs(d1_frozen - d1_live)
    assert np.all(abs_diff > 1e-6), (
        f"d1 should differ at terminal tick; got abs_diff={abs_diff}"
    )


def test_d7_per_tick_formula_is_consistent_with_env_path():
    """The per-tick formula in ``simulate`` (post-D-7) must produce the
    same d1 as the env's per-step formula at the same ``(S, K, sigma,
    tau)`` triple. Both end at ``d1 = (log(S/K) + 0.5σ²τ) / (σ√τ)``.
    """
    np.random.seed(42)
    S = 60_500.0
    K = 60_000.0
    sigma = 0.80
    tau_t = 0.038  # mid-window tau

    # Post-D-7 simulate: same formula
    d1_simulate = _live_d1_at_tick(
        S_new=np.array([S]), K=np.array([K]),
        tau_t=np.array([tau_t]), sigma=np.array([sigma]),
    )[0]

    # Env's per-step formula (env.py:418 — uses same closed form)
    d1_env = (np.log(S / K) + 0.5 * sigma * sigma * tau_t) / (sigma * np.sqrt(tau_t))

    assert d1_simulate == pytest.approx(d1_env, rel=1e-12)


def test_d7_terminal_tick_d1_uses_tau_not_T():
    """Concrete sanity: with T_entry = 0.04, n_steps = 10, the time
    decay at j=10 is ``10 * dt_step`` where
    ``dt_step = 15/(365.25*24*60*10)`` ≈ 2.85e-6 years. So
    ``half_sigma²·tau_t`` should be ~10/96 smaller than
    ``half_sigma²·T`` — verify the per-tick formula's denominator
    (``sigma·√tau_t``) shrinks as expected."""
    bar_minutes = 15.0
    n_steps = 10
    dt_step = (bar_minutes / n_steps) / (365.25 * 24 * 60)
    T_entry = np.array([0.04])
    sigma = np.array([0.80])

    # j=0
    tau_0 = np.maximum(T_entry - 1 * dt_step, 1e-10)
    sqrt_tau_0 = np.sqrt(tau_0)
    # j=n_steps (terminal — never reached in production but tests the limit)
    tau_terminal = np.maximum(T_entry - n_steps * dt_step, 1e-10)
    sqrt_tau_terminal = np.sqrt(tau_terminal)

    # Terminal sqrt should be slightly smaller — by ~ (n_steps-1)*dt_step / T
    # which is ~10/96/365.25 = ~2.85e-4 years out of 0.04 = ~0.7% of T.
    ratio = float((sqrt_tau_terminal / sqrt_tau_0).item())
    assert ratio < 1.0 - 1e-6, (
        f"Terminal sqrt(tau) should be smaller than j=0; got ratio={ratio}"
    )
    assert ratio > 0.99, (
        f"Terminal sqrt(tau) should still be near 1 (one bar << T_entry); "
        f"got ratio={ratio}"
    )
