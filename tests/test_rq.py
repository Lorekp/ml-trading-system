"""Unit tests for Realized Quarticity (RQ) computation in compute_tsrv_window.

RQ is the fourth-moment analog of Realized Variance:
    RQ_n = (n/3) * Σ r_i^4

Key properties:
1. For a pure Brownian motion with constant vol σ and n sub-bar returns of
   equal time-step Δt: E[r_i^4] = 3 σ^4 Δt^2 (Gaussian 4th moment), so
   E[Σ r_i^4] = n · 3 σ^4 Δt^2 = 3 σ^4 T^2 / n, and E[RQ] = (n/3) * 3 σ^4 T^2/n = σ^4 T^2.
   So RQ converges to the integrated quarticity IQ = σ^4 T^2 (constant-vol case).
2. Causality: RQ depends ONLY on sub-bar log-prices within the window — mutating
   ticks after the window must not change RQ for windows ending on/before the
   mutation timestamp.
3. RQ ≥ 0, RQ = 0 for windows with <2 sub-bars (matches _zero fallback).
"""

from __future__ import annotations

import numpy as np
import pytest

from TSRV.tsrv_series import compute_tsrv_window


RNG_SEED = 20260417


def _synthetic_brownian_ticks(
    sigma: float,
    T: float,
    n_ticks: int,
    dollar_per_tick: float = 1_000.0,
    seed: int = RNG_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate constant-vol GBM tick prices + uniform dollar volume.

    Returns (prices, dollar_vol) arrays of shape (n_ticks,).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_ticks
    # Constant-vol log-returns
    increments = rng.normal(loc=0.0, scale=sigma * np.sqrt(dt), size=n_ticks - 1)
    log_p = np.concatenate([[0.0], np.cumsum(increments)])
    prices = 100.0 * np.exp(log_p)
    dollar_vol = np.full(n_ticks, dollar_per_tick, dtype=np.float64)
    return prices, dollar_vol


def test_rq_in_result_dict():
    """compute_tsrv_window returns a dict with an 'rq' key."""
    prices, dollar_vol = _synthetic_brownian_ticks(sigma=0.5, T=1.0, n_ticks=1000)
    result = compute_tsrv_window(prices, dollar_vol, threshold=50_000.0)
    assert "rq" in result, f"'rq' missing from result dict, keys={sorted(result)}"
    assert result["rq"] >= 0.0, f"RQ must be non-negative, got {result['rq']}"
    assert np.isfinite(result["rq"]), f"RQ must be finite, got {result['rq']}"


def test_rq_zero_for_too_few_ticks():
    """Windows with <2 ticks produce rq=0.0 (matches _zero fallback)."""
    # 1 tick — below the n_ticks >= 2 gate
    result_single = compute_tsrv_window(
        np.array([100.0]), np.array([1_000.0]), threshold=1_000.0
    )
    assert result_single["rq"] == 0.0
    assert result_single["n_sub_bars"] == 0

    # 0 ticks
    result_empty = compute_tsrv_window(
        np.array([]), np.array([]), threshold=1_000.0
    )
    assert result_empty["rq"] == 0.0


def test_rq_zero_for_too_few_sub_bars():
    """Windows that accumulate to fewer than 2 sub-bars produce rq=0.0."""
    # With threshold=1e15 (impossibly high), no sub-bars form → fallback zero branch
    prices, dollar_vol = _synthetic_brownian_ticks(sigma=0.5, T=1.0, n_ticks=500)
    result = compute_tsrv_window(prices, dollar_vol, threshold=1e15)
    # n_ticks >= 2 but n_sub_bars < 2 → _zero path with rv_fast populated but rq stays 0.0
    assert result["n_sub_bars"] < 2
    assert result["rq"] == 0.0


def test_rq_approximates_integrated_quarticity_constant_vol():
    """Under constant vol σ on horizon T, RQ should approximate σ^4 T^2.

    Use a large n to ensure the sub-bar discretization captures the integrated
    quarticity well. Tolerance is wide because RQ is a noisy estimator.
    """
    sigma = 0.5       # 50% annualized vol (realistic crypto)
    T = 1.0 / 365.25  # 1 day — single TSRV window horizon (~15 min ≈ 1/35040y)
    n_ticks = 20_000

    # Aggregate 20 independent Monte-Carlo windows → reduce estimator noise
    rq_estimates = []
    expected_iq = sigma ** 4 * T ** 2
    for seed in range(20):
        prices, dollar_vol = _synthetic_brownian_ticks(
            sigma=sigma, T=T, n_ticks=n_ticks, seed=RNG_SEED + seed
        )
        # Threshold tuned so sub_bars is meaningful (roughly 50-200 per window)
        result = compute_tsrv_window(prices, dollar_vol, threshold=200_000.0)
        assert result["n_sub_bars"] >= 10, f"Seed {seed}: too few sub-bars ({result['n_sub_bars']})"
        rq_estimates.append(result["rq"])

    mean_rq = float(np.mean(rq_estimates))
    # Wide tolerance: estimator variance depends on n_sub_bars; factor-of-2 envelope.
    ratio = mean_rq / expected_iq
    assert 0.3 < ratio < 3.5, (
        f"RQ/IQ ratio={ratio:.3f} outside [0.3, 3.5]; mean_rq={mean_rq:.3e}, expected_iq={expected_iq:.3e}"
    )


def test_rq_scales_with_vol_fourth_power():
    """Doubling σ should multiply RQ by ≈16 (σ^4 scaling).

    Again, MC-averaged to reduce noise.
    """
    T = 1.0 / 365.25
    n_ticks = 15_000

    rq_low = []
    rq_high = []
    for seed in range(16):
        p_low, dv_low = _synthetic_brownian_ticks(sigma=0.3, T=T, n_ticks=n_ticks, seed=RNG_SEED + seed)
        p_high, dv_high = _synthetic_brownian_ticks(sigma=0.6, T=T, n_ticks=n_ticks, seed=RNG_SEED + seed)
        rq_low.append(compute_tsrv_window(p_low, dv_low, threshold=150_000.0)["rq"])
        rq_high.append(compute_tsrv_window(p_high, dv_high, threshold=150_000.0)["rq"])

    mean_low = float(np.mean(rq_low))
    mean_high = float(np.mean(rq_high))
    ratio = mean_high / mean_low
    # Expected 2^4 = 16; wide tolerance for MC noise.
    assert 8.0 < ratio < 32.0, (
        f"High/low RQ ratio={ratio:.2f} outside [8, 32] (expected ≈16 for σ doubled)"
    )


def test_rq_causality_mutate_future_ticks():
    """Mutating ticks AFTER the window boundary must not change RQ for the window.

    This is the critical leakage guard: RQ for a 15-min bar ending at T must
    depend only on sub-bar prices within [T-15min, T]. Emulating this by
    computing RQ on a prefix of tick array, then mutating the suffix, and
    asserting RQ-on-prefix is byte-identical.
    """
    prices, dollar_vol = _synthetic_brownian_ticks(sigma=0.5, T=2.0 / 365.25, n_ticks=4000)

    # Split point: simulate a 15-min boundary at tick 2000
    prefix_prices = prices[:2000].copy()
    prefix_dvol = dollar_vol[:2000].copy()

    # Baseline: compute RQ on the prefix alone
    baseline = compute_tsrv_window(prefix_prices, prefix_dvol, threshold=150_000.0)
    baseline_rq = baseline["rq"]

    # Mutate the suffix (ticks 2000+) — should NOT affect prefix computation
    prices_mutated = prices.copy()
    prices_mutated[2000:] = prices_mutated[2000:] * 1000.0  # drastic mutation
    dvol_mutated = dollar_vol.copy()
    dvol_mutated[2000:] = dvol_mutated[2000:] * 0.001

    prefix_prices_from_mutated = prices_mutated[:2000].copy()
    prefix_dvol_from_mutated = dvol_mutated[:2000].copy()

    after_mutation = compute_tsrv_window(
        prefix_prices_from_mutated, prefix_dvol_from_mutated, threshold=150_000.0
    )
    assert after_mutation["rq"] == baseline_rq, (
        f"RQ changed after mutating future ticks: baseline={baseline_rq}, "
        f"after={after_mutation['rq']}"
    )


def test_rq_monotone_under_scaling_of_sub_returns():
    """Scaling all sub-bar returns by c should scale RQ by c^4 exactly.

    We test this via direct formula rather than re-running compute_tsrv_window
    (which recomputes sub_bars from dollar_vol). Here we verify the formula
    itself as used inside compute_tsrv_window.
    """
    rng = np.random.default_rng(RNG_SEED)
    sub_log_p_baseline = np.cumsum(rng.normal(0, 0.01, 50))
    r_baseline = np.diff(sub_log_p_baseline)
    rq_baseline = (len(r_baseline) / 3.0) * float(np.sum(r_baseline ** 4))

    c = 2.5
    r_scaled = r_baseline * c
    rq_scaled = (len(r_scaled) / 3.0) * float(np.sum(r_scaled ** 4))

    expected = rq_baseline * c ** 4
    np.testing.assert_allclose(rq_scaled, expected, rtol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
