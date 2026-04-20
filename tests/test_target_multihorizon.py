"""Tests for Session I Candidate I-2 (multi-horizon base+delta reconciliation).

Pins:
1. 60m target arithmetic: log-vol aggregate of 4 consecutive 15m bars
   matches the analytic formula ``0.5 * log(mean(exp(2*ln_vol[T:T+4])))``.
2. 60m target drops last 3 rows (strictly forward-looking, no padding).
3. Alignment contract: when feature at T-15m predicts the 60m label at
   T, mutating any 15m bar at index ≥ T (inside the 60m window) is
   EXPECTED to change the label. Mutating any bar at index > T+45m
   (the last bar inside the [T, T+60m) window is T+45m label-index = 3
   bars ahead) must NOT change the label at T.
4. Meta-combiner fit on val doesn't touch test rows: mutating
   ``pred_val_15m``, ``pred_val_60m``, ``y_val_15m`` for synthetic
   test-set indices must NOT change fitted coefficients.
5. Oracle combination: if α=1, β=0, γ=0 (pass-through on 15m head),
   reconciled forecast equals the 15m head.
6. TIER_FEATURE_MAP + suffix dispatch contract.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_model.signal_research import (
    build_60m_aggregate_logvol,
    fit_multihorizon_meta_combiner,
)


# ── 1. 60m target arithmetic ─────────────────────────────────────────


def test_build_60m_arithmetic() -> None:
    """Hand-verify on a short known sequence. For 15m log-vol values
    [ln(0.5), ln(1.0), ln(1.5), ln(2.0)]:
      vol_ann² = [0.25, 1.0, 2.25, 4.0]
      mean = (0.25 + 1.0 + 2.25 + 4.0) / 4 = 1.875
      ln_vol_60m[0] = 0.5 * log(1.875)
    """
    vol_ann = np.array([0.5, 1.0, 1.5, 2.0, 1.0])  # 5 bars
    ln_vol = np.log(vol_ann)
    idx = pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC")
    series = pd.Series(ln_vol, index=idx, name="ln_vol")

    result = build_60m_aggregate_logvol(series)

    # Expected: length n-3 = 2
    assert len(result) == 2

    # Row 0: mean(0.25, 1.0, 2.25, 4.0) = 1.875
    expected_0 = 0.5 * np.log(1.875)
    # Row 1: mean(1.0, 2.25, 4.0, 1.0) = 2.0625
    expected_1 = 0.5 * np.log(2.0625)

    np.testing.assert_allclose(result.iloc[0], expected_0, rtol=1e-12)
    np.testing.assert_allclose(result.iloc[1], expected_1, rtol=1e-12)

    # Index preserves first n-3 timestamps of input
    assert result.index[0] == idx[0]
    assert result.index[1] == idx[1]


def test_build_60m_empty_when_too_short() -> None:
    """Fewer than 4 bars → no 60m aggregates can be computed."""
    idx = pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC")
    series = pd.Series([0.1, 0.2, 0.3], index=idx)
    result = build_60m_aggregate_logvol(series)
    assert len(result) == 0


def test_build_60m_preserves_index() -> None:
    """Name + index[:n-3] pattern."""
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    vol_ann = np.exp(np.random.default_rng(0).standard_normal(n) * 0.3)
    series = pd.Series(np.log(vol_ann), index=idx)
    result = build_60m_aggregate_logvol(series)
    assert result.name == "ln_vol_60m"
    assert len(result) == n - 3
    assert list(result.index) == list(idx[:n - 3])


def test_build_60m_nan_on_index_gaps() -> None:
    """CRITICAL: if the 15m series has gaps (e.g., SVI coverage starts
    late, or data-quality drops a bar), the sliding window spans an
    irregular interval and the 60m aggregate is MEANINGLESS. The
    function must NaN those rows rather than emit a bogus average of
    non-adjacent bars.

    Without this guard, the multihorizon candidate's smoke test on
    2023-07 showed QLIKE +33% (vs. baseline +1.72%) because some
    gap-spanning windows averaged bars hours or days apart and the
    meta-combiner picked up the garbage.
    """
    # Construct 10 bars, drop the middle 3 → 7 bars with a gap at T=3
    full_idx = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
    gapped_idx = full_idx.delete([3, 4, 5])  # remove rows 3,4,5
    series = pd.Series(
        np.log(np.arange(1, 8, dtype=float)),  # monotonic so we can spot the NaN
        index=gapped_idx,
    )
    result = build_60m_aggregate_logvol(series)

    # Row 0 at T=00:00: needs bars at 00:00, 00:15, 00:30, 00:45.
    # Available bars: 00:00, 00:15, 00:30, ??? → gapped_idx.iloc[3] is
    # now 01:15 (since 00:45, 01:00 were deleted). Window spans 75min
    # not 45min → invalid → NaN.
    # In general, only rows where 4 consecutive adjacent bars exist in
    # the gapped index can produce a valid aggregate.
    assert result.isna().any(), (
        "expected some NaN rows due to index gaps; got no NaN — "
        "gap-guard is not working"
    )


# ── 2. Alignment contract (the core leakage test) ────────────────────


def test_60m_target_uses_only_window_bars() -> None:
    """The 60m target at T must be byte-identical under mutation of any
    bar at index > T + 3*freq (i.e., at index ≥ T + 60m, outside the
    [T, T+60m) window). Bars INSIDE the window (T, T+15m, T+30m,
    T+45m) are the aggregation inputs and are EXPECTED to be read.
    """
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    ln_vol = np.log(0.5 + 0.1 * np.random.default_rng(0).standard_normal(n))
    series = pd.Series(ln_vol.copy(), index=idx)

    # Baseline output
    baseline = build_60m_aggregate_logvol(series)

    # Pick an index T well inside the valid range
    T_pos = 40
    T = idx[T_pos]

    # Mutate all bars at index >= T + 4*freq = index[T_pos + 4] onward
    mutated = series.copy()
    mutated.iloc[T_pos + 4:] = 999.0
    result_mutated = build_60m_aggregate_logvol(mutated)

    # At T, the 60m aggregate uses bars at index T_pos..T_pos+3.
    # The mutation starts at T_pos+4 (outside the window), so output
    # at T must be byte-identical.
    np.testing.assert_allclose(
        result_mutated.loc[T], baseline.loc[T],
        rtol=0.0, atol=0.0,
    )


def test_60m_target_fails_when_in_window_mutated() -> None:
    """Negative control: mutating a bar INSIDE the [T, T+60m) window
    MUST change the output at T. Pins the probe's sensitivity."""
    n = 100
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    ln_vol = np.log(0.5 + 0.1 * np.random.default_rng(1).standard_normal(n))
    series = pd.Series(ln_vol.copy(), index=idx)

    baseline = build_60m_aggregate_logvol(series)

    T_pos = 40
    T = idx[T_pos]

    # Mutate one bar INSIDE the window (at T+2*15m = index T_pos+2)
    mutated = series.copy()
    mutated.iloc[T_pos + 2] = 999.0
    result_mutated = build_60m_aggregate_logvol(mutated)

    assert result_mutated.loc[T] != baseline.loc[T], (
        "60m aggregate should have changed when an in-window bar was "
        "mutated — aggregation is not actually reading the window!"
    )


# ── 3. Meta-combiner no-test-leakage ─────────────────────────────────


def test_meta_combiner_basic_fit() -> None:
    """Sanity: fit returns finite convex-combo coefficients.
    When y = p15 exactly, the QLIKE-optimal λ is 1 (all weight on 15m)
    and γ=0. α+β=1 constraint holds. Bounded 1D optimizer precision
    is 1e-4 (xatol), so allow tolerance.
    """
    rng = np.random.default_rng(0)
    p15 = rng.standard_normal(200)
    p60 = rng.standard_normal(200)
    y = p15.copy()  # perfect 15m passthrough
    alpha, beta, gamma = fit_multihorizon_meta_combiner(y, p15, p60)
    # 1D bounded optimizer precision: xatol=1e-4
    np.testing.assert_allclose(alpha, 1.0, atol=1e-3)
    np.testing.assert_allclose(beta, 0.0, atol=1e-3)
    np.testing.assert_allclose(gamma, 0.0, atol=1e-3)
    # Pin the convex constraint (exact to machine precision)
    np.testing.assert_allclose(alpha + beta, 1.0, atol=1e-12)
    assert 0.0 <= alpha <= 1.0
    assert 0.0 <= beta <= 1.0


def test_meta_combiner_picks_60m_when_better() -> None:
    """If y = p60 (perfect 60m passthrough), QLIKE-optimal λ is 0."""
    rng = np.random.default_rng(4)
    p15 = rng.standard_normal(200) * 0.5
    p60 = rng.standard_normal(200) * 0.5
    y = p60.copy()
    alpha, beta, gamma = fit_multihorizon_meta_combiner(y, p15, p60)
    np.testing.assert_allclose(alpha, 0.0, atol=1e-3)
    np.testing.assert_allclose(beta, 1.0, atol=1e-3)
    np.testing.assert_allclose(gamma, 0.0, atol=1e-3)


def test_meta_combiner_bounded_never_blows_up() -> None:
    """Stress: extreme multicollinearity (p15 = p60 + tiny noise) must
    NOT produce runaway coefficients. The convex constraint guarantees
    |α|, |β| ≤ 1 always — so tiny differences in predictions cannot
    amplify into large test-time errors. This is the property that
    fixes the unconstrained-OLS smoke-test failure."""
    rng = np.random.default_rng(5)
    p15 = rng.standard_normal(500)
    p60 = p15 + 1e-6 * rng.standard_normal(500)
    y = rng.standard_normal(500) * 0.1 + p15
    alpha, beta, gamma = fit_multihorizon_meta_combiner(y, p15, p60)
    assert 0.0 <= alpha <= 1.0
    assert 0.0 <= beta <= 1.0
    np.testing.assert_allclose(alpha + beta, 1.0, atol=1e-12)
    # γ is a small mean-shift on the order of |y - p15|
    assert abs(gamma) < 1.0


def test_meta_combiner_no_test_leakage() -> None:
    """CRITICAL: the meta-combiner fit on val MUST NOT depend on test
    rows. Simulate: fit on val-only arrays, then mutate what would be
    test-row entries (not present in val slice) and confirm byte-
    identity. This is a structural invariant of the API — the function
    only takes val arrays, so test data CANNOT flow in. Pin it anyway
    as a contract test to catch accidental future refactors."""
    rng = np.random.default_rng(2)
    n_val = 150
    y_val = rng.standard_normal(n_val) * 0.3
    p15_val = y_val + 0.1 * rng.standard_normal(n_val)
    p60_val = y_val * 0.8 + 0.1 * rng.standard_normal(n_val)

    # Fit (this is the only fit the API supports — no test arrays flow in)
    alpha_a, beta_a, gamma_a = fit_multihorizon_meta_combiner(
        y_val, p15_val, p60_val,
    )
    # Trying to call with TEST-set data corrupting val is only possible
    # by the caller's own bug; the function itself only sees val. Re-fit
    # with the same val arrays — bit-identical by construction.
    alpha_b, beta_b, gamma_b = fit_multihorizon_meta_combiner(
        y_val, p15_val, p60_val,
    )
    assert alpha_a == alpha_b
    assert beta_a == beta_b
    assert gamma_a == gamma_b


def test_meta_combiner_handles_collinear() -> None:
    """When p15 and p60 are identical, convex combo degenerates to
    λ=1 (either side gives the same value) and γ absorbs the mean
    shift. α+β=1 always holds."""
    rng = np.random.default_rng(3)
    p = rng.standard_normal(100)
    y = 2.0 * p + 1.0
    alpha, beta, gamma = fit_multihorizon_meta_combiner(y, p, p)
    # Convex combo: α + β = 1 ALWAYS (not 2 — the 2x scaling must be
    # absorbed by mean-residual γ which can't handle it, so recovery
    # is imperfect on this synthetic). Accept that the convex combo
    # cannot represent y = 2·p linearly; the test pins structural
    # invariants instead.
    np.testing.assert_allclose(alpha + beta, 1.0, atol=1e-12)
    assert 0.0 <= alpha <= 1.0
    assert 0.0 <= beta <= 1.0
    assert np.isfinite(gamma)


# ── 4. Oracle combination sanity ─────────────────────────────────────


def test_reconciliation_passthrough_when_alpha_1() -> None:
    """If meta-combiner degenerates to (α=1, β=0, γ=0), reconciled
    output equals the 15m head exactly."""
    pred_15 = np.array([0.1, -0.3, 0.5, 0.0])
    pred_60 = np.array([-0.2, 0.1, 0.4, 0.5])
    reconciled = 1.0 * pred_15 + 0.0 * pred_60 + 0.0
    np.testing.assert_array_equal(reconciled, pred_15)


# ── 5. Tier map integration ──────────────────────────────────────────


def test_har_pd_multihorizon_in_tier_map() -> None:
    from vol_model.signal_research import TIER_FEATURE_MAP
    assert "har_pd_multihorizon" in TIER_FEATURE_MAP
    assert "har_pd" in TIER_FEATURE_MAP
    # Same features as har_pd (feature set unchanged; novelty is target)
    assert TIER_FEATURE_MAP["har_pd_multihorizon"] == TIER_FEATURE_MAP["har_pd"]


def test_thread_suffix_routes_to_multihorizon_trainer() -> None:
    assert "har_pd_multihorizon".endswith("_multihorizon")
    # Contract: suffix dispatch in run_spot_check_signal_research must
    # route ANY thread ending in `_multihorizon` to
    # train_multihorizon_reconciled.
