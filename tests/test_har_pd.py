"""Unit tests for HAR-PD path-dependent features (Tier 3 signal research 2026-04-17).

Verifies:
  1. Shape and output schema.
  2. Kernel normalization: sum(w_k) == 1 for all alpha.
  3. STRICT CAUSALITY: mutating TSRV values at times > T leaves R1(T), R2(T)
     byte-identical. This is the hard leakage guard.
  4. First K_BARS rows are NaN (insufficient history).
  5. Synthetic-data check: a constant signed_flow series produces R1 equal
     to that constant (from bar K onwards).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_model.features.har_pd import (
    ALPHA_GRID,
    BETA_GRID,
    K_BARS,
    HAR_PD_RAW_FEATURES,
    _exponential_kernel_weights,
    _weighted_rolling_sum,
    compute_har_pd_features,
)


def _make_synthetic_tsrv(n_bars: int = 500, seed: int = 42) -> pd.DataFrame:
    """Build a synthetic TSRV DataFrame with just enough columns."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="15min", tz="UTC")
    # Fake upside/downside RV pieces that sum to a realistic 15m variance
    rv_ann = rng.uniform(0.05, 2.0, size=n_bars)  # annualized variance
    rv_per_bar = rv_ann / (365.25 * 24 * 4)
    # Split per-bar RV into upside/downside randomly
    frac = rng.uniform(0.2, 0.8, size=n_bars)
    rs_up = rv_per_bar * frac
    rs_down = rv_per_bar * (1 - frac)
    df = pd.DataFrame(
        {"rs_up": rs_up, "rs_down": rs_down, "rv_ann": rv_ann},
        index=idx,
    )
    df.index.name = "timestamp"
    return df


def test_kernel_weights_sum_to_one():
    for alpha in ALPHA_GRID:
        w = _exponential_kernel_weights(alpha)
        assert abs(w.sum() - 1.0) < 1e-12
        assert len(w) == K_BARS
        # Weights monotonically decreasing (exponential decay)
        assert np.all(np.diff(w) <= 0)


def test_kernel_weights_reject_invalid_alpha():
    for bad in [0.0, 1.0, -0.1, 1.5]:
        with pytest.raises(ValueError):
            _exponential_kernel_weights(bad)


def test_output_schema():
    tsrv = _make_synthetic_tsrv(n_bars=500)
    feats = compute_har_pd_features(tsrv)
    assert list(feats.columns) == list(HAR_PD_RAW_FEATURES)
    assert feats.shape == (500, 6)
    assert (feats.index == tsrv.index).all()


def test_first_k_rows_are_nan():
    tsrv = _make_synthetic_tsrv(n_bars=500)
    feats = compute_har_pd_features(tsrv)
    # First K_BARS rows must be NaN (not enough history)
    for c in feats.columns:
        assert feats[c].iloc[:K_BARS].isna().all(), f"{c}: expected all NaN in first K rows"
    # Row K_BARS and beyond must be finite
    for c in feats.columns:
        assert feats[c].iloc[K_BARS:].notna().all(), f"{c}: expected finite values past K"


def test_causality_mutating_future_leaves_past_unchanged():
    """Critical leakage guard: mutating TSRV at indices > T leaves R1/R2 at T byte-identical."""
    tsrv = _make_synthetic_tsrv(n_bars=300, seed=1)
    original_feats = compute_har_pd_features(tsrv)

    # Mutate rows > T=200: random replacements for rs_up / rs_down / rv_ann
    T = 200
    mutated = tsrv.copy()
    rng = np.random.default_rng(99)
    for c in ["rs_up", "rs_down", "rv_ann"]:
        mutated.iloc[T + 1 :, mutated.columns.get_loc(c)] = rng.uniform(0.1, 10.0, size=len(mutated) - T - 1)

    mutated_feats = compute_har_pd_features(mutated)

    # At rows [0..T], original and mutated must be IDENTICAL
    for c in HAR_PD_RAW_FEATURES:
        a = original_feats[c].iloc[: T + 1].dropna().values
        b = mutated_feats[c].iloc[: T + 1].dropna().values
        assert np.array_equal(a, b), f"{c}: causality violation — values before T changed after mutating future"

    # At rows > T, mutated should differ (sanity — did we actually mutate?)
    diffs = (original_feats.iloc[T + 2 :] - mutated_feats.iloc[T + 2 :]).abs().sum().sum()
    assert diffs > 1e-6, "Mutation appears to have had no effect — test is broken"


def test_constant_input_yields_constant_output():
    """Weighted sum of a constant series equals that constant (weights sum to 1)."""
    n_bars = 200
    idx = pd.date_range("2024-01-01", periods=n_bars, freq="15min", tz="UTC")
    # Constant series: all rs_up = rs_down = 1e-5, all rv_ann = 1.0
    tsrv = pd.DataFrame({
        "rs_up": np.full(n_bars, 1e-5),
        "rs_down": np.full(n_bars, 1e-5),
        "rv_ann": np.full(n_bars, 1.0),
    }, index=idx)
    feats = compute_har_pd_features(tsrv)

    # R1 feeds on signed_flow = sqrt(rs_up) - sqrt(rs_down) = 0 (all zeros)
    for c in ("har_pd_R1_a005", "har_pd_R1_a01", "har_pd_R1_a03"):
        vals = feats[c].dropna().values
        assert np.allclose(vals, 0.0, atol=1e-12), f"{c}: expected all zeros, got {vals[:5]}"

    # R2 feeds on rv_ann = 1.0; weighted sum = 1.0 (weights sum to 1)
    for c in ("har_pd_R2_b005", "har_pd_R2_b01", "har_pd_R2_b03"):
        vals = feats[c].dropna().values
        assert np.allclose(vals, 1.0, atol=1e-10), f"{c}: expected all 1.0, got {vals[:5]}"


def test_weighted_rolling_sum_manual():
    """Hand-computed check: 3-bar kernel, 5-row input."""
    series = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0],
                       index=pd.date_range("2024-01-01", periods=5, freq="15min", tz="UTC"))
    weights = np.array([0.5, 0.3, 0.2])  # sum=1.0, K=3
    out = _weighted_rolling_sum(series, weights)
    # Rows 0, 1, 2 = NaN (K_BARS rolled bars needed)
    # Row 3: weighted sum of series[0..2] = 0.5 * series[2] + 0.3 * series[1] + 0.2 * series[0]
    #      = 0.5 * 30 + 0.3 * 20 + 0.2 * 10 = 15 + 6 + 2 = 23.0
    # Row 4: 0.5 * series[3] + 0.3 * series[2] + 0.2 * series[1]
    #      = 0.5 * 40 + 0.3 * 30 + 0.2 * 20 = 20 + 9 + 4 = 33.0
    # But note our function uses K_BARS=96 globally... let's verify by running locally with small K
    # Actually the inner _weighted_rolling_sum accepts any weights; we supply K=3 weights here.
    assert np.isnan(out.iloc[0])
    assert np.isnan(out.iloc[1])
    assert np.isnan(out.iloc[2])
    assert abs(out.iloc[3] - 23.0) < 1e-12, f"row 3: {out.iloc[3]}"
    assert abs(out.iloc[4] - 33.0) < 1e-12, f"row 4: {out.iloc[4]}"
