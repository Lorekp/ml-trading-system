"""Tests for vol_model.features.hawkes_ofi (Session F Candidate B).

Gates pinned here:

1. ``compute_hawkes_ofi_features`` emits the 7 expected columns as float64.
2. All columns are byte-identical copies of the sidecar.
3. Causality: ``probe_causality_by_mutation`` passes on synthetic data
   (the function is a pure projection, so this is trivial-but-documented).
4. Real-data causality: ``probe_causality_by_mutation`` passes on the
   built sidecar at mid-series T (requires
   ``vol_arb/output/hawkes_ofi_features_15m.parquet`` to exist).
5. Coverage on real sidecar ≥ 88% per column.
6. Parameter sanity: branching ratios ∈ [0, ~2], mu in reasonable range.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vol_model.features.hawkes_ofi import (
    HAWKES_OFI_DELTA_FEATURES,
    HAWKES_OFI_FEATURES,
    HAWKES_OFI_LEVEL_FEATURES,
    compute_hawkes_ofi_features,
    describe_coverage,
)
from vol_model.leakage_probes import probe_causality_by_mutation


# ── Fixtures ─────────────────────────────────────────────────────────


def _synthetic_sidecar(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Synthetic Hawkes-OFI sidecar with realistic magnitudes."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01 01:00", periods=n, freq="15min", tz="UTC")

    # Realistic-ish levels
    mu_b = 1.0 + 0.3 * rng.standard_normal(n).cumsum() * 0.1
    mu_s = 0.8 + 0.3 * rng.standard_normal(n).cumsum() * 0.1
    br_b = 0.5 + 0.1 * rng.standard_normal(n) + 0.05 * rng.standard_normal(n).cumsum() * 0.05
    br_s = 0.5 + 0.1 * rng.standard_normal(n) + 0.05 * rng.standard_normal(n).cumsum() * 0.05
    asym = 0.1 * rng.standard_normal(n)

    df = pd.DataFrame(
        {
            "hawkes_mu_buy": np.abs(mu_b),
            "hawkes_mu_sell": np.abs(mu_s),
            "hawkes_branching_buy": np.clip(br_b, 0, 1.5),
            "hawkes_branching_sell": np.clip(br_s, 0, 1.5),
            "hawkes_asymmetry": asym,
            "hawkes_branching_buy_d1h": np.r_[[np.nan] * 4, np.diff(np.clip(br_b, 0, 1.5), 4)],
            "hawkes_branching_sell_d1h": np.r_[[np.nan] * 4, np.diff(np.clip(br_s, 0, 1.5), 4)],
        },
        index=idx,
    )
    return df


# ── 1. Schema ────────────────────────────────────────────────────────


def test_compute_returns_expected_columns() -> None:
    sidecar = _synthetic_sidecar(n=200)
    feats = compute_hawkes_ofi_features(sidecar)
    assert list(feats.columns) == list(HAWKES_OFI_FEATURES)
    assert feats.index.equals(sidecar.index)
    # 5 levels + 2 deltas = 7
    assert len(HAWKES_OFI_FEATURES) == 7
    assert len(HAWKES_OFI_LEVEL_FEATURES) == 5
    assert len(HAWKES_OFI_DELTA_FEATURES) == 2
    # Every column float64
    for col in feats.columns:
        assert feats[col].dtype == np.float64, f"{col} is {feats[col].dtype}"


def test_compute_missing_column_raises() -> None:
    sidecar = _synthetic_sidecar(n=100).drop(columns=["hawkes_branching_buy"])
    with pytest.raises(KeyError, match="hawkes_branching_buy"):
        compute_hawkes_ofi_features(sidecar)


def test_compute_non_datetime_index_raises() -> None:
    sidecar = _synthetic_sidecar(n=100).reset_index(drop=True)
    with pytest.raises(TypeError, match="DatetimeIndex"):
        compute_hawkes_ofi_features(sidecar)


# ── 2. Byte-identity ─────────────────────────────────────────────────


def test_features_are_byte_identical_copies() -> None:
    sidecar = _synthetic_sidecar(n=200)
    feats = compute_hawkes_ofi_features(sidecar)
    for col in HAWKES_OFI_FEATURES:
        np.testing.assert_array_equal(
            feats[col].to_numpy(),
            sidecar[col].to_numpy().astype(float),
            err_msg=f"{col} is not a byte-identical copy",
        )


# ── 3. Gold-standard causality probe ─────────────────────────────────


def test_causality_by_mutation_synthetic() -> None:
    """Pure projection — passes causality probe trivially."""
    sidecar = _synthetic_sidecar(n=300)
    T = sidecar.index[150]
    result = probe_causality_by_mutation(compute_hawkes_ofi_features, sidecar, T)
    assert result.passed, result.detail
    assert result.metric == 0.0


# ── 4. Real-data probe + coverage ────────────────────────────────────


def test_causality_by_mutation_real_sidecar() -> None:
    """End-to-end: mutating all rows > T must not change feature[T].

    Runs only if the real Hawkes-OFI sidecar has been built. The calibrator
    at bucket T only looks at trades in [T-4h, T), so this MUST pass — if
    it fails, there's a stamp convention bug in the sidecar builder.
    """
    real_path = Path("vol_arb/output/hawkes_ofi_features_15m.parquet")
    if not real_path.exists():
        pytest.skip("real Hawkes-OFI sidecar not present; run build first")
    sidecar = pd.read_parquet(real_path)
    # Pick a mid-series T with data
    T = pd.Timestamp("2023-06-15 12:00:00", tz="UTC")
    if T not in sidecar.index:
        pytest.skip(f"T={T} not in sidecar index")
    result = probe_causality_by_mutation(compute_hawkes_ofi_features, sidecar, T)
    assert result.passed, result.detail


def test_coverage_on_real_sidecar() -> None:
    """On real data, every feature must have ≥ 88% non-NaN coverage."""
    real_path = Path("vol_arb/output/hawkes_ofi_features_15m.parquet")
    if not real_path.exists():
        pytest.skip("real Hawkes-OFI sidecar not present; run build first")
    sidecar = pd.read_parquet(real_path)
    cov = describe_coverage(sidecar)
    low_cov = cov[cov < 0.88]
    assert low_cov.empty, f"low-coverage features: {low_cov.to_dict()}"


# ── 5. Parameter sanity on real sidecar ──────────────────────────────


def test_branching_ratios_in_expected_range() -> None:
    """On real data, branching ratios should fall in [0, 2] (well below the
    MLE bounds of exp(log_bound_hi) ~ 1000 but allowing for occasional
    super-critical fits during FTX-type regimes)."""
    real_path = Path("vol_arb/output/hawkes_ofi_features_15m.parquet")
    if not real_path.exists():
        pytest.skip("real Hawkes-OFI sidecar not present; run build first")
    sidecar = pd.read_parquet(real_path)
    br_buy = sidecar["hawkes_branching_buy"].dropna()
    br_sell = sidecar["hawkes_branching_sell"].dropna()
    assert (br_buy >= 0).all(), "branching_buy has negative values"
    assert (br_sell >= 0).all(), "branching_sell has negative values"
    # 99th percentile should be well under 2 (rare super-criticality)
    assert br_buy.quantile(0.99) < 2.0, f"br_buy p99 = {br_buy.quantile(0.99)}"
    assert br_sell.quantile(0.99) < 2.0, f"br_sell p99 = {br_sell.quantile(0.99)}"
    # Median should be positive and < 1 (stable regime is typical)
    assert 0.1 < br_buy.median() < 1.5
    assert 0.1 < br_sell.median() < 1.5
