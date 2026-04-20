"""Tests for vol_model.features.vrp (Session H Candidate C).

Gates (modeled on tests/test_features_rough_heston.py):

1. ``compute_vrp_features`` emits 5 expected columns as float64.
2. All columns are byte-identical copies of the sidecar.
3. Causality: ``probe_causality_by_mutation`` passes on synthetic data.
4. Real-data causality: passes on the built sidecar at T=2023-06-15.
5. Coverage on real sidecar ≥ 90% per column (VRP depends on a 30-day
   rolling window, so warmup is substantial; 90% is conservative).
6. Feature-range sanity: iv_var_1m > 0, rv_var_30d > 0, vrp can be
   positive or negative (regime-dependent), z-score mean ≈ 0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vol_model.features.vrp import (
    VRP_DELTA_FEATURES,
    VRP_FEATURES,
    VRP_LEVEL_FEATURES,
    compute_vrp_features,
    describe_coverage,
)
from vol_model.leakage_probes import probe_causality_by_mutation


# ── Fixtures ─────────────────────────────────────────────────────────


def _synthetic_sidecar(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Synthetic VRP sidecar with realistic magnitudes.

    iv_var_1m ~ 0.25-1.0 (annualized variance, i.e. 50%-100% vol²)
    rv_var_30d ~ 0.2-1.2 (noisier)
    vrp_30d ~ N(0, 0.1) centered near 0
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01 01:00", periods=n, freq="15min", tz="UTC")

    iv_var = np.clip(0.5 + 0.1 * rng.standard_normal(n).cumsum() * 0.05, 0.05, 2.0)
    rv_var = iv_var + 0.1 * rng.standard_normal(n)  # similar scale
    rv_var = np.clip(rv_var, 0.05, 2.0)
    vrp = iv_var - rv_var
    # Synthetic z-score
    vrp_series = pd.Series(vrp, index=idx)
    vrp_roll = vrp_series.rolling(100, min_periods=20)
    vrp_z = (vrp_series - vrp_roll.mean()) / vrp_roll.std(ddof=0).replace(0.0, np.nan)
    vrp_d1d = vrp_series.diff(96)  # .diff(96) on 15m grid

    df = pd.DataFrame(
        {
            "iv_var_1m": iv_var,
            "rv_var_30d": rv_var,
            "vrp_30d": vrp,
            "vrp_30d_z30d": vrp_z.to_numpy(),
            "vrp_30d_d1d": vrp_d1d.to_numpy(),
        },
        index=idx,
    )
    return df.fillna(0.0)  # fill warmup for clean synthetic probe


# ── 1. Schema ────────────────────────────────────────────────────────


def test_compute_returns_expected_columns() -> None:
    sidecar = _synthetic_sidecar(n=200)
    feats = compute_vrp_features(sidecar)
    assert list(feats.columns) == list(VRP_FEATURES)
    assert feats.index.equals(sidecar.index)
    assert len(VRP_FEATURES) == 5
    assert len(VRP_LEVEL_FEATURES) == 4
    assert len(VRP_DELTA_FEATURES) == 1
    for col in feats.columns:
        assert feats[col].dtype == np.float64, f"{col} is {feats[col].dtype}"


def test_compute_missing_column_raises() -> None:
    sidecar = _synthetic_sidecar(n=100).drop(columns=["vrp_30d"])
    with pytest.raises(KeyError, match="vrp_30d"):
        compute_vrp_features(sidecar)


def test_compute_non_datetime_index_raises() -> None:
    sidecar = _synthetic_sidecar(n=100).reset_index(drop=True)
    with pytest.raises(TypeError, match="DatetimeIndex"):
        compute_vrp_features(sidecar)


# ── 2. Byte-identity ─────────────────────────────────────────────────


def test_features_are_byte_identical_copies() -> None:
    sidecar = _synthetic_sidecar(n=200)
    feats = compute_vrp_features(sidecar)
    for col in VRP_FEATURES:
        np.testing.assert_array_equal(
            feats[col].to_numpy(),
            sidecar[col].to_numpy().astype(float),
            err_msg=f"{col} is not a byte-identical copy",
        )


# ── 3. Causality probe (synthetic) ───────────────────────────────────


def test_causality_by_mutation_synthetic() -> None:
    sidecar = _synthetic_sidecar(n=300)
    T = sidecar.index[150]
    result = probe_causality_by_mutation(compute_vrp_features, sidecar, T)
    assert result.passed, result.detail
    assert result.metric == 0.0


# ── 4. Real-data probe + coverage ────────────────────────────────────


def test_causality_by_mutation_real_sidecar() -> None:
    real_path = Path("vol_arb/output/vrp_features_15m.parquet")
    if not real_path.exists():
        pytest.skip("real VRP sidecar not present; run build first")
    sidecar = pd.read_parquet(real_path)
    T = pd.Timestamp("2023-06-15 12:00:00", tz="UTC")
    if T not in sidecar.index:
        pytest.skip(f"T={T} not in sidecar index")
    result = probe_causality_by_mutation(compute_vrp_features, sidecar, T)
    assert result.passed, result.detail


def test_coverage_on_real_sidecar() -> None:
    """VRP depends on a 30-day rolling window. At the sidecar start,
    features have NaN during warmup. 90% threshold accounts for the
    initial ~30 days."""
    real_path = Path("vol_arb/output/vrp_features_15m.parquet")
    if not real_path.exists():
        pytest.skip("real VRP sidecar not present; run build first")
    sidecar = pd.read_parquet(real_path)
    cov = describe_coverage(sidecar)
    low_cov = cov[cov < 0.90]
    assert low_cov.empty, f"low-coverage features: {low_cov.to_dict()}"


# ── 5. Feature-range sanity ──────────────────────────────────────────


def test_feature_ranges_on_real_sidecar() -> None:
    real_path = Path("vol_arb/output/vrp_features_15m.parquet")
    if not real_path.exists():
        pytest.skip("real VRP sidecar not present; run build first")
    sidecar = pd.read_parquet(real_path)
    iv = sidecar["iv_var_1m"].dropna()
    rv = sidecar["rv_var_30d"].dropna()
    vrp = sidecar["vrp_30d"].dropna()
    vrp_z = sidecar["vrp_30d_z30d"].dropna()

    # IV² and RV² are positive
    assert (iv > 0).all(), "iv_var_1m must be positive"
    assert (rv > 0).all(), "rv_var_30d must be positive"
    # IV² and RV² are bounded sensibly (BTC annualized vol rarely >300%
    # sustained, so variance rarely > 9)
    assert iv.quantile(0.99) < 9.0, f"iv_var p99={iv.quantile(0.99)}"
    assert rv.quantile(0.99) < 9.0, f"rv_var p99={rv.quantile(0.99)}"

    # VRP can be positive or negative — regime-dependent
    # Almeida-Grith-Miftachov 2024: BTC VRP has flipped signs across 2021-2024
    assert vrp.min() < 0, "VRP should go negative at least once (crisis regimes)"
    assert vrp.max() > 0, "VRP should go positive at least once (calm regimes)"

    # z-score centers near 0; tails can be large
    assert abs(vrp_z.mean()) < 0.5, f"z-score mean off-center: {vrp_z.mean()}"
    assert vrp_z.std() > 0.5, f"z-score degenerate: std={vrp_z.std()}"


def test_vrp_arithmetic_consistency() -> None:
    """Verify VRP = IV² − RV² holds exactly on the real sidecar."""
    real_path = Path("vol_arb/output/vrp_features_15m.parquet")
    if not real_path.exists():
        pytest.skip("real VRP sidecar not present; run build first")
    sidecar = pd.read_parquet(real_path)
    # On rows where all three are defined, vrp_30d = iv_var_1m - rv_var_30d
    valid = sidecar[["iv_var_1m", "rv_var_30d", "vrp_30d"]].dropna()
    computed = valid["iv_var_1m"] - valid["rv_var_30d"]
    np.testing.assert_allclose(
        valid["vrp_30d"].to_numpy(), computed.to_numpy(),
        rtol=1e-10, atol=1e-12,
        err_msg="vrp_30d != iv_var_1m - rv_var_30d",
    )
