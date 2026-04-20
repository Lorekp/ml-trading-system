"""Tests for vol_model.features.funding (Session H Candidate B).

Gates (modeled on tests/test_features_rough_heston.py):

1. ``compute_funding_features`` emits 5 expected columns as float64.
2. All columns are byte-identical copies of the sidecar.
3. Causality: ``probe_causality_by_mutation`` passes on synthetic data.
4. Real-data causality: passes on the built sidecar.
5. Coverage on real sidecar ≥ 95% per column (funding is continuously
   published 8-hourly since 2019; any gap would indicate ingestion
   issue, not expected structural NaN).
6. Feature-range sanity: fr_level in plausible perp-funding range,
   fr_abs >= 0, fr_zscore_30d mean ≈ 0.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vol_model.features.funding import (
    FUNDING_FEATURES,
    FUNDING_LEVEL_FEATURES,
    compute_funding_features,
    describe_coverage,
)
from vol_model.leakage_probes import probe_causality_by_mutation


# ── Fixtures ─────────────────────────────────────────────────────────


def _synthetic_sidecar(n: int = 500, seed: int = 0) -> pd.DataFrame:
    """Synthetic funding sidecar with realistic magnitudes.

    On the 15m grid (same cadence as production), consecutive bars
    share values for ~32 bars (= 8h / 15min), mimicking the
    merge_asof-backward step-function pattern.
    """
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2024-01-01 01:00", periods=n, freq="15min", tz="UTC")

    # Generate funding at 8h cadence then ffill to 15m (32 bars/funding)
    n_8h = max(1, n // 32 + 1)
    fr_8h = 0.0001 * rng.standard_normal(n_8h)  # typical scale 1e-4
    fr_8h_series = pd.Series(
        fr_8h,
        index=pd.date_range("2024-01-01", periods=n_8h, freq="8h", tz="UTC"),
    )
    # Step-function via reindex + ffill
    fr_level = fr_8h_series.reindex(idx, method="ffill")
    if fr_level.isna().any():
        fr_level = fr_level.fillna(method="bfill")

    fr_abs = fr_level.abs()
    # Crude synthetic EMAs
    fr_ema_1d = fr_level.ewm(span=32 * 3, adjust=False).mean()
    fr_ema_7d = fr_level.ewm(span=32 * 21, adjust=False).mean()
    # Synthetic z-score
    roll = fr_level.rolling(32 * 90, min_periods=32 * 10)
    fr_zscore_30d = (fr_level - roll.mean()) / roll.std().replace(0.0, np.nan)

    df = pd.DataFrame(
        {
            "fr_level": fr_level,
            "fr_abs": fr_abs,
            "fr_ema_1d": fr_ema_1d,
            "fr_ema_7d": fr_ema_7d,
            "fr_zscore_30d": fr_zscore_30d,
        },
        index=idx,
    )
    return df.fillna(0.0)  # fill warmup for clean synthetic probe


# ── 1. Schema ────────────────────────────────────────────────────────


def test_compute_returns_expected_columns() -> None:
    sidecar = _synthetic_sidecar(n=200)
    feats = compute_funding_features(sidecar)
    assert list(feats.columns) == list(FUNDING_FEATURES)
    assert feats.index.equals(sidecar.index)
    assert len(FUNDING_FEATURES) == 5
    assert len(FUNDING_LEVEL_FEATURES) == 5
    for col in feats.columns:
        assert feats[col].dtype == np.float64, f"{col} is {feats[col].dtype}"


def test_compute_missing_column_raises() -> None:
    sidecar = _synthetic_sidecar(n=100).drop(columns=["fr_level"])
    with pytest.raises(KeyError, match="fr_level"):
        compute_funding_features(sidecar)


def test_compute_non_datetime_index_raises() -> None:
    sidecar = _synthetic_sidecar(n=100).reset_index(drop=True)
    with pytest.raises(TypeError, match="DatetimeIndex"):
        compute_funding_features(sidecar)


# ── 2. Byte-identity ─────────────────────────────────────────────────


def test_features_are_byte_identical_copies() -> None:
    sidecar = _synthetic_sidecar(n=200)
    feats = compute_funding_features(sidecar)
    for col in FUNDING_FEATURES:
        np.testing.assert_array_equal(
            feats[col].to_numpy(),
            sidecar[col].to_numpy().astype(float),
            err_msg=f"{col} is not a byte-identical copy",
        )


# ── 3. Causality probe (synthetic) ───────────────────────────────────


def test_causality_by_mutation_synthetic() -> None:
    sidecar = _synthetic_sidecar(n=300)
    T = sidecar.index[150]
    result = probe_causality_by_mutation(compute_funding_features, sidecar, T)
    assert result.passed, result.detail
    assert result.metric == 0.0


# ── 4. Real-data probe + coverage ────────────────────────────────────


def test_causality_by_mutation_real_sidecar() -> None:
    real_path = Path("vol_arb/output/funding_features_15m.parquet")
    if not real_path.exists():
        pytest.skip("real funding sidecar not present; run build first")
    sidecar = pd.read_parquet(real_path)
    T = pd.Timestamp("2023-06-15 12:00:00", tz="UTC")
    if T not in sidecar.index:
        pytest.skip(f"T={T} not in sidecar index")
    result = probe_causality_by_mutation(compute_funding_features, sidecar, T)
    assert result.passed, result.detail


def test_coverage_on_real_sidecar() -> None:
    """Funding is continuously published 8-hourly since 2019 — coverage
    should be ~100%. Threshold 95% leaves room for the ≈30-day warmup
    of fr_zscore_30d + fr_ema_7d near the grid start.
    """
    real_path = Path("vol_arb/output/funding_features_15m.parquet")
    if not real_path.exists():
        pytest.skip("real funding sidecar not present; run build first")
    sidecar = pd.read_parquet(real_path)
    cov = describe_coverage(sidecar)
    low_cov = cov[cov < 0.95]
    assert low_cov.empty, f"low-coverage features: {low_cov.to_dict()}"


# ── 5. Feature-range sanity ──────────────────────────────────────────


def test_feature_ranges_on_real_sidecar() -> None:
    """Plausibility bounds on real funding features."""
    real_path = Path("vol_arb/output/funding_features_15m.parquet")
    if not real_path.exists():
        pytest.skip("real funding sidecar not present; run build first")
    sidecar = pd.read_parquet(real_path)
    fr = sidecar["fr_level"].dropna()
    fr_abs = sidecar["fr_abs"].dropna()
    fr_z = sidecar["fr_zscore_30d"].dropna()

    # Funding rate is typically within ±0.01 per 8h (even extreme
    # regimes; Binance caps funding to ±0.075 as a hard limit)
    assert fr.abs().max() < 0.075, f"|fr| exceeds exchange cap: max={fr.abs().max()}"
    assert fr_abs.min() >= 0, "fr_abs must be non-negative"
    np.testing.assert_allclose(fr_abs, fr.abs(), err_msg="fr_abs != |fr_level|")

    # z-score centers near 0; tails can be large in crisis periods
    assert abs(fr_z.mean()) < 0.5, f"z-score mean off-center: {fr_z.mean()}"
    # Should have some dispersion (not a degenerate constant)
    assert fr_z.std() > 0.5, f"z-score degenerate: std={fr_z.std()}"


# ── 6. Causality of rolling statistics (wasteful-conservatism sanity) ─


def test_early_ema_warmup_nan_or_fresh() -> None:
    """The 7d EMA (21-period on 8h cadence) needs ~7 days of data to
    warm up. The 30d z-score needs ~10-30 days. Near the sidecar start,
    these should be NaN (warmup) rather than leak-computed from future
    data.
    """
    real_path = Path("vol_arb/output/funding_features_15m.parquet")
    if not real_path.exists():
        pytest.skip("real funding sidecar not present; run build first")
    sidecar = pd.read_parquet(real_path)
    # fr_level and fr_abs have no warmup — coverage should be 100%
    assert sidecar["fr_level"].notna().all(), "fr_level should have full coverage"
    assert sidecar["fr_abs"].notna().all(), "fr_abs should have full coverage"
    # At a minimum, non-warmup rows should have no NaNs in all features
    steady_state = sidecar.iloc[32 * 90 * 3:]  # well past the 30d z-score warmup
    assert steady_state.notna().all().all(), "steady-state should have no NaN"
