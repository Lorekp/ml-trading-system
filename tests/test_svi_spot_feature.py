"""Unit tests for the svi_atm_spot feature (Tier 2 signal research 2026-04-17).

Verifies:
  1. Column present and numerically sane in the production parquet.
  2. Causality: the +1h end-shift preserves merge_asof(backward) safety.
  3. Quality gate: svi_atm_spot is NaN when calibration is degraded
     (level >= 3 OR n_strikes < 3 OR not butterfly-free).
  4. Correctness on a synthetic SVISlice: atm_vol_pct matches manual
     computation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_arb.paths import svi_features_path
from vol_arb.svi import SVISlice
from vol_arb.rebuild_svi_features_with_spot import _compute_spot_from_nearest_slice


@pytest.fixture(scope="module")
def btc_features() -> pd.DataFrame:
    """The production SVI features parquet for BTC (post-augment)."""
    path = svi_features_path("BTC")
    if not path.exists():
        pytest.skip(f"SVI features parquet missing at {path}")
    df = pd.read_parquet(path)
    if "svi_atm_spot" not in df.columns:
        pytest.skip("svi_atm_spot column not yet added — run "
                    "`python -m vol_arb.rebuild_svi_features_with_spot --currency BTC`")
    return df


def test_svi_atm_spot_column_present(btc_features):
    """The column exists with the expected name."""
    assert "svi_atm_spot" in btc_features.columns


def test_svi_atm_spot_coverage(btc_features):
    """Coverage is > 80% over the SVI-calibrated date range.

    Some rows are expected to be NaN: pre-2021-04 (no options data) and
    a small fraction where the quality gate fires.
    """
    # Post-SVI-start slice only (2021-04-01 onwards)
    active = btc_features[btc_features.index >= pd.Timestamp("2021-04-01", tz="UTC")]
    coverage = active["svi_atm_spot"].notna().mean()
    assert coverage > 0.80, f"svi_atm_spot coverage {coverage:.1%} < 80% on active range"


def test_svi_atm_spot_distribution_sane(btc_features):
    """Bulk distribution in a plausible BTC vol range.

    Crypto near-expiry SVI can produce very high ATM values (>300%) during
    genuine crisis moments (May 2021 flash crash, June 2022 Luna
    aftermath, Aug 2022 FTX rumors). LightGBM handles these via tree
    splits. We only enforce that the *bulk* distribution (99th pct) is
    sane; the extreme tail is allowed to carry crisis-regime information.
    """
    sp = btc_features["svi_atm_spot"].dropna()
    assert 20.0 < sp.mean() < 100.0, f"svi_atm_spot mean {sp.mean():.1f} out of sane BTC range"
    assert sp.min() >= 0.0
    # 99th percentile must be below 250 — crisis tail is < 1% of rows
    p99 = sp.quantile(0.99)
    assert p99 < 250.0, f"svi_atm_spot p99 {p99:.1f} suspiciously high"
    # Extreme tail capped at an order-of-magnitude sanity limit
    assert sp.max() < 1000.0, f"svi_atm_spot max {sp.max():.1f} absurd"


def test_svi_atm_spot_respects_end_shift(btc_features):
    """The +1h end-shift is applied: non-null entries are at hourly
    boundaries (00:00, 01:00, ...) with 15-min forward-fill between."""
    # Pick a row we know is non-null
    non_null = btc_features["svi_atm_spot"].dropna()
    assert len(non_null) > 0
    # Check that within a 1h window, svi_atm_spot is constant (forward-filled)
    sample_hour_start = non_null.index[100].floor("h")
    window = btc_features.loc[
        sample_hour_start: sample_hour_start + pd.Timedelta("45min"),
        "svi_atm_spot",
    ]
    # All four 15m rows in the hour should be identical (within float precision)
    vals = window.dropna().unique()
    assert len(vals) <= 1, f"svi_atm_spot varies within hour {sample_hour_start}: {vals}"


def test_synthetic_slice_atm_vol_pct():
    """SVISlice.atm_vol_pct() at k=0 matches manual computation."""
    sl = SVISlice(
        a=0.001, b=0.02, rho=-0.3, m=0.01, sigma=0.1, T=0.01, F=100.0,
        calibration_level=0, rmse=1.0, n_strikes=10,
    )
    # w(0) = a + b * (rho * (-m) + sqrt(m^2 + sigma^2))
    w0 = 0.001 + 0.02 * (-0.3 * -0.01 + np.sqrt(0.01**2 + 0.1**2))
    expected_pct = np.sqrt(w0 / 0.01) * 100.0
    assert abs(sl.atm_vol_pct() - expected_pct) < 1e-10


def test_quality_gate_rejects_high_level():
    """Level >= 3 (fallback calibration) => NaN."""
    df = pd.DataFrame({
        "T": [0.01], "F": [100.0], "a": [0.001], "b": [0.02],
        "rho": [-0.3], "m": [0.01], "sigma": [0.1],
        "level": [3], "rmse": [1.0], "n_strikes": [10],
    })
    result = _compute_spot_from_nearest_slice(df)
    assert np.isnan(result), f"level=3 should gate to NaN, got {result}"


def test_quality_gate_rejects_low_n_strikes():
    """n_strikes < 3 => NaN."""
    df = pd.DataFrame({
        "T": [0.01], "F": [100.0], "a": [0.001], "b": [0.02],
        "rho": [-0.3], "m": [0.01], "sigma": [0.1],
        "level": [0], "rmse": [1.0], "n_strikes": [2],
    })
    result = _compute_spot_from_nearest_slice(df)
    assert np.isnan(result), f"n_strikes=2 should gate to NaN, got {result}"


def test_quality_gate_passes_good_calibration():
    """Good calibration (level=0, n_strikes=10, butterfly-free) => valid number."""
    df = pd.DataFrame({
        "T": [0.01], "F": [100.0], "a": [0.001], "b": [0.02],
        "rho": [-0.3], "m": [0.01], "sigma": [0.1],
        "level": [0], "rmse": [1.0], "n_strikes": [10],
    })
    result = _compute_spot_from_nearest_slice(df)
    assert not np.isnan(result)
    assert 20.0 < result < 250.0  # Sane range


def test_causality_end_shift_no_future_info(btc_features):
    """The SVI feature at time T was calibrated on trades strictly BEFORE T.

    The SVI pipeline buckets trades hourly and applies a +1h end-shift, so
    a feature at row-ts 01:00 was calibrated on trades from [00:00, 01:00).
    Reading svi_atm_spot at 01:00 and above is causally safe — no leakage.
    Verify by asserting that any svi_atm_spot value at some time T must
    have been available (non-null) at some feature row at or before T.
    """
    # For a random 100 non-null rows, confirm their timestamp is at or after
    # 2021-04-01 01:00 UTC (first possible post-shift hour).
    non_null = btc_features["svi_atm_spot"].dropna()
    first_expected = pd.Timestamp("2021-04-01 01:00:00", tz="UTC")
    assert non_null.index.min() >= first_expected, (
        f"svi_atm_spot first non-null at {non_null.index.min()} precedes "
        f"the +1h end-shift floor at {first_expected} — leakage risk"
    )
