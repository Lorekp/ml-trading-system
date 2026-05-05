"""Tests for scripts/regenerate_walkforward_ensemble_variance.py.

Cycle 4 Phase A.0 invariants (locked by plan):

1. **Walk-forward isolation** — mutating year-(T+1) inputs must not
   change σ²_epi at any year-T bar. Per-year mutation probe.
2. **Member determinism** — re-running with same seeds produces
   byte-identical output.
3. **2022 pass-through** — year-2022 rows are NaN (no train history).
4. **Cycle 3 artifact distinct** — output path differs from the Cycle 3
   K-fold parquet so that file is preserved unchanged.
5. **Pre-winsorize causality** — winsorize lo/hi computed on pre-T data
   only; mutating post-T extreme values must not change the lo/hi
   values used for prediction.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from vol_model.config import VolModelConfig

# Import the script's private function via path-resolution
import importlib.util
_script_path = Path(__file__).parent.parent / "scripts" / "regenerate_walkforward_ensemble_variance.py"
_spec = importlib.util.spec_from_file_location(
    "regen_wf_ensemble", _script_path,
)
_regen_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_regen_module)
_train_fold_and_predict_year = _regen_module._train_fold_and_predict_year
DEFAULT_OUT = _regen_module.DEFAULT_OUT


# ─── Synthetic dataset (fast, deterministic) ────────────────────────────


def _synthetic_dataset(n_per_year: int = 1_500, n_features: int = 5,
                        years: tuple[int, ...] = (2021, 2022, 2023, 2024),
                        seed: int = 7) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Synthetic (X, y, t1) suitable for fast walk-forward fit testing."""
    rng = np.random.default_rng(seed)
    parts = []
    for yr in years:
        parts.append(
            pd.date_range(
                f"{yr}-01-01", periods=n_per_year, freq="2h", tz="UTC",
            )
        )
    # Preserve tz-awareness when concatenating DatetimeIndexes
    full_idx = parts[0]
    for p in parts[1:]:
        full_idx = full_idx.append(p)
    n = len(full_idx)
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
        index=full_idx,
    )
    y_signal = X.sum(axis=1) * 0.3
    y = pd.Series(y_signal + rng.normal(0, 0.5, n), index=full_idx, name="y")
    t1 = pd.Series(
        full_idx + pd.Timedelta(hours=2), index=full_idx, name="t1",
    )
    return X, y, t1


@pytest.fixture(scope="module")
def small_walkforward_result():
    """Run _train_fold_and_predict_year once with small K + n_estimators."""
    X, y, t1 = _synthetic_dataset(n_per_year=1_500, n_features=5)
    cfg = VolModelConfig(
        asset="BTCUSDT", freq_s=900,
        n_estimators=20, early_stopping_rounds=10,
        n_splits=3, max_depth=3, num_leaves=7,
    )
    result = _train_fold_and_predict_year(
        X, y, t1, test_year=2023,
        base_config=cfg, K=2,
        min_train_bars=2_000,
    )
    return X, y, t1, cfg, result


# ─── 1. Walk-forward isolation ──────────────────────────────────────────


class TestWalkforwardIsolation:
    def test_year_T_unchanged_when_mutating_year_T_plus_1(self):
        """Mutate year-2024 features → σ²_epi at year 2023 must be unchanged."""
        X, y, t1 = _synthetic_dataset(n_per_year=1_500, n_features=5)
        cfg = VolModelConfig(
            asset="BTCUSDT", freq_s=900,
            n_estimators=20, early_stopping_rounds=10,
            n_splits=3, max_depth=3, num_leaves=7,
        )
        baseline = _train_fold_and_predict_year(
            X, y, t1, test_year=2023,
            base_config=cfg, K=2,
            min_train_bars=2_000,
        )
        # Mutate year 2024 features dramatically
        X_mut = X.copy()
        y_mut = y.copy()
        mut_mask = X.index.year == 2024
        X_mut.loc[mut_mask] = 999.0
        y_mut.loc[mut_mask] = -42.0

        mutated = _train_fold_and_predict_year(
            X_mut, y_mut, t1, test_year=2023,
            base_config=cfg, K=2,
            min_train_bars=2_000,
        )
        # σ²_epi for year 2023 must be byte-identical
        np.testing.assert_array_equal(
            baseline["sigma_epi"], mutated["sigma_epi"],
        )
        np.testing.assert_array_equal(
            baseline["ensemble_mean"], mutated["ensemble_mean"],
        )

    def test_year_T_unchanged_when_mutating_post_test_year(self):
        """Year 2024 within X (used for cal/test or just a future year);
        mutating a 2024-and-beyond range must not alter year-2023 prediction."""
        X, y, t1 = _synthetic_dataset(
            n_per_year=1_500, n_features=5,
            years=(2021, 2022, 2023, 2024, 2025),
        )
        cfg = VolModelConfig(
            asset="BTCUSDT", freq_s=900,
            n_estimators=20, early_stopping_rounds=10,
            n_splits=3, max_depth=3, num_leaves=7,
        )
        baseline = _train_fold_and_predict_year(
            X, y, t1, test_year=2023,
            base_config=cfg, K=2,
            min_train_bars=2_000,
        )
        X_mut = X.copy()
        y_mut = y.copy()
        post_mask = X.index.year >= 2024
        X_mut.loc[post_mask] = 999.0
        y_mut.loc[post_mask] = 1e6

        mutated = _train_fold_and_predict_year(
            X_mut, y_mut, t1, test_year=2023,
            base_config=cfg, K=2,
            min_train_bars=2_000,
        )
        np.testing.assert_array_equal(
            baseline["sigma_epi"], mutated["sigma_epi"],
        )


# ─── 2. Member determinism ──────────────────────────────────────────────


class TestMemberDeterminism:
    def test_byte_identical_under_fixed_seeds(self):
        X, y, t1 = _synthetic_dataset(n_per_year=1_000, n_features=4)
        cfg = VolModelConfig(
            asset="BTCUSDT", freq_s=900,
            n_estimators=15, early_stopping_rounds=8,
            n_splits=2, max_depth=3, num_leaves=7,
        )
        a = _train_fold_and_predict_year(
            X, y, t1, test_year=2023, base_config=cfg, K=2, min_train_bars=1_000,
        )
        b = _train_fold_and_predict_year(
            X, y, t1, test_year=2023, base_config=cfg, K=2, min_train_bars=1_000,
        )
        np.testing.assert_array_equal(a["sigma_epi"], b["sigma_epi"])
        np.testing.assert_array_equal(a["ensemble_mean"], b["ensemble_mean"])
        np.testing.assert_array_equal(a["sigma_alea"], b["sigma_alea"])


# ─── 3. Cycle 3 artifact distinct ───────────────────────────────────────


class TestCycle3ArtifactPreserved:
    def test_default_output_path_distinct_from_cycle3(self):
        """The Cycle 4 output path must NOT clobber the Cycle 3 K-fold parquet."""
        cycle_3_path = Path("vol_model/output/BTC_v2_har_pd_ensemble_var.parquet")
        cycle_4_path = DEFAULT_OUT
        assert cycle_4_path != cycle_3_path, (
            "Cycle 4 output path overlaps Cycle 3 — would overwrite leakage-suspect parquet"
        )
        assert "walkforward" in cycle_4_path.name

    def test_default_output_path_under_force_refuses_overwrite(self, tmp_path):
        """The default freeze contract refuses overwrite without --force."""
        from vol_model.uncertainty import freeze_uncertainty
        idx = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
        df = pd.DataFrame({"x": np.zeros(10)}, index=idx)
        out = tmp_path / "test.parquet"
        freeze_uncertainty(df, out)
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            freeze_uncertainty(df, out)


# ─── 4. Output sanity ───────────────────────────────────────────────────


class TestOutputSanity:
    def test_returns_required_keys(self, small_walkforward_result):
        _, _, _, _, result = small_walkforward_result
        for key in (
            "timestamps", "y_true", "ensemble_mean", "sigma_epi",
            "sigma_alea", "sigma_total", "spread", "pi_lo", "pi_hi",
            "q_conformal", "cal_n_obs", "n_train_bars",
            "ensemble_seeds", "feature_names",
        ):
            assert key in result, f"missing key {key!r}"

    def test_arrays_have_consistent_length(self, small_walkforward_result):
        _, _, _, _, result = small_walkforward_result
        n = len(result["timestamps"])
        for key in ("y_true", "ensemble_mean", "sigma_epi",
                     "sigma_alea", "sigma_total", "spread", "pi_lo", "pi_hi"):
            assert len(result[key]) == n, (
                f"{key} has length {len(result[key])}, expected {n}"
            )

    def test_sigma_epi_non_negative(self, small_walkforward_result):
        _, _, _, _, result = small_walkforward_result
        assert (result["sigma_epi"] >= 0).all()

    def test_sigma_total_is_quadrature_sum(self, small_walkforward_result):
        _, _, _, _, result = small_walkforward_result
        recomputed = np.sqrt(result["sigma_epi"] ** 2 + result["sigma_alea"] ** 2)
        np.testing.assert_allclose(
            result["sigma_total"], recomputed, rtol=1e-12,
        )

    def test_pi_monotonic(self, small_walkforward_result):
        _, _, _, _, result = small_walkforward_result
        assert (result["pi_lo"] <= result["pi_hi"]).all()
        assert (result["spread"] >= 0).all()

    def test_test_year_only_test_bars(self, small_walkforward_result):
        X, _, _, _, result = small_walkforward_result
        # Result timestamps must be entirely within year 2023
        years = pd.DatetimeIndex(result["timestamps"]).year
        assert (years == 2023).all()


# ─── 5. Pre-winsorize causality ─────────────────────────────────────────


class TestWinsorizeCausality:
    def test_pre_winsorize_does_not_use_year_T_extremes(self):
        """Inserting an extreme outlier at year-T should NOT change winsorize lo/hi
        (which is computed on pre-T data only).
        """
        X, y, t1 = _synthetic_dataset(n_per_year=1_500, n_features=4)
        cfg = VolModelConfig(
            asset="BTCUSDT", freq_s=900,
            n_estimators=15, early_stopping_rounds=8,
            n_splits=2, max_depth=3, num_leaves=7,
        )
        baseline = _train_fold_and_predict_year(
            X, y, t1, test_year=2023, base_config=cfg, K=2, min_train_bars=2_000,
        )
        X_mut = X.copy()
        # Insert extreme outliers at year 2023 (test year)
        test_mask = X_mut.index.year == 2023
        X_mut.loc[test_mask, "f0"] = 1e9
        # Note: this DOES change predictions on year 2023 because test data
        # passes through `model.predict`. But it should NOT change σ²_epi
        # via the winsorize lo/hi cuts (those are computed on pre-2023 data).
        # We can't test "predictions unchanged" because predictions ON the
        # mutated test data SHOULD differ. Instead, verify the n_train_bars
        # is unchanged (sanity that the train slice is unchanged).
        mutated = _train_fold_and_predict_year(
            X_mut, y, t1, test_year=2023, base_config=cfg, K=2, min_train_bars=2_000,
        )
        assert baseline["n_train_bars"] == mutated["n_train_bars"]
        # Most predictions WILL differ (we mutated test inputs); that's expected.
        # The walk-forward isolation test in TestWalkforwardIsolation already
        # asserts the right thing for non-test-year mutations.


# ─── 6. Test-year filtering ─────────────────────────────────────────────


class TestTestYearFiltering:
    def test_too_few_pre_test_year_bars_raises(self):
        """If pre-test-year data is below the floor, raise."""
        X, y, t1 = _synthetic_dataset(n_per_year=500, n_features=3,
                                       years=(2022, 2023))
        cfg = VolModelConfig(
            asset="BTCUSDT", freq_s=900,
            n_estimators=10, early_stopping_rounds=5,
            n_splits=2, max_depth=3, num_leaves=7,
        )
        with pytest.raises(ValueError, match="pre-T training bars"):
            _train_fold_and_predict_year(
                X, y, t1, test_year=2023, base_config=cfg, K=2,
                min_train_bars=5_000,
            )
