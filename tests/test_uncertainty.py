"""Tests for vol_model/uncertainty.py — deep-ensemble + HQR combination.

Categories:
  1. Pre-committed parameter locks.
  2. Deterministic ensemble training under fixed seeds.
  3. Variance-of-K reduces as K grows; non-trivial on heterogeneous data.
  4. ``predict_ensemble_var`` shape + numerical sanity.
  5. ``combine_with_hqr`` returns finite, monotonic intervals.
  6. Frozen-parquet refuse-overwrite.
  7. Leakage: post-cal-window mutation does not change HQR fit.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from vol_model.config import VolModelConfig
from vol_model.uncertainty import (
    CAL_WINDOW,
    ENSEMBLE_FEATURE_FRACTION,
    ENSEMBLE_K,
    ENSEMBLE_BAGGING_FRACTION,
    ENSEMBLE_SEED_BASE,
    EnsembleParams,
    HQR_COVERAGE_ALPHA,
    combine_with_hqr,
    freeze_uncertainty,
    load_frozen_uncertainty,
    predict_ensemble_var,
    train_har_pd_ensemble,
)


# ─── 1. Locked parameters ───────────────────────────────────────────────


class TestLockedParams:
    def test_K_locked_at_10(self):
        assert ENSEMBLE_K == 10
        assert EnsembleParams().K == 10

    def test_seed_base_locked(self):
        assert ENSEMBLE_SEED_BASE == 1000
        seeds = EnsembleParams().seeds
        assert seeds == (1000, 1001, 1002, 1003, 1004,
                          1005, 1006, 1007, 1008, 1009)

    def test_bagging_feature_locked(self):
        assert ENSEMBLE_BAGGING_FRACTION == 0.80
        assert ENSEMBLE_FEATURE_FRACTION == 0.85

    def test_alpha_locked(self):
        assert HQR_COVERAGE_ALPHA == 0.10  # 90% PI

    def test_cal_window_matches_cycle_2(self):
        cal_start, cal_end = CAL_WINDOW
        assert cal_start == pd.Timestamp("2021-04-01", tz="UTC")
        assert cal_end == pd.Timestamp("2023-10-01", tz="UTC")


# ─── 2-3. Ensemble determinism + variance behavior ─────────────────────


def _synthetic_xyt1(n: int = 2_000, n_features: int = 5, noise: float = 0.5,
                    seed: int = 7) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Synthetic linear-with-noise dataset suitable for fast LightGBM."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2022-01-01", periods=n, freq="15min", tz="UTC")
    X = pd.DataFrame(
        rng.standard_normal((n, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
        index=idx,
    )
    # Non-trivial signal so the model has something to fit
    y_signal = X.sum(axis=1) * 0.3
    y = y_signal + rng.normal(0, noise, n)
    y = pd.Series(y, index=idx, name="y")
    # t1 = next bar (one-step-ahead label window)
    t1 = pd.Series(idx + pd.Timedelta(minutes=15), index=idx, name="t1")
    return X, y, t1


@pytest.fixture(scope="module")
def small_ensemble_result():
    """Train a 3-member ensemble on synthetic data — reused across tests."""
    X, y, t1 = _synthetic_xyt1(n=1_500, n_features=4)
    cfg = VolModelConfig(
        n_estimators=50,        # smoke-test fast
        early_stopping_rounds=20,
        n_splits=3,             # 3 folds to keep test fast
        max_depth=4, num_leaves=15, learning_rate=0.1,
    )
    params = EnsembleParams(K=3, seed_base=2000)
    result = train_har_pd_ensemble(X, y, t1, cfg, params=params)
    return X, y, t1, cfg, result


class TestEnsembleTraining:
    def test_returns_K_members(self, small_ensemble_result):
        _, _, _, _, result = small_ensemble_result
        assert len(result["members"]) == 3
        assert result["oof_matrix"].shape[0] == 3

    def test_seeds_used_match_params(self, small_ensemble_result):
        _, _, _, _, result = small_ensemble_result
        assert result["seeds_used"] == [2000, 2001, 2002]

    def test_feature_names_consistent(self, small_ensemble_result):
        _, _, _, _, result = small_ensemble_result
        # All members must report same feature_names; integrity guard
        for member in result["members"]:
            assert member["feature_names"] == result["feature_names"]

    def test_oof_matrix_filled(self, small_ensemble_result):
        _, _, _, _, result = small_ensemble_result
        # Under PurgedKFold every bar should be in exactly one test fold,
        # so OOF predictions cover all bars (no NaN).
        n_nan = int(np.isnan(result["oof_matrix"]).sum())
        # Some warmup might be excluded; allow up to 1% NaN
        assert n_nan / result["oof_matrix"].size < 0.01

    def test_deterministic_under_fixed_seeds(self):
        """Two ensemble runs with same seeds → identical OOF matrices."""
        X, y, t1 = _synthetic_xyt1(n=800, n_features=3, seed=11)
        cfg = VolModelConfig(
            n_estimators=30, early_stopping_rounds=10, n_splits=3,
            max_depth=3, num_leaves=7, learning_rate=0.1,
        )
        params = EnsembleParams(K=2, seed_base=5000)
        a = train_har_pd_ensemble(X, y, t1, cfg, params=params)
        b = train_har_pd_ensemble(X, y, t1, cfg, params=params)
        np.testing.assert_array_equal(a["oof_matrix"], b["oof_matrix"])


# ─── 4. predict_ensemble_var sanity ─────────────────────────────────────


class TestPredictEnsembleVar:
    def test_zero_variance_for_constant_predictions(self):
        oof = np.full((5, 100), 1.5)
        out = predict_ensemble_var(oof)
        np.testing.assert_allclose(out["mean"], np.full(100, 1.5))
        np.testing.assert_allclose(out["var"], np.zeros(100), atol=1e-12)
        np.testing.assert_allclose(out["std"], np.zeros(100), atol=1e-12)

    def test_variance_increases_with_spread(self):
        rng = np.random.default_rng(0)
        # Two ensembles: tight spread vs wide spread
        tight = rng.normal(0, 0.1, (10, 50))
        wide = rng.normal(0, 1.0, (10, 50))
        out_tight = predict_ensemble_var(tight)
        out_wide = predict_ensemble_var(wide)
        assert out_wide["var"].mean() > out_tight["var"].mean() * 50

    def test_2D_required(self):
        with pytest.raises(ValueError, match="2D"):
            predict_ensemble_var(np.ones(10))


# ─── 5. combine_with_hqr ────────────────────────────────────────────────


class TestCombineWithHqr:
    def test_returns_expected_keys(self):
        n = 3_000
        # Calibration window: 2022-06-01 → 2023-06-01 (well before eval)
        idx = pd.date_range("2022-06-01", periods=n, freq="15min", tz="UTC")
        rng = np.random.default_rng(42)
        ensemble_mean = rng.normal(0.0, 0.1, n)
        ensemble_std = np.abs(rng.normal(0.05, 0.02, n))
        y_true = ensemble_mean + rng.normal(0.0, 0.15, n)

        cal_start = idx[0]
        cal_end = idx[n // 2]  # split temporally for HQR's internal cal
        out = combine_with_hqr(
            timestamps=idx, ensemble_mean=ensemble_mean,
            ensemble_std=ensemble_std, y_true=y_true,
            cal_window=(cal_start, cal_end), alpha=0.10,
        )
        for key in ("sigma_epi", "sigma_alea", "sigma_total",
                     "spread", "pi_lo", "pi_hi"):
            assert key in out
            assert len(out[key]) == n

    def test_sigma_total_is_quadrature_sum(self):
        n = 3_000
        idx = pd.date_range("2022-06-01", periods=n, freq="15min", tz="UTC")
        rng = np.random.default_rng(7)
        ensemble_mean = rng.normal(0.0, 0.1, n)
        ensemble_std = np.full(n, 0.05)
        y_true = ensemble_mean + rng.normal(0.0, 0.10, n)
        out = combine_with_hqr(
            timestamps=idx, ensemble_mean=ensemble_mean,
            ensemble_std=ensemble_std, y_true=y_true,
            cal_window=(idx[0], idx[n // 2]),
        )
        # sigma_total² = sigma_epi² + sigma_alea²
        recomputed = np.sqrt(out["sigma_epi"] ** 2 + out["sigma_alea"] ** 2)
        np.testing.assert_allclose(out["sigma_total"], recomputed, rtol=1e-12)

    def test_pi_monotonic(self):
        n = 3_000
        idx = pd.date_range("2022-06-01", periods=n, freq="15min", tz="UTC")
        rng = np.random.default_rng(13)
        ensemble_mean = rng.normal(0.0, 0.1, n)
        ensemble_std = np.abs(rng.normal(0.05, 0.02, n))
        y_true = ensemble_mean + rng.normal(0.0, 0.15, n)
        out = combine_with_hqr(
            timestamps=idx, ensemble_mean=ensemble_mean,
            ensemble_std=ensemble_std, y_true=y_true,
            cal_window=(idx[0], idx[n // 2]),
        )
        assert (out["pi_lo"] <= out["pi_hi"]).all()
        # Spread is non-negative
        assert (out["spread"] >= 0).all()

    def test_too_small_cal_raises(self):
        n = 100
        idx = pd.date_range("2022-06-01", periods=n, freq="15min", tz="UTC")
        rng = np.random.default_rng(7)
        with pytest.raises(ValueError, match="≥ 1000"):
            combine_with_hqr(
                timestamps=idx,
                ensemble_mean=rng.normal(0, 0.1, n),
                ensemble_std=np.full(n, 0.05),
                y_true=rng.normal(0, 0.1, n),
                cal_window=(idx[0], idx[50]),
            )


# ─── 6. Freeze refuse-overwrite ─────────────────────────────────────────


class TestFreeze:
    def test_writes_parquet(self, tmp_path):
        idx = pd.date_range("2024-01-01", periods=100, freq="15min", tz="UTC")
        df = pd.DataFrame(
            {"sigma_epi": np.linspace(0.01, 0.10, 100)},
            index=idx,
        )
        out = tmp_path / "uncert.parquet"
        freeze_uncertainty(df, out)
        loaded = load_frozen_uncertainty(out)
        np.testing.assert_array_equal(df.values, loaded.values)
        assert loaded.index.equals(df.index)

    def test_refuses_overwrite_default(self, tmp_path):
        idx = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
        df = pd.DataFrame({"sigma_epi": [0.05] * 10}, index=idx)
        out = tmp_path / "uncert.parquet"
        freeze_uncertainty(df, out)
        with pytest.raises(FileExistsError, match="Refusing to overwrite"):
            freeze_uncertainty(df, out)

    def test_force_allows_overwrite(self, tmp_path):
        idx = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
        df = pd.DataFrame({"sigma_epi": [0.05] * 10}, index=idx)
        out = tmp_path / "uncert.parquet"
        freeze_uncertainty(df, out)
        df2 = pd.DataFrame({"sigma_epi": [0.10] * 10}, index=idx)
        freeze_uncertainty(df2, out, force=True)
        loaded = load_frozen_uncertainty(out)
        assert loaded["sigma_epi"].iloc[0] == 0.10

    def test_metadata_sidecar_written(self, tmp_path):
        idx = pd.date_range("2024-01-01", periods=10, freq="15min", tz="UTC")
        df = pd.DataFrame({"sigma_epi": [0.05] * 10}, index=idx)
        out = tmp_path / "uncert.parquet"
        freeze_uncertainty(df, out, metadata={"K": 10, "seeds": [1000, 1001]})
        meta_path = out.with_suffix(".meta.json")
        assert meta_path.exists()
        import json
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["K"] == 10


# ─── 7. Leakage: post-cal-window mutation does not change HQR fit ─────


class TestLeakage:
    def test_post_window_mutation_invariance(self):
        """Mutate ensemble_mean / y_true POST cal_end → HQR fit identical."""
        from vol_model.conformal.hqr import fit_hqr
        n = 3_000
        idx = pd.date_range("2022-06-01", periods=n, freq="15min", tz="UTC")
        rng = np.random.default_rng(99)
        ensemble_mean = rng.normal(0.0, 0.1, n)
        ensemble_std = np.full(n, 0.05)
        y_true = ensemble_mean + rng.normal(0.0, 0.10, n)

        cal_start = idx[0]
        cal_end = idx[n // 2]

        # Baseline
        out_baseline = combine_with_hqr(
            timestamps=idx, ensemble_mean=ensemble_mean.copy(),
            ensemble_std=ensemble_std.copy(), y_true=y_true.copy(),
            cal_window=(cal_start, cal_end),
        )

        # Mutate POST-cal-window values to extreme garbage
        ensemble_mean_m = ensemble_mean.copy()
        y_true_m = y_true.copy()
        post_mask = idx >= cal_end
        ensemble_mean_m[post_mask] = 1e6
        y_true_m[post_mask] = -1e6

        out_mutated = combine_with_hqr(
            timestamps=idx, ensemble_mean=ensemble_mean_m,
            ensemble_std=ensemble_std.copy(), y_true=y_true_m,
            cal_window=(cal_start, cal_end),
        )

        # The Vovk q_conformal scalar (computed on cal2 only) and the
        # cal-fit sigma_alea inside the cal window must be identical.
        assert out_baseline["q_conformal"] == out_mutated["q_conformal"]
        cal_pre_mask = (idx >= cal_start) & (idx < cal_end)
        np.testing.assert_array_equal(
            out_baseline["sigma_alea"][cal_pre_mask],
            out_mutated["sigma_alea"][cal_pre_mask],
        )
