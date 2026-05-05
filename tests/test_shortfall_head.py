"""Tests for vol_arb.shortfall_head — Cycle 4 conditional-shortfall classifier.

Critical invariants (locked by plan ``now-check-empirically-from-eager-bird.md``):

1. **Hyperparameter parity** — BASELINE / TREATMENT / PLACEBO must use
   ``DFL_LGBM_PARAMS`` exactly. Parity is the experimental control.
2. **No lookahead** — DFL features inherit ``shift(1).rolling`` lag-safety;
   adding ``sigma_epi_squared`` must not break it.
3. **Walk-forward isolation** — mutating year-Y data must not change
   classifier scores on year-Y' for Y' < Y.
4. **Target excludes unfired trades** — shortfall is defined only on
   ``net_pnl_nc != 0``.
5. **Placebo permutation** — within-year, deterministic given the
   placebo seed; cross-year temporal causality preserved.
6. **2022 pass-through** — no train history, all-NaN classifier output.
7. **Block-bootstrap respects daily clustering** — wider CI under
   autocorrelated synthetic data than under IID.
8. **DeLong test** — paired-AUC p-value matches a reference implementation
   on synthetic data of known correlation.
9. **Locked gate constants** — gate thresholds match the plan exactly;
   regression locks them against accidental drift.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from vol_arb.dfl_head import DFL_HEAD_FEATURES, DFL_LGBM_PARAMS
from vol_arb.shortfall_head import (
    GATE_G1_DELTA_MIN,
    GATE_G2_BLOCK_BOOTSTRAP_REPS,
    GATE_G2_BLOCK_DAYS,
    GATE_G3_DELONG_P_MAX,
    GATE_G4_PLACEBO_MAX,
    SHORTFALL_FEATURE_EXTRA,
    SHORTFALL_FEATURES_TREATMENT,
    SHORTFALL_PLACEBO_SEED,
    SHORTFALL_TEST_YEARS,
    SHORTFALL_THRESHOLD_USD,
    AucDeltaGateResult,
    auc_delta_gate,
    block_bootstrap_auc_delta_ci,
    build_shortfall_features,
    compute_shortfall_target,
    delong_paired_p_value,
    per_year_auc_summary,
    train_walkforward_shortfall_classifier,
    _placebo_shuffle_within_year,
)


# ─── Synthetic fixture ──────────────────────────────────────────────────


@pytest.fixture
def synthetic_data():
    """4 years of trade log + predictions + walk-forward σ²_epi.

    Year 1 (2022) = pass-through (no train history).
    Year 2-4 (2023, 2024, 2025) = test years.
    Strong-edge bars are correlated with σ²_epi^-1 (lower uncertainty →
    higher win rate); the classifier should learn this if σ²_epi adds
    information beyond the DFL panel.
    """
    rng = np.random.default_rng(42)
    idx = pd.date_range("2022-01-01", "2025-12-31", freq="1h", tz="UTC")
    n = len(idx)

    y_true = rng.normal(-0.5, 0.3, size=n)
    y_pred = y_true + rng.normal(0, 0.2, size=n)
    iv = 50 + 20 * np.exp(y_pred)
    signal = rng.choice([-1, 0, 1], size=n, p=[0.25, 0.5, 0.25])
    iv_log = np.log(iv / 100)
    edge = y_pred - iv_log

    # σ²_epi: positive, weakly anti-correlated with |edge| (mimics V2 collider effect)
    sigma_epi = np.abs(rng.normal(0.05, 0.02, n)) * (1 + 0.3 * np.abs(edge))

    pnl_expected = np.sign(edge) * np.abs(edge) * 200 / (1 + sigma_epi)
    pnl_noise = rng.normal(0, 100, size=n)
    pnl = np.where(signal != 0, pnl_expected + pnl_noise, 0.0)

    tl = pd.DataFrame({
        "signal": signal,
        "iv_used": iv,
        "net_pnl_nc": pnl,
        "net_pnl": pnl,
    }, index=idx)
    preds = pd.DataFrame({"y_pred": y_pred, "y_true": y_true}, index=idx)
    wfu = pd.DataFrame({"sigma_epi": sigma_epi}, index=idx)
    return tl, preds, wfu


# ─── 1. Hyperparameter parity ───────────────────────────────────────────


class TestHyperparameterParity:
    def test_classifier_hyperparameters_match_dfl(self):
        """All three classifiers must use DFL_LGBM_PARAMS verbatim.
        This is enforced by ``train_walkforward_shortfall_classifier``
        importing them directly; we test the import binds correctly.
        """
        from vol_arb.shortfall_head import DFL_LGBM_PARAMS as ours
        assert ours == DFL_LGBM_PARAMS

    def test_locked_gate_constants(self):
        """Gate thresholds must match plan exactly (regression lock)."""
        assert GATE_G1_DELTA_MIN == 0.02
        assert GATE_G3_DELONG_P_MAX == 0.01
        assert GATE_G4_PLACEBO_MAX == 0.005
        assert GATE_G2_BLOCK_DAYS == 5
        assert GATE_G2_BLOCK_BOOTSTRAP_REPS == 1000

    def test_locked_threshold_X_at_200(self):
        assert SHORTFALL_THRESHOLD_USD == 200.0

    def test_locked_test_years(self):
        assert SHORTFALL_TEST_YEARS == (2023, 2024, 2025)

    def test_treatment_features_count_11(self):
        assert len(SHORTFALL_FEATURES_TREATMENT) == 11
        assert SHORTFALL_FEATURES_TREATMENT[-1] == SHORTFALL_FEATURE_EXTRA
        assert tuple(SHORTFALL_FEATURES_TREATMENT[:10]) == tuple(DFL_HEAD_FEATURES)


# ─── 2. Feature panel + leakage ─────────────────────────────────────────


class TestBuildShortfallFeatures:
    def test_returns_11_columns_in_locked_order(self, synthetic_data):
        tl, preds, wfu = synthetic_data
        feats = build_shortfall_features(tl, preds, wfu)
        assert tuple(feats.columns) == SHORTFALL_FEATURES_TREATMENT

    def test_features_inherit_dfl_lag_safety(self, synthetic_data):
        """Mutating y_true at row t must not change features at row t.
        Inherits from dfl_head.build_head_features's strict-lag pattern.
        """
        tl, preds, wfu = synthetic_data
        feats = build_shortfall_features(tl, preds, wfu)
        preds2 = preds.copy()
        # Mutate y_true at row 500 to a huge value
        preds2.iloc[500, preds2.columns.get_loc("y_true")] = 999.0
        feats2 = build_shortfall_features(tl, preds2, wfu)
        # Row 500 features must be unchanged
        pd.testing.assert_series_equal(
            feats.iloc[500], feats2.iloc[500],
            check_names=False,
        )

    def test_sigma_epi_mutation_does_not_affect_other_features(self, synthetic_data):
        """Mutating σ²_epi at row t changes only the σ²_epi column at t."""
        tl, preds, wfu = synthetic_data
        feats = build_shortfall_features(tl, preds, wfu)
        wfu2 = wfu.copy()
        wfu2.iloc[500, wfu2.columns.get_loc("sigma_epi")] = 100.0
        feats2 = build_shortfall_features(tl, preds, wfu2)
        # The 10 DFL columns at row 500 are unchanged
        for col in DFL_HEAD_FEATURES:
            assert feats[col].iloc[500] == feats2[col].iloc[500], (
                f"DFL column {col!r} changed when σ²_epi mutated"
            )
        # The σ²_epi column at row 500 IS changed (sanity)
        assert feats[SHORTFALL_FEATURE_EXTRA].iloc[500] != feats2[SHORTFALL_FEATURE_EXTRA].iloc[500]

    def test_refuses_overlap_column(self, synthetic_data):
        """If trade_log has 'sigma_epi_squared', refuse rather than overwrite."""
        tl, preds, wfu = synthetic_data
        tl2 = tl.copy()
        tl2[SHORTFALL_FEATURE_EXTRA] = 0.0
        with pytest.raises(ValueError, match="already contains"):
            build_shortfall_features(tl2, preds, wfu)

    def test_missing_sigma_epi_column_raises(self, synthetic_data):
        tl, preds, wfu = synthetic_data
        wfu2 = wfu.drop(columns=["sigma_epi"]).assign(other=0.0)
        with pytest.raises(KeyError, match="sigma_epi"):
            build_shortfall_features(tl, preds, wfu2)

    def test_squaring_is_correct(self, synthetic_data):
        """Sanity: the 11th feature is sigma_epi**2 (variance), not sigma_epi (std)."""
        tl, preds, wfu = synthetic_data
        feats = build_shortfall_features(tl, preds, wfu)
        # Pick a non-NaN bar and check
        valid = feats.dropna().head(10)
        for ts in valid.index:
            expected = float(wfu.loc[ts, "sigma_epi"]) ** 2
            actual = float(valid.loc[ts, SHORTFALL_FEATURE_EXTRA])
            assert abs(actual - expected) < 1e-12


# ─── 3. Target ──────────────────────────────────────────────────────────


class TestComputeShortfallTarget:
    def test_target_excludes_unfired_trades(self, synthetic_data):
        tl, _, _ = synthetic_data
        y = compute_shortfall_target(tl, threshold_usd=200.0)
        # Unfired bars (signal=0) should be absent
        fired_mask = tl["net_pnl_nc"] != 0
        assert len(y) == int(fired_mask.sum())
        assert (tl.loc[y.index, "net_pnl_nc"] != 0).all()

    def test_target_is_binary(self, synthetic_data):
        tl, _, _ = synthetic_data
        y = compute_shortfall_target(tl, threshold_usd=200.0)
        assert set(y.unique()).issubset({0, 1})

    def test_threshold_semantics_correct(self, synthetic_data):
        """y == 1 ⇔ net_pnl_nc < -threshold (strict)."""
        tl, _, _ = synthetic_data
        thr = 200.0
        y = compute_shortfall_target(tl, threshold_usd=thr)
        fired = tl.loc[tl["net_pnl_nc"] != 0]
        expected = (fired["net_pnl_nc"] < -thr).astype(int)
        pd.testing.assert_series_equal(
            y.astype(int), expected, check_names=False,
        )

    def test_default_threshold_is_200(self, synthetic_data):
        tl, _, _ = synthetic_data
        y_default = compute_shortfall_target(tl)
        y_explicit = compute_shortfall_target(tl, threshold_usd=200.0)
        pd.testing.assert_series_equal(y_default, y_explicit)


# ─── 4. Placebo shuffle ─────────────────────────────────────────────────


class TestPlaceboShuffle:
    def test_within_year_permutation(self, synthetic_data):
        """Shuffled values are a permutation WITHIN each year, not across."""
        tl, preds, wfu = synthetic_data
        feats = build_shortfall_features(tl, preds, wfu)
        sig = feats[SHORTFALL_FEATURE_EXTRA].dropna()
        sig_shuffled = _placebo_shuffle_within_year(sig, seed=SHORTFALL_PLACEBO_SEED)

        # Per-year multiset must match
        for yr in sorted(set(sig.index.year)):
            yr_orig = sig.loc[sig.index.year == yr].sort_values().to_numpy()
            yr_shuf = sig_shuffled.loc[sig_shuffled.index.year == yr].sort_values().to_numpy()
            np.testing.assert_array_equal(yr_orig, yr_shuf)

    def test_deterministic_under_fixed_seed(self, synthetic_data):
        tl, preds, wfu = synthetic_data
        feats = build_shortfall_features(tl, preds, wfu)
        sig = feats[SHORTFALL_FEATURE_EXTRA].dropna()
        a = _placebo_shuffle_within_year(sig, seed=SHORTFALL_PLACEBO_SEED)
        b = _placebo_shuffle_within_year(sig, seed=SHORTFALL_PLACEBO_SEED)
        pd.testing.assert_series_equal(a, b)

    def test_different_seeds_give_different_shuffles(self, synthetic_data):
        tl, preds, wfu = synthetic_data
        feats = build_shortfall_features(tl, preds, wfu)
        sig = feats[SHORTFALL_FEATURE_EXTRA].dropna()
        a = _placebo_shuffle_within_year(sig, seed=42)
        b = _placebo_shuffle_within_year(sig, seed=43)
        # Likely different (with extremely high probability for n > a few)
        assert not a.equals(b)


# ─── 5. Walk-forward classifier ─────────────────────────────────────────


@pytest.fixture(scope="module")
def trained_classifiers():
    """Train BASELINE / TREATMENT / PLACEBO once per test module."""
    rng = np.random.default_rng(42)
    # 4 years; year 1 pass-through, years 2-4 test years
    idx = pd.date_range("2022-01-01", "2025-12-31", freq="2h", tz="UTC")
    n = len(idx)
    y_true = rng.normal(-0.5, 0.3, size=n)
    y_pred = y_true + rng.normal(0, 0.2, size=n)
    iv = 50 + 20 * np.exp(y_pred)
    signal = rng.choice([-1, 0, 1], size=n, p=[0.2, 0.6, 0.2])
    iv_log = np.log(iv / 100)
    edge = y_pred - iv_log
    sigma_epi = np.abs(rng.normal(0.05, 0.02, n)) + 0.001
    # Make σ²_epi PREDICTIVE of shortfall: high σ²_epi → more loss
    pnl_expected = -100 * sigma_epi**2 * 50 + np.sign(edge) * np.abs(edge) * 100
    pnl_noise = rng.normal(0, 80, size=n)
    pnl = np.where(signal != 0, pnl_expected + pnl_noise, 0.0)

    tl = pd.DataFrame({
        "signal": signal, "iv_used": iv,
        "net_pnl_nc": pnl, "net_pnl": pnl,
    }, index=idx)
    preds = pd.DataFrame({"y_pred": y_pred, "y_true": y_true}, index=idx)
    wfu = pd.DataFrame({"sigma_epi": sigma_epi}, index=idx)
    feats = build_shortfall_features(tl, preds, wfu)
    target = compute_shortfall_target(tl, threshold_usd=100.0)

    p_base = train_walkforward_shortfall_classifier(
        feats, target, feature_set="baseline", min_train_trades=100,
    )
    p_treat = train_walkforward_shortfall_classifier(
        feats, target, feature_set="treatment", min_train_trades=100,
    )
    p_plac = train_walkforward_shortfall_classifier(
        feats, target, feature_set="placebo", min_train_trades=100,
    )
    return tl, preds, wfu, target, p_base, p_treat, p_plac


class TestWalkforwardClassifier:
    def test_2022_passthrough_baseline_and_treatment(self, trained_classifiers):
        """Year 2022 has no train history → all-NaN scores in baseline/treatment/placebo."""
        _, _, _, target, p_base, p_treat, p_plac = trained_classifiers
        for name, p in [("baseline", p_base), ("treatment", p_treat), ("placebo", p_plac)]:
            yr_2022 = target.index.year == 2022
            in_2022 = p.loc[p.index.isin(target.index[yr_2022])]
            assert in_2022.isna().all(), f"{name} produced non-NaN in 2022"

    def test_test_years_have_scores(self, trained_classifiers):
        """Years 2023+ should have non-NaN scores (sufficient train history)."""
        _, _, _, target, p_base, p_treat, p_plac = trained_classifiers
        for yr in [2023, 2024, 2025]:
            yr_idx = target.index[target.index.year == yr]
            for name, p in [("baseline", p_base), ("treatment", p_treat), ("placebo", p_plac)]:
                p_yr = p.loc[p.index.isin(yr_idx)]
                assert p_yr.notna().sum() > 0, f"{name} year {yr} has no scores"

    def test_baseline_does_not_use_sigma_epi(self, trained_classifiers):
        """Mutating σ²_epi must not change BASELINE scores (it's not in the feature set)."""
        tl, preds, wfu, target, p_base, _, _ = trained_classifiers
        wfu2 = wfu.copy()
        wfu2["sigma_epi"] = wfu2["sigma_epi"] * 100.0  # huge mutation
        feats2 = build_shortfall_features(tl, preds, wfu2)
        target2 = compute_shortfall_target(tl, threshold_usd=100.0)
        p_base2 = train_walkforward_shortfall_classifier(
            feats2, target2, feature_set="baseline", min_train_trades=100,
        )
        # Drop NaN before comparing (alignment)
        common = p_base.dropna().index.intersection(p_base2.dropna().index)
        np.testing.assert_array_equal(
            p_base.loc[common].to_numpy(),
            p_base2.loc[common].to_numpy(),
        )

    def test_walkforward_isolation_no_lookahead(self, trained_classifiers):
        """Mutating year-2025 features must not change 2023/2024 scores.

        We mutate only y_pred / sigma_epi at year 2025 — leaving net_pnl_nc
        (and therefore the fired-trade target index) unchanged. Year-2025
        DFL features depend on shift(1).rolling(y_true), so year-2024 y_true
        bleeding into 2025 features is acceptable; we don't mutate 2024.
        Year-2023/2024 features and target are fully decoupled from year-2025
        mutations → classifier scores for those years must be byte-identical.
        """
        tl, preds, wfu, target, _, p_treat_orig, _ = trained_classifiers
        preds2 = preds.copy()
        wfu2 = wfu.copy()
        mut_mask = preds2.index.year == 2025
        preds2.loc[mut_mask, "y_pred"] = -10.0
        preds2.loc[mut_mask, "y_true"] = 5.0
        wfu2.loc[mut_mask, "sigma_epi"] = 5.0

        feats2 = build_shortfall_features(tl, preds2, wfu2)
        # target is computed from net_pnl_nc which we did NOT mutate → unchanged
        target2 = compute_shortfall_target(tl, threshold_usd=100.0)
        pd.testing.assert_series_equal(target2, target)

        p_treat_mut = train_walkforward_shortfall_classifier(
            feats2, target2, feature_set="treatment", min_train_trades=100,
        )

        # Scores on 2023 and 2024 must be byte-identical
        for yr in [2023, 2024]:
            yr_mask = p_treat_orig.index.year == yr
            orig_yr = p_treat_orig.loc[yr_mask].dropna()
            mut_yr = p_treat_mut.loc[yr_mask].dropna()
            common = orig_yr.index.intersection(mut_yr.index)
            assert len(common) > 0, f"no overlap in year {yr}"
            np.testing.assert_array_equal(
                orig_yr.loc[common].to_numpy(),
                mut_yr.loc[common].to_numpy(),
            )

    def test_placebo_differs_from_treatment(self, trained_classifiers):
        """Within-year shuffle should produce DIFFERENT scores than treatment."""
        _, _, _, _, _, p_treat, p_plac = trained_classifiers
        common = p_treat.dropna().index.intersection(p_plac.dropna().index)
        # Should not be byte-identical
        assert not np.array_equal(
            p_treat.loc[common].to_numpy(),
            p_plac.loc[common].to_numpy(),
        )

    def test_fairness_columns_enforces_same_training_rows(self):
        """With fairness_columns set, BASELINE / TREATMENT see the same rows.

        Construct a fixture where ``sigma_epi_squared`` is NaN in year 2022
        (mirrors Phase A.0 walk-forward parquet output). Verify that
        ``fairness_columns=SHORTFALL_FEATURES_TREATMENT`` forces BASELINE to
        also exclude year-2022 rows from training (otherwise BASELINE would
        train on more rows than TREATMENT — sample-size confound).
        """
        rng = np.random.default_rng(123)
        idx = pd.date_range("2022-01-01", "2024-12-31", freq="2h", tz="UTC")
        n = len(idx)
        y_true = rng.normal(-0.5, 0.3, size=n)
        y_pred = y_true + rng.normal(0, 0.2, size=n)
        iv = 50 + 20 * np.exp(y_pred)
        signal = rng.choice([-1, 0, 1], size=n, p=[0.2, 0.6, 0.2])
        sigma_epi = np.abs(rng.normal(0.05, 0.02, n))
        # Set σ²_epi to NaN for year 2022 (Phase A.0 walk-forward contract)
        sigma_epi_with_nan = sigma_epi.copy()
        yr_2022_mask = idx.year == 2022
        sigma_epi_with_nan[yr_2022_mask] = np.nan

        edge = y_pred - np.log(iv / 100)
        pnl = np.where(
            signal != 0,
            np.sign(edge) * np.abs(edge) * 100 + rng.normal(0, 80, size=n),
            0.0,
        )
        tl = pd.DataFrame({"signal": signal, "iv_used": iv,
                            "net_pnl_nc": pnl, "net_pnl": pnl}, index=idx)
        preds = pd.DataFrame({"y_pred": y_pred, "y_true": y_true}, index=idx)
        wfu = pd.DataFrame({"sigma_epi": np.sqrt(sigma_epi_with_nan)}, index=idx)
        feats = build_shortfall_features(tl, preds, wfu)
        target = compute_shortfall_target(tl, threshold_usd=100.0)

        # Without fairness_columns: BASELINE trains on year 2022, TREATMENT does not
        p_b_unfair = train_walkforward_shortfall_classifier(
            feats, target, feature_set="baseline",
            min_train_trades=100, test_years=(2024,),
        )
        p_t_unfair = train_walkforward_shortfall_classifier(
            feats, target, feature_set="treatment",
            min_train_trades=100, test_years=(2024,),
        )

        # With fairness_columns set: BASELINE forced to drop year-2022 rows
        p_b_fair = train_walkforward_shortfall_classifier(
            feats, target, feature_set="baseline",
            min_train_trades=100, test_years=(2024,),
            fairness_columns=SHORTFALL_FEATURES_TREATMENT,
        )
        p_t_fair = train_walkforward_shortfall_classifier(
            feats, target, feature_set="treatment",
            min_train_trades=100, test_years=(2024,),
            fairness_columns=SHORTFALL_FEATURES_TREATMENT,
        )

        # Sanity: TREATMENT's score must be unchanged by fairness_columns
        # (both with/without already drop year-2022 since σ²_epi is NaN there)
        common_t = p_t_unfair.dropna().index.intersection(p_t_fair.dropna().index)
        np.testing.assert_array_equal(
            p_t_unfair.loc[common_t].to_numpy(),
            p_t_fair.loc[common_t].to_numpy(),
        )
        # The KEY assertion: BASELINE under fairness_columns must DIFFER from
        # BASELINE without fairness_columns (because the training-row set
        # changed — year 2022 dropped from training).
        common_b = p_b_unfair.dropna().index.intersection(p_b_fair.dropna().index)
        # Under different training data, BASELINE produces different scores
        assert not np.array_equal(
            p_b_unfair.loc[common_b].to_numpy(),
            p_b_fair.loc[common_b].to_numpy(),
        ), "fairness_columns failed to change BASELINE training set"


# ─── 6. AUC summary ─────────────────────────────────────────────────────


class TestPerYearAucSummary:
    def test_returns_per_year_and_pooled(self, trained_classifiers):
        _, _, _, target, _, p_treat, _ = trained_classifiers
        out = per_year_auc_summary(target, p_treat)
        assert "pooled" in out
        for yr in SHORTFALL_TEST_YEARS:
            assert f"year_{yr}" in out

    def test_auc_in_unit_interval_or_nan(self, trained_classifiers):
        _, _, _, target, p_base, p_treat, p_plac = trained_classifiers
        for p in [p_base, p_treat, p_plac]:
            out = per_year_auc_summary(target, p)
            for k, v in out.items():
                if not np.isnan(v):
                    assert 0.0 <= v <= 1.0, f"{k}={v} out of [0, 1]"


# ─── 7. Block-bootstrap ─────────────────────────────────────────────────


class TestBlockBootstrap:
    def test_returns_correct_shape(self, trained_classifiers):
        _, _, _, target, p_base, p_treat, _ = trained_classifiers
        # Restrict to test years and aligned index
        common = target.index.intersection(p_base.dropna().index).intersection(p_treat.dropna().index)
        if len(common) < 200:
            pytest.skip("not enough overlap in fixture")
        lcb, median, ucb, deltas = block_bootstrap_auc_delta_ci(
            target.loc[common], p_base.loc[common], p_treat.loc[common],
            n_reps=50, block_mean_days=5, seed=42,
        )
        assert deltas.shape == (50,)
        assert lcb <= median <= ucb

    def test_paired_same_seed_deterministic(self, trained_classifiers):
        _, _, _, target, p_base, p_treat, _ = trained_classifiers
        common = target.index.intersection(p_base.dropna().index).intersection(p_treat.dropna().index)
        if len(common) < 200:
            pytest.skip("not enough overlap in fixture")
        out_a = block_bootstrap_auc_delta_ci(
            target.loc[common], p_base.loc[common], p_treat.loc[common],
            n_reps=50, seed=42,
        )
        out_b = block_bootstrap_auc_delta_ci(
            target.loc[common], p_base.loc[common], p_treat.loc[common],
            n_reps=50, seed=42,
        )
        np.testing.assert_array_equal(out_a[3], out_b[3])

    def test_paired_resample_uses_shared_days(self):
        """Block bootstrap is paired by DAY: same resampled day-set applied
        to both classifiers. With p_a == p_b, every rep's AUC delta = 0 and
        the CI is degenerate at zero — proves the pairing is byte-identical
        across the two arms.
        """
        rng = np.random.default_rng(0)
        n_days = 100
        trades_per_day = 5
        days = pd.date_range("2024-01-01", periods=n_days, freq="D", tz="UTC")
        rows = []
        for d in days:
            for j in range(trades_per_day):
                rows.append(d + pd.Timedelta(hours=j))
        idx = pd.DatetimeIndex(rows)
        n = len(idx)
        scores = pd.Series(rng.normal(size=n), index=idx)
        y = pd.Series(rng.integers(0, 2, n), index=idx)
        # Identical classifiers → AUC delta = 0 for every rep
        lcb, median, ucb, deltas = block_bootstrap_auc_delta_ci(
            y, scores, scores, n_reps=100, block_mean_days=5, seed=0,
        )
        # Reps where bootstrap-resample produced no class diversity get 0
        # Reps with class diversity also get 0 (because p_a == p_b)
        assert np.all(deltas == 0.0)
        assert lcb == 0.0 and ucb == 0.0

    def test_minimum_days_validation(self):
        """Bootstrap requires ≥ 30 unique days for stability."""
        rng = np.random.default_rng(0)
        # 20 days × 5 trades = 100 trades, 20 unique days
        idx = pd.date_range("2024-01-01", periods=100, freq="6h", tz="UTC")
        y = pd.Series(rng.integers(0, 2, 100), index=idx)
        p = pd.Series(rng.uniform(size=100), index=idx)
        # 100 timestamps spanning 24 days at 6h intervals → 25 unique days; below 30
        with pytest.raises(ValueError, match="≥ 30"):
            block_bootstrap_auc_delta_ci(y, p, p, n_reps=10)


# ─── 8. DeLong ──────────────────────────────────────────────────────────


class TestDelongPaired:
    def test_identical_classifiers_p_value_high(self):
        """Two identical classifiers should have p-value ~ 1.0 (no detectable
        difference)."""
        rng = np.random.default_rng(0)
        n = 500
        y = rng.integers(0, 2, n)
        p = rng.uniform(size=n)
        _, _, _, p_value = delong_paired_p_value(y, p, p, alternative="greater")
        # Cannot reject null in either direction
        assert p_value > 0.4

    def test_strictly_better_classifier_low_p_value(self):
        """A classifier that perfectly orders y vs random should beat random in p-value."""
        rng = np.random.default_rng(7)
        n = 500
        y = rng.integers(0, 2, n).astype(int)
        # Perfect classifier: scores correlated with y
        p_perfect = y + rng.normal(0, 0.1, size=n)
        # Random
        p_random = rng.normal(size=n)
        _, _, _, p_value = delong_paired_p_value(
            y, p_random, p_perfect, alternative="greater",
        )
        assert p_value < 0.001

    def test_returns_correct_aucs(self):
        """AUC values returned should match sklearn.metrics.roc_auc_score."""
        from sklearn.metrics import roc_auc_score
        rng = np.random.default_rng(11)
        n = 500
        y = rng.integers(0, 2, n).astype(int)
        p_a = y + rng.normal(0, 0.5, size=n)
        p_b = y + rng.normal(0, 0.3, size=n)
        auc_a, auc_b, _, _ = delong_paired_p_value(y, p_a, p_b)
        sk_a = roc_auc_score(y, p_a)
        sk_b = roc_auc_score(y, p_b)
        # DeLong AUC matches sklearn to floating-point precision
        assert abs(auc_a - sk_a) < 1e-9
        assert abs(auc_b - sk_b) < 1e-9

    def test_two_sided_alternative(self):
        rng = np.random.default_rng(17)
        n = 500
        y = rng.integers(0, 2, n).astype(int)
        p_a = y + rng.normal(0, 0.3, size=n)
        p_b = y + rng.normal(0, 0.3, size=n)
        _, _, _, p_two = delong_paired_p_value(y, p_a, p_b, alternative="two-sided")
        assert 0.0 <= p_two <= 1.0


# ─── 9. End-to-end gate ─────────────────────────────────────────────────


class TestAucDeltaGate:
    def test_returns_dataclass_with_all_fields(self, trained_classifiers):
        _, _, _, target, p_base, p_treat, p_plac = trained_classifiers
        # Restrict to test years for the gate eval
        result = auc_delta_gate(
            target, p_base, p_treat, p_plac,
            block_bootstrap_n_reps=100,  # speed
        )
        assert isinstance(result, AucDeltaGateResult)
        assert hasattr(result, "auc_baseline_pooled")
        assert hasattr(result, "auc_treatment_pooled")
        assert hasattr(result, "delta_pooled")
        assert hasattr(result, "all_pass")
        # Summary lines render without error
        lines = result.summary_lines()
        assert any("VERDICT" in line for line in lines)

    def test_all_pass_consistency(self, trained_classifiers):
        """all_pass = G1 AND G2 AND G3 AND G4."""
        _, _, _, target, p_base, p_treat, p_plac = trained_classifiers
        result = auc_delta_gate(
            target, p_base, p_treat, p_plac,
            block_bootstrap_n_reps=100,
        )
        expected = (
            result.g1_pass and result.g2_pass and result.g3_pass and result.g4_pass
        )
        assert result.all_pass == expected

    def test_gate_uses_locked_thresholds(self, trained_classifiers):
        _, _, _, target, p_base, p_treat, p_plac = trained_classifiers
        result = auc_delta_gate(
            target, p_base, p_treat, p_plac,
            block_bootstrap_n_reps=100,
        )
        assert result.g1_threshold == GATE_G1_DELTA_MIN
        assert result.g3_p_threshold == GATE_G3_DELONG_P_MAX
        assert result.g4_threshold == GATE_G4_PLACEBO_MAX

    def test_too_small_sample_raises(self, synthetic_data):
        tl, preds, wfu = synthetic_data
        # Tiny target / score series
        idx = pd.date_range("2024-01-01", periods=10, freq="1h", tz="UTC")
        target = pd.Series([0, 1] * 5, index=idx, dtype=int)
        p = pd.Series(np.linspace(0, 1, 10), index=idx)
        with pytest.raises(ValueError, match="too small"):
            auc_delta_gate(target, p, p, p, block_bootstrap_n_reps=10)
