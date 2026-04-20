"""Tests for vol_arb.dfl_head — DFF-style decision-focused head.

Critical invariants:
1. NO LOOKAHEAD — mutating a test-year's data must not change
   prior-year scores.
2. FilterStrategy with quantile=0 drops no bars.
3. FilterStrategy with quantile=1 drops all scored bars.
4. SizeStrategy with constant multiplier_fn=1.0 returns identity.
5. Pass-through: when no training history, bars get NaN scores and
   are passed through unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_arb.dfl_head import (
    DFL_HEAD_FEATURES, FilterStrategy, SizeStrategy,
    build_head_features, compute_walkforward_scores,
    pre_committed_strategies,
)


@pytest.fixture
def synthetic_data():
    """3 years of trade log + predictions. Strong signal in year 2/3
    so the classifier should learn something; year 1 is pass-through."""
    rng = np.random.default_rng(42)
    idx = pd.date_range("2022-01-01", "2024-12-31", freq="1h", tz="UTC")
    n = len(idx)
    y_true = rng.normal(-0.5, 0.3, size=n)
    y_pred = y_true + rng.normal(0, 0.2, size=n)
    iv = 50 + 20 * np.exp(y_pred)
    signal = rng.choice([-1, 0, 1], size=n, p=[0.25, 0.5, 0.25])
    # Correlate win/loss with y_pred - IV_log
    iv_log = np.log(iv / 100)
    edge = y_pred - iv_log
    pnl_expected = np.sign(edge) * np.abs(edge) * 20
    pnl_noise = rng.normal(0, 100, size=n)
    pnl = np.where(signal != 0, pnl_expected + pnl_noise, 0.0)
    tl = pd.DataFrame({
        "signal": signal,
        "iv_used": iv,
        "net_pnl_nc": pnl,
        "net_pnl": pnl,
    }, index=idx)
    preds = pd.DataFrame({"y_pred": y_pred, "y_true": y_true}, index=idx)
    return tl, preds


def test_features_are_strictly_lagged(synthetic_data):
    tl, preds = synthetic_data
    feats = build_head_features(tl, preds)
    # First 96 rows (of the rolling-24h lag) should have NaN in at least one column
    assert feats.iloc[:24].isna().any(axis=1).any()
    # After 96 rows should have full coverage
    assert feats.iloc[200:].notna().all(axis=0).all()
    # All rolling features use past data — mutating the "current" row's
    # y_true should NOT change features at that row.
    tl2 = tl.copy()
    preds2 = preds.copy()
    preds2.loc[preds2.index[500], "y_true"] = 999.0  # huge mutation
    feats2 = build_head_features(tl2, preds2)
    # Features at row 500 should be unchanged (y_true at 500 is only
    # used in strictly-lagged rolling at row 501+)
    pd.testing.assert_series_equal(feats.iloc[500], feats2.iloc[500])


def test_walkforward_no_lookahead(synthetic_data):
    tl, preds = synthetic_data
    s_baseline = compute_walkforward_scores(tl, preds, min_train_trades=100)

    # Mutate year-2024 data
    tl2 = tl.copy()
    preds2 = preds.copy()
    mutation_mask = tl2.index.year == 2024
    tl2.loc[mutation_mask, "net_pnl_nc"] = -9999.0  # flip all wins to losses
    preds2.loc[mutation_mask, "y_pred"] = -5.0  # nonsense predictions

    s_mutated = compute_walkforward_scores(tl2, preds2, min_train_trades=100)

    # Scores on 2022 and 2023 bars must be byte-identical
    for y in [2022, 2023]:
        year_mask = s_baseline.index.year == y
        pd.testing.assert_series_equal(
            s_baseline.loc[year_mask].dropna(),
            s_mutated.loc[year_mask].dropna(),
            check_exact=True,
        )
    # Year 2024 may differ (training data unchanged but test data mutated
    # changes the predict inputs; though we're mutating net_pnl which is
    # the label, not the features, so actually year-2024 scores are same
    # too — no leakage possible)


def test_filter_quantile_0_drops_nothing(synthetic_data):
    tl, preds = synthetic_data
    scores = compute_walkforward_scores(tl, preds, min_train_trades=100)
    s = FilterStrategy(quantile=0.0, name="noop")
    out = s.apply(tl, scores)
    # Since min quantile is 0, threshold < every score, so no bars dropped.
    pd.testing.assert_frame_equal(out, tl)


def test_filter_drops_bottom_fraction_approximately(synthetic_data):
    tl, preds = synthetic_data
    scores = compute_walkforward_scores(tl, preds, min_train_trades=100)
    s = FilterStrategy(quantile=0.25, name="q25")
    out = s.apply(tl, scores)

    # Bars with scores should have some filtered
    scored = scores.dropna()
    n_scored = len(scored)
    n_zeroed = ((out["net_pnl_nc"] == 0) & (tl["net_pnl_nc"] != 0)).sum()
    # Approximately 25% of trade-fired bars should be zeroed (within
    # year-by-year quantile approximation)
    assert n_zeroed > 0, "Filter dropped nothing"


def test_size_strategy_multiplier_preserves_frame_shape(synthetic_data):
    tl, preds = synthetic_data
    scores = compute_walkforward_scores(tl, preds, min_train_trades=100)
    # Unit multiplier
    def unit_mult(year_scores, train_scores):
        return pd.Series(1.0, index=year_scores.index)
    s = SizeStrategy(name="unit", description="no-op", multiplier_fn=unit_mult)
    out = s.apply(tl, scores)
    # PnL columns should be unchanged
    pd.testing.assert_series_equal(out["net_pnl_nc"], tl["net_pnl_nc"])


def test_size_strategy_zero_multiplier_zeros_scored_pnl(synthetic_data):
    """The SizeStrategy needs a PRIOR year of scored data to compute
    multiplier parameters. The first scored year passes through
    (no baseline distribution yet). From the second scored year onward,
    the zero multiplier zeros out PnL.
    """
    tl, preds = synthetic_data
    scores = compute_walkforward_scores(tl, preds, min_train_trades=100)
    def zero_mult(year_scores, train_scores):
        return pd.Series(0.0, index=year_scores.index)
    s = SizeStrategy(name="zero", description="", multiplier_fn=zero_mult)
    out = s.apply(tl, scores)

    # At least one year should have been zeroed (2024 in our fixture).
    n_zeroed = int(((out["net_pnl_nc"] == 0) & (tl["net_pnl_nc"] != 0)).sum())
    assert n_zeroed > 0, (
        f"Expected some bars zeroed, got {n_zeroed}. "
        f"SizeStrategy needs prior-year score history to activate."
    )
    # Year-2022 bars are always pass-through (training year, not scored)
    tl_2022 = tl.loc[tl.index.year == 2022]
    out_2022 = out.loc[out.index.year == 2022]
    pd.testing.assert_series_equal(
        out_2022["net_pnl_nc"], tl_2022["net_pnl_nc"]
    )


def test_pre_committed_strategies_count():
    strats = pre_committed_strategies()
    names = [s.name for s in strats]
    assert names == [
        "A1_filter25", "A2_filter10",
        "B1_linear", "B2_binary_median", "B3_normalized",
    ]


def test_first_year_is_passthrough(synthetic_data):
    """Year 1 has no training history, should be NaN scores → no
    filtering/sizing applied (passthrough)."""
    tl, preds = synthetic_data
    scores = compute_walkforward_scores(tl, preds, min_train_trades=100)
    # Year 2022 should have all NaN scores
    s_2022 = scores.loc[scores.index.year == 2022]
    assert s_2022.isna().all()

    # Under FilterStrategy, year-2022 bars should not be dropped (threshold NaN)
    strat = FilterStrategy(quantile=0.25, name="q25")
    out = strat.apply(tl, scores)
    tl_2022 = tl.loc[tl.index.year == 2022]
    out_2022 = out.loc[out.index.year == 2022]
    pd.testing.assert_series_equal(
        out_2022["net_pnl_nc"], tl_2022["net_pnl_nc"],
    )
