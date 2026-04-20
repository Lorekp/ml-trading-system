"""Regression locks for BacktestConfig preset invariants.

Added 2026-04-17 foundation fix: prevent accidental drift of the V2 production
hold config back to 30m. The 15m hold was chosen to match the model's
``[D, D+15m]`` prediction horizon; see
``vol_model/output/FOUNDATION_ALIGNMENT_AUDIT.md``.
"""
from __future__ import annotations

from vol_arb.backtester import BacktestConfig


def test_production_v2_holding_minutes_is_15():
    """V2 production hold must stay aligned with prediction horizon.

    Prediction target: ln(vol_ann over [D, D+15m]). A 30-min hold would cover
    [D, D+30m] — first half signal-aligned, second half un-forecast theta
    carry. Locked at 15 min as of the foundation fix.
    """
    cfg = BacktestConfig().with_btc_production_filters_v2()
    assert cfg.holding_minutes == 15, (
        f"V2 holding_minutes must be 15 to match the model's "
        f"[D, D+15m] prediction window; got {cfg.holding_minutes}. "
        f"Reverting to 30 re-introduces the foundation misalignment — "
        f"see vol_model/output/FOUNDATION_ALIGNMENT_AUDIT.md."
    )


def test_production_v2_core_filters_preserved():
    """Sanity check: other V2 filters we rely on are still set as expected.

    Foundation-fix Step 6 should have ONLY touched holding_minutes. Any other
    field drifting here is a sign of an accidental edit.
    """
    cfg = BacktestConfig().with_btc_production_filters_v2()
    assert cfg.use_drl_hedging is True
    assert cfg.use_profit_scaling is True
    assert cfg.profit_scaling_floor == 0.3
    assert cfg.target_margin_utilization == 0.15
    assert cfg.entry_threshold_long == 12.0
    assert cfg.skip_long_if_term_slope_gt == 0.0


def test_production_v1_holding_minutes_is_15():
    """V1 hold was already 15m pre-fix — confirm we didn't break it."""
    cfg = BacktestConfig().with_btc_production_filters()
    assert cfg.holding_minutes == 15


def test_default_predictions_path_is_aligned_bias():
    """Production default adopts the walk-forward QLIKE-optimal bias-
    aligned predictions (2026-04-20).

    The underlying predictor is pruned_svi_har_pd_v1 (certified). The
    aligned_bias parquet additionally applies a per-month scalar shift
    β* = 0.5·log(mean(exp(2·residual))) fit on trailing 3-month OOS
    residuals (strictly non-leaking, leakage-tested in
    tests/test_prediction_aligner.py).

    Adopted as an ACCEPTED Sharpe-for-PnL tradeoff: -0.42 NC Sharpe
    (the honest signal-quality metric per feedback_noncompound_truth.md)
    in exchange for +$30K NC PnL (+3.7%). Not a Sharpe win — a dollar
    win at a per-trade-Sharpe cost.

    Override via --predictions-path or BacktestConfig(predictions_path=...)
    to reproduce pre-aligner results.
    """
    cfg = BacktestConfig()
    assert cfg.predictions_path.endswith("aligned_bias.parquet"), (
        f"Default predictions_path should point at the aligned_bias "
        f"parquet (walk-forward QLIKE-optimal scalar bias correction). "
        f"Got: {cfg.predictions_path}. Reverting to the raw predictor "
        f"would undo the +6.4% compound PnL adoption decision — see "
        f"memory/project_prediction_aligner_2026_04_20.md."
    )


def test_dfl_filter_defaults_off():
    """DFL filter must stay OFF by default so V2 baseline is byte-
    identical when the scores sidecar isn't passed."""
    cfg = BacktestConfig()
    assert cfg.dfl_scores_path is None
    # Quantile has a default value but is only consulted when path is set.
    assert cfg.dfl_filter_quantile == 0.25


def test_kelly_mult_bare_default_is_off():
    """The bare ``BacktestConfig()`` default keeps Kelly OFF.

    Only the V2 preset opts in. Plain ``BacktestConfig()`` is used
    as a generic starting point for non-BTC / non-V2 configs, where
    the K1 sidecar is invalid (it was computed FROM the V2 trade
    log; applying it to ETH/V1/etc. is wrong by construction).
    """
    cfg = BacktestConfig()
    assert cfg.kelly_mult_path is None


def test_production_v2_kelly_default_is_K1():
    """V2 preset MUST default to the K1 Kelly-shrinkage sidecar.

    Adopted 2026-04-21 as a deliberate Sharpe-for-PnL tradeoff:
    +$114K NC PnL (+13.5%) and +$2.84M compound PnL (+17.9%) in
    exchange for -0.94 NC Sharpe (the honest signal-quality metric
    per feedback_noncompound_truth.md) and -0.20 compound Sharpe.
    Compound MDD actually IMPROVED 0.62pp.

    Reverting the default to None reproduces V2-pre-Kelly baseline:
    NC PnL $845K, NC Sharpe 7.23, NC MDD -$8,288. See
    ``vol_arb/output/KELLY_SHRINKAGE_PHASE2_REPORT.md`` and
    ``memory/project_kelly_sizing_2026_04_21.md`` for the full
    decision record. Plan:
    ``C:/Users/laure/.claude/plans/starry-rolling-rabin.md``.
    """
    cfg = BacktestConfig().with_btc_production_filters_v2()
    expected = "vol_arb/output/BTC_v2_kelly_mult_K1.parquet"
    assert cfg.kelly_mult_path == expected, (
        f"V2 preset must auto-load the K1 Kelly-shrinkage sidecar "
        f"at {expected!r}; got {cfg.kelly_mult_path!r}. Reverting "
        f"to None re-introduces the +$114K PnL gap (long-side "
        f"high-|edge| boost) — see KELLY_SHRINKAGE_PHASE2_REPORT.md."
    )
