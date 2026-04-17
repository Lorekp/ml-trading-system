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
