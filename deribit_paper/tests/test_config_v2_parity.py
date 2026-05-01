"""Regression locks for ``PaperConfig.with_v2_preset()`` matching the
production V2 backtest preset field-by-field.

The single authoritative cross-check is
``test_v2_preset_matches_backtest_v2`` — it imports both factories and
asserts the shared field set is equal. Every other test in this module
guards a specific invariant individually, so when the cross-check fails
the regression has a clearly-named per-field witness.

C5 in plan ``C:\\Users\\laure\\.claude\\plans\\live-engine-h15-alignment-2026-04-26\\``.

Adding a new V2 invariant requires updating BOTH:
1. ``PaperConfig.with_v2_preset()`` (this module's test target).
2. ``BacktestConfig.with_btc_production_filters_v2()`` /
   ``_v2_horizon_invariants()`` (the source of truth in vol_arb).
The cross-check test will fail loudly until both sides agree.
"""
from __future__ import annotations

import pytest

from deribit_paper.config import PaperConfig
from vol_arb.backtester import BacktestConfig


@pytest.fixture
def v2_paper() -> PaperConfig:
    return PaperConfig.with_v2_preset()


@pytest.fixture
def v2_backtest() -> BacktestConfig:
    return BacktestConfig().with_btc_production_filters_v2()


@pytest.fixture
def default_paper() -> PaperConfig:
    return PaperConfig()


# ── Per-field invariants ───────────────────────────────────────────────────

def test_holding_minutes_15(v2_paper):
    """Foundation fix 2026-04-17 lock: model horizon == backtester hold."""
    assert v2_paper.holding_minutes == 15


def test_entry_threshold_long_is_12(v2_paper):
    """Phase C.5 lock: contango filter compensates lower long edge floor."""
    assert v2_paper.entry_threshold_long == 12.0


def test_entry_threshold_short_is_22(v2_paper):
    """Mirrors V2 ``entry_threshold=22`` for short straddles (set in V1
    production filters and inherited by V2)."""
    assert v2_paper.entry_threshold == 22.0


def test_skip_long_if_term_slope_gt_is_zero(v2_paper):
    """V2 contango filter — drop longs in IV contango (positive 1w-2w slope)."""
    assert v2_paper.skip_long_if_term_slope_gt == 0.0


def test_target_margin_utilization_is_15pct(v2_paper):
    """V2 capital-scale: 3x leverage vs default 5%."""
    assert v2_paper.target_margin_utilization == 0.15


def test_use_profit_scaling_enabled(v2_paper):
    """Conviction-weighted sizing on, floor=0.3."""
    assert v2_paper.use_profit_scaling is True
    assert v2_paper.profit_scaling_floor == 0.3


def test_kelly_mult_path_is_k1(v2_paper):
    """Kelly K1 sidecar adopted 2026-04-21."""
    assert v2_paper.kelly_mult_path == "vol_arb/output/BTC_v2_kelly_mult_K1.parquet"


def test_aligned_predictions_parquet_set(v2_paper):
    """Aligner adopted 2026-04-20 (default path: HAR-PD aligned-bias)."""
    assert v2_paper.aligned_predictions_parquet.endswith(
        "BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet"
    )


def test_aligner_betas_json_set(v2_paper):
    """Per-month β* JSON sidecar (extracted via scripts/extract_aligner_betas.py)."""
    assert v2_paper.aligner_betas_json.endswith("aligner_betas_by_month.json")


def test_bar_exec_quality_path_set(v2_paper):
    """realistic_taker empirical slippage sidecar."""
    assert v2_paper.bar_exec_quality_path == (
        "vol_arb/output/bar_execution_quality.parquet"
    )


def test_use_drl_hedging_implicit_via_engine(v2_paper):
    """SAC v4 hedging is always-on in the engine path; no flag needed.

    The live engine's ``engine.py:283-300`` always loads the DRL policy.
    BacktestConfig.use_drl_hedging is the symmetric flag; ``True`` in V2.
    Documenting the asymmetry here so a future contributor knows the
    live path doesn't gate hedging on a field.
    """
    # PaperConfig has no use_drl_hedging field — this is by design.
    # If a future change adds the field, it must default True under V2.
    assert not hasattr(v2_paper, "use_drl_hedging") or v2_paper.use_drl_hedging is True


# ── Default preset preservation (no surprise behaviour change) ─────────────

def test_default_preset_keeps_legacy_thresholds(default_paper):
    """Without --config-preset v2, live behaviour stays at legacy values."""
    assert default_paper.entry_threshold_long == 30.0
    assert default_paper.skip_long_if_term_slope_gt is None
    assert default_paper.target_margin_utilization == 0.05
    assert default_paper.use_profit_scaling is False
    assert default_paper.kelly_mult_path == ""
    assert default_paper.aligner_betas_json == ""
    assert default_paper.bar_exec_quality_path == ""


# ── Authoritative cross-check ──────────────────────────────────────────────

def test_v2_preset_matches_backtest_v2(v2_paper, v2_backtest):
    """Field-by-field equality on the shared V2 invariant set.

    This is the single test that guards "live and backtest don't drift".
    If a V2 invariant changes in BacktestConfig, this test fails until
    PaperConfig.with_v2_preset() catches up (and vice versa).
    """
    shared_invariants = [
        ("entry_threshold_long",         "entry_threshold_long"),
        ("target_margin_utilization",    "target_margin_utilization"),
        ("use_profit_scaling",           "use_profit_scaling"),
        ("profit_scaling_floor",         "profit_scaling_floor"),
        ("skip_long_if_term_slope_gt",   "skip_long_if_term_slope_gt"),
        ("entry_threshold",              "entry_threshold"),
        ("holding_minutes",              "holding_minutes"),
        ("kelly_mult_path",              "kelly_mult_path"),
    ]
    for live_field, backtest_field in shared_invariants:
        live_val = getattr(v2_paper, live_field)
        bt_val = getattr(v2_backtest, backtest_field)
        # Pydantic vs dataclass — both expose ``Field`` defaults the same way.
        assert live_val == bt_val, (
            f"V2 invariant drift: PaperConfig.{live_field}={live_val!r} "
            f"!= BacktestConfig.{backtest_field}={bt_val!r}"
        )


def test_v2_preset_predictions_path_matches_backtest_default(v2_paper, v2_backtest):
    """The live engine's aligned predictions parquet must equal the
    backtester's predictions_path (both consume the same aligner output).
    """
    assert v2_paper.aligned_predictions_parquet == v2_backtest.predictions_path


# ── E.2.1: predictions parquet bar frequency matches holding_minutes ──────


def test_aligned_predictions_bar_frequency_matches_holding_minutes(v2_paper):
    """Foundation-fix invariant on the LIVE side. Mirrors
    ``tests/test_backtester_config.py::test_v2_preset_horizon_matches_holding_minutes``
    but loads the parquet and infers bar frequency from the index.

    A 5-min predictions parquet paired with ``holding_minutes=15`` would
    silently throw away two-thirds of the predictor's information.
    Catching at startup is much cheaper than the ~$150K loss the
    foundation-fix fixed in 2026-04-17 (memory:project_foundation_fix).
    """
    from pathlib import Path

    import pandas as pd
    import pytest as _pt

    p = Path(v2_paper.aligned_predictions_parquet)
    if not p.exists():
        _pt.skip(f"{p} not on disk in this checkout — skipping runtime check")

    df = pd.read_parquet(p, columns=[])  # index only — fast
    if not isinstance(df.index, pd.DatetimeIndex) or len(df) < 100:
        _pt.skip(
            f"{p} index is not a DatetimeIndex or too short — skipping check"
        )

    # Sample the first 500 spacing diffs; mode is the bar duration.
    diffs = df.index[1:501] - df.index[:500]
    seconds = diffs.total_seconds().astype(int)
    bar_seconds_inferred = int(pd.Series(seconds).mode().iloc[0])
    expected_bar_seconds = v2_paper.holding_minutes * 60
    assert bar_seconds_inferred == expected_bar_seconds, (
        f"Predictions parquet at {p.name} has bar spacing "
        f"{bar_seconds_inferred}s != holding_minutes×60 = "
        f"{expected_bar_seconds}s. Re-introduces the foundation-fix "
        f"failure mode (model-horizon != backtester-hold)."
    )
