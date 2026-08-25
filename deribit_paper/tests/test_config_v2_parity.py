"""Regression locks for ``PaperConfig.historical_v2_preset()``.

These tests preserve forensic replay parity with the invalidated V2
backtest configuration. They do not validate performance or deployment.

The single authoritative cross-check is
``test_v2_preset_matches_backtest_v2`` — it imports both factories and
asserts the shared field set is equal. Every other test in this module
guards a specific invariant individually, so when the cross-check fails
the regression has a clearly-named per-field witness.

Adding a new V2 invariant requires updating BOTH:
1. ``PaperConfig.historical_v2_preset()`` (this module's test target).
2. ``BacktestConfig.with_historical_btc_v2()`` /
   ``_historical_v2_horizon_settings()`` (the source of truth in vol_arb).
The cross-check test will fail loudly until both sides agree.
"""
from __future__ import annotations

import pytest

from deribit_paper.config import PaperConfig
from vol_arb.backtester import BacktestConfig


@pytest.fixture
def v2_paper() -> PaperConfig:
    return PaperConfig.historical_v2_preset()


@pytest.fixture
def v2_backtest() -> BacktestConfig:
    return BacktestConfig().with_historical_btc_v2()


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


def test_entry_threshold_short_is_20(v2_paper):
    """Historical V2 retains its short-entry threshold."""
    assert v2_paper.entry_threshold == 20.0


def test_skip_long_if_term_slope_gt_is_zero(v2_paper):
    """V2 contango filter — drop longs in IV contango (positive 1w-2w slope)."""
    assert v2_paper.skip_long_if_term_slope_gt == 0.0


def test_skip_short_if_skew_1w_gt_disabled(v2_paper):
    """Historical V2 disables the short-skew filter."""
    assert v2_paper.skip_short_if_skew_1w_gt is None


def test_max_gross_notional_usd_is_200k(v2_paper):
    """Cycle-10 (2026-05-10): $30k default was sized for earlier BTC spot
    regime; at BTC=$80k a single ATM straddle is $40-97k notional and
    the $30k cap blocked 100% of fired signals (homeserver audit).
    Bumped to $200k (~2x equity) as defence-in-depth that allows
    legitimate trades. Backtester has no equivalent cap."""
    assert v2_paper.max_gross_notional_usd == 200_000.0


def test_target_margin_utilization_is_10pct(v2_paper):
    """Historical V2 retains ten-percent target utilization."""
    assert v2_paper.target_margin_utilization == 0.10


def test_use_profit_scaling_enabled(v2_paper):
    """Conviction-weighted sizing on, floor=0.3."""
    assert v2_paper.use_profit_scaling is True
    assert v2_paper.profit_scaling_floor == 0.3


def test_kelly_mult_path_is_k1_slope(v2_paper):
    """Historical V2 retains its K1+slope sidecar path."""
    assert v2_paper.kelly_mult_path == (
        "vol_arb/output/BTC_v2_kelly_mult_K1_slope.parquet"
    )


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


# ── Parity-cycle 2026-05-07: D1, D2, D6, D10 invariants ────────────────────

def test_dte_min_days_is_5(v2_paper):
    """D1 (parity 2026-05-07): V2 backtester ``min_dte_days=5.0``;
    live default of 1 lets the engine trade the 1-4 DTE band the
    backtester deliberately mutes (theta-decay-dominated).
    """
    assert v2_paper.dte_min_days == 5


def test_skip_long_hours_utc_is_us_cash_open(v2_paper):
    """D2 (parity 2026-05-07): V2 backtester mutes longs at US
    cash-open hours (14, 15, 16) UTC — the largest single dollar-impact
    parity divergence per the audit.
    """
    assert v2_paper.skip_long_hours_utc == (14, 15, 16)


def test_use_rnd_scaling_false_under_v2(v2_paper):
    """D6 (parity 2026-05-07): V2 backtester has no RND counterpart;
    the live RNDMonitor is gated behind ``use_rnd_scaling``, default
    ``False`` and stays False under V2 preset.
    """
    assert v2_paper.use_rnd_scaling is False


def test_max_single_trade_btc_disabled_under_v2(v2_paper):
    """D10 (parity 2026-05-07): V2 backtester ``max_position_btc=0.0``
    (= no per-trade hard cap, dynamic_btc_cap is the only ceiling).
    The V2 preset wires this through as 0.0 so RiskLimits and
    StraddleFlow both skip the static cap.
    """
    assert v2_paper.max_single_trade_btc == 0.0


def test_max_jump_fraction_for_shorts_disabled_under_v2(v2_paper):
    """Historical V2 disables the short jump-fraction breaker."""
    assert v2_paper.max_jump_fraction_for_shorts == 1.0


def test_spread_jump_k_disabled_under_v2(v2_paper):
    """Historical V2 disables exponential spread widening on jumps."""
    assert v2_paper.spread_jump_k == 0.0


def test_short_sizing_penalty_disabled_under_v2(v2_paper):
    """Historical V2 disables the short-notional penalty."""
    assert v2_paper.short_sizing_penalty == 1.0


def test_sizing_edge_cap_is_100_under_v2(v2_paper):
    """Historical V2 retains its effectively disabled edge cap."""
    assert v2_paper.sizing_edge_cap == 100.0


def test_sizing_dd_floor_is_1_under_v2(v2_paper):
    """Cycle-17 (2026-05-11, stale-defense L6): V2 preset deactivates the
    no-op ``sizing_dd_floor`` (0.25 -> 1.0). Pixel-identical results
    across [0, 0.75] in the sweep — never bound."""
    assert v2_paper.sizing_dd_floor == 1.0


def test_sizing_liq_cap_is_1_5_under_v2(v2_paper):
    """Historical V2 retains its liquidity cap."""
    assert v2_paper.sizing_liq_cap == 1.5


def test_kill_switch_daily_dd_pct_is_5_under_v2(v2_paper):
    """Historical V2 retains its five-percent daily drawdown threshold."""
    assert v2_paper.kill_switch_daily_dd_pct == 5.0


def test_for_small_account_uses_conservative_default_kill_switch():
    """Small-account paper/testnet config is independent of historical V2."""
    small = PaperConfig.for_small_account()
    assert small.kill_switch_daily_dd_pct == PaperConfig().kill_switch_daily_dd_pct
    assert small.kill_switch_daily_dd_pct == 2.0


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
    """The CLI-safe default does not silently apply historical V2 knobs."""
    assert default_paper.entry_threshold_long == 30.0
    assert default_paper.skip_long_if_term_slope_gt is None
    assert default_paper.target_margin_utilization == 0.05
    assert default_paper.use_profit_scaling is False
    assert default_paper.kelly_mult_path == ""
    assert default_paper.aligner_betas_json == ""
    assert default_paper.bar_exec_quality_path == ""
    # D1, D2, D6, D10 (parity 2026-05-07): default preset preserves
    # default behavior. Only the explicit historical replay factory flips these.
    assert default_paper.dte_min_days == 1
    assert default_paper.skip_long_hours_utc == ()
    assert default_paper.use_rnd_scaling is False
    assert default_paper.max_single_trade_btc == 2.0


# ── Authoritative cross-check ──────────────────────────────────────────────

def test_v2_preset_matches_backtest_v2(v2_paper, v2_backtest):
    """Field-by-field equality on the shared V2 invariant set.

    This guards the archived paper/backtest reconstruction contract. It
    does not establish result validity. If a historical invariant changes,
    both forensic factories must be reviewed together.
    """
    shared_invariants = [
        ("entry_threshold_long",         "entry_threshold_long"),
        ("target_margin_utilization",    "target_margin_utilization"),
        ("use_profit_scaling",           "use_profit_scaling"),
        ("profit_scaling_floor",         "profit_scaling_floor"),
        ("skip_long_if_term_slope_gt",   "skip_long_if_term_slope_gt"),
        ("skip_short_if_skew_1w_gt",     "skip_short_if_skew_1w_gt"),
        ("entry_threshold",              "entry_threshold"),
        ("holding_minutes",              "holding_minutes"),
        ("kelly_mult_path",              "kelly_mult_path"),
        # Parity-cycle 2026-05-07 (D2, D10): align long-hour mute and
        # per-trade size cap with V2 backtester. ``dte_min_days`` (D1)
        # has type int on PaperConfig and float on BacktestConfig
        # (``min_dte_days``); checked separately below to compare numeric
        # values without tripping the ``==`` type check on tuple types.
        ("skip_long_hours_utc",          "skip_long_hours_utc"),
        ("max_single_trade_btc",         "max_position_btc"),
        # Historical V2 disabled the jump-fraction circuit breaker.
        ("max_jump_fraction_for_shorts", "max_jump_fraction_for_shorts"),
        # Historical V2 disabled short-notional halving.
        ("short_sizing_penalty",         "short_sizing_penalty"),
        # Historical V2 disabled exponential spread widening on jumps.
        ("spread_jump_k",                "spread_jump_k"),
        # Historical composite-sizing parity fields.
        ("sizing_edge_cap",              "sizing_edge_cap"),
        ("sizing_dd_floor",              "sizing_dd_floor"),
        ("sizing_liq_cap",               "sizing_liq_cap"),
    ]
    for live_field, backtest_field in shared_invariants:
        live_val = getattr(v2_paper, live_field)
        bt_val = getattr(v2_backtest, backtest_field)
        # Pydantic vs dataclass — both expose ``Field`` defaults the same way.
        assert live_val == bt_val, (
            f"V2 invariant drift: PaperConfig.{live_field}={live_val!r} "
            f"!= BacktestConfig.{backtest_field}={bt_val!r}"
        )
    # D1 (parity 2026-05-07): live ``dte_min_days`` int vs backtester
    # ``min_dte_days`` float — compare numerically.
    assert float(v2_paper.dte_min_days) == float(v2_backtest.min_dte_days), (
        f"V2 invariant drift: PaperConfig.dte_min_days="
        f"{v2_paper.dte_min_days!r} != BacktestConfig.min_dte_days="
        f"{v2_backtest.min_dte_days!r}"
    )


def test_v2_preset_predictions_path_matches_backtest_default(v2_paper, v2_backtest):
    """Historical paper replay and backtest use the same prediction artifact.
    """
    assert v2_paper.aligned_predictions_parquet == v2_backtest.predictions_path


# ── E.2.1: predictions parquet bar frequency matches holding_minutes ──────


def test_aligned_predictions_bar_frequency_matches_holding_minutes(v2_paper):
    """The aligned prediction cadence must match the historical hold interval."""
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
