"""Regression locks for BacktestConfig preset invariants.

Added 2026-04-17 foundation fix: prevent accidental drift of the V2 production
hold config back to 30m. The 15m hold was chosen to match the model's
``[D, D+15m]`` prediction horizon; see
``vol_model/output/FOUNDATION_ALIGNMENT_AUDIT.md``.

Generalised 2026-04-21 (Phase 1g of starry-rolling-rabin.md) to cover
per-horizon presets — the parametrized invariant
``test_v2_preset_horizon_matches_holding_minutes`` lifts the original
"hold == 15" lock to "hold == prediction horizon" for any horizon the
plan adds. The legacy literal-15 test
(``test_production_v2_holding_minutes_is_15``) is kept verbatim because
it is the byte-identity guarantee for the existing h=15 production
default.
"""
from __future__ import annotations

import numpy as np
import pytest

from vol_arb.backtester import BacktestConfig, deribit_option_fees_btc
from vol_arb.paths import har_pd_predictions_path, kelly_sidecar_path


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


def test_production_v2_kelly_default_is_K1_slope():
    """V2 preset MUST default to the K1+slope Kelly sidecar (cycle-11).

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
    expected = "vol_arb/output/BTC_v2_kelly_mult_K1_slope.parquet"
    assert cfg.kelly_mult_path == expected, (
        f"V2 preset must auto-load the cycle-11 K1+slope Kelly sidecar "
        f"at {expected!r}; got {cfg.kelly_mult_path!r}. Reverting "
        f"to vanilla K1 forfeits the +$33K NC PnL / 5/5 per-year "
        f"strict / Sharpe-essentially-unchanged improvement — see "
        f"vol_arb/output/PHASE2_LONG_SIZING.md and "
        f"memory/project_phase2_long_sizing_2026_05_10.md."
    )


# ─────────────────────────────────────────────────────────────────────
# 2026-04-21 Phase 1g: generalized foundation invariant.
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("horizon_minutes", [15, 5, 30, 60])
def test_v2_preset_horizon_matches_holding_minutes(horizon_minutes):
    """Generalised foundation-fix invariant: the V2 BTC preset's
    ``holding_minutes`` MUST equal the horizon embedded in its
    ``predictions_path`` (and its ``kelly_mult_path``).

    Construction pattern under test (see ``BacktestConfig`` Phase 1f):
      - h=15: ``with_btc_production_filters_v2()`` (legacy entry point)
      - h≠15: ``with_btc_production_for_horizon(N)`` /
              ``with_btc_production_h{N}m()``

    The invariant catches the foundation-fix failure mode in any future
    horizon: pairing a 60-min predictions parquet with a 15-min hold
    silently throws away half the information, just as 30-min hold on
    15-min predictions did before 2026-04-17. See
    ``project_foundation_fix_2026_04_17.md``.

    NOTE: This test does NOT load the predictions parquet (which may
    not exist for new horizons until Phase 2); it asserts the
    config-level pairing only. The runtime check happens when the
    backtester opens the file in Phase 3.
    """
    if horizon_minutes == 15:
        cfg = BacktestConfig().with_btc_production_filters_v2()
    else:
        cfg = BacktestConfig.with_btc_production_for_horizon(horizon_minutes)
    assert cfg.holding_minutes == horizon_minutes, (
        f"V2 preset at h={horizon_minutes}m has "
        f"holding_minutes={cfg.holding_minutes}; this re-introduces the "
        f"foundation-fix failure mode (model-horizon != backtester-hold)."
    )
    expected_predictions = har_pd_predictions_path(
        "BTCUSDT", horizon_minutes, aligned_bias=True,
    ).as_posix()
    assert cfg.predictions_path == expected_predictions, (
        f"V2 preset at h={horizon_minutes}m has "
        f"predictions_path={cfg.predictions_path!r}; expected "
        f"{expected_predictions!r}. Mismatch means the backtester would "
        f"open a wrong-horizon predictions file silently."
    )
    if horizon_minutes == 15:
        # Cycle-11 (2026-05-10): production V2 at h=15 uses the K1+slope
        # sidecar (walk-forward K1 × 1.25 on long bars at extreme
        # backwardation). Other horizons keep vanilla K1 because the
        # slope-amp sidecar has not been regenerated for those.
        expected_kelly = "vol_arb/output/BTC_v2_kelly_mult_K1_slope.parquet"
    else:
        expected_kelly = kelly_sidecar_path(
            "BTC", horizon_minutes, variant="K1",
        ).as_posix()
    assert cfg.kelly_mult_path == expected_kelly, (
        f"V2 preset at h={horizon_minutes}m has "
        f"kelly_mult_path={cfg.kelly_mult_path!r}; expected "
        f"{expected_kelly!r}. Cross-horizon Kelly load is invalid by "
        f"construction (sidecar is fit to a specific trade log)."
    )


# ─────────────────────────────────────────────────────────────────────
# 2026-05-07 Deribit per-fill fee cap (Part 1 of plan
# c-claude-ml-trading-system-tsrv-pipelin-glistening-bunny.md).
# ─────────────────────────────────────────────────────────────────────


def test_fee_cap_unbounded_typical_v2_trade():
    """V2-typical case: DTE≥1d, ATM, IV≥30% — flat fee dominates, cap
    does NOT bind. Total fee = 4 × 0.0003 × S_exec for one BTC of
    straddle notional.
    """
    sigma = np.array([0.50])         # 50% IV
    T = np.array([1.0 / 365.0])      # 1 day
    S = np.array([100_000.0])
    fees = deribit_option_fees_btc(
        sigma_entry=sigma, T_entry=T,
        sigma_exit=sigma, T_exit=T - 15.0 / (365.25 * 24 * 60),
        S_exec=S, fee_flat=0.0003,
    )
    expected = 4 * 0.0003 * 100_000.0  # $120/BTC notional
    assert fees[0] == pytest.approx(expected, rel=1e-9), (
        "At DTE=1d / σ=50% the 12.5% cap (≈$1310/leg) is far above the "
        "0.0003 BTC flat (=$30/leg), so total fee per BTC must be "
        "exactly 4·0.0003·S_exec. Drift here means the cap formula or "
        "the 4-fill multiplier regressed."
    )


def test_fee_cap_binds_low_premium():
    """Edge case: low σ + short DTE ⇒ premium small ⇒ 12.5% cap binds.
    Per-fill fee = 0.125·premium, NOT the 0.0003 flat. Total fee should
    be strictly less than the unbounded 4·0.0003·S_exec.
    """
    sigma = np.array([0.20])              # 20% IV
    T = np.array([2.0 / (365.0 * 24)])    # 2 hours
    S = np.array([100_000.0])
    prem_btc = 0.4 * 0.20 * np.sqrt(2.0 / (365.0 * 24))  # ~0.00191
    expected_per_fill = 0.125 * prem_btc                  # ~0.000239 BTC < 0.0003 ⇒ binds
    assert expected_per_fill < 0.0003, "Test fixture lost its tail-case property"
    fees = deribit_option_fees_btc(
        sigma_entry=sigma, T_entry=T,
        sigma_exit=sigma, T_exit=T,  # use entry≈exit to isolate the cap behavior
        S_exec=S, fee_flat=0.0003,
    )
    expected = 4 * expected_per_fill * 100_000.0
    assert fees[0] == pytest.approx(expected, rel=1e-9)
    unbounded = 4 * 0.0003 * 100_000.0
    assert fees[0] < unbounded, (
        "Cap must bind for low-σ short-DTE bars; if this asserts the cap "
        "logic is no longer being applied per-fill."
    )


def test_fee_cap_per_fill_entry_vs_exit():
    """Structural property: when entry premium >> exit premium (e.g.
    intraday θ-decay or a fast-moving sigma), entry legs hit the flat
    fee while exit legs hit the cap — the total must reflect a SUM of
    two regimes, not a single-regime value.

    This is the core regression for the bug we just fixed: previously
    only the entry-time cap was applied to all 4 legs.
    """
    sigma_in = np.array([0.50])           # 50% IV at entry — flat dominates
    sigma_ex = np.array([0.20])           # 20% at exit — cap binds
    T_in = np.array([1.0 / 365.0])        # 1 day
    T_ex = np.array([2.0 / (365.0 * 24)]) # 2 hours
    S = np.array([100_000.0])
    fees = deribit_option_fees_btc(
        sigma_entry=sigma_in, T_entry=T_in,
        sigma_exit=sigma_ex, T_exit=T_ex,
        S_exec=S, fee_flat=0.0003,
    )
    fee_entry = 0.0003                                          # cap doesn't bind
    prem_exit_btc = 0.4 * 0.20 * np.sqrt(2.0 / (365.0 * 24))
    fee_exit = 0.125 * prem_exit_btc                            # cap binds
    expected = (2.0 * fee_entry + 2.0 * fee_exit) * 100_000.0
    assert fees[0] == pytest.approx(expected, rel=1e-9), (
        "Per-fill cap must mix entry and exit regimes independently. "
        "If this asserts, the fix at vol_arb/backtester.py:1639 has "
        "regressed back to the entry-only cap behavior."
    )

    # And it must differ from BOTH single-regime values, proving
    # the mix is real (not accidentally collapsing).
    all_flat = 4 * 0.0003 * 100_000.0
    all_capped = 4 * fee_exit * 100_000.0
    assert fees[0] != pytest.approx(all_flat, rel=1e-6)
    assert fees[0] != pytest.approx(all_capped, rel=1e-6)


def test_fee_cap_zero_T_safe():
    """Numerical safety: T → 0 (option about to expire) must not blow up
    or NaN. The function clamps T to 1e-6 internally.
    """
    sigma = np.array([0.5, 0.5])
    T = np.array([0.0, -1e-9])  # at-expiry and just-past
    S = np.array([100_000.0, 100_000.0])
    fees = deribit_option_fees_btc(
        sigma_entry=sigma, T_entry=T,
        sigma_exit=sigma, T_exit=T,
        S_exec=S, fee_flat=0.0003,
    )
    assert np.all(np.isfinite(fees))
    # At T=0 the cap binds aggressively — fee should be tiny, not negative.
    assert np.all(fees >= 0.0)
    assert np.all(fees < 4 * 0.0003 * 100_000.0)
