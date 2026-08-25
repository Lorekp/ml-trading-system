"""Regression locks for BacktestConfig preset invariants.

The historical V2 hold remains locked to the model horizon. These tests
preserve forensic configuration parity; they do not validate V2 performance.

Generalised 2026-04-21 to cover
per-horizon presets — the parametrized invariant
``test_v2_preset_horizon_matches_holding_minutes`` lifts the original
"hold == 15" lock to "hold == prediction horizon" for any horizon the
configuration supports. The literal-15 test
(``test_historical_v2_holding_minutes_is_15``) is kept verbatim because
it is the regression guarantee for the historical h=15 default.
"""
from __future__ import annotations

import numpy as np
import pytest

from vol_arb.backtester import BacktestConfig, deribit_option_fees_btc
from vol_arb.paths import har_pd_predictions_path, kelly_sidecar_path


def test_historical_v2_holding_minutes_is_15():
    """Historical V2 hold must stay aligned with prediction horizon.

    Prediction target: ln(vol_ann over [D, D+15m]). A 30-min hold would cover
    [D, D+30m] — first half signal-aligned, second half un-forecast theta
    carry. Locked at 15 min as of the foundation fix.
    """
    cfg = BacktestConfig().with_historical_btc_v2()
    assert cfg.holding_minutes == 15, (
        f"V2 holding_minutes must be 15 to match the model's "
        f"[D, D+15m] prediction window; got {cfg.holding_minutes}. "
        f"A different hold would re-introduce model/position misalignment."
    )


def test_historical_v2_core_filters_preserved():
    """Sanity check: other V2 filters we rely on are still set as expected.

    Foundation-fix Step 6 should have ONLY touched holding_minutes. Any other
    field drifting here is a sign of an accidental edit.
    """
    cfg = BacktestConfig().with_historical_btc_v2()
    assert cfg.use_drl_hedging is True
    assert cfg.use_profit_scaling is True
    assert cfg.profit_scaling_floor == 0.3
    assert cfg.target_margin_utilization == 0.10
    assert cfg.entry_threshold_long == 12.0
    assert cfg.skip_long_if_term_slope_gt == 0.0


def test_historical_v1_holding_minutes_is_15():
    """V1 hold was already 15m pre-fix — confirm we didn't break it."""
    cfg = BacktestConfig().with_historical_btc_v1()
    assert cfg.holding_minutes == 15


def test_default_predictions_path_is_aligned_bias():
    """The historical default retains its aligned-bias sidecar path."""
    cfg = BacktestConfig()
    assert cfg.predictions_path.endswith("aligned_bias.parquet"), (
        f"Default predictions_path should point at the aligned_bias "
        f"parquet used by the historical configuration. "
        f"Got: {cfg.predictions_path}."
    )


def test_v2_jump_fraction_breaker_disabled():
    """Historical V2 disables the breaker; the generic default does not."""
    cfg = BacktestConfig().with_historical_btc_v2()
    assert cfg.max_jump_fraction_for_shorts == 1.0, (
        f"V2 preset must set max_jump_fraction_for_shorts=1.0; "
        f"got {cfg.max_jump_fraction_for_shorts}."
    )
    # Bare default preserves legacy value for non-V2 callers.
    assert BacktestConfig().max_jump_fraction_for_shorts == 0.15


def test_v2_spread_jump_k_disabled():
    """Historical V2 removes jump widening; the generic default retains it."""
    cfg = BacktestConfig().with_historical_btc_v2()
    assert cfg.spread_jump_k == 0.0, (
        f"V2 preset must set spread_jump_k=0.0; got {cfg.spread_jump_k}."
    )
    # Bare default preserves legacy value for non-V2 callers.
    assert BacktestConfig().spread_jump_k == 3.0


def test_v2_short_sizing_penalty_disabled():
    """Historical V2 disables the penalty; the generic default retains it."""
    cfg = BacktestConfig().with_historical_btc_v2()
    assert cfg.short_sizing_penalty == 1.0, (
        f"V2 preset must set short_sizing_penalty=1.0; "
        f"got {cfg.short_sizing_penalty}."
    )
    # Bare default preserves legacy value for non-V2 callers.
    assert BacktestConfig().short_sizing_penalty == 0.5


def test_v2_sizing_edge_cap_is_100():
    """Cycle-17 (2026-05-11, stale-defense L6): V2 preset deactivates
    ``sizing_edge_cap`` by raising it from 3.0 to 100.0 (effectively off).

    This is a regression lock for invalidated historical settings, not an
    endorsement of the tuning decision.
    """
    cfg = BacktestConfig().with_historical_btc_v2()
    assert cfg.sizing_edge_cap == 100.0, (
        f"V2 preset must set sizing_edge_cap=100.0 (cycle-17); "
        f"got {cfg.sizing_edge_cap}."
    )


def test_v2_sizing_dd_floor_is_1():
    """Cycle-17 (2026-05-11, stale-defense L6): V2 preset deactivates
    ``sizing_dd_floor`` by raising it from 0.25 to 1.0 (effectively off).

    The 0.25 floor was a min-scale on drawdown but was pixel-identical
    in results across [0, 0.75] — it never bound under V2 because the
    composite_v2 strategy already clamps dd_scale before this floor.
    """
    cfg = BacktestConfig().with_historical_btc_v2()
    assert cfg.sizing_dd_floor == 1.0, (
        f"V2 preset must set sizing_dd_floor=1.0 (cycle-17); "
        f"got {cfg.sizing_dd_floor}."
    )


def test_v2_sizing_liq_cap_is_1_5():
    """Historical V2 retains ``sizing_liq_cap=1.5``."""
    cfg = BacktestConfig().with_historical_btc_v2()
    assert cfg.sizing_liq_cap == 1.5, (
        f"V2 preset must keep sizing_liq_cap=1.5; got {cfg.sizing_liq_cap}."
    )


def test_v2_base_spread_iv_under_realistic_costs_is_2():
    """Parity-r3 cycle-2 (2026-05-08): the historical V2 recipe is
    ``BacktestConfig().with_historical_btc_v2().with_realistic_costs()``,
    where ``with_realistic_costs()`` sets ``base_spread_iv = 2.0``.

    Live's V2 preset bakes this in directly; this test locks the
    backtester side to the same value when the realistic-costs
    composition is applied by the historical regeneration script.
    """
    cfg = BacktestConfig().with_historical_btc_v2().with_realistic_costs()
    assert cfg.base_spread_iv == 2.0, (
        f"Historical V2 recipe must yield base_spread_iv=2.0; "
        f"got {cfg.base_spread_iv}. NOTE: this is the FULL spread; "
        f"the backtester pipeline halves it at "
        f"backtester.py:3189-3190 before generate_signals. Live "
        f"mirrors this with the /2.0 tail in predictor.py:251 "
        f"(cycle-20 revert of cycle-19's wrong-direction fix)."
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


def test_historical_v2_kelly_default_is_K1_slope():
    """Historical V2 retains the K1+slope sidecar for forensic parity."""
    cfg = BacktestConfig().with_historical_btc_v2()
    expected = "vol_arb/output/BTC_v2_kelly_mult_K1_slope.parquet"
    assert cfg.kelly_mult_path == expected, (
        f"V2 preset must auto-load the cycle-11 K1+slope Kelly sidecar "
        f"at {expected!r}; got {cfg.kelly_mult_path!r}."
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
      - h=15: ``with_historical_btc_v2()`` (historical entry point)
      - h≠15: ``with_historical_btc_v2_for_horizon(N)`` /
              ``with_historical_btc_v2_h{N}m()``

    The invariant catches the foundation-fix failure mode in any future
    horizon: pairing a 60-min predictions parquet with a 15-min hold
    silently throws away half the information, just as 30-min hold on
    15-min predictions did before 2026-04-17.

    NOTE: This test does NOT load the predictions parquet (which may
    not exist for new horizons until Phase 2); it asserts the
    config-level pairing only. The runtime check happens when the
    backtester opens the file in Phase 3.
    """
    if horizon_minutes == 15:
        cfg = BacktestConfig().with_historical_btc_v2()
    else:
        cfg = BacktestConfig.with_historical_btc_v2_for_horizon(horizon_minutes)
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
        # Cycle-11 (2026-05-10): historical V2 at h=15 uses the K1+slope
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
# 2026-05-07 Deribit per-fill fee-cap regressions.
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


# ─────────────────────────────────────────────────────────────────────
# MF-3: per-horizon factory predictions-passthrough (eval plumbing fix)
# ─────────────────────────────────────────────────────────────────────
#
# Before MF-3, ``with_historical_btc_v2_for_horizon(H)`` HARDCODED
# predictions_path = aligned_bias(H) and silently dropped any
# caller-supplied path, so the downstream V2 backtest at h!=15 scored the
# WRONG (stale h15-tuned) predictions or crashed-then-swallowed where the
# aligned_bias file is missing. The fix adds a ``**overrides`` passthrough
# applied LAST via model_copy. These tests lock it for the full fleet.

# Historical fleet horizons (120m was excluded). The h=15 path uses the named
# historical preset, but this factory must still honor an h=15 override.
_FLEET_HORIZONS = [5, 15, 30, 60, 240, 480, 1440]


@pytest.mark.parametrize("h", _FLEET_HORIZONS)
def test_for_horizon_predictions_override_wins(h):
    """A caller-supplied predictions_path MUST survive into the final
    config at every fleet horizon (MF-3). This is the core regression:
    config.predictions_path == override for all 7 horizons."""
    override = f"vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions_h{h}m_no_tv.parquet"
    cfg = BacktestConfig.with_historical_btc_v2_for_horizon(
        h, predictions_path=override,
    )
    assert cfg.predictions_path == override, (
        f"h={h}: predictions override dropped — got {cfg.predictions_path!r}, "
        f"expected {override!r}. The MF-3 discard bug has regressed: the "
        f"per-horizon factory must apply **overrides LAST via model_copy."
    )


@pytest.mark.parametrize("h", _FLEET_HORIZONS)
def test_for_horizon_kelly_off_override(h):
    """``kelly_mult_path=None`` must propagate (None is a meaningful value:
    it disables the h15-tuned Kelly overlay for the bare V2 comparison
    stack). The horizon-aware default would otherwise point at a K1 sidecar
    that the cross-horizon job does NOT regenerate per horizon."""
    cfg = BacktestConfig.with_historical_btc_v2_for_horizon(
        h, kelly_mult_path=None,
    )
    assert cfg.kelly_mult_path is None, (
        f"h={h}: kelly_mult_path=None override dropped — got "
        f"{cfg.kelly_mult_path!r}. The bare V2 comparison stack requires "
        f"Kelly OFF; a falsey-filtering passthrough would silently keep the "
        f"K1 default."
    )


def test_for_horizon_no_override_preserves_horizon_defaults():
    """Without overrides the factory still composes the horizon-aware
    aligned_bias predictions + K1 kelly defaults (byte-identity for the
    pre-MF-3 callers / the named per-horizon presets)."""
    h = 30
    cfg = BacktestConfig.with_historical_btc_v2_for_horizon(h)
    expected_preds = har_pd_predictions_path(
        "BTCUSDT", h, aligned_bias=True,
    ).as_posix()
    expected_kelly = kelly_sidecar_path("BTC", h, variant="K1").as_posix()
    assert cfg.predictions_path == expected_preds
    assert cfg.kelly_mult_path == expected_kelly


def test_for_horizon_combined_bare_v2_comparison_stack():
    """The exact stack run_horizon_job.score_downstream builds for the
    cross-horizon comparison: fresh _no_tv predictions + Kelly OFF, at a
    non-15m horizon. Both overrides must land together."""
    h = 240
    preds = f"vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions_h{h}m_no_tv.parquet"
    cfg = BacktestConfig.with_historical_btc_v2_for_horizon(
        h, predictions_path=preds, kelly_mult_path=None,
    )
    assert cfg.predictions_path == preds
    assert cfg.kelly_mult_path is None
    # V2 horizon invariants must still come along.
    assert cfg.use_drl_hedging is True
    assert cfg.holding_minutes == h


def test_for_horizon_h15_named_preset_byte_identity_unchanged():
    """h=15 sacrosanct: the named legacy preset (what run_backtest.py uses
    at h=15) is untouched by the MF-3 factory change — still aligned_bias
    + K1_slope. The factory passthrough is additive only."""
    cfg = BacktestConfig().with_historical_btc_v2()
    assert cfg.predictions_path == (
        "vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet"
    )
    assert cfg.kelly_mult_path == "vol_arb/output/BTC_v2_kelly_mult_K1_slope.parquet"
