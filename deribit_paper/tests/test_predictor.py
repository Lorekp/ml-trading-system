"""Tests for Predictor — dynamic half-spread, circuit breaker, signal logic.

Exercises the parity-critical logic that mirrors the PRODUCTION pipeline in
``vol_arb.backtester.VolArbBacktester.run_backtest``:

  - ``compute_spread()`` returns ``base_spread_iv * exp(spread_jump_k * jf)``
    (backtester.py:1204)
  - the production pipeline then PASSES ``spread / 2.0`` to ``generate_signals``
    (backtester.py:3189-3190), so the effective half_spread used in the
    fire/flat decision is ``base_spread_iv * exp(k*jf) / 2.0``.
  - Live (this module) mirrors that with the same ``/ 2.0`` tail at
    ``predictor.py:251`` (cycle-19-REVERT 2026-05-11). Treating
    ``base_spread_iv`` as a half-spread directly would shift live's
    effective threshold by 1.0vp relative to backtester — discovered
    by the post-cycle-19 swarm audit.
  - asymmetric thresholds: long needs 30vp + half_spread, short needs 25vp + half_spread
  - circuit breaker: short signals killed when jump_fraction > max_jump_fraction_for_shorts
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.straddle import StraddleSelection
from deribit_paper.features.tsrv_live import BarTSRV
from deribit_paper.market_data.aggregator import Bar
from deribit_paper.signal.predictor import Predictor, Signal


# ── Fixtures ──────────────────────────────────────────────────────────

def _make_bar(index_price: float = 60_000.0) -> Bar:
    return Bar(
        timestamp=pd.Timestamp("2026-04-10 12:00:00", tz="UTC"),
        bar_minutes=15,
        perp_trades=[],
        option_trades=[],
        index_price=index_price,
    )


def _make_tsrv(jump_fraction: float = 0.0) -> BarTSRV:
    return BarTSRV(
        tsrv=0.001, rv_fast=0.001, rv_slow=0.001, bpv=0.001, medrv=0.001,
        continuous_var=0.001, jump_var=0.0,
        jump_fraction=jump_fraction,
        rs_down=0.0, rs_up=0.0, down_jump_fraction=0.0, up_jump_fraction=0.0,
        n_ticks=200, n_sub_bars=10, n_dash=9.0,
        vol_ann=0.80, vol_ann_continuous=0.80,
    )


def _make_straddle(strike: float = 60_000.0) -> StraddleSelection:
    return StraddleSelection(
        call_instrument="BTC-12APR26-60000-C",
        put_instrument="BTC-12APR26-60000-P",
        strike=strike,
        expiry_dt=pd.Timestamp("2026-04-12 08:00:00", tz="UTC"),
        index_price=60_000.0,
        moneyness=1.0,
        dte_days=2.0,
    )


def _make_predictor(rv_pred_pct: float, iv_used_pct: float) -> Predictor:
    """Build a Predictor stub that returns known rv_pred_pct and iv_used_pct.

    The FeatureAssembler, LightGBM, and SVI are all mocked so the signal
    logic is the only thing exercised.
    """
    # LightGBM stub: predict returns ln(rv_pred_decimal)
    lgbm = MagicMock()
    lgbm.predict.return_value = np.array([math.log(rv_pred_pct / 100.0)])

    # ModelBundle stub
    bundle = MagicMock()
    bundle.lgbm_model = lgbm

    # FeatureAssembler stub
    assembler = MagicMock()
    assembler._bundle = bundle
    assembler.build_row.return_value = pd.Series(np.zeros(51))

    # SVI calibrator stub: returns iv_used_pct
    svi = MagicMock()
    svi.has_params = True
    iv_series = pd.Series([iv_used_pct])
    svi.evaluate.return_value = iv_series

    return Predictor(
        feature_assembler=assembler,
        svi_calibrator=svi,
        entry_threshold_vol_pts=25.0,
        entry_threshold_long_vol_pts=30.0,
        base_spread_iv=1.5,
        spread_jump_k=3.0,
        max_jump_fraction_for_shorts=0.15,
    )


# ── Half-spread (low jump_fraction) ───────────────────────────────────

def test_flat_when_edge_just_below_long_threshold():
    """edge = 30.74vp < eff_long = half_spread(jf=0) + 30 = 0.75 + 30 = 30.75 → FLAT.

    Cycle-19-REVERT (2026-05-11): live's half_spread mirrors the backtester
    PRODUCTION pipeline (compute_spread() / 2.0). With base_spread_iv=1.5
    at jf=0, half_spread=0.75.
    """
    predictor = _make_predictor(rv_pred_pct=105.74, iv_used_pct=75.0)
    # rv - iv = 105.74 - 75 = 30.74; eff_long = 0.75 + 30 = 30.75
    bar = _make_bar()
    tsrv = _make_tsrv(jump_fraction=0.0)
    sig = predictor.predict(bar, tsrv, straddle=_make_straddle())
    assert sig.direction == 0, f"Expected FLAT, got {sig.reason}"


def test_long_when_edge_just_above_long_threshold():
    """edge = 30.76vp > eff_long = 30.75 → LONG (cycle-19-REVERT)."""
    predictor = _make_predictor(rv_pred_pct=105.76, iv_used_pct=75.0)
    bar = _make_bar()
    tsrv = _make_tsrv(jump_fraction=0.0)
    sig = predictor.predict(bar, tsrv, straddle=_make_straddle())
    assert sig.direction == 1, f"Expected LONG, got {sig.reason}"


def test_short_when_edge_just_below_short_threshold():
    """edge = -25.76vp < -eff_short = -(0.75 + 25) = -25.75 → SHORT.

    Cycle-19-REVERT (2026-05-11): eff_short = 25.75 mirroring backtester
    production pipeline (compute_spread / 2.0).
    """
    predictor = _make_predictor(rv_pred_pct=49.24, iv_used_pct=75.0)
    # rv - iv = 49.24 - 75 = -25.76; eff_short = 0.75 + 25 = 25.75
    bar = _make_bar()
    tsrv = _make_tsrv(jump_fraction=0.0)
    sig = predictor.predict(bar, tsrv, straddle=_make_straddle())
    assert sig.direction == -1, f"Expected SHORT, got {sig.reason}"


def test_flat_when_edge_just_inside_short_threshold():
    """edge = -25.74vp > -25.75 → FLAT (cycle-19-REVERT)."""
    predictor = _make_predictor(rv_pred_pct=49.26, iv_used_pct=75.0)
    bar = _make_bar()
    tsrv = _make_tsrv(jump_fraction=0.0)
    sig = predictor.predict(bar, tsrv, straddle=_make_straddle())
    assert sig.direction == 0, f"Expected FLAT, got {sig.reason}"


# ── Dynamic half-spread widens threshold at high jf ───────────────────

def test_long_signal_killed_by_wider_spread_at_high_jf():
    """edge = 31vp is LONG at jf=0 but FLAT at jf=0.3 (spread widens).

    Cycle-19-REVERT (2026-05-11) — half_spread = base_spread_iv * exp(k*jf) / 2:
    At jf=0.3: half_spread = 1.5 * exp(3.0 * 0.3) / 2 ≈ 1.845vp
    eff_long = 1.845 + 30 = 31.845 > 31 → FLAT.
    At jf=0.0: eff_long = 0.75 + 30 = 30.75 < 31 → LONG.
    """
    predictor = _make_predictor(rv_pred_pct=106.0, iv_used_pct=75.0)
    bar = _make_bar()

    sig_low_jf = predictor.predict(bar, _make_tsrv(jump_fraction=0.0), straddle=_make_straddle())
    sig_high_jf = predictor.predict(bar, _make_tsrv(jump_fraction=0.3), straddle=_make_straddle())

    assert sig_low_jf.direction == 1, f"Expected LONG at low jf, got {sig_low_jf.reason}"
    assert sig_high_jf.direction == 0, f"Expected FLAT at high jf, got {sig_high_jf.reason}"


# ── Circuit breaker ───────────────────────────────────────────────────

def test_circuit_breaker_kills_short_above_jf_threshold():
    """SHORT edge is killed when jump_fraction > 0.15 (circuit breaker)."""
    # edge = -40vp — clearly short without circuit breaker
    predictor = _make_predictor(rv_pred_pct=35.0, iv_used_pct=75.0)
    bar = _make_bar()

    sig_safe = predictor.predict(bar, _make_tsrv(jump_fraction=0.10), straddle=_make_straddle())
    sig_cascade = predictor.predict(bar, _make_tsrv(jump_fraction=0.20), straddle=_make_straddle())

    assert sig_safe.direction == -1, f"Expected SHORT at jf=0.10, got {sig_safe.reason}"
    assert sig_cascade.direction == 0, f"Expected FLAT (circuit breaker) at jf=0.20, got {sig_cascade.reason}"
    assert "circuit breaker" in sig_cascade.reason.lower()


def test_circuit_breaker_does_not_affect_long_signals():
    """The jump-fraction circuit breaker ONLY kills shorts, never longs."""
    # edge = +50vp — strong long signal
    predictor = _make_predictor(rv_pred_pct=125.0, iv_used_pct=75.0)
    bar = _make_bar()

    sig = predictor.predict(bar, _make_tsrv(jump_fraction=0.50), straddle=_make_straddle())
    assert sig.direction == 1, f"Expected LONG at high jf, got {sig.reason}"


def test_circuit_breaker_boundary_exactly_at_threshold():
    """jump_fraction exactly at 0.15 should NOT trip the circuit breaker."""
    predictor = _make_predictor(rv_pred_pct=35.0, iv_used_pct=75.0)
    bar = _make_bar()
    sig = predictor.predict(bar, _make_tsrv(jump_fraction=0.15), straddle=_make_straddle())
    # At exactly 0.15, the condition is jf > 0.15 → False → SHORT
    assert sig.direction == -1, f"Expected SHORT at jf=0.15 (boundary), got {sig.reason}"


# ── Flat signal when straddle / SVI unavailable ───────────────────────

def test_flat_when_no_straddle():
    """Without a straddle, the signal must be FLAT (no instrument to trade)."""
    predictor = _make_predictor(rv_pred_pct=125.0, iv_used_pct=75.0)
    bar = _make_bar()
    sig = predictor.predict(bar, _make_tsrv(), straddle=None)
    assert sig.direction == 0
    assert not sig.is_actionable


def test_predictor_falls_back_to_tsrv_proxy_when_svi_uncalibrated():
    """S-3 (parity-r3 cycle 2 — 2026-05-08): when SVI has no params yet,
    the predictor falls back to ``bar_tsrv.vol_ann_continuous × 100`` as
    a DVOL proxy and emits a signal with ``iv_source='tsrv_proxy'``.

    Pre-fix the predictor returned flat — silent emit/skip asymmetry vs
    the backtester's ``svi_iv_matched.fillna(iv).fillna(dvol_close)``
    chain (vol_arb/backtester.py:905-911).
    """
    predictor = _make_predictor(rv_pred_pct=125.0, iv_used_pct=75.0)
    predictor._svi.has_params = False  # simulate cold start
    bar = _make_bar()
    tsrv = _make_tsrv()
    sig = predictor.predict(bar, tsrv, straddle=_make_straddle())
    # S-3: signal NOT flat (fallback fired)
    assert sig.direction != 0, (
        "S-3 regression: predictor should fall back to TSRV proxy on "
        "SVI cold-start, not return flat"
    )
    assert sig.iv_source == "tsrv_proxy"
    # iv_used_pct should equal vol_ann_continuous × 100
    expected_iv = float(tsrv.vol_ann_continuous) * 100.0
    assert sig.iv_used_pct == pytest.approx(expected_iv, rel=1e-6)


def test_predictor_returns_flat_when_both_iv_tiers_unavailable():
    """When BOTH SVI is uncalibrated AND vol_ann_continuous is invalid,
    both fallback tiers fail and the predictor correctly returns flat
    with the iv-unavailable reason."""
    predictor = _make_predictor(rv_pred_pct=125.0, iv_used_pct=75.0)
    predictor._svi.has_params = False
    bar = _make_bar()
    bad_tsrv = BarTSRV(
        tsrv=0.0, rv_fast=0.0, rv_slow=0.0, bpv=0.0, medrv=0.0,
        continuous_var=0.0, jump_var=0.0, jump_fraction=0.0,
        rs_down=0.0, rs_up=0.0, down_jump_fraction=0.0, up_jump_fraction=0.0,
        n_ticks=0, n_sub_bars=0, n_dash=0.0,
        vol_ann=0.0, vol_ann_continuous=0.0,  # invalid → tier-2 fails
    )
    sig = predictor.predict(bar, bad_tsrv, straddle=_make_straddle())
    assert sig.direction == 0
    assert "unavailable" in sig.reason.lower()


# ── Cycle-19-REVERT parity lock: live half_spread must match the EFFECTIVE
# value the backtester pipeline actually passes to generate_signals. The
# previous version of this test compared live to `bt.compute_spread()`
# directly, which is RAW (full spread); the production pipeline at
# `vol_arb/backtester.py:3189-3190` divides by 2.0 before calling
# generate_signals. Cycle-19 (REMOVED) had aligned live to the raw value,
# making live 1vp STRICTER than the production backtester. Cycle-19-REVERT
# (2026-05-11) restores `/ 2.0` in live and aligns this test to the actual
# production effective half_spread.

def test_half_spread_formula_matches_backtester_no_jump():
    """Cycle-19-REVERT (2026-05-11) parity lock: live predictor's half_spread
    must equal the EFFECTIVE half_spread used in the backtester PRODUCTION
    pipeline (compute_spread / 2.0), not the raw compute_spread.

    Live (cycle-19-REVERT): half_spread = base * exp(k*jf) / 2.0
    Backtester production:  spread = base * exp(k*jf); pipeline passes
                            spread / 2.0 to generate_signals (L3189-3190).
    """
    import numpy as np
    from vol_arb.backtester import BacktestConfig, VolArbBacktester
    base = 2.0
    k = 0.0
    bt_cfg = BacktestConfig(base_spread_iv=base, spread_jump_k=k)
    bt = VolArbBacktester(bt_cfg)
    # Effective half_spread the backtester pipeline ACTUALLY uses:
    backtester_eff_half_spread = bt.compute_spread(np.array([0.0, 0.1, 0.3])) / 2.0

    # Live formula (with the /2.0 restored by cycle-19-REVERT)
    live_half_spread = np.array(
        [base * math.exp(k * jf) / 2.0 for jf in [0.0, 0.1, 0.3]]
    )

    np.testing.assert_allclose(
        live_half_spread, backtester_eff_half_spread,
        rtol=1e-12, atol=1e-12,
        err_msg=(
            "Cycle-19-REVERT parity gap: live half_spread != backtester "
            "effective half_spread (compute_spread / 2.0)"
        ),
    )


def test_signal_threshold_matches_backtester_at_boundary():
    """Cycle-19-REVERT (2026-05-11): given identical (edge, jf, thresholds,
    base_spread_iv, spread_jump_k), live and backtester must produce
    the same signal direction.

    CRITICAL: this test now goes through the BACKTESTER PRODUCTION
    pipeline path (compute_spread → spread/2.0 → generate_signals), not
    just `bt.generate_signals` called directly. The previous version of
    this test was a green-light cover for a broken cycle-19 fix because
    it called `bt.generate_signals(half_spread=[compute_spread()[0]])`
    bypassing the `/ 2.0` step at backtester.py:3189-3190.

    Probe: at base_spread_iv=2.0, k=0, entry_threshold=20.0,
    entry_threshold_long=12.0, the BACKTESTER PRODUCTION effective
    thresholds are eff_short = 21.0 and eff_long = 13.0 (because the
    pipeline halves compute_spread before generate_signals). A bar with
    edge = -21.5 should fire SHORT on both; edge = -20.5 should be FLAT
    on both.
    """
    import numpy as np
    from vol_arb.backtester import BacktestConfig, VolArbBacktester
    bt_cfg = BacktestConfig(
        base_spread_iv=2.0, spread_jump_k=0.0,
        entry_threshold=20.0, entry_threshold_long=12.0,
    )
    bt = VolArbBacktester(bt_cfg)
    # Effective half_spread the production pipeline passes to generate_signals
    half_effective = bt.compute_spread(np.array([0.0]))[0] / 2.0
    eff_short = half_effective + 20.0
    eff_long = half_effective + 12.0
    assert eff_short == pytest.approx(21.0)
    assert eff_long == pytest.approx(13.0)
    # Backtester signals at the production-effective boundary
    sig_inside_short = bt.generate_signals(
        edge=np.array([-20.5]),
        half_spread=np.array([half_effective]),
        jump_fraction=np.array([0.0]),
    )[0]
    sig_outside_short = bt.generate_signals(
        edge=np.array([-21.5]),
        half_spread=np.array([half_effective]),
        jump_fraction=np.array([0.0]),
    )[0]
    assert sig_inside_short == 0   # edge -20.5 > -21, FLAT
    assert sig_outside_short == -1 # edge -21.5 < -21, SHORT

    # Now live: at base=2.0, k=0, half_spread = 1.0 (cycle-19-REVERT),
    # eff_short = 21.0, so edge -20.5 must be FLAT and -21.5 must be SHORT.
    iv = 75.0
    predictor_inside = _make_predictor(rv_pred_pct=iv - 20.5, iv_used_pct=iv)
    predictor_inside._base_spread_iv = 2.0
    predictor_inside._spread_jump_k = 0.0
    predictor_inside._entry_threshold = 20.0
    predictor_inside._entry_threshold_long = 12.0
    sig_inside_live = predictor_inside.predict(
        _make_bar(), _make_tsrv(jump_fraction=0.0), straddle=_make_straddle()
    )
    assert sig_inside_live.direction == 0, (
        f"Cycle-19-REVERT parity gap: live fired at edge=-20.5 with eff_short=21.0; "
        f"backtester says FLAT. Got: {sig_inside_live.reason}"
    )

    predictor_outside = _make_predictor(rv_pred_pct=iv - 21.5, iv_used_pct=iv)
    predictor_outside._base_spread_iv = 2.0
    predictor_outside._spread_jump_k = 0.0
    predictor_outside._entry_threshold = 20.0
    predictor_outside._entry_threshold_long = 12.0
    sig_outside_live = predictor_outside.predict(
        _make_bar(), _make_tsrv(jump_fraction=0.0), straddle=_make_straddle()
    )
    assert sig_outside_live.direction == -1, (
        f"Edge -22.5 should fire SHORT in live (eff_short=22.0); got "
        f"{sig_outside_live.direction} ({sig_outside_live.reason})"
    )
