"""Tests for Predictor — dynamic half-spread, circuit breaker, signal logic.

Exercises the parity-critical logic that mirrors backtester.py:450-481:
  - dynamic half-spread = base_spread_iv * exp(spread_jump_k * jf) / 2
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
    """edge = 30.74vp < eff_long = half_spread(jf=0) + 30 = 0.75 + 30 = 30.75 → FLAT."""
    predictor = _make_predictor(rv_pred_pct=105.74, iv_used_pct=75.0)
    # rv - iv = 105.74 - 75 = 30.74; eff_long = 0.75 + 30 = 30.75
    bar = _make_bar()
    tsrv = _make_tsrv(jump_fraction=0.0)
    sig = predictor.predict(bar, tsrv, straddle=_make_straddle())
    assert sig.direction == 0, f"Expected FLAT, got {sig.reason}"


def test_long_when_edge_just_above_long_threshold():
    """edge = 30.76vp > eff_long = 30.75 → LONG."""
    predictor = _make_predictor(rv_pred_pct=105.76, iv_used_pct=75.0)
    bar = _make_bar()
    tsrv = _make_tsrv(jump_fraction=0.0)
    sig = predictor.predict(bar, tsrv, straddle=_make_straddle())
    assert sig.direction == 1, f"Expected LONG, got {sig.reason}"


def test_short_when_edge_just_below_short_threshold():
    """edge = -25.76vp < -eff_short = -(0.75 + 25) = -25.75 → SHORT."""
    predictor = _make_predictor(rv_pred_pct=49.24, iv_used_pct=75.0)
    # rv - iv = 49.24 - 75 = -25.76; eff_short = 0.75 + 25 = 25.75
    bar = _make_bar()
    tsrv = _make_tsrv(jump_fraction=0.0)
    sig = predictor.predict(bar, tsrv, straddle=_make_straddle())
    assert sig.direction == -1, f"Expected SHORT, got {sig.reason}"


def test_flat_when_edge_just_inside_short_threshold():
    """edge = -25.74vp > -25.75 → FLAT."""
    predictor = _make_predictor(rv_pred_pct=49.26, iv_used_pct=75.0)
    bar = _make_bar()
    tsrv = _make_tsrv(jump_fraction=0.0)
    sig = predictor.predict(bar, tsrv, straddle=_make_straddle())
    assert sig.direction == 0, f"Expected FLAT, got {sig.reason}"


# ── Dynamic half-spread widens threshold at high jf ───────────────────

def test_long_signal_killed_by_wider_spread_at_high_jf():
    """edge = 31vp is LONG at jf=0 but FLAT at jf=0.3 (spread widens).

    At jf=0.3: half_spread = 1.5 * exp(3.0 * 0.3) / 2 ≈ 1.84vp
    eff_long = 1.84 + 30 = 31.84 > 31 → FLAT.
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


def test_flat_when_svi_not_yet_calibrated():
    """If SVI has no params yet, fall back to FLAT (iv_used unavailable)."""
    predictor = _make_predictor(rv_pred_pct=125.0, iv_used_pct=75.0)
    predictor._svi.has_params = False  # simulate cold start
    bar = _make_bar()
    sig = predictor.predict(bar, _make_tsrv(), straddle=_make_straddle())
    assert sig.direction == 0
    assert "unavailable" in sig.reason.lower()
