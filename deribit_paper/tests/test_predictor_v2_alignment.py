"""Tests for V2-alignment wiring in :class:`Predictor`.

Covers C2 (β* per-month aligner) and the contango filter (skip-long
when ``svi_term_slope_1w_2w > threshold``) wired in
``deribit_paper/signal/predictor.py``. Plan:
``C:\\Users\\laure\\.claude\\plans\\live-engine-h15-alignment-2026-04-26\\``.

Two contracts under test:

1. **β* application**: ``Predictor`` adds the calendar-month β* to the
   raw LGBM log-vol output before exp(); the resulting ``rv_pred_pct``
   is shifted by ``exp(β*) - 1`` relative to the no-β baseline. Empty
   dict → no shift. Live month past sidecar → most-recent-prior fallback.

2. **Contango filter**: when ``skip_long_if_term_slope_gt`` is set and a
   LONG candidate fires at a regime where ``svi_atm_2w - svi_atm_1w`` >
   threshold, the signal is suppressed to FLAT; if ≤ threshold or
   threshold is ``None``, the LONG passes through.
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
from deribit_paper.signal.predictor import Predictor


def _make_bar(
    *, ts: str = "2026-04-15 12:00:00", index_price: float = 60_000.0,
) -> Bar:
    return Bar(
        timestamp=pd.Timestamp(ts, tz="UTC"),
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


def _make_straddle() -> StraddleSelection:
    return StraddleSelection(
        call_instrument="BTC-17APR26-60000-C",
        put_instrument="BTC-17APR26-60000-P",
        strike=60_000.0,
        expiry_dt=pd.Timestamp("2026-04-17 08:00:00", tz="UTC"),
        index_price=60_000.0,
        moneyness=1.0,
        dte_days=2.0,
    )


def _make_predictor(
    *,
    rv_pred_pct_target: float,
    iv_used_pct: float,
    aligner_betas: dict[str, float] | None = None,
    skip_long_if_term_slope_gt: float | None = None,
    atm_1w_pct: float = 70.0,
    atm_2w_pct: float = 70.0,
) -> Predictor:
    """Build a Predictor stub that targets a specific rv_pred_pct AFTER β*.

    The LGBM stub returns the raw ln(rv_pred_decimal) such that, after the
    predictor adds β*, the final rv_pred_pct equals ``rv_pred_pct_target``.
    """
    betas = dict(aligner_betas or {})
    if betas:
        from vol_arb.sidecars import lookup_beta_for_month
        ts = pd.Timestamp("2026-04-15 12:00:00", tz="UTC")
        beta = lookup_beta_for_month(betas, ts, fallback_zero_warn=False)
    else:
        beta = 0.0

    target_decimal = rv_pred_pct_target / 100.0
    raw_pred_ln = math.log(target_decimal) - beta

    lgbm = MagicMock()
    lgbm.predict.return_value = np.array([raw_pred_ln])

    bundle = MagicMock()
    bundle.lgbm_model = lgbm

    assembler = MagicMock()
    assembler._bundle = bundle
    assembler.build_row.return_value = pd.Series(np.zeros(51))

    svi = MagicMock()
    svi.has_params = True
    svi.evaluate.return_value = pd.Series([iv_used_pct])

    def _atm_vol_at_dte(dte: float) -> float:
        if dte <= 7.5:
            return atm_1w_pct
        return atm_2w_pct
    svi.atm_vol_at_dte.side_effect = _atm_vol_at_dte

    return Predictor(
        feature_assembler=assembler,
        svi_calibrator=svi,
        entry_threshold_vol_pts=22.0,
        entry_threshold_long_vol_pts=12.0,
        base_spread_iv=1.5,
        spread_jump_k=3.0,
        max_jump_fraction_for_shorts=0.15,
        skip_long_if_term_slope_gt=skip_long_if_term_slope_gt,
        aligner_betas=betas,
    )


# ── β* application ────────────────────────────────────────────────────────

def test_beta_star_shifts_rv_pred_upward():
    """β=+0.05 should make rv_pred_pct ~5% higher than the no-β baseline."""
    rv_target_no_beta = 70.0
    bar = _make_bar(ts="2026-04-15 12:00:00")
    tsrv = _make_tsrv()
    straddle = _make_straddle()
    iv = 60.0  # below rv → LONG candidate

    p_no_beta = _make_predictor(
        rv_pred_pct_target=rv_target_no_beta, iv_used_pct=iv,
    )
    sig_no_beta = p_no_beta.predict(bar, tsrv, straddle=straddle)

    p_with_beta = _make_predictor(
        rv_pred_pct_target=rv_target_no_beta * math.exp(0.05),
        iv_used_pct=iv,
        aligner_betas={"2026-04": 0.05},
    )
    sig_with_beta = p_with_beta.predict(bar, tsrv, straddle=straddle)

    assert sig_with_beta.rv_pred_pct > sig_no_beta.rv_pred_pct
    ratio = sig_with_beta.rv_pred_pct / sig_no_beta.rv_pred_pct
    assert 1.04 < ratio < 1.07


def test_empty_betas_is_passthrough():
    """No aligner sidecar → predict() must pass raw LGBM through unchanged."""
    bar = _make_bar()
    tsrv = _make_tsrv()
    straddle = _make_straddle()

    p = _make_predictor(
        rv_pred_pct_target=72.0, iv_used_pct=70.0,
        aligner_betas=None,
    )
    sig = p.predict(bar, tsrv, straddle=straddle)
    assert sig.rv_pred_pct == pytest.approx(72.0, rel=1e-9)


def test_beta_star_fallback_to_most_recent_prior():
    """Live April with sidecar covering only Jan-Mar → uses March β*."""
    bar = _make_bar(ts="2026-04-15 12:00:00")
    tsrv = _make_tsrv()
    straddle = _make_straddle()
    betas = {"2026-01": 0.03, "2026-02": 0.04, "2026-03": 0.05}

    p = _make_predictor(
        rv_pred_pct_target=70.0 * math.exp(0.05),
        iv_used_pct=60.0,
        aligner_betas=betas,
    )
    sig = p.predict(bar, tsrv, straddle=straddle)
    assert sig.rv_pred_pct == pytest.approx(70.0 * math.exp(0.05), rel=1e-9)


# ── Contango filter ───────────────────────────────────────────────────────

def test_long_suppressed_when_term_slope_positive():
    """LONG candidate + term slope > 0 + threshold=0 → FLAT."""
    p = _make_predictor(
        rv_pred_pct_target=85.0, iv_used_pct=60.0,
        skip_long_if_term_slope_gt=0.0,
        atm_1w_pct=70.0, atm_2w_pct=72.0,
    )
    bar = _make_bar()
    sig = p.predict(bar, _make_tsrv(), straddle=_make_straddle())
    assert sig.direction == 0
    assert "contango" in sig.reason.lower()


def test_long_passes_when_term_slope_zero():
    """Term slope = 0.0 is NOT > 0.0 (strict gt) → LONG passes."""
    p = _make_predictor(
        rv_pred_pct_target=85.0, iv_used_pct=60.0,
        skip_long_if_term_slope_gt=0.0,
        atm_1w_pct=70.0, atm_2w_pct=70.0,
    )
    bar = _make_bar()
    sig = p.predict(bar, _make_tsrv(), straddle=_make_straddle())
    assert sig.direction == 1


def test_long_passes_when_term_slope_negative():
    """Backwardation (atm_2w < atm_1w) → LONG passes."""
    p = _make_predictor(
        rv_pred_pct_target=85.0, iv_used_pct=60.0,
        skip_long_if_term_slope_gt=0.0,
        atm_1w_pct=72.0, atm_2w_pct=70.0,
    )
    bar = _make_bar()
    sig = p.predict(bar, _make_tsrv(), straddle=_make_straddle())
    assert sig.direction == 1


def test_short_unaffected_by_contango_filter():
    """Filter is long-side only; SHORT signals are unaffected."""
    p = _make_predictor(
        rv_pred_pct_target=30.0, iv_used_pct=60.0,
        skip_long_if_term_slope_gt=0.0,
        atm_1w_pct=70.0, atm_2w_pct=72.0,
    )
    bar = _make_bar()
    sig = p.predict(bar, _make_tsrv(), straddle=_make_straddle())
    assert sig.direction == -1


def test_filter_disabled_when_threshold_none():
    """skip_long_if_term_slope_gt=None → filter never fires."""
    p = _make_predictor(
        rv_pred_pct_target=85.0, iv_used_pct=60.0,
        skip_long_if_term_slope_gt=None,
        atm_1w_pct=70.0, atm_2w_pct=80.0,
    )
    bar = _make_bar()
    sig = p.predict(bar, _make_tsrv(), straddle=_make_straddle())
    assert sig.direction == 1


def test_filter_passes_through_when_atm_dte_returns_nan():
    """SVI calibrated but atm_vol_at_dte returns NaN → term_slope None → LONG passes.

    Defensive contract: don't suppress signals when the slope helper
    can't return a finite value, even though SVI itself is "calibrated".
    Testing this rather than ``has_params=False`` because the latter
    short-circuits at ``_lookup_iv`` (step 3) before the contango filter
    (step 4) is reached.
    """
    p = _make_predictor(
        rv_pred_pct_target=85.0, iv_used_pct=60.0,
        skip_long_if_term_slope_gt=0.0,
        atm_1w_pct=float("nan"),                # forces _compute_term_slope_1w_2w to return None
        atm_2w_pct=72.0,
    )
    bar = _make_bar()
    sig = p.predict(bar, _make_tsrv(), straddle=_make_straddle())
    assert sig.direction == 1


# ── E.1.8: term_slope cached on Signal at predict time ───────────────────


def test_term_slope_cached_on_signal():
    """The Signal carries the term_slope value sampled at predict time,
    not at trade-close time. Pre-E.1.8 the engine called the predictor's
    private ``_compute_term_slope_1w_2w`` 15 minutes later — different
    SVI state, different slope."""
    p = _make_predictor(
        rv_pred_pct_target=85.0, iv_used_pct=60.0,
        atm_1w_pct=70.0, atm_2w_pct=72.0,
    )
    bar = _make_bar()
    sig = p.predict(bar, _make_tsrv(), straddle=_make_straddle())
    assert sig.term_slope_1w_2w == pytest.approx(2.0)  # 72 - 70


def test_term_slope_none_when_svi_uncalibrated():
    """Uncalibrated SVI → ``term_slope_1w_2w`` is None on the Signal."""
    p = _make_predictor(
        rv_pred_pct_target=85.0, iv_used_pct=60.0,
        atm_1w_pct=70.0, atm_2w_pct=72.0,
    )
    p._svi.has_params = False
    bar = _make_bar()
    # has_params=False short-circuits _lookup_iv → flat. term_slope still
    # populated from the predict-time computation, but here also None
    # because the slope helper bails on uncalibrated SVI.
    sig = p.predict(bar, _make_tsrv(), straddle=_make_straddle())
    assert sig.term_slope_1w_2w is None


# ── E.1.14: require_aligner fail-fast on empty dict ──────────────────────


def test_require_aligner_raises_on_empty_dict():
    """V2 preset wires ``require_aligner=True``; empty β* dict at init
    raises rather than silently bypassing β* every bar."""
    from deribit_paper.signal.predictor import Predictor

    bundle = MagicMock()
    bundle.lgbm_model = MagicMock()
    assembler = MagicMock()
    assembler._bundle = bundle
    svi = MagicMock()
    svi.has_params = True

    with pytest.raises(RuntimeError, match="aligner"):
        Predictor(
            feature_assembler=assembler,
            svi_calibrator=svi,
            aligner_betas={},
            require_aligner=True,
        )


def test_require_aligner_passes_with_populated_dict():
    """Populated dict + require_aligner=True → no error."""
    from deribit_paper.signal.predictor import Predictor

    bundle = MagicMock()
    bundle.lgbm_model = MagicMock()
    assembler = MagicMock()
    assembler._bundle = bundle
    svi = MagicMock()
    svi.has_params = True

    Predictor(
        feature_assembler=assembler,
        svi_calibrator=svi,
        aligner_betas={"2026-01": 0.05},
        require_aligner=True,
    )


def test_require_aligner_default_false_allows_empty():
    """Default preset (require_aligner=False) accepts empty dict."""
    from deribit_paper.signal.predictor import Predictor

    bundle = MagicMock()
    bundle.lgbm_model = MagicMock()
    assembler = MagicMock()
    assembler._bundle = bundle
    svi = MagicMock()
    svi.has_params = True

    Predictor(
        feature_assembler=assembler,
        svi_calibrator=svi,
        aligner_betas={},
        require_aligner=False,
    )
