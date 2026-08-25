"""D2 (parity 2026-05-07): predictor mutes LONG signals at configured
UTC hours.

Mirrors the invalidated historical V2 backtester's
``_apply_entry_filters`` long-only filter. These checks preserve the entry
filter's live/offline parity behavior; they are not evidence that the strategy
or its historical PnL is valid.

Tests:
1. ``test_long_muted_at_skip_hour`` — bar at 14:00 UTC, edge above long
   threshold → direction flips from +1 to 0 (FLAT).
2. ``test_short_NOT_muted_at_skip_hour`` — bar at 14:00 UTC, edge below
   short threshold → SHORT (-1) fires unaffected.
3. ``test_long_fires_at_non_skip_hour`` — bar at 13:00 UTC, edge above
   long threshold → LONG (+1) fires.
4. ``test_empty_skip_hours_no_mute`` — empty tuple disables the filter
   on the predictor end (default constructor).
5. ``test_combined_with_contango_filter`` — both filters fire on the
   same bar; predictor stays FLAT (idempotent — both transitions to
   FLAT compose cleanly).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.straddle import StraddleSelection
from deribit_paper.features.tsrv_live import BarTSRV
from deribit_paper.market_data.aggregator import Bar
from deribit_paper.signal.predictor import Predictor


def _make_bar(timestamp_str: str) -> Bar:
    """Construct a minimal Bar at ``timestamp_str`` (UTC)."""
    return Bar(
        timestamp=pd.Timestamp(timestamp_str, tz="UTC"),
        bar_minutes=15,
        perp_trades=[],
        option_trades=[],
        index_price=100_000.0,
    )


def _make_tsrv() -> BarTSRV:
    """A neutral, finite BarTSRV that won't trip jump-fraction guards."""
    return BarTSRV(
        tsrv=0.04,
        rv_fast=0.04,
        rv_slow=0.04,
        bpv=0.04,
        medrv=0.04,
        continuous_var=0.04,
        jump_var=0.0,
        jump_fraction=0.0,
        rs_down=0.02,
        rs_up=0.02,
        down_jump_fraction=0.0,
        up_jump_fraction=0.0,
        n_ticks=100,
        n_sub_bars=10,
        n_dash=20.0,
        vol_ann=0.40,
        vol_ann_continuous=0.40,
    )


def _make_straddle(expiry_dt: str = "2026-05-21") -> StraddleSelection:
    """An ATM straddle pair with a 14-day DTE."""
    s = MagicMock(spec=StraddleSelection)
    s.strike = 100_000.0
    s.expiry_dt = pd.Timestamp(expiry_dt, tz="UTC")
    s.index_price = 100_000.0
    s.call_instrument = "BTC-21MAY26-100000-C"
    s.put_instrument = "BTC-21MAY26-100000-P"
    return s


def _make_predictor(
    *,
    skip_long_hours_utc: tuple[int, ...] = (),
    skip_long_if_term_slope_gt: float | None = None,
    rv_pred_ln: float = np.log(0.80),  # 80% vol-ann → much higher than 40% IV → LONG
    iv_used_pct: float = 40.0,
) -> Predictor:
    """Build a predictor with mocked LightGBM / SVI / assembler.

    The mocks return deterministic values so the only unknown in the
    signal direction is the hour-of-day filter under test.
    """
    # Mock bundle: predict() always returns ``rv_pred_ln`` so edge sign
    # is controlled by ``iv_used_pct``.
    bundle = MagicMock()
    bundle.lgbm_model.predict = MagicMock(
        return_value=np.array([rv_pred_ln], dtype=float),
    )

    # Mock assembler: build_row returns a non-None Series so the
    # NaN-feature short-circuit doesn't fire.
    assembler = MagicMock()
    assembler._bundle = bundle
    assembler.build_row = MagicMock(
        return_value=pd.Series(np.zeros(44), name="features"),
    )

    # Mock SVI calibrator: return iv_used_pct as a single-row DF with
    # value in % (predictor converts via /100 internally? No — predictor
    # uses ``iv`` directly as a percentage. Confirmed via _lookup_iv).
    svi = MagicMock()
    svi.has_params = True
    svi.evaluate = MagicMock(
        return_value=pd.Series([iv_used_pct], dtype=float),
    )
    # Term-slope: return None (no contango) unless test overrides
    svi.atm_vol_at_dte = MagicMock(return_value=40.0)

    return Predictor(
        feature_assembler=assembler,
        svi_calibrator=svi,
        entry_threshold_vol_pts=22.0,
        entry_threshold_long_vol_pts=12.0,
        base_spread_iv=1.5,
        spread_jump_k=3.0,
        max_jump_fraction_for_shorts=0.15,
        skip_long_if_term_slope_gt=skip_long_if_term_slope_gt,
        skip_long_hours_utc=skip_long_hours_utc,
        aligner_betas=None,
        require_aligner=False,
        log_every_bar=False,
    )


# ── 1. Long muted at skip hour ─────────────────────────────────────────────


@pytest.mark.parametrize("hour", [14, 15, 16])
def test_long_muted_at_skip_hour(hour):
    """At 14/15/16 UTC, a LONG candidate (edge >> threshold) → FLAT."""
    p = _make_predictor(skip_long_hours_utc=(14, 15, 16))
    bar = _make_bar(f"2026-05-07 {hour:02d}:00")
    sig = p.predict(bar, _make_tsrv(), straddle=_make_straddle())
    assert sig.direction == 0
    assert "V2 hour filter" in sig.reason
    assert f"{hour:02d}:00 UTC" in sig.reason


# ── 2. Short NOT muted at skip hour ────────────────────────────────────────


@pytest.mark.parametrize("hour", [14, 15, 16])
def test_short_not_muted_at_skip_hour(hour):
    """SHORT signal at the same hour fires (filter is long-only)."""
    # Flip rv_pred well below iv_used → strong short edge.
    p = _make_predictor(
        skip_long_hours_utc=(14, 15, 16),
        rv_pred_ln=np.log(0.10),  # 10% vol-ann
        iv_used_pct=40.0,
    )
    bar = _make_bar(f"2026-05-07 {hour:02d}:00")
    sig = p.predict(bar, _make_tsrv(), straddle=_make_straddle())
    assert sig.direction == -1
    assert "SHORT" in sig.reason


# ── 3. Long fires at non-skip hour ─────────────────────────────────────────


@pytest.mark.parametrize("hour", [0, 6, 13, 17, 22])
def test_long_fires_at_non_skip_hour(hour):
    """At hours not in the mute set, a LONG candidate fires normally."""
    p = _make_predictor(skip_long_hours_utc=(14, 15, 16))
    bar = _make_bar(f"2026-05-07 {hour:02d}:00")
    sig = p.predict(bar, _make_tsrv(), straddle=_make_straddle())
    assert sig.direction == 1
    assert "LONG" in sig.reason


# ── 4. Empty skip_hours disables filter ────────────────────────────────────


@pytest.mark.parametrize("hour", [14, 15, 16])
def test_empty_skip_hours_no_mute(hour):
    """Default empty tuple = no mute (preserves non-V2 behaviour)."""
    p = _make_predictor(skip_long_hours_utc=())
    bar = _make_bar(f"2026-05-07 {hour:02d}:00")
    sig = p.predict(bar, _make_tsrv(), straddle=_make_straddle())
    assert sig.direction == 1


# ── 5. Combined with contango filter ───────────────────────────────────────


def test_combined_with_contango_filter():
    """Contango filter mutes first; hour filter doesn't override the
    contango message but the result stays FLAT (idempotent)."""
    p = _make_predictor(
        skip_long_hours_utc=(14, 15, 16),
        skip_long_if_term_slope_gt=0.0,
    )
    # Force contango: atm_2w > atm_1w
    p._svi.atm_vol_at_dte = MagicMock(side_effect=[40.0, 50.0])  # 1w=40, 2w=50
    bar = _make_bar("2026-05-07 14:00")
    sig = p.predict(bar, _make_tsrv(), straddle=_make_straddle())
    assert sig.direction == 0
    # Contango filter fires first; reason mentions it. Hour filter is
    # gated on ``direction == 1`` so doesn't overwrite the message.
    assert "V2 contango filter" in sig.reason
