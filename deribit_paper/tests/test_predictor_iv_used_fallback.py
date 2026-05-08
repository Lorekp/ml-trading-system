"""S-3 (parity-r3 cycle 2 — 2026-05-08) regression tests.

Live ``Predictor._lookup_iv`` now mirrors the backtester's 2-tier fallback:
  1. SVI evaluation at the ATM strike (primary)
  2. ``bar_tsrv.vol_ann_continuous × 100`` as a DVOL substitute

Pre-fix the predictor returned None when SVI was uncalibrated and the
``predict()`` flow emitted a flat signal — silent emit/skip asymmetry vs
the backtester's ``svi_iv_matched.fillna(iv).fillna(dvol_close)`` chain
in ``vol_arb/backtester.py:905-911``.

Each emit now carries an ``iv_source`` tag so the harness can audit which
tier fired.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import pandas as pd
import pytest

from deribit_paper.signal.predictor import Predictor


def _stub_predictor(*, svi_returns: float | None) -> Predictor:
    """Build a Predictor whose SVI calibrator returns ``svi_returns``."""
    pred = Predictor.__new__(Predictor)
    svi = MagicMock()
    svi.has_params = svi_returns is not None
    if svi_returns is not None:
        svi.evaluate = MagicMock(return_value=pd.Series([float(svi_returns)]))
    else:
        svi.evaluate = MagicMock(return_value=pd.Series(dtype=float))
    pred._svi = svi
    return pred


def _make_bar(*, ts: pd.Timestamp, index: float = 60_000.0):
    return SimpleNamespace(timestamp=ts, index_price=index)


def _make_straddle(*, strike: float = 60_000.0):
    return SimpleNamespace(
        strike=strike,
        expiry_dt=pd.Timestamp("2026-04-22 08:00:00", tz="UTC"),
        index_price=60_000.0,
    )


def _make_bar_tsrv(*, vol_ann_continuous: float = 0.65):
    return SimpleNamespace(vol_ann_continuous=vol_ann_continuous)


def _assert_lookup(out: tuple[float, str] | None, *, iv: float, source: str) -> None:
    """Helper: tolerant float comparison, exact source string."""
    assert out is not None, "expected (iv, source) tuple, got None"
    iv_actual, source_actual = out
    assert source_actual == source
    assert iv_actual == pytest.approx(iv, rel=1e-9, abs=1e-9)


def test_lookup_iv_uses_svi_when_available():
    pred = _stub_predictor(svi_returns=72.5)
    out = pred._lookup_iv(
        bar=_make_bar(ts=pd.Timestamp("2026-04-15 12:00", tz="UTC")),
        straddle=_make_straddle(),
        bar_tsrv=_make_bar_tsrv(),
    )
    _assert_lookup(out, iv=72.5, source="svi")


def test_lookup_iv_falls_back_to_tsrv_proxy_when_svi_uncalibrated():
    pred = _stub_predictor(svi_returns=None)
    pred._svi.has_params = False
    out = pred._lookup_iv(
        bar=_make_bar(ts=pd.Timestamp("2026-04-15 12:00", tz="UTC")),
        straddle=_make_straddle(),
        bar_tsrv=_make_bar_tsrv(vol_ann_continuous=0.65),
    )
    # vol_ann_continuous (0.65) × 100 = 65%
    _assert_lookup(out, iv=65.0, source="tsrv_proxy")


def test_lookup_iv_falls_back_when_svi_returns_empty():
    """SVI calibrated but no slice for this expiry → empty result → fall through."""
    pred = _stub_predictor(svi_returns=None)
    pred._svi.has_params = True
    pred._svi.evaluate = MagicMock(return_value=pd.Series(dtype=float))
    out = pred._lookup_iv(
        bar=_make_bar(ts=pd.Timestamp("2026-04-15 12:00", tz="UTC")),
        straddle=_make_straddle(),
        bar_tsrv=_make_bar_tsrv(vol_ann_continuous=0.55),
    )
    _assert_lookup(out, iv=55.0, source="tsrv_proxy")


def test_lookup_iv_falls_back_when_svi_returns_non_finite():
    """SVI returns NaN / 0 / inf → fall through to TSRV proxy."""
    pred = _stub_predictor(svi_returns=None)
    pred._svi.has_params = True
    pred._svi.evaluate = MagicMock(return_value=pd.Series([float("nan")]))
    out = pred._lookup_iv(
        bar=_make_bar(ts=pd.Timestamp("2026-04-15 12:00", tz="UTC")),
        straddle=_make_straddle(),
        bar_tsrv=_make_bar_tsrv(vol_ann_continuous=0.70),
    )
    _assert_lookup(out, iv=70.0, source="tsrv_proxy")


def test_lookup_iv_returns_none_when_no_straddle():
    pred = _stub_predictor(svi_returns=72.5)
    out = pred._lookup_iv(
        bar=_make_bar(ts=pd.Timestamp("2026-04-15 12:00", tz="UTC")),
        straddle=None,
        bar_tsrv=_make_bar_tsrv(),
    )
    assert out is None


def test_lookup_iv_returns_none_when_svi_uncalibrated_and_no_bar_tsrv():
    """No SVI AND no bar_tsrv → both tiers exhausted → None."""
    pred = _stub_predictor(svi_returns=None)
    pred._svi.has_params = False
    out = pred._lookup_iv(
        bar=_make_bar(ts=pd.Timestamp("2026-04-15 12:00", tz="UTC")),
        straddle=_make_straddle(),
        bar_tsrv=None,
    )
    assert out is None


def test_lookup_iv_returns_none_when_tsrv_vol_ann_is_zero():
    """Defensive: bar_tsrv with vol_ann_continuous=0 (degenerate) →
    proxy is 0 → invalid → return None."""
    pred = _stub_predictor(svi_returns=None)
    pred._svi.has_params = False
    out = pred._lookup_iv(
        bar=_make_bar(ts=pd.Timestamp("2026-04-15 12:00", tz="UTC")),
        straddle=_make_straddle(),
        bar_tsrv=_make_bar_tsrv(vol_ann_continuous=0.0),
    )
    assert out is None
