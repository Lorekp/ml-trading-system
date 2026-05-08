"""S-1c (parity-r3 cycle 6 — 2026-05-08) — down_jump_frac alias fix.

Pre-cycle-6 the live ``_LIVE_HANDLERS`` registry mapped only
``down_jump_fraction`` (with the "ion" suffix, an internal name from
the BarTSRV dataclass). The 101-feature TransVol input column is
``down_jump_frac`` (no suffix) — without an alias this column resolved
to baseline-fallback every bar.

This test file locks the alias mapping + the 4 windowed variants.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.tsrv_features import TSRVFeatures
from deribit_paper.features.tsrv_history import TSRVHistoryBuffer
from deribit_paper.features.tsrv_live import BarTSRV
from deribit_paper.signal.feature_assembler import (
    _LIVE_HANDLERS,
    AssemblerContext,
)


def _make_bar_tsrv(*, down_jump: float = 0.04) -> BarTSRV:
    return BarTSRV(
        tsrv=0.05, rv_fast=0.04, rv_slow=0.05,
        bpv=0.045, medrv=0.045, continuous_var=0.04, jump_var=0.005,
        n_ticks=1000, n_sub_bars=10, n_dash=20.0,
        rs_down=0.02, rs_up=0.02,
        jump_fraction=0.05,
        down_jump_fraction=down_jump,
        up_jump_fraction=0.03,
        vol_ann=0.50, vol_ann_continuous=0.49,
    )


def _populate(n: int, *, seed: int = 0) -> TSRVHistoryBuffer:
    rng = np.random.default_rng(seed)
    buf = TSRVHistoryBuffer(max_size=max(n + 50, 200))
    base = pd.Timestamp("2026-04-01", tz="UTC")
    for i in range(n):
        ts = base + pd.Timedelta(minutes=15 * i)
        buf.append_bar(ts, _make_bar_tsrv(
            down_jump=float(0.01 + 0.04 * rng.random()),
        ))
    return buf


# ── Registry coverage ────────────────────────────────────────────


def test_down_jump_frac_alias_in_registry():
    """All 5 down_jump_frac TransVol input columns must have a live handler."""
    needed = [
        "down_jump_frac",
        "down_jump_frac_lag1", "down_jump_frac_lag3",
        "down_jump_frac_ema10", "down_jump_frac_ema50",
    ]
    for name in needed:
        assert name in _LIVE_HANDLERS, f"{name!r} missing from _LIVE_HANDLERS"


def test_down_jump_frac_handler_does_not_collide_with_down_jump_fraction():
    """The alias must coexist with the legacy ``down_jump_fraction``."""
    assert "down_jump_fraction" in _LIVE_HANDLERS
    assert "down_jump_frac" in _LIVE_HANDLERS
    # They are distinct callables.
    assert _LIVE_HANDLERS["down_jump_fraction"] is not _LIVE_HANDLERS["down_jump_frac"]


# ── Handler behaviour ────────────────────────────────────────────


def _ctx(buf: TSRVHistoryBuffer, *, with_tsrv_features: bool = True) -> AssemblerContext:
    base = pd.Timestamp("2026-04-01", tz="UTC")
    bar = type("StubBar", (), {})()
    bar.timestamp = base + pd.Timedelta(minutes=15 * (len(buf) - 1))
    bar.perp_trades = []
    bar.n_perp_trades = 0
    return AssemblerContext(
        bar=bar,
        bar_tsrv=_make_bar_tsrv(down_jump=0.123),
        svi_calibrator=None,
        baseline_row=pd.Series({"down_jump_frac": 0.0}),
        tsrv_history_buffer=buf,
        tsrv_features=TSRVFeatures(buf) if with_tsrv_features else None,
    )


def test_down_jump_frac_raw_reads_from_buffer():
    buf = _populate(50, seed=11)
    ctx = _ctx(buf)
    handler = _LIVE_HANDLERS["down_jump_frac"]
    expected = float(buf.latest().down_jump_fraction)
    assert handler(ctx) == pytest.approx(expected, abs=1e-15)


def test_down_jump_frac_falls_back_to_bar_tsrv_when_no_features():
    """Test/cold-start path: with TSRVFeatures missing the handler should
    use the current bar's BarTSRV.down_jump_fraction value rather than
    silently returning 0."""
    buf = _populate(0, seed=12)  # empty buffer
    ctx = _ctx(buf, with_tsrv_features=False)
    # ctx.bar_tsrv has down_jump_fraction=0.123 from the fixture
    handler = _LIVE_HANDLERS["down_jump_frac"]
    assert handler(ctx) == pytest.approx(0.123, abs=1e-15)


def test_down_jump_frac_lag1_via_handler():
    buf = _populate(100, seed=13)
    ctx = _ctx(buf)
    expected = float(buf.as_series("down_jump_fraction").iloc[-2])
    handler = _LIVE_HANDLERS["down_jump_frac_lag1"]
    assert handler(ctx) == pytest.approx(expected, abs=1e-15)


def test_down_jump_frac_lag3_via_handler():
    buf = _populate(100, seed=14)
    ctx = _ctx(buf)
    expected = float(buf.as_series("down_jump_fraction").iloc[-4])
    handler = _LIVE_HANDLERS["down_jump_frac_lag3"]
    assert handler(ctx) == pytest.approx(expected, abs=1e-15)


def test_down_jump_frac_ema10_via_handler():
    buf = _populate(100, seed=15)
    ctx = _ctx(buf)
    ser = buf.as_series("down_jump_fraction")
    expected = float(
        ser.ewm(span=10, adjust=True, min_periods=1).mean().iloc[-1]
    )
    handler = _LIVE_HANDLERS["down_jump_frac_ema10"]
    np.testing.assert_almost_equal(handler(ctx), expected, decimal=12)


def test_down_jump_frac_ema50_via_handler():
    buf = _populate(150, seed=16)
    ctx = _ctx(buf)
    ser = buf.as_series("down_jump_fraction")
    expected = float(
        ser.ewm(span=50, adjust=True, min_periods=1).mean().iloc[-1]
    )
    handler = _LIVE_HANDLERS["down_jump_frac_ema50"]
    np.testing.assert_almost_equal(handler(ctx), expected, decimal=12)
