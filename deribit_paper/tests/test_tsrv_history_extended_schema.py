"""S-1c (parity-r3 cycle 6 — 2026-05-08) — extended TSRV history schema.

Locks the cycle-6 extension that persists ``rv_fast``, ``n_ticks``, and
``down_jump_fraction`` on every ``TSRVHistoryRow``. These are needed to
compute live ``noise_level`` (= ``rv_fast / (2 × max(n_ticks, 1))``),
``down_jump_frac``, and (via existing ``bpv`` + ``rv_slow``)
``bpv_tsrv_ratio`` on the bar grid.

BarTSRV already carried these three values; pre-cycle-6 the buffer
silently dropped them, forcing the live engine onto baseline-fallback
for the corresponding 15 TransVol input columns.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.tsrv_history import (
    TSRVHistoryBuffer,
    TSRVHistoryRow,
)
from deribit_paper.features.tsrv_live import BarTSRV


def _bar_tsrv(
    *,
    rv_fast: float = 0.04,
    n_ticks: int = 1000,
    down_jump_fraction: float = 0.02,
) -> BarTSRV:
    """BarTSRV stub with the three new fields free to override."""
    return BarTSRV(
        tsrv=0.05, rv_fast=rv_fast, rv_slow=0.05,
        bpv=0.045, medrv=0.045, continuous_var=0.04, jump_var=0.005,
        jump_fraction=0.05, rs_down=0.02, rs_up=0.02,
        down_jump_fraction=down_jump_fraction,
        up_jump_fraction=0.03,
        n_ticks=n_ticks, n_sub_bars=10, n_dash=20.0,
        vol_ann=0.50, vol_ann_continuous=0.49,
    )


# ── Persistence on append_bar ──────────────────────────────────────


def test_append_bar_persists_rv_fast():
    buf = TSRVHistoryBuffer(max_size=5)
    ts = pd.Timestamp("2026-05-07 12:00", tz="UTC")
    buf.append_bar(ts, _bar_tsrv(rv_fast=0.0123))
    assert buf.latest().rv_fast == pytest.approx(0.0123)


def test_append_bar_persists_n_ticks_as_int():
    buf = TSRVHistoryBuffer(max_size=5)
    ts = pd.Timestamp("2026-05-07 12:00", tz="UTC")
    buf.append_bar(ts, _bar_tsrv(n_ticks=4567))
    last = buf.latest()
    assert last.n_ticks == 4567
    assert isinstance(last.n_ticks, int)


def test_append_bar_persists_down_jump_fraction():
    buf = TSRVHistoryBuffer(max_size=5)
    ts = pd.Timestamp("2026-05-07 12:00", tz="UTC")
    buf.append_bar(ts, _bar_tsrv(down_jump_fraction=0.0789))
    assert buf.latest().down_jump_fraction == pytest.approx(0.0789)


def test_append_bar_preserves_existing_fields():
    """Schema extension must not regress vol_ann/up_jump_fraction/bpv/rv_slow."""
    buf = TSRVHistoryBuffer(max_size=5)
    ts = pd.Timestamp("2026-05-07 12:00", tz="UTC")
    buf.append_bar(ts, _bar_tsrv())
    last = buf.latest()
    assert last.vol_ann == pytest.approx(0.50)
    assert last.up_jump_fraction == pytest.approx(0.03)
    assert last.bpv == pytest.approx(0.045)
    assert last.rv_slow == pytest.approx(0.05)


# ── as_series exposes new fields ───────────────────────────────────


def test_as_series_for_rv_fast():
    buf = TSRVHistoryBuffer(max_size=5)
    base = pd.Timestamp("2026-05-07 00:00", tz="UTC")
    for i in range(3):
        buf.append_bar(base + pd.Timedelta(minutes=15 * i),
                       _bar_tsrv(rv_fast=0.01 * (i + 1)))
    ser = buf.as_series("rv_fast")
    assert list(ser.values) == pytest.approx([0.01, 0.02, 0.03])


def test_as_series_for_n_ticks():
    buf = TSRVHistoryBuffer(max_size=5)
    base = pd.Timestamp("2026-05-07 00:00", tz="UTC")
    for i in range(3):
        buf.append_bar(base + pd.Timedelta(minutes=15 * i),
                       _bar_tsrv(n_ticks=100 + i))
    ser = buf.as_series("n_ticks")
    assert list(ser.values) == [100, 101, 102]


# ── prewarm_from_parquet round-trip ────────────────────────────────


def test_prewarm_loads_extended_schema(tmp_path):
    idx = pd.DatetimeIndex(
        pd.date_range("2026-05-07", periods=4, freq="15min", tz="UTC"),
        name="timestamp",
    )
    df = pd.DataFrame({
        "vol_ann":            [0.5, 0.55, 0.6, 0.65],
        "up_jump_fraction":   [0.01, 0.02, 0.03, 0.04],
        "down_jump_fraction": [0.005, 0.01, 0.015, 0.02],
        "bpv":                [0.04, 0.045, 0.05, 0.055],
        "rv_slow":            [0.05, 0.06, 0.07, 0.08],
        "rv_fast":            [0.03, 0.04, 0.05, 0.06],
        "n_ticks":            [100, 200, 300, 400],
    }, index=idx)
    p = tmp_path / "tsrv.parquet"
    df.to_parquet(p)
    buf = TSRVHistoryBuffer(max_size=10)
    n = buf.prewarm_from_parquet(p)
    assert n == 4
    rows = list(buf)
    assert rows[0].rv_fast == pytest.approx(0.03)
    assert rows[-1].rv_fast == pytest.approx(0.06)
    assert rows[-1].n_ticks == 400
    assert rows[-1].down_jump_fraction == pytest.approx(0.02)


def test_prewarm_rejects_missing_new_columns(tmp_path):
    """Parquet missing any of the 3 new columns → 0 rows + WARN log,
    not silent partial load."""
    idx = pd.DatetimeIndex(
        pd.date_range("2026-05-07", periods=2, freq="15min", tz="UTC"),
        name="timestamp",
    )
    # Old-schema parquet (cycle-5 era) that lacks the new columns.
    df = pd.DataFrame({
        "vol_ann":          [0.5, 0.55],
        "up_jump_fraction": [0.01, 0.02],
        "bpv":              [0.04, 0.045],
        "rv_slow":          [0.05, 0.06],
    }, index=idx)
    p = tmp_path / "old_schema.parquet"
    df.to_parquet(p)
    buf = TSRVHistoryBuffer(max_size=10)
    n = buf.prewarm_from_parquet(p)
    assert n == 0
    assert len(buf) == 0
