"""F-7 (parity-r3 cycle 3 — 2026-05-08) regression tests.

``_h_n_trades`` and ``_h_duration_s`` now prefer ``baseline_row``'s
last-sub-bar values over wall-clock 15-min window stats during warmup.

Pre-fix: warmup fallback for ``_h_n_trades`` returned the WALL-CLOCK
sum of perp trades in the 15-min bar; ``_h_duration_s`` synthesised
the mean inter-trade gap. Both diverged from offline's
``series.groupby(window_labels).last()`` which is the LAST sub-bar's
value within the window (a different quantity).

The baseline_row holds the offline last-sub-bar values built from
parquet at warmup, so using it keeps the warmup row monotonic with
offline. Buffer fallback unchanged for tests that explicitly avoid a
baseline.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from deribit_paper.signal.feature_assembler import (
    AssemblerContext,
    _h_duration_s,
    _h_n_trades,
)


def _make_ctx(
    *,
    baseline_row: pd.Series | None = None,
    perp_trades: list | None = None,
    n_perp_trades: int = 0,
) -> AssemblerContext:
    """Build an AssemblerContext with no buffer (warmup) and a chosen
    baseline_row + bar."""
    bar = SimpleNamespace(
        timestamp=pd.Timestamp("2026-05-08 12:00:00", tz="UTC"),
        n_perp_trades=n_perp_trades,
        n_option_trades=0,
        perp_trades=perp_trades or [],
        option_trades=[],
        index_price=60_000.0,
    )
    if baseline_row is None:
        baseline_row = pd.Series(dtype=float)
    return AssemblerContext(
        bar=bar,
        bar_tsrv=None,
        svi_calibrator=None,
        baseline_row=baseline_row,
        dollar_bar_buffer=None,
    )


# ── F-7: _h_n_trades warmup fallback prefers baseline_row ────────────


def test_h_n_trades_warmup_uses_baseline_row_when_available():
    """Buffer empty + baseline_row carries `n_trades` → use baseline."""
    baseline = pd.Series({"n_trades": 42.0, "duration_s": 30.0})
    ctx = _make_ctx(baseline_row=baseline, n_perp_trades=999)
    val = _h_n_trades(ctx)
    assert val == 42.0


def test_h_n_trades_warmup_falls_back_to_wall_clock_when_no_baseline():
    """Buffer empty + baseline_row missing key → legacy wall-clock fallback."""
    ctx = _make_ctx(baseline_row=pd.Series(dtype=float), n_perp_trades=200)
    val = _h_n_trades(ctx)
    assert val == 200.0


def test_h_n_trades_warmup_baseline_nan_falls_back():
    """Baseline value NaN → fall back to wall-clock count."""
    baseline = pd.Series({"n_trades": float("nan"), "duration_s": 30.0})
    ctx = _make_ctx(baseline_row=baseline, n_perp_trades=200)
    val = _h_n_trades(ctx)
    assert val == 200.0


# ── F-7: _h_duration_s warmup fallback prefers baseline_row ──────────


def test_h_duration_s_warmup_uses_baseline_row_when_available():
    """Buffer empty + baseline_row has `duration_s` > 0 → use baseline."""
    baseline = pd.Series({"n_trades": 42.0, "duration_s": 30.0})
    ctx = _make_ctx(baseline_row=baseline, perp_trades=[])
    val = _h_duration_s(ctx)
    assert val == 30.0


def test_h_duration_s_warmup_falls_back_to_inter_trade_gap_no_baseline():
    """No baseline value → synthesise mean inter-trade gap from
    perp_trades (legacy fallback for tests without a baseline row)."""
    trades = [
        SimpleNamespace(timestamp=pd.Timestamp("2026-05-08 12:00:00", tz="UTC")),
        SimpleNamespace(timestamp=pd.Timestamp("2026-05-08 12:00:30", tz="UTC")),
        SimpleNamespace(timestamp=pd.Timestamp("2026-05-08 12:01:00", tz="UTC")),
    ]
    # 60s span / 2 pairs = 30s mean gap.
    ctx = _make_ctx(baseline_row=pd.Series(dtype=float), perp_trades=trades)
    val = _h_duration_s(ctx)
    assert val == pytest.approx(30.0)


def test_h_duration_s_warmup_no_baseline_no_trades_returns_one():
    """Empty buffer + no baseline + no trades → return safe default 1.0."""
    ctx = _make_ctx(baseline_row=pd.Series(dtype=float), perp_trades=[])
    val = _h_duration_s(ctx)
    assert val == 1.0


def test_h_duration_s_warmup_baseline_zero_falls_back():
    """Baseline duration_s == 0 (or NaN) → fall back to inter-trade
    gap. Defensive: a zero baseline would erase forward signal."""
    baseline = pd.Series({"duration_s": 0.0})
    trades = [
        SimpleNamespace(timestamp=pd.Timestamp("2026-05-08 12:00:00", tz="UTC")),
        SimpleNamespace(timestamp=pd.Timestamp("2026-05-08 12:00:45", tz="UTC")),
    ]
    ctx = _make_ctx(baseline_row=baseline, perp_trades=trades)
    val = _h_duration_s(ctx)
    assert val == pytest.approx(45.0)
