"""Cycle-40 (2026-05-17): pin the time_to_funding bar-close clamp.

The cycle-36 parity trace found `time_to_funding_*` features diverge
massively (ema50 abs.mean 17.35, max 3226) between live and offline.

Root cause: the live `DollarBarBuffer` is fed continuously by the
Binance poll loop, so at `process_bar` time for bar T the buffer
already holds dollar bars that completed a few seconds AFTER T. The
cycle-6 funding handlers read `buf.latest_bar()` / `timestamp_at(-n)`
/ `as_index()` — all anchored on the ABSOLUTE buffer tail. Offline
computes per-dollar-bar then `groupby(index.floor("15min")).last()`,
whose window for the row covering [T-15min, T) strictly excludes
anything >= T. `time_to_funding` is a sawtooth resetting at the 8h
funding boundaries (08/16/24 UTC); reading a post-T bar when T is
near a boundary lands on the WRONG side of the reset — a ~28800s
error.

Cycle-40 clamps every funding handler to dollar bars STRICTLY BEFORE
`ctx.bar.timestamp` via `_dollar_bar_timestamps_before`.

Tests:
1. `_dollar_bar_timestamps_before` excludes bars at/after the bar close.
2. `_effective_timestamp_for_funding` returns the last PRE-T bar even
   when post-T bars are present in the buffer.
3. The boundary-straddle bug: with T at 16:00:00 and buffer bars on
   both sides, time_to_funding reads the pre-boundary (~small) value,
   not the post-boundary (~28800s) value.
4. Source sentinel: the funding handlers route through the clamp.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from deribit_paper.features.dollar_bar import DollarBar
from deribit_paper.features.dollar_bar_buffer import DollarBarBuffer


def _bar(ts: str) -> DollarBar:
    return DollarBar(
        timestamp=pd.Timestamp(ts), open=80_000.0, close=80_000.0,
        high=80_010.0, low=79_990.0, duration_s=12.0, n_trades=40,
        dollar_volume=500_000.0, buy_volume=260_000.0, sell_volume=240_000.0,
        vov=0.0, imbalance=0.04, log_return=0.0, rv_bar=1e-7,
    )


def _ctx(bar_close: str, dollar_bar_ts: list[str]):
    """Minimal AssemblerContext with a populated DollarBarBuffer."""
    from types import SimpleNamespace

    from deribit_paper.signal.feature_assembler import AssemblerContext

    buf = DollarBarBuffer(max_size=500)
    for t in dollar_bar_ts:
        buf.append(_bar(t))
    # AssemblerContext needs `bar` (with .timestamp) + `dollar_bar_buffer`.
    bar = SimpleNamespace(timestamp=pd.Timestamp(bar_close))
    return AssemblerContext(
        bar=bar,                       # type: ignore[arg-type]
        bar_tsrv=SimpleNamespace(),     # type: ignore[arg-type]
        svi_calibrator=None,
        baseline_row=pd.Series(dtype=float),
        dollar_bar_buffer=buf,
    )


def test_timestamps_before_excludes_post_close_bars() -> None:
    """`_dollar_bar_timestamps_before` drops bars at/after the bar close."""
    from deribit_paper.signal.feature_assembler import _dollar_bar_timestamps_before

    ctx = _ctx(
        "2026-05-14T16:00:00Z",
        ["2026-05-14T15:58:30Z", "2026-05-14T15:59:47Z",  # before
         "2026-05-14T16:00:11Z", "2026-05-14T16:00:25Z"],  # after (poll-loop)
    )
    before = _dollar_bar_timestamps_before(ctx)
    assert len(before) == 2
    assert before.max() < pd.Timestamp("2026-05-14T16:00:00Z")


def test_effective_timestamp_is_last_pre_close_bar() -> None:
    """`_effective_timestamp_for_funding` ignores post-T buffer bars."""
    from deribit_paper.signal.feature_assembler import (
        _effective_timestamp_for_funding,
    )

    ctx = _ctx(
        "2026-05-14T16:00:00Z",
        ["2026-05-14T15:58:30Z", "2026-05-14T15:59:47Z",
         "2026-05-14T16:00:11Z"],
    )
    eff = _effective_timestamp_for_funding(ctx)
    assert eff == pd.Timestamp("2026-05-14T15:59:47Z")


def test_funding_boundary_straddle_reads_pre_boundary_value() -> None:
    """The bug: T at a funding boundary + post-T bars → must NOT jump to 28800."""
    from deribit_paper.signal.feature_assembler import _h_time_to_funding

    # Bar close exactly at the 16:00 UTC funding boundary. Buffer has a
    # pre-boundary bar (15:59:47 → ~13s to funding) and post-boundary
    # bars (16:00:1x → ~28790s to the NEXT boundary).
    ctx = _ctx(
        "2026-05-14T16:00:00Z",
        ["2026-05-14T15:59:20Z", "2026-05-14T15:59:47Z",
         "2026-05-14T16:00:11Z", "2026-05-14T16:00:25Z"],
    )
    ttf = _h_time_to_funding(ctx)
    # 15:59:47 → h=15.9964 → (16-h)*3600 ≈ 13s. Must be small, NOT ~28800.
    assert ttf < 60.0, f"expected pre-boundary ~13s, got {ttf} (post-T leak)"

    # Sanity: if the buffer had ONLY the post-T bars, the old behaviour
    # would have returned the ~28800 value — prove the clamp matters.
    ctx_post = _ctx(
        "2026-05-14T16:00:00Z",
        ["2026-05-14T16:00:11Z", "2026-05-14T16:00:25Z"],
    )
    # No pre-T bar → falls back to ctx.bar.timestamp (16:00:00 exactly →
    # h=16.0 → not <16 → (24-16)*3600 = 28800).
    assert _h_time_to_funding(ctx_post) == pytest.approx(28800.0)


def test_lag_handlers_walk_pre_close_bars() -> None:
    """lag1/lag3 index back through PRE-T bars only."""
    from deribit_paper.signal.feature_assembler import (
        _h_time_to_funding_lag1,
        _time_to_funding_seconds,
    )

    ctx = _ctx(
        "2026-05-14T12:00:00Z",
        ["2026-05-14T11:58:00Z", "2026-05-14T11:59:00Z",
         "2026-05-14T11:59:50Z", "2026-05-14T12:00:30Z"],
    )
    # before = [11:58, 11:59, 11:59:50]; lag1 = before[-2] = 11:59:00.
    assert _h_time_to_funding_lag1(ctx) == _time_to_funding_seconds(
        pd.Timestamp("2026-05-14T11:59:00Z"),
    )


def test_source_routes_through_clamp() -> None:
    """Source sentinel: funding handlers use `_dollar_bar_timestamps_before`."""
    src = Path("deribit_paper/signal/feature_assembler.py").read_text(encoding="utf-8")
    assert "_dollar_bar_timestamps_before" in src
    # The pre-cycle-40 absolute-tail accessors must no longer drive the
    # funding handlers (latest_bar / timestamp_at / as_index in that
    # region). Cheap proxy: the helper is defined AND referenced ≥4×
    # (definition + effective_ts + lag1 + lag3 + ema).
    assert src.count("_dollar_bar_timestamps_before") >= 5
