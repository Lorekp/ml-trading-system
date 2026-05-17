"""Cycle-41 (2026-05-17): pin the dollar-bar buffer <T snapshot.

The cycle-36 parity trace found `n_trades` diverging +12.7 and
`duration_s` ~3 between live and offline, despite both sides using a
$500k dollar-bar threshold.

An empirical check (feed identical Binance aggTrades to the live
`DollarBarAggregator` and the offline `_accumulate_bars` /
`_build_bars_sequential`) proved the two builders are BYTE-IDENTICAL —
2060 bars, n_trades exactly equal. So the aggregator is correct.

Root cause is the same post-T-bar contamination cycle-40 found for
`time_to_funding`: the live `DollarBarBuffer` is fed continuously by
the async Binance poll loop, so at `build_row` time for decision bar
T it already holds dollar bars completed AFTER T. `_h_n_trades` /
`_h_duration_s` read `buf.latest_bar()` (the absolute tail = a post-T
bar); offline's `groupby(floor("15min")).last()` window for the
`[T-15min, T)` feature row excludes `>= T`.

Cycle-41 fix: `build_row` (and `assemble_transvol_input_row`) take ONE
`DollarBarBuffer.snapshot_before(bar.timestamp)` at the top; every
downstream consumer sees the clamped <T view.

Tests:
1. `snapshot_before` keeps only bars strictly before the cutoff.
2. The original buffer is unchanged (snapshot is a copy).
3. With post-T bars present, `_h_n_trades` / `_h_duration_s` via a
   ctx built from the snapshot return the LAST PRE-T bar's value.
4. Source sentinel: build_row + assemble_transvol_input_row snapshot.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from deribit_paper.features.dollar_bar import DollarBar
from deribit_paper.features.dollar_bar_buffer import DollarBarBuffer


def _bar(ts: str, n_trades: int, duration_s: float) -> DollarBar:
    return DollarBar(
        timestamp=pd.Timestamp(ts), open=80_000.0, close=80_010.0,
        high=80_020.0, low=79_990.0, duration_s=duration_s, n_trades=n_trades,
        dollar_volume=500_000.0, buy_volume=260_000.0, sell_volume=240_000.0,
        vov=0.0, imbalance=0.04, log_return=1e-4, rv_bar=1e-7,
    )


def test_snapshot_before_excludes_post_cutoff() -> None:
    """`snapshot_before` keeps only bars strictly before the cutoff."""
    buf = DollarBarBuffer(max_size=100)
    buf.append(_bar("2026-05-14T15:58:30Z", 50, 11.0))
    buf.append(_bar("2026-05-14T15:59:47Z", 64, 13.0))   # last pre-T
    buf.append(_bar("2026-05-14T16:00:11Z", 89, 9.0))    # post-T (poll loop)
    buf.append(_bar("2026-05-14T16:00:25Z", 102, 8.0))   # post-T

    snap = buf.snapshot_before(pd.Timestamp("2026-05-14T16:00:00Z"))
    assert len(snap) == 2
    assert snap.latest_bar().timestamp == pd.Timestamp("2026-05-14T15:59:47Z")
    assert snap.latest_bar().n_trades == 64


def test_snapshot_does_not_mutate_original() -> None:
    """The snapshot is a copy — the original buffer keeps all bars."""
    buf = DollarBarBuffer(max_size=100)
    for i, ts in enumerate(["2026-05-14T15:58:00Z", "2026-05-14T15:59:00Z",
                            "2026-05-14T16:00:30Z"]):
        buf.append(_bar(ts, 40 + i, 10.0))
    _ = buf.snapshot_before(pd.Timestamp("2026-05-14T16:00:00Z"))
    assert len(buf) == 3, "snapshot_before must not mutate the source buffer"


def test_n_trades_handler_reads_last_pre_t_bar() -> None:
    """`_h_n_trades`/`_h_duration_s` over a <T snapshot return the pre-T bar."""
    from types import SimpleNamespace

    from deribit_paper.signal.feature_assembler import (
        AssemblerContext, _h_duration_s, _h_n_trades,
    )

    buf = DollarBarBuffer(max_size=100)
    buf.append(_bar("2026-05-14T15:58:30Z", 50, 11.0))
    buf.append(_bar("2026-05-14T15:59:47Z", 64, 13.0))   # last pre-T
    buf.append(_bar("2026-05-14T16:00:11Z", 89, 9.0))    # post-T
    snap = buf.snapshot_before(pd.Timestamp("2026-05-14T16:00:00Z"))

    ctx = AssemblerContext(
        bar=SimpleNamespace(timestamp=pd.Timestamp("2026-05-14T16:00:00Z"),
                            n_perp_trades=0, perp_trades=[]),  # type: ignore[arg-type]
        bar_tsrv=SimpleNamespace(),                            # type: ignore[arg-type]
        svi_calibrator=None,
        baseline_row=pd.Series(dtype=float),
        dollar_bar_buffer=snap,
    )
    # Must read the 15:59:47 bar (n_trades=64), NOT the post-T 16:00:11 (89).
    assert _h_n_trades(ctx) == 64.0
    assert _h_duration_s(ctx) == 13.0


def test_build_row_takes_snapshot() -> None:
    """Source sentinel: both context builders snapshot the dollar-bar buffer."""
    src = Path("deribit_paper/signal/feature_assembler.py").read_text(encoding="utf-8")
    assert src.count("snapshot_before(bar.timestamp)") >= 2, (
        "build_row AND assemble_transvol_input_row must snapshot <T"
    )
    # The post-T-contaminated raw buffer must not be passed into the
    # feature context anymore.
    assert "dollar_bar_buffer=dollar_bar_buffer_snap" in src
