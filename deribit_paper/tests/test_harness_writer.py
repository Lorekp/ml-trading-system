"""Tests for ``deribit_paper/monitoring/harness_writer.py``.

Three contracts under test:

1. **Schema fidelity**: a fully-populated TradeOutcome → row matching the
   V2 backtest column order.
2. **Append**: writing two rows in succession produces a 2-row parquet
   with the schema preserved (no column drop / reorder on round-trip).
3. **Defensive paths**: outcome=None and outcome.signal=None silently
   skip without raising; bad parquet path logs an error and returns.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from deribit_paper.monitoring.harness_writer import (
    _HARNESS_COLUMNS,
    HarnessWriter,
    _row_from_outcome,
)


def _make_outcome(*, with_exit: bool = True) -> SimpleNamespace:
    """Duck-typed TradeOutcome stub."""
    started = pd.Timestamp("2026-04-15 12:00:00", tz="UTC")
    ended = pd.Timestamp("2026-04-15 12:15:00", tz="UTC")

    straddle = SimpleNamespace(
        strike=60_000.0,
        expiry_dt=pd.Timestamp("2026-04-17 08:00:00", tz="UTC"),
        index_price=60_500.0,
    )
    signal = SimpleNamespace(
        direction=1,
        edge_vol_pts=15.5,
        rv_pred_pct=72.0,
        iv_used_pct=58.0,
        straddle=straddle,
    )
    entry_call = SimpleNamespace(average_price=0.05)
    entry_put = SimpleNamespace(average_price=0.05)
    exit_call = SimpleNamespace(average_price=0.06) if with_exit else None
    exit_put = SimpleNamespace(average_price=0.04) if with_exit else None
    hedge_summary = SimpleNamespace(n_ticks_executed=10)

    return SimpleNamespace(
        trade_id="t-001",
        signal=signal,
        entry_call=entry_call,
        entry_put=entry_put,
        exit_call=exit_call,
        exit_put=exit_put,
        n_contracts_btc=0.3,
        status="completed",
        hedge_summary=hedge_summary,
        error=None,
        started_at=started,
        ended_at=ended,
    )


# ── Schema fidelity ────────────────────────────────────────────────────────

def test_row_columns_match_contract():
    out = _make_outcome()
    row = _row_from_outcome(
        out,
        realized_pnl_usd=120.5,
        kelly_mult=1.4,
        taker_slippage_pct=0.05,
        term_slope_1w_2w=2.3,
        config_preset="v2",
    )
    assert set(row.keys()) == set(_HARNESS_COLUMNS)


def test_row_values_populated():
    out = _make_outcome()
    row = _row_from_outcome(
        out,
        realized_pnl_usd=120.5,
        kelly_mult=1.4,
        taker_slippage_pct=0.05,
        term_slope_1w_2w=2.3,
        config_preset="v2",
    )
    assert row["signal"] == 1
    assert row["n_contracts_btc"] == 0.3
    # E.3.6: holding stored as int-valued float (V2 contract).
    assert row["holding_minutes"] == 15.0
    assert isinstance(row["holding_minutes"], float)
    assert row["kelly_mult"] == 1.4
    assert row["taker_slippage_pct"] == 0.05
    assert row["term_slope_1w_2w"] == 2.3
    assert row["config_preset"] == "v2"
    assert row["entry_call_price"] == 0.05
    assert row["exit_put_price"] == 0.04
    assert row["realized_pnl_usd"] == 120.5
    assert row["index_price_at_entry"] == 60_500.0
    assert row["hedge_ticks_executed"] == 10


def test_holding_minutes_floored_to_int():
    """E.3.6: holding_minutes stored as int-valued float so the V2
    harness diff doesn't show spurious ~0.03 deltas from sub-minute
    drift in the live ended_at vs the V2 backtester's bar-aligned hold."""
    out = _make_outcome()
    # A trade that took 14:58 (just under 15 minutes).
    out.ended_at = out.started_at + pd.Timedelta(seconds=14 * 60 + 58)
    row = _row_from_outcome(
        out,
        realized_pnl_usd=0.0,
    )
    assert row["holding_minutes"] == 14.0  # floored to int, not 14.967


def test_row_handles_missing_exit_legs():
    out = _make_outcome(with_exit=False)
    row = _row_from_outcome(
        out, realized_pnl_usd=None,
    )
    assert np.isnan(row["exit_call_price"])
    assert np.isnan(row["exit_put_price"])
    assert np.isnan(row["realized_pnl_usd"])


# ── Append + round-trip ───────────────────────────────────────────────────

import asyncio  # noqa: E402


def test_append_creates_parquet(tmp_path: Path):
    out = _make_outcome()
    writer = HarnessWriter(tmp_path / "harness.parquet")
    asyncio.run(writer.consolidate_and_write(
        out,
        realized_pnl_usd=100.0,
        kelly_mult=1.0,
        taker_slippage_pct=0.5,
        config_preset="default",
    ))
    df = pd.read_parquet(tmp_path / "harness.parquet")
    assert list(df.columns) == _HARNESS_COLUMNS
    assert len(df) == 1
    assert df.iloc[0]["trade_id"] == "t-001"


def test_append_preserves_existing_rows(tmp_path: Path):
    """Two appends → a 2-row parquet."""
    out1 = _make_outcome()
    out2 = _make_outcome()
    out2.trade_id = "t-002"
    writer = HarnessWriter(tmp_path / "harness.parquet")

    async def _go() -> None:
        await writer.consolidate_and_write(out1, realized_pnl_usd=100.0)
        await writer.consolidate_and_write(out2, realized_pnl_usd=-50.0)

    asyncio.run(_go())

    df = pd.read_parquet(tmp_path / "harness.parquet")
    assert len(df) == 2
    assert df["trade_id"].tolist() == ["t-001", "t-002"]
    assert df["realized_pnl_usd"].tolist() == [100.0, -50.0]


# ── Defensive paths ──────────────────────────────────────────────────────

def test_consolidate_with_none_outcome_silent(tmp_path: Path):
    writer = HarnessWriter(tmp_path / "harness.parquet")
    # Should not raise.
    asyncio.run(writer.consolidate_and_write(None, realized_pnl_usd=None))
    assert not (tmp_path / "harness.parquet").exists()


def test_consolidate_with_none_signal_silent(tmp_path: Path):
    out = SimpleNamespace(signal=None)
    writer = HarnessWriter(tmp_path / "harness.parquet")
    asyncio.run(writer.consolidate_and_write(out, realized_pnl_usd=None))
    assert not (tmp_path / "harness.parquet").exists()


# ── E.1.3: concurrency + atomic write ────────────────────────────────────


def test_concurrent_appends_no_row_loss(tmp_path: Path):
    """Two simultaneous tasks → both rows present in the final parquet.

    Pre-E.1.3, the read-modify-write pattern raced on this exact case:
    both tasks would read the empty parquet, build a 1-row frame, and
    both write — keeping only the second writer's row. The lock + atomic
    rename keeps both rows.
    """
    out1 = _make_outcome()
    out2 = _make_outcome()
    out2.trade_id = "t-002"
    out2.started_at = pd.Timestamp("2026-04-15 12:01:00", tz="UTC")
    out2.ended_at = pd.Timestamp("2026-04-15 12:16:00", tz="UTC")
    writer = HarnessWriter(tmp_path / "harness.parquet")

    async def _go() -> None:
        await asyncio.gather(
            writer.consolidate_and_write(out1, realized_pnl_usd=100.0),
            writer.consolidate_and_write(out2, realized_pnl_usd=200.0),
        )

    asyncio.run(_go())

    df = pd.read_parquet(tmp_path / "harness.parquet")
    assert len(df) == 2
    assert set(df["trade_id"]) == {"t-001", "t-002"}
    assert set(df["realized_pnl_usd"]) == {100.0, 200.0}


def test_atomic_write_leaves_no_tmp_on_success(tmp_path: Path):
    """After a successful write, no ``.tmp.*`` files remain."""
    out = _make_outcome()
    writer = HarnessWriter(tmp_path / "harness.parquet")
    asyncio.run(writer.consolidate_and_write(out, realized_pnl_usd=10.0))
    leftovers = list(tmp_path.glob("*.tmp*"))
    assert leftovers == []
