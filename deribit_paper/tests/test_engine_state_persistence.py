"""S1+S3 (parity-cycle round 2 — 2026-05-07): engine_state SQLite
persistence covers EquityTracker, edge_ema, and exec_cost_ema across
restarts. Pre-fix every restart re-seeded these fields at constructor
defaults, silently re-baselining the kill-switch daily-DD reference
and rebuilding the rolling EMAs from scratch.

Mirrors H4 (profit_scale_state) pattern.
"""

from __future__ import annotations

import pytest

from deribit_paper.position.equity import EquityTracker
from deribit_paper.position.store import SQLiteStore


# ── EquityTracker state_dict / load_state ────────────────────────────────


def test_equity_tracker_state_dict_roundtrip():
    """state_dict + fresh tracker + load_state = identical observable state."""
    tracker = EquityTracker(initial_equity_usd=100_000.0)
    tracker.on_trade_closed(1500.0)
    tracker.on_trade_closed(-500.0)
    tracker.on_trade_closed(2000.0)
    state = tracker.state_dict()
    assert state["current_equity_usd"] == pytest.approx(103_000.0)
    assert state["peak_equity_usd"] == pytest.approx(103_000.0)
    assert state["realized_pnl_usd"] == pytest.approx(3_000.0)
    assert state["n_trades_closed"] == 3

    fresh = EquityTracker(initial_equity_usd=100_000.0)
    fresh.load_state(state)
    assert fresh.current_equity == tracker.current_equity
    assert fresh.peak_equity == tracker.peak_equity
    assert fresh.realized_pnl == tracker.realized_pnl
    assert fresh.n_trades_closed == tracker.n_trades_closed
    assert fresh.drawdown_pct == pytest.approx(tracker.drawdown_pct)


def test_equity_tracker_load_state_preserves_drawdown():
    """A drawdown-state snapshot survives load_state."""
    tracker = EquityTracker(initial_equity_usd=100_000.0)
    tracker.on_trade_closed(5000.0)   # peak = 105k
    tracker.on_trade_closed(-3000.0)  # current = 102k, dd = -2.857%
    state = tracker.state_dict()

    fresh = EquityTracker(initial_equity_usd=100_000.0)
    fresh.load_state(state)
    assert fresh.peak_equity == pytest.approx(105_000.0)
    assert fresh.current_equity == pytest.approx(102_000.0)
    assert fresh.drawdown_pct == pytest.approx(-2.857142857142857, abs=1e-6)


# ── SQLiteStore engine_state ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_engine_state_save_load_roundtrip(tmp_path):
    db = SQLiteStore(tmp_path / "test.db")
    await db.open()

    state = {
        "current_equity_usd": 105_000.0,
        "peak_equity_usd": 110_000.0,
        "realized_pnl_usd": 5_000.0,
        "n_trades_closed": 42,
        "edge_ema": 35.7,
        "edge_warmup": [10.0, 20.0, 30.0],
        "exec_cost_ema_usd_per_btc": 14.5,
    }
    await db.save_engine_state(state)
    loaded = await db.load_engine_state()

    assert loaded is not None
    assert loaded["current_equity_usd"] == pytest.approx(105_000.0)
    assert loaded["peak_equity_usd"] == pytest.approx(110_000.0)
    assert loaded["realized_pnl_usd"] == pytest.approx(5_000.0)
    assert loaded["n_trades_closed"] == 42
    assert loaded["edge_ema"] == pytest.approx(35.7)
    assert loaded["edge_warmup"] == [10.0, 20.0, 30.0]
    assert loaded["exec_cost_ema_usd_per_btc"] == pytest.approx(14.5)
    assert loaded["last_updated"]  # timestamp string non-empty
    await db.close()


@pytest.mark.asyncio
async def test_engine_state_cold_start_returns_none(tmp_path):
    """Empty DB → load_engine_state returns None (engine keeps defaults)."""
    db = SQLiteStore(tmp_path / "cold.db")
    await db.open()
    loaded = await db.load_engine_state()
    assert loaded is None
    await db.close()


@pytest.mark.asyncio
async def test_engine_state_edge_ema_can_be_none(tmp_path):
    """During the 96-bar warmup, edge_ema is None — verify NULL persists."""
    db = SQLiteStore(tmp_path / "warmup.db")
    await db.open()
    await db.save_engine_state({
        "current_equity_usd": 100_000.0,
        "peak_equity_usd": 100_000.0,
        "realized_pnl_usd": 0.0,
        "n_trades_closed": 0,
        "edge_ema": None,                        # warmup not complete
        "edge_warmup": [10.0, 11.0, 12.0],       # 3 bars seen so far
        "exec_cost_ema_usd_per_btc": 12.0,
    })
    loaded = await db.load_engine_state()
    assert loaded is not None
    assert loaded["edge_ema"] is None
    assert loaded["edge_warmup"] == [10.0, 11.0, 12.0]
    await db.close()


@pytest.mark.asyncio
async def test_engine_state_idempotent_replace(tmp_path):
    """save_engine_state with id=1 overwrites prior row (INSERT OR REPLACE)."""
    db = SQLiteStore(tmp_path / "replace.db")
    await db.open()

    base = {
        "current_equity_usd": 100_000.0,
        "peak_equity_usd": 100_000.0,
        "realized_pnl_usd": 0.0,
        "n_trades_closed": 0,
        "edge_ema": None,
        "edge_warmup": [],
        "exec_cost_ema_usd_per_btc": 12.0,
    }
    await db.save_engine_state(base)
    base["current_equity_usd"] = 105_000.0
    base["n_trades_closed"] = 5
    await db.save_engine_state(base)

    loaded = await db.load_engine_state()
    assert loaded["current_equity_usd"] == pytest.approx(105_000.0)
    assert loaded["n_trades_closed"] == 5
    await db.close()


# ── S5+S8 (parity-cycle round 2 — 2026-05-07): kill_switch consecutive
# errors + svi_recal counters persist across restarts.


@pytest.mark.asyncio
async def test_engine_state_kill_switch_errors_persist(tmp_path):
    """S5: kill_switch_consecutive_errors round-trips via engine_state."""
    db = SQLiteStore(tmp_path / "ks_errors.db")
    await db.open()
    await db.save_engine_state({
        "current_equity_usd": 100_000.0,
        "peak_equity_usd": 100_000.0,
        "realized_pnl_usd": 0.0,
        "n_trades_closed": 0,
        "edge_ema": None,
        "edge_warmup": [],
        "exec_cost_ema_usd_per_btc": 12.0,
        "kill_switch_consecutive_errors": 2,  # 1 error away from tripping at threshold=3
    })
    loaded = await db.load_engine_state()
    assert loaded is not None
    assert loaded["kill_switch_consecutive_errors"] == 2
    await db.close()


@pytest.mark.asyncio
async def test_engine_state_svi_recal_counters_persist(tmp_path):
    """S8: svi_recal_count + svi_recal_failures round-trip via engine_state."""
    db = SQLiteStore(tmp_path / "svi_recal.db")
    await db.open()
    await db.save_engine_state({
        "current_equity_usd": 100_000.0,
        "peak_equity_usd": 100_000.0,
        "realized_pnl_usd": 0.0,
        "n_trades_closed": 0,
        "edge_ema": None,
        "edge_warmup": [],
        "exec_cost_ema_usd_per_btc": 12.0,
        "svi_recal_count": 47,
        "svi_recal_failures": 3,
    })
    loaded = await db.load_engine_state()
    assert loaded is not None
    assert loaded["svi_recal_count"] == 47
    assert loaded["svi_recal_failures"] == 3
    await db.close()


@pytest.mark.asyncio
async def test_engine_state_s5_s8_default_to_zero_when_unset(tmp_path):
    """Older callers that don't pass S5/S8 keys still write — fields default to 0."""
    db = SQLiteStore(tmp_path / "defaults.db")
    await db.open()
    await db.save_engine_state({
        "current_equity_usd": 100_000.0,
        "peak_equity_usd": 100_000.0,
        "realized_pnl_usd": 0.0,
        "n_trades_closed": 0,
        "edge_ema": None,
        "edge_warmup": [],
        "exec_cost_ema_usd_per_btc": 12.0,
        # S5/S8 fields omitted intentionally
    })
    loaded = await db.load_engine_state()
    assert loaded is not None
    assert loaded["kill_switch_consecutive_errors"] == 0
    assert loaded["svi_recal_count"] == 0
    assert loaded["svi_recal_failures"] == 0
    await db.close()


@pytest.mark.asyncio
async def test_engine_state_migration_adds_s5_s8_columns(tmp_path):
    """S5+S8 migration: opening a fresh DB applies ADD COLUMN idempotently.

    Sanity that two opens of the same DB don't raise on duplicate-column.
    """
    db = SQLiteStore(tmp_path / "migrate.db")
    await db.open()
    await db.close()
    db2 = SQLiteStore(tmp_path / "migrate.db")
    await db2.open()  # second open must not raise on the ADD COLUMN
    await db2.close()
