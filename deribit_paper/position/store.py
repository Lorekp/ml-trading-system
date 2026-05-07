"""SQLite persistence for trades, equity snapshots, kill-switch state.

Minimal Phase 6 implementation: records trades + equity + a single
kill-switch-state row so a restart can reconcile with the exchange
and decide whether to resume.  Full schema (orders, fills,
reconciliation log) is a Phase 8 hardening.

Tables:

- ``trades`` — one row per completed :class:`TradeOutcome` (entry
  fills, exit fills, hedge summary, realized PnL)
- ``equity_snapshots`` — one row per :class:`EquitySnapshot`
- ``kill_switch_state`` — single-row table holding current halt flag,
  reason, timestamp; survives process restart

All operations are via :mod:`aiosqlite` async API.  The store can be
shared across tasks since asyncio is single-threaded.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any

import aiosqlite
import pandas as pd

from deribit_paper.execution.straddle_flow import TradeOutcome
from deribit_paper.position.equity import EquitySnapshot

logger = logging.getLogger(__name__)


_CREATE_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    started_at TEXT NOT NULL,
    ended_at TEXT,
    signal_direction INTEGER NOT NULL,
    n_contracts_btc REAL NOT NULL,
    strike REAL NOT NULL,
    expiry_dt TEXT NOT NULL,
    status TEXT NOT NULL,
    realized_pnl_usd REAL,
    hedge_ticks_executed INTEGER,
    error TEXT,
    payload_json TEXT
);

CREATE INDEX IF NOT EXISTS idx_trades_started_at ON trades(started_at);
CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);

CREATE TABLE IF NOT EXISTS equity_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    equity_usd REAL NOT NULL,
    realized_pnl_usd REAL NOT NULL,
    peak_equity_usd REAL NOT NULL,
    drawdown_pct REAL NOT NULL,
    n_trades_closed INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_equity_timestamp ON equity_snapshots(timestamp);

CREATE TABLE IF NOT EXISTS kill_switch_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    halted INTEGER NOT NULL,
    reason TEXT,
    halted_at TEXT
);

CREATE TABLE IF NOT EXISTS open_positions (
    trade_id TEXT PRIMARY KEY,
    call_instrument TEXT NOT NULL,
    put_instrument TEXT NOT NULL,
    signal_direction INTEGER NOT NULL,
    n_contracts_btc REAL NOT NULL,
    strike REAL NOT NULL,
    expiry_dt TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    entry_call_price REAL NOT NULL DEFAULT 0.0,
    entry_put_price REAL NOT NULL DEFAULT 0.0
);

-- H8 (2026-05-06): bar_cursor records every bar timestamp that has
-- finished _on_bar_close. On restart, the engine skips bars whose
-- timestamps are already in this table, providing defence-in-depth
-- against re-processing a bar (e.g., a buggy aggregator re-firing a
-- past bar, or a manual replay accidentally hitting a live engine).
-- Bar timestamp is canonical exchange UTC (set by BarClock); same
-- timestamp from two restarts is byte-identical.
CREATE TABLE IF NOT EXISTS bar_cursor (
    bar_timestamp TEXT PRIMARY KEY,
    signal_direction INTEGER NOT NULL,
    signal_reason TEXT,
    fired_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_bar_cursor_fired_at ON bar_cursor(fired_at);

-- H4 (2026-05-06): profit_scale_state persists the
-- ProfitScaleEstimator EMAs across restarts. Without this row, the
-- engine restarted with gross_count=0, λ=0, est_gross collapsing to
-- est_theo for the first ~200 trades, drifting away from V2-parity.
-- Single-row table indexed at id=1 (mirrors the kill_switch_state
-- pattern). Updated after every closed trade.
CREATE TABLE IF NOT EXISTS profit_scale_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    gross_per_btc_ema_usd REAL NOT NULL,
    gross_count INTEGER NOT NULL,
    hedge_per_btc_ema_usd REAL NOT NULL,
    last_updated TEXT NOT NULL
);

-- S1+S3 (parity-cycle round 2 — 2026-05-07): engine_state persists
-- per-process state that pre-fix was lost on every restart:
--
--   * EquityTracker (S1): current_equity, peak_equity, realized_pnl,
--     n_trades_closed. Pre-fix every restart re-based at $100K,
--     silently re-baselining the kill-switch daily-DD reference
--     and producing different sizing decisions vs a continuous run.
--
--   * Edge-EMA (S3, D4): _edge_ema scalar + _edge_warmup buffer.
--     Pre-fix the EMA cold-started, taking ~500 trades to reconverge
--     to the V2 reference; live engine that restarts daily NEVER
--     converged.
--
--   * Exec-cost-EMA (S3): _exec_cost_ema scalar. Pre-fix re-seeded
--     at hardcoded 12.0 USD/BTC every restart.
--
-- Single-row table (id=1). Updated after every closed trade.
CREATE TABLE IF NOT EXISTS engine_state (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    -- EquityTracker
    current_equity_usd REAL NOT NULL,
    peak_equity_usd REAL NOT NULL,
    realized_pnl_usd REAL NOT NULL,
    n_trades_closed INTEGER NOT NULL,
    -- Edge EMA (D4 — composite_v2 sizing)
    edge_ema REAL,                          -- nullable: None during 96-bar warmup
    edge_warmup_json TEXT NOT NULL,          -- list of |edge| during warmup
    -- Execution-cost EMA
    exec_cost_ema_usd_per_btc REAL NOT NULL,
    -- S5 (parity-cycle round 2 — 2026-05-07): KillSwitch consecutive-error
    -- counter. Pre-S5 reset to 0 on every restart, so an engine that was
    -- 1 error from tripping got 2 free retries on restart.
    kill_switch_consecutive_errors INTEGER NOT NULL DEFAULT 0,
    -- S8 (parity-cycle round 2 — 2026-05-07): SVI recalibration counters
    -- so the heartbeat doesn't show "0 recals" right after restart.
    svi_recal_count INTEGER NOT NULL DEFAULT 0,
    svi_recal_failures INTEGER NOT NULL DEFAULT 0,
    last_updated TEXT NOT NULL
);
"""

# S5+S8: ALTER TABLE migrations for engine_state when an older row exists
# without these columns. Idempotent: ADD COLUMN raises ``duplicate column``
# on already-migrated DBs, which we swallow.
_ENGINE_STATE_S5_S8_MIGRATIONS: tuple[str, ...] = (
    "ALTER TABLE engine_state ADD COLUMN "
    "kill_switch_consecutive_errors INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE engine_state ADD COLUMN "
    "svi_recal_count INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE engine_state ADD COLUMN "
    "svi_recal_failures INTEGER NOT NULL DEFAULT 0",
)


class SQLiteStore:
    """Async SQLite-backed persistence for the paper-trading engine."""

    def __init__(self, db_path: str | Path):
        self.db_path = Path(db_path)
        self._conn: aiosqlite.Connection | None = None

    async def open(self) -> None:
        """Open the connection and ensure schema is in place."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = await aiosqlite.connect(str(self.db_path))
        # H14 (2026-05-06): WAL + busy_timeout + NORMAL synchronous.
        # Default DELETE journal mode raises ``database is locked`` on
        # any concurrent write contention; the live engine has multiple
        # asyncio tasks (trade-close persistence, kill-switch persist,
        # equity snapshots, profit-scale persist) hitting the DB
        # concurrently. WAL allows readers to coexist with writers and
        # busy_timeout=5000 makes contention block briefly instead of
        # raising. synchronous=NORMAL is safe for the engine's restart
        # idempotency model (bar_cursor + reconciliation) and ~3x
        # faster on every commit.
        for pragma in (
            "PRAGMA journal_mode=WAL",
            "PRAGMA synchronous=NORMAL",
            "PRAGMA busy_timeout=5000",
        ):
            try:
                async with self._conn.execute(pragma) as cur:
                    await cur.fetchone()  # journal_mode returns the new mode
            except Exception:  # noqa: BLE001
                logger.exception("SQLite pragma failed: %s", pragma)
        await self._conn.executescript(_CREATE_TABLES_SQL)
        # S5+S8 migrations on engine_state. Each ADD COLUMN is idempotent
        # via the duplicate-column check; SQLite raises OperationalError
        # for an already-existing column which we swallow.
        for _alter in _ENGINE_STATE_S5_S8_MIGRATIONS:
            try:
                await self._conn.execute(_alter)
            except aiosqlite.OperationalError as exc:
                if "duplicate column name" not in str(exc):
                    raise
        # Seed kill switch state if empty
        async with self._conn.execute(
            "SELECT COUNT(*) FROM kill_switch_state",
        ) as cursor:
            row = await cursor.fetchone()
            if row and row[0] == 0:
                await self._conn.execute(
                    "INSERT INTO kill_switch_state (id, halted, reason, halted_at) "
                    "VALUES (1, 0, NULL, NULL)",
                )
        await self._conn.commit()
        logger.info(
            "SQLiteStore opened at %s (WAL mode, busy_timeout=5s)",
            self.db_path,
        )

    async def close(self) -> None:
        if self._conn is not None:
            await self._conn.close()
            self._conn = None

    async def __aenter__(self) -> "SQLiteStore":
        await self.open()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    # ── trades ──────────────────────────────────────────────────────

    async def persist_trade(
        self,
        outcome: TradeOutcome,
        *,
        realized_pnl_usd: float | None = None,
    ) -> None:
        if self._conn is None:
            raise RuntimeError("SQLiteStore.open() must be called before persist_trade")

        signal = outcome.signal
        straddle = signal.straddle
        strike = float(straddle.strike) if straddle else 0.0
        expiry = pd.Timestamp(straddle.expiry_dt).isoformat() if straddle else ""
        hedge_ticks = (
            outcome.hedge_summary.n_ticks_executed
            if outcome.hedge_summary is not None else 0
        )
        payload = {
            "signal_reason": signal.reason,
            "edge_vol_pts": signal.edge_vol_pts,
            "rv_pred_pct": signal.rv_pred_pct,
            "iv_used_pct": signal.iv_used_pct,
            "entry_call_price": outcome.entry_call.average_price if outcome.entry_call else None,
            "entry_put_price": outcome.entry_put.average_price if outcome.entry_put else None,
            "exit_call_price": outcome.exit_call.average_price if outcome.exit_call else None,
            "exit_put_price": outcome.exit_put.average_price if outcome.exit_put else None,
        }
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO trades (
                trade_id, started_at, ended_at, signal_direction,
                n_contracts_btc, strike, expiry_dt, status,
                realized_pnl_usd, hedge_ticks_executed, error, payload_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                outcome.trade_id or f"abort_{outcome.started_at.value if outcome.started_at else 0}",
                outcome.started_at.isoformat() if outcome.started_at else "",
                outcome.ended_at.isoformat() if outcome.ended_at else None,
                int(signal.direction),
                float(outcome.n_contracts_btc),
                strike,
                expiry,
                outcome.status,
                realized_pnl_usd,
                hedge_ticks,
                outcome.error,
                json.dumps(payload, default=str),
            ),
        )
        await self._conn.commit()

    async def load_recent_trades(self, limit: int = 100) -> list[dict[str, Any]]:
        if self._conn is None:
            raise RuntimeError("SQLiteStore not opened")
        cursor = await self._conn.execute(
            "SELECT trade_id, started_at, ended_at, signal_direction, "
            "n_contracts_btc, strike, status, realized_pnl_usd, error "
            "FROM trades ORDER BY started_at DESC LIMIT ?",
            (limit,),
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            {
                "trade_id": r[0],
                "started_at": r[1],
                "ended_at": r[2],
                "signal_direction": r[3],
                "n_contracts_btc": r[4],
                "strike": r[5],
                "status": r[6],
                "realized_pnl_usd": r[7],
                "error": r[8],
            }
            for r in rows
        ]

    # ── equity ──────────────────────────────────────────────────────

    async def persist_equity_snapshot(self, snap: EquitySnapshot) -> None:
        if self._conn is None:
            raise RuntimeError("SQLiteStore not opened")
        await self._conn.execute(
            """
            INSERT INTO equity_snapshots (
                timestamp, equity_usd, realized_pnl_usd,
                peak_equity_usd, drawdown_pct, n_trades_closed
            ) VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                snap.timestamp.isoformat(),
                snap.equity_usd,
                snap.realized_pnl_usd,
                snap.peak_equity_usd,
                snap.drawdown_pct,
                snap.n_trades_closed,
            ),
        )
        await self._conn.commit()

    async def load_latest_equity(self) -> EquitySnapshot | None:
        if self._conn is None:
            raise RuntimeError("SQLiteStore not opened")
        cursor = await self._conn.execute(
            "SELECT timestamp, equity_usd, realized_pnl_usd, peak_equity_usd, "
            "drawdown_pct, n_trades_closed FROM equity_snapshots "
            "ORDER BY id DESC LIMIT 1",
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        return EquitySnapshot(
            timestamp=pd.Timestamp(row[0]),
            equity_usd=float(row[1]),
            realized_pnl_usd=float(row[2]),
            peak_equity_usd=float(row[3]),
            drawdown_pct=float(row[4]),
            n_trades_closed=int(row[5]),
        )

    # ── open positions (M3) ──────────────────────────────────────────

    async def persist_open_position(
        self,
        trade_id: str,
        call_instrument: str,
        put_instrument: str,
        signal_direction: int,
        n_contracts_btc: float,
        strike: float,
        expiry_dt: str,
        opened_at: str,
        entry_call_price: float = 0.0,
        entry_put_price: float = 0.0,
    ) -> None:
        """Record an open position so it survives engine crashes."""
        if self._conn is None:
            raise RuntimeError("SQLiteStore not opened")
        await self._conn.execute(
            """
            INSERT OR REPLACE INTO open_positions (
                trade_id, call_instrument, put_instrument,
                signal_direction, n_contracts_btc, strike,
                expiry_dt, opened_at, entry_call_price, entry_put_price
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                trade_id, call_instrument, put_instrument,
                signal_direction, n_contracts_btc, strike,
                expiry_dt, opened_at, entry_call_price, entry_put_price,
            ),
        )
        await self._conn.commit()

    async def remove_open_position(self, trade_id: str) -> None:
        """Remove a closed position from the open_positions table."""
        if self._conn is None:
            raise RuntimeError("SQLiteStore not opened")
        await self._conn.execute(
            "DELETE FROM open_positions WHERE trade_id = ?",
            (trade_id,),
        )
        await self._conn.commit()

    async def load_open_positions(self) -> list[dict[str, Any]]:
        """Load all open positions (for startup reconciliation)."""
        if self._conn is None:
            raise RuntimeError("SQLiteStore not opened")
        cursor = await self._conn.execute(
            "SELECT trade_id, call_instrument, put_instrument, "
            "signal_direction, n_contracts_btc, strike, expiry_dt, opened_at, "
            "entry_call_price, entry_put_price "
            "FROM open_positions",
        )
        rows = await cursor.fetchall()
        await cursor.close()
        return [
            {
                "trade_id": r[0],
                "call_instrument": r[1],
                "put_instrument": r[2],
                "signal_direction": r[3],
                "n_contracts_btc": r[4],
                "strike": r[5],
                "expiry_dt": r[6],
                "opened_at": r[7],
                "entry_call_price": r[8],
                "entry_put_price": r[9],
            }
            for r in rows
        ]

    # ── kill switch ────────────────────────────────────────────────

    async def get_kill_switch_state(self) -> tuple[bool, str | None, str | None]:
        if self._conn is None:
            raise RuntimeError("SQLiteStore not opened")
        cursor = await self._conn.execute(
            "SELECT halted, reason, halted_at FROM kill_switch_state WHERE id = 1",
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return (False, None, None)
        return (bool(row[0]), row[1], row[2])

    async def set_kill_switch_state(
        self,
        halted: bool,
        *,
        reason: str | None = None,
    ) -> None:
        if self._conn is None:
            raise RuntimeError("SQLiteStore not opened")
        halted_at = pd.Timestamp.now(tz="UTC").isoformat() if halted else None
        await self._conn.execute(
            "UPDATE kill_switch_state SET halted = ?, reason = ?, halted_at = ? "
            "WHERE id = 1",
            (int(halted), reason, halted_at),
        )
        await self._conn.commit()

    # ── bar cursor (H8) ──────────────────────────────────────────────

    async def is_bar_processed(self, bar_timestamp: pd.Timestamp) -> bool:
        """Return True iff a row exists in bar_cursor for this timestamp.

        Used by the engine on every ``_on_bar_close`` to refuse to
        re-process a bar it has already handled. Defence-in-depth
        against aggregator bugs / accidental replay against a live
        engine. The bar timestamp is the canonical exchange UTC
        boundary (set by BarClock); same restart, same boundary →
        same primary key.
        """
        if self._conn is None:
            raise RuntimeError("SQLiteStore not opened")
        cursor = await self._conn.execute(
            "SELECT 1 FROM bar_cursor WHERE bar_timestamp = ?",
            (bar_timestamp.isoformat(),),
        )
        row = await cursor.fetchone()
        await cursor.close()
        return row is not None

    async def mark_bar_processed(
        self,
        bar_timestamp: pd.Timestamp,
        *,
        signal_direction: int,
        signal_reason: str | None = None,
    ) -> None:
        """Record that a bar has finished its signal evaluation.

        Idempotent — repeated calls for the same bar timestamp are no-ops
        (``INSERT OR IGNORE``). Called from ``_on_bar_close`` AFTER the
        signal direction is decided but BEFORE any trade task is
        scheduled, so a crash mid-execution still leaves a cursor row
        and the next restart will skip the bar.
        """
        if self._conn is None:
            raise RuntimeError("SQLiteStore not opened")
        await self._conn.execute(
            "INSERT OR IGNORE INTO bar_cursor "
            "(bar_timestamp, signal_direction, signal_reason, fired_at) "
            "VALUES (?, ?, ?, ?)",
            (
                bar_timestamp.isoformat(),
                int(signal_direction),
                signal_reason,
                pd.Timestamp.now(tz="UTC").isoformat(),
            ),
        )
        await self._conn.commit()

    async def count_bars_processed(self) -> int:
        """Count of rows in bar_cursor — useful for stats/diagnostics."""
        if self._conn is None:
            raise RuntimeError("SQLiteStore not opened")
        cursor = await self._conn.execute("SELECT COUNT(*) FROM bar_cursor")
        row = await cursor.fetchone()
        await cursor.close()
        return int(row[0]) if row else 0

    # ── profit_scale_state (H4) ─────────────────────────────────────

    async def load_profit_scale_state(self) -> dict | None:
        """Return persisted ``ProfitScaleEstimator`` state, or None.

        Engine reads this at startup and calls ``load_state(state)`` on
        the estimator if non-None. Returns None if no row exists yet
        (cold start) so the estimator keeps its constructor defaults.
        """
        if self._conn is None:
            raise RuntimeError("SQLiteStore not opened")
        cursor = await self._conn.execute(
            "SELECT gross_per_btc_ema_usd, gross_count, hedge_per_btc_ema_usd, "
            "last_updated FROM profit_scale_state WHERE id = 1",
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        return {
            "gross_per_btc_ema_usd": float(row[0]),
            "gross_count": int(row[1]),
            "hedge_per_btc_ema_usd": float(row[2]),
            "last_updated": row[3],
        }

    async def save_profit_scale_state(self, state: dict) -> None:
        """Persist a state dict from ``ProfitScaleEstimator.state_dict()``.

        Idempotent — uses INSERT OR REPLACE keyed at id=1. Stamps
        ``last_updated`` with the current UTC timestamp so an operator
        can spot stale rows post-mortem.
        """
        if self._conn is None:
            raise RuntimeError("SQLiteStore not opened")
        await self._conn.execute(
            "INSERT OR REPLACE INTO profit_scale_state "
            "(id, gross_per_btc_ema_usd, gross_count, hedge_per_btc_ema_usd, last_updated) "
            "VALUES (1, ?, ?, ?, ?)",
            (
                float(state["gross_per_btc_ema_usd"]),
                int(state["gross_count"]),
                float(state["hedge_per_btc_ema_usd"]),
                pd.Timestamp.now(tz="UTC").isoformat(),
            ),
        )
        await self._conn.commit()

    # ── engine_state (S1+S3 parity-cycle round 2) ──────────────────

    async def load_engine_state(self) -> dict | None:
        """Return persisted engine state, or None.

        Covers (S1) EquityTracker fields, (S3) edge_ema / exec_cost_ema,
        (S5) kill_switch consecutive_errors, (S8) svi_recal counters.

        Engine reads at startup; if non-None, applies to ``EquityTracker``,
        ``self._edge_ema`` / ``_edge_warmup``, ``self._exec_cost_ema``,
        ``self.kill_switch._consecutive_errors``, and the SVI recal
        counters. Cold start (no row) → keep constructor defaults.
        """
        if self._conn is None:
            raise RuntimeError("SQLiteStore not opened")
        cursor = await self._conn.execute(
            "SELECT current_equity_usd, peak_equity_usd, realized_pnl_usd, "
            "n_trades_closed, edge_ema, edge_warmup_json, "
            "exec_cost_ema_usd_per_btc, "
            "kill_switch_consecutive_errors, svi_recal_count, "
            "svi_recal_failures, last_updated FROM engine_state "
            "WHERE id = 1",
        )
        row = await cursor.fetchone()
        await cursor.close()
        if row is None:
            return None
        import json as _json
        return {
            "current_equity_usd": float(row[0]),
            "peak_equity_usd": float(row[1]),
            "realized_pnl_usd": float(row[2]),
            "n_trades_closed": int(row[3]),
            "edge_ema": float(row[4]) if row[4] is not None else None,
            "edge_warmup": _json.loads(row[5]) if row[5] else [],
            "exec_cost_ema_usd_per_btc": float(row[6]),
            "kill_switch_consecutive_errors": int(row[7]),
            "svi_recal_count": int(row[8]),
            "svi_recal_failures": int(row[9]),
            "last_updated": row[10],
        }

    async def save_engine_state(self, state: dict) -> None:
        """Persist engine state. Idempotent (INSERT OR REPLACE at id=1).

        Optional keys default to 0 (S5/S8 counters) so older callers that
        don't yet pass them keep working post-migration.
        """
        if self._conn is None:
            raise RuntimeError("SQLiteStore not opened")
        import json as _json
        await self._conn.execute(
            "INSERT OR REPLACE INTO engine_state "
            "(id, current_equity_usd, peak_equity_usd, realized_pnl_usd, "
            "n_trades_closed, edge_ema, edge_warmup_json, "
            "exec_cost_ema_usd_per_btc, "
            "kill_switch_consecutive_errors, svi_recal_count, "
            "svi_recal_failures, last_updated) "
            "VALUES (1, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                float(state["current_equity_usd"]),
                float(state["peak_equity_usd"]),
                float(state["realized_pnl_usd"]),
                int(state["n_trades_closed"]),
                float(state["edge_ema"]) if state.get("edge_ema") is not None else None,
                _json.dumps(state.get("edge_warmup", [])),
                float(state["exec_cost_ema_usd_per_btc"]),
                int(state.get("kill_switch_consecutive_errors", 0)),
                int(state.get("svi_recal_count", 0)),
                int(state.get("svi_recal_failures", 0)),
                pd.Timestamp.now(tz="UTC").isoformat(),
            ),
        )
        await self._conn.commit()
