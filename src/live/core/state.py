"""Crash-safe SQLite state persistence.

Key improvements over the original state.py:
- pending_orders table for tracking orders placed but not yet filled
- Transactional writes (atomic position open + trade + checkpoint)
- strategy_id on all tables for multi-strategy support
- Equity computed from trades (checkpoint is just a cache)
- Audit trail for all state changes
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import aiosqlite

from live.core.interfaces import Position, PendingOrder

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    symbol TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    side TEXT NOT NULL,
    direction REAL NOT NULL,
    entry_price REAL NOT NULL,
    entry_time TEXT NOT NULL,
    size_usd REAL NOT NULL,
    hold_bars INTEGER NOT NULL,
    bars_held INTEGER DEFAULT 0,
    exit_price REAL,
    exit_time TEXT,
    pnl REAL,
    status TEXT DEFAULT 'open',
    fill_method TEXT DEFAULT '',
    order_id TEXT DEFAULT '',
    strategy_id TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    asset TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    action TEXT NOT NULL,
    price REAL NOT NULL,
    size_usd REAL NOT NULL,
    method TEXT DEFAULT '',
    pnl REAL,
    timeframe TEXT DEFAULT '',
    order_id TEXT DEFAULT '',
    strategy_id TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS pending_orders (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    symbol TEXT NOT NULL,
    exchange_order_id TEXT NOT NULL,
    side TEXT NOT NULL,
    qty REAL NOT NULL,
    price REAL,
    order_type TEXT NOT NULL,
    strategy_id TEXT DEFAULT '',
    placed_at TEXT NOT NULL,
    status TEXT DEFAULT 'pending'
);

CREATE TABLE IF NOT EXISTS tca_records (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    strategy_id TEXT DEFAULT '',
    decision_ts TEXT, submit_ts TEXT, first_fill_ts TEXT, complete_ts TEXT,
    decision_mid REAL, arrival_mid REAL, fill_price REAL,
    impl_shortfall_bps REAL,
    arrival_cost_bps REAL,
    spread_cost_bps REAL,
    timing_cost_bps REAL,
    method TEXT,
    maker_fill_rate REAL,
    total_qty REAL, maker_qty REAL, taker_qty REAL,
    decision_to_submit_ms REAL,
    submit_to_first_fill_ms REAL,
    submit_to_complete_ms REAL
);

CREATE INDEX IF NOT EXISTS idx_tca_timestamp ON tca_records(timestamp);
CREATE INDEX IF NOT EXISTS idx_tca_symbol ON tca_records(symbol);

CREATE TABLE IF NOT EXISTS daily_equity (
    date TEXT PRIMARY KEY,
    equity REAL NOT NULL,
    pnl REAL DEFAULT 0,
    n_trades INTEGER DEFAULT 0,
    n_signals INTEGER DEFAULT 0
);

CREATE TABLE IF NOT EXISTS checkpoints (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS strategy_state (
    strategy_id TEXT PRIMARY KEY,
    state_json TEXT NOT NULL,
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    event TEXT NOT NULL,
    details TEXT DEFAULT ''
);

CREATE TABLE IF NOT EXISTS confidence_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    asset TEXT NOT NULL,
    tf TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    confidence REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_strategy ON positions(strategy_id);
CREATE INDEX IF NOT EXISTS idx_trades_asset ON trades(asset);
CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_id);
CREATE INDEX IF NOT EXISTS idx_pending_status ON pending_orders(status);
CREATE INDEX IF NOT EXISTS idx_conf_asset_ts ON confidence_log(asset, timestamp);
"""


class StateDB:
    """Crash-safe async SQLite state persistence."""

    def __init__(self, db_path: str = "data/live/paper_state.db"):
        self._db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        Path(self._db_path).parent.mkdir(parents=True, exist_ok=True)
        self._db = await aiosqlite.connect(self._db_path)
        await self._db.execute("PRAGMA journal_mode=WAL")
        await self._db.execute("PRAGMA synchronous=NORMAL")
        await self._db.executescript(_SCHEMA)
        await self._db.commit()
        logger.info("StateDB initialized: %s", self._db_path)

    async def close(self) -> None:
        if self._db:
            await self._db.close()

    # ── Atomic operations (crash-safe) ──

    async def atomic_open_position(
        self, pos: Position, trade_data: dict, checkpoint_updates: dict
    ) -> int:
        """All-or-nothing: insert position + log trade + update checkpoints.

        If crash mid-write, nothing is committed (SQLite WAL rollback).
        """
        async with self._db.cursor() as cur:
            await self._db.execute("BEGIN IMMEDIATE")
            try:
                # Insert position
                await cur.execute(
                    """INSERT INTO positions
                       (asset, symbol, timeframe, side, direction, entry_price, entry_time,
                        size_usd, hold_bars, bars_held, status, fill_method, order_id, strategy_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?, ?)""",
                    (pos.asset, pos.symbol, pos.timeframe, pos.side, pos.direction,
                     pos.entry_price, pos.entry_time.isoformat() if pos.entry_time else "",
                     pos.size_usd, pos.hold_bars, pos.bars_held,
                     pos.fill_method, pos.order_id, pos.strategy_id),
                )
                pos_id = cur.lastrowid

                # Log trade
                await cur.execute(
                    """INSERT INTO trades
                       (timestamp, asset, symbol, side, action, price, size_usd,
                        method, pnl, timeframe, order_id, strategy_id)
                       VALUES (?, ?, ?, ?, 'open', ?, ?, ?, NULL, ?, ?, ?)""",
                    (trade_data["timestamp"], pos.asset, pos.symbol, trade_data["side"],
                     pos.entry_price, pos.size_usd, pos.fill_method,
                     pos.timeframe, pos.order_id, pos.strategy_id),
                )

                # Update checkpoints
                for k, v in checkpoint_updates.items():
                    await cur.execute(
                        "INSERT OR REPLACE INTO checkpoints (key, value) VALUES (?, ?)",
                        (k, str(v)),
                    )

                # Audit
                await cur.execute(
                    "INSERT INTO audit_log (timestamp, event, details) VALUES (?, ?, ?)",
                    (datetime.now().isoformat(), "position_opened",
                     f"{pos.asset} {pos.side} ${pos.size_usd:.0f} @ {pos.entry_price:.4f}"),
                )

                await self._db.execute("COMMIT")
                pos.id = pos_id
                return pos_id

            except Exception:
                await self._db.execute("ROLLBACK")
                raise

    async def atomic_close_position(
        self, pos_id: int, exit_price: float, exit_time: datetime,
        pnl: float, trade_data: dict, checkpoint_updates: dict
    ) -> None:
        """All-or-nothing: close position + log trade + update checkpoints."""
        async with self._db.cursor() as cur:
            await self._db.execute("BEGIN IMMEDIATE")
            try:
                await cur.execute(
                    "UPDATE positions SET exit_price=?, exit_time=?, pnl=?, status='closed' WHERE id=?",
                    (exit_price, exit_time.isoformat(), pnl, pos_id),
                )

                await cur.execute(
                    """INSERT INTO trades
                       (timestamp, asset, symbol, side, action, price, size_usd,
                        method, pnl, timeframe, order_id, strategy_id)
                       VALUES (?, ?, ?, ?, 'close', ?, ?, ?, ?, ?, ?, ?)""",
                    (trade_data["timestamp"], trade_data["asset"], trade_data["symbol"],
                     trade_data["side"], exit_price, trade_data["size_usd"],
                     trade_data.get("method", "taker"), pnl,
                     trade_data.get("timeframe", ""), trade_data.get("order_id", ""),
                     trade_data.get("strategy_id", "")),
                )

                for k, v in checkpoint_updates.items():
                    await cur.execute(
                        "INSERT OR REPLACE INTO checkpoints (key, value) VALUES (?, ?)",
                        (k, str(v)),
                    )

                await cur.execute(
                    "INSERT INTO audit_log (timestamp, event, details) VALUES (?, ?, ?)",
                    (datetime.now().isoformat(), "position_closed",
                     f"pos_id={pos_id} pnl=${pnl:.2f}"),
                )

                await self._db.execute("COMMIT")

            except Exception:
                await self._db.execute("ROLLBACK")
                raise

    # ── Positions ──

    async def get_open_positions(self, strategy_id: str = None) -> list[Position]:
        if strategy_id:
            cursor = await self._db.execute(
                "SELECT * FROM positions WHERE status='open' AND strategy_id=?",
                (strategy_id,))
        else:
            cursor = await self._db.execute(
                "SELECT * FROM positions WHERE status='open'")
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [self._row_to_position(dict(zip(cols, row))) for row in rows]

    async def update_bars_held(self, pos_id: int, bars_held: int) -> None:
        await self._db.execute(
            "UPDATE positions SET bars_held=? WHERE id=?", (bars_held, pos_id))
        await self._db.commit()

    # ── Pending Orders ──

    async def insert_pending_order(self, order: PendingOrder) -> int:
        cursor = await self._db.execute(
            """INSERT INTO pending_orders
               (symbol, exchange_order_id, side, qty, price, order_type,
                strategy_id, placed_at, status)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
            (order.symbol, order.exchange_order_id, order.side, order.qty,
             order.price, order.order_type, order.strategy_id,
             order.placed_at.isoformat() if order.placed_at else datetime.now().isoformat()),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def update_pending_order(self, order_id: int, status: str) -> None:
        await self._db.execute(
            "UPDATE pending_orders SET status=? WHERE id=?", (status, order_id))
        await self._db.commit()

    async def get_pending_orders(self, status: str = "pending") -> list[PendingOrder]:
        cursor = await self._db.execute(
            "SELECT * FROM pending_orders WHERE status=?", (status,))
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        result = []
        for row in rows:
            d = dict(zip(cols, row))
            result.append(PendingOrder(
                id=d["id"], symbol=d["symbol"],
                exchange_order_id=d["exchange_order_id"],
                side=d["side"], qty=d["qty"], price=d.get("price"),
                order_type=d["order_type"], strategy_id=d.get("strategy_id", ""),
                placed_at=datetime.fromisoformat(d["placed_at"]) if d.get("placed_at") else None,
                status=d["status"],
            ))
        return result

    # ── Equity from trades (authoritative, not from checkpoint) ──

    async def compute_equity_from_trades(self, starting_capital: float) -> float:
        """Recompute equity from all closed trade PnL. Authoritative on restart."""
        cursor = await self._db.execute(
            "SELECT COALESCE(SUM(pnl), 0) FROM trades WHERE action='close' AND pnl IS NOT NULL")
        row = await cursor.fetchone()
        total_pnl = row[0] if row else 0.0
        return starting_capital + total_pnl

    # ── Strategy State ──

    async def save_strategy_state(self, strategy_id: str, state: dict) -> None:
        await self._db.execute(
            """INSERT OR REPLACE INTO strategy_state
               (strategy_id, state_json, updated_at) VALUES (?, ?, ?)""",
            (strategy_id, json.dumps(state), datetime.now().isoformat()),
        )
        await self._db.commit()

    async def load_strategy_state(self, strategy_id: str) -> Optional[dict]:
        cursor = await self._db.execute(
            "SELECT state_json FROM strategy_state WHERE strategy_id=?",
            (strategy_id,))
        row = await cursor.fetchone()
        return json.loads(row[0]) if row else None

    # ── Daily Equity ──

    async def save_daily_equity(self, date: str, equity: float,
                                pnl: float, n_trades: int, n_signals: int) -> None:
        await self._db.execute(
            """INSERT OR REPLACE INTO daily_equity
               (date, equity, pnl, n_trades, n_signals) VALUES (?, ?, ?, ?, ?)""",
            (date, equity, pnl, n_trades, n_signals))
        await self._db.commit()

    async def get_equity_history(self) -> list[tuple]:
        cursor = await self._db.execute(
            "SELECT date, equity, pnl FROM daily_equity ORDER BY date")
        return await cursor.fetchall()

    # ── Checkpoints (fast cache, not authoritative) ──

    async def save_checkpoint(self, key: str, value: str) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO checkpoints (key, value) VALUES (?, ?)",
            (key, value))
        await self._db.commit()

    async def load_checkpoint(self, key: str) -> Optional[str]:
        cursor = await self._db.execute(
            "SELECT value FROM checkpoints WHERE key=?", (key,))
        row = await cursor.fetchone()
        return row[0] if row else None

    # ── TCA Records ──

    async def insert_tca(self, tca) -> int:
        """Insert a TCA record. Accepts TCARecord dataclass."""
        cursor = await self._db.execute(
            """INSERT INTO tca_records
               (timestamp, symbol, side, strategy_id,
                decision_ts, submit_ts, first_fill_ts, complete_ts,
                decision_mid, arrival_mid, fill_price,
                impl_shortfall_bps, arrival_cost_bps, spread_cost_bps, timing_cost_bps,
                method, maker_fill_rate, total_qty, maker_qty, taker_qty,
                decision_to_submit_ms, submit_to_first_fill_ms, submit_to_complete_ms)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (tca.timestamp.isoformat() if tca.timestamp else datetime.now().isoformat(),
             tca.symbol, tca.side, tca.strategy_id,
             tca.decision_ts.isoformat() if tca.decision_ts else None,
             tca.submit_ts.isoformat() if tca.submit_ts else None,
             tca.first_fill_ts.isoformat() if tca.first_fill_ts else None,
             tca.complete_ts.isoformat() if tca.complete_ts else None,
             tca.decision_mid, tca.arrival_mid, tca.fill_price,
             tca.impl_shortfall_bps, tca.arrival_cost_bps,
             tca.spread_cost_bps, tca.timing_cost_bps,
             tca.method, tca.maker_fill_rate,
             tca.total_qty, tca.maker_qty, tca.taker_qty,
             tca.decision_to_submit_ms, tca.submit_to_first_fill_ms,
             tca.submit_to_complete_ms),
        )
        await self._db.commit()
        return cursor.lastrowid

    async def get_recent_tca(self, hours: int = 24) -> list[dict]:
        """Get TCA records from the last N hours."""
        cutoff = (datetime.now() - timedelta(hours=hours)).isoformat()
        cursor = await self._db.execute(
            "SELECT * FROM tca_records WHERE timestamp >= ? ORDER BY timestamp",
            (cutoff,))
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in rows]

    # ── Audit ──

    async def audit(self, event: str, details: str = "") -> None:
        await self._db.execute(
            "INSERT INTO audit_log (timestamp, event, details) VALUES (?, ?, ?)",
            (datetime.now().isoformat(), event, details))
        await self._db.commit()

    # ── Confidence Log (for forward asset selection) ──

    async def log_confidence(self, asset: str, tf: str,
                             timestamp: datetime, confidence: float) -> None:
        """Log a signal confidence for asset selection ranking."""
        await self._db.execute(
            "INSERT INTO confidence_log (asset, tf, timestamp, confidence) VALUES (?, ?, ?, ?)",
            (asset, tf, timestamp.isoformat(), confidence))
        await self._db.commit()

    async def log_confidences_batch(self, records: list[tuple[str, str, datetime, float]]) -> None:
        """Batch insert confidence records: [(asset, tf, timestamp, confidence), ...]."""
        await self._db.executemany(
            "INSERT INTO confidence_log (asset, tf, timestamp, confidence) VALUES (?, ?, ?, ?)",
            [(a, tf, ts.isoformat(), c) for a, tf, ts, c in records])
        await self._db.commit()

    async def get_mean_confidences(self, since: datetime) -> dict[str, float]:
        """Return mean confidence per asset since the given date.

        Used by asset selector to rank expansion pool, matching
        yolo_compound.py:116-124 (mean confidence over 3-month window).
        """
        cursor = await self._db.execute(
            "SELECT asset, AVG(confidence) FROM confidence_log "
            "WHERE timestamp >= ? GROUP BY asset",
            (since.isoformat(),))
        rows = await cursor.fetchall()
        return {row[0]: row[1] for row in rows}

    # ── Helpers ──

    @staticmethod
    def _row_to_position(d: dict) -> Position:
        return Position(
            id=d["id"], asset=d["asset"], symbol=d["symbol"],
            timeframe=d["timeframe"], side=d["side"], direction=d["direction"],
            entry_price=d["entry_price"],
            entry_time=datetime.fromisoformat(d["entry_time"]) if d.get("entry_time") else None,
            size_usd=d["size_usd"], hold_bars=d["hold_bars"], bars_held=d["bars_held"],
            status=d["status"], fill_method=d.get("fill_method", ""),
            order_id=d.get("order_id", ""), strategy_id=d.get("strategy_id", ""),
        )
