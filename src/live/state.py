"""SQLite state persistence for the paper trading engine.

Stores positions, trades, breaker state, daily equity, and checkpoints.
Uses aiosqlite for async I/O with WAL mode for crash safety.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional

import aiosqlite

from live.portfolio import Position

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
    order_id TEXT DEFAULT ''
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
    order_id TEXT DEFAULT ''
);

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

CREATE TABLE IF NOT EXISTS breaker_state (
    asset TEXT PRIMARY KEY,
    recent_pnl TEXT NOT NULL,
    recent_notional TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_positions_status ON positions(status);
CREATE INDEX IF NOT EXISTS idx_positions_asset ON positions(asset);
CREATE INDEX IF NOT EXISTS idx_trades_asset ON trades(asset);
CREATE INDEX IF NOT EXISTS idx_trades_timestamp ON trades(timestamp);
"""


class StateDB:
    """Async SQLite state persistence."""

    def __init__(self, db_path: str = "data/live/paper_state.db"):
        self._db_path = db_path
        self._db: Optional[aiosqlite.Connection] = None

    async def initialize(self) -> None:
        """Open DB and create schema."""
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

    # ── Positions ──

    async def open_position(self, pos: Position) -> int:
        """Insert a new open position. Returns row ID."""
        cursor = await self._db.execute(
            """INSERT INTO positions
               (asset, symbol, timeframe, side, direction, entry_price, entry_time,
                size_usd, hold_bars, bars_held, status, fill_method, order_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open', ?, ?)""",
            (pos.asset, pos.symbol, pos.timeframe, pos.side, pos.direction,
             pos.entry_price, pos.entry_time.isoformat() if pos.entry_time else "",
             pos.size_usd, pos.hold_bars, pos.bars_held,
             pos.fill_method, pos.order_id),
        )
        await self._db.commit()
        pos.id = cursor.lastrowid
        return cursor.lastrowid

    async def close_position(self, pos_id: int, exit_price: float,
                             exit_time: datetime, pnl: float) -> None:
        await self._db.execute(
            """UPDATE positions
               SET exit_price=?, exit_time=?, pnl=?, status='closed'
               WHERE id=?""",
            (exit_price, exit_time.isoformat(), pnl, pos_id),
        )
        await self._db.commit()

    async def update_bars_held(self, pos_id: int, bars_held: int) -> None:
        await self._db.execute(
            "UPDATE positions SET bars_held=? WHERE id=?",
            (bars_held, pos_id),
        )
        await self._db.commit()

    async def get_open_positions(self) -> list[Position]:
        """Fetch all open positions."""
        cursor = await self._db.execute(
            "SELECT * FROM positions WHERE status='open'")
        rows = await cursor.fetchall()
        cols = [d[0] for d in cursor.description]
        positions = []
        for row in rows:
            d = dict(zip(cols, row))
            positions.append(Position(
                id=d["id"],
                asset=d["asset"],
                symbol=d["symbol"],
                timeframe=d["timeframe"],
                side=d["side"],
                direction=d["direction"],
                entry_price=d["entry_price"],
                entry_time=datetime.fromisoformat(d["entry_time"]) if d["entry_time"] else None,
                size_usd=d["size_usd"],
                hold_bars=d["hold_bars"],
                bars_held=d["bars_held"],
                status=d["status"],
                fill_method=d.get("fill_method", ""),
                order_id=d.get("order_id", ""),
            ))
        return positions

    async def get_recent_closed_trades(self, asset: str, limit: int = 10) -> list[dict]:
        """Get recent closed trades for an asset (for F1 breaker)."""
        cursor = await self._db.execute(
            """SELECT pnl, size_usd FROM trades
               WHERE asset=? AND action='close' AND pnl IS NOT NULL
               ORDER BY timestamp DESC LIMIT ?""",
            (asset, limit),
        )
        rows = await cursor.fetchall()
        return [{"pnl": r[0], "size_usd": r[1]} for r in rows]

    # ── Trades ──

    async def log_trade(self, asset: str, symbol: str, side: str,
                        action: str, price: float, size_usd: float,
                        method: str = "", pnl: Optional[float] = None,
                        timeframe: str = "", order_id: str = "") -> int:
        cursor = await self._db.execute(
            """INSERT INTO trades
               (timestamp, asset, symbol, side, action, price, size_usd,
                method, pnl, timeframe, order_id)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (datetime.now(tz=None).isoformat(), asset, symbol, side, action,
             price, size_usd, method, pnl, timeframe, order_id),
        )
        await self._db.commit()
        return cursor.lastrowid

    # ── Daily Equity ──

    async def save_daily_equity(self, date: str, equity: float,
                                pnl: float, n_trades: int, n_signals: int) -> None:
        await self._db.execute(
            """INSERT OR REPLACE INTO daily_equity
               (date, equity, pnl, n_trades, n_signals)
               VALUES (?, ?, ?, ?, ?)""",
            (date, equity, pnl, n_trades, n_signals),
        )
        await self._db.commit()

    async def get_equity_history(self) -> list[tuple]:
        cursor = await self._db.execute(
            "SELECT date, equity, pnl FROM daily_equity ORDER BY date")
        return await cursor.fetchall()

    # ── Checkpoints ──

    async def save_checkpoint(self, key: str, value: str) -> None:
        await self._db.execute(
            "INSERT OR REPLACE INTO checkpoints (key, value) VALUES (?, ?)",
            (key, value),
        )
        await self._db.commit()

    async def load_checkpoint(self, key: str) -> Optional[str]:
        cursor = await self._db.execute(
            "SELECT value FROM checkpoints WHERE key=?", (key,))
        row = await cursor.fetchone()
        return row[0] if row else None

    # ── Breaker State ──

    async def save_breaker_state(self, state: dict) -> None:
        for asset, data in state.items():
            await self._db.execute(
                """INSERT OR REPLACE INTO breaker_state
                   (asset, recent_pnl, recent_notional)
                   VALUES (?, ?, ?)""",
                (asset, json.dumps(data["pnl"]), json.dumps(data["notional"])),
            )
        await self._db.commit()

    async def load_breaker_state(self) -> dict:
        cursor = await self._db.execute("SELECT * FROM breaker_state")
        rows = await cursor.fetchall()
        state = {}
        for row in rows:
            state[row[0]] = {
                "pnl": json.loads(row[1]),
                "notional": json.loads(row[2]),
            }
        return state
