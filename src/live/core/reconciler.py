"""Startup reconciliation: sync DB state with actual exchange state.

Handles crash scenarios:
- Position on exchange but not in DB (crash after fill, before DB save)
- Position in DB but not on exchange (closed/liquidated externally)
- Stale pending orders (crash during MkrTkr60 60s wait)
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone, timedelta
from typing import Optional

from live.core.interfaces import (
    ExchangeAdapter, Position, PendingOrder, ReconcileReport,
)
from live.core.state import StateDB

logger = logging.getLogger(__name__)

STALE_ORDER_AGE = timedelta(minutes=5)


class Reconciler:
    """Sync local DB with exchange state on startup."""

    def __init__(self, exchange: ExchangeAdapter, state: StateDB,
                 stale_order_age: timedelta | None = None,
                 paper_mode: bool = False):
        self._exchange = exchange
        self._state = state
        self._stale_order_age = stale_order_age or STALE_ORDER_AGE
        self._paper_mode = paper_mode

    async def reconcile(self) -> ReconcileReport:
        """Full reconciliation. Call once on startup before trading begins."""
        report = ReconcileReport(
            orphaned_positions=[], ghost_positions=[],
            stale_orders_cancelled=0, adopted_positions=0, warnings=[],
        )

        # 1. Reconcile positions
        await self._reconcile_positions(report)

        # 2. Reconcile pending orders
        await self._reconcile_orders(report)

        # Log report
        if report.orphaned_positions:
            logger.warning("Found %d orphaned positions (on exchange, not in DB)",
                           len(report.orphaned_positions))
        if report.ghost_positions:
            logger.warning("Found %d ghost positions (in DB, not on exchange)",
                           len(report.ghost_positions))
        if report.stale_orders_cancelled:
            logger.info("Cancelled %d stale pending orders", report.stale_orders_cancelled)
        if report.adopted_positions:
            logger.info("Adopted %d orphaned positions into DB", report.adopted_positions)

        if not any([report.orphaned_positions, report.ghost_positions,
                     report.stale_orders_cancelled]):
            logger.info("Reconciliation clean — no mismatches")

        await self._state.audit("reconciliation",
                                f"orphaned={len(report.orphaned_positions)} "
                                f"ghost={len(report.ghost_positions)} "
                                f"stale_cancelled={report.stale_orders_cancelled} "
                                f"adopted={report.adopted_positions}")

        return report

    async def _reconcile_positions(self, report: ReconcileReport) -> None:
        """Compare exchange positions vs DB positions."""
        if self._paper_mode:
            db_positions = await self._state.get_open_positions()
            if db_positions:
                logger.info("Paper mode: preserving %d open positions from previous session",
                            len(db_positions))
            return

        try:
            exchange_positions = await self._exchange.get_open_positions()
        except Exception as e:
            report.warnings.append(f"Failed to fetch exchange positions: {e}")
            return

        db_positions = await self._state.get_open_positions()

        # Build lookup by symbol
        db_by_symbol = {}
        for pos in db_positions:
            db_by_symbol.setdefault(pos.symbol, []).append(pos)

        exchange_symbols = set(exchange_positions.keys())
        db_symbols = set(db_by_symbol.keys())

        # Orphaned: on exchange but not in DB
        for symbol in exchange_symbols - db_symbols:
            ep = exchange_positions[symbol]
            report.orphaned_positions.append({
                "symbol": symbol,
                "side": ep["side"],
                "size": ep["size"],
                "entry_price": ep["entry_price"],
            })

            # Adopt into DB as best-effort
            asset = symbol.replace("USDT", "")
            pos = Position(
                asset=asset, symbol=symbol, timeframe="unknown",
                side="long" if ep["side"] == "buy" else "short",
                direction=1.0 if ep["side"] == "buy" else -1.0,
                entry_price=ep["entry_price"],
                entry_time=datetime.now(tz=timezone.utc),
                size_usd=ep["size"] * ep["entry_price"],
                hold_bars=4,  # conservative default
                strategy_id="reconciled",
                fill_method="unknown",
            )
            try:
                await self._state.atomic_open_position(
                    pos,
                    {"timestamp": datetime.now().isoformat(), "side": ep["side"]},
                    {},
                )
                report.adopted_positions += 1
                logger.warning("Adopted orphaned position: %s %s %.4f",
                               symbol, ep["side"], ep["entry_price"])
            except Exception as e:
                report.warnings.append(f"Failed to adopt {symbol}: {e}")

        # Ghost: in DB but not on exchange
        for symbol in db_symbols - exchange_symbols:
            for pos in db_by_symbol[symbol]:
                report.ghost_positions.append({
                    "pos_id": pos.id,
                    "symbol": symbol,
                    "side": pos.side,
                    "entry_price": pos.entry_price,
                })
                # Mark as closed with unknown exit
                try:
                    await self._state.atomic_close_position(
                        pos.id, pos.entry_price, datetime.now(tz=timezone.utc), 0.0,
                        {"timestamp": datetime.now().isoformat(), "asset": pos.asset,
                         "symbol": pos.symbol, "side": "Sell" if pos.direction > 0 else "Buy",
                         "size_usd": pos.size_usd, "strategy_id": pos.strategy_id},
                        {},
                    )
                    logger.warning("Closed ghost position: %s (pos_id=%d)", symbol, pos.id)
                except Exception as e:
                    report.warnings.append(f"Failed to close ghost {symbol}: {e}")

    async def _reconcile_orders(self, report: ReconcileReport) -> None:
        """Cancel stale pending orders."""
        pending = await self._state.get_pending_orders("pending")
        now = datetime.now(tz=timezone.utc)

        for order in pending:
            age = now - (order.placed_at.replace(tzinfo=timezone.utc)
                         if order.placed_at else now)

            if age > self._stale_order_age:
                # Try to cancel on exchange
                try:
                    cancelled = await self._exchange.cancel_order(
                        order.symbol, order.exchange_order_id)
                    if cancelled:
                        await self._state.update_pending_order(order.id, "cancelled")
                        report.stale_orders_cancelled += 1
                        logger.info("Cancelled stale order: %s %s (age: %s)",
                                    order.symbol, order.exchange_order_id, age)
                    else:
                        # May have filled while we weren't looking
                        status = await self._exchange.get_order_status(
                            order.symbol, order.exchange_order_id)
                        if status.get("status") == "filled":
                            await self._state.update_pending_order(order.id, "filled")
                            report.warnings.append(
                                f"Stale order {order.symbol} was actually filled")
                        else:
                            await self._state.update_pending_order(order.id, "orphaned")
                except Exception as e:
                    report.warnings.append(
                        f"Failed to reconcile order {order.symbol}: {e}")
