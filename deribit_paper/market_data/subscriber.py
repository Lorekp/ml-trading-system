"""Wires the Deribit WebSocket client to the trade buffers.

Responsibilities:

1. Subscribe to the required Deribit channels:

   - ``trades.option.BTC.raw`` — all BTC option trades
   - ``trades.BTC-PERPETUAL.raw`` — perp trades (feeds TSRV)
   - ``deribit_price_index.btc_usd`` — BTC index price (1Hz)

2. Decode each push notification into a :class:`Trade` and append it
   to the appropriate buffer.

3. Parse option instrument names to extract (strike, expiry, type):
   Deribit uses ``BTC-28NOV25-60000-C`` format — a fixed regex handles this.

4. Filter incoming option trades by moneyness and DTE so the buffer
   stays focused on trades that matter to the straddle builder.

Keeps the main engine's ``on_message`` path fully async and lock-free.
"""

from __future__ import annotations

import logging
import re
from typing import Any

import pandas as pd

from deribit_paper.client.ws import DeribitWebSocketClient
from deribit_paper.market_data.buffer import (
    IndexPriceBuffer,
    Trade,
    TradeBuffer,
)

logger = logging.getLogger(__name__)


# Deribit option instrument name format: <ASSET>-<DDMMMYY>-<STRIKE>-<C|P>
# Example: BTC-28NOV25-60000-C
_OPTION_NAME_RE = re.compile(
    r"^(?P<asset>[A-Z]+)-(?P<expiry>\d{1,2}[A-Z]{3}\d{2})-(?P<strike>\d+(?:\.\d+)?)-(?P<type>[CP])$",
)


def parse_option_instrument(name: str) -> tuple[pd.Timestamp, float, str] | None:
    """Parse a Deribit option instrument name into (expiry_dt, strike, type).

    Returns ``None`` if the name doesn't match the expected format
    (e.g., a perpetual or future, not an option).
    """
    match = _OPTION_NAME_RE.match(name)
    if not match:
        return None
    expiry_str = match.group("expiry")  # e.g. "28NOV25"
    strike = float(match.group("strike"))
    option_type = match.group("type")  # "C" or "P"
    try:
        # Deribit expiries settle at 08:00 UTC on expiry day
        expiry_dt = pd.Timestamp(
            pd.to_datetime(expiry_str, format="%d%b%y"),
            tz="UTC",
        ) + pd.Timedelta(hours=8)
    except ValueError:
        return None
    return expiry_dt, strike, option_type


class MarketDataSubscriber:
    """Orchestrates Deribit WS subscriptions and routes to buffers.

    Usage::

        ws = DeribitWebSocketClient(ws_url)
        sub = MarketDataSubscriber(
            ws=ws,
            perp_buffer=perp_buffer,
            option_buffer=option_buffer,
            index_buffer=index_buffer,
            currency="BTC",
        )
        await sub.start()
        await ws.dispatch_forever()  # blocks, reading messages from the WS

    Can also be fed synthetic events via :meth:`handle_trade_event` for
    testing without a live WS connection.
    """

    def __init__(
        self,
        *,
        ws: DeribitWebSocketClient | None,
        perp_buffer: TradeBuffer,
        option_buffer: TradeBuffer,
        index_buffer: IndexPriceBuffer,
        currency: str = "BTC",
        moneyness_lo: float = 0.80,   # wider than signal filter for buffer retention
        moneyness_hi: float = 1.20,
        dte_min_days: float = 0.0,
        dte_max_days: float = 21.0,
        trades_interval: str = "raw",
    ):
        # ``trades_interval`` selects the trades-channel granularity:
        # - ``"raw"`` (default) gives every individual trade. Deribit
        #   gates the ``.raw`` variant behind WS auth in some
        #   configurations, so this requires credentials.
        # - ``"100ms"`` aggregates trades into 100ms snapshots and
        #   works on truly public WS connections (paper mode without
        #   credentials). At the 15-min bar level the granularity
        #   difference is invisible.
        if trades_interval not in {"raw", "100ms", "agg2"}:
            raise ValueError(
                f"trades_interval must be 'raw' | '100ms' | 'agg2', "
                f"got {trades_interval!r}",
            )
        self._ws = ws
        self.perp_buffer = perp_buffer
        self.option_buffer = option_buffer
        self.index_buffer = index_buffer
        self.currency = currency.upper()
        self.moneyness_lo = moneyness_lo
        self.moneyness_hi = moneyness_hi
        self.dte_min_days = dte_min_days
        self.dte_max_days = dte_max_days
        self.trades_interval = trades_interval
        self._kill_switch: Any = None  # injected after construction via set_kill_switch

        # Counters for observability
        self.stats = {
            "perp_added": 0,
            "perp_dup": 0,
            "option_added": 0,
            "option_dup": 0,
            "option_filtered_moneyness": 0,
            "option_filtered_dte": 0,
            "option_unparseable": 0,
            "index_updates": 0,
        }

    def set_kill_switch(self, ks: Any) -> None:
        """Inject the kill switch after construction (deferred wiring).

        Called from ``TradingEngine.initialize()`` once the kill switch
        has been constructed (Phase 6), after the subscriber (Phase 2).
        """
        self._kill_switch = ks

    async def start(self) -> None:
        """Subscribe to all required channels.  Requires an active ``ws``."""
        if self._ws is None:
            raise RuntimeError("MarketDataSubscriber.start() requires a ws client")
        perp_channel = f"trades.{self.currency}-PERPETUAL.{self.trades_interval}"
        option_channel = f"trades.option.{self.currency}.{self.trades_interval}"
        index_channel = f"deribit_price_index.{self.currency.lower()}_usd"

        await self._ws.subscribe([perp_channel], self._on_perp_trade_event)
        await self._ws.subscribe([option_channel], self._on_option_trade_event)
        await self._ws.subscribe([index_channel], self._on_index_event)
        logger.info(
            "MarketDataSubscriber subscribed to: %s, %s, %s",
            perp_channel, option_channel, index_channel,
        )

    # ── Event handlers (async callbacks from DeribitWebSocketClient) ──

    async def _on_perp_trade_event(self, params: dict[str, Any]) -> None:
        """Deribit trades channel sends a list of trades in ``params['data']``."""
        if self._kill_switch is not None:
            self._kill_switch.mark_ws_message()
        data = params.get("data") or []
        for raw in data:
            trade = self._parse_perp_trade(raw)
            if trade is not None:
                if self.perp_buffer.append(trade):
                    self.stats["perp_added"] += 1
                else:
                    self.stats["perp_dup"] += 1

    async def _on_option_trade_event(self, params: dict[str, Any]) -> None:
        if self._kill_switch is not None:
            self._kill_switch.mark_ws_message()
        data = params.get("data") or []
        for raw in data:
            trade = self._parse_option_trade(raw)
            if trade is None:
                continue
            if self._is_option_filtered(trade):
                continue
            if self.option_buffer.append(trade):
                self.stats["option_added"] += 1
            else:
                self.stats["option_dup"] += 1

    async def _on_index_event(self, params: dict[str, Any]) -> None:
        """Deribit ``deribit_price_index.*`` sends a single dict in ``data``."""
        if self._kill_switch is not None:
            self._kill_switch.mark_ws_message()
        data = params.get("data") or {}
        price = data.get("price")
        ts_ms = data.get("timestamp")
        if price is None or ts_ms is None:
            return
        ts = pd.Timestamp(ts_ms, unit="ms", tz="UTC")
        self.index_buffer.append(ts, float(price))
        self.stats["index_updates"] += 1

    # ── Synthetic-event entry points (for tests) ──────────────────────

    def handle_trade_event(
        self,
        channel: str,
        params: dict[str, Any],
    ) -> None:
        """Synchronous helper for tests — dispatches to the right async handler.

        Blocks until the handler completes.  Only safe to call outside an
        event loop, or wrapped in ``asyncio.run``.
        """
        import asyncio
        if channel.startswith("trades.") and "PERPETUAL" in channel:
            asyncio.get_event_loop().run_until_complete(
                self._on_perp_trade_event(params),
            )
        elif channel.startswith("trades.option."):
            asyncio.get_event_loop().run_until_complete(
                self._on_option_trade_event(params),
            )
        elif channel.startswith("deribit_price_index."):
            asyncio.get_event_loop().run_until_complete(
                self._on_index_event(params),
            )
        else:
            raise ValueError(f"Unknown channel: {channel}")

    # ── Parsing ───────────────────────────────────────────────────────

    @staticmethod
    def _direction_from_deribit(raw: dict[str, Any]) -> int:
        """Deribit uses ``'direction': 'buy' | 'sell'``."""
        d = raw.get("direction", "")
        return 1 if d == "buy" else (-1 if d == "sell" else 0)

    def _parse_perp_trade(self, raw: dict[str, Any]) -> Trade | None:
        try:
            return Trade(
                trade_id=str(raw["trade_id"]),
                timestamp=pd.Timestamp(raw["timestamp"], unit="ms", tz="UTC"),
                instrument_name=raw["instrument_name"],
                price=float(raw["price"]),
                amount=float(raw["amount"]),
                direction=self._direction_from_deribit(raw),
                index_price=raw.get("index_price"),
                mark_price=raw.get("mark_price"),
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("Unparseable perp trade: %s (%s)", raw, exc)
            return None

    def _parse_option_trade(self, raw: dict[str, Any]) -> Trade | None:
        try:
            name = raw["instrument_name"]
            parsed = parse_option_instrument(name)
            if parsed is None:
                self.stats["option_unparseable"] += 1
                return None
            expiry_dt, strike, option_type = parsed
            return Trade(
                trade_id=str(raw["trade_id"]),
                timestamp=pd.Timestamp(raw["timestamp"], unit="ms", tz="UTC"),
                instrument_name=name,
                price=float(raw["price"]),
                amount=float(raw["amount"]),
                direction=self._direction_from_deribit(raw),
                strike=strike,
                option_type=option_type,
                expiry_dt=expiry_dt,
                iv=float(raw["iv"]) if raw.get("iv") is not None else None,
                index_price=raw.get("index_price"),
                mark_price=raw.get("mark_price"),
            )
        except (KeyError, ValueError, TypeError) as exc:
            logger.debug("Unparseable option trade: %s (%s)", raw, exc)
            self.stats["option_unparseable"] += 1
            return None

    def _is_option_filtered(self, trade: Trade) -> bool:
        """Apply moneyness + DTE filters so the buffer stays focused."""
        if trade.index_price is None or trade.strike is None or trade.expiry_dt is None:
            return False  # can't filter without required fields; keep
        moneyness = trade.strike / trade.index_price
        if moneyness < self.moneyness_lo or moneyness > self.moneyness_hi:
            self.stats["option_filtered_moneyness"] += 1
            return True
        dte_days = (trade.expiry_dt - trade.timestamp).total_seconds() / 86_400.0
        if dte_days < self.dte_min_days or dte_days > self.dte_max_days:
            self.stats["option_filtered_dte"] += 1
            return True
        return False
