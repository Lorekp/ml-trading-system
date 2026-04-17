"""Telegram notifications for paper trading events."""

from __future__ import annotations

import asyncio
import logging
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


class TelegramNotifier:
    """Send notifications via Telegram bot API."""

    def __init__(self, bot_token: str = "", chat_id: str = "",
                 enabled: bool = True):
        self._bot_token = bot_token
        self._chat_id = chat_id
        self._enabled = enabled and bool(bot_token) and bool(chat_id)
        self._session: Optional[aiohttp.ClientSession] = None

    @classmethod
    def from_config(cls, config) -> TelegramNotifier:
        return cls(
            bot_token=config.bot_token,
            chat_id=config.chat_id,
            enabled=config.enabled,
        )

    async def send(self, message: str) -> bool:
        """Send a message. Returns True on success."""
        if not self._enabled:
            return False

        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()

        url = f"https://api.telegram.org/bot{self._bot_token}/sendMessage"
        payload = {
            "chat_id": self._chat_id,
            "text": message,
            "parse_mode": "HTML",
            "disable_web_page_preview": True,
        }

        try:
            async with self._session.post(url, json=payload, timeout=10) as resp:
                if resp.status == 200:
                    return True
                logger.warning("Telegram send failed: %d", resp.status)
                return False
        except Exception as e:
            logger.warning("Telegram send error: %s", e)
            return False

    async def notify_startup(self, capital: float, n_models: int,
                             n_assets: int, demo: bool) -> None:
        mode = "DEMO" if demo else "LIVE"
        await self.send(
            f"<b>Paper Trader Started [{mode}]</b>\n"
            f"Capital: ${capital:,.0f}\n"
            f"Models: {n_models}\n"
            f"Assets: {n_assets}"
        )

    async def notify_trade(self, asset: str, side: str, size_usd: float,
                           price: float, method: str, tf: str) -> None:
        await self.send(
            f"<b>TRADE</b> {side} {asset} ${size_usd:,.0f} @ {price:.4f}\n"
            f"Method: {method} | TF: {tf}"
        )

    async def notify_exit(self, asset: str, side: str, pnl: float,
                          entry_price: float, exit_price: float) -> None:
        emoji = "+" if pnl >= 0 else ""
        await self.send(
            f"<b>EXIT</b> {asset} {side}\n"
            f"PnL: {emoji}${pnl:,.2f}\n"
            f"Entry: {entry_price:.4f} → Exit: {exit_price:.4f}"
        )

    async def notify_daily_summary(self, date: str, equity: float,
                                   daily_pnl: float, n_trades: int,
                                   n_open: int, drawdown: float) -> None:
        await self.send(
            f"<b>Daily Summary {date}</b>\n"
            f"Equity: ${equity:,.0f}\n"
            f"Daily PnL: ${daily_pnl:,.2f}\n"
            f"Trades: {n_trades} | Open: {n_open}\n"
            f"Drawdown: {drawdown:.1f}%"
        )

    async def notify_error(self, error: str) -> None:
        await self.send(f"<b>ERROR</b>\n{error[:500]}")

    async def notify_heartbeat(self, equity: float, n_positions: int,
                               n_signals: int) -> None:
        await self.send(
            f"Heartbeat | Eq: ${equity:,.0f} | "
            f"Pos: {n_positions} | Sig: {n_signals}"
        )

    async def notify_taker_fallback(self, asset: str, reason: str) -> None:
        await self.send(f"<b>TAKER FALLBACK</b> {asset}\n{reason}")

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
