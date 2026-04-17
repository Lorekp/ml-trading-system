"""Paper execution with the backtest's calibrated slippage model.

Mirrors `experiments/drawdown_improvements.py:487-495` exactly. Entry/exit fill
prices are adjusted using `signal_engine.slippage.compute_slippage`, which
combines:
  - per-symbol half-spread from `data/execution_calibration.json`
  - Almgren-Chriss square-root market impact
  - VIP-tier maker/taker fee blend

Daily volume (USDT) is fetched via `exchange.fetch_ticker(symbol)["quote_volume_24h"]`
and cached with a TTL so we don't hammer the exchange on every bar.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from live.core.interfaces import ExchangeAdapter, TradeResult
from live.core.state import StateDB
from signal_engine.slippage import compute_slippage, load_per_symbol_calibration

logger = logging.getLogger(__name__)


class CalibratedPaperExecution:
    """Paper execution with calibrated, volume-aware slippage.

    The fill-price math matches the backtest line-for-line, so live trade PnL
    is directly comparable to backtest trade PnL when the signal stream is
    identical.

    Daily volume is cached per-symbol with a TTL (default 1h). On fetch
    failure the cache returns stale data if available; otherwise the slippage
    model falls back to `DEFAULT_TIER_PARAMS["tier4"]` (the conservative tier).
    """

    def __init__(
        self,
        exchange: ExchangeAdapter,
        state: StateDB,
        *,
        vip_tier: tuple[str, float, float],
        calibration_path: str | Path = "data/execution_calibration.json",
        daily_vol_ttl_s: int = 3600,
    ) -> None:
        self._exchange = exchange
        self._state = state
        self._vip_tier = vip_tier
        self._cal = load_per_symbol_calibration(Path(calibration_path))
        self._ttl = daily_vol_ttl_s
        # symbol -> (daily_volume_usd, fetched_at_unix)
        self._vol_cache: dict[str, tuple[float, float]] = {}

        if not self._cal:
            logger.warning(
                "CalibratedPaperExecution: no per-symbol calibration loaded "
                "from %s — falling back to DEFAULT_TIER_PARAMS.",
                calibration_path,
            )
        logger.info(
            "CalibratedPaperExecution ready: tier=%s maker=%.2fbp taker=%.2fbp cal=%d assets",
            vip_tier[0], vip_tier[1], vip_tier[2], len(self._cal),
        )

    async def _get_daily_volume(self, symbol: str, mid: float) -> float:
        """Fetch or read cached daily quote volume (USDT) for `symbol`.

        Returns 0.0 only when both the cache is cold and the fetch fails —
        callers interpret 0 as "no data, use the slippage model's fallback".
        """
        now = time.time()
        cached = self._vol_cache.get(symbol)
        if cached is not None:
            vol, fetched_at = cached
            if now - fetched_at < self._ttl:
                return vol

        # Refresh via ticker
        try:
            ticker = await self._exchange.fetch_ticker(symbol)
        except Exception as e:
            logger.warning("fetch_ticker failed for %s: %s (using cached=%s)",
                            symbol, e, cached is not None)
            return cached[0] if cached is not None else 0.0

        qv = float(ticker.get("quote_volume_24h", 0.0) or 0.0)
        if qv <= 0.0:
            # Some providers don't supply quote_volume_24h. Fall back to
            # base volume × mid price.
            base_vol = float(ticker.get("volume_24h", 0.0) or 0.0)
            qv = base_vol * mid if (base_vol > 0 and mid > 0) else 0.0

        self._vol_cache[symbol] = (qv, now)
        return qv

    async def execute_entry(
        self,
        symbol: str,
        side: str,
        qty: float,
        strategy_id: str = "",
    ) -> Optional[TradeResult]:
        ticker = await self._exchange.fetch_ticker(symbol)
        bid = float(ticker.get("bid", 0) or 0)
        ask = float(ticker.get("ask", 0) or 0)
        last = float(ticker.get("last_price", 0) or 0)
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else last

        if mid <= 0:
            logger.warning("Calibrated: no price for %s, skipping", symbol)
            return None

        daily_vol = await self._get_daily_volume(symbol, mid)

        # Strip USDT suffix to get asset name for the calibration lookup
        asset = symbol[:-4] if symbol.endswith("USDT") else symbol
        notional = qty * mid
        slip = compute_slippage(
            asset=asset,
            position_notional=notional,
            daily_volume=daily_vol,
            per_symbol_cal=self._cal,
            vip_tier=self._vip_tier,
        )

        # Buy side pays the ask (up-slip); sell side hits the bid (down-slip)
        if side in ("Buy", "open_long", "close_short"):
            fill_price = mid * (1.0 + slip)
        else:
            fill_price = mid * (1.0 - slip)

        return TradeResult(
            filled=True,
            fill_price=fill_price,
            fill_qty=qty,
            method="paper_calibrated",
            order_id=f"paper-{datetime.now(tz=timezone.utc).strftime('%H%M%S%f')}",
            decision_mid=mid,
        )

    async def execute_exit(
        self,
        symbol: str,
        side: str,
        qty: float,
    ) -> Optional[TradeResult]:
        # Exit uses the same math as entry; the passed `side` already
        # reflects the closing direction.
        return await self.execute_entry(symbol, side, qty)
