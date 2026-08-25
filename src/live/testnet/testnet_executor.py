"""Testnet execution algo — real testnet orders with MkrTkr60 semantics.

Mirrors ``MkrTkr60`` (live.execution.mkrtk60) but routes through
``BinanceFuturesTestnet`` so every fill is a real testnet fill.

Per-trade flow:
  1. Fetch ticker (bid/ask/last).
  2. If maker_wait_s > 0, place post-only limit on the maker side, poll
     every poll_interval_s, and fall back to taker after maker_wait_s.
  3. Record the fill + a slippage/fees snapshot for reconciliation.

Records are appended to ``self.fills`` as ``FillRecord`` instances — the
reconciler reads this list to compute net PnL / slippage / fee deltas vs the
backtest assumptions.

Safe-mode (no creds): if the exchange has no api_key/api_secret the executor
transparently delegates to ``CalibratedPaperExecution`` so the CLI is usable
in a smoke run without testnet creds.  The CLI logs a clear "PAPER FALLBACK"
banner so reconciliation numbers are tagged.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from live.core.interfaces import ExchangeAdapter, TradeResult

logger = logging.getLogger(__name__)


@dataclass
class FillRecord:
    """One fill observed by the testnet executor — feeds the reconciler."""
    symbol: str
    side: str                 # "Buy" / "Sell" (raw)
    intent_side: str          # "open_long" / "open_short" / "close_long" / "close_short"
    qty: float
    decision_mid: float
    fill_price: float
    method: str               # "maker", "taker", "blended", "paper_calibrated"
    order_id: str = ""
    exchange_order_id: str = ""
    slippage_bps: float = 0.0
    fee_bps: float = 0.0
    fee_usd: float = 0.0
    notional_usd: float = 0.0
    strategy_id: str = ""
    ts_unix: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return asdict(self)


class TestnetExecutor:
    """MkrTkr60-flavoured executor that posts real testnet orders.

    Parameters mirror ``MkrTkr60`` so the shared configuration maps directly:

    - ``maker_wait_s``: how long to wait for a maker fill before falling
      back to taker.  0 = pure taker (fastest, no queue position).
    - ``poll_interval_s``: poll cadence for the open order.
    - ``max_retries``: cancelled-then-retry budget for the maker order.
    - ``vip_maker_bps`` / ``vip_taker_bps``: for fee attribution only
      (Binance testnet still charges maker/taker fees).
    - ``fills_path``: optional JSONL to persist every fill (for the
      reconciler to read in a separate process).
    """

    def __init__(
        self,
        exchange: ExchangeAdapter,
        state,  # StateDB — kept for engine wire-up; not used here
        *,
        maker_wait_s: int = 0,         # 0 = pure taker; historical paper default
        poll_interval_s: float = 5.0,
        max_retries: int = 2,
        vip_maker_bps: float = 1.60,   # VIP2 default
        vip_taker_bps: float = 3.75,
        fills_path: Optional[Path] = None,
    ) -> None:
        self._exchange = exchange
        self._state = state
        self._maker_wait_s = maker_wait_s
        self._poll_interval_s = poll_interval_s
        self._max_retries = max_retries
        self._vip_maker_bps = vip_maker_bps
        self._vip_taker_bps = vip_taker_bps
        self._fills_path = Path(fills_path) if fills_path else None
        self._fills: list[FillRecord] = []

        # Detect paper fallback — engine still works but reconciliation
        # numbers will be paper, not real-testnet.
        self._is_paper = (
            not getattr(exchange, "_api_key", "") or
            not getattr(exchange, "_api_secret", "")
        )
        if self._is_paper:
            logger.warning(
                "TestnetExecutor: no creds on exchange — falling back to "
                "CalibratedPaperExecution behaviour. Set "
                "BINANCE_TESTNET_API_KEY/SECRET to hit testnet.binancefuture.com."
            )

    # ── public API ──

    @property
    def fills(self) -> list[FillRecord]:
        return list(self._fills)

    @property
    def is_paper(self) -> bool:
        return self._is_paper

    def export_fills(self, path: Path) -> None:
        """Dump fills to JSONL for the reconciler.

        NEW (B-031): atomic write via .tmp + os.replace to avoid corrupting
        the JSONL on Ctrl+C mid-write. The previous whole-file rewrite on
        every fill was O(N²) and corruption-prone.
        """
        import json
        import os
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            with tmp.open("w") as f:
                for rec in self._fills:
                    f.write(json.dumps(rec.to_dict()) + "\n")
            os.replace(tmp, path)
        except Exception:
            # Best-effort cleanup of tmp on failure
            try:
                tmp.unlink()
            except Exception:
                pass
            raise
        logger.info("export_fills: wrote %d records to %s", len(self._fills), path)

    async def execute_entry(
        self,
        symbol: str,
        side: str,
        qty: float,
        strategy_id: str = "",
    ) -> Optional[TradeResult]:
        return await self._execute(symbol, side, qty, "open", strategy_id)

    async def execute_exit(
        self,
        symbol: str,
        side: str,
        qty: float,
    ) -> Optional[TradeResult]:
        return await self._execute(symbol, side, qty, "close", "")

    # ── internals ──

    async def _execute(
        self,
        symbol: str,
        side: str,
        qty: float,
        intent: str,
        strategy_id: str,
    ) -> Optional[TradeResult]:
        ticker = await self._exchange.fetch_ticker(symbol)
        bid = float(ticker.get("bid", 0) or 0)
        ask = float(ticker.get("ask", 0) or 0)
        last = float(ticker.get("last_price", 0) or 0)
        mid = (bid + ask) / 2.0 if (bid > 0 and ask > 0) else last
        if mid <= 0:
            logger.warning("TestnetExec: no price for %s, skipping", symbol)
            return None

        notional = qty * mid
        fee_bps = self._vip_maker_bps  # default attribution; flipped on taker below
        method = "taker"
        order_id = f"testnet-{datetime.now(tz=timezone.utc).strftime('%H%M%S%f')}"
        exchange_order_id = ""
        fill_price = mid

        if self._is_paper:
            # Paper fallback — re-use the calibrated slippage model so the
            # signal stream + slippage math are identical to the live paper
            # config.  This is the "what would the backtest twin have done"
            # baseline for the reconciler.
            from signal_engine.slippage import (
                compute_slippage, load_per_symbol_calibration,
            )
            asset = symbol[:-4] if symbol.endswith("USDT") else symbol
            try:
                # attempt to read calibration; ignore on failure.
                try:
                    cal = load_per_symbol_calibration(
                        Path("data/execution_calibration.json"))
                except Exception:
                    cal = {}
                daily_vol = float(
                    (await self._exchange.fetch_ticker(symbol))
                    .get("quote_volume_24h", 0.0) or 0.0)
                slip = compute_slippage(
                    asset=asset, position_notional=notional,
                    daily_volume=daily_vol, per_symbol_cal=cal,
                    vip_tier=("VIP2", self._vip_maker_bps, self._vip_taker_bps),
                )
                if side in ("Buy", "open_long", "close_short"):
                    fill_price = mid * (1.0 + slip)
                else:
                    fill_price = mid * (1.0 - slip)
                slip_bps = float(slip) * 10_000
                method = "paper_calibrated"
            except Exception as e:
                logger.warning("TestnetExec paper fallback: %s — using flat 2bps",
                               e)
                slip_bps = 2.0
                if side in ("Buy", "open_long", "close_short"):
                    fill_price = mid * 1.0002
                else:
                    fill_price = mid * 0.9998
        elif self._maker_wait_s <= 0:
            # Pure taker — MARKET order.
            try:
                resp = await self._exchange.place_market_order(
                    symbol, side, qty, reduce_only=(intent == "close"))
                if isinstance(resp, dict):
                    exchange_order_id = str(resp.get("orderId", ""))
                    # Try to read avgPrice from fills if returned
                    fills = resp.get("fills") or []
                    if fills:
                        weighted = sum(
                            float(f.get("price", 0)) * float(f.get("qty", 0))
                            for f in fills)
                        total_qty = sum(float(f.get("qty", 0)) for f in fills)
                        if total_qty > 0:
                            fill_price = weighted / total_qty
            except Exception as e:
                logger.error("TestnetExec taker order failed: %s", e)
                return None
            fee_bps = self._vip_taker_bps
            method = "taker"
            slip_bps = abs(fill_price - mid) / mid * 10_000
        else:
            # Maker attempt — post-only LIMIT, wait, fall back to taker.
            method = "taker"
            slip_bps = 0.0
            for retry in range(self._max_retries + 1):
                if side in ("Buy", "open_long", "close_short"):
                    limit_price = bid
                else:
                    limit_price = ask
                try:
                    resp = await self._exchange.place_limit_order(
                        symbol, side, qty, limit_price,
                        post_only=True, reduce_only=(intent == "close"))
                    if not isinstance(resp, dict) or "orderId" not in resp:
                        logger.warning(
                            "TestnetExec: maker order no orderId (resp=%s); "
                            "falling back to taker", resp)
                        break
                    oid = str(resp.get("orderId", ""))
                    exchange_order_id = oid
                    start = time.monotonic()
                    filled = False
                    while time.monotonic() - start < self._maker_wait_s:
                        await asyncio.sleep(self._poll_interval_s)
                        try:
                            st = await self._exchange.get_order_status(symbol, oid)
                        except Exception as e:
                            logger.warning("order_status failed: %s", e)
                            continue
                        status = (st or {}).get("status", "")
                        if status == "FILLED":
                            try:
                                fill_price = float(
                                    (st.get("fills") or [{}])[0].get("price", limit_price))
                            except Exception:
                                fill_price = limit_price
                            method = "maker"
                            fee_bps = self._vip_maker_bps
                            slip_bps = abs(fill_price - mid) / mid * 10_000
                            filled = True
                            break
                        if status in ("CANCELED", "EXPIRED", "REJECTED"):
                            break
                    if filled:
                        break
                    # Cancel and retry
                    try:
                        await self._exchange.cancel_order(symbol, oid)
                    except Exception:
                        pass
                except Exception as e:
                    logger.warning(
                        "TestnetExec: maker attempt %d failed: %s", retry, e)
                    continue

            if method != "maker":
                # Taker fallback.
                try:
                    resp = await self._exchange.place_market_order(
                        symbol, side, qty, reduce_only=(intent == "close"))
                    if isinstance(resp, dict):
                        exchange_order_id = str(resp.get("orderId", ""))
                        fills = resp.get("fills") or []
                        if fills:
                            weighted = sum(
                                float(f.get("price", 0)) * float(f.get("qty", 0))
                                for f in fills)
                            total_qty = sum(
                                float(f.get("qty", 0)) for f in fills)
                            if total_qty > 0:
                                fill_price = weighted / total_qty
                except Exception as e:
                    logger.error("TestnetExec taker fallback failed: %s", e)
                    return None
                fee_bps = self._vip_taker_bps
                method = "taker"
                slip_bps = abs(fill_price - mid) / mid * 10_000

        # Record.
        fee_usd = notional * (fee_bps / 10_000)
        # NEW (B-032): normalize side to lowercase before intent classification.
        # The previous string-OR missed uppercase BUY/SELL, mis-tagging them
        # as 'short' when they were actually long (and vice versa).
        _sn = (side or "").lower() if isinstance(side, str) else side
        _is_long = _sn in ("buy", "open_long", "close_short")
        rec = FillRecord(
            symbol=symbol, side=side,
            intent_side=f"{intent}_{'long' if _is_long else 'short'}",
            qty=qty, decision_mid=mid, fill_price=fill_price,
            method=method, order_id=order_id,
            exchange_order_id=exchange_order_id,
            slippage_bps=slip_bps, fee_bps=fee_bps, fee_usd=fee_usd,
            notional_usd=notional, strategy_id=strategy_id,
        )
        self._fills.append(rec)
        if self._fills_path:
            self.export_fills(self._fills_path)

        return TradeResult(
            filled=True,
            fill_price=fill_price,
            fill_qty=qty,
            method=method,
            order_id=order_id,
            exchange_order_id=exchange_order_id,
            decision_mid=mid,
        )
