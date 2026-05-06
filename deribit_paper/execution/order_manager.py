"""Async order manager: place, track, cancel, poll fills.

Thin wrapper over :class:`DeribitRESTClient` that adds:

- **Idempotent sends**: every order carries a client-generated
  ``label`` (client_order_id) which survives retries without double-
  placing the order on Deribit's side.
- **Automatic retry** on transient failures (rate limit, 5xx) via the
  underlying REST client's retry logic.  The order manager adds a
  layer of its own retries for rejections that ARE retryable (e.g.,
  ``too_many_requests``, ``market_impact``).
- **Fill polling**: ``wait_for_fill`` polls ``get_order_state`` on a
  configurable cadence until the order state becomes ``filled``,
  ``cancelled``, or the timeout expires.
- **DRY_RUN honoring**: the underlying REST client's ``place_order``
  returns stub fills when DRY_RUN is on, so ``wait_for_fill`` returns
  instantly with a synthetic result.

The order manager does NOT decide WHAT to trade — that's the job of
:class:`StraddleFlow` (straddle_flow.py).  It just submits orders
reliably.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from deribit_paper.client.errors import (
    DeribitError,
    DeribitOrderRejected,
    DeribitRateLimitError,
)
from deribit_paper.client.rest import DeribitRESTClient
from deribit_paper.execution.amounts import (
    OPTION_MIN_CONTRACT_BTC,
    option_amount_btc,
)
from deribit_paper.execution.idempotency import (
    IdempotencyGuard,
    new_client_order_id,
)
from deribit_paper.risk.dry_run import get_dry_run

logger = logging.getLogger(__name__)


# Terminal order states — once reached, we stop polling
_TERMINAL_STATES: frozenset[str] = frozenset({
    "filled", "cancelled", "rejected", "expired", "untriggered",
})


def _merge_tier_results(a: "OrderResult", b: "OrderResult") -> "OrderResult":
    """Aggregate two tiered IOC results (filled_amount weighted average).

    Returns a fresh :class:`OrderResult` whose ``client_order_id``,
    ``order_id`` and ``state`` come from ``b`` (the latest tier);
    ``filled_amount`` is the sum, ``average_price`` is the size-weighted
    mean.

    Latency contract (E.2.5 2026-05-01): per-tier ``submit_to_ack_ms``
    and ``ack_to_terminal_ms`` reflect the LAST tier's call (``b``).
    The aggregate ``total_ms`` is set to ``None`` here — the wall-clock
    total is stamped by ``_place_tiered_taker`` from a single
    ``perf_counter_ns`` captured before tier-1 starts. Pre-E.2.5 this
    summed across tiers and a nested merge double-counted, producing
    misleading kill-switch numbers.
    """
    total_filled = a.filled_amount + b.filled_amount
    if total_filled > 0:
        avg_px = (
            a.average_price * a.filled_amount + b.average_price * b.filled_amount
        ) / total_filled
    else:
        # Pre-E.2.7: returned 0.0 here, propagating a fake fill price
        # downstream. NaN is the unambiguous "no fill" signal that
        # downstream consumers (HarnessWriter._avg_price, P&L recipe)
        # already handle via NaN checks.
        avg_px = float("nan")
    return OrderResult(
        client_order_id=b.client_order_id,
        order_id=b.order_id,
        instrument_name=b.instrument_name,
        side=b.side,
        amount_requested=a.amount_requested,         # original request
        filled_amount=total_filled,
        average_price=avg_px,
        state="filled" if total_filled >= a.amount_requested - 1e-9 else b.state,
        is_dry_run=a.is_dry_run or b.is_dry_run,
        error=b.error if b.state not in {"filled"} else None,
        raw=b.raw,
        submit_to_ack_ms=b.submit_to_ack_ms,
        ack_to_terminal_ms=b.ack_to_terminal_ms,
        total_ms=None,                  # set by _place_tiered_taker post-merge
    )


@dataclass
class OrderResult:
    """Final state of a submitted order.

    Returned by :meth:`OrderManager.place_and_wait`.  Even in DRY_RUN
    mode the result is populated with the stub fill so downstream
    logic can proceed without branching.
    """
    client_order_id: str
    order_id: str                       # Deribit's order ID (stub_* in dry-run)
    instrument_name: str
    side: str                           # "buy" / "sell"
    amount_requested: float
    filled_amount: float
    average_price: float
    state: str                          # "filled", "cancelled", etc.
    is_dry_run: bool
    error: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)
    # ── Latency telemetry (V2 wiring, I3 2026-04-26) ──
    submit_to_ack_ms: float | None = None   # time from .place_order() call → ack received
    ack_to_terminal_ms: float | None = None  # time from ack → state in _TERMINAL_STATES
    total_ms: float | None = None            # wall-clock from submit → terminal

    @property
    def is_filled(self) -> bool:
        return self.state == "filled" and self.filled_amount > 0.0

    @property
    def is_terminal(self) -> bool:
        return self.state in _TERMINAL_STATES


class OrderManager:
    """Submits orders + polls for fills.  Wraps a :class:`DeribitRESTClient`."""

    def __init__(
        self,
        *,
        rest: DeribitRESTClient,
        idempotency_guard: IdempotencyGuard | None = None,
        fill_poll_seconds: float = 1.5,
        max_rejection_retries: int = 2,
        retry_backoff_seconds: float = 2.0,
        latency_threshold_ms: float = float("inf"),
        on_latency_breach: "Callable[[float, float, str], None] | None" = None,
    ):
        self._rest = rest
        self._guard = idempotency_guard or IdempotencyGuard()
        self._fill_poll_seconds = fill_poll_seconds
        self._max_rejection_retries = max_rejection_retries
        self._retry_backoff_seconds = retry_backoff_seconds
        # E.2.8: optional latency-breach callback. Engine wires this to
        # ``KillSwitch.on_latency_breach``; OrderManager calls it with
        # ``(total_ms, threshold_ms, instrument_name)`` after each order
        # whose ``total_ms`` exceeds the threshold.
        self._latency_threshold_ms = float(latency_threshold_ms)
        self._on_latency_breach = on_latency_breach

    async def place_and_wait(
        self,
        *,
        instrument_name: str,
        side: str,
        amount: float,
        order_type: str = "market",
        price: float | None = None,
        label: str | None = None,
        timeout_seconds: float = 30.0,
        post_only: bool | None = None,
        reduce_only: bool | None = None,
        time_in_force: str | None = None,
    ) -> OrderResult:
        """Place an order and wait for it to reach a terminal state.

        Generates a client_order_id (carried in Deribit's ``label``
        field for idempotency), submits with retry, then polls until
        filled/cancelled/timeout.

        In DRY_RUN mode, the underlying REST client returns a stub
        fill synchronously, so this method effectively returns
        immediately with the stub result.
        """
        if label is None:
            label = new_client_order_id()

        # ── Latency telemetry start (V2 wiring, I3 2026-04-26) ──
        # Capture wall-clock submit-to-ack and total latencies on every
        # order. Used by the kill-switch latency threshold (E.2.8
        # 2026-05-01): when ``total_ms > cfg.kill_switch_latency_ms``,
        # ``self._on_latency_breach`` calls ``KillSwitch.on_latency_breach``
        # which feeds the consecutive-error counter. Testnet ships
        # ``kill_switch_latency_ms=inf`` (effectively disabled because
        # testnet latency varies wildly); mainnet operators set ~3000ms.
        _t0_submit_ns = time.perf_counter_ns()

        # Idempotency check — don't double-send the same label
        if self._guard.was_sent(label):
            logger.warning(
                "OrderManager.place_and_wait: label %s already sent (skipping)",
                label,
            )
            return OrderResult(
                client_order_id=label,
                order_id="",
                instrument_name=instrument_name,
                side=side,
                amount_requested=amount,
                filled_amount=0.0,
                average_price=0.0,
                state="rejected",
                is_dry_run=get_dry_run(),
                error="duplicate client_order_id",
            )

        submit_result: dict[str, Any] | None = None
        attempt = 0
        last_error: str | None = None
        while attempt <= self._max_rejection_retries:
            attempt += 1
            # E.3.2 (2026-05-01): re-stamp ``_t0_submit_ns`` BEFORE
            # each retry attempt so ``submit_to_ack_ms`` reflects only
            # the latest attempt's network round-trip — not the sum of
            # all retries plus their backoff sleeps. Pre-E.3.2 a
            # rejected order with 2× retries inflated the latency
            # number by ~4-6 seconds of sleep, drowning the real
            # network signal under retry-delay noise.
            _t0_submit_ns = time.perf_counter_ns()
            try:
                submit_result = await self._rest.place_order(
                    instrument_name=instrument_name,
                    side=side,
                    amount=amount,
                    type=order_type,
                    price=price,
                    label=label,
                    post_only=post_only,
                    reduce_only=reduce_only,
                    time_in_force=time_in_force,
                )
                break
            except DeribitRateLimitError as exc:
                last_error = str(exc)
                logger.warning(
                    "Order %s rate-limited (attempt %d/%d): %s",
                    label, attempt, self._max_rejection_retries + 1, exc,
                )
                if attempt <= self._max_rejection_retries:
                    await asyncio.sleep(self._retry_backoff_seconds * attempt)
                    continue
                break
            except DeribitOrderRejected as exc:
                last_error = str(exc)
                if exc.retryable and attempt <= self._max_rejection_retries:
                    logger.warning(
                        "Order %s rejected (retryable, attempt %d/%d): %s",
                        label, attempt, self._max_rejection_retries + 1, exc,
                    )
                    await asyncio.sleep(self._retry_backoff_seconds * attempt)
                    continue
                logger.error("Order %s rejected: %s", label, exc)
                break
            except DeribitError as exc:
                last_error = str(exc)
                logger.error("Order %s error: %s", label, exc)
                break

        if submit_result is None:
            return OrderResult(
                client_order_id=label,
                order_id="",
                instrument_name=instrument_name,
                side=side,
                amount_requested=amount,
                filled_amount=0.0,
                average_price=0.0,
                state="rejected",
                is_dry_run=get_dry_run(),
                error=last_error or "unknown submit error",
            )

        self._guard.mark_sent(label)
        _t1_ack_ns = time.perf_counter_ns()
        _submit_to_ack_ms = (_t1_ack_ns - _t0_submit_ns) / 1e6

        is_dry_run = bool(submit_result.get("_stub", False))
        order = submit_result.get("order", {}) or {}
        order_id = str(order.get("order_id", ""))
        state = str(order.get("order_state", "open"))
        filled_amount = float(order.get("filled_amount", 0.0) or 0.0)
        average_price = float(order.get("average_price", 0.0) or 0.0)

        if is_dry_run:
            # Dry-run stubs are already filled — short-circuit
            _total_ms = (time.perf_counter_ns() - _t0_submit_ns) / 1e6
            return OrderResult(
                client_order_id=label,
                order_id=order_id,
                instrument_name=instrument_name,
                side=side,
                amount_requested=amount,
                filled_amount=filled_amount,
                average_price=average_price,
                state=state,
                is_dry_run=True,
                raw=submit_result,
                submit_to_ack_ms=_submit_to_ack_ms,
                ack_to_terminal_ms=0.0,
                total_ms=_total_ms,
            )

        # Poll for terminal state
        result = await self._poll_until_terminal(
            order_id=order_id,
            client_order_id=label,
            instrument_name=instrument_name,
            side=side,
            amount_requested=amount,
            timeout_seconds=timeout_seconds,
        )
        # Stamp the latency telemetry onto the live result (the
        # _poll_until_terminal path can't see the original submit time).
        _t2_terminal_ns = time.perf_counter_ns()
        result.submit_to_ack_ms = _submit_to_ack_ms
        result.ack_to_terminal_ms = (_t2_terminal_ns - _t1_ack_ns) / 1e6
        result.total_ms = (_t2_terminal_ns - _t0_submit_ns) / 1e6
        if result.submit_to_ack_ms > 2000.0:
            logger.warning(
                "High submit-to-ack latency on %s: %.0fms — investigate "
                "Deribit congestion / network",
                label, result.submit_to_ack_ms,
            )
        # E.2.8: latency-breach callback (engine wires to KillSwitch).
        if (self._on_latency_breach is not None
                and result.total_ms is not None
                and result.total_ms > self._latency_threshold_ms):
            try:
                self._on_latency_breach(
                    result.total_ms, self._latency_threshold_ms, instrument_name,
                )
            except Exception:                  # noqa: BLE001 - callback non-critical
                logger.exception("on_latency_breach callback raised")
        return result

    # ── Smart taker execution ─────────────────────────────────

    async def place_ioc_taker(
        self,
        *,
        instrument_name: str,
        side: str,
        amount: float,
        max_slippage_pct: float = 0.5,
        label: str | None = None,
        reduce_only: bool | None = None,
        tiered: bool = False,
    ) -> OrderResult:
        """Taker order with IOC price protection.

        Fetches current best bid/ask, sets a limit price at
        best + max_slippage_pct, and sends as IOC (immediate-or-cancel).
        If the book is empty beyond the cap, partial fill is returned
        instead of a catastrophic price.

        When ``tiered=True`` (V2 wiring, I1 2026-04-26) escalates through
        a 3-tier ladder: tier-1 IOC at touch+1tick, tier-2 IOC at the
        full ``max_slippage_pct`` cap, tier-3 market on residual size.
        Each tier only fires for the *unfilled* portion of the order so
        early-tier fills aren't double-counted. Latency is captured per
        tier via the underlying ``place_and_wait`` telemetry (I3); the
        outermost result reflects the LAST tier that ran.
        """
        if tiered:
            return await self._place_tiered_taker(
                instrument_name=instrument_name,
                side=side,
                amount=amount,
                max_slippage_pct=max_slippage_pct,
                label=label,
                reduce_only=reduce_only,
            )
        if label is None:
            label = new_client_order_id()

        try:
            ticker = await self._rest.get_ticker(instrument_name)
        except Exception as exc:
            logger.warning(
                "IOC taker: get_ticker(%s) failed (%s) -- falling back to market",
                instrument_name, exc,
            )
            return await self.place_and_wait(
                instrument_name=instrument_name,
                side=side, amount=amount,
                order_type="market",
                label=label, reduce_only=reduce_only,
                timeout_seconds=30.0,
            )

        # Deribit option tick size is 0.0005 BTC
        tick = 0.0005

        if side == "buy":
            best = float(ticker.get("best_ask_price", 0.0) or 0.0)
            cap = best * (1.0 + max_slippage_pct / 100.0)
            cap = round(cap / tick) * tick  # round to tick
        else:
            best = float(ticker.get("best_bid_price", 0.0) or 0.0)
            cap = best * (1.0 - max_slippage_pct / 100.0)
            cap = round(cap / tick) * tick  # round to tick

        if best <= 0 or cap <= 0:
            logger.warning(
                "IOC taker: no valid quote for %s -- falling back to market",
                instrument_name,
            )
            return await self.place_and_wait(
                instrument_name=instrument_name,
                side=side, amount=amount,
                order_type="market",
                label=label, reduce_only=reduce_only,
                timeout_seconds=30.0,
            )

        spread = abs(float(ticker.get("best_ask_price", 0) or 0) -
                      float(ticker.get("best_bid_price", 0) or 0))
        mid = (float(ticker.get("best_ask_price", 0) or 0) +
               float(ticker.get("best_bid_price", 0) or 0)) / 2
        spread_pct = spread / mid * 100 if mid > 0 else 0

        logger.info(
            "IOC taker: %s %s %.4f @ cap %.6f (best=%.6f, spread=%.2f%%)",
            side, instrument_name, amount, cap, best, spread_pct,
        )

        return await self.place_and_wait(
            instrument_name=instrument_name,
            side=side, amount=amount,
            order_type="limit", price=cap,
            time_in_force="immediate_or_cancel",
            label=label, reduce_only=reduce_only,
            timeout_seconds=10.0,
        )

    async def _place_tiered_taker(
        self,
        *,
        instrument_name: str,
        side: str,
        amount: float,
        max_slippage_pct: float,
        label: str | None,
        reduce_only: bool | None,
    ) -> OrderResult:
        """3-tier IOC taker ladder. See :meth:`place_ioc_taker` ``tiered=True``.

        Tier 1: IOC at touch + 1 tick (cheapest, highest miss rate on testnet).
        Tier 2: IOC at touch ± ``max_slippage_pct``/100 (the C4 sidecar cap).
        Tier 3: market on whatever didn't fill in tiers 1+2.

        Each tier is a fresh ``place_and_wait`` call so the idempotency
        guard sees a unique label per tier. Returns the LAST tier's
        OrderResult with ``filled_amount`` aggregated across tiers.
        """
        if label is None:
            label = new_client_order_id()

        # E.2.5: capture wall-clock start so the merged total_ms reflects
        # actual elapsed time across all tiers, not the sum of per-tier
        # spans (which double-counted on nested merges).
        _ladder_t0_ns = time.perf_counter_ns()

        def _stamp_total(result: OrderResult) -> OrderResult:
            result.total_ms = (time.perf_counter_ns() - _ladder_t0_ns) / 1e6
            return result

        # Lookup the current touch + spread for tier-1 / tier-2 caps.
        try:
            ticker = await self._rest.get_ticker(instrument_name)
        except Exception as exc:
            logger.warning(
                "Tiered IOC: get_ticker(%s) failed (%s) -- jumping to market",
                instrument_name, exc,
            )
            return _stamp_total(await self.place_and_wait(
                instrument_name=instrument_name,
                side=side, amount=amount,
                order_type="market",
                label=label, reduce_only=reduce_only,
                timeout_seconds=30.0,
            ))

        tick = 0.0005   # Deribit option tick size in BTC.
        if side == "buy":
            best = float(ticker.get("best_ask_price", 0.0) or 0.0)
            tier1_cap = round((best + tick) / tick) * tick
            tier2_cap = round((best * (1.0 + max_slippage_pct / 100.0)) / tick) * tick
        else:
            best = float(ticker.get("best_bid_price", 0.0) or 0.0)
            tier1_cap = round((best - tick) / tick) * tick
            tier2_cap = round((best * (1.0 - max_slippage_pct / 100.0)) / tick) * tick

        if best <= 0:
            logger.warning(
                "Tiered IOC: no valid quote for %s -- jumping to market",
                instrument_name,
            )
            return _stamp_total(await self.place_and_wait(
                instrument_name=instrument_name,
                side=side, amount=amount,
                order_type="market",
                label=label, reduce_only=reduce_only,
                timeout_seconds=30.0,
            ))

        # ── Tier 1 ──
        logger.info(
            "Tiered IOC %s %s tier-1 @ %.6f (best=%.6f, amount=%.4f)",
            side, instrument_name, tier1_cap, best, amount,
        )
        r1 = await self.place_and_wait(
            instrument_name=instrument_name,
            side=side, amount=amount,
            order_type="limit", price=tier1_cap,
            time_in_force="immediate_or_cancel",
            label=f"{label}_t1", reduce_only=reduce_only,
            timeout_seconds=10.0,
        )
        residual_raw = amount - r1.filled_amount
        # H3 (2026-05-06): round residual to Deribit's 0.1-BTC option grid
        # before submitting tier-2. Without rounding, a partial tier-1 fill
        # left a non-grid residual (e.g. 0.57 BTC after tier-1 filled 0.43);
        # Deribit rejects with min_size/increment error, the tier-2 attempt
        # counts as a DeribitOrderRejected, and 3 of those in a row trips
        # the kill-switch on a moving market. option_amount_btc returns
        # 0.0 if abs(residual) < OPTION_MIN_CONTRACT_BTC.
        residual = option_amount_btc(residual_raw)
        if abs(residual_raw - residual) > 1e-9:
            logger.debug(
                "Tiered IOC %s %s tier-2 residual rounded: %.6f -> %.4f "
                "(0.1-BTC grid)",
                side, instrument_name, residual_raw, residual,
            )
        if residual < OPTION_MIN_CONTRACT_BTC:
            # Either tier-1 fully filled (residual_raw <= 1e-9 → rounded to
            # 0), or the partial leftover is below the minimum contract
            # size (e.g. 0.05 BTC). Either way we cannot legally submit
            # tier-2; treat as fully filled-enough.
            return _stamp_total(r1)

        # ── Tier 2 ──
        logger.info(
            "Tiered IOC %s %s tier-2 @ %.6f (residual=%.4f after tier-1)",
            side, instrument_name, tier2_cap, residual,
        )
        r2 = await self.place_and_wait(
            instrument_name=instrument_name,
            side=side, amount=residual,
            order_type="limit", price=tier2_cap,
            time_in_force="immediate_or_cancel",
            label=f"{label}_t2", reduce_only=reduce_only,
            timeout_seconds=10.0,
        )
        residual2_raw = residual - r2.filled_amount
        # H3: round tier-3 residual to grid as well.
        residual2 = option_amount_btc(residual2_raw)
        if abs(residual2_raw - residual2) > 1e-9:
            logger.debug(
                "Tiered IOC %s %s tier-3 residual rounded: %.6f -> %.4f",
                side, instrument_name, residual2_raw, residual2,
            )
        if residual2 < OPTION_MIN_CONTRACT_BTC:
            # Aggregate filled_amount + average_price across tiers 1+2.
            return _stamp_total(_merge_tier_results(r1, r2))

        # ── Tier 3 (bounded IOC, never naked market — E.1.7) ──
        # Pre-E.1.7 this dropped to a naked market order, defeating the
        # max_slippage_pct boundary the rest of the ladder enforces — a
        # manipulated thin book could fill the residual at any price.
        # Now: place a wider IOC at tier-2's cap × 1.5; if THAT misses,
        # alert the operator and skip the residual rather than ship a
        # fill at any price. Tail residual <10% of original is dropped
        # silently because residual is already small relative to
        # acceptable slippage.
        if residual2 < 0.1 * amount:
            logger.warning(
                "Tiered IOC %s %s tier-3 SKIP (residual=%.4f < 10%% of "
                "%.4f — tail too small to risk market fill)",
                side, instrument_name, residual2, amount,
            )
            return _stamp_total(_merge_tier_results(r1, r2))

        if side == "buy":
            tier3_cap = round((best * (1.0 + 1.5 * max_slippage_pct / 100.0)) / tick) * tick
        else:
            tier3_cap = round((best * (1.0 - 1.5 * max_slippage_pct / 100.0)) / tick) * tick
        logger.info(
            "Tiered IOC %s %s tier-3 IOC @ %.6f (residual=%.4f after "
            "tier-2, cap = best ±1.5×%.2f%%)",
            side, instrument_name, tier3_cap, residual2, max_slippage_pct,
        )
        r3 = await self.place_and_wait(
            instrument_name=instrument_name,
            side=side, amount=residual2,
            order_type="limit", price=tier3_cap,
            time_in_force="immediate_or_cancel",
            label=f"{label}_t3", reduce_only=reduce_only,
            timeout_seconds=10.0,
        )
        if not r3.is_filled and r3.filled_amount < residual2 - 1e-9:
            # Tier-3 still not filled at the wider cap. Don't fall back
            # to naked market — log + return what we have. Operator
            # alerting happens upstream via the OrderResult.state.
            logger.error(
                "Tiered IOC %s %s tier-3 INCOMPLETE: residual=%.4f, "
                "filled=%.4f at cap %.6f — skipping market fallback "
                "to defend against manipulated book",
                side, instrument_name, residual2, r3.filled_amount,
                tier3_cap,
            )
        return _stamp_total(_merge_tier_results(_merge_tier_results(r1, r2), r3))

    async def place_chunked_taker(
        self,
        *,
        instrument_name: str,
        side: str,
        amount: float,
        chunk_size: float = 1.0,
        delay_seconds: float = 2.0,
        max_slippage_pct: float = 0.5,
        label: str | None = None,
        reduce_only: bool | None = None,
    ) -> OrderResult:
        """Split large orders into chunks with IOC price protection."""
        if amount <= chunk_size:
            return await self.place_ioc_taker(
                instrument_name=instrument_name,
                side=side, amount=amount,
                max_slippage_pct=max_slippage_pct,
                label=label, reduce_only=reduce_only,
            )

        if label is None:
            label = new_client_order_id()

        remaining = amount
        total_filled = 0.0
        total_value = 0.0
        chunk_num = 0

        while remaining > 0.01:
            chunk = min(remaining, chunk_size)
            chunk_num += 1
            result = await self.place_ioc_taker(
                instrument_name=instrument_name,
                side=side, amount=chunk,
                max_slippage_pct=max_slippage_pct,
                label=f"{label}_c{chunk_num}",
                reduce_only=reduce_only,
            )
            total_filled += result.filled_amount
            total_value += result.filled_amount * result.average_price
            remaining -= chunk
            if remaining > 0.01 and result.filled_amount > 0:
                await asyncio.sleep(delay_seconds)
            elif result.filled_amount == 0:
                logger.warning(
                    "Chunk %d of %s got zero fill -- stopping early",
                    chunk_num, instrument_name,
                )
                break

        avg_price = total_value / total_filled if total_filled > 0 else 0.0
        return OrderResult(
            client_order_id=label,
            order_id=result.order_id if result else "",
            instrument_name=instrument_name,
            side=side,
            amount_requested=amount,
            filled_amount=total_filled,
            average_price=avg_price,
            state="filled" if total_filled > 0 else "cancelled",
            is_dry_run=get_dry_run(),
            raw={"chunks": chunk_num, "total_filled": total_filled},
        )

    async def cancel(
        self, order_id: str, *, max_retries: int = 3,
    ) -> dict[str, Any]:
        """Cancel an order by its Deribit order_id, with retry (M1)."""
        for attempt in range(1, max_retries + 1):
            try:
                return await self._rest.cancel_order(order_id)
            except DeribitError as exc:
                logger.warning(
                    "cancel(%s) attempt %d/%d failed: %s",
                    order_id, attempt, max_retries, exc,
                )
                if attempt < max_retries:
                    await asyncio.sleep(1.0 * attempt)
        return {"order_id": order_id, "order_state": "error", "error": "cancel failed after retries"}

    async def get_order_state(self, order_id: str) -> dict[str, Any]:
        """Fetch the current state of an order via REST."""
        return await self._rest._request(
            "GET",
            "/private/get_order_state",
            params={"order_id": order_id},
            authenticated=True,
        )

    async def _poll_until_terminal(
        self,
        *,
        order_id: str,
        client_order_id: str,
        instrument_name: str,
        side: str,
        amount_requested: float,
        timeout_seconds: float,
    ) -> OrderResult:
        """Poll the order state until it becomes terminal or timeout expires."""
        import time as _time
        deadline = _time.monotonic() + timeout_seconds
        last_raw: dict[str, Any] = {}
        while True:
            try:
                last_raw = await self.get_order_state(order_id)
            except DeribitError as exc:
                logger.warning("get_order_state(%s) failed: %s", order_id, exc)
                await asyncio.sleep(self._fill_poll_seconds)
                if _time.monotonic() >= deadline:
                    return OrderResult(
                        client_order_id=client_order_id,
                        order_id=order_id,
                        instrument_name=instrument_name,
                        side=side,
                        amount_requested=amount_requested,
                        filled_amount=0.0,
                        average_price=0.0,
                        state="unknown",
                        is_dry_run=False,
                        error=f"polling failed: {exc}",
                    )
                continue

            state = str(last_raw.get("order_state", "open"))
            if state in _TERMINAL_STATES:
                return OrderResult(
                    client_order_id=client_order_id,
                    order_id=order_id,
                    instrument_name=instrument_name,
                    side=side,
                    amount_requested=amount_requested,
                    filled_amount=float(last_raw.get("filled_amount", 0.0) or 0.0),
                    average_price=float(last_raw.get("average_price", 0.0) or 0.0),
                    state=state,
                    is_dry_run=False,
                    raw=last_raw,
                )

            if _time.monotonic() >= deadline:
                logger.warning(
                    "Order %s timed out after %.1fs in state %s — cancelling",
                    order_id, timeout_seconds, state,
                )
                await self.cancel(order_id)

                # C6: Check if the order actually filled despite non-terminal state.
                # Query order state one final time after cancel to get latest fill info.
                final_filled = float(last_raw.get("filled_amount", 0.0) or 0.0)
                final_price = float(last_raw.get("average_price", 0.0) or 0.0)
                try:
                    final_raw = await self.get_order_state(order_id)
                    final_filled = float(final_raw.get("filled_amount", 0.0) or 0.0)
                    final_price = float(final_raw.get("average_price", 0.0) or 0.0)
                    final_state = str(final_raw.get("order_state", "cancelled"))
                    if final_state == "filled" or final_filled >= amount_requested:
                        logger.warning(
                            "Order %s was actually filled (state=%s, filled=%.4f) "
                            "despite earlier non-terminal state — returning as filled",
                            order_id, final_state, final_filled,
                        )
                        return OrderResult(
                            client_order_id=client_order_id,
                            order_id=order_id,
                            instrument_name=instrument_name,
                            side=side,
                            amount_requested=amount_requested,
                            filled_amount=final_filled,
                            average_price=final_price,
                            state="filled",
                            is_dry_run=False,
                            raw=final_raw,
                        )
                except Exception:
                    pass  # Best-effort; fall through to cancelled result

                return OrderResult(
                    client_order_id=client_order_id,
                    order_id=order_id,
                    instrument_name=instrument_name,
                    side=side,
                    amount_requested=amount_requested,
                    filled_amount=final_filled,
                    average_price=final_price,
                    state="cancelled",
                    is_dry_run=False,
                    error=f"timeout after {timeout_seconds:.1f}s",
                    raw=last_raw,
                )

            await asyncio.sleep(self._fill_poll_seconds)

    # ── MkrTkr60 execution model ─────────────────────────────────

    async def place_maker_then_taker(
        self,
        *,
        instrument_name: str,
        side: str,
        amount: float,
        maker_wait_seconds: float = 60.0,
        label: str | None = None,
        reduce_only: bool | None = None,
    ) -> OrderResult:
        """MkrTkr60: post a limit order, wait for maker fill, fallback to taker.

        1. Fetch current best bid/ask via ``get_ticker()``
        2. Post a limit order joining the passive side (post_only=True)
        3. Poll for up to ``maker_wait_seconds``
        4. If filled: return (maker fill, zero exchange fee on options)
        5. If not filled / partially filled: cancel, send market for remainder

        Perp hedges should NOT use this — they stay market for latency.
        """
        if label is None:
            label = new_client_order_id()

        # 1. Fetch current quote for pricing
        try:
            ticker = await self._rest.get_ticker(instrument_name)
        except Exception as exc:
            logger.warning(
                "MkrTkr60: get_ticker(%s) failed (%s) -- falling back to market",
                instrument_name, exc,
            )
            return await self.place_and_wait(
                instrument_name=instrument_name,
                side=side,
                amount=amount,
                order_type="market",
                label=f"{label}_tkr",
                reduce_only=reduce_only,
                timeout_seconds=30.0,
            )

        # 2. Compute limit price: join the passive side
        if side == "buy":
            limit_price = float(ticker.get("best_bid_price", 0.0) or 0.0)
        else:
            limit_price = float(ticker.get("best_ask_price", 0.0) or 0.0)
        if limit_price <= 0:
            limit_price = float(ticker.get("mark_price", 0.0) or 0.0)
        if limit_price <= 0:
            logger.warning(
                "MkrTkr60: no valid price for %s -- falling back to market",
                instrument_name,
            )
            return await self.place_and_wait(
                instrument_name=instrument_name,
                side=side,
                amount=amount,
                order_type="market",
                label=f"{label}_tkr",
                reduce_only=reduce_only,
                timeout_seconds=30.0,
            )

        logger.info(
            "MkrTkr60: %s %s %.4f @ limit %.4f (wait %ds)",
            side, instrument_name, amount, limit_price, int(maker_wait_seconds),
        )

        # 3. Place limit order (post_only to guarantee maker)
        maker_result = await self.place_and_wait(
            instrument_name=instrument_name,
            side=side,
            amount=amount,
            order_type="limit",
            price=limit_price,
            label=f"{label}_mkr",
            post_only=True,
            reduce_only=reduce_only,
            timeout_seconds=maker_wait_seconds,
        )

        if maker_result.is_filled:
            logger.info(
                "MkrTkr60: MAKER fill %s @ %.4f (%.4f filled)",
                instrument_name, maker_result.average_price, maker_result.filled_amount,
            )
            if maker_result.raw is None:
                maker_result.raw = {}
            maker_result.raw["execution_type"] = "maker"
            return maker_result

        # 4. Not fully filled -- compute remainder
        filled_qty = maker_result.filled_amount
        remaining = amount - filled_qty

        if filled_qty > 0:
            logger.info(
                "MkrTkr60: partial maker fill %.4f / %.4f -- taker for remainder",
                filled_qty, amount,
            )
        else:
            logger.info(
                "MkrTkr60: maker timeout for %s -- taker fallback", instrument_name,
            )

        if remaining <= 0:
            if maker_result.raw is None:
                maker_result.raw = {}
            maker_result.raw["execution_type"] = "maker"
            return maker_result

        # 5. Taker fallback
        taker_result = await self.place_and_wait(
            instrument_name=instrument_name,
            side=side,
            amount=remaining,
            order_type="market",
            label=f"{label}_tkr",
            reduce_only=reduce_only,
            timeout_seconds=30.0,
        )

        # 6. Combine maker + taker into a single result
        total_filled = filled_qty + taker_result.filled_amount
        if total_filled > 0:
            maker_value = filled_qty * maker_result.average_price
            taker_value = taker_result.filled_amount * taker_result.average_price
            avg_price = (maker_value + taker_value) / total_filled
        else:
            avg_price = taker_result.average_price

        exec_type = "taker"
        if filled_qty >= amount:
            exec_type = "maker"
        elif filled_qty > 0:
            exec_type = "partial_maker"

        combined_raw = dict(taker_result.raw or {})
        combined_raw["execution_type"] = exec_type
        combined_raw["maker_filled"] = filled_qty
        combined_raw["taker_filled"] = taker_result.filled_amount

        return OrderResult(
            client_order_id=label,
            order_id=taker_result.order_id,
            instrument_name=instrument_name,
            side=side,
            amount_requested=amount,
            filled_amount=total_filled,
            average_price=avg_price,
            state="filled" if total_filled > 0 else taker_result.state,
            is_dry_run=taker_result.is_dry_run,
            raw=combined_raw,
        )
