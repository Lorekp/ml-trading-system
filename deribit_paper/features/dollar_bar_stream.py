"""Live dollar-bar stream from Binance USD-M Futures aggTrades REST polling.

F3 Tier 2 (parity-cycle round 3 — 2026-05-07; corrected cycle-31
2026-05-13): drives a ``DollarBarAggregator`` from the Binance USD-M
Futures REST endpoint (``https://fapi.binance.com/fapi/v1/aggTrades``).

**Venue history** (the class name ``DollarBarSpotStream`` is retained
for back-compat after cycle-31 corrected the venue):

* Original F3/F6/F8 (2026-05-07): pointed the stream at Binance SPOT
  (``api.binance.com/api/v3/aggTrades``). The accompanying docstring
  claimed the offline pipeline trained on Binance Vision SPOT aggTrades.
* Cycle-31 (2026-05-13) parity audit: discovered the canonical offline
  downloader (``scripts/download_binance_aggtrades.sh:10``) actually
  pulls from ``data.binance.vision/data/futures/um/monthly/aggTrades``
  i.e. Binance USD-M FUTURES. The F6/F8 claim was descriptively wrong.
* Cycle-31 flipped the engine's REST client wiring from
  ``BinanceRESTClient.for_spot()`` back to the default (futures), so
  live consumes the same venue the models were trained on.

The class name + the engine-side buffer name (``spot_trade_buffer``)
are retained as historical artefacts; functionally the stream and
buffer now hold Binance USD-M Futures trades. A future cleanup can
rename both to ``DollarBarBinanceStream`` / ``binance_futures_trade_buffer``
without behaviour change.

Round-4 F6/F8 fan-out hook (2026-05-07, semantics preserved
cycle-31): the same poller optionally fans every parsed trade out to
a second consumer via ``extra_trade_sink`` so the live TSRV
computation can read from the same venue the offline pipeline trains
on (closing the Deribit-perp-vs-Binance-futures venue gap on
TSRV-derived features).

The architecture mirrors :class:`LiveVPINStream._run_rest_loop`:
periodic ``GET /fapi/v1/aggTrades?fromId=last+1`` polling, feeding
each trade into the aggregator, with thread-safe locks on the
latest-bar property and an explicit shutdown event for graceful stop.

Key differences vs the VPIN stream:

* Drives a :class:`DollarBarAggregator`, not a bucket-VPIN accumulator.
* Same venue as VPIN now (Binance USD-M Futures) since cycle-31;
  prior to that they were different (VPIN=futures, DollarBar=spot).
* Each trade's ``dollar_volume = qty * price``.

The stream itself does NOT keep a history of emitted bars — that's
the consumer's job (the engine wires a :class:`DollarBarBuffer` via
the aggregator's ``on_bar_close`` callback).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time
from typing import Callable

import pandas as pd

from deribit_paper.client.binance_rest import (
    BinanceRESTClient,
    BinanceRESTError,
)
from deribit_paper.features.dollar_bar import DollarBar, DollarBarAggregator
from deribit_paper.market_data.buffer import Trade

# Round-4 F6/F8: callback signature for trade fan-out.  Receives a
# canonical :class:`Trade` for each Binance Spot aggTrade — used by
# the engine to feed ``spot_trade_buffer`` for live TSRV computation.
ExtraTradeSink = Callable[[Trade], None]

logger = logging.getLogger(__name__)


class DollarBarSpotStream:
    """REST-polling Binance Spot aggTrades feed driving a dollar-bar aggregator.

    Usage::

        rest = BinanceRESTClient.for_spot()
        await rest.connect()
        agg = DollarBarAggregator(threshold=500_000.0, on_bar_close=buffer.append)
        stream = DollarBarSpotStream(rest, agg, symbol="BTCUSDT", poll_seconds=3.0)
        task = asyncio.create_task(stream.run_forever())
        ...
        await stream.stop()
        await task
        await rest.close()

    Properties (thread-safe / lock-free read):

    * ``latest_bar`` — the most-recently-emitted :class:`DollarBar`,
      or ``None`` if the stream hasn't closed a bar yet.
    * ``n_bars_received`` — count of bars closed since construction.
    * ``is_stale`` — ``True`` after ``max_stale_seconds`` of silence
      (and after ``max_stale_seconds`` since construction, mirroring
      :class:`LiveVPINStream` semantics so a never-connected feed
      eventually trips the operator alert).
    * ``is_connected`` — flips ``True`` after the first successful
      poll page is fetched, regardless of whether it contained any
      trades.
    """

    def __init__(
        self,
        rest_client: BinanceRESTClient,
        aggregator: DollarBarAggregator,
        *,
        symbol: str = "BTCUSDT",
        poll_seconds: float = 3.0,
        max_stale_seconds: float = 60.0,
        extra_trade_sink: ExtraTradeSink | None = None,
    ) -> None:
        if poll_seconds <= 0:
            raise ValueError(f"poll_seconds must be positive, got {poll_seconds}")
        if max_stale_seconds <= 0:
            raise ValueError(
                f"max_stale_seconds must be positive, got {max_stale_seconds}"
            )
        self._rest = rest_client
        self._agg = aggregator
        self._symbol = symbol
        self._poll_seconds = float(poll_seconds)
        self._max_stale_seconds = float(max_stale_seconds)
        # Round-4 F6/F8: optional fan-out to a second consumer (typically
        # ``spot_trade_buffer.append`` on the engine).  Default ``None``
        # preserves pre-round-4 behaviour byte-identically — the existing
        # 846 deribit_paper tests don't need to know about this.
        self._extra_trade_sink: ExtraTradeSink | None = extra_trade_sink

        self._lock = threading.Lock()
        self._shutdown = asyncio.Event()
        self._connected = False
        self._latest_bar: DollarBar | None = None
        self._n_bars_received = 0
        self._last_agg_trade_id: int | None = None
        self._last_update_ts: float | None = None
        self._stream_start_ts: float = time.time()
        self._trades_processed = 0

        # Hook into the aggregator: every bar close updates our state.
        # We compose with whatever callback the caller already wired
        # (typically ``DollarBarBuffer.append``) so the consumer's
        # callback still fires.
        self._user_on_bar_close = aggregator._on_bar_close
        aggregator._on_bar_close = self._on_bar_close

    @property
    def latest_bar(self) -> DollarBar | None:
        with self._lock:
            return self._latest_bar

    @property
    def n_bars_received(self) -> int:
        with self._lock:
            return self._n_bars_received

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def is_stale(self) -> bool:
        """``True`` when the stream is past its staleness threshold.

        Two failure modes:

        1. Stream WAS receiving and went silent — last_update_ts is
           set but ``time.time() - last_update_ts > max_stale_seconds``.
        2. Stream NEVER received any trades AND has been running long
           enough that the silence is no longer warmup
           (``time.time() - stream_start_ts > max_stale_seconds``).

        Mirrors :class:`LiveVPINStream.is_stale` so operators see
        consistent semantics across feeds.
        """
        now = time.time()
        if self._last_update_ts is None:
            return (now - self._stream_start_ts) > self._max_stale_seconds
        return (now - self._last_update_ts) > self._max_stale_seconds

    @property
    def trades_processed(self) -> int:
        return self._trades_processed

    def _on_bar_close(self, bar: DollarBar) -> None:
        """Aggregator callback — updates buffer + fires user callback."""
        with self._lock:
            self._latest_bar = bar
            self._n_bars_received += 1
        if self._user_on_bar_close is not None:
            try:
                self._user_on_bar_close(bar)
            except Exception:  # noqa: BLE001 - callback failure must not stop the stream
                logger.exception(
                    "DollarBarSpotStream user on_bar_close raised",
                )

    async def run_forever(self) -> None:
        """Poll Binance Spot aggTrades until ``stop()`` is called.

        Each iteration:

        * fetches one page of aggTrades from ``last_seen + 1``;
        * feeds each trade through the aggregator
          (``timestamp = pd.Timestamp(t['T'], unit='ms', tz='UTC')``,
          ``price = float(t['p'])``, ``dollar_volume = qty * price``,
          ``is_buyer_maker = bool(t['m'])``);
        * sleeps ``poll_seconds`` (with shutdown-awareness).

        Connection-recovery is delegated to ``BinanceRESTClient`` —
        the client retries 429/5xx with exponential backoff. We catch
        ``BinanceRESTError`` and continue polling on the next tick,
        but flip ``is_connected=False`` so a downstream watchdog can
        notice. ``CancelledError`` propagates so the task can be
        cancelled cleanly by the engine.
        """
        logger.info(
            "DollarBarSpotStream starting: symbol=%s poll=%.1fs threshold=%s",
            self._symbol, self._poll_seconds, self._agg.threshold,
        )
        while not self._shutdown.is_set():
            try:
                trades = await self._rest.fetch_agg_trades(
                    self._symbol,
                    from_id=(
                        self._last_agg_trade_id + 1
                        if self._last_agg_trade_id is not None else None
                    ),
                )
            except asyncio.CancelledError:
                self._connected = False
                raise
            except BinanceRESTError as exc:
                self._connected = False
                logger.warning(
                    "DollarBarSpotStream: REST fetch failed (%s) — retry on next tick",
                    exc,
                )
                trades = []
            except Exception:
                self._connected = False
                logger.exception(
                    "DollarBarSpotStream: unexpected error during poll",
                )
                trades = []
            else:
                # A successful fetch (even with empty list) flips connection
                # health green. The accumulator/aggregator state advances
                # only when trades are present.
                self._connected = True

            if trades:
                self._process_page(trades)

            # Sleep with shutdown-awareness so stop() returns promptly.
            try:
                await asyncio.wait_for(
                    self._shutdown.wait(), timeout=self._poll_seconds,
                )
                return  # shutdown signaled during sleep
            except asyncio.TimeoutError:
                pass  # poll interval elapsed, loop

        logger.info(
            "DollarBarSpotStream stopped: trades=%d bars=%d",
            self._trades_processed, self._n_bars_received,
        )

    def _process_page(self, trades: list[dict]) -> None:
        """Feed one REST page of trades through the aggregator.

        Defensive parsing: malformed rows (missing keys / non-numeric
        fields) are silently skipped rather than raising, matching the
        :class:`LiveVPINStream._process_rest_page` pattern. Tracks the
        max ``aggTradeId`` seen so the next ``fetch_agg_trades`` call
        passes ``from_id = max + 1``.

        Round-4 F6/F8: when ``extra_trade_sink`` is wired, each parsed
        trade is also constructed into a canonical :class:`Trade` and
        forwarded — typically to ``spot_trade_buffer.append`` for live
        TSRV computation. Sink failures are logged and swallowed so a
        broken downstream consumer cannot stop the bar pipeline.
        """
        max_id = self._last_agg_trade_id
        for tr in trades:
            try:
                ts_ms = int(tr["T"])
                price = float(tr["p"])
                qty = float(tr["q"])
                is_buyer_maker = bool(tr["m"])
                agg_id = int(tr["a"])
            except (KeyError, ValueError, TypeError):
                continue  # silently skip malformed rows; defensive
            ts = pd.Timestamp(ts_ms, unit="ms", tz="UTC")
            dollar_volume = qty * price
            self._agg.add_trade(
                timestamp=ts,
                price=price,
                dollar_volume=dollar_volume,
                is_buyer_maker=is_buyer_maker,
            )
            # Round-4 F6/F8 fan-out — feed the same trade into a second
            # consumer (e.g., ``spot_trade_buffer`` for live TSRV).
            # ``is_buyer_maker == True`` ⇒ taker is the SELLER ⇒
            # direction = -1 (matches Deribit's perp ``direction`` sign
            # convention so ``compute_bar_tsrv`` is venue-agnostic).
            if self._extra_trade_sink is not None:
                try:
                    self._extra_trade_sink(
                        Trade(
                            trade_id=f"binance:{agg_id}",
                            timestamp=ts,
                            instrument_name=self._symbol,
                            price=price,
                            amount=qty,
                            direction=-1 if is_buyer_maker else 1,
                        )
                    )
                except Exception:  # noqa: BLE001 — sink failure must not stop the stream
                    logger.exception(
                        "DollarBarSpotStream extra_trade_sink raised",
                    )
            self._trades_processed += 1
            self._last_update_ts = time.time()
            if max_id is None or agg_id > max_id:
                max_id = agg_id
        if max_id is not None:
            self._last_agg_trade_id = max_id

    def stop(self) -> None:
        """Signal graceful shutdown — ``run_forever`` returns on the next tick."""
        self._shutdown.set()
        self._connected = False


__all__ = ["DollarBarSpotStream"]
