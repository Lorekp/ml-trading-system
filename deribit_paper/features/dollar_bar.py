"""Live dollar-bar reconstruction from streaming aggTrades.

F3 (parity-cycle round 3 — 2026-05-07): the offline feature pipeline
in ``vol_model/features/engineer.py`` builds dollar bars from Binance
spot aggTrades (one bar per ``$threshold`` of accumulated dollar
volume), then computes lag/EMA on the dollar-bar grid. The live engine
mirrors that here so the 24 dollar-bar-derived features can byte-identity
match offline.

The per-bar feature definitions and the accumulation kernel are byte-
identical to:

* ``TSRV/sampling.py::_accumulate_bars`` — numba kernel that emits a
  bar index every time cumulative dollar volume crosses ``threshold``,
  resetting cumulative to 0 after each emission.
* ``vol_model/features/engineer.py::_build_bars_sequential`` (lines
  77-99) — per-bar feature extraction.

Streaming semantics:

* trades arrive one at a time via ``add_trade``;
* an in-progress accumulator tracks open/high/low/close/buy_volume/
  sell_volume/cumulative_dollar/start_ts and a list of prices for VoV;
* when cumulative >= threshold, a :class:`DollarBar` is emitted via
  ``on_bar_close`` callback, the accumulator resets, and the cycle
  continues.

Key numerical conventions (mirrored from offline):

* ``vov = std(diff(log(prices)), ddof=1)`` if ``n_trades > 2`` else
  ``0.0``. Note the ``ddof=1`` — sample std, not population std.
* ``duration_s`` = ``(last_trade_ts - first_trade_ts) / timedelta(s)``.
* Bar timestamp = last trade's timestamp.
* ``buy_volume`` = sum of dollar_volume where ``~is_buyer_maker``
  (taker bought — aggressor was BUYER).
* ``sell_volume`` = sum where ``is_buyer_maker`` (aggressor was SELLER).
* ``imbalance`` = ``(buy - sell) / max(dollar_volume, 1.0)`` — emitted
  by the bar so downstream consumers don't need to recompute.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)


# Default BTC dollar-bar threshold ($500K) matches
# ``vol_model/config.py:22`` for BTCUSDT.
_DEFAULT_BTC_DOLLAR_BAR_THRESHOLD: float = 500_000.0


@dataclass(frozen=True)
class DollarBar:
    """One completed dollar bar.

    All fields mirror the columns produced by
    ``vol_model/features/engineer.py::_build_bars_sequential`` so a
    DataFrame of these is byte-identity-comparable to the offline
    ``BTCUSDT_bar_series.parquet``.

    ``log_return`` is set by the aggregator if a previous bar's close
    is available (``log(close / prev_close)``); ``NaN`` for the first
    bar. Mirrors the line ``bars["log_return"]`` in
    ``_build_bars_sequential`` after concat.
    """
    timestamp: pd.Timestamp
    open: float
    close: float
    high: float
    low: float
    duration_s: float
    n_trades: int
    dollar_volume: float
    buy_volume: float
    sell_volume: float
    vov: float
    imbalance: float
    log_return: float
    rv_bar: float


class DollarBarAggregator:
    """Stateful streaming dollar-bar accumulator.

    Usage::

        agg = DollarBarAggregator(
            threshold=500_000.0,
            on_bar_close=lambda bar: my_buffer.append(bar),
        )
        for trade in stream:
            agg.add_trade(
                timestamp=trade.timestamp,
                price=trade.price,
                dollar_volume=trade.dollar_volume,
                is_buyer_maker=trade.is_buyer_maker,
            )

    The aggregator does NOT keep a list of all emitted bars (that's the
    consumer's job — for the live engine, a `DollarBarBuffer` keeps the
    last ~300 bars for EMA50 + parkinson_1d 250-bar lookback).
    """

    def __init__(
        self,
        *,
        threshold: float = _DEFAULT_BTC_DOLLAR_BAR_THRESHOLD,
        on_bar_close: Callable[[DollarBar], None] | None = None,
    ) -> None:
        if threshold <= 0:
            raise ValueError(f"threshold must be positive, got {threshold}")
        self._threshold = float(threshold)
        self._on_bar_close = on_bar_close
        self._reset_accumulator()
        # Kernel cumulative — independent of the accumulator. Resets to
        # 0 at each kernel boundary (matches ``TSRV/sampling._accumulate_bars``).
        # The accumulator continues across boundaries because the boundary
        # trade is added to the next bar's accumulator (offline ``starts =
        # bar_idx[:-1]`` slicing). The kernel cumulative is what we use to
        # detect boundaries; the accumulator is what we emit.
        self._cum_kernel: float = 0.0
        self._prev_close: float | None = None
        self._n_bars_emitted: int = 0

    @property
    def threshold(self) -> float:
        return self._threshold

    @property
    def n_bars_emitted(self) -> int:
        return self._n_bars_emitted

    @property
    def cumulative_dollar(self) -> float:
        """Dollar volume accumulated in the kernel-cumulative (boundary detector).

        Resets to 0 immediately after each kernel boundary. NOT the same as
        the in-progress bar's dollar_volume — the bar includes the previous
        boundary trade as its first trade, but the kernel cumulative does
        not (matches ``TSRV/sampling._accumulate_bars`` semantics).
        """
        return self._cum_kernel

    def _reset_accumulator(self) -> None:
        self._first_ts: pd.Timestamp | None = None
        self._last_ts: pd.Timestamp | None = None
        self._open: float | None = None
        self._high: float = -math.inf
        self._low: float = math.inf
        self._buy_volume: float = 0.0
        self._sell_volume: float = 0.0
        self._n_trades: int = 0
        self._prices: list[float] = []

    def add_trade(
        self,
        *,
        timestamp: pd.Timestamp,
        price: float,
        dollar_volume: float,
        is_buyer_maker: bool,
    ) -> DollarBar | None:
        """Push one trade into the accumulator.

        Returns the emitted :class:`DollarBar` if this trade closed the
        in-progress bar; otherwise returns ``None``. Also fires the
        ``on_bar_close`` callback when a bar closes.

        Offline-convention quirk (verified empirically against
        ``TSRV/sampling._accumulate_bars`` + ``_build_bars_sequential``):

        The kernel maintains an internal cumulative that resets to 0
        every time it crosses the threshold. The trade that crosses is
        the kernel "boundary trade" — the kernel emits its index, then
        resets cumulative.

        The offline ``_build_bars_sequential`` then slices the trades
        with ``starts = [0] + bar_idx[:-1]`` and ``ends = bar_idx``, so:

        * bar 0 = trades ``[0, kernel_boundary[0])`` — excludes the
          boundary trade.
        * bar K (K>=1) = trades ``[kernel_boundary[K-1], kernel_boundary[K])``
          — INCLUDES the previous boundary trade as its first trade,
          excludes the current boundary trade.

        So bar 0 has dollar_volume ``< threshold`` (no previous boundary
        trade to seed it), while bars 1+ each have dollar_volume
        ``= threshold`` exactly (one trade in, one trade out, net zero).
        Live MUST replicate this for byte-identity to the offline parquet.

        Implementation: track the kernel cumulative (resets at boundary)
        SEPARATELY from the accumulator (which carries the boundary trade
        forward into the next bar).
        """
        if not (price > 0 and dollar_volume >= 0):
            # Defensive: reject malformed data without raising. The live
            # WS feed occasionally emits zero-quantity prints.
            return None

        new_cum_kernel = self._cum_kernel + float(dollar_volume)

        if new_cum_kernel >= self._threshold:
            # Kernel boundary detected. The current trade is the boundary
            # trade. Per the offline slicing, this trade is NOT in the
            # bar we're about to emit — it becomes the FIRST trade of
            # the NEXT bar.
            emitted: DollarBar | None = None
            if self._n_trades > 0:
                emitted = self._emit_bar()
            # Edge case: ``self._n_trades == 0`` means the accumulator
            # is empty when a single oversized trade hits the threshold
            # alone. Offline skips this bar (``e <= s: continue``); we
            # match by not emitting. The trade still seeds the next
            # accumulator below.
            self._cum_kernel = 0.0
            # Add this boundary trade as the first trade of the new bar.
            # The accumulator was reset by ``_emit_bar`` (or was already
            # empty in the edge case).
            self._add_to_accumulator(
                timestamp, float(price), float(dollar_volume), is_buyer_maker,
            )
            return emitted

        # Normal accumulation — no boundary.
        self._cum_kernel = new_cum_kernel
        self._add_to_accumulator(
            timestamp, float(price), float(dollar_volume), is_buyer_maker,
        )
        return None

    def _add_to_accumulator(
        self,
        timestamp: pd.Timestamp,
        price: float,
        dollar_volume: float,
        is_buyer_maker: bool,
    ) -> None:
        if self._first_ts is None:
            self._first_ts = timestamp
            self._open = price
        self._last_ts = timestamp
        if price > self._high:
            self._high = price
        if price < self._low:
            self._low = price
        if is_buyer_maker:
            self._sell_volume += dollar_volume
        else:
            self._buy_volume += dollar_volume
        self._n_trades += 1
        self._prices.append(price)

    def _emit_bar(self) -> DollarBar:
        """Compute the per-bar features, fire the callback, reset accumulator.

        Mirrors ``_build_bars_sequential`` lines 87-99 + 124-126 exactly.
        Note: does NOT reset ``self._cum_kernel`` — the caller does that
        because the kernel boundary semantics are split between the
        cumulative (resets) and the accumulator (carries the boundary
        trade forward).
        """
        if self._n_trades > 2:
            # std(diff(log(prices)), ddof=1) — Python loop matches numpy
            # numerics on the same float64 inputs to within ULP precision.
            log_prices = [math.log(p) for p in self._prices]
            log_returns = [
                log_prices[i] - log_prices[i - 1]
                for i in range(1, len(log_prices))
            ]
            mean_lr = sum(log_returns) / len(log_returns)
            sq_dev = sum((r - mean_lr) ** 2 for r in log_returns)
            # ddof=1 → divide by n-1
            vov = math.sqrt(sq_dev / (len(log_returns) - 1))
        else:
            vov = 0.0

        dollar_volume = self._buy_volume + self._sell_volume
        # imbalance uses ``max(dollar_volume, 1.0)`` clip per
        # ``_build_bars_sequential`` line 124 (avoids /0 and dwarfs the
        # imbalance to ~0 on tiny bars).
        denom = dollar_volume if dollar_volume >= 1.0 else 1.0
        imbalance = (self._buy_volume - self._sell_volume) / denom

        close = float(self._prices[-1])
        if self._prev_close is not None and self._prev_close > 0:
            log_return = math.log(close / self._prev_close)
        else:
            log_return = float("nan")
        rv_bar = log_return * log_return if math.isfinite(log_return) else float("nan")

        # Cast first/last ts to pandas Timestamp (the caller may have
        # passed numpy datetime64).
        last_ts = pd.Timestamp(self._last_ts)
        first_ts = pd.Timestamp(self._first_ts)
        duration_s = (last_ts - first_ts).total_seconds()

        bar = DollarBar(
            timestamp=last_ts,
            open=float(self._open) if self._open is not None else float("nan"),
            close=close,
            high=float(self._high),
            low=float(self._low),
            duration_s=float(duration_s),
            n_trades=int(self._n_trades),
            dollar_volume=float(dollar_volume),
            buy_volume=float(self._buy_volume),
            sell_volume=float(self._sell_volume),
            vov=float(vov),
            imbalance=float(imbalance),
            log_return=float(log_return),
            rv_bar=float(rv_bar),
        )

        self._prev_close = close
        self._n_bars_emitted += 1
        self._reset_accumulator()
        if self._on_bar_close is not None:
            try:
                self._on_bar_close(bar)
            except Exception:  # noqa: BLE001 - callback failures must not stop the stream
                logger.exception("DollarBarAggregator on_bar_close raised")
        return bar


__all__ = [
    "DollarBar",
    "DollarBarAggregator",
]
