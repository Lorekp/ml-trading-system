"""Rolling in-memory buffers for live market data.

Three buffer types:

- :class:`TradeBuffer` — rolling window of option or perp trades, keyed
  by ``timestamp``.  Supports append (O(1)), slice-by-window (O(N)
  where N is the number of trades in the window), and time-based
  eviction (O(1) amortised when called periodically).
- :class:`IndexPriceBuffer` — rolling window of ``(timestamp, price)``
  tuples for the BTC index.  Used to attach the close-of-bar spot
  reference to each bar.

Key invariants:

1. **Append is fast**: ``deque.append`` is O(1).
2. **Dedup on replay**: after a WS disconnect + resubscribe, Deribit
   may re-send trades we've already seen.  Each trade carries a
   ``trade_id``; we keep a bounded set of recent IDs and drop
   duplicates.  The dedup set is bounded so it can't grow forever.
3. **Eviction is lazy**: we only trim the head when ``evict_older()`` is
   called (typically on each bar tick).  This avoids per-append scans.
4. **Thread safety**: asyncio is single-threaded; no locks needed.
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


# Maximum number of trade_ids to keep in the dedup set.  Larger = more
# protection against reorder on reconnect; smaller = less memory.
# 50k is ample — it covers ~several hours of BTC option flow.
_DEDUP_RECENT_MAX = 50_000


@dataclass
class Trade:
    """Canonical trade record from Deribit.

    Mirrors the fields we care about from the ``trades.*`` channels.
    ``direction`` is ``+1`` for buys (taker bought), ``-1`` for sells.
    Instrument-specific fields like ``strike``, ``option_type``,
    ``expiry_dt`` are populated for option trades only.
    """
    trade_id: str
    timestamp: pd.Timestamp          # UTC
    instrument_name: str
    price: float                      # USD per BTC for perps; vol / USD per contract for options
    amount: float                     # base units (BTC)
    direction: int                    # +1 (buy) / -1 (sell)
    # Option-specific fields — None for perps
    strike: float | None = None
    option_type: str | None = None    # "C" or "P"
    expiry_dt: pd.Timestamp | None = None
    iv: float | None = None           # % (as Deribit reports)
    index_price: float | None = None
    mark_price: float | None = None


class TradeBuffer:
    """Rolling in-memory buffer of trades with time-based eviction + dedup.

    Two retention strategies are supported:

    - ``window_seconds`` — retain trades with timestamp newer than
      ``(now - window_seconds)``.  The buffer is trimmed lazily on
      ``evict_older()``.
    - ``max_len`` — hard cap; oldest trades are dropped when the cap
      is exceeded.

    Both can be active simultaneously.  The ``window_seconds`` value is
    the primary retention, ``max_len`` is a safety backstop to prevent
    unbounded growth if ``evict_older()`` is never called.
    """

    def __init__(
        self,
        *,
        name: str,
        window_seconds: float,
        max_len: int | None = None,
    ):
        self.name = name
        self.window_seconds = window_seconds
        self._trades: deque[Trade] = deque(maxlen=max_len)
        self._seen_ids: deque[str] = deque(maxlen=_DEDUP_RECENT_MAX)
        self._seen_set: set[str] = set()

    def __len__(self) -> int:
        return len(self._trades)

    def append(self, trade: Trade) -> bool:
        """Append a trade if not a duplicate.  Returns True if added."""
        if trade.trade_id in self._seen_set:
            return False

        # Evict oldest seen ID if the dedup deque is full
        if len(self._seen_ids) == self._seen_ids.maxlen:
            oldest = self._seen_ids[0]
            self._seen_set.discard(oldest)

        self._seen_ids.append(trade.trade_id)
        self._seen_set.add(trade.trade_id)
        self._trades.append(trade)
        return True

    def extend(self, trades: Iterable[Trade]) -> int:
        """Append many trades.  Returns the number actually added."""
        added = 0
        for t in trades:
            if self.append(t):
                added += 1
        return added

    def evict_older(self, now: pd.Timestamp) -> int:
        """Drop trades older than ``now - window_seconds``.

        Returns the number of evicted trades.  Call this periodically
        (on bar close is ideal) to keep memory bounded.
        """
        cutoff = now - pd.Timedelta(seconds=self.window_seconds)
        evicted = 0
        while self._trades and self._trades[0].timestamp < cutoff:
            self._trades.popleft()
            evicted += 1
        if evicted:
            logger.debug(
                "%s evicted %d trades older than %s (len now %d)",
                self.name, evicted, cutoff, len(self._trades),
            )
        return evicted

    def slice_window(
        self, start: pd.Timestamp, end: pd.Timestamp,
    ) -> list[Trade]:
        """Return trades with ``start <= timestamp < end``.

        Half-open interval to match the backtester's bar convention
        (a bar at ``T`` covers ``[T - 15min, T)``).
        """
        return [t for t in self._trades if start <= t.timestamp < end]

    def to_dataframe(self) -> pd.DataFrame:
        """Convert the entire buffer to a pandas DataFrame.

        Column order matches ``vol_arb/pipeline.py`` conventions so the
        live feature builder can pass the result straight into the batch
        pipeline's functions.
        """
        if not self._trades:
            return pd.DataFrame(columns=[
                "timestamp", "trade_id", "instrument_name", "price", "amount",
                "direction", "strike", "option_type", "expiry_dt", "iv",
                "index_price", "mark_price",
            ])
        rows = [
            {
                "timestamp": t.timestamp,
                "trade_id": t.trade_id,
                "instrument_name": t.instrument_name,
                "price": t.price,
                "amount": t.amount,
                "direction": t.direction,
                "strike": t.strike,
                "option_type": t.option_type,
                "expiry_dt": t.expiry_dt,
                "iv": t.iv,
                "index_price": t.index_price,
                "mark_price": t.mark_price,
            }
            for t in self._trades
        ]
        return pd.DataFrame(rows)

    def prewarm_from_parquet(
        self,
        path: str | "pathlib.Path",   # noqa: F821 — string forward-ref
        *,
        instrument_name: str = "BTCUSDT",
        trade_id_prefix: str = "warmstart",
        max_age_seconds: float | None = None,
        now: pd.Timestamp | None = None,
    ) -> int:
        """Bulk-load trades from a sidecar parquet at engine start.

        Round-4 F6/F8 (parity-cycle): the warm-start path for the live
        Binance Spot trade buffer that drives :func:`compute_bar_tsrv`.
        Mirrors :meth:`DollarBarBuffer.prewarm_from_parquet` (F3 Tier 4):
        load fast, fail loud on missing files, no silent partial loads.

        The parquet must carry, at minimum, ``timestamp``, ``price`` and
        ``amount`` columns (a Binance Vision aggTrades dump or the
        `scripts/warmstart_spot_trade_buffer.py` output both qualify).
        ``direction`` is read from ``is_buyer_maker`` if present (Binance
        Vision schema: ``+1`` when ``is_buyer_maker`` is False, ``-1``
        when True), otherwise defaults to ``+1``.

        ``trade_id`` is read from a ``trade_id`` or ``a`` (aggTradeId)
        column when present; otherwise we synthesise a stable id of the
        form ``f"{trade_id_prefix}:{i}"`` so dedup still works.

        ``max_age_seconds`` (when set) drops rows older than
        ``now - max_age_seconds``. Default ``None`` keeps every row.

        Returns the number of trades appended.
        """
        from pathlib import Path as _Path
        path = _Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"TradeBuffer prewarm: sidecar not found at {path}. "
                f"Run `scripts/warmstart_spot_trade_buffer.py` to generate it."
            )

        df = pd.read_parquet(path)
        if df.empty:
            logger.warning(
                "%s prewarm: parquet at %s is empty — buffer remains empty",
                self.name, path,
            )
            return 0

        required = {"timestamp", "price", "amount"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"TradeBuffer prewarm parquet at {path} missing columns: {missing}"
            )

        # Coerce timestamp to UTC pd.Timestamp; tolerate ms-int or pd.Timestamp.
        ts = df["timestamp"]
        if pd.api.types.is_integer_dtype(ts):
            ts = pd.to_datetime(ts, unit="ms", utc=True)
        elif pd.api.types.is_datetime64_any_dtype(ts):
            if getattr(ts.dtype, "tz", None) is None:
                ts = ts.dt.tz_localize("UTC")
        else:
            ts = pd.to_datetime(ts, utc=True)

        if max_age_seconds is not None:
            ref = now or pd.Timestamp.now(tz="UTC")
            cutoff = ref - pd.Timedelta(seconds=max_age_seconds)
            keep = ts >= cutoff
            df = df.loc[keep].reset_index(drop=True)
            ts = ts.loc[keep].reset_index(drop=True)

        prices = df["price"].to_numpy(dtype=float)
        amounts = df["amount"].to_numpy(dtype=float)

        # direction: prefer explicit column, fall back to is_buyer_maker
        if "direction" in df.columns:
            directions = df["direction"].to_numpy(dtype=int)
        elif "is_buyer_maker" in df.columns:
            directions = (~df["is_buyer_maker"].astype(bool)).astype(int) * 2 - 1
            directions = directions.to_numpy(dtype=int)
        else:
            directions = [1] * len(df)

        # trade_id: prefer explicit column, fall back to aggTradeId, then synth
        if "trade_id" in df.columns:
            ids = df["trade_id"].astype(str).tolist()
        elif "a" in df.columns:
            ids = [f"{trade_id_prefix}:{int(a)}" for a in df["a"].tolist()]
        else:
            ids = [f"{trade_id_prefix}:{i}" for i in range(len(df))]

        added = 0
        ts_list = ts.tolist()
        for i in range(len(df)):
            trade = Trade(
                trade_id=ids[i],
                timestamp=ts_list[i],
                instrument_name=instrument_name,
                price=float(prices[i]),
                amount=float(amounts[i]),
                direction=int(directions[i]),
            )
            if self.append(trade):
                added += 1
        logger.info(
            "TradeBuffer[%s] prewarmed: %d/%d rows from %s "
            "(window=%ds, dedup_pool=%d)",
            self.name, added, len(df), path,
            int(self.window_seconds), len(self._seen_set),
        )
        return added

    def clear(self) -> None:
        self._trades.clear()
        self._seen_ids.clear()
        self._seen_set.clear()


class IndexPriceBuffer:
    """Rolling buffer of ``(timestamp, index_price)`` pairs.

    Much lighter-weight than :class:`TradeBuffer` because there's no
    dedup requirement — index prices are published as a continuous
    stream, every sample is unique by timestamp.
    """

    def __init__(
        self,
        *,
        name: str = "index_price",
        window_seconds: float = 3600.0,
        max_len: int | None = 10_000,
    ):
        self.name = name
        self.window_seconds = window_seconds
        self._samples: deque[tuple[pd.Timestamp, float]] = deque(maxlen=max_len)

    def __len__(self) -> int:
        return len(self._samples)

    def append(self, timestamp: pd.Timestamp, price: float) -> None:
        self._samples.append((timestamp, price))

    def evict_older(self, now: pd.Timestamp) -> int:
        cutoff = now - pd.Timedelta(seconds=self.window_seconds)
        evicted = 0
        while self._samples and self._samples[0][0] < cutoff:
            self._samples.popleft()
            evicted += 1
        return evicted

    def latest(self) -> float | None:
        """Return the most recent index price, or ``None`` if empty.

        Note: this method does NOT check freshness — the returned value
        could be hours old if the index feed is silent. Use
        :meth:`latest_with_age` or :meth:`latest_fresh` when freshness
        matters (e.g., live hedging path). Kept for back-compat with
        callers that don't care about staleness (e.g., off-line
        aggregation).
        """
        if not self._samples:
            return None
        return self._samples[-1][1]

    def latest_with_age(
        self, now: pd.Timestamp | None = None,
    ) -> tuple[float | None, float | None]:
        """Return ``(price, age_seconds)`` for the most recent sample.

        ``age_seconds`` is None iff there are no samples. Lets the
        caller make the staleness decision per use-site.
        """
        if not self._samples:
            return (None, None)
        ts, price = self._samples[-1]
        ref = now or pd.Timestamp.now(tz="UTC")
        age = (ref - ts).total_seconds()
        return (price, age)

    def latest_fresh(
        self,
        max_age_seconds: float = 60.0,
        now: pd.Timestamp | None = None,
    ) -> float | None:
        """Return the most recent index price IFF younger than
        ``max_age_seconds``, else None.

        P1-4 (2026-05-07): the original ``latest()`` had no staleness
        check — a hedger calling it after the index feed went silent
        would receive a price from minutes/hours ago and trade against
        a stale delta. This wrapper makes the staleness contract
        explicit at the call site.
        """
        price, age = self.latest_with_age(now=now)
        if price is None or age is None:
            return None
        if age > max_age_seconds:
            return None
        return price

    def at_or_before(self, ts: pd.Timestamp) -> float | None:
        """Return the latest index price with ``timestamp <= ts``.

        Used to pin the close-of-bar index price when aggregating.
        Linear scan from the tail, so O(N) worst case but typical
        call site is the freshest-wins case → O(1).
        """
        # Walk from most recent backwards
        for t, p in reversed(self._samples):
            if t <= ts:
                return p
        return None

    def clear(self) -> None:
        self._samples.clear()
