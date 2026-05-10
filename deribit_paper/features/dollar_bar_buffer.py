"""Bounded rolling buffer of recently emitted ``DollarBar`` objects.

F3 Tier 2 (parity-cycle round 3 — 2026-05-07): the live engine needs
the last ~300 dollar bars in memory so feature handlers (EMA50,
parkinson_1d 250-bar window, lag3 returns, …) can compute on a
DataFrame view without re-querying anything else.

Why a bespoke buffer instead of a list:

* ``maxlen`` enforces eviction so memory stays bounded even on long
  uptimes.
* ``as_dataframe()`` exposes the buffer to feature code as a tidy
  DataFrame with the same schema as the offline parquet
  (``vol_model/output/BTCUSDT_bar_series.parquet``) — feature handlers
  can be written once and shared between offline and live without
  branching on data source.
* ``compute_vpin`` runs the offline ``_vpin_rolling`` reference and
  returns the LAST element so the byte-identity to offline is
  preserved.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from deribit_paper.features.dollar_bar import DollarBar
from vol_model.features.microstructure import _vpin_rolling

logger = logging.getLogger(__name__)


_COLUMNS: tuple[str, ...] = (
    "close",
    "open",
    "high",
    "low",
    "duration_s",
    "n_trades",
    "dollar_volume",
    "buy_volume",
    "sell_volume",
    "vov",
    "imbalance",
    "log_return",
    "rv_bar",
)


class DollarBarBuffer:
    """Bounded FIFO of :class:`DollarBar` objects.

    Construct with ``max_size`` (default 300) — older bars are evicted
    automatically when the buffer is full. The 300 cap matches the
    parkinson_1d 250-bar lookback used by the offline feature pipeline
    plus a safety margin so an EMA50 over the last column always has
    a valid window.

    Usage::

        buffer = DollarBarBuffer(max_size=300)
        agg = DollarBarAggregator(threshold=500_000.0, on_bar_close=buffer.append)
        # ... feed trades to agg ...
        df = buffer.as_dataframe()  # last <=300 bars as a DataFrame
        vpin = buffer.compute_vpin(window=50)  # latest VPIN, or None
    """

    def __init__(self, max_size: int = 300) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self._max_size = int(max_size)
        self._bars: deque[DollarBar] = deque(maxlen=self._max_size)

    @property
    def max_size(self) -> int:
        return self._max_size

    def __len__(self) -> int:
        return len(self._bars)

    def __iter__(self) -> Iterable[DollarBar]:
        return iter(self._bars)

    def append(self, bar: DollarBar) -> None:
        """Append a bar; oldest is evicted if buffer is at ``max_size``."""
        self._bars.append(bar)

    def prewarm_from_parquet(self, path: str | Path) -> int:
        """Bulk-load dollar bars from a sidecar parquet (F3 Tier 4).

        The parquet must have the schema produced by
        :meth:`as_dataframe` (also written by
        ``scripts/generate_dollar_bar_cache.py``):

            columns: [close, open, high, low, duration_s, n_trades,
                      dollar_volume, buy_volume, sell_volume,
                      vov, imbalance, log_return, rv_bar]
            index:   UTC DatetimeIndex

        Bars are appended chronologically; ``max_size`` eviction
        applies. Returns the number of bars loaded.

        Fail-soft on missing file (returns 0 + WARN log) so the engine
        can still start without a sidecar — the buffer will fill from
        live trades over ~37 minutes.
        """
        p = Path(path)
        if not p.exists():
            logger.warning(
                "DollarBarBuffer prewarm: sidecar %s does not exist; "
                "buffer will fill from live trades (~37min for 250 bars)",
                p,
            )
            return 0
        try:
            df = pd.read_parquet(p)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "DollarBarBuffer prewarm: failed to read %s (%s); "
                "starting cold",
                p, exc,
            )
            return 0
        required = set(_COLUMNS)
        missing = required - set(df.columns)
        if missing:
            logger.warning(
                "DollarBarBuffer prewarm: %s missing columns %s; starting cold",
                p, missing,
            )
            return 0
        if df.index.tz is None:
            df.index = pd.DatetimeIndex(df.index, tz="UTC")
        n_loaded = 0
        for ts, row in df.iterrows():
            bar = DollarBar(
                timestamp=pd.Timestamp(ts),
                open=float(row["open"]),
                close=float(row["close"]),
                high=float(row["high"]),
                low=float(row["low"]),
                duration_s=float(row["duration_s"]),
                n_trades=int(row["n_trades"]),
                dollar_volume=float(row["dollar_volume"]),
                buy_volume=float(row["buy_volume"]),
                sell_volume=float(row["sell_volume"]),
                vov=float(row["vov"]),
                imbalance=float(row["imbalance"]),
                log_return=float(row["log_return"]),
                rv_bar=float(row["rv_bar"]),
            )
            self._bars.append(bar)
            n_loaded += 1
        logger.info(
            "DollarBarBuffer prewarm: loaded %d bars from %s "
            "(window: %s → %s)",
            n_loaded, p,
            df.index[0] if n_loaded > 0 else "n/a",
            df.index[-1] if n_loaded > 0 else "n/a",
        )
        return n_loaded

    def latest_bar(self) -> DollarBar | None:
        """Return the most recently appended bar, or ``None`` if empty."""
        if not self._bars:
            return None
        return self._bars[-1]

    def timestamp_at(self, idx: int) -> pd.Timestamp | None:
        """S-1d (parity-r3 cycle 7 — 2026-05-08): timestamp of the bar
        at position ``idx``.

        Negative indices supported (``-1`` = latest, ``-2`` = previous,
        etc.) — mirrors Python's list-indexing convention. Returns
        ``None`` when ``idx`` is out of bounds; never raises.

        Used by the live ``time_to_funding_lag*`` handlers to walk
        back through actual sub-15-min dollar-bar timestamps (matches
        offline ``series.shift(n).groupby(window_labels).last()``
        which picks the n-th-previous dollar bar's timestamp within
        the current 15-min window).
        """
        n = len(self._bars)
        if n == 0:
            return None
        if idx < 0:
            idx = n + idx
        if idx < 0 or idx >= n:
            return None
        # deque doesn't support O(1) random access; convert once for
        # callers that may iterate through history. The cache below
        # makes repeat access free within a bar.
        return pd.Timestamp(self._bars[idx].timestamp)

    def as_index(self) -> pd.DatetimeIndex:
        """S-1d (parity-r3 cycle 7 — 2026-05-08): the buffer's
        timestamps as a UTC ``DatetimeIndex``.

        Used by :class:`TSRVDollarBarFeatures` to ffill TSRV scalars
        onto the dollar-bar grid without paying the full
        :meth:`as_dataframe` cost. The index is recomputed per call —
        callers are expected to cache for the bar's compute pass.
        """
        if not self._bars:
            return pd.DatetimeIndex([], tz="UTC", name="timestamp")
        timestamps = [bar.timestamp for bar in self._bars]
        # pandas 3.x raises "Tz-aware datetime.datetime cannot be converted
        # to datetime64 unless utc=True" when constructing DatetimeIndex from
        # a list of tz-aware Timestamps without explicit tz. Route through
        # pd.to_datetime(utc=True) — handles tz-aware (convert) and naive
        # (localize) inputs uniformly under both pandas 2.x and 3.x.
        idx = pd.DatetimeIndex(
            pd.to_datetime(timestamps, utc=True), name="timestamp",
        )
        return idx

    def tail(self, n: int) -> list[DollarBar]:
        """Return the last ``n`` bars as a list (newest at end).

        Returns ``[]`` when ``n <= 0``. Returns up to ``len(self)`` if
        ``n`` exceeds the buffer size — never raises.
        """
        if n <= 0:
            return []
        if n >= len(self._bars):
            return list(self._bars)
        # deque doesn't support negative slicing; islice from offset.
        from itertools import islice
        offset = len(self._bars) - n
        return list(islice(self._bars, offset, None))

    def as_dataframe(self) -> pd.DataFrame:
        """Return the buffer as a tidy DataFrame.

        Columns mirror :class:`DollarBar` field order:
        ``[close, open, high, low, duration_s, n_trades, dollar_volume,
        buy_volume, sell_volume, vov, imbalance, log_return, rv_bar]``.

        Index is a UTC ``DatetimeIndex`` named ``timestamp``. Empty
        buffer → empty DataFrame with the same schema (so downstream
        code can rely on column presence).
        """
        if not self._bars:
            empty_idx = pd.DatetimeIndex([], tz="UTC", name="timestamp")
            return pd.DataFrame(
                {col: pd.Series(dtype=_dtype_for(col)) for col in _COLUMNS},
                index=empty_idx,
            )

        timestamps = [bar.timestamp for bar in self._bars]
        data: dict[str, list] = {col: [] for col in _COLUMNS}
        for bar in self._bars:
            data["close"].append(bar.close)
            data["open"].append(bar.open)
            data["high"].append(bar.high)
            data["low"].append(bar.low)
            data["duration_s"].append(bar.duration_s)
            data["n_trades"].append(bar.n_trades)
            data["dollar_volume"].append(bar.dollar_volume)
            data["buy_volume"].append(bar.buy_volume)
            data["sell_volume"].append(bar.sell_volume)
            data["vov"].append(bar.vov)
            data["imbalance"].append(bar.imbalance)
            data["log_return"].append(bar.log_return)
            data["rv_bar"].append(bar.rv_bar)

        # See as_index() — pandas 3.x requires explicit utc handling when
        # building a DatetimeIndex from tz-aware Timestamps.
        idx = pd.DatetimeIndex(
            pd.to_datetime(timestamps, utc=True), name="timestamp",
        )
        df = pd.DataFrame(data, index=idx)
        # n_trades is integer in the dataclass; pandas infers int64.
        # Force the offline schema's float64 for OHLC etc. in case
        # NaN appeared somewhere (defensive — should not happen).
        for col in (
            "close", "open", "high", "low", "duration_s",
            "dollar_volume", "buy_volume", "sell_volume",
            "vov", "imbalance", "log_return", "rv_bar",
        ):
            df[col] = df[col].astype(np.float64)
        df["n_trades"] = df["n_trades"].astype(np.int64)
        return df

    def compute_vpin(self, window: int = 50) -> float | None:
        """Return the latest rolling VPIN over the last ``window`` bars.

        Calls the offline reference
        ``vol_model/features/microstructure._vpin_rolling`` on the
        buffer's ``buy_volume`` / ``sell_volume`` arrays and returns
        the last element. Returns ``None`` if the buffer has fewer
        than ``window`` bars (the offline routine fills those positions
        with NaN; we surface that as ``None`` so callers don't have to
        NaN-check floats).

        Byte-identity to offline is preserved as long as the buffer's
        bars themselves are byte-identical (Tier 1's aggregator test
        locks that contract).
        """
        if window <= 0:
            raise ValueError(f"window must be positive, got {window}")
        if len(self._bars) < window:
            return None
        buy = np.fromiter(
            (b.buy_volume for b in self._bars), dtype=np.float64,
            count=len(self._bars),
        )
        sell = np.fromiter(
            (b.sell_volume for b in self._bars), dtype=np.float64,
            count=len(self._bars),
        )
        vpin_arr = _vpin_rolling(buy, sell, window)
        last = vpin_arr[-1]
        if not np.isfinite(last):
            return None
        return float(last)


def _dtype_for(col: str) -> str:
    """Per-column dtype for the empty-DataFrame schema."""
    if col == "n_trades":
        return "int64"
    return "float64"


__all__ = ["DollarBarBuffer"]
