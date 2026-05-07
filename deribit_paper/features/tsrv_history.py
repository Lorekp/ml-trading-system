"""Live TSRV history buffer for the F3 Tier 5 TSRV-grid features.

The 6 TSRV-derived missing handlers are computed offline on the
dollar-bar grid (each TSRV value ffilled onto every dollar bar within
its 15-min window). For a live byte-identity match we'd need to keep
~1750 dollar bars in the buffer (1 week × 250 bars/day at $500K
threshold), then ffill TSRV onto each.

Tier 5 ships a PRAGMATIC 15-min-grid implementation instead: TSRV
features are computed directly on the live 15-min TSRV cadence with
windows scaled to that grid (96 bars = 1 day; 672 bars = 1 week). This
is mathematically equivalent to the offline computation when dollar
bars are evenly-spaced in time; deviates from offline byte-identity
when volume is bursty (dollar bars cluster during high-vol periods).
The deviation is documented + the live values are still substantially
better than baseline-frozen.

Buffer schema:

* ``timestamp`` (UTC, 15-min cadence)
* ``vol_ann`` (TSRV-based annualised vol — same as offline parquet column)
* ``up_jump_fraction``
* ``bpv``
* ``rv_slow``

Pre-warm via ``prewarm_from_parquet`` reads ``TSRV/output/<asset>_tsrv_15m.parquet``
which the engineer pipeline maintains as canonical.
"""

from __future__ import annotations

import logging
from collections import deque
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from deribit_paper.features.tsrv_live import BarTSRV

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TSRVHistoryRow:
    """One 15-min-bar's TSRV values, stored in :class:`TSRVHistoryBuffer`."""
    timestamp: pd.Timestamp
    vol_ann: float
    up_jump_fraction: float
    bpv: float
    rv_slow: float


_PARQUET_COLUMNS: tuple[str, ...] = ("vol_ann", "up_jump_fraction", "bpv", "rv_slow")


class TSRVHistoryBuffer:
    """Bounded deque of recent ``TSRVHistoryRow`` entries.

    Default ``max_size=672`` covers 1 week of 15-min bars — enough for
    the 1w-window features and EMA50 to converge.
    """

    def __init__(self, max_size: int = 672) -> None:
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        self._max_size = int(max_size)
        self._rows: deque[TSRVHistoryRow] = deque(maxlen=self._max_size)

    @property
    def max_size(self) -> int:
        return self._max_size

    def __len__(self) -> int:
        return len(self._rows)

    def __iter__(self) -> Iterable[TSRVHistoryRow]:
        return iter(self._rows)

    def append_bar(self, timestamp: pd.Timestamp, bar_tsrv: BarTSRV) -> None:
        """Append a fresh ``BarTSRV`` produced by ``compute_bar_tsrv``."""
        self._rows.append(TSRVHistoryRow(
            timestamp=pd.Timestamp(timestamp),
            vol_ann=float(bar_tsrv.vol_ann),
            up_jump_fraction=float(bar_tsrv.up_jump_fraction),
            bpv=float(bar_tsrv.bpv),
            rv_slow=float(bar_tsrv.rv_slow),
        ))

    def latest(self) -> TSRVHistoryRow | None:
        if not self._rows:
            return None
        return self._rows[-1]

    def as_series(self, field: str) -> pd.Series:
        """Return one column as a timestamp-indexed pandas Series."""
        if not self._rows:
            return pd.Series(
                dtype=np.float64,
                index=pd.DatetimeIndex([], tz="UTC", name="timestamp"),
            )
        ts = [r.timestamp for r in self._rows]
        vals = [getattr(r, field) for r in self._rows]
        idx = pd.DatetimeIndex(ts, tz="UTC", name="timestamp")
        return pd.Series(vals, index=idx, dtype=np.float64)

    def prewarm_from_parquet(self, path: str | Path) -> int:
        """Load up to ``max_size`` rows from a TSRV parquet (15-min cadence).

        Expected schema: ``vol_ann``, ``up_jump_fraction``, ``bpv``,
        ``rv_slow`` columns + UTC ``DatetimeIndex``. Older rows are
        evicted by the deque's ``maxlen`` (only the last ``max_size``
        rows are retained).

        Fail-soft: missing/corrupt/schema-incomplete file → 0 + WARN log.
        """
        p = Path(path)
        if not p.exists():
            logger.warning(
                "TSRVHistoryBuffer prewarm: parquet %s not found; "
                "buffer fills from live process_bar over ~1 week",
                p,
            )
            return 0
        try:
            df = pd.read_parquet(p)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "TSRVHistoryBuffer prewarm: failed to read %s (%s)",
                p, exc,
            )
            return 0
        missing = set(_PARQUET_COLUMNS) - set(df.columns)
        if missing:
            logger.warning(
                "TSRVHistoryBuffer prewarm: %s missing columns %s",
                p, missing,
            )
            return 0
        # Take last ``max_size`` rows; deque auto-evicts older.
        tail = df[list(_PARQUET_COLUMNS)].tail(self._max_size)
        n_loaded = 0
        for ts, row in tail.iterrows():
            self._rows.append(TSRVHistoryRow(
                timestamp=pd.Timestamp(ts),
                vol_ann=float(row["vol_ann"]),
                up_jump_fraction=float(row["up_jump_fraction"]),
                bpv=float(row["bpv"]),
                rv_slow=float(row["rv_slow"]),
            ))
            n_loaded += 1
        logger.info(
            "TSRVHistoryBuffer prewarm: loaded %d rows from %s "
            "(window: %s → %s)",
            n_loaded, p,
            tail.index[0] if n_loaded > 0 else "n/a",
            tail.index[-1] if n_loaded > 0 else "n/a",
        )
        return n_loaded


__all__ = ["TSRVHistoryBuffer", "TSRVHistoryRow"]
