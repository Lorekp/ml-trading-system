"""Per-bar metrics sink → rolling parquet file.

Each bar close writes one row to a parquet file under
``deribit_paper/state/live_metrics.parquet``.  The schema matches
(approximately) the backtester's ``trade_log`` columns so we can run
the same :func:`vol_arb.metrics.full_metrics_both` on the live data
once the paper run has enough samples.

Implementation: buffer rows in memory, flush to parquet every N rows
or on explicit :meth:`flush`.  Flushing appends a new row group to
the parquet file (creates it on first write).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)


class MetricsSink:
    """Append-only parquet sink for per-bar metrics rows.

    Usage::

        sink = MetricsSink("deribit_paper/state/live_metrics.parquet",
                            flush_every=20)
        sink.record({"timestamp": ..., "signal": ..., "edge_vp": ...})
        sink.flush()  # writes buffered rows to disk
    """

    def __init__(
        self,
        path: str | Path,
        *,
        flush_every: int = 20,
    ):
        self.path = Path(path)
        self.flush_every = flush_every
        self._buffer: list[dict[str, Any]] = []

    def record(self, row: dict[str, Any]) -> None:
        self._buffer.append(dict(row))
        if len(self._buffer) >= self.flush_every:
            self.flush()

    def flush(self) -> int:
        """Write the buffered rows to parquet.  Returns the number written."""
        if not self._buffer:
            return 0
        n = len(self._buffer)
        new_df = pd.DataFrame(self._buffer)
        self._buffer.clear()

        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            try:
                existing = pd.read_parquet(self.path)
                combined = pd.concat([existing, new_df], ignore_index=True)
                combined.to_parquet(self.path)
            except Exception:  # noqa: BLE001 - log + overwrite
                logger.exception(
                    "MetricsSink: failed to read existing %s — overwriting with new buffer",
                    self.path,
                )
                new_df.to_parquet(self.path)
        else:
            new_df.to_parquet(self.path)
        logger.info("MetricsSink flushed %d rows → %s", n, self.path)
        return n

    def __len__(self) -> int:
        return len(self._buffer)
