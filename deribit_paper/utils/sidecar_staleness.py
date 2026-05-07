"""S4 (parity-cycle round 2 — 2026-05-07): sidecar staleness via parquet
data-age, not file mtime.

Pre-S4 ``_check_sidecar_staleness`` in ``deribit_paper/engine.py`` used
``os.path.getmtime(p)`` to age-check sidecars. A ``cp`` deploy of an
older parquet refreshes mtime and masks the staleness — the engine
thinks the sidecar is fresh when its underlying data is stale.

This helper reads the parquet's row-group metadata to find the maximum
timestamp value in a designated time column, returning the age in
seconds versus ``now``. No full-data read — Parquet stores per-row-group
column statistics in the file footer, so the lookup is ~50ms even on a
1-GB parquet. Falls back to ``os.path.getmtime`` for parquets without
a recognisable time column.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import Iterable

import pandas as pd

logger = logging.getLogger(__name__)


_DEFAULT_TIME_COLUMNS: tuple[str, ...] = (
    "timestamp",
    "ts",
    "time",
    "bar_timestamp",
    "open_time",
)


def data_age_seconds(
    path: str | Path,
    *,
    time_columns: Iterable[str] = _DEFAULT_TIME_COLUMNS,
    fallback_to_mtime: bool = True,
) -> float:
    """Return the age (seconds since now) of the parquet's youngest row.

    Looks for a time column matching one of ``time_columns`` (case
    insensitive) and reads its max from the parquet's column
    statistics. If neither the parquet's columns nor index expose a
    parseable timestamp, falls back to file mtime.

    Parameters
    ----------
    path : str | Path
        Parquet file. Missing file → ``float('inf')`` (treated as
        infinitely stale, matches the existence-check semantics in
        ``engine._check_sidecar_staleness``).
    time_columns : iterable of str
        Candidate column names. The first one that exists in the
        parquet schema wins.
    fallback_to_mtime : bool
        When False, raise ``RuntimeError`` rather than fall back to
        mtime when no recognised time column is found. Useful for
        regression-locking sidecars that should always carry data-age
        metadata (e.g., V2 trade log).

    Returns
    -------
    float
        Age in seconds vs current wall-clock time. Negative for
        future-dated rows (clock-skew sanity check; caller may treat
        large negatives as a bug).
    """
    p = Path(path)
    if not p.exists():
        return float("inf")

    try:
        import pyarrow.parquet as pq
        pf = pq.ParquetFile(str(p))
        schema = pf.schema_arrow
        col_lookup = {c.lower(): c for c in schema.names}
        chosen_col: str | None = None
        for cand in time_columns:
            if cand.lower() in col_lookup:
                chosen_col = col_lookup[cand.lower()]
                break

        if chosen_col is not None:
            # Walk row-group statistics to find the global max for the
            # chosen column. Avoids a full-data read.
            max_ts: pd.Timestamp | None = None
            for rg_i in range(pf.num_row_groups):
                rg = pf.metadata.row_group(rg_i)
                col_idx = schema.get_field_index(chosen_col)
                if col_idx < 0:
                    continue
                col_stats = rg.column(col_idx).statistics
                if col_stats is None or not col_stats.has_min_max:
                    continue
                ts_max = pd.Timestamp(col_stats.max)
                if ts_max.tzinfo is None:
                    ts_max = ts_max.tz_localize("UTC")
                if max_ts is None or ts_max > max_ts:
                    max_ts = ts_max
            if max_ts is not None:
                now = pd.Timestamp.now(tz="UTC")
                return float((now - max_ts).total_seconds())
    except Exception:  # noqa: BLE001 - any pyarrow failure → fallback
        logger.debug("S4: data-age parquet probe failed, falling back", exc_info=True)

    if not fallback_to_mtime:
        raise RuntimeError(
            f"Sidecar {p} has no recognised time column "
            f"(searched: {list(time_columns)})"
        )
    return float(time.time() - os.path.getmtime(p))


__all__ = ["data_age_seconds"]
