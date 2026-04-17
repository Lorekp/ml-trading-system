"""HL L2 order book loader -- LZ4/JSON to contiguous numpy arrays.

Parses Hyperliquid l2Book snapshots from the raw archive:
    data/hl-bbo-raw/{COIN}/{YYYYMMDD}/{HH}/l2Book/{COIN}.lz4

Each LZ4 file contains ~22K JSON lines per hour, each with exactly
10 bid levels + 10 ask levels: {px, sz, n}.

Output: cached parquet with flat columns:
    timestamp_ms (int64), bid_px_0..9, bid_sz_0..9, ask_px_0..9, ask_sz_0..9
"""

from __future__ import annotations

import json
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

N_LEVELS = 10

# Column names for the cached parquet
_BID_PX_COLS = [f"bid_px_{i}" for i in range(N_LEVELS)]
_BID_SZ_COLS = [f"bid_sz_{i}" for i in range(N_LEVELS)]
_ASK_PX_COLS = [f"ask_px_{i}" for i in range(N_LEVELS)]
_ASK_SZ_COLS = [f"ask_sz_{i}" for i in range(N_LEVELS)]
ALL_COLS = ["timestamp_ms"] + _BID_PX_COLS + _BID_SZ_COLS + _ASK_PX_COLS + _ASK_SZ_COLS

try:
    import orjson
    _json_loads = orjson.loads
except ImportError:
    _json_loads = json.loads


def parse_hour_file(path: Path) -> np.ndarray | None:
    """Parse one LZ4 hour file into (N, 41) float64 array.

    Col 0: timestamp_ms (exchange time in ms since epoch)
    Cols 1-10: bid_px levels 0-9 (best to worst)
    Cols 11-20: bid_sz levels 0-9
    Cols 21-30: ask_px levels 0-9
    Cols 31-40: ask_sz levels 0-9

    Returns None if file is empty or unreadable.
    """
    import lz4.frame

    try:
        with lz4.frame.open(str(path), "rb") as f:
            raw = f.read()
    except Exception:
        return None

    if len(raw) < 10:
        return None

    lines = raw.decode("utf-8", errors="replace").strip().split("\n")
    if not lines:
        return None

    # Pre-allocate output
    out = np.empty((len(lines), 1 + 4 * N_LEVELS), dtype=np.float64)
    valid = 0

    for line in lines:
        if not line:
            continue
        try:
            snap = _json_loads(line)
            inner = snap.get("raw", snap) if isinstance(snap, dict) else {}
            data = inner.get("data", inner)
            ts = data.get("time", 0)
            levels = data.get("levels", [[], []])

            if len(levels) < 2:
                continue

            bids = levels[0]
            asks = levels[1]

            if len(bids) < N_LEVELS or len(asks) < N_LEVELS:
                continue

            row = out[valid]
            row[0] = float(ts)

            for L in range(N_LEVELS):
                row[1 + L] = float(bids[L]["px"])
                row[1 + N_LEVELS + L] = float(bids[L]["sz"])
                row[1 + 2 * N_LEVELS + L] = float(asks[L]["px"])
                row[1 + 3 * N_LEVELS + L] = float(asks[L]["sz"])

            valid += 1
        except (KeyError, ValueError, IndexError, TypeError):
            continue

    if valid == 0:
        return None

    return out[:valid]


def _iter_hour_files(data_dir: Path, coin: str):
    """Yield all hour file paths for a coin, sorted by date/hour."""
    coin_dir = data_dir / coin
    if not coin_dir.exists():
        return

    for date_dir in sorted(coin_dir.iterdir()):
        if not date_dir.is_dir() or len(date_dir.name) != 8:
            continue
        for hour in range(24):
            path = date_dir / str(hour) / "l2Book" / f"{coin}.lz4"
            if path.exists():
                yield path


def build_l2_cache(
    data_dir: Path,
    coin: str,
    cache_path: Path,
    max_workers: int = 8,
) -> None:
    """Parse all available L2 data and cache as parquet.

    Processes day-by-day to limit memory. Uses ThreadPoolExecutor
    for I/O parallelism across hour files within each day.
    """
    cache_path.parent.mkdir(parents=True, exist_ok=True)

    if cache_path.exists():
        logger.info("L2 cache already exists: %s", cache_path)
        return

    coin_dir = data_dir / coin
    if not coin_dir.exists():
        logger.warning("No L2 data directory: %s", coin_dir)
        return

    date_dirs = sorted(
        d for d in coin_dir.iterdir()
        if d.is_dir() and len(d.name) == 8 and d.name.isdigit()
    )
    logger.info("  L2 loader: %d dates for %s", len(date_dirs), coin)

    # Process day-by-day, accumulate chunks
    chunk_dir = cache_path.parent / "_l2_chunks"
    chunk_dir.mkdir(exist_ok=True)

    for d_idx, date_dir in enumerate(date_dirs):
        chunk_path = chunk_dir / f"{date_dir.name}.npy"
        if chunk_path.exists():
            if (d_idx + 1) % 50 == 0:
                logger.info("    [%d/%d] %s: cached", d_idx + 1, len(date_dirs), date_dir.name)
            continue

        # Collect hour file paths for this date
        hour_paths = []
        for hour in range(24):
            path = date_dir / str(hour) / "l2Book" / f"{coin}.lz4"
            if path.exists():
                hour_paths.append(path)

        if not hour_paths:
            continue

        # Parse hours in parallel
        arrays = []
        with ThreadPoolExecutor(max_workers=min(max_workers, len(hour_paths))) as pool:
            results = pool.map(parse_hour_file, hour_paths)
            for arr in results:
                if arr is not None:
                    arrays.append(arr)

        if not arrays:
            continue

        day_arr = np.concatenate(arrays)
        # Sort by timestamp
        order = day_arr[:, 0].argsort()
        day_arr = day_arr[order]

        np.save(str(chunk_path), day_arr)

        if (d_idx + 1) % 50 == 0 or d_idx == len(date_dirs) - 1:
            logger.info("    [%d/%d] %s: %d snapshots",
                        d_idx + 1, len(date_dirs), date_dir.name, len(day_arr))

    # Write chunks incrementally to parquet (avoids OOM on 149M+ rows)
    import pyarrow as pa
    import pyarrow.parquet as pq

    logger.info("  Writing chunks to parquet (streaming)...")
    chunks = sorted(chunk_dir.glob("*.npy"))
    if not chunks:
        logger.warning("  No L2 data parsed")
        return

    # Define schema
    fields = [pa.field("timestamp_ms", pa.int64())]
    for col in ALL_COLS[1:]:
        fields.append(pa.field(col, pa.float64()))
    schema = pa.schema(fields)

    total_rows = 0
    tmp_path = cache_path.with_suffix(".tmp")
    writer = pq.ParquetWriter(str(tmp_path), schema, compression="zstd")

    for i, chunk_path in enumerate(chunks):
        arr = np.load(str(chunk_path))
        # Build columnar arrays
        columns = [pa.array(arr[:, 0].astype(np.int64))]
        for j in range(1, arr.shape[1]):
            columns.append(pa.array(arr[:, j]))
        batch = pa.RecordBatch.from_arrays(columns, schema=schema)
        writer.write_batch(batch)
        total_rows += len(arr)
        del arr, columns, batch

        if (i + 1) % 100 == 0 or i == len(chunks) - 1:
            logger.info("    Written %d/%d chunks (%d rows)",
                        i + 1, len(chunks), total_rows)

    writer.close()
    tmp_path.rename(cache_path)
    logger.info("  Cached to %s (%d rows, %.1f MB)",
                cache_path, total_rows, cache_path.stat().st_size / 1e6)


def iter_l2_chunks(
    cache_dir: Path, coin: str,
) -> "Iterator[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]]":
    """Iterate over daily L2 chunks (memory-efficient).

    Yields (timestamps_ms, bid_px, bid_sz, ask_px, ask_sz) per day.
    Each array is C-contiguous float64/int64.
    Peak memory: ~50 MB per day chunk.
    """
    chunk_dir = cache_dir / "_cache" / "_l2_chunks"
    if not chunk_dir.exists():
        return

    for chunk_path in sorted(chunk_dir.glob("*.npy")):
        arr = np.load(str(chunk_path))
        ts = arr[:, 0].astype(np.int64)
        bp = np.ascontiguousarray(arr[:, 1:11])
        bs = np.ascontiguousarray(arr[:, 11:21])
        ap = np.ascontiguousarray(arr[:, 21:31])
        az = np.ascontiguousarray(arr[:, 31:41])
        yield ts, bp, bs, ap, az
        del arr


def get_l2_date_range(cache_dir: Path) -> tuple[int, int] | None:
    """Get timestamp range of cached L2 data without loading it all."""
    chunk_dir = cache_dir / "_cache" / "_l2_chunks"
    if not chunk_dir.exists():
        return None

    chunks = sorted(chunk_dir.glob("*.npy"))
    if not chunks:
        return None

    first = np.load(str(chunks[0]))
    last = np.load(str(chunks[-1]))
    ts_start = int(first[0, 0])
    ts_end = int(last[-1, 0])
    del first, last
    return ts_start, ts_end


def load_l2_arrays(
    cache_path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load cached L2 arrays. For small datasets only (<10M rows).

    For large datasets, use iter_l2_chunks() instead.

    Returns:
        timestamps_ms: (N,) int64 array of exchange timestamps (ms)
        bid_px: (N, 10) float64 bid prices per level
        bid_sz: (N, 10) float64 bid sizes per level
        ask_px: (N, 10) float64 ask prices per level
        ask_sz: (N, 10) float64 ask sizes per level
    """
    import pyarrow.parquet as pq
    table = pq.read_table(str(cache_path))

    timestamps_ms = table.column("timestamp_ms").to_numpy().astype(np.int64)
    bid_px = np.column_stack([table.column(c).to_numpy() for c in _BID_PX_COLS])
    bid_sz = np.column_stack([table.column(c).to_numpy() for c in _BID_SZ_COLS])
    ask_px = np.column_stack([table.column(c).to_numpy() for c in _ASK_PX_COLS])
    ask_sz = np.column_stack([table.column(c).to_numpy() for c in _ASK_SZ_COLS])

    bid_px = np.ascontiguousarray(bid_px)
    bid_sz = np.ascontiguousarray(bid_sz)
    ask_px = np.ascontiguousarray(ask_px)
    ask_sz = np.ascontiguousarray(ask_sz)

    logger.info("  Loaded L2 cache: %d snapshots (%s to %s)",
                len(timestamps_ms),
                pd.Timestamp(int(timestamps_ms[0]), unit="ms", tz="UTC"),
                pd.Timestamp(int(timestamps_ms[-1]), unit="ms", tz="UTC"))

    return timestamps_ms, bid_px, bid_sz, ask_px, ask_sz
