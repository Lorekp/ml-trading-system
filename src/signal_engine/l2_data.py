"""L2 data loader — streams from homeserver HDD over SSH/SFTP.

Reads Binance Vision bookTicker/bookDepth data from the homeserver's
/hdd/binance-l2/ directory. Uses paramiko SFTP for direct streaming
without mounting network drives.

Fallback: scp to temp file if paramiko is unavailable.
"""

from __future__ import annotations

import io
import logging
import os
import subprocess
import tempfile
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Default remote paths
REMOTE_HOST = "monitor"  # SSH alias
REMOTE_L2_DIR = "/hdd/binance-l2"
LOCAL_CACHE_DIR = Path("data/l2-cache")  # small local cache for frequently used files


def _sftp_read(remote_path: str) -> bytes:
    """Read a remote file. Uses scp for large files, ssh cat for small."""
    import tempfile

    # Check file size first
    size_check = subprocess.run(
        ["ssh", REMOTE_HOST, f"stat -c %s {remote_path} 2>/dev/null || echo 0"],
        capture_output=True, text=True, timeout=15,
    )
    file_size = int(size_check.stdout.strip()) if size_check.returncode == 0 else 0

    if file_size == 0:
        raise FileNotFoundError(f"Remote file not found: {remote_path}")

    if file_size > 10_000_000:  # >10MB: use scp (ssh cat corrupts large binary)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".zip") as tmp:
            tmp_path = tmp.name
        try:
            result = subprocess.run(
                ["scp", f"{REMOTE_HOST}:{remote_path}", tmp_path],
                capture_output=True, timeout=300,
            )
            if result.returncode != 0:
                raise FileNotFoundError(f"scp failed for {remote_path}")
            with open(tmp_path, "rb") as f:
                return f.read()
        finally:
            Path(tmp_path).unlink(missing_ok=True)
    else:
        result = subprocess.run(
            ["ssh", REMOTE_HOST, f"cat {remote_path}"],
            capture_output=True, timeout=120,
        )
        if result.returncode != 0:
            raise FileNotFoundError(f"Remote file not found: {remote_path}")
        return result.stdout


def _load_zip_csv(zip_bytes: bytes) -> pd.DataFrame:
    """Extract CSV from a zip file in memory."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            return pd.read_csv(f)


def _load_zip_csv_chunked(zip_bytes: bytes, chunksize: int = 1_000_000):
    """Yield DataFrame chunks from a zipped CSV."""
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        csv_name = zf.namelist()[0]
        with zf.open(csv_name) as f:
            for chunk in pd.read_csv(f, chunksize=chunksize):
                yield chunk


def load_book_ticker(
    symbol: str,
    date: str,
    remote_dir: str = REMOTE_L2_DIR,
) -> pd.DataFrame:
    """Load one day of bookTicker data (best bid/ask at tick level).

    Streams the zip from the homeserver, decompresses in memory.
    ~12M rows per day for BTCUSDT.

    Args:
        symbol: e.g. "BTCUSDT"
        date: e.g. "2024-01-01"
        remote_dir: base directory on homeserver

    Returns:
        DataFrame with columns: update_id, best_bid_price, best_bid_qty,
        best_ask_price, best_ask_qty, transaction_time, event_time
    """
    remote_path = f"{remote_dir}/bookTicker/{symbol}/{symbol}-bookTicker-{date}.zip"
    t0 = time.time()
    data = _sftp_read(remote_path)
    df = _load_zip_csv(data)
    elapsed = time.time() - t0
    logger.info("Loaded bookTicker %s %s: %d rows in %.1fs (%.0f MB)",
                symbol, date, len(df), elapsed, len(data) / 1e6)
    return df


def load_book_ticker_chunked(
    symbol: str,
    date: str,
    chunksize: int = 1_000_000,
    remote_dir: str = REMOTE_L2_DIR,
):
    """Stream bookTicker in chunks for memory-efficient processing.

    Yields DataFrames of `chunksize` rows each.
    """
    remote_path = f"{remote_dir}/bookTicker/{symbol}/{symbol}-bookTicker-{date}.zip"
    data = _sftp_read(remote_path)
    yield from _load_zip_csv_chunked(data, chunksize)


def load_book_depth(
    symbol: str,
    date: str,
    remote_dir: str = REMOTE_L2_DIR,
) -> pd.DataFrame:
    """Load one day of aggregate depth data.

    Small files (~29K rows, ~500KB).

    Returns:
        DataFrame with columns: timestamp, percentage, depth, notional
    """
    remote_path = f"{remote_dir}/bookDepth/{symbol}/{symbol}-bookDepth-{date}.zip"
    data = _sftp_read(remote_path)
    return _load_zip_csv(data)


def load_trades(
    symbol: str,
    date: str,
    remote_dir: str = REMOTE_L2_DIR,
) -> pd.DataFrame:
    """Load one day of individual trade data."""
    remote_path = f"{remote_dir}/trades/{symbol}/{symbol}-trades-{date}.zip"
    data = _sftp_read(remote_path)
    return _load_zip_csv(data)


def list_available_dates(
    symbol: str,
    data_type: str = "bookTicker",
    remote_dir: str = REMOTE_L2_DIR,
) -> list[str]:
    """List available dates for a symbol and data type."""
    remote_path = f"{remote_dir}/{data_type}/{symbol}/"
    result = subprocess.run(
        ["ssh", REMOTE_HOST, f"ls {remote_path}"],
        capture_output=True, text=True, timeout=30,
    )
    if result.returncode != 0:
        return []
    files = result.stdout.strip().split("\n")
    dates = []
    for f in files:
        if f.endswith(".zip") and "CHECKSUM" not in f:
            # Extract date from filename like BTCUSDT-bookTicker-2024-01-01.zip
            parts = f.replace(".zip", "").split("-")
            if len(parts) >= 5:
                date_str = f"{parts[-3]}-{parts[-2]}-{parts[-1]}"
                dates.append(date_str)
    return sorted(dates)


def compute_ofi_from_book_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Order Flow Imbalance from bookTicker data.

    OFI = change in best bid qty - change in best ask qty
    (Kolm, Turiel & Westray 2023)

    Args:
        df: bookTicker DataFrame with best_bid_price, best_bid_qty,
            best_ask_price, best_ask_qty columns

    Returns:
        DataFrame resampled to desired frequency with OFI features
    """
    df = df.copy()

    # Compute bid-side order flow
    bid_price_up = df["best_bid_price"] > df["best_bid_price"].shift(1)
    bid_price_same = df["best_bid_price"] == df["best_bid_price"].shift(1)
    bid_price_down = df["best_bid_price"] < df["best_bid_price"].shift(1)

    bid_of = np.where(bid_price_up, df["best_bid_qty"],
             np.where(bid_price_same, df["best_bid_qty"] - df["best_bid_qty"].shift(1),
             -df["best_bid_qty"].shift(1)))

    # Compute ask-side order flow
    ask_price_up = df["best_ask_price"] > df["best_ask_price"].shift(1)
    ask_price_same = df["best_ask_price"] == df["best_ask_price"].shift(1)
    ask_price_down = df["best_ask_price"] < df["best_ask_price"].shift(1)

    ask_of = np.where(ask_price_down, -df["best_ask_qty"],
             np.where(ask_price_same, -(df["best_ask_qty"] - df["best_ask_qty"].shift(1)),
             df["best_ask_qty"].shift(1)))

    df["ofi"] = bid_of - ask_of
    df["spread"] = df["best_ask_price"] - df["best_bid_price"]
    df["mid_price"] = (df["best_bid_price"] + df["best_ask_price"]) / 2
    df["bid_ask_imbalance"] = (df["best_bid_qty"] - df["best_ask_qty"]) / (
        df["best_bid_qty"] + df["best_ask_qty"] + 1e-10
    )

    return df
