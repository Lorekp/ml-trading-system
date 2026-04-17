"""Data loading and resampling for Binance 1-minute OHLCV CSVs.

Extracted from experiments/expanded_portfolio_test.py (lines 384-478).
Handles multiple data directories and both header/no-header CSV formats.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

# Standard Binance 1m CSV column names.
_COL_NAMES = [
    "open_time", "open", "high", "low", "close", "volume",
    "close_time", "quote_volume", "count",
    "taker_buy_vol", "taker_buy_quote_vol", "ignore",
]

_RAW_COLS = ["open_time", "open", "high", "low", "close", "volume"]


def _csv_has_header(csv_path: Path) -> bool:
    """Cheaply detect whether a Binance CSV contains a header row."""
    try:
        with csv_path.open("r", encoding="utf-8", errors="ignore") as handle:
            first_token = handle.readline().split(",", 1)[0].strip()
    except OSError:
        return False
    try:
        float(first_token)
        return False
    except (TypeError, ValueError):
        return True


def load_binance_1m(symbol: str, data_dirs: list[Path]) -> pd.DataFrame:
    """Load all monthly CSV files for a symbol from Binance 1-minute data.

    Checks multiple data directories and uses whichever has data first.
    Handles both header and no-header CSV formats (some 2025+ files
    include a header row).

    Args:
        symbol: Binance symbol, e.g. ``"BTCUSDT"``.
        data_dirs: Ordered list of directories to search.  Each directory is
            expected to contain a ``<symbol>/`` sub-directory with files
            matching ``<symbol>-1m-*.csv``.

    Returns:
        OHLCV DataFrame with a UTC ``DatetimeIndex`` named ``timestamp``.

    Raises:
        FileNotFoundError: If no CSV files are found in any directory.
    """
    files: list[Path] = []
    used_dir: Path | None = None

    for data_dir in data_dirs:
        data_dir = Path(data_dir)
        symbol_dir = data_dir / symbol
        pattern = f"{symbol}-1m-*.csv"
        found = sorted(symbol_dir.glob(pattern))
        if not found:
            found = sorted(data_dir.glob(pattern))
        if found:
            files = found
            used_dir = data_dir
            break

    if not files:
        raise FileNotFoundError(
            f"No files matching {symbol}-1m-*.csv in any of: "
            + ", ".join(str(d / symbol) for d in data_dirs)
        )

    frames: list[pd.DataFrame] = []
    for fp in files:
        try:
            read_kwargs = {
                "usecols": range(6),
                "low_memory": False,
                "memory_map": True,
            }
            if _csv_has_header(fp):
                df = pd.read_csv(fp, header=0, **read_kwargs)
                df.columns = _RAW_COLS[: len(df.columns)]
            else:
                df = pd.read_csv(fp, header=None, names=_RAW_COLS, **read_kwargs)

            frames.append(df)
        except Exception as exc:
            logger.warning("Failed to load %s: %s", fp.name, exc)
            continue

    if not frames:
        raise FileNotFoundError(f"No valid CSV files for {symbol}")

    combined = pd.concat(frames, ignore_index=True)

    combined = pd.concat(frames, ignore_index=True, copy=False)
    combined["open_time"] = pd.to_numeric(combined["open_time"], errors="coerce")
    for col in _RAW_COLS[1:]:
        combined[col] = pd.to_numeric(combined[col], errors="coerce")
    combined = combined.dropna(subset=_RAW_COLS)
    combined["timestamp"] = pd.to_datetime(
        combined["open_time"].astype("int64"), unit="ms", utc=True
    )
    combined = combined.set_index("timestamp").sort_index()

    # Keep OHLCV columns only.
    ohlcv = combined[_RAW_COLS[1:]].copy()

    # Drop duplicate timestamps
    ohlcv = ohlcv[~ohlcv.index.duplicated(keep="first")]

    dir_label = "primary" if used_dir == Path(data_dirs[0]) else "mirror"
    logger.info(
        "Loaded %s (%s): %s bars, %s to %s",
        symbol,
        dir_label,
        f"{len(ohlcv):,}",
        ohlcv.index.min().date(),
        ohlcv.index.max().date(),
    )
    return ohlcv


def resample_ohlcv(df: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Standard OHLCV resampling to a coarser timeframe.

    Args:
        df: OHLCV DataFrame with a ``DatetimeIndex``.
        timeframe: Pandas offset alias, e.g. ``"15min"``, ``"4h"``.

    Returns:
        Resampled OHLCV DataFrame with NaN rows dropped.
    """
    resampled = df.resample(timeframe).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()
    return resampled
