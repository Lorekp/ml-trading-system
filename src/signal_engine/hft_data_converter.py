"""Convert 1m OHLCV data to hftbacktest Event format.

Generates synthetic L2 depth and trade events from 1-minute OHLCV bars
for use with the hftbacktest library's tick-by-tick simulation engine.

Each 1m bar produces:
- 2 DEPTH_EVENTs: bid (close - tick/2) and ask (close + tick/2)
- 1-2 TRADE_EVENTs: volume split by bar direction (close > open = net buy)

This is a simplified model — real L2 data from the hl_l2_collector will
replace these synthetic events when available.

Usage:
    from signal_engine.hft_data_converter import convert_ohlcv_to_events
    events = convert_ohlcv_to_events(ohlcv_1m, tick_size=0.1)
    np.savez_compressed("output.npz", data=events)
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# hftbacktest event constants (matching src/data/hl_l2_collector.py)
DEPTH_EVENT = 1
TRADE_EVENT = 2
BUY_EVENT = 1 << 29
SELL_EVENT = 1 << 28
EXCH_EVENT = 1 << 31
LOCAL_EVENT = 1 << 30

EVENT_DTYPE = np.dtype([
    ("ev", "u8"),
    ("exch_ts", "i8"),
    ("local_ts", "i8"),
    ("px", "f8"),
    ("qty", "f8"),
    ("order_id", "u8"),
    ("ival", "i8"),
    ("fval", "f8"),
], align=True)

# Tick sizes for common assets (Hyperliquid)
TICK_SIZES = {
    "BTCUSDT": 0.1, "ETHUSDT": 0.01, "SOLUSDT": 0.001,
    "XRPUSDT": 0.0001, "BNBUSDT": 0.01, "ADAUSDT": 0.0001,
    "AVAXUSDT": 0.001, "DOTUSDT": 0.001, "LINKUSDT": 0.001,
    "NEARUSDT": 0.001, "UNIUSDT": 0.001, "AAVEUSDT": 0.01,
    "LTCUSDT": 0.01, "BCHUSDT": 0.01, "FILUSDT": 0.001,
}

LOT_SIZES = {
    "BTCUSDT": 0.001, "ETHUSDT": 0.01, "SOLUSDT": 0.01,
    "XRPUSDT": 1.0, "BNBUSDT": 0.01, "ADAUSDT": 1.0,
    "AVAXUSDT": 0.1, "DOTUSDT": 0.1, "LINKUSDT": 0.1,
    "NEARUSDT": 0.1, "UNIUSDT": 0.1, "AAVEUSDT": 0.01,
    "LTCUSDT": 0.01, "BCHUSDT": 0.01, "FILUSDT": 0.1,
}


def convert_ohlcv_to_events(
    ohlcv: pd.DataFrame,
    tick_size: float = 0.1,
    base_latency_ns: int = 10_000_000,  # 10ms
) -> np.ndarray:
    """Convert 1-minute OHLCV DataFrame to hftbacktest event array.

    Fully vectorized — no Python loops over bars.

    Args:
        ohlcv: DataFrame with columns [open, high, low, close, volume] and
               DatetimeIndex.
        tick_size: Minimum price increment for the asset.
        base_latency_ns: Simulated feed latency (exchange → local) in nanoseconds.

    Returns:
        Numpy structured array with EVENT_DTYPE, ready for np.savez_compressed().
    """
    n_bars = len(ohlcv)
    if n_bars == 0:
        return np.zeros(0, dtype=EVENT_DTYPE)

    # Extract arrays
    close = ohlcv["close"].values.astype(np.float64)
    volume = ohlcv["volume"].values.astype(np.float64)
    open_ = ohlcv["open"].values.astype(np.float64)

    # Timestamps: convert to nanoseconds
    ts_ns = ohlcv.index.astype(np.int64)  # already in nanoseconds for datetime64[ns]

    # Direction: close > open means net buying
    is_buy = close > open_

    # Each bar produces 3 events: bid depth, ask depth, trade
    n_events = n_bars * 3
    events = np.zeros(n_events, dtype=EVENT_DTYPE)

    # Bid depth events (even indices: 0, 3, 6, ...)
    bid_idx = np.arange(0, n_events, 3)
    events["ev"][bid_idx] = EXCH_EVENT | LOCAL_EVENT | DEPTH_EVENT | BUY_EVENT
    events["exch_ts"][bid_idx] = ts_ns
    events["local_ts"][bid_idx] = ts_ns + base_latency_ns
    events["px"][bid_idx] = close - tick_size / 2.0
    events["qty"][bid_idx] = volume * 0.5  # approximate bid depth as half volume

    # Ask depth events (indices: 1, 4, 7, ...)
    ask_idx = np.arange(1, n_events, 3)
    events["ev"][ask_idx] = EXCH_EVENT | LOCAL_EVENT | DEPTH_EVENT | SELL_EVENT
    events["exch_ts"][ask_idx] = ts_ns
    events["local_ts"][ask_idx] = ts_ns + base_latency_ns
    events["px"][ask_idx] = close + tick_size / 2.0
    events["qty"][ask_idx] = volume * 0.5

    # Trade events (indices: 2, 5, 8, ...)
    trade_idx = np.arange(2, n_events, 3)
    trade_side = np.where(is_buy, BUY_EVENT, SELL_EVENT)
    events["ev"][trade_idx] = EXCH_EVENT | LOCAL_EVENT | TRADE_EVENT | trade_side
    events["exch_ts"][trade_idx] = ts_ns + 30_000_000_000  # 30s into the bar
    events["local_ts"][trade_idx] = ts_ns + 30_000_000_000 + base_latency_ns
    events["px"][trade_idx] = close
    events["qty"][trade_idx] = volume

    return events


def convert_asset(
    symbol: str,
    ohlcv_1m: pd.DataFrame,
    output_dir: Path,
    tick_size: float | None = None,
) -> Path:
    """Convert one asset's 1m data to hftbacktest .npz files (one per month).

    Args:
        symbol: e.g. "BTCUSDT"
        ohlcv_1m: 1-minute OHLCV DataFrame
        output_dir: directory for output .npz files
        tick_size: override tick size (auto-detected from TICK_SIZES if None)

    Returns:
        Path to output directory
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    if tick_size is None:
        tick_size = TICK_SIZES.get(symbol, 0.01)

    # Group by month
    ohlcv_1m = ohlcv_1m.copy()
    months = ohlcv_1m.index.to_period("M").unique()

    for month in months:
        mask = ohlcv_1m.index.to_period("M") == month
        month_data = ohlcv_1m[mask]
        if len(month_data) < 100:
            continue

        events = convert_ohlcv_to_events(month_data, tick_size=tick_size)
        fname = output_dir / f"{symbol}_{month}.npz"
        np.savez_compressed(str(fname), data=events)

    logger.info("Converted %s: %d months -> %s", symbol, len(months), output_dir)
    return output_dir


def convert_all_assets(
    assets: dict[str, str],
    data_dirs: list[Path],
    output_dir: Path,
    max_workers: int = 8,
) -> dict[str, list[Path]]:
    """Convert all assets to hftbacktest format in parallel.

    Args:
        assets: {name: symbol} dict
        data_dirs: directories to search for 1m data
        output_dir: base output directory

    Returns:
        {symbol: [list of .npz file paths]}
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from signal_engine.data import load_binance_1m

    output_dir.mkdir(parents=True, exist_ok=True)
    result = {}

    def _convert_one(name, symbol):
        try:
            ohlcv_1m = load_binance_1m(symbol, data_dirs)
            asset_dir = output_dir / symbol
            convert_asset(symbol, ohlcv_1m, asset_dir)
            files = sorted(asset_dir.glob("*.npz"))
            return symbol, files
        except Exception as e:
            logger.warning("Failed to convert %s: %s", symbol, e)
            return symbol, []

    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_convert_one, name, sym): name
            for name, sym in assets.items()
        }
        for future in as_completed(futures):
            symbol, files = future.result()
            result[symbol] = files
            if files:
                logger.info("  %s: %d files", symbol, len(files))

    return result
