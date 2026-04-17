"""Local CSV/parquet data provider — loads from Binance 1m data on disk.

Implements DataProvider protocol. Useful for:
- Backtesting the live engine against historical data
- Fallback when exchange API is unavailable
- Parity testing (same data as experiments)
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import pandas as pd

from signal_engine.data import load_binance_1m, resample_ohlcv

logger = logging.getLogger(__name__)


class LocalCSVProvider:
    """Load OHLCV from local Binance 1m CSVs, resample to requested interval.

    Implements the DataProvider protocol.
    """

    def __init__(
        self,
        data_dirs: list[str] = None,
    ) -> None:
        self._data_dirs = [Path(d) for d in (data_dirs or [
            "data/binance-1m", "data/homeserver-mirror/binance-1m",
        ])]
        self._cache: dict[str, pd.DataFrame] = {}  # symbol -> 1m OHLCV

    async def fetch_ohlcv(
        self, symbol: str, interval: str, limit: int = 200
    ) -> pd.DataFrame:
        """Load 1m data from disk, resample to interval, return last `limit` bars."""
        # Load and cache 1m data
        if symbol not in self._cache:
            try:
                ohlcv_1m = load_binance_1m(symbol, self._data_dirs)
                self._cache[symbol] = ohlcv_1m
            except FileNotFoundError:
                logger.warning("No local data for %s", symbol)
                return pd.DataFrame()

        ohlcv_1m = self._cache[symbol]

        # Map interval strings
        resample_map = {
            "15m": "15min", "15min": "15min",
            "1h": "1h", "3h": "3h", "4h": "4h",
            "1d": "1D", "1D": "1D", "24h": "1D",
        }
        resample_str = resample_map.get(interval, interval)

        try:
            ohlcv_tf = resample_ohlcv(ohlcv_1m, resample_str)
        except Exception as e:
            logger.error("Resample %s %s: %s", symbol, interval, e)
            return pd.DataFrame()

        if ohlcv_tf.index.tz is not None:
            ohlcv_tf.index = ohlcv_tf.index.tz_localize(None)

        return ohlcv_tf.tail(limit)

    async def fetch_ticker(self, symbol: str) -> dict[str, float]:
        """Return last known price from local data."""
        if symbol not in self._cache:
            await self.fetch_ohlcv(symbol, "1h", 1)

        if symbol in self._cache:
            last = self._cache[symbol]["close"].iloc[-1]
            return {
                "last_price": float(last),
                "bid": float(last),
                "ask": float(last),
                "volume_24h": 0.0,
            }

        return {"last_price": 0.0, "bid": 0.0, "ask": 0.0, "volume_24h": 0.0}

    async def close(self) -> None:
        self._cache.clear()
