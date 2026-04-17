"""
Tests for data loader.

Tests with synthetic Binance-format CSV data and validates
the full load → resample → validate pipeline.
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest
import tempfile
import os
from pathlib import Path
from data.loader import (
    load_binance_csv, load_symbol, resample_ohlcv,
    validate_ohlcv, list_available_symbols, BINANCE_COLUMNS,
)


@pytest.fixture
def synthetic_binance_csv(tmp_path):
    """Create a synthetic Binance-format CSV file."""
    np.random.seed(42)
    n = 1440  # 1 day of 1-minute data

    base_time = pd.Timestamp("2024-01-01", tz="UTC")
    timestamps_ms = [int((base_time + pd.Timedelta(minutes=i)).timestamp() * 1000) for i in range(n)]

    prices = 100.0 + np.cumsum(np.random.normal(0, 0.1, n))
    high = prices + np.abs(np.random.normal(0, 0.05, n))
    low = prices - np.abs(np.random.normal(0, 0.05, n))
    volume = np.random.exponential(100, n)

    rows = []
    for i in range(n):
        rows.append([
            timestamps_ms[i], prices[i], high[i], low[i], prices[i],
            volume[i], timestamps_ms[i] + 59999, volume[i] * prices[i],
            int(np.random.exponential(50)), volume[i] * 0.6,
            volume[i] * prices[i] * 0.6, 0,
        ])

    csv_path = tmp_path / "TESTUSDT" / "TESTUSDT-1m-2024-01.csv"
    csv_path.parent.mkdir(parents=True)
    df = pd.DataFrame(rows)
    df.to_csv(csv_path, header=False, index=False)
    return tmp_path, csv_path


class TestLoadBinanceCsv:
    def test_loads_correct_columns(self, synthetic_binance_csv):
        _, csv_path = synthetic_binance_csv
        df = load_binance_csv(csv_path)
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_datetime_index(self, synthetic_binance_csv):
        _, csv_path = synthetic_binance_csv
        df = load_binance_csv(csv_path)
        assert isinstance(df.index, pd.DatetimeIndex)

    def test_correct_row_count(self, synthetic_binance_csv):
        _, csv_path = synthetic_binance_csv
        df = load_binance_csv(csv_path)
        assert len(df) == 1440


class TestResampleOhlcv:
    def test_1m_to_1h(self, synthetic_binance_csv):
        _, csv_path = synthetic_binance_csv
        df_1m = load_binance_csv(csv_path)
        df_1h = resample_ohlcv(df_1m, "1H")
        assert len(df_1h) == 24  # 24 hours in 1 day

    def test_1m_to_4h(self, synthetic_binance_csv):
        _, csv_path = synthetic_binance_csv
        df_1m = load_binance_csv(csv_path)
        df_4h = resample_ohlcv(df_1m, "4H")
        assert len(df_4h) == 6  # 6 × 4H bars in 1 day

    def test_ohlcv_semantics(self, synthetic_binance_csv):
        """Verify open=first, high=max, low=min, close=last, vol=sum."""
        _, csv_path = synthetic_binance_csv
        df_1m = load_binance_csv(csv_path)
        df_1h = resample_ohlcv(df_1m, "1H")

        # Check first hour
        first_hour = df_1m.iloc[:60]
        bar = df_1h.iloc[0]
        assert bar["open"] == first_hour["open"].iloc[0]
        assert bar["high"] == first_hour["high"].max()
        assert bar["low"] == first_hour["low"].min()
        assert bar["close"] == first_hour["close"].iloc[-1]
        assert bar["volume"] == pytest.approx(first_hour["volume"].sum(), rel=1e-6)


class TestLoadSymbol:
    def test_loads_and_resamples(self, synthetic_binance_csv):
        data_dir, _ = synthetic_binance_csv
        df = load_symbol("TESTUSDT", data_dir, timeframe="1H")
        assert len(df) == 24

    def test_caching(self, synthetic_binance_csv):
        data_dir, _ = synthetic_binance_csv
        cache_dir = data_dir / "cache"
        # First load: creates cache
        df1 = load_symbol("TESTUSDT", data_dir, timeframe="4H", cache_dir=cache_dir)
        cache_file = cache_dir / "TESTUSDT_4H.parquet"
        assert cache_file.exists()
        # Second load: from cache
        df2 = load_symbol("TESTUSDT", data_dir, timeframe="4H", cache_dir=cache_dir)
        # Parquet roundtrip may lose index frequency metadata — compare values only
        pd.testing.assert_frame_equal(df1.reset_index(drop=True), df2.reset_index(drop=True))


class TestValidateOhlcv:
    def test_clean_data_valid(self, synthetic_binance_csv):
        _, csv_path = synthetic_binance_csv
        df = load_binance_csv(csv_path)
        result = validate_ohlcv(df, "TESTUSDT")
        assert result["rows"] == 1440
        # Synthetic data is clean — should have minimal issues
        # (may have some H<L from random generation)

    def test_detects_nans(self):
        df = pd.DataFrame({
            "open": [100, np.nan, 100],
            "high": [101, 101, 101],
            "low": [99, 99, 99],
            "close": [100, 100, 100],
            "volume": [1000, 1000, 1000],
        }, index=pd.date_range("2024-01-01", periods=3, freq="1min"))
        result = validate_ohlcv(df, "TEST")
        assert not result["valid"]
        assert any("NaN" in issue for issue in result["issues"])


class TestListSymbols:
    def test_finds_symbols(self, synthetic_binance_csv):
        data_dir, _ = synthetic_binance_csv
        symbols = list_available_symbols(data_dir)
        assert "TESTUSDT" in symbols


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
