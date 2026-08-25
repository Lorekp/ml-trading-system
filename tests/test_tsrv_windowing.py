"""Regression tests for timestamp-unit-safe TSRV windowing."""

from __future__ import annotations

import numpy as np
import pandas as pd

from TSRV.backfill_rq import process_month_rq
from TSRV.tsrv_series import _process_month_tsrv
from TSRV.windowing import timestamp_window_ids


def _trades() -> pd.DataFrame:
    index = pd.date_range(
        "2024-01-01 00:00:00", periods=45, freq="1min", tz="UTC",
    )
    return pd.DataFrame(
        {
            "price": np.linspace(50_000.0, 50_100.0, len(index)),
            "dollar_volume": np.full(len(index), 1_000.0),
        },
        index=index,
    )


def test_timestamp_window_ids_normalize_datetime_units() -> None:
    index = _trades().index
    window_ids, freq_ns = timestamp_window_ids(index, pd.Timedelta("15min"))

    assert len(np.unique(window_ids)) == 3
    first = pd.Timestamp(int(window_ids[0]) * freq_ns, unit="ns", tz="UTC")
    assert first == pd.Timestamp("2024-01-01 00:00:00", tz="UTC")


def test_tsrv_month_processing_preserves_2024_windows(monkeypatch) -> None:
    trades = _trades()
    monkeypatch.setattr("TSRV.data_loader.load_trades", lambda *args: trades)

    rows = _process_month_tsrv(
        ".", "BTCUSDT", "2024-01", 1_000.0, pd.Timedelta("15min"), 1.0,
    )

    assert [row["timestamp"] for row in rows] == list(
        pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
    )


def test_rq_backfill_preserves_2024_windows(monkeypatch) -> None:
    trades = _trades()
    monkeypatch.setattr("TSRV.backfill_rq.load_trades", lambda *args: trades)

    result = process_month_rq(".", "BTCUSDT", "2024-01", 1_000.0)

    assert list(result.index) == list(
        pd.date_range("2024-01-01", periods=3, freq="15min", tz="UTC"),
    )
