"""F3 Tier 3 byte-identity tests for ``DollarBarFeatures``.

The 16 dollar-bar-derived handlers in ``deribit_paper/signal/feature_assembler.py``
all delegate to a :class:`DollarBarFeatures` wrapper around a live
:class:`DollarBarBuffer`. Each handler must reproduce the offline
``vol_model/features/engineer.py::_window_and_aggregate`` formula
applied at the latest dollar bar to within 1e-12 relative tolerance.

Two test halves:

1. **Synthetic-buffer unit tests** — small hand-crafted buffer where
   the expected value is computed inline. Verifies the per-method
   mechanics (latest_raw / ema / lag / parkinson_1d / parkinson_1d_ema).

2. **Byte-identity tests** on the same 200K-trade Binance Vision slice
   the Tier 1+2 fixtures use. The reference is computed via pandas
   directly on the buffer's ``as_dataframe()``; the live value comes
   from ``DollarBarFeatures``. Skips when the parquet isn't present.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.dollar_bar import DollarBar, DollarBarAggregator
from deribit_paper.features.dollar_bar_buffer import DollarBarBuffer
from deribit_paper.features.dollar_bar_features import DollarBarFeatures
from vol_model.features.microstructure import _vpin_rolling


_PARQUET_PATH = Path("data/binance-aggTrades-parquet/BTCUSDT/2026-04.parquet")


# ── synthetic buffer helper ──────────────────────────────────────────


def _make_synthetic_bar(
    i: int,
    *,
    price: float = 60_000.0,
    high: float = 60_100.0,
    low: float = 59_900.0,
    duration_s: float = 30.0,
    n_trades: int = 100,
    buy_volume: float = 250_000.0,
    sell_volume: float = 250_000.0,
    vov: float = 0.001,
) -> DollarBar:
    ts = pd.Timestamp("2026-05-07 00:00:00", tz="UTC") + pd.Timedelta(seconds=30 * i)
    dollar_volume = buy_volume + sell_volume
    imbalance = (buy_volume - sell_volume) / max(dollar_volume, 1.0)
    return DollarBar(
        timestamp=ts,
        open=price,
        close=price + 1.0,
        high=high,
        low=low,
        duration_s=duration_s,
        n_trades=n_trades,
        dollar_volume=dollar_volume,
        buy_volume=buy_volume,
        sell_volume=sell_volume,
        vov=vov,
        imbalance=imbalance,
        log_return=0.0001 if i > 0 else float("nan"),
        rv_bar=0.0001**2 if i > 0 else float("nan"),
    )


def _populate_buffer(n: int, buffer: DollarBarBuffer) -> None:
    for i in range(n):
        # Vary inputs deterministically so EMA/lag/parkinson values differ
        bar = _make_synthetic_bar(
            i,
            high=60_000.0 + 100.0 * (1.0 + 0.1 * i),
            low=60_000.0 - 100.0 * (1.0 + 0.05 * i),
            duration_s=30.0 + i,
            n_trades=100 + i,
            buy_volume=250_000.0 + 1000.0 * i,
            sell_volume=250_000.0 + 500.0 * i,
            vov=0.001 + 0.0001 * i,
        )
        buffer.append(bar)


# ── synthetic unit tests ─────────────────────────────────────────────


def test_latest_raw_returns_latest_bar_value():
    buf = DollarBarBuffer(max_size=10)
    _populate_buffer(5, buf)
    feats = DollarBarFeatures(buf)
    last = buf.latest_bar()
    assert feats.latest_raw("vov") == pytest.approx(last.vov)
    assert feats.latest_raw("imbalance") == pytest.approx(last.imbalance)
    assert feats.latest_raw("duration_s") == pytest.approx(last.duration_s)
    assert feats.latest_raw("n_trades") == pytest.approx(last.n_trades)


def test_latest_raw_empty_returns_none():
    buf = DollarBarBuffer(max_size=10)
    feats = DollarBarFeatures(buf)
    assert feats.latest_raw("vov") is None
    assert feats.latest_raw("vpin") is None


def test_latest_raw_unknown_name_raises():
    buf = DollarBarBuffer(max_size=10)
    _populate_buffer(2, buf)
    feats = DollarBarFeatures(buf)
    with pytest.raises(KeyError):
        feats.latest_raw("not_a_feature")


def test_ema_matches_pandas_reference_on_synthetic_buffer():
    buf = DollarBarBuffer(max_size=20)
    _populate_buffer(20, buf)
    feats = DollarBarFeatures(buf)
    expected = (
        buf.as_dataframe()["vov"]
        .ewm(span=10, adjust=True, min_periods=1).mean().iloc[-1]
    )
    assert feats.ema("vov", 10) == pytest.approx(expected, rel=1e-12)


def test_ema_zero_span_raises():
    buf = DollarBarBuffer(max_size=10)
    _populate_buffer(5, buf)
    feats = DollarBarFeatures(buf)
    with pytest.raises(ValueError):
        feats.ema("vov", 0)


def test_lag_returns_value_n_bars_before_latest():
    buf = DollarBarBuffer(max_size=20)
    _populate_buffer(10, buf)
    feats = DollarBarFeatures(buf)
    df = buf.as_dataframe()
    assert feats.lag("vov", 1) == pytest.approx(df["vov"].iloc[-2])
    assert feats.lag("vov", 3) == pytest.approx(df["vov"].iloc[-4])


def test_lag_underfull_returns_none():
    buf = DollarBarBuffer(max_size=10)
    _populate_buffer(2, buf)
    feats = DollarBarFeatures(buf)
    # Need 4 bars for lag(3); buffer has 2
    assert feats.lag("vov", 3) is None


def test_lag_zero_raises():
    buf = DollarBarBuffer(max_size=10)
    _populate_buffer(5, buf)
    feats = DollarBarFeatures(buf)
    with pytest.raises(ValueError):
        feats.lag("vov", 0)


def test_parkinson_1d_returns_none_when_underfull():
    buf = DollarBarBuffer(max_size=300)
    _populate_buffer(50, buf)
    feats = DollarBarFeatures(buf)
    assert feats.parkinson_1d(window=250) is None  # need 250


def test_parkinson_1d_matches_offline_formula_on_synthetic():
    """parkinson_bar = log(high/low)^2 / (4*log(2)); rolling-mean over window."""
    buf = DollarBarBuffer(max_size=300)
    _populate_buffer(250, buf)
    feats = DollarBarFeatures(buf)
    df = buf.as_dataframe()
    hl = (df["high"] / df["low"]).clip(lower=1.0 + 1e-10)
    park = (np.log(hl) ** 2 / (4.0 * np.log(2.0))).to_numpy()
    expected = float(np.mean(park[-250:]))
    assert feats.parkinson_1d(window=250) == pytest.approx(expected, rel=1e-12)


def test_parkinson_1d_ema_matches_offline_on_synthetic():
    buf = DollarBarBuffer(max_size=300)
    _populate_buffer(60, buf)
    feats = DollarBarFeatures(buf)
    df = buf.as_dataframe()
    hl = (df["high"] / df["low"]).clip(lower=1.0 + 1e-10)
    park = pd.Series(
        (np.log(hl) ** 2 / (4.0 * np.log(2.0))).to_numpy(dtype=np.float64)
    )
    expected = park.ewm(span=50, adjust=True, min_periods=1).mean().iloc[-1]
    assert feats.parkinson_1d_ema(span=50) == pytest.approx(
        expected, rel=1e-12,
    )


def test_vpin_underfull_returns_none():
    buf = DollarBarBuffer(max_size=100)
    _populate_buffer(40, buf)
    feats = DollarBarFeatures(buf)
    assert feats.latest_raw("vpin") is None
    assert feats.ema("vpin", 10) is None
    assert feats.lag("vpin", 1) is None


def test_vpin_ema_matches_pandas_reference_on_synthetic():
    """VPIN series via _vpin_rolling, then pandas ewm; first 49 entries are NaN."""
    buf = DollarBarBuffer(max_size=100)
    _populate_buffer(70, buf)
    feats = DollarBarFeatures(buf)
    df = buf.as_dataframe()
    buy = df["buy_volume"].to_numpy(dtype=np.float64)
    sell = df["sell_volume"].to_numpy(dtype=np.float64)
    vpin_arr = _vpin_rolling(buy, sell, 50)
    expected = (
        pd.Series(vpin_arr).ewm(span=10, adjust=True, min_periods=1)
        .mean().iloc[-1]
    )
    live = feats.ema("vpin", 10)
    assert live == pytest.approx(expected, rel=1e-12)


def test_cache_invalidates_on_buffer_growth():
    """After append, the next method call sees the new bar."""
    buf = DollarBarBuffer(max_size=10)
    _populate_buffer(3, buf)
    feats = DollarBarFeatures(buf)
    first_latest = feats.latest_raw("vov")
    # Append a bar with very different vov; cache should refresh
    new_bar = _make_synthetic_bar(99, vov=99.0)
    buf.append(new_bar)
    second_latest = feats.latest_raw("vov")
    assert second_latest == pytest.approx(99.0)
    assert second_latest != first_latest


# ── byte-identity vs offline pandas on real 200K trades ──────────────


def _build_buffer_from_parquet(n_trades: int = 200_000) -> DollarBarBuffer:
    if not _PARQUET_PATH.exists():
        pytest.skip(
            f"Binance aggTrades fixture not present at {_PARQUET_PATH}; "
            "this test only runs in dev environments with the data dir."
        )
    raw = pd.read_parquet(_PARQUET_PATH).head(n_trades)
    df = raw.copy()
    df.index = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df = df.sort_index()
    buf = DollarBarBuffer(max_size=2000)
    agg = DollarBarAggregator(threshold=500_000.0, on_bar_close=buf.append)
    for ts, row in df.iterrows():
        agg.add_trade(
            timestamp=ts,
            price=float(row["price"]),
            dollar_volume=float(row["dollar_volume"]),
            is_buyer_maker=bool(row["is_buyer_maker"]),
        )
    return buf


def test_byte_identity_vov_ema50():
    buf = _build_buffer_from_parquet()
    feats = DollarBarFeatures(buf)
    df = buf.as_dataframe()
    expected = df["vov"].ewm(span=50, adjust=True, min_periods=1).mean().iloc[-1]
    assert feats.ema("vov", 50) == pytest.approx(expected, rel=1e-12)


def test_byte_identity_vov_lag3():
    buf = _build_buffer_from_parquet()
    feats = DollarBarFeatures(buf)
    df = buf.as_dataframe()
    expected = float(df["vov"].iloc[-4])
    assert feats.lag("vov", 3) == pytest.approx(expected, rel=1e-12)


def test_byte_identity_imbalance_ema50():
    buf = _build_buffer_from_parquet()
    feats = DollarBarFeatures(buf)
    df = buf.as_dataframe()
    expected = (
        df["imbalance"].ewm(span=50, adjust=True, min_periods=1)
        .mean().iloc[-1]
    )
    assert feats.ema("imbalance", 50) == pytest.approx(expected, rel=1e-12)


def test_byte_identity_duration_s_ema10():
    buf = _build_buffer_from_parquet()
    feats = DollarBarFeatures(buf)
    df = buf.as_dataframe()
    expected = (
        df["duration_s"].ewm(span=10, adjust=True, min_periods=1)
        .mean().iloc[-1]
    )
    assert feats.ema("duration_s", 10) == pytest.approx(expected, rel=1e-12)


def test_byte_identity_duration_s_lag3():
    buf = _build_buffer_from_parquet()
    feats = DollarBarFeatures(buf)
    df = buf.as_dataframe()
    expected = float(df["duration_s"].iloc[-4])
    assert feats.lag("duration_s", 3) == pytest.approx(expected, rel=1e-12)


def test_byte_identity_n_trades_ema50():
    buf = _build_buffer_from_parquet()
    feats = DollarBarFeatures(buf)
    df = buf.as_dataframe()
    expected = (
        df["n_trades"].astype(np.float64)
        .ewm(span=50, adjust=True, min_periods=1).mean().iloc[-1]
    )
    assert feats.ema("n_trades", 50) == pytest.approx(expected, rel=1e-12)


def test_byte_identity_vpin_ema10():
    buf = _build_buffer_from_parquet()
    feats = DollarBarFeatures(buf)
    df = buf.as_dataframe()
    buy = df["buy_volume"].to_numpy(dtype=np.float64)
    sell = df["sell_volume"].to_numpy(dtype=np.float64)
    vpin_arr = _vpin_rolling(buy, sell, 50)
    expected = (
        pd.Series(vpin_arr).ewm(span=10, adjust=True, min_periods=1)
        .mean().iloc[-1]
    )
    assert feats.ema("vpin", 10) == pytest.approx(expected, rel=1e-12)


def test_byte_identity_vpin_lag1():
    buf = _build_buffer_from_parquet()
    feats = DollarBarFeatures(buf)
    df = buf.as_dataframe()
    buy = df["buy_volume"].to_numpy(dtype=np.float64)
    sell = df["sell_volume"].to_numpy(dtype=np.float64)
    vpin_arr = _vpin_rolling(buy, sell, 50)
    expected = float(vpin_arr[-2])
    assert feats.lag("vpin", 1) == pytest.approx(expected, rel=1e-12)


def test_byte_identity_parkinson_1d():
    buf = _build_buffer_from_parquet()
    feats = DollarBarFeatures(buf)
    df = buf.as_dataframe()
    hl = (df["high"] / df["low"]).clip(lower=1.0 + 1e-10)
    park = (np.log(hl) ** 2 / (4.0 * np.log(2.0))).to_numpy(dtype=np.float64)
    if len(park) >= 250:
        expected = float(park[-250:].mean())
        assert feats.parkinson_1d(window=250) == pytest.approx(expected, rel=1e-12)
    else:
        # Buffer not large enough — skip
        pytest.skip(f"buffer has only {len(park)} bars; need >=250")


def test_byte_identity_parkinson_1d_ema50():
    buf = _build_buffer_from_parquet()
    feats = DollarBarFeatures(buf)
    df = buf.as_dataframe()
    hl = (df["high"] / df["low"]).clip(lower=1.0 + 1e-10)
    park = pd.Series(
        (np.log(hl) ** 2 / (4.0 * np.log(2.0))).to_numpy(dtype=np.float64)
    )
    expected = park.ewm(span=50, adjust=True, min_periods=1).mean().iloc[-1]
    assert feats.parkinson_1d_ema(span=50) == pytest.approx(expected, rel=1e-12)
