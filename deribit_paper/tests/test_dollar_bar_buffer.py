"""Tests for :class:`DollarBarBuffer` — bounded ring of dollar bars.

F3 Tier 2 (2026-05-07): the buffer is the rolling-history view that
feature handlers read from. Unit tests cover eviction, schema, and
the offline-byte-identity contract on the VPIN convenience helper.

Byte-identity (the load-bearing test): a 200K-trade slice of
``data/binance-aggTrades-parquet/BTCUSDT/2026-04.parquet`` is run
through the live ``DollarBarAggregator`` + ``DollarBarBuffer`` and
``buffer.compute_vpin(50)`` is asserted against the offline
``_vpin_rolling`` reference on the same buy/sell volume arrays.
Tolerance: 1e-12 relative — pure float64 arithmetic on identical
inputs, no algorithmic divergence allowed.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.dollar_bar import DollarBar, DollarBarAggregator
from deribit_paper.features.dollar_bar_buffer import DollarBarBuffer


def _make_bar(
    ts: str,
    close: float = 50_000.0,
    *,
    buy: float = 300_000.0,
    sell: float = 200_000.0,
    n_trades: int = 50,
) -> DollarBar:
    """Construct a synthetic DollarBar for unit tests (non-parquet path)."""
    return DollarBar(
        timestamp=pd.Timestamp(ts, tz="UTC"),
        open=close - 10.0,
        close=close,
        high=close + 5.0,
        low=close - 15.0,
        duration_s=12.5,
        n_trades=n_trades,
        dollar_volume=buy + sell,
        buy_volume=buy,
        sell_volume=sell,
        vov=1.5e-4,
        imbalance=(buy - sell) / max(buy + sell, 1.0),
        log_return=0.0001,
        rv_bar=1e-8,
    )


# ── 1. Eviction at max_size ────────────────────────────────────────────────


def test_buffer_evicts_at_max_size():
    """Beyond ``max_size`` the oldest bar is evicted (FIFO)."""
    buf = DollarBarBuffer(max_size=3)
    for i in range(5):
        buf.append(_make_bar(f"2026-05-07 12:00:{i:02d}", close=50_000.0 + i))
    assert len(buf) == 3
    # Newest 3: indices 2, 3, 4
    bars = list(buf)
    assert bars[0].close == 50_002.0
    assert bars[1].close == 50_003.0
    assert bars[2].close == 50_004.0


def test_buffer_max_size_zero_raises():
    with pytest.raises(ValueError, match="max_size must be positive"):
        DollarBarBuffer(max_size=0)


# ── 2. Empty buffer behaviour ─────────────────────────────────────────────


def test_empty_buffer_latest_is_none():
    buf = DollarBarBuffer(max_size=10)
    assert len(buf) == 0
    assert buf.latest_bar() is None
    assert buf.tail(5) == []
    df = buf.as_dataframe()
    assert df.empty
    # Schema preserved on empty
    expected_cols = (
        "close", "open", "high", "low", "duration_s", "n_trades",
        "dollar_volume", "buy_volume", "sell_volume", "vov",
        "imbalance", "log_return", "rv_bar",
    )
    assert tuple(df.columns) == expected_cols
    assert df.index.name == "timestamp"
    assert df.index.tz is not None and str(df.index.tz) == "UTC"


# ── 3. tail() semantics ────────────────────────────────────────────────────


def test_tail_returns_last_n_bars_in_order():
    buf = DollarBarBuffer(max_size=10)
    for i in range(5):
        buf.append(_make_bar(f"2026-05-07 12:00:{i:02d}", close=50_000.0 + i))
    last_three = buf.tail(3)
    assert len(last_three) == 3
    # Newest at end
    assert last_three[0].close == 50_002.0
    assert last_three[2].close == 50_004.0

    # n > len → returns everything
    everything = buf.tail(100)
    assert len(everything) == 5

    # n <= 0 → empty list
    assert buf.tail(0) == []
    assert buf.tail(-3) == []


# ── 4. as_dataframe schema + ordering ─────────────────────────────────────


def test_as_dataframe_schema_and_ordering():
    buf = DollarBarBuffer(max_size=10)
    bars = [
        _make_bar("2026-05-07 12:00:00", close=50_000.0, buy=400_000.0, sell=100_000.0),
        _make_bar("2026-05-07 12:00:05", close=50_010.0, buy=200_000.0, sell=300_000.0),
        _make_bar("2026-05-07 12:00:10", close=50_020.0, buy=350_000.0, sell=150_000.0),
    ]
    for b in bars:
        buf.append(b)

    df = buf.as_dataframe()
    assert len(df) == 3
    # Index ordering matches insertion order
    assert df.index[0] == bars[0].timestamp
    assert df.index[-1] == bars[-1].timestamp
    # UTC index
    assert str(df.index.tz) == "UTC"
    # Per-row values match
    assert df.iloc[0]["close"] == 50_000.0
    assert df.iloc[1]["close"] == 50_010.0
    assert df.iloc[2]["buy_volume"] == 350_000.0
    assert df.iloc[2]["sell_volume"] == 150_000.0
    # Dtypes: float64 for numerics, int64 for n_trades
    assert df["close"].dtype == np.float64
    assert df["n_trades"].dtype == np.int64


def test_latest_bar_returns_most_recent():
    buf = DollarBarBuffer(max_size=10)
    b1 = _make_bar("2026-05-07 12:00:00", close=50_000.0)
    b2 = _make_bar("2026-05-07 12:00:05", close=50_010.0)
    buf.append(b1)
    buf.append(b2)
    assert buf.latest_bar() is b2


# ── 5. compute_vpin returns None when underfull ────────────────────────────


def test_compute_vpin_underfull_returns_none():
    """Buffer with fewer than ``window`` bars → None (offline NaN equivalent)."""
    buf = DollarBarBuffer(max_size=100)
    # 49 bars, window=50 → still underfull
    for i in range(49):
        buf.append(_make_bar(f"2026-05-07 12:{i // 60:02d}:{i % 60:02d}"))
    assert buf.compute_vpin(window=50) is None


def test_compute_vpin_window_zero_raises():
    buf = DollarBarBuffer(max_size=10)
    with pytest.raises(ValueError, match="window must be positive"):
        buf.compute_vpin(window=0)


# ── 6. Byte-identity VPIN test on real Binance Spot data ─────────────────


_PARQUET_PATH = Path(
    "data/binance-aggTrades-parquet/BTCUSDT/2026-04.parquet"
)
_REFERENCE_THRESHOLD = 500_000.0


def test_compute_vpin_byte_identity_to_offline():
    """Byte-identity test: ``buffer.compute_vpin(50)`` must equal
    ``_vpin_rolling(buy, sell, 50)[-1]`` to within 1e-12 relative
    tolerance, where the inputs come from a 200K-trade slice of the
    real Binance Vision parquet replayed through the live aggregator.

    Skip when the parquet isn't present (deploy contexts without the
    data/ dir).
    """
    if not _PARQUET_PATH.exists():
        pytest.skip(
            f"Binance aggTrades fixture not present at {_PARQUET_PATH}; "
            "this test only runs in dev environments with the data dir."
        )
    # Load 200K trades — same chunk as the Tier 1 byte-identity test.
    raw = pd.read_parquet(_PARQUET_PATH).head(200_000)
    df = raw.copy()
    df.index = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df = df.sort_index()
    df = df[["price", "quantity", "dollar_volume", "is_buyer_maker"]]

    # Drive the buffer through the aggregator.
    buf = DollarBarBuffer(max_size=10_000)
    agg = DollarBarAggregator(
        threshold=_REFERENCE_THRESHOLD,
        on_bar_close=buf.append,
    )
    for ts, row in df.iterrows():
        agg.add_trade(
            timestamp=ts,
            price=float(row["price"]),
            dollar_volume=float(row["dollar_volume"]),
            is_buyer_maker=bool(row["is_buyer_maker"]),
        )

    n_bars = len(buf)
    assert n_bars > 50, f"need at least 51 bars to compute VPIN(50), got {n_bars}"

    # Live: buffer.compute_vpin(50)
    live_vpin = buf.compute_vpin(window=50)
    assert live_vpin is not None
    assert math.isfinite(live_vpin)

    # Offline reference: _vpin_rolling on the same buy/sell arrays
    from vol_model.features.microstructure import _vpin_rolling
    buy = np.array([b.buy_volume for b in buf], dtype=np.float64)
    sell = np.array([b.sell_volume for b in buf], dtype=np.float64)
    offline_vpin = _vpin_rolling(buy, sell, 50)[-1]

    # 1e-12 relative tolerance — the path from buffer to compute_vpin
    # is just a fromiter + the same _vpin_rolling call, so identity
    # should hold modulo float64 ULP from the fromiter ordering.
    assert math.isfinite(offline_vpin)
    abs_diff = abs(live_vpin - offline_vpin)
    rel_diff = abs_diff / max(abs(offline_vpin), 1e-30)
    assert abs_diff < 1e-12 or rel_diff < 1e-12, (
        f"byte-identity VPIN mismatch: live={live_vpin} offline={offline_vpin} "
        f"abs_diff={abs_diff} rel_diff={rel_diff}"
    )
