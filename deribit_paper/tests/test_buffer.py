"""Tests for TradeBuffer and IndexPriceBuffer."""

from __future__ import annotations

import pandas as pd
import pytest

from deribit_paper.market_data.buffer import IndexPriceBuffer, Trade, TradeBuffer


def _make_trade(ts_str: str, trade_id: str, price: float = 50000.0, amount: float = 0.1) -> Trade:
    return Trade(
        trade_id=trade_id,
        timestamp=pd.Timestamp(ts_str, tz="UTC"),
        instrument_name="BTC-PERPETUAL",
        price=price,
        amount=amount,
        direction=1,
    )


# ─── TradeBuffer ─────────────────────────────────────────────────────

def test_buffer_appends_and_reports_length():
    buf = TradeBuffer(name="perp", window_seconds=3600)
    assert len(buf) == 0
    buf.append(_make_trade("2026-01-01 12:00:00", "t1"))
    buf.append(_make_trade("2026-01-01 12:00:01", "t2"))
    assert len(buf) == 2


def test_buffer_deduplicates_by_trade_id():
    buf = TradeBuffer(name="perp", window_seconds=3600)
    t1 = _make_trade("2026-01-01 12:00:00", "t1")
    assert buf.append(t1) is True
    assert buf.append(t1) is False  # duplicate
    assert len(buf) == 1


def test_buffer_extend_returns_added_count():
    buf = TradeBuffer(name="perp", window_seconds=3600)
    t1 = _make_trade("2026-01-01 12:00:00", "t1")
    t2 = _make_trade("2026-01-01 12:00:01", "t2")
    added = buf.extend([t1, t2, t1])  # t1 is duplicated
    assert added == 2
    assert len(buf) == 2


def test_buffer_evicts_old_trades():
    buf = TradeBuffer(name="perp", window_seconds=60)  # 1-min window
    buf.append(_make_trade("2026-01-01 12:00:00", "old"))
    buf.append(_make_trade("2026-01-01 12:00:30", "mid"))
    buf.append(_make_trade("2026-01-01 12:00:58", "recent"))

    now = pd.Timestamp("2026-01-01 12:01:30", tz="UTC")  # cutoff = 12:00:30
    evicted = buf.evict_older(now)
    assert evicted == 1   # only "old" (at 12:00:00) is strictly older
    assert len(buf) == 2


def test_buffer_slice_window_half_open():
    buf = TradeBuffer(name="perp", window_seconds=3600)
    buf.append(_make_trade("2026-01-01 12:00:00", "t0"))
    buf.append(_make_trade("2026-01-01 12:07:00", "t1"))
    buf.append(_make_trade("2026-01-01 12:14:59", "t2"))
    buf.append(_make_trade("2026-01-01 12:15:00", "t3_boundary"))
    buf.append(_make_trade("2026-01-01 12:20:00", "t4"))

    # Half-open [12:00, 12:15) — t3_boundary must NOT be included
    start = pd.Timestamp("2026-01-01 12:00:00", tz="UTC")
    end = pd.Timestamp("2026-01-01 12:15:00", tz="UTC")
    sliced = buf.slice_window(start, end)
    ids = [t.trade_id for t in sliced]
    assert ids == ["t0", "t1", "t2"]


def test_buffer_to_dataframe_schema():
    buf = TradeBuffer(name="opt", window_seconds=3600)
    t = Trade(
        trade_id="t1",
        timestamp=pd.Timestamp("2026-01-01 12:00:00", tz="UTC"),
        instrument_name="BTC-28NOV25-60000-C",
        price=0.05,
        amount=1.0,
        direction=1,
        strike=60000.0,
        option_type="C",
        expiry_dt=pd.Timestamp("2025-11-28 08:00:00", tz="UTC"),
        iv=75.0,
        index_price=59500.0,
    )
    buf.append(t)
    df = buf.to_dataframe()
    assert len(df) == 1
    for col in ("timestamp", "strike", "option_type", "expiry_dt", "iv"):
        assert col in df.columns
    assert df["strike"].iloc[0] == 60000.0
    assert df["option_type"].iloc[0] == "C"


def test_buffer_to_dataframe_empty():
    buf = TradeBuffer(name="perp", window_seconds=3600)
    df = buf.to_dataframe()
    assert df.empty
    assert "timestamp" in df.columns


def test_buffer_max_len_backstop():
    buf = TradeBuffer(name="perp", window_seconds=3600, max_len=3)
    for i in range(5):
        buf.append(_make_trade(f"2026-01-01 12:00:0{i}", f"t{i}"))
    assert len(buf) == 3  # hard cap


def test_buffer_dedup_set_bounded():
    """The dedup set must not grow unboundedly."""
    buf = TradeBuffer(name="perp", window_seconds=3600)
    # The private deque has maxlen = _DEDUP_RECENT_MAX; set mirrors the deque
    from deribit_paper.market_data.buffer import _DEDUP_RECENT_MAX
    # Add more than the cap
    n = _DEDUP_RECENT_MAX + 100
    for i in range(n):
        buf.append(_make_trade("2026-01-01 12:00:00", f"t{i}"))
    # Dedup set should be at or below the cap
    assert len(buf._seen_set) <= _DEDUP_RECENT_MAX


# ─── IndexPriceBuffer ────────────────────────────────────────────────

def test_index_buffer_latest():
    buf = IndexPriceBuffer()
    assert buf.latest() is None
    buf.append(pd.Timestamp("2026-01-01 12:00:00", tz="UTC"), 50000.0)
    buf.append(pd.Timestamp("2026-01-01 12:00:01", tz="UTC"), 50100.0)
    assert buf.latest() == 50100.0


def test_index_buffer_at_or_before():
    buf = IndexPriceBuffer()
    buf.append(pd.Timestamp("2026-01-01 12:00:00", tz="UTC"), 50000.0)
    buf.append(pd.Timestamp("2026-01-01 12:05:00", tz="UTC"), 50500.0)
    buf.append(pd.Timestamp("2026-01-01 12:10:00", tz="UTC"), 51000.0)

    # Exact match → returns that price
    assert buf.at_or_before(pd.Timestamp("2026-01-01 12:05:00", tz="UTC")) == 50500.0
    # Between two samples → returns the earlier one
    assert buf.at_or_before(pd.Timestamp("2026-01-01 12:07:00", tz="UTC")) == 50500.0
    # Before everything → None
    assert buf.at_or_before(pd.Timestamp("2026-01-01 11:00:00", tz="UTC")) is None
    # After everything → latest
    assert buf.at_or_before(pd.Timestamp("2026-01-01 13:00:00", tz="UTC")) == 51000.0


def test_index_buffer_eviction():
    buf = IndexPriceBuffer(window_seconds=60)
    buf.append(pd.Timestamp("2026-01-01 12:00:00", tz="UTC"), 50000.0)
    buf.append(pd.Timestamp("2026-01-01 12:00:30", tz="UTC"), 50100.0)
    buf.append(pd.Timestamp("2026-01-01 12:00:58", tz="UTC"), 50200.0)
    evicted = buf.evict_older(pd.Timestamp("2026-01-01 12:01:30", tz="UTC"))
    assert evicted == 1  # only the 12:00:00 sample is strictly older than cutoff
    assert len(buf) == 2
