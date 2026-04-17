"""Tests for BarAggregator — slicing, end-stamped window, eviction."""

from __future__ import annotations

import pandas as pd
import pytest

from deribit_paper.market_data.aggregator import Bar, BarAggregator
from deribit_paper.market_data.buffer import IndexPriceBuffer, Trade, TradeBuffer


def _perp(ts_str: str, trade_id: str) -> Trade:
    return Trade(
        trade_id=trade_id,
        timestamp=pd.Timestamp(ts_str, tz="UTC"),
        instrument_name="BTC-PERPETUAL",
        price=50000.0,
        amount=0.1,
        direction=1,
    )


def _option(ts_str: str, trade_id: str, strike: float = 60000.0) -> Trade:
    return Trade(
        trade_id=trade_id,
        timestamp=pd.Timestamp(ts_str, tz="UTC"),
        instrument_name=f"BTC-28NOV25-{int(strike)}-C",
        price=0.05,
        amount=1.0,
        direction=1,
        strike=strike,
        option_type="C",
        expiry_dt=pd.Timestamp("2025-11-28 08:00:00", tz="UTC"),
        iv=75.0,
        index_price=60000.0,
    )


@pytest.mark.asyncio
async def test_aggregator_end_stamped_window():
    perp = TradeBuffer(name="perp", window_seconds=3600)
    opt = TradeBuffer(name="opt", window_seconds=3600)
    idx = IndexPriceBuffer()

    # Perp trades spanning 12:00 → 12:20 (one bar [12:00, 12:15))
    perp.append(_perp("2026-04-10 12:00:00", "p0"))
    perp.append(_perp("2026-04-10 12:05:00", "p1"))
    perp.append(_perp("2026-04-10 12:14:59", "p2"))
    perp.append(_perp("2026-04-10 12:15:00", "p3_boundary"))  # NOT in this bar
    perp.append(_perp("2026-04-10 12:18:00", "p4"))

    opt.append(_option("2026-04-10 12:07:00", "o0"))
    opt.append(_option("2026-04-10 12:15:00", "o1_boundary"))  # NOT in this bar

    idx.append(pd.Timestamp("2026-04-10 12:14:30", tz="UTC"), 59800.0)
    idx.append(pd.Timestamp("2026-04-10 12:15:00", tz="UTC"), 59900.0)

    agg = BarAggregator(
        bar_minutes=15, perp_buffer=perp, option_buffer=opt, index_buffer=idx,
    )

    received: list[Bar] = []

    async def on_bar(bar: Bar) -> None:
        received.append(bar)

    agg.subscribe(on_bar)

    await agg.on_bar_close(pd.Timestamp("2026-04-10 12:15:00", tz="UTC"))

    assert len(received) == 1
    bar = received[0]
    assert bar.timestamp == pd.Timestamp("2026-04-10 12:15:00", tz="UTC")
    assert bar.bar_minutes == 15
    assert bar.window_start == pd.Timestamp("2026-04-10 12:00:00", tz="UTC")

    # End-stamped half-open window: [12:00, 12:15)
    perp_ids = {t.trade_id for t in bar.perp_trades}
    assert perp_ids == {"p0", "p1", "p2"}  # boundary trade excluded
    opt_ids = {t.trade_id for t in bar.option_trades}
    assert opt_ids == {"o0"}

    # Close-of-bar index uses at_or_before(12:15:00) → latest <= 12:15 (== 12:15)
    assert bar.index_price == 59900.0


@pytest.mark.asyncio
async def test_aggregator_empty_bar_still_emits():
    perp = TradeBuffer(name="perp", window_seconds=3600)
    opt = TradeBuffer(name="opt", window_seconds=3600)
    idx = IndexPriceBuffer()
    agg = BarAggregator(bar_minutes=15, perp_buffer=perp, option_buffer=opt, index_buffer=idx)

    received = []
    async def on_bar(bar): received.append(bar)
    agg.subscribe(on_bar)

    await agg.on_bar_close(pd.Timestamp("2026-04-10 12:15:00", tz="UTC"))

    assert len(received) == 1
    assert received[0].n_perp_trades == 0
    assert received[0].n_option_trades == 0
    assert received[0].index_price is None


@pytest.mark.asyncio
async def test_aggregator_evicts_buffers_after_emission():
    perp = TradeBuffer(name="perp", window_seconds=60)  # 1-min retention
    opt = TradeBuffer(name="opt", window_seconds=60)
    idx = IndexPriceBuffer(window_seconds=60)

    # Old trades that should be evicted
    perp.append(_perp("2026-04-10 12:00:00", "old_p"))
    opt.append(_option("2026-04-10 12:00:00", "old_o"))
    idx.append(pd.Timestamp("2026-04-10 12:00:00", tz="UTC"), 50000.0)

    # Recent trades that should survive
    perp.append(_perp("2026-04-10 12:14:30", "new_p"))
    opt.append(_option("2026-04-10 12:14:30", "new_o"))
    idx.append(pd.Timestamp("2026-04-10 12:14:30", tz="UTC"), 51000.0)

    agg = BarAggregator(bar_minutes=15, perp_buffer=perp, option_buffer=opt, index_buffer=idx)

    received = []
    async def on_bar(bar): received.append(bar)
    agg.subscribe(on_bar)

    await agg.on_bar_close(pd.Timestamp("2026-04-10 12:15:00", tz="UTC"))

    # The bar window is [12:00, 12:15) so BOTH trades are inside the window
    assert received[0].n_perp_trades == 2
    # But after emission, the 1-min eviction kicks in → only the 12:14:30 trades survive
    assert len(perp) == 1
    assert len(opt) == 1


@pytest.mark.asyncio
async def test_aggregator_dataframe_conversions():
    perp = TradeBuffer(name="perp", window_seconds=3600)
    opt = TradeBuffer(name="opt", window_seconds=3600)
    idx = IndexPriceBuffer()

    perp.append(_perp("2026-04-10 12:07:00", "p0"))
    opt.append(_option("2026-04-10 12:08:00", "o0", strike=60000.0))
    opt.append(_option("2026-04-10 12:09:00", "o1", strike=62000.0))

    agg = BarAggregator(bar_minutes=15, perp_buffer=perp, option_buffer=opt, index_buffer=idx)

    received = []
    async def on_bar(bar): received.append(bar)
    agg.subscribe(on_bar)

    await agg.on_bar_close(pd.Timestamp("2026-04-10 12:15:00", tz="UTC"))

    bar = received[0]
    perp_df = bar.perp_dataframe()
    opt_df = bar.option_dataframe()

    assert len(perp_df) == 1
    assert "price" in perp_df.columns
    assert len(opt_df) == 2
    assert set(opt_df["strike"]) == {60000.0, 62000.0}
    assert set(opt_df["option_type"]) == {"C"}


@pytest.mark.asyncio
async def test_aggregator_subscriber_exception_isolated():
    perp = TradeBuffer(name="perp", window_seconds=3600)
    opt = TradeBuffer(name="opt", window_seconds=3600)
    idx = IndexPriceBuffer()
    agg = BarAggregator(bar_minutes=15, perp_buffer=perp, option_buffer=opt, index_buffer=idx)

    calls_a = []
    calls_b = []

    async def bad_sub(bar):
        calls_a.append(bar)
        raise RuntimeError("boom")

    async def good_sub(bar):
        calls_b.append(bar)

    agg.subscribe(bad_sub)
    agg.subscribe(good_sub)

    await agg.on_bar_close(pd.Timestamp("2026-04-10 12:15:00", tz="UTC"))
    assert len(calls_a) == 1
    assert len(calls_b) == 1  # exception from bad_sub must not prevent good_sub
