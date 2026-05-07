"""Round-4 F6/F8 — TradeBuffer.prewarm_from_parquet round-trip tests.

Scope:

* A parquet written in the warm-start schema (``timestamp, price, amount,
  is_buyer_maker, a``) is round-tripped through
  :meth:`TradeBuffer.prewarm_from_parquet` into Trade objects with the
  expected fields.
* Missing files raise :class:`FileNotFoundError` (fail-loud).
* Partial-schema parquets raise :class:`ValueError` listing missing cols.
* ``max_age_seconds`` filter drops stale rows.
* Pre-warmed trades dedup against a subsequent live append carrying the
  same trade_id (so the live engine doesn't double-count the warm-start
  window).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deribit_paper.market_data.buffer import Trade, TradeBuffer


@pytest.fixture()
def warmstart_parquet(tmp_path: Path) -> Path:
    """A 200-trade synthetic warm-start parquet."""
    base_ts = pd.Timestamp("2026-05-07 12:00:00", tz="UTC")
    n = 200
    df = pd.DataFrame({
        "timestamp": pd.date_range(base_ts, periods=n, freq="100ms"),
        "price": np.linspace(60_000.0, 60_500.0, n),
        "amount": np.full(n, 0.5),
        "is_buyer_maker": [i % 3 == 0 for i in range(n)],
        "a": np.arange(1000, 1000 + n, dtype=np.int64),
    })
    out = tmp_path / "spot_warmstart.parquet"
    df.to_parquet(out, index=False)
    return out


def _new_buffer() -> TradeBuffer:
    return TradeBuffer(name="spot", window_seconds=3600.0)


def test_prewarm_basic_round_trip(warmstart_parquet: Path):
    """200 trades load, all fields preserved, dedup cache populated."""
    buf = _new_buffer()
    added = buf.prewarm_from_parquet(warmstart_parquet)
    assert added == 200
    assert len(buf) == 200
    # Smoke-check first and last
    df = buf.to_dataframe().sort_values("timestamp").reset_index(drop=True)
    assert df.loc[0, "price"] == 60_000.0
    assert df.loc[199, "price"] == 60_500.0
    # is_buyer_maker -> direction sign: True (maker buys) -> -1, False -> +1
    assert df.loc[0, "direction"] == -1   # i=0, is_buyer_maker=True
    assert df.loc[1, "direction"] == 1    # i=1, is_buyer_maker=False


def test_prewarm_missing_file_raises(tmp_path: Path):
    buf = _new_buffer()
    with pytest.raises(FileNotFoundError, match="sidecar not found"):
        buf.prewarm_from_parquet(tmp_path / "does-not-exist.parquet")


def test_prewarm_missing_cols_raises(tmp_path: Path):
    """A parquet missing required columns raises ValueError, not silent
    partial load."""
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp.now(tz="UTC")],
        # missing price + amount
    })
    out = tmp_path / "broken.parquet"
    df.to_parquet(out, index=False)
    buf = _new_buffer()
    with pytest.raises(ValueError, match="missing columns"):
        buf.prewarm_from_parquet(out)


def test_prewarm_max_age_drops_stale_rows(warmstart_parquet: Path):
    """max_age_seconds=N drops rows older than now - N."""
    buf = _new_buffer()
    # The fixture's timestamps span ~20s starting at 2026-05-07 12:00.
    # Pretend "now" is 12:00:10 UTC and ask for a 5s window — only the
    # last ~50 trades should remain.
    fake_now = pd.Timestamp("2026-05-07 12:00:10", tz="UTC")
    added = buf.prewarm_from_parquet(
        warmstart_parquet, max_age_seconds=5.0, now=fake_now,
    )
    assert added < 200
    # Surviving trades all have ts >= fake_now - 5s.
    cutoff = fake_now - pd.Timedelta(seconds=5)
    for t in list(buf._trades):  # type: ignore[attr-defined]
        assert t.timestamp >= cutoff


def test_prewarm_dedup_against_live_append(warmstart_parquet: Path):
    """A live append of a trade whose trade_id matches a warm-start row
    must be deduped (returns False, no duplicate in buffer)."""
    buf = _new_buffer()
    buf.prewarm_from_parquet(warmstart_parquet)
    n_before = len(buf)
    # Synthesize a live trade with a colliding trade_id (the prewarm
    # uses ``trade_id_prefix:agg_id`` as the id).
    live_trade = Trade(
        trade_id="warmstart:1000",  # collides with first warm-start row
        timestamp=pd.Timestamp("2026-05-07 12:01:00", tz="UTC"),
        instrument_name="BTCUSDT",
        price=99_999.0,
        amount=1.0,
        direction=1,
    )
    accepted = buf.append(live_trade)
    assert accepted is False, "duplicate trade_id should be rejected"
    assert len(buf) == n_before


def test_prewarm_empty_parquet_warns_and_returns_zero(tmp_path: Path):
    """Empty parquet returns 0, doesn't raise."""
    df = pd.DataFrame({
        "timestamp": pd.to_datetime([], utc=True),
        "price": np.array([], dtype=np.float64),
        "amount": np.array([], dtype=np.float64),
    })
    out = tmp_path / "empty.parquet"
    df.to_parquet(out, index=False)
    buf = _new_buffer()
    added = buf.prewarm_from_parquet(out)
    assert added == 0
    assert len(buf) == 0


def test_prewarm_int_ms_timestamps_are_coerced(tmp_path: Path):
    """A parquet with integer-ms timestamps (Binance Vision schema) is
    coerced to UTC pd.Timestamp on load."""
    df = pd.DataFrame({
        "timestamp": pd.Series([1700_000_000_000, 1700_000_001_000],
                                dtype=np.int64),
        "price": [60_000.0, 60_001.0],
        "amount": [1.0, 2.0],
    })
    out = tmp_path / "int_ms.parquet"
    df.to_parquet(out, index=False)
    buf = _new_buffer()
    added = buf.prewarm_from_parquet(out)
    assert added == 2
    trades = list(buf._trades)  # type: ignore[attr-defined]
    assert all(t.timestamp.tz is not None for t in trades)
    assert trades[0].timestamp == pd.Timestamp(
        1700_000_000_000, unit="ms", tz="UTC",
    )
