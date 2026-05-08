"""F3 Tier 4 tests: warm-start sidecar + DollarBarBuffer.prewarm_from_parquet.

The sidecar shipped at ``vol_arb/output/BTC_dollar_bar_cache.parquet``
contains the last ~300 dollar bars in the offline schema. The live
engine pre-loads this at startup so parkinson_1d (250-bar window) and
EMA50 features can fire on the FIRST live bar instead of waiting 37+
minutes for live trades to fill the buffer.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.dollar_bar import DollarBar, DollarBarAggregator
from deribit_paper.features.dollar_bar_buffer import DollarBarBuffer


_PARQUET_PATH = Path("data/binance-aggTrades-parquet/BTCUSDT/2026-04.parquet")


def _make_synthetic_bar(i: int) -> DollarBar:
    ts = pd.Timestamp("2026-05-07 00:00:00", tz="UTC") + pd.Timedelta(seconds=30 * i)
    return DollarBar(
        timestamp=ts,
        open=60_000.0 + i,
        close=60_000.0 + i + 1.0,
        high=60_100.0 + i,
        low=59_900.0 + i,
        duration_s=30.0 + i,
        n_trades=100 + i,
        dollar_volume=500_000.0 + i,
        buy_volume=250_000.0 + 1000.0 * i,
        sell_volume=250_000.0 + 500.0 * i,
        vov=0.001 + 0.0001 * i,
        imbalance=0.05 + 0.001 * i,
        log_return=0.0001 if i > 0 else float("nan"),
        rv_bar=0.0001**2 if i > 0 else float("nan"),
    )


# ── prewarm_from_parquet tests ───────────────────────────────────────


def test_prewarm_loads_synthetic_sidecar(tmp_path):
    """Round-trip: synthetic buffer → as_dataframe → parquet → prewarm → buffer."""
    src = DollarBarBuffer(max_size=50)
    for i in range(20):
        src.append(_make_synthetic_bar(i))
    out = tmp_path / "sidecar.parquet"
    src.as_dataframe().to_parquet(out)

    dst = DollarBarBuffer(max_size=50)
    n = dst.prewarm_from_parquet(out)
    assert n == 20
    assert len(dst) == 20

    src_last = src.latest_bar()
    dst_last = dst.latest_bar()
    assert dst_last is not None
    assert dst_last.timestamp == src_last.timestamp
    assert dst_last.close == pytest.approx(src_last.close)
    assert dst_last.duration_s == pytest.approx(src_last.duration_s)
    assert dst_last.n_trades == src_last.n_trades
    assert dst_last.vov == pytest.approx(src_last.vov)


def test_prewarm_missing_file_returns_zero(tmp_path):
    """Fail-soft: missing sidecar → return 0, no raise."""
    buf = DollarBarBuffer(max_size=50)
    n = buf.prewarm_from_parquet(tmp_path / "does_not_exist.parquet")
    assert n == 0
    assert len(buf) == 0


def test_prewarm_corrupted_file_returns_zero(tmp_path):
    """Defensive: unreadable parquet → 0, no raise."""
    bogus = tmp_path / "bogus.parquet"
    bogus.write_bytes(b"not a parquet file")
    buf = DollarBarBuffer(max_size=50)
    n = buf.prewarm_from_parquet(bogus)
    assert n == 0


def test_prewarm_missing_columns_returns_zero(tmp_path):
    """Schema validation: parquet missing required columns → fail-soft."""
    out = tmp_path / "incomplete.parquet"
    df = pd.DataFrame({"close": [1.0, 2.0]}, index=pd.DatetimeIndex(
        ["2026-05-07", "2026-05-08"], tz="UTC", name="timestamp",
    ))
    df.to_parquet(out)
    buf = DollarBarBuffer(max_size=50)
    n = buf.prewarm_from_parquet(out)
    assert n == 0


def test_prewarm_respects_max_size_eviction(tmp_path):
    """Loading more bars than max_size keeps only the most recent."""
    src = DollarBarBuffer(max_size=100)
    for i in range(50):
        src.append(_make_synthetic_bar(i))
    out = tmp_path / "sidecar.parquet"
    src.as_dataframe().to_parquet(out)

    dst = DollarBarBuffer(max_size=20)  # smaller than the 50-bar sidecar
    n = dst.prewarm_from_parquet(out)
    assert n == 50
    assert len(dst) == 20  # evicted to max_size
    # Most recent bars retained; oldest dropped
    assert dst.latest_bar().n_trades == src.latest_bar().n_trades


# ── shipped sidecar parquet ─────────────────────────────────────────


_SHIPPED_SIDECAR = Path("vol_arb/output/BTC_dollar_bar_cache.parquet")


def test_shipped_sidecar_has_expected_schema():
    """The committed sidecar must match the buffer's expected schema."""
    if not _SHIPPED_SIDECAR.exists():
        pytest.skip(f"Sidecar {_SHIPPED_SIDECAR} not present (regenerate via script)")
    df = pd.read_parquet(_SHIPPED_SIDECAR)
    expected_cols = {
        "close", "open", "high", "low", "duration_s", "n_trades",
        "dollar_volume", "buy_volume", "sell_volume",
        "vov", "imbalance", "log_return", "rv_bar",
    }
    assert set(df.columns) == expected_cols
    assert df.index.tz is not None
    assert df["n_trades"].dtype == np.int64
    # S-1d (parity-r3 cycle 7 — 2026-05-08): cache extended from
    # ~300 bars (~1.6h) to 600000 bars (~35 days). 30+ days is
    # required so the trailing-30-day ``bpd`` estimate (offline
    # ``avg_bars_per_day``) matches offline byte-equivalent — needed
    # for tsrv_mean_1d/1w byte-equal closure of C1.
    assert 500_000 <= len(df) <= 1_000_000


def test_shipped_sidecar_loads_into_buffer():
    """End-to-end: shipped sidecar → buffer → 300 bars ready for handlers."""
    if not _SHIPPED_SIDECAR.exists():
        pytest.skip(f"Sidecar {_SHIPPED_SIDECAR} not present")
    buf = DollarBarBuffer(max_size=300)
    n = buf.prewarm_from_parquet(_SHIPPED_SIDECAR)
    assert n > 0
    assert len(buf) == min(n, 300)
    last = buf.latest_bar()
    assert last is not None
    assert last.n_trades > 0
    assert math.isfinite(last.dollar_volume)
    assert last.dollar_volume > 0


# ── byte-identity round-trip on real data ────────────────────────────


def test_round_trip_byte_identity_real_trades():
    """Aggregator → buffer.as_dataframe → parquet → prewarm → DollarBars match.

    Confirms the prewarm path reconstructs DollarBar objects byte-identical
    (within float64 precision) to the originals emitted by the aggregator.
    """
    if not _PARQUET_PATH.exists():
        pytest.skip(f"Binance fixture not present at {_PARQUET_PATH}")

    raw = pd.read_parquet(_PARQUET_PATH).head(50_000)
    df = raw.copy()
    df.index = pd.to_datetime(df["transact_time"], unit="ms", utc=True)
    df = df.sort_index()

    src = DollarBarBuffer(max_size=2000)
    agg = DollarBarAggregator(threshold=500_000.0, on_bar_close=src.append)
    for ts, row in df.iterrows():
        agg.add_trade(
            timestamp=ts,
            price=float(row["price"]),
            dollar_volume=float(row["dollar_volume"]),
            is_buyer_maker=bool(row["is_buyer_maker"]),
        )
    assert len(src) > 0
    src_bars = list(src)

    # Round-trip via parquet
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".parquet", delete=False) as f:
        out_path = Path(f.name)
    try:
        src.as_dataframe().to_parquet(out_path)
        dst = DollarBarBuffer(max_size=2000)
        n = dst.prewarm_from_parquet(out_path)
    finally:
        out_path.unlink(missing_ok=True)

    assert n == len(src)
    dst_bars = list(dst)
    assert len(src_bars) == len(dst_bars)
    for src_bar, dst_bar in zip(src_bars, dst_bars):
        assert src_bar.timestamp == dst_bar.timestamp
        for field in (
            "open", "close", "high", "low",
            "duration_s", "dollar_volume", "buy_volume", "sell_volume",
            "vov", "imbalance",
        ):
            assert getattr(dst_bar, field) == pytest.approx(
                getattr(src_bar, field), rel=1e-12,
            ), f"{field} mismatch at {src_bar.timestamp}"
        assert src_bar.n_trades == dst_bar.n_trades
        # log_return / rv_bar can be NaN for the first bar
        if math.isnan(src_bar.log_return):
            assert math.isnan(dst_bar.log_return)
        else:
            assert dst_bar.log_return == pytest.approx(src_bar.log_return, rel=1e-12)
