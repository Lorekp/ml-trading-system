"""S-1c (parity-r3 cycle 6 — 2026-05-08) — funding_basis live handler.

Locks the 5 ``funding_basis_*`` handlers' integration with
``FundingRateProvider`` via ``AssemblerContext.funding_provider``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deribit_paper.market_data.funding import FundingRateProvider
from deribit_paper.signal.feature_assembler import (
    _LIVE_HANDLERS,
    AssemblerContext,
)


def _make_parquet(tmp_path: Path, rows: list[tuple[str, float]]) -> Path:
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz="UTC") for t, _ in rows], name="fundingTime")
    df = pd.DataFrame({"fundingRate": [r for _, r in rows]}, index=idx)
    p = tmp_path / "funding.parquet"
    df.to_parquet(p)
    return p


def _ctx(prov: FundingRateProvider | None, *, baseline: float | None = None) -> AssemblerContext:
    bar = type("StubBar", (), {})()
    bar.timestamp = pd.Timestamp("2026-05-01 12:00", tz="UTC")
    bar.perp_trades = []
    bar.n_perp_trades = 0
    series_idx = ["funding_basis", "funding_basis_lag1", "funding_basis_lag3",
                  "funding_basis_ema10", "funding_basis_ema50"]
    if baseline is None:
        # NaN baseline (per F9 — propagate to fail-fast guard).
        baseline_row = pd.Series({k: float("nan") for k in series_idx})
    else:
        baseline_row = pd.Series({k: float(baseline) for k in series_idx})
    return AssemblerContext(
        bar=bar,
        bar_tsrv=None,
        svi_calibrator=None,
        baseline_row=baseline_row,
        funding_provider=prov,
    )


def test_funding_basis_raw_handler_reads_provider(tmp_path):
    p = _make_parquet(tmp_path, [
        ("2026-05-01 00:00", 0.0001),
        ("2026-05-01 08:00", 0.0099),
    ])
    prov = FundingRateProvider(parquet_path=p, rest_fetcher=lambda: None)
    prov.advance(pd.Timestamp("2026-05-01 12:00", tz="UTC"))
    ctx = _ctx(prov)
    assert _LIVE_HANDLERS["funding_basis"](ctx) == pytest.approx(0.0099)


def test_funding_basis_lag1_handler(tmp_path):
    p = _make_parquet(tmp_path, [
        ("2026-05-01 00:00", 0.0001),
        ("2026-05-01 08:00", 0.0099),
    ])
    prov = FundingRateProvider(parquet_path=p, rest_fetcher=lambda: None)
    prov.advance(pd.Timestamp("2026-05-01 04:00", tz="UTC"))   # 0.0001
    prov.advance(pd.Timestamp("2026-05-01 12:00", tz="UTC"))   # 0.0099
    ctx = _ctx(prov)
    assert _LIVE_HANDLERS["funding_basis_lag1"](ctx) == pytest.approx(0.0001)


def test_funding_basis_lag3_handler(tmp_path):
    p = _make_parquet(tmp_path, [("2026-05-01 00:00", 0.0001)])
    prov = FundingRateProvider(parquet_path=p, rest_fetcher=lambda: None)
    for h in (0, 4, 8, 12):
        prov.advance(pd.Timestamp(f"2026-05-01 {h:02d}:00", tz="UTC"))
    ctx = _ctx(prov)
    # 4 advances; lag3 = first one
    assert _LIVE_HANDLERS["funding_basis_lag3"](ctx) == pytest.approx(0.0001)


def test_funding_basis_ema10_handler_matches_inline(tmp_path):
    p = _make_parquet(tmp_path, [
        ("2026-05-01 00:00", 0.0001),
        ("2026-05-01 08:00", 0.0005),
        ("2026-05-01 16:00", 0.0003),
    ])
    prov = FundingRateProvider(parquet_path=p, rest_fetcher=lambda: None)
    timestamps = pd.date_range("2026-05-01 04:00", periods=12, freq="2h", tz="UTC")
    for ts in timestamps:
        prov.advance(ts)
    snap = prov.history_snapshot()
    series = pd.Series([v for _, v in snap])
    expected = series.ewm(span=10, adjust=True, min_periods=1).mean().iloc[-1]
    ctx = _ctx(prov)
    np.testing.assert_almost_equal(
        _LIVE_HANDLERS["funding_basis_ema10"](ctx), expected, decimal=12,
    )


def test_funding_basis_ema50_handler_distinct_from_ema10(tmp_path):
    p = _make_parquet(tmp_path, [("2026-05-01 00:00", 0.0001)])
    prov = FundingRateProvider(parquet_path=p, rest_fetcher=lambda: None)
    rng = np.random.default_rng(0)
    base = pd.Timestamp("2026-05-01 04:00", tz="UTC")
    for i in range(60):
        ts = base + pd.Timedelta(hours=2 * i)
        # Manually inject varying parquet observations by overriding the
        # ffill via a force_rest:
        from deribit_paper.market_data.funding import FundingObservation
        prov._force_rest_observation(FundingObservation(
            funding_time=ts,
            funding_rate=float(0.0001 * rng.standard_normal()),
        ))
        prov.advance(ts)
    ctx = _ctx(prov)
    e10 = _LIVE_HANDLERS["funding_basis_ema10"](ctx)
    e50 = _LIVE_HANDLERS["funding_basis_ema50"](ctx)
    assert np.isfinite(e10) and np.isfinite(e50)
    assert e10 != pytest.approx(e50, abs=1e-9)


def test_funding_basis_falls_back_to_baseline_when_provider_none():
    """No provider → handler returns the baseline_row value (NaN)."""
    ctx = _ctx(prov=None)
    val = _LIVE_HANDLERS["funding_basis"](ctx)
    assert np.isnan(val)


def test_funding_basis_baseline_value_returned_when_provider_none_with_baseline():
    """Baseline row carries a finite value → fallback returns it."""
    ctx = _ctx(prov=None, baseline=0.00042)
    val = _LIVE_HANDLERS["funding_basis"](ctx)
    assert val == pytest.approx(0.00042, abs=1e-15)


def test_funding_basis_returns_baseline_when_provider_history_empty():
    """Provider with no advances yet → handler falls back to baseline."""
    prov = FundingRateProvider(parquet_path=None, rest_fetcher=lambda: None)
    # No advance() called → history empty → latest() returns None
    ctx = _ctx(prov, baseline=0.123)
    val = _LIVE_HANDLERS["funding_basis"](ctx)
    assert val == pytest.approx(0.123, abs=1e-15)


def test_funding_basis_byte_equal_to_offline_ffill(tmp_path):
    """Round-trip: parquet → live provider → handler → equals
    offline-style ``fundingRate.reindex(bar_grid).ffill()`` lookup."""
    rows = [
        ("2026-05-01 00:00", 0.0001),
        ("2026-05-01 08:00", 0.0002),
        ("2026-05-01 16:00", 0.0003),
        ("2026-05-02 00:00", 0.0004),
        ("2026-05-02 08:00", 0.0005),
    ]
    p = _make_parquet(tmp_path, rows)
    prov = FundingRateProvider(parquet_path=p, rest_fetcher=lambda: None)
    bar_ts = pd.Timestamp("2026-05-01 18:30", tz="UTC")
    prov.advance(bar_ts)
    # Offline-style: latest funding obs with funding_time <= bar_ts
    fr = pd.Series({pd.Timestamp(t, tz="UTC"): r for t, r in rows})
    offline_val = float(fr[fr.index <= bar_ts].iloc[-1])
    ctx = _ctx(prov)
    live_val = _LIVE_HANDLERS["funding_basis"](ctx)
    assert live_val == pytest.approx(offline_val, abs=1e-15)


def test_funding_basis_handler_count_in_registry():
    """Lock: 5 funding_basis_* handlers registered after Phase 2."""
    matched = [k for k in _LIVE_HANDLERS if k.startswith("funding_basis")]
    assert sorted(matched) == [
        "funding_basis", "funding_basis_ema10", "funding_basis_ema50",
        "funding_basis_lag1", "funding_basis_lag3",
    ]
