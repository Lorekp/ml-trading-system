"""S-1c (parity-r3 cycle 6 — 2026-05-08) — funding-rate live provider.

Locks the two-path live funding mechanism: REST refresh (Path A,
default Binance ``premiumIndex``) + offline parquet fallback (Path B).
The provider exposes per-bar latest / lag / EMA accessors backed by a
bar-grid history deque so the 5 ``funding_basis_*`` TransVol input
columns can be computed live byte-equivalent to offline.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deribit_paper.market_data.funding import (
    FundingObservation,
    FundingRateProvider,
)


# ── Path B (parquet) loader ────────────────────────────────────────


def _make_parquet(tmp_path: Path, rows: list[tuple[str, float]]) -> Path:
    idx = pd.DatetimeIndex([pd.Timestamp(t, tz="UTC") for t, _ in rows], name="fundingTime")
    df = pd.DataFrame({
        "fundingRate": [r for _, r in rows],
    }, index=idx)
    p = tmp_path / "funding.parquet"
    df.to_parquet(p)
    return p


def test_parquet_load_provides_ffill_value(tmp_path):
    """Latest funding rate at-or-before the queried timestamp."""
    p = _make_parquet(tmp_path, [
        ("2026-05-01 00:00", 0.0001),
        ("2026-05-01 08:00", 0.0002),
        ("2026-05-01 16:00", 0.0003),
    ])
    prov = FundingRateProvider(parquet_path=p, rest_fetcher=lambda: None)
    # Bar between 08:00 and 16:00 → ffilled value is 0.0002
    val = prov.advance(pd.Timestamp("2026-05-01 12:30", tz="UTC"))
    assert val == pytest.approx(0.0002, abs=1e-15)


def test_parquet_load_default_when_no_observations_le_ts(tmp_path):
    """Bar earlier than any funding obs → default 0.0."""
    p = _make_parquet(tmp_path, [
        ("2026-05-01 16:00", 0.0003),
    ])
    prov = FundingRateProvider(parquet_path=p, rest_fetcher=lambda: None)
    val = prov.advance(pd.Timestamp("2026-05-01 00:00", tz="UTC"))
    assert val == 0.0


def test_parquet_path_disabled_when_file_missing(tmp_path):
    """Missing file → Path B disabled, default returned."""
    prov = FundingRateProvider(
        parquet_path=tmp_path / "missing.parquet",
        rest_fetcher=lambda: None,
    )
    val = prov.advance(pd.Timestamp("2026-05-01 12:00", tz="UTC"))
    assert val == 0.0


# ── Path A (REST) refresh + TTL ────────────────────────────────────


def test_rest_observation_overrides_parquet_when_newer(tmp_path):
    """REST has a fresher fundingTime → it wins the ffill."""
    p = _make_parquet(tmp_path, [("2026-05-01 00:00", 0.0001)])
    rest_obs = FundingObservation(
        funding_time=pd.Timestamp("2026-05-01 08:00", tz="UTC"),
        funding_rate=0.0099,
    )
    prov = FundingRateProvider(parquet_path=p, rest_fetcher=lambda: rest_obs)
    val = prov.advance(pd.Timestamp("2026-05-01 12:00", tz="UTC"))
    assert val == pytest.approx(0.0099, abs=1e-15)


def test_rest_ttl_skips_extra_fetches(tmp_path):
    """Within the TTL window the closure must not be invoked again."""
    fetch_count = [0]

    def _counted_fetch():
        fetch_count[0] += 1
        return FundingObservation(
            funding_time=pd.Timestamp("2026-05-01 08:00", tz="UTC"),
            funding_rate=0.0050,
        )

    prov = FundingRateProvider(
        parquet_path=None,
        rest_fetcher=_counted_fetch,
        rest_ttl_seconds=10.0,
    )
    prov.advance(pd.Timestamp("2026-05-01 09:00", tz="UTC"))
    prov.advance(pd.Timestamp("2026-05-01 09:15", tz="UTC"))
    prov.advance(pd.Timestamp("2026-05-01 09:30", tz="UTC"))
    assert fetch_count[0] == 1  # one fetch within TTL


def test_rest_returns_none_falls_back_to_parquet(tmp_path):
    """REST fetcher returns None (network error sim) → parquet wins."""
    p = _make_parquet(tmp_path, [("2026-05-01 00:00", 0.0007)])
    prov = FundingRateProvider(parquet_path=p, rest_fetcher=lambda: None)
    val = prov.advance(pd.Timestamp("2026-05-01 12:00", tz="UTC"))
    assert val == pytest.approx(0.0007, abs=1e-15)


# ── Bar-grid history (lag / EMA) ───────────────────────────────────


def test_advance_appends_to_bar_history(tmp_path):
    p = _make_parquet(tmp_path, [
        ("2026-05-01 00:00", 0.0001),
        ("2026-05-01 08:00", 0.0002),
        ("2026-05-01 16:00", 0.0003),
    ])
    prov = FundingRateProvider(parquet_path=p, rest_fetcher=lambda: None)
    for h in (4, 8, 12, 16, 20):
        prov.advance(pd.Timestamp(f"2026-05-01 {h:02d}:00", tz="UTC"))
    assert prov.history_size == 5
    snap = prov.history_snapshot()
    # Bar at 04:00 inherits 00:00's rate; 08:00 picks 08:00's value; etc.
    assert [round(v, 6) for _, v in snap] == [0.0001, 0.0002, 0.0002, 0.0003, 0.0003]


def test_latest_returns_last_bar_value(tmp_path):
    p = _make_parquet(tmp_path, [("2026-05-01 00:00", 0.0001)])
    prov = FundingRateProvider(parquet_path=p, rest_fetcher=lambda: None)
    prov.advance(pd.Timestamp("2026-05-01 04:00", tz="UTC"))
    prov.advance(pd.Timestamp("2026-05-01 12:00", tz="UTC"))
    assert prov.latest() == pytest.approx(0.0001, abs=1e-15)


def test_latest_lag_n_returns_value_n_bars_back(tmp_path):
    p = _make_parquet(tmp_path, [
        ("2026-05-01 00:00", 0.0001),
        ("2026-05-01 08:00", 0.0002),
    ])
    prov = FundingRateProvider(parquet_path=p, rest_fetcher=lambda: None)
    prov.advance(pd.Timestamp("2026-05-01 04:00", tz="UTC"))   # 0.0001
    prov.advance(pd.Timestamp("2026-05-01 09:00", tz="UTC"))   # 0.0002
    prov.advance(pd.Timestamp("2026-05-01 12:00", tz="UTC"))   # 0.0002
    assert prov.latest_lag(0) == pytest.approx(0.0002)
    assert prov.latest_lag(1) == pytest.approx(0.0002)
    assert prov.latest_lag(2) == pytest.approx(0.0001)


def test_latest_ema_matches_pandas_ewm(tmp_path):
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
    np.testing.assert_almost_equal(prov.latest_ema(10), expected, decimal=12)


def test_history_bounded_by_history_bars(tmp_path):
    p = _make_parquet(tmp_path, [("2026-05-01 00:00", 0.0001)])
    prov = FundingRateProvider(
        parquet_path=p, rest_fetcher=lambda: None, history_bars=4,
    )
    for h in range(10):
        prov.advance(pd.Timestamp(f"2026-05-01 {h:02d}:00", tz="UTC"))
    assert prov.history_size == 4


def test_advance_naive_timestamp_is_localized(tmp_path):
    """Bar timestamps without tz must be auto-localized to UTC."""
    p = _make_parquet(tmp_path, [("2026-05-01 00:00", 0.0001)])
    prov = FundingRateProvider(parquet_path=p, rest_fetcher=lambda: None)
    val = prov.advance(pd.Timestamp("2026-05-01 12:00"))  # naive
    assert val == pytest.approx(0.0001, abs=1e-15)


def test_lag_n_negative_raises(tmp_path):
    p = _make_parquet(tmp_path, [("2026-05-01 00:00", 0.0001)])
    prov = FundingRateProvider(parquet_path=p, rest_fetcher=lambda: None)
    prov.advance(pd.Timestamp("2026-05-01 12:00", tz="UTC"))
    with pytest.raises(ValueError, match="lag n must be >= 0"):
        prov.latest_lag(-1)


def test_ema_span_zero_raises(tmp_path):
    p = _make_parquet(tmp_path, [("2026-05-01 00:00", 0.0001)])
    prov = FundingRateProvider(parquet_path=p, rest_fetcher=lambda: None)
    prov.advance(pd.Timestamp("2026-05-01 12:00", tz="UTC"))
    with pytest.raises(ValueError, match="span must be positive"):
        prov.latest_ema(0)


def test_force_rest_observation_test_hook(tmp_path):
    """The test hook injects a REST observation bypassing TTL gating."""
    p = _make_parquet(tmp_path, [("2026-05-01 00:00", 0.0001)])
    prov = FundingRateProvider(parquet_path=p, rest_fetcher=lambda: None)
    prov._force_rest_observation(FundingObservation(
        funding_time=pd.Timestamp("2026-05-01 08:00", tz="UTC"),
        funding_rate=0.0099,
    ))
    val = prov.advance(pd.Timestamp("2026-05-01 12:00", tz="UTC"))
    assert val == pytest.approx(0.0099, abs=1e-15)
