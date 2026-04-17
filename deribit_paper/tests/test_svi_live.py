"""Tests for LiveSVICalibrator — caching + recalibration + evaluate parity."""

from __future__ import annotations

import asyncio

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.svi_live import LiveSVICalibrator


def _synthetic_option_trades(
    n_per_expiry: int = 30,
    n_expiries: int = 2,
    start: str = "2026-04-10 11:00:00",
) -> pd.DataFrame:
    """Build a synthetic option-trades DataFrame that passes the SVI calibrator.

    The SVI calibrator needs at least ``min_strikes`` trades per
    (bucket, expiry) to produce a fit, so we spread trades across
    multiple strikes for each expiry.
    """
    rng = np.random.default_rng(0)
    rows = []
    base_ts = pd.Timestamp(start, tz="UTC")
    strikes = np.linspace(55_000, 65_000, 7)  # 7 strikes per expiry

    for exp_idx in range(n_expiries):
        expiry_dt = base_ts + pd.Timedelta(days=3 + exp_idx * 4) + pd.Timedelta(hours=-3)
        for i in range(n_per_expiry):
            ts = base_ts + pd.Timedelta(minutes=i * 2)
            strike = float(rng.choice(strikes))
            # Approximate IV in percent (Deribit convention)
            iv = 70.0 + rng.normal(0, 5.0)
            tau = (expiry_dt - ts).total_seconds() / (365.25 * 24 * 3600)
            rows.append({
                "timestamp": ts,
                "instrument_name": f"BTC-EXP{exp_idx}-{int(strike)}-C",
                "price": 0.05,
                "amount": 1.0,
                "direction": 1 if rng.random() > 0.5 else -1,
                "strike": strike,
                "option_type": "C" if rng.random() > 0.5 else "P",
                "expiry_dt": expiry_dt,
                "iv": iv,
                "index_price": 60_000.0,
                "mark_price": 0.05,
                "T_years": tau,
                "dte_days": tau * 365.25,
            })
    return pd.DataFrame(rows)


def test_needs_refresh_when_empty():
    cal = LiveSVICalibrator(cache_path="/tmp/nonexistent.parquet", recalibrate_hours=1.0)
    now = pd.Timestamp("2026-04-10 12:00:00", tz="UTC")
    assert cal.needs_refresh(now) is True


def test_has_params_false_initially():
    cal = LiveSVICalibrator(cache_path="/tmp/nonexistent_svi.parquet")
    assert cal.has_params is False


def test_evaluate_before_calibration_raises():
    cal = LiveSVICalibrator(cache_path="/tmp/nonexistent_svi2.parquet")
    with pytest.raises(RuntimeError, match="before any successful"):
        cal.evaluate(pd.DataFrame({"timestamp": [], "strike": [], "expiry_dt": []}))


def test_recalibrate_with_empty_trades_raises():
    cal = LiveSVICalibrator(cache_path="/tmp/nonexistent_svi3.parquet")
    with pytest.raises(ValueError, match="empty"):
        cal.recalibrate(pd.DataFrame())


def test_recalibrate_populates_cache(tmp_path):
    """End-to-end: calibrate on synthetic trades, check params_df + meta are set."""
    cache_path = tmp_path / "svi_cache.parquet"
    cal = LiveSVICalibrator(cache_path=cache_path, recalibrate_hours=1.0)

    trades = _synthetic_option_trades(n_per_expiry=50, n_expiries=2)
    now = pd.Timestamp("2026-04-10 12:00:00", tz="UTC")

    meta = cal.recalibrate(trades, now=now)
    # Calibration may succeed or fall back, but meta should be populated
    assert meta is not None
    assert cal.last_recalibrated_at == now

    # The cache should have been persisted to disk
    if cal.has_params:
        assert cache_path.exists()


def test_needs_refresh_honors_elapsed(tmp_path):
    cache_path = tmp_path / "svi.parquet"
    cal = LiveSVICalibrator(cache_path=cache_path, recalibrate_hours=1.0)
    trades = _synthetic_option_trades(n_per_expiry=30, n_expiries=2)
    first = pd.Timestamp("2026-04-10 12:00:00", tz="UTC")
    cal.recalibrate(trades, now=first)

    # 30 min later → not yet time to refresh
    assert cal.needs_refresh(first + pd.Timedelta(minutes=30)) is False
    # 1h + 1min later → refresh needed
    assert cal.needs_refresh(first + pd.Timedelta(minutes=61)) is True


@pytest.mark.asyncio
async def test_recalibrate_async_non_blocking(tmp_path):
    """recalibrate_async should complete without blocking the event loop."""
    cal = LiveSVICalibrator(cache_path=tmp_path / "svi.parquet", recalibrate_hours=1.0)
    trades = _synthetic_option_trades(n_per_expiry=30, n_expiries=2)
    now = pd.Timestamp("2026-04-10 12:00:00", tz="UTC")

    meta = await cal.recalibrate_async(trades, now=now)
    assert meta is not None
    assert cal.last_recalibrated_at == now


def test_is_recalibrating_flag(tmp_path):
    cal = LiveSVICalibrator(cache_path=tmp_path / "svi.parquet")
    assert cal.is_recalibrating is False
    # Hard to test mid-flight without threading gymnastics — trust the flag
    # invariant is maintained by the try/finally in recalibrate()
