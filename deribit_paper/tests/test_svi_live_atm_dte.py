"""F7 (parity-cycle round 2 — 2026-05-07): ``LiveSVICalibrator.atm_vol_at_dte``
returns NaN when the latest bucket has fewer than 2 cached expiries.

Pre-F7 the surface degenerated to a single-slice approximation,
returning that one expiry's ATM regardless of the requested DTE — a
silent wrong-magnitude value passed through the F9 fallback because
the value WAS technically finite.

Also validates S7 atomic-cache-persist (write-and-replace pattern).
"""

from __future__ import annotations

import math
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from deribit_paper.features.svi_live import LiveSVICalibrator, SVICacheMeta


def _build_params_df(expiry_dts: list[pd.Timestamp], bucket_ts: pd.Timestamp):
    return pd.DataFrame([
        {
            "timestamp": bucket_ts,
            "expiry_dt": e,
            "T": 7.0 / 365.25,
            "F": 50_000.0,
            "a": 0.001,
            "b": 0.1,
            "rho": -0.2,
            "m": 0.0,
            "sigma": 0.3,
        }
        for e in expiry_dts
    ])


def test_atm_vol_at_dte_returns_nan_with_single_expiry():
    """F7: only one cached expiry → return NaN (degraded mode)."""
    bucket_ts = pd.Timestamp("2026-05-07 12:00", tz="UTC")
    cal = LiveSVICalibrator(cache_path="/tmp/_test_svi_only.parquet")
    cal._params_df = _build_params_df(
        [pd.Timestamp("2026-05-09", tz="UTC")], bucket_ts,
    )
    cal._meta = SVICacheMeta(
        last_recalibrated_at=bucket_ts, n_buckets=1, n_params_rows=1,
        window_start=bucket_ts, window_end=bucket_ts,
    )
    out = cal.atm_vol_at_dte(7.0)
    assert math.isnan(out)


def test_atm_vol_at_dte_finite_with_two_expiries():
    """Two cached expiries → genuine surface evaluation, finite output."""
    bucket_ts = pd.Timestamp("2026-05-07 12:00", tz="UTC")
    cal = LiveSVICalibrator(cache_path="/tmp/_test_svi_two.parquet")
    cal._params_df = _build_params_df(
        [
            pd.Timestamp("2026-05-09", tz="UTC"),
            pd.Timestamp("2026-05-16", tz="UTC"),
        ],
        bucket_ts,
    )
    cal._meta = SVICacheMeta(
        last_recalibrated_at=bucket_ts, n_buckets=1, n_params_rows=2,
        window_start=bucket_ts, window_end=bucket_ts,
    )
    out = cal.atm_vol_at_dte(7.0)
    assert math.isfinite(out)
    assert out > 0.0


def test_atm_vol_at_dte_returns_nan_on_uncalibrated():
    cal = LiveSVICalibrator(cache_path="/tmp/_test_svi_empty.parquet")
    assert math.isnan(cal.atm_vol_at_dte(7.0))


# ── S7: atomic SVI cache persist ─────────────────────────────────────


def test_persist_cache_atomic_write_succeeds(tmp_path):
    """Happy path: persist writes both files; both exist post-write."""
    bucket_ts = pd.Timestamp("2026-05-07 12:00", tz="UTC")
    cache_path = tmp_path / "svi_cache.parquet"
    cal = LiveSVICalibrator(cache_path=cache_path)
    cal._params_df = _build_params_df(
        [pd.Timestamp("2026-05-09", tz="UTC")], bucket_ts,
    )
    cal._meta = SVICacheMeta(
        last_recalibrated_at=bucket_ts, n_buckets=1, n_params_rows=1,
        window_start=bucket_ts, window_end=bucket_ts,
    )
    cal._persist_cache()
    assert cache_path.exists()
    assert cache_path.with_suffix(".meta.parquet").exists()
    # Tmp files cleaned up
    assert not (cache_path.with_suffix(cache_path.suffix + ".tmp").exists())


def test_persist_cache_failure_cleans_up_tmps(tmp_path):
    """If meta write raises, params .tmp must not leak onto disk."""
    bucket_ts = pd.Timestamp("2026-05-07 12:00", tz="UTC")
    cache_path = tmp_path / "svi_cache.parquet"
    cal = LiveSVICalibrator(cache_path=cache_path)
    cal._params_df = _build_params_df(
        [pd.Timestamp("2026-05-09", tz="UTC")], bucket_ts,
    )
    cal._meta = SVICacheMeta(
        last_recalibrated_at=bucket_ts, n_buckets=1, n_params_rows=1,
        window_start=bucket_ts, window_end=bucket_ts,
    )

    # Force the SECOND write (meta) to raise; params tmp may have been
    # written successfully, so the cleanup should remove it.
    real_to_parquet = pd.DataFrame.to_parquet
    call = {"n": 0}

    def maybe_raising(self, *args, **kwargs):
        call["n"] += 1
        if call["n"] == 2:  # the meta write
            raise IOError("simulated disk-full on meta write")
        return real_to_parquet(self, *args, **kwargs)

    with patch.object(pd.DataFrame, "to_parquet", maybe_raising):
        cal._persist_cache()  # logs warning, doesn't raise

    # Final cache should not exist (replace never ran)
    # Tmp leftover should be cleaned up
    params_tmp = cache_path.with_suffix(cache_path.suffix + ".tmp")
    assert not params_tmp.exists()
    meta_tmp = cache_path.with_suffix(".meta.parquet").with_suffix(".meta.parquet.tmp")
    assert not meta_tmp.exists()
