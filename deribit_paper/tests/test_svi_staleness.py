"""Tests for SVI staleness detection (C3)."""

from __future__ import annotations

import pandas as pd
import pytest

from deribit_paper.features.svi_live import LiveSVICalibrator, SVICacheMeta


def _make_calibrator(**kwargs) -> LiveSVICalibrator:
    return LiveSVICalibrator(
        cache_path="nonexistent_test_path.parquet",
        **kwargs,
    )


def test_is_stale_no_meta_returns_false_within_grace():
    """Before any calibration, within the warmup grace period
    (``max_stale_hours`` since construction), is_stale returns False.

    Past the grace period, see ``test_silent_failure_fixes.py`` —
    the post-fix behaviour is that is_stale returns True after grace
    even with no meta (closes the silent-failure pattern that hid
    permanently-broken SVI as "warmup").
    """
    started = pd.Timestamp("2026-04-13 11:30", tz="UTC")
    cal = _make_calibrator(started_at=started)
    now = pd.Timestamp("2026-04-13 12:00", tz="UTC")  # 30 min after start
    assert cal.is_stale(now, max_stale_hours=4.0) is False


def test_is_stale_recent_calibration():
    """A calibration 30 minutes ago is NOT stale (default 4h)."""
    cal = _make_calibrator()
    now = pd.Timestamp("2026-04-13 12:30", tz="UTC")
    cal._meta = SVICacheMeta(
        last_recalibrated_at=pd.Timestamp("2026-04-13 12:00", tz="UTC"),
        n_buckets=5, n_params_rows=10,
        window_start=pd.Timestamp("2026-04-13 11:00", tz="UTC"),
        window_end=pd.Timestamp("2026-04-13 12:00", tz="UTC"),
    )
    assert cal.is_stale(now) is False


def test_is_stale_old_calibration():
    """A calibration 5 hours ago IS stale (default 4h)."""
    cal = _make_calibrator()
    now = pd.Timestamp("2026-04-13 17:00", tz="UTC")
    cal._meta = SVICacheMeta(
        last_recalibrated_at=pd.Timestamp("2026-04-13 12:00", tz="UTC"),
        n_buckets=5, n_params_rows=10,
        window_start=pd.Timestamp("2026-04-13 11:00", tz="UTC"),
        window_end=pd.Timestamp("2026-04-13 12:00", tz="UTC"),
    )
    assert cal.is_stale(now) is True


def test_is_stale_custom_threshold():
    """Custom max_stale_hours is respected."""
    cal = _make_calibrator()
    now = pd.Timestamp("2026-04-13 14:00", tz="UTC")
    cal._meta = SVICacheMeta(
        last_recalibrated_at=pd.Timestamp("2026-04-13 12:00", tz="UTC"),
        n_buckets=5, n_params_rows=10,
        window_start=pd.Timestamp("2026-04-13 11:00", tz="UTC"),
        window_end=pd.Timestamp("2026-04-13 12:00", tz="UTC"),
    )
    assert cal.is_stale(now, max_stale_hours=1.5) is True
    assert cal.is_stale(now, max_stale_hours=3.0) is False


def test_stale_age_hours():
    """stale_age_hours returns correct elapsed time."""
    cal = _make_calibrator()
    now = pd.Timestamp("2026-04-13 15:30", tz="UTC")
    cal._meta = SVICacheMeta(
        last_recalibrated_at=pd.Timestamp("2026-04-13 12:00", tz="UTC"),
        n_buckets=5, n_params_rows=10,
        window_start=pd.Timestamp("2026-04-13 11:00", tz="UTC"),
        window_end=pd.Timestamp("2026-04-13 12:00", tz="UTC"),
    )
    assert cal.stale_age_hours(now) == pytest.approx(3.5)


def test_stale_age_hours_no_meta():
    """When the calibrator has never produced ``_meta``, stale_age_hours
    returns time-since-construction (NOT 0.0 as the pre-fix code emitted).

    Pre-fix this returned 0.0, masking the "calibrator alive for hours
    but never calibrated" failure mode in the operator-facing
    heartbeat. Now it reports the honest elapsed.
    """
    started = pd.Timestamp("2026-04-13 11:00", tz="UTC")
    cal = _make_calibrator(started_at=started)
    now = pd.Timestamp("2026-04-13 12:00", tz="UTC")
    # 1 hour since startup, no calibration yet
    assert cal.stale_age_hours(now) == pytest.approx(1.0)
