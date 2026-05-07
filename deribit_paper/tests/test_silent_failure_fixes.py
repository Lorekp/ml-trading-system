"""Tests for the 4 sister silent-failure bug fixes (2026-05-07).

After fixing ``LiveVPINStream.is_stale`` (the original VPIN-shows-
warm-start-value-forever bug), an Opus code-review found 4 more
places with the SAME pattern: a check that returns "healthy" when
the underlying state has never transitioned out of "initial".

This file pins the contract for each of the four fixes:

1. ``LiveSVICalibrator.is_stale`` — returns True after grace period
   when ``_meta`` was never set (no calibration ever succeeded).
2. ``KillSwitch.check_before_signal`` — trips on network outage when
   ``_last_ws_message_at is None`` AND startup elapsed >
   ``network_outage_seconds`` (Deribit subscribed but no events ever).
3. ``RNDMonitor.size_scale`` — degrades to ``size_min`` (not 1.0)
   when ``_latest is None`` AND grace elapsed (broken upstream).
4. ``IndexPriceBuffer.latest_fresh`` / ``latest_with_age`` — surfaces
   the freshness of the most recent sample so callers don't trade
   off hours-old prices.

Each test injects an artificial ``started_at`` / ``now`` so we don't
have to actually wait for grace periods to elapse.
"""
from __future__ import annotations

import pandas as pd
import pytest

from deribit_paper.features.svi_live import LiveSVICalibrator
from deribit_paper.market_data.buffer import IndexPriceBuffer
from deribit_paper.position.equity import EquityTracker
from deribit_paper.risk.kill_switch import KillSwitch
from deribit_paper.risk.rnd_monitor import RNDConfig, RNDMonitor


# ── P0-1: LiveSVICalibrator.is_stale ─────────────────────────────────


def test_svi_is_stale_false_within_grace_when_never_calibrated(tmp_path):
    """During warmup grace, no calibration is OK — is_stale returns False."""
    started = pd.Timestamp("2026-05-07 10:00:00", tz="UTC")
    cal = LiveSVICalibrator(
        cache_path=tmp_path / "cache.parquet",
        started_at=started,
    )
    now = started + pd.Timedelta(hours=2)
    assert cal.is_stale(now, max_stale_hours=4.0) is False


def test_svi_is_stale_true_after_grace_when_never_calibrated(tmp_path):
    """Past warmup grace and still no calibration is broken — stale."""
    started = pd.Timestamp("2026-05-07 10:00:00", tz="UTC")
    cal = LiveSVICalibrator(
        cache_path=tmp_path / "cache.parquet",
        started_at=started,
    )
    now = started + pd.Timedelta(hours=5)  # > 4 hours threshold
    assert cal.is_stale(now, max_stale_hours=4.0) is True


def test_svi_stale_age_hours_uses_startup_when_never_calibrated(tmp_path):
    """``stale_age_hours`` reports time-since-startup, not 0.0, so the
    operator's heartbeat shows an honest "we've been waiting N hours"
    rather than the misleading 0.0 the pre-fix code emitted."""
    started = pd.Timestamp("2026-05-07 10:00:00", tz="UTC")
    cal = LiveSVICalibrator(
        cache_path=tmp_path / "cache.parquet",
        started_at=started,
    )
    now = started + pd.Timedelta(hours=3.5)
    assert cal.stale_age_hours(now) == pytest.approx(3.5, abs=1e-6)


# ── P0-2: KillSwitch network-outage check on cold-start ──────────────


def _make_ks(*, started: pd.Timestamp, network_outage_seconds: float = 120.0) -> KillSwitch:
    return KillSwitch(
        equity_tracker=EquityTracker(initial_equity_usd=100_000.0),
        network_outage_seconds=network_outage_seconds,
        started_at=started,
    )


def test_ks_no_trip_within_outage_window_with_no_message():
    """During the 120s grace, no WS message yet is OK."""
    started = pd.Timestamp("2026-05-07 10:00:00", tz="UTC")
    ks = _make_ks(started=started, network_outage_seconds=120.0)
    now = started + pd.Timedelta(seconds=60)
    assert ks.check_before_signal(now=now) is False
    assert ks.is_halted() is False


def test_ks_trips_on_cold_start_silence_after_grace():
    """Past the 120s grace with NO WS message ever → outage trip.

    Pre-fix this was the silent-failure case: ``_last_ws_message_at is
    None`` skipped the check, so the engine ran for hours with the kill
    switch inert."""
    started = pd.Timestamp("2026-05-07 10:00:00", tz="UTC")
    ks = _make_ks(started=started, network_outage_seconds=120.0)
    now = started + pd.Timedelta(seconds=200)  # 80s past threshold
    halted = ks.check_before_signal(now=now)
    assert halted is True
    assert ks.is_halted() is True
    assert "WS messages never arrived" in (ks.state.reason or "")


def test_ks_existing_path_still_works():
    """The previously-flowing-then-silent path is unchanged."""
    started = pd.Timestamp("2026-05-07 10:00:00", tz="UTC")
    ks = _make_ks(started=started, network_outage_seconds=120.0)
    # First message arrives 30s in
    msg_time = started + pd.Timedelta(seconds=30)
    ks.mark_ws_message(at=msg_time)
    # Then silence for 200s
    now = msg_time + pd.Timedelta(seconds=200)
    halted = ks.check_before_signal(now=now)
    assert halted is True
    assert "since last WS message" in (ks.state.reason or "")


# ── P1-3: RNDMonitor.size_scale on broken upstream ──────────────────


def test_rnd_size_scale_full_within_warmup_when_never_updated():
    """Within grace, ``_latest is None`` is legitimate warmup —
    size_scale returns 1.0 (full size)."""
    started = pd.Timestamp("2026-05-07 10:00:00", tz="UTC")
    cfg = RNDConfig(bar_minutes=15)
    monitor = RNDMonitor(config=cfg, started_at=started)
    # Use a small grace period to make the test fast: 1 hour
    monitor._broken_grace_seconds = 3600.0
    now = started + pd.Timedelta(minutes=30)
    assert monitor.size_scale(now=now) == pytest.approx(1.0)
    assert monitor.is_warmup_failed(now=now) is False


def test_rnd_size_scale_degrades_when_never_updated_past_grace():
    """Past grace with ``_latest is None`` is broken upstream —
    size_scale returns ``size_min`` (default 0.25), NOT 1.0."""
    started = pd.Timestamp("2026-05-07 10:00:00", tz="UTC")
    cfg = RNDConfig(bar_minutes=15, size_min=0.25)
    monitor = RNDMonitor(config=cfg, started_at=started)
    monitor._broken_grace_seconds = 3600.0
    now = started + pd.Timedelta(hours=2)  # past grace
    assert monitor.is_warmup_failed(now=now) is True
    assert monitor.size_scale(now=now) == pytest.approx(0.25)


def test_rnd_default_grace_period_matches_warmup_bars():
    """Grace defaults to ``baseline_warmup_bars × bar_minutes × 60``,
    so a fresh engine has time to receive at least one SVI calibration
    cycle before this trips."""
    cfg = RNDConfig(bar_minutes=15)  # baseline_warmup_bars defaults to 1 day
    monitor = RNDMonitor(config=cfg)
    # baseline_warmup_bars = 1440 // 15 = 96 bars × 15 min × 60 s = 86400 s
    assert monitor._broken_grace_seconds == pytest.approx(86400.0)


# ── P1-4: IndexPriceBuffer staleness ─────────────────────────────────


def test_index_buffer_latest_unchanged_no_age_check():
    """Back-compat: ``latest()`` still returns the most recent price
    regardless of age."""
    buf = IndexPriceBuffer(window_seconds=3600.0)
    old_ts = pd.Timestamp("2026-05-07 09:00:00", tz="UTC")
    buf.append(old_ts, 60_000.0)
    assert buf.latest() == 60_000.0


def test_index_buffer_latest_with_age_returns_seconds():
    """``latest_with_age`` exposes the (price, age_seconds) tuple."""
    buf = IndexPriceBuffer(window_seconds=3600.0)
    sample_ts = pd.Timestamp("2026-05-07 09:00:00", tz="UTC")
    buf.append(sample_ts, 60_000.0)
    now = pd.Timestamp("2026-05-07 09:01:30", tz="UTC")  # 90s later
    price, age = buf.latest_with_age(now=now)
    assert price == 60_000.0
    assert age == pytest.approx(90.0)


def test_index_buffer_latest_fresh_returns_price_when_young():
    """Within ``max_age_seconds``, latest_fresh returns the price."""
    buf = IndexPriceBuffer(window_seconds=3600.0)
    sample_ts = pd.Timestamp("2026-05-07 09:00:00", tz="UTC")
    buf.append(sample_ts, 60_000.0)
    now = sample_ts + pd.Timedelta(seconds=30)
    assert buf.latest_fresh(max_age_seconds=60.0, now=now) == 60_000.0


def test_index_buffer_latest_fresh_returns_none_when_stale():
    """Past ``max_age_seconds``, latest_fresh returns None — caller
    should fall back to a REST query."""
    buf = IndexPriceBuffer(window_seconds=3600.0)
    sample_ts = pd.Timestamp("2026-05-07 09:00:00", tz="UTC")
    buf.append(sample_ts, 60_000.0)
    now = sample_ts + pd.Timedelta(seconds=120)  # past 60s threshold
    assert buf.latest_fresh(max_age_seconds=60.0, now=now) is None


def test_index_buffer_latest_fresh_returns_none_when_empty():
    """Empty buffer → None regardless of threshold."""
    buf = IndexPriceBuffer(window_seconds=3600.0)
    now = pd.Timestamp("2026-05-07 09:00:00", tz="UTC")
    assert buf.latest_fresh(max_age_seconds=60.0, now=now) is None


def test_index_buffer_latest_with_age_empty_returns_none_pair():
    """Empty buffer → ``(None, None)`` so callers can destructure
    without needing to check len() first."""
    buf = IndexPriceBuffer(window_seconds=3600.0)
    price, age = buf.latest_with_age()
    assert price is None
    assert age is None
