"""Tests for KillSwitch trip conditions."""

from __future__ import annotations

import pandas as pd
import pytest

from deribit_paper.position.equity import EquityTracker
from deribit_paper.risk.kill_switch import KillSwitch, KillSwitchState


def _make_switch(**overrides) -> tuple[KillSwitch, EquityTracker, list[str]]:
    equity = EquityTracker(initial_equity_usd=100_000.0)
    alerts: list[str] = []
    defaults = dict(
        equity_tracker=equity,
        max_daily_drawdown_pct=2.0,
        max_consecutive_errors=3,
        network_outage_seconds=120.0,
        alert_callback=alerts.append,
    )
    defaults.update(overrides)
    ks = KillSwitch(**defaults)
    return ks, equity, alerts


def test_initial_state_not_halted():
    ks, _, _ = _make_switch()
    assert ks.is_halted() is False
    assert ks.state.reason is None


def test_trips_on_daily_drawdown():
    """R-1 (round-3 parity audit, 2026-05-08): the trip condition is
    daily PnL (since 00:00 UTC), NOT lifetime-peak drawdown. Pre-fix
    this test set up peak=105k / current=101k (a +1% DAILY gain after
    a +5% then −4% sequence) and expected a trip — that codified the
    bug where peak-DD was being read as if it were daily DD.

    Post-fix: a single −3% intra-day loss DOES trip; the prior fake
    "peak then partial recovery" sequence does NOT (today is +1% on
    net). Both directions now match the config name + docstring +
    runbook.
    """
    ks, equity, alerts = _make_switch(max_daily_drawdown_pct=2.0)
    # Single −3% intra-day loss → daily PnL = −3% < −2% threshold → trip.
    equity.on_trade_closed(-3_000.0)
    halted = ks.check_before_signal()
    assert halted is True
    assert ks.is_halted() is True
    # Reason references daily PnL, not lifetime peak.
    assert "daily" in (ks.state.reason or "").lower()
    assert "-3.00%" in (ks.state.reason or "")
    assert len(alerts) == 1


def test_does_not_trip_on_peak_dd_with_positive_daily_pnl():
    """R-1 contract verification: a +5K then −4K sequence leaves the
    account at +1% DAILY but with a 3.8% DRAWDOWN-FROM-PEAK. The
    kill-switch must NOT trip on this (pre-fix it did, conflating the
    two metrics). Daily PnL is the contract.
    """
    ks, equity, _ = _make_switch(max_daily_drawdown_pct=2.0)
    equity.on_trade_closed(5_000.0)   # peak 105k
    equity.on_trade_closed(-4_000.0)  # 101k → daily PnL +1%
    assert ks.check_before_signal() is False
    assert ks.is_halted() is False


def test_does_not_trip_on_small_drawdown():
    ks, equity, _ = _make_switch(max_daily_drawdown_pct=5.0)
    equity.on_trade_closed(5_000.0)
    equity.on_trade_closed(-1_000.0)  # daily PnL = +4% (well within 5% threshold either way)
    assert ks.check_before_signal() is False
    assert ks.is_halted() is False


def test_trips_on_consecutive_errors():
    ks, _, alerts = _make_switch(max_consecutive_errors=3)
    ks.on_trade_result(success=False)
    ks.on_trade_result(success=False)
    assert ks.is_halted() is False
    ks.on_trade_result(success=False)  # third → trip
    assert ks.is_halted() is True
    assert "consecutive" in (ks.state.reason or "")
    assert len(alerts) == 1


def test_success_resets_error_counter():
    ks, _, _ = _make_switch(max_consecutive_errors=3)
    ks.on_trade_result(success=False)
    ks.on_trade_result(success=False)
    ks.on_trade_result(success=True)  # reset
    ks.on_trade_result(success=False)
    ks.on_trade_result(success=False)  # only 2 consecutive errors
    assert ks.is_halted() is False


def test_trips_on_network_outage():
    ks, _, _ = _make_switch(network_outage_seconds=10.0)
    now = pd.Timestamp("2026-04-10 12:00:00", tz="UTC")
    ks.mark_ws_message(at=now)
    # 15 seconds later — outage
    later = now + pd.Timedelta(seconds=15)
    halted = ks.check_before_signal(now=later)
    assert halted is True
    assert "network" in (ks.state.reason or "")


def test_manual_trip():
    ks, _, alerts = _make_switch()
    ks.trip("emergency — margin call detected")
    assert ks.is_halted() is True
    assert "emergency" in (ks.state.reason or "")
    assert len(alerts) == 1


def test_trip_is_idempotent():
    ks, _, alerts = _make_switch()
    ks.trip("reason 1")
    ks.trip("reason 2")  # second call is a no-op
    assert ks.state.reason == "reason 1"
    assert len(alerts) == 1


def test_reset_requires_manual_action():
    ks, _, _ = _make_switch()
    ks.trip("test")
    assert ks.is_halted() is True
    ks.reset(operator="test")
    assert ks.is_halted() is False
    assert ks.state.reason is None


def test_alert_callback_exception_does_not_block_trip():
    def bad_alert(_):
        raise RuntimeError("telegram down")
    ks, _, _ = _make_switch(alert_callback=bad_alert)
    ks.trip("test")
    assert ks.is_halted() is True  # trip still succeeds


def test_network_outage_not_masked_by_bar_ticks():
    """Verify that the kill switch trips when WS data stops arriving,
    even though bar ticks continue on schedule.

    Regression test for the 2026-04-13 incident where mark_ws_message
    was called on every bar tick, masking a dead WS feed for 11 hours.
    """
    ks, _, _ = _make_switch(network_outage_seconds=60.0)
    t0 = pd.Timestamp("2026-04-10 12:00:00", tz="UTC")
    ks.mark_ws_message(at=t0)

    # Simulate 5 bar ticks at 15-min intervals WITHOUT any WS marks.
    # The very first check (15 min = 900s) exceeds the 60s threshold.
    for i in range(1, 6):
        bar_time = t0 + pd.Timedelta(minutes=15 * i)
        halted = ks.check_before_signal(now=bar_time)
        if halted:
            break
    assert ks.is_halted() is True
    assert "network" in (ks.state.reason or "")


# ── C2: Emergency close flag tests ────────────────────────────────


def test_trip_sets_needs_emergency_close():
    """C2: trip() sets needs_emergency_close flag for engine to pick up."""
    ks, _, _ = _make_switch()
    assert ks.needs_emergency_close is False
    ks.trip("test")
    assert ks.needs_emergency_close is True


def test_reset_does_not_clear_emergency_flag():
    """needs_emergency_close is cleared by the engine, not by reset."""
    ks, _, _ = _make_switch()
    ks.trip("test")
    ks.needs_emergency_close = False  # engine cleared it
    ks.reset(operator="test")
    assert ks.needs_emergency_close is False


def test_second_trip_does_not_re_set_emergency_flag():
    """Only the FIRST trip sets the flag; second trip is idempotent."""
    ks, _, _ = _make_switch()
    ks.trip("first")
    ks.needs_emergency_close = False  # engine cleared it
    ks.trip("second")  # no-op since already halted
    assert ks.needs_emergency_close is False
