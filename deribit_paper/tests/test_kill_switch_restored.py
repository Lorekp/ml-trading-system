"""S6 (parity-cycle round 2 — 2026-05-07): ``KillSwitch.trip(restored=True)``
restores the halted state from a persisted row WITHOUT re-firing the
emergency-close path that the original trip already executed.

Pre-S6 the engine called ``kill_switch.trip(restored_msg)`` at startup
when SQLite reported a prior-halted state. That set
``needs_emergency_close=True``, which the first bar's risk-check then
acted on by issuing a fresh emergency close-all. The original session
had already flattened on the trip; the second close-all sent a
redundant Telegram alert and risked placing reduce-only orders against
zero positions if the ledger reconciliation lagged.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pandas as pd
import pytest

from deribit_paper.position.equity import EquityTracker
from deribit_paper.risk.kill_switch import KillSwitch


def _make_kill_switch(*, alert_callback=None, persist_callback=None) -> KillSwitch:
    return KillSwitch(
        equity_tracker=EquityTracker(initial_equity_usd=100_000.0),
        max_daily_drawdown_pct=2.0,
        max_consecutive_errors=3,
        network_outage_seconds=120.0,
        alert_callback=alert_callback,
        persist_callback=persist_callback,
    )


def test_normal_trip_sets_needs_emergency_close():
    ks = _make_kill_switch()
    assert ks.needs_emergency_close is False
    ks.trip("manual halt")
    assert ks.is_halted()
    assert ks.needs_emergency_close is True


def test_restored_trip_halts_without_emergency_close():
    """S6: restored=True suppresses needs_emergency_close."""
    ks = _make_kill_switch()
    ks.trip("[restored from <ts>] previous halt", restored=True)
    assert ks.is_halted()
    assert ks.needs_emergency_close is False


def test_restored_trip_skips_alert_callback():
    """Restored path must not re-fire the operator alert (already sent at original trip)."""
    alert = MagicMock()
    ks = _make_kill_switch(alert_callback=alert)
    ks.trip("[restored from <ts>] previous halt", restored=True)
    alert.assert_not_called()


def test_restored_trip_skips_persist_callback():
    """Restored path must not re-fire SQLite persist (already there)."""
    persist = MagicMock()
    ks = _make_kill_switch(persist_callback=persist)
    ks.trip("[restored from <ts>] previous halt", restored=True)
    persist.assert_not_called()


def test_normal_trip_then_restored_no_double_emergency_close():
    """If trip fires THEN we mark needs_emergency_close consumed,
    a restored trip on a fresh ks should NOT re-fire it."""
    ks = _make_kill_switch()
    # simulate the first session's normal trip + first-bar consumption
    ks.trip("session 1 trip")
    assert ks.needs_emergency_close
    ks.needs_emergency_close = False  # engine consumed the flag

    # session 2: fresh ks loaded from disk
    ks2 = _make_kill_switch()
    ks2.trip("[restored from session 1 trip ts] session 1 trip", restored=True)
    assert ks2.is_halted()
    assert ks2.needs_emergency_close is False


def test_restored_trip_records_reason_and_timestamp():
    ks = _make_kill_switch()
    now = pd.Timestamp("2026-05-07 12:00", tz="UTC")
    ks.trip("network outage 600s", restored=True, now=now)
    assert ks.state.halted is True
    assert ks.state.reason == "network outage 600s"
    assert ks.state.halted_at == now


def test_restored_trip_idempotent_when_already_halted():
    """Pre-existing halt should not be overwritten by a duplicate restored=True call."""
    ks = _make_kill_switch()
    ks.trip("first trip")
    first_reason = ks.state.reason
    ks.trip("second trip [restored]", restored=True)
    # The class doesn't update reason on a second trip — preserves the first
    assert ks.state.reason == first_reason
