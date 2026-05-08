"""Tests for ``KillSwitch.on_latency_breach`` (E.2.8).

Three contracts under test:

1. **Single breach**: WARN logged, ``_consecutive_latency_breaches``
   increments by 1, but no trip below the consecutive threshold.
2. **Threshold breach**: at ``max_consecutive_errors`` consecutive
   latency breaches, the kill switch trips with a clear reason.
3. **Below threshold latency**: WARN does NOT fire,
   ``_consecutive_latency_breaches`` stays at 0.

R-8 (parity-r3 cycle 8 — 2026-05-08): latency breaches are now
tracked in a dedicated ``_consecutive_latency_breaches`` counter
rather than the shared ``_consecutive_errors``. ``on_trade_result
(success=True)`` no longer resets latency breaches; ``on_latency_clean()``
is the symmetric reset path.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from deribit_paper.position.equity import EquityTracker
from deribit_paper.risk.kill_switch import KillSwitch


@pytest.fixture
def ks() -> KillSwitch:
    return KillSwitch(
        equity_tracker=EquityTracker(initial_equity_usd=100_000.0),
        max_daily_drawdown_pct=10.0,
        max_consecutive_errors=3,
    )


def test_single_breach_does_not_trip(ks: KillSwitch):
    ks.on_latency_breach(
        total_ms=3000.0, threshold_ms=1000.0, instrument_name="BTC-OPT",
    )
    assert not ks.is_halted()
    assert ks._consecutive_latency_breaches == 1
    # Execution-error counter is independent.
    assert ks._consecutive_errors == 0


def test_consecutive_breaches_trip_at_threshold(ks: KillSwitch):
    for _ in range(3):
        ks.on_latency_breach(
            total_ms=3000.0, threshold_ms=1000.0, instrument_name="BTC-OPT",
        )
    assert ks.is_halted()
    assert "latency" in (ks.state.reason or "").lower()


def test_below_threshold_no_increment(ks: KillSwitch):
    ks.on_latency_breach(
        total_ms=500.0, threshold_ms=1000.0, instrument_name="BTC-OPT",
    )
    assert ks._consecutive_latency_breaches == 0
    assert not ks.is_halted()


def test_non_finite_latency_no_increment(ks: KillSwitch):
    """Defensive: NaN/inf total_ms is ignored, not treated as a breach."""
    ks.on_latency_breach(
        total_ms=float("nan"), threshold_ms=1000.0, instrument_name="BTC-OPT",
    )
    ks.on_latency_breach(
        total_ms=float("inf"), threshold_ms=1000.0, instrument_name="BTC-OPT",
    )
    assert ks._consecutive_latency_breaches == 0


def test_successful_trade_does_not_reset_latency_counter(ks: KillSwitch):
    """R-8 (parity-r3 cycle 8 — 2026-05-08): latency breaches and
    execution errors track in INDEPENDENT counters. A successful trade
    no longer resets the latency counter — that requires the explicit
    ``on_latency_clean()`` symmetric reset path."""
    ks.on_latency_breach(
        total_ms=3000.0, threshold_ms=1000.0, instrument_name="BTC-OPT",
    )
    assert ks._consecutive_latency_breaches == 1
    ks.on_trade_result(success=True)
    # latency counter is independent — does NOT reset on trade success
    assert ks._consecutive_latency_breaches == 1
    # exec counter unaffected (was never set)
    assert ks._consecutive_errors == 0


def test_on_latency_clean_resets_latency_counter(ks: KillSwitch):
    """R-8: ``on_latency_clean()`` is the symmetric reset path for the
    latency counter, paralleling ``on_trade_result(success=True)`` for
    the execution-error counter."""
    ks.on_latency_breach(
        total_ms=3000.0, threshold_ms=1000.0, instrument_name="BTC-OPT",
    )
    assert ks._consecutive_latency_breaches == 1
    ks.on_latency_clean()
    assert ks._consecutive_latency_breaches == 0


def test_default_disabled_threshold_does_not_trip(ks: KillSwitch):
    """When the engine wires ``kill_switch_latency_ms=inf`` (default,
    testnet-friendly), no breach can ever fire because total_ms < inf."""
    ks.on_latency_breach(
        total_ms=1e9, threshold_ms=float("inf"), instrument_name="BTC-OPT",
    )
    assert ks._consecutive_latency_breaches == 0
