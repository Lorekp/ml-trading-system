"""Tests for ``KillSwitch.on_latency_breach`` (E.2.8).

Three contracts under test:

1. **Single breach**: WARN logged, ``_consecutive_errors`` increments
   by 1, but no trip below the consecutive threshold.
2. **Threshold breach**: at ``max_consecutive_errors`` consecutive
   latency breaches, the kill switch trips with a clear reason.
3. **Below threshold latency**: WARN does NOT fire, ``_consecutive_errors``
   stays at 0.
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
    assert ks._consecutive_errors == 1


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
    assert ks._consecutive_errors == 0
    assert not ks.is_halted()


def test_non_finite_latency_no_increment(ks: KillSwitch):
    """Defensive: NaN/inf total_ms is ignored, not treated as a breach."""
    ks.on_latency_breach(
        total_ms=float("nan"), threshold_ms=1000.0, instrument_name="BTC-OPT",
    )
    ks.on_latency_breach(
        total_ms=float("inf"), threshold_ms=1000.0, instrument_name="BTC-OPT",
    )
    # inf > threshold is True technically, but inf isn't finite — should
    # be filtered. Both calls should be no-ops.
    assert ks._consecutive_errors == 0


def test_successful_trade_resets_counter(ks: KillSwitch):
    """A successful trade clears the error counter — same path as
    consecutive-error recovery, so a single breach followed by a clean
    fill un-arms the trip."""
    ks.on_latency_breach(
        total_ms=3000.0, threshold_ms=1000.0, instrument_name="BTC-OPT",
    )
    assert ks._consecutive_errors == 1
    ks.on_trade_result(success=True)
    assert ks._consecutive_errors == 0


def test_default_disabled_threshold_does_not_trip(ks: KillSwitch):
    """When the engine wires ``kill_switch_latency_ms=inf`` (default,
    testnet-friendly), no breach can ever fire because total_ms < inf."""
    # Simulate the OrderManager check: `total_ms > threshold_ms`.
    # With threshold=inf, this is never True, so on_latency_breach
    # never gets called. Confirm by directly calling with finite value
    # < inf — would be a no-op since total_ms 1e9 is still finite.
    ks.on_latency_breach(
        total_ms=1e9, threshold_ms=float("inf"), instrument_name="BTC-OPT",
    )
    # 1e9 is finite, but 1e9 > inf is False so the helper's first guard
    # short-circuits.
    assert ks._consecutive_errors == 0
