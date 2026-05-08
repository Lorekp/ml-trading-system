"""R-1 + R-2 regression tests (round-3 parity audit, fixes 2026-05-08).

R-1: kill-switch daily-DD trip path now uses ``daily_return_pct`` (PnL
since 00:00 UTC), not ``drawdown_pct`` (lifetime-peak DD). Pre-fix the
implementation diverged from the config name, docstring, and runbook.

R-2: ``EquityTracker.set_equity`` only advances ``peak_equity`` on
``reason="deposit"``. Restart reconciliation (default reason
``"reconcile"``) deliberately does NOT advance peak — pre-fix it did,
silently re-basing the kill-switch reference upward after a winning
week + crash-restart.

Together these close the round-3 audit's two P0 risk-accounting
findings.
"""

from __future__ import annotations

import pandas as pd
import pytest

from deribit_paper.position.equity import EquityTracker
from deribit_paper.risk.kill_switch import KillSwitch


# ── R-2: set_equity peak advancement ─────────────────────────────


def test_set_equity_reconcile_does_not_advance_peak():
    """Default ``reason="reconcile"`` must not advance peak.
    (This is the path most likely to silently re-base after restart.)
    """
    tracker = EquityTracker(initial_equity_usd=100_000.0)
    # Simulate trade that creates a peak.
    tracker.on_trade_closed(pnl_usd=10_000.0)  # 110K, peak = 110K
    assert tracker.peak_equity == 110_000.0

    # Simulate a 2% intra-day loss (109,800).
    tracker.on_trade_closed(pnl_usd=-200.0)
    assert tracker.peak_equity == 110_000.0  # peak unchanged
    assert tracker.current_equity == 109_800.0

    # Now simulate engine restart: exchange reports equity = 113K
    # (e.g., a missed trade or a reconciliation gain). Peak MUST NOT
    # advance — otherwise the kill-switch drawdown reference floats up
    # silently every restart.
    tracker.set_equity(113_000.0, reason="reconcile")
    assert tracker.peak_equity == 110_000.0, (
        f"R-2: reconcile must not advance peak; got {tracker.peak_equity}"
    )
    assert tracker.current_equity == 113_000.0


def test_set_equity_deposit_advances_peak():
    """Deposits ARE genuine new highs and DO advance peak."""
    tracker = EquityTracker(initial_equity_usd=100_000.0)
    tracker.on_trade_closed(pnl_usd=10_000.0)  # peak = 110K
    assert tracker.peak_equity == 110_000.0

    # User deposits 50K. New equity = 160K.
    tracker.set_equity(160_000.0, reason="deposit")
    assert tracker.peak_equity == 160_000.0  # peak ADVANCED
    assert tracker.current_equity == 160_000.0


def test_set_equity_withdrawal_does_not_advance_peak():
    """Withdrawals reduce equity; they certainly don't make a new peak."""
    tracker = EquityTracker(initial_equity_usd=100_000.0)
    tracker.on_trade_closed(pnl_usd=20_000.0)  # peak = 120K
    assert tracker.peak_equity == 120_000.0

    # User withdraws 5K.
    tracker.set_equity(115_000.0, reason="withdrawal")
    assert tracker.peak_equity == 120_000.0  # unchanged
    assert tracker.current_equity == 115_000.0


def test_set_equity_unknown_reason_raises():
    """Operator typo protection: an unknown reason raises rather than
    silently defaulting to reconcile semantics."""
    tracker = EquityTracker(initial_equity_usd=100_000.0)
    with pytest.raises(ValueError, match="not recognised"):
        tracker.set_equity(105_000.0, reason="reconcile_after_deposit")  # typo


# ── R-1: kill-switch daily-DD semantics ──────────────────────────


def _build_kill_switch(
    max_dd_pct: float = 2.0,
    *,
    ws_heartbeat_at: pd.Timestamp | None = None,
) -> tuple[KillSwitch, EquityTracker]:
    """Construct a kill-switch wired to a fresh EquityTracker.

    ``ws_heartbeat_at`` should be just-before the test's ``now`` so the
    network-outage path doesn't fire incidentally.
    """
    tracker = EquityTracker(initial_equity_usd=100_000.0)
    ks = KillSwitch(
        equity_tracker=tracker,
        max_daily_drawdown_pct=max_dd_pct,
        max_consecutive_errors=3,
        network_outage_seconds=300,
    )
    if ws_heartbeat_at is not None:
        ks.mark_ws_message(at=ws_heartbeat_at)
    return ks, tracker


def test_kill_switch_trips_on_daily_loss_only():
    """Daily PnL < -2% trips. Lifetime-peak DD alone does NOT trip
    if today's PnL is positive or flat.
    """
    day2 = pd.Timestamp("2026-05-08 14:00:00", tz="UTC")
    ks, tracker = _build_kill_switch(
        max_dd_pct=2.0,
        ws_heartbeat_at=day2 - pd.Timedelta(seconds=10),
    )
    # Seed snapshots at start of day 2 (after a multi-day winning streak)
    # Day 2 start: 110K.
    tracker._snapshots.clear()  # clear seeds; rebuild
    from deribit_paper.position.equity import EquitySnapshot
    tracker._snapshots.append(EquitySnapshot(
        timestamp=pd.Timestamp("2026-05-08 00:00:00", tz="UTC"),
        equity_usd=110_000.0,
        realized_pnl_usd=10_000.0,
        peak_equity_usd=110_000.0,
        drawdown_pct=0.0,
        n_trades_closed=1,
    ))
    tracker._current = 110_000.0
    tracker._peak = 110_000.0
    tracker._realized_pnl = 10_000.0

    # Lose 1.5% intra-day (no trip — under threshold).
    tracker._current = 110_000.0 * 0.985
    assert not ks.check_before_signal(now=day2), (
        "1.5% daily loss < 2% threshold should not trip"
    )

    # Lose another 1% (total 2.5% intra-day) → should trip.
    tracker._current = 110_000.0 * 0.975
    tripped = ks.check_before_signal(now=day2)
    assert tripped, "2.5% daily loss should trip the 2% threshold"
    assert "daily PnL" in (ks._state.reason or ""), (
        f"trip reason should mention daily PnL, got: {ks._state.reason}"
    )


def test_kill_switch_does_not_trip_on_lifetime_peak_dd_alone():
    """A 5% drawdown from lifetime peak that occurred days ago does NOT
    trip if today's PnL is flat / mildly positive — the contract is
    DAILY DD, not lifetime peak DD.
    """
    day3 = pd.Timestamp("2026-05-09 14:00:00", tz="UTC")
    ks, tracker = _build_kill_switch(
        max_dd_pct=2.0,
        ws_heartbeat_at=day3 - pd.Timedelta(seconds=10),
    )
    from deribit_paper.position.equity import EquitySnapshot
    tracker._snapshots.clear()

    # Lifetime peak: 120K (set on day 1).
    tracker._peak = 120_000.0
    tracker._realized_pnl = 14_000.0
    # Day 3 start: 114K (5% below lifetime peak).
    tracker._snapshots.append(EquitySnapshot(
        timestamp=pd.Timestamp("2026-05-09 00:00:00", tz="UTC"),
        equity_usd=114_000.0,
        realized_pnl_usd=14_000.0,
        peak_equity_usd=120_000.0,
        drawdown_pct=-5.0,
        n_trades_closed=10,
    ))
    # Today's 'current' = 114,800 (a +0.7% intra-day gain).
    tracker._current = 114_800.0

    tripped = ks.check_before_signal(now=day3)
    assert not tripped, (
        "5% lifetime-peak DD with +0.7% today should NOT trip — "
        "the contract is daily PnL, not lifetime DD."
    )
    # Sanity: drawdown_pct from lifetime peak is still -4.33% (post-gain).
    assert tracker.drawdown_pct < -4.0
    # But daily_return_pct since start-of-day is positive.
    assert tracker.daily_return_pct(now=day3) > 0


def test_kill_switch_trip_message_uses_daily_pct_not_peak_dd():
    """Sanity: the trip reason string must reference daily PnL, not
    peak drawdown — operator messaging needs to match the actual gate.
    """
    day = pd.Timestamp("2026-05-08 14:00:00", tz="UTC")
    ks, tracker = _build_kill_switch(
        max_dd_pct=2.0,
        ws_heartbeat_at=day - pd.Timedelta(seconds=10),
    )
    from deribit_paper.position.equity import EquitySnapshot
    tracker._snapshots.clear()
    tracker._snapshots.append(EquitySnapshot(
        timestamp=pd.Timestamp("2026-05-08 00:00:00", tz="UTC"),
        equity_usd=100_000.0,
        realized_pnl_usd=0.0,
        peak_equity_usd=100_000.0,
        drawdown_pct=0.0,
        n_trades_closed=0,
    ))
    tracker._current = 100_000.0
    tracker._peak = 100_000.0
    tracker._realized_pnl = 0.0

    # Lose 3% today → trip.
    tracker._current = 97_000.0
    ks.check_before_signal(now=day)
    assert "daily PnL" in (ks._state.reason or "")
    assert "-3.00%" in (ks._state.reason or "")
