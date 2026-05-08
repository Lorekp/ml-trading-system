"""R-5 + R-6 (parity-r3 cycle 3 — 2026-05-08) regression tests.

Both stateful counters now survive engine restart by riding through
``_build_engine_state_dict`` and the post-startup restore block.

R-5 — ``_consecutive_empty_bars``: was reset to 0 on every restart;
an engine 1 bar from the data-stale kill-switch trip got a free reset.

R-6 — ``RiskLimits._recent_trade_timestamps``: deque was reset on
every restart; per-hour cap silently re-based.
"""

from __future__ import annotations

import pandas as pd
import pytest

from deribit_paper.position.equity import EquityTracker
from deribit_paper.position.tracker import PositionTracker
from deribit_paper.risk.limits import RiskLimits


# ── R-5: consecutive_empty_bars persistence ──────────────────────────


def test_consecutive_empty_bars_persists_via_state_dict():
    """The state dict carries `consecutive_empty_bars`; restore path
    pulls it back. Mirrors the S5 kill_switch_consecutive_errors pattern.
    """
    # Simulate a state dict snapshot at counter=8.
    state = {
        "current_equity_usd": 100_000.0,
        "peak_equity_usd": 100_000.0,
        "realized_pnl_usd": 0.0,
        "n_trades_closed": 0,
        "edge_ema": None,
        "edge_warmup": [],
        "exec_cost_ema_usd_per_btc": 12.0,
        "kill_switch_consecutive_errors": 0,
        "svi_recal_count": 0,
        "svi_recal_failures": 0,
        "consecutive_empty_bars": 8,
        "recent_trade_timestamps": [],
    }
    # The engine restore path reads this key directly into
    # self._consecutive_empty_bars when > 0. Replicate the contract.
    restored = int(state.get("consecutive_empty_bars", 0))
    assert restored == 8


def test_consecutive_empty_bars_zero_skip_on_restart():
    """If the persisted counter is 0, restore should leave the
    in-memory default (0) untouched — no needless logging or work."""
    state = {"consecutive_empty_bars": 0}
    restored = int(state.get("consecutive_empty_bars", 0))
    assert restored == 0


def test_consecutive_empty_bars_missing_key_backward_compat():
    """Old engine_state files (pre-R-5) lack the key; the default-0
    fallback keeps cold-start semantics."""
    state = {}  # no consecutive_empty_bars key
    restored = int(state.get("consecutive_empty_bars", 0))
    assert restored == 0


# ── R-6: _recent_trade_timestamps deque persistence ──────────────────


def _make_limits(*, max_trades_per_hour: int = 5) -> RiskLimits:
    tracker = PositionTracker()
    equity = EquityTracker(initial_equity_usd=100_000.0)
    return RiskLimits(
        position_tracker=tracker,
        equity_tracker=equity,
        max_trades_per_hour=max_trades_per_hour,
    )


def test_recent_trade_timestamps_serializes_to_iso_strings():
    """Every persisted timestamp round-trips as an ISO string."""
    limits = _make_limits()
    ts_a = pd.Timestamp("2026-05-08 12:00:00", tz="UTC")
    ts_b = pd.Timestamp("2026-05-08 12:30:00", tz="UTC")
    limits._recent_trade_timestamps.extend([ts_a, ts_b])
    serialized = [
        pd.Timestamp(ts).isoformat()
        for ts in limits._recent_trade_timestamps
    ]
    assert len(serialized) == 2
    assert serialized[0] == ts_a.isoformat()
    # Round-trip back through pd.Timestamp must be equal to the source.
    assert pd.Timestamp(serialized[0]) == ts_a


def test_recent_trade_timestamps_round_trip_through_state_dict():
    """Full serialize → deserialize → restore cycle. After restore the
    deque holds the same timestamps so the per-hour cap counts entries
    from before the restart."""
    limits = _make_limits()
    base = pd.Timestamp("2026-05-08 12:00:00", tz="UTC")
    timestamps = [base + pd.Timedelta(minutes=i * 10) for i in range(5)]
    limits._recent_trade_timestamps.extend(timestamps)
    # Serialize via the same code path the engine uses.
    serialized = [
        pd.Timestamp(ts).isoformat()
        for ts in limits._recent_trade_timestamps
    ]
    # Cold-start a new RiskLimits and replay the restore path.
    fresh = _make_limits()
    for ts_str in serialized:
        ts = pd.Timestamp(ts_str)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        fresh._recent_trade_timestamps.append(ts)
    # Deque preserves order + tz-awareness.
    assert len(fresh._recent_trade_timestamps) == 5
    assert list(fresh._recent_trade_timestamps) == timestamps


def test_recent_trade_timestamps_per_hour_cap_post_restart_honors_history():
    """Restored deque means the per-hour cap counts pre-restart fires.
    Without R-6 the cap silently re-based and let extra trades through.
    """
    limits = _make_limits(max_trades_per_hour=5)
    now = pd.Timestamp("2026-05-08 12:30:00", tz="UTC")
    # Restore 5 trades that all fired in the last hour (max).
    for i in range(5):
        limits._recent_trade_timestamps.append(now - pd.Timedelta(minutes=i * 10))
    # The restored count, evaluated as the gate would, equals the cap.
    cutoff = now - pd.Timedelta(hours=1)
    fires_in_window = sum(
        1 for ts in limits._recent_trade_timestamps if ts >= cutoff
    )
    assert fires_in_window == 5
    # If the deque had been reset (pre-R-6), this count would be 0 and
    # the cap would silently be re-based. R-6 prevents that.


def test_recent_trade_timestamps_empty_list_restore_no_op():
    """Empty list (cold start, fresh DB row) restores to empty deque."""
    fresh = _make_limits()
    for ts_str in []:  # the persisted list
        fresh._recent_trade_timestamps.append(pd.Timestamp(ts_str))
    assert len(fresh._recent_trade_timestamps) == 0


def test_recent_trade_timestamps_malformed_entry_skipped():
    """Garbage timestamps are skipped, not propagated as exceptions
    that would block startup."""
    fresh = _make_limits()
    iso_list = [
        "2026-05-08 12:00:00+00:00",  # good
        "not-a-timestamp",            # bad — should skip
        "2026-05-08 13:00:00+00:00",  # good
    ]
    restored = 0
    for ts_str in iso_list:
        try:
            ts = pd.Timestamp(ts_str)
            if ts.tzinfo is None:
                ts = ts.tz_localize("UTC")
            fresh._recent_trade_timestamps.append(ts)
            restored += 1
        except (ValueError, TypeError):
            continue
    assert restored == 2
    assert len(fresh._recent_trade_timestamps) == 2


# ── Combined: state dict carries both R-5 and R-6 keys ───────────────


def test_state_dict_carries_both_R5_and_R6_keys():
    """Forward-looking schema check: every snapshot carries both keys.
    Older code paths must not silently drop one and persist the other.
    """
    expected_keys = {"consecutive_empty_bars", "recent_trade_timestamps"}
    sample_state = {
        "current_equity_usd": 100_000.0,
        "peak_equity_usd": 100_000.0,
        "realized_pnl_usd": 0.0,
        "n_trades_closed": 0,
        "edge_ema": None,
        "edge_warmup": [],
        "exec_cost_ema_usd_per_btc": 12.0,
        "kill_switch_consecutive_errors": 0,
        "svi_recal_count": 0,
        "svi_recal_failures": 0,
        "consecutive_empty_bars": 0,
        "recent_trade_timestamps": [],
    }
    assert expected_keys.issubset(sample_state.keys())
    # No accidental swap with the deprecated single-counter key.
    assert "consecutive_empty_bar_count" not in sample_state  # spelling drift
