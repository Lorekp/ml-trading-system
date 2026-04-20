"""Tests for vol_arb.pnl_throttle.

Critical invariants under test:
1. NO LOOKAHEAD — schedule for day D uses only P&L through D-1.
2. ``None`` params returns identity (byte-identical P&L column).
3. Cooldown duration and scale semantics are exact.
4. Monotonicity — lower threshold → fewer days throttled.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_arb.pnl_throttle import (
    ThrottleParams,
    apply_throttle,
    compute_throttle_schedule,
    walk_forward_select,
)


# ── Fixtures ──────────────────────────────────────────────────

@pytest.fixture
def simple_daily_pnl() -> pd.Series:
    """10 days, alternating gains and losses."""
    idx = pd.date_range("2024-01-01", periods=10, freq="1D", tz="UTC")
    vals = [500.0, -1500.0, 600.0, 400.0, -1200.0, 800.0, -800.0, 300.0, 900.0, -2000.0]
    return pd.Series(vals, index=idx, name="daily_pnl")


@pytest.fixture
def simple_trade_log() -> pd.DataFrame:
    """Mini trade log: 10 trades on 10 different days."""
    idx = pd.date_range("2024-01-01 12:00", periods=10, freq="1D", tz="UTC")
    return pd.DataFrame({
        "signal": [1, -1, 1, 1, -1, 1, -1, 1, -1, 1],
        "net_pnl_nc": [500.0, -1500.0, 600.0, 400.0, -1200.0, 800.0, -800.0, 300.0, 900.0, -2000.0],
        "gross_pnl_nc": [510.0, -1490.0, 610.0, 410.0, -1190.0, 810.0, -790.0, 310.0, 910.0, -1990.0],
        "vega": [50.0] * 10,
    }, index=idx)


# ── 1. No-lookahead invariant ────────────────────────────────

def test_schedule_uses_only_prior_pnl(simple_daily_pnl):
    """Mutating day-D P&L must not change the scale on day D itself.

    This is the gold-standard non-lookahead probe.
    """
    params = ThrottleParams(threshold_usd=-1000.0, scale=0.5, duration_days=2)

    baseline_schedule = compute_throttle_schedule(simple_daily_pnl, params)

    # Mutate day 4 (the -1200 loss) to be a win
    mutated = simple_daily_pnl.copy()
    mutated.iloc[4] = 5000.0
    mutated_schedule = compute_throttle_schedule(mutated, params)

    # Schedule on days 0..4 must be byte-identical (day 4's scale is
    # based on day 3's P&L, which we didn't touch).
    np.testing.assert_array_equal(
        baseline_schedule.iloc[:5].to_numpy(),
        mutated_schedule.iloc[:5].to_numpy(),
        err_msg="Mutating day 4 P&L changed earlier schedule — LEAK!",
    )

    # Day 5 (the day AFTER day 4) may differ — day 5's scale uses day 4's P&L.
    # If day 4 is -1200 (loss) → day 5 triggered. If day 4 is +5000 (win) → not.
    # So day 5's scale should differ. (Sanity check the mutation actually propagates.)
    assert baseline_schedule.iloc[5] != mutated_schedule.iloc[5], (
        "Test design error: mutation should have changed day-5 scale"
    )


def test_day_zero_never_throttled(simple_daily_pnl):
    """Day 0 has no prior P&L, so it must always be at scale=1.0."""
    for thr in (-100.0, -1000.0, -10000.0):
        s = compute_throttle_schedule(
            simple_daily_pnl, ThrottleParams(threshold_usd=thr)
        )
        assert s.iloc[0] == 1.0


# ── 2. Identity under None params ─────────────────────────────

def test_none_params_returns_identity(simple_trade_log):
    out = apply_throttle(simple_trade_log, None)
    pd.testing.assert_frame_equal(out, simple_trade_log.copy())


# ── 3. Cooldown semantics ─────────────────────────────────────

def test_cooldown_duration_exact(simple_daily_pnl):
    """After a trigger, exactly `duration_days` days are throttled."""
    params = ThrottleParams(threshold_usd=-1000.0, scale=0.3, duration_days=3)
    s = compute_throttle_schedule(simple_daily_pnl, params)
    # Day 1 has P&L -1500 → triggers. Day 2,3,4 should be scale=0.3
    # (duration_days=3 from day 2 start).
    assert s.iloc[2] == 0.3, f"Expected 0.3 on day 2, got {s.iloc[2]}"
    assert s.iloc[3] == 0.3, f"Expected 0.3 on day 3, got {s.iloc[3]}"
    assert s.iloc[4] == 0.3, f"Expected 0.3 on day 4, got {s.iloc[4]}"
    # Day 5 has P&L -1200 on day 4 → re-triggers, cooldown extended.
    assert s.iloc[5] == 0.3, f"Expected 0.3 on day 5 (re-trigger), got {s.iloc[5]}"


def test_scale_zero_fully_disables(simple_trade_log):
    params = ThrottleParams(threshold_usd=-1000.0, scale=0.0, duration_days=2)
    out = apply_throttle(simple_trade_log, params)
    # Day 1 P&L is -1500 → triggers cooldown on day 2. With duration=2,
    # days 2 and 3 are throttled; day 4 is back to normal (cooldown expired).
    assert out["net_pnl_nc"].iloc[2] == 0.0  # throttled (cooldown start)
    assert out["net_pnl_nc"].iloc[3] == 0.0  # throttled (cooldown end)
    # Day 4: the day-1 trigger has expired. Day 4's prev is day 3 P&L = 400 > threshold.
    # So day 4 is NOT throttled.
    assert out["net_pnl_nc"].iloc[4] == simple_trade_log["net_pnl_nc"].iloc[4]
    # Day 4's P&L is -1200 → triggers cooldown for days 5 and 6.
    assert out["net_pnl_nc"].iloc[5] == 0.0  # throttled
    assert out["net_pnl_nc"].iloc[6] == 0.0  # throttled (note: day 6 prev=day 5 = 800 > thr; but cooldown from day 4 still active)


# ── 4. Monotonicity ───────────────────────────────────────────

def test_lower_threshold_throttles_fewer_days(simple_daily_pnl):
    """Tighter (more negative) threshold → fewer throttle trigger days."""
    s_loose = compute_throttle_schedule(
        simple_daily_pnl, ThrottleParams(threshold_usd=-500.0)
    )
    s_tight = compute_throttle_schedule(
        simple_daily_pnl, ThrottleParams(threshold_usd=-2000.0)
    )
    assert (s_loose < 1.0).sum() >= (s_tight < 1.0).sum()


# ── 5. Validation ─────────────────────────────────────────────

def test_invalid_scale_raises():
    with pytest.raises(ValueError):
        ThrottleParams(threshold_usd=-1000.0, scale=-0.1)
    with pytest.raises(ValueError):
        ThrottleParams(threshold_usd=-1000.0, scale=1.5)


def test_invalid_duration_raises():
    with pytest.raises(ValueError):
        ThrottleParams(threshold_usd=-1000.0, duration_days=0)


def test_non_monotonic_index_raises():
    pnl = pd.Series(
        [1.0, 2.0, 3.0],
        index=pd.DatetimeIndex(["2024-01-03", "2024-01-01", "2024-01-02"]),
    )
    with pytest.raises(ValueError):
        compute_throttle_schedule(pnl, ThrottleParams(threshold_usd=-100.0))


# ── 6. Apply-to-trade-log smoke ──────────────────────────────

def test_apply_preserves_non_pnl_columns(simple_trade_log):
    params = ThrottleParams(threshold_usd=-1000.0, scale=0.5, duration_days=2)
    out = apply_throttle(simple_trade_log, params)
    # Non-pnl columns unchanged
    pd.testing.assert_series_equal(out["signal"], simple_trade_log["signal"])
    pd.testing.assert_series_equal(out["vega"], simple_trade_log["vega"])


def test_apply_scales_all_pnl_components(simple_trade_log):
    params = ThrottleParams(threshold_usd=-1000.0, scale=0.5, duration_days=2)
    out = apply_throttle(simple_trade_log, params)
    # Day 1 has -1500 → day 2 and 3 are scaled by 0.5
    assert out["net_pnl_nc"].iloc[2] == pytest.approx(simple_trade_log["net_pnl_nc"].iloc[2] * 0.5)
    assert out["gross_pnl_nc"].iloc[2] == pytest.approx(simple_trade_log["gross_pnl_nc"].iloc[2] * 0.5)
    # Day 0 and 1 untouched
    assert out["net_pnl_nc"].iloc[0] == simple_trade_log["net_pnl_nc"].iloc[0]
    assert out["net_pnl_nc"].iloc[1] == simple_trade_log["net_pnl_nc"].iloc[1]


# ── 7. Leakage probe on real trade-log-shaped data ───────────

def test_leakage_on_real_shaped_data():
    """Build a random trade log and mutate a test-month's P&L.
    Prior-month's throttle schedule must not change.
    """
    rng = np.random.default_rng(42)
    idx = pd.date_range("2024-01-01", periods=200, freq="6h", tz="UTC")
    df = pd.DataFrame({
        "signal": rng.choice([-1, 1, 0], size=200, p=[0.4, 0.2, 0.4]),
        "net_pnl_nc": rng.normal(100, 500, size=200),
        "vega": np.abs(rng.normal(50, 20, size=200)),
    }, index=idx)

    params = ThrottleParams(threshold_usd=-500.0, scale=0.5, duration_days=2)

    # Get the throttle schedule
    trades_only = df.loc[df["signal"] != 0]
    daily = trades_only["net_pnl_nc"].resample("1D").sum()
    s_baseline = compute_throttle_schedule(daily, params)

    # Mutate the LAST 20 days of P&L dramatically
    mutated = df.copy()
    last_20_days_cutoff = mutated.index[-20]
    mask = mutated.index >= last_20_days_cutoff
    mutated.loc[mask, "net_pnl_nc"] = -99999.0  # extreme losses

    trades_mut = mutated.loc[mutated["signal"] != 0]
    daily_mut = trades_mut["net_pnl_nc"].resample("1D").sum()
    s_mutated = compute_throttle_schedule(daily_mut, params)

    # The throttle schedule up to the day BEFORE the mutation must be
    # byte-identical. (The day OF the mutation isn't affected because
    # schedule[D] depends on P&L[D-1] which we didn't touch on that
    # first mutated day — but to stay safe, check up to the day BEFORE
    # the mutation boundary.)
    mutation_day = pd.Timestamp("2024-01-01", tz="UTC") + pd.Timedelta(
        days=int((last_20_days_cutoff - idx[0]) / pd.Timedelta(days=1))
    )
    # Compare schedule values for days strictly before the mutation day
    pre_days = [d for d in s_baseline.index if d < mutation_day]
    for d in pre_days:
        if d in s_mutated.index:
            assert s_baseline.loc[d] == s_mutated.loc[d], (
                f"LEAKAGE: schedule[{d}] changed from "
                f"{s_baseline.loc[d]} to {s_mutated.loc[d]}"
            )


# ── 8. Walk-forward selection smoke ──────────────────────────

def test_walk_forward_first_months_passthrough():
    """First `train_months` months must be pass-through (no history)."""
    rng = np.random.default_rng(123)
    idx = pd.date_range("2024-01-01", periods=15 * 30, freq="6h", tz="UTC")
    df = pd.DataFrame({
        "signal": rng.choice([-1, 1, 0], size=len(idx), p=[0.4, 0.2, 0.4]),
        "net_pnl_nc": rng.normal(50, 300, size=len(idx)),
        "vega": np.abs(rng.normal(40, 15, size=len(idx))),
    }, index=idx)

    grid = [
        ThrottleParams(threshold_usd=-500, scale=0.5, duration_days=2),
        ThrottleParams(threshold_usd=-1000, scale=0.5, duration_days=2),
        ThrottleParams(threshold_usd=-1e9, scale=1.0, duration_days=1),  # pass-through
    ]
    result = walk_forward_select(df, grid, train_months=12)

    # First 12 rows should be pass-through (scale=1.0)
    first12 = result.iloc[:12]
    assert (first12["scale"] == 1.0).all()
    assert first12["passthrough"].all()

    # Later rows may pick any grid candidate
    later = result.iloc[12:]
    assert not later["passthrough"].any()
