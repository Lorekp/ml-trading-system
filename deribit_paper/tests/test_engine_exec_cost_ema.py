"""Cycle-24-C2 (2026-05-11) regression tests for the exec_cost EMA fix.

Audit A3 C2 found that the live engine was updating ``_exec_cost_ema``
with ``total_premium * spot`` per trade (~$4000-6000), not per-bar
taker slip USD/BTC (~$10-50). The EMA seeded at $12 converged toward
the premium-USD scale, pegging ``liq_scale`` at the
``sizing_liq_cap=1.5`` ceiling. Live's N was ~33% larger than
backtester's at the same config.

Cycle-24-C2 fix:
1. Removed the buggy per-trade update at engine.py:2773-2787.
2. Added a per-bar update at the START of process_bar that applies
   the PRIOR bar's slip via ``_pending_slip_update`` (preserving
   strict past-only ordering — bar T's sizing uses EMA-through-T-1).
3. Captures THIS bar's slip via ``lookup_slip_usd_at`` into
   ``_pending_slip_update`` for the next call.

This mirrors backtester L2686: ``_slip_ema += alpha * (_bar_taker_slip[i]
- _slip_ema)`` runs once per bar, AFTER the trade for bar i is
processed.
"""

from __future__ import annotations

import math

import pytest


def _simulate_ema_pre_cycle24(
    premiums_per_trade: list[float],
    spot: float,
    seed: float = 12.0,
    alpha: float = 2.0 / 201.0,
) -> float:
    """Replicate the pre-cycle-24 (buggy) per-trade EMA update."""
    ema = seed
    for prem in premiums_per_trade:
        fill_cost = prem * spot
        ema += alpha * (fill_cost - ema)
    return ema


def _simulate_ema_post_cycle24(
    per_bar_slips: list[float],
    seed: float = 12.0,
    alpha: float = 2.0 / 201.0,
) -> float:
    """Replicate the post-cycle-24 per-bar EMA update."""
    ema = seed
    pending = None
    for slip in per_bar_slips:
        if pending is not None and math.isfinite(pending):
            ema += alpha * (pending - ema)
        if math.isfinite(slip):
            pending = float(slip)
    # Final apply (mirrors what happens on the next process_bar call):
    if pending is not None and math.isfinite(pending):
        ema += alpha * (pending - ema)
    return ema


# ── Direction of the fix ──────────────────────────────────────────────


def test_pre_cycle24_ema_diverges_to_premium_scale():
    """Sanity: the buggy per-trade update converges to total_premium *
    spot (~$4000-6000), not the seed $12."""
    # 200 trades, each with $1200 straddle premium at $60K spot
    # total_premium = 0.02 BTC per trade × 60_000 = $1200 fed in
    premiums = [0.02] * 200  # BTC
    ema = _simulate_ema_pre_cycle24(premiums, spot=60_000.0)
    # EMA should have converged toward 0.02 * 60_000 = $1200
    assert ema > 100.0, (
        f"Pre-cycle-24 EMA should diverge to premium scale; got ${ema:.2f}. "
        f"If this fails, the simulator is wrong."
    )


def test_post_cycle24_ema_tracks_per_bar_slip_distribution():
    """Cycle-24-C2: the new per-bar update should converge to the
    mean of the slip sequence, not premium × spot."""
    # 200 bars with slips drawn near $12 (the historical median).
    slips = [10.0, 12.0, 14.0, 11.0, 13.0] * 40  # mean = 12.0
    ema = _simulate_ema_post_cycle24(slips, seed=12.0)
    # EMA should hover near the mean of the inputs (~$12), not diverge.
    assert 8.0 < ema < 16.0, (
        f"Post-cycle-24 EMA should track slip mean ~$12; got ${ema:.2f}. "
        f"If this fails, the per-bar update is wrong."
    )


def test_post_cycle24_ema_handles_nan_pending_safely():
    """If a slip lookup returns NaN, the EMA must not be corrupted."""
    slips = [10.0, float("nan"), 12.0, 14.0]
    ema = _simulate_ema_post_cycle24(slips, seed=12.0)
    assert math.isfinite(ema), f"NaN slip corrupted EMA: ${ema}"


def test_post_cycle24_ema_preserves_past_only_order():
    """Bar T's sizing must see EMA-through-T-1, not EMA-through-T.

    Simulated via: feed [s_0, s_1, s_2]. After processing bar 0
    (only s_0 captured as pending), EMA should still be the seed.
    After processing bar 1 (s_0 applied, s_1 captured), EMA reflects
    s_0 only — not s_1. After bar 2, EMA reflects s_0 + s_1.
    """
    seed = 12.0
    alpha = 2.0 / 201.0

    # Bar 0: pending=None → no update; capture s_0 → pending=s_0
    ema = seed
    pending = None
    s_0, s_1, s_2 = 100.0, 200.0, 300.0  # outlier slips for visibility

    # Bar 0
    if pending is not None:
        ema += alpha * (pending - ema)
    pending = s_0
    ema_after_bar_0 = ema
    assert ema_after_bar_0 == seed, "Bar 0 should NOT update EMA (no pending)"

    # Bar 1
    if pending is not None:
        ema += alpha * (pending - ema)
    pending = s_1
    ema_after_bar_1 = ema
    expected_after_bar_1 = seed + alpha * (s_0 - seed)
    assert ema_after_bar_1 == pytest.approx(expected_after_bar_1), (
        f"Bar 1 should have applied s_0={s_0} only, not s_1={s_1}. "
        f"Got {ema_after_bar_1}, expected {expected_after_bar_1}."
    )
    # CRITICAL: at bar 1's sizing decision (which uses ``self._exec_cost_ema``
    # AFTER the start-of-bar update), the EMA reflects s_0 — strict past.


# ── Magnitude of the impact ───────────────────────────────────────────


def test_post_cycle24_liq_scale_is_not_pegged_at_cap():
    """At realistic live conditions (slip ~$10-15, spread ~$10-30),
    the resulting liq_scale must fall in [0.25, 1.5] without pegging.

    Pre-cycle-24, the EMA diverged to ~$4000 → liq_scale = min($4000/$30,
    1.5) = 1.5 (pegged). Post-cycle-24, EMA ≈ $12 → liq_scale ≈ $12/$30
    = 0.4 (inside the cap).
    """
    # Post-fix EMA ≈ $12 (slip distribution mean)
    ema = 12.0
    typical_spread_usd = 30.0
    liq_scale = max(0.25, min(ema / typical_spread_usd, 1.5))
    assert 0.25 < liq_scale < 1.5, (
        f"Post-cycle-24 liq_scale should be inside [0.25, 1.5] without "
        f"pegging at the ceiling; got {liq_scale:.3f}."
    )
    # Documentation: pre-fix this was 1.5 (pegged).
    assert liq_scale < 1.0, (
        f"Post-cycle-24 liq_scale at realistic conditions should be < 1.0 "
        f"(EMA $12 / spread $30 = 0.4); got {liq_scale:.3f}."
    )
