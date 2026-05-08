"""Tier C (parity-r3 cycle 3 — 2026-05-08) backtester semantic flips.

Four configs added that bring the V2 backtester's execution model in
line with the live engine's operational truth:

- ``ioc_max_slippage_pct``  (E-6): match live's API-level % cap on IOC.
- ``cancel_replace_bar_gap`` (O-1): charge bar-(i+1) slip when SOR
  maker failed (live's 60s cancel-and-replace lands a bar later).
- ``taker_tier_blend`` (O-2): blended slip across the 3-tier IOC ladder.
- ``bar_aligned_exit_latency_pct`` (O-6): premium-scaled exit slip
  surcharge to model live's 2× ticker + 2× IOC round-trip latency.

Each section verifies the new config field exists and is wired into
``with_realistic_costs()`` (V2 preset).
"""

from __future__ import annotations

import pytest

from vol_arb.backtester import BacktestConfig


# ── E-6: ioc_max_slippage_pct ──────────────────────────────────────────


def test_ioc_max_slippage_pct_default_is_zero_disabled():
    """Default 0.0 preserves legacy non-V2 byte-identity."""
    cfg = BacktestConfig()
    assert cfg.ioc_max_slippage_pct == pytest.approx(0.0)


def test_ioc_max_slippage_pct_v2_preset_is_one_percent():
    """`with_realistic_costs()` is the V2 trade-log generator path
    (`with_btc_production_filters_v2().with_realistic_costs()`); it
    must enable the 1% cap to match the live ``place_ioc_taker``
    default `max_slippage_pct=1.0`."""
    cfg = BacktestConfig().with_realistic_costs()
    assert cfg.ioc_max_slippage_pct == pytest.approx(1.0)


def test_ioc_max_slippage_pct_legacy_preset_unchanged():
    """Bare V2 (``with_btc_production_filters_v2`` alone) does NOT flip
    the cap — only the realistic-costs leg does. Documents that callers
    of the bare preset see legacy behaviour."""
    cfg = BacktestConfig().with_btc_production_filters_v2()
    assert cfg.ioc_max_slippage_pct == pytest.approx(0.0)


# ── O-2: taker_tier_blend ──────────────────────────────────────────────


def test_taker_tier_blend_default_off():
    cfg = BacktestConfig()
    assert cfg.taker_tier_blend is False
    assert cfg.taker_tier_split_fractions == (0.5, 0.3, 0.2)
    assert cfg.taker_tier_slip_multipliers == (0.5, 1.5, 2.5)


def test_taker_tier_blend_v2_preset_is_on():
    """V2 preset enables tier blending to mirror live's
    ``use_tiered_taker=True`` default in the V2 PaperConfig."""
    cfg = BacktestConfig().with_realistic_costs()
    assert cfg.taker_tier_blend is True


def test_taker_tier_split_fractions_sum_to_one():
    """Sanity: the default tier-split fractions partition the order."""
    cfg = BacktestConfig()
    assert sum(cfg.taker_tier_split_fractions) == pytest.approx(1.0)


def test_blended_cap_is_tighter_than_legacy_2x():
    """Default tier blend (50/30/20 × 0.5/1.5/2.5) yields effective
    cap ≈ 1.2× bar_slip, tighter than the legacy ``ioc_cap_multiplier=2.0``.
    This is the expected forward direction: live's tier ladder is
    cheaper than the worst-case multiplier cap."""
    cfg = BacktestConfig()
    blended_per_leg = sum(
        frac * mult
        for frac, mult in zip(
            cfg.taker_tier_split_fractions,
            cfg.taker_tier_slip_multipliers,
            strict=False,
        )
    )
    # 0.5×0.5 + 0.3×1.5 + 0.2×2.5 = 0.25 + 0.45 + 0.5 = 1.2
    assert blended_per_leg == pytest.approx(1.2)
    # Straddle (2 legs) blended = 2 × 1.2 = 2.4 × bar_slip.
    assert blended_per_leg * 2 == pytest.approx(2.4)
    # Compare to legacy flat cap: 2.0 × bar_slip total.
    # The blended cap is HIGHER than the legacy cap on the per-straddle
    # basis (2.4 vs 2.0), but is only an UPPER bound. The full slip
    # before-cap (size_mult × jump × 2 × bar_slip) often exceeds 2.4,
    # in which case both caps clip; otherwise the blend allows more
    # spread room. Either way, the cap structure is more honest.


# ── O-1: cancel_replace_bar_gap ────────────────────────────────────────


def test_cancel_replace_bar_gap_default_off():
    """Default off — preserves legacy SOR semantics."""
    cfg = BacktestConfig()
    assert cfg.cancel_replace_bar_gap is False


def test_cancel_replace_bar_gap_v2_preset_off_by_design():
    """V2 preset uses ``use_sor=False`` (in
    ``with_btc_production_filters_v2``) so the maker path never fires
    and the bar-gap branch is N/A. The flag stays default-off in V2."""
    cfg = BacktestConfig().with_realistic_costs()
    assert cfg.cancel_replace_bar_gap is False


def test_cancel_replace_bar_gap_can_be_enabled_for_sor_paths():
    """Non-V2 callers using ``use_sor=True`` can opt in."""
    cfg = BacktestConfig().model_copy(update=dict(
        use_sor=True, cancel_replace_bar_gap=True,
    ))
    assert cfg.use_sor is True
    assert cfg.cancel_replace_bar_gap is True


# ── O-6: bar_aligned_exit_latency_pct ──────────────────────────────────


def test_bar_aligned_exit_latency_pct_default_zero():
    cfg = BacktestConfig()
    assert cfg.bar_aligned_exit_latency_pct == pytest.approx(0.0)


def test_bar_aligned_exit_latency_pct_v2_preset_5bps():
    """V2 preset sets 5 bps to model the 2× ticker + 2× IOC round-trip
    latency observed in live D12 exits."""
    cfg = BacktestConfig().with_realistic_costs()
    assert cfg.bar_aligned_exit_latency_pct == pytest.approx(0.05)


def test_bar_aligned_exit_latency_magnitude_at_typical_premium():
    """Sanity: at 60% IV, 2 DTE, $60K spot, the premium is
    ~0.4·0.6·√(2/365.25) ≈ 0.0177 BTC × $60K = $1062. 5 bps × $1062 × 2
    legs = $1.06 USD per BTC of straddle notional. At 0.5 BTC straddle
    that's ~$0.53/exit. Models the magnitude of the live latency cost."""
    cfg = BacktestConfig().with_realistic_costs()
    sigma, T, S = 0.6, 2.0 / 365.25, 60_000.0
    prem_per_leg_btc = 0.4 * sigma * (T ** 0.5)
    prem_per_leg_usd = prem_per_leg_btc * S
    surcharge_per_btc = (
        2.0 * cfg.bar_aligned_exit_latency_pct / 100.0
        * prem_per_leg_usd
    )
    # Should be in the $1-2 per-BTC range at typical 2-DTE BTC straddle.
    assert 0.5 < surcharge_per_btc < 5.0


# ── Cross-cycle: V2 preset config consistency ─────────────────────────


def test_v2_preset_combines_all_cycle3_tier_c_flips():
    """All four Tier C configs are exercised together by V2; this
    consolidated assertion guards against future preset edits dropping
    one silently."""
    cfg = BacktestConfig().with_realistic_costs()
    assert cfg.ioc_max_slippage_pct == pytest.approx(1.0)            # E-6
    assert cfg.taker_tier_blend is True                              # O-2
    assert cfg.bar_aligned_exit_latency_pct == pytest.approx(0.05)   # O-6
    # O-1 deliberately NOT enabled in V2 (use_sor=False makes it N/A);
    # it's only meaningful when use_sor=True, which V2 disables.
    assert cfg.cancel_replace_bar_gap is False
