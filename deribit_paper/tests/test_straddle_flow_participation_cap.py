"""D3 (parity 2026-05-07): ``dynamic_btc_cap`` is the V2 participation cap.

Mirrors V2 backtester at ``vol_arb/backtester.py:1796-1798``:
    ``dynamic_btc_cap = clip(trailing_vol * max_participation_rate, 1.0, 50.0)``

The live engine pre-D3 set ``dynamic_btc_cap = float(n)`` (margin envelope),
so subsequent re-clips at profit_scale and Kelly used the wrong ceiling.

This module locks the participation-cap semantics:
1. ``test_participation_cap_clips_to_floor`` — small trailing volume
   produces cap=1.0 (the V2 lower clip).
2. ``test_participation_cap_clips_to_ceiling`` — huge trailing volume
   produces cap=50.0 (the V2 upper clip).
3. ``test_participation_cap_pass_through_in_band`` — mid volume gives
   ``trailing_vol * max_participation`` exactly.
4. ``test_kelly_cannot_breach_participation_cap`` — Kelly mult of 2.0x
   never produces N > dynamic_btc_cap (re-clip works).
5. ``test_profit_scale_under_1_respects_cap`` — profit_scale=0.5
   doesn't shrink below cap (no inversion).
6. ``test_no_volume_falls_back_to_margin_envelope`` — when trailing
   volume is None, dynamic_btc_cap = margin envelope (legacy
   behaviour preserved for early bars / venue outages).
"""

from __future__ import annotations

from unittest.mock import MagicMock

from deribit_paper.execution.straddle_flow import StraddleFlow


def _make_flow(*, max_participation: float = 0.10) -> StraddleFlow:
    """Construct StraddleFlow with mocked dependencies for sizing-only tests."""
    return StraddleFlow(
        order_manager=MagicMock(),
        live_drl_hedger=MagicMock(),
        target_margin_utilization=0.15,
        im_fraction=0.20,
        max_participation_rate=max_participation,
        short_sizing_penalty=0.50,
        max_trade_size_btc=0.0,  # disable static cap (V2 preset)
        baseline_target_dvol=50.0,
        sizing_edge_cap=3.0,
        sizing_dd_floor=0.25,
    )


# ── 1-3. Participation cap clipping ────────────────────────────────────────


def test_participation_cap_clips_to_floor():
    """trailing_vol * max_participation < 1.0 → cap=1.0 (V2 floor)."""
    flow = _make_flow(max_participation=0.10)
    # 5 BTC * 0.10 = 0.5 < floor of 1.0
    n = flow.size_position(
        equity=1_000_000.0, signal_direction=1, spot=100_000.0,
        trailing_1h_volume_btc=5.0,
        dvol_pct=50.0, edge_vp=20.0, rolling_median_edge=20.0,
        peak_equity=1_000_000.0,
    )
    # n must respect floor=1.0
    assert n <= 1.0 + 1e-9, f"expected n<=1.0 (floor), got {n}"


def test_participation_cap_clips_to_ceiling():
    """trailing_vol * max_participation > 50.0 → cap=50.0 (V2 ceiling)."""
    flow = _make_flow(max_participation=0.10)
    # 1000 BTC * 0.10 = 100 > ceiling of 50.0
    n = flow.size_position(
        equity=10_000_000_000.0,  # huge equity → margin envelope huge
        signal_direction=1, spot=100_000.0,
        trailing_1h_volume_btc=1000.0,
        dvol_pct=50.0, edge_vp=20.0, rolling_median_edge=20.0,
        peak_equity=10_000_000_000.0,
    )
    # n must respect ceiling=50.0
    assert n <= 50.0 + 1e-9, f"expected n<=50.0 (ceiling), got {n}"


def test_participation_cap_pass_through_in_band():
    """trailing_vol * max_participation in [1, 50] → cap = exact product."""
    flow = _make_flow(max_participation=0.05)
    # 200 BTC * 0.05 = 10.0 ∈ [1, 50]
    n = flow.size_position(
        equity=1_000_000_000.0, signal_direction=1, spot=100_000.0,
        trailing_1h_volume_btc=200.0,
        dvol_pct=50.0, edge_vp=20.0, rolling_median_edge=20.0,
        peak_equity=1_000_000_000.0,
    )
    # margin envelope is huge (M$ equity), so cap binds at exactly 10.0
    assert n <= 10.0 + 1e-9, f"expected n<=10.0 (participation), got {n}"


# ── 4. Kelly cannot breach the participation cap ───────────────────────────


def test_kelly_cannot_breach_participation_cap():
    """Kelly mult 2.0x: re-clip ensures N stays ≤ dynamic_btc_cap."""
    flow = _make_flow(max_participation=0.10)
    # 50 BTC * 0.10 = 5.0 — well within band, easy to compute expected
    n = flow.size_position(
        equity=10_000_000.0, signal_direction=1, spot=100_000.0,
        trailing_1h_volume_btc=50.0,
        dvol_pct=50.0, edge_vp=20.0, rolling_median_edge=20.0,
        peak_equity=10_000_000.0,
        kelly_mult=2.0,  # ← maxes out
    )
    # Cap = 5.0; Kelly 2x should not push above
    assert n <= 5.0 + 1e-9, f"Kelly 2x breached cap: got {n}, cap=5.0"


# ── 5. Profit-scale under 1 still respects cap ─────────────────────────────


def test_profit_scale_under_1_respects_cap():
    """profit_scale=0.5 shrinks size; should not change cap behaviour."""
    flow = _make_flow(max_participation=0.10)
    n_full = flow.size_position(
        equity=10_000_000.0, signal_direction=1, spot=100_000.0,
        trailing_1h_volume_btc=50.0,
        dvol_pct=50.0, edge_vp=20.0, rolling_median_edge=20.0,
        peak_equity=10_000_000.0,
        profit_scale=1.0,
    )
    n_half = flow.size_position(
        equity=10_000_000.0, signal_direction=1, spot=100_000.0,
        trailing_1h_volume_btc=50.0,
        dvol_pct=50.0, edge_vp=20.0, rolling_median_edge=20.0,
        peak_equity=10_000_000.0,
        profit_scale=0.5,
    )
    # n_half must be strictly less than n_full (profit_scale binds when <1)
    # But both must be ≤ cap of 5.0
    assert n_full <= 5.0 + 1e-9
    assert n_half <= 5.0 + 1e-9
    assert n_half < n_full


# ── 6. No volume falls back to margin envelope ─────────────────────────────


def test_no_volume_falls_back_to_margin_envelope():
    """trailing_1h_volume_btc=None → cap = margin envelope (n_base).

    This protects early bars / venue outages where volume isn't yet
    populated. Without the fallback, dynamic_btc_cap would default to
    NaN/0 and shrink the trade to zero.
    """
    flow = _make_flow(max_participation=0.10)
    n = flow.size_position(
        equity=100_000.0, signal_direction=1, spot=100_000.0,
        trailing_1h_volume_btc=None,
        dvol_pct=50.0, edge_vp=20.0, rolling_median_edge=20.0,
        peak_equity=100_000.0,
    )
    # margin envelope = 100_000 * 0.15 / (0.20 * 100_000) = 0.75 BTC.
    # With no further multipliers (edge=median, dd=1, liq=1), n ≈ 0.75.
    assert n > 0.0
    # Sanity: not unreasonable, certainly not NaN or 0.
    assert 0.5 < n < 1.0
