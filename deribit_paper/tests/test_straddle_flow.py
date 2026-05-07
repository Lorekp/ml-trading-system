"""Tests for StraddleFlow — sizing, lifecycle orchestration, abort paths."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from deribit_paper.execution.order_manager import OrderResult
from deribit_paper.execution.straddle_flow import StraddleFlow, TradeOutcome
from deribit_paper.features.straddle import StraddleSelection
from deribit_paper.features.tsrv_live import BarTSRV
from deribit_paper.signal.drl_hedger_live import LiveDRLHedger
from deribit_paper.signal.model_loader import ModelBundle
from deribit_paper.signal.predictor import Signal


def _make_signal(direction: int = -1) -> Signal:
    straddle = StraddleSelection(
        call_instrument="BTC-12APR26-60000-C",
        put_instrument="BTC-12APR26-60000-P",
        strike=60_000.0,
        expiry_dt=pd.Timestamp("2026-04-12 08:00:00", tz="UTC"),
        index_price=60_000.0,
        moneyness=1.0,
        dte_days=2.0,
    )
    bar_tsrv = BarTSRV(
        tsrv=0.001, rv_fast=0.001, rv_slow=0.001, bpv=0.001, medrv=0.001,
        continuous_var=0.001, jump_var=0.0, jump_fraction=0.05,
        rs_down=0.0005, rs_up=0.0005, down_jump_fraction=0.0, up_jump_fraction=0.0,
        n_ticks=100, n_sub_bars=10, n_dash=9.0,
        vol_ann=0.8, vol_ann_continuous=0.8,
    )
    return Signal(
        timestamp=pd.Timestamp("2026-04-10 12:00:00", tz="UTC"),
        direction=direction,
        edge_vol_pts=-1.5 if direction == -1 else 1.5,
        rv_pred_pct=73.0,
        iv_used_pct=75.0,
        bar_tsrv=bar_tsrv,
        straddle=straddle,
        reason="test signal",
    )


def _stub_bundle() -> ModelBundle:
    drl = MagicMock()
    drl.predict.return_value = (np.array([[0.0]], dtype=np.float32), None)
    return ModelBundle(
        month_key="stub",
        lgbm_model=None, lgbm_features=[],
        transvol_model=None, transvol_config=None,
        transvol_feat_mean=None, transvol_feat_std=None,
        transvol_time_mean=None, transvol_time_std=None,
        drl_policy=drl,
        drl_vec_normalize_mean=None,
        drl_vec_normalize_var=None,
        drl_vec_normalize_clip=10.0,
    )


def _fill_result(instrument: str, side: str, amount: float, fill_price: float = 50_000.0) -> OrderResult:
    return OrderResult(
        client_order_id=f"stub_{instrument}",
        order_id=f"oid_{instrument}",
        instrument_name=instrument,
        side=side,
        amount_requested=amount,
        filled_amount=amount,
        average_price=fill_price,
        state="filled",
        is_dry_run=True,
    )


# ─── Sizing ──────────────────────────────────────────────────────────

def test_size_position_basic_long():
    flow = StraddleFlow(
        order_manager=None, live_drl_hedger=None,
        target_margin_utilization=0.02,
        im_fraction=0.10,
        short_sizing_penalty=0.5,
    )
    # equity=10 BTC, spot=$60k → 10 * 0.02 / (0.10 * 60000) = 3.33e-5 — too small
    # equity=$10M (in USD-equivalent), spot=$60k — wait, equity here is in BTC
    # Actually the backtester uses USD equity. Let me reread.
    # n = equity_USD * margin_util / (im * spot_USD)
    # With equity=$1_000_000, margin=0.02, im=0.1, spot=60000
    #   n = 1_000_000 * 0.02 / (0.1 * 60000) = 20000/6000 = 3.33 BTC
    n = flow.size_position(
        equity=1_000_000.0,
        signal_direction=1,
        spot=60_000.0,
    )
    assert n == pytest.approx(2.0)  # capped at max_trade_size 2.0 BTC


def test_size_position_short_penalty_applied():
    flow = StraddleFlow(
        order_manager=None, live_drl_hedger=None,
        target_margin_utilization=0.05,  # production default; 0.02 is below the 0.05 DVOL floor
        im_fraction=0.10,
        short_sizing_penalty=0.5,
        max_trade_size_btc=10.0,
    )
    # n = 1M * 0.05 / (0.10 * 60K) = 8.33 BTC × 0.5 short penalty = 4.17 → rounded to 4.2
    n = flow.size_position(
        equity=1_000_000.0,
        signal_direction=-1,
        spot=60_000.0,
    )
    assert n == pytest.approx(4.2)


def test_size_position_participation_cap():
    """D3 (parity 2026-05-07): dynamic_btc_cap is the V2 participation
    cap with floor=1.0 / ceiling=50.0. trailing_vol=5.0 BTC ×
    max_participation=0.10 = 0.5 → clipped UP to 1.0 (V2 floor at
    backtester.py:1796-1798). Pre-D3 the live engine produced 0.5
    here; under V2-parity the cap floor binds at 1.0.
    """
    flow = StraddleFlow(
        order_manager=None, live_drl_hedger=None,
        target_margin_utilization=0.02,
        im_fraction=0.10,
        max_participation_rate=0.10,
        max_trade_size_btc=10.0,
    )
    # margin envelope: 1M * 0.02 / (0.10 * 60K) = 3.33 BTC
    # participation cap: clip(5.0 * 0.10, 1.0, 50.0) = clip(0.5, 1.0, 50.0) = 1.0
    # n = min(3.33, 1.0) = 1.0
    n = flow.size_position(
        equity=1_000_000.0,
        signal_direction=1,
        spot=60_000.0,
        trailing_1h_volume_btc=5.0,
    )
    assert n == pytest.approx(1.0)


def test_size_position_below_min_returns_zero():
    flow = StraddleFlow(
        order_manager=None, live_drl_hedger=None,
        target_margin_utilization=0.02,
        im_fraction=0.10,
    )
    # Tiny equity → tiny N → below 0.1 BTC minimum
    n = flow.size_position(
        equity=100.0,
        signal_direction=1,
        spot=60_000.0,
    )
    assert n == 0.0


def test_size_position_floor_returns_minimum_at_small_equity():
    """With min_trade_floor=True, returns 0.1 BTC even when formula gives 0."""
    flow = StraddleFlow(
        order_manager=None, live_drl_hedger=None,
        target_margin_utilization=0.05,
        im_fraction=0.10,
        min_trade_floor=True,
    )
    n = flow.size_position(
        equity=750.0,
        signal_direction=-1,
        spot=100_000.0,
    )
    assert n == 0.1


def test_size_position_floor_off_returns_zero():
    """Without min_trade_floor, same equity returns 0.0."""
    flow = StraddleFlow(
        order_manager=None, live_drl_hedger=None,
        target_margin_utilization=0.05,
        im_fraction=0.10,
        min_trade_floor=False,
    )
    n = flow.size_position(
        equity=750.0,
        signal_direction=-1,
        spot=100_000.0,
    )
    assert n == 0.0


def test_size_position_formula_takes_over_at_100k():
    """At $100K equity, formula produces 0.5 BTC (backtester parity)."""
    flow = StraddleFlow(
        order_manager=None, live_drl_hedger=None,
        target_margin_utilization=0.05,
        im_fraction=0.10,
        min_trade_floor=True,
    )
    n = flow.size_position(
        equity=100_000.0,
        signal_direction=1,
        spot=100_000.0,
    )
    assert n == 0.5  # formula: 100k * 0.05 / (0.10 * 100k) = 0.5


def test_size_position_floor_transition_at_20k():
    """At $20K equity, formula produces exactly 0.1 — floor is non-binding."""
    flow = StraddleFlow(
        order_manager=None, live_drl_hedger=None,
        target_margin_utilization=0.05,
        im_fraction=0.10,
        min_trade_floor=True,
    )
    n = flow.size_position(
        equity=20_000.0,
        signal_direction=1,
        spot=100_000.0,
    )
    assert n == 0.1  # formula: 20k * 0.05 / (0.10 * 100k) = 0.1


# ─── Realistic Deribit IM (im_fraction=0.20) ─────────────────────────


def test_deribit_im_floor_at_750():
    """With realistic IM (0.20), $750 equity still gets floor 0.1 BTC."""
    flow = StraddleFlow(
        order_manager=None, live_drl_hedger=None,
        target_margin_utilization=0.05,
        im_fraction=0.20,
        min_trade_floor=True,
    )
    n = flow.size_position(
        equity=750.0, signal_direction=-1, spot=100_000.0,
    )
    assert n == 0.1  # formula: 750*0.05/(0.20*100k)=0.001875 → 0.0 → floor


def test_deribit_im_formula_transition_at_40k():
    """With realistic IM (0.20), formula produces exactly 0.1 at $40K."""
    flow = StraddleFlow(
        order_manager=None, live_drl_hedger=None,
        target_margin_utilization=0.05,
        im_fraction=0.20,
        min_trade_floor=True,
    )
    n = flow.size_position(
        equity=40_000.0, signal_direction=1, spot=100_000.0,
    )
    assert n == 0.1  # formula: 40k * 0.05 / (0.20 * 100k) = 0.1


def test_deribit_im_at_100k_equity():
    """With realistic IM (0.20), $100K produces 0.2 BTC (more conservative than backtester)."""
    flow = StraddleFlow(
        order_manager=None, live_drl_hedger=None,
        target_margin_utilization=0.05,
        im_fraction=0.20,
        min_trade_floor=True,
    )
    n = flow.size_position(
        equity=100_000.0, signal_direction=1, spot=100_000.0,
    )
    # formula: 100k * 0.05 / (0.20 * 100k) = 0.25 → rounds to 0.2
    assert n == 0.2


# ─── composite_v2 sizing ─────────────────────────────────────────────

def test_dvol_scaling_high_dvol_reduces_margin():
    """At DVOL = 2× baseline (100%), dynamic margin scales to half, floored at 0.05."""
    flow = StraddleFlow(
        order_manager=None, live_drl_hedger=None,
        target_margin_utilization=0.10,  # > 0.05 floor so scaling is visible
        im_fraction=0.10,
        baseline_target_dvol=50.0,
        max_trade_size_btc=100.0,
    )
    # dvol=100 → margin = 0.10 * (50/100) = 0.05 (hits the 0.05 floor exactly)
    n_high_dvol = flow.size_position(
        equity=1_000_000.0, signal_direction=1, spot=60_000.0, dvol_pct=100.0,
    )
    # dvol=50 (baseline) → margin = 0.10 * (50/50) = 0.10
    n_baseline = flow.size_position(
        equity=1_000_000.0, signal_direction=1, spot=60_000.0, dvol_pct=50.0,
    )
    # High-DVOL should produce half the notional
    assert n_baseline == pytest.approx(2 * n_high_dvol, rel=0.05)


def test_dvol_scaling_capped_at_target():
    """Low DVOL cannot push margin above target_margin_utilization."""
    flow = StraddleFlow(
        order_manager=None, live_drl_hedger=None,
        target_margin_utilization=0.05,
        im_fraction=0.10,
        baseline_target_dvol=50.0,
        max_trade_size_btc=100.0,
    )
    # dvol=10 → uncapped = 0.05 * (50/10) = 0.25, but capped at target=0.05
    n = flow.size_position(
        equity=1_000_000.0, signal_direction=1, spot=60_000.0, dvol_pct=10.0,
    )
    # Same as baseline (0.05 margin either way)
    n_baseline = flow.size_position(
        equity=1_000_000.0, signal_direction=1, spot=60_000.0, dvol_pct=50.0,
    )
    assert n == pytest.approx(n_baseline)


def test_edge_scale_above_median_increases_size():
    """edge_vp > rolling_median_edge → edge_scale > 1 → larger position.

    D3 (parity 2026-05-07): pass ``trailing_1h_volume_btc=500.0`` so
    the participation cap = 50.0 BTC (V2 ceiling), keeping the cap from
    binding on the edge_scale=3 case (margin envelope 8.33 BTC × 3 = 25
    BTC < 50 cap). Tests edge_scale in isolation.
    """
    flow = StraddleFlow(
        order_manager=None, live_drl_hedger=None,
        target_margin_utilization=0.05,
        im_fraction=0.10,
        sizing_edge_cap=3.0,
        max_trade_size_btc=100.0,
    )
    # Baseline: edge = median → edge_scale = 1.0
    n_median = flow.size_position(
        equity=1_000_000.0, signal_direction=1, spot=60_000.0,
        trailing_1h_volume_btc=500.0,  # cap = 50.0 (ceiling), non-binding
        edge_vp=30.0, rolling_median_edge=30.0,
    )
    # 3× median edge → edge_scale = 3.0 (capped at sizing_edge_cap)
    n_high = flow.size_position(
        equity=1_000_000.0, signal_direction=1, spot=60_000.0,
        trailing_1h_volume_btc=500.0,
        edge_vp=90.0, rolling_median_edge=30.0,
    )
    assert n_high == pytest.approx(3.0 * n_median, rel=0.05)


def test_edge_scale_below_median_reduces_size():
    """edge_vp < rolling_median_edge → edge_scale < 1 (floor at 0.25)."""
    flow = StraddleFlow(
        order_manager=None, live_drl_hedger=None,
        target_margin_utilization=0.05,
        im_fraction=0.10,
        max_trade_size_btc=100.0,
    )
    # Very small edge → edge_scale floored at 0.25
    n_low = flow.size_position(
        equity=1_000_000.0, signal_direction=1, spot=60_000.0,
        edge_vp=1.0, rolling_median_edge=100.0,
    )
    n_median = flow.size_position(
        equity=1_000_000.0, signal_direction=1, spot=60_000.0,
        edge_vp=100.0, rolling_median_edge=100.0,
    )
    assert n_low == pytest.approx(0.25 * n_median, rel=0.05)


def test_dd_scale_in_drawdown():
    """During drawdown, size scales down; at floor=0.25 with -37.5% DD.

    Compare two calls at the SAME equity to isolate dd_scale:
    - no drawdown (peak == equity) → dd_scale = 1.0
    - -37.5% drawdown (peak >> equity) → dd_scale = clip(1 + 2*(-0.375)) = 0.25
    """
    flow = StraddleFlow(
        order_manager=None, live_drl_hedger=None,
        target_margin_utilization=0.05,
        im_fraction=0.10,
        sizing_dd_floor=0.25,
        max_trade_size_btc=100.0,
    )
    equity = 625_000.0
    # No drawdown at this equity level
    n_no_dd = flow.size_position(
        equity=equity, signal_direction=1, spot=60_000.0,
        peak_equity=equity,  # equity == peak → dd_scale = 1.0
    )
    # -37.5% drawdown → dd = -0.375 → dd_scale = 0.25 (floor)
    n_dd = flow.size_position(
        equity=equity, signal_direction=1, spot=60_000.0,
        peak_equity=1_000_000.0,
    )
    assert n_dd == pytest.approx(0.25 * n_no_dd, rel=0.05)


def test_dd_scale_capped_at_one_above_peak():
    """equity >= peak_equity → dd_scale = 1.0 (no drawdown upscaling).

    Both equity=peak and equity>peak should yield the same position since
    the condition is (equity < peak_equity) — equity above peak is treated
    as 'no drawdown'.
    """
    flow = StraddleFlow(
        order_manager=None, live_drl_hedger=None,
        target_margin_utilization=0.05,
        im_fraction=0.10,
        max_trade_size_btc=100.0,
    )
    equity = 1_100_000.0
    # equity > peak → dd_scale = 1.0
    n_above = flow.size_position(
        equity=equity, signal_direction=1, spot=60_000.0,
        peak_equity=1_000_000.0,
    )
    # equity == peak → dd_scale = 1.0
    n_at_peak = flow.size_position(
        equity=equity, signal_direction=1, spot=60_000.0,
        peak_equity=equity,
    )
    assert n_above == pytest.approx(n_at_peak)


# ─── Execute flow — aborted entry paths ──────────────────────────────

@pytest.mark.asyncio
async def test_execute_rejects_non_actionable_signal():
    signal = _make_signal(direction=0)
    signal.straddle = None  # flat signal has no straddle

    flow = StraddleFlow(order_manager=None, live_drl_hedger=None)
    outcome = await flow.execute(signal=signal, equity=1_000_000.0)
    assert outcome.status == "aborted_entry"
    assert outcome.error == "signal not actionable"


@pytest.mark.asyncio
async def test_execute_rejects_zero_size():
    signal = _make_signal(direction=-1)
    flow = StraddleFlow(order_manager=None, live_drl_hedger=None)
    outcome = await flow.execute(signal=signal, equity=100.0)  # too small
    assert outcome.status == "aborted_entry"
    assert "minimum" in (outcome.error or "").lower()


@pytest.mark.asyncio
async def test_execute_aborts_when_entry_leg_rejected():
    signal = _make_signal(direction=-1)
    manager = AsyncMock()
    # Call leg filled; put leg rejected
    manager.place_ioc_taker = AsyncMock(side_effect=[
        _fill_result("BTC-12APR26-60000-C", "sell", 1.7),
        OrderResult(
            client_order_id="p_fail",
            order_id="",
            instrument_name="BTC-12APR26-60000-P",
            side="sell",
            amount_requested=1.7,
            filled_amount=0.0,
            average_price=0.0,
            state="rejected",
            is_dry_run=True,
            error="test rejection",
        ),
    ])
    # Unilateral close of the filled leg uses place_and_wait (market)
    manager.place_and_wait = AsyncMock(side_effect=[
        _fill_result("BTC-12APR26-60000-C", "buy", 1.7),
    ])

    bundle = _stub_bundle()
    drl = LiveDRLHedger(bundle=bundle)

    flow = StraddleFlow(
        order_manager=manager,
        live_drl_hedger=drl,
        holding_minutes=15,
        max_trade_size_btc=10.0,
    )
    outcome = await flow.execute(signal=signal, equity=1_000_000.0)
    assert outcome.status == "aborted_entry"
    assert "not filled" in (outcome.error or "").lower()
    # 2 maker calls (entry_C + entry_P) + 1 market close
    assert manager.place_ioc_taker.await_count == 2
    assert manager.place_and_wait.await_count == 1


# ─── Execute flow — full lifecycle (both legs fill, no hedges needed) ─

@pytest.mark.asyncio
async def test_execute_completes_full_lifecycle_when_hedger_skips_all_ticks():
    """End-to-end: entry → hedge loop (all dust) → exit → completed.

    Uses a stub DRL policy that outputs action=0 so every hedge tick
    is a no-op.  This exercises the full lifecycle without the DRL
    actually placing perp orders.  Holding period is set to ~100ms
    so the test runs in <1 sec.
    """
    signal = _make_signal(direction=-1)

    # Entry + exit use place_ioc_taker; abort close uses place_and_wait
    manager = AsyncMock()
    manager.place_ioc_taker = AsyncMock(side_effect=[
        _fill_result("BTC-12APR26-60000-C", "sell", 4.2),
        _fill_result("BTC-12APR26-60000-P", "sell", 4.2),
        _fill_result("BTC-12APR26-60000-C", "buy", 4.2),
        _fill_result("BTC-12APR26-60000-P", "buy", 4.2),
    ])

    # DRL stub that always outputs 0 (no hedge trade)
    bundle = _stub_bundle()
    drl = LiveDRLHedger(bundle=bundle, n_steps_per_episode=10)

    async def fake_spot():
        return 60_000.0

    flow = StraddleFlow(
        order_manager=manager,
        live_drl_hedger=drl,
        im_fraction=0.10,  # pin for test stability
        holding_minutes=15,
        max_trade_size_btc=10.0,
        n_hedge_ticks=2,  # only 2 ticks for the test
        spot_fetcher=fake_spot,
    )
    # Speed up: override the holding sleep via monkey-patch
    original_sleep = asyncio.sleep

    async def fast_sleep(seconds):
        await original_sleep(min(seconds, 0.01))

    import deribit_paper.execution.straddle_flow as sf_mod
    import deribit_paper.execution.perp_hedger as ph_mod
    sf_mod.asyncio.sleep = fast_sleep
    ph_mod_original = ph_mod.asyncio.sleep
    try:
        outcome = await flow.execute(signal=signal, equity=1_000_000.0)
    finally:
        sf_mod.asyncio.sleep = original_sleep
        ph_mod.asyncio.sleep = ph_mod_original

    assert outcome.status == "completed"
    assert outcome.entry_call.is_filled
    assert outcome.entry_put.is_filled
    assert outcome.exit_call.is_filled
    assert outcome.exit_put.is_filled
    assert outcome.n_contracts_btc == pytest.approx(4.2)
    assert outcome.trade_id != ""
    # Hedge summary presence verified; actual tick count depends on the
    # race between the monkey-patched holding sleep and the hedger task
    # start-up, so we only assert the object is present (the hedger's
    # own unit tests cover tick execution in detail).
    assert outcome.hedge_summary is not None
    assert outcome.hedge_summary.trade_id == outcome.trade_id
