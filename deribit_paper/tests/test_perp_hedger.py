"""Tests for PerpHedger — tick execution, dust filtering, cancellation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from deribit_paper.execution.order_manager import OrderResult
from deribit_paper.execution.perp_hedger import HedgeRunSummary, PerpHedger
from deribit_paper.signal.drl_hedger_live import LiveDRLHedger
from deribit_paper.signal.model_loader import ModelBundle


def _stub_bundle_with_action(action: float) -> ModelBundle:
    drl = MagicMock()
    drl.predict.return_value = (np.array([[action]], dtype=np.float32), None)
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


def _filled_perp_order(amount_usd: float, side: str = "buy", fill_price: float = 60_000.0) -> OrderResult:
    return OrderResult(
        client_order_id="stub",
        order_id="oid_perp",
        instrument_name="BTC-PERPETUAL",
        side=side,
        amount_requested=amount_usd,
        filled_amount=amount_usd,
        average_price=fill_price,
        state="filled",
        is_dry_run=True,
    )


async def _instant_sleep(seconds):
    """Inject as ``clock_sleep_fn`` so tests don't wait real time."""
    await asyncio.sleep(0)


async def _make_hedger_trade(bundle, signal_direction: int = -1) -> tuple[LiveDRLHedger, str]:
    """Build a LiveDRLHedger + start one trade for the test to hedge."""
    drl = LiveDRLHedger(bundle=bundle, n_steps_per_episode=3)
    tid = drl.begin_trade(
        signal=signal_direction,
        S_entry=60_000.0,
        K=60_000.0,
        vol_ann_entry=0.8,
        iv_entry=0.75,
        tau_entry=0.04,
        V_bg=100.0,
        jump_fraction=0.0,
        vpin=0.5,
    )
    return drl, tid


# ─── Dust filtering ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hedger_skips_dust_when_action_is_zero():
    """With action=0, the DRL never moves inventory → every tick is dust."""
    bundle = _stub_bundle_with_action(0.0)
    drl, tid = await _make_hedger_trade(bundle)

    order_manager = AsyncMock()
    order_manager.place_and_wait = AsyncMock()

    async def spot(): return 60_000.0

    hedger = PerpHedger(
        order_manager=order_manager,
        live_drl_hedger=drl,
        bar_minutes=15,
        n_ticks_per_bar=3,
        spot_fetcher=spot,
        clock_sleep_fn=_instant_sleep,
    )
    summary = await hedger.run(tid, entry_time=pd.Timestamp.now(tz="UTC"))

    assert summary.n_ticks_executed == 0
    assert summary.n_ticks_skipped_dust == 3
    assert summary.total_usd_traded == 0.0
    order_manager.place_and_wait.assert_not_called()


# ─── Active hedging ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hedger_submits_orders_when_delta_exceeds_dust():
    """Action=+1 with a significant delta → perp order placed every tick."""
    bundle = _stub_bundle_with_action(1.0)
    drl, tid = await _make_hedger_trade(bundle, signal_direction=-1)

    order_manager = AsyncMock()
    order_manager.place_and_wait = AsyncMock(side_effect=[
        _filled_perp_order(amount_usd=5_000.0, side="buy"),
        _filled_perp_order(amount_usd=1_000.0, side="buy"),
        _filled_perp_order(amount_usd=500.0, side="buy"),
    ])

    spot_values = [60_500.0, 60_800.0, 61_000.0]
    idx = [0]
    async def spot():
        v = spot_values[idx[0]]
        idx[0] += 1
        return v

    hedger = PerpHedger(
        order_manager=order_manager,
        live_drl_hedger=drl,
        bar_minutes=15,
        n_ticks_per_bar=3,
        dust_threshold_btc=0.001,  # very low threshold to force orders
        spot_fetcher=spot,
        clock_sleep_fn=_instant_sleep,
    )
    summary = await hedger.run(tid, entry_time=pd.Timestamp.now(tz="UTC"))

    # All 3 ticks should have triggered an order
    assert summary.n_ticks_executed == 3
    assert summary.total_usd_traded == pytest.approx(6_500.0)
    assert order_manager.place_and_wait.await_count == 3


# ─── Cancellation ───────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_hedger_exits_on_cancel():
    bundle = _stub_bundle_with_action(1.0)
    drl, tid = await _make_hedger_trade(bundle)

    order_manager = AsyncMock()
    order_manager.place_and_wait = AsyncMock(
        return_value=_filled_perp_order(amount_usd=1_000.0),
    )

    async def spot(): return 60_000.0

    hedger = PerpHedger(
        order_manager=order_manager,
        live_drl_hedger=drl,
        bar_minutes=15,
        n_ticks_per_bar=10,
        dust_threshold_btc=0.001,
        spot_fetcher=spot,
        clock_sleep_fn=_instant_sleep,
    )
    # Cancel before running
    hedger.cancel()
    summary = await hedger.run(tid, entry_time=pd.Timestamp.now(tz="UTC"))
    # The cancel check is AFTER the first sleep, so the first iteration's
    # sleep runs, then the cancel check fires before any work.  Expect 0
    # ticks executed.
    assert summary.n_ticks_executed == 0


# ─── Spot fetcher failures ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_hedger_survives_spot_fetch_exception():
    """If the spot fetcher raises, the hedger should log + skip the tick."""
    bundle = _stub_bundle_with_action(0.5)
    drl, tid = await _make_hedger_trade(bundle)

    order_manager = AsyncMock()
    order_manager.place_and_wait = AsyncMock(
        return_value=_filled_perp_order(amount_usd=1_000.0),
    )

    call_count = [0]
    async def spot():
        call_count[0] += 1
        if call_count[0] == 1:
            raise RuntimeError("network flake")
        return 60_000.0

    hedger = PerpHedger(
        order_manager=order_manager,
        live_drl_hedger=drl,
        bar_minutes=15,
        n_ticks_per_bar=2,
        dust_threshold_btc=0.001,  # low so second tick does hedge
        spot_fetcher=spot,
        clock_sleep_fn=_instant_sleep,
    )
    summary = await hedger.run(tid, entry_time=pd.Timestamp.now(tz="UTC"))
    # First tick raised; second tick ran (fetched spot + called DRL + placed order)
    assert call_count[0] == 2
    assert summary.n_ticks_executed == 1


@pytest.mark.asyncio
async def test_hedger_requires_spot_fetcher():
    bundle = _stub_bundle_with_action(0.0)
    drl, tid = await _make_hedger_trade(bundle)

    hedger = PerpHedger(
        order_manager=AsyncMock(),
        live_drl_hedger=drl,
        spot_fetcher=None,
        clock_sleep_fn=_instant_sleep,
    )
    with pytest.raises(RuntimeError, match="spot_fetcher"):
        await hedger.run(tid, entry_time=pd.Timestamp.now(tz="UTC"))
