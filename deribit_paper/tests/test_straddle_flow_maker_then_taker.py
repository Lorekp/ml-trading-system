"""E-5 + E-8 (parity-r3 cycle 2 — 2026-05-08) regression tests.

Live entry path now uses ``place_maker_then_taker`` (mirroring the V2
backtester's SOR which simulates maker fills on ~30-60% of bars). The
exit path stays on ``place_ioc_taker`` for fast bar-aligned timing.

Recovery paths (``_close_if_filled`` and the holding-period CancelledError
cancel-exit) use ``place_ioc_taker(tiered=True)`` instead of naked
market orders — respecting the engine's "never naked market" contract
from E.1.7 even during shutdown.
"""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock

import numpy as np
import pandas as pd
import pytest

from deribit_paper.execution.order_manager import OrderResult
from deribit_paper.execution.straddle_flow import StraddleFlow
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


def _filled(instrument: str, side: str, amount: float = 1.0, px: float = 0.05) -> OrderResult:
    return OrderResult(
        client_order_id=f"stub_{instrument}_{side}",
        order_id=f"oid_{instrument}",
        instrument_name=instrument,
        side=side,
        amount_requested=amount,
        filled_amount=amount,
        average_price=px,
        state="filled",
        is_dry_run=True,
    )


def _make_flow(order_manager) -> StraddleFlow:
    """Construct a StraddleFlow plumbed for execute() flow tests.

    Holding period is set to 0 so we don't actually sleep; spot fetcher
    and DRL hedger are stubs.
    """
    bundle = _stub_bundle()
    drl = LiveDRLHedger(bundle=bundle, n_steps_per_episode=10, bar_minutes=15)
    return StraddleFlow(
        order_manager=order_manager,
        live_drl_hedger=drl,
        holding_minutes=0,
        target_margin_utilization=0.05,
        im_fraction=0.20,
        max_trade_size_btc=10.0,
        n_hedge_ticks=1,
        spot_fetcher=AsyncMock(return_value=60_000.0),
        maker_wait_seconds=10.0,
        cancel_exit_max_slippage_pct=1.0,
        use_tiered_taker=False,
        min_trade_floor=True,
    )


# ── E-5: entry path uses maker-then-taker ────────────────────────────


@pytest.mark.asyncio
async def test_execute_entry_calls_maker_then_taker():
    om = MagicMock()
    om._rest = MagicMock()
    om._rest.get_ticker = AsyncMock(return_value={
        "best_ask_price": 0.06, "best_bid_price": 0.058, "mark_price": 0.059,
    })
    om.place_maker_then_taker = AsyncMock(side_effect=lambda **kwargs: _filled(
        kwargs["instrument_name"], kwargs["side"],
        amount=kwargs["amount"], px=0.06,
    ))
    om.place_ioc_taker = AsyncMock(side_effect=lambda **kwargs: _filled(
        kwargs["instrument_name"], kwargs["side"],
        amount=kwargs["amount"], px=0.06,
    ))

    flow = _make_flow(om)
    outcome = await flow.execute(
        signal=_make_signal(direction=-1),
        equity=1_000_000.0,
        trailing_1h_volume_btc=20.0,
    )

    # Entry path used maker-then-taker
    assert om.place_maker_then_taker.call_count == 2  # call + put
    # Exit path used ioc_taker (fast fill at bar boundary)
    assert om.place_ioc_taker.call_count == 2  # call + put
    assert outcome.status == "completed"


@pytest.mark.asyncio
async def test_execute_passes_taker_slippage_pct_through_to_maker_then_taker():
    """The taker fallback slippage cap flows from execute() into
    place_maker_then_taker so the cancel-and-fall-back path enforces
    the same cap as the steady-state taker."""
    om = MagicMock()
    om._rest = MagicMock()
    om._rest.get_ticker = AsyncMock(return_value={
        "best_ask_price": 0.06, "best_bid_price": 0.058, "mark_price": 0.059,
    })
    om.place_maker_then_taker = AsyncMock(side_effect=lambda **kwargs: _filled(
        kwargs["instrument_name"], kwargs["side"], amount=kwargs["amount"],
    ))
    om.place_ioc_taker = AsyncMock(side_effect=lambda **kwargs: _filled(
        kwargs["instrument_name"], kwargs["side"], amount=kwargs["amount"],
    ))

    flow = _make_flow(om)
    await flow.execute(
        signal=_make_signal(direction=-1),
        equity=1_000_000.0,
        trailing_1h_volume_btc=20.0,
        taker_slippage_pct=0.7,
    )

    for call in om.place_maker_then_taker.call_args_list:
        assert call.kwargs["taker_max_slippage_pct"] == pytest.approx(0.7)
        # tiered defaults to use_tiered_taker (False here)
        assert call.kwargs["taker_tiered"] is False


# ── E-8: cancel-exit uses tiered IOC, not naked market ───────────────


@pytest.mark.asyncio
async def test_cancel_exit_uses_tiered_ioc_not_market():
    """When the holding-period sleep is cancelled (asyncio.CancelledError
    from kill switch / shutdown), the cancel-exit places via
    place_ioc_taker(tiered=True), NOT place_and_wait(market)."""
    om = MagicMock()
    om._rest = MagicMock()
    om._rest.get_ticker = AsyncMock(return_value={
        "best_ask_price": 0.06, "best_bid_price": 0.058, "mark_price": 0.059,
    })
    om.place_maker_then_taker = AsyncMock(side_effect=lambda **kwargs: _filled(
        kwargs["instrument_name"], kwargs["side"], amount=kwargs["amount"],
    ))
    om.place_ioc_taker = AsyncMock(side_effect=lambda **kwargs: _filled(
        kwargs["instrument_name"], kwargs["side"], amount=kwargs["amount"],
    ))
    om.place_and_wait = AsyncMock(return_value=_filled("BTC", "sell"))

    # Holding period > 0 so the sleep is cancellable
    bundle = _stub_bundle()
    drl = LiveDRLHedger(bundle=bundle, n_steps_per_episode=10, bar_minutes=15)
    flow = StraddleFlow(
        order_manager=om,
        live_drl_hedger=drl,
        holding_minutes=15,
        target_margin_utilization=0.05,
        im_fraction=0.20,
        max_trade_size_btc=10.0,
        n_hedge_ticks=1,
        spot_fetcher=AsyncMock(return_value=60_000.0),
        maker_wait_seconds=10.0,
        cancel_exit_max_slippage_pct=1.5,
        use_tiered_taker=False,
        min_trade_floor=True,
    )

    task = asyncio.create_task(flow.execute(
        signal=_make_signal(direction=-1),
        equity=1_000_000.0,
        trailing_1h_volume_btc=20.0,
    ))
    # Give the task time to enter the holding-period sleep
    await asyncio.sleep(0.1)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    # Cancel-exit path: 2 IOC tiered calls (call + put), 0 naked market
    cancel_exit_calls = [
        c for c in om.place_ioc_taker.call_args_list
        if c.kwargs["label"].startswith("cancel_exit_")
    ]
    assert len(cancel_exit_calls) == 2
    for call in cancel_exit_calls:
        assert call.kwargs["tiered"] is True
        assert call.kwargs["max_slippage_pct"] == pytest.approx(1.5)
        assert call.kwargs["reduce_only"] is True

    # No naked market in the cancel path
    cancel_market_calls = [
        c for c in om.place_and_wait.call_args_list
        if c.kwargs.get("order_type") == "market"
        and c.kwargs.get("label", "").startswith("cancel_exit_")
    ]
    assert cancel_market_calls == []


# ── E-8: _close_if_filled uses tiered IOC ────────────────────────────


@pytest.mark.asyncio
async def test_close_if_filled_uses_tiered_ioc():
    """Recovery path: if leg-1 fills but leg-2 fails, ``_close_if_filled``
    closes leg-1 via tiered IOC, not naked market."""
    om = MagicMock()
    om.place_ioc_taker = AsyncMock(return_value=_filled("BTC-30APR26-60000-C", "buy"))
    om.place_and_wait = AsyncMock()

    bundle = _stub_bundle()
    drl = LiveDRLHedger(bundle=bundle, n_steps_per_episode=10, bar_minutes=15)
    flow = StraddleFlow(
        order_manager=om,
        live_drl_hedger=drl,
        cancel_exit_max_slippage_pct=1.2,
    )

    # Simulate a filled leg
    filled_leg = _filled("BTC-30APR26-60000-C", "sell", amount=1.0)
    await flow._close_if_filled(
        order=filled_leg,
        instrument="BTC-30APR26-60000-C",
        entry_side="sell",
        n_contracts=1.0,
    )

    # Tiered IOC close happened, not place_and_wait(market)
    assert om.place_ioc_taker.call_count == 1
    call = om.place_ioc_taker.call_args
    assert call.kwargs["tiered"] is True
    assert call.kwargs["max_slippage_pct"] == pytest.approx(1.2)
    assert call.kwargs["reduce_only"] is True
    # No naked market
    market_calls = [
        c for c in om.place_and_wait.call_args_list
        if c.kwargs.get("order_type") == "market"
    ]
    assert market_calls == []


@pytest.mark.asyncio
async def test_close_if_filled_no_op_on_unfilled_order():
    """If the entry leg never filled, no close attempt fires."""
    om = MagicMock()
    om.place_ioc_taker = AsyncMock()

    bundle = _stub_bundle()
    drl = LiveDRLHedger(bundle=bundle, n_steps_per_episode=10, bar_minutes=15)
    flow = StraddleFlow(order_manager=om, live_drl_hedger=drl)

    # Order is rejected — filled_amount == 0
    rejected = OrderResult(
        client_order_id="x", order_id="x",
        instrument_name="BTC", side="sell",
        amount_requested=1.0, filled_amount=0.0,
        average_price=0.0, state="rejected", is_dry_run=True,
    )
    await flow._close_if_filled(
        order=rejected,
        instrument="BTC",
        entry_side="sell",
        n_contracts=1.0,
    )
    assert om.place_ioc_taker.call_count == 0
