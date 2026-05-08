"""All 4 production IOC sites must pass ``tiered=True`` under V2 preset.

E.1.6 in plan ``C:\\Users\\laure\\.claude\\plans\\starry-rolling-rabin.md``.

Pre-E.1.6 the 3-tier IOC ladder was implemented but unreachable from
production because :meth:`StraddleFlow.execute` always used the
``tiered=False`` default. This test injects a recording mock
order_manager and verifies all 4 call sites (entry call, entry put,
exit call, exit put) pass the kwarg through under V2 preset and not
under default.
"""
from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pandas as pd
import pytest

from deribit_paper.config import PaperConfig
from deribit_paper.execution.order_manager import OrderResult
from deribit_paper.execution.straddle_flow import StraddleFlow
from deribit_paper.signal.predictor import Signal


def _make_filled_result() -> OrderResult:
    return OrderResult(
        client_order_id="cli",
        order_id="o",
        instrument_name="BTC-X",
        side="buy",
        amount_requested=0.3,
        filled_amount=0.3,
        average_price=0.05,
        state="filled",
        is_dry_run=True,
    )


def _make_flow(*, use_tiered: bool) -> tuple[StraddleFlow, list[dict]]:
    """Return a flow instance plus a list of recorded entry+exit calls.

    E-5 (parity-r3 cycle 2 — 2026-05-08): entry path uses
    ``place_maker_then_taker``; exit path uses ``place_ioc_taker``.
    Both record into the same ``calls`` list with a ``_method`` key
    so assertions can distinguish entry-vs-exit.
    """
    calls: list[dict] = []

    async def _record_maker_then_taker(**kwargs) -> OrderResult:
        calls.append({**kwargs, "_method": "maker_then_taker"})
        return _make_filled_result()

    async def _record_ioc_taker(**kwargs) -> OrderResult:
        calls.append({**kwargs, "_method": "ioc_taker"})
        return _make_filled_result()

    om = MagicMock()
    om.place_maker_then_taker = _record_maker_then_taker
    om.place_ioc_taker = _record_ioc_taker
    om._rest = None             # disables the spread-fetch branch in execute
    om.place_and_wait = AsyncMock(return_value=_make_filled_result())

    drl = MagicMock()
    drl.begin_trade = MagicMock(return_value="t-001")
    drl.close_trade = MagicMock()
    drl.predict = MagicMock(return_value=(None, None))

    async def _spot() -> float:
        return 60_000.0

    flow = StraddleFlow(
        order_manager=om,
        live_drl_hedger=drl,
        target_margin_utilization=0.15,
        im_fraction=0.20,
        max_trade_size_btc=10.0,
        min_trade_size_btc=0.1,
        holding_minutes=0,                  # bypass the asyncio.sleep
        baseline_target_dvol=50.0,
        sizing_edge_cap=3.0,
        sizing_dd_floor=0.25,
        use_tiered_taker=use_tiered,
        spot_fetcher=_spot,
    )
    return flow, calls


def _make_actionable_signal() -> Signal:
    started = pd.Timestamp("2026-04-15 12:00:00", tz="UTC")
    straddle = SimpleNamespace(
        call_instrument="BTC-CALL",
        put_instrument="BTC-PUT",
        strike=60_000.0,
        expiry_dt=pd.Timestamp("2026-04-22 08:00:00", tz="UTC"),
        index_price=60_000.0,
    )
    bar_tsrv = SimpleNamespace(
        vol_ann_continuous=0.6,
        jump_fraction=0.0,
    )
    return Signal(
        timestamp=started,
        direction=1,
        edge_vol_pts=15.0,
        rv_pred_pct=72.0,
        iv_used_pct=58.0,
        bar_tsrv=bar_tsrv,
        straddle=straddle,
        reason="test",
    )


@pytest.mark.parametrize("use_tiered,expected_tiered", [
    (True, True),
    (False, False),
])
def test_all_four_ioc_sites_propagate_tiered_flag(
    use_tiered: bool, expected_tiered: bool,
):
    flow, calls = _make_flow(use_tiered=use_tiered)
    signal = _make_actionable_signal()

    asyncio.run(flow.execute(
        signal=signal,
        equity=100_000.0,
        trailing_1h_volume_btc=None,
        dvol_pct=50.0,
        edge_vp=15.0,
        rolling_median_edge=10.0,
        peak_equity=100_000.0,
        kelly_mult=1.0,
    ))

    # Expect 4 calls total: 2 entry maker_then_taker + 2 exit ioc_taker.
    # E-5 (parity-r3 cycle 2 — 2026-05-08): the tiered flag arrives at
    # entry as ``taker_tiered`` (controlling the maker→taker fallback),
    # at exit as ``tiered`` (controlling the steady-state IOC ladder).
    assert len(calls) == 4, calls
    entry_calls = [c for c in calls if c["_method"] == "maker_then_taker"]
    exit_calls = [c for c in calls if c["_method"] == "ioc_taker"]
    assert len(entry_calls) == 2
    assert len(exit_calls) == 2
    for c in entry_calls:
        assert c.get("taker_tiered") == expected_tiered, c
    for c in exit_calls:
        assert c.get("tiered") == expected_tiered, c


def test_v2_preset_flips_use_tiered_taker():
    """Sanity: ``with_v2_preset`` activates the ladder; ``default`` does not."""
    default_cfg = PaperConfig()
    v2_cfg = PaperConfig.with_v2_preset()
    assert default_cfg.use_tiered_taker is False
    assert v2_cfg.use_tiered_taker is True


# ── E.1.9: live spread plumbed to slippage_calc override ─────────────────


def test_order_labels_unique_across_concurrent_trades():
    """E.1.13: two trades fired at the same wall-clock timestamp produce
    distinct order labels at every IOC site. Pre-E.1.13 the timestamp
    alone was the entire label, so simultaneous trades collided.
    """
    seen_labels: list[str] = []

    async def _record(**kwargs) -> OrderResult:
        seen_labels.append(kwargs["label"])
        return _make_filled_result()

    async def _spot() -> float:
        return 60_000.0

    def _build_flow():
        om = MagicMock()
        # E-5 (parity-r3 cycle 2 — 2026-05-08): entry uses
        # place_maker_then_taker, exit uses place_ioc_taker.
        om.place_maker_then_taker = _record
        om.place_ioc_taker = _record
        om._rest = None
        om.place_and_wait = AsyncMock(return_value=_make_filled_result())
        drl = MagicMock()
        drl.begin_trade = MagicMock(return_value="t-001")
        drl.close_trade = MagicMock()
        drl.predict = MagicMock(return_value=(None, None))
        return StraddleFlow(
            order_manager=om,
            live_drl_hedger=drl,
            target_margin_utilization=0.15,
            im_fraction=0.20,
            max_trade_size_btc=10.0,
            min_trade_size_btc=0.1,
            holding_minutes=0,
            baseline_target_dvol=50.0,
            sizing_edge_cap=3.0,
            sizing_dd_floor=0.25,
            spot_fetcher=_spot,
        )

    async def _go() -> None:
        f1 = _build_flow()
        f2 = _build_flow()
        s = _make_actionable_signal()
        await asyncio.gather(
            f1.execute(
                signal=s, equity=100_000.0,
                trailing_1h_volume_btc=None, dvol_pct=50.0,
                edge_vp=15.0, rolling_median_edge=10.0,
                peak_equity=100_000.0, kelly_mult=1.0,
            ),
            f2.execute(
                signal=s, equity=100_000.0,
                trailing_1h_volume_btc=None, dvol_pct=50.0,
                edge_vp=15.0, rolling_median_edge=10.0,
                peak_equity=100_000.0, kelly_mult=1.0,
            ),
        )

    asyncio.run(_go())
    # 2 trades × 4 sites = 8 IOC labels. All must be unique.
    assert len(seen_labels) == 8
    assert len(set(seen_labels)) == 8


def test_observed_spread_widens_taker_slippage_pct_in_execute():
    """Wide observed spread on the ticker triggers the override in
    ``live_slippage_pct``, raising ``taker_slippage_pct`` for the IOC
    call sites. Caller passed ``taker_slippage_pct=0.05`` (sidecar-tight),
    but a $300 observed spread at $60K spot = 0.5% > 1.5×0.05% = 0.075%
    → override fires → ~0.5% used at the IOC.

    E-5 (parity-r3 cycle 2 — 2026-05-08): entry path uses
    ``place_maker_then_taker``; the override flows to entry via
    ``taker_max_slippage_pct`` and to exit via ``max_slippage_pct``.
    """
    calls: list[dict] = []

    async def _record_maker_then_taker(**kwargs) -> OrderResult:
        calls.append({**kwargs, "_method": "maker_then_taker"})
        return _make_filled_result()

    async def _record_ioc_taker(**kwargs) -> OrderResult:
        calls.append({**kwargs, "_method": "ioc_taker"})
        return _make_filled_result()

    rest = MagicMock()
    rest.get_ticker = AsyncMock(return_value={
        "best_ask_price": 0.055,         # 0.005 wide => 0.005 × 60K = $300
        "best_bid_price": 0.050,
    })

    om = MagicMock()
    om.place_maker_then_taker = _record_maker_then_taker
    om.place_ioc_taker = _record_ioc_taker
    om._rest = rest
    om.place_and_wait = AsyncMock(return_value=_make_filled_result())

    drl = MagicMock()
    drl.begin_trade = MagicMock(return_value="t-001")
    drl.close_trade = MagicMock()
    drl.predict = MagicMock(return_value=(None, None))

    async def _spot() -> float:
        return 60_000.0

    flow = StraddleFlow(
        order_manager=om,
        live_drl_hedger=drl,
        target_margin_utilization=0.15,
        im_fraction=0.20,
        max_trade_size_btc=10.0,
        min_trade_size_btc=0.1,
        holding_minutes=0,
        baseline_target_dvol=50.0,
        sizing_edge_cap=3.0,
        sizing_dd_floor=0.25,
        spot_fetcher=_spot,
    )

    signal = _make_actionable_signal()
    asyncio.run(flow.execute(
        signal=signal,
        equity=100_000.0,
        trailing_1h_volume_btc=None,
        dvol_pct=50.0,
        edge_vp=15.0,
        rolling_median_edge=10.0,
        peak_equity=100_000.0,
        kelly_mult=1.0,
        taker_slippage_pct=0.05,            # tight sidecar value
    ))

    # All 4 sites got the OVERRIDE pct, not the original 0.05.
    # Override pct ≈ 0.5% (300/60000 × 100).
    assert len(calls) == 4
    for c in calls:
        if c["_method"] == "maker_then_taker":
            # Entry: pct flows through ``taker_max_slippage_pct``
            assert c["taker_max_slippage_pct"] > 0.4
            assert c["taker_max_slippage_pct"] < 1.0
        else:
            # Exit: pct flows through ``max_slippage_pct``
            assert c["max_slippage_pct"] > 0.4
            assert c["max_slippage_pct"] < 1.0
