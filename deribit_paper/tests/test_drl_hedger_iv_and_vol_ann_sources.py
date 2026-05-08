"""D-8 + D-9 (parity-r3 cycle 2 — 2026-05-08) regression tests.

D-8: live ``vol_ann_entry`` was the CURRENT bar's ``vol_ann_continuous``
(jump-corrected); the training env (``vol_arb/drl_hedging/env.py:172``)
and offline backtester (``DRLHedger.simulate``) consume the PREVIOUS
bar's RAW ``vol_ann`` (``df["vol_ann"].shift(1)``). Two-fold mismatch:
wrong lag, wrong column.

D-9: live ``iv_entry`` was ``signal.iv_used_pct/100`` — usually agrees
with ``slice.iv(K)`` but could fall through to a DVOL backstop the env
was never trained on. The env's reset semantics evaluate the slice at
K explicitly (``env.py:412-415``).

Both are P1 alignment fixes — small magnitude in the median bar, but
introduce systematic bias on the tail (high-vol bars + uncalibrated SVI
respectively).
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from deribit_paper.signal.drl_hedger_live import LiveDRLHedger
from deribit_paper.signal.model_loader import ModelBundle


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


# ── D-9: iv_entry sourced from SVI slice when available ──────────────


def test_d9_iv_entry_uses_slice_when_provided():
    """When ``svi_slice_params`` is provided, ``begin_trade`` evaluates
    the slice at K and uses that for ``iv_entry`` — overriding the
    fallback ``iv_entry`` kwarg the caller passed."""
    hedger = LiveDRLHedger(bundle=_stub_bundle(), bar_minutes=15)

    # SVI slice with a, b, rho, m, sigma, T, F such that slice.iv(60_000) ≈ known value.
    # Use a moderate slice: a=0.04, b=0.10, rho=-0.3, m=-0.05, sigma=0.20, T=0.04, F=60000.
    # k = log(60000/60000) = 0; km = -0.05; w = 0.04 + 0.10*(-0.3*-0.05 + sqrt(0.0025 + 0.04))
    #   = 0.04 + 0.10*(0.015 + 0.21213) = 0.04 + 0.0227 = 0.0627
    # iv = sqrt(0.0627 / 0.04) = sqrt(1.568) = 1.252 → out of [0,5] band? No, valid.
    # Let me adjust to get a more realistic IV. Use a=0.005, b=0.05.
    # w = 0.005 + 0.05*(-0.3*-0.05 + sqrt(0.0025 + 0.04)) = 0.005 + 0.05 * 0.227 = 0.0163
    # iv = sqrt(0.0163 / 0.04) = sqrt(0.408) = 0.639 → 64% IV, realistic.
    svi_params = {
        "a": 0.005, "b": 0.05, "rho": -0.3, "m": -0.05, "sigma": 0.20,
        "T": 0.04, "F": 60_000.0,
    }
    trade_id = hedger.begin_trade(
        signal=-1, S_entry=60_000.0, K=60_000.0,
        vol_ann_entry=0.80,
        iv_entry=0.99,  # FALLBACK value — should be overridden by slice.iv(60000)
        tau_entry=0.04, V_bg=120.0,
        jump_fraction=0.05, vpin=0.30,
        svi_slice_params=svi_params,
    )

    state = hedger._trades[trade_id]
    # The actual slice.iv(60000) should land in the realistic 0.5-0.8 range,
    # NOT the fallback 0.99. Verify it's NOT the fallback.
    assert state.iv_entry != pytest.approx(0.99)
    assert 0.4 < state.iv_entry < 0.9


def test_d9_iv_entry_keeps_fallback_when_no_slice():
    """No ``svi_slice_params`` → uses the fallback ``iv_entry`` arg."""
    hedger = LiveDRLHedger(bundle=_stub_bundle(), bar_minutes=15)
    trade_id = hedger.begin_trade(
        signal=-1, S_entry=60_000.0, K=60_000.0,
        vol_ann_entry=0.80,
        iv_entry=0.75,
        tau_entry=0.04, V_bg=120.0,
        jump_fraction=0.05, vpin=0.30,
        svi_slice_params=None,
    )
    state = hedger._trades[trade_id]
    assert state.iv_entry == pytest.approx(0.75)


def test_d9_iv_entry_skips_slice_when_slice_returns_invalid():
    """Slice with degenerate params (T=0) returns slice.iv(K)=0.0 → must
    fall back to the passed ``iv_entry``, not the zero."""
    hedger = LiveDRLHedger(bundle=_stub_bundle(), bar_minutes=15)
    bad_slice = {
        "a": 0.005, "b": 0.05, "rho": -0.3, "m": -0.05, "sigma": 0.20,
        "T": 0.0,  # invalid → slice.iv returns 0.0
        "F": 60_000.0,
    }
    trade_id = hedger.begin_trade(
        signal=-1, S_entry=60_000.0, K=60_000.0,
        vol_ann_entry=0.80,
        iv_entry=0.75,
        tau_entry=0.04, V_bg=120.0,
        jump_fraction=0.05, vpin=0.30,
        svi_slice_params=bad_slice,
    )
    state = hedger._trades[trade_id]
    # slice returned 0.0 → fall back to passed iv_entry
    assert state.iv_entry == pytest.approx(0.75)


def test_d9_delta_entry_recomputed_with_slice_iv():
    """``delta_entry`` is computed AFTER iv_entry override, so it
    reflects the slice-derived IV — not the fallback."""
    hedger = LiveDRLHedger(bundle=_stub_bundle(), bar_minutes=15)
    svi_params = {
        "a": 0.005, "b": 0.05, "rho": -0.3, "m": -0.05, "sigma": 0.20,
        "T": 0.04, "F": 60_000.0,
    }
    tid_with_slice = hedger.begin_trade(
        signal=-1, S_entry=60_000.0, K=60_000.0,
        vol_ann_entry=0.80,
        iv_entry=0.99, tau_entry=0.04, V_bg=120.0,
        jump_fraction=0.05, vpin=0.30,
        svi_slice_params=svi_params,
    )
    delta_with_slice = hedger._trades[tid_with_slice].delta_entry

    # Now begin_trade WITHOUT a slice — same fallback iv_entry value.
    hedger2 = LiveDRLHedger(bundle=_stub_bundle(), bar_minutes=15)
    tid_no_slice = hedger2.begin_trade(
        signal=-1, S_entry=60_000.0, K=60_000.0,
        vol_ann_entry=0.80,
        iv_entry=0.99, tau_entry=0.04, V_bg=120.0,
        jump_fraction=0.05, vpin=0.30,
        svi_slice_params=None,
    )
    delta_no_slice = hedger2._trades[tid_no_slice].delta_entry

    # Different IVs → different deltas.
    assert delta_with_slice != pytest.approx(delta_no_slice, rel=1e-6)


# ── D-8: vol_ann_entry sourced from lagged buffer ────────────────────


def test_d8_straddle_flow_uses_lagged_vol_ann_when_provided():
    """End-to-end: when the engine passes ``vol_ann_entry_lagged``,
    the straddle_flow plumbs it into begin_trade as ``vol_ann_entry``,
    overriding the current-bar ``signal.bar_tsrv.vol_ann_continuous``.
    """
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import pandas as pd

    from deribit_paper.execution.order_manager import OrderResult
    from deribit_paper.execution.straddle_flow import StraddleFlow
    from deribit_paper.signal.predictor import Signal

    om = MagicMock()
    om._rest = None
    om.place_maker_then_taker = AsyncMock(return_value=OrderResult(
        client_order_id="x", order_id="y", instrument_name="BTC",
        side="sell", amount_requested=1.0, filled_amount=1.0,
        average_price=0.05, state="filled", is_dry_run=True,
    ))
    om.place_ioc_taker = AsyncMock(return_value=OrderResult(
        client_order_id="x", order_id="y", instrument_name="BTC",
        side="buy", amount_requested=1.0, filled_amount=1.0,
        average_price=0.05, state="filled", is_dry_run=True,
    ))

    captured: dict = {}

    drl = MagicMock()

    def _begin_trade(**kwargs):
        captured.update(kwargs)
        return "t-001"

    drl.begin_trade = _begin_trade
    drl.close_trade = MagicMock()

    flow = StraddleFlow(
        order_manager=om,
        live_drl_hedger=drl,
        target_margin_utilization=0.05,
        im_fraction=0.20,
        max_trade_size_btc=10.0,
        holding_minutes=0,
        spot_fetcher=AsyncMock(return_value=60_000.0),
        min_trade_floor=True,
    )

    started_at = pd.Timestamp("2026-04-15 12:00:00", tz="UTC")
    straddle = SimpleNamespace(
        call_instrument="BTC-CALL",
        put_instrument="BTC-PUT",
        strike=60_000.0,
        expiry_dt=pd.Timestamp("2026-04-22 08:00:00", tz="UTC"),
        index_price=60_000.0,
    )
    bar_tsrv = SimpleNamespace(
        vol_ann_continuous=0.80,  # CURRENT bar's vol_ann_continuous
        jump_fraction=0.05,
    )
    sig = Signal(
        timestamp=started_at, direction=-1,
        edge_vol_pts=-2.0, rv_pred_pct=72.0, iv_used_pct=58.0,
        bar_tsrv=bar_tsrv, straddle=straddle, reason="test",
    )

    asyncio.run(flow.execute(
        signal=sig, equity=1_000_000.0,
        trailing_1h_volume_btc=20.0,
        vol_ann_entry_lagged=0.65,  # PRIOR bar's raw vol_ann
    ))

    # vol_ann_entry should be the lagged value (0.65), NOT the current
    # bar's vol_ann_continuous (0.80).
    assert captured["vol_ann_entry"] == pytest.approx(0.65)


def test_d8_straddle_flow_falls_back_to_current_bar_when_no_lagged():
    """No lagged value (cold start, buffer empty) → fall back to
    ``signal.bar_tsrv.vol_ann_continuous`` (pre-D-8 behaviour). Trades
    can still fire at cold start with this fallback."""
    import asyncio
    from types import SimpleNamespace
    from unittest.mock import AsyncMock

    import pandas as pd

    from deribit_paper.execution.order_manager import OrderResult
    from deribit_paper.execution.straddle_flow import StraddleFlow
    from deribit_paper.signal.predictor import Signal

    om = MagicMock()
    om._rest = None
    om.place_maker_then_taker = AsyncMock(return_value=OrderResult(
        client_order_id="x", order_id="y", instrument_name="BTC",
        side="sell", amount_requested=1.0, filled_amount=1.0,
        average_price=0.05, state="filled", is_dry_run=True,
    ))
    om.place_ioc_taker = AsyncMock(return_value=OrderResult(
        client_order_id="x", order_id="y", instrument_name="BTC",
        side="buy", amount_requested=1.0, filled_amount=1.0,
        average_price=0.05, state="filled", is_dry_run=True,
    ))

    captured: dict = {}

    drl = MagicMock()

    def _begin_trade(**kwargs):
        captured.update(kwargs)
        return "t-001"

    drl.begin_trade = _begin_trade
    drl.close_trade = MagicMock()

    flow = StraddleFlow(
        order_manager=om,
        live_drl_hedger=drl,
        target_margin_utilization=0.05,
        im_fraction=0.20,
        max_trade_size_btc=10.0,
        holding_minutes=0,
        spot_fetcher=AsyncMock(return_value=60_000.0),
        min_trade_floor=True,
    )

    started_at = pd.Timestamp("2026-04-15 12:00:00", tz="UTC")
    straddle = SimpleNamespace(
        call_instrument="BTC-CALL", put_instrument="BTC-PUT",
        strike=60_000.0,
        expiry_dt=pd.Timestamp("2026-04-22 08:00:00", tz="UTC"),
        index_price=60_000.0,
    )
    bar_tsrv = SimpleNamespace(vol_ann_continuous=0.80, jump_fraction=0.05)
    sig = Signal(
        timestamp=started_at, direction=-1,
        edge_vol_pts=-2.0, rv_pred_pct=72.0, iv_used_pct=58.0,
        bar_tsrv=bar_tsrv, straddle=straddle, reason="test",
    )

    asyncio.run(flow.execute(
        signal=sig, equity=1_000_000.0,
        trailing_1h_volume_btc=20.0,
        # vol_ann_entry_lagged=None  # cold start
    ))

    # No lagged value → falls back to current bar (pre-D-8 behaviour).
    assert captured["vol_ann_entry"] == pytest.approx(0.80)
