"""D4 (parity-cycle round 2 — 2026-05-07): ``LiveDRLHedger.step``
skips the ``hedge_inventory_btc`` update when ``|trade_size| <
dust_threshold_btc`` so internal inventory doesn't drift from the
actual perp position.

Pre-D4 the inventory updated unconditionally, but the downstream
``PerpHedger`` silently no-op'd sub-dust orders. Result: each sub-dust
tick added the un-placed delta to the policy's internal inventory,
the obs vector then reported a fake hedge_position, and over many
sub-dust ticks the policy ran out-of-distribution.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np

from deribit_paper.signal.drl_hedger_live import LiveDRLHedger
from deribit_paper.signal.model_loader import ModelBundle


def _make_stub_bundle(action_value: float) -> ModelBundle:
    drl = MagicMock()
    drl.predict.return_value = (np.array([[action_value]], dtype=np.float32), None)
    return ModelBundle(
        month_key="stub",
        lgbm_model=None, lgbm_features=[],
        transvol_model=None, transvol_config=None,
        transvol_feat_mean=None, transvol_feat_std=None,
        transvol_time_mean=None, transvol_time_std=None,
        drl_policy=drl,
        drl_vec_normalize_mean=None, drl_vec_normalize_var=None,
        drl_vec_normalize_clip=10.0,
    )


def test_dust_filter_default_threshold_is_0_01():
    """Default dust threshold matches the engine's perp_hedger default."""
    bundle = _make_stub_bundle(0.5)
    h = LiveDRLHedger(bundle=bundle)
    assert h._dust_threshold_btc == 0.01


def test_sub_dust_action_does_not_advance_inventory():
    """Tiny action × tiny |delta_required| → trade_size ~ 0 → inventory unchanged."""
    # action_value=0.001 → 0.1% of |delta_required|. With small delta_required
    # (which we engineer via a small spot move from S_entry), trade_size is sub-dust.
    bundle = _make_stub_bundle(action_value=0.001)
    h = LiveDRLHedger(bundle=bundle, dust_threshold_btc=0.01)
    tid = h.begin_trade(
        signal=-1, S_entry=50_000, K=50_000,
        vol_ann_entry=0.8, iv_entry=0.75, tau_entry=0.04,
        V_bg=100.0, jump_fraction=0.0, vpin=0.5,
    )
    # Move spot a small amount → small delta_theo change → small delta_required
    inv_post = h.step(tid, current_spot=50_010.0)
    # action_value 0.001 × small delta_required → trade_size < 0.01 → skip
    assert inv_post == 0.0  # inventory remained at zero


def test_full_action_with_meaningful_delta_does_update_inventory():
    """Sanity: a full hedge action with a meaningful delta still updates inventory."""
    bundle = _make_stub_bundle(action_value=1.0)
    h = LiveDRLHedger(bundle=bundle, dust_threshold_btc=0.01)
    tid = h.begin_trade(
        signal=-1, S_entry=50_000, K=50_000,
        vol_ann_entry=0.8, iv_entry=0.75, tau_entry=0.04,
        V_bg=100.0, jump_fraction=0.0, vpin=0.5,
    )
    # Big spot move → big delta_required → trade_size well above dust
    inv_post = h.step(tid, current_spot=52_000.0)
    assert abs(inv_post) > 0.01


def test_dust_filter_zero_threshold_back_compat():
    """dust_threshold_btc=0.0 → fall through to pre-D4 behaviour (always update)."""
    bundle = _make_stub_bundle(action_value=0.001)
    h = LiveDRLHedger(bundle=bundle, dust_threshold_btc=0.0)
    tid = h.begin_trade(
        signal=-1, S_entry=50_000, K=50_000,
        vol_ann_entry=0.8, iv_entry=0.75, tau_entry=0.04,
        V_bg=100.0, jump_fraction=0.0, vpin=0.5,
    )
    inv_post = h.step(tid, current_spot=50_010.0)
    # With dust threshold 0, even a tiny trade is recorded
    assert inv_post != 0.0


def test_dust_filter_compounding_drift_eliminated():
    """10 sub-dust ticks in a row → inventory still 0 (no compound drift)."""
    bundle = _make_stub_bundle(action_value=0.001)
    h = LiveDRLHedger(bundle=bundle, dust_threshold_btc=0.01)
    tid = h.begin_trade(
        signal=-1, S_entry=50_000, K=50_000,
        vol_ann_entry=0.8, iv_entry=0.75, tau_entry=0.04,
        V_bg=100.0, jump_fraction=0.0, vpin=0.5,
    )
    for tick_spot in (50_005.0, 50_010.0, 50_015.0, 50_020.0, 50_025.0,
                      50_030.0, 50_035.0, 50_040.0, 50_045.0, 50_050.0):
        h.step(tid, current_spot=tick_spot)
    summary = h.close_trade(tid)
    assert summary["final_inventory_btc"] == 0.0
    assert summary["total_trade_size_btc"] == 0.0
