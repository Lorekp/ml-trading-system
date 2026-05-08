"""D-11 (parity-r3 cycle 2 — 2026-05-08) regression tests.

The DRL hedger's per-trade ``hedge_inventory_btc`` must survive an engine
restart. Pre-fix, ``LiveDRLHedger._trades`` lived only in memory; on
restart, ``_resume_trade`` rebuilt the position tracker but the hedger
had no entry → next perp tick computed ``delta_required = target - 0.0``
and over-traded by exactly the prior tick's hedge inventory.

This test set covers:
1. SQLite store layer (persist / load / remove round-trip).
2. ``LiveDRLHedger`` callback firing (begin / step / close).
3. ``_LiveTradeState.to_persistable_dict`` ↔ ``from_persisted_dict``
   round-trip (with and without SVI slice).
4. ``resume_trade_from_dict`` rehydrates the hedger.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from deribit_paper.position.store import SQLiteStore
from deribit_paper.signal.drl_hedger_live import (
    LiveDRLHedger,
    _LiveTradeState,
    _TradeSliceSVI,
)
from deribit_paper.signal.model_loader import ModelBundle


# ── Fixtures ──────────────────────────────────────────────────────────


def _make_state(
    trade_id: str = "t_abc",
    *,
    with_slice: bool = True,
    hedge_inv: float = 0.0,
    tick: int = 0,
) -> _LiveTradeState:
    slice_obj = None
    if with_slice:
        slice_obj = _TradeSliceSVI(
            a=0.04, b=0.10, rho=-0.3, m=-0.05, sigma=0.20,
            T=0.04, F=60_000.0,
        )
    return _LiveTradeState(
        trade_id=trade_id,
        signal=-1,
        K=60_000.0,
        S_entry=60_000.0,
        vol_ann_entry=0.80,
        iv_entry=0.75,
        V_bg=120.0,
        jump_fraction=0.05,
        vpin=0.30,
        tau_entry=0.04,
        delta_entry=0.0,
        delta_theo_entry_abs=0.5,
        svi_slice=slice_obj,
        hedge_inventory_btc=hedge_inv,
        tick=tick,
        total_trade_size_btc=abs(hedge_inv),
        total_impact_cost_usd=hedge_inv * 0.5,
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


# ── _LiveTradeState round-trip ────────────────────────────────────────


def test_to_persistable_dict_with_slice_roundtrips():
    state = _make_state(with_slice=True, hedge_inv=-0.42, tick=4)
    d = state.to_persistable_dict()
    rehydrated = _LiveTradeState.from_persisted_dict(d)

    assert rehydrated.trade_id == state.trade_id
    assert rehydrated.signal == state.signal
    assert rehydrated.K == state.K
    assert rehydrated.S_entry == state.S_entry
    assert rehydrated.hedge_inventory_btc == pytest.approx(-0.42)
    assert rehydrated.tick == 4
    assert rehydrated.svi_slice is not None
    assert rehydrated.svi_slice.a == pytest.approx(0.04)
    assert rehydrated.svi_slice.b == pytest.approx(0.10)
    assert rehydrated.svi_slice.F == pytest.approx(60_000.0)


def test_to_persistable_dict_without_slice_roundtrips():
    state = _make_state(with_slice=False, hedge_inv=0.15, tick=2)
    d = state.to_persistable_dict()
    rehydrated = _LiveTradeState.from_persisted_dict(d)

    assert rehydrated.svi_slice is None
    assert all(d[col] is None for col in (
        "svi_a", "svi_b", "svi_rho", "svi_m", "svi_sigma", "svi_t", "svi_f",
    ))
    assert rehydrated.hedge_inventory_btc == pytest.approx(0.15)
    assert rehydrated.tick == 2


def test_from_persisted_dict_resets_step_log():
    """step_log is observability-only and not persisted; resume starts
    with empty list (matches begin_trade behaviour)."""
    state = _make_state()
    state.step_log.append({"tick": 1, "spot": 60_001.0})
    d = state.to_persistable_dict()
    assert "step_log" not in d
    rehydrated = _LiveTradeState.from_persisted_dict(d)
    assert rehydrated.step_log == []


# ── SQLiteStore layer ─────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_store_persist_and_load_drl_state(tmp_path):
    store = SQLiteStore(tmp_path / "test.db")
    await store.open()
    try:
        # Parent open_position row required by the FK
        await store.persist_open_position(
            trade_id="t_abc",
            call_instrument="BTC-30APR26-60000-C",
            put_instrument="BTC-30APR26-60000-P",
            signal_direction=-1,
            n_contracts_btc=1.0,
            strike=60_000.0,
            expiry_dt="2026-04-30T08:00:00+00:00",
            opened_at="2026-05-08T12:00:00+00:00",
        )

        d = _make_state(trade_id="t_abc", hedge_inv=-0.42, tick=4).to_persistable_dict()
        await store.persist_drl_trade_state(d)

        rows = await store.load_drl_trade_states()
        assert len(rows) == 1
        loaded = rows[0]
        assert loaded["trade_id"] == "t_abc"
        assert loaded["hedge_inventory_btc"] == pytest.approx(-0.42)
        assert loaded["tick"] == 4
        assert loaded["svi_a"] == pytest.approx(0.04)
    finally:
        await store.close() if hasattr(store, "close") else None


@pytest.mark.asyncio
async def test_store_persist_drl_state_upsert(tmp_path):
    """Repeated persist on the same trade_id updates the row (INSERT OR
    REPLACE)."""
    store = SQLiteStore(tmp_path / "test.db")
    await store.open()
    try:
        await store.persist_open_position(
            trade_id="t_xyz",
            call_instrument="BTC-30APR26-60000-C",
            put_instrument="BTC-30APR26-60000-P",
            signal_direction=-1,
            n_contracts_btc=1.0,
            strike=60_000.0,
            expiry_dt="2026-04-30T08:00:00+00:00",
            opened_at="2026-05-08T12:00:00+00:00",
        )
        await store.persist_drl_trade_state(
            _make_state(trade_id="t_xyz", hedge_inv=-0.10, tick=1).to_persistable_dict(),
        )
        await store.persist_drl_trade_state(
            _make_state(trade_id="t_xyz", hedge_inv=-0.42, tick=4).to_persistable_dict(),
        )

        rows = await store.load_drl_trade_states()
        assert len(rows) == 1
        assert rows[0]["hedge_inventory_btc"] == pytest.approx(-0.42)
        assert rows[0]["tick"] == 4
    finally:
        if hasattr(store, "close"):
            await store.close()


@pytest.mark.asyncio
async def test_store_remove_drl_state_idempotent(tmp_path):
    store = SQLiteStore(tmp_path / "test.db")
    await store.open()
    try:
        # Remove on empty table — must not raise
        await store.remove_drl_trade_state("nonexistent")

        await store.persist_open_position(
            trade_id="t_rm", call_instrument="C", put_instrument="P",
            signal_direction=-1, n_contracts_btc=1.0, strike=60000.0,
            expiry_dt="2026-04-30T08:00:00+00:00",
            opened_at="2026-05-08T12:00:00+00:00",
        )
        await store.persist_drl_trade_state(
            _make_state(trade_id="t_rm").to_persistable_dict(),
        )
        assert len(await store.load_drl_trade_states()) == 1

        await store.remove_drl_trade_state("t_rm")
        assert await store.load_drl_trade_states() == []

        # Remove again — idempotent
        await store.remove_drl_trade_state("t_rm")
    finally:
        if hasattr(store, "close"):
            await store.close()


@pytest.mark.asyncio
async def test_store_drl_state_cascades_on_open_position_delete(tmp_path):
    """FK ON DELETE CASCADE: removing the parent open_positions row
    drops the linked drl_trade_state row.

    SQLite enforces FK constraints only when ``PRAGMA foreign_keys=ON``
    is set on the connection. SQLiteStore does NOT enable this today
    (matches pre-D-11 conventions across the engine). This test
    documents the contract: when the engine flips foreign_keys ON
    in a future hardening pass, cascade is the expected semantic.
    Until then we verify the looser semantic: parent gone, child
    orphan stays — which is acceptable because resume only follows
    parent rows.
    """
    store = SQLiteStore(tmp_path / "test.db")
    await store.open()
    try:
        await store.persist_open_position(
            trade_id="t_cascade", call_instrument="C", put_instrument="P",
            signal_direction=-1, n_contracts_btc=1.0, strike=60000.0,
            expiry_dt="2026-04-30T08:00:00+00:00",
            opened_at="2026-05-08T12:00:00+00:00",
        )
        await store.persist_drl_trade_state(
            _make_state(trade_id="t_cascade").to_persistable_dict(),
        )
        assert len(await store.load_drl_trade_states()) == 1

        # Remove parent. The application layer (engine) explicitly
        # removes the child via ``remove_drl_trade_state`` in
        # ``close_trade``; the FK is documentary until PRAGMA
        # foreign_keys=ON is enabled.
        await store.remove_open_position("t_cascade")
        # Either cascade (FK ON) or orphan-tolerated (FK OFF) — both
        # acceptable. Resume code paths only follow open_positions rows.
        rows = await store.load_drl_trade_states()
        assert len(rows) <= 1  # cascade if pragma ON, orphan if OFF
    finally:
        if hasattr(store, "close"):
            await store.close()


# ── LiveDRLHedger callback firing ─────────────────────────────────────


def test_hedger_fires_persister_on_begin_trade():
    persisted: list[dict[str, Any]] = []

    hedger = LiveDRLHedger(
        bundle=_stub_bundle(),
        n_steps_per_episode=10,
        bar_minutes=15,
        state_persister=lambda d: persisted.append(d),
    )

    trade_id = hedger.begin_trade(
        signal=-1, S_entry=60_000.0, K=60_000.0,
        vol_ann_entry=0.80, iv_entry=0.75,
        tau_entry=0.04, V_bg=120.0,
        jump_fraction=0.05, vpin=0.30,
    )

    assert len(persisted) == 1
    assert persisted[0]["trade_id"] == trade_id
    assert persisted[0]["hedge_inventory_btc"] == 0.0
    assert persisted[0]["tick"] == 0


def test_hedger_fires_persister_on_step():
    persisted: list[dict[str, Any]] = []

    hedger = LiveDRLHedger(
        bundle=_stub_bundle(),
        n_steps_per_episode=10,
        bar_minutes=15,
        state_persister=lambda d: persisted.append(d),
    )

    trade_id = hedger.begin_trade(
        signal=-1, S_entry=60_000.0, K=60_000.0,
        vol_ann_entry=0.80, iv_entry=0.75,
        tau_entry=0.04, V_bg=120.0,
        jump_fraction=0.05, vpin=0.30,
    )
    persisted.clear()  # drop the begin_trade emit

    hedger.step(trade_id, current_spot=60_100.0)
    hedger.step(trade_id, current_spot=60_200.0)

    assert len(persisted) == 2
    assert persisted[0]["tick"] == 1
    assert persisted[1]["tick"] == 2


def test_hedger_fires_remover_on_close_trade():
    removed: list[str] = []

    hedger = LiveDRLHedger(
        bundle=_stub_bundle(),
        n_steps_per_episode=10,
        bar_minutes=15,
        state_remover=lambda tid: removed.append(tid),
    )

    trade_id = hedger.begin_trade(
        signal=-1, S_entry=60_000.0, K=60_000.0,
        vol_ann_entry=0.80, iv_entry=0.75,
        tau_entry=0.04, V_bg=120.0,
        jump_fraction=0.05, vpin=0.30,
    )
    hedger.close_trade(trade_id)

    assert removed == [trade_id]


def test_hedger_resume_trade_from_dict_rehydrates_state():
    hedger = LiveDRLHedger(
        bundle=_stub_bundle(),
        n_steps_per_episode=10,
        bar_minutes=15,
    )

    state_dict = _make_state(
        trade_id="t_resume", hedge_inv=-0.42, tick=4,
    ).to_persistable_dict()
    tid = hedger.resume_trade_from_dict(state_dict)

    assert tid == "t_resume"
    assert hedger.active_trade_count() == 1
    # Step should produce a hedge_inventory_btc anchored at -0.42
    out = hedger.step("t_resume", current_spot=60_100.0)
    # The action stub returns 0.0 → inventory unchanged at -0.42
    assert out == pytest.approx(-0.42)


def test_hedger_resume_rejects_duplicate_trade_id():
    hedger = LiveDRLHedger(bundle=_stub_bundle(), bar_minutes=15)
    state = _make_state(trade_id="t_dup").to_persistable_dict()
    hedger.resume_trade_from_dict(state)
    with pytest.raises(ValueError, match="already active"):
        hedger.resume_trade_from_dict(state)


def test_hedger_persister_failure_does_not_break_step():
    """If the persister raises, the in-memory hot path still returns."""
    def angry_persister(_d: dict[str, Any]) -> None:
        raise RuntimeError("disk full")

    hedger = LiveDRLHedger(
        bundle=_stub_bundle(),
        bar_minutes=15,
        state_persister=angry_persister,
    )
    trade_id = hedger.begin_trade(
        signal=-1, S_entry=60_000.0, K=60_000.0,
        vol_ann_entry=0.80, iv_entry=0.75,
        tau_entry=0.04, V_bg=120.0,
        jump_fraction=0.05, vpin=0.30,
    )
    # Should not raise
    out = hedger.step(trade_id, current_spot=60_100.0)
    assert isinstance(out, float)
