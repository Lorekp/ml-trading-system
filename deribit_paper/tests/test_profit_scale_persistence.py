"""Tests for H4 ProfitScaleEstimator state persistence (2026-05-06).

Why this exists
---------------

Before H4, the ProfitScaleEstimator's EMAs were in-memory only.
On restart, ``_gross_count`` dropped to 0, the shrinkage weight
λ = n / (n + τ) collapsed to 0, and ``est_gross`` decayed to the
theoretical estimate (``est_theo``) for the first ~200 trades. The
V2 backtester does NOT reset its prior mid-stream — so live drift
silently diverged from backtest after every engine restart.

These tests pin the H4 contract:

1. ``state_dict()`` returns a serializable dict with the three
   tracked variables (gross_ema, gross_count, hedge_ema).
2. ``load_state(d)`` restores these values; subsequent estimates
   match the post-load lambda.
3. The SQLiteStore round-trip preserves values exactly.
4. Defensive: corrupt persisted values (NaN, negative count, etc.)
   are silently skipped — fall back to cold-start.
5. End-to-end: trip → save → close → reopen → load → verify EMA
   continuity.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest

from deribit_paper.execution.profit_scale import ProfitScaleEstimator
from deribit_paper.position.store import SQLiteStore


def test_state_dict_returns_three_tracked_vars():
    est = ProfitScaleEstimator()
    # Drive a few realised trades to move EMAs off zero
    for _ in range(5):
        est.update_realized(gross_pnl_usd=1500.0, hedge_cost_usd=20.0, n_btc=1.0)

    state = est.state_dict()
    assert set(state.keys()) == {
        "gross_per_btc_ema_usd", "gross_count", "hedge_per_btc_ema_usd",
    }
    assert state["gross_count"] == 5
    assert state["gross_per_btc_ema_usd"] > 0.0
    assert state["hedge_per_btc_ema_usd"] > 0.0


def test_load_state_restores_emas_and_count():
    est_a = ProfitScaleEstimator()
    for _ in range(50):
        est_a.update_realized(gross_pnl_usd=2000.0, hedge_cost_usd=30.0, n_btc=1.0)
    state = est_a.state_dict()

    est_b = ProfitScaleEstimator()
    assert est_b.gross_count == 0
    est_b.load_state(state)

    assert est_b.gross_count == est_a.gross_count
    assert est_b.gross_per_btc_ema_usd == pytest.approx(est_a.gross_per_btc_ema_usd)
    assert est_b.hedge_per_btc_ema_usd == pytest.approx(est_a.hedge_per_btc_ema_usd)
    # And the shrinkage weight is identical
    assert est_b.shrinkage_lambda == pytest.approx(est_a.shrinkage_lambda)


def test_load_state_with_nan_gross_falls_back_to_cold():
    est = ProfitScaleEstimator()
    est.load_state({
        "gross_per_btc_ema_usd": math.nan,
        "gross_count": 100,
        "hedge_per_btc_ema_usd": 50.0,
    })
    # Bad value rejected, others applied.
    assert est.gross_per_btc_ema_usd == 0.0   # cold default
    assert est.gross_count == 100
    assert est.hedge_per_btc_ema_usd == 50.0


def test_load_state_with_negative_count_falls_back():
    est = ProfitScaleEstimator()
    est.load_state({
        "gross_per_btc_ema_usd": 1500.0,
        "gross_count": -5,    # corrupt
        "hedge_per_btc_ema_usd": 30.0,
    })
    assert est.gross_count == 0  # cold default preserved


def test_load_state_with_negative_hedge_falls_back():
    est = ProfitScaleEstimator()
    est.load_state({
        "gross_per_btc_ema_usd": 1500.0,
        "gross_count": 50,
        "hedge_per_btc_ema_usd": -10.0,   # corrupt
    })
    assert est.hedge_per_btc_ema_usd == 0.0  # cold default preserved


def test_load_state_with_missing_keys_does_not_raise():
    """Defensive against schema drift / partial dicts."""
    est = ProfitScaleEstimator()
    est.load_state({})
    assert est.gross_count == 0
    assert est.gross_per_btc_ema_usd == 0.0
    assert est.hedge_per_btc_ema_usd == 0.0


# ── End-to-end SQLite round-trip ──────────────────────────────────────


@pytest.mark.asyncio
async def test_sqlite_round_trip_preserves_state(tmp_path):
    db = tmp_path / "ps.db"

    # Lifecycle 1: drive estimator + save state
    async with SQLiteStore(db) as store:
        est_a = ProfitScaleEstimator()
        for i in range(100):
            est_a.update_realized(
                gross_pnl_usd=1500.0 + i, hedge_cost_usd=20.0, n_btc=1.0,
            )
        await store.save_profit_scale_state(est_a.state_dict())

    # Lifecycle 2: reopen + load + verify
    async with SQLiteStore(db) as store:
        loaded = await store.load_profit_scale_state()
        assert loaded is not None
        assert loaded["gross_count"] == 100
        assert loaded["gross_per_btc_ema_usd"] == pytest.approx(est_a.gross_per_btc_ema_usd)
        assert loaded["hedge_per_btc_ema_usd"] == pytest.approx(est_a.hedge_per_btc_ema_usd)
        assert loaded["last_updated"] is not None

        est_b = ProfitScaleEstimator()
        est_b.load_state(loaded)
        assert est_b.gross_count == 100
        assert est_b.shrinkage_lambda == pytest.approx(est_a.shrinkage_lambda)


@pytest.mark.asyncio
async def test_load_returns_none_on_fresh_db(tmp_path):
    db = tmp_path / "fresh.db"
    async with SQLiteStore(db) as store:
        loaded = await store.load_profit_scale_state()
        assert loaded is None


@pytest.mark.asyncio
async def test_save_is_idempotent(tmp_path):
    db = tmp_path / "ps.db"
    async with SQLiteStore(db) as store:
        await store.save_profit_scale_state({
            "gross_per_btc_ema_usd": 1000.0,
            "gross_count": 50,
            "hedge_per_btc_ema_usd": 20.0,
        })
        await store.save_profit_scale_state({
            "gross_per_btc_ema_usd": 2000.0,
            "gross_count": 100,
            "hedge_per_btc_ema_usd": 40.0,
        })
        # Only the latest state is kept (single-row table).
        loaded = await store.load_profit_scale_state()
        assert loaded["gross_count"] == 100


# ── Continuity check: persisted estimator gives same estimate ──


@pytest.mark.asyncio
async def test_post_restart_estimate_matches_pre_restart(tmp_path):
    """The whole point of H4: an estimate computed pre-restart and
    post-restart on the same inputs must match (up to FP). Without
    H4, the post-restart lambda would be 0 and the estimate would
    collapse to est_theo."""
    db = tmp_path / "ps.db"

    args = dict(
        edge_vp=2.5,
        vega_per_vol_pt_usd=200.0,
        spot=60_000.0,
        jump_fraction=0.1,
        slip_usd_per_btc=12.0,
        n_btc=1.0,
    )

    async with SQLiteStore(db) as store:
        est_a = ProfitScaleEstimator()
        for _ in range(200):
            est_a.update_realized(gross_pnl_usd=1800.0, hedge_cost_usd=25.0, n_btc=1.0)
        ps_pre = est_a.estimate(**args)
        await store.save_profit_scale_state(est_a.state_dict())

    # Simulate a restart
    async with SQLiteStore(db) as store:
        est_b = ProfitScaleEstimator()
        loaded = await store.load_profit_scale_state()
        est_b.load_state(loaded)
        ps_post = est_b.estimate(**args)

    assert ps_post == pytest.approx(ps_pre, abs=1e-9), (
        f"H4 continuity violated: pre={ps_pre}, post={ps_post}"
    )
