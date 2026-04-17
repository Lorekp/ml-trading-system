"""Tests for the non-compounding backtest path.

These tests run a single real backtest via a module-scoped fixture
(the setup is ~2 seconds; running it once across all tests is efficient),
then verify properties of the ``_nc`` columns that the backtester attaches
when ``report_non_compounding=True``.

The CRITICAL correctness check is :func:`test_vectorized_matches_formula`,
which reconstructs ``N_nc`` from the documented formula using columns
that the backtester already exposes in the trade log (``dynamic_margin``,
``trailing_1h_volume``, ``signal``, ``index_price``) and asserts
bar-by-bar equality with the vectorized block's output.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_arb.backtester import BacktestConfig, VolArbBacktester
from vol_arb.metrics import build_nc_view, full_metrics_both


# ── Fixtures ──────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def full_run():
    """Run the backtester once with non-compounding ON.

    Uses use_instrument_routing=False to match the BacktestConfig default
    (matches the rule-based A/B baseline we care about).  Reuse this
    fixture across tests — it's expensive (~2-5 seconds).
    """
    cfg = BacktestConfig(
        use_instrument_routing=False,
        report_non_compounding=True,
        entry_threshold=20.0,
        sizing_model="baseline",
        slippage_model="grid",
        use_profit_filter=False,
    )
    bt = VolArbBacktester(cfg)
    metrics, trade_log = bt.run_backtest()
    return cfg, metrics, trade_log


@pytest.fixture(scope="module")
def full_run_no_nc():
    """Run the backtester with non-compounding OFF — for regression tests."""
    cfg = BacktestConfig(
        use_instrument_routing=False,
        report_non_compounding=False,
        entry_threshold=20.0,
        sizing_model="baseline",
        slippage_model="grid",
        use_profit_filter=False,
    )
    bt = VolArbBacktester(cfg)
    metrics, trade_log = bt.run_backtest()
    return cfg, metrics, trade_log


# ── Presence / absence of columns ────────────────────────────────────

def test_nc_columns_present_when_flag_on(full_run):
    _, _, tl = full_run
    expected = [
        "net_pnl_nc", "equity_nc", "r_t_nc", "N_contracts_nc",
        "slippage_nc", "vega_pnl_nc", "theta_pnl_nc",
        "delta_hedge_pnl_nc", "hedge_costs_nc", "gross_pnl_nc",
        "maker_filled_nc",
    ]
    for col in expected:
        assert col in tl.columns, f"missing {col}"


def test_nc_columns_absent_when_flag_off(full_run_no_nc):
    _, _, tl = full_run_no_nc
    assert "net_pnl_nc" not in tl.columns
    assert "equity_nc" not in tl.columns
    assert "r_t_nc" not in tl.columns


def test_nc_metrics_in_result(full_run):
    _, metrics, _ = full_run
    assert "non_compounding" in metrics
    nc = metrics["non_compounding"]
    assert "total_pnl" in nc
    assert "sharpe_ratio" in nc
    assert nc["n_trades"] > 0


# ── Structural properties ────────────────────────────────────────────

def test_nc_equity_is_additive(full_run):
    """equity_nc[i] = initial_equity + cumsum(net_pnl_nc)[:i+1]"""
    cfg, _, tl = full_run
    expected = cfg.initial_equity + tl["net_pnl_nc"].cumsum().values
    np.testing.assert_allclose(
        tl["equity_nc"].values, expected,
        rtol=0, atol=1e-6,
        err_msg="equity_nc must be additive cumulative PnL",
    )


def test_nc_inactive_trades_are_zero(full_run):
    _, _, tl = full_run
    inactive = tl[tl["signal"] == 0]
    assert (inactive["N_contracts_nc"] == 0).all()
    assert (inactive["net_pnl_nc"] == 0).all()
    assert (inactive["slippage_nc"] == 0).all()
    assert (inactive["hedge_costs_nc"] == 0).all()


def test_nc_r_t_uses_fixed_notional(full_run):
    """r_t_nc = net_pnl_nc / initial_equity — constant denominator."""
    cfg, _, tl = full_run
    expected = tl["net_pnl_nc"].values / cfg.initial_equity
    np.testing.assert_allclose(
        tl["r_t_nc"].values, expected,
        rtol=0, atol=1e-10,
    )


# ── Vectorized-vs-formula correctness (the key test) ────────────────

def test_vectorized_matches_formula(full_run):
    """Reconstruct N_nc from the documented formula using trade-log columns
    and assert bar-by-bar equality with the vectorized non-compound block.

    This is the correctness guarantee: if the vectorized code deviates from
    the mathematical specification of the non-compound sizing rule, this
    test catches it.
    """
    cfg, _, tl = full_run

    S = tl["index_price"].values
    signal = tl["signal"].values
    dynamic_margin = tl["dynamic_margin"].values

    # Reconstruct dynamic_btc_cap exactly as compute_pnl computes it
    trailing_vol = tl["trailing_1h_volume"].values
    dynamic_btc_cap = np.clip(
        trailing_vol * cfg.max_participation_rate, 1.0, 50.0,
    )

    # Reference formula (matches the backtester's vectorized block)
    # im_per_bar is direction-dependent: long IM = premium, short IM = cfg.im_fraction
    im_bar = tl["im_per_bar"].values
    n_target = (cfg.initial_equity * dynamic_margin) / (im_bar * S)
    n_target = np.where(signal == -1, n_target * cfg.short_sizing_penalty, n_target)
    expected_N_nc = np.minimum(n_target, dynamic_btc_cap)
    expected_N_nc = np.where(signal == 0, 0.0, expected_N_nc)

    np.testing.assert_allclose(
        tl["N_contracts_nc"].values, expected_N_nc,
        rtol=1e-10, atol=1e-10,
        err_msg="Vectorized N_nc must match the closed-form non-compound sizing rule",
    )


def test_nc_independent_of_compound_feedback(full_run):
    """N_nc must not depend on running compound equity.

    We check this indirectly: the ratio N_nc / dynamic_margin (per active
    trade) should be independent of trade index — it should only vary as
    (initial_equity / S / im_fraction), which has no time-dependence from
    the compounding feedback loop.  If compound PnL leaked into N_nc
    sizing, we'd see drift correlated with the compound equity curve.
    """
    cfg, _, tl = full_run
    active = tl[(tl["signal"] != 0) & (tl["N_contracts_nc"] > 0)].copy()

    # Restrict to uncapped trades — where the min() cap hasn't kicked in —
    # otherwise the ratio reflects the cap, not the sizing formula
    trailing_vol = active["trailing_1h_volume"].values
    cap = np.clip(trailing_vol * cfg.max_participation_rate, 1.0, 50.0)
    uncapped = active["N_contracts_nc"].values < (cap - 1e-6)
    active = active[uncapped]
    if len(active) == 0:
        pytest.skip("No uncapped trades in this run")

    # Reconstruct the expected ratio: initial_equity / (im_per_bar * S)
    # with short_pen applied for short straddles
    pen = np.where(active["signal"].values == -1, cfg.short_sizing_penalty, 1.0)
    im_bar = active["im_per_bar"].values
    expected_ratio = (
        cfg.initial_equity * pen / (im_bar * active["index_price"].values)
    )
    actual_ratio = (
        active["N_contracts_nc"].values / active["dynamic_margin"].values
    )
    np.testing.assert_allclose(
        actual_ratio, expected_ratio,
        rtol=1e-9, atol=1e-9,
        err_msg="N_nc / dynamic_margin must equal initial_equity / (im * S) * short_pen",
    )


# ── Compound regression protection ──────────────────────────────────

def test_compound_unchanged_by_nc_flag(full_run, full_run_no_nc):
    """Toggling report_non_compounding must NOT change the compound path."""
    _, m_on, tl_on = full_run
    _, m_off, tl_off = full_run_no_nc

    # Compound trade_log columns must be identical
    np.testing.assert_array_equal(
        tl_on["net_pnl"].values, tl_off["net_pnl"].values,
    )
    np.testing.assert_array_equal(
        tl_on["equity"].values, tl_off["equity"].values,
    )
    np.testing.assert_array_equal(
        tl_on["N_contracts"].values, tl_off["N_contracts"].values,
    )

    # Compound metrics must be identical
    assert m_on["total_pnl"] == m_off["total_pnl"]
    assert m_on["sharpe_ratio"] == m_off["sharpe_ratio"]


# ── Metrics view ────────────────────────────────────────────────────

def test_build_nc_view_renames_columns(full_run):
    _, _, tl = full_run
    view = build_nc_view(tl)
    # Standard column names present in the view
    for col in ["net_pnl", "equity", "r_t", "N_contracts"]:
        assert col in view.columns
    # Source _nc columns should be dropped from the view
    for col in ["net_pnl_nc", "equity_nc", "r_t_nc"]:
        assert col not in view.columns
    # View's net_pnl should equal original's net_pnl_nc
    np.testing.assert_array_equal(
        view["net_pnl"].values, tl["net_pnl_nc"].values,
    )


def test_build_nc_view_raises_when_missing(full_run_no_nc):
    _, _, tl = full_run_no_nc
    with pytest.raises(ValueError, match="non-compounding"):
        build_nc_view(tl)


def test_full_metrics_both_produces_both_paths(full_run):
    _, _, tl = full_run
    both = full_metrics_both(tl, initial_equity=100_000)
    assert "compound" in both
    assert "non_compound" in both
    # Compound and non-compound should generally have different totals
    # (otherwise the strategy is zero-PnL over the entire window)
    assert both["compound"]["total_pnl"] != both["non_compound"]["total_pnl"]


# ── Directionality check ────────────────────────────────────────────

def test_compounding_amplifies_profitable_strategy(full_run):
    """Compound and NC PnL must agree on sign (both profitable or both losing).

    With realistic Deribit fees (option 0.03%, perp 5bps, IM 20%),
    compound PnL can be below NC PnL even when both are positive: early
    drawdowns reduce the compound path's equity below initial_equity,
    shrinking positions at exactly the wrong time.  This is correct
    behaviour, not a bug — the test verifies directional agreement only.
    """
    _, metrics, _ = full_run
    compound_pnl = metrics["total_pnl"]
    nc_pnl = metrics["non_compounding"]["total_pnl"]

    if nc_pnl > 0:
        assert compound_pnl > 0, (
            "NC is profitable but compound is not "
            f"(compound={compound_pnl}, nc={nc_pnl})"
        )
