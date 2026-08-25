"""Tests for vol_arb/honest_pnl.py — the real-fill premium mark that mirrors
the live engine. Synthetic, deterministic; no data files.

Locks: the long/short sign convention, the per-fill fee cap, the tiering
(real fill -> mark_price -> uncovered), and a no-lookahead probe (the exit
mark must not use a trade that printed after the exit target).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_arb.honest_pnl import (
    mark_trade_log_to_real_premium,
    strict_pnl_realism_gate,
)


def _tape(rows):
    """rows: list of (ts, strike, expiry, type, price, mark)."""
    return pd.DataFrame(
        rows, columns=["timestamp", "strike", "expiry_dt", "option_type",
                       "price", "mark_price"],
    ).assign(
        timestamp=lambda d: pd.to_datetime(d["timestamp"], utc=True),
        expiry_dt=lambda d: pd.to_datetime(d["expiry_dt"], utc=True),
    )


def _fired(rows):
    """rows: list of (entry_ts, strike, expiry, signal, N, index, exit_ts)."""
    idx = pd.to_datetime([r[0] for r in rows], utc=True)
    return pd.DataFrame({
        "strike": [r[1] for r in rows],
        "expiry_dt": pd.to_datetime([r[2] for r in rows], utc=True),
        "signal": [r[3] for r in rows],
        "N_contracts_nc": [r[4] for r in rows],
        "index_price": [r[5] for r in rows],
        "exit_target": pd.to_datetime([r[6] for r in rows], utc=True),
    }, index=idx)


E = "2023-06-09 08:00:00"
K = 100.0


def test_short_straddle_premium_fall_is_a_win():
    # premium_in=5 (C2+P3), premium_out=2 (C1+P1); short profits when premium falls.
    tape = _tape([
        ("2023-06-01 11:15:00", K, E, "C", 2.0, 2.0),
        ("2023-06-01 11:15:00", K, E, "P", 3.0, 3.0),
        ("2023-06-01 11:30:00", K, E, "C", 1.0, 1.0),
        ("2023-06-01 11:30:00", K, E, "P", 1.0, 1.0),
    ])
    fired = _fired([("2023-06-01 11:15:00", K, E, -1, 1.0, 100.0, "2023-06-01 11:30:00")])
    res = mark_trade_log_to_real_premium(fired, tape, tol_minutes=15.0)
    row = res.trade_log.iloc[0]
    assert row["premium_tier"] == 1
    # pnl_btc = (5-2)*1 = 3 ; usd = 300 ; fees = 0.0003*4*100 = 0.12
    assert row["real_option_pnl_nc"] == pytest.approx(300.0 - 0.12, abs=1e-6)
    assert res.coverage_frac == 1.0


def test_long_straddle_flips_sign():
    tape = _tape([
        ("2023-06-01 11:15:00", K, E, "C", 2.0, 2.0),
        ("2023-06-01 11:15:00", K, E, "P", 3.0, 3.0),
        ("2023-06-01 11:30:00", K, E, "C", 1.0, 1.0),
        ("2023-06-01 11:30:00", K, E, "P", 1.0, 1.0),
    ])
    fired = _fired([("2023-06-01 11:15:00", K, E, +1, 1.0, 100.0, "2023-06-01 11:30:00")])
    res = mark_trade_log_to_real_premium(fired, tape, tol_minutes=15.0)
    # long loses when premium falls: -(5-2)*100 = -300, minus fees
    assert res.trade_log.iloc[0]["real_option_pnl_nc"] == pytest.approx(-300.0 - 0.12, abs=1e-6)


def test_fee_cap_binds_on_cheap_legs():
    # price 0.001 -> 0.125*0.001 = 0.000125 < flat 0.0003 -> cap binds.
    tape = _tape([
        ("2023-06-01 11:15:00", K, E, "C", 0.001, 0.001),
        ("2023-06-01 11:15:00", K, E, "P", 0.001, 0.001),
        ("2023-06-01 11:30:00", K, E, "C", 0.001, 0.001),
        ("2023-06-01 11:30:00", K, E, "P", 0.001, 0.001),
    ])
    fired = _fired([("2023-06-01 11:15:00", K, E, -1, 1.0, 100.0, "2023-06-01 11:30:00")])
    res = mark_trade_log_to_real_premium(fired, tape, tol_minutes=15.0)
    # premium flat -> 0 option pnl; fee = 4 * 0.000125 * 100 = 0.05
    assert res.trade_log.iloc[0]["real_option_pnl_nc"] == pytest.approx(-0.05, abs=1e-9)


def test_mark_price_fallback_is_tier2():
    # entry call has NO fill price (NaN) but a mark_price -> tier 2.
    tape = _tape([
        ("2023-06-01 11:15:00", K, E, "C", np.nan, 2.0),
        ("2023-06-01 11:15:00", K, E, "P", 3.0, 3.0),
        ("2023-06-01 11:30:00", K, E, "C", 1.0, 1.0),
        ("2023-06-01 11:30:00", K, E, "P", 1.0, 1.0),
    ])
    fired = _fired([("2023-06-01 11:15:00", K, E, -1, 1.0, 100.0, "2023-06-01 11:30:00")])
    res = mark_trade_log_to_real_premium(fired, tape, tol_minutes=15.0)
    assert res.trade_log.iloc[0]["premium_tier"] == 2
    assert res.n_tier2 == 1 and res.n_tier1 == 0


def test_no_lookahead_exit_ignores_future_trade():
    # The only exit-side trades print AFTER exit_target -> backward match finds
    # nothing within tolerance at exit -> uncovered (tier 0), NOT a future fill.
    tape = _tape([
        ("2023-06-01 11:15:00", K, E, "C", 2.0, 2.0),
        ("2023-06-01 11:15:00", K, E, "P", 3.0, 3.0),
        # exit_target is 11:30; these are at 11:40, beyond a backward match
        ("2023-06-01 11:40:00", K, E, "C", 1.0, 1.0),
        ("2023-06-01 11:40:00", K, E, "P", 1.0, 1.0),
    ])
    fired = _fired([("2023-06-01 11:15:00", K, E, -1, 1.0, 100.0, "2023-06-01 11:30:00")])
    res = mark_trade_log_to_real_premium(fired, tape, tol_minutes=5.0)
    assert res.trade_log.iloc[0]["premium_tier"] == 0
    assert np.isnan(res.trade_log.iloc[0]["real_option_pnl_nc"])


def test_uncovered_when_contract_never_trades():
    tape = _tape([
        ("2023-06-01 11:15:00", 999.0, E, "C", 2.0, 2.0),  # wrong strike
        ("2023-06-01 11:15:00", 999.0, E, "P", 3.0, 3.0),
    ])
    fired = _fired([("2023-06-01 11:15:00", K, E, -1, 1.0, 100.0, "2023-06-01 11:30:00")])
    res = mark_trade_log_to_real_premium(fired, tape, tol_minutes=15.0)
    assert res.trade_log.iloc[0]["premium_tier"] == 0
    assert res.coverage_frac == 0.0


def test_strict_realism_gate_passes_independent_tier1_agreement():
    real = np.linspace(10.0, 50.0, 40)
    marked = pd.DataFrame({
        "premium_tier": np.ones(40, dtype=np.int8),
        "net_pnl_nc": real * 1.05,
        "real_option_pnl_nc": real,
    })

    gate = strict_pnl_realism_gate(marked)

    assert gate.passed
    assert gate.detail["n_tier1_finite"] == 40
    assert gate.detail["tier1_coverage"] == 1.0


def test_strict_realism_gate_rejects_mark_price_only_coverage():
    marked = pd.DataFrame({
        "premium_tier": np.full(40, 2, dtype=np.int8),
        "net_pnl_nc": np.linspace(10.0, 50.0, 40),
        "real_option_pnl_nc": np.linspace(10.0, 50.0, 40),
    })

    gate = strict_pnl_realism_gate(marked)

    assert not gate.passed
    assert gate.detail["n_tier1_finite"] == 0


def test_strict_realism_gate_rejects_unrealistic_pnl_shape():
    real = np.linspace(10.0, 50.0, 40)
    model = real[::-1]
    marked = pd.DataFrame({
        "premium_tier": np.ones(40, dtype=np.int8),
        "net_pnl_nc": model,
        "real_option_pnl_nc": real,
    })

    gate = strict_pnl_realism_gate(marked)

    assert not gate.passed
    assert gate.detail["corr"] < 0.85
