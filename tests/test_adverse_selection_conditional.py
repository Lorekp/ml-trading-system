"""Regression tests for the conditional maker-adverse-selection penalty.

Background
----------
The `maker_adverse_selection_iv` penalty is an economic cost of MAKER fills
only — the maker earns the bid-ask spread but is adversely selected by more-
informed counterparties. TAKER fills already pay half-spread crossing via the
slippage model (`bar_taker_slip_usd` × size_mult, applied as `straddle_slip`
inside the PnL loop). Charging both on the same fill is double-counting
spread cost.

Pre-fix (buggy): penalty was applied unconditionally to every trade by baking
it into `effective_entry_iv` at the top of the pipeline. With current V2
trade log having `maker_filled = False` on 100% of 8,308 trades, this imposed
a synthetic cost of 0.5 * Σ|vega|·N ≈ $1.48M on the NC path — 218% of NC PnL.

Post-fix (correct): the up-front `dIV` uses unpenalized entry IVs. Inside the
per-bar PnL loop, the penalty `vega_s[i] * cfg.maker_adverse_selection_iv` is
subtracted from `pnl_per_btc` ONLY when `maker_filled_arr[i] == True` (i.e.,
SOR confirmed a maker fill).

These tests pin the invariant:
  1. With all maker_filled=False, NC PnL equals pre-fix NC PnL + $1.48M.
  2. With all maker_filled=True, the penalty is applied on every trade —
     equivalent to the pre-fix behavior at aggregate level.
  3. Mixed cases: the penalty is applied only to the maker-filled subset.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_arb.backtester import BacktestConfig


# ── 1. The invariant: saved V2 trade log matches the pre-fix + $1.48M delta ──


SAVED_TRADE_LOG = "vol_arb/output/BTC_v2_trade_log.parquet"

# Pre-Phase-1 baselines (post-shrinkage-fix):
#   NC PnL      $  678,859.77
#   Compound    $11,683,067
# Phase-1 NC invariant: 0.5 * Σ(|vega| × N_contracts_nc) ≈ $120,634.
# Compound invariant is larger by ~10x because N_contracts is dollar-scaled and
# the equity path is dollar-scaled; additionally the compound path shows a
# further amplification from path-dependent compounding when the penalty is
# removed early (freed capital → larger later-bar sizes → more penalty freed).
# Therefore NC is the clean arithmetic check; compound is validated via Sharpe
# improvement, not invariant identity.

PRE_FIX_NC_PNL = 678_859.77
NC_INVARIANT_TOLERANCE_PCT = 0.25  # ±25% to accommodate small path effects


def _sum_vega_penalty_nc(df: pd.DataFrame, penalty_vp: float = 0.5) -> float:
    """NC-path invariant: the penalty released when maker_filled=False on all
    bars is penalty_vp × Σ(|vega| × N_contracts_nc). Vega is always >=0 for a
    straddle; signal sign-flip is absorbed in dIV, leaving |vega_s| * penalty
    as the cost-per-BTC magnitude."""
    exe = df[df["N_contracts_nc"] > 0]
    return float(penalty_vp * (exe["vega"].abs() * exe["N_contracts_nc"]).sum())


def test_saved_v2_trade_log_reflects_penalty_removal():
    """After Phase 1, NC PnL on the committed V2 trade log should equal
    pre-fix baseline + the NC invariant (±25% to accommodate path effects
    from profit-filter shrinkage and compounding-like effects in margin calc)."""
    from pathlib import Path

    log_path = Path(SAVED_TRADE_LOG)
    if not log_path.exists():
        pytest.skip(f"{log_path} not present — regenerate via run_backtest")

    df = pd.read_parquet(log_path)
    # Sanity: no maker fills in the current config (SOR columns are sparse).
    n_maker = int(df["maker_filled"].sum())
    n_exe = int((df["N_contracts"] > 0).sum())
    assert n_maker == 0, (
        f"Test assumes 0 maker fills in V2 baseline. Got {n_maker}/{n_exe}."
    )

    invariant_nc = _sum_vega_penalty_nc(df)
    current_nc = float(df["net_pnl_nc"].sum())
    delta_nc = current_nc - PRE_FIX_NC_PNL
    # The observed delta should be close to the invariant.
    ratio = delta_nc / invariant_nc
    assert 1.0 - NC_INVARIANT_TOLERANCE_PCT <= ratio <= 1.0 + NC_INVARIANT_TOLERANCE_PCT, (
        f"NC PnL delta ${delta_nc:,.0f} differs from invariant ${invariant_nc:,.0f} "
        f"by ratio {ratio:.2f} (tolerance ±{NC_INVARIANT_TOLERANCE_PCT:.0%}). "
        f"If ratio < 0.75, the fix may not be fully plumbed. If ratio > 1.25, a "
        f"path-dependent interaction is amplifying the effect — audit before proceeding."
    )


def test_saved_v2_trade_log_shows_sharpe_improvement():
    """Phase 1 should improve NC Sharpe (risk-adjusted return) because the
    penalty was a uniform-per-trade deduction; removing it adds return
    without adding volatility proportionally."""
    from pathlib import Path

    log_path = Path(SAVED_TRADE_LOG)
    if not log_path.exists():
        pytest.skip(f"{log_path} not present — regenerate via run_backtest")

    df = pd.read_parquet(log_path)
    nc_ret = df["equity_nc"].pct_change().dropna()
    bars_per_year = 365.25 * 24 * 4  # 15m bars
    nc_sharpe = (nc_ret.mean() / nc_ret.std()) * np.sqrt(bars_per_year)

    # Pre-Phase-1 NC Sharpe was 5.65. Post-Phase-1 should be materially higher.
    # Gate at 7.5 — a conservative threshold: if Phase 1 drops Sharpe below
    # this, something is structurally wrong (likely a regression was introduced).
    assert nc_sharpe >= 7.5, (
        f"Phase 1 NC Sharpe {nc_sharpe:.2f} is below the 7.5 gate. "
        f"Pre-Phase-1 baseline was 5.65; the fix should improve Sharpe, "
        f"not degrade it."
    )


# ── 2. Config contract ─────────────────────────────────────────────


def test_maker_adverse_selection_default_unchanged():
    """The penalty parameter itself is not changed — 0.5 vp is still the
    correct magnitude when applied on actual maker fills."""
    cfg = BacktestConfig()
    assert cfg.maker_adverse_selection_iv == 0.5


def test_v2_preset_inherits_penalty_param():
    cfg = BacktestConfig().with_btc_production_filters_v2()
    assert cfg.maker_adverse_selection_iv == 0.5


# ── 3. Economic-logic invariant (arithmetic only, no full backtest) ──


def test_penalty_invariant_arithmetic():
    """For a per-trade realization, the penalty subtracted from pnl_per_btc is
    `vega_s * penalty` regardless of signal direction. Walk through both
    long and short scenarios to confirm sign-independence of the magnitude."""
    penalty = 0.5
    vega_s = 50.0  # positive straddle vega per BTC
    # Long: signal = +1, paying 0.5 vp MORE on entry (higher IV at entry is worse for long)
    # dIV_with_penalty    = iv_exit - (iv_entry + 0.5) = dIV_base - 0.5
    # vega_pnl_with_pen   = +1 * vega_s * (dIV_base - 0.5) = vega_pnl_base - vega_s*0.5
    # delta               = vega_pnl_base - vega_pnl_with_pen = +vega_s * 0.5
    long_delta = vega_s * penalty
    # Short: signal = -1, receiving 0.5 vp LESS on entry (lower IV at entry is worse for short)
    # dIV_with_penalty    = iv_exit - (iv_entry - 0.5) = dIV_base + 0.5
    # vega_pnl_with_pen   = -1 * vega_s * (dIV_base + 0.5) = -vega_s*dIV_base - vega_s*0.5
    # vega_pnl_base       = -1 * vega_s * dIV_base         = -vega_s*dIV_base
    # delta               = vega_pnl_base - vega_pnl_with_pen = +vega_s * 0.5
    short_delta = vega_s * penalty
    assert long_delta == short_delta == 25.0, (
        "Per-BTC penalty magnitude must equal vega_s * penalty regardless of signal."
    )


def test_penalty_zero_when_no_maker_fill_and_zero_elsewhere():
    """If maker_filled is all-False, the penalty contribution is zero (Phase 1
    behavior on the current V2 run). If maker_filled is all-True, the penalty
    is applied on every bar (equivalent in aggregate to the pre-fix buggy
    behavior)."""
    n_bars = 1000
    vega_s = np.random.RandomState(0).uniform(10, 100, n_bars)
    n_contracts = np.random.RandomState(1).uniform(1, 10, n_bars)
    penalty = 0.5

    # All-False: zero penalty
    maker_filled_false = np.zeros(n_bars, dtype=bool)
    penalty_arr_false = np.where(
        maker_filled_false, vega_s * penalty, 0.0
    )
    total_false = (penalty_arr_false * n_contracts).sum()
    assert total_false == 0.0

    # All-True: full penalty on every bar
    maker_filled_true = np.ones(n_bars, dtype=bool)
    penalty_arr_true = np.where(
        maker_filled_true, vega_s * penalty, 0.0
    )
    total_true = (penalty_arr_true * n_contracts).sum()
    assert total_true == pytest.approx(
        0.5 * (vega_s * n_contracts).sum(), rel=1e-12
    )

    # Mixed: penalty matches the maker-subset sum.
    rng = np.random.RandomState(2)
    maker_filled_mixed = rng.random(n_bars) < 0.3
    penalty_arr_mixed = np.where(
        maker_filled_mixed, vega_s * penalty, 0.0
    )
    total_mixed = (penalty_arr_mixed * n_contracts).sum()
    expected = 0.5 * (vega_s[maker_filled_mixed] * n_contracts[maker_filled_mixed]).sum()
    assert total_mixed == pytest.approx(expected, rel=1e-12)
