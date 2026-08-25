"""Partial-window byte-identity parity test (2026-05-11).

Formal contract for the partial-window feature: bar T's trade-log row
MUST be byte-identical whether the backtest's end_dt is at T+holding
or at the full sample's end — for the same start_dt.

This is the determinism guarantee that lets the user run partial
backtests (e.g. "stop at 2026-05-11 12:15:00") and diff the result
against the live engine's trade log without worrying that the
backtester is using future data to influence past bars.

If this test fires, someone has introduced a full-sample-dependent
computation in the backtester pipeline. Likely culprits: a new
sidecar lookup, a full-sample normalization, an .expanding() stat.
"""
from __future__ import annotations

import pandas as pd
import pytest

from vol_arb.backtester import BacktestConfig, VolArbBacktester


# Compare ONLY the deterministic, decision-relevant columns. The trade
# log has some path-dependent equity-running columns (equity, equity_nc,
# r_t, r_t_nc) whose VALUE at bar T depends on which prior bars were in
# the slice — both runs start fresh from initial_equity at their slice
# start. We exclude those from the strict-identity comparison.
DETERMINISTIC_COLUMNS = [
    # Inputs
    "y_pred", "rv_pred_pct", "iv_used", "edge", "adjusted_iv",
    "spread_iv", "vrp_baseline", "jump_fraction",
    # Signal & exit alignment
    "signal", "iv_exit", "price_exit",
    # Per-trade decisions
    "N_contracts_nc",
    # NC PnL components (path-independent: each trade is fixed-notional)
    "vega_pnl_nc", "theta_pnl_nc", "delta_hedge_pnl_nc",
    "option_fees_nc", "slippage_nc", "hedge_costs_nc",
    "net_pnl_nc",
]


def _run(start_dt: str, end_dt: str | None) -> pd.DataFrame:
    """Run V2 backtester with the given window, return the trade log."""
    cfg = (BacktestConfig()
           .with_historical_btc_v2()
           .with_realistic_costs())
    cfg.start_dt = start_dt
    cfg.end_dt = end_dt
    _, tl = VolArbBacktester(cfg).run_backtest()
    return tl


@pytest.mark.slow
def test_partial_window_byte_identity_at_shared_bars():
    """Determinism contract: bar T's row must be byte-identical whether
    the backtest ends at T+15min or at the slice's natural end — for
    the same start_dt.

    Test setup:
    - Both runs share start_dt = 2025-10-01 (gives ~7m of bars,
      includes a full DFL-filter year boundary).
    - Run A ends at 2026-04-15 (full window).
    - Run B ends at 2026-01-15 (~3.5m partial).
    - Shared rows must agree to floating-point precision on all
      deterministic columns.
    """
    start = "2025-10-01"
    end_full    = "2026-04-15"
    end_partial = "2026-01-15"

    tl_full    = _run(start, end_full)
    tl_partial = _run(start, end_partial)

    shared = tl_partial.index.intersection(tl_full.index)
    assert len(shared) > 100, (
        f"Partial run too short to validate parity: only {len(shared)} "
        f"shared rows. Test window may have shrunk."
    )

    cols = [c for c in DETERMINISTIC_COLUMNS
            if c in tl_partial.columns and c in tl_full.columns]
    assert len(cols) >= 10, (
        f"Too few deterministic columns present in trade log: {cols}. "
        f"Schema change?"
    )

    pd.testing.assert_frame_equal(
        tl_partial.loc[shared, cols].sort_index(),
        tl_full.loc[shared, cols].sort_index(),
        check_exact=False,
        rtol=1e-12,
        atol=1e-12,
        check_dtype=False,
    )


@pytest.mark.slow
def test_partial_window_signal_count_consistency():
    """Sanity sub-check: number of signals fired up to end_partial is
    the same in the partial and full runs (the partial run can't
    invent or suppress signals from past bars).
    """
    start = "2025-10-01"
    end_full    = "2026-04-15"
    end_partial = "2026-01-15"

    tl_full    = _run(start, end_full)
    tl_partial = _run(start, end_partial)

    cutoff = pd.Timestamp(end_partial)
    if tl_full.index.tz is not None and cutoff.tz is None:
        cutoff = cutoff.tz_localize(tl_full.index.tz)

    full_in_window = tl_full.loc[tl_full.index <= cutoff]
    n_signals_full    = int((full_in_window["signal"] != 0).sum())
    n_signals_partial = int((tl_partial["signal"] != 0).sum())

    # Allow the partial run to be missing the LAST holding_minutes worth
    # of bars (they have no exit IV available within the partial slice
    # and get dropped by align_exits).
    diff = abs(n_signals_full - n_signals_partial)
    assert diff <= 5, (
        f"Partial-run signal count diverges from full run within the "
        f"shared window: full={n_signals_full}, partial={n_signals_partial}, "
        f"diff={diff}. Should agree modulo the holding_minutes tail drop."
    )
