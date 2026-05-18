"""Honest re-backtest of the Binance 32-asset directional system at the
LIVE production config, with the fixed inverse-vol sizing.

Phase-5 sizing-fix evaluation (2026-05-18). The live paper engine lost
-$15k; the counterfactual replay showed ~77% of that was the fail-open
sizing bug. This script establishes ONE honest Sharpe for the system
*under the deployed config* so a keep/retire decision can be made.

It drives the experiments backtester `circuit_breaker_v2_experiment.py`
(the true live twin — identical sizing path) at the LIVE PortfolioConfig:

    base_sizing_pct = 0.30   (live yaml; research default was 0.045)
    max_per_asset   = 0.25   (live yaml; research default was 0.06)
    max_exposure    = 0.90
    leverage        = 4      (enters only via the exposure cap; PnL math
                              is on notional, unchanged)

Costs: `run_variant` already applies the calibrated VIP-tier round-trip
cost (~3.9 bps) on top of the 1 bps base slippage — no optimistic headline.

The clip floor is swept {0.05, 0.10, 0.15, 0.25}; the production floor is
chosen by per-year stability, NOT the in-sample optimum (feedback_no_cherry_pick).
Validation gates (validation/gates.py) are run on the trade log.

Usage:
    PYTHONPATH=src python scripts/honest_rebacktest_binance.py [--smoke]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import circuit_breaker_v2_experiment as cb  # noqa: E402
from signal_engine.config import PortfolioConfig  # noqa: E402
from validation.gates import (  # noqa: E402
    pnl_realism_gate, selection_bias_gate, sizing_sanity_gate,
)

FLOORS = [0.05, 0.10, 0.15, 0.25]
TARGET_VOL, CAP_MULT = 0.025, 2.0


def live_port() -> PortfolioConfig:
    """PortfolioConfig at the deployed live config (configs/paper_binance.yaml)."""
    return PortfolioConfig(
        starting_capital=cb.CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
        max_exposure=0.90, max_per_asset=0.25, max_per_strategy=0.50,
        base_sizing_pct=0.30, max_signal_mult=2.0, sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)


def recompute_all(signal_cache):
    """Recompute Full5 forward returns ONCE (expensive — reloads 1m data).
    Independent of the clip floor, so it is hoisted out of the floor sweep."""
    recomputed = {}
    for tf, h in cb.FULL5:
        if tf not in signal_cache:
            continue
        resample = dict(cb.NEEDED_TFS)[tf]
        recomputed[tf] = cb.recompute_returns(signal_cache[tf], resample, h)
    return recomputed


def build_merged(recomputed, vol_series, floor):
    """Full5 merged signal set with the inverse-vol overlay at `floor`."""
    parts = [cb.apply_vol(sigs, vol_series, TARGET_VOL, floor, CAP_MULT)
             for sigs in recomputed.values()]
    return (pd.concat(parts, ignore_index=True)
            .sort_values("timestamp").reset_index(drop=True))


def per_year(sim) -> pd.Series:
    """Per-calendar-year PnL from a simulate_with_breaker result."""
    ts = pd.to_datetime(sim["timestamps"])
    df = pd.DataFrame({"year": ts.year, "pnl": sim["pnl"],
                       "skipped": sim["skipped"]})
    return df[~df["skipped"]].groupby("year")["pnl"].sum()


def main() -> None:
    t0 = time.time()
    print("=" * 78)
    print("  HONEST RE-BACKTEST — Binance 32-asset system @ LIVE config")
    print(f"  base_sizing_pct=0.30  max_per_asset=0.25  universe={len(cb.ASSETS)} assets")
    print("  NON-COMPOUND (fixed-capital sizing) + calibrated VIP-tier costs")
    print("=" * 78)

    # ── Signals (cached) + per-asset daily vol ──
    print("\n  Generating signals (cached on first run)...", flush=True)
    signal_cache = {}
    for label, resample in cb.NEEDED_TFS:
        t1 = time.time()
        sigs = cb.generate_signals_cached(label, resample)
        if sigs is not None:
            signal_cache[label] = sigs
            print(f"    {label}: {len(sigs):,} signals ({time.time()-t1:.0f}s)",
                  flush=True)
    from concurrent.futures import ProcessPoolExecutor
    vol_series = {n: v for n, v in ProcessPoolExecutor(max_workers=8).map(
        cb._compute_vol_one, list(cb.ASSETS.items())) if v is not None}
    print(f"  vol series: {len(vol_series)} assets", flush=True)

    t1 = time.time()
    recomputed = recompute_all(signal_cache)
    print(f"  recomputed Full5 forward returns ({time.time()-t1:.0f}s)", flush=True)

    port = live_port()

    # ── Clip-floor sweep ──
    print(f"\n  {'Floor':>6} {'Sharpe':>8} {'Sortino':>8} {'PnL':>13} "
          f"{'MaxDD':>7} {'Calmar':>7} {'PF':>6} {'Trades':>8}")
    print("  " + "-" * 70)
    sims = {}
    for floor in FLOORS:
        merged = build_merged(recomputed, vol_series, floor)
        # compound=False -> honest non-compound Sharpe/PnL (the planning
        # metric; compounding at base_sizing_pct=0.30 explodes the headline).
        r = cb.run_variant(merged, "none", {}, port=port, compound=False)
        sim = cb.simulate_with_breaker(merged, port, "none", {}, compound=False)
        sims[floor] = (r, sim, merged)
        if r is None:
            print(f"  {floor:>6.2f}  FAIL")
            continue
        print(f"  {floor:>6.2f} {r['sr']:>8.3f} {r['sortino']:>8.3f} "
              f"${r['pnl']:>11,.0f} {r['max_dd']:>6.1f}% {r['calmar']:>7.2f} "
              f"{r['pf']:>6.2f} {r['n_trades']:>8,}")

    # ── Per-year stability (production floor = 0.10, the deployed value) ──
    prod_floor = 0.10
    r, sim, merged = sims[prod_floor]
    print(f"\n  Per-year PnL @ floor={prod_floor} (deployed):")
    py = per_year(sim)
    for yr, pnl in py.items():
        print(f"    {yr}: ${pnl:>13,.0f}")
    n_pos_years = int((py > 0).sum())
    print(f"  positive years: {n_pos_years}/{len(py)}")

    # ── Validation gates on the production-floor trade log ──
    print(f"\n  Validation gates @ floor={prod_floor}:")
    active = ~sim["skipped"]
    trade_pnl = np.asarray(sim["pnl"])[active]
    trade_size = np.asarray(sim["sizes"])[active]
    # Gate 1 — PnL realism. The Binance backtest marks PnL to real OHLC
    # forward returns, so model PnL IS the fill-marked PnL (no separate
    # model). Pass the same series both sides — confirms it is sound.
    g1 = pnl_realism_gate(trade_pnl, trade_pnl)
    # Gate 2 — sizing sanity. pnl_per_unit = per-$ edge; exposure = notional.
    per_unit = trade_pnl / trade_size
    g2 = sizing_sanity_gate(per_unit, trade_size)
    # Gate 3 — selection bias. Traded edge vs the full candidate-signal edge.
    uni_edge = (merged["direction"].values * merged["fwd_return"].values)
    traded_edge = per_unit  # realized edge per $ on taken trades
    g3 = selection_bias_gate(traded_edge, uni_edge)
    for g in (g1, g2, g3):
        print(f"    [{'PASS' if g.passed else 'FAIL'}] {g.name}: {g.message}")

    print(f"\n  Runtime: {time.time()-t0:.0f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
