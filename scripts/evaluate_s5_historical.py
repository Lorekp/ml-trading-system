"""Historical S5 evaluation for the Binance directional system.

This runner replays the 32-asset research system with the corrected inverse-vol
sizing path and the configuration values used by the former paper experiment.
It is a historical configuration evaluation, not a live or deployment
certification.

The earlier paper run lost approximately $15k, and a counterfactual replay
attributed much of that loss to a fail-open sizing bug. This script evaluates
the corrected sizing path under the same historical portfolio settings so the
research can be reassessed on a consistent basis.

It drives ``experiments/circuit_breaker_v2_experiment.py`` at:

    base_sizing_pct = 0.30
    max_per_asset   = 0.25
    max_exposure    = 0.90
    leverage        = 4      (enters only via the exposure cap; PnL math
                              is on notional, unchanged)

``run_variant`` applies the historical VIP-tier cost assumptions on top of the
1 bps base slippage assumption. Those assumptions are part of the replay and
have not been independently reconciled to an execution ledger.

The clip floor is swept over {0.05, 0.10, 0.15, 0.25}. The 0.10 reference is
the value from the historical configuration; the sweep is a sensitivity check,
not a procedure for selecting a new setting.

Independent PnL realism is not evaluated because this runner has no separate
execution/fill PnL series. Sizing-sanity and selection-bias diagnostics are
still run because they compare distinct quantities available in the replay.

Usage:
    PYTHONPATH=src python scripts/evaluate_s5_historical.py [--smoke]
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
from validation.gates import selection_bias_gate, sizing_sanity_gate  # noqa: E402

FLOORS = [0.05, 0.10, 0.15, 0.25]
TARGET_VOL, CAP_MULT = 0.025, 2.0
REFERENCE_FLOOR = 0.10


def _parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="Historical S5 evaluation (Binance)")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Quick smoke run (BTC/ETH/SOL only)",
    )
    return parser.parse_args()


def historical_portfolio_config() -> PortfolioConfig:
    """Return the portfolio settings used by the historical S5 replay."""
    return PortfolioConfig(
        starting_capital=cb.CAP,
        max_dd=0.10,
        delever_start=0.05,
        slippage=0.0001,
        max_exposure=0.90,
        max_per_asset=0.25,
        max_per_strategy=0.50,
        base_sizing_pct=0.30,
        max_signal_mult=2.0,
        sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True,
        regime_exposure_floor=0.50,
    )


def recompute_all(signal_cache):
    """Recompute Full5 forward returns once, independent of the floor sweep."""
    recomputed = {}
    for tf, horizon in cb.FULL5:
        if tf not in signal_cache:
            continue
        resample = dict(cb.NEEDED_TFS)[tf]
        recomputed[tf] = cb.recompute_returns(signal_cache[tf], resample, horizon)
    return recomputed


def build_merged(recomputed, vol_series, floor):
    """Build the Full5 signal set with the inverse-vol overlay at ``floor``."""
    parts = [
        cb.apply_vol(signals, vol_series, TARGET_VOL, floor, CAP_MULT)
        for signals in recomputed.values()
    ]
    return (
        pd.concat(parts, ignore_index=True)
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def per_year(sim) -> pd.Series:
    """Return per-calendar-year PnL from a replay result."""
    timestamps = pd.to_datetime(sim["timestamps"])
    frame = pd.DataFrame(
        {"year": timestamps.year, "pnl": sim["pnl"], "skipped": sim["skipped"]}
    )
    return frame[~frame["skipped"]].groupby("year")["pnl"].sum()


def main() -> None:
    args = _parse_args()
    if args.smoke:
        cb.SMOKE = True
        smoke_assets = list(cb.ASSETS.keys())[:3]
        cb.ASSETS = {asset: cb.ASSETS[asset] for asset in smoke_assets}
        print("  --smoke: BTC/ETH/SOL only (3 assets)")

    started_at = time.time()
    print("=" * 78)
    print("  HISTORICAL S5 EVALUATION — Binance directional system")
    print(
        f"  base_sizing_pct=0.30  max_per_asset=0.25  "
        f"universe={len(cb.ASSETS)} assets"
    )
    print("  NON-COMPOUND (fixed-capital sizing) + historical cost assumptions")
    print("=" * 78)

    print("\n  Generating signals (cached on first run)...", flush=True)
    signal_cache = {}
    for label, resample in cb.NEEDED_TFS:
        signal_started_at = time.time()
        signals = cb.generate_signals_cached(label, resample)
        if signals is not None:
            signal_cache[label] = signals
            print(
                f"    {label}: {len(signals):,} signals "
                f"({time.time() - signal_started_at:.0f}s)",
                flush=True,
            )

    from concurrent.futures import ProcessPoolExecutor

    vol_series = {
        name: values
        for name, values in ProcessPoolExecutor(max_workers=8).map(
            cb._compute_vol_one, list(cb.ASSETS.items())
        )
        if values is not None
    }
    print(f"  vol series: {len(vol_series)} assets", flush=True)

    recompute_started_at = time.time()
    recomputed = recompute_all(signal_cache)
    print(
        "  recomputed Full5 forward returns "
        f"({time.time() - recompute_started_at:.0f}s)",
        flush=True,
    )

    portfolio = historical_portfolio_config()

    print(
        f"\n  {'Floor':>6} {'Sharpe':>8} {'Sortino':>8} {'PnL':>13} "
        f"{'MaxDD':>7} {'Calmar':>7} {'PF':>6} {'Trades':>8}"
    )
    print("  " + "-" * 70)
    simulations = {}
    for floor in FLOORS:
        merged = build_merged(recomputed, vol_series, floor)
        result = cb.run_variant(
            merged, "none", {}, port=portfolio, compound=False
        )
        simulation = cb.simulate_with_breaker(
            merged, portfolio, "none", {}, compound=False
        )
        simulations[floor] = (result, simulation, merged)
        if result is None:
            print(f"  {floor:>6.2f}  FAIL")
            continue
        print(
            f"  {floor:>6.2f} {result['sr']:>8.3f} "
            f"{result['sortino']:>8.3f} ${result['pnl']:>11,.0f} "
            f"{result['max_dd']:>6.1f}% {result['calmar']:>7.2f} "
            f"{result['pf']:>6.2f} {result['n_trades']:>8,}"
        )

    _, simulation, merged = simulations[REFERENCE_FLOOR]
    print(f"\n  Per-year PnL @ historical floor={REFERENCE_FLOOR}:")
    yearly_pnl = per_year(simulation)
    for year, pnl in yearly_pnl.items():
        print(f"    {year}: ${pnl:>13,.0f}")
    positive_years = int((yearly_pnl > 0).sum())
    print(f"  positive years: {positive_years}/{len(yearly_pnl)}")

    print(f"\n  Historical diagnostics @ floor={REFERENCE_FLOOR}:")
    print(
        "    [N/A] pnl_realism: independent PnL realism NOT EVALUATED; "
        "no separate execution/fill PnL series is available"
    )
    active = ~simulation["skipped"]
    trade_pnl = np.asarray(simulation["pnl"])[active]
    trade_size = np.asarray(simulation["sizes"])[active]

    per_unit = trade_pnl / trade_size
    sizing_diagnostic = sizing_sanity_gate(per_unit, trade_size)

    universe_edge = merged["direction"].values * merged["fwd_return"].values
    selection_diagnostic = selection_bias_gate(per_unit, universe_edge)
    for diagnostic in (sizing_diagnostic, selection_diagnostic):
        print(
            f"    [{'PASS' if diagnostic.passed else 'FAIL'}] "
            f"{diagnostic.name}: {diagnostic.message}"
        )

    print(f"\n  Runtime: {time.time() - started_at:.0f}s")
    print("=" * 78)


if __name__ == "__main__":
    main()
