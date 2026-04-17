#!/usr/bin/env python
"""Slippage filter sweep: compare 4 approaches to slippage-aware trade filtering.

Tests A1 (profit filter), A2 (slippage-adjusted threshold), A3 (profit scaling),
A4 (liquidity-regime filter), and key combinations. Outputs a comparison table
with Sharpe, PnL, slippage, trade count, and per-year stability.

Usage:
    PYTHONPATH=. python scripts/slippage_filter_sweep.py
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.backtester import BacktestConfig, VolArbBacktester

# ── Shared production baseline ──
BASE = dict(
    entry_threshold=25,
    entry_threshold_long=30,
    slippage_model="realistic_taker",
    sizing_model="composite_v2",
    sizing_liq_cap=1.5,
    ioc_cap_multiplier=2.0,
    report_non_compounding=True,
)

CONFIGS = {
    # Reference
    "baseline":         {},

    # ─��� A1: Profit filter (rolling EMA of gross PnL vs estimated costs) ──
    "A1_r00":           dict(use_profit_filter=True, profit_filter_min_ratio=0.0),
    "A1_r10":           dict(use_profit_filter=True, profit_filter_min_ratio=0.1),
    "A1_r20":           dict(use_profit_filter=True, profit_filter_min_ratio=0.2),
    "A1_r30":           dict(use_profit_filter=True, profit_filter_min_ratio=0.3),
    "A1_r-25":          dict(use_profit_filter=True, profit_filter_min_ratio=-0.25),

    # ── A2: Slippage-adjusted threshold (weak for this strategy, included for completeness) ──
    "A2_15_20":         dict(use_slip_adjusted_threshold=True, slip_threshold_base_short=15, slip_threshold_base_long=20),
    "A2_25_30":         dict(use_slip_adjusted_threshold=True, slip_threshold_base_short=25, slip_threshold_base_long=30),

    # ── A3: Profit-ratio position scaling ──
    "A3_f10":           dict(use_profit_scaling=True, profit_scaling_floor=0.1),
    "A3_f25":           dict(use_profit_scaling=True, profit_scaling_floor=0.25),
    "A3_f50":           dict(use_profit_scaling=True, profit_scaling_floor=0.50),

    # ── A4: Liquidity-regime filter ──
    "A4_p95":           dict(use_liquidity_regime_filter=True, liquidity_regime_percentile=95),
    "A4_p90":           dict(use_liquidity_regime_filter=True, liquidity_regime_percentile=90),
    "A4_p85":           dict(use_liquidity_regime_filter=True, liquidity_regime_percentile=85),
    "A4_p80":           dict(use_liquidity_regime_filter=True, liquidity_regime_percentile=80),

    # ── Combinations ──
    "A1r00+A4p90":      dict(use_profit_filter=True, profit_filter_min_ratio=0.0,
                             use_liquidity_regime_filter=True, liquidity_regime_percentile=90),
    "A1r10+A4p90":      dict(use_profit_filter=True, profit_filter_min_ratio=0.1,
                             use_liquidity_regime_filter=True, liquidity_regime_percentile=90),
    "A1r00+A4p85":      dict(use_profit_filter=True, profit_filter_min_ratio=0.0,
                             use_liquidity_regime_filter=True, liquidity_regime_percentile=85),
    "A4p90+A3f25":      dict(use_liquidity_regime_filter=True, liquidity_regime_percentile=90,
                             use_profit_scaling=True, profit_scaling_floor=0.25),
    "A1r00+A4p90+A3f25": dict(use_profit_filter=True, profit_filter_min_ratio=0.0,
                              use_liquidity_regime_filter=True, liquidity_regime_percentile=90,
                              use_profit_scaling=True, profit_scaling_floor=0.25),
    "A1r-25+A4p95":     dict(use_profit_filter=True, profit_filter_min_ratio=-0.25,
                              use_liquidity_regime_filter=True, liquidity_regime_percentile=95),
    "A1r-25+A4p90":     dict(use_profit_filter=True, profit_filter_min_ratio=-0.25,
                              use_liquidity_regime_filter=True, liquidity_regime_percentile=90),
}


def run_config(name: str, overrides: dict) -> dict:
    """Run a single backtest config and collect metrics."""
    cfg_kw = {**BASE, **overrides}
    bt = VolArbBacktester(BacktestConfig(**cfg_kw))
    metrics, tl = bt.run_backtest()

    n_signalled = int((tl["signal"] != 0).sum())
    traded = tl[(tl["signal"] != 0) & (tl["N_contracts"] > 0)]
    n_traded = len(traded)

    # Compound metrics
    daily = traded["net_pnl"].resample("D").sum()
    daily = daily[daily != 0]
    sharpe = float((daily.mean() / daily.std()) * np.sqrt(365)) if len(daily) > 1 else 0
    pnl = traded["net_pnl"].sum()
    gross = traded["gross_pnl"].sum()
    slip = traded["slippage"].sum()
    dd = ((tl["equity"].cummax() - tl["equity"]) / tl["equity"].cummax()).max() * 100

    # Non-compound metrics
    nc_metrics = metrics.get("non_compounding", {})
    nc_sharpe = nc_metrics.get("sharpe_ratio") or 0.0
    nc_pnl = nc_metrics.get("total_pnl") or 0.0

    # Filter diagnostics
    pf = int(tl["profit_filtered"].sum())
    st = int(tl["slip_thresh_filtered"].sum())
    rf = int(tl["regime_filtered"].sum())
    ps = int((tl["profit_scale"] < 1.0).sum())

    # Per-year compound Sharpe
    year_sharpes = {}
    for year in sorted(traded.index.year.unique()):
        yd = traded[traded.index.year == year]["net_pnl"].resample("D").sum()
        yd = yd[yd != 0]
        if len(yd) > 10:
            year_sharpes[year] = float((yd.mean() / yd.std()) * np.sqrt(365))

    # Slippage-to-gross ratio
    slip_pct = 100 * slip / gross if gross > 0 else 0

    return {
        "name": name,
        "sharpe": sharpe,
        "nc_sharpe": nc_sharpe,
        "pnl_M": pnl / 1e6,
        "nc_pnl_K": nc_pnl / 1e3,
        "maxdd_pct": dd,
        "trades": n_traded,
        "filtered": n_signalled - n_traded,
        "slip_M": slip / 1e6,
        "slip_pct": slip_pct,
        "slip_per_trade": slip / max(n_traded, 1),
        "pf_filtered": pf,
        "st_filtered": st,
        "rf_filtered": rf,
        "ps_scaled": ps,
        **{f"sharpe_{y}": s for y, s in year_sharpes.items()},
    }


def main():
    results = []
    t0 = time.time()

    for i, (name, overrides) in enumerate(CONFIGS.items()):
        t1 = time.time()
        print(f"[{i+1}/{len(CONFIGS)}] Running {name}...", end=" ", flush=True)
        row = run_config(name, overrides)
        results.append(row)
        print(f"done ({time.time()-t1:.1f}s) — Sharpe {row['sharpe']:.2f}")

    elapsed = time.time() - t0
    print(f"\nAll {len(CONFIGS)} configs done in {elapsed:.0f}s\n")

    df = pd.DataFrame(results)

    # Delta from baseline
    base = df[df["name"] == "baseline"].iloc[0]
    df["d_sharpe"] = df["sharpe"] - base["sharpe"]
    df["d_nc_sharpe"] = df["nc_sharpe"] - base["nc_sharpe"]
    df["d_pnl_M"] = df["pnl_M"] - base["pnl_M"]
    df["d_slip_M"] = df["slip_M"] - base["slip_M"]

    # Print results table
    print("=" * 140)
    print(f"{'Config':25s} {'Sharpe':>7s} {'dSharpe':>8s} {'NC_Sh':>7s} {'dNC':>6s} "
          f"{'PnL_M':>7s} {'dPnL':>6s} {'DD%':>6s} {'Trades':>7s} {'Filt':>5s} "
          f"{'Slip_M':>7s} {'dSlip':>6s} {'S%Gr':>5s} {'$/trd':>6s}")
    print("-" * 140)
    for _, r in df.iterrows():
        print(f"{r['name']:25s} {r['sharpe']:7.2f} {r['d_sharpe']:+8.2f} "
              f"{r['nc_sharpe']:7.2f} {r['d_nc_sharpe']:+6.2f} "
              f"{r['pnl_M']:7.1f} {r['d_pnl_M']:+6.1f} {r['maxdd_pct']:6.1f} "
              f"{int(r['trades']):7,} {int(r['filtered']):5,} "
              f"{r['slip_M']:7.1f} {r['d_slip_M']:+6.1f} {r['slip_pct']:5.1f} "
              f"{r['slip_per_trade']:6.0f}")
    print("=" * 140)

    # Filter breakdown for non-baseline configs
    print(f"\n{'Config':25s} {'ProfFilt':>8s} {'SlipThr':>8s} {'Regime':>8s} {'Scaled':>8s}")
    print("-" * 60)
    for _, r in df.iterrows():
        if r["name"] == "baseline":
            continue
        print(f"{r['name']:25s} {int(r['pf_filtered']):8,} {int(r['st_filtered']):8,} "
              f"{int(r['rf_filtered']):8,} {int(r['ps_scaled']):8,}")

    # Per-year Sharpe stability
    year_cols = [c for c in df.columns if c.startswith("sharpe_")]
    if year_cols:
        print(f"\n{'Config':25s}", end="")
        for c in sorted(year_cols):
            print(f" {c.replace('sharpe_',''):>6s}", end="")
        print()
        print("-" * (25 + 7 * len(year_cols)))
        for _, r in df.iterrows():
            print(f"{r['name']:25s}", end="")
            for c in sorted(year_cols):
                v = r.get(c, float("nan"))
                print(f" {v:6.1f}" if np.isfinite(v) else "    -  ", end="")
            print()

    # Save
    out_path = Path("vol_arb/output/slippage_filter_sweep.csv")
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
