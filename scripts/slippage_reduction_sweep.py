#!/usr/bin/env python
"""Slippage reduction experiment: test 4 ideas against production baseline.

Anti-overfitting: few configs, motivated by data analysis, report per-year stability.
"""
import warnings, logging
warnings.filterwarnings("ignore")
logging.disable(logging.WARNING)

import numpy as np
from vol_arb.backtester import BacktestConfig, VolArbBacktester

BASE = dict(
    entry_threshold=25,
    slippage_model="realistic_taker",
    sizing_model="composite_v2",
    sizing_liq_cap=1.5,
    ioc_cap_multiplier=3.0,
    report_non_compounding=True,
)

CONFIGS = {
    "baseline":         {},

    # Idea 1: Asymmetric threshold (longs cost more, require more edge)
    "long_thresh_30":   dict(entry_threshold_long=30),
    "long_thresh_35":   dict(entry_threshold_long=35),

    # Idea 2: Hard BTC cap (kill convex tail of sqrt impact)
    "max_15btc":        dict(max_position_btc=15.0),
    "max_10btc":        dict(max_position_btc=10.0),

    # Idea 3: Tighter IOC cap (cut catastrophic fills)
    "ioc_2.5x":         dict(ioc_cap_multiplier=2.5),
    "ioc_2.0x":         dict(ioc_cap_multiplier=2.0),

    # Idea 4: Raise threshold globally (fewer but higher-quality trades)
    "thresh_30":        dict(entry_threshold=30),
    "thresh_35":        dict(entry_threshold=35),

    # Combos (only 2, motivated by analysis)
    "long30_max15":     dict(entry_threshold_long=30, max_position_btc=15.0),
    "long30_ioc2.5":    dict(entry_threshold_long=30, ioc_cap_multiplier=2.5),
}

results = []
for name, overrides in CONFIGS.items():
    params = {**BASE, **overrides}
    cfg = BacktestConfig(**params)
    bt = VolArbBacktester(cfg)
    metrics, tl = bt.run_backtest()

    active = tl[tl["signal"] != 0]
    nc = metrics.get("non_compounding", {})

    # Per-year Sharpe for stability check
    daily = active["net_pnl"].resample("D").sum()
    yearly_sharpe = {}
    for year in sorted(active.index.year.unique()):
        yr = daily[daily.index.year == year]
        yr = yr[yr != 0]
        if len(yr) > 30:
            yearly_sharpe[year] = float((yr.mean() / yr.std()) * np.sqrt(365))

    # Slippage stats
    total_slip = active["slippage"].sum()
    long_slip = active[active["signal"] == 1]["slippage"].sum()

    results.append({
        "name": name,
        "sharpe": metrics["sharpe_ratio"],
        "pnl": metrics["total_pnl"],
        "max_dd": metrics["max_drawdown_pct"],
        "nc_pnl": nc.get("total_pnl", 0),
        "trades": metrics["n_trades"],
        "total_slip": total_slip,
        "long_slip": long_slip,
        "slip_per_trade": total_slip / max(metrics["n_trades"], 1),
        "slip_pct_gross": 100 * total_slip / max(abs(active["gross_pnl"].sum()), 1),
        "yearly_sharpe": yearly_sharpe,
    })

# Print results
print()
print("=" * 120)
print("  SLIPPAGE REDUCTION EXPERIMENT")
print("=" * 120)
print(f"  {'Config':<18s} {'Sharpe':>7s} {'PnL($K)':>10s} {'MaxDD':>7s} {'NC_PnL($K)':>11s} "
      f"{'Trades':>7s} {'Slip($K)':>9s} {'$/trade':>8s} {'Slip%':>6s} {'LongSlip%':>10s}")
print("-" * 120)

baseline = results[0]
for r in results:
    slip_delta = r["total_slip"] - baseline["total_slip"]
    pnl_delta = r["pnl"] - baseline["pnl"]
    print(f"  {r['name']:<18s} {r['sharpe']:>7.2f} {r['pnl']/1000:>10,.0f} {r['max_dd']:>7.1f}% "
          f"{r['nc_pnl']/1000:>10,.0f} {r['trades']:>7,} {r['total_slip']/1000:>9,.0f} "
          f"{r['slip_per_trade']:>8.0f} {r['slip_pct_gross']:>5.1f}% "
          f"{100*r['long_slip']/max(r['total_slip'],1):>9.0f}%")

# Stability: per-year Sharpe
print()
print("=" * 120)
print("  PER-YEAR SHARPE (stability check)")
print("=" * 120)
years = sorted(set().union(*(r["yearly_sharpe"].keys() for r in results)))
header = f"  {'Config':<18s}" + "".join(f" {y:>6d}" for y in years) + "   StdDev"
print(header)
print("-" * len(header))
for r in results:
    vals = [r["yearly_sharpe"].get(y, float("nan")) for y in years]
    valid = [v for v in vals if np.isfinite(v)]
    std = np.std(valid) if len(valid) > 1 else 0
    row = f"  {r['name']:<18s}" + "".join(f" {v:>6.1f}" for v in vals) + f"   {std:>5.1f}"
    print(row)
print("=" * 120)

# Delta table
print()
print("  DELTA vs BASELINE:")
print(f"  {'Config':<18s} {'dSharpe':>8s} {'dPnL($K)':>10s} {'dSlip($K)':>10s} {'dDD':>7s}")
print("-" * 60)
for r in results[1:]:
    ds = r["sharpe"] - baseline["sharpe"]
    dp = (r["pnl"] - baseline["pnl"]) / 1000
    dsl = (r["total_slip"] - baseline["total_slip"]) / 1000
    ddd = r["max_dd"] - baseline["max_dd"]
    marker = " **" if ds > 0 and dsl < 0 else ""
    print(f"  {r['name']:<18s} {ds:>+8.2f} {dp:>+10,.0f} {dsl:>+10,.0f} {ddd:>+7.1f}%{marker}")


if __name__ == "__main__":
    pass
