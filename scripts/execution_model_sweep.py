#!/usr/bin/env python
"""A/B sweep: execution model improvements vs baseline.

Tests all combinations of slippage model, sizing strategy, and IOC cap
to quantify the impact of each intervention.

Usage:
    PYTHONPATH=. python scripts/execution_model_sweep.py
"""

from __future__ import annotations

import logging
import time

import numpy as np
import pandas as pd

from vol_arb.backtester import BacktestConfig, VolArbBacktester

logging.basicConfig(level=logging.WARNING)

CONFIGS = [
    # Reference: grid + baseline
    ("grid_baseline", dict(slippage_model="grid", sizing_model="baseline", use_sor=False)),
    # Current realistic: pre-fix
    # (can't test pre-fix since code is already patched — use ioc_cap=6 to approximate)
    ("real_baseline_ioc6", dict(slippage_model="realistic_taker", sizing_model="baseline", use_sor=False, ioc_cap_multiplier=6.0)),
    # Fixes only: sqrt floor + IOC cap 4x
    ("real_baseline_fixed", dict(slippage_model="realistic_taker", sizing_model="baseline", use_sor=False, ioc_cap_multiplier=4.0)),
    # Liquidity-conditioned
    ("real_liq", dict(slippage_model="realistic_taker", sizing_model="liq", use_sor=False, ioc_cap_multiplier=4.0)),
    # Cost-edge
    ("real_cost_edge", dict(slippage_model="realistic_taker", sizing_model="cost_edge", use_sor=False, ioc_cap_multiplier=4.0)),
    # Composite v2
    ("real_comp_v2", dict(slippage_model="realistic_taker", sizing_model="composite_v2", use_sor=False, ioc_cap_multiplier=4.0)),
    # Composite v2 at 10% margin
    ("real_comp_v2_10", dict(slippage_model="realistic_taker", sizing_model="composite_v2", use_sor=False, ioc_cap_multiplier=4.0, target_margin_utilization=0.10)),
    # IOC cap sweep
    ("real_ioc3", dict(slippage_model="realistic_taker", sizing_model="baseline", use_sor=False, ioc_cap_multiplier=3.0)),
    ("real_ioc5", dict(slippage_model="realistic_taker", sizing_model="baseline", use_sor=False, ioc_cap_multiplier=5.0)),
    # Liq cap sweep
    ("real_liq_cap15", dict(slippage_model="realistic_taker", sizing_model="liq", use_sor=False, ioc_cap_multiplier=4.0, sizing_liq_cap=1.5)),
    ("real_liq_cap30", dict(slippage_model="realistic_taker", sizing_model="liq", use_sor=False, ioc_cap_multiplier=4.0, sizing_liq_cap=3.0)),
]


def main() -> None:
    results = []
    t0 = time.time()

    for name, kwargs in CONFIGS:
        cfg = BacktestConfig(
            initial_equity=100_000,
            entry_threshold=25,
            report_non_compounding=True,
            **kwargs,
        )
        bt = VolArbBacktester(cfg)
        metrics, df = bt.run_backtest()

        # Extract compound + non-compound metrics
        nc = metrics.get("non_compounding", {})
        max_dd_pct = metrics.get("max_drawdown_pct", np.nan)
        sharpe = metrics.get("sharpe_ratio", np.nan)
        calmar = abs(sharpe / max_dd_pct * 100) if max_dd_pct and max_dd_pct != 0 else np.nan
        row = {
            "name": name,
            "sharpe": sharpe,
            "sortino": metrics.get("sortino_ratio", np.nan),
            "pnl": metrics.get("total_pnl", np.nan),
            "max_dd_pct": max_dd_pct,
            "calmar": calmar,
            "n_trades": metrics.get("n_trades", 0),
            "avg_trade_pnl": metrics.get("avg_trade_pnl", np.nan),
            "nc_sharpe": nc.get("sharpe_ratio", np.nan),
            "nc_pnl": nc.get("total_pnl", np.nan),
            "nc_max_dd_pct": nc.get("max_drawdown_pct", np.nan),
        }

        # Cost statistics
        slip = df["slippage"].values
        trades = df["signal"].values != 0
        if trades.any():
            trade_slips = slip[trades]
            row["slip_mean"] = float(np.mean(trade_slips))
            row["slip_std"] = float(np.std(trade_slips))
            row["slip_total"] = float(np.sum(trade_slips))
        else:
            row["slip_mean"] = row["slip_std"] = row["slip_total"] = 0.0

        results.append(row)
        print(f"  {name:25s}  Sharpe={row['sharpe']:6.2f}  PnL=${row['pnl']:>10,.0f}  "
              f"DD={row['max_dd_pct']:>6.1f}%  NC_Sharpe={row['nc_sharpe']:6.2f}  "
              f"trades={row['n_trades']:,}  slip_std=${row['slip_std']:.1f}")

    elapsed = time.time() - t0
    print(f"\nTotal sweep time: {elapsed:.0f}s\n")

    # Summary table
    print("=" * 120)
    print("  EXECUTION MODEL SWEEP RESULTS")
    print("=" * 120)
    print(f"  {'Name':<25s} {'Sharpe':>7} {'PnL':>12} {'MaxDD%':>7} {'Calmar':>7} "
          f"{'NC_Sharpe':>10} {'NC_PnL':>12} {'Trades':>7} {'Slip_Std':>9}")
    print("-" * 120)
    for r in results:
        print(f"  {r['name']:<25s} {r['sharpe']:>7.2f} ${r['pnl']:>10,.0f} "
              f"{r['max_dd_pct']:>7.1f}% {r['calmar']:>7.1f} "
              f"{r['nc_sharpe']:>10.2f} ${r['nc_pnl']:>10,.0f} "
              f"{r['n_trades']:>7,} ${r['slip_std']:>8.1f}")
    print("=" * 120)

    # Save
    out = pd.DataFrame(results)
    out_path = "vol_arb/output/execution_model_sweep.csv"
    out.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
