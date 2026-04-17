#!/usr/bin/env python
"""Threshold sweep for the 30m BTC model — generates comparison table + best dashboard."""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from vol_arb.backtester import BacktestConfig, VolArbBacktester, print_tearsheet
from vol_arb.metrics import build_nc_view

THRESHOLDS = [10, 15, 20, 25, 30, 35, 40]

BASE = dict(
    vol_arb_path="vol_arb/output/BTC_vol_arb_svi_30m.parquet",
    predictions_path="vol_model/output/BTCUSDT_expanding_30m_predictions.parquet",
    initial_equity=200_000.0,
    slippage_model="realistic_taker",
    sizing_model="composite_v2",
    ioc_cap_multiplier=2.0,
    use_profit_filter=True,
    profit_filter_min_ratio=0.0,
    execution_delay_s=1.0,
)

results = []
best_nc_sharpe = -np.inf
best_tl = None
best_thresh = None
best_metrics = None

print(f"{'thresh':>8}  {'trades':>7}  {'comp_sh':>8}  {'nc_sh':>7}  {'nc_pnl':>14}  {'nc_dd':>7}  {'wr':>6}  {'pf':>6}")
print("-" * 85)

for thresh in THRESHOLDS:
    cfg = BacktestConfig.for_timeframe(
        bar_minutes=30,
        **BASE,
        entry_threshold=float(thresh),
        entry_threshold_long=float(thresh + 5),
    )
    bt = VolArbBacktester(cfg)
    metrics, tl = bt.run_backtest()

    active = tl[(tl["signal"] != 0) & (tl["N_contracts"] > 0)]
    n_trades = len(active)

    comp_sharpe = metrics.get("sharpe", 0.0)
    comp_pnl = metrics.get("total_pnl", 0.0)

    has_nc = "net_pnl_nc" in tl.columns
    if has_nc:
        nc_view = build_nc_view(tl)  # renames net_pnl_nc -> net_pnl
        nc_daily = nc_view["net_pnl"].resample("D").sum()
        nc_daily = nc_daily[nc_daily != 0]
        nc_sharpe = float(
            (nc_daily.mean() / nc_daily.std()) * np.sqrt(365)
        ) if len(nc_daily) > 1 and nc_daily.std() > 0 else 0.0
        nc_pnl = nc_view["net_pnl"].sum()
        nc_equity = cfg.initial_equity + nc_view["net_pnl"].cumsum()
        nc_peak = nc_equity.cummax()
        nc_dd = ((nc_equity - nc_peak) / nc_peak * 100).min()
    else:
        nc_sharpe = nc_pnl = nc_dd = 0.0

    row = {
        "threshold": thresh,
        "threshold_long": thresh + 5,
        "trades": n_trades,
        "comp_sharpe": comp_sharpe,
        "comp_pnl": comp_pnl,
        "nc_sharpe": nc_sharpe,
        "nc_pnl": nc_pnl,
        "nc_dd": nc_dd,
        "win_rate": metrics.get("win_rate", 0.0),
        "profit_factor": metrics.get("profit_factor", 0.0),
    }
    results.append(row)

    if nc_sharpe > best_nc_sharpe:
        best_nc_sharpe = nc_sharpe
        best_tl = tl
        best_thresh = thresh
        best_metrics = metrics
        best_cfg = cfg

    print(f"  {thresh:>2d}/{thresh+5:>2d}  {n_trades:>7,d}  {comp_sharpe:>8.1f}  {nc_sharpe:>7.2f}  "
          f"${nc_pnl:>12,.0f}  {nc_dd:>6.2f}%  {row['win_rate']:>5.1f}%  {row['profit_factor']:>5.2f}")

print("\n" + "=" * 85)
print(f"  BEST NC SHARPE: threshold={best_thresh}/{best_thresh+5} -> NC Sharpe {best_nc_sharpe:.2f}")
print("=" * 85)

# Save sweep table
df = pd.DataFrame(results)
df.to_csv("vol_arb/output/30m_threshold_sweep.csv", index=False)
print(f"\nSweep results saved to vol_arb/output/30m_threshold_sweep.csv")

# Save best trade log + print tearsheet
if best_tl is not None:
    best_tl.to_parquet("vol_arb/output/backtest_tradelog_30m_best.parquet")
    print(f"Best trade log saved to vol_arb/output/backtest_tradelog_30m_best.parquet")
    print(f"\n{'='*85}")
    print(f"TEARSHEET — threshold={best_thresh}/{best_thresh+5} (30m bars)")
    print(f"{'='*85}")
    print_tearsheet(best_metrics, best_cfg)
