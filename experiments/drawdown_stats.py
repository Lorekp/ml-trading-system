"""Drawdown statistics for the baseline pipeline."""

import sys, os
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

from signal_engine.config import PortfolioConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats

warnings.filterwarnings("ignore")

# Reuse signal generation from test_original_pipeline
import test_original_pipeline as orig


def main():
    t0 = time.time()
    W = max(1, os.cpu_count() - 1)

    print("Generating signals...", flush=True)
    all_results = {}
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(orig._process_asset, (a, s)): a for a, s in orig.ASSETS.items()}
        for f in as_completed(futures):
            asset, results = f.result()
            all_results[asset] = results

    # Combine + NCO
    sigs = []
    for ar in all_results.values():
        for ms, sd in ar.get(8, []):
            sigs.append(pd.DataFrame(sd))
    combined = pd.concat(sigs).sort_values("timestamp").reset_index(drop=True)
    combined = orig.apply_nco_wf(combined)

    pcfg = orig.get_pcfg()
    sim = simulate_portfolio(combined, pcfg)
    dpnl = compute_daily_pnl_series(sim)
    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"],
                               sim["n_trades"] / len(dpnl))

    print(f"Signals done in {time.time()-t0:.0f}s")
    print(f"Trades: {sim['n_trades']}, Days: {stats['n_days']}")

    # Build equity curve
    equity = pd.Series(
        sim["equity"][~sim["skipped"]],
        index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]),
    )
    peak = equity.expanding().max()
    dd = (equity - peak) / peak

    print(f"\n{'='*80}")
    print("DRAWDOWN STATISTICS")
    print(f"{'='*80}")

    print(f"\n  Sharpe:          {stats['sharpe']:.3f}")
    print(f"  Sortino:         {stats.get('sortino', 0):.3f}")
    print(f"  Calmar:          {stats.get('calmar', 0):.3f}")
    print(f"  Max Drawdown:    {stats['max_dd_pct']:.2f}%")
    print(f"  Final Equity:    ${stats['final_equity']:,.2f}")
    print(f"  Total Return:    {stats['total_return_pct']:.1f}%")

    # DD distribution
    dd_vals = dd[dd < -0.001]
    print(f"\n  --- Drawdown Distribution ---")
    print(f"  Mean DD:         {dd_vals.mean()*100:.2f}%")
    print(f"  Median DD:       {dd_vals.median()*100:.2f}%")
    print(f"  P95 DD:          {dd_vals.quantile(0.05)*100:.2f}%")
    print(f"  P99 DD:          {dd_vals.quantile(0.01)*100:.2f}%")
    print(f"  Max DD:          {dd.min()*100:.2f}%")
    print(f"  Max DD date:     {dd.idxmin()}")

    # Find DD periods
    dd_periods = []
    in_dd = False
    dd_start = dd_trough = dd_trough_date = None
    for ts_i, dd_val in dd.items():
        if dd_val < -0.001 and not in_dd:
            in_dd = True
            dd_start = ts_i
            dd_trough = dd_val
            dd_trough_date = ts_i
        elif dd_val < -0.001 and in_dd:
            if dd_val < dd_trough:
                dd_trough = dd_val
                dd_trough_date = ts_i
        elif dd_val >= -0.001 and in_dd:
            in_dd = False
            dd_periods.append({
                "start": dd_start, "trough_date": dd_trough_date,
                "end": ts_i, "trough_pct": dd_trough * 100,
                "duration": (ts_i - dd_start).days,
                "recovery": (ts_i - dd_trough_date).days,
            })
    if in_dd and dd_start is not None:
        dd_periods.append({
            "start": dd_start, "trough_date": dd_trough_date,
            "end": dd.index[-1], "trough_pct": dd_trough * 100,
            "duration": (dd.index[-1] - dd_start).days,
            "recovery": -1,  # not recovered
        })

    dd_periods.sort(key=lambda x: x["trough_pct"])
    print(f"\n  --- Top 10 Drawdowns ---")
    print(f"  {'#':<3s} {'Start':<12s} {'Trough':<12s} {'End':<12s} {'Depth':>8s} {'Dur':>6s} {'Recov':>6s}")
    print("  " + "-" * 63)
    for i, p in enumerate(dd_periods[:10]):
        rec = f"{p['recovery']}d" if p["recovery"] >= 0 else "OPEN"
        print(f"  {i+1:<3d} {str(p['start'])[:10]:<12s} {str(p['trough_date'])[:10]:<12s} "
              f"{str(p['end'])[:10]:<12s} {p['trough_pct']:>7.2f}% {p['duration']:>5d}d {rec:>6s}")

    # Underwater stats
    durations = [p["duration"] for p in dd_periods]
    if durations:
        print(f"\n  --- Underwater Periods ---")
        print(f"  Count:           {len(dd_periods)}")
        print(f"  Avg duration:    {np.mean(durations):.1f} days")
        print(f"  Median duration: {np.median(durations):.1f} days")
        print(f"  Max duration:    {max(durations)} days")
        print(f"  P90 duration:    {np.percentile(durations, 90):.0f} days")

    # Daily PnL
    print(f"\n  --- Daily PnL ---")
    print(f"  Mean:            ${dpnl.mean():.2f}")
    print(f"  Std:             ${dpnl.std():.2f}")
    print(f"  Skew:            {dpnl.skew():.3f}")
    print(f"  Kurtosis:        {dpnl.kurtosis():.1f}")
    print(f"  VaR 95%:         ${np.percentile(dpnl, 5):.2f}")
    print(f"  CVaR 95%:        ${dpnl[dpnl <= np.percentile(dpnl, 5)].mean():.2f}")
    print(f"  Best day:        ${dpnl.max():.2f}")
    print(f"  Worst day:       ${dpnl.min():.2f}")

    # Streaks
    daily_win = (dpnl > 0).astype(int)
    streaks = []
    cur = 1
    for i in range(1, len(daily_win)):
        if daily_win.iloc[i] == daily_win.iloc[i-1]:
            cur += 1
        else:
            streaks.append((daily_win.iloc[i-1], cur))
            cur = 1
    streaks.append((daily_win.iloc[-1], cur))
    win_s = [s for v, s in streaks if v == 1]
    loss_s = [s for v, s in streaks if v == 0]

    print(f"\n  --- Streaks ---")
    print(f"  Win rate:        {(dpnl>0).mean()*100:.1f}%")
    print(f"  Max win streak:  {max(win_s) if win_s else 0} days")
    print(f"  Max loss streak: {max(loss_s) if loss_s else 0} days")
    print(f"  Avg win streak:  {np.mean(win_s):.1f} days")
    print(f"  Avg loss streak: {np.mean(loss_s):.1f} days")

    # Monthly
    monthly = dpnl.resample("ME").sum()
    pos_m = (monthly > 0).sum()
    print(f"\n  --- Monthly ---")
    print(f"  Positive:        {pos_m}/{len(monthly)} ({pos_m/len(monthly)*100:.0f}%)")
    print(f"  Best:            ${monthly.max():.2f} ({monthly.idxmax().strftime('%Y-%m')})")
    print(f"  Worst:           ${monthly.min():.2f} ({monthly.idxmin().strftime('%Y-%m')})")
    print(f"  Avg:             ${monthly.mean():.2f}")
    print(f"  Monthly SR:      {monthly.mean()/monthly.std()*np.sqrt(12):.3f}")

    # Yearly
    yearly = dpnl.resample("YE").sum()
    print(f"\n  --- Yearly ---")
    for dt, pnl in yearly.items():
        print(f"  {dt.year}: ${pnl:>10.2f}")

    print(f"\n  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
