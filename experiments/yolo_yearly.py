#!/usr/bin/env python
"""Config C year-by-year breakdown with quarterly detail.

Usage:
    PYTHONPATH="src;." python experiments/yolo_yearly.py
"""

import sys, os, time, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from experiments.drawdown_improvements import simulate_with_improvements, CACHE_DIR
from experiments.circuit_breaker_v2_experiment import apply_vol
from experiments.dynamic_asset_v2_experiment import assemble_signals, OBVIOUS_CORE
from signal_engine.portfolio import compute_daily_pnl_series
from signal_engine.config import PortfolioConfig

CAP = 100.0


def main():
    T0 = time.time()

    with open(CACHE_DIR / "v2_prep_cache.pkl", "rb") as f:
        cached = pickle.load(f)
    hold_sigs = cached["hold_sigs"]
    vol_series = cached["vol_series"]
    asset_volumes = cached["asset_volumes"]

    months = pd.period_range("2020-07", "2026-02", freq="M")

    def _regime_sel(ms):
        btc_vol = vol_series.get("BTC")
        n_assets = 30
        if btc_vol is not None:
            avail = btc_vol.loc[:ms - pd.Timedelta(days=1)].dropna()
            if len(avail) > 60:
                current = avail.iloc[-1]
                if current > avail.quantile(0.80): n_assets = 15
                elif current < avail.quantile(0.20): n_assets = 40
        cutoff = ms - pd.DateOffset(months=3)
        scored = []
        for name in hold_sigs:
            confs = []
            for tf in hold_sigs[name]:
                s = hold_sigs[name][tf]
                ts = pd.to_datetime(s["timestamp"])
                if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
                w = s[((ts >= cutoff) & (ts < ms)).values]
                if "confidence" in w.columns and len(w) > 5:
                    confs.extend(w["confidence"].values)
            if not confs: continue
            vol = asset_volumes.get(name)
            if vol is None: continue
            avail_v = vol.loc[:ms - pd.Timedelta(days=1)]
            if len(avail_v) == 0: continue
            scored.append((name, float(np.mean(confs)) * float(avail_v.iloc[-1])))
        scored.sort(key=lambda x: -x[1])
        return [a for a, _ in scored[:n_assets]]

    def _s8_sel(ms):
        active_core = [a for a in OBVIOUS_CORE if a in hold_sigs]
        full = _regime_sel(ms)
        return active_core + [a for a in full if a not in OBVIOUS_CORE]

    merged = assemble_signals(hold_sigs, _s8_sel, months)
    prepared = merged.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in prepared.columns: prepared[c] = 1.0
        prepared[c] = prepared[c].fillna(1.0)
    prepared["probability"] = prepared["probability"].astype(float)
    prepared = apply_vol(prepared.sort_values("timestamp").reset_index(drop=True), vol_series)

    port = PortfolioConfig(
        starting_capital=CAP, max_dd=1.00, delever_start=1.00, slippage=0.0001,
        max_exposure=0.95, max_per_asset=0.25, max_per_strategy=0.50,
        base_sizing_pct=0.200, max_signal_mult=2.0, sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)
    breaker = {"window": 10, "threshold": -0.01, "floor": 0.10,
               "up_threshold": 0.01, "cap": 1.5}
    improvements = {"max_signals_per_bar": 20}

    sim = simulate_with_improvements(prepared, port, breaker, improvements)
    dpnl = compute_daily_pnl_series(sim)
    if dpnl.index.tz is not None:
        dpnl.index = dpnl.index.tz_localize(None)

    # Build compounding equity curve
    dr = dpnl / CAP
    eq = (1 + dr).cumprod()
    eq_dollar = CAP * eq

    print("=" * 95)
    print("  CONFIG C YEAR-BY-YEAR: 20% sizing, no delever, no CT, $100 start")
    print(f"  Period: {dpnl.index[0].date()} to {dpnl.index[-1].date()}")
    print("=" * 95)

    # Year-by-year
    years = sorted(dpnl.index.year.unique())
    yearly_aprs = []

    print(f"\n  {'Year':<6} {'Days':>5} {'SR':>6} {'PnL':>10} {'Return%':>8} "
          f"{'APR%':>7} {'MaxDD%':>7} {'WR%':>5} {'Trades':>7}"
          f"  {'Best Q':>18} {'Worst Q':>18}")
    print("  " + "-" * 110)

    for yr in years:
        yr_mask = dpnl.index.year == yr
        yr_pnl = dpnl[yr_mask]
        yr_dr = yr_pnl / CAP
        n_days = len(yr_pnl)
        if n_days < 5:
            continue

        yr_sr = float(yr_dr.mean() / yr_dr.std() * np.sqrt(365.25)) if yr_dr.std() > 0 else 0
        yr_total = float(yr_pnl.sum())
        yr_ret = yr_total / CAP * 100
        yr_apr = yr_ret * (365.25 / n_days)  # annualize

        yr_eq = (1 + yr_dr).cumprod()
        yr_dd = (yr_eq.cummax() - yr_eq) / yr_eq.cummax()
        yr_maxdd = float(yr_dd.max() * 100)
        yr_wr = float((yr_pnl > 0).mean() * 100)

        # Count trades in this year from sim
        ts_col = pd.to_datetime(prepared["timestamp"])
        if ts_col.dt.tz is not None:
            ts_col = ts_col.dt.tz_localize(None)
        yr_sigs = prepared[ts_col.dt.year == yr]
        yr_trades = len(yr_sigs[yr_sigs.get("executed", pd.Series(True, index=yr_sigs.index)) == True]) if "executed" in yr_sigs.columns else len(yr_sigs)

        yearly_aprs.append(yr_apr)

        # Quarterly breakdown
        quarters = {}
        for q in [1, 2, 3, 4]:
            q_mask = yr_mask & (dpnl.index.quarter == q)
            q_pnl = dpnl[q_mask]
            if len(q_pnl) > 0:
                q_ret = float(q_pnl.sum()) / CAP * 100
                quarters[f"Q{q}"] = q_ret

        if quarters:
            best_q = max(quarters, key=quarters.get)
            worst_q = min(quarters, key=quarters.get)
            best_str = f"{best_q} {quarters[best_q]:>+.1f}%"
            worst_str = f"{worst_q} {quarters[worst_q]:>+.1f}%"
        else:
            best_str = worst_str = "N/A"

        print(f"  {yr:<6} {n_days:>5} {yr_sr:>6.2f} ${yr_total:>8.2f} {yr_ret:>7.1f}% "
              f"{yr_apr:>6.0f}% {yr_maxdd:>6.1f}% {yr_wr:>4.0f}% {yr_trades:>7,}"
              f"  {best_str:>18} {worst_str:>18}")

    # Summary stats
    print("  " + "-" * 110)
    total_pnl = float(dpnl.sum())
    total_days = len(dpnl)
    total_ret = total_pnl / CAP * 100
    overall_apr = total_ret * (365.25 / total_days)
    overall_sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    overall_wr = float((dpnl > 0).mean() * 100)
    overall_dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)

    print(f"  {'TOTAL':<6} {total_days:>5} {overall_sr:>6.2f} ${total_pnl:>8.2f} {total_ret:>7.1f}% "
          f"{overall_apr:>6.0f}% {overall_dd:>6.1f}% {overall_wr:>4.0f}% {sim['n_trades']:>7,}")

    mean_apr = np.mean(yearly_aprs)
    median_apr = np.median(yearly_aprs)
    min_apr = np.min(yearly_aprs)
    max_apr = np.max(yearly_aprs)

    print(f"\n  APR SUMMARY")
    print(f"  " + "-" * 40)
    print(f"  Mean APR:     {mean_apr:.0f}%")
    print(f"  Median APR:   {median_apr:.0f}%")
    print(f"  Min APR:      {min_apr:.0f}% ({years[np.argmin(yearly_aprs)]})")
    print(f"  Max APR:      {max_apr:.0f}% ({years[np.argmax(yearly_aprs)]})")
    print(f"  Std APR:      {np.std(yearly_aprs):.0f}%")

    # Full quarterly table
    print(f"\n  QUARTERLY RETURNS (%)")
    print(f"  " + "-" * 55)
    print(f"  {'Year':<6} {'Q1':>10} {'Q2':>10} {'Q3':>10} {'Q4':>10}")
    print(f"  " + "-" * 55)
    for yr in years:
        yr_mask = dpnl.index.year == yr
        cols = []
        for q in [1, 2, 3, 4]:
            q_mask = yr_mask & (dpnl.index.quarter == q)
            q_pnl = dpnl[q_mask]
            if len(q_pnl) > 0:
                q_ret = float(q_pnl.sum()) / CAP * 100
                cols.append(f"{q_ret:>+9.1f}%")
            else:
                cols.append(f"{'--':>10}")
        print(f"  {yr:<6} {''.join(cols)}")

    # Negative quarters
    neg_quarters = []
    for yr in years:
        yr_mask = dpnl.index.year == yr
        for q in [1, 2, 3, 4]:
            q_mask = yr_mask & (dpnl.index.quarter == q)
            q_pnl = dpnl[q_mask]
            if len(q_pnl) > 0:
                q_ret = float(q_pnl.sum()) / CAP * 100
                if q_ret < 0:
                    neg_quarters.append((yr, q, q_ret))

    print(f"\n  NEGATIVE QUARTERS: {len(neg_quarters)} total")
    for yr, q, ret in neg_quarters:
        print(f"    {yr} Q{q}: {ret:>+.2f}%")

    # Compounding equity at year-end
    print(f"\n  COMPOUNDING EQUITY (year-end, $100 start)")
    print(f"  " + "-" * 40)
    for yr in years:
        yr_mask = dpnl.index.year == yr
        yr_last = eq_dollar[yr_mask].iloc[-1] if yr_mask.any() else None
        if yr_last is not None:
            print(f"  End {yr}:  ${yr_last:>10.2f}")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
