#!/usr/bin/env python
"""2025+2026 performance breakdown for top configs."""
import sys, os, warnings, numpy as np, pandas as pd
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from experiments.circuit_breaker_v2_experiment import (
    ASSETS, NEEDED_TFS, FULL5, CAP, BASELINE_WR, VIP_TIERS, SPREAD_BPS,
    generate_signals_cached, recompute_returns, apply_vol, _compute_vol_one,
    simulate_with_breaker, get_vip_tier,
)
from signal_engine.portfolio import compute_daily_pnl_series
from signal_engine.config import PortfolioConfig

VARIANTS = [
    ("Baseline",          "none",      {}),
    ("A1: decay50@3",     "streak",    {"threshold": 3, "decay": 0.50}),
    ("E2: ema a=0.85",    "ema",       {"alpha": 0.85, "floor": 0.10}),
    ("F1: pnl10 t=-1%",   "pnl_ratio", {"window": 10, "threshold": -0.01, "floor": 0.10}),
    ("F3: pnl10 t=-0.5%", "pnl_ratio", {"window": 10, "threshold": -0.005, "floor": 0.10}),
]


def run_dpnl(merged, btype, bparams):
    port = PortfolioConfig(
        starting_capital=CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
        max_exposure=0.90, max_per_asset=0.06, max_per_strategy=0.50,
        base_sizing_pct=0.045, max_signal_mult=2.0, sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)
    prepared = merged.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in prepared.columns: prepared[c] = 1.0
        prepared[c] = prepared[c].fillna(1.0)
    prepared["probability"] = prepared["probability"].astype(float)
    prepared = prepared.sort_values("timestamp").reset_index(drop=True)
    sim = simulate_with_breaker(prepared, port, btype, bparams)
    if sim is None: return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None: return None
    if dpnl.index.tz is not None:
        dpnl = dpnl.copy(); dpnl.index = dpnl.index.tz_localize(None)
    mfp = 0.90; active = ~sim["skipped"]; sizes = sim["sizes"]
    ts_arr = pd.to_datetime(prepared["timestamp"].values)
    if ts_arr.tz is not None: ts_arr = ts_arr.tz_localize(None)
    dates_arr = ts_arr.date; daily_vol = {}
    for i in range(len(prepared)):
        if active[i]: daily_vol[dates_arr[i]] = daily_vol.get(dates_arr[i], 0) + abs(sizes[i]) * 2
    vol_s = pd.Series(daily_vol).sort_index(); vol_30d = vol_s.rolling(30, min_periods=1).sum()
    eq_curve = CAP + dpnl.cumsum(); dpnl_adj = dpnl.copy()
    for d in dpnl.index:
        dd = d.date(); eq = eq_curve.get(d, CAP); v30 = vol_30d.get(dd, 0)
        _, mbps, tbps = get_vip_tier(eq, v30)
        rt = mfp * 2 * mbps + (1 - mfp) * 2 * tbps + SPREAD_BPS; extra = rt - 1.0
        nd = sum(1 for i in range(len(prepared)) if active[i] and dates_arr[i] == dd)
        avg = daily_vol.get(dd, 0) / max(nd * 2, 1)
        dpnl_adj[d] -= nd * avg * extra / 10000
    return dpnl_adj


def main():
    vol_series = {n: v for n, v in (ProcessPoolExecutor(max_workers=8).map(
        _compute_vol_one, list(ASSETS.items()))) if v is not None}

    signal_cache = {}
    for label, resample in NEEDED_TFS:
        sigs = generate_signals_cached(label, resample)
        if sigs is not None: signal_cache[label] = sigs

    hold_cache = {}
    for tf, h in FULL5:
        if tf not in signal_cache: continue
        resample = dict(NEEDED_TFS)[tf]
        hold_cache[(tf, h)] = apply_vol(recompute_returns(signal_cache[tf], resample, h), vol_series)
    parts = [hold_cache[(tf, h)] for tf, h in FULL5 if (tf, h) in hold_cache]
    merged = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

    print("=" * 100)
    print("  2025 + 2026 PERFORMANCE - TOP CONFIGS")
    print("=" * 100)

    results = {}
    for name, btype, bparams in VARIANTS:
        dpnl = run_dpnl(merged, btype, bparams)
        if dpnl is not None:
            results[name] = dpnl
            print(f"  {name}: computed", flush=True)

    names = list(results.keys())
    shorts = [n.split(":")[0].strip() for n in names]
    col_w = 14

    for period_label, start, end in [("2025 QUARTERLY", "2025-01-01", "2026-01-01"),
                                      ("2026 YTD", "2026-01-01", "2027-01-01")]:
        print(f"\n{'='*100}")
        print(f"  {period_label}")
        print(f"{'='*100}")

        header = f"\n  {'':>12}"
        for s in shorts:
            header += f" {s:>{col_w}}"
        print(header)
        print("  " + "-" * (12 + (col_w + 1) * len(names)))

        ref = list(results.values())[0]
        d_period = ref[(ref.index >= start) & (ref.index < end)]
        if len(d_period) == 0:
            print("  No data"); continue
        monthly = d_period.resample("MS").sum()

        for m in monthly.index:
            m_start = str(m.date())
            m_end = str((m + pd.offsets.MonthBegin(1)).date())
            row = f"  {m.strftime('%Y-%m'):>12}"
            for name in names:
                dpnl = results[name]
                val = float(dpnl[(dpnl.index >= m_start) & (dpnl.index < m_end)].sum())
                row += f" ${val:>{col_w-1},.0f}"
            print(row)

        if "2025" in period_label:
            print()
            for q, qs, qe in [("Q1", "2025-01-01", "2025-04-01"), ("Q2", "2025-04-01", "2025-07-01"),
                               ("Q3", "2025-07-01", "2025-10-01"), ("Q4", "2025-10-01", "2026-01-01")]:
                row = f"  {q:>12}"
                for name in names:
                    val = float(results[name][(results[name].index >= qs) & (results[name].index < qe)].sum())
                    row += f" ${val:>{col_w-1},.0f}"
                print(row)

        print("  " + "-" * (12 + (col_w + 1) * len(names)))

        row = f"  {'Total':>12}"
        for name in names:
            d = results[name][(results[name].index >= start) & (results[name].index < end)]
            row += f" ${float(d.sum()):>{col_w-1},.0f}"
        print(row)

        row = f"  {'Sharpe':>12}"
        for name in names:
            d = results[name][(results[name].index >= start) & (results[name].index < end)]
            dr = d / CAP
            sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if len(dr) > 20 and dr.std() > 0 else 0
            row += f" {sr:>{col_w}.3f}"
        print(row)

        row = f"  {'MaxDD':>12}"
        for name in names:
            d = results[name][(results[name].index >= start) & (results[name].index < end)]
            dr = d / CAP; eq = (1 + dr).cumprod()
            dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)
            row += f" {dd:>{col_w-1}.2f}%"
        print(row)

        row = f"  {'Win Rate':>12}"
        for name in names:
            d = results[name][(results[name].index >= start) & (results[name].index < end)]
            wr = float((d > 0).sum() / len(d)) if len(d) > 0 else 0
            row += f" {wr:>{col_w-1}.1%}"
        print(row)

        row = f"  {'Best Day':>12}"
        for name in names:
            d = results[name][(results[name].index >= start) & (results[name].index < end)]
            row += f" ${float(d.max()):>{col_w-1},.0f}"
        print(row)

        row = f"  {'Worst Day':>12}"
        for name in names:
            d = results[name][(results[name].index >= start) & (results[name].index < end)]
            row += f" ${float(d.min()):>{col_w-1},.0f}"
        print(row)


if __name__ == "__main__":
    main()
