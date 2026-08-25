#!/usr/bin/env python
"""Config C vs C+: independent $10K sim per year, YOLO vs with safety rails.

Usage:
    PYTHONPATH="src;." python experiments/yolo_per_year.py
"""

import sys, os, time, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import OrderedDict

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from experiments.drawdown_improvements import simulate_with_improvements, CACHE_DIR
from experiments.circuit_breaker_v2_experiment import apply_vol
from experiments.dynamic_asset_v2_experiment import assemble_signals, OBVIOUS_CORE
from signal_engine.portfolio import compute_daily_pnl_series
from signal_engine.config import PortfolioConfig

CAP = 200_000.0
USE_REALISTIC_SLIPPAGE = True


def run_year(yr_signals, port, breaker, improvements, slippage_model=None, asset_volumes=None):
    sim = simulate_with_improvements(yr_signals, port, breaker, improvements,
                                     slippage_model=slippage_model, asset_volumes=asset_volumes)
    if sim is None or sim["n_trades"] < 5:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5:
        return None
    if dpnl.index.tz is not None:
        dpnl.index = dpnl.index.tz_localize(None)

    pnl = float(dpnl.sum())
    n_days = len(dpnl)

    # Proper equity-relative returns
    running_eq = CAP
    daily_rets = []
    eq_series = []
    for dp in dpnl.values:
        r = dp / running_eq if running_eq > 0 else 0.0
        daily_rets.append(r)
        running_eq = max(running_eq + dp, 0.0)
        eq_series.append(running_eq)
    daily_rets = np.array(daily_rets)
    eq_arr = np.array(eq_series)

    sr = float(daily_rets.mean() / daily_rets.std() * np.sqrt(365.25)) if daily_rets.std() > 0 else 0
    peak = np.maximum.accumulate(eq_arr)
    dd_arr = np.where(peak > 0, (peak - eq_arr) / peak, 0)
    max_dd = float(dd_arr.max() * 100)
    worst_day_pct = float(daily_rets.min() * 100)

    # Quarters
    quarters = {}
    for q in [1, 2, 3, 4]:
        q_pnl = dpnl[dpnl.index.quarter == q]
        if len(q_pnl) > 0:
            quarters[q] = float(q_pnl.sum())

    monthly = dpnl.resample("MS").sum()
    neg_months = int((monthly < 0).sum())

    return {
        "days": n_days, "sr": sr, "pnl": pnl,
        "ret_pct": pnl / CAP * 100, "max_dd": max_dd,
        "worst_day_pct": worst_day_pct,
        "win_rate": float((dpnl > 0).mean() * 100),
        "trades": sim["n_trades"], "quarters": quarters,
        "neg_months": neg_months, "tot_months": len(monthly),
    }


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
    prepared_all = merged.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in prepared_all.columns: prepared_all[c] = 1.0
        prepared_all[c] = prepared_all[c].fillna(1.0)
    prepared_all["probability"] = prepared_all["probability"].astype(float)
    prepared_all = apply_vol(prepared_all.sort_values("timestamp").reset_index(drop=True), vol_series)

    ts_all = pd.to_datetime(prepared_all["timestamp"])
    if ts_all.dt.tz is not None:
        ts_all = ts_all.dt.tz_localize(None)
    prepared_all["_ts"] = ts_all
    years = sorted(ts_all.dt.year.unique())

    breaker = {"window": 10, "threshold": -0.01, "floor": 0.10,
               "up_threshold": 0.01, "cap": 1.5}

    configs = OrderedDict()
    # YOLO: delever at 20% DD, no CT
    configs["YOLO"] = (
        PortfolioConfig(
            starting_capital=CAP, max_dd=0.30, delever_start=0.20, slippage=0.0001,
            max_exposure=0.95, max_per_asset=0.25, max_per_strategy=0.50,
            base_sizing_pct=0.200, max_signal_mult=2.0, sizing_mode="probability_edge",
            use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        {"max_signals_per_bar": 20})

    # With safety: delever at 10% DD, halt at 25%, CT90
    configs["SAFE"] = (
        PortfolioConfig(
            starting_capital=CAP, max_dd=0.25, delever_start=0.10, slippage=0.0001,
            max_exposure=0.95, max_per_asset=0.25, max_per_strategy=0.50,
            base_sizing_pct=0.200, max_signal_mult=2.0, sizing_mode="probability_edge",
            use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        {"max_signals_per_bar": 20,
         "correlation_throttle": 0.90, "correlation_direction": "both", "correlation_scale": 0.5})

    # Slippage model
    slippage_model = None
    _asset_volumes = None
    slip_label = "flat 0.01bps"
    if USE_REALISTIC_SLIPPAGE:
        from signal_engine.slippage import load_per_symbol_calibration
        from signal_engine.execution import get_vip_tier
        per_symbol_cal = load_per_symbol_calibration()
        vip_name, maker_bps, taker_bps = get_vip_tier(CAP)
        slippage_model = {
            "calibration": per_symbol_cal,
            "vip_tier": (vip_name, maker_bps, taker_bps),
            "max_participation_rate": 0.02,
            "guard_pct": 0.50,
        }
        _asset_volumes = asset_volumes
        slip_label = f"realistic ({vip_name}, guard 50%)"

    cap_label = f"${CAP/1000:,.0f}K" if CAP < 1_000_000 else f"${CAP/1e6:,.0f}M"
    print("=" * 100)
    print(f"  CONFIG C: {cap_label} per year | 20% sizing | YOLO vs SAFE | slippage: {slip_label}")
    print("=" * 100)

    # Run both configs per year
    all_results = {}
    for cfg_name, (port, imp) in configs.items():
        all_results[cfg_name] = {}
        for yr in years:
            yr_signals = prepared_all[prepared_all["_ts"].dt.year == yr].drop(columns=["_ts"]).reset_index(drop=True)
            if len(yr_signals) < 10:
                continue
            r = run_year(yr_signals, port, breaker, imp,
                         slippage_model=slippage_model, asset_volumes=_asset_volumes)
            if r:
                all_results[cfg_name][yr] = r

    # Side-by-side table
    print(f"\n  {'':>6} {'-------- YOLO (no safety) --------':>48} {'| -------- SAFE (delever+CT) --------':>50}")
    print(f"  {'Year':<6} {'SR':>5} {'PnL':>10} {'Ret%':>7} {'APR%':>7} {'DD%':>6} {'Worst':>6}"
          f" | {'SR':>5} {'PnL':>10} {'Ret%':>7} {'APR%':>7} {'DD%':>6} {'Worst':>6}")
    print("  " + "-" * 110)

    yolo_pnls = []
    safe_pnls = []
    yolo_srs = []
    safe_srs = []
    yolo_aprs = []
    safe_aprs = []

    for yr in years:
        y = all_results["YOLO"].get(yr)
        s = all_results["SAFE"].get(yr)
        if not y or not s:
            continue

        yolo_pnls.append(y["pnl"])
        safe_pnls.append(s["pnl"])
        yolo_srs.append(y["sr"])
        safe_srs.append(s["sr"])
        y_apr = y["ret_pct"] * (365.25 / y["days"])
        s_apr = s["ret_pct"] * (365.25 / s["days"])
        yolo_aprs.append(y_apr)
        safe_aprs.append(s_apr)

        print(f"  {yr:<6} {y['sr']:>5.2f} ${y['pnl']:>8,.0f} {y['ret_pct']:>6.1f}% {y_apr:>6.0f}% {y['max_dd']:>5.1f}% {y['worst_day_pct']:>5.1f}%"
              f" | {s['sr']:>5.2f} ${s['pnl']:>8,.0f} {s['ret_pct']:>6.1f}% {s_apr:>6.0f}% {s['max_dd']:>5.1f}% {s['worst_day_pct']:>5.1f}%")

    print("  " + "-" * 110)
    print(f"  {'MEAN':<6} {np.mean(yolo_srs):>5.2f} ${np.mean(yolo_pnls):>8,.0f} {np.mean([r['ret_pct'] for r in all_results['YOLO'].values()]):>6.1f}% {np.mean(yolo_aprs):>6.0f}%"
          f" {'':>13}"
          f" | {np.mean(safe_srs):>5.2f} ${np.mean(safe_pnls):>8,.0f} {np.mean([r['ret_pct'] for r in all_results['SAFE'].values()]):>6.1f}% {np.mean(safe_aprs):>6.0f}%")
    print(f"  {'MEDIAN':<6} {np.median(yolo_srs):>5.2f} ${np.median(yolo_pnls):>8,.0f} {np.median([r['ret_pct'] for r in all_results['YOLO'].values()]):>6.1f}% {np.median(yolo_aprs):>6.0f}%"
          f" {'':>13}"
          f" | {np.median(safe_srs):>5.2f} ${np.median(safe_pnls):>8,.0f} {np.median([r['ret_pct'] for r in all_results['SAFE'].values()]):>6.1f}% {np.median(safe_aprs):>6.0f}%")

    # Quarterly detail for both
    for cfg_name in configs:
        res = all_results[cfg_name]
        print(f"\n  QUARTERLY RETURNS % — {cfg_name}")
        print(f"  {'Year':<6} {'Q1':>10} {'Q2':>10} {'Q3':>10} {'Q4':>10} {'Full':>10}")
        print(f"  " + "-" * 60)
        neg_q = 0
        tot_q = 0
        for yr in years:
            r = res.get(yr)
            if not r:
                continue
            cols = []
            for q in [1, 2, 3, 4]:
                if q in r["quarters"]:
                    v = r["quarters"][q] / CAP * 100
                    tot_q += 1
                    if v < 0: neg_q += 1
                    cols.append(f"{v:>+9.1f}%")
                else:
                    cols.append(f"{'--':>10}")
            cols.append(f"{r['ret_pct']:>+9.1f}%")
            print(f"  {yr:<6} {''.join(cols)}")
        print(f"  Negative quarters: {neg_q}/{tot_q}")

    # Monthly consistency comparison
    print(f"\n  MONTHLY CONSISTENCY")
    print(f"  {'Year':<6} {'YOLO pos/tot':>13} {'SAFE pos/tot':>13}")
    print(f"  " + "-" * 35)
    y_pos_total = 0; y_tot_total = 0
    s_pos_total = 0; s_tot_total = 0
    for yr in years:
        y = all_results["YOLO"].get(yr)
        s = all_results["SAFE"].get(yr)
        if not y or not s:
            continue
        yp = y["tot_months"] - y["neg_months"]
        sp = s["tot_months"] - s["neg_months"]
        y_pos_total += yp; y_tot_total += y["tot_months"]
        s_pos_total += sp; s_tot_total += s["tot_months"]
        print(f"  {yr:<6} {yp:>5}/{y['tot_months']:<5}       {sp:>5}/{s['tot_months']:<5}")
    print(f"  " + "-" * 35)
    print(f"  {'TOTAL':<6} {y_pos_total:>5}/{y_tot_total:<5} ({y_pos_total/y_tot_total*100:.0f}%)"
          f"  {s_pos_total:>5}/{s_tot_total:<5} ({s_pos_total/s_tot_total*100:.0f}%)")

    # Best/worst quarter per year per config
    for cfg_name in configs:
        res = all_results[cfg_name]
        print(f"\n  BEST / WORST QUARTER — {cfg_name}")
        print(f"  {'Year':<6} {'Best':>18} {'Worst':>18}")
        print(f"  " + "-" * 45)
        for yr in years:
            r = res.get(yr)
            if not r or not r["quarters"]:
                continue
            qs = {f"Q{q}": v / CAP * 100 for q, v in r["quarters"].items()}
            best_q = max(qs, key=qs.get)
            worst_q = min(qs, key=qs.get)
            print(f"  {yr:<6} {best_q} {qs[best_q]:>+8.1f}%      {worst_q} {qs[worst_q]:>+8.1f}%")

    # Bottom line
    print(f"\n  BOTTOM LINE ({cap_label} capital, median year)")
    print(f"  " + "=" * 60)
    print(f"  YOLO:  ${np.median(yolo_pnls):>8,.0f} profit  ({np.median([r['ret_pct'] for r in all_results['YOLO'].values()]):.0f}%)"
          f"  SR {np.median(yolo_srs):.2f}  APR {np.median(yolo_aprs):.0f}%")
    print(f"  SAFE:  ${np.median(safe_pnls):>8,.0f} profit  ({np.median([r['ret_pct'] for r in all_results['SAFE'].values()]):.0f}%)"
          f"  SR {np.median(safe_srs):.2f}  APR {np.median(safe_aprs):.0f}%")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
