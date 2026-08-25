#!/usr/bin/env python
"""Aggressive config sweep for $100 personal account.

Ignores prop-firm drawdown rules. Finds max-return configurations.

Usage:
    PYTHONPATH="src;." python experiments/aggressive_sweep.py
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

CAP = 100.0  # $100 personal account

def run_config(prepared, vol_series, label, port, breaker, improvements):
    """Run one config, return stats dict."""
    try:
        sim = simulate_with_improvements(prepared, port, breaker, improvements)
        if sim is None:
            return None
        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None or len(dpnl) < 30:
            return None
        if dpnl.index.tz is not None:
            dpnl.index = dpnl.index.tz_localize(None)

        dr = dpnl / CAP
        sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if dr.std() > 0 else 0
        eq = (1 + dr).cumprod()
        dd = (eq.cummax() - eq) / eq.cummax()
        pnl = float(dpnl.sum())
        final_eq = CAP + pnl
        max_dd = float(dd.max() * 100)
        cagr = float((eq.iloc[-1]) ** (365.25 / len(dpnl)) - 1) * 100
        worst_day = float((dpnl / CAP * 100).min())

        return {
            "label": label,
            "sr": sr,
            "pnl": pnl,
            "final_eq": final_eq,
            "return_pct": pnl / CAP * 100,
            "cagr": cagr,
            "max_dd": max_dd,
            "worst_day": worst_day,
            "trades": sim["n_trades"],
            "pnl_day": pnl / len(dpnl),
        }
    except Exception as e:
        print(f"  {label}: ERROR — {e}")
        return None


def main():
    T0 = time.time()
    print("=" * 110)
    print("  AGGRESSIVE CONFIG SWEEP — $100 personal account, max returns")
    print("=" * 110)

    # Load cached signals
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
    print(f"  {len(prepared):,} signals ({time.time()-T0:.0f}s)\n")

    # ── Define configs to sweep ──
    # Production baseline (scaled to $100)
    configs = []

    # 1. Production baseline
    configs.append(("PROD baseline",
        PortfolioConfig(starting_capital=CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
                        max_exposure=0.90, max_per_asset=0.06, max_per_strategy=0.50,
                        base_sizing_pct=0.060, max_signal_mult=2.0, sizing_mode="probability_edge",
                        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        {"correlation_throttle": 0.90, "correlation_direction": "both", "correlation_scale": 0.5,
         "max_signals_per_bar": 15}))

    # 2. No safety rails (remove delever, higher exposure)
    configs.append(("No delever",
        PortfolioConfig(starting_capital=CAP, max_dd=1.00, delever_start=1.00, slippage=0.0001,
                        max_exposure=0.95, max_per_asset=0.10, max_per_strategy=0.50,
                        base_sizing_pct=0.060, max_signal_mult=2.0, sizing_mode="probability_edge",
                        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        {"correlation_throttle": 0.90, "correlation_direction": "both", "correlation_scale": 0.5,
         "max_signals_per_bar": 15}))

    # 3. 10% sizing (big positions)
    configs.append(("10% sizing",
        PortfolioConfig(starting_capital=CAP, max_dd=1.00, delever_start=1.00, slippage=0.0001,
                        max_exposure=0.95, max_per_asset=0.15, max_per_strategy=0.50,
                        base_sizing_pct=0.100, max_signal_mult=2.0, sizing_mode="probability_edge",
                        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        {"max_signals_per_bar": 15}))

    # 4. 15% sizing
    configs.append(("15% sizing",
        PortfolioConfig(starting_capital=CAP, max_dd=1.00, delever_start=1.00, slippage=0.0001,
                        max_exposure=0.95, max_per_asset=0.20, max_per_strategy=0.50,
                        base_sizing_pct=0.150, max_signal_mult=2.0, sizing_mode="probability_edge",
                        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        {"max_signals_per_bar": 15}))

    # 5. 20% sizing, no CT
    configs.append(("20% no CT",
        PortfolioConfig(starting_capital=CAP, max_dd=1.00, delever_start=1.00, slippage=0.0001,
                        max_exposure=0.95, max_per_asset=0.25, max_per_strategy=0.50,
                        base_sizing_pct=0.200, max_signal_mult=2.0, sizing_mode="probability_edge",
                        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        {"max_signals_per_bar": 20}))

    # 6. 25% sizing, no CT, high mult
    configs.append(("25% 3x mult",
        PortfolioConfig(starting_capital=CAP, max_dd=1.00, delever_start=1.00, slippage=0.0001,
                        max_exposure=0.95, max_per_asset=0.30, max_per_strategy=0.50,
                        base_sizing_pct=0.250, max_signal_mult=3.0, sizing_mode="probability_edge",
                        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        {"max_signals_per_bar": 25}))

    # 7. 30% YOLO
    configs.append(("30% YOLO",
        PortfolioConfig(starting_capital=CAP, max_dd=1.00, delever_start=1.00, slippage=0.0001,
                        max_exposure=1.00, max_per_asset=0.40, max_per_strategy=1.00,
                        base_sizing_pct=0.300, max_signal_mult=3.0, sizing_mode="probability_edge",
                        use_regime_exposure_multiplier=False, regime_exposure_floor=1.00),
        {"max_signals_per_bar": 30}))

    # 8-10. Leverage variants on 15% sizing (simulated as 2x/3x/5x capital)
    for lev in [2, 3, 5]:
        configs.append((f"15% {lev}x lev",
            PortfolioConfig(starting_capital=CAP, max_dd=1.00, delever_start=1.00, slippage=0.0001,
                            max_exposure=0.95 * lev, max_per_asset=0.20 * lev, max_per_strategy=0.50,
                            base_sizing_pct=0.150 * lev, max_signal_mult=2.0, sizing_mode="probability_edge",
                            use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
            {"max_signals_per_bar": 15}))

    # 11-12. Conservative leverage (10% sizing with 2x/3x)
    for lev in [2, 3]:
        configs.append((f"10% {lev}x lev",
            PortfolioConfig(starting_capital=CAP, max_dd=1.00, delever_start=1.00, slippage=0.0001,
                            max_exposure=0.95 * lev, max_per_asset=0.15 * lev, max_per_strategy=0.50,
                            base_sizing_pct=0.100 * lev, max_signal_mult=2.0, sizing_mode="probability_edge",
                            use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
            {"max_signals_per_bar": 15}))

    breaker = {"window": 10, "threshold": -0.01, "floor": 0.10,
               "up_threshold": 0.01, "cap": 1.5}

    # ── Run all configs ──
    results = []
    for label, port, imp in configs:
        t1 = time.time()
        r = run_config(prepared, vol_series, label, port, breaker, imp)
        if r:
            results.append(r)
            print(f"  {label:<18} SR={r['sr']:.2f}  PnL=${r['pnl']:,.0f}  "
                  f"CAGR={r['cagr']:.0f}%  DD={r['max_dd']:.1f}%  "
                  f"Trades={r['trades']:,}  ({time.time()-t1:.1f}s)")

    # ── Summary table sorted by total PnL ──
    results.sort(key=lambda x: -x["pnl"])
    print(f"\n{'=' * 110}")
    print("  RESULTS RANKED BY TOTAL PNL ($100 starting capital)")
    print(f"{'=' * 110}")
    print(f"  {'Config':<18} {'Sharpe':>7} {'TotalPnL':>10} {'FinalEq':>10} {'Return%':>8} "
          f"{'CAGR%':>7} {'MaxDD%':>7} {'WorstDay':>9} {'Trades':>7} {'$/day':>7}")
    print("  " + "-" * 105)
    for r in results:
        print(f"  {r['label']:<18} {r['sr']:>7.2f} ${r['pnl']:>8,.0f} ${r['final_eq']:>8,.0f} "
              f"{r['return_pct']:>7.0f}% {r['cagr']:>6.0f}% {r['max_dd']:>6.1f}% "
              f"{r['worst_day']:>8.1f}% {r['trades']:>7,} ${r['pnl_day']:>6.1f}")

    # ── Best risk-adjusted (Sharpe * sqrt(return)) ──
    print(f"\n  RECOMMENDATION:")
    best_sr = max(results, key=lambda x: x["sr"])
    best_pnl = max(results, key=lambda x: x["pnl"])
    best_bang = max(results, key=lambda x: x["sr"] * np.sqrt(max(x["return_pct"], 1)))
    print(f"  Best Sharpe:      {best_sr['label']} (SR={best_sr['sr']:.2f}, ${best_sr['pnl']:,.0f})")
    print(f"  Best Total PnL:   {best_pnl['label']} (${best_pnl['pnl']:,.0f}, DD={best_pnl['max_dd']:.1f}%)")
    print(f"  Best bang/buck:    {best_bang['label']} (SR={best_bang['sr']:.2f}, ${best_bang['pnl']:,.0f})")

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
