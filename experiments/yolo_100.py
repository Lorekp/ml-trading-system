#!/usr/bin/env python
"""$100 YOLO analysis with Monte Carlo bankruptcy probability.

Finds the max-return config that doesn't reliably blow up,
then runs MC to estimate P(ruin) at various thresholds.

Usage:
    PYTHONPATH="src;." python experiments/yolo_100.py
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
N_MC = 10000
SEED = 42


def mc_bankruptcy(dr_values, n_sims=N_MC, thresholds=(0.50, 0.75, 0.90, 0.95)):
    """Block bootstrap MC — returns bankruptcy/drawdown stats.

    Thresholds: probability of losing X% of starting capital at any point.
    """
    rng = np.random.default_rng(SEED)
    n = len(dr_values)
    bs = 5  # block size
    nb = n // bs

    starts = rng.integers(0, n - bs, size=(n_sims, nb))
    off = np.arange(bs)
    idx = (starts[:, :, None] + off[None, None, :]).reshape(n_sims, -1)
    boot = dr_values[idx]

    # Cumulative equity curves
    cum = np.cumprod(1.0 + boot, axis=1)
    final_eq = cum[:, -1]

    # Running max and drawdowns
    rm = np.maximum.accumulate(cum, axis=1)
    dds = (rm - cum) / rm  # drawdown series
    max_dds = dds.max(axis=1)  # max DD per sim

    # Running min equity (for bankruptcy)
    min_eq = np.minimum.accumulate(cum, axis=1).min(axis=1)

    # Worst single day
    worst_days = boot.min(axis=1)

    results = {
        "n_sims": n_sims,
        "period_days": n,
        # Final equity stats
        "final_eq_mean": float(final_eq.mean()) * CAP,
        "final_eq_median": float(np.median(final_eq)) * CAP,
        "final_eq_p5": float(np.percentile(final_eq, 5)) * CAP,
        "final_eq_p10": float(np.percentile(final_eq, 10)) * CAP,
        "final_eq_p25": float(np.percentile(final_eq, 25)) * CAP,
        "final_eq_p75": float(np.percentile(final_eq, 75)) * CAP,
        "final_eq_p90": float(np.percentile(final_eq, 90)) * CAP,
        "final_eq_p95": float(np.percentile(final_eq, 95)) * CAP,
        # Return stats
        "return_mean": float((final_eq.mean() - 1) * 100),
        "return_median": float((np.median(final_eq) - 1) * 100),
        "return_p5": float((np.percentile(final_eq, 5) - 1) * 100),
        "return_p95": float((np.percentile(final_eq, 95) - 1) * 100),
        # Drawdown stats
        "max_dd_mean": float(max_dds.mean() * 100),
        "max_dd_median": float(np.median(max_dds) * 100),
        "max_dd_p95": float(np.percentile(max_dds, 95) * 100),
        "max_dd_p99": float(np.percentile(max_dds, 99) * 100),
        # Worst day
        "worst_day_p1": float(np.percentile(worst_days, 1) * 100),
        "worst_day_p5": float(np.percentile(worst_days, 5) * 100),
        # Bankruptcy probabilities (equity ever drops below threshold)
        "p_lose_50pct": float((min_eq < 0.50).mean() * 100),
        "p_lose_75pct": float((min_eq < 0.25).mean() * 100),
        "p_lose_90pct": float((min_eq < 0.10).mean() * 100),
        "p_lose_95pct": float((min_eq < 0.05).mean() * 100),
        "p_lose_99pct": float((min_eq < 0.01).mean() * 100),
        # End negative
        "p_end_negative": float((final_eq < 1.0).mean() * 100),
        "p_end_below_50": float((final_eq < 0.5).mean() * 100),
        # Profit probability
        "p_double": float((final_eq >= 2.0).mean() * 100),
        "p_5x": float((final_eq >= 5.0).mean() * 100),
        "p_10x": float((final_eq >= 10.0).mean() * 100),
    }
    return results


def run_sim(prepared, label, port, breaker, imp):
    """Run backtest + MC, return everything."""
    try:
        sim = simulate_with_improvements(prepared, port, breaker, imp)
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
        dd_s = (eq.cummax() - eq) / eq.cummax()
        pnl = float(dpnl.sum())
        max_dd = float(dd_s.max() * 100)
        worst_day = float(dr.min() * 100)
        win_rate = float((dpnl > 0).mean() * 100)

        # MC
        mc = mc_bankruptcy(dr.values)

        # Yearly breakdown
        yearly = {}
        for yr in sorted(dpnl.index.year.unique()):
            yr_pnl = dpnl[dpnl.index.year == yr]
            yr_dr = yr_pnl / CAP
            yr_sr = float(yr_dr.mean() / yr_dr.std() * np.sqrt(365.25)) if yr_dr.std() > 0 else 0
            yearly[yr] = {"sr": yr_sr, "pnl": float(yr_pnl.sum()), "ret": float(yr_pnl.sum() / CAP * 100)}

        return {
            "label": label,
            "sr": sr, "pnl": pnl, "final_eq": CAP + pnl,
            "max_dd": max_dd, "worst_day": worst_day,
            "win_rate": win_rate, "trades": sim["n_trades"],
            "days": len(dpnl),
            "start": str(dpnl.index[0].date()),
            "end": str(dpnl.index[-1].date()),
            "mc": mc, "yearly": yearly,
            "dpnl": dpnl,  # keep for detailed analysis
        }
    except Exception as e:
        print(f"  {label}: ERROR — {e}")
        import traceback; traceback.print_exc()
        return None


def main():
    T0 = time.time()
    print("=" * 100)
    print("  $100 YOLO ANALYSIS — balls to the wall, with MC bankruptcy estimates")
    print("=" * 100)

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
    print(f"  {len(prepared):,} signals, period 2020-07 to 2026-02 ({time.time()-T0:.0f}s)\n")

    breaker = {"window": 10, "threshold": -0.01, "floor": 0.10,
               "up_threshold": 0.01, "cap": 1.5}

    # ── CONFIGS ──
    configs = []

    # A. Conservative aggressive (10% sizing, no delever)
    configs.append(("A: 10% aggro",
        PortfolioConfig(starting_capital=CAP, max_dd=1.00, delever_start=1.00, slippage=0.0001,
                        max_exposure=0.95, max_per_asset=0.15, max_per_strategy=0.50,
                        base_sizing_pct=0.100, max_signal_mult=2.0, sizing_mode="probability_edge",
                        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        {"max_signals_per_bar": 15}))

    # B. Medium (15% sizing)
    configs.append(("B: 15% medium",
        PortfolioConfig(starting_capital=CAP, max_dd=1.00, delever_start=1.00, slippage=0.0001,
                        max_exposure=0.95, max_per_asset=0.20, max_per_strategy=0.50,
                        base_sizing_pct=0.150, max_signal_mult=2.0, sizing_mode="probability_edge",
                        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        {"max_signals_per_bar": 15}))

    # C. Hot (20% sizing, no CT, more signals)
    configs.append(("C: 20% hot",
        PortfolioConfig(starting_capital=CAP, max_dd=1.00, delever_start=1.00, slippage=0.0001,
                        max_exposure=0.95, max_per_asset=0.25, max_per_strategy=0.50,
                        base_sizing_pct=0.200, max_signal_mult=2.0, sizing_mode="probability_edge",
                        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        {"max_signals_per_bar": 20}))

    # D. 10% + 2x leverage
    configs.append(("D: 10% 2x lev",
        PortfolioConfig(starting_capital=CAP, max_dd=1.00, delever_start=1.00, slippage=0.0001,
                        max_exposure=1.90, max_per_asset=0.30, max_per_strategy=1.00,
                        base_sizing_pct=0.200, max_signal_mult=2.0, sizing_mode="probability_edge",
                        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        {"max_signals_per_bar": 15}))

    # E. 15% + 2x leverage
    configs.append(("E: 15% 2x lev",
        PortfolioConfig(starting_capital=CAP, max_dd=1.00, delever_start=1.00, slippage=0.0001,
                        max_exposure=1.90, max_per_asset=0.40, max_per_strategy=1.00,
                        base_sizing_pct=0.300, max_signal_mult=2.0, sizing_mode="probability_edge",
                        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        {"max_signals_per_bar": 15}))

    # F. 20% + 2x leverage (the real YOLO)
    configs.append(("F: 20% 2x lev",
        PortfolioConfig(starting_capital=CAP, max_dd=1.00, delever_start=1.00, slippage=0.0001,
                        max_exposure=1.90, max_per_asset=0.50, max_per_strategy=1.00,
                        base_sizing_pct=0.400, max_signal_mult=2.0, sizing_mode="probability_edge",
                        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        {"max_signals_per_bar": 20}))

    # G. 10% + 3x leverage
    configs.append(("G: 10% 3x lev",
        PortfolioConfig(starting_capital=CAP, max_dd=1.00, delever_start=1.00, slippage=0.0001,
                        max_exposure=2.85, max_per_asset=0.45, max_per_strategy=1.00,
                        base_sizing_pct=0.300, max_signal_mult=2.0, sizing_mode="probability_edge",
                        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        {"max_signals_per_bar": 15}))

    # H. Full send: 20% + 3x
    configs.append(("H: 20% 3x lev",
        PortfolioConfig(starting_capital=CAP, max_dd=1.00, delever_start=1.00, slippage=0.0001,
                        max_exposure=2.85, max_per_asset=0.60, max_per_strategy=1.00,
                        base_sizing_pct=0.600, max_signal_mult=2.0, sizing_mode="probability_edge",
                        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        {"max_signals_per_bar": 20}))

    # ── Run all ──
    results = []
    for label, port, imp in configs:
        t1 = time.time()
        r = run_sim(prepared, label, port, breaker, imp)
        if r:
            results.append(r)
            mc = r["mc"]
            print(f"  {label:<18} SR={r['sr']:.2f}  ${r['final_eq']:>8,.0f}  "
                  f"DD={r['max_dd']:.1f}%  P(ruin>50%)={mc['p_lose_50pct']:.1f}%  "
                  f"Trades={r['trades']:,}  ({time.time()-t1:.1f}s)")

    # ── DETAILED RESULTS ──
    print(f"\n{'=' * 100}")
    print(f"  BACKTEST PERIOD: 2020-07-02 to 2026-02-28 ({results[0]['days']} days / {results[0]['days']/365.25:.1f} years)")
    print(f"  Starting capital: $100 | {N_MC:,} MC simulations per config")
    print(f"{'=' * 100}")

    print(f"\n  {'Config':<18} {'SR':>5} {'BT Final$':>9} {'BT MaxDD':>8} {'Trades':>7} {'WR':>5} {'BT Worst':>8}")
    print("  " + "-" * 65)
    for r in results:
        print(f"  {r['label']:<18} {r['sr']:>5.2f} ${r['final_eq']:>7,.0f} {r['max_dd']:>7.1f}% "
              f"{r['trades']:>7,} {r['win_rate']:>4.0f}% {r['worst_day']:>7.1f}%")

    # ── MC RESULTS TABLE ──
    print(f"\n{'=' * 100}")
    print("  MONTE CARLO: RETURN DISTRIBUTION (10K sims)")
    print(f"{'=' * 100}")
    print(f"  {'Config':<18} {'Median$':>8} {'Mean$':>8} {'p5$':>8} {'p25$':>8} {'p75$':>8} {'p95$':>8}")
    print("  " + "-" * 70)
    for r in results:
        mc = r["mc"]
        print(f"  {r['label']:<18} ${mc['final_eq_median']:>6,.0f} ${mc['final_eq_mean']:>6,.0f} "
              f"${mc['final_eq_p5']:>6,.0f} ${mc['final_eq_p25']:>6,.0f} "
              f"${mc['final_eq_p75']:>6,.0f} ${mc['final_eq_p95']:>6,.0f}")

    print(f"\n{'=' * 100}")
    print("  MONTE CARLO: RISK OF RUIN")
    print(f"{'=' * 100}")
    print(f"  {'Config':<18} {'P(lose>50%)':>11} {'P(lose>75%)':>11} {'P(lose>90%)':>11} "
          f"{'P(lose>95%)':>11} {'P(end neg)':>10} {'MC MaxDD':>8}")
    print("  " + "-" * 85)
    for r in results:
        mc = r["mc"]
        print(f"  {r['label']:<18} {mc['p_lose_50pct']:>10.1f}% {mc['p_lose_75pct']:>10.1f}% "
              f"{mc['p_lose_90pct']:>10.1f}% {mc['p_lose_95pct']:>10.1f}% "
              f"{mc['p_end_negative']:>9.1f}% {mc['max_dd_p95']:>7.1f}%")

    print(f"\n{'=' * 100}")
    print("  MONTE CARLO: UPSIDE PROBABILITY")
    print(f"{'=' * 100}")
    print(f"  {'Config':<18} {'P(2x)':>7} {'P(5x)':>7} {'P(10x)':>7} {'MC median ret':>13}")
    print("  " + "-" * 55)
    for r in results:
        mc = r["mc"]
        print(f"  {r['label']:<18} {mc['p_double']:>6.1f}% {mc['p_5x']:>6.1f}% {mc['p_10x']:>6.1f}% "
              f"{mc['return_median']:>12.0f}%")

    # ── YEARLY BREAKDOWN for top 2 configs ──
    top2 = sorted(results, key=lambda x: x["mc"]["final_eq_median"], reverse=True)[:3]
    print(f"\n{'=' * 100}")
    print("  YEARLY BREAKDOWN (top 3 configs)")
    print(f"{'=' * 100}")
    for r in top2:
        print(f"\n  {r['label']}:")
        print(f"    {'Year':>6} {'Sharpe':>7} {'PnL':>10} {'Return':>8}")
        for yr in sorted(r["yearly"]):
            y = r["yearly"][yr]
            print(f"    {yr:>6} {y['sr']:>7.2f} ${y['pnl']:>8,.0f} {y['ret']:>7.0f}%")

    # ── RECOMMENDATION ──
    print(f"\n{'=' * 100}")
    print("  RECOMMENDATION")
    print(f"{'=' * 100}")
    # Find best median MC return with <5% ruin probability
    safe_ish = [r for r in results if r["mc"]["p_lose_50pct"] < 5.0]
    if safe_ish:
        best = max(safe_ish, key=lambda x: x["mc"]["final_eq_median"])
        mc = best["mc"]
        print(f"\n  BEST (P(ruin) < 5%):  {best['label']}")
        print(f"    MC Median outcome:  ${mc['final_eq_median']:,.0f}  ({mc['return_median']:.0f}% return)")
        print(f"    MC 5th percentile:  ${mc['final_eq_p5']:,.0f}")
        print(f"    MC 95th percentile: ${mc['final_eq_p95']:,.0f}")
        print(f"    P(lose >50%):       {mc['p_lose_50pct']:.1f}%")
        print(f"    P(double money):    {mc['p_double']:.1f}%")
        print(f"    P(10x):             {mc['p_10x']:.1f}%")

    # Absolute best returns regardless of ruin
    best_yolo = max(results, key=lambda x: x["mc"]["final_eq_median"])
    mc = best_yolo["mc"]
    print(f"\n  MAX RETURNS (any risk):  {best_yolo['label']}")
    print(f"    MC Median outcome:  ${mc['final_eq_median']:,.0f}  ({mc['return_median']:.0f}% return)")
    print(f"    MC 5th percentile:  ${mc['final_eq_p5']:,.0f}")
    print(f"    MC 95th percentile: ${mc['final_eq_p95']:,.0f}")
    print(f"    P(lose >50%):       {mc['p_lose_50pct']:.1f}%")
    print(f"    P(lose >90%):       {mc['p_lose_90pct']:.1f}%")
    print(f"    P(double money):    {mc['p_double']:.1f}%")
    print(f"    P(10x):             {mc['p_10x']:.1f}%")

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
