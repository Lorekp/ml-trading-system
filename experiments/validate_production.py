#!/usr/bin/env python
"""Validate production config: CT90 + maxSig15 + 6.0% sizing.

Runs the exact production backtest, reports period-specific performance,
and validates with 1000 Monte Carlo DD simulations.

Usage:
    PYTHONPATH="src;." python experiments/validate_production.py
"""

import sys, os, time, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from experiments.drawdown_improvements import simulate_with_improvements, CACHE_DIR
from experiments.circuit_breaker_v2_experiment import apply_vol, CAP
from experiments.dynamic_asset_v2_experiment import assemble_signals, OBVIOUS_CORE
from signal_engine.portfolio import compute_daily_pnl_series
from signal_engine.config import PortfolioConfig

N_MC = 1000
SEED = 42

# ── Production config (matches configs/bybit_demo.yaml exactly) ──
IMPROVEMENTS = {
    "correlation_throttle": 0.90,
    "correlation_direction": "both",
    "correlation_scale": 0.5,
    "max_signals_per_bar": 15,
    "max_risk_per_trade": 0.03,
    "sl_cap": 0.75,
}

PORT = PortfolioConfig(
    starting_capital=CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
    max_exposure=0.90, max_per_asset=0.06, max_per_strategy=0.50,
    base_sizing_pct=0.060, max_signal_mult=2.0, sizing_mode="probability_edge",
    use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)

BREAKER = {"window": 10, "threshold": -0.01, "floor": 0.10,
           "up_threshold": 0.01, "cap": 1.5}


def mc_vectorized(dr_values, n_sims=N_MC):
    """Block bootstrap MC — returns detailed breach stats."""
    rng = np.random.default_rng(SEED)
    n = len(dr_values)
    bs = 5
    nb = n // bs
    starts = rng.integers(0, n - bs, size=(n_sims, nb))
    off = np.arange(bs)
    idx = (starts[:, :, None] + off[None, None, :]).reshape(n_sims, -1)
    boot = dr_values[idx]
    cum = np.cumprod(1.0 + boot, axis=1)
    rm = np.maximum.accumulate(cum, axis=1)
    dds = ((rm - cum) / rm).max(axis=1) * 100
    days = boot.min(axis=1) * 100
    return {
        "worst_dd": float(dds.max()),
        "worst_day": float(days.min()),
        "breach_10": float((dds > 10).mean() * 100),
        "breach_8": float((dds > 8).mean() * 100),
        "breach_5day": float((days < -5).mean() * 100),
        "breach_3day": float((days < -3).mean() * 100),
        "dd_p50": float(np.percentile(dds, 50)),
        "dd_p95": float(np.percentile(dds, 95)),
        "dd_p99": float(np.percentile(dds, 99)),
        "day_p1": float(np.percentile(days, 1)),
        "day_p5": float(np.percentile(days, 5)),
        "passes": bool((dds <= 10).all() and (days >= -5).all()),
    }


def period_stats(dpnl, label, cap=CAP):
    """Compute stats for a PnL slice."""
    if len(dpnl) < 5:
        return None
    dr = dpnl / cap
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if dr.std() > 0 else 0
    eq = (1 + dr).cumprod()
    dd = (eq.cummax() - eq) / eq.cummax()
    pnl = float(dpnl.sum())
    win_rate = float((dpnl > 0).mean() * 100)
    return {
        "label": label, "days": len(dpnl), "sr": sr, "pnl": pnl,
        "max_dd": float(dd.max() * 100), "win_rate": win_rate,
        "pnl_day": pnl / len(dpnl),
    }


def main():
    T0 = time.time()
    print("=" * 100)
    print("  PRODUCTION CONFIG VALIDATION")
    print(f"  CT90 both 50% + maxSig15 + 6.0% sizing | $250K | {N_MC} MC sims")
    print("=" * 100)

    # ── Build signals (same as left_tail_reduction.py) ──
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
    print(f"  {len(prepared):,} signals assembled ({time.time()-T0:.0f}s)")

    # ── Run simulation (without compliance SL) ──
    t1 = time.time()
    sim = simulate_with_improvements(prepared, PORT, BREAKER, IMPROVEMENTS)
    if sim is None:
        print("  ERROR: simulate returned None")
        return
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None:
        print("  ERROR: no daily PnL")
        return
    if dpnl.index.tz is not None:
        dpnl.index = dpnl.index.tz_localize(None)

    # ── Also run with SL variants for comparison ──
    sl_results = {}
    # Dynamic SL (production config)
    imp_dyn = {**IMPROVEMENTS}  # already has max_risk_per_trade
    sim_dyn = simulate_with_improvements(prepared, PORT, BREAKER, imp_dyn)
    if sim_dyn:
        dpnl_dyn = compute_daily_pnl_series(sim_dyn)
        if dpnl_dyn is not None:
            if dpnl_dyn.index.tz is not None:
                dpnl_dyn.index = dpnl_dyn.index.tz_localize(None)
            dr_dyn = dpnl_dyn / CAP
            sr_dyn = float(dr_dyn.mean() / dr_dyn.std() * np.sqrt(365.25)) if dr_dyn.std() > 0 else 0
            eq_dyn = (1 + dr_dyn).cumprod()
            dd_dyn = float(((eq_dyn.cummax() - eq_dyn) / eq_dyn.cummax()).max() * 100)
            sl_results["dyn3%"] = {
                "sr": sr_dyn, "pnl": float(dpnl_dyn.sum()), "max_dd": dd_dyn,
                "trades": sim_dyn["n_trades"], "sl_triggers": sim_dyn.get("sl_triggers", 0),
            }
    # Fixed SL comparisons
    for sl_pct in [0.25, 0.50, 0.75]:
        imp_no_dyn = {k: v for k, v in IMPROVEMENTS.items()
                      if k not in ("max_risk_per_trade", "sl_cap")}
        imp_no_dyn["compliance_sl_pct"] = sl_pct
        sim_sl = simulate_with_improvements(prepared, PORT, BREAKER, imp_no_dyn)
        if sim_sl:
            dpnl_sl = compute_daily_pnl_series(sim_sl)
            if dpnl_sl is not None:
                if dpnl_sl.index.tz is not None:
                    dpnl_sl.index = dpnl_sl.index.tz_localize(None)
                dr_sl = dpnl_sl / CAP
                sr_sl = float(dr_sl.mean() / dr_sl.std() * np.sqrt(365.25)) if dr_sl.std() > 0 else 0
                eq_sl = (1 + dr_sl).cumprod()
                dd_sl = float(((eq_sl.cummax() - eq_sl) / eq_sl.cummax()).max() * 100)
                sl_results[f"fix{sl_pct*100:.0f}%"] = {
                    "sr": sr_sl, "pnl": float(dpnl_sl.sum()), "max_dd": dd_sl,
                    "trades": sim_sl["n_trades"], "sl_triggers": sim_sl.get("sl_triggers", 0),
                }
    print(f"  Simulations done ({time.time()-t1:.0f}s)")

    dr = dpnl / CAP

    # ── 1. Full-period stats ──
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    eq = (1 + dr).cumprod()
    dd_s = (eq.cummax() - eq) / eq.cummax()
    max_dd = float(dd_s.max() * 100)
    pnl = float(dpnl.sum())
    calmar = sr / max_dd * 100 if max_dd > 0 else 0
    win_rate = float((dpnl > 0).mean() * 100)
    monthly = dpnl.resample("MS").sum()
    daily_dd_pct = dpnl / (CAP + dpnl.cumsum()).shift(1).fillna(CAP) * 100
    worst_daily = float(daily_dd_pct.min())

    print(f"\n{'=' * 100}")
    print("  1. FULL PERIOD PERFORMANCE")
    print(f"{'=' * 100}")
    print(f"  Period:       {dpnl.index[0].date()} to {dpnl.index[-1].date()} ({len(dpnl)} days)")
    print(f"  Sharpe:       {sr:.2f}")
    print(f"  Total PnL:    ${pnl:,.0f}")
    print(f"  Final Equity: ${CAP + pnl:,.0f}")
    print(f"  Max DD:       {max_dd:.2f}%")
    print(f"  Worst Day:    {worst_daily:.2f}%")
    print(f"  Calmar:       {calmar:.2f}")
    print(f"  Win Rate:     {win_rate:.1f}%")
    print(f"  Trades:       {sim['n_trades']:,}")
    print(f"  PnL/month:    ${pnl / max(len(monthly), 1):,.0f}")
    print(f"  PnL/day:      ${pnl / len(dpnl):,.0f}")

    # ─��� 2. Period breakdown ─���
    print(f"\n{'=' * 100}")
    print("  2. PERIOD BREAKDOWN")
    print(f"{'=' * 100}")
    print(f"  {'Period':<20} {'Days':>5} {'Sharpe':>7} {'PnL':>12} {'$/day':>8} {'MaxDD':>7} {'WinR':>6}")
    print("  " + "-" * 75)

    slices = [
        ("Last 30d",  dpnl.iloc[-30:]  if len(dpnl) >= 30  else dpnl),
        ("Last 90d",  dpnl.iloc[-90:]  if len(dpnl) >= 90  else dpnl),
        ("Last 180d", dpnl.iloc[-180:] if len(dpnl) >= 180 else dpnl),
        ("Last 365d", dpnl.iloc[-365:] if len(dpnl) >= 365 else dpnl),
        ("Full",      dpnl),
    ]
    for label, sl in slices:
        s = period_stats(sl, label)
        if s:
            print(f"  {s['label']:<20} {s['days']:>5} {s['sr']:>7.2f}"
                  f" ${s['pnl']:>10,.0f} ${s['pnl_day']:>7,.0f}"
                  f" {s['max_dd']:>6.1f}% {s['win_rate']:>5.1f}%")

    # ── 3. Monthly PnL table ──
    print(f"\n{'=' * 100}")
    print("  3. MONTHLY PNL")
    print(f"{'=' * 100}")
    monthly_dr = dpnl.resample("MS").sum() / CAP
    print(f"  {'Month':<12} {'PnL':>10} {'Return':>8} {'Sharpe':>8}")
    print("  " + "-" * 42)
    neg_months = 0
    for dt, mp in monthly.items():
        m_dr = dpnl.loc[dpnl.index.to_period("M") == dt.to_period("M")] / CAP
        m_sr = float(m_dr.mean() / m_dr.std() * np.sqrt(365.25)) if m_dr.std() > 0 else 0
        marker = " ***" if mp < 0 else ""
        if mp < 0: neg_months += 1
        print(f"  {dt.strftime('%Y-%m'):<12} ${mp:>9,.0f} {mp/CAP*100:>7.2f}% {m_sr:>7.2f}{marker}")
    print(f"\n  Positive months: {len(monthly) - neg_months}/{len(monthly)}"
          f" ({(len(monthly) - neg_months)/len(monthly)*100:.0f}%)")

    # ── 4. Worst days ──
    print(f"\n{'=' * 100}")
    print("  4. WORST 10 DAYS")
    print(f"{'=' * 100}")
    worst = dpnl.nsmallest(10)
    for dt, val in worst.items():
        pct = val / CAP * 100
        print(f"  {dt.strftime('%Y-%m-%d')}  ${val:>10,.0f}  ({pct:>+.2f}%)")

    # ── 5. Monte Carlo DD validation ──
    print(f"\n{'=' * 100}")
    print(f"  5. MONTE CARLO VALIDATION ({N_MC} sims)")
    print(f"{'=' * 100}")
    t2 = time.time()
    mc = mc_vectorized(dr.values)
    print(f"  Computed in {time.time()-t2:.1f}s")
    print(f"  Worst MC MaxDD:   {mc['worst_dd']:.2f}%")
    print(f"  Worst MC Day:     {mc['worst_day']:.2f}%")
    print(f"  DD p50/p95/p99:   {mc['dd_p50']:.1f}% / {mc['dd_p95']:.1f}% / {mc['dd_p99']:.1f}%")
    print(f"  Day p1/p5:        {mc['day_p1']:.2f}% / {mc['day_p5']:.2f}%")
    print(f"  Breach >10% DD:   {mc['breach_10']:.1f}%")
    print(f"  Breach >8% DD:    {mc['breach_8']:.1f}%")
    print(f"  Breach <-5% day:  {mc['breach_5day']:.1f}%")
    print(f"  Breach <-3% day:  {mc['breach_3day']:.1f}%")
    print(f"  PASSES 10%/5%:    {'YES' if mc['passes'] else 'NO'}")

    # ── 6. Recent regime check ──
    print(f"\n{'=' * 100}")
    print("  6. REGIME STABILITY CHECK")
    print(f"{'=' * 100}")
    yearly_sr = {}
    for yr in sorted(dpnl.index.year.unique()):
        yr_pnl = dpnl[dpnl.index.year == yr]
        yr_dr = yr_pnl / CAP
        if yr_dr.std() > 0 and len(yr_dr) > 20:
            yearly_sr[yr] = float(yr_dr.mean() / yr_dr.std() * np.sqrt(365.25))
    for yr, ysr in yearly_sr.items():
        bar = "+" * max(int(ysr), 0)
        print(f"  {yr}: Sharpe {ysr:>6.2f}  {bar}")

    # ── 7. COMPLIANCE STOP LOSS IMPACT ──
    print(f"\n{'=' * 100}")
    print("  7. COMPLIANCE STOP LOSS IMPACT")
    print(f"{'=' * 100}")
    print(f"  {'Config':<22} {'Sharpe':>7} {'PnL':>12} {'MaxDD':>7} {'Trades':>7} {'SL Hits':>8} {'Delta PnL':>10}")
    print("  " + "-" * 80)

    # Baseline is now the dynamic SL run (since that's production config)
    # Show no-SL for reference
    imp_nosl = {k: v for k, v in IMPROVEMENTS.items()
                if k not in ("max_risk_per_trade", "sl_cap", "compliance_sl_pct")}
    sim_nosl = simulate_with_improvements(prepared, PORT, BREAKER, imp_nosl)
    dpnl_nosl = compute_daily_pnl_series(sim_nosl)
    if dpnl_nosl.index.tz is not None:
        dpnl_nosl.index = dpnl_nosl.index.tz_localize(None)
    nosl_pnl = float(dpnl_nosl.sum())
    nosl_dr = dpnl_nosl / CAP
    nosl_sr = float(nosl_dr.mean() / nosl_dr.std() * np.sqrt(365.25))
    nosl_dd = float(((1 + nosl_dr).cumprod().pipe(lambda e: (e.cummax() - e) / e.cummax())).max() * 100)
    print(f"  {'No SL':<22} {nosl_sr:>7.2f} ${nosl_pnl:>10,.0f} {nosl_dd:>6.2f}%"
          f" {sim_nosl['n_trades']:>7,}       -          -")

    for label in ["dyn3%", "fix25%", "fix50%", "fix75%"]:
        if label in sl_results:
            r = sl_results[label]
            delta = r["pnl"] - nosl_pnl
            print(f"  {label:<22} {r['sr']:>7.2f} ${r['pnl']:>10,.0f} {r['max_dd']:>6.2f}%"
                  f" {r['trades']:>7,} {r['sl_triggers']:>8} ${delta:>+9,.0f}")

    # ── 8. GO/NO-GO ──
    print(f"\n{'=' * 100}")
    print("  8. GO/NO-GO ASSESSMENT")
    print(f"{'=' * 100}")
    last_90 = period_stats(dpnl.iloc[-90:] if len(dpnl) >= 90 else dpnl, "90d")
    checks = [
        ("MC 0% breach at 10%/5%", mc["passes"]),
        ("Full Sharpe > 2.0", sr > 2.0),
        ("Last 90d Sharpe > 1.0", last_90["sr"] > 1.0 if last_90 else False),
        ("Win rate > 50%", win_rate > 50),
        ("Max DD < 8% (backtest)", max_dd < 8),
    ]
    all_pass = True
    for label, passed in checks:
        status = "PASS" if passed else "FAIL"
        if not passed: all_pass = False
        print(f"  [{status}] {label}")

    print(f"\n  VERDICT: {'GO — ready for prop challenge' if all_pass else 'NO-GO — see failures above'}")
    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
