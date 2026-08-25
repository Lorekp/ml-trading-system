#!/usr/bin/env python
"""Sweep compliance SL from 1% to 100% — find stable region.

Usage:
    PYTHONPATH="src;." python experiments/compliance_sl_sweep.py
"""

import sys, os, time, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

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

PORT = PortfolioConfig(
    starting_capital=CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
    max_exposure=0.90, max_per_asset=0.06, max_per_strategy=0.50,
    base_sizing_pct=0.060, max_signal_mult=2.0, sizing_mode="probability_edge",
    use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)

BREAKER = {"window": 10, "threshold": -0.01, "floor": 0.10,
           "up_threshold": 0.01, "cap": 1.5}

BASE_IMP = {
    "correlation_throttle": 0.90,
    "correlation_direction": "both",
    "correlation_scale": 0.5,
    "max_signals_per_bar": 15,
}

_CACHE = {}


def mc_vectorized(dr_values, n_sims=N_MC):
    rng = np.random.default_rng(SEED)
    n = len(dr_values)
    bs, nb = 5, n // 5
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
        "passes": bool((dds <= 10).all() and (days >= -5).all()),
    }


def _worker(prepared_path, sl_pct):
    warnings.filterwarnings("ignore")
    if prepared_path not in _CACHE:
        with open(prepared_path, "rb") as f:
            _CACHE[prepared_path] = pickle.load(f)
    prepared = _CACHE[prepared_path]

    imp = {**BASE_IMP}
    if sl_pct is not None:
        imp["compliance_sl_pct"] = sl_pct

    sim = simulate_with_improvements(prepared, PORT, BREAKER, imp)
    if sim is None:
        return None

    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None:
        return None
    if dpnl.index.tz is not None:
        dpnl.index = dpnl.index.tz_localize(None)

    dr = dpnl / CAP
    if dr.std() == 0 or len(dr) < 30:
        return None

    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    eq = (1 + dr).cumprod()
    dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)
    pnl = float(dpnl.sum())
    mc = mc_vectorized(dr.values)

    return {
        "sl_pct": sl_pct, "sr": sr, "pnl": pnl, "max_dd": dd,
        "trades": sim["n_trades"], "sl_triggers": sim.get("sl_triggers", 0),
        "mc_worst_dd": mc["worst_dd"], "mc_worst_day": mc["worst_day"],
        "mc_passes": mc["passes"],
    }


def main():
    T0 = time.time()
    print("=" * 110)
    print("  COMPLIANCE SL SWEEP — 1% to 100%")
    print("=" * 110)

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

    pp = str(CACHE_DIR / "_sl_sweep_prepared.pkl")
    with open(pp, "wb") as f:
        pickle.dump(prepared, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  {len(prepared):,} signals ({time.time()-T0:.0f}s)\n")

    # SL levels: 1% to 100% in 1% steps + no SL baseline
    sl_levels = [None] + [round(x/100, 2) for x in range(1, 101)]

    print(f"  Running {len(sl_levels)} configs...", flush=True)
    t1 = time.time()

    results = []
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count())) as pool:
        futs = {pool.submit(_worker, pp, sl): sl for sl in sl_levels}
        for f in as_completed(futs):
            r = f.result()
            if r:
                results.append(r)

    results.sort(key=lambda x: (x["sl_pct"] is None, x["sl_pct"] or 0))
    print(f"  Done ({time.time()-t1:.0f}s)\n")

    # Table
    print(f"  {'SL%':>5} {'Sharpe':>7} {'PnL':>12} {'MaxDD':>7} {'SL Hits':>8}"
          f" {'MC wDD':>7} {'MC wDay':>8} {'MC Pass':>8}")
    print("  " + "-" * 75)

    baseline_pnl = None
    for r in results:
        sl_str = "None" if r["sl_pct"] is None else f"{r['sl_pct']*100:.0f}%"
        if r["sl_pct"] is None:
            baseline_pnl = r["pnl"]
        mc_pass = "YES" if r["mc_passes"] else "NO"
        print(f"  {sl_str:>5} {r['sr']:>7.2f} ${r['pnl']:>10,.0f} {r['max_dd']:>6.2f}%"
              f" {r['sl_triggers']:>8} {r['mc_worst_dd']:>6.1f}% {r['mc_worst_day']:>7.2f}%"
              f" {mc_pass:>8}")

    # Find stable region
    print(f"\n{'=' * 110}")
    print("  ANALYSIS")
    print(f"{'=' * 110}")

    if baseline_pnl:
        # Find first SL level where PnL drops >5% vs baseline
        threshold = baseline_pnl * 0.95
        for r in results:
            if r["sl_pct"] is not None and r["pnl"] < threshold:
                print(f"  PnL drops >5% below {r['sl_pct']*100:.0f}% SL")
                break

        # Find best SL (max PnL)
        sl_only = [r for r in results if r["sl_pct"] is not None and r["mc_passes"]]
        if sl_only:
            best = max(sl_only, key=lambda x: x["pnl"])
            print(f"  Best SL: {best['sl_pct']*100:.0f}% (PnL=${best['pnl']:,.0f},"
                  f" SR={best['sr']:.2f}, {best['sl_triggers']} triggers)")

        # Find stable region (SL levels where PnL is within 1% of baseline)
        stable = [r for r in results if r["sl_pct"] is not None
                  and abs(r["pnl"] - baseline_pnl) / baseline_pnl < 0.01
                  and r["mc_passes"]]
        if stable:
            lo = min(r["sl_pct"] for r in stable)
            hi = max(r["sl_pct"] for r in stable)
            print(f"  Stable region (<1% PnL impact): {lo*100:.0f}%-{hi*100:.0f}%")
            mid = stable[len(stable)//2]
            print(f"  Recommended: {mid['sl_pct']*100:.0f}% SL"
                  f" (PnL=${mid['pnl']:,.0f}, {mid['sl_triggers']} triggers)")

    # Cleanup
    Path(pp).unlink(missing_ok=True)
    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
