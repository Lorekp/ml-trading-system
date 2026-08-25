#!/usr/bin/env python
"""Find max PnL config that NEVER breaches DD limits in 1000 MC sims.

Parallelized: each (mode, sizing) combo runs in a separate process.
Workers load shared data from pickle cache (no large DataFrame pickling).

Usage:
    PYTHONPATH="src;." python experiments/optimal_sizing_search.py
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

# Module-level cache for worker processes
_WORKER_CACHE = {}


def build_signals():
    """Load cached data and build S8 merged signals."""
    with open(CACHE_DIR / "v2_prep_cache.pkl", "rb") as f:
        cached = pickle.load(f)
    hold_sigs = cached["hold_sigs"]
    vol_series = cached["vol_series"]
    asset_volumes = cached["asset_volumes"]

    months = pd.period_range("2020-07", "2026-02", freq="M")
    CORE = OBVIOUS_CORE

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
        active_core = [a for a in CORE if a in hold_sigs]
        full = _regime_sel(ms)
        return active_core + [a for a in full if a not in CORE]

    merged = assemble_signals(hold_sigs, _s8_sel, months)
    return merged, vol_series


def run_sim_get_daily(merged, vol_series, base_pct, corr_thresh=0.90,
                      corr_direction="both", corr_scale=0.5):
    """Run one full sim, return daily return series."""
    port = PortfolioConfig(
        starting_capital=CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
        max_exposure=0.90, max_per_asset=0.06, max_per_strategy=0.50,
        base_sizing_pct=base_pct, max_signal_mult=2.0,
        sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)

    prepared = merged.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in prepared.columns: prepared[c] = 1.0
        prepared[c] = prepared[c].fillna(1.0)
    prepared["probability"] = prepared["probability"].astype(float)
    prepared = apply_vol(prepared.sort_values("timestamp").reset_index(drop=True), vol_series)

    breaker_params = {"window": 10, "threshold": -0.01, "floor": 0.10,
                      "up_threshold": 0.01, "cap": 1.5}
    improvements = {"correlation_throttle": corr_thresh,
                    "correlation_direction": corr_direction,
                    "correlation_scale": corr_scale}

    sim = simulate_with_improvements(prepared, port, breaker_params, improvements)
    if sim is None:
        return None

    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None:
        return None
    if dpnl.index.tz is not None:
        dpnl.index = dpnl.index.tz_localize(None)

    dr = dpnl / CAP
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if dr.std() > 0 else 0
    eq = (1 + dr).cumprod()
    dd_s = (eq.cummax() - eq) / eq.cummax()
    max_dd = float(dd_s.max() * 100)
    pnl = float(dpnl.sum())
    daily_dd = float((dpnl / (CAP + dpnl.cumsum()).shift(1).fillna(CAP) * 100).min())
    monthly = dpnl.resample("MS").sum()

    return {
        "sr": sr, "pnl": pnl, "max_dd": max_dd, "daily_dd": daily_dd,
        "n_trades": sim["n_trades"],
        "pnl_mo": pnl / max(len(monthly), 1),
        "dr": dr.values,
    }


def mc_vectorized(dr_values, dd_limit, day_limit, n_sims=N_MC):
    """Fully vectorized block bootstrap MC. Returns breach counts."""
    rng = np.random.default_rng(SEED)
    n_days = len(dr_values)
    block_size = 5
    n_blocks = n_days // block_size

    # Build all bootstrap paths at once
    block_starts = rng.integers(0, n_days - block_size, size=(n_sims, n_blocks))
    # Vectorized block construction
    offsets = np.arange(block_size)
    all_indices = (block_starts[:, :, None] + offsets[None, None, :]).reshape(n_sims, -1)
    boot = dr_values[all_indices]

    cum = np.cumprod(1.0 + boot, axis=1)
    rmax = np.maximum.accumulate(cum, axis=1)
    dds = (rmax - cum) / rmax
    max_dds = dds.max(axis=1) * 100
    worst_days = boot.min(axis=1) * 100

    return {
        "n_dd": int((max_dds > dd_limit).sum()),
        "n_day": int((worst_days < -day_limit).sum()),
        "worst_dd": float(max_dds.max()),
        "worst_day": float(worst_days.min()),
        "passes": bool((max_dds <= dd_limit).all() and (worst_days >= -day_limit).all()),
    }


def _worker_run(prepared_path, base_pct, dd_limit, day_limit,
                corr_thresh, corr_direction, corr_scale):
    """Worker function for ProcessPoolExecutor. Loads data from pickle."""
    warnings.filterwarnings("ignore")

    # Load prepared signals (cached, fast)
    if prepared_path not in _WORKER_CACHE:
        with open(prepared_path, "rb") as f:
            _WORKER_CACHE[prepared_path] = pickle.load(f)
    prepared = _WORKER_CACHE[prepared_path]

    port = PortfolioConfig(
        starting_capital=CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
        max_exposure=0.90, max_per_asset=0.06, max_per_strategy=0.50,
        base_sizing_pct=base_pct, max_signal_mult=2.0,
        sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)

    breaker_params = {"window": 10, "threshold": -0.01, "floor": 0.10,
                      "up_threshold": 0.01, "cap": 1.5}
    improvements = {"correlation_throttle": corr_thresh,
                    "correlation_direction": corr_direction,
                    "correlation_scale": corr_scale}

    sim = simulate_with_improvements(prepared, port, breaker_params, improvements)
    if sim is None:
        return None, None

    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None:
        return None, None
    if dpnl.index.tz is not None:
        dpnl.index = dpnl.index.tz_localize(None)

    dr = dpnl / CAP
    if dr.std() == 0 or len(dr) < 30:
        return None, None

    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    eq = (1 + dr).cumprod()
    dd_s = (eq.cummax() - eq) / eq.cummax()
    max_dd = float(dd_s.max() * 100)
    pnl = float(dpnl.sum())
    daily_dd = float((dpnl / (CAP + dpnl.cumsum()).shift(1).fillna(CAP) * 100).min())
    monthly = dpnl.resample("MS").sum()

    mc = mc_vectorized(dr.values, dd_limit, day_limit)

    summary = {"sr": sr, "pnl": pnl, "max_dd": max_dd, "daily_dd": daily_dd,
               "n_trades": sim["n_trades"], "pnl_mo": pnl / max(len(monthly), 1)}
    return summary, mc


def main():
    T0 = time.time()
    print("=" * 115)
    print(f"  OPTIMAL SIZING — short-only vs both CorrThrottle ({N_MC} MC sims)")
    print("=" * 115)

    merged, vol_series = build_signals()
    print(f"  {len(merged):,} signals loaded ({time.time()-T0:.0f}s)")

    DD_LIM = 10.0   # full prop firm limit
    DAY_LIM = 5.0
    sizing_range = np.arange(0.045, 0.096, 0.001)

    modes = [
        ("No throttle",               "both",  1.00, 1.0),
        ("Corr90 both 50%cut",        "both",  0.90, 0.5),
        ("Corr90 short-only 50%cut",  "short", 0.90, 0.5),
        ("Corr85 short-only 50%cut",  "short", 0.85, 0.5),
        ("Corr80 short-only 50%cut",  "short", 0.80, 0.5),
        ("Corr90 short-only 25%cut",  "short", 0.90, 0.25),
        ("Corr90 short-only 75%cut",  "short", 0.90, 0.75),
    ]

    # Save prepared signals to temp pickle for worker processes
    prepared_path = CACHE_DIR / "_opt_prepared.pkl"
    prepared = merged.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in prepared.columns: prepared[c] = 1.0
        prepared[c] = prepared[c].fillna(1.0)
    prepared["probability"] = prepared["probability"].astype(float)
    prepared = apply_vol(prepared.sort_values("timestamp").reset_index(drop=True), vol_series)
    with open(prepared_path, "wb") as f:
        pickle.dump(prepared, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Prepared signals cached ({len(prepared):,} rows)")

    # Build all work items: coarse 0.5% steps (11 per mode)
    work = []
    for mi, (mode_label, corr_dir, corr_thresh, corr_sc) in enumerate(modes):
        for bp_x10 in range(45, 96, 5):
            work.append((mi, round(bp_x10/1000, 4), corr_dir, corr_thresh, corr_sc))

    print(f"  {len(work)} coarse configs, {os.cpu_count()} CPUs...", flush=True)
    t1 = time.time()

    results_by_mode = {mi: {} for mi in range(len(modes))}

    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count())) as pool:
        futs = {}
        for item in work:
            mi, bp, cd, ct, cs = item
            fut = pool.submit(_worker_run, str(prepared_path), bp,
                              DD_LIM, DAY_LIM, ct, cd, cs)
            futs[fut] = (mi, bp)

        done = 0
        for fut in as_completed(futs):
            mi, bp = futs[fut]
            summary, mc = fut.result()
            if summary is not None:
                results_by_mode[mi][bp] = (summary, mc)
            done += 1
            if done % 20 == 0:
                print(f"    {done}/{len(work)} ({time.time()-t1:.0f}s)", flush=True)

    print(f"  Coarse sweep done ({time.time()-t1:.0f}s)")

    # Fine sweep around each mode's boundary
    fine_work = []
    for mi, (mode_label, corr_dir, corr_thresh, corr_sc) in enumerate(modes):
        last_pass = None
        for bp in sorted(results_by_mode[mi].keys()):
            summary, mc = results_by_mode[mi][bp]
            if mc["passes"]:
                last_pass = bp
        if last_pass is None:
            continue
        for bp_x100 in range(int(last_pass*1000)-4, int(last_pass*1000)+6):
            bp = round(bp_x100/1000, 4)
            if bp < 0.04 or bp > 0.10 or bp in results_by_mode[mi]:
                continue
            fine_work.append((mi, bp, corr_dir, corr_thresh, corr_sc))

    if fine_work:
        print(f"  {len(fine_work)} fine configs...", flush=True)
        t2 = time.time()
        with ProcessPoolExecutor(max_workers=min(8, os.cpu_count())) as pool:
            futs = {}
            for item in fine_work:
                mi, bp, cd, ct, cs = item
                fut = pool.submit(_worker_run, str(prepared_path), bp,
                                  DD_LIM, DAY_LIM, ct, cd, cs)
                futs[fut] = (mi, bp)
            for fut in as_completed(futs):
                mi, bp = futs[fut]
                summary, mc = fut.result()
                if summary is not None:
                    results_by_mode[mi][bp] = (summary, mc)
        print(f"  Fine sweep done ({time.time()-t2:.0f}s)")

    # Clean up temp file
    prepared_path.unlink(missing_ok=True)

    # Find best passing config per mode
    all_bests = []
    for mi, (mode_label, *_) in enumerate(modes):
        best = None
        for bp in sorted(results_by_mode[mi].keys()):
            summary, mc = results_by_mode[mi][bp]
            if mc["passes"]:
                best = (bp, summary, mc)
        if best:
            bp, r, mc = best
            all_bests.append((mode_label, bp, r, mc))
            print(f"  {mode_label:<30} -> {bp*100:.1f}% | SR={r['sr']:.2f}"
                  f" | ${r['pnl']:,.0f} | DD={mc['worst_dd']:.1f}%")
        else:
            print(f"  {mode_label:<30} -> NO PASSING CONFIG")

    print(f"\n{'=' * 115}")
    print(f"  FINAL RANKING by PnL (MaxDD < {DD_LIM}%, DailyDD > -{DAY_LIM}%)")
    print(f"{'=' * 115}")
    print(f"  {'Mode':<30} {'Size':>5} {'SR':>5} {'PnL':>11} {'$/mo':>9}"
          f" {'BT DD':>6} {'MC wDD':>7} {'MC wDay':>8}")
    print("  " + "-" * 95)

    for mode_label, bp, r, mc in sorted(all_bests, key=lambda x: -x[2]["pnl"]):
        print(f"  {mode_label:<30} {bp*100:>4.1f}% {r['sr']:>5.2f}"
              f" ${r['pnl']:>9,.0f} ${r['pnl_mo']:>8,.0f}"
              f" {r['max_dd']:>5.1f}% {mc['worst_dd']:>6.1f}%"
              f" {mc['worst_day']:>7.2f}%")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
