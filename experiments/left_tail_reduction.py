#!/usr/bin/env python
"""Left tail reduction experiment — find methods that let us size bigger.

Tests 6 methods individually + combos with CorrThrottle, finds max sizing
at 10%/5% DD limits for each.

Parallelized: ~20 configs per method × ProcessPoolExecutor.

Usage:
    PYTHONPATH="src;." python experiments/left_tail_reduction.py
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
DD_LIM = 10.0
DAY_LIM = 5.0

_CACHE = {}


def mc_vectorized(dr_values, dd_limit=DD_LIM, day_limit=DAY_LIM, n_sims=N_MC):
    rng = np.random.default_rng(SEED)
    n_days = len(dr_values)
    block_size = 5
    n_blocks = n_days // block_size
    block_starts = rng.integers(0, n_days - block_size, size=(n_sims, n_blocks))
    offsets = np.arange(block_size)
    all_idx = (block_starts[:, :, None] + offsets[None, None, :]).reshape(n_sims, -1)
    boot = dr_values[all_idx]
    cum = np.cumprod(1.0 + boot, axis=1)
    rmax = np.maximum.accumulate(cum, axis=1)
    dds = (rmax - cum) / rmax
    max_dds = dds.max(axis=1) * 100
    worst_days = boot.min(axis=1) * 100
    return {
        "worst_dd": float(max_dds.max()),
        "worst_day": float(worst_days.min()),
        "passes": bool((max_dds <= dd_limit).all() and (worst_days >= -day_limit).all()),
    }


def _worker(prepared_path, base_pct, max_per_asset, improvements):
    """Worker: run sim + MC, return summary."""
    warnings.filterwarnings("ignore")
    if prepared_path not in _CACHE:
        with open(prepared_path, "rb") as f:
            _CACHE[prepared_path] = pickle.load(f)
    prepared = _CACHE[prepared_path]

    port = PortfolioConfig(
        starting_capital=CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
        max_exposure=0.90, max_per_asset=max_per_asset, max_per_strategy=0.50,
        base_sizing_pct=base_pct, max_signal_mult=2.0,
        sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)

    breaker_params = {"window": 10, "threshold": -0.01, "floor": 0.10,
                      "up_threshold": 0.01, "cap": 1.5}

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

    mc = mc_vectorized(dr.values)

    return {
        "sr": sr, "pnl": pnl, "max_dd": max_dd, "daily_dd": daily_dd,
        "n_trades": sim["n_trades"], "pnl_mo": pnl / max(len(monthly), 1),
    }, mc


def find_max_sizing(prepared_path, improvements, max_per_asset=0.06, label=""):
    """Binary search: coarse (0.5%) then fine (0.1%) for max passing sizing."""
    # Coarse: 4.5% to 9.5% in 0.5% steps
    coarse = [(round(x/1000, 4), max_per_asset, improvements)
              for x in range(45, 96, 5)]

    last_pass = None
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count())) as pool:
        futs = {pool.submit(_worker, prepared_path, bp, mpa, imp): bp
                for bp, mpa, imp in coarse}
        for fut in as_completed(futs):
            bp = futs[fut]
            summary, mc = fut.result()
            if summary is not None and mc is not None and mc["passes"]:
                if last_pass is None or bp > last_pass:
                    last_pass = bp

    if last_pass is None:
        return None

    # Fine: around boundary
    fine_start = int(last_pass * 1000) - 4
    fine_end = int(last_pass * 1000) + 6
    best = None

    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count())) as pool:
        futs = {}
        for bp_x100 in range(fine_start, fine_end):
            bp = round(bp_x100 / 1000, 4)
            if bp < 0.04 or bp > 0.10:
                continue
            fut = pool.submit(_worker, prepared_path, bp, max_per_asset, improvements)
            futs[fut] = bp
        for fut in as_completed(futs):
            bp = futs[fut]
            summary, mc = fut.result()
            if summary is not None and mc is not None and mc["passes"]:
                best = (bp, summary, mc)

    return best


def main():
    T0 = time.time()
    print("=" * 110)
    print(f"  LEFT TAIL REDUCTION — find methods that unlock more PnL")
    print(f"  Constraint: MaxDD < {DD_LIM}%, DailyDD > -{DAY_LIM}% in {N_MC} MC sims")
    print("=" * 110)

    # Build and cache prepared signals
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
    prepared = merged.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in prepared.columns: prepared[c] = 1.0
        prepared[c] = prepared[c].fillna(1.0)
    prepared["probability"] = prepared["probability"].astype(float)
    prepared = apply_vol(prepared.sort_values("timestamp").reset_index(drop=True), vol_series)

    pp = CACHE_DIR / "_lt_prepared.pkl"
    with open(pp, "wb") as f:
        pickle.dump(prepared, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  {len(prepared):,} signals cached ({time.time()-T0:.0f}s)")

    # ── Methods to test ──
    # All include CorrThrottle90 as baseline (proven essential)
    CT90 = {"correlation_throttle": 0.90, "correlation_direction": "both", "correlation_scale": 0.5}

    methods = [
        # Reference
        ("CorrThrottle90 only",        0.06, {**CT90}),
        # 1. Lower max_per_asset
        ("CT90 + maxAsset=4%",          0.04, {**CT90}),
        ("CT90 + maxAsset=5%",          0.05, {**CT90}),
        ("CT90 + maxAsset=3%",          0.03, {**CT90}),
        # 2. Max signals per bar
        ("CT90 + maxSig=10/bar",        0.06, {**CT90, "max_signals_per_bar": 10}),
        ("CT90 + maxSig=15/bar",        0.06, {**CT90, "max_signals_per_bar": 15}),
        ("CT90 + maxSig=20/bar",        0.06, {**CT90, "max_signals_per_bar": 20}),
        # 3. Vol regime hard cut
        ("CT90 + volCut=0.015",         0.06, {**CT90, "vol_regime_cut": 0.015}),
        ("CT90 + volCut=0.020",         0.06, {**CT90, "vol_regime_cut": 0.020}),
        # 4. Rolling Sharpe gate
        ("CT90 + SharpeGate",           0.06, {**CT90, "rolling_sharpe_gate": True}),
        # 5. Max daily notional
        ("CT90 + dailyNot=30%",         0.06, {**CT90, "max_daily_notional_pct": 0.30}),
        ("CT90 + dailyNot=50%",         0.06, {**CT90, "max_daily_notional_pct": 0.50}),
        # 6. Combos
        ("CT90+maxAsset4%+maxSig15",    0.04, {**CT90, "max_signals_per_bar": 15}),
        ("CT90+maxAsset4%+SharpeGate",  0.04, {**CT90, "rolling_sharpe_gate": True}),
        ("CT90+maxAsset5%+maxSig15",    0.05, {**CT90, "max_signals_per_bar": 15}),
        ("CT90+mA4%+mS15+SG",          0.04, {**CT90, "max_signals_per_bar": 15, "rolling_sharpe_gate": True}),
    ]

    print(f"\n  Testing {len(methods)} methods (coarse+fine search each)...\n")
    print(f"  {'Method':<32} {'Size':>5} {'SR':>5} {'PnL':>10} {'$/mo':>8}"
          f" {'MaxDD':>6} {'MC wDD':>7} {'MC wDay':>8}")
    print("  " + "-" * 95)

    results = []
    for label, max_pa, improvements in methods:
        t1 = time.time()
        best = find_max_sizing(str(pp), improvements, max_per_asset=max_pa)
        elapsed = time.time() - t1

        if best:
            bp, r, mc = best
            results.append((label, bp, r, mc))
            print(f"  {label:<32} {bp*100:>4.1f}% {r['sr']:>5.2f} ${r['pnl']:>8,.0f}"
                  f" ${r['pnl_mo']:>7,.0f} {r['max_dd']:>5.1f}%"
                  f" {mc['worst_dd']:>6.1f}% {mc['worst_day']:>7.2f}%"
                  f"  ({elapsed:.0f}s)")
        else:
            print(f"  {label:<32} NO PASSING CONFIG ({elapsed:.0f}s)")

    # Cleanup
    pp.unlink(missing_ok=True)

    # Ranking
    print(f"\n{'=' * 110}")
    print(f"  RANKING BY PNL")
    print(f"{'=' * 110}")
    results.sort(key=lambda x: -x[2]["pnl"])
    for i, (label, bp, r, mc) in enumerate(results, 1):
        print(f"  {i:>2}. {label:<32} {bp*100:>4.1f}% SR={r['sr']:.2f}"
              f" PnL=${r['pnl']:,.0f} (${r['pnl_mo']:,.0f}/mo)"
              f" DD={mc['worst_dd']:.1f}%")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
