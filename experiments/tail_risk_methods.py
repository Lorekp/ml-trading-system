#!/usr/bin/env python
"""Test all tail risk reduction methods on CT90 base.

Methods: CPPI, TUW, Kurtosis, SharpeGate, maxSig, combos.
Parallelized coarse+fine sizing search per method.

Usage:
    PYTHONPATH="src;." python experiments/tail_risk_methods.py
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

CT90 = {"correlation_throttle": 0.90, "correlation_direction": "both", "correlation_scale": 0.5}


def mc_check(dr_values):
    rng = np.random.default_rng(SEED)
    n = len(dr_values)
    bs = 5
    nb = n // bs
    starts = rng.integers(0, n - bs, size=(N_MC, nb))
    off = np.arange(bs)
    idx = (starts[:, :, None] + off[None, None, :]).reshape(N_MC, -1)
    boot = dr_values[idx]
    cum = np.cumprod(1.0 + boot, axis=1)
    rm = np.maximum.accumulate(cum, axis=1)
    dds = ((rm - cum) / rm).max(axis=1) * 100
    days = boot.min(axis=1) * 100
    return {
        "worst_dd": float(dds.max()),
        "worst_day": float(days.min()),
        "passes": bool((dds <= DD_LIM).all() and (days >= -DAY_LIM).all()),
    }


def _worker(pp, bp, mpa, imp):
    warnings.filterwarnings("ignore")
    if pp not in _CACHE:
        with open(pp, "rb") as f:
            _CACHE[pp] = pickle.load(f)
    prepared = _CACHE[pp]

    port = PortfolioConfig(
        starting_capital=CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
        max_exposure=0.90, max_per_asset=mpa, max_per_strategy=0.50,
        base_sizing_pct=bp, max_signal_mult=2.0, sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)

    breaker = {"window": 10, "threshold": -0.01, "floor": 0.10,
               "up_threshold": 0.01, "cap": 1.5}

    sim = simulate_with_improvements(prepared, port, breaker, imp)
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
    pnl = float(dpnl.sum())
    monthly = dpnl.resample("MS").sum()

    mc = mc_check(dr.values)

    return {
        "sr": sr, "pnl": pnl,
        "max_dd": float(dd_s.max() * 100),
        "daily_dd": float((dpnl / (CAP + dpnl.cumsum()).shift(1).fillna(CAP) * 100).min()),
        "n_trades": sim["n_trades"],
        "pnl_mo": pnl / max(len(monthly), 1),
    }, mc


def find_best(pp, imp, mpa=0.06):
    """Coarse+fine search for max passing sizing."""
    # Coarse: 0.5% steps
    last_pass = None
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count())) as pool:
        futs = {pool.submit(_worker, pp, round(x/1000, 4), mpa, imp): round(x/1000, 4)
                for x in range(45, 96, 5)}
        for f in as_completed(futs):
            bp = futs[f]
            s, mc = f.result()
            if s and mc and mc["passes"]:
                if last_pass is None or bp > last_pass:
                    last_pass = bp

    if last_pass is None:
        return None

    # Fine: 0.1% steps around boundary
    best = None
    with ProcessPoolExecutor(max_workers=min(8, os.cpu_count())) as pool:
        futs = {}
        for x in range(int(last_pass*1000)-4, int(last_pass*1000)+6):
            bp = round(x/1000, 4)
            if 0.04 <= bp <= 0.10:
                futs[pool.submit(_worker, pp, bp, mpa, imp)] = bp
        for f in as_completed(futs):
            bp = futs[f]
            s, mc = f.result()
            if s and mc and mc["passes"]:
                best = (bp, s, mc)

    return best


def main():
    T0 = time.time()
    print("=" * 110)
    print(f"  TAIL RISK METHODS — all on CT90 base ({N_MC} MC, DD<{DD_LIM}%, Day<{DAY_LIM}%)")
    print("=" * 110)

    # Build and cache signals
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

    pp = str(CACHE_DIR / "_tr_prepared.pkl")
    with open(pp, "wb") as f:
        pickle.dump(prepared, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  {len(prepared):,} signals ({time.time()-T0:.0f}s)\n")

    # ── Methods ──
    methods = [
        # Reference
        ("CT90 only",                  0.06, {**CT90}),
        ("CT90+maxSig15",              0.06, {**CT90, "max_signals_per_bar": 15}),
        # CPPI variants
        ("CT90+CPPI M=1.5",           0.06, {**CT90, "cppi_multiplier": 1.5}),
        ("CT90+CPPI M=2.0",           0.06, {**CT90, "cppi_multiplier": 2.0}),
        ("CT90+CPPI M=2.5",           0.06, {**CT90, "cppi_multiplier": 2.5}),
        ("CT90+CPPI M=3.0",           0.06, {**CT90, "cppi_multiplier": 3.0}),
        # TUW variants
        ("CT90+TUW5d",                0.06, {**CT90, "tuw_threshold_days": 5}),
        ("CT90+TUW10d",               0.06, {**CT90, "tuw_threshold_days": 10}),
        ("CT90+TUW5d fast",           0.06, {**CT90, "tuw_threshold_days": 5, "tuw_decay_rate": 0.08}),
        # Kurtosis variants
        ("CT90+Kurt>3",               0.06, {**CT90, "kurtosis_threshold": 3.0}),
        ("CT90+Kurt>5",               0.06, {**CT90, "kurtosis_threshold": 5.0}),
        ("CT90+Kurt>7",               0.06, {**CT90, "kurtosis_threshold": 7.0}),
        # SharpeGate
        ("CT90+SharpeGate",           0.06, {**CT90, "rolling_sharpe_gate": True}),
        # Combos: CPPI + others
        ("CT90+CPPI2+TUW5",           0.06, {**CT90, "cppi_multiplier": 2.0, "tuw_threshold_days": 5}),
        ("CT90+CPPI2+Kurt5",          0.06, {**CT90, "cppi_multiplier": 2.0, "kurtosis_threshold": 5.0}),
        ("CT90+CPPI2+TUW5+Kurt5",     0.06, {**CT90, "cppi_multiplier": 2.0, "tuw_threshold_days": 5, "kurtosis_threshold": 5.0}),
        ("CT90+CPPI2+mS15",           0.06, {**CT90, "cppi_multiplier": 2.0, "max_signals_per_bar": 15}),
        ("CT90+CPPI2+mS15+TUW5",      0.06, {**CT90, "cppi_multiplier": 2.0, "max_signals_per_bar": 15, "tuw_threshold_days": 5}),
        ("CT90+CPPI2+SG",             0.06, {**CT90, "cppi_multiplier": 2.0, "rolling_sharpe_gate": True}),
        # Kitchen sink
        ("CT90+CPPI2+mS15+TUW5+K5",   0.06, {**CT90, "cppi_multiplier": 2.0, "max_signals_per_bar": 15,
                                               "tuw_threshold_days": 5, "kurtosis_threshold": 5.0}),
    ]

    print(f"  Testing {len(methods)} methods...\n")
    print(f"  {'Method':<32} {'Size':>5} {'SR':>5} {'PnL':>10} {'$/mo':>8}"
          f" {'BT DD':>6} {'MC wDD':>7} {'MC wDay':>8}")
    print("  " + "-" * 95)

    results = []
    for label, mpa, imp in methods:
        t1 = time.time()
        best = find_best(pp, imp, mpa)
        el = time.time() - t1
        if best:
            bp, r, mc = best
            results.append((label, bp, r, mc))
            print(f"  {label:<32} {bp*100:>4.1f}% {r['sr']:>5.2f} ${r['pnl']:>8,.0f}"
                  f" ${r['pnl_mo']:>7,.0f} {r['max_dd']:>5.1f}%"
                  f" {mc['worst_dd']:>6.1f}% {mc['worst_day']:>7.2f}%  ({el:.0f}s)")
        else:
            print(f"  {label:<32} NO PASSING CONFIG ({el:.0f}s)")

    # Cleanup
    Path(pp).unlink(missing_ok=True)

    # Ranking
    print(f"\n{'=' * 110}")
    print("  RANKING BY PNL")
    print(f"{'=' * 110}")
    results.sort(key=lambda x: -x[2]["pnl"])
    for i, (label, bp, r, mc) in enumerate(results, 1):
        delta = r["pnl"] - (results[-1][2]["pnl"] if results else 0)
        print(f"  {i:>2}. {label:<32} {bp*100:>4.1f}% SR={r['sr']:.2f}"
              f" ${r['pnl']:,.0f} (${r['pnl_mo']:,.0f}/mo)"
              f" DD={mc['worst_dd']:.1f}%")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
