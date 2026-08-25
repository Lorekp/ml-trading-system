"""T15 tweak grid: find feasible variant of the $97/cal_day config."""
import sys, os
_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import pickle, numpy as np, pandas as pd
from pathlib import Path
from signal_engine.config import PortfolioConfig
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
import test_original_pipeline as orig

CACHE_DIR = Path("data/optuna/signal_cache")
W = [("W1", 245), ("W2", 244), ("W3", 122)]
C_VAL = 0.04925

def af(df, thr, fwd, cr):
    if df.empty: return df
    f = df[df["confidence"] > thr].copy()
    if len(f) < 10: return f
    if cr == "fixed":
        f["fwd_return"] = f[f"ret_{fwd}"]
    elif cr == "decay_T2":
        f["fwd_return"] = f.apply(lambda r: r["ret_2"] if r["ret_2"] < 0 else r[f"ret_{fwd}"], axis=1)
    elif cr == "decay_T3":
        f["fwd_return"] = f.apply(lambda r: r["ret_3"] if r["ret_3"] < 0 else r[f"ret_{fwd}"], axis=1)
    return f

def run(thr, cr, fwd, lev, dlv):
    tp = tc = tt = ttd = 0
    wd = wdy = 0.0
    for wl, cd in W:
        cf = CACHE_DIR / f"C{C_VAL:.5f}_lb16_{wl}.pkl"
        if not cf.exists(): return None
        with open(cf, "rb") as f:
            raw = pickle.load(f)
        fi = af(raw, thr, fwd, cr)
        if len(fi) < 10: continue
        co = orig.apply_nco_wf(fi)
        pc = PortfolioConfig(
            starting_capital=10000.0, max_dd=0.10, delever_start=dlv,
            slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
            base_sizing_pct=0.04 * lev, max_signal_mult=3.0,
        )
        s = simulate_portfolio(co, pc)
        if s is None: continue
        dp = compute_daily_pnl_series(s)
        if dp is None or len(dp) < 1: continue
        eq = pd.Series(s["equity"][~s["skipped"]],
                       index=pd.to_datetime(s["timestamps"][~s["skipped"]]))
        dd = ((eq - eq.expanding().max()) / eq.expanding().max())
        dr = dp / pc.starting_capital
        tp += s["equity"][-1] - 10000
        tc += cd; tt += s["n_trades"]; ttd += len(dp)
        wd = max(wd, abs(dd.min()))
        wdy = max(wdy, abs(dr.min()))
    if tc == 0: return None
    pcal = tp / tc
    d2t = int(1000 / pcal) if pcal > 0 else 9999
    act = ttd / tc * 100
    ok = wd < 0.10 and wdy < 0.05
    return pcal, d2t, wd, wdy, tt, act, ok

configs = [
    ("T15 original",         0.04, "fixed",   8, 2.0, 0.07),
    ("lev=1.9",              0.04, "fixed",   8, 1.9, 0.07),
    ("lev=1.8",              0.04, "fixed",   8, 1.8, 0.07),
    ("lev=1.7",              0.04, "fixed",   8, 1.7, 0.07),
    ("lev=1.5",              0.04, "fixed",   8, 1.5, 0.07),
    ("dlv=0.06",             0.04, "fixed",   8, 2.0, 0.06),
    ("dlv=0.065",            0.04, "fixed",   8, 2.0, 0.065),
    ("dlv=0.05",             0.04, "fixed",   8, 2.0, 0.05),
    ("thr=0.05",             0.05, "fixed",   8, 2.0, 0.07),
    ("thr=0.06",             0.06, "fixed",   8, 2.0, 0.07),
    ("fwd=6",                0.04, "fixed",   6, 2.0, 0.07),
    ("fwd=4",                0.04, "fixed",   4, 2.0, 0.07),
    ("decay_T2",             0.04, "decay_T2",8, 2.0, 0.07),
    ("decay_T3",             0.04, "decay_T3",8, 2.0, 0.07),
    ("lev=1.9+dlv=0.065",   0.04, "fixed",   8, 1.9, 0.065),
    ("lev=1.8+dlv=0.06",    0.04, "fixed",   8, 1.8, 0.06),
    ("lev=1.9+thr=0.05",    0.05, "fixed",   8, 1.9, 0.07),
    ("lev=1.8+thr=0.05",    0.05, "fixed",   8, 1.8, 0.07),
    ("decay_T2+lev=1.8",    0.04, "decay_T2",8, 1.8, 0.07),
    ("decay_T3+lev=1.9",    0.04, "decay_T3",8, 1.9, 0.07),
    ("thr=0.05+lev=1.9",    0.05, "fixed",   8, 1.9, 0.07),
    ("thr=0.05+dlv=0.065",  0.05, "fixed",   8, 2.0, 0.065),
    ("thr=0.06+lev=1.9",    0.06, "fixed",   8, 1.9, 0.07),
    ("thr=0.05+fwd=6",      0.05, "fixed",   6, 2.0, 0.07),
]

print("T15 TWEAKS (C=0.04925, lb=16)")
print(f"{'Config':<30s} {'$/cal':>6s} {'Days':>5s} {'DD%':>6s} {'WD%':>6s} {'Act%':>5s} {'OK':>4s}")
print("-" * 65)

best_feasible = None
for label, thr, cr, fwd, lev, dlv in configs:
    r = run(thr, cr, fwd, lev, dlv)
    if r is None:
        print(f"{label:<30s} NO CACHE")
        continue
    pcal, d2t, dd, wdy, tt, act, ok = r
    marker = "YES" if ok else "no"
    print(f"{label:<30s} {pcal:>6.2f} {d2t:>5d} {dd*100:>5.1f}% {wdy*100:>5.1f}% {act:>4.0f}% {marker:>4s}")
    if ok and (best_feasible is None or pcal > best_feasible[1]):
        best_feasible = (label, pcal, d2t, dd, act)

if best_feasible:
    print(f"\nBEST FEASIBLE: {best_feasible[0]}")
    print(f"  ${best_feasible[1]:.2f}/cal_day, ~{best_feasible[2]} days, DD={best_feasible[3]:.1%}, act={best_feasible[4]:.0f}%")
else:
    print("\nNo feasible tweaks found")
