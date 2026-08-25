"""T16 tweak grid: push the best feasible config further."""
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

def run(C_str, thr, cr, fwd, lev, dlv):
    tp = tc = tt = ttd = 0
    wd = wdy = 0.0
    pw = []
    for wl, cd in W:
        cf = CACHE_DIR / f"C{C_str}_lb16_{wl}.pkl"
        if not cf.exists():
            return None
        with open(cf, "rb") as f:
            raw = pickle.load(f)
        fi = af(raw, thr, fwd, cr)
        if len(fi) < 5:
            pw.append((wl, 0, 0, 0, 0))
            continue
        co = orig.apply_nco_wf(fi)
        pc = PortfolioConfig(
            starting_capital=10000.0, max_dd=0.10, delever_start=dlv,
            slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
            base_sizing_pct=0.04 * lev, max_signal_mult=3.0,
        )
        s = simulate_portfolio(co, pc)
        if s is None:
            pw.append((wl, 0, 0, 0, 0))
            continue
        dp = compute_daily_pnl_series(s)
        if dp is None or len(dp) < 1:
            pw.append((wl, 0, 0, 0, 0))
            continue
        eq = pd.Series(s["equity"][~s["skipped"]],
                       index=pd.to_datetime(s["timestamps"][~s["skipped"]]))
        dd = ((eq - eq.expanding().max()) / eq.expanding().max())
        dr = dp / pc.starting_capital
        pnl = s["equity"][-1] - 10000
        pw.append((wl, pnl, s["n_trades"], abs(dd.min()), abs(dr.min())))
        tp += pnl; tc += cd; tt += s["n_trades"]; ttd += len(dp)
        wd = max(wd, abs(dd.min())); wdy = max(wdy, abs(dr.min()))
    if tc == 0:
        return None
    pcal = tp / tc
    d2t = int(1000 / pcal) if pcal > 0 else 9999
    act = ttd / tc * 100
    ok = wd < 0.10 and wdy < 0.05
    return pcal, d2t, wd, wdy, tt, act, ok, pw

# T16 base: C=0.01928, thr=0.08, lb=16, fixed, fwd=4, lev=2.0
# Also try nearby C values that are cached
cached_c_lb16 = []
for f in sorted(CACHE_DIR.glob("C*_lb16_W1.pkl")):
    c_str = f.stem.split("_")[0][1:]  # e.g. "0.01928"
    cached_c_lb16.append(c_str)

print("T16 TWEAKS + NEARBY C VALUES (lb=16)")
print(f"Cached C values for lb=16: {cached_c_lb16}")
print()
print(f"{'Config':<35s} {'$/cal':>6s} {'Days':>5s} {'DD%':>6s} {'WD%':>6s} {'Act%':>5s} {'OK':>4s}")
print("-" * 70)

best = None
configs = []

# T16 variants
for c_str in cached_c_lb16:
    for thr in [0.06, 0.08, 0.10, 0.12]:
        for cr in ["fixed", "decay_T2", "decay_T3"]:
            for fwd in [4, 6, 8]:
                for lev in [1.5, 2.0, 2.5]:
                    for dlv in [0.07, 0.08, 0.085, 0.09]:
                        configs.append((c_str, thr, cr, fwd, lev, dlv))

# Run all and collect feasible
feasible = []
for c_str, thr, cr, fwd, lev, dlv in configs:
    r = run(c_str, thr, cr, fwd, lev, dlv)
    if r is None:
        continue
    pcal, d2t, dd, wdy, tt, act, ok, pw = r
    if ok:
        label = f"C={c_str} t={thr} {cr} f={fwd} l={lev} d={dlv}"
        feasible.append((label, pcal, d2t, dd, wdy, act, tt))

# Sort by PnL/cal_day
feasible.sort(key=lambda x: -x[1])

print(f"\nFEASIBLE CONFIGS: {len(feasible)} out of {len(configs)} tested")
print()
print(f"{'Config':<55s} {'$/cal':>6s} {'Days':>5s} {'DD%':>6s} {'WD%':>6s} {'Act%':>5s}")
print("-" * 85)

for label, pcal, d2t, dd, wdy, act, tt in feasible[:25]:
    print(f"{label:<55s} {pcal:>6.2f} {d2t:>5d} {dd*100:>5.1f}% {wdy*100:>5.1f}% {act:>4.0f}%")

if feasible:
    b = feasible[0]
    print(f"\nBEST: {b[0]}")
    print(f"  ${b[1]:.2f}/cal_day, ~{b[2]} days to 10%, DD={b[3]:.1%}, worst_day={b[4]:.1%}, activity={b[5]:.0f}%")
