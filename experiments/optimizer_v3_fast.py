"""Optimizer v3 FAST: flat-bet screening + top-N portfolio sim.

Step 1: Sweep ALL filter combos with flat-bet stats (<0.01s each) — screens 1,674+ combos in seconds
Step 2: Run portfolio sim only on top 200 by flat-bet PnL/cal_day (~200s)
Step 3: Scale leverage up/down analytically to meet DD constraints
Step 4: Sort by effective PnL/calendar_day
"""

import sys, os, time, warnings, pickle
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import numpy as np, pandas as pd
from signal_engine.config import PortfolioConfig
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
CACHE_DIR = Path("data/optuna/signal_cache")
W = [("W1", 245), ("W2", 244), ("W3", 122)]
REF_LEV = 2.0
DD_LIMIT = 0.10
DAILY_DD_LIMIT = 0.05
TOP_N = 200  # only run portfolio sim on top N

THRESHOLDS = [0.04, 0.05, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
CLOSE_RULES = ["fixed", "decay_T2", "decay_T3"]
FWD_BARS = [2, 4, 6, 8]


def discover_cached():
    combos = set()
    for f in CACHE_DIR.glob("C*_lb*_W1.pkl"):
        parts = f.stem.split("_")
        c_str = parts[0][1:]
        lb = int(parts[1][2:])
        if all((CACHE_DIR / f"C{c_str}_lb{lb}_{wl}.pkl").exists() for wl, _ in W):
            combos.add((c_str, lb))
    return sorted(combos)


def load_cached(c_str, lb):
    dfs = {}
    for wl, cd in W:
        with open(CACHE_DIR / f"C{c_str}_lb{lb}_{wl}.pkl", "rb") as f:
            dfs[wl] = pickle.load(f)
    return dfs


def apply_filters(df, thr, fwd, cr):
    if df.empty: return df
    f = df[df["confidence"] > thr]
    if len(f) < 5: return f
    # Vectorized close rule (no .apply lambda)
    if cr == "fixed":
        fwd_ret = f[f"ret_{fwd}"].values
    elif cr == "decay_T2":
        ret2 = f["ret_2"].values
        retN = f[f"ret_{fwd}"].values
        fwd_ret = np.where(ret2 < 0, ret2, retN)
    elif cr == "decay_T3":
        ret3 = f["ret_3"].values
        retN = f[f"ret_{fwd}"].values
        fwd_ret = np.where(ret3 < 0, ret3, retN)
    else:
        fwd_ret = f[f"ret_{fwd}"].values
    f = f.copy()
    f["fwd_return"] = fwd_ret
    return f


def flat_bet_cal_pnl(window_dfs, thr, cr, fwd):
    """Ultra-fast screening: flat-bet PnL per calendar day across all windows."""
    total_pnl_bps = 0
    total_cal = 0
    total_signals = 0
    total_trading_days = 0

    for (wl, cd), raw in zip(W, [window_dfs.get("W1"), window_dfs.get("W2"), window_dfs.get("W3")]):
        if raw is None or raw.empty: continue
        filtered = apply_filters(raw, thr, fwd, cr)
        if len(filtered) < 5: continue

        pnls = filtered["direction"].values * filtered["fwd_return"].values - 0.0001
        ts = pd.to_datetime(filtered["timestamp"], utc=True)
        daily = pd.Series(pnls, index=ts.values).resample("1D").sum()
        daily = daily[daily != 0]

        total_pnl_bps += pnls.sum() * 5.0  # $5 flat bet
        total_cal += cd
        total_signals += len(filtered)
        total_trading_days += len(daily)

    if total_cal == 0 or total_signals < 20:
        return None

    return {
        "pnl_cal": total_pnl_bps / total_cal,
        "n_signals": total_signals,
        "activity": total_trading_days / total_cal * 100,
    }


def _portfolio_eval_one(args):
    """Worker for parallel portfolio eval. Picklable top-level function."""
    c_str, lb, thr, cr, fwd, flat_pnl = args
    import sys, os, warnings, pickle
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
    sys.path.insert(0, "src"); sys.path.insert(0, "experiments")
    warnings.filterwarnings("ignore")
    import numpy as np, pandas as pd
    from signal_engine.config import PortfolioConfig
    from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series

    W_local = [("W1", 245), ("W2", 244), ("W3", 122)]
    CACHE = Path("data/optuna/signal_cache")
    REF = 2.0

    total_pnl = tc = tt = ttd = 0
    mdd = mwd = 0.0

    for wl, cd in W_local:
        cf = CACHE / f"C{c_str}_lb{lb}_{wl}.pkl"
        if not cf.exists(): continue
        with open(cf, "rb") as f: raw = pickle.load(f)
        if raw.empty: continue
        fi = raw[raw["confidence"] > thr]
        if len(fi) < 5: continue
        fi = fi.copy()
        if cr == "fixed": fi["fwd_return"] = fi[f"ret_{fwd}"].values
        elif cr == "decay_T2": fi["fwd_return"] = np.where(fi["ret_2"].values < 0, fi["ret_2"].values, fi[f"ret_{fwd}"].values)
        elif cr == "decay_T3": fi["fwd_return"] = np.where(fi["ret_3"].values < 0, fi["ret_3"].values, fi[f"ret_{fwd}"].values)
        if len(fi) < 10: continue

        import test_original_pipeline as orig
        combined = orig.apply_nco_wf(fi)
        pcfg = PortfolioConfig(
            starting_capital=10000.0, max_dd=0.20, delever_start=0.15,
            slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
            base_sizing_pct=0.04 * REF, max_signal_mult=3.0,
        )
        sim = simulate_portfolio(combined, pcfg)
        if sim is None or sim["n_trades"] < 5: continue
        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None or len(dpnl) < 1: continue
        eq = pd.Series(sim["equity"][~sim["skipped"]], index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
        dd = ((eq - eq.expanding().max()) / eq.expanding().max())
        dr = dpnl / pcfg.starting_capital
        total_pnl += sim["equity"][-1] - 10000; tc += cd; tt += sim["n_trades"]; ttd += len(dpnl)
        mdd = max(mdd, abs(dd.min())); mwd = max(mwd, abs(dr.min()))

    if tc == 0 or total_pnl <= 0: return None
    rpnl = total_pnl / tc; act = ttd / tc * 100
    dds = 0.10 / mdd if mdd > 0 else 10.0
    dys = 0.05 / mwd if mwd > 0 else 10.0
    sc = min(dds, dys)
    epnl = rpnl * sc; olev = REF * sc
    d2t = int(1000 / epnl) if epnl > 0 else 9999
    bind = "DD" if dds < dys else "DayDD"
    return {
        "c_str": c_str, "lb": lb, "thr": thr, "cr": cr, "fwd": fwd,
        "flat_pnl": flat_pnl, "raw_pnl_cal": rpnl, "effective_pnl": epnl,
        "optimal_lev": olev, "days_to_10": d2t, "raw_dd": mdd, "raw_worst_day": mwd,
        "scale": sc, "binding": bind, "trades": tt, "activity": act,
    }


def main():
    t0 = time.time()
    print("=" * 105)
    print("OPTIMIZER v3 FAST: flat-bet screen + top-N portfolio sim + leverage scaling")
    print(f"  Ref leverage: {REF_LEV}x, DD limit: {DD_LIMIT:.0%}, Daily DD: {DAILY_DD_LIMIT:.0%}")
    print(f"  Top {TOP_N} screened configs get full portfolio sim")
    print("=" * 105)

    combos = discover_cached()
    n_filter = len(THRESHOLDS) * len(CLOSE_RULES) * len(FWD_BARS)
    total = len(combos) * n_filter
    print(f"\n  Cached combos: {len(combos)}, Filter combos: {n_filter}, Total: {total}")

    # STEP 1: Flat-bet screening
    print(f"\n  Step 1: Flat-bet screening ({total} combos)...", flush=True)
    t1 = time.time()
    screen_results = []

    for c_str, lb in combos:
        wdfs = load_cached(c_str, lb)
        for thr in THRESHOLDS:
            for cr in CLOSE_RULES:
                for fwd in FWD_BARS:
                    r = flat_bet_cal_pnl(wdfs, thr, cr, fwd)
                    if r is not None and r["pnl_cal"] > 0:
                        screen_results.append({
                            "c_str": c_str, "lb": lb, "thr": thr, "cr": cr, "fwd": fwd,
                            **r,
                        })

    screen_results.sort(key=lambda x: -x["pnl_cal"])
    print(f"    {len(screen_results)} viable configs in {time.time()-t1:.1f}s")

    # STEP 2: Portfolio sim on top N
    top_n = screen_results[:TOP_N]
    print(f"\n  Step 2: Portfolio sim on top {len(top_n)} configs...", flush=True)
    t2 = time.time()

    # Build args for parallel execution
    eval_args = [
        (s["c_str"], s["lb"], s["thr"], s["cr"], s["fwd"], s["pnl_cal"])
        for s in top_n
    ]

    from concurrent.futures import ProcessPoolExecutor, as_completed
    N_WORKERS = min(8, os.cpu_count() or 4)
    print(f"    Using {N_WORKERS} parallel workers", flush=True)

    final_results = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
        futures = {pool.submit(_portfolio_eval_one, a): a for a in eval_args}
        done = 0
        for future in as_completed(futures):
            done += 1
            r = future.result()
            if r is not None:
                final_results.append(r)
            if done % 50 == 0:
                print(f"    {done}/{len(eval_args)} done", flush=True)

    final_results.sort(key=lambda x: -x["effective_pnl"])
    print(f"    {len(final_results)} results in {time.time()-t2:.0f}s", flush=True)

    # RESULTS
    print(f"\n{'='*105}")
    print(f"TOP 30 CONFIGS (by effective PnL/calendar_day after leverage scaling)")
    print(f"{'='*105}")
    print(f"\n  {'#':>3s} {'EffPnL':>7s} {'~Days':>5s} {'OptLev':>6s} {'RawPnL':>7s} {'RawDD':>6s} {'RawWD':>6s} {'Scale':>5s} {'Bind':>5s} "
          f"{'Act%':>5s} {'C':>8s} {'Thr':>5s} {'LB':>3s} {'Close':>8s} {'Fwd':>4s}")
    print("  " + "-" * 105)

    for i, r in enumerate(final_results[:30]):
        print(f"  {i+1:>3d} ${r['effective_pnl']:>6.2f} {r['days_to_10']:>5d} {r['optimal_lev']:>5.2f}x "
              f"${r['raw_pnl_cal']:>6.2f} {r['raw_dd']*100:>5.1f}% {r['raw_worst_day']*100:>5.1f}% "
              f"{r['scale']:>5.2f} {r['binding']:>5s} "
              f"{r['activity']:>4.0f}% {r['c_str']:>8s} {r['thr']:>5.2f} {r['lb']:>3d} "
              f"{r['cr']:>8s} {r['fwd']:>4d}")

    if final_results:
        b = final_results[0]
        print(f"\n  BEST CONFIG:")
        print(f"    C={b['c_str']}, threshold={b['thr']}, lookback={b['lb']}, "
              f"close={b['cr']}, fwd_bars={b['fwd']}")
        print(f"    Optimal leverage: {b['optimal_lev']:.2f}x")
        print(f"    Effective PnL/cal_day: ${b['effective_pnl']:.2f}")
        print(f"    Days to 10%: ~{b['days_to_10']}")
        print(f"    Raw DD at {REF_LEV}x: {b['raw_dd']:.1%}, Raw worst day: {b['raw_worst_day']:.1%}")
        print(f"    Scale factor: {b['scale']:.2f} (binding: {b['binding']})")
        print(f"    Activity: {b['activity']:.0f}%, Trades: {b['trades']}")

    dd_bound = sum(1 for r in final_results if r["binding"] == "DD")
    day_bound = sum(1 for r in final_results if r["binding"] == "DayDD")
    print(f"\n  Binding: DD={dd_bound}, DayDD={day_bound}")
    print(f"  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
