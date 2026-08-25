"""Optimizer v3: Scale-down approach + exhaustive sweep from cache.

1. Use all cached signals (66+ C×lb combos already generated)
2. Sweep ALL filter combos (threshold × close × fwd) — instant from cache
3. Run portfolio sim at 2x reference leverage
4. Scale DOWN leverage to meet DD constraints
5. Sort by effective PnL/calendar_day (= fastest to 10%)

No Optuna needed — pure exhaustive sweep with analytical leverage scaling.
"""

import sys, os, time, warnings, pickle
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

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
REF_LEV = 2.0  # reference leverage for raw sim
DD_LIMIT = 0.10
DAILY_DD_LIMIT = 0.05

THRESHOLDS = [0.04, 0.06, 0.08, 0.10, 0.12, 0.14]
CLOSE_RULES = ["fixed", "decay_T2", "decay_T3"]
FWD_BARS = [4, 6, 8]


def discover_cached_combos():
    """Find all unique (C_str, lb) combos in the cache."""
    combos = set()
    for f in CACHE_DIR.glob("C*_lb*_W1.pkl"):
        parts = f.stem.split("_")
        c_str = parts[0][1:]  # e.g. "0.01000"
        lb = int(parts[1][2:])  # e.g. 16
        # Check all 3 windows exist
        if all((CACHE_DIR / f"C{c_str}_lb{lb}_{wl}.pkl").exists() for wl, _ in W):
            combos.add((c_str, lb))
    return sorted(combos)


def load_cached(c_str, lb):
    """Load all 3 windows for a (C, lb) combo."""
    dfs = {}
    for wl, cd in W:
        cf = CACHE_DIR / f"C{c_str}_lb{lb}_{wl}.pkl"
        with open(cf, "rb") as f:
            dfs[wl] = pickle.load(f)
    return dfs


def apply_filters(df, thr, fwd, cr):
    if df.empty: return df
    f = df[df["confidence"] > thr].copy()
    if len(f) < 5: return f
    if cr == "fixed":
        f["fwd_return"] = f[f"ret_{fwd}"]
    elif cr == "decay_T2":
        f["fwd_return"] = f.apply(lambda r: r["ret_2"] if r["ret_2"] < 0 else r[f"ret_{fwd}"], axis=1)
    elif cr == "decay_T3":
        f["fwd_return"] = f.apply(lambda r: r["ret_3"] if r["ret_3"] < 0 else r[f"ret_{fwd}"], axis=1)
    return f


def evaluate_combo(c_str, lb, thr, cr, fwd, window_dfs):
    """Run portfolio sim at reference leverage, return raw metrics per calendar day."""
    total_pnl = 0
    total_cal = 0
    total_trades = 0
    total_trading_days = 0
    max_dd = 0
    worst_day = 0

    for (wl, cd), raw in zip(W, [window_dfs.get("W1"), window_dfs.get("W2"), window_dfs.get("W3")]):
        if raw is None or raw.empty:
            continue

        filtered = apply_filters(raw, thr, fwd, cr)
        if len(filtered) < 10:
            continue

        combined = orig.apply_nco_wf(filtered)
        pcfg = PortfolioConfig(
            starting_capital=10000.0, max_dd=0.15, delever_start=0.12,
            slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
            base_sizing_pct=0.04 * REF_LEV, max_signal_mult=3.0,
        )
        sim = simulate_portfolio(combined, pcfg)
        if sim is None or sim["n_trades"] < 5:
            continue

        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None or len(dpnl) < 1:
            continue

        equity = pd.Series(sim["equity"][~sim["skipped"]],
                           index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
        dd = ((equity - equity.expanding().max()) / equity.expanding().max())
        daily_ret = dpnl / pcfg.starting_capital

        total_pnl += sim["equity"][-1] - 10000
        total_cal += cd
        total_trades += sim["n_trades"]
        total_trading_days += len(dpnl)
        max_dd = max(max_dd, abs(dd.min()))
        worst_day = max(worst_day, abs(daily_ret.min()))

    if total_cal == 0 or total_pnl <= 0:
        return None

    raw_pnl_cal = total_pnl / total_cal
    activity = total_trading_days / total_cal * 100

    # Scale-down to meet constraints
    dd_scale = DD_LIMIT / max_dd if max_dd > 0 else 10.0
    day_scale = DAILY_DD_LIMIT / worst_day if worst_day > 0 else 10.0
    scale = min(dd_scale, day_scale)  # most binding constraint (NO cap — scale UP if room)

    effective_pnl = raw_pnl_cal * scale
    optimal_lev = REF_LEV * scale
    days_to_10 = int(1000 / effective_pnl) if effective_pnl > 0 else 9999
    binding = "DD" if dd_scale < day_scale else "DayDD"

    return {
        "c_str": c_str, "lb": lb, "thr": thr, "cr": cr, "fwd": fwd,
        "raw_pnl_cal": raw_pnl_cal,
        "effective_pnl": effective_pnl,
        "optimal_lev": optimal_lev,
        "days_to_10": days_to_10,
        "raw_dd": max_dd, "raw_worst_day": worst_day,
        "scale": scale, "binding": binding,
        "trades": total_trades, "activity": activity,
    }


def main():
    t0 = time.time()
    print("=" * 100)
    print("OPTIMIZER v3: SCALE-DOWN APPROACH")
    print(f"  Reference leverage: {REF_LEV}x")
    print(f"  DD limit: {DD_LIMIT:.0%}, Daily DD limit: {DAILY_DD_LIMIT:.0%}")
    print(f"  Thresholds: {THRESHOLDS}")
    print(f"  Close rules: {CLOSE_RULES}")
    print(f"  Fwd bars: {FWD_BARS}")
    print("=" * 100)

    # Discover cached combos
    combos = discover_cached_combos()
    print(f"\n  Found {len(combos)} cached (C, lookback) combos")

    # Exhaustive sweep
    n_filter_combos = len(THRESHOLDS) * len(CLOSE_RULES) * len(FWD_BARS)
    total_evals = len(combos) * n_filter_combos
    print(f"  Filter combos per (C,lb): {n_filter_combos}")
    print(f"  Total evaluations: {total_evals}")
    print(f"\n  Running...", flush=True)

    results = []
    done = 0
    for c_str, lb in combos:
        window_dfs = load_cached(c_str, lb)

        for thr in THRESHOLDS:
            for cr in CLOSE_RULES:
                for fwd in FWD_BARS:
                    r = evaluate_combo(c_str, lb, thr, cr, fwd, window_dfs)
                    if r is not None:
                        results.append(r)

        done += 1
        if done % 10 == 0:
            print(f"    {done}/{len(combos)} combos processed, {len(results)} viable configs", flush=True)

    print(f"\n  Processed {total_evals} evaluations in {time.time()-t0:.0f}s")
    print(f"  Viable configs: {len(results)}")

    # Sort by effective PnL (= fastest to 10%)
    results.sort(key=lambda x: -x["effective_pnl"])

    # Results
    print(f"\n{'='*100}")
    print("TOP 30 CONFIGS (by effective PnL/calendar_day after scaling)")
    print(f"{'='*100}")
    print(f"\n  {'#':>3s} {'EffPnL':>7s} {'~Days':>5s} {'OptLev':>6s} {'RawPnL':>7s} {'RawDD':>6s} {'RawWD':>6s} {'Scale':>5s} {'Bind':>5s} "
          f"{'Act%':>5s} {'C':>8s} {'Thr':>5s} {'LB':>3s} {'Close':>8s} {'Fwd':>4s}")
    print("  " + "-" * 105)

    for i, r in enumerate(results[:30]):
        print(f"  {i+1:>3d} ${r['effective_pnl']:>6.2f} {r['days_to_10']:>5d} {r['optimal_lev']:>5.2f}x "
              f"${r['raw_pnl_cal']:>6.2f} {r['raw_dd']*100:>5.1f}% {r['raw_worst_day']*100:>5.1f}% "
              f"{r['scale']:>5.2f} {r['binding']:>5s} "
              f"{r['activity']:>4.0f}% {r['c_str']:>8s} {r['thr']:>5.2f} {r['lb']:>3d} "
              f"{r['cr']:>8s} {r['fwd']:>4d}")

    # Best config details
    if results:
        b = results[0]
        print(f"\n  BEST CONFIG:")
        print(f"    C={b['c_str']}, threshold={b['thr']}, lookback={b['lb']}, "
              f"close={b['cr']}, fwd_bars={b['fwd']}")
        print(f"    Optimal leverage: {b['optimal_lev']:.2f}x (scaled from {REF_LEV}x)")
        print(f"    Effective PnL/cal_day: ${b['effective_pnl']:.2f}")
        print(f"    Days to 10%: {b['days_to_10']}")
        print(f"    Raw DD at {REF_LEV}x: {b['raw_dd']:.1%} (binding: {b['binding']})")
        print(f"    Raw worst day at {REF_LEV}x: {b['raw_worst_day']:.1%}")
        print(f"    Activity: {b['activity']:.0f}% of calendar days")
        print(f"    Trades: {b['trades']}")

    # Distribution of binding constraints
    dd_bound = sum(1 for r in results if r["binding"] == "DD")
    day_bound = sum(1 for r in results if r["binding"] == "DayDD")
    print(f"\n  Binding constraint: DD={dd_bound}, DayDD={day_bound}")

    print(f"\n  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
