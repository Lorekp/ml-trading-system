"""Optuna v6: 10K trials, 10 dimensions, max PnL/calendar_day.

Every tunable parameter exposed. Uses all 114 cached (C, lb) combos.
~3s per trial × 10,000 = ~8 hours.
"""

import sys, os, time, warnings, pickle
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import numpy as np, pandas as pd, optuna
from signal_engine.config import PortfolioConfig
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

CACHE_DIR = Path("data/optuna/signal_cache")
W = [("W1", 245), ("W2", 244), ("W3", 122)]

# Discover and pre-load
CACHED_COMBOS = {}
for f in sorted(CACHE_DIR.glob("C*_lb*_W1.pkl")):
    parts = f.stem.split("_"); c_str = parts[0][1:]; lb = int(parts[1][2:])
    if all((CACHE_DIR / f"C{c_str}_lb{lb}_{wl}.pkl").exists() for wl, _ in W):
        CACHED_COMBOS[f"{c_str}_lb{lb}"] = (c_str, lb)
COMBO_KEYS = sorted(CACHED_COMBOS.keys())

CACHE_DATA = {}
t_load = time.time()
for key, (c_str, lb) in CACHED_COMBOS.items():
    dfs = {}
    for wl, cd in W:
        with open(CACHE_DIR / f"C{c_str}_lb{lb}_{wl}.pkl", "rb") as f:
            dfs[wl] = pickle.load(f)
    CACHE_DATA[key] = dfs
print(f"Loaded {len(COMBO_KEYS)} combos in {time.time()-t_load:.0f}s")


def evaluate(combo_key, thr, cr, fwd, lev, dlv, max_exp, max_pa, slip, sizing_base):
    dfs = CACHE_DATA[combo_key]
    tp = tc = tt = ttd = 0; mdd = mwd = 0.0

    for (wl, cd) in W:
        raw = dfs.get(wl)
        if raw is None or raw.empty: continue
        fi = raw[raw["confidence"] > thr]
        if len(fi) < 10: continue
        fi = fi.copy()
        if cr == "fixed": fi["fwd_return"] = fi[f"ret_{fwd}"].values
        elif cr == "decay_T2": fi["fwd_return"] = np.where(fi["ret_2"].values < 0, fi["ret_2"].values, fi[f"ret_{fwd}"].values)
        elif cr == "decay_T3": fi["fwd_return"] = np.where(fi["ret_3"].values < 0, fi["ret_3"].values, fi[f"ret_{fwd}"].values)

        combined = orig.apply_nco_wf(fi)
        pcfg = PortfolioConfig(
            starting_capital=10000.0, max_dd=0.10, delever_start=dlv,
            slippage=slip, max_exposure=max_exp, max_per_asset=max_pa,
            base_sizing_pct=sizing_base * lev, max_signal_mult=3.0,
        )
        sim = simulate_portfolio(combined, pcfg)
        if sim is None or sim["n_trades"] < 5: continue
        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None or len(dpnl) < 1: continue
        eq = pd.Series(sim["equity"][~sim["skipped"]],
                       index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
        dd = ((eq - eq.expanding().max()) / eq.expanding().max())
        dr = dpnl / 10000.0
        tp += sim["equity"][-1] - 10000; tc += cd; tt += sim["n_trades"]; ttd += len(dpnl)
        mdd = max(mdd, abs(dd.min())); mwd = max(mwd, abs(dr.min()))

    if tc == 0 or tp <= 0: return None
    return {"pnl_cal": tp / tc, "max_dd": mdd, "worst_day": mwd,
            "trades": tt, "activity": ttd / tc * 100,
            "d2t": int(1000 / (tp / tc)) if tp > 0 else 9999}


def constraints(trial):
    return [
        trial.user_attrs.get("max_dd", 1.0) - 0.10,
        trial.user_attrs.get("worst_day", 1.0) - 0.05,
    ]


def objective(trial):
    combo = trial.suggest_categorical("combo", COMBO_KEYS)
    thr = trial.suggest_float("threshold", 0.04, 0.20, step=0.01)
    cr = trial.suggest_categorical("close_rule", ["fixed", "decay_T2", "decay_T3"])
    fwd = trial.suggest_categorical("fwd_bars", [2, 4, 6, 8])
    lev = trial.suggest_float("leverage", 0.5, 10.0, step=0.5)
    dlv = trial.suggest_float("delever_start", 0.03, 0.095, step=0.005)
    max_exp = trial.suggest_float("max_exposure", 0.30, 0.95, step=0.05)
    max_pa = trial.suggest_float("max_per_asset", 0.03, 0.10, step=0.01)
    slip = trial.suggest_categorical("slippage", [0.0001, 0.0002, 0.0005])
    sizing = trial.suggest_categorical("sizing_mode", ["base", "aggressive"])
    sizing_base = 0.04 if sizing == "base" else 0.06

    r = evaluate(combo, thr, cr, fwd, lev, dlv, max_exp, max_pa, slip, sizing_base)
    if r is None:
        trial.set_user_attr("max_dd", 1.0); trial.set_user_attr("worst_day", 1.0)
        return float("-inf")

    trial.set_user_attr("max_dd", r["max_dd"])
    trial.set_user_attr("worst_day", r["worst_day"])
    trial.set_user_attr("trades", r["trades"])
    trial.set_user_attr("activity", r["activity"])
    trial.set_user_attr("d2t", r["d2t"])
    return r["pnl_cal"]


def main():
    n_trials = 10000
    t0 = time.time()
    print("=" * 110)
    print(f"OPTUNA v6: 10K TRIALS, 10 DIMENSIONS, MAX PnL/CAL_DAY")
    print(f"  {len(COMBO_KEYS)} combos, {n_trials} trials")
    print(f"  Params: combo, threshold(0.01 step), close, fwd, leverage(0.5-10), delever(0.03-0.095),")
    print(f"          max_exposure(0.30-0.95), max_per_asset(0.03-0.10), slippage, sizing_mode")
    print(f"  Constraints: DD<10%, DayDD<5%")
    print("=" * 110)

    storage = "sqlite:///data/optuna/study_v6.db"
    Path("data/optuna").mkdir(parents=True, exist_ok=True)

    try:
        sampler = optuna.samplers.GPSampler(constraints_func=constraints)
    except Exception:
        sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=30)

    study = optuna.create_study(
        study_name="max_pnl_v6", storage=storage, direction="maximize",
        sampler=sampler, load_if_exists=True,
    )

    # Seed T165 and variants
    if len(study.trials) == 0:
        seeds = [
            # T165 exact
            {"combo": "0.02000_lb2", "threshold": 0.10, "close_rule": "fixed", "fwd_bars": 6,
             "leverage": 5.0, "delever_start": 0.05, "max_exposure": 0.90, "max_per_asset": 0.05,
             "slippage": 0.0001, "sizing_mode": "base"},
            # T165 aggressive sizing
            {"combo": "0.02000_lb2", "threshold": 0.10, "close_rule": "fixed", "fwd_bars": 6,
             "leverage": 5.0, "delever_start": 0.05, "max_exposure": 0.90, "max_per_asset": 0.05,
             "slippage": 0.0001, "sizing_mode": "aggressive"},
            # T165 with higher exposure
            {"combo": "0.02000_lb2", "threshold": 0.10, "close_rule": "fixed", "fwd_bars": 6,
             "leverage": 5.0, "delever_start": 0.05, "max_exposure": 0.95, "max_per_asset": 0.08,
             "slippage": 0.0001, "sizing_mode": "base"},
            # T165 with tighter delever
            {"combo": "0.02000_lb2", "threshold": 0.10, "close_rule": "fixed", "fwd_bars": 6,
             "leverage": 5.0, "delever_start": 0.08, "max_exposure": 0.90, "max_per_asset": 0.05,
             "slippage": 0.0001, "sizing_mode": "base"},
            # lb=2 decay_T2 fwd=2 (short hold)
            {"combo": "0.02000_lb2", "threshold": 0.10, "close_rule": "decay_T2", "fwd_bars": 2,
             "leverage": 5.0, "delever_start": 0.05, "max_exposure": 0.90, "max_per_asset": 0.05,
             "slippage": 0.0001, "sizing_mode": "base"},
            # lb=4 variant
            {"combo": "0.10000_lb4", "threshold": 0.10, "close_rule": "fixed", "fwd_bars": 6,
             "leverage": 5.0, "delever_start": 0.05, "max_exposure": 0.90, "max_per_asset": 0.05,
             "slippage": 0.0001, "sizing_mode": "base"},
        ]
        for s in seeds:
            study.enqueue_trial(s)
        print(f"  Seeded {len(seeds)} configs")

    completed_before = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    print(f"  Resuming from {completed_before} completed trials\n", flush=True)

    last_print = [0]
    best_so_far = [float("-inf")]

    def callback(study, trial):
        if trial.state != optuna.trial.TrialState.COMPLETE: return
        n = len([t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE])

        ok = trial.user_attrs.get("max_dd", 1) < 0.10 and trial.user_attrs.get("worst_day", 1) < 0.05
        is_new_best = ok and trial.value > best_so_far[0]
        if is_new_best: best_so_far[0] = trial.value

        # Print every 100 trials or on new best
        if n % 100 == 0 or is_new_best:
            m = "NEW BEST" if is_new_best else ("F" if ok else "X")
            d2t = trial.user_attrs.get("d2t", 9999)
            act = trial.user_attrs.get("activity", 0)
            p = trial.params
            print(f"  [{m:>8s}] T{trial.number} (#{n}): ${trial.value:.2f}/cal, ~{d2t}d, "
                  f"DD={trial.user_attrs.get('max_dd',0):.1%}, WD={trial.user_attrs.get('worst_day',0):.2%}, "
                  f"act={act:.0f}%, lev={p['leverage']:.1f}, dlv={p['delever_start']:.3f}, "
                  f"exp={p['max_exposure']:.2f}, pa={p['max_per_asset']:.2f}, "
                  f"slip={p['slippage']}, sz={p['sizing_mode']}, "
                  f"{p['combo']}, thr={p['threshold']:.2f}, {p['close_rule']}, fwd={p['fwd_bars']}",
                  flush=True)

    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    # Results
    print(f"\n{'='*110}")
    all_t = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    feasible = [t for t in all_t if t.user_attrs.get("max_dd", 1) < 0.10 and t.user_attrs.get("worst_day", 1) < 0.05]
    feasible.sort(key=lambda t: t.value, reverse=True)
    print(f"Completed: {len(all_t)}, Feasible: {len(feasible)}")

    if feasible:
        print(f"\nTOP 20 FEASIBLE:")
        print(f"  {'#':>3s} {'$/cal':>7s} {'~Days':>5s} {'DD':>5s} {'WD':>6s} {'Act':>4s} {'Lev':>4s} {'Dlv':>5s} "
              f"{'Exp':>4s} {'PA':>4s} {'Slip':>6s} {'Sz':>4s} {'Combo':<18s} {'Thr':>4s} {'Cl':>8s} {'Fwd':>3s}")
        print("  " + "-" * 115)
        for i, t in enumerate(feasible[:20]):
            p = t.params
            print(f"  {i+1:>3d} ${t.value:>6.2f} {t.user_attrs.get('d2t',9999):>5d} "
                  f"{t.user_attrs['max_dd']:>4.1%} {t.user_attrs['worst_day']:>5.2%} "
                  f"{t.user_attrs.get('activity',0):>3.0f}% {p['leverage']:>4.1f} {p['delever_start']:>5.3f} "
                  f"{p['max_exposure']:>4.2f} {p['max_per_asset']:>4.2f} {p['slippage']:>6.4f} "
                  f"{p['sizing_mode']:>4s} {p['combo']:<18s} {p['threshold']:>4.2f} "
                  f"{p['close_rule']:>8s} {p['fwd_bars']:>3d}")

        b = feasible[0]
        print(f"\nBEST: ${b.value:.2f}/cal_day, ~{b.user_attrs.get('d2t',9999)} days to 10%")
        for k, v in b.params.items():
            print(f"  {k}: {v}")

    try:
        imp = optuna.importance.get_param_importances(study)
        print(f"\nPARAMETER IMPORTANCE:")
        for k, v in sorted(imp.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v:.3f}")
    except Exception: pass

    print(f"\nTotal: {time.time()-t0:.0f}s ({len(all_t)} trials, {len(all_t)/(time.time()-t0):.1f} trials/s)")


if __name__ == "__main__":
    main()
