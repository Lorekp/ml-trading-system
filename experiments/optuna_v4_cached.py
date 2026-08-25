"""Optuna v4: Cache-only search — every trial instant.

Uses ONLY cached (C, lb) combos. No signal generation needed.
Each trial: load cache → filter → NCO → portfolio sim → ~3s.
300 trials in ~15 minutes.

Objective: maximize PnL per CALENDAR day.
Constraints: max DD < 10%, daily DD < 5%.
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

# Discover all cached (C_str, lb) combos
CACHED_COMBOS = {}  # (c_str, lb) -> {W1: df, W2: df, W3: df}
for f in sorted(CACHE_DIR.glob("C*_lb*_W1.pkl")):
    parts = f.stem.split("_")
    c_str = parts[0][1:]
    lb = int(parts[1][2:])
    if all((CACHE_DIR / f"C{c_str}_lb{lb}_{wl}.pkl").exists() for wl, _ in W):
        key = f"{c_str}_lb{lb}"
        CACHED_COMBOS[key] = (c_str, lb)

COMBO_KEYS = sorted(CACHED_COMBOS.keys())
print(f"Found {len(COMBO_KEYS)} cached (C, lb) combos")

# Pre-load ALL cached data into memory for speed
CACHE_DATA = {}
t_load = time.time()
for key, (c_str, lb) in CACHED_COMBOS.items():
    dfs = {}
    for wl, cd in W:
        with open(CACHE_DIR / f"C{c_str}_lb{lb}_{wl}.pkl", "rb") as f:
            dfs[wl] = pickle.load(f)
    CACHE_DATA[key] = dfs
print(f"Pre-loaded all data in {time.time()-t_load:.0f}s")


def evaluate(combo_key, thr, cr, fwd, lev, dlv):
    """Full eval: filter → NCO → portfolio sim → scale leverage."""
    dfs = CACHE_DATA[combo_key]
    total_pnl = tc = tt = ttd = 0
    mdd = mwd = 0.0

    for (wl, cd) in W:
        raw = dfs.get(wl)
        if raw is None or raw.empty: continue

        # Vectorized filter
        fi = raw[raw["confidence"] > thr]
        if len(fi) < 10: continue
        fi = fi.copy()
        if cr == "fixed":
            fi["fwd_return"] = fi[f"ret_{fwd}"].values
        elif cr == "decay_T2":
            fi["fwd_return"] = np.where(fi["ret_2"].values < 0, fi["ret_2"].values, fi[f"ret_{fwd}"].values)
        elif cr == "decay_T3":
            fi["fwd_return"] = np.where(fi["ret_3"].values < 0, fi["ret_3"].values, fi[f"ret_{fwd}"].values)

        # NCO allocation
        combined = orig.apply_nco_wf(fi)

        pcfg = PortfolioConfig(
            starting_capital=10000.0, max_dd=0.10, delever_start=dlv,
            slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
            base_sizing_pct=0.04 * lev, max_signal_mult=3.0,
        )
        sim = simulate_portfolio(combined, pcfg)
        if sim is None or sim["n_trades"] < 5: continue

        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None or len(dpnl) < 1: continue

        eq = pd.Series(sim["equity"][~sim["skipped"]],
                       index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
        dd = ((eq - eq.expanding().max()) / eq.expanding().max())
        dr = dpnl / pcfg.starting_capital

        total_pnl += sim["equity"][-1] - 10000
        tc += cd; tt += sim["n_trades"]; ttd += len(dpnl)
        mdd = max(mdd, abs(dd.min()))
        mwd = max(mwd, abs(dr.min()))

    if tc == 0 or total_pnl <= 0:
        return None

    return {
        "pnl_cal": total_pnl / tc,
        "max_dd": mdd,
        "worst_day": mwd,
        "trades": tt,
        "activity": ttd / tc * 100,
        "d2t": int(1000 / (total_pnl / tc)) if total_pnl > 0 else 9999,
    }


def constraints(trial):
    return [
        trial.user_attrs.get("max_dd", 1.0) - 0.10,
        trial.user_attrs.get("worst_day", 1.0) - 0.05,
    ]


def objective(trial):
    combo_key = trial.suggest_categorical("combo", COMBO_KEYS)
    thr = trial.suggest_float("threshold", 0.04, 0.20, step=0.02)
    cr = trial.suggest_categorical("close_rule", ["fixed", "decay_T2", "decay_T3"])
    fwd = trial.suggest_categorical("fwd_bars", [2, 4, 6, 8])
    lev = trial.suggest_float("leverage", 0.5, 5.0, step=0.5)
    dlv = trial.suggest_float("delever_start", 0.05, 0.095, step=0.005)

    r = evaluate(combo_key, thr, cr, fwd, lev, dlv)

    if r is None:
        trial.set_user_attr("max_dd", 1.0)
        trial.set_user_attr("worst_day", 1.0)
        return float("-inf")

    trial.set_user_attr("max_dd", r["max_dd"])
    trial.set_user_attr("worst_day", r["worst_day"])
    trial.set_user_attr("trades", r["trades"])
    trial.set_user_attr("activity", r["activity"])
    trial.set_user_attr("d2t", r["d2t"])

    return r["pnl_cal"]


def main():
    t0 = time.time()
    n_trials = 1500
    print("=" * 105)
    print(f"OPTUNA v4: CACHE-ONLY (every trial instant)")
    print(f"  {len(COMBO_KEYS)} (C,lb) combos, {n_trials} trials")
    print(f"  Objective: max PnL/calendar_day, constraints: DD<10%, DayDD<5%")
    print("=" * 105)

    storage = f"sqlite:///data/optuna/study_v4.db"
    Path("data/optuna").mkdir(parents=True, exist_ok=True)

    try:
        sampler = optuna.samplers.GPSampler(constraints_func=constraints)
    except Exception:
        sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=20)

    study = optuna.create_study(
        study_name="max_pnl_v4",
        storage=storage,
        direction="maximize",
        sampler=sampler,
        load_if_exists=True,
    )

    # Seed T16 and variants
    if len(study.trials) == 0:
        for key in COMBO_KEYS:
            if "0.01928" in key or "0.01000" in key:
                for thr in [0.06, 0.08, 0.10]:
                    study.enqueue_trial({
                        "combo": key, "threshold": thr, "close_rule": "fixed",
                        "fwd_bars": 4, "leverage": 2.0, "delever_start": 0.085,
                    })
        print(f"  Seeded {len(study.trials)} configs")

    def callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            ok = trial.user_attrs.get("max_dd", 1) < 0.10 and trial.user_attrs.get("worst_day", 1) < 0.05
            m = "F" if ok else "X"
            d2t = trial.user_attrs.get("d2t", 9999)
            act = trial.user_attrs.get("activity", 0)
            p = trial.params
            combo = p["combo"]
            print(f"  [{m}] T{trial.number}: ${trial.value:.2f}/cal, ~{d2t}d, "
                  f"DD={trial.user_attrs.get('max_dd',0):.1%}, "
                  f"act={act:.0f}%, lev={p['leverage']:.1f}, dlv={p['delever_start']:.3f}, "
                  f"{combo}, thr={p['threshold']:.2f}, {p['close_rule']}, fwd={p['fwd_bars']}",
                  flush=True)

    print(f"\nRunning {n_trials} trials...\n", flush=True)
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    # Results
    print(f"\n{'='*105}")
    all_t = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    feasible = [t for t in all_t if t.user_attrs.get("max_dd", 1) < 0.10 and t.user_attrs.get("worst_day", 1) < 0.05]
    feasible.sort(key=lambda t: t.value, reverse=True)
    print(f"Completed: {len(all_t)}, Feasible: {len(feasible)}")

    if feasible:
        print(f"\nTOP 20 FEASIBLE:")
        print(f"  {'#':>3s} {'$/cal':>6s} {'~Days':>5s} {'DD':>5s} {'WD':>5s} {'Act':>4s} {'Lev':>4s} {'Dlv':>5s} {'Combo':<25s} {'Thr':>4s} {'Close':>8s} {'Fwd':>4s}")
        print("  " + "-" * 95)
        for i, t in enumerate(feasible[:20]):
            p = t.params
            print(f"  {i+1:>3d} ${t.value:>5.2f} {t.user_attrs.get('d2t',9999):>5d} "
                  f"{t.user_attrs['max_dd']:>4.1%} {t.user_attrs['worst_day']:>4.1%} "
                  f"{t.user_attrs.get('activity',0):>3.0f}% {p['leverage']:>4.1f} "
                  f"{p['delever_start']:>5.3f} {p['combo']:<25s} {p['threshold']:>4.2f} "
                  f"{p['close_rule']:>8s} {p['fwd_bars']:>4d}")

        b = feasible[0]
        print(f"\nBEST: ${b.value:.2f}/cal_day, ~{b.user_attrs.get('d2t',9999)} days to 10%")
        for k, v in b.params.items():
            print(f"  {k}: {v}")

    try:
        imp = optuna.importance.get_param_importances(study)
        print(f"\nIMPORTANCE:")
        for k, v in sorted(imp.items(), key=lambda x: -x[1]):
            print(f"  {k}: {v:.3f}")
    except Exception: pass

    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
