"""Optuna v5: Minimize CVaR (tail risk) with minimum trade frequency.

Find the safest possible strategy that trades at least 1000x/year.
Then lever the hell out of it.

Objective: minimize CVaR(5%) of daily returns
Constraint: total trades >= 1000/year (across all windows)
Search: all cached (C, lb) combos + filter params
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
from signal_engine.allocation import compute_cvar
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

CACHE_DIR = Path("data/optuna/signal_cache")
W = [("W1", 245), ("W2", 244), ("W3", 122)]
MIN_TRADES_PER_YEAR = 1000
TOTAL_CAL_DAYS = sum(cd for _, cd in W)  # 611 days
MIN_TRADES_TOTAL = int(MIN_TRADES_PER_YEAR * TOTAL_CAL_DAYS / 365.25)  # ~1673

# Discover and pre-load cache
CACHED_COMBOS = {}
for f in sorted(CACHE_DIR.glob("C*_lb*_W1.pkl")):
    parts = f.stem.split("_")
    c_str = parts[0][1:]
    lb = int(parts[1][2:])
    if all((CACHE_DIR / f"C{c_str}_lb{lb}_{wl}.pkl").exists() for wl, _ in W):
        CACHED_COMBOS[f"{c_str}_lb{lb}"] = (c_str, lb)

COMBO_KEYS = sorted(CACHED_COMBOS.keys())
print(f"Found {len(COMBO_KEYS)} cached combos")

CACHE_DATA = {}
t0 = time.time()
for key, (c_str, lb) in CACHED_COMBOS.items():
    dfs = {}
    for wl, cd in W:
        with open(CACHE_DIR / f"C{c_str}_lb{lb}_{wl}.pkl", "rb") as f:
            dfs[wl] = pickle.load(f)
    CACHE_DATA[key] = dfs
print(f"Pre-loaded in {time.time()-t0:.0f}s")


def evaluate(combo_key, thr, cr, fwd):
    """Evaluate at 1x leverage. Returns CVaR, PnL, DD, trades."""
    dfs = CACHE_DATA[combo_key]
    all_daily_rets = []
    total_pnl = 0; total_cal = 0; total_trades = 0; total_days = 0
    max_dd = 0; worst_day = 0

    for (wl, cd) in W:
        raw = dfs.get(wl)
        if raw is None or raw.empty: continue
        fi = raw[raw["confidence"] > thr]
        if len(fi) < 5: continue
        fi = fi.copy()
        if cr == "fixed": fi["fwd_return"] = fi[f"ret_{fwd}"].values
        elif cr == "decay_T2": fi["fwd_return"] = np.where(fi["ret_2"].values < 0, fi["ret_2"].values, fi[f"ret_{fwd}"].values)
        elif cr == "decay_T3": fi["fwd_return"] = np.where(fi["ret_3"].values < 0, fi["ret_3"].values, fi[f"ret_{fwd}"].values)
        if len(fi) < 10: continue

        combined = orig.apply_nco_wf(fi)
        # Run at 1x — we'll lever after finding min CVaR
        pcfg = PortfolioConfig(
            starting_capital=10000.0, max_dd=0.50, delever_start=0.40,
            slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
            base_sizing_pct=0.04, max_signal_mult=3.0,
        )
        sim = simulate_portfolio(combined, pcfg)
        if sim is None or sim["n_trades"] < 5: continue
        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None or len(dpnl) < 1: continue

        daily_ret = dpnl / 10000.0
        all_daily_rets.append(daily_ret)

        eq = pd.Series(sim["equity"][~sim["skipped"]],
                       index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
        dd = ((eq - eq.expanding().max()) / eq.expanding().max())

        total_pnl += sim["equity"][-1] - 10000
        total_cal += cd; total_trades += sim["n_trades"]; total_days += len(dpnl)
        max_dd = max(max_dd, abs(dd.min()))
        worst_day = max(worst_day, abs(daily_ret.min()))

    if total_cal == 0 or not all_daily_rets:
        return None

    combined_rets = pd.concat(all_daily_rets).sort_index()
    cvar = compute_cvar(combined_rets.values, alpha=0.05)
    pnl_cal = total_pnl / total_cal
    activity = total_days / total_cal * 100

    # Compute max leverage that stays within DD limits
    # DD and worst_day scale linearly with leverage
    if max_dd > 0:
        max_lev_dd = 0.10 / max_dd
    else:
        max_lev_dd = 20.0
    if worst_day > 0:
        max_lev_wd = 0.05 / worst_day
    else:
        max_lev_wd = 20.0
    max_safe_lev = min(max_lev_dd, max_lev_wd, 20.0)
    effective_pnl = pnl_cal * max_safe_lev
    d2t = int(1000 / effective_pnl) if effective_pnl > 0 else 9999

    return {
        "cvar": cvar, "pnl_cal_1x": pnl_cal, "max_dd_1x": max_dd,
        "worst_day_1x": worst_day, "trades": total_trades,
        "activity": activity, "max_safe_lev": max_safe_lev,
        "effective_pnl": effective_pnl, "d2t": d2t,
    }


def constraints(trial):
    trades = trial.user_attrs.get("trades", 0)
    return [MIN_TRADES_TOTAL - trades]  # positive = violated


def objective(trial):
    combo_key = trial.suggest_categorical("combo", COMBO_KEYS)
    thr = trial.suggest_float("threshold", 0.04, 0.20, step=0.02)
    cr = trial.suggest_categorical("close_rule", ["fixed", "decay_T2", "decay_T3"])
    fwd = trial.suggest_categorical("fwd_bars", [2, 4, 6, 8])

    r = evaluate(combo_key, thr, cr, fwd)
    if r is None:
        trial.set_user_attr("trades", 0)
        return float("inf")  # minimize CVaR, inf = worst

    trial.set_user_attr("trades", r["trades"])
    trial.set_user_attr("pnl_cal_1x", r["pnl_cal_1x"])
    trial.set_user_attr("max_dd_1x", r["max_dd_1x"])
    trial.set_user_attr("worst_day_1x", r["worst_day_1x"])
    trial.set_user_attr("max_safe_lev", r["max_safe_lev"])
    trial.set_user_attr("effective_pnl", r["effective_pnl"])
    trial.set_user_attr("d2t", r["d2t"])
    trial.set_user_attr("activity", r["activity"])

    return r["cvar"]  # MINIMIZE tail risk


def main():
    n_trials = 1500
    t0 = time.time()
    print("=" * 105)
    print(f"OPTUNA v5: MINIMIZE CVaR (find safest strategy, then lever up)")
    print(f"  Min trades: {MIN_TRADES_PER_YEAR}/year ({MIN_TRADES_TOTAL} total across {TOTAL_CAL_DAYS} days)")
    print(f"  {len(COMBO_KEYS)} combos, {n_trials} trials")
    print(f"  After finding min-CVaR: compute max safe leverage for DD<10%, DayDD<5%")
    print("=" * 105)

    storage = "sqlite:///data/optuna/study_v5.db"
    Path("data/optuna").mkdir(parents=True, exist_ok=True)

    try:
        sampler = optuna.samplers.GPSampler(constraints_func=constraints)
    except Exception:
        sampler = optuna.samplers.TPESampler(multivariate=True, n_startup_trials=20)

    study = optuna.create_study(
        study_name="min_cvar_v5",
        storage=storage,
        direction="minimize",  # MINIMIZE CVaR
        sampler=sampler,
        load_if_exists=True,
    )

    def callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            trades = trial.user_attrs.get("trades", 0)
            enough = trades >= MIN_TRADES_TOTAL
            m = "F" if enough else "X"
            lev = trial.user_attrs.get("max_safe_lev", 0)
            epnl = trial.user_attrs.get("effective_pnl", 0)
            d2t = trial.user_attrs.get("d2t", 9999)
            dd1x = trial.user_attrs.get("max_dd_1x", 0)
            wd1x = trial.user_attrs.get("worst_day_1x", 0)
            act = trial.user_attrs.get("activity", 0)
            p = trial.params
            print(f"  [{m}] T{trial.number}: CVaR={trial.value:.5f}, "
                  f"1x_DD={dd1x:.1%} 1x_WD={wd1x:.2%}, "
                  f"maxLev={lev:.1f}x, effPnL=${epnl:.2f}/cal, ~{d2t}d, "
                  f"trades={trades}, act={act:.0f}%, "
                  f"{p['combo']}, thr={p['threshold']:.2f}, {p['close_rule']}, fwd={p['fwd_bars']}",
                  flush=True)

    print(f"\nRunning {n_trials} trials...\n", flush=True)
    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    # Results
    print(f"\n{'='*105}")
    all_t = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    feasible = [t for t in all_t if t.user_attrs.get("trades", 0) >= MIN_TRADES_TOTAL]
    feasible.sort(key=lambda t: t.value)  # sort by CVaR ascending (lowest = safest)

    print(f"Completed: {len(all_t)}, Feasible (>={MIN_TRADES_TOTAL} trades): {len(feasible)}")

    if feasible:
        print(f"\nTOP 20 SAFEST (lowest CVaR, then levered up):")
        print(f"  {'#':>3s} {'CVaR':>8s} {'1xDD':>5s} {'1xWD':>6s} {'MaxLev':>6s} {'EffPnL':>7s} {'~Days':>5s} "
              f"{'Trades':>6s} {'Act%':>5s} {'Combo':<20s} {'Thr':>4s} {'Close':>8s} {'Fwd':>4s}")
        print("  " + "-" * 105)

        for i, t in enumerate(feasible[:20]):
            p = t.params
            print(f"  {i+1:>3d} {t.value:>8.5f} {t.user_attrs['max_dd_1x']:>4.1%} "
                  f"{t.user_attrs['worst_day_1x']:>5.2%} "
                  f"{t.user_attrs['max_safe_lev']:>5.1f}x "
                  f"${t.user_attrs['effective_pnl']:>6.2f} "
                  f"{t.user_attrs['d2t']:>5d} "
                  f"{t.user_attrs['trades']:>6d} {t.user_attrs['activity']:>4.0f}% "
                  f"{p['combo']:<20s} {p['threshold']:>4.2f} {p['close_rule']:>8s} {p['fwd_bars']:>4d}")

        b = feasible[0]
        print(f"\n  SAFEST CONFIG (lowest CVaR):")
        for k, v in b.params.items():
            print(f"    {k}: {v}")
        print(f"    CVaR(5%): {b.value:.5f}")
        print(f"    1x DD: {b.user_attrs['max_dd_1x']:.1%}, 1x worst day: {b.user_attrs['worst_day_1x']:.2%}")
        print(f"    Max safe leverage: {b.user_attrs['max_safe_lev']:.1f}x")
        print(f"    Effective PnL at max lev: ${b.user_attrs['effective_pnl']:.2f}/cal_day")
        print(f"    Days to 10%: ~{b.user_attrs['d2t']}")
        print(f"    Trades: {b.user_attrs['trades']}")

        # Also show: sorted by effective PnL (best leveraged returns)
        by_pnl = sorted(feasible, key=lambda t: -t.user_attrs.get("effective_pnl", 0))
        print(f"\n  TOP 10 BY LEVERAGED PnL:")
        print(f"  {'#':>3s} {'EffPnL':>7s} {'~Days':>5s} {'MaxLev':>6s} {'CVaR':>8s} {'1xDD':>5s} {'1xWD':>6s}")
        print("  " + "-" * 50)
        for i, t in enumerate(by_pnl[:10]):
            print(f"  {i+1:>3d} ${t.user_attrs['effective_pnl']:>6.2f} "
                  f"{t.user_attrs['d2t']:>5d} "
                  f"{t.user_attrs['max_safe_lev']:>5.1f}x "
                  f"{t.value:>8.5f} "
                  f"{t.user_attrs['max_dd_1x']:>4.1%} "
                  f"{t.user_attrs['worst_day_1x']:>5.2%}")

    try:
        imp = optuna.importance.get_param_importances(study)
        print(f"\n  IMPORTANCE:")
        for k, v in sorted(imp.items(), key=lambda x: -x[1]):
            print(f"    {k}: {v:.3f}")
    except Exception: pass

    print(f"\n  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
