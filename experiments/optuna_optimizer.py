"""Optuna-based parameter optimization for max PnL under DD constraints.

Uses GPSampler with hard constraints (max DD < 10%, daily DD < 5%).
Objective: maximize CVaR-ratio (return / expected shortfall at 5%).
Walk-forward validation across 3 windows to prevent single-period overfit.
Signal cache makes most trials near-instant after initial generation.

Usage:
    python experiments/optuna_optimizer.py [--n-trials 200] [--n-jobs 4]
"""

import sys, os, argparse, time, warnings, hashlib, pickle
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import numpy as np, pandas as pd, sklearn, optuna
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from signal_engine.config import load_config, PortfolioConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from signal_engine.allocation import compute_cvar
from signal_engine.asset_selection import compute_monthly_dollar_volume
from afml.bet_sizing import get_signal, discretize_signal
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
DATA_DIR = Path("data/binance-1m")
TOP_N = 30
MAX_BARS = 8
CACHE_DIR = Path("data/optuna/signal_cache")

# Walk-forward windows: train up to train_end, test from test_start to test_end
WINDOWS = [
    {"label": "W1", "test_start": "2024-07", "test_end": "2025-02"},
    {"label": "W2", "test_start": "2025-03", "test_end": "2025-10"},
    {"label": "W3", "test_start": "2025-11", "test_end": "2026-02"},
]

# Global data (loaded once)
ASSET_DATA = {}
ASSET_VOLUMES = {}
SIGNAL_CACHE = {}


def make_logreg(C):
    if NEW_SKLEARN:
        return LogisticRegression(C=C, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=C, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def load_all_data():
    """Load all asset data once into global dicts."""
    global ASSET_DATA, ASSET_VOLUMES
    if ASSET_DATA:
        return

    cfg = load_config(Path("configs/default.yaml"))
    t0 = time.time()

    for d in sorted(DATA_DIR.iterdir()):
        if not d.is_dir() or not d.name.endswith("USDT"): continue
        symbol = d.name
        try:
            ohlcv_1m = load_binance_1m(symbol, [DATA_DIR])
            ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
            if len(ohlcv_15m) < 5000: continue
            close = ohlcv_15m["close"]
            features = compute_features(ohlcv_15m, cfg.features)
            feat = features[cfg.features.feature_list].copy()
            for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
                if col in feat.columns:
                    feat[col] = feat[col].clip(feat[col].quantile(0.01), feat[col].quantile(0.99))
            fc = feat.dropna()
            fwd_ret = close.pct_change(4).shift(-4)
            y_all = fwd_ret.reindex(fc.index).dropna()
            valid = fc.index.intersection(y_all.index)
            if len(valid) < 3000: continue
            vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

            ASSET_DATA[symbol] = {
                "close": close, "close_arr": ohlcv_15m["close"].values.astype(np.float64),
                "fc": fc, "fwd_ret": fwd_ret, "y_all": y_all, "valid": valid,
                "vol": vol, "timestamps": ohlcv_15m.index,
                "start_date": ohlcv_15m.index.min(), "end_date": ohlcv_15m.index.max(),
                "median_gap": ohlcv_15m.index.to_series().diff().dropna().median(),
            }
            ASSET_VOLUMES[symbol] = compute_monthly_dollar_volume(ohlcv_15m)
        except Exception: pass

    print(f"Loaded {len(ASSET_DATA)} assets in {time.time()-t0:.0f}s")


def cache_key(C, lookback, window_label):
    """Deterministic cache key for (C, lookback, window)."""
    return f"C{C:.5f}_lb{lookback}_{window_label}"


def generate_signals(C, lookback, window):
    """Generate raw signals for a (C, lookback, window) combo. Uses cache."""
    key = cache_key(C, lookback, window["label"])
    if key in SIGNAL_CACHE:
        return SIGNAL_CACHE[key]

    # Check disk cache
    cache_file = CACHE_DIR / f"{key}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            df = pickle.load(f)
        SIGNAL_CACHE[key] = df
        return df

    sample_idx = next(iter(ASSET_VOLUMES.values())).index
    test_start = pd.Timestamp(window["test_start"] + "-01")
    test_end_str = window["test_end"]
    if sample_idx.tz is not None:
        test_start = test_start.tz_localize(sample_idx.tz)

    end_date = max(d["end_date"] for d in ASSET_DATA.values())
    test_end = pd.Timestamp(test_end_str + "-28")
    if sample_idx.tz is not None:
        test_end = test_end.tz_localize(sample_idx.tz)
    test_end = min(test_end, end_date)

    test_months = pd.period_range(start=test_start.to_period("M"),
                                   end=test_end.to_period("M"), freq="M")

    all_signals = []
    for month in test_months:
        ms = pd.Timestamp(month.start_time)
        if sample_idx.tz is not None:
            ms = ms.tz_localize(sample_idx.tz)

        lookback_end = ms - pd.Timedelta(days=1)
        vol_candidates = {}
        for symbol, vol_s in ASSET_VOLUMES.items():
            avail = vol_s.loc[:lookback_end]
            if len(avail) < 45: continue
            latest = avail.iloc[-1]
            if np.isnan(latest) or latest <= 0: continue
            if len(avail) >= 90:
                m90 = avail.iloc[-90:].median()
                if m90 > 0 and latest / m90 > 5.0: continue
            vol_candidates[symbol] = latest
        selected = [s for s, _ in sorted(vol_candidates.items(), key=lambda x: -x[1])[:TOP_N]]

        for symbol in selected:
            if symbol not in ASSET_DATA: continue
            ad = ASSET_DATA[symbol]
            close_arr = ad["close_arr"]; n = len(close_arr)
            fc = ad["fc"]; y_all = ad["y_all"]; valid = ad["valid"]
            vol = ad["vol"]; timestamps_idx = ad["timestamps"]

            ms_local = ms; me_local = ms + pd.offsets.MonthEnd(0)
            if fc.index.tz is None and ms.tzinfo is not None:
                ms_local = ms.tz_localize(None); me_local = me_local.tz_localize(None)

            te_all = valid[(valid >= ms_local) & (valid <= me_local)]
            vol_train_start = ms_local - pd.DateOffset(months=3)
            vol_train = vol.loc[(vol.index >= vol_train_start) & (vol.index < ms_local)].dropna()
            if len(vol_train) < 50: continue
            vol_quantiles = {q: vol_train.quantile(q) for q in [0.0, 0.25, 0.40, 0.50, 0.60, 0.80]}
            test_vol = vol.reindex(te_all)

            train_start = ms_local - pd.DateOffset(months=lookback)
            if train_start < ad["start_date"]: continue
            embargo_td = ad["median_gap"] * 5
            train_cutoff = ms_local - embargo_td
            tr = valid[(valid >= train_start) & (valid < train_cutoff)]
            if len(tr) < 200: continue

            try:
                sw = compute_sample_weights(fc.loc[tr].index.values.astype(np.int64), fwd_bars=4)
                scaler = StandardScaler()
                Xtr = scaler.fit_transform(fc.loc[tr])
                model = make_logreg(C)
                model.fit(Xtr, (y_all.loc[tr] > 0).astype(int), sample_weight=sw)
                Xte = scaler.transform(fc.loc[te_all])
                yp = model.predict_proba(Xte)[:, 1]
            except Exception: continue

            conf = np.abs(yp - 0.5)
            dirs = np.where(yp > 0.5, 1.0, -1.0)
            sigs = get_signal(yp, num_classes=2)
            sd = discretize_signal(sigs, step_size=0.1)
            ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))
            asset_name = symbol.replace("USDT", "")

            for i in range(len(te_all)):
                bar_idx = timestamps_idx.get_loc(te_all[i])
                if bar_idx + MAX_BARS >= n: continue
                ep = close_arr[bar_idx]
                if ep <= 0: continue
                d = dirs[i]
                rets = {k: d * (close_arr[bar_idx + k] - ep) / ep for k in range(1, MAX_BARS + 1)}
                vol_val = float(test_vol.get(te_all[i], np.nan))

                all_signals.append({
                    "timestamp": te_all[i], "direction": d,
                    "confidence": conf[i], "probability": yp[i],
                    "signal_strength": float(ss[i]),
                    "asset": asset_name, "vol_val": vol_val,
                    **{f"ret_{k}": rets[k] for k in range(1, MAX_BARS + 1)},
                    **{f"vol_q_{q}": vol_quantiles[q] for q in vol_quantiles},
                })

    df = pd.DataFrame(all_signals).sort_values("timestamp").reset_index(drop=True) if all_signals else pd.DataFrame()

    # Cache to disk
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f:
        pickle.dump(df, f)
    SIGNAL_CACHE[key] = df
    return df


def apply_filters(df, threshold, vol_q, fwd_bars, close_rule):
    if df.empty: return df
    filtered = df[df["confidence"] > threshold].copy()
    if vol_q > 0:
        q_col = f"vol_q_{vol_q}"
        if q_col in filtered.columns:
            filtered = filtered[filtered["vol_val"] >= filtered[q_col]]
    if len(filtered) < 20: return filtered
    if close_rule == "fixed":
        filtered["fwd_return"] = filtered[f"ret_{fwd_bars}"]
    elif close_rule == "decay_T2":
        filtered["fwd_return"] = filtered.apply(
            lambda r: r["ret_2"] if r["ret_2"] < 0 else r[f"ret_{fwd_bars}"], axis=1)
    elif close_rule == "decay_T3":
        filtered["fwd_return"] = filtered.apply(
            lambda r: r["ret_3"] if r["ret_3"] < 0 else r[f"ret_{fwd_bars}"], axis=1)
    return filtered


def evaluate_window(raw, threshold, vol_q, close_rule, fwd_bars, leverage=1.0):
    """Evaluate one window → return metrics dict or None."""
    filtered = apply_filters(raw, threshold, vol_q, fwd_bars, close_rule)
    if len(filtered) < 30: return None

    combined = orig.apply_nco_wf(filtered)
    pcfg = PortfolioConfig(
        starting_capital=10000.0, max_dd=0.20, delever_start=0.10,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
        base_sizing_pct=0.04 * leverage, max_signal_mult=3.0,
    )
    sim = simulate_portfolio(combined, pcfg)
    if sim is None or sim["n_trades"] < 10: return None

    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 5: return None

    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"],
                               sim["n_trades"] / len(dpnl))

    equity = pd.Series(sim["equity"][~sim["skipped"]],
                       index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
    dd = ((equity - equity.expanding().max()) / equity.expanding().max())
    daily_ret = dpnl / pcfg.starting_capital

    # CVaR at 5%
    cvar = compute_cvar(daily_ret.values, alpha=0.05)
    annual_ret = stats.get("cagr_pct", 0) / 100
    cvar_ratio = annual_ret / max(cvar, 1e-6)

    return {
        "sharpe": stats["sharpe"],
        "calmar": stats.get("calmar", 0),
        "cvar_ratio": cvar_ratio,
        "cvar": cvar,
        "annual_ret": annual_ret,
        "max_dd": abs(dd.min()),
        "worst_day": abs(daily_ret.min()),
        "n_trades": sim["n_trades"],
        "pnl_day": (stats["final_equity"] - pcfg.starting_capital) / max(stats["n_days"], 1),
    }


def constraints(trial):
    """Hard constraints: max DD < 10%, daily DD < 5%."""
    return [
        trial.user_attrs.get("max_dd", 1.0) - 0.10,
        trial.user_attrs.get("worst_day", 1.0) - 0.05,
    ]


def objective(trial):
    """Optuna objective: maximize mean CVaR-ratio across walk-forward windows."""
    load_all_data()

    C = trial.suggest_float("C", 0.001, 1.0, log=True)
    threshold = trial.suggest_float("threshold", 0.04, 0.20, step=0.02)
    lookback = trial.suggest_int("lookback", 6, 16, step=2)
    vol_q = trial.suggest_float("vol_q", 0.0, 0.80, step=0.2)
    close_rule = trial.suggest_categorical("close_rule", ["fixed", "decay_T2", "decay_T3"])
    fwd_bars = trial.suggest_categorical("fwd_bars", [2, 4, 6, 8])

    cvar_ratios = []
    max_dds = []
    worst_days = []
    all_trades = 0

    for i, window in enumerate(WINDOWS):
        raw = generate_signals(C, lookback, window)
        result = evaluate_window(raw, threshold, vol_q, close_rule, fwd_bars)

        if result is None:
            trial.set_user_attr("max_dd", 1.0)
            trial.set_user_attr("worst_day", 1.0)
            return float("-inf")

        cvar_ratios.append(result["cvar_ratio"])
        max_dds.append(result["max_dd"])
        worst_days.append(result["worst_day"])
        all_trades += result["n_trades"]

        # Pruning: report intermediate score
        trial.report(sum(cvar_ratios) / len(cvar_ratios), step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    mean_cvar_ratio = sum(cvar_ratios) / len(cvar_ratios)

    trial.set_user_attr("max_dd", max(max_dds))
    trial.set_user_attr("worst_day", max(worst_days))
    trial.set_user_attr("mean_sharpe", sum(r["sharpe"] for r in [None] * 0) if False else 0)
    trial.set_user_attr("total_trades", all_trades)
    trial.set_user_attr("cvar_ratios", cvar_ratios)

    return mean_cvar_ratio


def main():
    parser = argparse.ArgumentParser(description="Optuna parameter optimizer")
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--n-jobs", type=int, default=1)
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 95)
    print("OPTUNA PARAMETER OPTIMIZATION — CVaR-RATIO + DD CONSTRAINTS")
    print(f"  Trials: {args.n_trials}, Workers: {args.n_jobs}")
    print(f"  Objective: maximize return / CVaR(5%)")
    print(f"  Constraints: max DD < 10%, daily DD < 5%")
    print(f"  Walk-forward: {len(WINDOWS)} windows")
    for w in WINDOWS:
        print(f"    {w['label']}: test [{w['test_start']}, {w['test_end']}]")
    print("=" * 95)

    # Load data
    print("\nLoading data...", flush=True)
    load_all_data()

    # Setup Optuna
    Path("data/optuna").mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///data/optuna/study.db"

    try:
        sampler = optuna.samplers.GPSampler(constraints_func=constraints)
    except Exception:
        # Fallback if GPSampler doesn't support constraints in this version
        sampler = optuna.samplers.TPESampler(
            multivariate=True,
            constant_liar=True,
            n_startup_trials=15,
        )
        print("  (Using TPESampler — GPSampler constraints not available)")

    study = optuna.create_study(
        study_name="strategy_cvar_v1",
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        load_if_exists=True,
    )

    # Seed with known good configs
    existing = len(study.trials)
    if existing == 0:
        study.enqueue_trial({
            "C": 0.01, "threshold": 0.14, "lookback": 12,
            "vol_q": 0.4, "close_rule": "decay_T2", "fwd_bars": 4,
        })
        study.enqueue_trial({
            "C": 0.005, "threshold": 0.14, "lookback": 8,
            "vol_q": 0.4, "close_rule": "fixed", "fwd_bars": 8,
        })
        study.enqueue_trial({
            "C": 0.01, "threshold": 0.14, "lookback": 14,
            "vol_q": 0.4, "close_rule": "decay_T3", "fwd_bars": 4,
        })
        print("  Seeded 3 known-good configs")
    else:
        print(f"  Resuming: {existing} existing trials")

    # Run optimization
    print(f"\nRunning {args.n_trials} trials...\n", flush=True)

    def callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            feasible = (trial.user_attrs.get("max_dd", 1) < 0.10 and
                       trial.user_attrs.get("worst_day", 1) < 0.05)
            marker = "F" if feasible else "X"
            print(f"  [{marker}] Trial {trial.number}: CVaR-ratio={trial.value:.3f}, "
                  f"DD={trial.user_attrs.get('max_dd',0):.1%}, "
                  f"C={trial.params['C']:.4f}, thr={trial.params['threshold']:.2f}, "
                  f"lb={trial.params['lookback']}, close={trial.params['close_rule']}, "
                  f"fwd={trial.params['fwd_bars']}", flush=True)

    study.optimize(objective, n_trials=args.n_trials, callbacks=[callback])

    # ==================== RESULTS ====================
    print(f"\n{'='*95}")
    print("RESULTS")
    print(f"{'='*95}")

    # Filter feasible trials
    all_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    feasible = [t for t in all_trials
                if t.user_attrs.get("max_dd", 1) < 0.10
                and t.user_attrs.get("worst_day", 1) < 0.05]

    print(f"\n  Total trials: {len(all_trials)}")
    print(f"  Feasible (DD constraints met): {len(feasible)}")
    print(f"  Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")

    if feasible:
        feasible.sort(key=lambda t: t.value, reverse=True)

        print(f"\n  TOP 10 FEASIBLE CONFIGS:")
        print(f"  {'#':>3s} {'CVaR-R':>7s} {'MaxDD':>6s} {'DayDD':>6s} "
              f"{'C':>7s} {'Thr':>5s} {'LB':>3s} {'VQ':>4s} {'Close':>8s} {'Fwd':>4s}")
        print("  " + "-" * 65)

        for i, t in enumerate(feasible[:10]):
            p = t.params
            print(f"  {i+1:>3d} {t.value:>7.2f} {t.user_attrs['max_dd']:>5.1%} "
                  f"{t.user_attrs['worst_day']:>5.1%} "
                  f"{p['C']:>7.4f} {p['threshold']:>5.2f} {p['lookback']:>3d} "
                  f"{p['vol_q']:>4.1f} {p['close_rule']:>8s} {p['fwd_bars']:>4d}")

        # Best config
        best = feasible[0]
        print(f"\n  BEST CONFIG:")
        for k, v in best.params.items():
            print(f"    {k}: {v}")
        print(f"    CVaR-ratio: {best.value:.3f}")
        print(f"    Max DD: {best.user_attrs['max_dd']:.1%}")
        print(f"    Worst day: {best.user_attrs['worst_day']:.1%}")

        # Leverage analysis on best
        print(f"\n  LEVERAGE ANALYSIS (best config):")
        for lev in [1.0, 1.5, 2.0, 2.5, 3.0]:
            results = []
            for window in WINDOWS:
                raw = generate_signals(best.params["C"], best.params["lookback"], window)
                r = evaluate_window(raw, best.params["threshold"], best.params["vol_q"],
                                   best.params["close_rule"], best.params["fwd_bars"], leverage=lev)
                if r: results.append(r)
            if results:
                avg_dd = max(r["max_dd"] for r in results)
                avg_pnl = sum(r["pnl_day"] for r in results) / len(results)
                avg_sr = sum(r["sharpe"] for r in results) / len(results)
                feasible_lev = avg_dd < 0.10 and max(r["worst_day"] for r in results) < 0.05
                marker = "OK" if feasible_lev else "DD!"
                print(f"    {lev:.1f}x: Sharpe={avg_sr:.2f}, PnL/d=${avg_pnl:.2f}, "
                      f"DD={avg_dd:.1%} [{marker}]")

    # Parameter importance
    try:
        importance = optuna.importance.get_param_importances(study)
        print(f"\n  PARAMETER IMPORTANCE:")
        for param, imp in importance.items():
            print(f"    {param}: {imp:.3f}")
    except Exception:
        pass

    print(f"\n  Total time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
