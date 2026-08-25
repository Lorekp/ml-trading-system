"""Optuna v2: Minimize days to 10% profit target.

Correct objective: maximize PnL/day (fastest to 10%).
Leverage and delever_start are in the search space.
Hard constraints: max DD < 10%, daily DD < 5%.
Reuses signal cache from v1.
"""

import sys, os, argparse, time, warnings, pickle
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import numpy as np, pandas as pd, sklearn, optuna
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from signal_engine.config import load_config, PortfolioConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from signal_engine.asset_selection import compute_monthly_dollar_volume
from afml.bet_sizing import get_signal, discretize_signal
import test_original_pipeline as orig

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
DATA_DIR = Path("data/binance-1m")
TOP_N = 30; MAX_BARS = 8
CACHE_DIR = Path("data/optuna/signal_cache")

WINDOWS = [
    {"label": "W1", "test_start": "2024-07", "test_end": "2025-02", "cal_days": 245},
    {"label": "W2", "test_start": "2025-03", "test_end": "2025-10", "cal_days": 244},
    {"label": "W3", "test_start": "2025-11", "test_end": "2026-02", "cal_days": 122},
]

ASSET_DATA = {}; ASSET_VOLUMES = {}; SIGNAL_CACHE = {}


def make_logreg(C):
    if NEW_SKLEARN:
        return LogisticRegression(C=C, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=C, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def load_all_data():
    global ASSET_DATA, ASSET_VOLUMES
    if ASSET_DATA: return
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


def generate_signals(C, lookback, window):
    key = f"C{C:.5f}_lb{lookback}_{window['label']}"
    if key in SIGNAL_CACHE: return SIGNAL_CACHE[key]
    cache_file = CACHE_DIR / f"{key}.pkl"
    if cache_file.exists():
        with open(cache_file, "rb") as f:
            df = pickle.load(f)
        SIGNAL_CACHE[key] = df
        return df

    sample_idx = next(iter(ASSET_VOLUMES.values())).index
    test_start = pd.Timestamp(window["test_start"] + "-01")
    end_date = max(d["end_date"] for d in ASSET_DATA.values())
    test_end = pd.Timestamp(window["test_end"] + "-28")
    if sample_idx.tz is not None:
        test_start = test_start.tz_localize(sample_idx.tz)
        test_end = test_end.tz_localize(sample_idx.tz)
    test_end = min(test_end, end_date)
    test_months = pd.period_range(start=test_start.to_period("M"), end=test_end.to_period("M"), freq="M")

    all_signals = []
    for month in test_months:
        ms = pd.Timestamp(month.start_time)
        if sample_idx.tz is not None: ms = ms.tz_localize(sample_idx.tz)
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
                ep = close_arr[bar_idx];
                if ep <= 0: continue
                d = dirs[i]
                rets = {k: d * (close_arr[bar_idx + k] - ep) / ep for k in range(1, MAX_BARS + 1)}
                vol_val = float(test_vol.get(te_all[i], np.nan))
                all_signals.append({
                    "timestamp": te_all[i], "direction": d, "confidence": conf[i],
                    "probability": yp[i], "signal_strength": float(ss[i]),
                    "asset": asset_name, "vol_val": vol_val,
                    **{f"ret_{k}": rets[k] for k in range(1, MAX_BARS + 1)},
                    **{f"vol_q_{q}": vol_quantiles[q] for q in vol_quantiles},
                })
    df = pd.DataFrame(all_signals).sort_values("timestamp").reset_index(drop=True) if all_signals else pd.DataFrame()
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    with open(cache_file, "wb") as f: pickle.dump(df, f)
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


def evaluate_window(raw, threshold, vol_q, close_rule, fwd_bars, leverage, delever_start, cal_days):
    """Evaluate one window with prop firm DD limits. Returns PnL per CALENDAR day."""
    filtered = apply_filters(raw, threshold, vol_q, fwd_bars, close_rule)
    if len(filtered) < 20: return None
    combined = orig.apply_nco_wf(filtered)
    pcfg = PortfolioConfig(
        starting_capital=10000.0,
        max_dd=0.10,                   # PROP FIRM HARD LIMIT
        delever_start=delever_start,   # TUNABLE
        slippage=0.0001,
        max_exposure=0.90,
        max_per_asset=0.05,
        base_sizing_pct=0.04 * leverage,
        max_signal_mult=3.0,
    )
    sim = simulate_portfolio(combined, pcfg)
    if sim is None or sim["n_trades"] < 5: return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 3: return None
    stats = compute_full_stats(dpnl, pcfg.starting_capital, sim["n_trades"],
                               sim["n_trades"] / len(dpnl))
    equity = pd.Series(sim["equity"][~sim["skipped"]],
                       index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
    dd = ((equity - equity.expanding().max()) / equity.expanding().max())
    daily_ret = dpnl / pcfg.starting_capital

    # PnL per CALENDAR day (not trading day)
    total_pnl = stats["final_equity"] - pcfg.starting_capital
    pnl_per_cal_day = total_pnl / max(cal_days, 1)
    trading_days = len(dpnl)

    return {
        "pnl_cal_day": pnl_per_cal_day,
        "pnl_trade_day": total_pnl / max(trading_days, 1),
        "total_pnl": total_pnl,
        "sharpe": stats["sharpe"],
        "max_dd": abs(dd.min()),
        "worst_day": abs(daily_ret.min()),
        "n_trades": sim["n_trades"],
        "trading_days": trading_days,
        "cal_days": cal_days,
        "activity_pct": trading_days / max(cal_days, 1) * 100,
        "halted": sim.get("halted", False),
    }


def constraints(trial):
    return [
        trial.user_attrs.get("max_dd", 1.0) - 0.10,
        trial.user_attrs.get("worst_day", 1.0) - 0.05,
    ]


def objective(trial):
    """Maximize PnL/day (= minimize days to 10%) across walk-forward windows."""
    load_all_data()

    C = trial.suggest_float("C", 0.001, 1.0, log=True)
    threshold = trial.suggest_float("threshold", 0.04, 0.20, step=0.02)
    lookback = trial.suggest_int("lookback", 6, 16, step=2)
    vol_q = trial.suggest_float("vol_q", 0.0, 0.80, step=0.2)
    close_rule = trial.suggest_categorical("close_rule", ["fixed", "decay_T2", "decay_T3"])
    fwd_bars = trial.suggest_categorical("fwd_bars", [2, 4, 6, 8])
    leverage = trial.suggest_float("leverage", 1.0, 5.0, step=0.5)
    delever_start = trial.suggest_float("delever_start", 0.05, 0.095, step=0.005)

    pnl_cal_days = []
    max_dds = []
    worst_days = []
    all_trades = 0
    all_trading_days = 0
    all_cal_days = 0

    for i, window in enumerate(WINDOWS):
        raw = generate_signals(C, lookback, window)
        result = evaluate_window(raw, threshold, vol_q, close_rule, fwd_bars,
                                  leverage, delever_start, window["cal_days"])

        if result is None:
            trial.set_user_attr("max_dd", 1.0)
            trial.set_user_attr("worst_day", 1.0)
            return float("-inf")

        pnl_cal_days.append(result["pnl_cal_day"])
        max_dds.append(result["max_dd"])
        worst_days.append(result["worst_day"])
        all_trades += result["n_trades"]
        all_trading_days += result["trading_days"]
        all_cal_days += result["cal_days"]

        # Pruning on calendar-day PnL
        trial.report(sum(pnl_cal_days) / len(pnl_cal_days), step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()

    # Weighted average PnL/cal_day (weight by window length)
    total_pnl = sum(r * w["cal_days"] for r, w in zip(pnl_cal_days, WINDOWS))
    mean_pnl_cal = total_pnl / all_cal_days

    trial.set_user_attr("max_dd", max(max_dds))
    trial.set_user_attr("worst_day", max(worst_days))
    trial.set_user_attr("total_trades", all_trades)
    trial.set_user_attr("trading_days", all_trading_days)
    trial.set_user_attr("activity_pct", all_trading_days / max(all_cal_days, 1) * 100)
    trial.set_user_attr("days_to_10pct", int(1000 / mean_pnl_cal) if mean_pnl_cal > 0 else 9999)

    return mean_pnl_cal  # MAXIMIZE PnL per CALENDAR day


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-trials", type=int, default=300)
    args = parser.parse_args()

    t0 = time.time()
    print("=" * 95)
    print("OPTUNA v2: MAXIMIZE PnL/DAY (fastest to 10%)")
    print(f"  Objective: max PnL/day across 3 walk-forward windows")
    print(f"  Constraints: max DD < 10%, daily DD < 5%")
    print(f"  Search: C, threshold, lookback, vol_q, close, fwd, leverage, delever_start")
    print(f"  Trials: {args.n_trials}")
    print("=" * 95)

    load_all_data()
    Path("data/optuna").mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///data/optuna/study_v2.db"

    try:
        sampler = optuna.samplers.GPSampler(constraints_func=constraints)
    except Exception:
        sampler = optuna.samplers.TPESampler(multivariate=True, constant_liar=True, n_startup_trials=15)

    study = optuna.create_study(
        study_name="max_pnl_v2",
        storage=storage,
        direction="maximize",
        sampler=sampler,
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=1),
        load_if_exists=True,
    )

    # Seed with known configs + aggressive leverage variants
    if len(study.trials) == 0:
        seeds = [
            {"C": 0.01, "threshold": 0.14, "lookback": 12, "vol_q": 0.4, "close_rule": "decay_T2", "fwd_bars": 4, "leverage": 2.0, "delever_start": 0.085},
            {"C": 0.01, "threshold": 0.14, "lookback": 12, "vol_q": 0.4, "close_rule": "decay_T2", "fwd_bars": 4, "leverage": 3.0, "delever_start": 0.09},
            {"C": 0.01, "threshold": 0.14, "lookback": 12, "vol_q": 0.4, "close_rule": "decay_T2", "fwd_bars": 4, "leverage": 4.0, "delever_start": 0.095},
            {"C": 0.01, "threshold": 0.12, "lookback": 12, "vol_q": 0.4, "close_rule": "fixed", "fwd_bars": 4, "leverage": 2.5, "delever_start": 0.085},
            {"C": 0.01, "threshold": 0.10, "lookback": 10, "vol_q": 0.4, "close_rule": "fixed", "fwd_bars": 6, "leverage": 2.0, "delever_start": 0.09},
        ]
        for s in seeds:
            study.enqueue_trial(s)
        print(f"  Seeded {len(seeds)} configs")

    print(f"\nRunning {args.n_trials} trials...\n", flush=True)

    def callback(study, trial):
        if trial.state == optuna.trial.TrialState.COMPLETE:
            feasible = (trial.user_attrs.get("max_dd", 1) < 0.10 and
                       trial.user_attrs.get("worst_day", 1) < 0.05)
            marker = "F" if feasible else "X"
            d2t = trial.user_attrs.get("days_to_10pct", 9999)
            act = trial.user_attrs.get("activity_pct", 0)
            print(f"  [{marker}] T{trial.number}: ${trial.value:.2f}/cal_d, "
                  f"~{d2t}d to 10%, DD={trial.user_attrs.get('max_dd',0):.1%}, "
                  f"act={act:.0f}%, lev={trial.params.get('leverage',0):.1f}, "
                  f"C={trial.params['C']:.4f}, thr={trial.params['threshold']:.2f}, "
                  f"lb={trial.params['lookback']}, {trial.params['close_rule']}, "
                  f"fwd={trial.params['fwd_bars']}", flush=True)

    study.optimize(objective, n_trials=args.n_trials, callbacks=[callback])

    # RESULTS
    print(f"\n{'='*95}")
    print("RESULTS")
    print(f"{'='*95}")

    all_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    feasible = [t for t in all_trials
                if t.user_attrs.get("max_dd", 1) < 0.10
                and t.user_attrs.get("worst_day", 1) < 0.05]
    feasible.sort(key=lambda t: t.value, reverse=True)

    print(f"\n  Completed: {len(all_trials)}, Feasible: {len(feasible)}, "
          f"Pruned: {len([t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED])}")

    if feasible:
        print(f"\n  TOP 15 FEASIBLE (by PnL per CALENDAR day):")
        print(f"  {'#':>3s} {'$/cal':>6s} {'~Days':>5s} {'DD':>5s} {'Act%':>5s} {'Lev':>4s} {'Dlv':>5s} "
              f"{'C':>7s} {'Thr':>4s} {'LB':>3s} {'Close':>8s} {'Fwd':>4s}")
        print("  " + "-" * 78)

        for i, t in enumerate(feasible[:15]):
            p = t.params
            d2t = t.user_attrs.get("days_to_10pct", 9999)
            act = t.user_attrs.get("activity_pct", 0)
            print(f"  {i+1:>3d} ${t.value:>5.2f} {d2t:>5d} {t.user_attrs['max_dd']:>4.1%} "
                  f"{act:>4.0f}% {p['leverage']:>4.1f} "
                  f"{p['delever_start']:>5.3f} {p['C']:>7.4f} {p['threshold']:>4.2f} "
                  f"{p['lookback']:>3d} {p['close_rule']:>8s} {p['fwd_bars']:>4d}")

        best = feasible[0]
        print(f"\n  BEST CONFIG (max PnL/calendar_day, DD < 10%):")
        for k, v in best.params.items():
            print(f"    {k}: {v}")
        print(f"    PnL/calendar_day: ${best.value:.2f}")
        print(f"    Est. days to 10%: {best.user_attrs.get('days_to_10pct', 9999)}")
        print(f"    Max DD: {best.user_attrs['max_dd']:.1%}")
        print(f"    Worst day: {best.user_attrs['worst_day']:.1%}")
        print(f"    Trades: {best.user_attrs['total_trades']}")
        print(f"    Activity: {best.user_attrs.get('activity_pct', 0):.0f}% of calendar days")

    try:
        importance = optuna.importance.get_param_importances(study)
        print(f"\n  PARAMETER IMPORTANCE:")
        for param, imp in sorted(importance.items(), key=lambda x: -x[1]):
            print(f"    {param}: {imp:.3f}")
    except Exception: pass

    print(f"\n  Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
