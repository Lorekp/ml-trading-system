"""ML Pair Manager — XGBoost learns which pairs to add/drop.

Loads training data from coint_ml_generate_data.py output.
Walk-forward: expanding train window, predict next month.
Uses predictions to manage a rolling pair book.
Bar-level spread simulation for selected pairs.
"""

import sys, os, time, warnings, math
from pathlib import Path

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

warnings.filterwarnings("ignore")
CACHE_DIR = Path("data/cache_15m")
DATA_PATH = Path("results/coint_ml/training_data.parquet")

FEATURES = [
    "eg_pvalue", "half_life_days", "hurst", "beta_stability", "return_corr",
    "spread_vol", "eg_pvalue_3mo_trend", "half_life_trend",
    "consecutive_passes", "consecutive_fails", "months_since_last_pass",
    "spread_mr_speed", "volume_ratio", "sector_match", "spread_skewness",
]

CAPITAL = 10000.0; SIZE_PCT = 0.02; COST_BPS = 5.0
MAX_HOLD = 96 * 30; MIN_HOLD = 96; EMBARGO = 96 * 7
ENTRY_Z = 2.0; CAL_DAYS = 611


def ols_fit(y, x):
    xm, ym = x.mean(), y.mean(); xd = x - xm; d = np.dot(xd, xd)
    if d <= 0: return np.nan, np.nan
    b = np.dot(xd, y - ym) / d; return float(b), float(ym - b * xm)


def simulate_book_month(book_pairs, close_dict, month_start, month_end):
    """Simulate all pairs in book for one month. Returns daily PnL dict."""
    daily_pnl = {}; n_trades = 0

    for pair_info in book_pairs:
        y_name = pair_info["y"]; x_name = pair_info["x"]
        beta = pair_info["beta"]; mu = pair_info.get("mu", 0); sigma = pair_info.get("sigma", 1)
        if not np.isfinite(beta) or sigma <= 0: continue
        if y_name not in close_dict or x_name not in close_dict: continue

        y_s = close_dict[y_name]; x_s = close_dict[x_name]
        common = y_s.index.intersection(x_s.index)
        y = y_s.reindex(common); x = x_s.reindex(common)

        mask = (y.index >= month_start) & (y.index <= month_end)
        y_m = y[mask].values.astype(float); x_m = x[mask].values.astype(float)
        if len(y_m) < 96 * 10: continue

        log_y = np.log(y_m); log_x = np.log(x_m)
        y_ret = np.zeros(len(y_m)); x_ret = np.zeros(len(x_m))
        y_ret[1:] = y_m[1:] / y_m[:-1] - 1; x_ret[1:] = x_m[1:] / x_m[:-1] - 1

        # Confidence-scaled sizing
        ml_prob = pair_info.get("ml_prob", 0.5)
        confidence_mult = max(0.5, min(2.0, (ml_prob - 0.5) * 4 + 1.0))
        notional = CAPITAL * SIZE_PCT * confidence_mult

        pos = 0; entry_i = 0
        bar_dates = y.index[mask]
        cost = COST_BPS / 10000 * (1 + abs(beta)) * notional

        for i in range(1, len(y_m)):
            spread = log_y[i] - beta * log_x[i] - pair_info.get("intercept", 0)
            z = (spread - mu) / sigma if sigma > 0 else 0

            bar_pnl = 0.0
            if pos != 0:
                bar_pnl = pos * (y_ret[i] - beta * x_ret[i]) * notional
                bh = i - entry_i
                should_exit = False
                if bh >= MIN_HOLD:
                    if pos == 1 and z >= 0: should_exit = True
                    elif pos == -1 and z <= 0: should_exit = True
                if bh >= MAX_HOLD: should_exit = True
                if abs(z) > 5 and bh > 48: should_exit = True
                if should_exit: bar_pnl -= cost; pos = 0

            if pos == 0:
                if z <= -ENTRY_Z: pos = 1; entry_i = i; bar_pnl -= cost; n_trades += 1
                elif z >= ENTRY_Z: pos = -1; entry_i = i; bar_pnl -= cost; n_trades += 1

            if i < len(bar_dates):
                day = bar_dates[i].normalize()
                daily_pnl[day] = daily_pnl.get(day, 0) + bar_pnl

    return daily_pnl, n_trades


def run_ml_backtest(df, close_dict, model_class, model_params, config):
    """Walk-forward ML pair manager backtest."""
    add_thresh = config.get("add_thresh", 0.6)
    drop_thresh = config.get("drop_thresh", 0.4)
    max_pairs = config.get("max_pairs", 8)
    retrain_every = config.get("retrain_every", 3)  # months

    months = sorted(df["month"].unique())
    test_start_idx = len([m for m in months if m < "2025-03"])  # OOS starts 2025-03

    book = {}  # pair_key -> {y, x, beta, mu, sigma, consec_above, consec_below, ml_prob}
    candidates = {}  # pair_key -> consec_above
    model = None; last_train_month = None
    all_daily_pnl = {}; total_trades = 0
    monthly_log = []

    for mi in range(test_start_idx, len(months)):
        month = months[mi]
        month_start = pd.Timestamp(f"{month}-01", tz="UTC")
        month_end = month_start + pd.offsets.MonthEnd(0)

        # Train model (expanding window, retrain every N months)
        months_since_train = (mi - test_start_idx) if last_train_month is None else mi - last_train_month
        if model is None or months_since_train >= retrain_every:
            train_data = df[df["month"] < month]
            if len(train_data) < 100: continue

            X_train = train_data[FEATURES].values
            y_train = train_data["label"].values

            # Handle NaN/inf
            X_train = np.nan_to_num(X_train, nan=0, posinf=100, neginf=-100)

            model = model_class(**model_params)
            model.fit(X_train, y_train)
            last_train_month = mi

        # Predict for this month's pairs
        month_data = df[df["month"] == month]
        if len(month_data) == 0: continue

        X_month = np.nan_to_num(month_data[FEATURES].values, nan=0, posinf=100, neginf=-100)
        probs = model.predict_proba(X_month)[:, 1]

        pair_probs = {}
        for idx, (_, row) in enumerate(month_data.iterrows()):
            pair_key = row["pair"]
            pair_probs[pair_key] = {
                "prob": float(probs[idx]),
                "y": row["y"], "x": row["x"],
                "beta": row["beta"],
            }

        # Book management: keep/drop/add
        # Update existing pairs
        for pk in list(book.keys()):
            if pk in pair_probs:
                prob = pair_probs[pk]["prob"]
                book[pk]["ml_prob"] = prob
                if prob >= drop_thresh:
                    book[pk]["consec_below"] = 0
                else:
                    book[pk]["consec_below"] = book[pk].get("consec_below", 0) + 1
                    if book[pk]["consec_below"] >= 2:
                        book.pop(pk)  # DROP
            else:
                # Pair not in universe anymore
                book[pk]["consec_below"] = book[pk].get("consec_below", 0) + 1
                if book[pk]["consec_below"] >= 2:
                    book.pop(pk)

        # Candidates: track consecutive high predictions
        for pk, info in pair_probs.items():
            if pk in book: continue
            if info["prob"] >= add_thresh:
                candidates[pk] = candidates.get(pk, 0) + 1
                if candidates[pk] >= 2 and len(book) < max_pairs:
                    # ADD: fit OLS and add to book
                    y_name, x_name = info["y"], info["x"]
                    if y_name in close_dict and x_name in close_dict:
                        y_s = close_dict[y_name]; x_s = close_dict[x_name]
                        common = y_s.index.intersection(x_s.index)
                        train_end = month_start - pd.Timedelta(minutes=15 * EMBARGO)
                        train_start = train_end - pd.DateOffset(months=6)
                        tmask = (common >= train_start) & (common < train_end)
                        y_train = y_s.reindex(common[tmask]).values.astype(float)
                        x_train = x_s.reindex(common[tmask]).values.astype(float)
                        if len(y_train) >= 96 * 30:
                            log_y = np.log(y_train); log_x = np.log(x_train)
                            beta, intercept = ols_fit(log_y, log_x)
                            if np.isfinite(beta) and beta > 0:
                                resid = log_y - beta * log_x - intercept
                                book[pk] = {
                                    "y": y_name, "x": x_name,
                                    "beta": beta, "intercept": intercept,
                                    "mu": float(np.mean(resid)), "sigma": float(np.std(resid)),
                                    "ml_prob": info["prob"], "consec_below": 0,
                                }
                    candidates.pop(pk, None)
            else:
                candidates.pop(pk, None)

        # Refit OLS for all book pairs
        for pk, pair_info in book.items():
            y_name, x_name = pair_info["y"], pair_info["x"]
            if y_name in close_dict and x_name in close_dict:
                y_s = close_dict[y_name]; x_s = close_dict[x_name]
                common = y_s.index.intersection(x_s.index)
                train_end = month_start - pd.Timedelta(minutes=15 * EMBARGO)
                train_start = train_end - pd.DateOffset(months=6)
                tmask = (common >= train_start) & (common < train_end)
                y_t = y_s.reindex(common[tmask]).values.astype(float)
                x_t = x_s.reindex(common[tmask]).values.astype(float)
                if len(y_t) >= 96 * 30:
                    log_y = np.log(y_t); log_x = np.log(x_t)
                    beta, intercept = ols_fit(log_y, log_x)
                    if np.isfinite(beta) and beta > 0:
                        resid = log_y - beta * log_x - intercept
                        pair_info["beta"] = beta; pair_info["intercept"] = intercept
                        pair_info["mu"] = float(np.mean(resid))
                        pair_info["sigma"] = float(np.std(resid))

        # Simulate this month
        dpnl, nt = simulate_book_month(list(book.values()), close_dict, month_start, month_end)
        for day, pnl in dpnl.items():
            all_daily_pnl[day] = all_daily_pnl.get(day, 0) + pnl
        total_trades += nt

        pairs_in_book = [pk for pk in book]
        monthly_log.append({
            "month": month, "n_book": len(book), "n_trades": nt,
            "pairs": pairs_in_book[:5],
        })

    # Metrics
    dpnl = pd.Series(all_daily_pnl)
    if len(dpnl) < 10: return None

    cum = dpnl.cumsum(); dr = dpnl / CAPITAL
    sharpe = dr.mean() / dr.std() * np.sqrt(365.25) if dr.std() > 0 else 0
    peak = (cum + CAPITAL).cummax(); dd = abs(((cum + CAPITAL - peak) / peak).min())
    wd = abs(dr.min()) if len(dr) > 0 else 0
    total_pnl = cum.iloc[-1]; pnl_cal = total_pnl / CAL_DAYS
    monthly_pnl = dpnl.resample("ME").sum()
    pos_months = (monthly_pnl > 0).sum()

    return {
        "sharpe": sharpe, "pnl_cal": pnl_cal, "max_dd": dd, "worst_day": wd,
        "n_trades": total_trades,
        "pos_months": f"{pos_months}/{len(monthly_pnl)} ({pos_months/max(len(monthly_pnl),1)*100:.0f}%)",
        "monthly_log": monthly_log, "monthly_pnl": monthly_pnl,
        "model": model,
    }


def main():
    t0 = time.time()
    print("=" * 110)
    print("ML PAIR MANAGER — XGBoost Walk-Forward Backtest")
    print("=" * 110)

    # Load training data
    if not DATA_PATH.exists():
        print(f"ERROR: Training data not found at {DATA_PATH}")
        print("Run coint_ml_generate_data.py first.")
        return

    df = pd.read_parquet(DATA_PATH)
    print(f"\nTraining data: {len(df):,} rows, {df['month'].nunique()} months, "
          f"{df['pair'].nunique()} pairs, {df['label'].mean()*100:.1f}% positive")

    # Load close data for simulation
    print("Loading close data...", end=" ", flush=True)
    close_dict = {}
    for f in sorted(CACHE_DIR.glob("*.parquet")):
        name = f.stem.replace("USDT", "")
        try:
            d = pd.read_parquet(f, columns=["close"])
            if len(d) >= 2000: close_dict[name] = d["close"]
        except: pass
    print(f"{len(close_dict)} assets", flush=True)

    # Feature importance (quick, on full train set)
    print("\n--- Feature Importance (full training set) ---")
    train_data = df[df["month"] < "2025-03"]
    X_all = np.nan_to_num(train_data[FEATURES].values, nan=0, posinf=100, neginf=-100)
    y_all = train_data["label"].values

    xgb = XGBClassifier(max_depth=4, n_estimators=100, learning_rate=0.1,
                        subsample=0.8, colsample_bytree=0.8,
                        eval_metric="logloss", verbosity=0, random_state=42)
    xgb.fit(X_all, y_all)
    imp = sorted(zip(FEATURES, xgb.feature_importances_), key=lambda x: -x[1])
    for feat, score in imp:
        print(f"  {feat:<30s} {score:.4f}")

    # In-sample accuracy
    preds = xgb.predict(X_all)
    print(f"\n  In-sample accuracy: {accuracy_score(y_all, preds)*100:.1f}%")
    print(f"  Base rate: {y_all.mean()*100:.1f}%")

    # Walk-forward backtests with different configs
    configs = [
        ("XGB add=0.6 drop=0.4 max=8", XGBClassifier,
         {"max_depth": 4, "n_estimators": 100, "learning_rate": 0.1, "subsample": 0.8,
          "colsample_bytree": 0.8, "eval_metric": "logloss", "verbosity": 0, "random_state": 42},
         {"add_thresh": 0.6, "drop_thresh": 0.4, "max_pairs": 8}),
        ("XGB add=0.5 drop=0.3 max=8", XGBClassifier,
         {"max_depth": 4, "n_estimators": 100, "learning_rate": 0.1, "subsample": 0.8,
          "colsample_bytree": 0.8, "eval_metric": "logloss", "verbosity": 0, "random_state": 42},
         {"add_thresh": 0.5, "drop_thresh": 0.3, "max_pairs": 8}),
        ("XGB add=0.7 drop=0.5 max=5", XGBClassifier,
         {"max_depth": 4, "n_estimators": 100, "learning_rate": 0.1, "subsample": 0.8,
          "colsample_bytree": 0.8, "eval_metric": "logloss", "verbosity": 0, "random_state": 42},
         {"add_thresh": 0.7, "drop_thresh": 0.5, "max_pairs": 5}),
        ("LogReg add=0.6 drop=0.4", LogisticRegression,
         {"max_iter": 1000, "random_state": 42},
         {"add_thresh": 0.6, "drop_thresh": 0.4, "max_pairs": 8}),
        ("RF add=0.6 drop=0.4", RandomForestClassifier,
         {"n_estimators": 100, "max_depth": 5, "random_state": 42},
         {"add_thresh": 0.6, "drop_thresh": 0.4, "max_pairs": 8}),
    ]

    results = []
    for name, model_class, model_params, config in configs:
        print(f"\nRunning: {name}...", end=" ", flush=True)
        et0 = time.time()
        r = run_ml_backtest(df, close_dict, model_class, model_params, config)
        if r:
            r["name"] = name
            results.append(r)
            print(f"Sharpe {r['sharpe']:.2f}, ${r['pnl_cal']:.2f}/cal, DD {r['max_dd']*100:.1f}%, "
                  f"Trades {r['n_trades']}, +Mo {r['pos_months']} ({time.time()-et0:.0f}s)")
        else:
            print(f"NO DATA ({time.time()-et0:.0f}s)")

    # Final comparison
    print(f"\n{'='*110}")
    print("ML PAIR MANAGER RESULTS")
    print(f"{'='*110}")

    print(f"\n{'Config':<35s} {'$/cal':>7s} {'Sharpe':>7s} {'DD%':>6s} {'WD%':>6s} "
          f"{'Trades':>6s} {'+Mo':>12s}")
    print("-" * 90)
    for r in sorted(results, key=lambda x: -x["sharpe"]):
        print(f"{r['name']:<35s} ${r['pnl_cal']:>6.2f} {r['sharpe']:>7.2f} "
              f"{r['max_dd']*100:>5.1f}% {r['worst_day']*100:>5.2f}% "
              f"{r['n_trades']:>6d} {r['pos_months']:>12s}")

    print(f"\nBaselines: Static=Sharpe 1.58 $3.25/cal | R2-1b=Sharpe 0.90 $1.10/cal")

    # Best method monthly PnL
    if results:
        best = max(results, key=lambda r: r["sharpe"])
        print(f"\nBest: {best['name']}")
        if best.get("monthly_pnl") is not None:
            print(f"\n{'Month':<10s} {'PnL':>8s} {'Book':>5s}")
            for i, dt in enumerate(best["monthly_pnl"].index):
                v = best["monthly_pnl"].loc[dt]
                n_book = best["monthly_log"][i]["n_book"] if i < len(best["monthly_log"]) else 0
                flag = " <<< NEG" if v < -50 else ""
                print(f"{dt.strftime('%Y-%m'):<10s} ${v:>7.0f} {n_book:>5d}{flag}")

    print(f"\nRuntime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
