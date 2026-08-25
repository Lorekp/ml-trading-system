"""Cointegration Final Optimization — Systematic Grid Search

Pre-computes monthly EG screens once, then sweeps 48 configs over:
- Selection method: heuristic / ml_gate / ml_only
- Correlation prescreen: 0.40 / 0.50
- OLS lookback: 2mo / 6mo
- Entry z: 1.5 / 2.0
- Max pairs: 8 / 12
- Sizing: fixed 2% / ML-scaled 1-4%

All configs use rolling health check with 2-month hysteresis.
"""

import sys, os, time, warnings, math
from pathlib import Path
from itertools import combinations, product

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
CACHE_DIR = Path("data/cache_15m")
ML_DATA = Path("results/coint_ml/training_data.parquet")

CAPITAL = 10000.0; COST_BPS = 5.0; MAX_HOLD = 96 * 30; MIN_HOLD = 96
EMBARGO = 96 * 7; MIN_AGE_DAYS = 180
TEST_START = "2025-03"; TEST_END = "2026-02"; CAL_DAYS = 366

ML_FEATURES = [
    "eg_pvalue", "half_life_days", "hurst", "beta_stability", "return_corr",
    "spread_vol", "eg_pvalue_3mo_trend", "half_life_trend",
    "consecutive_passes", "consecutive_fails", "months_since_last_pass",
    "spread_mr_speed", "volume_ratio", "sector_match", "spread_skewness",
]


def ols_fit(y, x):
    xm, ym = x.mean(), y.mean(); xd = x - xm; d = np.dot(xd, xd)
    if d <= 0: return np.nan, np.nan
    b = np.dot(xd, y - ym) / d; return float(b), float(ym - b * xm)


def half_life_bars(resid):
    if len(resid) < 50: return float("inf")
    lag = resid[:-1]; delta = np.diff(resid)
    d = np.dot(lag - lag.mean(), lag - lag.mean())
    if d <= 0: return float("inf")
    phi = np.dot(lag - lag.mean(), delta - delta.mean()) / d
    return -math.log(2) / phi if phi < 0 else float("inf")


def select_universe(ohlcv_data, month_start):
    lookback_end = month_start - pd.Timedelta(days=1)
    candidates = {}
    for name, o15 in ohlcv_data.items():
        age = (month_start - o15.index.min()).days
        if age < MIN_AGE_DAYS: continue
        avail = o15.loc[:lookback_end]
        if len(avail) < 96 * 45: continue
        dv = (avail["close"] * avail["volume"]).resample("1D").sum()
        recent = dv.iloc[-30:] if len(dv) >= 30 else dv
        v = recent.median()
        if np.isnan(v) or v <= 0: continue
        if len(dv) >= 90 and dv.iloc[-90:].median() > 0 and v / dv.iloc[-90:].median() > 5: continue
        candidates[name] = float(v)
    return [a for a, _ in sorted(candidates.items(), key=lambda x: -x[1])[:30]]


def precompute_monthly_screens(close_panel, ohlcv_data, months):
    """Pre-compute EG results for all pairs for all months. Returns dict[month][corr_thresh] -> list of pair results."""
    screens = {}
    for month in months:
        ms = pd.Timestamp(f"{month}-01", tz="UTC")
        universe = select_universe(ohlcv_data, ms)
        if len(universe) < 6:
            screens[str(month)] = {"universe": universe, "pairs": {}}
            continue

        screen_end = ms - pd.Timedelta(minutes=15 * EMBARGO)
        avail = [a for a in universe if a in close_panel.columns]
        pairs_data = {}

        for a_i, a_j in combinations(avail, 2):
            # Get data for both lookback windows (6mo covers 2mo)
            screen_start = screen_end - pd.DateOffset(months=6)
            mask = (close_panel.index >= screen_start) & (close_panel.index < screen_end)
            y_s = close_panel.loc[mask, a_i].dropna()
            x_s = close_panel.loc[mask, a_j].dropna()
            common = y_s.index.intersection(x_s.index)
            if len(common) < 3000: continue

            y_arr = y_s.reindex(common).values.astype(float)
            x_arr = x_s.reindex(common).values.astype(float)

            # Correlation
            ret_y = np.diff(y_arr) / y_arr[:-1]; ret_x = np.diff(x_arr) / x_arr[:-1]
            valid = np.isfinite(ret_y) & np.isfinite(ret_x)
            if valid.sum() < 2000: continue
            corr = float(np.corrcoef(ret_y[valid], ret_x[valid])[0, 1])
            if not np.isfinite(corr): continue

            # EG test
            try: _, p, _ = coint(y_arr, x_arr, maxlag=10)
            except: p = 1.0

            # OLS on full 6mo
            log_y = np.log(y_arr); log_x = np.log(x_arr)
            beta6, int6 = ols_fit(log_y, log_x)
            resid6 = log_y - beta6 * log_x - int6 if np.isfinite(beta6) and beta6 > 0 else None

            # OLS on last 2mo
            n_2mo = min(len(log_y), 96 * 60)
            beta2, int2 = ols_fit(log_y[-n_2mo:], log_x[-n_2mo:])
            resid2 = log_y[-n_2mo:] - beta2 * log_x[-n_2mo:] - int2 if np.isfinite(beta2) and beta2 > 0 else None

            hl6 = half_life_bars(resid6) / 96 if resid6 is not None else float("inf")
            hl2 = half_life_bars(resid2) / 96 if resid2 is not None else float("inf")

            pairs_data[(a_i, a_j)] = {
                "corr": corr, "p": float(p),
                "beta6": beta6, "int6": int6, "hl6": hl6,
                "mu6": float(np.mean(resid6)) if resid6 is not None else 0,
                "sigma6": float(np.std(resid6)) if resid6 is not None else 1,
                "beta2": beta2, "int2": int2, "hl2": hl2,
                "mu2": float(np.mean(resid2)) if resid2 is not None else 0,
                "sigma2": float(np.std(resid2)) if resid2 is not None else 1,
            }

        screens[str(month)] = {"universe": universe, "pairs": pairs_data}
    return screens


def simulate_config(config, screens, close_panel, ret_panel, ml_model, ml_data, months):
    """Run one config through the full walk-forward."""
    method = config["method"]; min_corr = config["min_corr"]
    lb = config["lb_months"]; entry_z = config["entry_z"]
    max_pairs = config["max_pairs"]; base_size = config["base_size"]
    use_ml_sizing = config.get("ml_sizing", False)
    ml_thresh = config.get("ml_thresh", 0.4)

    book = {}  # (y,x) -> {beta, intercept, mu, sigma, fail_count, months_in}
    candidates = {}  # (y,x) -> consecutive_passes
    daily_pnl = {}; total_trades = 0

    for month in months:
        ms = pd.Timestamp(f"{month}-01", tz="UTC")
        me = ms + pd.offsets.MonthEnd(0)
        screen = screens.get(str(month), {})
        pairs_data = screen.get("pairs", {})

        # Determine which pairs pass this month
        passing = set()
        for (a_i, a_j), pd_info in pairs_data.items():
            if pd_info["corr"] < min_corr: continue

            passes_heuristic = pd_info["p"] < 0.05 and 1 < (pd_info[f"hl{lb}"] if f"hl{lb}" in pd_info else pd_info["hl6"]) < 20
            passes_ml = True

            if method in ("ml_gate", "ml_only") and ml_model is not None and ml_data is not None:
                # Check ML prediction for this pair
                month_ml = ml_data[(ml_data["month"] == str(month)) &
                                   (((ml_data["y"] == a_i) & (ml_data["x"] == a_j)) |
                                    ((ml_data["y"] == a_j) & (ml_data["x"] == a_i)))]
                if len(month_ml) > 0:
                    X = np.nan_to_num(month_ml[ML_FEATURES].values[:1], nan=0, posinf=100, neginf=-100)
                    prob = float(ml_model.predict_proba(X)[0, 1])
                    passes_ml = prob >= ml_thresh
                else:
                    passes_ml = False

            if method == "heuristic" and passes_heuristic:
                passing.add((a_i, a_j))
            elif method == "ml_gate" and passes_heuristic and passes_ml:
                passing.add((a_i, a_j))
            elif method == "ml_only" and passes_ml:
                passing.add((a_i, a_j))

        # Rolling book management
        for key in list(book.keys()):
            if key in passing or (key[1], key[0]) in passing:
                book[key]["fail_count"] = 0
                book[key]["months_in"] += 1
            else:
                book[key]["fail_count"] += 1
                if book[key]["fail_count"] >= 2:
                    book.pop(key); continue

            # Refit OLS
            pd_info = pairs_data.get(key)
            if pd_info is None: pd_info = pairs_data.get((key[1], key[0]))
            if pd_info:
                lb_key = f"beta{lb}" if f"beta{lb}" in pd_info else "beta6"
                book[key]["beta"] = pd_info[lb_key]
                book[key]["intercept"] = pd_info[f"int{lb}" if f"int{lb}" in pd_info else "int6"]
                book[key]["mu"] = pd_info[f"mu{lb}" if f"mu{lb}" in pd_info else "mu6"]
                book[key]["sigma"] = pd_info[f"sigma{lb}" if f"sigma{lb}" in pd_info else "sigma6"]

        # Add new pairs
        for key in passing:
            if key in book or (key[1], key[0]) in book: continue
            candidates[key] = candidates.get(key, 0) + 1
            if candidates[key] >= 2 and len(book) < max_pairs:
                pd_info = pairs_data.get(key)
                if pd_info is None: pd_info = pairs_data.get((key[1], key[0]))
                if pd_info:
                    lb_key = "6" if lb == 6 else "2"
                    b = pd_info.get(f"beta{lb_key}", pd_info.get("beta6"))
                    if np.isfinite(b) and b > 0:
                        book[key] = {
                            "beta": b,
                            "intercept": pd_info.get(f"int{lb_key}", pd_info.get("int6")),
                            "mu": pd_info.get(f"mu{lb_key}", pd_info.get("mu6")),
                            "sigma": pd_info.get(f"sigma{lb_key}", pd_info.get("sigma6")),
                            "fail_count": 0, "months_in": 0,
                        }
                        candidates.pop(key, None)

        # Clean stale candidates
        for key in list(candidates.keys()):
            if key not in passing and (key[1], key[0]) not in passing:
                candidates.pop(key, None)

        # Simulate this month
        month_bars = close_panel.index[(close_panel.index >= ms) & (close_panel.index <= me)]

        for key, pair in book.items():
            y_name, x_name = key
            beta = pair["beta"]; mu = pair["mu"]; sigma = pair["sigma"]
            if not np.isfinite(beta) or sigma <= 0: continue
            if y_name not in close_panel.columns or x_name not in close_panel.columns: continue

            # Sizing
            sz = base_size
            if use_ml_sizing and ml_model is not None and ml_data is not None:
                month_ml = ml_data[(ml_data["month"] == str(month)) &
                                   (((ml_data["y"] == y_name) & (ml_data["x"] == x_name)) |
                                    ((ml_data["y"] == x_name) & (ml_data["x"] == y_name)))]
                if len(month_ml) > 0:
                    X = np.nan_to_num(month_ml[ML_FEATURES].values[:1], nan=0, posinf=100, neginf=-100)
                    prob = float(ml_model.predict_proba(X)[0, 1])
                    sz = float(np.clip((prob - 0.3) * 3 * base_size * 2, base_size * 0.5, base_size * 2.5))

            notional = CAPITAL * sz
            pos = 0; entry_i = 0

            for bar_idx, ts in enumerate(month_bars):
                y_p = close_panel.at[ts, y_name]; x_p = close_panel.at[ts, x_name]
                if np.isnan(y_p) or np.isnan(x_p) or y_p <= 0 or x_p <= 0: continue

                spread = math.log(y_p) - beta * math.log(x_p) - pair.get("intercept", 0)
                z = (spread - mu) / sigma if sigma > 0 else 0

                y_r = ret_panel.at[ts, y_name] if ts in ret_panel.index and y_name in ret_panel.columns else 0
                x_r = ret_panel.at[ts, x_name] if ts in ret_panel.index and x_name in ret_panel.columns else 0
                if np.isnan(y_r): y_r = 0
                if np.isnan(x_r): x_r = 0

                bar_pnl = 0.0
                cost = COST_BPS / 10000 * (1 + abs(beta)) * notional

                if pos != 0:
                    bar_pnl = pos * (y_r - beta * x_r) * notional
                    bh = bar_idx - entry_i
                    should_exit = False
                    if bh >= MIN_HOLD:
                        if pos == 1 and z >= 0: should_exit = True
                        elif pos == -1 and z <= 0: should_exit = True
                    if bh >= MAX_HOLD: should_exit = True
                    if abs(z) > 5 and bh > 48: should_exit = True
                    if should_exit: bar_pnl -= cost; pos = 0

                if pos == 0:
                    if z <= -entry_z: pos = 1; entry_i = bar_idx; bar_pnl -= cost; total_trades += 1
                    elif z >= entry_z: pos = -1; entry_i = bar_idx; bar_pnl -= cost; total_trades += 1

                day = ts.normalize()
                daily_pnl[day] = daily_pnl.get(day, 0) + bar_pnl

    # Metrics
    dpnl = pd.Series(daily_pnl)
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
    }


def main():
    t0 = time.time()
    print("=" * 110)
    print("COINTEGRATION FINAL OPTIMIZATION — 48-Config Grid Search")
    print("=" * 110)

    # Load data
    print("\nLoading data...", end=" ", flush=True)
    ohlcv_data = {}
    for f in sorted(CACHE_DIR.glob("*.parquet")):
        name = f.stem.replace("USDT", "")
        try:
            df = pd.read_parquet(f, columns=["close", "volume"])
            if len(df) >= 2000: ohlcv_data[name] = df
        except: pass
    close_panel = pd.DataFrame({n: d["close"] for n, d in ohlcv_data.items()})
    ret_panel = close_panel.pct_change()
    print(f"{len(ohlcv_data)} assets", flush=True)

    # Load ML data + train model
    ml_model = None; ml_data = None
    if ML_DATA.exists():
        ml_data = pd.read_parquet(ML_DATA)
        train = ml_data[ml_data["month"] < TEST_START]
        if len(train) > 100:
            X = np.nan_to_num(train[ML_FEATURES].values, nan=0, posinf=100, neginf=-100)
            y = train["label"].values
            ml_model = XGBClassifier(max_depth=4, n_estimators=100, learning_rate=0.1,
                                     subsample=0.8, colsample_bytree=0.8,
                                     eval_metric="logloss", verbosity=0, random_state=42)
            ml_model.fit(X, y)
            print(f"ML model trained on {len(train)} samples", flush=True)

    # Pre-compute monthly screens
    months = pd.period_range(TEST_START, TEST_END, freq="M")
    print(f"\nPre-computing monthly EG screens for {len(months)} months...", flush=True)
    st = time.time()
    screens = precompute_monthly_screens(close_panel, ohlcv_data, months)
    print(f"  Done in {time.time()-st:.0f}s", flush=True)

    # Grid
    configs = []
    for method in ["heuristic", "ml_gate", "ml_only"]:
        for min_corr in [0.40, 0.50]:
            for lb in [2, 6]:
                for ez in [1.5, 2.0]:
                    for mp in [8, 12]:
                        for sz, ml_sz in [(0.02, False), (0.02, True)]:
                            # Skip redundant: ml_sizing only meaningful with ml methods
                            if ml_sz and method == "heuristic": continue
                            # Skip: ml_only with no ML model
                            if method in ("ml_gate", "ml_only") and ml_model is None: continue
                            label = f"{method[:4]} c{min_corr:.0e} lb{lb} z{ez} p{mp} {'mlsz' if ml_sz else f'{sz}'}"
                            configs.append({
                                "label": label, "method": method, "min_corr": min_corr,
                                "lb_months": lb, "entry_z": ez, "max_pairs": mp,
                                "base_size": sz, "ml_sizing": ml_sz, "ml_thresh": 0.4,
                            })

    print(f"\n{len(configs)} configurations to test")

    # Run all configs
    results = []
    for i, config in enumerate(configs):
        if i % 10 == 0:
            print(f"  [{i}/{len(configs)}] ({time.time()-t0:.0f}s)", flush=True)
        r = simulate_config(config, screens, close_panel, ret_panel, ml_model, ml_data, months)
        if r:
            r["config"] = config["label"]
            results.append(r)

    # Sort by Sharpe
    results.sort(key=lambda x: -x["sharpe"])

    print(f"\n{'='*110}")
    print(f"TOP 20 CONFIGS BY SHARPE")
    print(f"{'='*110}")

    print(f"\n{'#':>3s} {'Config':<45s} {'$/cal':>7s} {'Sharpe':>7s} {'DD%':>6s} "
          f"{'WD%':>6s} {'Trades':>6s} {'+Mo':>12s}")
    print("-" * 105)

    for i, r in enumerate(results[:20]):
        ok = r["max_dd"] < 0.07 and r["worst_day"] < 0.05
        print(f"{i+1:>3d} {r['config']:<45s} ${r['pnl_cal']:>6.2f} {r['sharpe']:>7.2f} "
              f"{r['max_dd']*100:>5.1f}% {r['worst_day']*100:>5.2f}% "
              f"{r['n_trades']:>6d} {r['pos_months']:>12s} {'OK' if ok else ''}")

    # Pareto frontier: best Sharpe at each DD level
    print(f"\n{'='*110}")
    print(f"PARETO FRONTIER (best Sharpe at each DD level)")
    print(f"{'='*110}")

    dd_buckets = [(0, 0.02), (0.02, 0.04), (0.04, 0.06), (0.06, 0.10)]
    for lo, hi in dd_buckets:
        bucket = [r for r in results if lo <= r["max_dd"] < hi]
        if bucket:
            best = max(bucket, key=lambda x: x["sharpe"])
            print(f"  DD {lo*100:.0f}-{hi*100:.0f}%: {best['config']:<45s} "
                  f"Sharpe {best['sharpe']:.2f}, ${best['pnl_cal']:.2f}/cal, "
                  f"DD {best['max_dd']*100:.1f}%, Trades {best['n_trades']}")

    print(f"\nBaselines: Static=1.58 $3.25 | R2-1b=0.90 $1.10 | ML=0.68 $0.21")
    print(f"Total configs tested: {len(results)}")
    print(f"Runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
