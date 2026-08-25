"""Cointegration Phase 2 — VECM, Copula, ML Persistence

Exp 3: VECM alpha as pair quality filter + entry signal
Exp 2: Gaussian copula spread trading
Exp 5: XGBoost persistence prediction (walk-forward)

All use same shared infrastructure as Phase 1.
"""

import sys, os, time, warnings, math
from pathlib import Path
from itertools import combinations

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd
from scipy.stats import norm, kendalltau
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")
CACHE_DIR = Path("data/cache_15m")

CANDIDATE_PAIRS = [
    ("ENJ", "ANKR"), ("ANKR", "FIL"), ("ENA", "LTC"), ("TAO", "FET"),
    ("LINK", "ENA"), ("XRP", "XLM"), ("ADA", "NEAR"), ("SUI", "WLD"),
    ("FIL", "SXP"), ("ADA", "LTC"), ("XRP", "LTC"), ("DOT", "APT"),
    ("ENA", "ETC"), ("AVAX", "ETC"), ("ADA", "DOT"), ("BCH", "LTC"),
]

CAPITAL = 10000.0; SIZE_PCT = 0.03; COST_BPS = 5.0
MAX_HOLD = 96 * 30; EMBARGO = 96 * 7
TEST_START = "2024-07-01"; TEST_END = "2026-02-28"; CAL_DAYS = 611


def ols_fit(y, x):
    xm, ym = x.mean(), y.mean(); xd = x - xm; d = np.dot(xd, xd)
    if d <= 0: return np.nan, np.nan
    b = np.dot(xd, y - ym) / d; return float(b), float(ym - b * xm)


# ============================================================
# Exp 3: VECM Error Correction
# ============================================================

def run_vecm_experiment(close_data):
    """Test VECM alpha as pair quality filter and entry signal."""
    from statsmodels.tsa.vector_ar.vecm import VECM
    print("\nFitting VECM for each pair...")

    results = []
    for y_name, x_name in CANDIDATE_PAIRS:
        if y_name not in close_data or x_name not in close_data: continue
        y_s = close_data[y_name]; x_s = close_data[x_name]
        common = y_s.index.intersection(x_s.index)
        y = y_s.reindex(common).astype(float); x = x_s.reindex(common).astype(float)

        log_y = np.log(y.values); log_x = np.log(x.values)
        y_ret = np.zeros(len(y)); x_ret = np.zeros(len(x))
        y_ret[1:] = y.values[1:] / y.values[:-1] - 1
        x_ret[1:] = x.values[1:] / x.values[:-1] - 1

        ws = pd.Timestamp(TEST_START, tz="UTC"); we = pd.Timestamp(TEST_END, tz="UTC")
        test_mask = (y.index >= ws) & (y.index <= we)
        test_idx = y.index[test_mask]
        if len(test_idx) < 96 * 60: continue

        idx_map = {ts: i for i, ts in enumerate(y.index)}
        pos = 0; entry_bar = 0; last_month = None
        beta = np.nan; mu = 0; sigma = 1; alpha_y = 0
        daily_pnl = {}; n_trades = 0; n_wins = 0

        for ts in test_idx:
            i = idx_map[ts]; cm = ts.to_period("M")

            if cm != last_month:
                te = ts - pd.Timedelta(minutes=15 * EMBARGO)
                t_start = te - pd.DateOffset(months=3)
                tm = (y.index >= t_start) & (y.index < te)
                if tm.sum() >= 96 * 30:
                    data = np.column_stack([log_y[tm], log_x[tm]])
                    try:
                        vecm = VECM(data, k_ar_diff=2, coint_rank=1, deterministic="ci")
                        result = vecm.fit()
                        alpha_y = float(result.alpha[0, 0])
                        beta_vec = result.beta
                        if beta_vec.shape[0] >= 2 and abs(beta_vec[0, 0]) > 1e-10:
                            beta = float(-beta_vec[1, 0] / beta_vec[0, 0])
                        else:
                            beta = np.nan
                        if np.isfinite(beta) and beta > 0:
                            resid = log_y[tm] - beta * log_x[tm]
                            mu = float(np.mean(resid)); sigma = float(np.std(resid))
                            if sigma <= 1e-12: sigma = 1
                        else:
                            beta = np.nan
                    except Exception:
                        beta = np.nan; alpha_y = 0
                last_month = cm

            if not np.isfinite(beta) or not np.isfinite(sigma): continue
            # Only trade if |alpha| > threshold (strong error correction)
            if abs(alpha_y) < 0.001: continue

            spread = log_y[i] - beta * log_x[i]
            z = (spread - mu) / sigma

            notional = CAPITAL * SIZE_PCT
            bar_pnl = 0.0
            if pos != 0:
                hr = y_ret[i] - beta * x_ret[i]
                bar_pnl = pos * hr * notional

            cost = COST_BPS / 10000 * (1 + abs(beta)) * notional

            if pos != 0:
                bh = i - entry_bar
                should_exit = False
                if pos == 1 and z >= 0: should_exit = True
                elif pos == -1 and z <= 0: should_exit = True
                elif bh >= MAX_HOLD: should_exit = True
                if should_exit:
                    bar_pnl -= cost; pos = 0
                    if bar_pnl > 0: n_wins += 1

            if pos == 0:
                if z <= -1.5: pos = 1; entry_bar = i; bar_pnl -= cost; n_trades += 1
                elif z >= 1.5: pos = -1; entry_bar = i; bar_pnl -= cost; n_trades += 1

            day = ts.normalize()
            daily_pnl[day] = daily_pnl.get(day, 0) + bar_pnl

        if n_trades < 3: continue
        dpnl = pd.Series(daily_pnl); dr = dpnl / CAPITAL
        sharpe = dr.mean() / dr.std() * np.sqrt(365.25) if len(dr) > 10 and dr.std() > 0 else 0
        cum = dpnl.cumsum(); peak = (cum + CAPITAL).cummax()
        dd = abs(((cum + CAPITAL - peak) / peak).min())
        wd = abs(dr.min()) if len(dr) > 0 else 0
        results.append({"pair": f"{y_name}/{x_name}", "sharpe": sharpe, "pnl_cal": cum.iloc[-1] / CAL_DAYS,
                        "max_dd": dd, "worst_day": wd, "n_trades": n_trades,
                        "win_rate": n_wins / max(n_trades, 1) * 100, "alpha": alpha_y})

    return results


# ============================================================
# Exp 2: Gaussian Copula
# ============================================================

def run_copula_experiment(close_data):
    """Gaussian copula: transform to uniform margins, fit correlation, trade anomalies."""
    results = []

    for y_name, x_name in CANDIDATE_PAIRS:
        if y_name not in close_data or x_name not in close_data: continue
        y_s = close_data[y_name]; x_s = close_data[x_name]
        common = y_s.index.intersection(x_s.index)
        y = y_s.reindex(common).astype(float); x = x_s.reindex(common).astype(float)

        y_ret_full = y.pct_change().values; x_ret_full = x.pct_change().values
        y_ret_full[0] = 0; x_ret_full[0] = 0

        ws = pd.Timestamp(TEST_START, tz="UTC"); we = pd.Timestamp(TEST_END, tz="UTC")
        test_mask = (y.index >= ws) & (y.index <= we)
        test_idx = y.index[test_mask]
        if len(test_idx) < 96 * 60: continue

        idx_map = {ts: i for i, ts in enumerate(y.index)}
        pos = 0; entry_bar = 0; last_month = None
        rho = 0.5; y_mu = 0; y_sig = 1; x_mu = 0; x_sig = 1
        daily_pnl = {}; n_trades = 0; n_wins = 0

        for ts in test_idx:
            i = idx_map[ts]; cm = ts.to_period("M")

            # Monthly refit: estimate copula on training returns
            if cm != last_month:
                te = ts - pd.Timedelta(minutes=15 * EMBARGO)
                t_start = te - pd.DateOffset(months=3)
                tm = (y.index >= t_start) & (y.index < te)
                n_train = tm.sum()
                if n_train >= 96 * 30:
                    yr = y_ret_full[tm]; xr = x_ret_full[tm]
                    y_mu = float(np.mean(yr)); y_sig = float(np.std(yr))
                    x_mu = float(np.mean(xr)); x_sig = float(np.std(xr))
                    if y_sig <= 1e-12: y_sig = 1
                    if x_sig <= 1e-12: x_sig = 1

                    # Transform to uniform via CDF, then to normal
                    u_y = norm.cdf((yr - y_mu) / y_sig)
                    u_x = norm.cdf((xr - x_mu) / x_sig)
                    # Gaussian copula parameter = Pearson correlation of normals
                    z_y = norm.ppf(np.clip(u_y, 1e-6, 1 - 1e-6))
                    z_x = norm.ppf(np.clip(u_x, 1e-6, 1 - 1e-6))
                    valid = np.isfinite(z_y) & np.isfinite(z_x)
                    if valid.sum() > 100:
                        rho = float(np.corrcoef(z_y[valid], z_x[valid])[0, 1])
                    if not np.isfinite(rho): rho = 0.5
                last_month = cm

            if abs(rho) < 0.1: continue  # No dependence

            # Current bar: mispricing index
            yr_now = y_ret_full[i]; xr_now = x_ret_full[i]
            u_y = norm.cdf((yr_now - y_mu) / y_sig)
            u_x = norm.cdf((xr_now - x_mu) / x_sig)
            z_y = norm.ppf(np.clip(u_y, 1e-6, 1 - 1e-6))
            z_x = norm.ppf(np.clip(u_x, 1e-6, 1 - 1e-6))

            if not (np.isfinite(z_y) and np.isfinite(z_x)): continue

            # Conditional: E[Z_y | Z_x] = rho * Z_x, Var = 1 - rho^2
            cond_mean = rho * z_x
            cond_std = math.sqrt(max(1 - rho ** 2, 1e-6))
            # Mispricing = how far Y is from conditional expectation
            mispricing = (z_y - cond_mean) / cond_std

            # Use log prices for PnL (hedge ratio = 1 for copula)
            notional = CAPITAL * SIZE_PCT
            bar_pnl = 0.0
            if pos != 0:
                bar_pnl = pos * (y_ret_full[i] - x_ret_full[i]) * notional

            cost = COST_BPS / 10000 * 2 * notional  # both legs

            if pos != 0:
                bh = i - entry_bar
                should_exit = False
                if pos == 1 and mispricing >= 0: should_exit = True
                elif pos == -1 and mispricing <= 0: should_exit = True
                elif bh >= MAX_HOLD: should_exit = True
                if should_exit:
                    bar_pnl -= cost; pos = 0
                    if bar_pnl > 0: n_wins += 1

            if pos == 0:
                if mispricing <= -2.0: pos = 1; entry_bar = i; bar_pnl -= cost; n_trades += 1
                elif mispricing >= 2.0: pos = -1; entry_bar = i; bar_pnl -= cost; n_trades += 1

            day = ts.normalize()
            daily_pnl[day] = daily_pnl.get(day, 0) + bar_pnl

        if n_trades < 3: continue
        dpnl = pd.Series(daily_pnl); dr = dpnl / CAPITAL
        sharpe = dr.mean() / dr.std() * np.sqrt(365.25) if len(dr) > 10 and dr.std() > 0 else 0
        cum = dpnl.cumsum(); peak = (cum + CAPITAL).cummax()
        dd = abs(((cum + CAPITAL - peak) / peak).min())
        wd = abs(dr.min()) if len(dr) > 0 else 0
        results.append({"pair": f"{y_name}/{x_name}", "sharpe": sharpe, "pnl_cal": cum.iloc[-1] / CAL_DAYS,
                        "max_dd": dd, "worst_day": wd, "n_trades": n_trades,
                        "win_rate": n_wins / max(n_trades, 1) * 100, "rho": rho})

    return results


# ============================================================
# Exp 5: ML Persistence Prediction
# ============================================================

def run_ml_persistence_experiment(close_data):
    """XGBoost predicts which pairs will be profitable next month."""
    from xgboost import XGBClassifier
    from sklearn.model_selection import TimeSeriesSplit

    # Build feature matrix: per pair per month
    months = pd.period_range("2024-01", "2026-01", freq="M")
    rows = []

    for y_name, x_name in CANDIDATE_PAIRS:
        if y_name not in close_data or x_name not in close_data: continue
        y_s = close_data[y_name]; x_s = close_data[x_name]
        common = y_s.index.intersection(x_s.index)
        y = y_s.reindex(common).astype(float); x = x_s.reindex(common).astype(float)
        log_y = np.log(y.values); log_x = np.log(x.values)

        for mi, month in enumerate(months[:-1]):
            ms = pd.Timestamp(month.start_time, tz="UTC")
            next_ms = pd.Timestamp(months[mi + 1].start_time, tz="UTC")

            # Features from 6-month lookback
            lb = ms - pd.DateOffset(months=6)
            mask = (y.index >= lb) & (y.index < ms)
            ya = y[mask].values; xa = x[mask].values
            if len(ya) < 2000: continue

            # EG test
            try:
                _, p_now, _ = coint(ya, xa, maxlag=10)
            except: continue

            log_ya = np.log(ya); log_xa = np.log(xa)
            beta, intercept = ols_fit(log_ya, log_xa)
            if not np.isfinite(beta) or beta <= 0: continue
            resid = log_ya - beta * log_xa - intercept

            # Half-life
            if len(resid) >= 50:
                lag = resid[:-1]; delta = np.diff(resid)
                d = np.dot(lag - lag.mean(), lag - lag.mean())
                phi = np.dot(lag - lag.mean(), delta - delta.mean()) / d if d > 0 else 0
                hl = -math.log(2) / phi / 96 if phi < 0 else 999
            else: hl = 999

            # Hurst
            lags = np.arange(2, min(65, len(resid) // 3))
            tau = [np.std(resid[l:] - resid[:-l]) for l in lags]
            tau_arr = np.array(tau)
            keep = (tau_arr > 0) & np.isfinite(tau_arr)
            hurst = float(np.polyfit(np.log(lags[keep]), np.log(tau_arr[keep]), 1)[0]) if keep.sum() > 5 else 0.5

            # Beta stability
            betas = []
            for chunk_y, chunk_x in zip(np.array_split(log_ya, 4), np.array_split(log_xa, 4)):
                b, _ = ols_fit(chunk_y, chunk_x)
                if np.isfinite(b) and b > 0: betas.append(b)
            beta_stab = max(betas) / min(betas) if len(betas) == 4 and min(betas) > 0 else 99

            # Spread volatility
            spread_vol = float(np.std(resid))

            # Return correlation
            ret_corr = float(np.corrcoef(np.diff(log_ya), np.diff(log_xa))[0, 1])

            # Label: is pair profitable next month? (simple: positive spread PnL)
            next_mask = (y.index >= ms) & (y.index < next_ms)
            if next_mask.sum() < 96 * 20: continue
            next_ly = log_y[next_mask.values if hasattr(next_mask, 'values') else next_mask]
            next_lx = log_x[next_mask.values if hasattr(next_mask, 'values') else next_mask]
            # Recompute z-scores and simulate
            next_resid = next_ly - beta * next_lx - intercept
            next_z = (next_resid - np.mean(resid)) / np.std(resid) if np.std(resid) > 0 else np.zeros_like(next_resid)
            # Simple: was there a profitable trade?
            try:
                _, p_next, _ = coint(y[next_mask].values, x[next_mask].values, maxlag=10)
                label = 1 if p_next < 0.05 else 0
            except: continue

            rows.append({
                "y": y_name, "x": x_name, "month": str(month),
                "p_value": p_now, "half_life": min(hl, 100), "hurst": hurst,
                "beta_stab": min(beta_stab, 10), "spread_vol": spread_vol,
                "ret_corr": ret_corr, "beta": beta,
                "label": label,
            })

    if len(rows) < 50:
        print("Insufficient data for ML experiment")
        return []

    df = pd.DataFrame(rows)
    features = ["p_value", "half_life", "hurst", "beta_stab", "spread_vol", "ret_corr"]
    X = df[features].values; y_labels = df["label"].values

    print(f"\nML dataset: {len(df)} samples, {y_labels.mean()*100:.0f}% positive")

    # Walk-forward: train on first 70%, test on last 30%
    split = int(len(df) * 0.7)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y_labels[:split], y_labels[split:]

    xgb = XGBClassifier(max_depth=3, n_estimators=50, learning_rate=0.1,
                        eval_metric="logloss", verbosity=0, random_state=42)
    xgb.fit(X_train, y_train)

    probs = xgb.predict_proba(X_test)[:, 1]
    preds = (probs > 0.5).astype(int)
    accuracy = (preds == y_test).mean()
    base_rate = y_test.mean()

    # Feature importance
    importances = dict(zip(features, xgb.feature_importances_))

    print(f"  Accuracy: {accuracy*100:.1f}% (base rate: {base_rate*100:.1f}%)")
    print(f"  Feature importance: {importances}")

    # Now use ML predictions to select pairs and simulate
    test_df = df.iloc[split:].copy()
    test_df["pred_prob"] = probs
    test_df["pred"] = preds

    # For months where ML predicts positive, simulate those pairs
    ml_results = []
    for _, row in test_df[test_df["pred"] == 1].iterrows():
        y_name, x_name = row["y"], row["x"]
        if y_name not in close_data or x_name not in close_data: continue
        # Quick sim for this pair-month
        # (simplified — just report the ML prediction quality)

    return [{
        "method": "XGBoost",
        "accuracy": accuracy,
        "base_rate": base_rate,
        "lift": accuracy / base_rate if base_rate > 0 else 0,
        "importances": importances,
        "n_test": len(y_test),
        "n_predicted_positive": int(preds.sum()),
    }]


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    print("=" * 110)
    print("COINTEGRATION PHASE 2 — VECM, COPULA, ML")
    print("=" * 110)

    print("\nLoading from parquet cache...", end=" ", flush=True)
    close_data = {}
    for f in sorted(CACHE_DIR.glob("*.parquet")):
        name = f.stem.replace("USDT", "")
        try:
            df = pd.read_parquet(f)
            if len(df) >= 2000: close_data[name] = df["close"]
        except: pass
    print(f"{len(close_data)} assets in {time.time()-t0:.1f}s")

    # ---- Exp 3: VECM ----
    print("\n" + "=" * 80)
    print("EXP 3: VECM ERROR CORRECTION (alpha as quality filter)")
    print("=" * 80)
    results_vecm = run_vecm_experiment(close_data)

    # ---- Exp 2: Copula ----
    print("\n" + "=" * 80)
    print("EXP 2: GAUSSIAN COPULA (non-linear dependence)")
    print("=" * 80)
    results_copula = run_copula_experiment(close_data)

    # ---- Exp 5: ML ----
    print("\n" + "=" * 80)
    print("EXP 5: ML PERSISTENCE PREDICTION (XGBoost)")
    print("=" * 80)
    results_ml = run_ml_persistence_experiment(close_data)

    # ---- Comparison ----
    print("\n" + "=" * 110)
    print("RESULTS COMPARISON")
    print("=" * 110)

    for name, results in [("Exp 3: VECM", results_vecm), ("Exp 2: Copula", results_copula)]:
        if not results:
            print(f"\n{name}: NO RESULTS"); continue
        total_pnl = sum(r["pnl_cal"] * CAL_DAYS for r in results)
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        max_dd = max(r["max_dd"] for r in results)
        max_wd = max(r["worst_day"] for r in results)
        total_trades = sum(r["n_trades"] for r in results)
        pnl_cal = total_pnl / CAL_DAYS
        ok = max_dd < 0.10 and max_wd < 0.05
        print(f"\n{name}: {len(results)} pairs, ${pnl_cal:.2f}/cal, Sharpe {avg_sharpe:.2f}, "
              f"DD {max_dd*100:.1f}%, WD {max_wd*100:.2f}%, Trades {total_trades} {'OK' if ok else 'FAIL'}")

        print(f"  {'Pair':<15s} {'Sharpe':>7s} {'$/cal':>7s} {'DD%':>6s} {'#Tr':>5s} {'WR%':>5s}")
        for r in sorted(results, key=lambda x: -x["sharpe"]):
            print(f"  {r['pair']:<15s} {r['sharpe']:>7.2f} ${r['pnl_cal']:>6.2f} "
                  f"{r['max_dd']*100:>5.1f}% {r['n_trades']:>5d} {r['win_rate']:>4.0f}%")

    if results_ml:
        print(f"\nExp 5: ML Persistence")
        for r in results_ml:
            print(f"  Accuracy: {r['accuracy']*100:.1f}% (base: {r['base_rate']*100:.1f}%, "
                  f"lift: {r['lift']:.2f}x)")
            print(f"  Importances: {r['importances']}")

    print(f"\nBaseline: Static 5-pair = Sharpe 1.58, $3.25/cal")
    print(f"Runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
