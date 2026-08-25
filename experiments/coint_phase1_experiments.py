"""Cointegration Phase 1 — 4 Independent Pair Selection Experiments

Each experiment tests a different method for reliably finding/trading cointegrated pairs.
All share the same data, evaluation framework, and walk-forward windows.

Exp 1: Kalman Filter (adaptive hedge ratio, state covariance breakdown detection)
Exp 4: Persistence-Weighted Selection (rank pairs by rolling EG pass rate)
Exp 6: Distance Method + GHE (Gatev normalized distance + Hurst filter)
Exp 7: Sector-Constrained + Min Streak

Compared against: Static 5-pair baseline (Sharpe 1.58)
"""

import sys, os, time, warnings, math
from pathlib import Path
from itertools import combinations

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")
CACHE_DIR = Path("data/cache_15m")

# All candidate pairs (from screening + expansion)
CANDIDATE_PAIRS = [
    ("ENJ", "ANKR"), ("ANKR", "FIL"), ("ENA", "LTC"), ("TAO", "FET"),
    ("LINK", "ENA"), ("XRP", "XLM"), ("ADA", "NEAR"), ("SUI", "WLD"),
    ("FIL", "SXP"), ("ADA", "LTC"), ("XRP", "LTC"), ("DOT", "APT"),
    ("ENA", "ETC"), ("AVAX", "ETC"), ("ADA", "DOT"), ("BCH", "LTC"),
]

SECTORS = {
    "L1": ["BTC", "ETH", "SOL", "ADA", "AVAX", "NEAR", "DOT", "APT", "SUI"],
    "DeFi": ["UNI", "AAVE", "LINK"],
    "AI": ["FET", "TAO"],
    "Storage": ["FIL", "ANKR"],
    "Gaming": ["ENJ"],
    "Exchange": ["BNB", "FTM", "TRX"],
    "Payment": ["XRP", "XLM", "LTC", "BCH"],
    "Other": ["ENA", "WLD", "SXP", "HBAR", "ETC", "DOGE", "DOT"],
}
ASSET_SECTOR = {}
for sector, assets in SECTORS.items():
    for a in assets:
        ASSET_SECTOR[a] = sector

# Shared config
CAPITAL = 10000.0; SIZE_PCT = 0.03; COST_BPS = 5.0
MAX_HOLD = 96 * 30; EMBARGO = 96 * 7
TEST_START = "2024-07-01"; TEST_END = "2026-02-28"; CAL_DAYS = 611


def ols_fit(y, x):
    xm, ym = x.mean(), y.mean(); xd = x - xm; d = np.dot(xd, xd)
    if d <= 0: return np.nan, np.nan
    b = np.dot(xd, y - ym) / d; return float(b), float(ym - b * xm)


def hurst_exponent(series, max_lag=64):
    """Generalized Hurst Exponent. H < 0.5 = mean-reverting."""
    if len(series) < max_lag * 3: return 0.5
    lags = np.arange(2, min(max_lag + 1, len(series) // 3))
    tau = []
    for lag in lags:
        diff = series[lag:] - series[:-lag]
        s = np.std(diff)
        if s > 0 and np.isfinite(s): tau.append(s)
        else: tau.append(np.nan)
    tau_arr = np.array(tau)
    keep = np.isfinite(tau_arr) & (tau_arr > 0)
    if keep.sum() < 5: return 0.5
    slope, _ = np.polyfit(np.log(lags[keep]), np.log(tau_arr[keep]), 1)
    return float(np.clip(slope, 0, 1))


# ============================================================
# Shared simulation function
# ============================================================

def simulate_pair_generic(
    y_close, x_close, y_name, x_name,
    get_signal_func,  # function(log_y, log_x, i, state) -> (z_score, position_size)
    state_init_func,  # function() -> state dict
    refit_func=None,  # function(state, train_log_y, train_log_x) -> state
    refit_monthly=True,
    lb_months=2,
):
    """Generic bar-level simulation with pluggable signal generation."""
    common = y_close.index.intersection(x_close.index)
    y = y_close.reindex(common).astype(float); x = x_close.reindex(common).astype(float)
    log_y = np.log(y.values); log_x = np.log(x.values)
    y_ret = np.zeros(len(y)); x_ret = np.zeros(len(x))
    y_ret[1:] = y.values[1:] / y.values[:-1] - 1
    x_ret[1:] = x.values[1:] / x.values[:-1] - 1

    ws = pd.Timestamp(TEST_START, tz="UTC"); we = pd.Timestamp(TEST_END, tz="UTC")
    test_mask = (y.index >= ws) & (y.index <= we)
    test_idx = y.index[test_mask]
    if len(test_idx) < 96 * 30: return None

    idx_map = {ts: i for i, ts in enumerate(y.index)}
    state = state_init_func()
    last_month = None
    pos = 0; entry_bar = 0
    daily_pnl = {}; n_trades = 0; n_wins = 0; equity = 0.0

    for ts in test_idx:
        i = idx_map[ts]
        cm = ts.to_period("M")

        # Monthly refit
        if refit_monthly and cm != last_month and refit_func:
            te = ts - pd.Timedelta(minutes=15 * EMBARGO)
            t_start = te - pd.DateOffset(months=lb_months)
            tm = (y.index >= t_start) & (y.index < te)
            if tm.sum() >= 96 * 30:
                state = refit_func(state, log_y[tm], log_x[tm])
            last_month = cm

        # Get signal
        sig = get_signal_func(log_y, log_x, i, state)
        if sig is None: continue
        z, beta = sig

        notional = CAPITAL * SIZE_PCT
        bar_pnl = 0.0
        if pos != 0 and np.isfinite(beta):
            hr = y_ret[i] - beta * x_ret[i]
            bar_pnl = pos * hr * notional

        cost = COST_BPS / 10000 * (1 + abs(beta)) * notional if np.isfinite(beta) else 0

        # Exit
        if pos != 0:
            bh = i - entry_bar
            should_exit = False
            if pos == 1 and z >= 0: should_exit = True
            elif pos == -1 and z <= 0: should_exit = True
            elif bh >= MAX_HOLD: should_exit = True
            elif abs(z) > 5.0 and bh > 48: should_exit = True  # breakdown
            if should_exit:
                bar_pnl -= cost; pos = 0
                if bar_pnl > 0: n_wins += 1

        # Entry
        entry_z = state.get("entry_z", 1.5)
        if pos == 0 and np.isfinite(z):
            if z <= -entry_z: pos = 1; entry_bar = i; bar_pnl -= cost; n_trades += 1
            elif z >= entry_z: pos = -1; entry_bar = i; bar_pnl -= cost; n_trades += 1

        equity += bar_pnl
        day = ts.normalize()
        daily_pnl[day] = daily_pnl.get(day, 0) + bar_pnl

    if n_trades < 3: return None
    dpnl = pd.Series(daily_pnl); dr = dpnl / CAPITAL
    sharpe = dr.mean() / dr.std() * np.sqrt(365.25) if len(dr) > 10 and dr.std() > 0 else 0
    cum = dpnl.cumsum(); peak = (cum + CAPITAL).cummax()
    dd = abs(((cum + CAPITAL - peak) / peak).min())
    wd = abs(dr.min()) if len(dr) > 0 else 0
    return {"sharpe": sharpe, "pnl": cum.iloc[-1], "pnl_cal": cum.iloc[-1] / CAL_DAYS,
            "max_dd": dd, "worst_day": wd, "n_trades": n_trades, "win_rate": n_wins / max(n_trades, 1) * 100}


# ============================================================
# Exp 1: Kalman Filter
# ============================================================

def kalman_state_init():
    return {"theta": np.array([1.0, 0.0]), "P": np.eye(2), "delta": 1e-5, "R": 1e-3,
            "entry_z": 2.0, "mu": 0, "sigma": 1, "n": 0}

def kalman_refit(state, train_log_y, train_log_x):
    # Run Kalman on training data to initialize state
    n = len(train_log_y)
    theta = np.array([1.0, 0.0]); P = np.eye(2)
    delta = state["delta"]; R = state["R"]
    errors = []
    for t in range(1, n):
        F = np.array([[train_log_x[t], 1.0]])
        P_pred = P + delta * np.eye(2)
        y_pred = F @ theta
        S = F @ P_pred @ F.T + R
        K = P_pred @ F.T / S[0, 0]
        e = train_log_y[t] - y_pred[0]
        theta = theta + K.flatten() * e
        P = (np.eye(2) - K @ F) @ P_pred
        if t > n // 2: errors.append(e)
    state["theta"] = theta; state["P"] = P
    if errors:
        state["mu"] = float(np.mean(errors))
        state["sigma"] = float(np.std(errors))
        if state["sigma"] <= 1e-12: state["sigma"] = 1
    state["n"] = 0
    return state

def kalman_signal(log_y, log_x, i, state):
    if i < 1: return None
    theta = state["theta"]; P = state["P"]
    delta = state["delta"]; R = state["R"]
    F = np.array([[log_x[i], 1.0]])
    P_pred = P + delta * np.eye(2)
    y_pred = F @ theta
    S = float((F @ P_pred @ F.T + R)[0, 0])  # Innovation variance
    K = P_pred @ F.T / S
    e = float(log_y[i] - y_pred[0])
    state["theta"] = theta + K.flatten() * e
    state["P"] = (np.eye(2) - K @ F) @ P_pred
    state["n"] += 1
    # Z-score = innovation / sqrt(innovation_variance) — CORRECT normalization
    z = e / math.sqrt(S) if S > 0 else 0
    return (z, float(state["theta"][0]))


# ============================================================
# Exp 4: Persistence-Weighted (uses standard OLS, filtered by persistence)
# ============================================================

def persistence_state_init():
    return {"beta": np.nan, "intercept": np.nan, "mu": 0, "sigma": 1, "entry_z": 1.5}

def persistence_refit(state, train_log_y, train_log_x):
    beta, intercept = ols_fit(train_log_y, train_log_x)
    if np.isfinite(beta) and beta > 0:
        resid = train_log_y - beta * train_log_x - intercept
        state["beta"] = beta; state["intercept"] = intercept
        state["mu"] = float(np.mean(resid)); state["sigma"] = float(np.std(resid))
        if state["sigma"] <= 1e-12: state["sigma"] = 1
    return state

def persistence_signal(log_y, log_x, i, state):
    if not np.isfinite(state["beta"]): return None
    spread = log_y[i] - state["beta"] * log_x[i] - state["intercept"]
    z = (spread - state["mu"]) / state["sigma"]
    return (z, state["beta"])


# ============================================================
# Exp 6: Distance Method + GHE
# ============================================================

def distance_state_init():
    return {"y_norm_base": np.nan, "x_norm_base": np.nan,
            "dist_mean": 0, "dist_std": 1, "entry_z": 2.0, "beta": 1.0}

def distance_refit(state, train_log_y, train_log_x):
    # Normalize by first value in training
    y_norm = np.exp(train_log_y - train_log_y[0])
    x_norm = np.exp(train_log_x - train_log_x[0])
    dist = y_norm - x_norm
    state["y_norm_base"] = train_log_y[-1]
    state["x_norm_base"] = train_log_x[-1]
    state["dist_mean"] = float(np.mean(dist))
    state["dist_std"] = float(np.std(dist))
    if state["dist_std"] <= 1e-12: state["dist_std"] = 1
    # GHE on distance series
    h = hurst_exponent(dist)
    state["hurst"] = h
    state["beta"] = 1.0  # distance method: equal weight
    return state

def distance_signal(log_y, log_x, i, state):
    if np.isnan(state["y_norm_base"]): return None
    # Only trade when Hurst < 0.45 (mean-reverting regime)
    if state.get("hurst", 0.5) >= 0.45: return None
    y_norm = np.exp(log_y[i] - state["y_norm_base"])
    x_norm = np.exp(log_x[i] - state["x_norm_base"])
    dist = y_norm - x_norm
    z = (dist - state["dist_mean"]) / state["dist_std"]
    return (z, state["beta"])


# ============================================================
# Main
# ============================================================

def run_experiment(exp_name, pairs, signal_func, state_init, refit_func, close_data, lb_months=2):
    """Run one experiment across all specified pairs."""
    results = []
    for y_name, x_name in pairs:
        if y_name not in close_data or x_name not in close_data: continue
        r = simulate_pair_generic(
            close_data[y_name], close_data[x_name], y_name, x_name,
            signal_func, state_init, refit_func, lb_months=lb_months,
        )
        if r: r["pair"] = f"{y_name}/{x_name}"; results.append(r)
    return results


def compute_persistence_scores(close_data, pairs):
    """Compute rolling EG persistence for each pair."""
    months = pd.period_range("2024-01", "2026-02", freq="M")
    persistence = {}
    for y_name, x_name in pairs:
        if y_name not in close_data or x_name not in close_data: continue
        y_s = close_data[y_name]; x_s = close_data[x_name]
        common = y_s.index.intersection(x_s.index)
        y = y_s.reindex(common).astype(float); x = x_s.reindex(common).astype(float)
        passes = []
        for month in months:
            ms = pd.Timestamp(month.start_time, tz="UTC")
            lb = ms - pd.DateOffset(months=6)
            mask = (y.index >= lb) & (y.index < ms)
            ya = y[mask].values; xa = x[mask].values
            if len(ya) < 2000: passes.append(False); continue
            try:
                _, p, _ = coint(ya, xa, maxlag=10)
                passes.append(p < 0.05)
            except: passes.append(False)
        # Persistence = fraction of last 6 months that passed
        key = (y_name, x_name)
        recent = passes[-6:] if len(passes) >= 6 else passes
        persistence[key] = sum(recent) / max(len(recent), 1)
    return persistence


def main():
    t0 = time.time()
    print("=" * 110)
    print("COINTEGRATION PHASE 1 — 4 EXPERIMENTS")
    print("=" * 110)

    # Load data from parquet cache
    print("\nLoading from parquet cache...", end=" ", flush=True)
    close_data = {}
    for f in sorted(CACHE_DIR.glob("*.parquet")):
        name = f.stem.replace("USDT", "")
        try:
            df = pd.read_parquet(f)
            if len(df) >= 2000: close_data[name] = df["close"]
        except: pass
    print(f"{len(close_data)} assets in {time.time()-t0:.1f}s")

    # ---- Exp 1: Kalman Filter ----
    print("\n" + "=" * 80)
    print("EXP 1: KALMAN FILTER (adaptive hedge ratio, state covariance breakdown)")
    print("=" * 80)

    results_kalman = run_experiment("Kalman", CANDIDATE_PAIRS,
        kalman_signal, kalman_state_init, kalman_refit, close_data)

    # ---- Exp 4: Persistence-Weighted ----
    print("\n" + "=" * 80)
    print("EXP 4: PERSISTENCE-WEIGHTED SELECTION (rank by rolling EG pass rate)")
    print("=" * 80)

    pers_scores = compute_persistence_scores(close_data, CANDIDATE_PAIRS)
    # Only trade pairs with persistence > 0.5
    high_pers_pairs = [p for p in CANDIDATE_PAIRS if pers_scores.get(p, 0) > 0.5]
    print(f"\nPairs with persistence > 0.5: {len(high_pers_pairs)}")
    for p in high_pers_pairs:
        print(f"  {p[0]}/{p[1]}: {pers_scores[p]:.2f}")

    results_persistence = run_experiment("Persistence", high_pers_pairs,
        persistence_signal, persistence_state_init, persistence_refit, close_data)

    # ---- Exp 6: Distance + GHE ----
    print("\n" + "=" * 80)
    print("EXP 6: DISTANCE METHOD + GHE (Gatev + Hurst filter)")
    print("=" * 80)

    results_distance = run_experiment("Distance+GHE", CANDIDATE_PAIRS,
        distance_signal, distance_state_init, distance_refit, close_data)

    # ---- Exp 7: Sector + Min Streak ----
    print("\n" + "=" * 80)
    print("EXP 7: SECTOR-CONSTRAINED + MIN STREAK")
    print("=" * 80)

    # Filter: same sector only
    sector_pairs = [(y, x) for y, x in CANDIDATE_PAIRS
                    if ASSET_SECTOR.get(y) == ASSET_SECTOR.get(x)]
    # Filter: persistence > 0.5 (min streak proxy)
    sector_pers_pairs = [p for p in sector_pairs if pers_scores.get(p, 0) > 0.3]
    print(f"\nSame-sector pairs: {len(sector_pairs)}")
    print(f"Same-sector + persistence > 0.3: {len(sector_pers_pairs)}")
    for p in sector_pers_pairs:
        sec = ASSET_SECTOR.get(p[0], "?")
        print(f"  {p[0]}/{p[1]} ({sec}): pers={pers_scores.get(p, 0):.2f}")

    # Also scan ALL same-sector pairs from universe
    all_sector_pairs = []
    sector_assets = {}
    for name in close_data:
        sec = ASSET_SECTOR.get(name)
        if sec: sector_assets.setdefault(sec, []).append(name)
    for sec, assets in sector_assets.items():
        if len(assets) >= 2:
            for y, x in combinations(assets, 2):
                all_sector_pairs.append((y, x))
    print(f"\nAll same-sector pairs from universe: {len(all_sector_pairs)}")

    results_sector = run_experiment("Sector+Streak", sector_pers_pairs if sector_pers_pairs else all_sector_pairs[:10],
        persistence_signal, persistence_state_init, persistence_refit, close_data)

    # ---- Comparison ----
    print("\n" + "=" * 110)
    print("RESULTS COMPARISON")
    print("=" * 110)

    experiments = [
        ("Exp 1: Kalman", results_kalman),
        ("Exp 4: Persistence", results_persistence),
        ("Exp 6: Distance+GHE", results_distance),
        ("Exp 7: Sector+Streak", results_sector),
    ]

    print(f"\n{'Experiment':<25s} {'Pairs':>5s} {'$/cal':>7s} {'AvgSR':>7s} {'MaxDD':>6s} "
          f"{'MaxWD':>6s} {'Trades':>6s} {'Status':>6s}")
    print("-" * 80)

    for name, results in experiments:
        if not results:
            print(f"{name:<25s}     0  NO RESULTS")
            continue
        total_pnl = sum(r["pnl"] for r in results)
        avg_sharpe = np.mean([r["sharpe"] for r in results])
        max_dd = max(r["max_dd"] for r in results)
        max_wd = max(r["worst_day"] for r in results)
        total_trades = sum(r["n_trades"] for r in results)
        pnl_cal = total_pnl / CAL_DAYS
        ok = max_dd < 0.10 and max_wd < 0.05
        print(f"{name:<25s} {len(results):>5d} ${pnl_cal:>6.2f} {avg_sharpe:>7.2f} "
              f"{max_dd*100:>5.1f}% {max_wd*100:>5.2f}% {total_trades:>6d} {'OK' if ok else 'FAIL'}")

    print(f"\nBaseline: Static 5-pair = Sharpe 1.58, $3.25/cal, DD 4.0%, WD 2.23%")

    # Per-pair detail for best experiment
    for name, results in experiments:
        if not results: continue
        print(f"\n--- {name} per-pair ---")
        print(f"{'Pair':<15s} {'Sharpe':>7s} {'$/cal':>7s} {'DD%':>6s} {'WD%':>6s} {'#Tr':>5s} {'WR%':>5s}")
        for r in sorted(results, key=lambda x: -x["sharpe"]):
            print(f"{r['pair']:<15s} {r['sharpe']:>7.2f} ${r['pnl_cal']:>6.2f} "
                  f"{r['max_dd']*100:>5.1f}% {r['worst_day']*100:>5.2f}% "
                  f"{r['n_trades']:>5d} {r['win_rate']:>4.0f}%")

    print(f"\nTotal runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
