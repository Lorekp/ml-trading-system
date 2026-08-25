"""Cointegration — Full Universe Walk-Forward Scan

NO pre-selected pairs. Every month:
1. Take top 30 assets by volume (dynamic universe)
2. Screen ALL 435 pairs with the method under test
3. Select top K that pass
4. Trade for that month
5. Repeat

Tests 5 methods side-by-side on the SAME monthly scans:
A) Standard EG (baseline dynamic — already showed Sharpe 0.69)
B) VECM alpha filter (only pairs with strong error correction)
C) Persistence filter (only pairs that passed 4+ of last 6 months)
D) Distance + GHE (Gatev method + Hurst < 0.45)
E) Combined: VECM alpha > 0.001 AND persistence > 0.5

All methods use the same universe, same walk-forward, same bar-level PnL.
"""

import sys, os, time, warnings, math
from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")
CACHE_DIR = Path("data/cache_15m")

TOP_N = 30; MAX_PAIRS = 8; ENTRY_Z = 1.5; COST_BPS = 5.0
MAX_HOLD = 96 * 30; EMBARGO = 96 * 7; LB_MONTHS = 2
CAPITAL = 10000.0; SIZE_PCT = 0.03; MIN_AGE_DAYS = 180
COINT_P = 0.05; MIN_HL = 0.25; MAX_HL = 20
TEST_START = "2024-07-01"; TEST_END = "2026-02-28"; CAL_DAYS = 611


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


def hurst_exponent(series, max_lag=64):
    if len(series) < max_lag * 3: return 0.5
    lags = np.arange(2, min(max_lag + 1, len(series) // 3))
    tau = []
    for lag in lags:
        s = np.std(series[lag:] - series[:-lag])
        tau.append(s if s > 0 and np.isfinite(s) else np.nan)
    tau_arr = np.array(tau); keep = np.isfinite(tau_arr) & (tau_arr > 0)
    if keep.sum() < 5: return 0.5
    return float(np.clip(np.polyfit(np.log(lags[keep]), np.log(tau_arr[keep]), 1)[0], 0, 1))


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
        med_vol = recent.median()
        if np.isnan(med_vol) or med_vol <= 0: continue
        if len(dv) >= 90 and dv.iloc[-90:].median() > 0 and med_vol / dv.iloc[-90:].median() > 5: continue
        candidates[name] = float(med_vol)
    ranked = sorted(candidates.items(), key=lambda x: -x[1])
    return [a for a, _ in ranked[:TOP_N]]


def screen_pair_basic(y_arr, x_arr):
    """Basic EG screen: returns (p_value, half_life_days, beta) or None."""
    if len(y_arr) < 3000: return None
    try:
        _, p, _ = coint(y_arr, x_arr, maxlag=10)
    except: return None
    if p > COINT_P: return None
    log_y, log_x = np.log(y_arr), np.log(x_arr)
    beta, intercept = ols_fit(log_y, log_x)
    if not np.isfinite(beta) or beta <= 0: return None
    resid = log_y - beta * log_x - intercept
    hl = half_life_bars(resid) / 96
    if hl < MIN_HL or hl > MAX_HL: return None
    return {"p": float(p), "hl": hl, "beta": beta, "intercept": intercept,
            "mu": float(np.mean(resid)), "sigma": float(np.std(resid)),
            "hurst": hurst_exponent(resid)}


def screen_pair_vecm(y_arr, x_arr):
    """VECM screen: standard EG + alpha > 0.001."""
    base = screen_pair_basic(y_arr, x_arr)
    if base is None: return None
    from statsmodels.tsa.vector_ar.vecm import VECM
    log_y, log_x = np.log(y_arr), np.log(x_arr)
    try:
        data = np.column_stack([log_y[-5000:], log_x[-5000:]])
        vecm = VECM(data, k_ar_diff=2, coint_rank=1, deterministic="ci")
        result = vecm.fit()
        alpha = float(result.alpha[0, 0])
    except: return None
    if abs(alpha) < 0.001: return None
    base["alpha"] = alpha
    return base


def screen_pair_distance_ghe(y_arr, x_arr):
    """Distance + GHE: normalized spread, Hurst < 0.45."""
    if len(y_arr) < 3000: return None
    y_norm = y_arr / y_arr[0]; x_norm = x_arr / x_arr[0]
    dist = y_norm - x_norm
    h = hurst_exponent(dist)
    if h >= 0.45: return None
    mu = float(np.mean(dist)); sigma = float(np.std(dist))
    if sigma <= 1e-12: return None
    return {"p": 0, "hl": 0, "beta": 1.0, "intercept": 0,
            "mu": mu, "sigma": sigma, "hurst": h, "distance_mode": True}


def monthly_scan(close_panel, ohlcv_data, month_start, method,
                 persistence_cache=None):
    """Scan full universe for one month with given method. Returns list of fitted pairs."""
    universe = select_universe(ohlcv_data, month_start)
    if len(universe) < 6: return [], universe

    screen_end = month_start - pd.Timedelta(minutes=15 * EMBARGO)
    screen_start = screen_end - pd.DateOffset(months=max(LB_MONTHS, 6))
    mask = (close_panel.index >= screen_start) & (close_panel.index < screen_end)
    avail = [a for a in universe if a in close_panel.columns]
    screen = close_panel.loc[mask, avail].dropna(axis=1, how="all")
    assets = list(screen.columns)
    if len(assets) < 4: return [], universe

    # Vectorized correlation prescreen
    ret = screen.pct_change().dropna()
    if len(ret) < 2000: return [], universe
    corr = ret.corr()

    # Build candidate list (prescreen by correlation)
    candidates = []
    for i_idx, j_idx in combinations(range(len(assets)), 2):
        c = corr.iloc[i_idx, j_idx]
        if not np.isfinite(c) or c < 0.45: continue
        pair_data = screen[[assets[i_idx], assets[j_idx]]].dropna()
        if len(pair_data) < 3000: continue
        candidates.append((i_idx, j_idx, pair_data.iloc[:, 0].values.astype(float),
                          pair_data.iloc[:, 1].values.astype(float)))

    # Screen candidates (sequential — coint() not picklable for ProcessPool)
    valid_pairs = []
    for i_idx, j_idx, y_arr, x_arr in candidates:
        if method == "eg_basic":
            result = screen_pair_basic(y_arr, x_arr)
        elif method == "vecm":
            result = screen_pair_vecm(y_arr, x_arr)
        elif method == "distance_ghe":
            result = screen_pair_distance_ghe(y_arr, x_arr)
        elif method == "persistence":
            result = screen_pair_basic(y_arr, x_arr)
            if result and persistence_cache:
                key = (assets[i_idx], assets[j_idx])
                pers = persistence_cache.get(key, 0)
                if pers < 0.5: result = None
        elif method == "combined":
            result = screen_pair_vecm(y_arr, x_arr)
            if result and persistence_cache:
                key = (assets[i_idx], assets[j_idx])
                pers = persistence_cache.get(key, 0)
                if pers < 0.3: result = None
        else:
            result = screen_pair_basic(y_arr, x_arr)

        if result:
            result["y"] = assets[i_idx]; result["x"] = assets[j_idx]
            valid_pairs.append(result)

    # Rank by half-life (faster = better) for cointegration methods,
    # by hurst for distance method
    if method == "distance_ghe":
        valid_pairs.sort(key=lambda p: p.get("hurst", 1))
    else:
        valid_pairs.sort(key=lambda p: p.get("hl", 999))

    return valid_pairs[:MAX_PAIRS], universe


def compute_rolling_persistence(close_panel, month_start, universe):
    """Compute persistence scores for all pairs in universe over last 6 months."""
    cache = {}
    check_months = pd.period_range(
        (month_start - pd.DateOffset(months=6)).to_period("M"),
        month_start.to_period("M") - 1, freq="M"
    )
    for a_i, a_j in combinations(universe, 2):
        if a_i not in close_panel.columns or a_j not in close_panel.columns: continue
        passes = 0; total = 0
        for m in check_months:
            ms = pd.Timestamp(m.start_time, tz="UTC")
            lb = ms - pd.DateOffset(months=6)
            mask = (close_panel.index >= lb) & (close_panel.index < ms)
            ya = close_panel.loc[mask, a_i].dropna().values
            xa = close_panel.loc[mask, a_j].dropna().values
            if len(ya) < 2000 or len(xa) < 2000: continue
            total += 1
            try:
                _, p, _ = coint(ya.astype(float), xa.astype(float), maxlag=10)
                if p < 0.05: passes += 1
            except: pass
        if total > 0:
            cache[(a_i, a_j)] = passes / total
            cache[(a_j, a_i)] = passes / total
    return cache


def simulate_month(close_panel, ret_panel, active_pairs, month_start, month_end):
    """Simulate all active pairs for one month. Returns daily PnL dict and trade count."""
    month_bars = close_panel.index[(close_panel.index >= month_start) & (close_panel.index <= month_end)]
    daily_pnl = {}; n_trades = 0; n_breakdowns = 0

    for pair in active_pairs:
        y_name, x_name = pair["y"], pair["x"]
        beta = pair["beta"]; mu = pair["mu"]; sigma = pair["sigma"]
        if not np.isfinite(beta) or not np.isfinite(sigma) or sigma <= 0: continue

        is_distance = pair.get("distance_mode", False)
        pos = 0; entry_bar_i = 0

        for bar_idx, ts in enumerate(month_bars):
            if y_name not in close_panel.columns or x_name not in close_panel.columns: continue
            y_p = close_panel.at[ts, y_name]; x_p = close_panel.at[ts, x_name]
            if np.isnan(y_p) or np.isnan(x_p) or y_p <= 0 or x_p <= 0: continue

            if is_distance:
                y_norm = y_p / pair.get("y_base", y_p)
                x_norm = x_p / pair.get("x_base", x_p)
                z = (y_norm - x_norm - mu) / sigma
            else:
                spread = math.log(y_p) - beta * math.log(x_p) - pair.get("intercept", 0)
                z = (spread - mu) / sigma

            y_r = ret_panel.at[ts, y_name] if ts in ret_panel.index and y_name in ret_panel.columns else 0
            x_r = ret_panel.at[ts, x_name] if ts in ret_panel.index and x_name in ret_panel.columns else 0
            if np.isnan(y_r): y_r = 0
            if np.isnan(x_r): x_r = 0

            notional = CAPITAL * SIZE_PCT
            bar_pnl = 0.0
            if pos != 0:
                bar_pnl = pos * (y_r - beta * x_r) * notional

            cost = COST_BPS / 10000 * (1 + abs(beta)) * notional

            # Exit
            if pos != 0:
                bh = bar_idx - entry_bar_i
                should_exit = False
                if pos == 1 and z >= 0: should_exit = True
                elif pos == -1 and z <= 0: should_exit = True
                elif bh >= MAX_HOLD: should_exit = True
                elif abs(z) > 5 and bh > 48: should_exit = True; n_breakdowns += 1
                if should_exit: bar_pnl -= cost; pos = 0

            # Entry
            entry_z = ENTRY_Z if not is_distance else 2.0
            if pos == 0:
                if z <= -entry_z: pos = 1; entry_bar_i = bar_idx; bar_pnl -= cost; n_trades += 1
                elif z >= entry_z: pos = -1; entry_bar_i = bar_idx; bar_pnl -= cost; n_trades += 1

            day = ts.normalize()
            daily_pnl[day] = daily_pnl.get(day, 0) + bar_pnl

    return daily_pnl, n_trades, n_breakdowns


def run_full_walkforward(close_panel, ret_panel, ohlcv_data, method_name, method_key):
    """Full walk-forward: monthly scan + simulate. Returns metrics."""
    ws = pd.Timestamp(TEST_START, tz="UTC"); we = pd.Timestamp(TEST_END, tz="UTC")
    months = pd.period_range(start=ws.to_period("M"), end=we.to_period("M"), freq="M")

    all_daily_pnl = {}; total_trades = 0; total_breakdowns = 0
    monthly_log = []

    for month in months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = ms + pd.offsets.MonthEnd(0)
        mt0 = time.time()

        # Persistence cache (needed for persistence/combined methods)
        universe = select_universe(ohlcv_data, ms)
        pers_cache = None
        if method_key in ("persistence", "combined"):
            pers_cache = compute_rolling_persistence(close_panel, ms, universe)

        pairs, univ = monthly_scan(close_panel, ohlcv_data, ms, method_key, pers_cache)

        # Set base prices for distance method
        for p in pairs:
            if p.get("distance_mode"):
                if p["y"] in close_panel.columns and p["x"] in close_panel.columns:
                    recent = close_panel.loc[:ms - pd.Timedelta(days=1)]
                    if len(recent) > 0:
                        p["y_base"] = float(recent[p["y"]].dropna().iloc[-1])
                        p["x_base"] = float(recent[p["x"]].dropna().iloc[-1])

        dpnl, nt, nb = simulate_month(close_panel, ret_panel, pairs, ms, me)
        for day, pnl in dpnl.items():
            all_daily_pnl[day] = all_daily_pnl.get(day, 0) + pnl
        total_trades += nt; total_breakdowns += nb

        elapsed = time.time() - mt0
        monthly_log.append({
            "month": str(month), "univ": len(univ), "pairs": len(pairs),
            "trades": nt, "time": elapsed,
        })
        print(f"    {str(month)} univ={len(univ)} pairs={len(pairs)} trades={nt} ({elapsed:.0f}s)", flush=True)

    # Metrics
    dpnl = pd.Series(all_daily_pnl)
    if len(dpnl) < 10: return None

    cum = dpnl.cumsum(); dr = dpnl / CAPITAL
    sharpe = dr.mean() / dr.std() * np.sqrt(365.25) if dr.std() > 0 else 0
    peak = (cum + CAPITAL).cummax(); dd = abs(((cum + CAPITAL - peak) / peak).min())
    wd = abs(dr.min()) if len(dr) > 0 else 0
    total_pnl = cum.iloc[-1]; pnl_cal = total_pnl / CAL_DAYS
    monthly_pnl = dpnl.resample("ME").sum()
    pos_months = (monthly_pnl > 0).sum(); total_months = len(monthly_pnl)

    return {
        "method": method_name, "sharpe": sharpe, "pnl_cal": pnl_cal,
        "max_dd": dd, "worst_day": wd, "n_trades": total_trades,
        "n_breakdowns": total_breakdowns,
        "pos_months": f"{pos_months}/{total_months} ({pos_months/total_months*100:.0f}%)",
        "monthly_log": monthly_log, "monthly_pnl": monthly_pnl,
    }


def main():
    t0 = time.time()
    print("=" * 110)
    print("COINTEGRATION — FULL UNIVERSE WALK-FORWARD (no pre-selected pairs)")
    print("=" * 110)

    print("\nLoading from parquet cache...", end=" ", flush=True)
    ohlcv_data = {}  # name -> DataFrame with close + volume only
    for f in sorted(CACHE_DIR.glob("*.parquet")):
        name = f.stem.replace("USDT", "")
        try:
            df = pd.read_parquet(f, columns=["close", "volume"])
            if len(df) >= 2000: ohlcv_data[name] = df
        except: pass
    close_panel = pd.DataFrame({n: d["close"] for n, d in ohlcv_data.items()})
    ret_panel = close_panel.pct_change()
    print(f"{len(ohlcv_data)} assets in {time.time()-t0:.1f}s", flush=True)

    methods = [
        ("A) EG Basic (baseline)", "eg_basic"),
        ("B) VECM alpha>0.001", "vecm"),
        ("C) Persistence>0.5", "persistence"),
        ("D) Distance+GHE<0.45", "distance_ghe"),
        ("E) VECM+Persistence", "combined"),
    ]

    all_results = []
    for method_name, method_key in methods:
        print(f"\n{'='*80}")
        print(f"Running: {method_name}")
        print(f"{'='*80}")

        r = run_full_walkforward(close_panel, ret_panel, ohlcv_data, method_name, method_key)
        if r:
            all_results.append(r)
            # Print monthly log
            for m in r["monthly_log"]:
                pass  # Suppress verbose output
            print(f"  Sharpe: {r['sharpe']:.2f}, $/cal: ${r['pnl_cal']:.2f}, "
                  f"DD: {r['max_dd']*100:.1f}%, Trades: {r['n_trades']}, "
                  f"+Months: {r['pos_months']}")
        else:
            print(f"  NO RESULTS")

    # ---- Final comparison ----
    print(f"\n{'='*110}")
    print("FINAL COMPARISON — Full Universe Walk-Forward")
    print(f"{'='*110}")

    print(f"\n{'Method':<30s} {'$/cal':>7s} {'Sharpe':>7s} {'DD%':>6s} {'WD%':>6s} "
          f"{'Trades':>6s} {'BDs':>4s} {'+Mo':>12s} {'Status':>6s}")
    print("-" * 100)

    for r in all_results:
        ok = r["max_dd"] < 0.10 and r["worst_day"] < 0.05
        print(f"{r['method']:<30s} ${r['pnl_cal']:>6.2f} {r['sharpe']:>7.2f} "
              f"{r['max_dd']*100:>5.1f}% {r['worst_day']*100:>5.2f}% "
              f"{r['n_trades']:>6d} {r['n_breakdowns']:>4d} {r['pos_months']:>12s} "
              f"{'OK' if ok else 'FAIL'}")

    print(f"\nStatic 5-pair baseline: Sharpe 1.58, $3.25/cal, DD 4.0%, WD 2.23%")

    # Monthly PnL for best method
    if all_results:
        best = max(all_results, key=lambda r: r["sharpe"])
        print(f"\nBest method: {best['method']}")
        print(f"\n{'Month':<10s} {'PnL':>8s}")
        print("-" * 20)
        for dt in best["monthly_pnl"].index:
            v = best["monthly_pnl"].loc[dt]
            flag = " <<< NEG" if v < -50 else ""
            print(f"{dt.strftime('%Y-%m'):<10s} ${v:>7.0f}{flag}")

    print(f"\nRuntime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
