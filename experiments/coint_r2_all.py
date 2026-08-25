"""Cointegration Round 2 — Rolling Health Check, 8 Experiments

Key fix: monthly scan is a HEALTH CHECK, not a rotation.
- KEEP pairs that still pass cointegration
- DROP only after 2 consecutive monthly failures
- ADD only after 2 consecutive monthly passes (at small size)
- Core book is sticky → minimal turnover

8 experiments test different signal/sizing/hedge methods on top of this.
"""

import sys, os, time, warnings, math
from pathlib import Path
from itertools import combinations
from dataclasses import dataclass, field

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

warnings.filterwarnings("ignore")
CACHE_DIR = Path("data/cache_15m")

CAPITAL = 10000.0; COST_BPS = 5.0; MAX_HOLD = 96 * 30
EMBARGO = 96 * 7; LB_MONTHS = 2; MIN_AGE_DAYS = 180
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


@dataclass
class PairBook:
    """Rolling pair book with hysteresis."""
    y: str; x: str
    consecutive_passes: int = 0
    consecutive_fails: int = 0
    months_in_book: int = 0
    beta: float = np.nan; intercept: float = np.nan
    mu: float = 0; sigma: float = 1
    position: int = 0; entry_bar: int = 0
    # Kalman state (for R2-3)
    kalman_theta: np.ndarray = field(default_factory=lambda: np.array([1.0, 0.0]))
    kalman_P: np.ndarray = field(default_factory=lambda: np.eye(2))


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
    return [a for a, _ in ranked[:30]]


def monthly_screen(close_panel, month_start, universe, min_corr=0.55, min_p=0.01):
    """Screen all pairs. Returns set of (y,x) that pass."""
    screen_end = month_start - pd.Timedelta(minutes=15 * EMBARGO)
    screen_start = screen_end - pd.DateOffset(months=6)
    mask = (close_panel.index >= screen_start) & (close_panel.index < screen_end)
    avail = [a for a in universe if a in close_panel.columns]
    screen = close_panel.loc[mask, avail].dropna(axis=1, how="all")
    assets = list(screen.columns)
    if len(assets) < 4: return set()

    ret = screen.pct_change().dropna()
    if len(ret) < 2000: return set()
    corr = ret.corr()

    passing = set()
    for i, j in combinations(range(len(assets)), 2):
        c = corr.iloc[i, j]
        if not np.isfinite(c) or c < min_corr: continue
        pair_data = screen[[assets[i], assets[j]]].dropna()
        if len(pair_data) < 3000: continue
        y_arr = pair_data.iloc[:, 0].values.astype(float)
        x_arr = pair_data.iloc[:, 1].values.astype(float)
        try:
            _, p, _ = coint(y_arr, x_arr, maxlag=10)
        except: continue
        if p < 0.05:  # Loose check for KEEP (existing pairs)
            passing.add((assets[i], assets[j]))
            passing.add((assets[j], assets[i]))  # Both orientations
    return passing


def fit_pair_ols(close_panel, y_name, x_name, month_start, lb_months=2):
    """Fit OLS on training window with embargo."""
    train_end = month_start - pd.Timedelta(minutes=15 * EMBARGO)
    train_start = train_end - pd.DateOffset(months=lb_months)
    mask = (close_panel.index >= train_start) & (close_panel.index < train_end)
    if y_name not in close_panel.columns or x_name not in close_panel.columns:
        return None
    train = close_panel.loc[mask, [y_name, x_name]].dropna()
    if len(train) < 96 * 30: return None
    log_y = np.log(train[y_name].values); log_x = np.log(train[x_name].values)
    beta, intercept = ols_fit(log_y, log_x)
    if not np.isfinite(beta) or beta <= 0: return None
    resid = log_y - beta * log_x - intercept
    mu = float(np.mean(resid)); sigma = float(np.std(resid))
    if sigma <= 1e-12: return None
    return {"beta": beta, "intercept": intercept, "mu": mu, "sigma": sigma}


def run_experiment(name, close_panel, ret_panel, ohlcv_data, config):
    """Run one experiment with rolling book management."""
    ws = pd.Timestamp(TEST_START, tz="UTC"); we = pd.Timestamp(TEST_END, tz="UTC")
    months = pd.period_range(start=ws.to_period("M"), end=we.to_period("M"), freq="M")

    max_pairs = config.get("max_pairs", 5)
    entry_z = config.get("entry_z", 2.0)
    exit_z = config.get("exit_z", 0.0)
    size_pct = config.get("size_pct", 0.03)
    lb_months = config.get("lb_months", 2)
    min_corr = config.get("min_corr", 0.55)
    use_kalman = config.get("use_kalman", False)
    use_persistence_sizing = config.get("use_persistence_sizing", False)
    use_vecm_gate = config.get("use_vecm_gate", False)

    book: dict[tuple, PairBook] = {}  # (y,x) -> PairBook
    candidates: dict[tuple, int] = {}  # (y,x) -> consecutive passes (not yet in book)
    daily_pnl = {}; total_trades = 0

    for month in months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = ms + pd.offsets.MonthEnd(0)

        # Monthly health check
        universe = select_universe(ohlcv_data, ms)
        passing = monthly_screen(close_panel, ms, universe, min_corr=min_corr)

        # Update book: KEEP or mark for DROP
        for key in list(book.keys()):
            pair = book[key]
            if key in passing or (key[1], key[0]) in passing:
                pair.consecutive_fails = 0
                pair.consecutive_passes += 1
                pair.months_in_book += 1
            else:
                pair.consecutive_fails += 1
                pair.consecutive_passes = 0
                if pair.consecutive_fails >= 2:
                    # DROP: close position, remove from book
                    book.pop(key)
                    continue

            # Refit OLS monthly
            fit = fit_pair_ols(close_panel, key[0], key[1], ms, lb_months)
            if fit:
                pair.beta = fit["beta"]; pair.intercept = fit["intercept"]
                pair.mu = fit["mu"]; pair.sigma = fit["sigma"]

        # Candidates: track consecutive passes for new pairs
        for key in passing:
            if key in book or (key[1], key[0]) in book: continue
            candidates[key] = candidates.get(key, 0) + 1
            if candidates[key] >= 2 and len(book) < max_pairs:
                # ADD: new pair enters book
                fit = fit_pair_ols(close_panel, key[0], key[1], ms, lb_months)
                if fit:
                    book[key] = PairBook(y=key[0], x=key[1], months_in_book=0,
                                         consecutive_passes=2, **{k: v for k, v in fit.items()})
                    candidates.pop(key, None)

        # Clean old candidates
        for key in list(candidates.keys()):
            if key not in passing and (key[1], key[0]) not in passing:
                candidates.pop(key, None)

        # VECM gate: compute alpha for active pairs (if enabled)
        vecm_ok = {}
        if use_vecm_gate:
            from statsmodels.tsa.vector_ar.vecm import VECM as _VECM
            for key, pair in book.items():
                train_end = ms - pd.Timedelta(minutes=15 * EMBARGO)
                train_start = train_end - pd.DateOffset(months=3)
                tmask = (close_panel.index >= train_start) & (close_panel.index < train_end)
                if pair.y in close_panel.columns and pair.x in close_panel.columns:
                    train = close_panel.loc[tmask, [pair.y, pair.x]].dropna()
                    if len(train) >= 96 * 30:
                        try:
                            data = np.column_stack([np.log(train.iloc[:, 0].values),
                                                    np.log(train.iloc[:, 1].values)])
                            vecm = _VECM(data, k_ar_diff=2, coint_rank=1, deterministic="ci")
                            result = vecm.fit()
                            alpha = abs(float(result.alpha[0, 0]))
                            vecm_ok[key] = alpha > 0.001
                        except: vecm_ok[key] = True
                    else: vecm_ok[key] = True
                else: vecm_ok[key] = True

        # Bar-level trading within month
        month_bars = close_panel.index[(close_panel.index >= ms) & (close_panel.index <= me)]

        for ts in month_bars:
            for key, pair in book.items():
                if not np.isfinite(pair.beta) or not np.isfinite(pair.sigma) or pair.sigma <= 0:
                    continue
                if pair.y not in close_panel.columns or pair.x not in close_panel.columns:
                    continue

                y_p = close_panel.at[ts, pair.y]; x_p = close_panel.at[ts, pair.x]
                if np.isnan(y_p) or np.isnan(x_p) or y_p <= 0 or x_p <= 0: continue

                # Z-score
                if use_kalman:
                    # Kalman update
                    delta_k = 1e-4; R_k = 1e-3
                    F = np.array([[math.log(x_p), 1.0]])
                    P_pred = pair.kalman_P + delta_k * np.eye(2)
                    y_pred = F @ pair.kalman_theta
                    S = float((F @ P_pred @ F.T + R_k)[0, 0])
                    K = P_pred @ F.T / S
                    e = math.log(y_p) - y_pred[0]
                    pair.kalman_theta = pair.kalman_theta + K.flatten() * e
                    pair.kalman_P = (np.eye(2) - K @ F) @ P_pred
                    z = e / math.sqrt(S) if S > 0 else 0
                    beta_eff = float(pair.kalman_theta[0])
                else:
                    spread = math.log(y_p) - pair.beta * math.log(x_p) - pair.intercept
                    z = (spread - pair.mu) / pair.sigma
                    beta_eff = pair.beta

                # Sizing
                if use_persistence_sizing:
                    if pair.months_in_book >= 6: sz = 0.05
                    elif pair.months_in_book >= 3: sz = 0.03
                    else: sz = 0.015
                else:
                    sz = size_pct

                notional = CAPITAL * sz
                y_r = ret_panel.at[ts, pair.y] if ts in ret_panel.index and pair.y in ret_panel.columns else 0
                x_r = ret_panel.at[ts, pair.x] if ts in ret_panel.index and pair.x in ret_panel.columns else 0
                if np.isnan(y_r): y_r = 0
                if np.isnan(x_r): x_r = 0

                bar_pnl = 0.0
                if pair.position != 0:
                    bar_pnl = pair.position * (y_r - beta_eff * x_r) * notional

                cost = COST_BPS / 10000 * (1 + abs(beta_eff)) * notional

                # Exit
                if pair.position != 0:
                    if isinstance(pair.entry_bar, pd.Timestamp):
                        bh = (ts - pair.entry_bar).total_seconds() / (15 * 60)
                    else:
                        bh = 0
                    should_exit = False
                    if bh >= 96:  # Min hold: 1 day (96 bars) to prevent cycling
                        if pair.position == 1 and z >= exit_z: should_exit = True
                        elif pair.position == -1 and z <= -exit_z: should_exit = True
                    if bh >= MAX_HOLD: should_exit = True
                    if abs(z) > 5 and bh > 48: should_exit = True  # breakdown override
                    if should_exit: bar_pnl -= cost; pair.position = 0

                # Entry (with optional VECM gate)
                if pair.position == 0:
                    gate_ok = True
                    if use_vecm_gate: gate_ok = vecm_ok.get(key, True)
                    if gate_ok:
                        if z <= -entry_z:
                            pair.position = 1; pair.entry_bar = ts; bar_pnl -= cost; total_trades += 1
                        elif z >= entry_z:
                            pair.position = -1; pair.entry_bar = ts; bar_pnl -= cost; total_trades += 1

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

    return {"name": name, "sharpe": sharpe, "pnl_cal": pnl_cal, "max_dd": dd,
            "worst_day": wd, "n_trades": total_trades,
            "pos_months": f"{pos_months}/{len(monthly_pnl)} ({pos_months/len(monthly_pnl)*100:.0f}%)",
            "monthly_pnl": monthly_pnl}


def main():
    t0 = time.time()
    print("=" * 110)
    print("COINTEGRATION ROUND 2 — Rolling Health Check, 8 Experiments")
    print("=" * 110)

    print("\nLoading from parquet cache...", end=" ", flush=True)
    ohlcv_data = {}
    for f in sorted(CACHE_DIR.glob("*.parquet")):
        name = f.stem.replace("USDT", "")
        try:
            df = pd.read_parquet(f, columns=["close", "volume"])
            if len(df) >= 2000: ohlcv_data[name] = df
        except: pass
    close_panel = pd.DataFrame({n: d["close"] for n, d in ohlcv_data.items()})
    ret_panel = close_panel.pct_change()
    print(f"{len(ohlcv_data)} assets in {time.time()-t0:.1f}s", flush=True)

    experiments = [
        ("R2-1: Rolling EG tight", {"max_pairs": 5, "entry_z": 2.0, "min_corr": 0.55}),
        ("R2-2: Rolling+PersSizing", {"max_pairs": 5, "entry_z": 2.0, "use_persistence_sizing": True}),
        ("R2-3: Rolling+Kalman", {"max_pairs": 5, "entry_z": 1.5, "use_kalman": True}),
        ("R2-5: Rolling+6mo OLS", {"max_pairs": 5, "entry_z": 2.0, "lb_months": 6}),
        ("R2-6: Rolling+exit z=0.5", {"max_pairs": 5, "entry_z": 2.0, "exit_z": 0.5}),
        ("R2-7: Rolling+VECM gate", {"max_pairs": 5, "entry_z": 2.0, "use_vecm_gate": True}),
        ("R2-8: Best combo", {"max_pairs": 5, "entry_z": 2.0, "use_persistence_sizing": True,
                               "use_kalman": True, "exit_z": 0.3}),
        ("R2-1b: Loose (corr>0.45)", {"max_pairs": 8, "entry_z": 1.5, "min_corr": 0.45, "size_pct": 0.02}),
    ]

    results = []
    for name, config in experiments:
        print(f"\nRunning: {name}...", end=" ", flush=True)
        et0 = time.time()
        r = run_experiment(name, close_panel, ret_panel, ohlcv_data, config)
        elapsed = time.time() - et0
        if r:
            results.append(r)
            ok = r["max_dd"] < 0.07 and r["worst_day"] < 0.05
            print(f"Sharpe {r['sharpe']:.2f}, ${r['pnl_cal']:.2f}/cal, DD {r['max_dd']*100:.1f}%, "
                  f"Trades {r['n_trades']}, +Mo {r['pos_months']} ({elapsed:.0f}s) {'OK' if ok else 'FAIL'}")
        else:
            print(f"NO DATA ({elapsed:.0f}s)")

    # Final comparison
    print(f"\n{'='*110}")
    print("ROUND 2 RESULTS")
    print(f"{'='*110}")

    print(f"\n{'Experiment':<30s} {'$/cal':>7s} {'Sharpe':>7s} {'DD%':>6s} {'WD%':>6s} "
          f"{'Trades':>6s} {'+Mo':>12s}")
    print("-" * 85)
    for r in sorted(results, key=lambda x: -x["sharpe"]):
        print(f"{r['name']:<30s} ${r['pnl_cal']:>6.2f} {r['sharpe']:>7.2f} "
              f"{r['max_dd']*100:>5.1f}% {r['worst_day']*100:>5.2f}% "
              f"{r['n_trades']:>6d} {r['pos_months']:>12s}")

    print(f"\nBaselines: Static=Sharpe 1.58 $3.25/cal | R1 best=Sharpe 0.32 $0.49/cal")

    # Best method monthly PnL
    if results:
        best = max(results, key=lambda r: r["sharpe"])
        print(f"\nBest: {best['name']}")
        print(f"\n{'Month':<10s} {'PnL':>8s}")
        for dt in best["monthly_pnl"].index:
            v = best["monthly_pnl"].loc[dt]
            flag = " <<< NEG" if v < -50 else ""
            print(f"{dt.strftime('%Y-%m'):<10s} ${v:>7.0f}{flag}")

    print(f"\nRuntime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
