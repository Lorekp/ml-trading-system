"""Dynamic Cointegration Strategy — Fully Parallelized Backtest

Monthly cycle:
1. Dynamic universe: top 30 by dollar volume (age/spike filtered)
2. Parallel pair screening: EG coint + half-life + beta stability
3. Top K pairs by half-life, OLS fit with embargo
4. Bar-level z-score trading with breakdown detection

All screening parallelized via ProcessPoolExecutor.
All data loaded from parquet cache.
"""

import sys, os, time, warnings, math
from pathlib import Path
from itertools import combinations
from concurrent.futures import ProcessPoolExecutor, as_completed

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
CACHE_DIR = Path("data/cache_15m")

# Config
TOP_N = 30; MAX_PAIRS = 10; MIN_CORR = 0.45; COINT_P = 0.05
MIN_HL = 0.25; MAX_HL = 20; MAX_BETA_STAB = 3.0; ENTRY_Z = 1.5
MAX_HOLD = 96 * 30; EMBARGO = 96 * 7; LB_MONTHS = 2; COST_BPS = 5.0
CAPITAL = 10000.0; SIZE_PCT = 0.03; BREAKDOWN_Z = 5.0; BREAKDOWN_BARS = 48
MIN_AGE_DAYS = 180; TEST_START = "2024-07-01"; TEST_END = "2026-02-28"; CAL_DAYS = 611


def ols_fit(y, x):
    xm, ym = x.mean(), y.mean(); xd = x - xm; d = np.dot(xd, xd)
    if d <= 0: return np.nan, np.nan
    return float(np.dot(xd, y - ym) / d), float(ym - np.dot(xd, y - ym) / d * xm)


def half_life_bars(resid):
    if len(resid) < 50: return float("inf")
    lag = resid[:-1]; delta = np.diff(resid)
    d = np.dot(lag - lag.mean(), lag - lag.mean())
    if d <= 0: return float("inf")
    phi = np.dot(lag - lag.mean(), delta - delta.mean()) / d
    return -math.log(2) / phi if phi < 0 else float("inf")


def beta_stability(y, x, splits=4):
    if len(y) < splits * 250: return float("inf")
    betas = []
    for ys, xs in zip(np.array_split(y, splits), np.array_split(x, splits)):
        b, _ = ols_fit(ys, xs)
        if not np.isfinite(b) or b <= 0: return float("inf")
        betas.append(abs(b))
    return max(betas) / min(betas) if min(betas) > 0 else float("inf")


# ---- Parallel pair screening worker ----
def _screen_one_pair(args):
    """Worker: test one pair for cointegration. Returns dict or None."""
    a_i, a_j, y_arr, x_arr = args
    from statsmodels.tsa.stattools import coint as _coint
    if len(y_arr) < 3000: return None
    try:
        _, p, _ = _coint(y_arr, x_arr, maxlag=10)
    except Exception:
        return None
    if p > COINT_P: return None

    log_y, log_x = np.log(y_arr), np.log(x_arr)
    xm, ym = log_x.mean(), log_y.mean(); xd = log_x - xm
    d = np.dot(xd, xd)
    if d <= 0: return None
    beta = float(np.dot(xd, log_y - ym) / d)
    if not np.isfinite(beta) or beta <= 0: return None
    intercept = float(ym - beta * xm)

    resid = log_y - beta * log_x - intercept
    hl = half_life_bars(resid) / 96
    if hl < MIN_HL or hl > MAX_HL: return None

    # Beta stability
    if len(log_y) >= 1000:
        betas = []
        for ys, xs in zip(np.array_split(log_y, 4), np.array_split(log_x, 4)):
            xm2, ym2 = xs.mean(), ys.mean(); xd2 = xs - xm2
            d2 = np.dot(xd2, xd2)
            if d2 <= 0: return None
            betas.append(abs(float(np.dot(xd2, ys - ym2) / d2)))
        if min(betas) <= 0 or max(betas) / min(betas) > MAX_BETA_STAB:
            return None

    return {"y": a_i, "x": a_j, "p": float(p), "hl": hl}


def screen_pairs_parallel(close_panel, month_start, universe, n_workers=8):
    """Screen all pairs from universe using parallel coint tests."""
    screen_end = month_start
    screen_start = month_start - pd.DateOffset(months=12)
    mask = (close_panel.index >= screen_start) & (close_panel.index < screen_end)
    avail = [a for a in universe if a in close_panel.columns]
    screen = close_panel.loc[mask, avail].dropna(axis=1, how="all")

    if screen.shape[1] < 4: return []

    # Vectorized correlation prescreen (instant)
    ret = screen.pct_change().dropna()
    if len(ret) < 2000: return []
    corr = ret.corr()
    assets = list(screen.columns)

    # Build tasks only for pairs passing correlation prescreen
    tasks = []
    for i, j in combinations(range(len(assets)), 2):
        c = corr.iloc[i, j]
        if not np.isfinite(c) or c < MIN_CORR: continue
        pair_data = screen[[assets[i], assets[j]]].dropna()
        if len(pair_data) < 3000: continue
        tasks.append((
            assets[i], assets[j],
            pair_data.iloc[:, 0].values.astype(np.float64),
            pair_data.iloc[:, 1].values.astype(np.float64),
        ))

    if not tasks: return []

    # Parallel coint tests
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for r in pool.map(_screen_one_pair, tasks, chunksize=max(1, len(tasks) // n_workers)):
            if r is not None:
                results.append(r)

    results.sort(key=lambda p: p["hl"])
    return results[:MAX_PAIRS]


def select_universe(close_panel, ohlcv_data, month_start):
    """Dynamic universe: top N by volume with age/spike filters."""
    lookback_end = month_start - pd.Timedelta(days=1)
    candidates = {}

    for name, o15 in ohlcv_data.items():
        # Age filter
        age = (month_start - o15.index.min()).days
        if age < MIN_AGE_DAYS: continue

        # Volume: 30-day median dollar volume
        avail = o15.loc[:lookback_end]
        if len(avail) < 96 * 45: continue  # 45 days min
        dv = (avail["close"] * avail["volume"]).resample("1D").sum()
        recent = dv.iloc[-30:] if len(dv) >= 30 else dv
        med_vol = recent.median()
        if np.isnan(med_vol) or med_vol <= 0: continue

        # Spike filter
        if len(dv) >= 90:
            m90 = dv.iloc[-90:].median()
            if m90 > 0 and med_vol / m90 > 5.0: continue

        candidates[name] = float(med_vol)

    ranked = sorted(candidates.items(), key=lambda x: -x[1])
    return [a for a, _ in ranked[:TOP_N]]


def main():
    t0 = time.time()
    print("=" * 110)
    print("DYNAMIC COINTEGRATION — Parallelized, Parquet-Cached")
    print("=" * 110)

    # Load ALL from parquet cache
    print("\nLoading from parquet cache...", end=" ", flush=True)
    ohlcv_data = {}  # name -> DataFrame
    for f in sorted(CACHE_DIR.glob("*.parquet")):
        name = f.stem.replace("USDT", "")
        try:
            df = pd.read_parquet(f)
            if len(df) >= 2000:
                ohlcv_data[name] = df
        except Exception:
            pass

    close_panel = pd.DataFrame({n: d["close"] for n, d in ohlcv_data.items()})
    print(f"{len(ohlcv_data)} assets in {time.time()-t0:.1f}s")

    # Pre-compute returns for fast simulation
    ret_panel = close_panel.pct_change()

    # Walk-forward monthly
    ws = pd.Timestamp(TEST_START, tz="UTC"); we = pd.Timestamp(TEST_END, tz="UTC")
    months = pd.period_range(start=ws.to_period("M"), end=we.to_period("M"), freq="M")

    active_pairs = []
    daily_pnl = {}
    n_trades = 0; n_breakdowns = 0
    monthly_log = []
    n_workers = min(8, os.cpu_count() or 4)

    print(f"\nSimulating {len(months)} months with {n_workers}-core parallel screening...")

    for month in months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = ms + pd.offsets.MonthEnd(0)
        mt0 = time.time()

        # Dynamic universe
        universe = select_universe(close_panel, ohlcv_data, ms)
        if len(universe) < 6:
            monthly_log.append({"month": str(month), "univ": len(universe), "found": 0, "active": 0})
            continue

        # Parallel pair screening
        new_pairs = screen_pairs_parallel(close_panel, ms, universe, n_workers)
        new_keys = {(p["y"], p["x"]) for p in new_pairs}

        # Smooth rotation
        kept = [p for p in active_pairs if (p["y"], p["x"]) in new_keys]
        added = [p for p in new_pairs if (p["y"], p["x"]) not in {(q["y"], q["x"]) for q in active_pairs}]
        exited = [p for p in active_pairs if (p["y"], p["x"]) not in new_keys]

        # Fit OLS for all active pairs
        fitted = []
        for pair_info in new_pairs:
            y_name, x_name = pair_info["y"], pair_info["x"]
            if y_name not in close_panel.columns or x_name not in close_panel.columns: continue

            train_end = ms - pd.Timedelta(minutes=15 * EMBARGO)
            train_start = train_end - pd.DateOffset(months=LB_MONTHS)
            tmask = (close_panel.index >= train_start) & (close_panel.index < train_end)
            train = close_panel.loc[tmask, [y_name, x_name]].dropna()
            if len(train) < 96 * 30: continue

            log_y = np.log(train[y_name].values); log_x = np.log(train[x_name].values)
            beta, intercept = ols_fit(log_y, log_x)
            if not np.isfinite(beta) or beta <= 0: continue
            resid = log_y - beta * log_x - intercept
            mu = float(np.mean(resid)); sigma = float(np.std(resid))
            if sigma <= 1e-12: continue

            # Preserve position state from existing pair
            existing = [p for p in kept if p["y"] == y_name and p["x"] == x_name]
            pos = existing[0]["pos"] if existing else 0
            entry_ts = existing[0].get("entry_ts") if existing else None
            bd_count = existing[0].get("bd_count", 0) if existing else 0

            fitted.append({
                "y": y_name, "x": x_name, "beta": beta, "intercept": intercept,
                "mu": mu, "sigma": sigma, "pos": pos, "entry_ts": entry_ts,
                "bd_count": bd_count, "hl": pair_info["hl"],
            })

        active_pairs = fitted
        screen_time = time.time() - mt0

        monthly_log.append({
            "month": str(month), "univ": len(universe), "found": len(new_pairs),
            "active": len(active_pairs), "added": len(added), "exited": len(exited),
            "screen_s": screen_time,
        })

        # Bar-level trading within month
        month_bars = close_panel.index[(close_panel.index >= ms) & (close_panel.index <= me)]

        for ts in month_bars:
            for pair in active_pairs:
                y_name, x_name = pair["y"], pair["x"]
                if ts not in close_panel.index: continue

                y_p = close_panel.at[ts, y_name] if y_name in close_panel.columns else np.nan
                x_p = close_panel.at[ts, x_name] if x_name in close_panel.columns else np.nan
                if np.isnan(y_p) or np.isnan(x_p) or y_p <= 0 or x_p <= 0: continue

                z = (math.log(y_p) - pair["beta"] * math.log(x_p) - pair["intercept"] - pair["mu"]) / pair["sigma"]

                y_r = ret_panel.at[ts, y_name] if ts in ret_panel.index and y_name in ret_panel.columns else 0
                x_r = ret_panel.at[ts, x_name] if ts in ret_panel.index and x_name in ret_panel.columns else 0
                if np.isnan(y_r): y_r = 0
                if np.isnan(x_r): x_r = 0

                notional = CAPITAL * SIZE_PCT
                bar_pnl = 0.0
                if pair["pos"] != 0:
                    bar_pnl = pair["pos"] * (y_r - pair["beta"] * x_r) * notional

                cost = COST_BPS / 10000 * (1 + abs(pair["beta"])) * notional

                # Breakdown detection
                if abs(z) > BREAKDOWN_Z:
                    pair["bd_count"] = pair.get("bd_count", 0) + 1
                else:
                    pair["bd_count"] = 0

                # Exit
                if pair["pos"] != 0:
                    bh = (ts - pair["entry_ts"]).total_seconds() / (15 * 60) if pair.get("entry_ts") else MAX_HOLD + 1
                    should_exit = False
                    if pair["pos"] == 1 and z >= 0: should_exit = True
                    elif pair["pos"] == -1 and z <= 0: should_exit = True
                    elif bh >= MAX_HOLD: should_exit = True
                    elif pair.get("bd_count", 0) >= BREAKDOWN_BARS:
                        should_exit = True; n_breakdowns += 1
                    if should_exit:
                        bar_pnl -= cost; pair["pos"] = 0

                # Entry
                if pair["pos"] == 0 and pair.get("bd_count", 0) < BREAKDOWN_BARS:
                    if z <= -ENTRY_Z:
                        pair["pos"] = 1; pair["entry_ts"] = ts; bar_pnl -= cost; n_trades += 1
                    elif z >= ENTRY_Z:
                        pair["pos"] = -1; pair["entry_ts"] = ts; bar_pnl -= cost; n_trades += 1

                day = ts.normalize()
                daily_pnl[day] = daily_pnl.get(day, 0) + bar_pnl

    # ---- Results ----
    print(f"\n{'='*110}")
    print("MONTHLY SCREENING LOG")
    print(f"{'='*110}")
    print(f"\n{'Month':<10s} {'Univ':>5s} {'Found':>6s} {'Active':>7s} {'Added':>6s} {'Exited':>7s} {'Scr(s)':>7s}")
    print("-" * 55)
    for m in monthly_log:
        print(f"{m['month']:<10s} {m['univ']:>5d} {m['found']:>6d} {m['active']:>7d} "
              f"{m.get('added',0):>6d} {m.get('exited',0):>7d} {m.get('screen_s',0):>6.1f}s")

    dpnl = pd.Series(daily_pnl)
    if len(dpnl) < 10:
        print("\nInsufficient data"); return

    cum = dpnl.cumsum(); dr = dpnl / CAPITAL
    sharpe = dr.mean() / dr.std() * np.sqrt(365.25) if dr.std() > 0 else 0
    peak = (cum + CAPITAL).cummax(); dd = abs(((cum + CAPITAL - peak) / peak).min())
    wd = abs(dr.min()); total_pnl = cum.iloc[-1]; pnl_cal = total_pnl / CAL_DAYS
    monthly_pnl = dpnl.resample("ME").sum()

    print(f"\n{'='*110}")
    print("DYNAMIC STRATEGY RESULTS")
    print(f"{'='*110}")
    print(f"Total PnL:     ${total_pnl:.0f}")
    print(f"$/cal_day:     ${pnl_cal:.2f}")
    print(f"Sharpe:        {sharpe:.2f}")
    print(f"Max DD:        {dd*100:.1f}%")
    print(f"Worst day:     {wd*100:.2f}%")
    print(f"Trades:        {n_trades}")
    print(f"Breakdowns:    {n_breakdowns}")
    print(f"+Months:       {(monthly_pnl > 0).sum()}/{len(monthly_pnl)} ({(monthly_pnl > 0).mean()*100:.0f}%)")
    ok = dd < 0.10 and wd < 0.05
    print(f"Risk check:    {'PASS' if ok else 'FAIL'}")

    print(f"\n{'Month':<10s} {'PnL':>8s}  {'Pairs':>6s}")
    print("-" * 28)
    for i, dt in enumerate(monthly_pnl.index):
        v = monthly_pnl.loc[dt]
        n_active = monthly_log[i]["active"] if i < len(monthly_log) else 0
        flag = " <<< NEG" if v < -50 else ""
        print(f"{dt.strftime('%Y-%m'):<10s} ${v:>7.0f}  {n_active:>6d}{flag}")

    print(f"\nCOMPARISON vs STATIC 5-pair:")
    print(f"  Static:  Sharpe 1.58, $3.25/cal, DD 4.0%, WD 2.23%, worst month -$186")
    print(f"  Dynamic: Sharpe {sharpe:.2f}, ${pnl_cal:.2f}/cal, DD {dd*100:.1f}%, WD {wd*100:.2f}%, worst month ${monthly_pnl.min():.0f}")

    print(f"\nRuntime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
