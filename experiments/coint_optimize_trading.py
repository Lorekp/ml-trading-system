"""Cointegration Trading Parameter Optimization

Best screening config is FIXED (heur, corr>0.40, lb6, z=2.0, 8 pairs).
Sweeps the TRADING parameters that have never been optimized:
- Position sizing (1-5% per leg)
- Exit z-score (-0.3 to 0.5)
- Min hold (12h to 2d)
- Max hold (15d to 45d)
- Add/drop months (1-3)
- Z-proportional sizing
- Per-pair quality sizing
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

CAPITAL = 10000.0; COST_BPS = 5.0; EMBARGO = 96 * 7; MIN_AGE_DAYS = 180
ENTRY_Z = 2.0; MIN_CORR = 0.40
TEST_START = "2025-03"; TEST_END = "2026-02"; CAL_DAYS = 366


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


def precompute_screens(close_panel, ohlcv_data, months):
    """Pre-compute EG + OLS for all pairs per month (6mo lookback only)."""
    screens = {}
    for month in months:
        ms = pd.Timestamp(f"{month}-01", tz="UTC")
        universe = select_universe(ohlcv_data, ms)
        if len(universe) < 6: screens[str(month)] = {}; continue

        screen_end = ms - pd.Timedelta(minutes=15 * EMBARGO)
        screen_start = screen_end - pd.DateOffset(months=6)
        avail = [a for a in universe if a in close_panel.columns]
        pairs = {}

        for a_i, a_j in combinations(avail, 2):
            mask = (close_panel.index >= screen_start) & (close_panel.index < screen_end)
            y_s = close_panel.loc[mask, a_i].dropna()
            x_s = close_panel.loc[mask, a_j].dropna()
            common = y_s.index.intersection(x_s.index)
            if len(common) < 3000: continue

            y_arr = y_s.reindex(common).values.astype(float)
            x_arr = x_s.reindex(common).values.astype(float)

            ret_y = np.diff(y_arr) / y_arr[:-1]; ret_x = np.diff(x_arr) / x_arr[:-1]
            valid = np.isfinite(ret_y) & np.isfinite(ret_x)
            if valid.sum() < 2000: continue
            corr = float(np.corrcoef(ret_y[valid], ret_x[valid])[0, 1])
            if not np.isfinite(corr) or corr < MIN_CORR: continue

            try: _, p, _ = coint(y_arr, x_arr, maxlag=10)
            except: p = 1.0

            log_y = np.log(y_arr); log_x = np.log(x_arr)
            beta, intercept = ols_fit(log_y, log_x)
            if not np.isfinite(beta) or beta <= 0: continue
            resid = log_y - beta * log_x - intercept
            hl = half_life_bars(resid) / 96

            passes = p < 0.05 and 1 < hl < 20
            pairs[(a_i, a_j)] = {
                "p": float(p), "beta": beta, "intercept": intercept, "hl": hl,
                "mu": float(np.mean(resid)), "sigma": float(np.std(resid)),
                "passes": passes,
            }

        screens[str(month)] = pairs
    return screens


def simulate(screens, close_panel, ret_panel, months, cfg):
    """Simulate one config through walk-forward."""
    base_size = cfg.get("base_size", 0.02)
    exit_z = cfg.get("exit_z", 0.0)
    min_hold = cfg.get("min_hold", 96)
    max_hold = cfg.get("max_hold", 96 * 30)
    add_months = cfg.get("add_months", 2)
    drop_months = cfg.get("drop_months", 2)
    breakdown_z = cfg.get("breakdown_z", 5.0)
    z_proportional = cfg.get("z_proportional", False)
    quality_sizing = cfg.get("quality_sizing", False)
    max_pairs = 8

    book = {}; candidates = {}
    daily_pnl = {}; total_trades = 0

    for month in months:
        ms = pd.Timestamp(f"{month}-01", tz="UTC")
        me = ms + pd.offsets.MonthEnd(0)
        pairs_data = screens.get(str(month), {})

        passing = {k for k, v in pairs_data.items() if v["passes"]}

        # Book management
        for key in list(book.keys()):
            if key in passing or (key[1], key[0]) in passing:
                book[key]["fail_count"] = 0
                book[key]["months_in"] = book[key].get("months_in", 0) + 1
            else:
                book[key]["fail_count"] = book[key].get("fail_count", 0) + 1
                if book[key]["fail_count"] >= drop_months:
                    # Close any open position before dropping (pay exit cost)
                    if book[key].get("_pos", 0) != 0:
                        beta_drop = book[key].get("beta", 1)
                        cost_drop = COST_BPS / 10000 * (1 + abs(beta_drop)) * CAPITAL * base_size
                        day_drop = ms.normalize()
                        daily_pnl[day_drop] = daily_pnl.get(day_drop, 0) - cost_drop
                    book.pop(key); continue

            pd_info = pairs_data.get(key) or pairs_data.get((key[1], key[0]))
            if pd_info:
                book[key]["beta"] = pd_info["beta"]
                book[key]["intercept"] = pd_info["intercept"]
                book[key]["mu"] = pd_info["mu"]
                book[key]["sigma"] = pd_info["sigma"]

        for key in passing:
            if key in book or (key[1], key[0]) in book: continue
            candidates[key] = candidates.get(key, 0) + 1
            if candidates[key] >= add_months and len(book) < max_pairs:
                pd_info = pairs_data.get(key)
                if pd_info and np.isfinite(pd_info["beta"]) and pd_info["beta"] > 0:
                    book[key] = {**pd_info, "fail_count": 0, "months_in": 0}
                    candidates.pop(key, None)

        for key in list(candidates.keys()):
            if key not in passing and (key[1], key[0]) not in passing:
                candidates.pop(key, None)

        # Simulate month
        month_bars = close_panel.index[(close_panel.index >= ms) & (close_panel.index <= me)]

        for key, pair in book.items():
            y_name, x_name = key
            beta = pair["beta"]; mu = pair["mu"]; sigma = pair["sigma"]
            if not np.isfinite(beta) or sigma <= 0: continue
            if y_name not in close_panel.columns or x_name not in close_panel.columns: continue

            # Sizing
            sz = base_size
            if quality_sizing:
                mi = pair.get("months_in", 0)
                if mi >= 6: sz = base_size * 1.5
                elif mi >= 3: sz = base_size * 1.0
                else: sz = base_size * 0.5

            # CARRY position across months (stored in book, not local)
            pos = pair.get("_pos", 0)
            entry_ts = pair.get("_entry_ts", None)
            entry_z_actual = pair.get("_entry_z", 0)

            for bar_idx, ts in enumerate(month_bars):
                y_p = close_panel.at[ts, y_name]; x_p = close_panel.at[ts, x_name]
                if np.isnan(y_p) or np.isnan(x_p) or y_p <= 0 or x_p <= 0: continue

                spread = math.log(y_p) - beta * math.log(x_p) - pair.get("intercept", 0)
                z = (spread - mu) / sigma if sigma > 0 else 0

                # Z-proportional sizing
                if z_proportional and pos == 0 and abs(z) >= ENTRY_Z:
                    z_mult = min(abs(z) / ENTRY_Z, 2.0)
                    notional = CAPITAL * sz * z_mult
                elif z_proportional and pos != 0:
                    notional = CAPITAL * sz * min(abs(entry_z_actual) / ENTRY_Z, 2.0)
                else:
                    notional = CAPITAL * sz

                y_r = ret_panel.at[ts, y_name] if ts in ret_panel.index and y_name in ret_panel.columns else 0
                x_r = ret_panel.at[ts, x_name] if ts in ret_panel.index and x_name in ret_panel.columns else 0
                if np.isnan(y_r): y_r = 0
                if np.isnan(x_r): x_r = 0

                bar_pnl = 0.0
                cost = COST_BPS / 10000 * (1 + abs(beta)) * notional

                if pos != 0:
                    bar_pnl = pos * (y_r - beta * x_r) * notional
                    # Bars held using absolute timestamps
                    if entry_ts is not None and isinstance(entry_ts, pd.Timestamp):
                        bh = int((ts - entry_ts).total_seconds() / (15 * 60))
                    else:
                        bh = min_hold + 1  # allow exit if no entry_ts
                    should_exit = False
                    if bh >= min_hold:
                        if pos == 1 and z >= exit_z: should_exit = True
                        elif pos == -1 and z <= -exit_z: should_exit = True
                    if bh >= max_hold: should_exit = True
                    if abs(z) > breakdown_z and bh > 48: should_exit = True
                    if should_exit: bar_pnl -= cost; pos = 0; entry_ts = None

                if pos == 0:
                    if z <= -ENTRY_Z:
                        pos = 1; entry_ts = ts; entry_z_actual = z
                        bar_pnl -= cost; total_trades += 1
                    elif z >= ENTRY_Z:
                        pos = -1; entry_ts = ts; entry_z_actual = z
                        bar_pnl -= cost; total_trades += 1

                day = ts.normalize()
                daily_pnl[day] = daily_pnl.get(day, 0) + bar_pnl

            # Save position state back to book for next month
            pair["_pos"] = pos
            pair["_entry_ts"] = entry_ts
            pair["_entry_z"] = entry_z_actual

    dpnl = pd.Series(daily_pnl)
    if len(dpnl) < 10: return None
    cum = dpnl.cumsum(); dr = dpnl / CAPITAL
    sharpe = dr.mean() / dr.std() * np.sqrt(365.25) if dr.std() > 0 else 0
    peak = (cum + CAPITAL).cummax(); dd = abs(((cum + CAPITAL - peak) / peak).min())
    wd = abs(dr.min()) if len(dr) > 0 else 0
    pnl_cal = cum.iloc[-1] / CAL_DAYS
    monthly_pnl = dpnl.resample("ME").sum()
    pos_months = (monthly_pnl > 0).sum()

    return {"sharpe": sharpe, "pnl_cal": pnl_cal, "max_dd": dd, "worst_day": wd,
            "n_trades": total_trades,
            "pos_months": f"{pos_months}/{len(monthly_pnl)} ({pos_months/max(len(monthly_pnl),1)*100:.0f}%)"}


def main():
    t0 = time.time()
    print("=" * 110)
    print("COINTEGRATION TRADING PARAMETER OPTIMIZATION")
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

    months = pd.period_range(TEST_START, TEST_END, freq="M")
    print(f"Pre-computing screens for {len(months)} months...", end=" ", flush=True)
    screens = precompute_screens(close_panel, ohlcv_data, months)
    print(f"done ({time.time()-t0:.0f}s)", flush=True)

    # ---- Sweep 1: Core grid (size × exit × min_hold) ----
    configs = []
    for size in [0.01, 0.02, 0.03, 0.05]:
        for exit_z in [-0.3, 0.0, 0.3, 0.5]:
            for min_hold in [48, 96, 192]:
                label = f"sz={size*100:.0f}% ex={exit_z:+.1f} mh={min_hold//96:.0f}d"
                configs.append({"label": label, "base_size": size, "exit_z": exit_z, "min_hold": min_hold})

    # ---- Sweep 2: Max hold (on best sizing) ----
    for max_hold_days in [15, 30, 45]:
        label = f"sz=3% ex=0.0 maxh={max_hold_days}d"
        configs.append({"label": label, "base_size": 0.03, "max_hold": 96 * max_hold_days})

    # ---- Sweep 3: Book management ----
    for add_m in [1, 2, 3]:
        for drop_m in [1, 2, 3]:
            if add_m == 2 and drop_m == 2: continue  # default
            label = f"sz=3% add={add_m} drop={drop_m}"
            configs.append({"label": label, "base_size": 0.03, "add_months": add_m, "drop_months": drop_m})

    # ---- Sweep 4: Advanced sizing ----
    configs.append({"label": "sz=3% z-proportional", "base_size": 0.03, "z_proportional": True})
    configs.append({"label": "sz=3% quality-sized", "base_size": 0.03, "quality_sizing": True})
    configs.append({"label": "sz=3% z-prop+quality", "base_size": 0.03, "z_proportional": True, "quality_sizing": True})
    configs.append({"label": "sz=5% z-prop", "base_size": 0.05, "z_proportional": True})
    configs.append({"label": "sz=5% quality", "base_size": 0.05, "quality_sizing": True})

    print(f"\n{len(configs)} configurations to test")

    results = []
    for i, cfg in enumerate(configs):
        if i % 20 == 0: print(f"  [{i}/{len(configs)}] ({time.time()-t0:.0f}s)", flush=True)
        r = simulate(screens, close_panel, ret_panel, months, cfg)
        if r: r["config"] = cfg["label"]; results.append(r)

    results.sort(key=lambda x: -x["sharpe"])

    print(f"\n{'='*110}")
    print(f"TOP 25 BY SHARPE")
    print(f"{'='*110}")

    print(f"\n{'#':>3s} {'Config':<35s} {'$/cal':>7s} {'Sharpe':>7s} {'DD%':>6s} "
          f"{'WD%':>6s} {'Trades':>6s} {'+Mo':>12s}")
    print("-" * 95)

    for i, r in enumerate(results[:25]):
        ok = r["max_dd"] < 0.07 and r["worst_day"] < 0.05
        print(f"{i+1:>3d} {r['config']:<35s} ${r['pnl_cal']:>6.2f} {r['sharpe']:>7.2f} "
              f"{r['max_dd']*100:>5.1f}% {r['worst_day']*100:>5.2f}% "
              f"{r['n_trades']:>6d} {r['pos_months']:>12s} {'OK' if ok else ''}")

    # Top by $/cal
    by_pnl = sorted(results, key=lambda x: -x["pnl_cal"])
    print(f"\n{'='*110}")
    print(f"TOP 10 BY $/CAL")
    print(f"{'='*110}")
    for i, r in enumerate(by_pnl[:10]):
        print(f"{i+1:>3d} {r['config']:<35s} ${r['pnl_cal']:>6.2f} {r['sharpe']:>7.2f} "
              f"{r['max_dd']*100:>5.1f}% {r['worst_day']*100:>5.2f}% {r['n_trades']:>6d}")

    # Pareto frontier
    print(f"\n{'='*110}")
    print(f"PARETO FRONTIER")
    print(f"{'='*110}")
    for lo, hi in [(0, 0.03), (0.03, 0.05), (0.05, 0.07), (0.07, 0.10)]:
        bucket = [r for r in results if lo <= r["max_dd"] < hi]
        if bucket:
            best = max(bucket, key=lambda x: x["sharpe"])
            print(f"  DD {lo*100:.0f}-{hi*100:.0f}%: {best['config']:<35s} "
                  f"SR={best['sharpe']:.2f} ${best['pnl_cal']:.2f}/cal DD={best['max_dd']*100:.1f}%")

    print(f"\nBaseline: heur lb6 z2.0 sz=2% = Sharpe 0.89, $0.77/cal, DD 3.1%")
    print(f"Runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
