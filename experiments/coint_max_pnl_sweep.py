"""Max PnL sweep: size x leverage x exit_z with DD<10% and WD<5% constraint."""

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
CAPITAL = 10000.0; COST_BPS = 5.0; EMBARGO = 96*7; MIN_AGE_DAYS = 180
MIN_CORR = 0.40; MIN_HOLD = 96; MAX_HOLD = 96*30; BREAKDOWN_Z = 5.0
DROP_MONTHS = 3; ADD_MONTHS = 2; MAX_PAIRS = 8
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


def select_universe(ohlcv_data, ms):
    le = ms - pd.Timedelta(days=1); c = {}
    for n, o in ohlcv_data.items():
        if (ms - o.index.min()).days < MIN_AGE_DAYS: continue
        a = o.loc[:le]
        if len(a) < 96*45: continue
        dv = (a["close"]*a["volume"]).resample("1D").sum()
        r = dv.iloc[-30:] if len(dv) >= 30 else dv; v = r.median()
        if np.isnan(v) or v <= 0: continue
        if len(dv) >= 90 and dv.iloc[-90:].median() > 0 and v / dv.iloc[-90:].median() > 5: continue
        c[n] = float(v)
    return [a for a, _ in sorted(c.items(), key=lambda x: -x[1])[:30]]


def precompute(close_panel, ohlcv_data, months):
    screens = {}
    for month in months:
        ms = pd.Timestamp(f"{month}-01", tz="UTC")
        universe = select_universe(ohlcv_data, ms)
        if len(universe) < 6: screens[str(month)] = {}; continue
        se = ms - pd.Timedelta(minutes=15*EMBARGO); ss = se - pd.DateOffset(months=6)
        avail = [a for a in universe if a in close_panel.columns]; pairs = {}
        for ai, aj in combinations(avail, 2):
            m = (close_panel.index >= ss) & (close_panel.index < se)
            ys = close_panel.loc[m, ai].dropna(); xs = close_panel.loc[m, aj].dropna()
            co = ys.index.intersection(xs.index)
            if len(co) < 3000: continue
            ya = ys.reindex(co).values.astype(float); xa = xs.reindex(co).values.astype(float)
            ry = np.diff(ya)/ya[:-1]; rx = np.diff(xa)/xa[:-1]
            v = np.isfinite(ry) & np.isfinite(rx)
            if v.sum() < 2000: continue
            corr = float(np.corrcoef(ry[v], rx[v])[0, 1])
            if not np.isfinite(corr) or corr < MIN_CORR: continue
            try:
                _, p, _ = coint(ya, xa, maxlag=10)
            except:
                p = 1.0
            ly = np.log(ya); lx = np.log(xa)
            beta, intercept = ols_fit(ly, lx)
            if not np.isfinite(beta) or beta <= 0: continue
            resid = ly - beta*lx - intercept; hl = half_life_bars(resid)/96
            passes = p < 0.05 and 1 < hl < 20
            pairs[(ai, aj)] = {
                "p": float(p), "beta": beta, "intercept": intercept, "hl": hl,
                "mu": float(np.mean(resid)), "sigma": float(np.std(resid)), "passes": passes,
            }
        screens[str(month)] = pairs
    return screens


def simulate(screens, close_panel, ret_panel, months, size_pct, entry_z, exit_z, leverage=1.0):
    book = {}; candidates = {}; daily_pnl = {}; total_trades = 0
    for month in months:
        ms = pd.Timestamp(f"{month}-01", tz="UTC"); me = ms + pd.offsets.MonthEnd(0)
        pd2 = screens.get(str(month), {}); passing = {k for k, v in pd2.items() if v["passes"]}
        for key in list(book.keys()):
            if key in passing or (key[1], key[0]) in passing:
                book[key]["fc"] = 0; book[key]["mi"] = book[key].get("mi", 0) + 1
            else:
                book[key]["fc"] = book[key].get("fc", 0) + 1
                if book[key]["fc"] >= DROP_MONTHS:
                    if book[key].get("_pos", 0) != 0:
                        ct = COST_BPS/10000*(1+abs(book[key].get("beta", 1)))*CAPITAL*size_pct*leverage
                        d = ms.normalize(); daily_pnl[d] = daily_pnl.get(d, 0) - ct
                    book.pop(key); continue
            pi = pd2.get(key) or pd2.get((key[1], key[0]))
            if pi:
                book[key].update({"beta": pi["beta"], "intercept": pi["intercept"],
                                  "mu": pi["mu"], "sigma": pi["sigma"]})
        for key in passing:
            if key in book or (key[1], key[0]) in book: continue
            candidates[key] = candidates.get(key, 0) + 1
            if candidates[key] >= ADD_MONTHS and len(book) < MAX_PAIRS:
                pi = pd2.get(key)
                if pi and np.isfinite(pi["beta"]) and pi["beta"] > 0:
                    book[key] = {**pi, "fc": 0, "mi": 0, "_pos": 0, "_ets": None}
                    candidates.pop(key, None)
        for key in list(candidates.keys()):
            if key not in passing and (key[1], key[0]) not in passing:
                candidates.pop(key, None)

        month_bars = close_panel.index[(close_panel.index >= ms) & (close_panel.index <= me)]
        for key, pair in book.items():
            yn, xn = key; beta = pair["beta"]; mu = pair["mu"]; sigma = pair["sigma"]
            if not np.isfinite(beta) or sigma <= 0: continue
            if yn not in close_panel.columns or xn not in close_panel.columns: continue
            pos = pair.get("_pos", 0); ets = pair.get("_ets", None)
            notional = CAPITAL * size_pct * leverage
            for bar_idx, ts in enumerate(month_bars):
                yp = close_panel.at[ts, yn]; xp = close_panel.at[ts, xn]
                if np.isnan(yp) or np.isnan(xp) or yp <= 0 or xp <= 0: continue
                spread = math.log(yp) - beta*math.log(xp) - pair.get("intercept", 0)
                z = (spread - mu) / sigma if sigma > 0 else 0
                yr = ret_panel.at[ts, yn] if ts in ret_panel.index and yn in ret_panel.columns else 0
                xr = ret_panel.at[ts, xn] if ts in ret_panel.index and xn in ret_panel.columns else 0
                if np.isnan(yr): yr = 0
                if np.isnan(xr): xr = 0
                bar_pnl = 0.0; cost = COST_BPS/10000*(1+abs(beta))*notional
                if pos != 0:
                    bar_pnl = pos*(yr - beta*xr)*notional
                    bh = int((ts - ets).total_seconds()/(15*60)) if isinstance(ets, pd.Timestamp) else MIN_HOLD+1
                    ex = False
                    if bh >= MIN_HOLD:
                        if pos == 1 and z >= exit_z: ex = True
                        elif pos == -1 and z <= -exit_z: ex = True
                    if bh >= MAX_HOLD: ex = True
                    if abs(z) > BREAKDOWN_Z and bh > 48: ex = True
                    if ex: bar_pnl -= cost; pos = 0; ets = None
                if pos == 0:
                    if z <= -entry_z: pos = 1; ets = ts; bar_pnl -= cost; total_trades += 1
                    elif z >= entry_z: pos = -1; ets = ts; bar_pnl -= cost; total_trades += 1
                day = ts.normalize(); daily_pnl[day] = daily_pnl.get(day, 0) + bar_pnl
            pair["_pos"] = pos; pair["_ets"] = ets

    dpnl = pd.Series(daily_pnl)
    if len(dpnl) < 10: return None
    cum = dpnl.cumsum(); dr = dpnl / CAPITAL
    sharpe = dr.mean()/dr.std()*np.sqrt(365.25) if dr.std() > 0 else 0
    peak = (cum + CAPITAL).cummax(); dd = abs(((cum + CAPITAL - peak) / peak).min())
    wd = abs(dr.min()) if len(dr) > 0 else 0
    return {"sharpe": sharpe, "pnl_cal": cum.iloc[-1]/CAL_DAYS, "max_dd": dd,
            "worst_day": wd, "n_trades": total_trades}


def main():
    t0 = time.time()
    print("=" * 100)
    print("MAX PNL SWEEP: size x leverage x exit_z  |  DD<10% and WD<5%")
    print("=" * 100)

    print("\nLoading...", end=" ", flush=True)
    ohlcv_data = {}
    for f in sorted(CACHE_DIR.glob("*.parquet")):
        n = f.stem.replace("USDT", "")
        try:
            df = pd.read_parquet(f, columns=["close", "volume"])
            if len(df) >= 2000: ohlcv_data[n] = df
        except: pass
    close_panel = pd.DataFrame({n: d["close"] for n, d in ohlcv_data.items()})
    ret_panel = close_panel.pct_change()
    print(f"{len(ohlcv_data)} assets")

    months = pd.period_range(TEST_START, TEST_END, freq="M")
    print("Screens...", end=" ", flush=True)
    screens = precompute(close_panel, ohlcv_data, months)
    print(f"done ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n{'Config':<50s} {'$/cal':>7s} {'SR':>6s} {'DD%':>6s} {'WD%':>6s} {'Tr':>5s}")
    print("-" * 85)

    feasible = []
    for size in [0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]:
        for lev in [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
            for ez in [0.0, 0.3, 0.5]:
                eff = size * lev
                if eff > 0.30: continue  # max 30% per leg
                r = simulate(screens, close_panel, ret_panel, months, size, 2.0, ez, lev)
                if r is None: continue
                ok = r["max_dd"] < 0.10 and r["worst_day"] < 0.05
                label = f"sz={size*100:.0f}% lev={lev:.1f}x ex={ez} eff={eff*100:.0f}%/leg"
                if ok:
                    feasible.append({**r, "config": label})
                    print(f"{label:<50s} ${r['pnl_cal']:>6.2f} {r['sharpe']:>6.2f} "
                          f"{r['max_dd']*100:>5.1f}% {r['worst_day']*100:>5.2f}% {r['n_trades']:>5d}")

    if feasible:
        top10 = sorted(feasible, key=lambda x: -x["pnl_cal"])[:10]
        print(f"\n{'='*100}")
        print("TOP 10 FEASIBLE (max PnL, DD<10%, WD<5%):")
        print(f"{'='*100}")
        for i, r in enumerate(top10):
            print(f"  {i+1:>2d}. {r['config']:<50s} ${r['pnl_cal']:>6.2f}/cal  "
                  f"SR={r['sharpe']:.2f}  DD={r['max_dd']*100:.1f}%  WD={r['worst_day']*100:.2f}%  "
                  f"Tr={r['n_trades']}")

        # Also show: best Sharpe among feasible
        by_sr = sorted(feasible, key=lambda x: -x["sharpe"])[:5]
        print(f"\nTOP 5 BY SHARPE (within DD<10% WD<5%):")
        for i, r in enumerate(by_sr):
            print(f"  {i+1:>2d}. {r['config']:<50s} SR={r['sharpe']:.2f}  "
                  f"${r['pnl_cal']:.2f}/cal  DD={r['max_dd']*100:.1f}%")

    print(f"\nTotal feasible: {len(feasible)} configs")
    print(f"Runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
