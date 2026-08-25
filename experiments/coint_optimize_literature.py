"""Literature-Based Improvements on Winning Cointegration Config

Baseline: Sharpe 1.44, $2.12/cal, DD 3.0% (heur, corr>0.40, lb6, z=2.0, exit=0.5, 3%, drop=3)

Tests 6 improvements in isolation, then stacks winners:
L1: Kalman hedge ratio (replaces monthly OLS)
L2: Stop-loss on spread PnL (2/3/5/8%)
L3: Asymmetric entry/exit
L4: OU-optimal per-pair thresholds
L5: Vol-inverse sizing
L6: Cross-pair correlation limit
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
BASE_ENTRY_Z = 2.0; BASE_EXIT_Z = 0.5; BASE_SIZE = 0.03
MIN_HOLD = 96; MAX_HOLD = 96 * 30; MIN_CORR = 0.40
DROP_MONTHS = 3; ADD_MONTHS = 2; MAX_PAIRS = 8; BREAKDOWN_Z = 5.0
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
        r = dv.iloc[-30:] if len(dv)>=30 else dv; v = r.median()
        if np.isnan(v) or v<=0: continue
        if len(dv)>=90 and dv.iloc[-90:].median()>0 and v/dv.iloc[-90:].median()>5: continue
        c[n] = float(v)
    return [a for a,_ in sorted(c.items(), key=lambda x:-x[1])[:30]]


def precompute_screens(close_panel, ohlcv_data, months):
    screens = {}
    for month in months:
        ms = pd.Timestamp(f"{month}-01", tz="UTC")
        universe = select_universe(ohlcv_data, ms)
        if len(universe) < 6: screens[str(month)] = {}; continue
        se = ms - pd.Timedelta(minutes=15*EMBARGO)
        ss = se - pd.DateOffset(months=6)
        avail = [a for a in universe if a in close_panel.columns]
        pairs = {}
        for ai, aj in combinations(avail, 2):
            m = (close_panel.index >= ss) & (close_panel.index < se)
            ys = close_panel.loc[m, ai].dropna(); xs = close_panel.loc[m, aj].dropna()
            c = ys.index.intersection(xs.index)
            if len(c) < 3000: continue
            ya = ys.reindex(c).values.astype(float); xa = xs.reindex(c).values.astype(float)
            ry = np.diff(ya)/ya[:-1]; rx = np.diff(xa)/xa[:-1]
            v = np.isfinite(ry) & np.isfinite(rx)
            if v.sum()<2000: continue
            corr = float(np.corrcoef(ry[v],rx[v])[0,1])
            if not np.isfinite(corr) or corr<MIN_CORR: continue
            try: _, p, _ = coint(ya, xa, maxlag=10)
            except: p=1.0
            ly=np.log(ya); lx=np.log(xa)
            beta, intercept = ols_fit(ly, lx)
            if not np.isfinite(beta) or beta<=0: continue
            resid = ly - beta*lx - intercept
            hl = half_life_bars(resid)/96
            sigma = float(np.std(resid))
            passes = p<0.05 and 1<hl<20
            pairs[(ai,aj)] = {"p":float(p),"beta":beta,"intercept":intercept,"hl":hl,
                              "mu":float(np.mean(resid)),"sigma":sigma,"passes":passes}
        screens[str(month)] = pairs
    return screens


def simulate(screens, close_panel, ret_panel, months, cfg):
    """Simulate with all literature improvements as flags."""
    use_kalman = cfg.get("kalman", False)
    stop_loss_pct = cfg.get("stop_loss", None)  # e.g. 0.03 = 3%
    asym_entry = cfg.get("asym_entry", None)  # (long_z, short_z) or None
    asym_exit = cfg.get("asym_exit", None)  # (long_exit, short_exit) or None
    ou_adaptive = cfg.get("ou_adaptive", False)
    vol_inverse = cfg.get("vol_inverse", False)
    corr_limit = cfg.get("corr_limit", None)  # max correlation between spreads
    entry_z = cfg.get("entry_z", BASE_ENTRY_Z)
    exit_z = cfg.get("exit_z", BASE_EXIT_Z)
    base_size = cfg.get("base_size", BASE_SIZE)

    book = {}; candidates = {}
    daily_pnl = {}; total_trades = 0

    # Collect all pair sigmas for vol-inverse median
    all_sigmas = []
    for m_pairs in screens.values():
        for pi in m_pairs.values():
            if pi.get("passes") and pi["sigma"] > 0:
                all_sigmas.append(pi["sigma"])
    median_sigma = float(np.median(all_sigmas)) if all_sigmas else 0.1

    # Collect all HLs for OU-adaptive median
    all_hls = []
    for m_pairs in screens.values():
        for pi in m_pairs.values():
            if pi.get("passes") and 1 < pi["hl"] < 20:
                all_hls.append(pi["hl"])
    median_hl = float(np.median(all_hls)) if all_hls else 5.0

    for month in months:
        ms = pd.Timestamp(f"{month}-01", tz="UTC")
        me = ms + pd.offsets.MonthEnd(0)
        pairs_data = screens.get(str(month), {})
        passing = {k for k,v in pairs_data.items() if v["passes"]}

        # Book management (same rolling health check)
        for key in list(book.keys()):
            if key in passing or (key[1],key[0]) in passing:
                book[key]["fc"] = 0; book[key]["mi"] = book[key].get("mi",0)+1
            else:
                book[key]["fc"] = book[key].get("fc",0)+1
                if book[key]["fc"] >= DROP_MONTHS:
                    if book[key].get("_pos",0) != 0:
                        c = COST_BPS/10000*(1+abs(book[key].get("beta",1)))*CAPITAL*base_size
                        d = ms.normalize(); daily_pnl[d] = daily_pnl.get(d,0)-c
                    book.pop(key); continue
            pi = pairs_data.get(key) or pairs_data.get((key[1],key[0]))
            if pi:
                book[key].update({"beta":pi["beta"],"intercept":pi["intercept"],
                                  "mu":pi["mu"],"sigma":pi["sigma"],"hl":pi["hl"]})

        # Cross-pair correlation limit (L6)
        if corr_limit and len(book) > 1:
            # Simplified: no action on existing book, just limit new adds
            pass

        for key in passing:
            if key in book or (key[1],key[0]) in book: continue
            candidates[key] = candidates.get(key,0)+1
            if candidates[key]>=ADD_MONTHS and len(book)<MAX_PAIRS:
                pi = pairs_data.get(key)
                if pi and np.isfinite(pi["beta"]) and pi["beta"]>0:
                    # Cross-pair correlation check for new adds
                    add_ok = True
                    if corr_limit and len(book) > 0:
                        # Check if new pair's spread is correlated with existing
                        # Simplified: check if assets overlap
                        new_assets = {key[0], key[1]}
                        for bk in book:
                            overlap = new_assets & {bk[0], bk[1]}
                            if len(overlap) > 0:
                                # Shared asset — skip if already have 2 pairs with this asset
                                shared_count = sum(1 for bk2 in book if overlap & {bk2[0], bk2[1]})
                                if shared_count >= 2: add_ok = False; break
                    if add_ok:
                        book[key] = {**pi,"fc":0,"mi":0,"_pos":0,"_ets":None,"_ez":0,
                                     "_kalman_theta":np.array([pi["beta"],pi["intercept"]]),
                                     "_kalman_P":np.eye(2),"_trade_pnl":0.0}
                        candidates.pop(key,None)

        for key in list(candidates.keys()):
            if key not in passing and (key[1],key[0]) not in passing:
                candidates.pop(key,None)

        # Bar-level simulation
        month_bars = close_panel.index[(close_panel.index>=ms)&(close_panel.index<=me)]
        for key, pair in book.items():
            yn,xn = key
            beta = pair["beta"]; mu = pair["mu"]; sigma = pair["sigma"]
            if not np.isfinite(beta) or sigma<=0: continue
            if yn not in close_panel.columns or xn not in close_panel.columns: continue

            pos = pair.get("_pos",0); ets = pair.get("_ets",None)
            ez_actual = pair.get("_ez",0); trade_pnl = pair.get("_trade_pnl",0.0)

            # Per-pair thresholds (L4: OU-adaptive)
            p_entry_z = entry_z
            p_exit_z = exit_z
            if ou_adaptive and pair.get("hl",5) > 0:
                hl_ratio = math.sqrt(pair["hl"] / median_hl)
                p_entry_z = entry_z * max(0.5, min(hl_ratio, 2.0))
                p_exit_z = exit_z * max(0.3, min(hl_ratio, 1.5))

            # Asymmetric (L3)
            long_entry_z = p_entry_z; short_entry_z = p_entry_z
            long_exit_z = p_exit_z; short_exit_z = p_exit_z
            if asym_entry: long_entry_z, short_entry_z = asym_entry
            if asym_exit: long_exit_z, short_exit_z = asym_exit

            # Vol-inverse sizing (L5)
            sz = base_size
            if vol_inverse and sigma > 0 and median_sigma > 0:
                sz = base_size * float(np.clip(median_sigma / sigma, 0.5, 2.0))
            notional = CAPITAL * sz

            for bar_idx, ts in enumerate(month_bars):
                yp = close_panel.at[ts,yn]; xp = close_panel.at[ts,xn]
                if np.isnan(yp) or np.isnan(xp) or yp<=0 or xp<=0: continue

                # Z-score: Kalman (L1) or OLS
                if use_kalman:
                    delta_k=1e-4; R_k=1e-3
                    th = pair.get("_kalman_theta",np.array([beta,pair.get("intercept",0)]))
                    P = pair.get("_kalman_P",np.eye(2))
                    F = np.array([[math.log(xp),1.0]])
                    Pp = P + delta_k*np.eye(2)
                    yhat = F@th; S = float((F@Pp@F.T+R_k)[0,0])
                    K = Pp@F.T/S; e = math.log(yp)-yhat[0]
                    pair["_kalman_theta"] = th + K.flatten()*e
                    pair["_kalman_P"] = (np.eye(2)-K@F)@Pp
                    z = e/math.sqrt(S) if S>0 else 0
                    beta_eff = float(pair["_kalman_theta"][0])
                else:
                    spread = math.log(yp)-beta*math.log(xp)-pair.get("intercept",0)
                    z = (spread-mu)/sigma if sigma>0 else 0
                    beta_eff = beta

                yr = ret_panel.at[ts,yn] if ts in ret_panel.index and yn in ret_panel.columns else 0
                xr = ret_panel.at[ts,xn] if ts in ret_panel.index and xn in ret_panel.columns else 0
                if np.isnan(yr): yr=0
                if np.isnan(xr): xr=0

                bar_pnl = 0.0; cost = COST_BPS/10000*(1+abs(beta_eff))*notional

                if pos != 0:
                    bp = pos*(yr-beta_eff*xr)*notional
                    bar_pnl = bp; trade_pnl += bp
                    bh = int((ts-ets).total_seconds()/(15*60)) if isinstance(ets,pd.Timestamp) else MIN_HOLD+1
                    should_exit = False

                    if bh >= MIN_HOLD:
                        if pos==1 and z>=long_exit_z: should_exit=True
                        elif pos==-1 and z<=-short_exit_z: should_exit=True
                    if bh>=MAX_HOLD: should_exit=True
                    if abs(z)>BREAKDOWN_Z and bh>48: should_exit=True

                    # Stop-loss (L2)
                    if stop_loss_pct and trade_pnl < -stop_loss_pct*notional:
                        should_exit = True

                    if should_exit:
                        bar_pnl -= cost; pos=0; ets=None; trade_pnl=0.0

                if pos==0:
                    if z<=-long_entry_z:
                        pos=1; ets=ts; ez_actual=z; trade_pnl=0.0
                        bar_pnl-=cost; total_trades+=1
                    elif z>=short_entry_z:
                        pos=-1; ets=ts; ez_actual=z; trade_pnl=0.0
                        bar_pnl-=cost; total_trades+=1

                day = ts.normalize()
                daily_pnl[day] = daily_pnl.get(day,0)+bar_pnl

            pair["_pos"]=pos; pair["_ets"]=ets; pair["_ez"]=ez_actual; pair["_trade_pnl"]=trade_pnl

    dpnl = pd.Series(daily_pnl)
    if len(dpnl)<10: return None
    cum=dpnl.cumsum(); dr=dpnl/CAPITAL
    sharpe = dr.mean()/dr.std()*np.sqrt(365.25) if dr.std()>0 else 0
    peak=(cum+CAPITAL).cummax(); dd=abs(((cum+CAPITAL-peak)/peak).min())
    wd=abs(dr.min()) if len(dr)>0 else 0
    mpnl=dpnl.resample("ME").sum(); pm=(mpnl>0).sum()
    return {"sharpe":sharpe,"pnl_cal":cum.iloc[-1]/CAL_DAYS,"max_dd":dd,"worst_day":wd,
            "n_trades":total_trades,"pos_months":f"{pm}/{len(mpnl)} ({pm/max(len(mpnl),1)*100:.0f}%)"}


def main():
    t0 = time.time()
    print("="*110)
    print("LITERATURE-BASED IMPROVEMENTS — 6 experiments + stacking")
    print("="*110)

    print("\nLoading...", end=" ", flush=True)
    ohlcv_data = {}
    for f in sorted(CACHE_DIR.glob("*.parquet")):
        n = f.stem.replace("USDT","")
        try:
            df = pd.read_parquet(f, columns=["close","volume"])
            if len(df)>=2000: ohlcv_data[n]=df
        except: pass
    close_panel = pd.DataFrame({n:d["close"] for n,d in ohlcv_data.items()})
    ret_panel = close_panel.pct_change()
    print(f"{len(ohlcv_data)} assets", flush=True)

    months = pd.period_range(TEST_START, TEST_END, freq="M")
    print(f"Pre-computing screens...", end=" ", flush=True)
    screens = precompute_screens(close_panel, ohlcv_data, months)
    print(f"done ({time.time()-t0:.0f}s)", flush=True)

    configs = [
        ("BASELINE (sz=3% ex=0.5 drop=3)", {}),
        # L1: Kalman hedge
        ("L1: Kalman hedge", {"kalman": True}),
        # L2: Stop-loss
        ("L2: SL 2%", {"stop_loss": 0.02}),
        ("L2: SL 3%", {"stop_loss": 0.03}),
        ("L2: SL 5%", {"stop_loss": 0.05}),
        ("L2: SL 8%", {"stop_loss": 0.08}),
        # L3: Asymmetric
        ("L3: asym entry (L2.0/S1.5)", {"asym_entry": (2.0, 1.5)}),
        ("L3: asym entry (L1.5/S2.0)", {"asym_entry": (1.5, 2.0)}),
        ("L3: asym exit (L0.5/S0.3)", {"asym_exit": (0.5, 0.3)}),
        ("L3: asym exit (L0.3/S0.5)", {"asym_exit": (0.3, 0.5)}),
        # L4: OU-adaptive thresholds
        ("L4: OU-adaptive z", {"ou_adaptive": True}),
        # L5: Vol-inverse sizing
        ("L5: vol-inverse sizing", {"vol_inverse": True}),
        # L6: Cross-pair correlation limit
        ("L6: corr limit (max 2 shared)", {"corr_limit": True}),
        # Combinations
        ("L1+L2(3%): Kalman+SL", {"kalman": True, "stop_loss": 0.03}),
        ("L1+L4: Kalman+OU-adapt", {"kalman": True, "ou_adaptive": True}),
        ("L1+L5: Kalman+vol-inv", {"kalman": True, "vol_inverse": True}),
        ("L4+L5: OU+vol-inv", {"ou_adaptive": True, "vol_inverse": True}),
        ("L1+L4+L5: Kalman+OU+vol", {"kalman": True, "ou_adaptive": True, "vol_inverse": True}),
        ("ALL: L1+L2+L4+L5+L6", {"kalman": True, "stop_loss": 0.03, "ou_adaptive": True,
                                   "vol_inverse": True, "corr_limit": True}),
        # At 5% sizing
        ("5% BASELINE", {"base_size": 0.05}),
        ("5% L1+L4+L5", {"base_size": 0.05, "kalman": True, "ou_adaptive": True, "vol_inverse": True}),
    ]

    results = []
    for name, cfg in configs:
        et0 = time.time()
        r = simulate(screens, close_panel, ret_panel, months, cfg)
        if r:
            r["config"] = name; results.append(r)
            ok = r["max_dd"]<0.07 and r["worst_day"]<0.05
            print(f"  {name:<40s} SR={r['sharpe']:>5.2f} ${r['pnl_cal']:>5.2f}/cal "
                  f"DD={r['max_dd']*100:>4.1f}% Tr={r['n_trades']:>4d} +Mo={r['pos_months']} "
                  f"({time.time()-et0:.0f}s) {'OK' if ok else ''}", flush=True)
        else:
            print(f"  {name:<40s} NO DATA", flush=True)

    results.sort(key=lambda x:-x["sharpe"])
    print(f"\n{'='*110}")
    print("RANKED RESULTS")
    print(f"{'='*110}")
    print(f"\n{'#':>3s} {'Config':<42s} {'$/cal':>7s} {'Sharpe':>7s} {'DD%':>6s} {'WD%':>6s} {'Tr':>5s} {'+Mo':>10s}")
    print("-"*95)
    for i,r in enumerate(results):
        ok = r["max_dd"]<0.07 and r["worst_day"]<0.05
        print(f"{i+1:>3d} {r['config']:<42s} ${r['pnl_cal']:>6.2f} {r['sharpe']:>7.2f} "
              f"{r['max_dd']*100:>5.1f}% {r['worst_day']*100:>5.2f}% {r['n_trades']:>5d} "
              f"{r['pos_months']:>10s} {'OK' if ok else ''}")

    print(f"\nRuntime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
