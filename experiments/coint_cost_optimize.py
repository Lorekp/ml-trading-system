"""Coint Cost Optimization — Make coint net profitable in production

Problem: coint loses -$1.65/cal because 1591 trades at 10bps roundtrip = $1.56/cal drag.
Solution: fewer, higher-quality trades via cost-aware entry, AFML sizing, higher thresholds.

Runs coint standalone AND combined with LogReg T1 (loaded from signal cache).
"""

import sys, os, time, warnings, math, pickle
from pathlib import Path
from itertools import combinations

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import numpy as np
import pandas as pd
from scipy.stats import norm
from statsmodels.tsa.stattools import coint
from signal_engine.config import PortfolioConfig
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
import test_original_pipeline as orig

warnings.filterwarnings("ignore")

PCACHE = Path("data/cache_15m")
SCACHE = Path("data/optuna/signal_cache")
CAPITAL = 10000.0
COST_BPS = 5.0  # per entry OR exit (10 bps effective roundtrip)

# Coint base params
COINT_MH = 96; COINT_MXH = 96*30; COINT_EMB = 96*7
COINT_MCORR = 0.40; COINT_DROP = 3; COINT_ADD = 2; COINT_MPAIRS = 8

W = [("W1", 245), ("W2", 244), ("W3", 122)]


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


def afml_bet_size(z, entry_z):
    """AFML-style: map z-score to bet size via probability transform."""
    # Larger |z| → higher probability of mean reversion → bigger bet
    p = norm.cdf(abs(z) - entry_z + 0.5)  # shift so entry_z → p≈0.5
    p = max(0.5, min(p, 0.95))
    z_score = (p - 0.5) / math.sqrt(p * (1 - p)) if p > 0.5 else 0
    signal = 2 * norm.cdf(z_score) - 1
    return float(max(0.1, min(signal, 1.0)))


def run_coint(cfg, screens, close_panel, ret_panel, months):
    """Run one coint config. Returns daily PnL dict and trade count."""
    entry_z = cfg.get("entry_z", 2.0)
    exit_z = cfg.get("exit_z", 0.5)
    base_size = cfg.get("base_size", 0.03)
    use_afml = cfg.get("afml_sizing", False)
    cost_aware = cfg.get("cost_aware", False)
    min_profit = cfg.get("min_profit", False)

    book = {}; candidates = {}; daily_pnl = {}; n_trades = 0

    for month in months:
        ms = pd.Timestamp(month.start_time, tz="UTC"); me = ms + pd.offsets.MonthEnd(0)
        pd2 = screens.get(str(month), {}); passing = {k for k, v in pd2.items() if v["passes"]}

        for key in list(book.keys()):
            if key in passing or (key[1],key[0]) in passing:
                book[key]["fc"]=0
            else:
                book[key]["fc"]=book[key].get("fc",0)+1
                if book[key]["fc"]>=COINT_DROP:
                    if book[key].get("_pos",0)!=0:
                        ct=COST_BPS/10000*(1+abs(book[key].get("beta",1)))*CAPITAL*base_size
                        d=ms.normalize().tz_localize(None); daily_pnl[d]=daily_pnl.get(d,0)-ct
                    book.pop(key); continue
            pi=pd2.get(key) or pd2.get((key[1],key[0]))
            if pi: book[key].update(pi)

        for key in passing:
            if key in book or (key[1],key[0]) in book: continue
            candidates[key]=candidates.get(key,0)+1
            if candidates[key]>=COINT_ADD and len(book)<COINT_MPAIRS:
                pi=pd2.get(key)
                if pi and np.isfinite(pi["beta"]) and pi["beta"]>0:
                    book[key]={**pi,"fc":0,"_pos":0,"_ets":None}; candidates.pop(key,None)
        for key in list(candidates.keys()):
            if key not in passing and (key[1],key[0]) not in passing: candidates.pop(key,None)

        month_bars = close_panel.index[(close_panel.index>=ms)&(close_panel.index<=me)]
        for key, pair in book.items():
            yn,xn=key; beta=pair["beta"]; mu=pair["mu"]; sigma=pair["sigma"]
            if not np.isfinite(beta) or sigma<=0: continue
            if yn not in close_panel.columns or xn not in close_panel.columns: continue
            pos=pair.get("_pos",0); ets=pair.get("_ets",None)

            for ts in month_bars:
                yp=close_panel.at[ts,yn]; xp=close_panel.at[ts,xn]
                if np.isnan(yp) or np.isnan(xp) or yp<=0 or xp<=0: continue
                spread=math.log(yp)-beta*math.log(xp)-pair.get("intercept",0)
                z=(spread-mu)/sigma if sigma>0 else 0
                yr=ret_panel.at[ts,yn] if ts in ret_panel.index and yn in ret_panel.columns else 0
                xr=ret_panel.at[ts,xn] if ts in ret_panel.index and xn in ret_panel.columns else 0
                if np.isnan(yr): yr=0
                if np.isnan(xr): xr=0

                # Sizing
                sz = base_size
                if use_afml and pos == 0 and abs(z) >= entry_z:
                    sz = base_size * afml_bet_size(z, entry_z)

                notional = CAPITAL * sz
                cost = COST_BPS/10000*(1+abs(beta))*notional
                bar_pnl = 0.0

                if pos != 0:
                    bar_pnl = pos*(yr-beta*xr)*CAPITAL*base_size  # PnL on original notional
                    bh=int((ts-ets).total_seconds()/(15*60)) if isinstance(ets,pd.Timestamp) else COINT_MH+1
                    ex=False
                    if bh>=COINT_MH:
                        if pos==1 and z>=exit_z: ex=True
                        elif pos==-1 and z<=-exit_z: ex=True
                    if bh>=COINT_MXH: ex=True
                    if abs(z)>5.0 and bh>48: ex=True
                    if ex: bar_pnl-=cost; pos=0; ets=None

                if pos==0 and abs(z)>=entry_z:
                    # Cost-aware filter
                    enter = True
                    if cost_aware:
                        # Expected profit in dollar terms (rough estimate)
                        # When z reverts from entry to exit_z, spread moves by (|z|-exit_z)*sigma
                        # Hedged return ≈ spread_move / beta_adjusted_notional
                        expected_move = (abs(z) - exit_z) * sigma
                        expected_pnl = notional * expected_move
                        if expected_pnl < 1.5 * cost:
                            enter = False

                    if min_profit and enter:
                        expected_move = (abs(z) - exit_z) * sigma
                        expected_pnl = notional * expected_move
                        if expected_pnl < 2.0 * cost:
                            enter = False

                    if enter:
                        if z<=-entry_z: pos=1; ets=ts; bar_pnl-=cost; n_trades+=1
                        elif z>=entry_z: pos=-1; ets=ts; bar_pnl-=cost; n_trades+=1

                day=ts.normalize().tz_localize(None)
                daily_pnl[day]=daily_pnl.get(day,0)+bar_pnl
            pair["_pos"]=pos; pair["_ets"]=ets

    return daily_pnl, n_trades


def precompute_screens(close_panel, ohlcv_data, months):
    screens = {}
    for month in months:
        ms=pd.Timestamp(month.start_time,tz="UTC")
        le=ms-pd.Timedelta(days=1); uc={}
        for n,o in ohlcv_data.items():
            if (ms-o.index.min()).days<180: continue
            a=o.loc[:le]
            if len(a)<96*45: continue
            dv=(a["close"]*a["volume"]).resample("1D").sum()
            r=dv.iloc[-30:] if len(dv)>=30 else dv; v=r.median()
            if np.isnan(v) or v<=0: continue
            if len(dv)>=90 and dv.iloc[-90:].median()>0 and v/dv.iloc[-90:].median()>5: continue
            uc[n]=float(v)
        universe=[a for a,_ in sorted(uc.items(),key=lambda x:-x[1])[:30]]
        if len(universe)<6: screens[str(month)]={}; continue
        se=ms-pd.Timedelta(minutes=15*COINT_EMB); ss=se-pd.DateOffset(months=6)
        avail=[a for a in universe if a in close_panel.columns]; pairs={}
        for ai,aj in combinations(avail,2):
            m=(close_panel.index>=ss)&(close_panel.index<se)
            ys=close_panel.loc[m,ai].dropna(); xs=close_panel.loc[m,aj].dropna()
            co=ys.index.intersection(xs.index)
            if len(co)<3000: continue
            ya=ys.reindex(co).values.astype(float); xa=xs.reindex(co).values.astype(float)
            ry=np.diff(ya)/ya[:-1]; rx=np.diff(xa)/xa[:-1]
            v=np.isfinite(ry)&np.isfinite(rx)
            if v.sum()<2000: continue
            corr=float(np.corrcoef(ry[v],rx[v])[0,1])
            if not np.isfinite(corr) or corr<COINT_MCORR: continue
            try: _,p,_=coint(ya,xa,maxlag=10)
            except: p=1.0
            ly=np.log(ya); lx=np.log(xa)
            beta,intercept=ols_fit(ly,lx)
            if not np.isfinite(beta) or beta<=0: continue
            resid=ly-beta*lx-intercept; hl=half_life_bars(resid)/96
            passes=p<0.05 and 1<hl<20
            pairs[(ai,aj)]={"p":float(p),"beta":beta,"intercept":intercept,"hl":hl,
                            "mu":float(np.mean(resid)),"sigma":float(np.std(resid)),"passes":passes}
        screens[str(month)]=pairs
    return screens


def load_logreg_t1_dpnl():
    """Load LogReg T1 daily PnL from signal cache (exact same as production_combined_v2)."""
    dfs = {}
    for wl, cd in W:
        p = SCACHE / f"C0.02000_lb2_{wl}.pkl"
        if not p.exists(): continue
        with open(p, "rb") as f: dfs[wl] = pickle.load(f)
    if not dfs: return pd.Series(dtype=float)

    all_dpnl = {}
    for wl, cd in W:
        raw = dfs.get(wl)
        if raw is None or raw.empty: continue
        fi = raw[raw["confidence"] > 0.10]
        if len(fi) < 10: continue
        fi = fi.copy(); fi["fwd_return"] = fi["ret_6"].values
        combined = orig.apply_nco_wf(fi)
        pcfg = PortfolioConfig(starting_capital=CAPITAL, max_dd=0.10, delever_start=0.05,
            slippage=0.0001, max_exposure=0.90, max_per_asset=0.05,
            base_sizing_pct=0.30, max_signal_mult=3.0)
        sim = simulate_portfolio(combined, pcfg)
        if sim is None: continue
        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None: continue
        for dt, val in dpnl.items():
            dt2 = dt.tz_localize(None) if hasattr(dt, "tz") and dt.tz is not None else dt
            all_dpnl[dt2] = all_dpnl.get(dt2, 0) + val

    return pd.Series(all_dpnl).sort_index()


def main():
    t0 = time.time()
    print("=" * 110)
    print("COINT COST OPTIMIZATION — Make it profitable")
    print("=" * 110)

    # Load data
    print("\nLoading...", end=" ", flush=True)
    ohlcv_data = {}
    for f in sorted(PCACHE.glob("*.parquet")):
        n = f.stem.replace("USDT","")
        try:
            df = pd.read_parquet(f, columns=["close","volume"])
            if len(df)>=2000: ohlcv_data[n]=df
        except: pass
    close_panel = pd.DataFrame({n:d["close"] for n,d in ohlcv_data.items()})
    ret_panel = close_panel.pct_change()
    print(f"{len(ohlcv_data)} assets")

    months = pd.period_range("2024-07","2026-02",freq="M")
    print("Screens...", end=" ", flush=True)
    screens = precompute_screens(close_panel, ohlcv_data, months)
    print(f"done ({time.time()-t0:.0f}s)")

    # Load LogReg T1
    print("Loading LogReg T1...", end=" ", flush=True)
    lr_dpnl = load_logreg_t1_dpnl()
    print(f"{len(lr_dpnl)} days")

    # Configs to test
    configs = [
        ("Base z=2.0 3%", {"entry_z": 2.0, "exit_z": 0.5, "base_size": 0.03}),
        ("A: z=2.5 3%", {"entry_z": 2.5, "exit_z": 0.5, "base_size": 0.03}),
        ("B: z=3.0 3%", {"entry_z": 3.0, "exit_z": 0.5, "base_size": 0.03}),
        ("C: z=2.0 5%", {"entry_z": 2.0, "exit_z": 0.5, "base_size": 0.05}),
        ("D: z=3.0 5%", {"entry_z": 3.0, "exit_z": 0.5, "base_size": 0.05}),
        ("E: z=2.0 AFML", {"entry_z": 2.0, "exit_z": 0.5, "base_size": 0.05, "afml_sizing": True}),
        ("F: cost-aware z=2.0", {"entry_z": 2.0, "exit_z": 0.5, "base_size": 0.03, "cost_aware": True}),
        ("G: min-profit z=2.0", {"entry_z": 2.0, "exit_z": 0.5, "base_size": 0.03, "min_profit": True}),
        ("H: z=3.0 AFML 5%", {"entry_z": 3.0, "exit_z": 0.5, "base_size": 0.05, "afml_sizing": True}),
        ("I: z=2.5 cost-aware 5%", {"entry_z": 2.5, "exit_z": 0.5, "base_size": 0.05, "cost_aware": True}),
        ("J: z=3.0 min-profit 5%", {"entry_z": 3.0, "exit_z": 0.5, "base_size": 0.05, "min_profit": True}),
        ("K: z=4.0 5%", {"entry_z": 4.0, "exit_z": 0.5, "base_size": 0.05}),
    ]

    cal_days = 611
    print(f"\n{'Config':<30s} {'C$/cal':>7s} {'CSR':>6s} {'CDD':>5s} {'CTr':>5s} "
          f"{'Cmb$/c':>7s} {'CmbSR':>6s} {'CmbDD':>6s} {'CmbWD':>6s}")
    print("-" * 100)

    for name, cfg in configs:
        dpnl_dict, n_tr = run_coint(cfg, screens, close_panel, ret_panel, months)
        c_dpnl = pd.Series(dpnl_dict).sort_index()

        # Coint standalone metrics
        if len(c_dpnl) < 5:
            print(f"{name:<30s} NO DATA"); continue
        c_cum = c_dpnl.cumsum(); c_dr = c_dpnl / CAPITAL
        c_sr = c_dr.mean()/c_dr.std()*np.sqrt(365.25) if c_dr.std()>0 else 0
        c_peak = (c_cum+CAPITAL).cummax(); c_dd = abs(((c_cum+CAPITAL-c_peak)/c_peak).min())
        c_pnl_cal = c_cum.iloc[-1]/cal_days

        # Combined with LogReg
        all_d = lr_dpnl.index.union(c_dpnl.index).sort_values()
        combo = lr_dpnl.reindex(all_d).fillna(0) + c_dpnl.reindex(all_d).fillna(0)
        cb_dr = combo/CAPITAL
        cb_sr = cb_dr.mean()/cb_dr.std()*np.sqrt(365.25) if cb_dr.std()>0 else 0
        cb_cum = combo.cumsum()
        cb_peak = (cb_cum+CAPITAL).cummax(); cb_dd = abs(((cb_cum+CAPITAL-cb_peak)/cb_peak).min())
        cb_wd = abs(cb_dr.min())
        cb_pnl = cb_cum.iloc[-1]/cal_days

        ok = cb_dd < 0.10 and cb_wd < 0.05
        better = c_pnl_cal > 0

        print(f"{name:<30s} ${c_pnl_cal:>6.2f} {c_sr:>6.2f} {c_dd*100:>4.1f}% {n_tr:>5d} "
              f"${cb_pnl:>6.2f} {cb_sr:>6.2f} {cb_dd*100:>5.1f}% {cb_wd*100:>5.2f}% "
              f"{'OK' if ok else '  '} {'$+' if better else '  '}")

    print(f"\nLogReg T1 standalone: ${lr_dpnl.sum()/cal_days:.2f}/cal, SR={lr_dpnl.mean()/lr_dpnl.std()*np.sqrt(365.25):.2f}")
    print(f"Runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
