"""Production Combined Backtest v2 — Uses EXACT optuna_v6 pipeline + T1 config

LogReg: loads pre-cached signals from data/optuna/signal_cache/
        applies T1 params: C=0.02, lb=2, thr=0.10, fwd=6, fixed close,
        5x leverage, NCO allocation, simulate_portfolio with dynamic sizing
Coint:  rolling health check, optimal config, 5 bps/trade cost
"""

import sys, os, time, warnings, pickle, math
from pathlib import Path
from itertools import combinations

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))
sys.path.insert(0, os.path.join(_root, "experiments"))

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import coint

from signal_engine.config import PortfolioConfig
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
import test_original_pipeline as orig

warnings.filterwarnings("ignore")

CACHE_DIR = Path("data/optuna/signal_cache")
PCACHE_DIR = Path("data/cache_15m")
CAPITAL = 10000.0  # Match T1 validated config

# T1 exact params from CONFIG.md
T1_COMBO = "0.02000_lb2"
T1_THR = 0.10; T1_FWD = 6; T1_CLOSE = "fixed"
T1_LEV = 5.0; T1_DLV = 0.05; T1_MAXEXP = 0.90; T1_MAXPA = 0.05
T1_SLIP = 0.0001; T1_BASE = 0.06  # 0.06 * 5x = 0.30 effective

# Coint params
COINT_COST = 5.0  # bps per entry/exit (10 bps effective roundtrip)
COINT_SIZE = 0.03; COINT_EZ = 2.0; COINT_XZ = 0.5
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


def run_logreg_t1():
    """Run LogReg with EXACT T1 config using cached signals + optuna_v6 pipeline."""
    # Load cached signals
    dfs = {}
    for wl, cd in W:
        cache_path = CACHE_DIR / f"C{T1_COMBO}_{wl}.pkl"
        if not cache_path.exists():
            print(f"  WARNING: {cache_path} not found"); continue
        with open(cache_path, "rb") as f:
            dfs[wl] = pickle.load(f)
    if not dfs:
        return {}, pd.Series(dtype=float)

    # Per-window simulation (EXACT same as optuna_v6 evaluate())
    results = {}
    all_dpnl = {}

    for wl, cd in W:
        raw = dfs.get(wl)
        if raw is None or raw.empty: continue

        # Apply T1 threshold
        fi = raw[raw["confidence"] > T1_THR]
        if len(fi) < 10: continue
        fi = fi.copy()

        # Close rule: fixed
        fi["fwd_return"] = fi[f"ret_{T1_FWD}"].values

        # NCO allocation (walk-forward, exact same as original)
        combined = orig.apply_nco_wf(fi)

        # Portfolio simulation with T1 sizing
        pcfg = PortfolioConfig(
            starting_capital=CAPITAL, max_dd=0.10, delever_start=T1_DLV,
            slippage=T1_SLIP, max_exposure=T1_MAXEXP, max_per_asset=T1_MAXPA,
            base_sizing_pct=T1_BASE * T1_LEV, max_signal_mult=3.0,
        )
        sim = simulate_portfolio(combined, pcfg)
        if sim is None or sim["n_trades"] < 5: continue

        dpnl = compute_daily_pnl_series(sim)
        if dpnl is None or len(dpnl) < 1: continue

        # Collect daily PnL
        for dt, val in dpnl.items():
            # Normalize TZ
            if hasattr(dt, "tz") and dt.tz is not None:
                dt = dt.tz_localize(None)
            all_dpnl[dt] = all_dpnl.get(dt, 0) + val

        eq = pd.Series(sim["equity"][~sim["skipped"]],
                       index=pd.to_datetime(sim["timestamps"][~sim["skipped"]]))
        dd = ((eq - eq.expanding().max()) / eq.expanding().max())
        dr = dpnl / CAPITAL

        pnl = sim["equity"][-1] - CAPITAL
        results[wl] = {
            "pnl": pnl, "pnl_cal": pnl / cd,
            "max_dd": abs(dd.min()), "worst_day": abs(dr.min()),
            "n_trades": sim["n_trades"], "sharpe": dr.mean() / dr.std() * np.sqrt(365.25) if dr.std() > 0 else 0,
        }
        print(f"  {wl}: ${pnl/cd:.2f}/cal, SR={results[wl]['sharpe']:.2f}, "
              f"DD={results[wl]['max_dd']*100:.1f}%, WD={results[wl]['worst_day']*100:.2f}%, "
              f"Tr={sim['n_trades']}")

    return results, pd.Series(all_dpnl).sort_index()


def run_coint():
    """Coint strategy with optimal config. Returns daily PnL Series."""
    ohlcv_data = {}
    for f in sorted(PCACHE_DIR.glob("*.parquet")):
        n = f.stem.replace("USDT","")
        try:
            df = pd.read_parquet(f, columns=["close","volume"])
            if len(df)>=2000: ohlcv_data[n]=df
        except: pass
    close_panel = pd.DataFrame({n:d["close"] for n,d in ohlcv_data.items()})
    ret_panel = close_panel.pct_change()

    months = pd.period_range("2024-07","2026-02",freq="M")
    book={}; candidates={}; daily_pnl={}; n_trades=0

    for month in months:
        ms=pd.Timestamp(month.start_time,tz="UTC"); me=ms+pd.offsets.MonthEnd(0)
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

        se=ms-pd.Timedelta(minutes=15*COINT_EMB); ss=se-pd.DateOffset(months=6)
        avail=[a for a in universe if a in close_panel.columns]
        passing=set(); pi={}
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
            if p<0.05 and 1<hl<20:
                passing.add((ai,aj))
                pi[(ai,aj)]={"beta":beta,"intercept":intercept,
                    "mu":float(np.mean(resid)),"sigma":float(np.std(resid))}

        for key in list(book.keys()):
            if key in passing or (key[1],key[0]) in passing:
                book[key]["fc"]=0
            else:
                book[key]["fc"]=book[key].get("fc",0)+1
                if book[key]["fc"]>=COINT_DROP:
                    if book[key].get("_pos",0)!=0:
                        ct=COINT_COST/10000*(1+abs(book[key].get("beta",1)))*CAPITAL*COINT_SIZE
                        d=ms.normalize().tz_localize(None)
                        daily_pnl[d]=daily_pnl.get(d,0)-ct
                    book.pop(key); continue
            info=pi.get(key) or pi.get((key[1],key[0]))
            if info: book[key].update(info)

        for key in passing:
            if key in book or (key[1],key[0]) in book: continue
            candidates[key]=candidates.get(key,0)+1
            if candidates[key]>=COINT_ADD and len(book)<COINT_MPAIRS:
                info=pi.get(key)
                if info: book[key]={**info,"fc":0,"_pos":0,"_ets":None}; candidates.pop(key,None)
        for key in list(candidates.keys()):
            if key not in passing and (key[1],key[0]) not in passing: candidates.pop(key,None)

        month_bars=close_panel.index[(close_panel.index>=ms)&(close_panel.index<=me)]
        for key,pair in book.items():
            yn,xn=key; beta=pair["beta"]; mu=pair["mu"]; sigma=pair["sigma"]
            if not np.isfinite(beta) or sigma<=0: continue
            if yn not in close_panel.columns or xn not in close_panel.columns: continue
            pos=pair.get("_pos",0); ets=pair.get("_ets",None)
            notional=CAPITAL*COINT_SIZE
            for ts in month_bars:
                yp=close_panel.at[ts,yn]; xp=close_panel.at[ts,xn]
                if np.isnan(yp) or np.isnan(xp) or yp<=0 or xp<=0: continue
                spread=math.log(yp)-beta*math.log(xp)-pair.get("intercept",0)
                z=(spread-mu)/sigma if sigma>0 else 0
                yr=ret_panel.at[ts,yn] if ts in ret_panel.index and yn in ret_panel.columns else 0
                xr=ret_panel.at[ts,xn] if ts in ret_panel.index and xn in ret_panel.columns else 0
                if np.isnan(yr): yr=0
                if np.isnan(xr): xr=0
                bar_pnl=0.0; cost=COINT_COST/10000*(1+abs(beta))*notional
                if pos!=0:
                    bar_pnl=pos*(yr-beta*xr)*notional
                    bh=int((ts-ets).total_seconds()/(15*60)) if isinstance(ets,pd.Timestamp) else COINT_MH+1
                    ex=False
                    if bh>=COINT_MH:
                        if pos==1 and z>=COINT_XZ: ex=True
                        elif pos==-1 and z<=-COINT_XZ: ex=True
                    if bh>=COINT_MXH: ex=True
                    if abs(z)>5.0 and bh>48: ex=True
                    if ex: bar_pnl-=cost; pos=0; ets=None
                if pos==0:
                    if z<=-COINT_EZ: pos=1; ets=ts; bar_pnl-=cost; n_trades+=1
                    elif z>=COINT_EZ: pos=-1; ets=ts; bar_pnl-=cost; n_trades+=1
                day=ts.normalize().tz_localize(None)
                daily_pnl[day]=daily_pnl.get(day,0)+bar_pnl
            pair["_pos"]=pos; pair["_ets"]=ets

    print(f"  Coint: {n_trades} trades across {len(book)} pairs")
    return pd.Series(daily_pnl).sort_index(), n_trades


def metrics(dpnl, label, capital=CAPITAL, cal_days=611):
    if len(dpnl)<5: return
    dr=dpnl/capital
    sharpe=dr.mean()/dr.std()*np.sqrt(365.25) if dr.std()>0 else 0
    cum=dpnl.cumsum(); peak=(cum+capital).cummax()
    dd=abs(((cum+capital-peak)/peak).min()); wd=abs(dr.min())
    pnl_cal=cum.iloc[-1]/cal_days
    monthly=dpnl.resample("ME").sum(); pm=(monthly>0).sum()
    print(f"  {label:<15s} SR={sharpe:>5.2f}  ${pnl_cal:>7.2f}/cal  "
          f"DD={dd*100:>5.1f}%  WD={wd*100:>5.2f}%  +Mo={pm}/{len(monthly)} ({pm/len(monthly)*100:.0f}%)")
    return {"sharpe":sharpe,"pnl_cal":pnl_cal,"dd":dd,"wd":wd}


def main():
    t0=time.time()
    print("="*110)
    print("PRODUCTION COMBINED BACKTEST v2")
    print(f"LogReg T1: C=0.02 lb=2 fwd=6 thr=0.10 lev=5x (from optuna_v6 signal cache)")
    print(f"Coint: corr>0.40 lb6 z=2.0 ex=0.5 3%/leg drop=3 (5 bps/trade)")
    print(f"Capital: ${CAPITAL:,.0f}")
    print("="*110)

    # LogReg
    print("\n--- LogReg T1 ---")
    lr_results, lr_dpnl = run_logreg_t1()

    # Coint
    print("\n--- Coint Spread ---")
    coint_dpnl, coint_trades = run_coint()

    # Normalize TZ
    if len(lr_dpnl)>0 and lr_dpnl.index.tz is not None:
        lr_dpnl.index=lr_dpnl.index.tz_localize(None)
    if len(coint_dpnl)>0 and coint_dpnl.index.tz is not None:
        coint_dpnl.index=coint_dpnl.index.tz_localize(None)

    # Combined
    all_dates=lr_dpnl.index.union(coint_dpnl.index).sort_values() if len(lr_dpnl)>0 else coint_dpnl.index
    combined=lr_dpnl.reindex(all_dates).fillna(0)+coint_dpnl.reindex(all_dates).fillna(0)

    print(f"\n{'='*110}")
    print("AGGREGATE METRICS")
    print(f"{'='*110}")
    metrics(lr_dpnl, "LogReg T1")
    metrics(coint_dpnl, "Coint Spread")
    metrics(combined, "COMBINED")

    if len(lr_dpnl)>0 and len(coint_dpnl)>0:
        common=lr_dpnl.index.intersection(coint_dpnl.index)
        if len(common)>20:
            rho=float(lr_dpnl.reindex(common).corr(coint_dpnl.reindex(common)))
            print(f"\n  Correlation: {rho:.3f}")

    # Monthly
    print(f"\n{'Month':<10s} {'LogReg':>10s} {'Coint':>10s} {'Combined':>10s}")
    print("-"*45)
    lr_m=lr_dpnl.resample("ME").sum() if len(lr_dpnl)>0 else pd.Series(dtype=float)
    c_m=coint_dpnl.resample("ME").sum()
    combo_m=combined.resample("ME").sum()
    for dt in combo_m.index:
        l=lr_m.get(dt,0); c=c_m.get(dt,0); x=l+c
        flag=" <<<" if x<0 else ""
        print(f"{dt.strftime('%Y-%m'):<10s} ${l:>9.0f} ${c:>9.0f} ${x:>9.0f}{flag}")

    # Risk
    if len(combined)>0:
        dr=combined/CAPITAL; cum=combined.cumsum()
        peak=(cum+CAPITAL).cummax(); dd=abs(((cum+CAPITAL-peak)/peak).min()); wd=abs(dr.min())
        print(f"\n--- RISK CHECK ---")
        print(f"  Max DD: {dd*100:.1f}% {'PASS' if dd<0.10 else 'FAIL'} (limit 10%)")
        print(f"  Worst day: {wd*100:.2f}% {'PASS' if wd<0.05 else 'FAIL'} (limit 5%)")

    print(f"\nRuntime: {time.time()-t0:.0f}s")


if __name__=="__main__":
    main()
