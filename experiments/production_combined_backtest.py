"""Production Combined Backtest: LogReg T1 + Coint Spread

Full proper run with:
- LogReg: per-asset monthly retrain, T1 params (C=0.02, lb=2, fwd=6, thr=0.10)
- Coint: rolling health check, optimal config (corr>0.40, 6mo OLS, z=2.0, ex=0.5)
- Realistic costs: 3 bps LogReg, 7 bps Coint
- Fixed: winsorization uses training quantiles only
- Capital: $100K
- Walk-forward: W1/W2/W3
"""

import sys, os, time, warnings, math
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
os.chdir(_root)
sys.path.insert(0, os.path.join(_root, "src"))

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.stattools import coint

from signal_engine.config import load_config, ConfidenceFilterConfig, PortfolioConfig
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.stats import compute_full_stats
from afml.bet_sizing import get_signal, discretize_signal

warnings.filterwarnings("ignore")
NEW_SKLEARN = tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8)
DATA_DIR = Path("data/binance-1m")
CACHE_DIR = Path("data/cache_15m")
CAPITAL = 100000.0

# T1 LogReg params
LR_C = 0.02
LR_LOOKBACK = 2
LR_FWD_BARS = 6
LR_THRESHOLD = 0.10
LR_SLIPPAGE = 0.0002  # 2 bps (MEXC 0% maker + 1bps slippage + 1bps spread)

# Coint params — 5 bps per entry/exit (= 10 bps effective roundtrip, matching validated results)
COINT_COST_BPS = 5.0
COINT_SIZE = 0.03; COINT_ENTRY_Z = 2.0; COINT_EXIT_Z = 0.5
COINT_MIN_HOLD = 96; COINT_MAX_HOLD = 96*30; COINT_EMBARGO = 96*7
COINT_MIN_CORR = 0.40; COINT_DROP = 3; COINT_ADD = 2; COINT_MAX_PAIRS = 8

ASSETS = {
    "BTC": "BTCUSDT", "ETH": "ETHUSDT", "SOL": "SOLUSDT", "XRP": "XRPUSDT",
    "BNB": "BNBUSDT", "ADA": "ADAUSDT", "AVAX": "AVAXUSDT", "LINK": "LINKUSDT",
    "DOT": "DOTUSDT", "SUI": "SUIUSDT", "NEAR": "NEARUSDT", "LTC": "LTCUSDT",
    "BCH": "BCHUSDT", "UNI": "UNIUSDT", "AAVE": "AAVEUSDT", "FIL": "FILUSDT",
    "APT": "APTUSDT", "FET": "FETUSDT", "ENA": "ENAUSDT", "HBAR": "HBARUSDT",
    "XLM": "XLMUSDT", "TRX": "TRXUSDT", "FTM": "FTMUSDT", "WLD": "WLDUSDT",
    "ZEC": "ZECUSDT", "TAO": "TAOUSDT", "ENJ": "ENJUSDT", "ANKR": "ANKRUSDT",
    "SXP": "SXPUSDT", "BNX": "BNXUSDT",
}

WINDOWS = [
    {"label": "W1", "start": "2024-07", "end": "2025-02", "cal": 245},
    {"label": "W2", "start": "2025-03", "end": "2025-10", "cal": 244},
    {"label": "W3", "start": "2025-11", "end": "2026-02", "cal": 122},
]


# ============================================================
# LogReg Pipeline (T1 config, fixed winsorization)
# ============================================================

def make_logreg():
    if NEW_SKLEARN:
        return LogisticRegression(C=LR_C, l1_ratio=1.0, solver="saga", max_iter=1000, random_state=42)
    return LogisticRegression(C=LR_C, penalty="l1", solver="saga", max_iter=1000, random_state=42)


def process_asset_t1(args):
    """Per-asset walk-forward with T1 params. Fixed winsorization."""
    asset, symbol = args
    warnings.filterwarnings("ignore")
    sys.path.insert(0, "src")
    from signal_engine.config import load_config
    from signal_engine.data import load_binance_1m, resample_ohlcv
    from signal_engine.features import compute_features
    from signal_engine.weights import compute_sample_weights
    from afml.bet_sizing import get_signal, discretize_signal

    cfg = load_config(Path("configs/default.yaml"))
    try:
        ohlcv_1m = load_binance_1m(symbol, [Path("data/binance-1m")])
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
    except Exception:
        return asset, []

    close = ohlcv_15m["close"]
    features = compute_features(ohlcv_15m, cfg.features)
    feat = features[cfg.features.feature_list].copy()

    # Label: fwd_bars=6 for T1
    fwd_ret = close.pct_change(LR_FWD_BARS).shift(-LR_FWD_BARS)

    # Vol filter (Parkinson-style via EWM std — matches T1)
    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

    fc = feat.dropna()
    y_all = fwd_ret.reindex(fc.index).dropna()
    valid = fc.index.intersection(y_all.index)
    if len(valid) < 3000:
        return asset, []

    start_date = ohlcv_15m.index.min()
    end_date = ohlcv_15m.index.max()
    median_gap = ohlcv_15m.index.to_series().diff().dropna().median()
    embargo_td = median_gap * 5

    all_months = pd.period_range(
        start=(start_date + pd.DateOffset(months=LR_LOOKBACK)).to_period("M"),
        end=end_date.to_period("M"), freq="M",
    )

    all_signals = []

    for month in all_months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if fc.index.tz is None:
            ms = ms.tz_localize(None); me = me.tz_localize(None)

        te = valid[(valid >= ms) & (valid <= me)]

        # Vol filter: only trade high-vol bars (from prior 3 months)
        vol_train_start = ms - pd.DateOffset(months=3)
        vol_train = vol.loc[(vol.index >= vol_train_start) & (vol.index < ms)].dropna()
        if len(vol_train) < 50: continue
        vt = vol_train.quantile(0.50)
        test_vol = vol.reindex(te)
        hv = (test_vol > vt).fillna(False)
        te = te[hv.reindex(te).fillna(False).values]
        if len(te) < 5: continue

        # Training window: T1 uses 2-month lookback
        train_start = ms - pd.DateOffset(months=LR_LOOKBACK)
        if train_start < start_date: continue
        train_cutoff = ms - embargo_td
        tr = valid[(valid >= train_start) & (valid < train_cutoff)]
        if len(tr) < 200: continue

        try:
            # FIX: Winsorize using TRAINING quantiles only
            fc_tr = fc.loc[tr].copy()
            winsor_lo = {}; winsor_hi = {}
            for col in ["ret_1", "ret_3", "ret_7", "ret_14"]:
                if col in fc_tr.columns:
                    winsor_lo[col] = fc_tr[col].quantile(0.01)
                    winsor_hi[col] = fc_tr[col].quantile(0.99)
                    fc_tr[col] = fc_tr[col].clip(winsor_lo[col], winsor_hi[col])

            fc_te = fc.loc[te].copy()
            for col in winsor_lo:
                if col in fc_te.columns:
                    fc_te[col] = fc_te[col].clip(winsor_lo[col], winsor_hi[col])

            sw = compute_sample_weights(fc_tr.index.values.astype(np.int64), fwd_bars=LR_FWD_BARS)
            scaler = StandardScaler()
            Xtr = scaler.fit_transform(fc_tr)
            model = make_logreg()
            model.fit(Xtr, (y_all.loc[tr] > 0).astype(int), sample_weight=sw)
            Xte = scaler.transform(fc_te)
            yp = model.predict_proba(Xte)[:, 1]
        except Exception:
            continue

        conf = np.abs(yp - 0.5)
        dirs = np.where(yp > 0.5, 1.0, -1.0)
        sigs = get_signal(yp, num_classes=2)
        sd = discretize_signal(sigs, step_size=0.1)
        ss = np.abs(sd) if np.ndim(sd) > 0 else np.full(len(yp), abs(float(sd)))
        fv = fwd_ret.reindex(te).values
        finite = np.isfinite(fv)
        if not finite.any(): continue

        # Filter by confidence threshold (T1: 0.10)
        mask = conf >= LR_THRESHOLD
        combined_mask = finite & mask
        if not combined_mask.any(): continue

        signals = pd.DataFrame({
            "timestamp": te[combined_mask], "direction": dirs[combined_mask],
            "fwd_return": fv[combined_mask], "confidence": conf[combined_mask],
            "probability": yp[combined_mask], "signal_strength": ss[combined_mask],
        })
        signals["asset"] = asset
        if len(signals) > 0:
            all_signals.append(signals)

    return asset, all_signals


# ============================================================
# Coint Pipeline (optimal config, reused from prior experiments)
# ============================================================

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


def run_coint_strategy(test_start, test_end):
    """Full coint strategy with rolling health check. Returns daily PnL Series."""
    ohlcv_data = {}
    for f in sorted(CACHE_DIR.glob("*.parquet")):
        n = f.stem.replace("USDT", "")
        try:
            df = pd.read_parquet(f, columns=["close", "volume"])
            if len(df) >= 2000: ohlcv_data[n] = df
        except: pass

    close_panel = pd.DataFrame({n: d["close"] for n, d in ohlcv_data.items()})
    ret_panel = close_panel.pct_change()

    months = pd.period_range(test_start[:7], test_end[:7], freq="M")
    book = {}; candidates = {}; daily_pnl = {}; total_trades = 0

    for month in months:
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = ms + pd.offsets.MonthEnd(0)

        # Universe selection
        le = ms - pd.Timedelta(days=1); univ_cands = {}
        for n, o in ohlcv_data.items():
            if (ms - o.index.min()).days < 180: continue
            a = o.loc[:le]
            if len(a) < 96*45: continue
            dv = (a["close"]*a["volume"]).resample("1D").sum()
            r = dv.iloc[-30:] if len(dv)>=30 else dv; v = r.median()
            if np.isnan(v) or v<=0: continue
            if len(dv)>=90 and dv.iloc[-90:].median()>0 and v/dv.iloc[-90:].median()>5: continue
            univ_cands[n] = float(v)
        universe = [a for a,_ in sorted(univ_cands.items(), key=lambda x:-x[1])[:30]]

        # Screen pairs
        se = ms - pd.Timedelta(minutes=15*COINT_EMBARGO); ss = se - pd.DateOffset(months=6)
        avail = [a for a in universe if a in close_panel.columns]
        passing = set(); pairs_info = {}
        for ai, aj in combinations(avail, 2):
            m = (close_panel.index >= ss) & (close_panel.index < se)
            ys = close_panel.loc[m, ai].dropna(); xs = close_panel.loc[m, aj].dropna()
            co = ys.index.intersection(xs.index)
            if len(co) < 3000: continue
            ya = ys.reindex(co).values.astype(float); xa = xs.reindex(co).values.astype(float)
            ry = np.diff(ya)/ya[:-1]; rx = np.diff(xa)/xa[:-1]
            v = np.isfinite(ry) & np.isfinite(rx)
            if v.sum()<2000: continue
            corr = float(np.corrcoef(ry[v],rx[v])[0,1])
            if not np.isfinite(corr) or corr < COINT_MIN_CORR: continue
            try: _, p, _ = coint(ya, xa, maxlag=10)
            except: p=1.0
            ly=np.log(ya); lx=np.log(xa)
            beta, intercept = ols_fit(ly, lx)
            if not np.isfinite(beta) or beta<=0: continue
            resid = ly-beta*lx-intercept; hl = half_life_bars(resid)/96
            if p<0.05 and 1<hl<20:
                passing.add((ai,aj))
                pairs_info[(ai,aj)] = {"beta":beta,"intercept":intercept,
                    "mu":float(np.mean(resid)),"sigma":float(np.std(resid))}

        # Book management
        for key in list(book.keys()):
            if key in passing or (key[1],key[0]) in passing:
                book[key]["fc"]=0
            else:
                book[key]["fc"]=book[key].get("fc",0)+1
                if book[key]["fc"]>=COINT_DROP:
                    if book[key].get("_pos",0)!=0:
                        ct=COINT_COST_BPS/10000*(1+abs(book[key].get("beta",1)))*CAPITAL*COINT_SIZE
                        daily_pnl[ms.normalize()]=daily_pnl.get(ms.normalize(),0)-ct
                    book.pop(key); continue
            pi=pairs_info.get(key) or pairs_info.get((key[1],key[0]))
            if pi: book[key].update(pi)

        for key in passing:
            if key in book or (key[1],key[0]) in book: continue
            candidates[key]=candidates.get(key,0)+1
            if candidates[key]>=COINT_ADD and len(book)<COINT_MAX_PAIRS:
                pi=pairs_info.get(key)
                if pi:
                    book[key]={**pi,"fc":0,"_pos":0,"_ets":None}
                    candidates.pop(key,None)
        for key in list(candidates.keys()):
            if key not in passing and (key[1],key[0]) not in passing:
                candidates.pop(key,None)

        # Trade
        month_bars = close_panel.index[(close_panel.index>=ms)&(close_panel.index<=me)]
        for key, pair in book.items():
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
                bar_pnl=0.0; cost=COINT_COST_BPS/10000*(1+abs(beta))*notional
                if pos!=0:
                    bar_pnl=pos*(yr-beta*xr)*notional
                    bh=int((ts-ets).total_seconds()/(15*60)) if isinstance(ets,pd.Timestamp) else COINT_MIN_HOLD+1
                    ex=False
                    if bh>=COINT_MIN_HOLD:
                        if pos==1 and z>=COINT_EXIT_Z: ex=True
                        elif pos==-1 and z<=-COINT_EXIT_Z: ex=True
                    if bh>=COINT_MAX_HOLD: ex=True
                    if abs(z)>5.0 and bh>48: ex=True
                    if ex: bar_pnl-=cost; pos=0; ets=None
                if pos==0:
                    if z<=-COINT_ENTRY_Z: pos=1; ets=ts; bar_pnl-=cost; total_trades+=1
                    elif z>=COINT_ENTRY_Z: pos=-1; ets=ts; bar_pnl-=cost; total_trades+=1
                day=ts.normalize()
                daily_pnl[day]=daily_pnl.get(day,0)+bar_pnl
            pair["_pos"]=pos; pair["_ets"]=ets

    return pd.Series(daily_pnl).sort_index(), total_trades


# ============================================================
# Main
# ============================================================

def main():
    t0 = time.time()
    print("=" * 110)
    print("PRODUCTION COMBINED BACKTEST — LogReg T1 + Coint Spread")
    print(f"Capital: ${CAPITAL:,.0f}  |  LogReg slippage: {LR_SLIPPAGE*10000:.0f}bps  |  Coint slippage: {COINT_COST_BPS/10000*10000:.0f}bps")
    print("=" * 110)

    # ---- Part 1: LogReg T1 ----
    print("\n--- Part 1: LogReg T1 (per-asset walk-forward) ---")
    print(f"C={LR_C}, lb={LR_LOOKBACK}mo, fwd={LR_FWD_BARS}, thr={LR_THRESHOLD}", flush=True)

    W = max(1, os.cpu_count() - 1)
    tasks = list(ASSETS.items())
    all_lr_signals = []

    print(f"Processing {len(tasks)} assets with {W} workers...", flush=True)
    with ProcessPoolExecutor(max_workers=W) as pool:
        futures = {pool.submit(process_asset_t1, t): t for t in tasks}
        for f in as_completed(futures):
            asset, sigs = f.result()
            if sigs:
                all_lr_signals.extend(sigs)
                total = sum(len(s) for s in sigs)
                print(f"  {asset}: {total} signals", flush=True)

    if not all_lr_signals:
        print("ERROR: No LogReg signals generated"); return

    lr_combined = pd.concat(all_lr_signals).sort_values("timestamp").reset_index(drop=True)
    print(f"\nLogReg total: {len(lr_combined)} signals across {lr_combined['asset'].nunique()} assets")

    # Portfolio simulation with T1 sizing
    lr_pcfg = PortfolioConfig(
        starting_capital=CAPITAL, max_dd=0.10, delever_start=0.05,
        slippage=LR_SLIPPAGE, max_exposure=0.90, max_per_asset=0.05,
        base_sizing_pct=0.06, max_signal_mult=3.0,  # 5x leverage via 0.06 * 5 = 0.30
    )
    lr_sim = simulate_portfolio(lr_combined, lr_pcfg)
    lr_dpnl = compute_daily_pnl_series(lr_sim) if lr_sim else pd.Series(dtype=float)

    # ---- Part 2: Coint Spread ----
    print("\n--- Part 2: Coint Spread (rolling health check) ---", flush=True)
    coint_dpnl, coint_trades = run_coint_strategy("2024-07-01", "2026-02-28")
    print(f"Coint total: {coint_trades} trades")

    # ---- Part 3: Combined ----
    print("\n" + "=" * 110)
    print("RESULTS")
    print("=" * 110)

    # Per-window analysis
    # Normalize TZ: strip tz from both series for clean comparison
    if len(lr_dpnl) > 0 and lr_dpnl.index.tz is None:
        pass  # already naive
    elif len(lr_dpnl) > 0:
        lr_dpnl.index = lr_dpnl.index.tz_localize(None)
    if len(coint_dpnl) > 0 and coint_dpnl.index.tz is not None:
        coint_dpnl.index = coint_dpnl.index.tz_localize(None)

    for w in WINDOWS:
        ws = pd.Timestamp(f"{w['start']}-01")
        we = pd.Timestamp(f"{w['end']}-28")

        lr_w = lr_dpnl[(lr_dpnl.index >= ws) & (lr_dpnl.index <= we)] if len(lr_dpnl) > 0 else pd.Series(dtype=float)
        c_w = coint_dpnl[(coint_dpnl.index >= ws) & (coint_dpnl.index <= we)]

        all_dates = lr_w.index.union(c_w.index).sort_values()
        combined_w = lr_w.reindex(all_dates).fillna(0) + c_w.reindex(all_dates).fillna(0)

        for label, dpnl in [("LogReg", lr_w), ("Coint", c_w), ("COMBINED", combined_w)]:
            if len(dpnl) < 5: print(f"  {w['label']} {label:<12s} insufficient data"); continue
            dr = dpnl / CAPITAL
            sharpe = dr.mean()/dr.std()*np.sqrt(365.25) if dr.std()>0 else 0
            cum = dpnl.cumsum()
            peak = (cum+CAPITAL).cummax(); dd = abs(((cum+CAPITAL-peak)/peak).min())
            wd = abs(dr.min())
            pnl_cal = cum.iloc[-1]/w["cal"]
            print(f"  {w['label']} {label:<12s} SR={sharpe:>5.2f}  ${pnl_cal:>7.2f}/cal  "
                  f"DD={dd*100:>5.1f}%  WD={wd*100:>5.2f}%")

    # Aggregate
    print(f"\n--- AGGREGATE (all windows) ---")
    all_dates = lr_dpnl.index.union(coint_dpnl.index).sort_values() if len(lr_dpnl) > 0 else coint_dpnl.index
    combined_full = lr_dpnl.reindex(all_dates).fillna(0) + coint_dpnl.reindex(all_dates).fillna(0)
    total_cal = sum(w["cal"] for w in WINDOWS)

    for label, dpnl in [("LogReg", lr_dpnl), ("Coint", coint_dpnl), ("COMBINED", combined_full)]:
        if len(dpnl) < 5: continue
        dr = dpnl / CAPITAL
        sharpe = dr.mean()/dr.std()*np.sqrt(365.25) if dr.std()>0 else 0
        cum = dpnl.cumsum()
        peak = (cum+CAPITAL).cummax(); dd = abs(((cum+CAPITAL-peak)/peak).min())
        wd = abs(dr.min())
        pnl_cal = cum.iloc[-1]/total_cal
        monthly = dpnl.resample("ME").sum(); pm = (monthly>0).sum()
        print(f"  {label:<12s} SR={sharpe:>5.2f}  ${pnl_cal:>7.2f}/cal  "
              f"DD={dd*100:>5.1f}%  WD={wd*100:>5.2f}%  +Mo={pm}/{len(monthly)} ({pm/len(monthly)*100:.0f}%)")

    # Correlation
    if len(lr_dpnl) > 0 and len(coint_dpnl) > 0:
        common = lr_dpnl.index.intersection(coint_dpnl.index)
        if len(common) > 20:
            rho = float(lr_dpnl.reindex(common).corr(coint_dpnl.reindex(common)))
            print(f"\n  Daily PnL correlation: {rho:.3f}")

    # Monthly breakdown
    print(f"\n--- MONTHLY PNL ---")
    lr_monthly = lr_dpnl.resample("ME").sum() if len(lr_dpnl) > 0 else pd.Series(dtype=float)
    c_monthly = coint_dpnl.resample("ME").sum()
    combo_monthly = combined_full.resample("ME").sum()

    print(f"{'Month':<10s} {'LogReg':>10s} {'Coint':>10s} {'Combined':>10s}")
    print("-" * 45)
    for dt in combo_monthly.index:
        lr_v = lr_monthly.get(dt, 0); c_v = c_monthly.get(dt, 0)
        combo = lr_v + c_v
        flag = " <<<" if combo < 0 else ""
        print(f"{dt.strftime('%Y-%m'):<10s} ${lr_v:>9.0f} ${c_v:>9.0f} ${combo:>9.0f}{flag}")

    # Risk check
    if len(combined_full) > 0:
        dr = combined_full / CAPITAL
        cum = combined_full.cumsum()
        peak = (cum+CAPITAL).cummax(); dd = abs(((cum+CAPITAL-peak)/peak).min())
        wd = abs(dr.min())
        print(f"\n--- RISK CHECK (HyroTrader constraints) ---")
        print(f"  Max DD: {dd*100:.1f}% {'PASS' if dd < 0.10 else 'FAIL'} (limit 10%)")
        print(f"  Worst day: {wd*100:.2f}% {'PASS' if wd < 0.05 else 'FAIL'} (limit 5%)")

    print(f"\nRuntime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
