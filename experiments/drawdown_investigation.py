#!/usr/bin/env python
"""Drawdown Investigation — Analyze what causes large drawdowns.

For each combo, identifies drawdown periods and breaks down:
  - Which assets contributed most to losses
  - Which timeframes contributed most
  - What was happening in the market (BTC/ETH returns, volatility)
  - Whether regime filter was on/off
  - Signal direction (long/short) during losses

Usage:
    PYTHONPATH=src python experiments/drawdown_investigation.py
"""
import sys, os, time, warnings, gc
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.model import WalkForwardModel
from signal_engine.filters import apply_regime_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.config import *

DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
ASSETS = {
    "BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT","XRP":"XRPUSDT",
    "BNB":"BNBUSDT","ADA":"ADAUSDT","AVAX":"AVAXUSDT","DOT":"DOTUSDT",
    "LINK":"LINKUSDT","NEAR":"NEARUSDT","UNI":"UNIUSDT","AAVE":"AAVEUSDT",
    "LTC":"LTCUSDT","BCH":"BCHUSDT","HBAR":"HBARUSDT","TRX":"TRXUSDT",
    "SUI":"SUIUSDT","APT":"APTUSDT","ENJ":"ENJUSDT","SXP":"SXPUSDT",
    "ZEC":"ZECUSDT","XLM":"XLMUSDT","FET":"FETUSDT","WLD":"WLDUSDT",
    "ANKR":"ANKRUSDT","TAO":"TAOUSDT","ENA":"ENAUSDT",
    "FTM":"FTMUSDT","FIL":"FILUSDT","BNX":"BNXUSDT",
}
OHLCV_FEATURES = [
    "ret_1","ret_3","ret_7","ret_14","bar_range","vol_zscore","rsi_14",
    "stoch_rsi","macd_hist","bb_pct_b","atr_14","dist_swing_high","dist_swing_low"]
VIP_TIERS = [
    ("VIP0",0,0,2.00,5.50),("VIP1",100_000,10_000_000,1.80,4.00),
    ("VIP2",250_000,25_000_000,1.60,3.75),("VIP3",500_000,50_000_000,1.40,3.50),
    ("VIP4",1_000_000,100_000_000,1.20,3.20),("VIP5",2_000_000,250_000_000,1.00,3.20),
    ("Supreme",5_000_000,500_000_000,0.00,3.00),
]
SPREAD_BPS = 0.5; CAP = 250_000

NEEDED_TFS = [("15m","15min"),("1h","1h"),("3h","3h"),("4h","4h"),("24h","1D")]
# Focus on 3 key combos
COMBOS = [
    ("Current",    [("15m",4),("1h",4),("4h",6)]),
    ("New trio",   [("15m",2),("3h",16),("24h",12)]),
    ("Full5",      [("15m",2),("1h",4),("3h",16),("4h",6),("24h",12)]),
]


def get_vip_tier(eq, v30):
    r = VIP_TIERS[0]
    for t in VIP_TIERS:
        if eq >= t[1] or v30 >= t[2]: r = t
    return r[0], r[3], r[4]


def _gen_one(args):
    name, sym, tf_resample = args
    warnings.filterwarnings("ignore")
    config = SystemConfig(
        model=ModelConfig(C=0.10,penalty="elasticnet",solver="saga",max_iter=1000,
                          seed=42,fwd_bars=4,signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=list(OHLCV_FEATURES),winsor_lo=0.01,winsor_hi=0.99,
                               vol_filter_quantile=0.50,vol_zscore_window=30,swing_lookback=20),
        regime_filter=RegimeFilterConfig(enabled=True,timeframe="4h",ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile",threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS],timeframe=tf_resample,regime_timeframe="4h"),
        assets=ASSETS)
    try: ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
    except FileNotFoundError: return None
    ohlcv_tf = resample_ohlcv(ohlcv_1m, tf_resample)
    ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
    feat = compute_features(ohlcv_tf, config.features)
    wf = WalkForwardModel(config, force_monthly_retrain=True)
    sigs = wf.generate_signals_batch(ohlcv_tf, feat, name)
    if len(sigs) == 0: return None
    sigs = apply_regime_filter(sigs, ohlcv_4h, config.regime_filter)
    return sigs if len(sigs) > 0 else None


def generate_signals(label, tf_resample):
    tasks = [(n,s,tf_resample) for n,s in ASSETS.items()]
    all_sigs = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_gen_one,t): t for t in tasks}
        for f in as_completed(futs):
            r = f.result()
            if r is not None and len(r) > 0: all_sigs.append(r)
    if not all_sigs: return None
    m = pd.concat(all_sigs, ignore_index=True)
    m["strategy"] = label; m["hold_bars"] = 4
    if "confidence" in m.columns: m = m[m["confidence"]>=0.08].reset_index(drop=True)
    return m if len(m) > 0 else None


def recompute_returns(signals, tf_resample, hold_bars):
    if hold_bars == 4: return signals.copy()
    close_cache = {}
    for a in signals["asset"].unique():
        sym = ASSETS.get(a)
        if not sym: continue
        try:
            c = resample_ohlcv(load_binance_1m(sym,DATA_DIRS),tf_resample)["close"]
            if c.index.tz is not None: c.index = c.index.tz_localize(None)
            close_cache[a] = c
        except: continue
    sig_ts = pd.to_datetime(signals["timestamp"])
    if sig_ts.dt.tz is not None: sig_ts = sig_ts.dt.tz_localize(None)
    out = signals.copy(); fwd = np.full(len(signals), np.nan)
    for a, close in close_cache.items():
        mask = (signals["asset"]==a).values; pos = np.where(mask)[0]
        idx = close.index.searchsorted(sig_ts.iloc[pos].values)
        for k, p in enumerate(pos):
            bi,ei = idx[k], idx[k]+hold_bars
            if 0<=bi and ei<len(close):
                fwd[p] = (close.values[ei]-close.values[bi])/close.values[bi]
    out["fwd_return"]=fwd; out["hold_bars"]=hold_bars
    return out.dropna(subset=["fwd_return"]).reset_index(drop=True)


def _compute_vol_one(args):
    name, sym = args
    warnings.filterwarnings("ignore")
    try:
        o = resample_ohlcv(load_binance_1m(sym,DATA_DIRS),"1D")["close"]
        if o.index.tz is not None: o.index = o.index.tz_localize(None)
        return name, o.pct_change().rolling(20).std()
    except: return name, None


def apply_vol(signals, vol_series):
    out = signals.copy()
    if "sizing_mult" not in out.columns: out["sizing_mult"]=1.0
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
    for i in range(len(out)):
        a = out.iloc[i].get("asset","")
        if a not in vol_series: continue
        vs = vol_series[a]; idx = vs.index.searchsorted(ts.iloc[i])
        if 0<idx<len(vs):
            v = vs.iloc[idx-1]
            if pd.notna(v) and v>0:
                out.iat[i,out.columns.get_loc("sizing_mult")]*=np.clip(0.025/v,0.3,1.5)
    return out


def run_sim_detailed(signals_prepared, starting_capital):
    """Run sim and return per-trade PnL details + daily PnL."""
    port = PortfolioConfig(
        starting_capital=starting_capital,max_dd=0.10,delever_start=0.05,
        slippage=0.0001,max_exposure=0.90,max_per_asset=0.06,
        max_per_strategy=0.50,base_sizing_pct=0.045,max_signal_mult=2.0,
        sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True,regime_exposure_floor=0.50)
    merged = signals_prepared.copy()
    for c in ["regime_exposure_mult","sizing_mult"]:
        if c not in merged.columns: merged[c]=1.0
        merged[c]=merged[c].fillna(1.0)
    merged["probability"]=merged["probability"].astype(float)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    sim = simulate_portfolio(merged, port)
    if sim is None: return None, None, None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None: return None, None, None
    if dpnl.index.tz is not None: dpnl=dpnl.copy(); dpnl.index=dpnl.index.tz_localize(None)

    # VIP fee overlay
    mfp = 0.90; active = ~sim["skipped"]; sizes = sim["sizes"]
    ts_arr = pd.to_datetime(merged["timestamp"].values)
    if ts_arr.tz is not None: ts_arr = ts_arr.tz_localize(None)
    dates_arr = ts_arr.date
    daily_vol = {}
    for i in range(len(merged)):
        if active[i]: daily_vol[dates_arr[i]]=daily_vol.get(dates_arr[i],0)+abs(sizes[i])*2
    vol_s=pd.Series(daily_vol).sort_index()
    vol_30d=vol_s.rolling(30,min_periods=1).sum()
    eq_curve=starting_capital+dpnl.cumsum()
    dpnl_adj=dpnl.copy()
    for d in dpnl.index:
        dd=d.date(); eq=eq_curve.get(d,starting_capital); v30=vol_30d.get(dd,0)
        _,mbps,tbps=get_vip_tier(eq,v30)
        rt=mfp*2*mbps+(1-mfp)*2*tbps+SPREAD_BPS; extra=rt-1.0
        nd=sum(1 for i in range(len(merged)) if active[i] and dates_arr[i]==dd)
        avg=daily_vol.get(dd,0)/max(nd*2,1)
        dpnl_adj[d]-=nd*avg*extra/10000

    # Build per-trade detail
    trade_details = merged.copy()
    trade_details["active"] = active
    trade_details["size_usd"] = sizes
    trade_details["trade_date"] = dates_arr
    # Compute per-trade PnL (direction * fwd_return * size - costs)
    trade_details["gross_pnl"] = trade_details["direction"] * trade_details["fwd_return"] * np.abs(sizes)
    trade_details["trade_pnl"] = trade_details["gross_pnl"] - np.abs(sizes) * 0.0001  # approx slippage

    return dpnl_adj, trade_details, eq_curve


def main():
    T0 = time.time()
    print("="*110)
    print("  DRAWDOWN INVESTIGATION")
    print("="*110)

    # Market context: load BTC and ETH daily returns
    print("\n  Loading market context...", flush=True)
    btc_1d = resample_ohlcv(load_binance_1m("BTCUSDT", DATA_DIRS), "1D")
    eth_1d = resample_ohlcv(load_binance_1m("ETHUSDT", DATA_DIRS), "1D")
    if btc_1d.index.tz is not None: btc_1d.index = btc_1d.index.tz_localize(None)
    if eth_1d.index.tz is not None: eth_1d.index = eth_1d.index.tz_localize(None)
    btc_ret = btc_1d["close"].pct_change()
    eth_ret = eth_1d["close"].pct_change()
    btc_vol = btc_ret.rolling(20).std()

    # Signal gen
    vol_series = {n:v for n,v in (ProcessPoolExecutor(max_workers=8).map(_compute_vol_one, list(ASSETS.items()))) if v is not None}
    print("  Generating signals...", flush=True)
    signal_cache = {}
    for label, resample in NEEDED_TFS:
        t1=time.time()
        sigs = generate_signals(label, resample)
        if sigs is not None:
            signal_cache[label]=sigs
            print(f"    {label:<6} {len(sigs):>8,} ({time.time()-t1:.0f}s)", flush=True)

    # Hold variants
    hold_cache = {}
    needed = set()
    for _,components in COMBOS:
        for tf,h in components: needed.add((tf,h))
    for tf,h in needed:
        if tf not in signal_cache: continue
        resample = dict(NEEDED_TFS)[tf]
        sigs_h = recompute_returns(signal_cache[tf], resample, h)
        hold_cache[(tf,h)] = apply_vol(sigs_h, vol_series)
    print("  Hold variants ready.\n", flush=True)

    # Analyze each combo
    for combo_name, components in COMBOS:
        print(f"\n{'='*110}")
        print(f"  {combo_name}")
        print(f"{'='*110}")

        parts = [hold_cache[(tf,h)] for tf,h in components if (tf,h) in hold_cache]
        merged = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        dpnl_adj, trades, eq_curve = run_sim_detailed(merged, CAP)
        if dpnl_adj is None: print("  FAILED"); continue

        eq = CAP + dpnl_adj.cumsum()
        dr = dpnl_adj / CAP
        eq_adj = (1+dr).cumprod()
        dd_series = (eq_adj.cummax() - eq_adj) / eq_adj.cummax()

        # Find top 5 drawdown PERIODS (not just single days)
        # Find peaks and troughs in equity curve
        running_max = eq_adj.cummax()
        in_dd = dd_series > 0.001  # at least 0.1% drawdown

        # Identify distinct drawdown episodes
        episodes = []
        start = None
        for i, (dt, is_dd) in enumerate(in_dd.items()):
            if is_dd and start is None:
                start = dt
            elif not is_dd and start is not None:
                end = in_dd.index[i-1]
                dd_max = float(dd_series[start:end].max() * 100)
                trough_dt = dd_series[start:end].idxmax()
                pnl_loss = float(dpnl_adj[start:end].sum())
                n_days = (end - start).days
                episodes.append((start, end, trough_dt, dd_max, pnl_loss, n_days))
                start = None
        if start is not None:
            end = in_dd.index[-1]
            dd_max = float(dd_series[start:end].max()*100)
            trough_dt = dd_series[start:end].idxmax()
            pnl_loss = float(dpnl_adj[start:end].sum())
            n_days = (end-start).days
            episodes.append((start, end, trough_dt, dd_max, pnl_loss, n_days))

        episodes.sort(key=lambda x: x[3], reverse=True)

        # Top 5 drawdown episodes
        print(f"\n  TOP 5 DRAWDOWN EPISODES")
        print(f"  {'#':>3} {'Start':>12} {'Trough':>12} {'End':>12} {'Days':>5} {'MaxDD':>7} {'Loss':>12}"
              f" {'BTC':>8} {'Vol20d':>8}")
        print("  "+"-"*90)

        for rank, (start, end, trough, dd_max, pnl_loss, n_days) in enumerate(episodes[:5], 1):
            btc_r = float(btc_ret[start:end].sum()*100) if start in btc_ret.index or len(btc_ret[start:end])>0 else 0
            vol20 = float(btc_vol.asof(trough)*100) if trough in btc_vol.index or True else 0
            try: vol20 = float(btc_vol[btc_vol.index <= trough].iloc[-1]*100)
            except: vol20 = 0
            print(f"  {rank:>3} {str(start.date()):>12} {str(trough.date()):>12} {str(end.date()):>12}"
                  f" {n_days:>5} {dd_max:>6.2f}% ${pnl_loss:>10,.0f} {btc_r:>+7.1f}% {vol20:>7.1f}%")

        # Deep dive into top 3 episodes
        for rank, (start, end, trough, dd_max, pnl_loss, n_days) in enumerate(episodes[:3], 1):
            window_start = start - pd.Timedelta(days=2)
            window_end = end + pd.Timedelta(days=2)

            print(f"\n  --- Episode {rank}: {start.date()} to {end.date()} (DD={dd_max:.2f}%, ${pnl_loss:,.0f}) ---")

            # Market context
            btc_window = btc_ret[window_start:window_end]
            eth_window = eth_ret[window_start:window_end]
            btc_px = btc_1d["close"][window_start:window_end]
            print(f"\n    Market: BTC {float(btc_window.sum()*100):+.1f}%, ETH {float(eth_window.sum()*100):+.1f}%")
            if len(btc_px) > 0:
                print(f"    BTC price: ${float(btc_px.iloc[0]):,.0f} -> ${float(btc_px.iloc[-1]):,.0f}")

            # Trades active during this period
            active_trades = trades[trades["active"]].copy()
            ts = pd.to_datetime(active_trades["timestamp"])
            if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
            active_trades["ts_clean"] = ts

            # Trades that were entered during the drawdown period
            dd_trades = active_trades[(active_trades["ts_clean"] >= start) &
                                      (active_trades["ts_clean"] <= end)]

            if len(dd_trades) == 0:
                print("    No trades during this period")
                continue

            n_trades = len(dd_trades)
            n_long = (dd_trades["direction"] > 0).sum()
            n_short = (dd_trades["direction"] < 0).sum()
            avg_fwd = float((dd_trades["direction"] * dd_trades["fwd_return"]).mean() * 10000)

            print(f"    Trades: {n_trades} ({n_long} long, {n_short} short)")
            print(f"    Avg directional return: {avg_fwd:+.1f} bps")

            # Loss by asset
            dd_trades_pnl = dd_trades.copy()
            dd_trades_pnl["dir_ret"] = dd_trades_pnl["direction"] * dd_trades_pnl["fwd_return"]
            asset_pnl = dd_trades_pnl.groupby("asset").agg(
                n=("dir_ret", "count"),
                avg_ret=("dir_ret", "mean"),
                total_gross=("gross_pnl", "sum"),
            ).sort_values("total_gross")

            print(f"\n    Loss by asset (worst first):")
            print(f"    {'Asset':<10} {'Trades':>7} {'Avg Ret':>10} {'Gross PnL':>12}")
            print("    "+"-"*42)
            for asset, row in asset_pnl.head(5).iterrows():
                print(f"    {asset:<10} {int(row['n']):>7} {row['avg_ret']*10000:>+9.1f}bp ${row['total_gross']:>10,.0f}")
            if len(asset_pnl) > 5:
                rest = asset_pnl.iloc[5:]
                print(f"    {'(rest)':<10} {int(rest['n'].sum()):>7} {'':>10} ${rest['total_gross'].sum():>10,.0f}")

            # Loss by strategy/TF
            strat_pnl = dd_trades_pnl.groupby("strategy").agg(
                n=("dir_ret", "count"),
                avg_ret=("dir_ret", "mean"),
                total_gross=("gross_pnl", "sum"),
                pct_long=("direction", lambda x: (x>0).mean()),
            ).sort_values("total_gross")

            print(f"\n    Loss by timeframe:")
            print(f"    {'TF':<6} {'Trades':>7} {'Avg Ret':>10} {'Gross PnL':>12} {'%Long':>7}")
            print("    "+"-"*45)
            for strat, row in strat_pnl.iterrows():
                print(f"    {strat:<6} {int(row['n']):>7} {row['avg_ret']*10000:>+9.1f}bp"
                      f" ${row['total_gross']:>10,.0f} {row['pct_long']:>6.0%}")

            # Direction analysis: were we on the wrong side?
            if len(btc_window) > 0:
                btc_dir = "UP" if float(btc_window.sum()) > 0 else "DOWN"
                pct_long = n_long / max(n_trades, 1)
                mismatch = (btc_dir == "DOWN" and pct_long > 0.6) or (btc_dir == "UP" and pct_long < 0.4)
                print(f"\n    Direction: Market went {btc_dir}, we were {pct_long:.0%} long"
                      f" {'** MISMATCH **' if mismatch else '(aligned)'}")

        # ── Pattern analysis across all episodes ──
        print(f"\n  PATTERN ANALYSIS (all {len(episodes)} drawdown episodes)")

        if len(episodes) >= 3:
            # Collect stats across episodes
            ep_btc_rets = []
            ep_vols = []
            ep_durations = []
            ep_dds = []

            for start, end, trough, dd_max, pnl_loss, n_days in episodes:
                br = float(btc_ret[start:end].sum()) if len(btc_ret[start:end]) > 0 else 0
                try: vol20 = float(btc_vol[btc_vol.index <= trough].iloc[-1])
                except: vol20 = 0
                ep_btc_rets.append(br)
                ep_vols.append(vol20)
                ep_durations.append(n_days)
                ep_dds.append(dd_max)

            # Correlation: BTC return vs drawdown severity
            if len(ep_btc_rets) >= 5:
                corr = np.corrcoef(ep_btc_rets[:20], ep_dds[:20])[0,1]
                print(f"    BTC return vs DD severity correlation: {corr:.3f}")

            # High-vol vs low-vol drawdowns
            med_vol = np.median(ep_vols) if ep_vols else 0
            high_vol_eps = [(d, dd) for d, dd, v in zip(ep_durations, ep_dds, ep_vols) if v > med_vol]
            low_vol_eps = [(d, dd) for d, dd, v in zip(ep_durations, ep_dds, ep_vols) if v <= med_vol]

            if high_vol_eps and low_vol_eps:
                avg_dd_hv = np.mean([dd for _, dd in high_vol_eps])
                avg_dd_lv = np.mean([dd for _, dd in low_vol_eps])
                print(f"    Avg DD in high-vol periods: {avg_dd_hv:.2f}% ({len(high_vol_eps)} episodes)")
                print(f"    Avg DD in low-vol periods:  {avg_dd_lv:.2f}% ({len(low_vol_eps)} episodes)")

            # Duration distribution
            print(f"    DD duration: median={np.median(ep_durations):.0f}d, "
                  f"mean={np.mean(ep_durations):.0f}d, max={max(ep_durations)}d")

            # Fraction of time in drawdown
            total_dd_days = sum(ep_durations)
            total_days = (dpnl_adj.index[-1] - dpnl_adj.index[0]).days
            print(f"    Time in drawdown: {total_dd_days}/{total_days} days ({total_dd_days/total_days:.0%})")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
