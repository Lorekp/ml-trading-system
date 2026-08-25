#!/usr/bin/env python
"""Detailed stats for optimized combos — max daily DD, quarterly 2025, 2026 YTD.

Usage:
    PYTHONPATH=src python experiments/combo_detailed_stats.py
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
SPREAD_BPS = 0.5
CAP = 250_000

NEEDED_TFS = [("15m","15min"),("1h","1h"),("3h","3h"),("4h","4h"),("24h","1D")]
COMBOS = [
    ("Current: 15m(4)+1h(4)+4h(6)",          [("15m",4),("1h",4),("4h",6)]),
    ("New trio: 15m(2)+3h(16)+24h(12)",       [("15m",2),("3h",16),("24h",12)]),
    ("Full5: 15m(2)+1h(4)+3h(16)+4h(6)+24h(12)", [("15m",2),("1h",4),("3h",16),("4h",6),("24h",12)]),
    ("Quad: 15m(2)+1h(4)+3h(16)+4h(6)",      [("15m",2),("1h",4),("3h",16),("4h",6)]),
    ("Quad-B: 15m(2)+3h(16)+4h(6)+24h(12)",  [("15m",2),("3h",16),("4h",6),("24h",12)]),
]


def get_vip_tier(equity, volume_30d):
    result = VIP_TIERS[0]
    for tier in VIP_TIERS:
        if equity >= tier[1] or volume_30d >= tier[2]:
            result = tier
    return result[0], result[3], result[4]


def _gen_one(args):
    name, sym, tf_resample = args
    warnings.filterwarnings("ignore")
    config = SystemConfig(
        model=ModelConfig(C=0.10, penalty="elasticnet", solver="saga", max_iter=1000,
                          seed=42, fwd_bars=4, signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=list(OHLCV_FEATURES), winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS], timeframe=tf_resample,
                        regime_timeframe="4h"),
        assets=ASSETS)
    try:
        ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
    except FileNotFoundError:
        return None
    ohlcv_tf = resample_ohlcv(ohlcv_1m, tf_resample)
    ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
    feat = compute_features(ohlcv_tf, config.features)
    wf = WalkForwardModel(config, force_monthly_retrain=True)
    sigs = wf.generate_signals_batch(ohlcv_tf, feat, name)
    if len(sigs) == 0: return None
    sigs = apply_regime_filter(sigs, ohlcv_4h, config.regime_filter)
    return sigs if len(sigs) > 0 else None


def generate_signals(label, tf_resample):
    tasks = [(n, s, tf_resample) for n, s in ASSETS.items()]
    all_sigs = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_gen_one, t): t for t in tasks}
        for f in as_completed(futs):
            r = f.result()
            if r is not None and len(r) > 0: all_sigs.append(r)
    if not all_sigs: return None
    merged = pd.concat(all_sigs, ignore_index=True)
    merged["strategy"] = label; merged["hold_bars"] = 4
    if "confidence" in merged.columns:
        merged = merged[merged["confidence"] >= 0.08].reset_index(drop=True)
    return merged if len(merged) > 0 else None


def recompute_returns(signals, tf_resample, hold_bars):
    if hold_bars == 4: return signals.copy()
    assets_in = signals["asset"].unique()
    close_cache = {}
    for a in assets_in:
        sym = ASSETS.get(a)
        if not sym: continue
        try:
            ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
            c = resample_ohlcv(ohlcv_1m, tf_resample)["close"]
            if c.index.tz is not None: c.index = c.index.tz_localize(None)
            close_cache[a] = c; del ohlcv_1m
        except: continue
    sig_ts = pd.to_datetime(signals["timestamp"])
    if sig_ts.dt.tz is not None: sig_ts = sig_ts.dt.tz_localize(None)
    sigs_out = signals.copy()
    new_fwd = np.full(len(signals), np.nan)
    for a, close in close_cache.items():
        mask = (signals["asset"] == a).values
        positions = np.where(mask)[0]
        idx = close.index.searchsorted(sig_ts.iloc[positions].values)
        for k, pos in enumerate(positions):
            bi, ei = idx[k], idx[k] + hold_bars
            if 0 <= bi and ei < len(close):
                new_fwd[pos] = (close.values[ei] - close.values[bi]) / close.values[bi]
    sigs_out["fwd_return"] = new_fwd; sigs_out["hold_bars"] = hold_bars
    return sigs_out.dropna(subset=["fwd_return"]).reset_index(drop=True)


def _compute_vol_one(args):
    name, sym = args
    warnings.filterwarnings("ignore")
    try:
        ohlcv = resample_ohlcv(load_binance_1m(sym, DATA_DIRS), "1D")["close"]
        if ohlcv.index.tz is not None: ohlcv.index = ohlcv.index.tz_localize(None)
        return name, ohlcv.pct_change().rolling(20).std()
    except: return name, None


def compute_vol_parallel():
    vol = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for name, vs in pool.map(_compute_vol_one, list(ASSETS.items())):
            if vs is not None: vol[name] = vs
    return vol


def apply_vol(signals, vol_series):
    out = signals.copy()
    if "sizing_mult" not in out.columns: out["sizing_mult"] = 1.0
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
    for i in range(len(out)):
        asset = out.iloc[i].get("asset", "")
        if asset not in vol_series: continue
        vs = vol_series[asset]
        idx = vs.index.searchsorted(ts.iloc[i])
        if 0 < idx < len(vs):
            v = vs.iloc[idx - 1]
            if pd.notna(v) and v > 0:
                out.iat[i, out.columns.get_loc("sizing_mult")] *= np.clip(0.025 / v, 0.3, 1.5)
    return out


def run_sim(signals_prepared, starting_capital, max_per_strategy=0.50):
    port = PortfolioConfig(
        starting_capital=starting_capital, max_dd=0.10, delever_start=0.05,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
        max_per_strategy=max_per_strategy, base_sizing_pct=0.045, max_signal_mult=2.0,
        sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)
    merged = signals_prepared.copy()
    for c in ["regime_exposure_mult","sizing_mult"]:
        if c not in merged.columns: merged[c] = 1.0
        merged[c] = merged[c].fillna(1.0)
    merged["probability"] = merged["probability"].astype(float)
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    sim = simulate_portfolio(merged, port)
    if sim is None: return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None: return None
    if dpnl.index.tz is not None: dpnl = dpnl.copy(); dpnl.index = dpnl.index.tz_localize(None)

    maker_fill_pct = 0.90
    active = ~sim["skipped"]; sizes = sim["sizes"]
    ts_arr = pd.to_datetime(merged["timestamp"].values)
    if ts_arr.tz is not None: ts_arr = ts_arr.tz_localize(None)
    dates_arr = ts_arr.date
    daily_vol = {}
    for i in range(len(merged)):
        if active[i]: daily_vol[dates_arr[i]] = daily_vol.get(dates_arr[i], 0) + abs(sizes[i]) * 2
    vol_s = pd.Series(daily_vol).sort_index()
    vol_30d = vol_s.rolling(30, min_periods=1).sum()
    eq_curve = starting_capital + dpnl.cumsum()
    dpnl_adj = dpnl.copy()
    for d in dpnl.index:
        d_date = d.date()
        eq = eq_curve.get(d, starting_capital)
        v30 = vol_30d.get(d_date, 0)
        _, maker_bps, taker_bps = get_vip_tier(eq, v30)
        rt_bps = maker_fill_pct*2*maker_bps + (1-maker_fill_pct)*2*taker_bps + SPREAD_BPS
        extra_bps = rt_bps - 1.0
        n_day = sum(1 for i in range(len(merged)) if active[i] and dates_arr[i] == d_date)
        avg_sz = daily_vol.get(d_date, 0) / max(n_day * 2, 1)
        dpnl_adj[d] -= n_day * avg_sz * extra_bps / 10000
    return dpnl_adj


def compute_stats(dpnl_adj, starting_capital):
    """Compute full stats from daily PnL series."""
    dr = dpnl_adj / starting_capital
    if dr.std() == 0 or len(dr) < 30: return None

    # Equity curve
    eq = starting_capital + dpnl_adj.cumsum()

    # Standard metrics
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0
    eq_adj = (1 + dr).cumprod()
    dd_series = (eq_adj.cummax() - eq_adj) / eq_adj.cummax()
    dd = float(dd_series.max() * 100)
    calmar = float(dr.mean() * 365.25 / (dd / 100)) if dd > 0 else 0
    pnl = float(dpnl_adj.sum())
    pf = float(dpnl_adj[dpnl_adj > 0].sum() / abs(dpnl_adj[dpnl_adj < 0].sum())) if (dpnl_adj < 0).any() else 99
    wr = float((dpnl_adj > 0).sum() / len(dpnl_adj))
    monthly = dpnl_adj.resample("MS").sum()
    pos_months = int((monthly > 0).sum())
    tot_months = len(monthly)

    # Max daily DD (worst single day loss as % of equity at start of day)
    daily_dd_pct = dpnl_adj / eq.shift(1).fillna(starting_capital) * 100
    max_daily_dd = float(daily_dd_pct.min())  # most negative day
    worst_day = daily_dd_pct.idxmin()
    worst_day_pnl = float(dpnl_adj[worst_day])

    # Top 5 worst days
    worst_5 = daily_dd_pct.nsmallest(5)

    # Top 5 best days
    best_5 = daily_dd_pct.nlargest(5)

    # Consecutive losing days
    losing_streak = 0; max_losing = 0
    for v in dpnl_adj:
        if v < 0: losing_streak += 1; max_losing = max(max_losing, losing_streak)
        else: losing_streak = 0

    # Quarterly stats
    quarterly = dpnl_adj.resample("QS").sum()

    return {
        "sr": sr, "sortino": sortino, "dd": dd, "calmar": calmar,
        "pnl": pnl, "pf": pf, "wr": wr,
        "pnl_mo": pnl / max(tot_months, 1),
        "pos_months": pos_months, "tot_months": tot_months,
        "max_daily_dd_pct": max_daily_dd, "worst_day": worst_day,
        "worst_day_pnl": worst_day_pnl,
        "worst_5": worst_5, "best_5": best_5,
        "max_losing_streak": max_losing,
        "monthly": monthly, "quarterly": quarterly,
        "daily_pnl": dpnl_adj, "eq": eq,
    }


def main():
    T0 = time.time()
    print("=" * 110)
    print("  DETAILED COMBO STATS — Max Daily DD, 2025 Quarterly, 2026 YTD")
    print(f"  30 assets, {len(COMBOS)} combos, MkrTkr60, $250K")
    print("=" * 110)

    vol_series = compute_vol_parallel()
    print(f"  Vol computed ({time.time()-T0:.0f}s)")

    # Signal gen
    print(f"\n  Generating signals...", flush=True)
    signal_cache = {}
    for label, resample in NEEDED_TFS:
        t1 = time.time()
        sigs = generate_signals(label, resample)
        if sigs is not None:
            signal_cache[label] = sigs
            print(f"    {label:<6} {len(sigs):>8,} signals ({time.time()-t1:.0f}s)", flush=True)

    # Hold variants
    print(f"\n  Computing hold variants...", flush=True)
    hold_cache = {}
    needed = set()
    for _, components in COMBOS:
        for tf, h in components: needed.add((tf, h))
    by_tf = {}
    for tf, h in needed: by_tf.setdefault(tf, set()).add(h)
    for tf_label, holds in by_tf.items():
        if tf_label not in signal_cache: continue
        sigs = signal_cache[tf_label]
        resample = dict(NEEDED_TFS)[tf_label]
        for h in holds:
            sigs_h = recompute_returns(sigs, resample, h)
            hold_cache[(tf_label, h)] = apply_vol(sigs_h, vol_series)
    print(f"    Done", flush=True)

    # Run combos
    combo_stats = {}
    for name, components in COMBOS:
        parts = [hold_cache[(tf, h)] for tf, h in components if (tf, h) in hold_cache]
        merged = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
        dpnl = run_sim(merged, CAP, max_per_strategy=0.50)
        if dpnl is not None:
            combo_stats[name] = compute_stats(dpnl, CAP)
            combo_stats[name]["n_signals"] = len(merged)

    # ── Overview ──
    print(f"\n{'='*110}")
    print("  OVERVIEW")
    print(f"{'='*110}")
    print(f"\n  {'Combo':<45} {'Sharpe':>7} {'PnL':>11} {'MaxDD':>6} {'MaxDayDD':>9} {'MaxLoss$':>10} {'Streak':>7} {'$/mo':>8}")
    print("  " + "-" * 105)
    for name, s in combo_stats.items():
        print(f"  {name:<45} {s['sr']:>7.3f} ${s['pnl']:>9,.0f} {s['dd']:>5.1f}%"
              f" {s['max_daily_dd_pct']:>8.2f}% ${s['worst_day_pnl']:>8,.0f} {s['max_losing_streak']:>5}d ${s['pnl_mo']:>7,.0f}")

    # ── Worst days per combo ──
    print(f"\n{'='*110}")
    print("  TOP 5 WORST DAYS")
    print(f"{'='*110}")
    for name, s in combo_stats.items():
        print(f"\n  {name}")
        print(f"    {'Date':<12} {'Daily DD%':>10} {'PnL':>12} {'Equity':>12}")
        print("    " + "-" * 48)
        for dt, pct in s["worst_5"].items():
            pnl_d = float(s["daily_pnl"][dt])
            eq_d = float(s["eq"][dt])
            print(f"    {str(dt.date()):<12} {pct:>9.2f}% ${pnl_d:>10,.0f} ${eq_d:>10,.0f}")

    # ── Best days per combo ──
    print(f"\n{'='*110}")
    print("  TOP 5 BEST DAYS")
    print(f"{'='*110}")
    for name, s in combo_stats.items():
        print(f"\n  {name}")
        print(f"    {'Date':<12} {'Daily Gain%':>12} {'PnL':>12}")
        print("    " + "-" * 38)
        for dt, pct in s["best_5"].items():
            pnl_d = float(s["daily_pnl"][dt])
            print(f"    {str(dt.date()):<12} {pct:>11.2f}% ${pnl_d:>10,.0f}")

    # ── 2025 Quarterly ──
    print(f"\n{'='*110}")
    print("  2025 QUARTERLY BREAKDOWN")
    print(f"{'='*110}")

    q_labels = ["2025-Q1","2025-Q2","2025-Q3","2025-Q4","2025 Total"]
    header = f"  {'':>18}"
    for name in combo_stats:
        short = name.split(":")[0].strip()
        header += f" {short:>14}"
    print(f"\n{header}")
    print("  " + "-" * (18 + 15 * len(combo_stats)))

    for qi, ql in enumerate(q_labels):
        row = f"  {ql:>18}"
        for name, s in combo_stats.items():
            q = s["quarterly"]
            q_2025 = q[(q.index >= "2025-01-01") & (q.index < "2026-01-01")]
            if ql == "2025 Total":
                val = float(q_2025.sum())
            elif qi < len(q_2025):
                val = float(q_2025.iloc[qi])
            else:
                val = 0
            row += f" ${val:>12,.0f}"
        print(row)

    # 2025 annualized Sharpe
    row = f"  {'2025 Sharpe':>18}"
    for name, s in combo_stats.items():
        d = s["daily_pnl"]
        d_2025 = d[(d.index >= "2025-01-01") & (d.index < "2026-01-01")]
        if len(d_2025) > 30:
            dr = d_2025 / CAP
            sr_25 = float(dr.mean() / dr.std() * np.sqrt(365.25))
            row += f" {sr_25:>14.3f}"
        else:
            row += f" {'N/A':>14}"
    print(row)

    # ── 2026 YTD ──
    print(f"\n{'='*110}")
    print("  2026 YTD")
    print(f"{'='*110}")

    header = f"  {'':>18}"
    for name in combo_stats:
        short = name.split(":")[0].strip()
        header += f" {short:>14}"
    print(f"\n{header}")
    print("  " + "-" * (18 + 15 * len(combo_stats)))

    # Monthly 2026
    for name_s, s in combo_stats.items():
        break  # just get one to find months
    m_2026 = s["monthly"][(s["monthly"].index >= "2026-01-01")]
    for m in m_2026.index:
        row = f"  {m.strftime('%Y-%m'):>18}"
        for name, s in combo_stats.items():
            val = s["monthly"].get(m, 0)
            row += f" ${val:>12,.0f}"
        print(row)

    row = f"  {'2026 YTD Total':>18}"
    for name, s in combo_stats.items():
        d = s["daily_pnl"]
        d_2026 = d[(d.index >= "2026-01-01")]
        row += f" ${float(d_2026.sum()):>12,.0f}"
    print(row)

    row = f"  {'2026 Sharpe (ann)':>18}"
    for name, s in combo_stats.items():
        d = s["daily_pnl"]
        d_2026 = d[(d.index >= "2026-01-01")]
        if len(d_2026) > 20:
            dr = d_2026 / CAP
            sr_26 = float(dr.mean() / dr.std() * np.sqrt(365.25))
            row += f" {sr_26:>14.3f}"
        else:
            row += f" {'N/A':>14}"
    print(row)

    row = f"  {'2026 MaxDD':>18}"
    for name, s in combo_stats.items():
        d = s["daily_pnl"]
        d_2026 = d[(d.index >= "2026-01-01")]
        dr = d_2026 / CAP
        eq = (1 + dr).cumprod()
        dd26 = float(((eq.cummax() - eq) / eq.cummax()).max() * 100)
        row += f" {dd26:>13.2f}%"
    print(row)

    row = f"  {'2026 Win Rate':>18}"
    for name, s in combo_stats.items():
        d = s["daily_pnl"]
        d_2026 = d[(d.index >= "2026-01-01")]
        if len(d_2026) > 0:
            wr = (d_2026 > 0).sum() / len(d_2026)
            row += f" {wr:>13.1%}"
        else:
            row += f" {'N/A':>14}"
    print(row)

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
