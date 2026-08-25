#!/usr/bin/env python
"""HL Multi-Level OFI Experiment — 20-level depth features for 5 major assets.

Tests whether Hyperliquid's 20-level depth OFI adds edge over L1-only features.
Data: BTC, ETH, SOL, XRP, BNB — Apr 2023 to Feb 2026 (2.8 years, 16 features).

Variants:
  A: Baseline (OHLCV only, Elastic Net)
  B: +HL OFI Full (16 features)
  C: +HL OFI Top5 (ofi_L10, ofi_weighted, depth_imb_L10, spread_L1, depth_resilience)

Only runs on the 5 HL assets (not all 30) for clean comparison.

Usage:
    PYTHONPATH=src python experiments/hl_ofi_experiment.py
"""
import sys, os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import wilcoxon

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
L2_DIR = Path("data/l2-features")

# Only HL assets
ASSETS = {"BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT","XRP":"XRPUSDT","BNB":"BNBUSDT"}

OHLCV = ["ret_1","ret_3","ret_7","ret_14","bar_range","vol_zscore","rsi_14",
         "stoch_rsi","macd_hist","bb_pct_b","atr_14","dist_swing_high","dist_swing_low"]

HL_OFI_TOP5 = ["ofi_L10", "ofi_weighted", "depth_imb_L10", "spread_L1", "depth_resilience"]

VIP_TIERS = [
    ("VIP0",0,0,2.00,5.50),("VIP1",100_000,10_000_000,1.80,4.00),
    ("VIP2",250_000,25_000_000,1.60,3.75),("VIP3",500_000,50_000_000,1.40,3.50),
    ("VIP4",1_000_000,100_000_000,1.20,3.20),("VIP5",2_000_000,250_000_000,1.00,3.20),
    ("Supreme",5_000_000,500_000_000,0.00,3.00),
]
SPREAD_BPS = 0.5
CAP = 250_000

# Pre-load HL OFI data
_HL_CACHE = {}

def _preload_hl():
    for name, sym in ASSETS.items():
        # HL OFI uses short names (BTC, not BTCUSDT)
        p = L2_DIR / f"{name}_ofi_15m.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            _HL_CACHE[name] = df


def _resample_hl(df, tf):
    agg = {}
    for col in df.columns:
        if "sum" in col or "count" in col: agg[col] = "sum"
        elif "max" in col: agg[col] = "max"
        elif "std" in col or "skew" in col: agg[col] = "last"
        else: agg[col] = "mean"
    return df.resample(tf).agg(agg).dropna(how="all")


def _gen(args):
    name, sym, tf, fwd, C, hl_mode = args
    warnings.filterwarnings("ignore")
    feature_list = list(OHLCV)

    config = SystemConfig(
        model=ModelConfig(C=C, penalty="elasticnet", solver="saga", max_iter=1000, seed=42,
                          fwd_bars=fwd, signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=feature_list, winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS], timeframe=tf, regime_timeframe="4h"),
        assets=ASSETS)
    try:
        ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
    except FileNotFoundError:
        return None
    ohlcv_tf = resample_ohlcv(ohlcv_1m, tf)
    ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
    feat = compute_features(ohlcv_tf, config.features)

    # Merge HL OFI features
    if hl_mode != "none" and name in _HL_CACHE:
        hl = _HL_CACHE[name].copy()
        if hl_mode == "top5":
            hl = hl[[c for c in HL_OFI_TOP5 if c in hl.columns]]
        if tf != "15min":
            hl = _resample_hl(hl, tf)
        # Prefix columns to avoid collision
        hl.columns = [f"hl_{c}" for c in hl.columns]
        if feat.index.tz is not None and hl.index.tz is None:
            hl.index = hl.index.tz_localize(feat.index.tz)
        elif feat.index.tz is None and hl.index.tz is not None:
            hl.index = hl.index.tz_localize(None)
        feat = feat.join(hl, how="left")
        hl_cols = [c for c in feat.columns if c.startswith("hl_")]
        feat[hl_cols] = feat[hl_cols].fillna(0)
        feature_list = list(OHLCV) + hl_cols
        config.features.feature_list = feature_list
        # Restrict to HL overlap
        hl_start = hl.index.min()
        if feat.index.tz is not None and hl_start.tzinfo is None:
            hl_start = hl_start.tz_localize(feat.index.tz)
        elif feat.index.tz is None and hl_start.tzinfo is not None:
            hl_start = hl_start.tz_localize(None)
        ohlcv_tf = ohlcv_tf[ohlcv_tf.index >= hl_start]
        feat = feat[feat.index >= hl_start]

    wf = WalkForwardModel(config, force_monthly_retrain=True)
    sigs = wf.generate_signals_batch(ohlcv_tf, feat, name)
    if len(sigs) == 0: return None
    sigs = apply_regime_filter(sigs, ohlcv_4h, config.regime_filter)
    return sigs if len(sigs) > 0 else None


def get_vip_tier(equity, volume_30d):
    result = VIP_TIERS[0]
    for tier in VIP_TIERS:
        if equity >= tier[1] or volume_30d >= tier[2]: result = tier
    return result[0], result[3], result[4]


def _compute_vol_one(args):
    name, sym = args
    warnings.filterwarnings("ignore")
    try:
        ohlcv = resample_ohlcv(load_binance_1m(sym, DATA_DIRS), "1D")["close"]
        if ohlcv.index.tz is not None: ohlcv.index = ohlcv.index.tz_localize(None)
        return name, ohlcv.pct_change().rolling(20).std()
    except: return name, None


def apply_vol(signals, vol_series):
    out = signals.copy()
    if "sizing_mult" not in out.columns: out["sizing_mult"] = 1.0
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
    for i in range(len(out)):
        asset = out.iloc[i].get("asset", "")
        if asset not in vol_series: continue
        vs = vol_series[asset]
        t = ts.iloc[i]
        idx = vs.index.searchsorted(t)
        if 0 < idx < len(vs):
            v = vs.iloc[idx - 1]
            if pd.notna(v) and v > 0:
                out.iat[i, out.columns.get_loc("sizing_mult")] *= np.clip(0.025 / v, 0.3, 1.5)
    return out


def run_sim(signals_prepared, starting_capital):
    port = PortfolioConfig(
        starting_capital=starting_capital, max_dd=0.10, delever_start=0.05,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.15,  # higher per-asset for 5 assets
        max_per_strategy=0.50, base_sizing_pct=0.045, max_signal_mult=2.0,
        sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)

    merged = signals_prepared.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in merged.columns: merged[c] = 1.0
        merged[c] = merged[c].fillna(1.0)
    merged["probability"] = merged["probability"].astype(float)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    sim = simulate_portfolio(merged, port)
    if sim is None: return None, None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None: return None, None
    if dpnl.index.tz is not None:
        dpnl = dpnl.copy(); dpnl.index = dpnl.index.tz_localize(None)

    # VIP fees
    active = ~sim["skipped"]
    sizes = sim["sizes"]
    ts_arr = pd.to_datetime(merged["timestamp"].values)
    if ts_arr.tz is not None: ts_arr = ts_arr.tz_localize(None)
    dates_arr = ts_arr.date
    daily_vol = {}
    for i in range(len(merged)):
        if active[i]:
            d = dates_arr[i]; daily_vol[d] = daily_vol.get(d, 0) + abs(sizes[i]) * 2
    vol_s = pd.Series(daily_vol).sort_index()
    vol_30d = vol_s.rolling(30, min_periods=1).sum()
    eq_curve = starting_capital + dpnl.cumsum()
    dpnl_adj = dpnl.copy()
    for d in dpnl.index:
        d_date = d.date()
        eq = eq_curve.get(d, starting_capital)
        v30 = vol_30d.get(d_date, 0)
        _, maker_bps, _ = get_vip_tier(eq, v30)
        extra_bps = 2 * maker_bps + SPREAD_BPS - 1.0
        n_day = sum(1 for i in range(len(merged)) if active[i] and dates_arr[i] == d_date)
        avg_sz = daily_vol.get(d_date, 0) / max(n_day * 2, 1)
        dpnl_adj[d] -= n_day * avg_sz * extra_bps / 10000

    dr = dpnl_adj / starting_capital
    if dr.std() == 0 or len(dr) < 30: return None, None
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0
    eq_adj = (1 + dr).cumprod()
    dd = float(((eq_adj.cummax() - eq_adj) / eq_adj.cummax()).max() * 100)
    calmar = float(dr.mean() * 365.25 / (dd / 100)) if dd > 0 else 0
    pnl = float(dpnl_adj.sum())
    monthly = dpnl_adj.resample("MS").sum()
    pf = float(dpnl_adj[dpnl_adj > 0].sum() / abs(dpnl_adj[dpnl_adj < 0].sum())) if (dpnl_adj < 0).any() else 99
    pos_m = int((monthly > 0).sum())
    tot_m = len(monthly)
    rng = np.random.RandomState(42)
    bsr = np.array([np.mean(rng.choice(dr.values, len(dr))) / max(np.std(rng.choice(dr.values, len(dr))), 1e-10) * np.sqrt(365.25) for _ in range(3000)])
    ci = (float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5)))
    return dpnl_adj, {"sr": sr, "ci": ci, "sortino": sortino, "dd": dd, "calmar": calmar,
        "pnl": pnl, "pf": pf, "pos_months": f"{pos_m}/{tot_m}", "n_trades": sim["n_trades"], "monthly": monthly}


def main():
    T0 = time.time()
    print("=" * 80)
    print("  HL MULTI-LEVEL OFI EXPERIMENT")
    print("  5 assets (BTC/ETH/SOL/XRP/BNB) | 16 features | 2.8 years")
    print("=" * 80)

    _preload_hl()
    print(f"\n  HL OFI cached: {len(_HL_CACHE)} assets", flush=True)
    for name, df in _HL_CACHE.items():
        print(f"    {name}: {df.index.min().date()} to {df.index.max().date()}, {len(df):,} bars")

    vol_series = {}
    with ProcessPoolExecutor(max_workers=5) as pool:
        for name, vs in pool.map(_compute_vol_one, list(ASSETS.items())):
            if vs is not None: vol_series[name] = vs

    VARIANTS = [
        ("A: Baseline", "none"),
        ("B: +HL OFI Full (16)", "full"),
        ("C: +HL OFI Top5", "top5"),
    ]

    all_results = {}
    for vname, hl_mode in VARIANTS:
        t0 = time.time()
        print(f"\n  Generating [{vname}]...", flush=True)
        strat_sigs = {}
        for sname, tf, fwd, C in [("15m","15min",4,0.10),("1h","1h",4,0.10),("4h","4h",6,0.20)]:
            tasks = [(n, s, tf, fwd, C, hl_mode) for n, s in ASSETS.items()]
            all_sigs = []
            with ProcessPoolExecutor(max_workers=5) as pool:
                futs = {pool.submit(_gen, t): t for t in tasks}
                for f in as_completed(futs):
                    r = f.result()
                    if r is not None and len(r) > 0: all_sigs.append(r)
            if all_sigs:
                m = pd.concat(all_sigs, ignore_index=True)
                m["strategy"] = sname; m["hold_bars"] = fwd
                if "confidence" in m.columns:
                    m = m[m["confidence"] >= 0.08].reset_index(drop=True)
                strat_sigs[sname] = m
                print(f"    {sname}: {len(m):,}", flush=True)

        combined = pd.concat(list(strat_sigs.values()), ignore_index=True) if strat_sigs else pd.DataFrame()
        if len(combined) == 0: print(f"    No signals!"); continue
        combined = apply_vol(combined, vol_series)
        print(f"    Total: {len(combined):,} ({time.time()-t0:.0f}s)", flush=True)

        dpnl, stats = run_sim(combined, CAP)
        if stats:
            all_results[vname] = {"dpnl": dpnl, "stats": stats}
            print(f"    Sharpe={stats['sr']:.3f}, PnL=${stats['pnl']:,.0f}", flush=True)

    # Report
    if len(all_results) < 2: print("\nNot enough results"); return
    vnames = list(all_results.keys())
    W = 18
    print(f"\n{'='*80}")
    print(f"  RESULTS -- 5 HL assets, $250K")
    print(f"{'='*80}")
    print(f"\n  {'Metric':<18}" + "".join(f" {v:>{W}}" for v in vnames))
    print("  " + "-" * (18 + (W+1) * len(vnames)))

    for metric, label, fmt in [
        ("sr","Sharpe",lambda v:f"{v:.3f}"),("sortino","Sortino",lambda v:f"{v:.3f}"),
        ("pnl","Total PnL",lambda v:f"${v:,.0f}"),("dd","Max DD %",lambda v:f"{v:.2f}%"),
        ("calmar","Calmar",lambda v:f"{v:.2f}"),("pf","Profit Factor",lambda v:f"{v:.2f}"),
        ("pos_months","Pos Months",lambda v:v),("n_trades","Trades",lambda v:f"{v:,}"),
    ]:
        row = f"  {label:<18}"
        for vn in vnames:
            row += f" {fmt(all_results[vn]['stats'][metric]):>{W}}"
        print(row)

    bl = all_results.get("A: Baseline")
    if bl:
        print(f"\n  vs Baseline (monthly Wilcoxon):")
        bl_m = bl["stats"]["monthly"]
        for vn in vnames:
            if "Baseline" in vn: continue
            v_m = all_results[vn]["stats"]["monthly"]
            common = bl_m.index.intersection(v_m.index)
            diffs = (v_m[common] - bl_m[common]).values
            wins = int((diffs > 0).sum()); losses = int((diffs < 0).sum())
            if len(diffs) >= 10 and np.any(diffs != 0):
                _, p = wilcoxon(diffs)
                print(f"    {vn}: W={wins} L={losses}/{len(common)}  p={p:.4f} ({'SIG' if p < 0.05 else 'ns'})")
            else:
                print(f"    {vn}: W={wins} L={losses}/{len(common)}")

        print(f"\n  Year-by-Year Sharpe:")
        years = sorted(bl["dpnl"].index.year.unique())
        print(f"  {'Year':<6}" + "".join(f" {v[:12]:>12}" for v in vnames))
        for yr in years:
            row = f"  {yr:<6}"
            for vn in vnames:
                d = all_results[vn]["dpnl"]
                y = d[d.index.year == yr] / CAP
                sr = float(y.mean() / y.std() * np.sqrt(365.25)) if y.std() > 0 and len(y) > 10 else 0
                row += f" {sr:>12.3f}"
            print(row)

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
