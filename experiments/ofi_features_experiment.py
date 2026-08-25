#!/usr/bin/env python
"""OFI Microstructure Features Experiment — BERT replacement test.

Tests whether handcrafted OFI/microstructure features add edge on top of
the OHLCV Elastic Net baseline. 5 approaches:
  1. Baseline (13 OHLCV, Elastic Net)
  2. +OFI Core (3 extra: ofi_sum, spread_mean, imbalance_mean)
  3. +Micro Core (5 extra: trade_imbalance, vpin, depth_imb, kyle, amihud)
  4. +Full L2 (~27 extra: all bookTicker + micro + depth features)
  5. OFI Filter (baseline signals filtered by OFI direction agreement)

Usage:
    PYTHONPATH=src python experiments/ofi_features_experiment.py [--smoke]
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

# ============================================================
# Config
# ============================================================
SMOKE = "--smoke" in sys.argv

DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
L2_DIR = Path("data/l2-features")

ALL_ASSETS = {
    "BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT","XRP":"XRPUSDT",
    "BNB":"BNBUSDT","ADA":"ADAUSDT","AVAX":"AVAXUSDT","DOT":"DOTUSDT",
    "LINK":"LINKUSDT","NEAR":"NEARUSDT","UNI":"UNIUSDT","AAVE":"AAVEUSDT",
    "LTC":"LTCUSDT","BCH":"BCHUSDT","HBAR":"HBARUSDT","TRX":"TRXUSDT",
    "SUI":"SUIUSDT","APT":"APTUSDT","ENJ":"ENJUSDT","SXP":"SXPUSDT",
    "ZEC":"ZECUSDT","XLM":"XLMUSDT","FET":"FETUSDT","WLD":"WLDUSDT",
    "ANKR":"ANKRUSDT","TAO":"TAOUSDT","ENA":"ENAUSDT",
    "FTM":"FTMUSDT","FIL":"FILUSDT","BNX":"BNXUSDT",
}
SMOKE_ASSETS = {"BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT"}
ASSETS = SMOKE_ASSETS if SMOKE else ALL_ASSETS

OHLCV = ["ret_1","ret_3","ret_7","ret_14","bar_range","vol_zscore","rsi_14",
         "stoch_rsi","macd_hist","bb_pct_b","atr_14","dist_swing_high","dist_swing_low"]

OFI_CORE = ["l2_ofi_sum", "l2_spread_mean", "l2_imbalance_mean"]
MICRO_CORE = ["ms_trade_imbalance", "ms_vpin", "ms_depth_imb_2pct",
              "ms_kyle_lambda", "ms_amihud"]
# Extended trades micro: 6 features with 4.5 years of coverage (vs 10 months for full L2)
TRADES_MICRO = ["ms_trade_imbalance", "ms_vpin", "ms_amihud",
                "ms_kyle_lambda", "ms_trade_count", "ms_large_trade_frac"]

VIP_TIERS = [
    ("VIP0",    0,           0,             2.00, 5.50),
    ("VIP1",    100_000,     10_000_000,    1.80, 4.00),
    ("VIP2",    250_000,     25_000_000,    1.60, 3.75),
    ("VIP3",    500_000,     50_000_000,    1.40, 3.50),
    ("VIP4",    1_000_000,   100_000_000,   1.20, 3.20),
    ("VIP5",    2_000_000,   250_000_000,   1.00, 3.20),
    ("Supreme", 5_000_000,   500_000_000,   0.00, 3.00),
]
SPREAD_BPS = 0.5
CAP = 250_000

# Pre-load all L2 parquets into memory (fast — ~50MB total)
_L2_CACHE = {}

def _preload_l2():
    """Load all L2 parquet files into memory once.
    Prefers extended trades_micro (2019-2024) over old micro (2023-2024)."""
    for name, sym in ASSETS.items():
        for src, pat in [("bt", f"{sym}_bookTicker_15m.parquet"),
                         ("bd", f"{sym}_bookDepth_15m.parquet")]:
            p = L2_DIR / pat
            if p.exists():
                df = pd.read_parquet(p)
                if df.index.tz is not None: df.index = df.index.tz_localize(None)
                _L2_CACHE[(sym, src)] = df
        # Micro: prefer extended trades_micro (4.5 years) over old micro (10 months)
        ext_p = L2_DIR / f"{sym}_trades_micro_15m.parquet"
        old_p = L2_DIR / f"{sym}_micro_15m.parquet"
        if ext_p.exists():
            df = pd.read_parquet(ext_p)
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            _L2_CACHE[(sym, "mc")] = df
        elif old_p.exists():
            df = pd.read_parquet(old_p)
            if df.index.tz is not None: df.index = df.index.tz_localize(None)
            _L2_CACHE[(sym, "mc")] = df


def _resample_l2(df, tf):
    """Resample 15m L2 to coarser timeframe."""
    agg = {}
    for col in df.columns:
        if "sum" in col or "count" in col: agg[col] = "sum"
        elif "max" in col: agg[col] = "max"
        elif "std" in col or "skew" in col: agg[col] = "last"
        else: agg[col] = "mean"
    return df.resample(tf).agg(agg).dropna(how="all")


def _get_l2(sym, tf, mode):
    """Get L2 features for an asset/timeframe/mode from cache."""
    dfs = []
    if mode in ("ofi_core", "full"):
        bt = _L2_CACHE.get((sym, "bt"))
        if bt is not None:
            dfs.append(bt[OFI_CORE] if mode == "ofi_core" else bt)
    if mode in ("micro_core", "full", "trades_micro"):
        mc = _L2_CACHE.get((sym, "mc"))
        if mc is not None:
            if mode == "micro_core":
                cols = [c for c in MICRO_CORE if c in mc.columns]
                dfs.append(mc[cols])
            elif mode == "trades_micro":
                cols = [c for c in TRADES_MICRO if c in mc.columns]
                dfs.append(mc[cols])
            else:
                dfs.append(mc)
    if mode == "full":
        bd = _L2_CACHE.get((sym, "bd"))
        if bd is not None:
            dfs.append(bd)
    if not dfs:
        return None
    combined = pd.concat(dfs, axis=1)
    combined = combined.loc[:, ~combined.columns.duplicated()]
    if tf != "15min":
        combined = _resample_l2(combined, tf)
    return combined


# ============================================================
# Signal generation (runs in worker processes)
# ============================================================

def _gen(args):
    """Generate signals for one asset/timeframe/l2_mode. Runs in ProcessPool."""
    name, sym, tf, fwd, C, l2_mode = args
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

    # Merge L2 features if requested
    if l2_mode != "none":
        l2 = _get_l2(sym, tf, l2_mode)
        if l2 is not None and not l2.empty:
            if feat.index.tz is not None and l2.index.tz is None:
                l2 = l2.copy(); l2.index = l2.index.tz_localize(feat.index.tz)
            elif feat.index.tz is None and l2.index.tz is not None:
                l2 = l2.copy(); l2.index = l2.index.tz_localize(None)
            feat = feat.join(l2, how="left")
            l2_cols = [c for c in feat.columns if c.startswith("l2_") or c.startswith("ms_")]
            feat[l2_cols] = feat[l2_cols].fillna(0)
            feature_list = list(OHLCV) + l2_cols
            config.features.feature_list = feature_list
            # Restrict to L2 overlap period so model sees real values in training.
            # Without this, L2 features are 0 for 90% of training -> model ignores them.
            l2_start = l2.index.min()
            if feat.index.tz is not None and l2_start.tzinfo is None:
                l2_start = l2_start.tz_localize(feat.index.tz)
            elif feat.index.tz is None and l2_start.tzinfo is not None:
                l2_start = l2_start.tz_localize(None)
            ohlcv_tf = ohlcv_tf[ohlcv_tf.index >= l2_start]
            feat = feat[feat.index >= l2_start]

    wf = WalkForwardModel(config, force_monthly_retrain=True)
    sigs = wf.generate_signals_batch(ohlcv_tf, feat, name)
    if len(sigs) == 0: return None
    sigs = apply_regime_filter(sigs, ohlcv_4h, config.regime_filter)
    return sigs if len(sigs) > 0 else None


# ============================================================
# Portfolio helpers (identical to l1_vs_elasticnet.py)
# ============================================================

def get_vip_tier(equity, volume_30d):
    result = VIP_TIERS[0]
    for tier in VIP_TIERS:
        if equity >= tier[1] or volume_30d >= tier[2]:
            result = tier
    return result[0], result[3], result[4]


def _compute_vol_one(args):
    """Compute vol series for one asset. For parallel execution."""
    name, sym = args
    warnings.filterwarnings("ignore")
    try:
        ohlcv = resample_ohlcv(load_binance_1m(sym, DATA_DIRS), "1D")["close"]
        if ohlcv.index.tz is not None: ohlcv.index = ohlcv.index.tz_localize(None)
        return name, ohlcv.pct_change().rolling(20).std()
    except Exception:
        return name, None


def compute_per_asset_vol_parallel():
    """Parallelize vol computation across all assets."""
    vol_series = {}
    tasks = list(ASSETS.items())
    with ProcessPoolExecutor(max_workers=8) as pool:
        for name, vs in pool.map(_compute_vol_one, tasks):
            if vs is not None:
                vol_series[name] = vs
    return vol_series


def apply_per_asset_vol(signals, vol_series):
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


def apply_depth_scale(signals, ofi_data):
    out = signals.copy()
    if "sizing_mult" not in out.columns: out["sizing_mult"] = 1.0
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
    for i in range(len(out)):
        asset = out.iloc[i].get("asset", "")
        sym = asset + "USDT" if not asset.endswith("USDT") else asset
        if sym not in ofi_data or ofi_data[sym] is None: continue
        ofi_df = ofi_data[sym]
        t = ts.iloc[i]
        idx = ofi_df.index.searchsorted(t)
        if idx == 0 or idx > len(ofi_df): continue
        row = ofi_df.iloc[idx - 1]
        depth_val = row.get("ms_depth_imb_2pct", None) if "ms_depth_imb_2pct" in ofi_df.columns else None
        if depth_val is None or pd.isna(depth_val): continue
        direction = out.iloc[i]["direction"]
        agreement = depth_val * direction
        mult = np.clip(0.3 + agreement * 8, 0.2, 2.0)
        out.iat[i, out.columns.get_loc("sizing_mult")] *= mult
    return out


def apply_ofi_filter(signals, ofi_data):
    """Keep only signals where OFI direction agrees with model direction."""
    out = signals.copy()
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
    keep = np.ones(len(out), dtype=bool)
    for i in range(len(out)):
        asset = out.iloc[i].get("asset", "")
        sym = asset + "USDT" if not asset.endswith("USDT") else asset
        if sym not in ofi_data or ofi_data[sym] is None: continue
        ofi_df = ofi_data[sym]
        t = ts.iloc[i]
        idx = ofi_df.index.searchsorted(t)
        if idx == 0 or idx > len(ofi_df): continue
        ofi_val = ofi_df.iloc[idx - 1].get("ms_trade_imbalance", 0)
        if pd.isna(ofi_val) or ofi_val == 0: continue
        direction = out.iloc[i]["direction"]
        if (ofi_val > 0) != (direction > 0):
            keep[i] = False
    return out[keep].reset_index(drop=True)


def run_sim(signals_prepared, starting_capital):
    """Run simulate_portfolio with VIP fees."""
    port = PortfolioConfig(
        starting_capital=starting_capital, max_dd=0.10, delever_start=0.05,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
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

    # VIP fee adjustment
    active = ~sim["skipped"]
    sizes = sim["sizes"]
    ts_arr = pd.to_datetime(merged["timestamp"].values)
    if ts_arr.tz is not None: ts_arr = ts_arr.tz_localize(None)
    dates_arr = ts_arr.date
    daily_vol = {}
    for i in range(len(merged)):
        if active[i]:
            d = dates_arr[i]
            daily_vol[d] = daily_vol.get(d, 0) + abs(sizes[i]) * 2
    vol_s = pd.Series(daily_vol).sort_index()
    vol_30d = vol_s.rolling(30, min_periods=1).sum()
    eq_curve = starting_capital + dpnl.cumsum()

    dpnl_adj = dpnl.copy()
    for d in dpnl.index:
        d_date = d.date()
        eq = eq_curve.get(d, starting_capital)
        v30 = vol_30d.get(d_date, 0)
        _, maker_bps, _ = get_vip_tier(eq, v30)
        rt_bps = 2 * maker_bps + SPREAD_BPS
        extra_bps = rt_bps - 1.0
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
    bsr = np.empty(3000)
    for b in range(3000):
        s = rng.choice(dr.values, size=len(dr), replace=True)
        bsr[b] = s.mean() / s.std() * np.sqrt(365.25) if s.std() > 0 else 0
    ci = (float(np.percentile(bsr, 2.5)), float(np.percentile(bsr, 97.5)))

    return dpnl_adj, {
        "sr": sr, "ci": ci, "sortino": sortino, "dd": dd, "calmar": calmar,
        "pnl": pnl, "pf": pf, "pos_months": f"{pos_m}/{tot_m}",
        "n_trades": sim["n_trades"], "monthly": monthly,
    }


# ============================================================
# Signal generation (parallelized, identical pattern to l1_vs_elasticnet)
# ============================================================

def generate_all_signals(label, l2_mode):
    print(f"\n  Generating signals [{label}]...", flush=True)
    strat_sigs = {}
    tfs = [("15m", "15min", 4, 0.10)] if SMOKE else [
        ("15m", "15min", 4, 0.10), ("1h", "1h", 4, 0.10), ("4h", "4h", 6, 0.20)]
    for sname, tf, fwd, C in tfs:
        tasks = [(n, s, tf, fwd, C, l2_mode) for n, s in ASSETS.items()]
        all_sigs = []
        with ProcessPoolExecutor(max_workers=8) as pool:
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
            print(f"    {sname}: {len(m):,} signals", flush=True)
    return strat_sigs


# ============================================================
# Main
# ============================================================

def main():
    T0 = time.time()
    mode_str = "SMOKE TEST (3 assets, 15m only)" if SMOKE else "FULL (30 assets, 15m+1h+4h+cs)"
    print("=" * 80)
    print(f"  OFI MICROSTRUCTURE FEATURES EXPERIMENT -- {mode_str}")
    print("=" * 80)

    # Pre-load L2 data into memory
    print("\nPre-loading L2 features...", flush=True)
    t0 = time.time()
    _preload_l2()
    print(f"  Cached {len(_L2_CACHE)} L2 files in {time.time()-t0:.1f}s", flush=True)

    # Parallel vol computation
    print("Computing per-asset vol (parallel)...", flush=True)
    t0 = time.time()
    vol_series = compute_per_asset_vol_parallel()
    print(f"  {len(vol_series)} assets in {time.time()-t0:.1f}s", flush=True)

    # Load depth data for scaling
    ofi_data = {}
    for name, sym in ASSETS.items():
        if (sym, "mc") in _L2_CACHE:
            mc_df = _L2_CACHE[(sym, "mc")]
            ofi_cols = [c for c in ["ms_trade_imbalance", "ms_depth_imb_2pct"] if c in mc_df.columns]
            if ofi_cols:
                ofi_data[sym] = mc_df[ofi_cols]

    # CS momentum
    cs_path = Path("results/signals_cache/cs_momentum.parquet")
    scs = pd.read_parquet(cs_path) if (cs_path.exists() and not SMOKE) else pd.DataFrame()

    # ============================================================
    # Generate signals for each variant
    # ============================================================
    VARIANTS = [
        ("Baseline (EN)", "none"),
        ("+Trades Micro (6)", "trades_micro"),
        ("+OFI Core (3)", "ofi_core"),
        ("+Micro Core (5)", "micro_core"),
        ("+Full L2 (27)", "full"),
    ]

    all_results = {}
    baseline_combined = None

    for vname, l2_mode in VARIANTS:
        t0 = time.time()
        strat_sigs = generate_all_signals(vname, l2_mode)
        parts = list(strat_sigs.values())
        if len(scs) > 0: parts.append(scs)
        combined = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
        if len(combined) == 0:
            print(f"    {vname}: no signals!", flush=True); continue
        combined = apply_per_asset_vol(combined, vol_series)
        combined = apply_depth_scale(combined, ofi_data)
        print(f"    Total: {len(combined):,} signals ({time.time()-t0:.0f}s)", flush=True)

        if "Baseline" in vname:
            baseline_combined = combined.copy()

        dpnl, stats = run_sim(combined, CAP)
        if stats:
            all_results[vname] = {"dpnl": dpnl, "stats": stats}
            print(f"    Sharpe={stats['sr']:.3f}, PnL=${stats['pnl']:,.0f}", flush=True)

    # OFI Filter variant
    if baseline_combined is not None:
        print(f"\n  [OFI Filter] Filtering baseline signals...", flush=True)
        filtered = apply_ofi_filter(baseline_combined, ofi_data)
        pct = len(filtered) / max(len(baseline_combined), 1) * 100
        print(f"    Kept {len(filtered):,}/{len(baseline_combined):,} ({pct:.0f}%)", flush=True)
        dpnl, stats = run_sim(filtered, CAP)
        if stats:
            all_results["OFI Filter"] = {"dpnl": dpnl, "stats": stats}
            print(f"    Sharpe={stats['sr']:.3f}, PnL=${stats['pnl']:,.0f}", flush=True)

    # ============================================================
    # Comparison table
    # ============================================================
    if len(all_results) < 2:
        print("\n  Not enough results to compare"); return

    print("\n" + "=" * 80)
    print(f"  RESULTS -- ${CAP//1000}K Capital")
    print("=" * 80)

    vnames = list(all_results.keys())
    W = 14
    print(f"\n  {'Metric':<18}" + "".join(f" {v[:W]:>{W}}" for v in vnames))
    print("  " + "-" * (18 + (W + 1) * len(vnames)))

    for metric, label, fmt in [
        ("sr",      "Sharpe",        lambda v: f"{v:.3f}"),
        ("sortino", "Sortino",       lambda v: f"{v:.3f}"),
        ("pnl",     "Total PnL",     lambda v: f"${v:,.0f}"),
        ("dd",      "Max DD %",      lambda v: f"{v:.2f}%"),
        ("calmar",  "Calmar",        lambda v: f"{v:.2f}"),
        ("pf",      "Profit Factor", lambda v: f"{v:.2f}"),
        ("pos_months","Pos Months",  lambda v: v),
        ("n_trades","Trades",        lambda v: f"{v:,}"),
    ]:
        row = f"  {label:<18}"
        for vn in vnames:
            row += f" {fmt(all_results[vn]['stats'][metric]):>{W}}"
        print(row)

    # Sharpe CI
    print(f"\n  Sharpe 95% CI:")
    for vn in vnames:
        ci = all_results[vn]["stats"]["ci"]
        print(f"    {vn}: [{ci[0]:.3f}, {ci[1]:.3f}]")

    # Wilcoxon vs baseline
    bl = all_results.get("Baseline (EN)")
    if bl:
        print(f"\n  vs Baseline (monthly Wilcoxon):")
        bl_m = bl["stats"]["monthly"]
        for vn in vnames:
            if vn == "Baseline (EN)": continue
            v_m = all_results[vn]["stats"]["monthly"]
            common = bl_m.index.intersection(v_m.index)
            diffs = (v_m[common] - bl_m[common]).values
            wins = int((diffs > 0).sum())
            losses = int((diffs < 0).sum())
            if len(diffs) >= 10 and np.any(diffs != 0):
                _, p = wilcoxon(diffs)
                sig = "SIG" if p < 0.05 else "ns"
                print(f"    {vn}: W={wins} L={losses}/{len(common)}  p={p:.4f} ({sig})")
            else:
                print(f"    {vn}: W={wins} L={losses}/{len(common)}  (too few for test)")

    # Year-by-year
    if bl:
        print(f"\n  Year-by-Year Sharpe:")
        years = sorted(bl["dpnl"].index.year.unique())
        print(f"  {'Year':<6}" + "".join(f" {v[:10]:>10}" for v in vnames))
        print("  " + "-" * (6 + 11 * len(vnames)))
        for yr in years:
            row = f"  {yr:<6}"
            for vn in vnames:
                d = all_results[vn]["dpnl"]
                y = d[d.index.year == yr] / CAP
                sr = float(y.mean() / y.std() * np.sqrt(365.25)) if y.std() > 0 and len(y) > 10 else 0
                row += f" {sr:>10.3f}"
            print(row)

    # Overlap period (Nov 2023 - Mar 2024)
    print(f"\n  {'='*60}")
    print(f"  OVERLAP PERIOD (Nov 2023 - Mar 2024) -- L2 data dense")
    print(f"  {'='*60}")
    os_start, os_end = pd.Timestamp("2023-11-01"), pd.Timestamp("2024-04-01")
    print(f"  {'Variant':<18} {'Sharpe':>8} {'PnL':>12}")
    print("  " + "-" * 38)
    for vn in vnames:
        d = all_results[vn]["dpnl"]
        ov = d[(d.index >= os_start) & (d.index < os_end)]
        if len(ov) < 10: continue
        dr = ov / CAP
        sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if dr.std() > 0 else 0
        print(f"  {vn:<18} {sr:>8.3f} ${ov.sum():>10,.0f}")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
