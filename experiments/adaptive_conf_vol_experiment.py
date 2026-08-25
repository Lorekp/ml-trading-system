#!/usr/bin/env python
"""Experiment A: Adaptive Confidence Threshold.

Tests adaptive IQR-scaled confidence threshold vs production baseline.
Matches production config EXACTLY (elasticnet, force monthly retrain,
70th pctl + 0.08 absolute filter, per-asset 0.025/vol scaling).

Variants:
  1. Baseline  — production config (WalkForwardModel + 0.08 hard filter)
  2. Adaptive  — IQR-scaled percentile threshold + 0.08 hard floor

Generates RAW signals once (no confidence filter), attaches per-window
metadata (thresholds, IQR), then applies filter variants as post-processing.

Success criteria:
  Sharpe >= baseline + 0.10, no trade count drop > 20%

Usage:
    PYTHONPATH=src python experiments/adaptive_conf_vol_experiment.py [--smoke]
"""
import sys, os, time, warnings, gc
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import wilcoxon

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from afml.bet_sizing import discretize_signal, get_signal
from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.weights import compute_sample_weights
from signal_engine.filters import apply_regime_filter
from signal_engine.portfolio import simulate_portfolio, compute_daily_pnl_series
from signal_engine.config import *

# ============================================================
SMOKE = "--smoke" in sys.argv
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]

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


def get_vip_tier(equity, volume_30d):
    result = VIP_TIERS[0]
    for tier in VIP_TIERS:
        if equity >= tier[1] or volume_30d >= tier[2]:
            result = tier
    return result[0], result[3], result[4]


# ============================================================
# RAW signal generation — no confidence filter, emits per-window metadata
# ============================================================

def _gen_raw(args):
    """Walk-forward signal gen with NO confidence filter.

    Returns ALL signals with per-window metadata columns:
      window_conf_p70  — 70th percentile threshold for baseline comparison
      window_conf_iqr  — IQR of training confidences (for adaptive scaling)
      window_idx       — integer window ID
    """
    name, sym, tf_resample, fwd_bars, C = args
    warnings.filterwarnings("ignore")

    config = SystemConfig(
        model=ModelConfig(C=C, penalty="elasticnet", solver="saga", max_iter=1000,
                          seed=42, fwd_bars=fwd_bars, signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=list(OHLCV_FEATURES), winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS], timeframe=tf_resample,
                        regime_timeframe="4h"),
        assets=ASSETS)

    mcfg = config.model
    fcfg = config.features
    scfg = config.signal
    feature_cols = fcfg.feature_list

    try:
        ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
    except FileNotFoundError:
        return None

    ohlcv_tf = resample_ohlcv(ohlcv_1m, tf_resample)
    ohlcv_4h = resample_ohlcv(ohlcv_1m, "4h")
    features = compute_features(ohlcv_tf, config.features)

    close = ohlcv_tf["close"]
    fwd_ret = close.pct_change(fwd_bars).shift(-fwd_bars)
    vol = close.pct_change().dropna().ewm(span=50, min_periods=20).std()

    start_date = ohlcv_tf.index.min()
    end_date = ohlcv_tf.index.max()
    if (end_date - start_date).days < 180:
        return None

    first_test_month = start_date + pd.DateOffset(months=6)
    all_months = pd.period_range(
        start=first_test_month.to_period("M"),
        end=end_date.to_period("M"), freq="M")
    if len(all_months) == 0:
        return None

    retrain_every = 1  # force monthly retrain (matches production)

    if len(ohlcv_tf) > 1:
        median_gap = ohlcv_tf.index.to_series().diff().dropna().median()
        embargo_td = median_gap * (fwd_bars + 1)
    else:
        embargo_td = pd.Timedelta(hours=2)

    rows = []
    cached_model = None
    cached_scaler = None
    cached_conf_p70 = 0.0
    cached_conf_iqr = 0.0
    last_retrain_idx = -999

    for month_idx, month in enumerate(all_months):
        ms = pd.Timestamp(month.start_time, tz="UTC")
        me = pd.Timestamp(month.end_time, tz="UTC")
        if features.index.tz is None:
            ms = ms.tz_localize(None)
            me = me.tz_localize(None)

        embargo_cutoff = ms - embargo_td
        feat_cols = [c for c in feature_cols if c in features.columns]
        if len(feat_cols) < 8:
            continue

        feat = features[feat_cols].copy()
        train_mask = feat.index < embargo_cutoff
        for col in ["ret_1","ret_3","ret_7","ret_14"]:
            if col in feat.columns:
                tv = feat.loc[train_mask, col]
                if len(tv.dropna()) > 10:
                    lo, hi = tv.quantile(fcfg.winsor_lo), tv.quantile(fcfg.winsor_hi)
                    feat[col] = feat[col].clip(lo, hi)

        fc = feat.dropna()
        y_all = fwd_ret.reindex(fc.index).dropna()
        valid = fc.index.intersection(y_all.index)
        tr = valid[valid < embargo_cutoff]
        te = valid[(valid >= ms) & (valid <= me)]
        if len(tr) < 200 or len(te) < 10:
            continue

        # Vol filter on test set
        if vol is not None and len(vol) > 0:
            tv = vol.loc[vol.index.intersection(tr)].dropna()
            if len(tv) < 20:
                continue
            vt = tv.quantile(fcfg.vol_filter_quantile)
            vol_aligned = vol.reindex(te)
            hv = (vol_aligned > vt).fillna(False)
            te = te[hv.reindex(te).fillna(False).values]
        if len(te) < 5:
            continue

        X_train = fc.loc[tr]
        y_train = (y_all.loc[tr] > 0).astype(int)
        X_test = fc.loc[te]

        need_retrain = (month_idx - last_retrain_idx) >= retrain_every or cached_model is None

        if need_retrain:
            train_timestamps = X_train.index.values.astype(np.int64)
            sample_w = compute_sample_weights(train_timestamps, fwd_bars=fwd_bars)

        try:
            if need_retrain:
                scaler = StandardScaler()
                X_tr_scaled = scaler.fit_transform(X_train)
                import sklearn
                if tuple(int(x) for x in sklearn.__version__.split(".")[:2]) >= (1, 8):
                    l1_ratio = 1.0 if mcfg.penalty == "l1" else (0.5 if mcfg.penalty == "elasticnet" else 0.0)
                    model = LogisticRegression(C=mcfg.C, l1_ratio=l1_ratio,
                                               solver=mcfg.solver, max_iter=mcfg.max_iter,
                                               random_state=mcfg.seed)
                else:
                    model = LogisticRegression(C=mcfg.C, penalty=mcfg.penalty,
                                               solver=mcfg.solver, max_iter=mcfg.max_iter,
                                               random_state=mcfg.seed)
                model.fit(X_tr_scaled, y_train, sample_weight=sample_w)
                cached_model = model
                cached_scaler = scaler
                last_retrain_idx = month_idx

                # Compute per-window confidence statistics
                train_proba = model.predict_proba(X_tr_scaled)[:, 1]
                train_conf = np.abs(train_proba - 0.5)
                cached_conf_p70 = float(np.percentile(train_conf, 70.0))
                p75 = float(np.percentile(train_conf, 75.0))
                p25 = float(np.percentile(train_conf, 25.0))
                cached_conf_iqr = max(p75 - p25, 1e-8)
            else:
                scaler = cached_scaler
                model = cached_model

            X_te_scaled = scaler.transform(X_test)
            yp = model.predict_proba(X_te_scaled)[:, 1]
        except Exception:
            continue

        conf = np.abs(yp - 0.5)
        fwd_ret_vals = fwd_ret.reindex(te)
        directions = np.where(yp > 0.5, 1.0, -1.0)
        signals = get_signal(yp, num_classes=2)
        signals_disc = discretize_signal(signals, step_size=scfg.step_size)

        fr_vals = fwd_ret_vals.values if hasattr(fwd_ret_vals, "values") else np.array(fwd_ret_vals)
        finite_mask = np.isfinite(fr_vals)
        if finite_mask.any():
            sig_str = signals_disc if np.ndim(signals_disc) > 0 else np.full(len(yp), float(signals_disc))
            batch = pd.DataFrame({
                "timestamp": te[finite_mask],
                "direction": directions[finite_mask],
                "fwd_return": fr_vals[finite_mask],
                "confidence": conf[finite_mask],
                "probability": yp[finite_mask],
                "signal_strength": np.abs(sig_str[finite_mask]),
                "window_conf_p70": cached_conf_p70,
                "window_conf_iqr": cached_conf_iqr,
                "window_idx": month_idx,
            })
            rows.append(batch)

    if not rows:
        return None
    df = pd.concat(rows, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["asset"] = name

    # Apply regime filter
    df = apply_regime_filter(df, ohlcv_4h, config.regime_filter)
    return df if len(df) > 0 else None


def generate_all_raw_signals():
    """Generate raw multi-TF signals for all assets."""
    strat_sigs = {}
    tfs = [("15m", "15min", 4, 0.10)] if SMOKE else [
        ("15m", "15min", 4, 0.10), ("1h", "1h", 4, 0.10), ("4h", "4h", 6, 0.20)]

    for sname, tf, fwd, C in tfs:
        tasks = [(n, sym, tf, fwd, C) for n, sym in ASSETS.items()]
        all_sigs = []
        with ProcessPoolExecutor(max_workers=6) as pool:
            futs = {pool.submit(_gen_raw, t): t for t in tasks}
            for f in as_completed(futs):
                r = f.result()
                if r is not None and len(r) > 0:
                    all_sigs.append(r)
        if all_sigs:
            m = pd.concat(all_sigs, ignore_index=True)
            m["strategy"] = sname
            m["hold_bars"] = fwd
            strat_sigs[sname] = m
            print(f"    {sname}: {len(m):,} raw signals", flush=True)
    return strat_sigs


# ============================================================
# Confidence filter variants
# ============================================================

HARD_CONF_FLOOR = 0.08  # production hard confidence floor


def filter_baseline(signals):
    """Production baseline: per-window 70th percentile + 0.08 absolute floor."""
    df = signals.copy()
    mask = (df["confidence"] >= df["window_conf_p70"]) & (df["confidence"] >= HARD_CONF_FLOOR)
    return df[mask].reset_index(drop=True)


def filter_adaptive(signals):
    """Adaptive: IQR-scaled HARD FLOOR (the dominant filter).

    The 0.08 hard floor dominates (cuts 94%+ of signals). The 70th pctl
    threshold is usually below 0.08 and thus redundant. So the adaptive
    approach must scale the HARD FLOOR itself.

    When IQR is low (noisy window, tight confidence cluster) → raise floor
    above 0.08 (more selective, fewer noisy trades).
    When IQR is high (strong signal window) → lower floor below 0.08
    (let more confident signals through).

    Uses expanding median IQR as baseline (walk-forward safe: only past data).
    adaptive_floor = 0.08 * (expanding_median_iqr / current_iqr), clamped [0.06, 0.12]
    """
    df = signals.copy()

    # Compute expanding median IQR per window (walk-forward: only use past windows)
    window_iqrs = df.groupby("window_idx")["window_conf_iqr"].first().sort_index()
    expanding_median = window_iqrs.expanding().median()

    # Map expanding median back to each signal
    df["_exp_median_iqr"] = df["window_idx"].map(expanding_median)

    # Adaptive floor: scale 0.08 by ratio of median IQR to current IQR
    iqr_ratio = df["_exp_median_iqr"] / df["window_conf_iqr"].clip(lower=1e-8)
    adaptive_floor = (HARD_CONF_FLOOR * iqr_ratio).clip(lower=0.06, upper=0.12)

    # Effective threshold: max of adaptive floor and per-window 70th pctl
    effective_thresh = np.maximum(adaptive_floor, df["window_conf_p70"])

    mask = df["confidence"] >= effective_thresh
    df = df.drop(columns=["_exp_median_iqr"])
    return df[mask].reset_index(drop=True)


# ============================================================
# Vol overlay variants
# ============================================================

def _compute_daily_vol(args):
    """Compute per-asset daily vol series."""
    name, sym = args
    warnings.filterwarnings("ignore")
    try:
        ohlcv = resample_ohlcv(load_binance_1m(sym, DATA_DIRS), "1D")["close"]
        if ohlcv.index.tz is not None:
            ohlcv.index = ohlcv.index.tz_localize(None)
        return name, ohlcv.pct_change().rolling(20).std()
    except Exception:
        return name, None


def compute_vol_all():
    """Compute daily vol for all assets."""
    vol = {}
    with ProcessPoolExecutor(max_workers=6) as pool:
        for name, vs in pool.map(_compute_daily_vol, list(ASSETS.items())):
            if vs is not None:
                vol[name] = vs
    return vol


def apply_vol_scaling(signals, vol_series):
    """Production vol scaling: sizing_mult *= clip(0.025/vol, 0.3, 1.5)."""
    out = signals.copy()
    if "sizing_mult" not in out.columns:
        out["sizing_mult"] = 1.0
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)
    for i in range(len(out)):
        asset = out.iloc[i].get("asset", "")
        if asset not in vol_series:
            continue
        vs = vol_series[asset]
        idx = vs.index.searchsorted(ts.iloc[i])
        if 0 < idx < len(vs):
            v = vs.iloc[idx - 1]
            if pd.notna(v) and v > 0:
                out.iat[i, out.columns.get_loc("sizing_mult")] *= np.clip(0.025 / v, 0.3, 1.5)
    return out


# ============================================================
# Portfolio sim with MkrTkr60 VIP fees
# ============================================================

def run_sim(signals_prepared, starting_capital):
    port = PortfolioConfig(
        starting_capital=starting_capital, max_dd=0.10, delever_start=0.05,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
        max_per_strategy=0.50, base_sizing_pct=0.045, max_signal_mult=2.0,
        sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)

    merged = signals_prepared.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in merged.columns:
            merged[c] = 1.0
        merged[c] = merged[c].fillna(1.0)
    merged["probability"] = merged["probability"].astype(float)
    merged = merged.sort_values("timestamp").reset_index(drop=True)

    sim = simulate_portfolio(merged, port)
    if sim is None:
        return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None:
        return None
    if dpnl.index.tz is not None:
        dpnl = dpnl.copy()
        dpnl.index = dpnl.index.tz_localize(None)

    # MkrTkr60 VIP fee overlay
    maker_fill_pct = 0.90
    active = ~sim["skipped"]
    sizes = sim["sizes"]
    ts_arr = pd.to_datetime(merged["timestamp"].values)
    if ts_arr.tz is not None:
        ts_arr = ts_arr.tz_localize(None)
    dates_arr = ts_arr.date
    daily_vol = {}
    for i in range(len(merged)):
        if active[i]:
            daily_vol[dates_arr[i]] = daily_vol.get(dates_arr[i], 0) + abs(sizes[i]) * 2
    vol_s = pd.Series(daily_vol).sort_index()
    vol_30d = vol_s.rolling(30, min_periods=1).sum()
    eq_curve = starting_capital + dpnl.cumsum()

    dpnl_adj = dpnl.copy()
    for d in dpnl.index:
        d_date = d.date()
        eq = eq_curve.get(d, starting_capital)
        v30 = vol_30d.get(d_date, 0)
        _, maker_bps, taker_bps = get_vip_tier(eq, v30)
        rt_bps = (maker_fill_pct * 2 * maker_bps +
                  (1 - maker_fill_pct) * 2 * taker_bps + SPREAD_BPS)
        extra_bps = rt_bps - 1.0
        n_day = sum(1 for i in range(len(merged)) if active[i] and dates_arr[i] == d_date)
        avg_sz = daily_vol.get(d_date, 0) / max(n_day * 2, 1)
        dpnl_adj[d] -= n_day * avg_sz * extra_bps / 10000

    dr = dpnl_adj / starting_capital
    if dr.std() == 0 or len(dr) < 30:
        return None
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0
    eq_adj = (1 + dr).cumprod()
    dd = float(((eq_adj.cummax() - eq_adj) / eq_adj.cummax()).max() * 100)
    calmar = float(dr.mean() * 365.25 / (dd / 100)) if dd > 0 else 0
    pnl = float(dpnl_adj.sum())
    pf = float(dpnl_adj[dpnl_adj > 0].sum() / abs(dpnl_adj[dpnl_adj < 0].sum())) if (dpnl_adj < 0).any() else 99
    monthly = dpnl_adj.resample("MS").sum()
    pos_months = int((monthly > 0).sum())
    tot_months = len(monthly)
    return {"sr": sr, "sortino": sortino, "dd": dd, "calmar": calmar,
            "pnl": pnl, "pf": pf, "n_trades": sim["n_trades"], "monthly": monthly,
            "pos_months": pos_months, "tot_months": tot_months,
            "pnl_mo": pnl / max(tot_months, 1)}


# ============================================================
# Main
# ============================================================

def main():
    T0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print("=" * 80)
    print(f"  ADAPTIVE CONFIDENCE + PER-ASSET VOL EXPERIMENT ({mode})")
    print(f"  {len(ASSETS)} assets, raw signal gen then 4 variants")
    print("=" * 80)

    # Phase 1: Generate raw signals (no confidence filter)
    print(f"\n  Phase 1: Generating raw signals...", flush=True)
    t0 = time.time()
    strat_sigs = generate_all_raw_signals()
    if not strat_sigs:
        print("  ERROR: No signals generated. Exiting.")
        return
    raw = pd.concat(list(strat_sigs.values()), ignore_index=True)
    raw = raw.sort_values("timestamp").reset_index(drop=True)
    print(f"  Total raw: {len(raw):,} signals ({time.time()-t0:.0f}s)", flush=True)

    # Signal counts per asset
    print(f"\n  Raw signals per asset:")
    for name in sorted(ASSETS.keys()):
        n = len(raw[raw["asset"] == name])
        if n > 0:
            print(f"    {name:<10} {n:>6,}")

    # Phase 2: Compute vol series
    print(f"\n  Phase 2: Computing vol series...", flush=True)
    t0 = time.time()
    vol_series = compute_vol_all()
    print(f"  Done ({time.time()-t0:.0f}s)")

    # Diagnostics: confidence threshold distribution
    print(f"\n  Confidence threshold diagnostics:")
    window_stats = raw.groupby("window_idx").agg(
        p70=("window_conf_p70", "first"),
        iqr=("window_conf_iqr", "first"),
        n_signals=("confidence", "count"),
        mean_conf=("confidence", "mean"),
    )
    print(f"    Windows:       {len(window_stats)}")
    print(f"    p70 range:     [{window_stats['p70'].min():.4f}, {window_stats['p70'].max():.4f}]")
    print(f"    p70 median:    {window_stats['p70'].median():.4f}")
    print(f"    IQR range:     [{window_stats['iqr'].min():.4f}, {window_stats['iqr'].max():.4f}]")
    print(f"    IQR median:    {window_stats['iqr'].median():.4f}")
    pct_below_floor = (window_stats['p70'] < HARD_CONF_FLOOR).mean() * 100
    print(f"    Windows where p70 < 0.08: {pct_below_floor:.0f}%")

    # Phase 3: Apply filter variants + production vol scaling
    print(f"\n  Phase 3: Applying filter variants + vol scaling...", flush=True)

    variants = {}

    # 1. Baseline: 70th pctl + 0.08 floor + 0.025/vol (production)
    base_filtered = filter_baseline(raw)
    base_filtered = apply_vol_scaling(base_filtered, vol_series)
    variants["1-Baseline"] = base_filtered
    print(f"    1-Baseline:  {len(base_filtered):>8,} signals")

    # 2. Adaptive floor + 0.025/vol
    adapt_filtered = filter_adaptive(raw)
    adapt_filtered = apply_vol_scaling(adapt_filtered, vol_series)
    variants["2-Adaptive"] = adapt_filtered
    print(f"    2-Adaptive:  {len(adapt_filtered):>8,} signals")
    trade_delta = len(adapt_filtered) - len(base_filtered)
    print(f"    Delta:       {trade_delta:>+8,} ({trade_delta/max(len(base_filtered),1):.1%})")

    # Phase 4: Portfolio sims
    print(f"\n{'='*80}")
    print("  PORTFOLIO RESULTS (MkrTkr60, VIP fees, $250K)")
    print(f"{'='*80}")

    results = {}
    for vname, vsigs in variants.items():
        t0 = time.time()
        r = run_sim(vsigs, CAP)
        results[vname] = r
        if r:
            print(f"\n  {vname}:")
            print(f"    Sharpe:    {r['sr']:.3f}")
            print(f"    Sortino:   {r['sortino']:.3f}")
            print(f"    PnL:       ${r['pnl']:,.0f}")
            print(f"    MaxDD:     {r['dd']:.2f}%")
            print(f"    Calmar:    {r['calmar']:.2f}")
            print(f"    PF:        {r['pf']:.2f}")
            print(f"    Trades:    {r['n_trades']:,}")
            print(f"    $/month:   ${r['pnl_mo']:,.0f}")
            print(f"    Pos mo:    {r['pos_months']}/{r['tot_months']}")
        else:
            print(f"\n  {vname}: FAILED (no trades)")

    # Phase 5: Statistical comparison
    baseline_r = results.get("1-Baseline")
    if baseline_r is None:
        print("\n  Baseline failed — cannot compare.")
        return

    print(f"\n{'='*80}")
    print("  STATISTICAL COMPARISON vs BASELINE")
    print(f"{'='*80}")

    print(f"\n  {'Variant':<20} {'Sharpe':>8} {'Delta':>8} {'PnL':>12} {'Trades':>8} {'Wilcoxon p':>12}")
    print("  " + "-" * 72)

    for vname, r in results.items():
        if r is None:
            print(f"  {vname:<20} {'FAILED':>8}")
            continue
        delta = r["sr"] - baseline_r["sr"]

        # Wilcoxon test on monthly PnL
        if vname == "1-Baseline":
            p_str = "—"
        else:
            try:
                bm = baseline_r["monthly"]
                vm = r["monthly"]
                common = bm.index.intersection(vm.index)
                if len(common) > 10:
                    _, p = wilcoxon(vm.reindex(common).values - bm.reindex(common).values)
                    p_str = f"p={p:.4f}" + (" *" if p < 0.10 else " " + ("**" if p < 0.05 else ""))
                else:
                    p_str = "n/a"
            except Exception:
                p_str = "n/a"

        print(f"  {vname:<20} {r['sr']:>8.3f} {delta:>+8.3f} ${r['pnl']:>10,.0f} {r['n_trades']:>8,} {p_str:>12}")

    # Phase 6: Year-by-year breakdown
    print(f"\n{'='*80}")
    print("  YEAR-BY-YEAR SHARPE")
    print(f"{'='*80}")

    years = set()
    for r in results.values():
        if r and r["monthly"] is not None:
            years.update(r["monthly"].index.year)
    years = sorted(years)

    header = f"  {'Year':<6}" + "".join(f"{v:>16}" for v in results.keys())
    print(header)
    print("  " + "-" * (6 + 16 * len(results)))
    for yr in years:
        row = f"  {yr:<6}"
        for vname, r in results.items():
            if r is None:
                row += f"{'—':>16}"
                continue
            m = r["monthly"]
            yr_m = m[m.index.year == yr]
            if len(yr_m) < 2:
                row += f"{'—':>16}"
                continue
            dr = yr_m / CAP
            yr_sr = float(dr.mean() / dr.std() * np.sqrt(12)) if dr.std() > 0 else 0
            row += f"{yr_sr:>16.3f}"
        print(row)

    # Phase 7: Success/Kill criteria evaluation
    print(f"\n{'='*80}")
    print("  EXPERIMENT VERDICT")
    print(f"{'='*80}")

    adapt_r = results.get("2-Adaptive")

    if adapt_r:
        delta_a = adapt_r["sr"] - baseline_r["sr"]
        trade_drop = 1.0 - adapt_r["n_trades"] / max(baseline_r["n_trades"], 1)
        pass_a = delta_a >= 0.10 and trade_drop <= 0.20
        print(f"\n  Exp A (Adaptive Confidence):")
        print(f"    Sharpe delta:    {delta_a:+.3f} (need >= +0.10)")
        print(f"    Trade drop:      {trade_drop:.1%} (need <= 20%)")
        print(f"    DD change:       {adapt_r['dd'] - baseline_r['dd']:+.2f}pp")
        print(f"    PnL change:      ${adapt_r['pnl'] - baseline_r['pnl']:+,.0f}")
        print(f"    VERDICT:         {'PASS' if pass_a else 'FAIL/KILL' if delta_a < 0 else 'MARGINAL'}")

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
