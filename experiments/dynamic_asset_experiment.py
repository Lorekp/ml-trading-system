#!/usr/bin/env python
"""Dynamic Asset Selection — Test with current Full5 + F1 config.

Instead of fixed 30 assets, select top-N each month by 30d median dollar volume.
Compare: fixed-30 vs dynamic-20 vs dynamic-30 vs dynamic-40.

Usage:
    PYTHONPATH=src python experiments/dynamic_asset_experiment.py [--smoke]
"""
import sys, os, time, warnings, gc, math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.model import WalkForwardModel
from signal_engine.filters import apply_regime_filter
from signal_engine.portfolio import compute_daily_pnl_series
from signal_engine.asset_selection import (
    compute_monthly_dollar_volume, select_top_n_for_month,
)
from signal_engine.config import *

SMOKE = "--smoke" in sys.argv
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
OHLCV_FEATURES = [
    "ret_1","ret_3","ret_7","ret_14","bar_range","vol_zscore","rsi_14",
    "stoch_rsi","macd_hist","bb_pct_b","atr_14","dist_swing_high","dist_swing_low"]
VIP_TIERS = [
    ("VIP0",0,0,2.00,5.50),("VIP1",100_000,10_000_000,1.80,4.00),
    ("VIP2",250_000,25_000_000,1.60,3.75),("VIP3",500_000,50_000_000,1.40,3.50),
    ("VIP4",1_000_000,100_000_000,1.20,3.20),("VIP5",2_000_000,250_000_000,1.00,3.20),
    ("Supreme",5_000_000,500_000_000,0.00,3.00),
]
SPREAD_BPS = 0.5; CAP = 250_000; BASELINE_WR = 0.65

# Fixed 30 (current production)
FIXED_30 = {
    "BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT","XRP":"XRPUSDT",
    "BNB":"BNBUSDT","ADA":"ADAUSDT","AVAX":"AVAXUSDT","DOT":"DOTUSDT",
    "LINK":"LINKUSDT","NEAR":"NEARUSDT","UNI":"UNIUSDT","AAVE":"AAVEUSDT",
    "LTC":"LTCUSDT","BCH":"BCHUSDT","HBAR":"HBARUSDT","TRX":"TRXUSDT",
    "SUI":"SUIUSDT","APT":"APTUSDT","ENJ":"ENJUSDT","SXP":"SXPUSDT",
    "ZEC":"ZECUSDT","XLM":"XLMUSDT","FET":"FETUSDT","WLD":"WLDUSDT",
    "ANKR":"ANKRUSDT","TAO":"TAOUSDT","ENA":"ENAUSDT",
    "FTM":"FTMUSDT","FIL":"FILUSDT","BNX":"BNXUSDT",
}

NEEDED_TFS = [("15m","15min"),("1h","1h"),("3h","3h"),("4h","4h"),("24h","1D")]
FULL5_HOLDS = {"15m": 2, "1h": 4, "3h": 16, "4h": 6, "24h": 12}


def get_vip_tier(eq, v30):
    r = VIP_TIERS[0]
    for t in VIP_TIERS:
        if eq >= t[1] or v30 >= t[2]: r = t
    return r[0], r[3], r[4]


def _edge_from_probability(prob, sig):
    return float(np.clip(abs(2.0 * prob - 1.0), 0.0, 1.0)) if np.isfinite(prob) else float(np.clip(abs(sig), 0.0, 1.0))


# ── Signal generation per asset ──

def _gen_one_asset(name, sym, tf_resample, fwd_bars=4, C=0.10):
    """Generate signals for one asset at one timeframe."""
    warnings.filterwarnings("ignore")
    config = SystemConfig(
        model=ModelConfig(C=C, penalty="elasticnet", solver="saga", max_iter=1000,
                          seed=42, fwd_bars=fwd_bars, signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=list(OHLCV_FEATURES), winsor_lo=0.01, winsor_hi=0.99,
                               vol_filter_quantile=0.50, vol_zscore_window=30, swing_lookback=20),
        regime_filter=RegimeFilterConfig(enabled=True, timeframe="4h", ema_span=20),
        confidence_filter=ConfidenceFilterConfig(mode="percentile", threshold=70.0),
        signal=SignalConfig(step_size=0.1),
        data=DataConfig(data_dirs=[str(d) for d in DATA_DIRS], timeframe=tf_resample, regime_timeframe="4h"),
        assets=FIXED_30)  # doesn't matter, just needs valid config
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


def _gen_wrapper(args):
    return _gen_one_asset(*args)


# ── Hold recomputation ──

def recompute_returns(signals, tf_resample, hold_bars):
    if hold_bars == 4: return signals.copy()
    close_cache = {}
    for a in signals["asset"].unique():
        sym = a + "USDT" if not a.endswith("USDT") else a
        try:
            c = resample_ohlcv(load_binance_1m(sym, DATA_DIRS), tf_resample)["close"]
            if c.index.tz is not None: c.index = c.index.tz_localize(None)
            close_cache[a] = c
        except: continue
    sig_ts = pd.to_datetime(signals["timestamp"])
    if sig_ts.dt.tz is not None: sig_ts = sig_ts.dt.tz_localize(None)
    out = signals.copy(); fwd = np.full(len(signals), np.nan)
    for a, close in close_cache.items():
        mask = (signals["asset"] == a).values; pos = np.where(mask)[0]
        idx = close.index.searchsorted(sig_ts.iloc[pos].values)
        for k, p in enumerate(pos):
            bi, ei = idx[k], idx[k] + hold_bars
            if 0 <= bi and ei < len(close): fwd[p] = (close.values[ei] - close.values[bi]) / close.values[bi]
    out["fwd_return"] = fwd; out["hold_bars"] = hold_bars
    return out.dropna(subset=["fwd_return"]).reset_index(drop=True)


# ── Vol scaling ──

def _compute_vol_one(args):
    name, sym = args
    warnings.filterwarnings("ignore")
    try:
        o = resample_ohlcv(load_binance_1m(sym, DATA_DIRS), "1D")["close"]
        if o.index.tz is not None: o.index = o.index.tz_localize(None)
        return name, o.pct_change().rolling(20).std()
    except: return name, None


def apply_vol(signals, vol_series):
    out = signals.copy()
    if "sizing_mult" not in out.columns: out["sizing_mult"] = 1.0
    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
    for i in range(len(out)):
        a = out.iloc[i].get("asset", "")
        if a not in vol_series: continue
        vs = vol_series[a]; idx = vs.index.searchsorted(ts.iloc[i])
        if 0 < idx < len(vs):
            v = vs.iloc[idx - 1]
            if pd.notna(v) and v > 0:
                out.iat[i, out.columns.get_loc("sizing_mult")] *= np.clip(0.025 / v, 0.3, 1.5)
    return out


# ── F1 breaker sim ──

def simulate_f1(signals_df, config):
    if signals_df is None or len(signals_df) < 10: return None
    sub = signals_df.sort_values("timestamp").reset_index(drop=True)
    n = len(sub)
    timestamps = sub["timestamp"].values
    assets_arr = sub["asset"].values if "asset" in sub.columns else np.full(n, "SINGLE")
    directions = sub["direction"].values.astype(np.float64)
    fwd_returns = sub["fwd_return"].values.astype(np.float64)
    signal_strengths = sub["signal_strength"].values.astype(np.float64) if "signal_strength" in sub.columns else None
    probabilities = sub["probability"].values.astype(np.float64) if "probability" in sub.columns else np.full(n, np.nan)
    regime_mults = sub["regime_exposure_mult"].values.astype(np.float64) if "regime_exposure_mult" in sub.columns else np.ones(n)
    sizing_mults = sub["sizing_mult"].values.astype(np.float64) if "sizing_mult" in sub.columns else np.ones(n)
    hold_bars_arr = sub["hold_bars"].values.astype(np.int64) if "hold_bars" in sub.columns else np.full(n, 4, dtype=np.int64)
    strategies_arr = sub["strategy"].values if "strategy" in sub.columns else np.full(n, "default")
    equity_arr = np.empty(n); pnl_arr = np.zeros(n); dd_arr = np.empty(n)
    size_arr = np.zeros(n); skipped = np.ones(n, dtype=bool)
    equity = config.starting_capital; peak_equity = equity; halted = False
    active_positions = []; unique_ts = pd.Series(timestamps).unique()
    ts_to_idx = {ts: idx for idx, ts in enumerate(unique_ts)}
    use_prob = signal_strengths is not None
    asset_recent_pnl = defaultdict(lambda: deque(maxlen=10))
    asset_recent_not = defaultdict(lambda: deque(maxlen=10))

    for i in range(n):
        current_ts = timestamps[i]; current_ts_idx = ts_to_idx.get(current_ts, 0)
        active_positions = [(e, sz, a, st) for e, sz, a, st in active_positions if e > current_ts_idx]
        current_exposure = sum(sz for _, sz, _, _ in active_positions)
        asset_exposure = {}; strategy_exposure = {}
        for _, sz, a, st in active_positions:
            asset_exposure[a] = asset_exposure.get(a, 0.0) + sz
            strategy_exposure[st] = strategy_exposure.get(st, 0.0) + sz
        if halted: equity_arr[i] = equity; dd_arr[i] = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0; continue
        dd = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        if dd >= config.max_dd: halted = True; equity_arr[i] = equity; dd_arr[i] = dd; continue
        size_mult = max(1.0 - (dd - config.delever_start) / (config.max_dd - config.delever_start), 0.0) if dd > config.delever_start else 1.0
        if use_prob:
            sig_abs = abs(signal_strengths[i])
            if config.sizing_mode == "probability_edge": sig_abs = _edge_from_probability(probabilities[i], sig_abs)
            if config.use_regime_exposure_multiplier: sig_abs *= float(np.clip(regime_mults[i], config.regime_exposure_floor, 1.0))
            raw_size = equity * config.base_sizing_pct * sig_abs * size_mult * sizing_mults[i]
            raw_size = min(raw_size, equity * config.base_sizing_pct * config.max_signal_mult * size_mult)
        else:
            raw_size = equity * config.base_sizing_pct * size_mult
        # F1 breaker
        asset = assets_arr[i]
        rp = asset_recent_pnl[asset]; rn = asset_recent_not[asset]
        if len(rp) >= 10:
            ratio = sum(rp) / sum(rn) if sum(rn) > 0 else 0
            if ratio < -0.01: raw_size *= max(1.0 + ratio / 0.01, 0.10)
        # Exposure limits
        ae = asset_exposure.get(asset, 0.0)
        if ae + raw_size > equity * config.max_per_asset: raw_size = max(equity * config.max_per_asset - ae, 0.0)
        strategy = strategies_arr[i]; mps = getattr(config, "max_per_strategy", 1.0)
        se = strategy_exposure.get(strategy, 0.0)
        if mps < 1.0 and se + raw_size > equity * mps: raw_size = max(equity * mps - se, 0.0)
        if current_exposure + raw_size > equity * config.max_exposure: raw_size = max(equity * config.max_exposure - current_exposure, 0.0)
        if raw_size < 1.0: equity_arr[i] = equity; dd_arr[i] = dd; continue
        trade_return = directions[i] * fwd_returns[i] - config.slippage
        pnl = raw_size * trade_return; equity += pnl
        if equity > peak_equity: peak_equity = equity
        active_positions.append((current_ts_idx + int(hold_bars_arr[i]), raw_size, asset, strategy))
        equity_arr[i] = equity; pnl_arr[i] = pnl
        dd_arr[i] = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        size_arr[i] = raw_size; skipped[i] = False
        asset_recent_pnl[asset].append(pnl); asset_recent_not[asset].append(raw_size)

    return {"equity": equity_arr, "pnl": pnl_arr, "dd": dd_arr, "sizes": size_arr,
            "skipped": skipped, "timestamps": timestamps, "halted": halted,
            "final_equity": equity, "n_trades": int((~skipped).sum())}


def run_sim(merged, vol_series):
    prepared = apply_vol(merged, vol_series)
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in prepared.columns: prepared[c] = 1.0
        prepared[c] = prepared[c].fillna(1.0)
    prepared["probability"] = prepared["probability"].astype(float)
    prepared = prepared.sort_values("timestamp").reset_index(drop=True)
    port = PortfolioConfig(
        starting_capital=CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
        max_exposure=0.90, max_per_asset=0.06, max_per_strategy=0.50,
        base_sizing_pct=0.045, max_signal_mult=2.0, sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)
    sim = simulate_f1(prepared, port)
    if sim is None: return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None: return None
    if dpnl.index.tz is not None: dpnl = dpnl.copy(); dpnl.index = dpnl.index.tz_localize(None)
    # VIP fees
    mfp = 0.90; active = ~sim["skipped"]; sizes = sim["sizes"]
    ts_arr = pd.to_datetime(prepared["timestamp"].values)
    if ts_arr.tz is not None: ts_arr = ts_arr.tz_localize(None)
    dates_arr = ts_arr.date; daily_vol = {}
    for i in range(len(prepared)):
        if active[i]: daily_vol[dates_arr[i]] = daily_vol.get(dates_arr[i], 0) + abs(sizes[i]) * 2
    vol_s = pd.Series(daily_vol).sort_index(); vol_30d = vol_s.rolling(30, min_periods=1).sum()
    eq_curve = CAP + dpnl.cumsum(); dpnl_adj = dpnl.copy()
    for d in dpnl.index:
        dd = d.date(); eq = eq_curve.get(d, CAP); v30 = vol_30d.get(dd, 0)
        _, mbps, tbps = get_vip_tier(eq, v30)
        rt = mfp * 2 * mbps + (1 - mfp) * 2 * tbps + SPREAD_BPS; extra = rt - 1.0
        nd = sum(1 for i in range(len(prepared)) if active[i] and dates_arr[i] == dd)
        avg = daily_vol.get(dd, 0) / max(nd * 2, 1)
        dpnl_adj[d] -= nd * avg * extra / 10000
    dr = dpnl_adj / CAP
    if dr.std() == 0 or len(dr) < 30: return None
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0
    eq_adj = (1 + dr).cumprod(); dd_s = (eq_adj.cummax() - eq_adj) / eq_adj.cummax()
    max_dd = float(dd_s.max() * 100)
    calmar = float(dr.mean() * 365.25 / (max_dd / 100)) if max_dd > 0 else 0
    pnl = float(dpnl_adj.sum())
    pf = float(dpnl_adj[dpnl_adj > 0].sum() / abs(dpnl_adj[dpnl_adj < 0].sum())) if (dpnl_adj < 0).any() else 99
    daily_dd_pct = dpnl_adj / (CAP + dpnl_adj.cumsum()).shift(1).fillna(CAP) * 100
    monthly = dpnl_adj.resample("MS").sum()
    return {"sr": sr, "sortino": sortino, "max_dd": max_dd, "calmar": calmar,
            "pnl": pnl, "pf": pf, "n_trades": sim["n_trades"],
            "max_daily_dd": float(daily_dd_pct.min()),
            "pnl_mo": pnl / max(len(monthly), 1), "n_assets": len(merged["asset"].unique())}


def main():
    T0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print("=" * 100)
    print(f"  DYNAMIC ASSET SELECTION EXPERIMENT ({mode})")
    print(f"  Full5 config + F1 breaker, MkrTkr60")
    print("=" * 100)

    # Phase 1: Discover all available symbols
    data_dir = DATA_DIRS[0]
    all_symbols = {}
    for d in sorted(data_dir.iterdir()):
        if d.is_dir() and d.name.endswith("USDT"):
            all_symbols[d.name.replace("USDT", "")] = d.name
    print(f"  Available symbols: {len(all_symbols)}")

    if SMOKE:
        all_symbols = {k: v for k, v in list(all_symbols.items())[:10]}

    # Phase 2: Compute dollar volumes for all symbols
    print(f"\n  Computing dollar volumes...", flush=True)
    asset_volumes = {}
    asset_first_dates = {}
    for name, sym in sorted(all_symbols.items()):
        try:
            ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
            ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
            vol = compute_monthly_dollar_volume(ohlcv_15m)
            if vol.index.tz is not None: vol.index = vol.index.tz_localize(None)
            asset_volumes[name] = vol
            first = ohlcv_15m.index[0]
            if hasattr(first, 'tz') and first.tz is not None:
                first = first.tz_localize(None)
            asset_first_dates[name] = first
        except:
            pass
    print(f"  Computed volumes for {len(asset_volumes)} symbols")

    # Phase 3: Build monthly schedules for different top_n
    top_ns = [20, 30, 40] if not SMOKE else [10]

    # Compute vol series for all symbols (for vol scaling)
    print(f"\n  Computing vol series...", flush=True)
    vol_pairs = [(n, all_symbols[n]) for n in asset_volumes.keys() if n in all_symbols]
    vol_series = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for name, vs in pool.map(_compute_vol_one, vol_pairs):
            if vs is not None: vol_series[name] = vs

    # Generate signals for ALL available assets (cache per asset)
    print(f"\n  Generating signals for all assets...", flush=True)
    # Per-asset signal cache: asset_sigs[name][tf_label] = DataFrame
    asset_sigs = defaultdict(dict)
    for tf_label, tf_resample in NEEDED_TFS:
        t1 = time.time()
        tasks = [(name, all_symbols[name], tf_resample) for name in asset_volumes.keys() if name in all_symbols]
        with ProcessPoolExecutor(max_workers=8) as pool:
            futs = {pool.submit(_gen_wrapper, t): t[0] for t in tasks}
            for f in as_completed(futs):
                name = futs[f]
                r = f.result()
                if r is not None and len(r) > 0:
                    r["strategy"] = tf_label; r["hold_bars"] = 4
                    if "confidence" in r.columns:
                        r = r[r["confidence"] >= 0.08].reset_index(drop=True)
                    if len(r) > 0:
                        asset_sigs[name][tf_label] = r
        n_with = sum(1 for a in asset_sigs if tf_label in asset_sigs[a])
        print(f"    {tf_label}: {n_with} assets with signals ({time.time()-t1:.0f}s)", flush=True)
    gc.collect()

    # Phase 4: Recompute returns ONCE per asset per TF (not per month)
    print(f"\n  Recomputing hold returns (once per asset)...", flush=True)
    # hold_sigs[name][tf_label] = DataFrame with correct hold_bars and fwd_return
    hold_sigs = defaultdict(dict)
    for tf_label, tf_resample in NEEDED_TFS:
        hold = FULL5_HOLDS[tf_label]
        # Gather all assets that have signals for this TF
        all_assets_this_tf = [name for name in asset_sigs if tf_label in asset_sigs[name]]
        if not all_assets_this_tf:
            continue
        if hold == 4:
            # No recomputation needed
            for name in all_assets_this_tf:
                s = asset_sigs[name][tf_label].copy()
                s["hold_bars"] = hold
                hold_sigs[name][tf_label] = s
        else:
            # Batch recompute: concat all assets, recompute once
            all_s = pd.concat([asset_sigs[name][tf_label] for name in all_assets_this_tf], ignore_index=True)
            all_s_fixed = recompute_returns(all_s, tf_resample, hold)
            # Split back by asset
            for name in all_assets_this_tf:
                mask = all_s_fixed["asset"] == name
                s = all_s_fixed[mask].reset_index(drop=True)
                if len(s) > 0:
                    hold_sigs[name][tf_label] = s
        print(f"    {tf_label} (h={hold}): {len(all_assets_this_tf)} assets", flush=True)

    # Phase 5: Assemble and run variants
    print(f"\n{'='*100}")
    print("  RESULTS")
    print(f"{'='*100}")

    variants = [("Fixed-30", None)]
    for tn in top_ns:
        variants.append((f"Dynamic-{tn}", tn))

    print(f"\n  {'Variant':<16} {'Sharpe':>7} {'Sortino':>8} {'PnL':>11} {'MaxDD':>6} {'DayDD':>7}"
          f" {'Calmar':>7} {'PF':>6} {'Trades':>7} {'Assets':>7} {'$/mo':>8}")
    print("  " + "-" * 100)

    for var_name, top_n in variants:
        t1 = time.time()
        if top_n is None:
            # Fixed-30: concat signals for FIXED_30 assets only
            parts = []
            for name in FIXED_30.keys():
                for tf_label, _ in NEEDED_TFS:
                    if name in hold_sigs and tf_label in hold_sigs[name]:
                        parts.append(hold_sigs[name][tf_label])
            if not parts:
                print(f"  {var_name:<16} NO SIGNALS"); continue
            merged = pd.concat(parts, ignore_index=True)
        else:
            # Dynamic: build monthly schedule, filter pre-computed signals by month
            months = pd.period_range("2020-07", "2026-02", freq="M")
            parts = []
            for month in months:
                ms = pd.Timestamp(month.start_time)
                selected = select_top_n_for_month(
                    asset_volumes, asset_first_dates, ms, top_n=top_n, min_history_days=180)
                me = ms + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
                for name in selected:
                    for tf_label, _ in NEEDED_TFS:
                        if name not in hold_sigs or tf_label not in hold_sigs[name]:
                            continue
                        s = hold_sigs[name][tf_label]
                        ts = pd.to_datetime(s["timestamp"])
                        if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
                        mask = (ts >= ms) & (ts < me)
                        s_month = s[mask.values]
                        if len(s_month) > 0:
                            parts.append(s_month)
            if not parts:
                print(f"  {var_name:<16} NO SIGNALS"); continue
            merged = pd.concat(parts, ignore_index=True)
            # Deduplicate (a signal could appear in multiple months if asset stays selected)
            merged = merged.drop_duplicates(subset=["timestamp", "asset", "strategy"]).reset_index(drop=True)

        merged = merged.sort_values("timestamp").reset_index(drop=True)
        r = run_sim(merged, vol_series)
        if r is None:
            print(f"  {var_name:<16} FAIL"); continue
        print(f"  {var_name:<16} {r['sr']:>7.3f} {r['sortino']:>8.3f} ${r['pnl']:>9,.0f} {r['max_dd']:>5.1f}%"
              f" {r['max_daily_dd']:>6.2f}% {r['calmar']:>7.2f} {r['pf']:>5.2f}"
              f" {r['n_trades']:>7,} {r['n_assets']:>7} ${r['pnl_mo']:>7,.0f}")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
