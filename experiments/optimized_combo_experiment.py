#!/usr/bin/env python
"""Optimized Combo Test — Compare current vs optimized hold combos.

Tests specific TF+hold combinations identified by hold_sweep_experiment.

Usage:
    PYTHONPATH=src python experiments/optimized_combo_experiment.py
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

# TFs we need (label, resample)
NEEDED_TFS = [
    ("15m", "15min"), ("1h", "1h"), ("3h", "3h"), ("4h", "4h"), ("24h", "1D"),
]

# Combos to test: list of (name, [(tf_label, hold_bars), ...])
COMBOS = [
    ("Current: 15m(4)+1h(4)+4h(6)",       [("15m", 4),  ("1h", 4), ("4h", 6)]),
    ("Optimized: 15m(2)+1h(4)+4h(6)",     [("15m", 2),  ("1h", 4), ("4h", 6)]),
    ("New trio: 15m(2)+3h(16)+24h(12)",    [("15m", 2),  ("3h", 16), ("24h", 12)]),
    ("Quad: 15m(2)+1h(4)+3h(16)+4h(6)",   [("15m", 2),  ("1h", 4), ("3h", 16), ("4h", 6)]),
    ("Quad-B: 15m(2)+3h(16)+4h(6)+24h(12)", [("15m", 2), ("3h", 16), ("4h", 6), ("24h", 12)]),
    ("Full5: 15m(2)+1h(4)+3h(16)+4h(6)+24h(12)", [("15m", 2), ("1h", 4), ("3h", 16), ("4h", 6), ("24h", 12)]),
]


def get_vip_tier(equity, volume_30d):
    result = VIP_TIERS[0]
    for tier in VIP_TIERS:
        if equity >= tier[1] or volume_30d >= tier[2]:
            result = tier
    return result[0], result[3], result[4]


# ── Signal generation ──

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
    if len(sigs) == 0:
        return None
    sigs = apply_regime_filter(sigs, ohlcv_4h, config.regime_filter)
    return sigs if len(sigs) > 0 else None


def generate_signals(label, tf_resample):
    tasks = [(n, s, tf_resample) for n, s in ASSETS.items()]
    all_sigs = []
    with ProcessPoolExecutor(max_workers=8) as pool:
        futs = {pool.submit(_gen_one, t): t for t in tasks}
        for f in as_completed(futs):
            r = f.result()
            if r is not None and len(r) > 0:
                all_sigs.append(r)
    if not all_sigs:
        return None
    merged = pd.concat(all_sigs, ignore_index=True)
    merged["strategy"] = label
    merged["hold_bars"] = 4
    if "confidence" in merged.columns:
        merged = merged[merged["confidence"] >= 0.08].reset_index(drop=True)
    return merged if len(merged) > 0 else None


# ── Recompute returns ──

def recompute_returns_for_hold(signals, tf_resample, hold_bars):
    """Recompute fwd_return from OHLCV at a specific hold_bars."""
    assets_in = signals["asset"].unique()
    close_cache = {}
    for asset_name in assets_in:
        sym = ASSETS.get(asset_name)
        if sym is None:
            continue
        try:
            ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
            ohlcv_tf = resample_ohlcv(ohlcv_1m, tf_resample)
            close = ohlcv_tf["close"]
            if close.index.tz is not None:
                close.index = close.index.tz_localize(None)
            close_cache[asset_name] = close
            del ohlcv_1m, ohlcv_tf
        except Exception:
            continue

    sig_ts = pd.to_datetime(signals["timestamp"])
    if sig_ts.dt.tz is not None:
        sig_ts = sig_ts.dt.tz_localize(None)

    sigs_out = signals.copy()
    new_fwd = np.full(len(signals), np.nan)

    for asset_name, close in close_cache.items():
        mask = (signals["asset"] == asset_name).values
        positions = np.where(mask)[0]
        idx = close.index.searchsorted(sig_ts.iloc[positions].values)
        for k, pos in enumerate(positions):
            bi = idx[k]
            ei = bi + hold_bars
            if bi >= 0 and ei < len(close):
                new_fwd[pos] = (close.values[ei] - close.values[bi]) / close.values[bi]

    sigs_out["fwd_return"] = new_fwd
    sigs_out["hold_bars"] = hold_bars
    sigs_out = sigs_out.dropna(subset=["fwd_return"]).reset_index(drop=True)
    return sigs_out


# ── Vol scaling ──

def _compute_vol_one(args):
    name, sym = args
    warnings.filterwarnings("ignore")
    try:
        ohlcv = resample_ohlcv(load_binance_1m(sym, DATA_DIRS), "1D")["close"]
        if ohlcv.index.tz is not None:
            ohlcv.index = ohlcv.index.tz_localize(None)
        return name, ohlcv.pct_change().rolling(20).std()
    except Exception:
        return name, None


def compute_vol_parallel():
    vol = {}
    with ProcessPoolExecutor(max_workers=8) as pool:
        for name, vs in pool.map(_compute_vol_one, list(ASSETS.items())):
            if vs is not None:
                vol[name] = vs
    return vol


def apply_vol(signals, vol_series):
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


# ── Portfolio sim ──

def run_sim(signals_prepared, starting_capital, max_per_strategy=0.50):
    port = PortfolioConfig(
        starting_capital=starting_capital, max_dd=0.10, delever_start=0.05,
        slippage=0.0001, max_exposure=0.90, max_per_asset=0.06,
        max_per_strategy=max_per_strategy, base_sizing_pct=0.045, max_signal_mult=2.0,
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
    wr = float((dpnl_adj > 0).sum() / len(dpnl_adj)) if len(dpnl_adj) > 0 else 0
    monthly = dpnl_adj.resample("MS").sum()
    pos_months = int((monthly > 0).sum())
    tot_months = len(monthly)
    return {"sr": sr, "sortino": sortino, "dd": dd, "calmar": calmar,
            "pnl": pnl, "pf": pf, "wr": wr, "n_trades": sim["n_trades"],
            "monthly": monthly, "pos_months": pos_months, "tot_months": tot_months,
            "pnl_mo": pnl / max(tot_months, 1), "daily_pnl": dpnl_adj}


# ── Main ──

def main():
    T0 = time.time()
    print("=" * 100)
    print("  OPTIMIZED COMBO TEST")
    print(f"  30 assets, {len(COMBOS)} combos, MkrTkr60, $250K")
    print("=" * 100)

    # Vol
    t0 = time.time()
    vol_series = compute_vol_parallel()
    print(f"  Vol computed ({time.time()-t0:.0f}s)")

    # Generate & cache signals per TF
    print(f"\n  Generating signals...", flush=True)
    signal_cache = {}  # tf_label -> DataFrame
    for label, resample in NEEDED_TFS:
        t1 = time.time()
        sigs = generate_signals(label, resample)
        if sigs is not None:
            signal_cache[label] = sigs
            print(f"    {label:<6} {len(sigs):>8,} signals ({time.time()-t1:.0f}s)", flush=True)

    # Precompute hold variants: cache[(tf, hold)] -> vol-scaled signals
    print(f"\n  Computing hold variants...", flush=True)
    hold_cache = {}  # (tf_label, hold_bars) -> vol-scaled signals
    needed_holds = set()
    for _, components in COMBOS:
        for tf, h in components:
            needed_holds.add((tf, h))

    # Group by TF to minimize OHLCV reloads
    by_tf = {}
    for tf, h in needed_holds:
        by_tf.setdefault(tf, set()).add(h)

    for tf_label, holds in by_tf.items():
        if tf_label not in signal_cache:
            continue
        sigs = signal_cache[tf_label]
        resample = dict(NEEDED_TFS)[tf_label]
        for h in holds:
            t1 = time.time()
            if h == 4:
                # Original hold, no recomputation needed
                sigs_h = sigs.copy()
            else:
                sigs_h = recompute_returns_for_hold(sigs, resample, h)
            sigs_h = apply_vol(sigs_h, vol_series)
            hold_cache[(tf_label, h)] = sigs_h
            print(f"    {tf_label}(h={h}): {len(sigs_h):,} signals ({time.time()-t1:.0f}s)", flush=True)

    # Run combos
    print(f"\n{'='*100}")
    print("  RESULTS")
    print(f"{'='*100}")

    combo_results = {}
    print(f"\n  {'Combo':<45} {'Sigs':>7} {'Sharpe':>8} {'Sortino':>8} {'PnL':>12}"
          f" {'MaxDD':>7} {'Calmar':>7} {'PF':>6} {'WR':>6} {'$/mo':>8}")
    print("  " + "-" * 118)

    for name, components in COMBOS:
        parts = []
        for tf, h in components:
            key = (tf, h)
            if key in hold_cache:
                parts.append(hold_cache[key])
        if not parts:
            continue
        merged = pd.concat(parts, ignore_index=True)
        merged = merged.sort_values("timestamp").reset_index(drop=True)

        r = run_sim(merged, CAP, max_per_strategy=0.50)
        if r is None:
            print(f"  {name:<45} FAIL")
            continue
        combo_results[name] = r
        r["n_signals"] = len(merged)
        print(f"  {name:<45} {len(merged):>7,} {r['sr']:>8.3f} {r['sortino']:>8.3f}"
              f" ${r['pnl']:>10,.0f} {r['dd']:>6.2f}% {r['calmar']:>7.2f}"
              f" {r['pf']:>5.2f} {r['wr']:>5.1%} ${r['pnl_mo']:>7,.0f}")

    # Allocation sensitivity on top combos
    print(f"\n{'='*100}")
    print("  ALLOCATION SENSITIVITY (max_per_strategy sweep)")
    print(f"{'='*100}")

    ranked = sorted(combo_results.items(), key=lambda x: x[1]["sr"], reverse=True)
    top_combos = [name for name, _ in ranked[:3]]
    caps = [0.30, 0.40, 0.50, 0.60, 0.70, 1.0]

    print(f"\n  {'Combo':<45} {'Cap':>5} {'Sharpe':>8} {'PnL':>12} {'MaxDD':>7} {'$/mo':>8}")
    print("  " + "-" * 90)

    for combo_name in top_combos:
        components = dict(COMBOS)[combo_name]
        parts = [hold_cache[(tf, h)] for tf, h in components if (tf, h) in hold_cache]
        merged = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)

        best_sr, best_cap = -999, 0.50
        for cap in caps:
            r = run_sim(merged, CAP, max_per_strategy=cap)
            if r is None:
                continue
            is_best = r["sr"] > best_sr
            if is_best:
                best_sr, best_cap = r["sr"], cap
            mark = " *" if is_best else ""
            print(f"  {combo_name:<45} {cap:>5.2f} {r['sr']:>8.3f}"
                  f" ${r['pnl']:>10,.0f} {r['dd']:>6.2f}% ${r['pnl_mo']:>7,.0f}{mark}")
        print()

    # Monthly PnL comparison (top 3 + current baseline)
    print(f"\n{'='*100}")
    print("  MONTHLY PnL COMPARISON")
    print(f"{'='*100}")

    show = list(combo_results.keys())[:4]  # first 4 combos
    if not show:
        return

    # Truncate names for column headers
    short = {s: s.split(":")[0].strip() for s in show}

    all_months = sorted(set().union(*(
        combo_results[s]["monthly"].index for s in show)))

    header = f"  {'Month':<10}"
    for s in show:
        header += f" {short[s]:>16}"
    print(f"\n{header}")
    print("  " + "-" * (10 + 17 * len(show)))

    yearly = {s: {} for s in show}
    for m in all_months:
        row = f"  {m.strftime('%Y-%m'):<10}"
        for s in show:
            val = combo_results[s]["monthly"].get(m, 0)
            row += f" ${val:>14,.0f}"
            yr = m.year
            yearly[s][yr] = yearly[s].get(yr, 0) + val
        print(row)

    all_years = sorted(set().union(*(y.keys() for y in yearly.values())))
    print("  " + "-" * (10 + 17 * len(show)))
    for yr in all_years:
        row = f"  {yr:<10}"
        for s in show:
            val = yearly[s].get(yr, 0)
            row += f" ${val:>14,.0f}"
        print(row)

    print("  " + "=" * (10 + 17 * len(show)))
    row = f"  {'TOTAL':<10}"
    for s in show:
        row += f" ${combo_results[s]['pnl']:>14,.0f}"
    print(row)

    row = f"  {'Pos months':<10}"
    for s in show:
        r = combo_results[s]
        row += f" {r['pos_months']:>11}/{r['tot_months']:<3}"
    print(row)

    # Pairwise correlation of combos
    print(f"\n{'='*100}")
    print("  DAILY PnL CORRELATION BETWEEN COMBOS")
    print(f"{'='*100}")

    pnl_df = pd.DataFrame({short[s]: combo_results[s]["daily_pnl"] for s in show}).fillna(0)
    corr = pnl_df.corr()
    col_w = max(max(len(v) for v in short.values()) + 1, 8)
    header = f"  {'':>{col_w}}"
    for s in show:
        header += f" {short[s]:>{col_w}}"
    print(f"\n{header}")
    print("  " + "-" * (col_w + 1 + (col_w + 1) * len(show)))
    for r_s in show:
        line = f"  {short[r_s]:>{col_w}}"
        for c_s in show:
            v = corr.loc[short[r_s], short[c_s]]
            if r_s == c_s:
                line += f" {'1.00':>{col_w}}"
            else:
                line += f" {v:>{col_w}.3f}"
        print(line)

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
