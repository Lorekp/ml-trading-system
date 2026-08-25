#!/usr/bin/env python
"""Timeframe Combination Experiment.

Tests which timeframe combinations maximize portfolio Sharpe under MkrTkr60.

Design: Cache-and-Combine
  Phase 1: Generate signals per (TF, fwd_bars, C) config, cache
  Phase 2: Evaluate each config solo
  Phase 3: Test all TF combinations using best config per TF
  Phase 4: Allocation sensitivity (max_per_strategy sweep)
  Phase 5: Winner summary with monthly PnL

Baseline: 15m+1h+4h, MkrTkr60, Sharpe 2.929, $321K on $250K.

Usage:
    PYTHONPATH=src python experiments/timeframe_combination_experiment.py [--smoke]
"""
import sys, os, time, warnings, gc, itertools
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
ASSETS = {"BTC":"BTCUSDT","ETH":"ETHUSDT","SOL":"SOLUSDT"} if SMOKE else ALL_ASSETS

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

# Signal gen configs: (label, tf_resample, fwd_bars, C)
ALL_CONFIGS = [
    ("15m_f4_C010", "15min", 4, 0.10),
    ("15m_f2_C010", "15min", 2, 0.10),
    ("15m_f6_C010", "15min", 6, 0.10),
    ("30m_f4_C010", "30min", 4, 0.10),
    ("30m_f4_C020", "30min", 4, 0.20),
    ("1h_f4_C010",  "1h",   4, 0.10),
    ("1h_f2_C010",  "1h",   2, 0.10),
    ("1h_f6_C010",  "1h",   6, 0.10),
    ("2h_f4_C010",  "2h",   4, 0.10),
    ("2h_f4_C020",  "2h",   4, 0.20),
    ("4h_f6_C020",  "4h",   6, 0.20),
    ("4h_f4_C020",  "4h",   4, 0.20),
    ("4h_f4_C010",  "4h",   4, 0.10),
]
SMOKE_CONFIGS = [
    ("15m_f4_C010", "15min", 4, 0.10),
    ("1h_f4_C010",  "1h",   4, 0.10),
    ("4h_f6_C020",  "4h",   6, 0.20),
]
CONFIGS = SMOKE_CONFIGS if SMOKE else ALL_CONFIGS

TF_NAMES = ["15m", "30m", "1h", "2h", "4h"]


def get_vip_tier(equity, volume_30d):
    result = VIP_TIERS[0]
    for tier in VIP_TIERS:
        if equity >= tier[1] or volume_30d >= tier[2]:
            result = tier
    return result[0], result[3], result[4]


def tf_label(resample_str):
    """Convert resample string to short label: '15min'->'15m', '30min'->'30m', etc."""
    return resample_str.replace("min", "m")


# ── Signal generation ──

def _gen_one(args):
    """Generate signals for one asset at one (tf, fwd, C) config."""
    name, sym, tf_resample, fwd_bars, C = args
    warnings.filterwarnings("ignore")
    config = SystemConfig(
        model=ModelConfig(C=C, penalty="elasticnet", solver="saga", max_iter=1000,
                          seed=42, fwd_bars=fwd_bars, signal_model_type="asset_logistic"),
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


def generate_signals_for_config(label, tf_resample, fwd_bars, C, assets):
    """Generate signals for one config across all assets. Returns DataFrame or None."""
    tfl = tf_label(tf_resample)
    tasks = [(n, s, tf_resample, fwd_bars, C) for n, s in assets.items()]
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
    merged["strategy"] = tfl
    merged["hold_bars"] = fwd_bars
    if "confidence" in merged.columns:
        merged = merged[merged["confidence"] >= 0.08].reset_index(drop=True)
    return merged if len(merged) > 0 else None


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


# ── Portfolio sim with VIP fees (MkrTkr60) ──

def run_sim(signals_prepared, starting_capital, max_per_strategy=0.50):
    """Run portfolio sim with MkrTkr60 VIP fee overlay."""
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

    # MkrTkr60 VIP fee overlay (90% maker, 10% taker)
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
    n_months = max(tot_months, 1)
    return {"sr": sr, "sortino": sortino, "dd": dd, "calmar": calmar,
            "pnl": pnl, "pf": pf, "wr": wr, "n_trades": sim["n_trades"],
            "monthly": monthly, "pos_months": pos_months, "tot_months": tot_months,
            "pnl_mo": pnl / n_months}


# ── Main ──

def main():
    T0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print("=" * 90)
    print(f"  TIMEFRAME COMBINATION EXPERIMENT ({mode})")
    print(f"  {len(ASSETS)} assets, {len(CONFIGS)} configs, MkrTkr60 execution")
    print("=" * 90)

    # ── Phase 1: Generate & cache signals ──
    print(f"\n{'='*90}")
    print("  PHASE 1: Generate & Cache Signals")
    print(f"{'='*90}")

    t0 = time.time()
    vol_series = compute_vol_parallel()
    print(f"  Vol series computed ({time.time()-t0:.0f}s)")

    cache = {}  # key: config label -> vol-scaled DataFrame
    for label, tf_resample, fwd, C in CONFIGS:
        t1 = time.time()
        sigs = generate_signals_for_config(label, tf_resample, fwd, C, ASSETS)
        if sigs is not None and len(sigs) > 0:
            sigs_vol = apply_vol(sigs, vol_series)
            cache[label] = sigs_vol
            print(f"  {label}: {len(sigs_vol):>6,} signals ({time.time()-t1:.0f}s)")
        else:
            print(f"  {label}: 0 signals (skipped)")
        gc.collect()

    print(f"\n  Total configs cached: {len(cache)}")
    print(f"  Phase 1 time: {time.time()-t0:.0f}s")

    if len(cache) == 0:
        print("  ERROR: No signals generated. Exiting.")
        return

    # ── Phase 2: Solo evaluation ──
    print(f"\n{'='*90}")
    print("  PHASE 2: Individual Config Evaluation (solo, no strategy cap)")
    print(f"{'='*90}")

    solo_results = {}
    print(f"\n  {'Config':<20} {'Signals':>8} {'Sharpe':>8} {'Sortino':>8} {'PnL':>12}"
          f" {'MaxDD':>7} {'Calmar':>7} {'PF':>6} {'WR':>6} {'$/mo':>8}")
    print("  " + "-" * 95)

    for label, sigs in sorted(cache.items()):
        r = run_sim(sigs, CAP, max_per_strategy=1.0)
        if r is not None:
            solo_results[label] = r
            print(f"  {label:<20} {len(sigs):>8,} {r['sr']:>8.3f} {r['sortino']:>8.3f}"
                  f" ${r['pnl']:>10,.0f} {r['dd']:>6.2f}% {r['calmar']:>7.2f}"
                  f" {r['pf']:>5.2f} {r['wr']:>5.1%} ${r['pnl_mo']:>7,.0f}")
        else:
            print(f"  {label:<20} {len(sigs):>8,}     FAIL")

    # Pick best config per timeframe base
    tf_best = {}  # tf_name -> (label, sharpe)
    for label, r in solo_results.items():
        tf_base = label.split("_")[0]  # "15m", "30m", "1h", "2h", "4h"
        if tf_base not in tf_best or r["sr"] > tf_best[tf_base][1]:
            tf_best[tf_base] = (label, r["sr"])

    print(f"\n  Best config per timeframe:")
    for tf in TF_NAMES:
        if tf in tf_best:
            label, sr = tf_best[tf]
            print(f"    {tf}: {label} (Sharpe {sr:.3f})")

    viable_tfs = sorted(tf_best.keys(), key=lambda x: TF_NAMES.index(x))
    print(f"\n  Viable timeframes: {', '.join(viable_tfs)}")

    # ── Phase 3: Combination search ──
    print(f"\n{'='*90}")
    print("  PHASE 3: Combination Search (max_per_strategy=0.50)")
    print(f"{'='*90}")

    combos = []
    for r in range(1, len(viable_tfs) + 1):
        for combo in itertools.combinations(viable_tfs, r):
            combos.append(combo)

    combo_results = {}
    print(f"\n  Testing {len(combos)} combinations...")
    print(f"\n  {'#':>3} {'Combination':<30} {'Signals':>8} {'Sharpe':>8} {'Sortino':>8}"
          f" {'PnL':>12} {'MaxDD':>7} {'Calmar':>7} {'PF':>6} {'$/mo':>8}")
    print("  " + "-" * 100)

    for combo in combos:
        labels = [tf_best[tf][0] for tf in combo]
        merged = pd.concat([cache[l] for l in labels], ignore_index=True)
        merged = merged.sort_values("timestamp").reset_index(drop=True)
        r = run_sim(merged, CAP, max_per_strategy=0.50)
        combo_key = "+".join(combo)
        if r is not None:
            combo_results[combo_key] = r
            r["n_signals"] = len(merged)

    # Sort by Sharpe, print
    ranked = sorted(combo_results.items(), key=lambda x: x[1]["sr"], reverse=True)
    for i, (key, r) in enumerate(ranked, 1):
        baseline = " <-- BASELINE" if key == "15m+1h+4h" else ""
        best = " <-- BEST" if i == 1 and key != "15m+1h+4h" else ""
        print(f"  {i:>3} {key:<30} {r['n_signals']:>8,} {r['sr']:>8.3f} {r['sortino']:>8.3f}"
              f" ${r['pnl']:>10,.0f} {r['dd']:>6.2f}% {r['calmar']:>7.2f}"
              f" {r['pf']:>5.2f} ${r['pnl_mo']:>7,.0f}{baseline}{best}")

    # ── Phase 4: Allocation sensitivity ──
    print(f"\n{'='*90}")
    print("  PHASE 4: Allocation Sensitivity (top 3 combos)")
    print(f"{'='*90}")

    top_3 = [key for key, _ in ranked[:3]]
    caps = [0.30, 0.40, 0.50, 0.60, 0.70, 1.0]

    print(f"\n  {'Combination':<30} {'Cap':>5} {'Sharpe':>8} {'Sortino':>8}"
          f" {'PnL':>12} {'MaxDD':>7} {'$/mo':>8}")
    print("  " + "-" * 85)

    alloc_results = {}
    for combo_key in top_3:
        tfs = combo_key.split("+")
        labels = [tf_best[tf][0] for tf in tfs]
        merged = pd.concat([cache[l] for l in labels], ignore_index=True)
        merged = merged.sort_values("timestamp").reset_index(drop=True)
        best_sr = -999
        best_cap = 0.50
        for cap in caps:
            r = run_sim(merged, CAP, max_per_strategy=cap)
            if r is None:
                continue
            is_best = r["sr"] > best_sr
            if is_best:
                best_sr = r["sr"]
                best_cap = cap
            alloc_results[(combo_key, cap)] = r
            mark = " *" if is_best else ""
            print(f"  {combo_key:<30} {cap:>5.2f} {r['sr']:>8.3f} {r['sortino']:>8.3f}"
                  f" ${r['pnl']:>10,.0f} {r['dd']:>6.2f}% ${r['pnl_mo']:>7,.0f}{mark}")
        print()

    # ── Phase 5: Winner summary ──
    print(f"\n{'='*90}")
    print("  PHASE 5: Winner Summary")
    print(f"{'='*90}")

    # Find overall best (combo, cap)
    best_key = None
    best_overall = -999
    for (combo_key, cap), r in alloc_results.items():
        if r["sr"] > best_overall:
            best_overall = r["sr"]
            best_key = (combo_key, cap)

    if best_key is None:
        print("  No valid results.")
        return

    winner_combo, winner_cap = best_key
    winner = alloc_results[best_key]

    # Also get baseline
    baseline_r = combo_results.get("15m+1h+4h")
    baseline_key = "15m+1h+4h (cap=0.50)"

    print(f"\n  WINNER: {winner_combo} (max_per_strategy={winner_cap:.2f})")
    print(f"    Sharpe:   {winner['sr']:.3f}")
    print(f"    Sortino:  {winner['sortino']:.3f}")
    print(f"    PnL:      ${winner['pnl']:,.0f}")
    print(f"    MaxDD:    {winner['dd']:.2f}%")
    print(f"    Calmar:   {winner['calmar']:.2f}")
    print(f"    PF:       {winner['pf']:.2f}")
    print(f"    $/month:  ${winner['pnl_mo']:,.0f}")
    print(f"    Pos months: {winner['pos_months']}/{winner['tot_months']}")

    if baseline_r:
        delta_sr = winner["sr"] - baseline_r["sr"]
        delta_pnl = winner["pnl"] - baseline_r["pnl"]
        print(f"\n  vs Baseline ({baseline_key}):")
        print(f"    Sharpe delta: {delta_sr:+.3f}")
        print(f"    PnL delta:    ${delta_pnl:+,.0f}")

    # Monthly PnL breakdown
    print(f"\n  MONTHLY PnL BREAKDOWN")
    print(f"  {'Month':<10} {'Winner':>14} {'Baseline':>14}")
    print("  " + "-" * 40)

    w_monthly = winner["monthly"]
    b_monthly = baseline_r["monthly"] if baseline_r else pd.Series(dtype=float)
    all_months = sorted(set(w_monthly.index) | set(b_monthly.index))
    yearly_w, yearly_b = {}, {}
    for m in all_months:
        wv = w_monthly.get(m, 0)
        bv = b_monthly.get(m, 0) if baseline_r else 0
        yearly_w[m.year] = yearly_w.get(m.year, 0) + wv
        yearly_b[m.year] = yearly_b.get(m.year, 0) + bv
        print(f"  {m.strftime('%Y-%m'):<10} ${wv:>12,.0f} ${bv:>12,.0f}")

    print("  " + "-" * 40)
    for yr in sorted(yearly_w.keys()):
        print(f"  {yr:<10} ${yearly_w[yr]:>12,.0f} ${yearly_b.get(yr, 0):>12,.0f}")
    print("  " + "=" * 40)
    print(f"  {'TOTAL':<10} ${winner['pnl']:>12,.0f} ${baseline_r['pnl'] if baseline_r else 0:>12,.0f}")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
