#!/usr/bin/env python
"""Execution Strategy Ranking — Production Pipeline.

Uses the EXACT same signal generation as ofi_features_experiment.py:
  - 30 assets, 15m + 1h + 4h timeframes, Elastic Net
  - Hard confidence filter (>= 0.08)
  - Per-asset vol scaling (0.025/vol)
  - VIP fee progression
  - PortfolioConfig with max_per_strategy=0.50

Then tests maker/taker/hybrid strategies on real BBO data.

Usage:
    PYTHONPATH=src python experiments/execution_strategy_ranking.py [--smoke]
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

SMOKE = "--smoke" in sys.argv
PARQUET_DIR = Path("data/bookTicker_parquet")
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

OHLCV = ["ret_1","ret_3","ret_7","ret_14","bar_range","vol_zscore","rsi_14",
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

TICK_SIZES = {
    "BTCUSDT":0.1,"ETHUSDT":0.01,"SOLUSDT":0.001,"XRPUSDT":0.0001,
    "BNBUSDT":0.01,"ADAUSDT":0.0001,"AVAXUSDT":0.001,"DOTUSDT":0.001,
    "LINKUSDT":0.001,"NEARUSDT":0.001,"UNIUSDT":0.001,"AAVEUSDT":0.01,
    "LTCUSDT":0.01,"BCHUSDT":0.01,"FILUSDT":0.001,"FTMUSDT":0.0001,
    "HBARUSDT":0.00001,"TRXUSDT":0.0001,"SUIUSDT":0.0001,"APTUSDT":0.001,
    "ENJUSDT":0.0001,"SXPUSDT":0.0001,"ZECUSDT":0.01,"XLMUSDT":0.00001,
    "FETUSDT":0.0001,"WLDUSDT":0.001,"BNXUSDT":0.001,"ANKRUSDT":0.00001,
    "TAOUSDT":0.01,"ENAUSDT":0.0001,
}


def get_vip_tier(equity, volume_30d):
    result = VIP_TIERS[0]
    for tier in VIP_TIERS:
        if equity >= tier[1] or volume_30d >= tier[2]:
            result = tier
    return result[0], result[3], result[4]


# ── Signal generation (identical to ofi_features_experiment) ──

def _gen(args):
    name, sym, tf, fwd, C = args
    warnings.filterwarnings("ignore")
    config = SystemConfig(
        model=ModelConfig(C=C, penalty="elasticnet", solver="saga", max_iter=1000,
                          seed=42, fwd_bars=fwd, signal_model_type="asset_logistic"),
        features=FeatureConfig(feature_list=list(OHLCV), winsor_lo=0.01, winsor_hi=0.99,
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
    wf = WalkForwardModel(config, force_monthly_retrain=True)
    sigs = wf.generate_signals_batch(ohlcv_tf, feat, name)
    if len(sigs) == 0: return None
    sigs = apply_regime_filter(sigs, ohlcv_4h, config.regime_filter)
    return sigs if len(sigs) > 0 else None


def generate_all_signals():
    """Generate multi-timeframe signals (identical to ofi_features Baseline EN)."""
    strat_sigs = {}
    tfs = [("15m", "15min", 4, 0.10)] if SMOKE else [
        ("15m", "15min", 4, 0.10), ("1h", "1h", 4, 0.10), ("4h", "4h", 6, 0.20)]
    for sname, tf, fwd, C in tfs:
        tasks = [(n, s, tf, fwd, C) for n, s in ASSETS.items()]
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


# ── Vol scaling (identical to ofi_features_experiment) ──

def _compute_vol_one(args):
    name, sym = args
    warnings.filterwarnings("ignore")
    try:
        ohlcv = resample_ohlcv(load_binance_1m(sym, DATA_DIRS), "1D")["close"]
        if ohlcv.index.tz is not None: ohlcv.index = ohlcv.index.tz_localize(None)
        return name, ohlcv.pct_change().rolling(20).std()
    except Exception:
        return name, None


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


# ── Portfolio sim with VIP fees (identical to ofi_features_experiment) ──

def run_sim(signals_prepared, starting_capital, fee_mode="maker", maker_fill_pct=0.90):
    """Run portfolio sim with VIP fee overlay.

    fee_mode: 'maker' (limit orders), 'taker' (market orders),
              'hybrid' (maker_fill_pct maker + rest taker)
    """
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
    if sim is None: return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None: return None
    if dpnl.index.tz is not None: dpnl = dpnl.copy(); dpnl.index = dpnl.index.tz_localize(None)

    # VIP fee overlay — fee type depends on execution strategy
    active = ~sim["skipped"]
    sizes = sim["sizes"]
    ts_arr = pd.to_datetime(merged["timestamp"].values)
    if ts_arr.tz is not None: ts_arr = ts_arr.tz_localize(None)
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
        if fee_mode == "maker":
            rt_bps = 2 * maker_bps + SPREAD_BPS
        elif fee_mode == "taker":
            rt_bps = 2 * taker_bps + SPREAD_BPS
        else:  # hybrid
            rt_bps = (maker_fill_pct * 2 * maker_bps +
                      (1 - maker_fill_pct) * 2 * taker_bps + SPREAD_BPS)
        extra_bps = rt_bps - 1.0  # extra cost above 1bps slippage already in sim
        n_day = sum(1 for i in range(len(merged)) if active[i] and dates_arr[i] == d_date)
        avg_sz = daily_vol.get(d_date, 0) / max(n_day * 2, 1)
        dpnl_adj[d] -= n_day * avg_sz * extra_bps / 10000

    dr = dpnl_adj / starting_capital
    if dr.std() == 0 or len(dr) < 30: return None
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0
    eq_adj = (1 + dr).cumprod()
    dd = float(((eq_adj.cummax() - eq_adj) / eq_adj.cummax()).max() * 100)
    calmar = float(dr.mean() * 365.25 / (dd / 100)) if dd > 0 else 0
    pnl = float(dpnl_adj.sum())
    pf = float(dpnl_adj[dpnl_adj > 0].sum() / abs(dpnl_adj[dpnl_adj < 0].sum())) if (dpnl_adj < 0).any() else 99
    monthly = dpnl_adj.resample("MS").sum()
    return {"sr": sr, "sortino": sortino, "dd": dd, "calmar": calmar,
            "pnl": pnl, "pf": pf, "n_trades": sim["n_trades"], "monthly": monthly}


# ── BBO fill checking ──

def _check_maker(bts, bbid, bask, times_ns, dirs, timeout_ns, tick):
    n = len(dirs)
    filled = np.zeros(n, dtype=bool)
    limit_px = np.full(n, np.nan)
    for i in range(n):
        s = np.searchsorted(bts, times_ns[i], side="right") - 1
        if s < 0 or s >= len(bbid): continue
        e = np.searchsorted(bts, times_ns[i] + timeout_ns, side="right")
        if dirs[i] > 0:
            lp = bbid[s]
            if np.any(bask[s:e] <= lp + tick * 0.01):
                filled[i] = True; limit_px[i] = lp
        else:
            lp = bask[s]
            if np.any(bbid[s:e] >= lp - tick * 0.01):
                filled[i] = True; limit_px[i] = lp
    return filled, limit_px


def _check_reprice(bts, bbid, bask, times_ns, dirs, int_ns, tot_ns, tick):
    n = len(dirs)
    filled = np.zeros(n, dtype=bool)
    fill_px = np.full(n, np.nan)
    for i in range(n):
        t = times_ns[i]; t_end = t + tot_ns
        while t < t_end:
            s = np.clip(np.searchsorted(bts, t, side="right") - 1, 0, len(bbid) - 1)
            e = np.searchsorted(bts, min(t + int_ns, t_end), side="right")
            if dirs[i] > 0:
                lp = bbid[s]
                if np.any(bask[s:e] <= lp + tick * 0.01):
                    filled[i] = True; fill_px[i] = lp; break
            else:
                lp = bask[s]
                if np.any(bbid[s:e] >= lp - tick * 0.01):
                    filled[i] = True; fill_px[i] = lp; break
            t = min(t + int_ns, t_end)
    return filled, fill_px


def analyze_chunk(bts, bbid, bask, dirs, sizes, fwd_rets, edges,
                  sig_ns, exit_ns, maker_fee, taker_fee, tick):
    n = len(dirs)
    if n == 0: return None
    si = np.clip(np.searchsorted(bts, sig_ns, side="right") - 1, 0, len(bbid) - 1)
    xi = np.clip(np.searchsorted(bts, exit_ns, side="right") - 1, 0, len(bbid) - 1)
    entry_mid = (bbid[si] + bask[si]) / 2.0
    entry_taker = np.where(dirs > 0, bask[si], bbid[si])
    exit_taker = np.where(dirs > 0, bbid[xi], bask[xi])
    qty = sizes / entry_mid
    mf, tf = maker_fee, taker_fee

    e30, ep30 = _check_maker(bts, bbid, bask, sig_ns, dirs, 30_000_000_000, tick)
    e60, ep60 = _check_maker(bts, bbid, bask, sig_ns, dirs, 60_000_000_000, tick)
    x30, xp30 = _check_maker(bts, bbid, bask, exit_ns, dirs * -1, 30_000_000_000, tick)
    x60, xp60 = _check_maker(bts, bbid, bask, exit_ns, dirs * -1, 60_000_000_000, tick)
    er, erp = _check_reprice(bts, bbid, bask, sig_ns, dirs, 10_000_000_000, 60_000_000_000, tick)
    xr, xrp = _check_reprice(bts, bbid, bask, exit_ns, dirs * -1, 10_000_000_000, 60_000_000_000, tick)

    def _pnl(e_px, e_fee, x_px, x_fee, mask=None):
        if mask is None: mask = np.ones(n, dtype=bool)
        if mask.sum() == 0: return 0.0, 0.0, 0
        d, q = dirs[mask], qty[mask]
        ep, xp = e_px[mask], x_px[mask]
        ef = e_fee[mask] if isinstance(e_fee, np.ndarray) else np.full(mask.sum(), e_fee)
        xf = x_fee[mask] if isinstance(x_fee, np.ndarray) else np.full(mask.sum(), x_fee)
        gross = np.where(d > 0, (xp - ep) * q, (ep - xp) * q)
        fees = ef * ep * q + xf * xp * q
        return float((gross - fees).sum()), float(fees.sum()), int(mask.sum())

    results = {}
    # Taker/Taker
    net, fee, cnt = _pnl(entry_taker, tf, exit_taker, tf)
    results["Taker/Taker"] = (net, fee, cnt, n)
    # Maker60/Maker60
    both60 = e60 & x60
    net, fee, cnt = _pnl(ep60, mf, xp60, mf, both60)
    results["Maker60/Maker60"] = (net, fee, cnt, n)
    # MkrTkr30
    net, fee, cnt = _pnl(np.where(e30, ep30, entry_taker), np.where(e30, mf, tf),
                         np.where(x30, xp30, exit_taker), np.where(x30, mf, tf))
    results["MkrTkr30/MkrTkr30"] = (net, fee, cnt, n)
    # MkrTkr60
    net, fee, cnt = _pnl(np.where(e60, ep60, entry_taker), np.where(e60, mf, tf),
                         np.where(x60, xp60, exit_taker), np.where(x60, mf, tf))
    results["MkrTkr60/MkrTkr60"] = (net, fee, cnt, n)
    # Reprice
    net, fee, cnt = _pnl(np.where(er, erp, entry_taker), np.where(er, mf, tf),
                         np.where(xr, xrp, exit_taker), np.where(xr, mf, tf))
    results["Reprice10/60"] = (net, fee, cnt, n)
    # Maker60skip/Taker
    net, fee, cnt = _pnl(ep60, mf, exit_taker, tf, e60)
    results["Maker60skip/Taker"] = (net, fee, cnt, n)
    # EdgeRoute
    med = np.median(edges) if n > 1 else 0.5
    hi = edges > med
    net, fee, cnt = _pnl(
        np.where(hi, entry_taker, np.where(e60, ep60, entry_taker)),
        np.where(hi, tf, np.where(e60, mf, tf)),
        np.where(hi, exit_taker, np.where(x60, xp60, exit_taker)),
        np.where(hi, tf, np.where(x60, mf, tf)))
    results["EdgeRoute(med)"] = (net, fee, cnt, n)

    adv = {}
    if e60.sum() > 0 and (~e60).sum() > 0:
        adv["filled_ret"] = float((dirs[e60] * fwd_rets[e60]).mean() * 10000)
        adv["missed_ret"] = float((dirs[~e60] * fwd_rets[~e60]).mean() * 10000)
        adv["filled_n"] = int(e60.sum())
        adv["missed_n"] = int((~e60).sum())
    fills = {k: float(v.mean()) for k, v in [
        ("e30", e30), ("x30", x30), ("e60", e60), ("x60", x60), ("erp", er), ("xrp", xr)]}
    return results, adv, fills


# ── Main ──

def main():
    T0 = time.time()
    print("=" * 80)
    print("  EXECUTION STRATEGY RANKING -- Production Pipeline")
    print("  30 assets, 15m+1h+4h, EN, vol scaling, VIP fees")
    print("=" * 80)

    # 1. Generate multi-timeframe signals
    print("\n  Generating signals (multi-timeframe)...", flush=True)
    t0 = time.time()
    strat_sigs = generate_all_signals()
    combined = pd.concat(list(strat_sigs.values()), ignore_index=True)
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    total = len(combined)
    print(f"  Total: {total:,} signals ({time.time()-t0:.0f}s)", flush=True)

    # 2. Vol scaling
    print("\n  Computing per-asset vol...", flush=True)
    t0 = time.time()
    vol_series = compute_vol_parallel()
    combined = apply_vol(combined, vol_series)
    print(f"  Done ({time.time()-t0:.0f}s)", flush=True)

    # 3. Production baseline (Sharpe with VIP maker fees)
    print(f"\n{'='*80}")
    print("  PRODUCTION BASELINE (1bps slippage + VIP maker fees)")
    print(f"{'='*80}")
    baseline = run_sim(combined, CAP, fee_mode="maker")
    if baseline:
        print(f"  Sharpe={baseline['sr']:.3f}  Sortino={baseline['sortino']:.3f}"
              f"  PnL=${baseline['pnl']:,.0f}  MaxDD={baseline['dd']:.2f}%"
              f"  Trades={baseline['n_trades']:,}")

    # 4. BBO strategy comparison
    print(f"\n{'='*80}")
    print("  BBO STRATEGY COMPARISON (BTC+ETH overlap period)")
    print(f"{'='*80}")

    bbo_assets = {}
    for sym in TICK_SIZES:
        pq_dir = PARQUET_DIR / sym
        if pq_dir.exists():
            dates = sorted(f.stem for f in pq_dir.glob("*.parquet"))
            if dates: bbo_assets[sym] = dates

    print(f"\n  BBO data: {', '.join(f'{s}({len(d)}d)' for s, d in bbo_assets.items())}")

    _, maker_bps, taker_bps = get_vip_tier(CAP, 0)
    mf_dec = maker_bps / 10000
    tf_dec = taker_bps / 10000
    print(f"  VIP tier at ${CAP:,}: maker={maker_bps:.1f}bps taker={taker_bps:.1f}bps")

    # Filter signals to BBO overlap period
    bbo_sigs = {}
    for sym, dates in bbo_assets.items():
        # Find matching signals (across all timeframes)
        name = sym.replace("USDT", "")
        asset_sigs = combined[combined["asset"] == name].copy()
        if len(asset_sigs) == 0: continue
        ts = pd.to_datetime(asset_sigs["timestamp"])
        if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
        start, end = pd.Timestamp(dates[0]), pd.Timestamp(dates[-1]) + pd.Timedelta(days=1)
        mask = (ts >= start) & (ts < end)
        filtered = asset_sigs[mask.values].reset_index(drop=True)
        if len(filtered) >= 5: bbo_sigs[sym] = filtered

    bbo_total = sum(len(s) for s in bbo_sigs.values())
    print(f"  BBO overlap signals: {bbo_total:,}")

    agg = {}
    agg_adv = {"fr": [], "mr": [], "fn": 0, "mn": 0}
    agg_fill = {}

    for sym, sigs in sorted(bbo_sigs.items()):
        dates = bbo_assets[sym]
        tick = TICK_SIZES.get(sym, 0.01)
        sigs_s = sigs.sort_values("timestamp").reset_index(drop=True)
        ts_col = pd.to_datetime(sigs_s["timestamp"])
        if ts_col.dt.tz is not None: ts_col = ts_col.dt.tz_localize(None)

        # Hold period depends on strategy timeframe
        hold_bars = sigs_s["hold_bars"].values.astype(np.int64)
        # Compute per-signal hold in ns
        # 15m: hold_bars=4 → 4*15*60*1e9, 1h: hold_bars=4 → 4*60*60*1e9, 4h: hold_bars=6 → 6*4*60*60*1e9
        strategy_col = sigs_s["strategy"].values
        hold_ns_arr = np.zeros(len(sigs_s), dtype=np.int64)
        for i in range(len(sigs_s)):
            hb = hold_bars[i]
            st = strategy_col[i]
            if st == "15m":
                hold_ns_arr[i] = hb * 15 * 60 * 1_000_000_000
            elif st == "1h":
                hold_ns_arr[i] = hb * 60 * 60 * 1_000_000_000
            elif st == "4h":
                hold_ns_arr[i] = hb * 4 * 60 * 60 * 1_000_000_000
            else:
                hold_ns_arr[i] = hb * 15 * 60 * 1_000_000_000

        sig_ns = ts_col.values.astype(np.int64)
        exit_ns = sig_ns + hold_ns_arr
        dirs = sigs_s["direction"].values.astype(np.float64)
        fwd = sigs_s["fwd_return"].values.astype(np.float64)
        prob = sigs_s["probability"].values.astype(np.float64)
        edges = np.clip(np.abs(2.0 * prob - 1.0), 0, 1)
        sizing_mult = sigs_s["sizing_mult"].values.astype(np.float64) if "sizing_mult" in sigs_s.columns else np.ones(len(sigs_s))
        sizes = (CAP * 0.045 * edges * sizing_mult).astype(np.float64)

        sig_dates = sorted(set(str(t.date()) for t in ts_col))
        n = len(sigs_s)
        print(f"\n  {sym}: {n} signals on {len(sig_dates)} days", flush=True)

        chunk_size = 3 if SMOKE else 7
        processed = 0
        for ci in range(0, len(sig_dates), chunk_size):
            chunk_dates = sig_dates[ci:ci + chunk_size]
            needed = set(chunk_dates)
            for d in chunk_dates:
                needed.add(str((pd.Timestamp(d) + pd.Timedelta(days=2)).date()))
                needed.add(str((pd.Timestamp(d) + pd.Timedelta(days=1)).date()))

            first_ns = pd.Timestamp(chunk_dates[0]).value
            last_ns = pd.Timestamp(chunk_dates[-1]).value + 86_400_000_000_000
            mask = (sig_ns >= first_ns) & (sig_ns < last_ns)
            idx = np.where(mask)[0]
            if len(idx) == 0: continue

            chunks = []
            for d in sorted(needed):
                pq = PARQUET_DIR / sym / f"{d}.parquet"
                if pq.exists(): chunks.append(pd.read_parquet(pq))
            if not chunks: continue
            bbo = pd.concat(chunks, ignore_index=True).sort_values("transaction_time")
            bts = bbo["transaction_time"].values.astype(np.int64) * 1_000_000
            bbid = bbo["best_bid_price"].values.astype(np.float64)
            bask = bbo["best_ask_price"].values.astype(np.float64)
            del bbo, chunks; gc.collect()

            res = analyze_chunk(bts, bbid, bask, dirs[idx], sizes[idx], fwd[idx],
                                edges[idx], sig_ns[idx], exit_ns[idx], mf_dec, tf_dec, tick)
            del bts, bbid, bask; gc.collect()
            if res is None: continue
            cr, ca, cf = res

            for name, vals in cr.items():
                if name not in agg: agg[name] = [0.0, 0.0, 0, 0]
                agg[name][0] += vals[0]; agg[name][1] += vals[1]
                agg[name][2] += vals[2]; agg[name][3] += vals[3]
            if ca:
                agg_adv["fr"].append(ca.get("filled_ret", 0))
                agg_adv["mr"].append(ca.get("missed_ret", 0))
                agg_adv["fn"] += ca.get("filled_n", 0)
                agg_adv["mn"] += ca.get("missed_n", 0)
            for k, v in cf.items():
                agg_fill.setdefault(k, []).append(v)
            processed += len(idx)
        print(f"    Processed: {processed}/{n}", flush=True)

    # 5. Strategy ranking
    print(f"\n{'='*80}")
    print("  STRATEGY RANKING (BBO overlap, VIP2 fees)")
    print(f"{'='*80}")

    ranked = sorted(agg.items(), key=lambda x: x[1][0], reverse=True)
    taker_net = agg.get("Taker/Taker", [0])[0]
    taker_fees = agg.get("Taker/Taker", [0, 0])[1]
    total_notional = taker_fees / (2 * tf_dec) if tf_dec > 0 and taker_fees > 0 else 1

    print(f"\n  {'#':>2} {'Strategy':<26} {'Traded':>7} {'Net PnL':>10} {'Fees':>10}"
          f" {'Fee bps':>8} {'vs Taker':>10}")
    print("  " + "-" * 78)
    for rank, (name, vals) in enumerate(ranked, 1):
        net, fee, cnt, nsig = vals
        not_est = total_notional * cnt / max(nsig, 1) if nsig > 0 else 1
        fbps = fee / not_est * 10000 if not_est > 0 else 0
        vs = net - taker_net
        best = rank == 1
        print(f"  {rank:>2} {name:<26} {cnt:>7} ${net:>9,.0f} ${fee:>9,.0f}"
              f" {fbps:>7.1f} ${vs:>+9,.0f}{' <-- BEST' if best else ''}")

    # Adverse selection
    if agg_adv["fn"] > 0 and agg_adv["mn"] > 0:
        fr = np.mean(agg_adv["fr"]); mr = np.mean(agg_adv["mr"])
        print(f"\n  Adverse Selection (maker60 entry):")
        print(f"    Filled ({agg_adv['fn']:>4}): avg fwd_ret = {fr:>+.2f} bps")
        print(f"    Missed ({agg_adv['mn']:>4}): avg fwd_ret = {mr:>+.2f} bps")
        print(f"    Gap: {fr - mr:>+.1f} bps ({'ADVERSE' if fr < mr else 'OK'})")

    # Fill rates
    print(f"\n  Fill rates:")
    for label, key in [("Entry maker30", "e30"), ("Exit maker30", "x30"),
                       ("Entry maker60", "e60"), ("Exit maker60", "x60"),
                       ("Entry reprice", "erp"), ("Exit reprice", "xrp")]:
        if key in agg_fill:
            print(f"    {label:<16}: {np.mean(agg_fill[key]):.0%}")

    # 6. Full portfolio projection (different slippage assumptions)
    print(f"\n{'='*80}")
    print("  FULL PORTFOLIO PROJECTION (all assets, all timeframes, VIP fees)")
    print(f"{'='*80}")

    # Each scenario uses correct fee model (maker/taker/hybrid) in VIP overlay
    scenarios = [
        ("Maker", "maker", 1.0),
        ("MkrTkr60 (90%mkr)", "hybrid", 0.90),
        ("MkrTkr30 (85%mkr)", "hybrid", 0.85),
        ("Hybrid 50/50", "hybrid", 0.50),
        ("Taker", "taker", 1.0),
    ]
    sim_results = {}
    print(f"\n  {'Scenario':<22} {'Sharpe':>8} {'Sortino':>8} {'PnL':>12}"
          f" {'MaxDD':>7} {'Calmar':>7} {'PF':>6} {'$/mo':>8}")
    print("  " + "-" * 82)
    for label, mode, fill_pct in scenarios:
        s = run_sim(combined, CAP, fee_mode=mode, maker_fill_pct=fill_pct)
        if s is None: continue
        sim_results[label] = s
        n_months = len(s["monthly"]) if s["monthly"] is not None else 1
        pnl_mo = s["pnl"] / max(n_months, 1)
        print(f"  {label:<22} {s['sr']:>8.3f} {s['sortino']:>8.3f}"
              f" ${s['pnl']:>10,.0f} {s['dd']:>6.2f}% {s['calmar']:>7.2f}"
              f" {s['pf']:>5.2f} ${pnl_mo:>7,.0f}")

    # 7. Monthly PnL breakdown for key scenarios
    print(f"\n{'='*80}")
    print("  MONTHLY PnL BREAKDOWN")
    print(f"{'='*80}")

    show = ["Maker", "MkrTkr60 (90%mkr)", "Taker"]
    show = [s for s in show if s in sim_results]
    if show:
        # Collect all months across scenarios
        all_months = sorted(set().union(*(
            sim_results[s]["monthly"].index for s in show if sim_results[s]["monthly"] is not None)))

        header = f"  {'Month':<10}"
        for s in show:
            header += f" {s:>18}"
        print(f"\n{header}")
        print("  " + "-" * (10 + 19 * len(show)))

        yearly = {s: {} for s in show}
        for m in all_months:
            row = f"  {m.strftime('%Y-%m'):<10}"
            for s in show:
                monthly = sim_results[s]["monthly"]
                val = monthly.get(m, 0) if monthly is not None else 0
                row += f" ${val:>16,.0f}"
                yr = m.year
                yearly[s][yr] = yearly[s].get(yr, 0) + val
            print(row)

        # Year totals
        all_years = sorted(set().union(*(yearly[s].keys() for s in show)))
        print("  " + "-" * (10 + 19 * len(show)))
        for yr in all_years:
            row = f"  {yr:<10}"
            for s in show:
                val = yearly[s].get(yr, 0)
                row += f" ${val:>16,.0f}"
            print(row)

        # Grand total
        print("  " + "=" * (10 + 19 * len(show)))
        row = f"  {'TOTAL':<10}"
        for s in show:
            row += f" ${sim_results[s]['pnl']:>16,.0f}"
        print(row)

        # Positive months
        row = f"  {'Pos months':<10}"
        for s in show:
            monthly = sim_results[s]["monthly"]
            pos = int((monthly > 0).sum()); tot = len(monthly)
            row += f" {pos:>13}/{tot:<3}"
        print(row)

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
