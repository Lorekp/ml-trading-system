#!/usr/bin/env python
"""Dynamic Asset Selection V2 — Multiple approaches to improve Dynamic-30.

A: Multi-factor scoring (volume + spread + stability + diversification)
B: Volume regime filter (asset must be above own historical median)
C: Rebalance frequency (weekly, bi-monthly, buffer zone)
D: Lookback window (14d, 60d vs default 30d)
E: Combined best

All use Full5 + F1 breaker. Signals loaded from disk cache.
Supports --realistic flag for MkrTkr60 BBO-based execution costs.

Usage:
    PYTHONPATH="src;." python experiments/dynamic_asset_v2_experiment.py [--smoke] [--realistic]
"""
import sys, os, time, warnings, gc, math, pickle
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from signal_engine.data import load_binance_1m, resample_ohlcv
from signal_engine.features import compute_features
from signal_engine.asset_selection import (
    compute_monthly_dollar_volume, compute_asset_score,
    select_top_n_for_month, select_assets_for_month, diversification_filter,
)
from experiments.circuit_breaker_v2_experiment import (
    NEEDED_TFS, FULL5, CAP, VIP_TIERS, SPREAD_BPS,
    generate_signals_cached, recompute_returns, apply_vol, _compute_vol_one,
    simulate_with_breaker, get_vip_tier,
)
from signal_engine.portfolio import compute_daily_pnl_series
from signal_engine.config import *

SMOKE = "--smoke" in sys.argv
REALISTIC = "--realistic" in sys.argv
DATA_DIRS = [Path("data/binance-1m"), Path("data/homeserver-mirror/binance-1m")]
FULL5_HOLDS = {"15m": 2, "1h": 4, "3h": 16, "4h": 6, "24h": 12}
CACHE_DIR = Path("experiments/.cache")

# OVERFIT — uses assets selected with hindsight. Kept only as upper bound.
FIXED_30_OVERFIT = {
    "BTC","ETH","SOL","XRP","BNB","ADA","AVAX","DOT","LINK","NEAR",
    "UNI","AAVE","LTC","BCH","HBAR","TRX","SUI","APT","ENJ","SXP",
    "ZEC","XLM","FET","WLD","ANKR","TAO","ENA","FTM","FIL","BNX",
}

# FAIR — assets that existed at backtest start (July 2020), ranked by volume at that time.
# No lookahead. Includes dead assets (ONT, BAT, IOST) because we didn't know they'd fade.
FAIR_FIXED = {
    "BTC","ETH","ADA","BCH","LINK","VET","XRP","XTZ","LTC","IOST",
    "ETC","TRX","BNB","ONT","XLM","ZEC","BAT","NEO","THETA","ATOM",
    "XMR","IOTA","QTUM","DASH",
}

# OBVIOUS CORE — assets anyone would have traded in mid-2020. No hindsight needed.
OBVIOUS_CORE = {"BTC","ETH","XRP","ADA","LINK","LTC","BNB","BCH","ETC"}


def run_sim(merged, vol_series, base_sizing=0.045, symmetric=None, max_signal_mult=2.0):
    """Run F1 breaker sim with execution cost modeling.

    Args:
        symmetric: None (F1 down-only) or dict with keys:
            up_threshold, down_threshold, cap — for symmetric adaptive sizing
        base_sizing: base position sizing percentage
        max_signal_mult: max sizing multiplier from signal strength
    """
    prepared = apply_vol(merged, vol_series)
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in prepared.columns: prepared[c] = 1.0
        prepared[c] = prepared[c].fillna(1.0)
    prepared["probability"] = prepared["probability"].astype(float)
    prepared = prepared.sort_values("timestamp").reset_index(drop=True)

    use_realistic = REALISTIC and "exec_cost_bps" in prepared.columns

    if use_realistic:
        costs = prepared["exec_cost_bps"].fillna(0).values
        prepared["fwd_return"] = prepared["fwd_return"].astype(float) - costs / 10000
        slippage = 0.0
    else:
        slippage = 0.0001

    port = PortfolioConfig(
        starting_capital=CAP, max_dd=0.10, delever_start=0.05, slippage=slippage,
        max_exposure=0.90, max_per_asset=0.06, max_per_strategy=0.50,
        base_sizing_pct=base_sizing, max_signal_mult=max_signal_mult,
        sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)

    # Breaker config: F1 down-only, or symmetric (scale up winners too)
    breaker_type = "pnl_ratio"
    breaker_params = {"window": 10, "threshold": -0.01, "floor": 0.10}
    if symmetric:
        breaker_type = "pnl_ratio_symmetric"
        breaker_params = {
            "window": 10,
            "threshold": symmetric.get("down_threshold", -0.01),
            "floor": 0.10,
            "up_threshold": symmetric.get("up_threshold", 0.01),
            "cap": symmetric.get("cap", 1.5),
        }

    sim = simulate_with_breaker(prepared, port, breaker_type, breaker_params)
    if sim is None: return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None: return None
    if dpnl.index.tz is not None: dpnl = dpnl.copy(); dpnl.index = dpnl.index.tz_localize(None)

    if use_realistic:
        # Costs already in fwd_return — no post-hoc adjustment needed
        dpnl_adj = dpnl
    else:
        # Original post-hoc VIP fee overlay
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


def _score_one(args):
    """Compute volumes + scores for one asset (module-level for pickling)."""
    name, sym = args
    warnings.filterwarnings("ignore")
    try:
        ohlcv_1m = load_binance_1m(sym, DATA_DIRS)
        ohlcv_15m = resample_ohlcv(ohlcv_1m, "15min")
        if len(ohlcv_15m) < 5000: return name, None
        vol30 = compute_monthly_dollar_volume(ohlcv_15m)
        dv = (ohlcv_15m["close"] * ohlcv_15m["volume"]).resample("1D").sum()
        vol14 = dv.rolling(14, min_periods=10).median()
        vol60 = dv.rolling(60, min_periods=40).median()
        for v in [vol30, vol14, vol60]:
            if v.index.tz is not None: v.index = v.index.tz_localize(None)
        first = ohlcv_15m.index[0]
        if hasattr(first, 'tz') and first.tz is not None: first = first.tz_localize(None)
        scores = compute_asset_score(ohlcv_15m)
        for k, v in scores.items():
            if v.index.tz is not None: scores[k] = v.tz_localize(None)
        close = ohlcv_15m["close"]
        dr = close.resample("1D").last().pct_change().dropna()
        if dr.index.tz is not None: dr.index = dr.index.tz_localize(None)
        return name, {"vol30": vol30, "vol14": vol14, "vol60": vol60,
                      "first": first, "scores": scores, "dr": dr}
    except: return name, None


def _recompute_tf(args):
    """Recompute hold returns for one TF (module-level for pickling)."""
    tf_label, tf_resample, asset_sigs_tf = args
    warnings.filterwarnings("ignore")
    hold = {"15m": 2, "1h": 4, "3h": 16, "4h": 6, "24h": 12}[tf_label]
    if not asset_sigs_tf: return tf_label, hold, {}
    result = {}
    if hold == 4:
        for n, sigs in asset_sigs_tf.items():
            s = sigs.copy(); s["hold_bars"] = hold
            result[n] = s
    else:
        all_s = pd.concat(list(asset_sigs_tf.values()), ignore_index=True)
        fixed = recompute_returns(all_s, tf_resample, hold)
        for n in asset_sigs_tf:
            mask = fixed["asset"] == n
            s = fixed[mask].reset_index(drop=True)
            if len(s) > 0: result[n] = s
    return tf_label, hold, result


def assemble_signals(hold_sigs, schedule_func, months):
    """Given a schedule_func(month_start) -> list[asset_names], assemble signals."""
    parts = []
    for month in months:
        ms = pd.Timestamp(month.start_time)
        selected = schedule_func(ms)
        me = ms + pd.offsets.MonthEnd(0) + pd.Timedelta(days=1)
        for name in selected:
            for tf_label, _ in NEEDED_TFS:
                if name not in hold_sigs or tf_label not in hold_sigs[name]: continue
                s = hold_sigs[name][tf_label]
                if "timeframe" not in s.columns:
                    s = s.copy()
                    s["timeframe"] = tf_label
                ts = pd.to_datetime(s["timestamp"])
                if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
                mask = (ts >= ms) & (ts < me)
                s_m = s[mask.values]
                if len(s_m) > 0: parts.append(s_m)
    if not parts: return None
    merged = pd.concat(parts, ignore_index=True)
    return merged.drop_duplicates(subset=["timestamp", "asset", "strategy"]).reset_index(drop=True)


def assemble_weekly(hold_sigs, schedule_func, start, end):
    """Weekly rebalance version."""
    parts = []
    weeks = pd.date_range(start, end, freq="W-MON")
    for ws in weeks:
        selected = schedule_func(ws)
        we = ws + pd.Timedelta(days=7)
        for name in selected:
            for tf_label, _ in NEEDED_TFS:
                if name not in hold_sigs or tf_label not in hold_sigs[name]: continue
                s = hold_sigs[name][tf_label]
                if "timeframe" not in s.columns:
                    s = s.copy()
                    s["timeframe"] = tf_label
                ts = pd.to_datetime(s["timestamp"])
                if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
                mask = (ts >= ws) & (ts < we)
                s_w = s[mask.values]
                if len(s_w) > 0: parts.append(s_w)
    if not parts: return None
    merged = pd.concat(parts, ignore_index=True)
    return merged.drop_duplicates(subset=["timestamp", "asset", "strategy"]).reset_index(drop=True)


def main():
    T0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    exec_mode = "MkrTkr60 REALISTIC" if REALISTIC else "Simplified overlay"
    print("=" * 110)
    print(f"  DYNAMIC ASSET SELECTION V2 ({mode})")
    print(f"  Full5 + F1 breaker, 15 variants, Execution: {exec_mode}")
    print("=" * 110)

    # ── Phase 1: Load cached signals (direct parquet, no ASSETS filter) ──
    print("\n  Loading cached signals...", flush=True)
    all_symbols = {}
    for d in DATA_DIRS:
        if d.exists():
            for sd in sorted(d.iterdir()):
                if sd.is_dir() and sd.name.endswith("USDT"):
                    name = sd.name.replace("USDT", "")
                    if name not in all_symbols:
                        all_symbols[name] = sd.name
    if SMOKE: all_symbols = {k: v for k, v in list(all_symbols.items())[:15]}

    signal_cache = {}
    for label, resample in NEEDED_TFS:
        cache_path = CACHE_DIR / f"sigs_{label}_fwd4.parquet"
        if cache_path.exists():
            sigs = pd.read_parquet(cache_path)
            if SMOKE and "asset" in sigs.columns:
                sigs = sigs[sigs["asset"].isin(all_symbols.keys())].reset_index(drop=True)
            if len(sigs) > 0:
                signal_cache[label] = sigs
    print(f"    {len(signal_cache)} TFs loaded, {sum(len(v) for v in signal_cache.values()):,} total signals")

    # Per-asset signal dict
    asset_sigs = defaultdict(dict)
    for tf_label in signal_cache:
        df = signal_cache[tf_label]
        for name in df["asset"].unique():
            mask = df["asset"] == name
            asset_sigs[name][tf_label] = df[mask].reset_index(drop=True)

    # ── Phases 2-4: Cached (scores + holds + vol series) ──
    prep_cache = CACHE_DIR / "v2_prep_cache.pkl"
    if prep_cache.exists() and "--no-cache" not in sys.argv:
        print("\n  Loading cached prep data...", flush=True)
        t1 = time.time()
        with open(prep_cache, "rb") as f:
            cached = pickle.load(f)
        asset_volumes = cached["asset_volumes"]
        asset_volumes_14d = cached["asset_volumes_14d"]
        asset_volumes_60d = cached["asset_volumes_60d"]
        asset_first_dates = cached["asset_first_dates"]
        asset_scores = cached["asset_scores"]
        daily_returns = cached["daily_returns"]
        hold_sigs = cached["hold_sigs"]
        vol_series = cached["vol_series"]
        print(f"    Loaded {len(asset_scores)} assets, {len(hold_sigs)} with signals ({time.time()-t1:.0f}s)")
    else:
        # Phase 2: Asset scores
        print("\n  Computing asset scores (volume + spread + stability)...", flush=True)
        t1 = time.time()
        asset_volumes, asset_volumes_14d, asset_volumes_60d = {}, {}, {}
        asset_first_dates, asset_scores, daily_returns = {}, {}, {}
        with ProcessPoolExecutor(max_workers=12) as pool:
            futs = {pool.submit(_score_one, (n, s)): n for n, s in sorted(all_symbols.items())}
            for f in as_completed(futs):
                name, result = f.result()
                if result is None: continue
                asset_volumes[name] = result["vol30"]
                asset_volumes_14d[name] = result["vol14"]
                asset_volumes_60d[name] = result["vol60"]
                asset_first_dates[name] = result["first"]
                asset_scores[name] = result["scores"]
                daily_returns[name] = result["dr"]
        print(f"    {len(asset_scores)} assets scored ({time.time()-t1:.0f}s)")

        # Phase 3: Hold returns
        print("\n  Recomputing hold returns...", flush=True)
        t1 = time.time()
        hold_sigs = defaultdict(dict)
        for tf_label, tf_resample in NEEDED_TFS:
            hold = FULL5_HOLDS[tf_label]
            assets_this_tf = [n for n in asset_sigs if tf_label in asset_sigs[n]]
            if not assets_this_tf: continue
            if hold == 4:
                for n in assets_this_tf:
                    s = asset_sigs[n][tf_label].copy(); s["hold_bars"] = hold
                    hold_sigs[n][tf_label] = s
            else:
                all_s = pd.concat([asset_sigs[n][tf_label] for n in assets_this_tf], ignore_index=True)
                fixed = recompute_returns(all_s, tf_resample, hold)
                for n in assets_this_tf:
                    mask = fixed["asset"] == n
                    s = fixed[mask].reset_index(drop=True)
                    if len(s) > 0: hold_sigs[n][tf_label] = s
            print(f"    {tf_label} (h={hold}): {len(assets_this_tf)} assets", flush=True)

        # Phase 4: Vol series
        vol_pairs = [(n, all_symbols[n]) for n in asset_scores.keys() if n in all_symbols]
        vol_series = {}
        with ProcessPoolExecutor(max_workers=12) as pool:
            for name, vs in pool.map(_compute_vol_one, vol_pairs):
                if vs is not None: vol_series[name] = vs
        print(f"    Done ({time.time()-t1:.0f}s)")

        # Save cache
        print("  Saving prep cache...", flush=True)
        with open(prep_cache, "wb") as f:
            pickle.dump({"asset_volumes": asset_volumes, "asset_volumes_14d": asset_volumes_14d,
                         "asset_volumes_60d": asset_volumes_60d, "asset_first_dates": asset_first_dates,
                         "asset_scores": asset_scores, "daily_returns": daily_returns,
                         "hold_sigs": dict(hold_sigs), "vol_series": vol_series}, f)
        print(f"    Saved to {prep_cache}")

    # ── Phase 4b: Precompute execution costs (if realistic mode) ──
    exec_cache_path = CACHE_DIR / "v2_exec_costs.pkl"
    if REALISTIC and exec_cache_path.exists() and "--no-cache" not in sys.argv:
        print("\n  Loading cached execution costs...", flush=True)
        t1 = time.time()
        with open(exec_cache_path, "rb") as f:
            exec_cached = pickle.load(f)
        for name in hold_sigs:
            for tf_label in hold_sigs[name]:
                key = (name, tf_label)
                if key in exec_cached:
                    hold_sigs[name][tf_label] = hold_sigs[name][tf_label].copy()
                    hold_sigs[name][tf_label]["exec_cost_bps"] = exec_cached[key]
        print(f"    Loaded ({time.time()-t1:.0f}s)")
    elif REALISTIC:
        print("\n  Precomputing execution costs (per-asset BBO + calibrated fallback)...", flush=True)
        t1 = time.time()
        from signal_engine.execution import (
            check_maker_fill, get_vip_tier, load_calibration,
            BBO_DIRS, DEFAULT_TICK_SIZES)

        calibration = load_calibration()
        _, maker_bps, taker_bps = get_vip_tier(CAP)

        # Build per-symbol calibration lookup from calibration file
        per_sym_cal = {}
        for entry in calibration.get("per_symbol", []):
            per_sym_cal[entry["symbol"]] = {
                "fill_rate": entry["fill_rate"],
                "spread_bps": entry["spread_median_bps"],
            }

        # Tier-based fallback for symbols not in calibration
        tier_cal = calibration.get("calibration", calibration)

        n_bbo_exact, n_bbo_fallback, n_stat, n_total = 0, 0, 0, 0
        all_costs = []

        for name in hold_sigs:
            sym = f"{name}USDT"

            # Find which dates this asset's signals span
            all_sig_dates = set()
            for tf_label in hold_sigs[name]:
                ts = pd.to_datetime(hold_sigs[name][tf_label]["timestamp"].values)
                if hasattr(ts, "tz") and ts.tz is not None: ts = ts.tz_localize(None)
                all_sig_dates.update(ts.strftime("%Y-%m-%d"))

            # Sample up to 15 BBO days to measure fill rate (avoids OOM)
            sample_fill_rate = None
            for bbo_dir in BBO_DIRS:
                sym_dir = bbo_dir / sym
                if not sym_dir.exists(): continue
                avail = sorted(set(f.stem for f in sym_dir.glob("*.parquet")) & all_sig_dates)
                if not avail: continue
                # Sample evenly across available dates
                sample_n = min(15, len(avail))
                indices = np.linspace(0, len(avail) - 1, sample_n, dtype=int)
                sample_dates = [avail[i] for i in indices]
                fills = []
                for date_str in sample_dates:
                    pq = sym_dir / f"{date_str}.parquet"
                    try:
                        df = pd.read_parquet(pq)
                        bts = df["transaction_time"].values.astype(np.int64) * 1_000_000
                        bbid = df["best_bid_price"].values.astype(np.float64)
                        bask = df["best_ask_price"].values.astype(np.float64)
                        # Generate test signals every 15 min
                        interval = 15 * 60 * 1_000_000_000
                        sig_times = np.arange(bts[0] + interval, bts[-1] - 60_000_000_000, interval)
                        if len(sig_times) < 5: continue
                        dirs = np.ones(len(sig_times))
                        filled, _ = check_maker_fill(bts, bbid, bask, sig_times, dirs, 60_000_000_000, tick)
                        fills.append(float(filled.mean()))
                        del df, bts, bbid, bask
                    except: pass
                if fills:
                    sample_fill_rate = float(np.mean(fills))
                break

            tick = DEFAULT_TICK_SIZES.get(sym, 0.01)

            # Per-asset calibrated fallback cost
            if sym in per_sym_cal:
                # Use this asset's actual measured fill rate + spread
                cal = per_sym_cal[sym]
                fallback_cost = (cal["fill_rate"] * 2 * maker_bps +
                                 (1 - cal["fill_rate"]) * 2 * taker_bps +
                                 cal["spread_bps"])
            else:
                # Use volume-tier fallback
                vol = asset_volumes.get(name)
                avg_vol = float(vol.median()) if vol is not None and len(vol) > 0 else 10_000_000
                # Find tier
                fill_rate, spread = 0.77, 1.2  # overall average defaults
                for tier_name in ["tier1", "tier2", "tier3", "tier4"]:
                    t = tier_cal.get(tier_name, {})
                    if avg_vol >= t.get("volume_min", 0):
                        fill_rate = t.get("fill_rate", 0.77)
                        spread = t.get("spread_bps", 1.2)
                        break
                fallback_cost = (fill_rate * 2 * maker_bps +
                                 (1 - fill_rate) * 2 * taker_bps + spread)

            # Compute cost using sampled fill rate or calibration fallback
            if sample_fill_rate is not None:
                # BBO-measured fill rate for this asset
                spread = per_sym_cal.get(sym, {}).get("spread_bps", 1.0)
                bbo_cost = (sample_fill_rate * 2 * maker_bps +
                            (1 - sample_fill_rate) * 2 * taker_bps + spread)
                use_bbo = True
            else:
                bbo_cost = fallback_cost
                use_bbo = False

            for tf_label in hold_sigs[name]:
                sigs = hold_sigs[name][tf_label]
                n = len(sigs)
                costs = np.full(n, bbo_cost)

                if use_bbo:
                    n_bbo_exact += n
                else:
                    n_stat += n

                hold_sigs[name][tf_label] = sigs.copy()
                hold_sigs[name][tf_label]["exec_cost_bps"] = costs
                all_costs.append(costs)
                n_total += n

        all_costs_arr = np.concatenate(all_costs)
        avg_cost = float(np.mean(all_costs_arr))
        pct_bbo = n_bbo_exact / n_total * 100 if n_total > 0 else 0
        print(f"    {n_total:,} signals: {pct_bbo:.0f}% BBO-measured, "
              f"{100 - pct_bbo:.0f}% calibrated-fallback, avg cost {avg_cost:.1f} bps ({time.time()-t1:.0f}s)")
        print(f"    Fees: maker {maker_bps:.1f} bps, taker {taker_bps:.1f} bps")

        # Cache execution costs
        exec_to_cache = {}
        for name in hold_sigs:
            for tf_label in hold_sigs[name]:
                if "exec_cost_bps" in hold_sigs[name][tf_label].columns:
                    exec_to_cache[(name, tf_label)] = hold_sigs[name][tf_label]["exec_cost_bps"].values
        with open(exec_cache_path, "wb") as f:
            pickle.dump(exec_to_cache, f)
        print(f"    Exec costs cached to {exec_cache_path}")

    # ── Phase 5: Define variants ──
    months = pd.period_range("2020-07", "2026-02", freq="M")

    # Core sets — no hindsight
    CORE = OBVIOUS_CORE  # 9 assets anyone would trade in 2020

    # Per-asset execution quality from calibration
    import json as _json
    _cal_path = Path("data/execution_calibration.json")
    _fill_rates, _spreads = {}, {}
    if _cal_path.exists():
        _cal = _json.load(open(_cal_path))
        for e in _cal.get("per_symbol", []):
            _fill_rates[e["symbol"]] = e["fill_rate"]
            _spreads[e["symbol"]] = e["spread_median_bps"]

    # --- Selection functions ---

    def make_vol_selector(volumes, top_n=30, min_hist=180):
        def sel(ms):
            return select_top_n_for_month(volumes, asset_first_dates, ms, top_n=top_n, min_history_days=min_hist)
        return sel

    def make_vol_regime_selector(volumes, lookback_months=6, top_n=30):
        def sel(ms):
            base = select_top_n_for_month(volumes, asset_first_dates, ms, top_n=50, min_history_days=180)
            filtered = []
            for name in base:
                vol = volumes.get(name)
                if vol is None: continue
                avail = vol.loc[:ms - pd.Timedelta(days=1)]
                if len(avail) < 30 * lookback_months: filtered.append(name); continue
                hist_median = avail.iloc[-(30 * lookback_months):].median()
                current = avail.iloc[-1]
                if current >= hist_median:
                    filtered.append(name)
                if len(filtered) >= top_n: break
            return filtered[:top_n]
        return sel

    def make_hybrid_selector(core, volumes, n_total=30):
        """Fix core assets, fill remaining slots from volume ranking."""
        def sel(ms):
            pool = select_top_n_for_month(volumes, asset_first_dates, ms, top_n=60, min_history_days=90)
            # Core assets that have signals
            active_core = [a for a in core if a in hold_sigs]
            dynamic = [a for a in pool if a not in core][:n_total - len(active_core)]
            return active_core + dynamic
        return sel

    def make_exec_aware_selector(core, volumes, n_total=30):
        """Fix core, fill dynamic slots ranked by volume × fill_rate / (spread + 1)."""
        def sel(ms):
            pool = select_top_n_for_month(volumes, asset_first_dates, ms, top_n=80, min_history_days=90)
            active_core = [a for a in core if a in hold_sigs]
            non_core = [a for a in pool if a not in core]
            scored = []
            for a in non_core:
                vol = volumes.get(a)
                if vol is None: continue
                avail = vol.loc[:ms - pd.Timedelta(days=1)]
                if len(avail) == 0: continue
                v = float(avail.iloc[-1])
                fr = _fill_rates.get(f"{a}USDT", 0.77)
                sp = _spreads.get(f"{a}USDT", 1.5)
                scored.append((a, v * fr / (sp + 1)))
            scored.sort(key=lambda x: -x[1])
            dynamic = [a for a, _ in scored[:n_total - len(active_core)]]
            return active_core + dynamic
        return sel

    def make_variable_n_selector(volumes, min_fill=0.70, max_spread=3.0):
        """Include all assets meeting quality thresholds. N varies."""
        def sel(ms):
            pool = select_top_n_for_month(volumes, asset_first_dates, ms, top_n=100, min_history_days=90)
            return [a for a in pool
                    if _fill_rates.get(f"{a}USDT", 0.5) >= min_fill
                    and _spreads.get(f"{a}USDT", 5.0) <= max_spread]
        return sel

    def make_variable_n_core_selector(core, volumes, min_fill=0.70, max_spread=3.0):
        """Variable-N with core always included."""
        def sel(ms):
            pool = select_top_n_for_month(volumes, asset_first_dates, ms, top_n=100, min_history_days=90)
            qualified = [a for a in pool
                         if _fill_rates.get(f"{a}USDT", 0.5) >= min_fill
                         and _spreads.get(f"{a}USDT", 5.0) <= max_spread]
            active_core = [a for a in core if a in hold_sigs]
            return list(set(active_core + qualified))
        return sel

    def make_rolling_perf_selector(volumes, hold_sigs_ref, lookback_months=6, top_n=30):
        """Dynamic-30 minus assets with negative rolling Sharpe."""
        perf_cache = {}  # {name: list of monthly pnl}
        def sel(ms):
            base = select_top_n_for_month(volumes, asset_first_dates, ms, top_n=50, min_history_days=180)
            # Compute rolling perf from hold_sigs
            cutoff = ms - pd.DateOffset(months=lookback_months)
            good = []
            for name in base:
                if name not in hold_sigs_ref: continue
                # Aggregate PnL for this asset across TFs in lookback window
                total_pnl = 0
                n_trades = 0
                for tf in hold_sigs_ref[name]:
                    s = hold_sigs_ref[name][tf]
                    ts = pd.to_datetime(s["timestamp"])
                    if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
                    mask = (ts >= cutoff) & (ts < ms)
                    window = s[mask.values]
                    if len(window) > 0 and "fwd_return" in window.columns:
                        total_pnl += float(window["fwd_return"].sum())
                        n_trades += len(window)
                # Include if positive or insufficient history
                if n_trades < 20 or total_pnl >= 0:
                    good.append(name)
                if len(good) >= top_n:
                    break
            return good[:top_n]
        return sel

    # --- Variant definitions ---

    # --- S-series: Signal-quality and regime-adaptive selection ---

    def make_confidence_selector(volumes, n=30, lookback_months=3):
        """S1: Rank by trailing OOS confidence × volume."""
        def sel(ms):
            cutoff = ms - pd.DateOffset(months=lookback_months)
            scored = []
            for name in hold_sigs:
                confs = []
                for tf in hold_sigs[name]:
                    s = hold_sigs[name][tf]
                    ts = pd.to_datetime(s["timestamp"])
                    if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
                    mask = (ts >= cutoff) & (ts < ms)
                    w = s[mask.values]
                    if "confidence" in w.columns and len(w) > 5:
                        confs.extend(w["confidence"].values)
                if not confs: continue
                avg_conf = float(np.mean(confs))
                vol = volumes.get(name)
                if vol is None: continue
                avail = vol.loc[:ms - pd.Timedelta(days=1)]
                if len(avail) == 0: continue
                scored.append((name, avg_conf * float(avail.iloc[-1])))
            scored.sort(key=lambda x: -x[1])
            return [a for a, _ in scored[:n]]
        return sel

    def make_winrate_selector(volumes, min_wr=0.55, n=30, lookback_months=3):
        """S2: Only include assets with trailing win rate > threshold, fill rest by volume."""
        def sel(ms):
            cutoff = ms - pd.DateOffset(months=lookback_months)
            winners, others = [], []
            pool = select_top_n_for_month(volumes, asset_first_dates, ms, top_n=60, min_history_days=90)
            for name in pool:
                if name not in hold_sigs: others.append(name); continue
                wins, total = 0, 0
                for tf in hold_sigs[name]:
                    s = hold_sigs[name][tf]
                    ts = pd.to_datetime(s["timestamp"])
                    if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
                    mask = (ts >= cutoff) & (ts < ms)
                    w = s[mask.values]
                    if "fwd_return" in w.columns and len(w) > 0:
                        wins += int((w["fwd_return"] > 0).sum())
                        total += len(w)
                if total < 20:
                    others.append(name)
                elif wins / total >= min_wr:
                    winners.append(name)
                else:
                    others.append(name)
            return (winners + others)[:n]
        return sel

    def make_conf_exec_selector(volumes, n=30, lookback_months=3):
        """S3: Rank by confidence × fill_rate / (spread + 1)."""
        def sel(ms):
            cutoff = ms - pd.DateOffset(months=lookback_months)
            scored = []
            for name in hold_sigs:
                confs = []
                for tf in hold_sigs[name]:
                    s = hold_sigs[name][tf]
                    ts = pd.to_datetime(s["timestamp"])
                    if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
                    mask = (ts >= cutoff) & (ts < ms)
                    w = s[mask.values]
                    if "confidence" in w.columns and len(w) > 5:
                        confs.extend(w["confidence"].values)
                if not confs: continue
                sym = f"{name}USDT"
                fr = _fill_rates.get(sym, 0.77)
                sp = _spreads.get(sym, 1.5)
                score = float(np.mean(confs)) * fr / (sp + 1)
                scored.append((name, score))
            scored.sort(key=lambda x: -x[1])
            return [a for a, _ in scored[:n]]
        return sel

    def make_regime_adaptive_selector(volumes, lookback_months=3):
        """S4: Fewer assets in high-vol, more in low-vol. Ranked by confidence."""
        btc_vol = vol_series.get("BTC")
        def sel(ms):
            # Determine regime from BTC realized vol
            n_assets = 30  # default
            if btc_vol is not None:
                avail = btc_vol.loc[:ms - pd.Timedelta(days=1)].dropna()
                if len(avail) > 60:
                    current = avail.iloc[-1]
                    p80 = avail.quantile(0.80)
                    p20 = avail.quantile(0.20)
                    if current > p80:
                        n_assets = 15  # concentrate
                    elif current < p20:
                        n_assets = 40  # expand
            # Select by confidence × volume
            cutoff = ms - pd.DateOffset(months=lookback_months)
            scored = []
            for name in hold_sigs:
                confs = []
                for tf in hold_sigs[name]:
                    s = hold_sigs[name][tf]
                    ts = pd.to_datetime(s["timestamp"])
                    if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
                    mask = (ts >= cutoff) & (ts < ms)
                    w = s[mask.values]
                    if "confidence" in w.columns and len(w) > 5:
                        confs.extend(w["confidence"].values)
                if not confs: continue
                vol = volumes.get(name)
                if vol is None: continue
                avail_v = vol.loc[:ms - pd.Timedelta(days=1)]
                if len(avail_v) == 0: continue
                scored.append((name, float(np.mean(confs)) * float(avail_v.iloc[-1])))
            scored.sort(key=lambda x: -x[1])
            return [a for a, _ in scored[:n_assets]]
        return sel

    def make_momentum_selector(volumes, positive_only=True, n=30):
        """S5/S6: Cross-sectional momentum selection."""
        def sel(ms):
            pool = select_top_n_for_month(volumes, asset_first_dates, ms, top_n=50, min_history_days=90)
            scored = []
            for name in pool:
                dr = daily_returns.get(name)
                if dr is None: continue
                avail = dr.loc[:ms - pd.Timedelta(days=1)]
                if len(avail) < 20: continue
                mom = float(avail.iloc[-30:].sum())  # 30d return
                if positive_only and mom <= 0: continue
                scored.append((name, mom))
            if positive_only:
                scored.sort(key=lambda x: -x[1])  # best momentum first
            else:
                scored.sort(key=lambda x: x[1])  # worst first (contrarian)
            result = [a for a, _ in scored[:n]]
            # If not enough, fill from volume pool
            if len(result) < n:
                for a in pool:
                    if a not in result:
                        result.append(a)
                    if len(result) >= n: break
            return result[:n]
        return sel

    def make_core_confidence_selector(core, volumes, n_total=30, lookback_months=3):
        """S7: Core + confidence-ranked dynamic."""
        conf_sel = make_confidence_selector(volumes, n=80, lookback_months=lookback_months)
        def sel(ms):
            active_core = [a for a in core if a in hold_sigs]
            dynamic = [a for a in conf_sel(ms) if a not in core][:n_total - len(active_core)]
            return active_core + dynamic
        return sel

    def make_core_regime_selector(core, volumes, lookback_months=3):
        """S8: Core + regime-adaptive dynamic."""
        regime_sel = make_regime_adaptive_selector(volumes, lookback_months)
        def sel(ms):
            active_core = [a for a in core if a in hold_sigs]
            full = regime_sel(ms)
            dynamic = [a for a in full if a not in core]
            # Ensure core is always present
            return active_core + dynamic
        return sel

    # --- Variant definitions ---
    # Format: (name, mode, sel_func, exec_weighted, base_sizing, symmetric_cfg)

    _s8_sel = make_core_regime_selector(CORE, asset_volumes)
    _s7_sel = make_core_confidence_selector(CORE, asset_volumes, 30)
    _s1_sel = make_confidence_selector(asset_volumes, 30)

    variants = [
        # References
        ("Overfit-30 (ceil)",    "fixed_set", FIXED_30_OVERFIT, False, 0.045, None),
        # V4 winners (selection only, F1 breaker)
        ("S8: Core+Regime",      "monthly", _s8_sel, False, 0.045, None),
        ("S7: Core+ConfDyn",     "monthly", _s7_sel, False, 0.045, None),
        ("S1: Conf-ranked",      "monthly", _s1_sel, False, 0.045, None),
        # Execution-weighted sizing on best selections
        ("S8+ExecSize",          "monthly", _s8_sel, True, 0.045, None),
        ("S7+ExecSize",          "monthly", _s7_sel, True, 0.045, None),
        # Symmetric sizing (scale UP winners) on best selections
        ("S8+Sym1.5",            "monthly", _s8_sel, False, 0.045, {"up_threshold": 0.01, "down_threshold": -0.01, "cap": 1.5}),
        ("S8+Sym2.0",            "monthly", _s8_sel, False, 0.045, {"up_threshold": 0.01, "down_threshold": -0.01, "cap": 2.0}),
        ("S7+Sym1.5",            "monthly", _s7_sel, False, 0.045, {"up_threshold": 0.01, "down_threshold": -0.01, "cap": 1.5}),
        # Higher base sizing
        ("S8+Size5.5%",          "monthly", _s8_sel, False, 0.055, None),
        ("S8+Size6.0%",          "monthly", _s8_sel, False, 0.060, None),
        # Combined: exec-weighted + symmetric
        ("S8+Exec+Sym1.5",      "monthly", _s8_sel, True, 0.045, {"up_threshold": 0.01, "down_threshold": -0.01, "cap": 1.5}),
        # Combined: higher sizing + symmetric
        ("S8+5.5%+Sym1.5",      "monthly", _s8_sel, False, 0.055, {"up_threshold": 0.01, "down_threshold": -0.01, "cap": 1.5}),
        # Kitchen sink: exec + higher sizing + sym
        ("S8+Exec+5.5%+Sym1.5", "monthly", _s8_sel, True, 0.055, {"up_threshold": 0.01, "down_threshold": -0.01, "cap": 1.5}),
    ]
    if SMOKE:
        variants = [variants[0], variants[1], variants[6], variants[9], variants[11]]

    # ── Phase 6: Run variants (PARALLEL) ──
    from concurrent.futures import ThreadPoolExecutor as _TPE

    def _eval_one(var_tuple):
        var_name, mode, sel_func, exec_weighted, base_sizing, sym_cfg = var_tuple
        try:
            if mode == "fixed_set":
                parts = []
                for name in sel_func:
                    for tf_label, _ in NEEDED_TFS:
                        if name in hold_sigs and tf_label in hold_sigs[name]:
                            parts.append(hold_sigs[name][tf_label])
                merged = pd.concat(parts, ignore_index=True) if parts else None
            elif mode == "monthly":
                merged = assemble_signals(hold_sigs, sel_func, months)
            elif mode == "weekly":
                merged = assemble_weekly(hold_sigs, sel_func, "2020-07-01", "2026-03-01")
            else:
                merged = None

            if merged is None or len(merged) < 100:
                return var_name, None

            if exec_weighted and "exec_cost_bps" in merged.columns:
                costs = merged["exec_cost_bps"].fillna(merged["exec_cost_bps"].median())
                median_cost = costs.median()
                if median_cost > 0:
                    mult = np.clip(median_cost / costs.replace(0, median_cost), 0.3, 2.0)
                    if "sizing_mult" not in merged.columns: merged["sizing_mult"] = 1.0
                    merged["sizing_mult"] = merged["sizing_mult"].astype(float) * mult.values

            merged = merged.sort_values("timestamp").reset_index(drop=True)
            r = run_sim(merged, vol_series, base_sizing=base_sizing, symmetric=sym_cfg)
            return var_name, r
        except Exception as e:
            return var_name, {"error": str(e)}

    print(f"\n  Evaluating {len(variants)} variants in parallel...", flush=True)
    t_eval = time.time()

    all_results = {}
    with _TPE(max_workers=min(len(variants), 8)) as pool:
        futs = {pool.submit(_eval_one, v): v[0] for v in variants}
        for fut in as_completed(futs):
            name, r = fut.result()
            if r is not None and "error" not in r:
                all_results[name] = r

    print(f"  {len(all_results)}/{len(variants)} completed ({time.time()-t_eval:.0f}s)")

    # Print results table
    print(f"\n{'='*110}")
    print("  RESULTS")
    print(f"{'='*110}")
    print(f"\n  {'Variant':<24} {'Sharpe':>7} {'Sortino':>8} {'PnL':>11} {'MaxDD':>6} {'DayDD':>7}"
          f" {'Calmar':>7} {'PF':>6} {'Trades':>7} {'Assets':>7} {'$/mo':>8}")
    print("  " + "-" * 110)
    for var_name, *_ in variants:
        r = all_results.get(var_name)
        if r is None:
            print(f"  {var_name:<24} NO SIGNALS / FAIL")
        elif "error" in r:
            print(f"  {var_name:<24} ERROR: {r['error'][:60]}")
        else:
            print(f"  {var_name:<24} {r['sr']:>7.3f} {r['sortino']:>8.3f} ${r['pnl']:>9,.0f} {r['max_dd']:>5.1f}%"
                  f" {r['max_daily_dd']:>6.2f}% {r['calmar']:>7.2f} {r['pf']:>5.2f}"
                  f" {r['n_trades']:>7,} {r['n_assets']:>7} ${r['pnl_mo']:>7,.0f}")

    # Ranking
    print(f"\n{'='*110}")
    print("  RANKING BY SHARPE")
    print(f"{'='*110}")
    base_sr = all_results.get("Dyn-30 (baseline)", {}).get("sr", 0)
    ranked = sorted(all_results.items(), key=lambda x: x[1]["sr"], reverse=True)
    print(f"\n  {'#':>3} {'Variant':<24} {'Sharpe':>7} {'dSh':>6} {'PnL':>11} {'MaxDD':>6} {'Calmar':>7} {'$/mo':>8}")
    print("  " + "-" * 80)
    for i, (name, r) in enumerate(ranked, 1):
        ds = r["sr"] - base_sr
        print(f"  {i:>3} {name:<24} {r['sr']:>7.3f} {ds:>+5.2f} ${r['pnl']:>9,.0f} {r['max_dd']:>5.1f}%"
              f" {r['calmar']:>7.2f} ${r['pnl_mo']:>7,.0f}")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
