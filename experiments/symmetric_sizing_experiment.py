#!/usr/bin/env python
"""Symmetric Adaptive Sizing — Scale down losers AND scale up winners.

Extends F1 (PnL-weighted breaker) with upside boost when asset is on a hot streak.

Usage:
    PYTHONPATH=src python experiments/symmetric_sizing_experiment.py [--smoke]
"""
import sys, os, time, warnings, gc, math
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict, deque
from concurrent.futures import ProcessPoolExecutor, as_completed

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from experiments.circuit_breaker_v2_experiment import (
    ASSETS, NEEDED_TFS, FULL5, CAP, BASELINE_WR, VIP_TIERS, SPREAD_BPS,
    generate_signals_cached, recompute_returns, apply_vol, _compute_vol_one,
    get_vip_tier,
)
from signal_engine.portfolio import compute_daily_pnl_series
from signal_engine.config import PortfolioConfig

SMOKE = "--smoke" in sys.argv

VARIANTS = [
    ("Baseline (no CB)",  "none",      {}),
    ("F1 (down-only)",    "pnl_ratio", {"window": 10, "threshold": -0.01, "floor": 0.10}),
    # Symmetric
    ("S1: 1%/1% cap1.5",  "symmetric", {"window": 10, "down_threshold": -0.01, "up_threshold": 0.01, "floor": 0.10, "cap": 1.5, "reset_mode": "smooth"}),
    ("S2: 1%/1% cap2.0",  "symmetric", {"window": 10, "down_threshold": -0.01, "up_threshold": 0.01, "floor": 0.10, "cap": 2.0, "reset_mode": "smooth"}),
    ("S3: .5%/.5% cap1.5", "symmetric", {"window": 10, "down_threshold": -0.005, "up_threshold": 0.005, "floor": 0.10, "cap": 1.5, "reset_mode": "smooth"}),
    # Asymmetric
    ("S4: 1%/0.5% cap1.5", "symmetric", {"window": 10, "down_threshold": -0.01, "up_threshold": 0.005, "floor": 0.10, "cap": 1.5, "reset_mode": "smooth"}),
    ("S5: 1%/0.3% cap2.0", "symmetric", {"window": 10, "down_threshold": -0.01, "up_threshold": 0.003, "floor": 0.10, "cap": 2.0, "reset_mode": "smooth"}),
    # Reset modes
    ("S6: S2+hard reset",  "symmetric", {"window": 10, "down_threshold": -0.01, "up_threshold": 0.01, "floor": 0.10, "cap": 2.0, "reset_mode": "hard_loss"}),
    ("S7: S2+thresh reset", "symmetric", {"window": 10, "down_threshold": -0.01, "up_threshold": 0.01, "floor": 0.10, "cap": 2.0, "reset_mode": "threshold"}),
]
if SMOKE:
    VARIANTS = [VARIANTS[0], VARIANTS[1], VARIANTS[2], VARIANTS[4]]


def _edge_from_probability(prob, sig):
    return float(np.clip(abs(2.0 * prob - 1.0), 0.0, 1.0)) if np.isfinite(prob) else float(np.clip(abs(sig), 0.0, 1.0))


def simulate_symmetric(signals_df, config, breaker_type, breaker_params):
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
    active_positions = []
    unique_ts = pd.Series(timestamps).unique()
    ts_to_idx = {ts: idx for idx, ts in enumerate(unique_ts)}
    use_prob = signal_strengths is not None

    w = breaker_params.get("window", 10)
    asset_recent_pnl = defaultdict(lambda: deque(maxlen=w))
    asset_recent_notional = defaultdict(lambda: deque(maxlen=w))
    asset_last_loss = defaultdict(bool)
    cb_boosts = 0; cb_cuts = 0

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

        asset = assets_arr[i]
        cb_mult = 1.0

        if breaker_type == "pnl_ratio":
            recent_pnl = asset_recent_pnl[asset]
            recent_not = asset_recent_notional[asset]
            if len(recent_pnl) >= w:
                total_pnl = sum(recent_pnl); total_not = sum(recent_not)
                if total_not > 0:
                    ratio = total_pnl / total_not
                    t = breaker_params["threshold"]
                    if ratio < t:
                        cb_mult = max(1.0 + ratio / abs(t), breaker_params.get("floor", 0.10))
                        cb_cuts += 1

        elif breaker_type == "symmetric":
            recent_pnl = asset_recent_pnl[asset]
            recent_not = asset_recent_notional[asset]
            if len(recent_pnl) >= w:
                total_pnl = sum(recent_pnl); total_not = sum(recent_not)
                if total_not > 0:
                    ratio = total_pnl / total_not
                    dt = breaker_params["down_threshold"]
                    ut = breaker_params["up_threshold"]
                    floor = breaker_params.get("floor", 0.10)
                    cap = breaker_params.get("cap", 1.5)
                    reset = breaker_params.get("reset_mode", "smooth")

                    if ratio < dt:
                        cb_mult = max(1.0 + ratio / abs(dt), floor)
                        cb_cuts += 1
                    elif ratio > ut:
                        boost = 1.0 + (ratio - ut) / ut
                        cb_mult = min(boost, cap)

                        if reset == "hard_loss" and asset_last_loss.get(asset, False):
                            cb_mult = 1.0
                        elif reset == "threshold":
                            pass  # already handled: ratio > ut means we're above threshold

                        if cb_mult > 1.0:
                            cb_boosts += 1

        raw_size *= cb_mult

        # Exposure limits
        asset_exp = asset_exposure.get(asset, 0.0)
        if asset_exp + raw_size > equity * config.max_per_asset:
            raw_size = max(equity * config.max_per_asset - asset_exp, 0.0)
        strategy = strategies_arr[i]; mps = getattr(config, "max_per_strategy", 1.0)
        strat_exp = strategy_exposure.get(strategy, 0.0)
        if mps < 1.0 and strat_exp + raw_size > equity * mps:
            raw_size = max(equity * mps - strat_exp, 0.0)
        if current_exposure + raw_size > equity * config.max_exposure:
            raw_size = max(equity * config.max_exposure - current_exposure, 0.0)
        if raw_size < 1.0: equity_arr[i] = equity; dd_arr[i] = dd; continue

        trade_return = directions[i] * fwd_returns[i] - config.slippage
        pnl = raw_size * trade_return; equity += pnl
        if equity > peak_equity: peak_equity = equity
        active_positions.append((current_ts_idx + int(hold_bars_arr[i]), raw_size, asset, strategy))
        equity_arr[i] = equity; pnl_arr[i] = pnl
        dd_arr[i] = (peak_equity - equity) / peak_equity if peak_equity > 0 else 0.0
        size_arr[i] = raw_size; skipped[i] = False

        asset_last_loss[asset] = pnl < 0
        asset_recent_pnl[asset].append(pnl)
        asset_recent_notional[asset].append(raw_size)

    return {"equity": equity_arr, "pnl": pnl_arr, "dd": dd_arr, "sizes": size_arr,
            "skipped": skipped, "timestamps": timestamps, "halted": halted,
            "final_equity": equity, "n_trades": int((~skipped).sum()),
            "cb_boosts": cb_boosts, "cb_cuts": cb_cuts}


def run_variant(merged, name, btype, bparams):
    port = PortfolioConfig(
        starting_capital=CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
        max_exposure=0.90, max_per_asset=0.06, max_per_strategy=0.50,
        base_sizing_pct=0.045, max_signal_mult=2.0, sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)
    prepared = merged.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in prepared.columns: prepared[c] = 1.0
        prepared[c] = prepared[c].fillna(1.0)
    prepared["probability"] = prepared["probability"].astype(float)
    prepared = prepared.sort_values("timestamp").reset_index(drop=True)
    sim = simulate_symmetric(prepared, port, btype, bparams)
    if sim is None: return None
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None: return None
    if dpnl.index.tz is not None: dpnl = dpnl.copy(); dpnl.index = dpnl.index.tz_localize(None)

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
    eq_adj = (1 + dr).cumprod()
    dd_s = (eq_adj.cummax() - eq_adj) / eq_adj.cummax()
    max_dd = float(dd_s.max() * 100)
    calmar = float(dr.mean() * 365.25 / (max_dd / 100)) if max_dd > 0 else 0
    pnl = float(dpnl_adj.sum())
    pf = float(dpnl_adj[dpnl_adj > 0].sum() / abs(dpnl_adj[dpnl_adj < 0].sum())) if (dpnl_adj < 0).any() else 99
    daily_dd_pct = dpnl_adj / (CAP + dpnl_adj.cumsum()).shift(1).fillna(CAP) * 100
    max_daily_dd = float(daily_dd_pct.min())
    worst_day_pnl = float(dpnl_adj[daily_dd_pct.idxmin()])
    monthly = dpnl_adj.resample("MS").sum()

    # 2025 + 2026 stats
    for yr_label, yr_start, yr_end in [("2025", "2025-01-01", "2026-01-01"), ("2026", "2026-01-01", "2027-01-01")]:
        d = dpnl_adj[(dpnl_adj.index >= yr_start) & (dpnl_adj.index < yr_end)]
        dr_yr = d / CAP
        sr_yr = float(dr_yr.mean() / dr_yr.std() * np.sqrt(365.25)) if len(dr_yr) > 20 and dr_yr.std() > 0 else 0

    return {"sr": sr, "sortino": sortino, "max_dd": max_dd, "calmar": calmar,
            "pnl": pnl, "pf": pf, "n_trades": sim["n_trades"],
            "cb_boosts": sim["cb_boosts"], "cb_cuts": sim["cb_cuts"],
            "max_daily_dd": max_daily_dd, "worst_day_pnl": worst_day_pnl,
            "pnl_mo": pnl / max(len(monthly), 1), "daily_pnl": dpnl_adj}


def main():
    T0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print("=" * 120)
    print(f"  SYMMETRIC ADAPTIVE SIZING ({mode})")
    print(f"  {len(ASSETS)} assets, Full5 combo, {len(VARIANTS)} variants")
    print("=" * 120)

    vol_series = {n: v for n, v in (ProcessPoolExecutor(max_workers=8).map(
        _compute_vol_one, list(ASSETS.items()))) if v is not None}

    signal_cache = {}
    for label, resample in NEEDED_TFS:
        sigs = generate_signals_cached(label, resample)
        if sigs is not None: signal_cache[label] = sigs; print(f"  {label}: {len(sigs):,}", flush=True)

    hold_cache = {}
    for tf, h in FULL5:
        if tf not in signal_cache: continue
        resample = dict(NEEDED_TFS)[tf]
        hold_cache[(tf, h)] = apply_vol(recompute_returns(signal_cache[tf], resample, h), vol_series)
    parts = [hold_cache[(tf, h)] for tf, h in FULL5 if (tf, h) in hold_cache]
    merged = pd.concat(parts, ignore_index=True).sort_values("timestamp").reset_index(drop=True)
    print(f"  Full5: {len(merged):,} signals\n")

    # Run all variants
    print(f"  {'Variant':<24} {'Sharpe':>7} {'Sortino':>8} {'PnL':>11} {'MaxDD':>6} {'DayDD':>7}"
          f" {'WorstDay$':>10} {'Calmar':>7} {'PF':>6} {'Trades':>7} {'Boosts':>7} {'Cuts':>7} {'$/mo':>8}")
    print("  " + "-" * 125)

    all_results = {}
    for name, btype, bparams in VARIANTS:
        r = run_variant(merged, name, btype, bparams)
        if r is None: print(f"  {name:<24} FAIL"); continue
        all_results[name] = r
        print(f"  {name:<24} {r['sr']:>7.3f} {r['sortino']:>8.3f} ${r['pnl']:>9,.0f} {r['max_dd']:>5.1f}%"
              f" {r['max_daily_dd']:>6.2f}% ${r['worst_day_pnl']:>8,.0f} {r['calmar']:>7.2f}"
              f" {r['pf']:>5.2f} {r['n_trades']:>7,} {r['cb_boosts']:>7,} {r['cb_cuts']:>7,}"
              f" ${r['pnl_mo']:>7,.0f}")

    # Ranking
    print(f"\n{'='*120}")
    print("  RANKING BY SHARPE")
    print(f"{'='*120}")
    f1 = all_results.get("F1 (down-only)", all_results.get("Baseline (no CB)"))
    ranked = sorted(all_results.items(), key=lambda x: x[1]["sr"], reverse=True)
    print(f"\n  {'#':>3} {'Variant':<24} {'Sharpe':>7} {'dSh':>6} {'PnL':>11} {'MaxDD':>6} {'DayDD':>7} {'$/mo':>8}")
    print("  " + "-" * 80)
    for i, (name, r) in enumerate(ranked, 1):
        ds = r["sr"] - f1["sr"]
        print(f"  {i:>3} {name:<24} {r['sr']:>7.3f} {ds:>+5.2f} ${r['pnl']:>9,.0f} {r['max_dd']:>5.1f}%"
              f" {r['max_daily_dd']:>6.2f}% ${r['pnl_mo']:>7,.0f}")

    # 2025 + 2026 for top variants
    print(f"\n{'='*120}")
    print("  2025 + 2026 YEARLY PERFORMANCE")
    print(f"{'='*120}")

    show = [name for name, _ in ranked[:4]]
    shorts = [n.split(":")[0].split("(")[0].strip() for n in show]
    col_w = 14

    for yr, start, end in [("2025", "2025-01-01", "2026-01-01"), ("2026 YTD", "2026-01-01", "2027-01-01")]:
        header = f"\n  {yr + ':':>12}"
        for s in shorts: header += f" {s:>{col_w}}"
        print(header)
        print("  " + "-" * (12 + (col_w + 1) * len(show)))

        # Total
        row = f"  {'PnL':>12}"
        for name in show:
            d = all_results[name]["daily_pnl"]
            val = float(d[(d.index >= start) & (d.index < end)].sum())
            row += f" ${val:>{col_w-1},.0f}"
        print(row)

        # Sharpe
        row = f"  {'Sharpe':>12}"
        for name in show:
            d = all_results[name]["daily_pnl"]
            d_yr = d[(d.index >= start) & (d.index < end)]
            dr = d_yr / CAP
            sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if len(dr) > 20 and dr.std() > 0 else 0
            row += f" {sr:>{col_w}.3f}"
        print(row)

        # MaxDD
        row = f"  {'MaxDD':>12}"
        for name in show:
            d = all_results[name]["daily_pnl"]
            d_yr = d[(d.index >= start) & (d.index < end)]
            dr = d_yr / CAP; eq = (1 + dr).cumprod()
            dd = float(((eq.cummax() - eq) / eq.cummax()).max() * 100) if len(eq) > 0 else 0
            row += f" {dd:>{col_w-1}.2f}%"
        print(row)

        # Worst day
        row = f"  {'Worst Day':>12}"
        for name in show:
            d = all_results[name]["daily_pnl"]
            d_yr = d[(d.index >= start) & (d.index < end)]
            row += f" ${float(d_yr.min()) if len(d_yr) > 0 else 0:>{col_w-1},.0f}"
        print(row)

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
