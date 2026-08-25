#!/usr/bin/env python
"""$200K compounding run — full period, reinvesting profits.

Usage:
    PYTHONPATH="src;." python experiments/yolo_compound.py
"""

import sys, os, time, warnings, pickle, logging
import numpy as np
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)
from collections import OrderedDict

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from experiments.drawdown_improvements import (
    simulate_with_improvements, apply_bar_level_sl, CACHE_DIR, TF_RESAMPLE,
)
from experiments.circuit_breaker_v2_experiment import apply_vol
from experiments.dynamic_asset_v2_experiment import assemble_signals, OBVIOUS_CORE
from signal_engine.portfolio import compute_daily_pnl_series
from signal_engine.config import PortfolioConfig
from signal_engine.slippage import load_per_symbol_calibration
from signal_engine.execution import get_vip_tier

CAP = 200_000.0


def run_full(prepared, config_label, port, breaker, imp, slippage_model, asset_volumes,
             close_cache=None, export_trade_log=False):
    # Apply bar-level SL if close cache available
    if close_cache:
        prepared, sl_count = apply_bar_level_sl(
            prepared, close_cache,
            max_risk_per_trade=imp.get("max_risk_per_trade", 0.03),
            sl_cap=imp.get("sl_cap", 0.75),
            base_sizing_pct=port.base_sizing_pct,
        )
        if sl_count > 0:
            logger.info("%s: bar-level SL triggered on %d / %d signals",
                        config_label, sl_count, len(prepared))

    sim = simulate_with_improvements(
        prepared, port, breaker, imp,
        slippage_model=slippage_model, asset_volumes=asset_volumes)
    if sim is None:
        print(f"  {config_label}: sim returned None")
        return None

    # Export per-trade log
    if export_trade_log:
        _export_trade_log(prepared, sim, config_label)

    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None or len(dpnl) < 10:
        print(f"  {config_label}: no daily PnL")
        return None
    if dpnl.index.tz is not None:
        dpnl.index = dpnl.index.tz_localize(None)

    # Build equity curve from actual equity (compounding)
    running_eq = CAP
    daily_rets = []
    eq_series = []
    for dp in dpnl.values:
        r = dp / running_eq if running_eq > 0 else 0.0
        daily_rets.append(r)
        running_eq = max(running_eq + dp, 0.0)
        eq_series.append(running_eq)
    daily_rets = np.array(daily_rets)
    eq_arr = np.array(eq_series)
    eq_s = pd.Series(eq_arr, index=dpnl.index)

    sr = float(daily_rets.mean() / daily_rets.std() * np.sqrt(365.25)) if daily_rets.std() > 0 else 0
    peak = np.maximum.accumulate(eq_arr)
    dd_arr = np.where(peak > 0, (peak - eq_arr) / peak, 0)
    dd_s = pd.Series(dd_arr, index=dpnl.index)
    max_dd = float(dd_arr.max() * 100)
    final_eq = eq_arr[-1]
    total_ret = (final_eq / CAP - 1) * 100
    n_days = len(dpnl)
    cagr = ((final_eq / CAP) ** (365.25 / n_days) - 1) * 100

    return {
        "label": config_label, "sr": sr, "final_eq": final_eq,
        "total_ret": total_ret, "cagr": cagr, "max_dd": max_dd,
        "n_days": n_days, "trades": sim["n_trades"],
        "win_rate": float((dpnl > 0).mean() * 100),
        "dpnl": dpnl, "eq_s": eq_s, "dd_s": dd_s,
        "daily_rets": daily_rets,
    }


def _export_trade_log(prepared, sim, config_label):
    """Export per-trade CSV from simulation results."""
    out_dir = Path("experiments/trade_logs")
    out_dir.mkdir(exist_ok=True)

    cols = {
        "timestamp": prepared["timestamp"].values,
        "asset": prepared["asset"].values,
        "direction": prepared["direction"].values,
        "probability": prepared["probability"].values,
        "sizing_mult": prepared["sizing_mult"].values,
        "size_usd": sim["sizes"],
        "pnl": sim["pnl"],
        "equity": sim["equity"],
        "skipped": sim["skipped"],
        "halted": sim["halted"],
    }
    for opt in ["strategy", "timeframe", "confidence", "hold_bars"]:
        if opt in prepared.columns:
            cols[opt] = prepared[opt].values
    trades = pd.DataFrame(cols)
    # Keep only executed trades
    executed = trades[~trades["skipped"]].copy()
    executed["side"] = executed["direction"].map({1: "long", -1: "short"})
    executed["cumulative_pnl"] = executed["pnl"].cumsum()
    executed = executed.drop(columns=["skipped", "halted", "direction"])

    path = out_dir / f"trade_log_{config_label.lower()}.csv"
    executed.to_csv(path, index=False, float_format="%.4f")
    print(f"  Trade log: {path} ({len(executed):,} trades)")


def main():
    T0 = time.time()

    with open(CACHE_DIR / "v2_prep_cache.pkl", "rb") as f:
        cached = pickle.load(f)
    hold_sigs = cached["hold_sigs"]
    vol_series = cached["vol_series"]
    asset_volumes = cached["asset_volumes"]

    months = pd.period_range("2020-07", "2026-04", freq="M")

    def _regime_sel(ms):
        btc_vol = vol_series.get("BTC")
        n_assets = 30
        if btc_vol is not None:
            avail = btc_vol.loc[:ms - pd.Timedelta(days=1)].dropna()
            if len(avail) > 60:
                current = avail.iloc[-1]
                if current > avail.quantile(0.80): n_assets = 15
                elif current < avail.quantile(0.20): n_assets = 40
        cutoff = ms - pd.DateOffset(months=3)
        scored = []
        for name in hold_sigs:
            confs = []
            for tf in hold_sigs[name]:
                s = hold_sigs[name][tf]
                ts = pd.to_datetime(s["timestamp"])
                if ts.dt.tz is not None: ts = ts.dt.tz_localize(None)
                w = s[((ts >= cutoff) & (ts < ms)).values]
                if "confidence" in w.columns and len(w) > 5:
                    confs.extend(w["confidence"].values)
            if not confs: continue
            vol = asset_volumes.get(name)
            if vol is None: continue
            avail_v = vol.loc[:ms - pd.Timedelta(days=1)]
            if len(avail_v) == 0: continue
            scored.append((name, float(np.mean(confs)) * float(avail_v.iloc[-1])))
        scored.sort(key=lambda x: -x[1])
        return [a for a, _ in scored[:n_assets]]

    def _s8_sel(ms):
        active_core = [a for a in OBVIOUS_CORE if a in hold_sigs]
        full = _regime_sel(ms)
        return active_core + [a for a in full if a not in OBVIOUS_CORE]

    merged = assemble_signals(hold_sigs, _s8_sel, months)
    prepared = merged.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in prepared.columns: prepared[c] = 1.0
        prepared[c] = prepared[c].fillna(1.0)
    prepared["probability"] = prepared["probability"].astype(float)
    prepared = apply_vol(prepared.sort_values("timestamp").reset_index(drop=True), vol_series)

    # Filter to 2026 only
    EXCLUDE_YEARS = set()
    ONLY_YEAR = None  # Set to e.g. 2026 to filter to a single year
    if EXCLUDE_YEARS:
        _ts = pd.to_datetime(prepared["timestamp"])
        if _ts.dt.tz is not None:
            _ts = _ts.dt.tz_localize(None)
        prepared = prepared[~_ts.dt.year.isin(EXCLUDE_YEARS)].reset_index(drop=True)
        print(f"  Excluded years: {EXCLUDE_YEARS} — {len(prepared):,} signals remaining\n")
    if ONLY_YEAR:
        _ts = pd.to_datetime(prepared["timestamp"])
        if _ts.dt.tz is not None:
            _ts = _ts.dt.tz_localize(None)
        prepared = prepared[_ts.dt.year == ONLY_YEAR].reset_index(drop=True)
        print(f"  Filtered to {ONLY_YEAR} only — {len(prepared):,} signals remaining\n")

    # ── Bar-level SL: build close price cache from 1m data ──
    close_cache_path = CACHE_DIR / "close_cache.pkl"
    if close_cache_path.exists():
        with open(close_cache_path, "rb") as f:
            close_cache = pickle.load(f)
        print(f"  Close cache: {len(close_cache)} (asset, tf) pairs loaded from disk")
    else:
        from signal_engine.data import load_binance_1m, resample_ohlcv
        DATA_DIRS = [Path("data/binance-1m")]
        close_cache = {}
        assets_needed = sorted(hold_sigs.keys())
        print(f"  Building close cache for {len(assets_needed)} assets...")
        for idx, asset in enumerate(assets_needed):
            sym = f"{asset}USDT"
            try:
                raw = load_binance_1m(sym, DATA_DIRS)
            except Exception:
                continue
            for tf_label, tf_resample in TF_RESAMPLE.items():
                try:
                    ohlcv = resample_ohlcv(raw, tf_resample)
                    close_cache[(asset, tf_label)] = ohlcv["close"]
                except Exception:
                    pass
            if (idx + 1) % 50 == 0:
                print(f"    {idx + 1}/{len(assets_needed)} assets done")
        with open(close_cache_path, "wb") as f:
            pickle.dump(close_cache, f)
        print(f"  Close cache: {len(close_cache)} (asset, tf) pairs built and saved")

    # Slippage model
    per_symbol_cal = load_per_symbol_calibration()
    vip_name, maker_bps, taker_bps = get_vip_tier(CAP)
    slippage_model = {
        "calibration": per_symbol_cal,
        "vip_tier": (vip_name, maker_bps, taker_bps),
        "max_participation_rate": 0.02,
        "guard_pct": 0.50,
    }

    breaker = {"window": 10, "threshold": -0.01, "floor": 0.10,
               "up_threshold": 0.01, "cap": 1.5}

    configs = OrderedDict()
    configs["YOLO"] = (
        PortfolioConfig(
            starting_capital=CAP, max_dd=0.30, delever_start=0.20, slippage=0.0001,
            max_exposure=0.95, max_per_asset=0.25, max_per_strategy=0.50,
            base_sizing_pct=0.200, max_signal_mult=2.0, sizing_mode="probability_edge",
            use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        {"max_signals_per_bar": 20, "max_risk_per_trade": 0.03, "sl_cap": 0.75})

    configs["PROD"] = (
        PortfolioConfig(
            starting_capital=CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
            max_exposure=0.90, max_per_asset=0.25, max_per_strategy=0.50,
            base_sizing_pct=0.300, max_signal_mult=2.0, sizing_mode="probability_edge",
            use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        {"max_signals_per_bar": 20, "max_risk_per_trade": 0.03, "sl_cap": 0.75})

    # PROD with HyroTrader compliance rules (swing DD, margin/notional limits)
    configs["HYRO"] = (
        PortfolioConfig(
            starting_capital=CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
            max_exposure=0.90, max_per_asset=0.25, max_per_strategy=0.50,
            base_sizing_pct=0.300, max_signal_mult=2.0, sizing_mode="probability_edge",
            use_regime_exposure_multiplier=True, regime_exposure_floor=0.50),
        {"max_signals_per_bar": 20, "max_risk_per_trade": 0.03, "sl_cap": 0.75,
         "daily_dd_limit": 0.05, "leverage": 4,
         "max_margin_pct": 0.25, "max_notional_mult": 2.0,
         "risk_basis": "initial", "max_loss_from_initial": 0.10})

    print("=" * 100)
    print(f"  $200K COMPOUNDING — full period, profits reinvested")
    print(f"  Realistic slippage ({vip_name}) | guard 50% | max participation 2%")
    print(f"  PROD_BINANCE: 1 bps flat | PROD_BYBIT: 4 bps flat (+3 bps cross-exchange)")
    print("=" * 100)

    results = {}
    for cfg_name, (port, imp) in configs.items():
        t1 = time.time()
        r = run_full(prepared.copy(), cfg_name, port, breaker, imp, slippage_model,
                     asset_volumes, close_cache=close_cache, export_trade_log=True)
        if r:
            results[cfg_name] = r
            print(f"  {cfg_name}: ${r['final_eq']:,.0f} final | SR {r['sr']:.2f} | "
                  f"CAGR {r['cagr']:.0f}% | DD {r['max_dd']:.1f}% | ({time.time()-t1:.1f}s)")

    # Full-period summary
    print(f"\n{'=' * 100}")
    print("  FULL PERIOD SUMMARY")
    print(f"{'=' * 100}")
    for cfg_name, r in results.items():
        print(f"\n  {cfg_name}:")
        print(f"    Period:       {r['dpnl'].index[0].date()} to {r['dpnl'].index[-1].date()} ({r['n_days']} days)")
        print(f"    Starting:     ${CAP:,.0f}")
        print(f"    Final Equity: ${r['final_eq']:,.0f}")
        print(f"    Total Return: {r['total_ret']:,.0f}%")
        print(f"    CAGR:         {r['cagr']:.0f}%")
        print(f"    Sharpe:       {r['sr']:.2f}")
        print(f"    Max DD:       {r['max_dd']:.1f}%")
        print(f"    Win Rate:     {r['win_rate']:.0f}%")
        print(f"    Trades:       {r['trades']:,}")

    # Year-end equity snapshots
    print(f"\n{'=' * 100}")
    print("  YEAR-END EQUITY")
    print(f"{'=' * 100}")
    print(f"  {'Date':<12}", end="")
    for cfg_name in results:
        print(f" {cfg_name + ' Equity':>18} {cfg_name + ' DD':>10}", end="")
    print()
    print("  " + "-" * (12 + 28 * len(results)))

    # Start
    print(f"  {'Start':<12}", end="")
    for cfg_name in results:
        print(f" ${CAP:>16,.0f} {'0.0%':>10}", end="")
    print()

    all_years = set()
    for r in results.values():
        all_years.update(r['eq_s'].index.year.unique())

    for yr in sorted(all_years):
        print(f"  {'End ' + str(yr):<12}", end="")
        for cfg_name, r in results.items():
            yr_eq = r['eq_s'][r['eq_s'].index.year == yr]
            yr_dd = r['dd_s'][r['dd_s'].index.year == yr]
            if len(yr_eq) > 0:
                eq_val = yr_eq.iloc[-1]
                dd_val = yr_dd.max() * 100
                print(f" ${eq_val:>16,.0f} {dd_val:>9.1f}%", end="")
            else:
                print(f" {'--':>18} {'--':>10}", end="")
        print()

    # Yearly returns (on compounding equity)
    print(f"\n{'=' * 100}")
    print("  YEARLY RETURNS (on compounding equity)")
    print(f"{'=' * 100}")
    print(f"  {'Year':<6}", end="")
    for cfg_name in results:
        print(f" {cfg_name + ' Ret%':>12} {cfg_name + ' SR':>10}", end="")
    print()
    print("  " + "-" * (6 + 22 * len(results)))

    for yr in sorted(all_years):
        print(f"  {yr:<6}", end="")
        for cfg_name, r in results.items():
            yr_mask = r['dpnl'].index.year == yr
            yr_dpnl = r['dpnl'][yr_mask]
            yr_eq = r['eq_s'][yr_mask]
            if len(yr_dpnl) > 0 and len(yr_eq) > 0:
                # Start-of-year equity
                eq_idx = r['eq_s'].index
                yr_start_idx = eq_idx[eq_idx.year == yr][0]
                pos = eq_idx.get_loc(yr_start_idx)
                soy_eq = r['eq_s'].iloc[pos - 1] if pos > 0 else CAP
                yr_ret = float(yr_dpnl.sum()) / soy_eq * 100 if soy_eq > 0 else 0

                yr_rets = yr_dpnl.values / np.maximum(
                    np.concatenate([[soy_eq], r['eq_s'][yr_mask].values[:-1]]), 1.0)
                yr_sr = float(yr_rets.mean() / yr_rets.std() * np.sqrt(365.25)) if yr_rets.std() > 0 else 0
                print(f" {yr_ret:>+11.1f}% {yr_sr:>10.2f}", end="")
            else:
                print(f" {'--':>12} {'--':>10}", end="")
        print()

    # Quarterly returns
    print(f"\n{'=' * 100}")
    print("  QUARTERLY RETURNS (on compounding equity)")
    print(f"{'=' * 100}")
    for cfg_name, r in results.items():
        print(f"\n  {cfg_name}:")
        print(f"  {'Year':<6} {'Q1':>10} {'Q2':>10} {'Q3':>10} {'Q4':>10} {'Full':>10}")
        print(f"  " + "-" * 60)
        neg_q = 0
        tot_q = 0
        for yr in sorted(all_years):
            cols = []
            yr_mask = r['dpnl'].index.year == yr

            # Start-of-year equity for computing returns
            eq_idx = r['eq_s'].index
            yr_dates = eq_idx[eq_idx.year == yr]
            if len(yr_dates) == 0:
                continue
            pos = eq_idx.get_loc(yr_dates[0])
            soy_eq = r['eq_s'].iloc[pos - 1] if pos > 0 else CAP
            yr_dpnl = r['dpnl'][yr_mask]
            yr_ret = float(yr_dpnl.sum()) / soy_eq * 100 if soy_eq > 0 else 0

            for q in [1, 2, 3, 4]:
                q_mask = yr_mask & (r['dpnl'].index.quarter == q)
                q_pnl = r['dpnl'][q_mask]
                if len(q_pnl) > 0:
                    q_ret = float(q_pnl.sum()) / soy_eq * 100 if soy_eq > 0 else 0
                    tot_q += 1
                    if q_ret < 0:
                        neg_q += 1
                    cols.append(f"{q_ret:>+9.1f}%")
                else:
                    cols.append(f"{'--':>10}")
            cols.append(f"{yr_ret:>+9.1f}%")
            print(f"  {yr:<6} {''.join(cols)}")
        print(f"  Negative quarters: {neg_q}/{tot_q}")

    # Monthly consistency
    print(f"\n{'=' * 100}")
    print("  MONTHLY CONSISTENCY")
    print(f"{'=' * 100}")
    for cfg_name, r in results.items():
        monthly = r['dpnl'].resample("MS").sum()
        pos = (monthly > 0).sum()
        tot = len(monthly)
        best = float(monthly.max())
        worst = float(monthly.min())
        # Best/worst as % of equity at that time
        print(f"  {cfg_name}: {pos}/{tot} positive ({pos/tot*100:.0f}%)")

    # Worst drawdown periods
    print(f"\n{'=' * 100}")
    print("  WORST 5 DRAWDOWN PERIODS")
    print(f"{'=' * 100}")
    for cfg_name, r in results.items():
        print(f"\n  {cfg_name}:")
        worst5 = r['dpnl'].nsmallest(5)
        for dt, val in worst5.items():
            eq_at = r['eq_s'].loc[:dt].iloc[-1] if dt in r['eq_s'].index else r['eq_s'].iloc[-1]
            pct = val / (eq_at - val) * 100 if (eq_at - val) > 0 else 0
            print(f"    {dt.strftime('%Y-%m-%d')}  ${val:>+12,.0f}  ({pct:>+.2f}% of equity)")

    # Bottom line
    print(f"\n{'=' * 100}")
    print("  BOTTOM LINE: $200K compounding over 5.5 years")
    print(f"{'=' * 100}")
    for cfg_name, r in results.items():
        mult = r['final_eq'] / CAP
        print(f"  {cfg_name}: ${CAP:,.0f} -> ${r['final_eq']:,.0f}  ({mult:.0f}x)  "
              f"CAGR {r['cagr']:.0f}%  SR {r['sr']:.2f}  MaxDD {r['max_dd']:.1f}%")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
