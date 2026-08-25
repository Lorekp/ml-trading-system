#!/usr/bin/env python
"""Config C backtest: 2026 only, $100 starting capital.

Usage:
    PYTHONPATH="src;." python experiments/yolo_2026.py
"""

import sys, os, time, warnings, pickle
import numpy as np
import pandas as pd
from pathlib import Path

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src"); sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

from experiments.drawdown_improvements import simulate_with_improvements, CACHE_DIR
from experiments.circuit_breaker_v2_experiment import apply_vol
from experiments.dynamic_asset_v2_experiment import assemble_signals, OBVIOUS_CORE
from signal_engine.portfolio import compute_daily_pnl_series
from signal_engine.config import PortfolioConfig

CAP = 100.0
N_MC = 10000
SEED = 42

def mc_bankruptcy(dr_values, n_sims=N_MC):
    rng = np.random.default_rng(SEED)
    n = len(dr_values)
    bs = min(5, n // 2)
    nb = n // bs
    if nb < 2:
        return None
    starts = rng.integers(0, max(n - bs, 1), size=(n_sims, nb))
    off = np.arange(bs)
    idx = (starts[:, :, None] + off[None, None, :]).reshape(n_sims, -1)
    boot = dr_values[idx]
    cum = np.cumprod(1.0 + boot, axis=1)
    final_eq = cum[:, -1]
    rm = np.maximum.accumulate(cum, axis=1)
    dds = (rm - cum) / rm
    max_dds = dds.max(axis=1)
    min_eq = np.minimum.accumulate(cum, axis=1).min(axis=1)
    worst_days = boot.min(axis=1)

    return {
        "final_eq_median": float(np.median(final_eq)) * CAP,
        "final_eq_mean": float(final_eq.mean()) * CAP,
        "final_eq_p5": float(np.percentile(final_eq, 5)) * CAP,
        "final_eq_p10": float(np.percentile(final_eq, 10)) * CAP,
        "final_eq_p25": float(np.percentile(final_eq, 25)) * CAP,
        "final_eq_p75": float(np.percentile(final_eq, 75)) * CAP,
        "final_eq_p90": float(np.percentile(final_eq, 90)) * CAP,
        "final_eq_p95": float(np.percentile(final_eq, 95)) * CAP,
        "return_median": float((np.median(final_eq) - 1) * 100),
        "max_dd_median": float(np.median(max_dds) * 100),
        "max_dd_p95": float(np.percentile(max_dds, 95) * 100),
        "p_lose_25pct": float((min_eq < 0.75).mean() * 100),
        "p_lose_50pct": float((min_eq < 0.50).mean() * 100),
        "p_lose_75pct": float((min_eq < 0.25).mean() * 100),
        "p_lose_90pct": float((min_eq < 0.10).mean() * 100),
        "p_end_negative": float((final_eq < 1.0).mean() * 100),
        "p_double": float((final_eq >= 2.0).mean() * 100),
        "p_5x": float((final_eq >= 5.0).mean() * 100),
        "p_10x": float((final_eq >= 10.0).mean() * 100),
        "worst_day_p1": float(np.percentile(worst_days, 1) * 100),
        "worst_day_p5": float(np.percentile(worst_days, 5) * 100),
    }


def main():
    T0 = time.time()
    print("=" * 90)
    print("  CONFIG C — 2026 ONLY — $100 starting capital")
    print("  20% sizing | no delever | no CT | maxSig 20")
    print("=" * 90)

    with open(CACHE_DIR / "v2_prep_cache.pkl", "rb") as f:
        cached = pickle.load(f)
    hold_sigs = cached["hold_sigs"]
    vol_series = cached["vol_series"]
    asset_volumes = cached["asset_volumes"]

    # Use months ending at 2026-02 (latest in data)
    # But we need signals assembled with history for model training
    months = pd.period_range("2020-07", "2026-02", freq="M")

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
    ts = pd.to_datetime(prepared["timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)
    prepared_2026 = prepared[ts >= "2026-01-01"].copy().reset_index(drop=True)
    print(f"  {len(prepared):,} total signals, {len(prepared_2026):,} in 2026 ({time.time()-T0:.0f}s)\n")

    # Config C
    port = PortfolioConfig(
        starting_capital=CAP, max_dd=1.00, delever_start=1.00, slippage=0.0001,
        max_exposure=0.95, max_per_asset=0.25, max_per_strategy=0.50,
        base_sizing_pct=0.200, max_signal_mult=2.0, sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)

    breaker = {"window": 10, "threshold": -0.01, "floor": 0.10,
               "up_threshold": 0.01, "cap": 1.5}

    improvements = {"max_signals_per_bar": 20}

    # Run backtest
    sim = simulate_with_improvements(prepared_2026, port, breaker, improvements)
    if sim is None:
        print("  ERROR: sim returned None")
        return
    dpnl = compute_daily_pnl_series(sim)
    if dpnl is None:
        print("  ERROR: no daily PnL")
        return
    if dpnl.index.tz is not None:
        dpnl.index = dpnl.index.tz_localize(None)

    dr = dpnl / CAP
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25)) if dr.std() > 0 else 0
    eq = (1 + dr).cumprod()
    dd_s = (eq.cummax() - eq) / eq.cummax()
    pnl = float(dpnl.sum())
    max_dd = float(dd_s.max() * 100)
    worst_day = float(dr.min() * 100)
    win_rate = float((dpnl > 0).mean() * 100)
    final_eq = CAP + pnl

    print(f"  BACKTEST RESULTS")
    print(f"  " + "-" * 50)
    print(f"  Period:       {dpnl.index[0].date()} to {dpnl.index[-1].date()} ({len(dpnl)} days)")
    print(f"  Sharpe:       {sr:.2f}")
    print(f"  Total PnL:    ${pnl:,.2f}")
    print(f"  Final Equity: ${final_eq:,.2f}")
    print(f"  Return:       {pnl/CAP*100:.1f}%")
    print(f"  Max DD:       {max_dd:.2f}%")
    print(f"  Worst Day:    {worst_day:.2f}%")
    print(f"  Win Rate:     {win_rate:.1f}%")
    print(f"  Trades:       {sim['n_trades']:,}")
    print(f"  PnL/day:      ${pnl/len(dpnl):.2f}")

    # Equity curve
    eq_dollar = CAP * eq
    print(f"\n  EQUITY CURVE (weekly snapshots)")
    print(f"  " + "-" * 50)
    weekly = eq_dollar.resample("W").last()
    for dt, val in weekly.items():
        bar_len = max(int((val - CAP) / CAP * 50), 0) if val >= CAP else 0
        bar = "#" * min(bar_len, 60)
        dd_now = float(dd_s.loc[:dt].iloc[-1] * 100) if dt in dd_s.index or len(dd_s.loc[:dt]) > 0 else 0
        print(f"  {dt.strftime('%Y-%m-%d')}  ${val:>7.2f}  {bar}")

    # Daily PnL
    print(f"\n  WORST 5 DAYS")
    print(f"  " + "-" * 50)
    worst5 = dpnl.nsmallest(5)
    for dt, val in worst5.items():
        print(f"  {dt.strftime('%Y-%m-%d')}  ${val:>+8.2f}  ({val/CAP*100:>+.2f}%)")

    print(f"\n  BEST 5 DAYS")
    print(f"  " + "-" * 50)
    best5 = dpnl.nlargest(5)
    for dt, val in best5.items():
        print(f"  {dt.strftime('%Y-%m-%d')}  ${val:>+8.2f}  ({val/CAP*100:>+.2f}%)")

    # Monthly
    monthly = dpnl.resample("MS").sum()
    print(f"\n  MONTHLY PNL")
    print(f"  " + "-" * 50)
    cum_pnl = 0
    for dt, mp in monthly.items():
        cum_pnl += mp
        print(f"  {dt.strftime('%Y-%m')}  ${mp:>+8.2f}  (cum: ${cum_pnl:>8.2f})")

    # MC
    print(f"\n  MONTE CARLO ({N_MC:,} sims)")
    print(f"  " + "-" * 50)
    mc = mc_bankruptcy(dr.values)
    if mc is None:
        print("  Not enough data for MC")
        return

    print(f"  Final equity distribution:")
    print(f"    p5  (worst 5%):   ${mc['final_eq_p5']:>8.2f}")
    print(f"    p10:              ${mc['final_eq_p10']:>8.2f}")
    print(f"    p25:              ${mc['final_eq_p25']:>8.2f}")
    print(f"    MEDIAN:           ${mc['final_eq_median']:>8.2f}")
    print(f"    p75:              ${mc['final_eq_p75']:>8.2f}")
    print(f"    p90:              ${mc['final_eq_p90']:>8.2f}")
    print(f"    p95 (best 5%):    ${mc['final_eq_p95']:>8.2f}")

    print(f"\n  Risk of ruin:")
    print(f"    P(ever lose >25%):  {mc['p_lose_25pct']:.1f}%")
    print(f"    P(ever lose >50%):  {mc['p_lose_50pct']:.1f}%")
    print(f"    P(ever lose >75%):  {mc['p_lose_75pct']:.1f}%")
    print(f"    P(ever lose >90%):  {mc['p_lose_90pct']:.1f}%")
    print(f"    P(end in the red):  {mc['p_end_negative']:.1f}%")

    print(f"\n  Upside probability:")
    print(f"    P(double $100->$200): {mc['p_double']:.1f}%")
    print(f"    P(5x $100->$500):     {mc['p_5x']:.1f}%")
    print(f"    P(10x $100->$1000):   {mc['p_10x']:.1f}%")

    print(f"\n  Drawdown:")
    print(f"    Median max DD:       {mc['max_dd_median']:.1f}%")
    print(f"    p95 max DD:          {mc['max_dd_p95']:.1f}%")
    print(f"    Worst day p1:        {mc['worst_day_p1']:.2f}%")
    print(f"    Worst day p5:        {mc['worst_day_p5']:.2f}%")

    print(f"\n  Total runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
