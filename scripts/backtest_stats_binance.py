"""Full backtest stats — Binance 32-asset system @ LIVE config.

Runs the experiments backtester (the true live twin) ONCE for signal+vol
prep, then simulates twice: compound=True (gross-compound) and compound=False
(non-compound, fixed-capital sizing — the honest planning metric).

Reports for each mode:
  - Total PnL
  - Annualized Sharpe (daily log-returns for compound; daily PnL / CAP for NC)
  - Sortino (downside-deviation Sharpe)
  - Calmar (annualized return / |MaxDD|)
  - MaxDD % (peak-to-trough on the actual equity curve)
  - Probabilistic Sharpe (PSR vs SR0=0)
  - Deflated Sharpe (DSR vs N=100 implicit trials)
  - n_trades, win-rate, profit factor
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import circuit_breaker_v2_experiment as cb  # noqa: E402
from signal_engine.config import PortfolioConfig  # noqa: E402


TARGET_VOL, FLOOR, CAP_MULT = 0.025, 0.10, 2.0
DEFLATE_N_TRIALS = 100  # rough estimate of variant count tried over project life


def live_port() -> PortfolioConfig:
    return PortfolioConfig(
        starting_capital=cb.CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
        max_exposure=0.90, max_per_asset=0.25, max_per_strategy=0.50,
        base_sizing_pct=0.30, max_signal_mult=2.0, sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)


def build_merged(signal_cache, vol_series):
    parts = []
    for tf, h in cb.FULL5:
        if tf not in signal_cache:
            continue
        resample = dict(cb.NEEDED_TFS)[tf]
        sigs = cb.recompute_returns(signal_cache[tf], resample, h)
        sigs = cb.apply_vol(sigs, vol_series, TARGET_VOL, FLOOR, CAP_MULT)
        parts.append(sigs)
    return (pd.concat(parts, ignore_index=True)
            .sort_values("timestamp").reset_index(drop=True))


def cost_adjusted_daily_pnl(sim, prepared):
    """Replicate run_variant's calibrated VIP-tier cost adjustment but
    return the daily PnL series (in $)."""
    dpnl = cb.compute_daily_pnl_series(sim)
    if dpnl is None:
        return None
    if dpnl.index.tz is not None:
        dpnl = dpnl.copy()
        dpnl.index = dpnl.index.tz_localize(None)
    mfp = 0.90
    active = ~sim["skipped"]
    sizes = sim["sizes"]
    ts_arr = pd.to_datetime(prepared["timestamp"].values)
    if ts_arr.tz is not None:
        ts_arr = ts_arr.tz_localize(None)
    dates_arr = ts_arr.date
    daily_vol = {}
    for i in range(len(prepared)):
        if active[i]:
            daily_vol[dates_arr[i]] = daily_vol.get(dates_arr[i], 0) + abs(sizes[i]) * 2
    vol_s = pd.Series(daily_vol).sort_index()
    vol_30d = vol_s.rolling(30, min_periods=1).sum()
    eq_curve = cb.CAP + dpnl.cumsum()
    dpnl_adj = dpnl.copy()
    for d in dpnl.index:
        dd = d.date()
        eq = eq_curve.get(d, cb.CAP)
        v30 = vol_30d.get(dd, 0)
        _, mbps, tbps = cb.get_vip_tier(eq, v30)
        rt = mfp * 2 * mbps + (1 - mfp) * 2 * tbps + cb.SPREAD_BPS
        extra = rt - 1.0
        nd = sum(1 for i in range(len(prepared)) if active[i] and dates_arr[i] == dd)
        avg = daily_vol.get(dd, 0) / max(nd * 2, 1)
        dpnl_adj[d] -= nd * avg * extra / 10000
    return dpnl_adj


def deflated_sharpe(sr_obs, T, skew, kurt, n_trials=1):
    """Bailey & López de Prado (2014). Returns DSR (probability the true
    Sharpe exceeds the expected best of `n_trials` random trials).
    With n_trials=1 this is PSR (vs SR0=0)."""
    if n_trials > 1:
        # E[max SR | n_trials random] under SR_true=0 and σ(SR)=1
        gamma = 0.5772156649  # Euler-Mascheroni
        sr0 = ((1 - gamma) * norm.ppf(1 - 1.0 / n_trials)
               + gamma * norm.ppf(1 - 1.0 / (n_trials * np.e)))
    else:
        sr0 = 0.0
    # Sharpe estimator variance (Mertens 2002)
    denom = np.sqrt(max(1 - skew * sr_obs + (kurt - 1) / 4 * sr_obs ** 2, 1e-9))
    z = (sr_obs - sr0) * np.sqrt(T - 1) / denom
    return float(norm.cdf(z)), sr0


def stats(sim, prepared, mode: str) -> dict:
    eq = np.asarray(sim["equity"], dtype=float)
    ts = pd.to_datetime(sim["timestamps"])
    if ts.tz is not None:
        ts = ts.tz_localize(None)
    eq_curve = pd.Series(eq, index=ts).sort_index()
    eq_daily = eq_curve.resample("D").last().ffill()
    peak = eq_daily.cummax()
    dd = (eq_daily - peak) / peak
    max_dd_pct = float(dd.min())
    # Cost-adjusted daily PnL (replicates run_variant's cost loop)
    dpnl_adj = cost_adjusted_daily_pnl(sim, prepared)
    n_years = (eq_daily.index[-1] - eq_daily.index[0]).days / 365.25
    # Returns series for Sharpe
    if mode == "compound":
        # Apply daily cost adjustment to the equity curve
        # (subtract cumulative extra cost from the equity track)
        cost_eq = eq_daily.copy()
        cost_adj_cum = (cb.compute_daily_pnl_series(sim) - dpnl_adj).cumsum().reindex(eq_daily.index, method="ffill").fillna(0)
        cost_eq = cost_eq - cost_adj_cum
        daily_ret = np.log(cost_eq / cost_eq.shift(1)).dropna()
        total_pnl = float(cost_eq.iloc[-1] - cb.CAP)
        cagr = float((cost_eq.iloc[-1] / cost_eq.iloc[0]) ** (1.0 / n_years) - 1.0)
    else:
        daily_ret = (dpnl_adj / cb.CAP).dropna()
        total_pnl = float(dpnl_adj.sum())
        # NC "annualized return" = mean daily * 365 (no compounding)
        cagr = float(daily_ret.mean() * 365.25)
    T = len(daily_ret)
    mu, sigma = float(daily_ret.mean()), float(daily_ret.std())
    sr = mu / sigma * np.sqrt(365.25) if sigma > 0 else 0.0
    down = daily_ret[daily_ret < 0]
    sortino = (mu / down.std() * np.sqrt(365.25)) if (len(down) > 0 and down.std() > 0) else 0.0
    calmar = (cagr / abs(max_dd_pct)) if max_dd_pct < 0 else float("inf")
    skew = float(daily_ret.skew())
    # pandas .kurt() returns EXCESS kurtosis; the DSR formula uses kurtosis
    kurt = float(daily_ret.kurt()) + 3.0
    psr, _ = deflated_sharpe(sr, T, skew, kurt, n_trials=1)
    dsr, sr0 = deflated_sharpe(sr, T, skew, kurt, n_trials=DEFLATE_N_TRIALS)
    # Trade-level
    active = ~sim["skipped"]
    pnl = np.asarray(sim["pnl"])[active]
    win_rate = float((pnl > 0).mean()) if len(pnl) else 0.0
    profit_factor = (pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum())) if (pnl < 0).any() else float("inf")
    return {
        "mode": mode, "n_trades": int(active.sum()),
        "n_days": T, "n_years": round(n_years, 2),
        "total_pnl": total_pnl, "cagr_or_ann_ret": cagr,
        "sharpe": sr, "sortino": sortino, "calmar": calmar,
        "max_dd_pct": max_dd_pct * 100, "win_rate": win_rate,
        "profit_factor": float(profit_factor),
        "psr": psr, "dsr_n100": dsr, "dsr_sr0": sr0,
        "skew": skew, "kurt": kurt,
    }


def fmt(s: dict) -> str:
    return (f"  n_trades        {s['n_trades']:>12,d}\n"
            f"  span (days/yrs) {s['n_days']:>6,d} d  /  {s['n_years']:.2f} yrs\n"
            f"  total PnL       ${s['total_pnl']:>14,.0f}\n"
            f"  CAGR / ann.ret  {s['cagr_or_ann_ret']*100:>13.2f} %\n"
            f"  MaxDD (P->T)    {s['max_dd_pct']:>13.2f} %\n"
            f"  Sharpe (ann)    {s['sharpe']:>13.3f}\n"
            f"  Sortino (ann)   {s['sortino']:>13.3f}\n"
            f"  Calmar          {s['calmar']:>13.3f}\n"
            f"  Profit factor   {s['profit_factor']:>13.2f}\n"
            f"  Win rate        {s['win_rate']*100:>13.2f} %\n"
            f"  PSR (SR0=0)     {s['psr']*100:>13.2f} %  (prob true SR > 0)\n"
            f"  DSR (N=100)     {s['dsr_n100']*100:>13.2f} %  (SR0={s['dsr_sr0']:.3f})\n"
            f"  skew / kurt     {s['skew']:+.3f} / {s['kurt']:.3f}")


def main() -> None:
    t0 = time.time()
    print("=" * 78)
    print(f"  BACKTEST STATS — Binance {len(cb.ASSETS)}-asset @ LIVE config")
    print(f"  base_sizing_pct=0.30  max_per_asset=0.25  floor={FLOOR} cap={CAP_MULT}")
    print(f"  calibrated VIP-tier costs (~3.9 bps RT)")
    print("=" * 78)
    print("  loading signals...", flush=True)
    sigs = {l: cb.generate_signals_cached(l, r) for l, r in cb.NEEDED_TFS}
    sigs = {k: v for k, v in sigs.items() if v is not None}
    from concurrent.futures import ProcessPoolExecutor
    vol = {n: v for n, v in ProcessPoolExecutor(max_workers=8).map(
        cb._compute_vol_one, list(cb.ASSETS.items())) if v is not None}
    print(f"  vol series for {len(vol)} assets", flush=True)
    t1 = time.time()
    merged = build_merged(sigs, vol)
    # Prepare once (same for both modes)
    prepared = merged.copy()
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in prepared.columns:
            prepared[c] = 1.0
        prepared[c] = prepared[c].fillna(1.0)
    prepared["probability"] = prepared["probability"].astype(float)
    prepared = prepared.sort_values("timestamp").reset_index(drop=True)
    print(f"  prepared {len(prepared):,} signals ({time.time()-t1:.0f}s)", flush=True)
    port = live_port()

    print("\n  running compound sim...", flush=True)
    sim_c = cb.simulate_with_breaker(prepared, port, "none", {}, compound=True)
    print("  running non-compound sim...", flush=True)
    sim_n = cb.simulate_with_breaker(prepared, port, "none", {}, compound=False)

    s_c = stats(sim_c, prepared, "compound")
    s_n = stats(sim_n, prepared, "non-compound")

    print("\n" + "=" * 78)
    print("  COMPOUND (size grows with equity — late-period dominates)")
    print("=" * 78)
    print(fmt(s_c))
    print("\n" + "=" * 78)
    print("  NON-COMPOUND (size off fixed $250k — the honest planning metric)")
    print("=" * 78)
    print(fmt(s_n))
    print(f"\n  Total runtime: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
