"""Measure the dollar value of the BTC realized-vol forecast (h60) used as a
VOL-TARGETING SIZING OVERLAY on the S5 Binance directional book.

Non-destructive: imports the historical research backtester
(circuit_breaker_v2_experiment as `cb`) and recreates the S5 historical
PortfolioConfig, runs the FLAT reference ONCE, then post-processes three
sizings of the SAME book by scaling each trade's gross PnL and notional by a
per-period vol-target multiplier s_t and re-deriving the historical cost-adjusted
metrics (Sharpe/Sortino/PnL/MaxDD/Calmar) with the SAME VIP-tier cost model as
run_variant (costs scale ~linearly with size).

Sizings:
  FLAT      s_t = 1                                   (= historical reference)
  FORECAST  s_t = clip(target/forecast_vol_t, lo, hi) (causal candidate overlay)
  REALIZED  s_t = clip(target/realized_vol_t, lo, hi) (oracle, forward-looking)

forecast_vol_t = exp(y_pred) of the BTC h60 model, aggregated to a DAILY
regime scaler and LAGGED one day (uses only forecasts timestamped strictly
before the trading day) -> no lookahead. BTC vol is used as a MARKET-WIDE
regime scaler for the 30-asset book (proxy limitation, reported).

Usage:  PYTHONPATH=src python scripts/forecast_overlay_value.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

import circuit_breaker_v2_experiment as cb  # noqa: E402
from signal_engine.config import PortfolioConfig  # noqa: E402

CAP = cb.CAP                     # 250_000
SPREAD_BPS = cb.SPREAD_BPS
FORECAST_PARQUET = ROOT / "vol_model/output/BTCUSDT_expanding_60m_predictions.parquet"


def historical_portfolio_config() -> PortfolioConfig:
    """Historical S5 settings, matching ``evaluate_s5_historical.py``."""
    return PortfolioConfig(
        starting_capital=CAP, max_dd=0.10, delever_start=0.05, slippage=0.0001,
        max_exposure=0.90, max_per_asset=0.25, max_per_strategy=0.50,
        base_sizing_pct=0.30, max_signal_mult=2.0, sizing_mode="probability_edge",
        use_regime_exposure_multiplier=True, regime_exposure_floor=0.50)


# ── Build the FLAT reference signal set as in the historical evaluator ──
def build_flat_merged(port):
    print("  Generating signals (cached)...", flush=True)
    signal_cache = {}
    for label, resample in cb.NEEDED_TFS:
        sigs = cb.generate_signals_cached(label, resample)
        if sigs is not None:
            signal_cache[label] = sigs
            print(f"    {label}: {len(sigs):,} signals", flush=True)

    from concurrent.futures import ProcessPoolExecutor
    vol_series = {n: v for n, v in ProcessPoolExecutor(max_workers=6).map(
        cb._compute_vol_one, list(cb.ASSETS.items())) if v is not None}
    print(f"  per-asset vol series: {len(vol_series)} assets", flush=True)

    parts = []
    for tf, h in cb.FULL5:
        if tf not in signal_cache:
            continue
        resample = dict(cb.NEEDED_TFS)[tf]
        rec = cb.recompute_returns(signal_cache[tf], resample, h)
        # Per-asset inverse-vol overlay at the historical settings
        # (TARGET_VOL=0.025, floor=0.10, cap=2.0), matching
        # evaluate_s5_historical.py.
        parts.append(cb.apply_vol(rec, vol_series, 0.025, 0.10, 2.0))
    merged = (pd.concat(parts, ignore_index=True)
              .sort_values("timestamp").reset_index(drop=True))
    # prepare cols exactly like run_variant
    for c in ["regime_exposure_mult", "sizing_mult"]:
        if c not in merged.columns:
            merged[c] = 1.0
        merged[c] = merged[c].fillna(1.0)
    merged["probability"] = merged["probability"].astype(float)
    merged = merged.sort_values("timestamp").reset_index(drop=True)
    return merged


# ── Daily forecast/realized vol scaler, lagged (no lookahead) ──
def daily_vol_scalers(target_vol, lo, hi):
    """Return two daily Series (forecast_scale, realized_scale), indexed by
    calendar date, LAGGED one day. forecast_vol = exp(y_pred); realized = exp(y_true)."""
    df = pd.read_parquet(FORECAST_PARQUET).dropna()
    idx = df.index
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    df.index = idx
    fvol = np.exp(df["y_pred"])     # forecast annualized vol level
    rvol = np.exp(df["y_true"])     # realized annualized vol level
    # daily mean of hourly vol
    f_daily = fvol.groupby(fvol.index.normalize()).mean()
    r_daily = rvol.groupby(rvol.index.normalize()).mean()

    def scaler(vol_daily):
        # LAG one day: the scaler applied on day D uses the vol aggregate from
        # day D-1 and earlier only (shift(1)) -> strictly point-in-time.
        v = vol_daily.shift(1)
        s = (target_vol / v).clip(lo, hi)
        s.index = s.index.date  # key by python date for fast lookup
        return s

    return scaler(f_daily), scaler(r_daily), f_daily, r_daily


# ── Historical cost-adjusted metrics for a per-trade scale vector ──
def metrics_for_scale(sim, merged, scale_by_date):
    """Replicate run_variant's daily cost-adjusted metric pipeline, but scale
    each active trade's gross PnL and notional by scale_by_date[trade_date].
    scale_by_date: dict date->float (default 1.0 if missing)."""
    active = ~sim["skipped"]
    pnl = np.asarray(sim["pnl"], dtype=float)
    sizes = np.asarray(sim["sizes"], dtype=float)
    ts_arr = pd.to_datetime(np.asarray(sim["timestamps"]))
    if ts_arr.tz is not None:
        ts_arr = ts_arr.tz_localize(None)
    dates_arr = ts_arr.date

    # per-trade scale
    s = np.array([scale_by_date.get(d, 1.0) for d in dates_arr], dtype=float)

    scaled_pnl = pnl * s
    scaled_size = sizes * s

    # daily gross PnL (scaled), active days only aggregated by resample like
    # compute_daily_pnl_series (which sums active-trade pnl then resamples 1D)
    daily = (pd.Series(scaled_pnl[active], index=pd.to_datetime(sim["timestamps"])[active])
             .resample("1D").sum())
    if daily.index.tz is not None:
        daily.index = daily.index.tz_localize(None)
    t0 = pd.Timestamp(pd.to_datetime(sim["timestamps"])[active][0])
    if t0.tz is not None:
        t0 = t0.tz_localize(None)
    daily = daily[daily.index >= t0]
    if len(daily) < 30:
        return None

    # ── VIP-tier cost model (mirrors run_variant exactly, scaled sizes) ──
    mfp = 0.90
    daily_vol = {}
    for i in range(len(scaled_size)):
        if active[i]:
            daily_vol[dates_arr[i]] = daily_vol.get(dates_arr[i], 0.0) + abs(scaled_size[i]) * 2
    vol_s = pd.Series(daily_vol).sort_index()
    vol_30d = vol_s.rolling(30, min_periods=1).sum()
    eq_curve = CAP + daily.cumsum()
    dpnl_adj = daily.copy()
    # precompute per-date active count
    ndate = {}
    for i in range(len(dates_arr)):
        if active[i]:
            ndate[dates_arr[i]] = ndate.get(dates_arr[i], 0) + 1
    for d in daily.index:
        dd = d.date()
        eq = eq_curve.get(d, CAP)
        v30 = vol_30d.get(dd, 0)
        _, mbps, tbps = cb.get_vip_tier(eq, v30)
        rt = mfp * 2 * mbps + (1 - mfp) * 2 * tbps + SPREAD_BPS
        extra = rt - 1.0
        nd = ndate.get(dd, 0)
        avg = daily_vol.get(dd, 0) / max(nd * 2, 1)
        dpnl_adj[d] -= nd * avg * extra / 10000

    dr = dpnl_adj / CAP
    if dr.std() == 0 or len(dr) < 30:
        return None
    sr = float(dr.mean() / dr.std() * np.sqrt(365.25))
    down = dr[dr < 0]
    sortino = float(dr.mean() / down.std() * np.sqrt(365.25)) if len(down) > 0 and down.std() > 0 else 0.0
    eq_adj = (1 + dr).cumprod()
    dd_s = (eq_adj.cummax() - eq_adj) / eq_adj.cummax()
    max_dd = float(dd_s.max() * 100)
    calmar = float(dr.mean() * 365.25 / (max_dd / 100)) if max_dd > 0 else 0.0
    pnl = float(dpnl_adj.sum())
    # per-year
    py = dpnl_adj.groupby(dpnl_adj.index.year).sum()
    # per-year Sharpe
    pys = {}
    for yr, grp in dr.groupby(dr.index.year):
        if grp.std() > 0 and len(grp) > 5:
            pys[yr] = float(grp.mean() / grp.std() * np.sqrt(365.25))
        else:
            pys[yr] = float("nan")
    return {"sr": sr, "sortino": sortino, "max_dd": max_dd, "calmar": calmar,
            "pnl": pnl, "per_year_pnl": py, "per_year_sr": pys,
            "avg_scale": float(np.mean(s[active])), "daily": dpnl_adj}


def main():
    t0 = time.time()
    out_lines = []

    def log(msg=""):
        print(msg, flush=True)
        out_lines.append(msg)

    log("=" * 80)
    log("  FORECAST VOL-TARGET OVERLAY VALUE on S5 (Binance 30-asset)")
    log("=" * 80)

    port = historical_portfolio_config()
    merged = build_flat_merged(port)
    log(f"  FLAT merged signals: {len(merged):,}")

    # FLAT reference simulation (non-compound, historical config) — one run
    sim = cb.simulate_with_breaker(merged, port, "none", {}, compound=False)
    log(f"  baseline sim: n_trades={sim['n_trades']:,}  halted={sim['halted']}")

    # Persist the sim + minimal merged cols so any windowed/sensitivity
    # post-processing is instant (no expensive re-backtest).
    import pickle
    cache = ROOT / "vol_arb" / "output" / "_overlay_sim_cache.pkl"
    cache.parent.mkdir(parents=True, exist_ok=True)
    with open(cache, "wb") as fh:
        pickle.dump({"pnl": np.asarray(sim["pnl"]), "sizes": np.asarray(sim["sizes"]),
                     "skipped": np.asarray(sim["skipped"]),
                     "timestamps": np.asarray(sim["timestamps"]),
                     "n_trades": sim["n_trades"], "halted": sim["halted"]}, fh)
    log(f"  sim cache -> {cache}")

    # Reproduction check against the previously reported historical result.
    flat = metrics_for_scale(sim, merged, {})  # empty -> all 1.0
    log(f"\n  FLAT (historical reference check): Sharpe={flat['sr']:.3f}  PnL=${flat['pnl']:,.0f}  "
        f"MaxDD={flat['max_dd']:.1f}%  Calmar={flat['calmar']:.2f}")
    log("  (reported reference, not independently certified: "
        "Sharpe 3.40, PnL $1,707,603, MaxDD 32.8%)")

    # forecast/realized scalers across a few clip/target choices (sensitivity)
    f_daily_med = None
    configs = []
    # target = median forecast vol so avg scale ~ 1
    for (lo, hi, tmode) in [(0.33, 3.0, "median"), (0.50, 2.0, "median"),
                            (0.33, 3.0, "p40")]:
        configs.append((lo, hi, tmode))

    log("\n" + "=" * 80)
    log("  SENSITIVITY (clip lo/hi, target choice) — FORECAST vs FLAT vs REALIZED(oracle)")
    log("=" * 80)

    results = {}
    for (lo, hi, tmode) in configs:
        # need the daily median to set target
        df = pd.read_parquet(FORECAST_PARQUET).dropna()
        idx = df.index
        if idx.tz is not None:
            idx = idx.tz_localize(None)
        df.index = idx
        fvol = np.exp(df["y_pred"])
        f_daily = fvol.groupby(fvol.index.normalize()).mean()
        if tmode == "median":
            target = float(f_daily.median())
        elif tmode == "p40":
            target = float(f_daily.quantile(0.40))
        f_scale, r_scale, _, _ = daily_vol_scalers(target, lo, hi)
        fdict = {d: v for d, v in f_scale.items() if np.isfinite(v)}
        rdict = {d: v for d, v in r_scale.items() if np.isfinite(v)}

        m_f = metrics_for_scale(sim, merged, fdict)
        m_r = metrics_for_scale(sim, merged, rdict)
        tag = f"clip[{lo},{hi}] tgt={tmode}"
        results[tag] = (m_f, m_r, target)
        log(f"\n  {tag}  (target_vol={target:.3f})")
        log(f"    {'sizing':<10}{'Sharpe':>8}{'Sortino':>9}{'PnL($)':>14}"
            f"{'MaxDD%':>8}{'Calmar':>8}{'avgScale':>9}")
        log(f"    {'FLAT':<10}{flat['sr']:>8.3f}{flat['sortino']:>9.3f}"
            f"{flat['pnl']:>14,.0f}{flat['max_dd']:>8.1f}{flat['calmar']:>8.2f}{1.0:>9.2f}")
        log(f"    {'FORECAST':<10}{m_f['sr']:>8.3f}{m_f['sortino']:>9.3f}"
            f"{m_f['pnl']:>14,.0f}{m_f['max_dd']:>8.1f}{m_f['calmar']:>8.2f}{m_f['avg_scale']:>9.2f}")
        log(f"    {'REALIZED':<10}{m_r['sr']:>8.3f}{m_r['sortino']:>9.3f}"
            f"{m_r['pnl']:>14,.0f}{m_r['max_dd']:>8.1f}{m_r['calmar']:>8.2f}{m_r['avg_scale']:>9.2f}")
        log(f"    DELTA forecast-vs-flat: Sharpe {m_f['sr']-flat['sr']:+.3f}  "
            f"PnL ${m_f['pnl']-flat['pnl']:+,.0f}")
        log(f"    DELTA realized-vs-flat (oracle ceiling): Sharpe {m_r['sr']-flat['sr']:+.3f}  "
            f"PnL ${m_r['pnl']-flat['pnl']:+,.0f}")

    # per-year stability for the primary config clip[0.33,3.0] tgt=median
    primary = "clip[0.33,3.0] tgt=median"
    m_f, m_r, target = results[primary]
    log("\n" + "=" * 80)
    log(f"  PER-YEAR STABILITY  (primary {primary})")
    log("=" * 80)
    log(f"  {'year':<6}{'FLAT PnL':>14}{'FCST PnL':>14}{'delta':>14}"
        f"{'FLAT SR':>9}{'FCST SR':>9}")
    yrs = sorted(set(flat['per_year_pnl'].index) | set(m_f['per_year_pnl'].index))
    n_better = 0
    for yr in yrs:
        fp = flat['per_year_pnl'].get(yr, 0.0)
        cp = m_f['per_year_pnl'].get(yr, 0.0)
        fsr = flat['per_year_sr'].get(yr, float('nan'))
        csr = m_f['per_year_sr'].get(yr, float('nan'))
        d = cp - fp
        if d > 0:
            n_better += 1
        log(f"  {yr:<6}{fp:>14,.0f}{cp:>14,.0f}{d:>+14,.0f}{fsr:>9.2f}{csr:>9.2f}")
    log(f"  forecast beats flat in {n_better}/{len(yrs)} years")

    # final verdict numbers
    log("\n" + "=" * 80)
    log("  HEADLINE (primary config)")
    log("=" * 80)
    log(f"  Forecast overlay $ value = PnL(forecast) - PnL(flat) = "
        f"${m_f['pnl']-flat['pnl']:+,.0f}")
    log(f"  Sharpe delta             = {m_f['sr']-flat['sr']:+.3f}")
    log(f"  Oracle (realized) ceiling: $ {m_r['pnl']-flat['pnl']:+,.0f}, "
        f"Sharpe {m_r['sr']-flat['sr']:+.3f}")
    log(f"\n  Runtime: {time.time()-t0:.0f}s")

    # save raw log to scratchpad for the report
    (ROOT / "vol_arb" / "output").mkdir(parents=True, exist_ok=True)
    rawlog = ROOT / "vol_arb" / "output" / "_forecast_overlay_raw.txt"
    rawlog.write_text("\n".join(out_lines), encoding="utf-8")
    print(f"\n  raw log -> {rawlog}")


if __name__ == "__main__":
    main()
