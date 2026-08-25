"""LONG straddle sweep at hold=4h on the realistic engine.

Evaluates LONG straddle (signal>0) on the full fired set at hold=4h:
  (a) UN-HEDGED directional bet (total_unhedged_usd)
  (b) DELTA-HEDGED gamma scalp (total_hedged_usd), sweeping the hedge band.

Reports per variant: n_trades, total USD, NON-OVERLAP Sharpe, Newey-West,
MaxDD, per-year signs, gate_no_fake_profit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vol_arb.realistic_eval import RealisticEvalConfig, evaluate_realistic
from vol_arb.honest_pnl import gate_no_fake_profit
from scripts.longhold_hedged_vrp import (
    _sharpe_iid, _newey_west_sharpe, _nonoverlap_sharpe, _max_drawdown,
)

HOLD = "4h"
BANDS = [0.05, 0.10, 0.20, 0.35]
TRADE_LOG = "vol_arb/output/BTC_v2_trade_log.parquet"


def _per_year(pnl, entry_ts):
    yr = entry_ts.year
    out = {}
    for y in sorted(set(yr)):
        out[int(y)] = float(np.nansum(pnl[yr == y]))
    return out


def _gate(f, model_col):
    if "real_option_pnl_nc" in f.columns:
        try:
            gr = gate_no_fake_profit(f, model_col=model_col, real_col="real_option_pnl_nc")
            return bool(gr.passed), gr.detail
        except Exception as ex:
            return None, f"gate error: {ex}"
    return None, "real_option_pnl_nc absent"


def evaluate_variant(tl, band):
    cfg = RealisticEvalConfig(
        hold=HOLD,
        real_bs_delta=True,
        fill_tol_minutes=60.0,
        spread_source="tape",
        delta_hedge_band=band,
    )
    res = evaluate_realistic(tl, cfg)
    f = res.trade_log
    # LONG subset: signal > 0, covered
    cov = f[(f["premium_tier"] > 0) & (f["signal"] > 0)].copy().sort_index()
    return res, f, cov


def report(label, pnl, entry_ts, exit_ts, f_for_gate, gate_col):
    pnl = np.asarray(pnl, float)
    mask = np.isfinite(pnl)
    pnl_v = pnl[mask]
    e = entry_ts[mask]
    x = exit_ts[mask]
    total = float(np.nansum(pnl_v))
    n = len(pnl_v)
    maxdd = _max_drawdown(pnl_v)
    naive = _sharpe_iid(pnl_v)
    nw = _newey_west_sharpe(pnl_v, e, x)
    nonov, n_nonov = _nonoverlap_sharpe(pnl_v, e, x)
    py = _per_year(pnl_v, e)
    gate_passed, gate_msg = _gate(f_for_gate, gate_col)
    print(f"\n### {label}")
    print(f"  n_trades            = {n}")
    print(f"  total USD           = ${total:,.0f}")
    print(f"  NAIVE IID Sharpe    = {naive:.3f}")
    print(f"  NON-OVERLAP Sharpe  = {nonov:.3f}  (n_kept={n_nonov})")
    print(f"  Newey-West Sharpe   = {nw:.3f}")
    print(f"  MaxDD               = ${maxdd:,.0f}")
    print(f"  worst single        = ${float(np.nanmin(pnl_v)) if n else float('nan'):,.0f}")
    py_str = "  ".join(f"{y}:{'+' if s>0 else '-'}(${s:,.0f})" for y, s in py.items())
    print(f"  per-year            = {py_str}")
    print(f"  gate_no_fake_profit = {gate_passed}")
    return dict(label=label, n=n, total=total, naive=naive, nonov=nonov,
                n_nonov=n_nonov, nw=nw, maxdd=maxdd, per_year=py, gate=gate_passed)


def main():
    tl = pd.read_parquet(TRADE_LOG)
    print("=" * 74)
    print(f"LONG STRADDLE SWEEP — hold={HOLD}  (signal>0, full fired set)")
    print("=" * 74)

    results = {}

    # ---- UN-HEDGED (band-independent option leg; compute once via band=0.05) ----
    res, f, cov = evaluate_variant(tl, BANDS[0])
    print(f"\nn_fired={res.n_fired}  coverage={res.coverage_frac:.2%}")
    print(f"LONG covered trades = {len(cov)}")
    entry_ts = pd.DatetimeIndex(pd.to_datetime(cov.index, utc=True))
    exit_ts = entry_ts + pd.Timedelta(minutes=240)
    # clip exit to expiry
    expiry = pd.DatetimeIndex(pd.to_datetime(cov["expiry_dt"].values, utc=True))
    exit_ts = pd.DatetimeIndex(np.where(exit_ts < expiry, exit_ts, expiry - pd.Timedelta(minutes=1)))

    f_long = f[(f["premium_tier"] > 0) & (f["signal"] > 0)]
    results["unhedged"] = report(
        "UN-HEDGED (directional)", cov["total_unhedged_usd"].to_numpy(float),
        entry_ts, exit_ts, f_long, "total_unhedged_usd")

    # ---- HEDGED sweep ----
    for band in BANDS:
        res, f, cov = evaluate_variant(tl, band)
        entry_ts = pd.DatetimeIndex(pd.to_datetime(cov.index, utc=True))
        exit_ts = entry_ts + pd.Timedelta(minutes=240)
        expiry = pd.DatetimeIndex(pd.to_datetime(cov["expiry_dt"].values, utc=True))
        exit_ts = pd.DatetimeIndex(np.where(exit_ts < expiry, exit_ts, expiry - pd.Timedelta(minutes=1)))
        f_long = f[(f["premium_tier"] > 0) & (f["signal"] > 0)]
        results[f"hedged_{band}"] = report(
            f"HEDGED gamma-scalp band={band}", cov["total_hedged_usd"].to_numpy(float),
            entry_ts, exit_ts, f_long, "total_hedged_usd")

    print("\n" + "=" * 74)
    print("SUMMARY")
    print("=" * 74)
    # best by total USD
    best = max(results.values(), key=lambda r: r["total"])
    for r in results.values():
        print(f"  {r['label']:38s} total=${r['total']:>12,.0f}  nonov_S={r['nonov']:.2f}  gate={r['gate']}")
    print(f"\nBEST variant by total USD: {best['label']}  (${best['total']:,.0f})")
    any_pos = any(r["total"] > 0 and r["gate"] for r in results.values())
    print(f"ANY long variant positive & gate-passing: {any_pos}")


if __name__ == "__main__":
    main()
