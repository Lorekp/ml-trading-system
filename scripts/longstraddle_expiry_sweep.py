"""LONG straddle (signal>0) to-expiry sweep — un-hedged + delta-hedged band sweep.

Mirror of the historical SHORT VRP candidate. Evaluates the LONG straddle on the full
fired set at hold=expiry with the full realistic Deribit fee stack:
  (a) UN-HEDGED long (directional/convexity)  -> total_unhedged_usd
  (b) DELTA-HEDGED long (gamma scalp), band sweep {0.05,0.10,0.20,0.35}
                                              -> total_hedged_usd

Per variant: n_trades(covered), total USD, NON-OVERLAPPING Sharpe (honest headline),
Newey-West Sharpe, MaxDD, per-calendar-year signs, gate_no_fake_profit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vol_arb.realistic_eval import RealisticEvalConfig, evaluate_realistic
from vol_arb.honest_pnl import gate_no_fake_profit
from scripts.longhold_hedged_vrp import (
    _sharpe_iid, _newey_west_sharpe, _nonoverlap_sharpe, _max_drawdown,
)

TRADE_LOG = "vol_arb/output/BTC_v2_trade_log.parquet"
BANDS = [0.05, 0.10, 0.20, 0.35]
FILL_TOL = 60.0


def _long_covered(f: pd.DataFrame) -> pd.DataFrame:
    cov = f[(f["signal"] > 0) & (f["premium_tier"] > 0)].copy()
    cov = cov.sort_index()
    return cov


def _stats(pnl, entry_ts, exit_ts):
    n = len(pnl)
    total = float(np.nansum(pnl))
    maxdd = _max_drawdown(pnl)
    worst = float(np.nanmin(pnl)) if n else float("nan")
    nonov, n_nonov = _nonoverlap_sharpe(pnl, entry_ts, exit_ts)
    nw = _newey_west_sharpe(pnl, entry_ts, exit_ts)
    yr = entry_ts.year
    per_year = {int(y): float(np.nansum(pnl[yr == y])) for y in sorted(set(yr))}
    return dict(n=n, total=total, maxdd=maxdd, worst=worst, nonov=nonov,
                n_nonov=n_nonov, nw=nw, per_year=per_year)


def _gate(f_long):
    if "real_option_pnl_nc" not in f_long.columns:
        return None, "real_option_pnl_nc absent"
    try:
        gr = gate_no_fake_profit(f_long, model_col="total_hedged_usd",
                                 real_col="real_option_pnl_nc")
        return bool(gr.passed), gr.detail
    except Exception as ex:  # noqa
        return None, f"gate error: {ex}"


def main():
    tl = pd.read_parquet(TRADE_LOG)

    # ---- UN-HEDGED (band-invariant; run once at default band) ----
    cfg0 = RealisticEvalConfig(hold="expiry", real_bs_delta=True,
                               fill_tol_minutes=FILL_TOL, spread_source="tape",
                               delta_hedge_band=BANDS[0])
    res0 = evaluate_realistic(tl, cfg0)
    cov0 = _long_covered(res0.trade_log)
    e0 = pd.DatetimeIndex(pd.to_datetime(cov0.index, utc=True))
    x0 = pd.DatetimeIndex(pd.to_datetime(cov0["expiry_dt"].values, utc=True))
    unh = cov0["total_unhedged_usd"].to_numpy(float)
    us = _stats(unh, e0, x0)
    gate_unh = _gate(cov0)

    print("=" * 78)
    print("LONG STRADDLE (signal>0) — hold=EXPIRY — realistic fee stack")
    print(f"fill_tol={FILL_TOL}min   n_long_fired={int((res0.trade_log['signal']>0).sum())}")
    print("=" * 78)
    print(f"[UN-HEDGED]  n_cov={us['n']}  total=${us['total']:,.0f}  "
          f"MaxDD=${us['maxdd']:,.0f}  worst=${us['worst']:,.0f}")
    print(f"   nonoverlap_Sharpe={us['nonov']:.3f} (n_kept={us['n_nonov']})  "
          f"NW_Sharpe={us['nw']:.3f}")
    print(f"   per_year={ {y: round(v) for y,v in us['per_year'].items()} }")
    print(f"   gate_no_fake_profit={gate_unh[0]}")
    print("-" * 78)

    results = {"unhedged": (us, gate_unh)}

    # ---- DELTA-HEDGED band sweep ----
    for b in BANDS:
        cfg = RealisticEvalConfig(hold="expiry", real_bs_delta=True,
                                  fill_tol_minutes=FILL_TOL, spread_source="tape",
                                  delta_hedge_band=b)
        res = evaluate_realistic(tl, cfg)
        cov = _long_covered(res.trade_log)
        e = pd.DatetimeIndex(pd.to_datetime(cov.index, utc=True))
        x = pd.DatetimeIndex(pd.to_datetime(cov["expiry_dt"].values, utc=True))
        hed = cov["total_hedged_usd"].to_numpy(float)
        st = _stats(hed, e, x)
        gate = _gate(cov)
        results[f"hedged_band{b}"] = (st, gate)
        print(f"[HEDGED band={b:.2f}]  n_cov={st['n']}  total=${st['total']:,.0f}  "
              f"MaxDD=${st['maxdd']:,.0f}  worst=${st['worst']:,.0f}")
        print(f"   nonoverlap_Sharpe={st['nonov']:.3f} (n_kept={st['n_nonov']})  "
              f"NW_Sharpe={st['nw']:.3f}")
        print(f"   per_year={ {y: round(v) for y,v in st['per_year'].items()} }")
        print(f"   gate_no_fake_profit={gate[0]}")
        print("-" * 78)

    # ---- best variant ----
    best_key, best_total = None, -1e18
    for k, (st, _) in results.items():
        if st["total"] > best_total:
            best_total, best_key = st["total"], k
    print("=" * 78)
    print(f"BEST variant by total USD = {best_key}  (${best_total:,.0f})")
    any_pos = any(st["total"] > 0 and g[0] for st, g in results.values())
    print(f"ANY long variant positive & gate-passing? {any_pos}")
    print("=" * 78)


if __name__ == "__main__":
    main()
