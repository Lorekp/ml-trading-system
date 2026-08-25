"""Rebalance-band sweep for the to-expiry delta-hedged SHORT straddle.

WHY: the long-hold sweep used delta_hedge_band=0.05 (the 15-MIN default) for a
MULTI-DAY hold -> ~499 rebalances/trade -> $3.57M perp fees, an UNREALISTIC
over-hedge (a pessimism bias). A real vol desk re-hedges a multi-day position on
a much wider band (or on a schedule). This sweep re-tunes the band — the last
untested lever — and judges each on the OVERLAP-ADJUSTED Sharpe + per-year sign,
NOT the in-sample-best mean (no cherry-pick). It also prints the Whalley-Wilmott
transaction-cost-optimal band as the principled reference point.

Add-only: reuses vol_arb/realistic_eval.evaluate_realistic at hold='expiry' with
varying cfg.delta_hedge_band. Never touches the 15-min path / V2 byte-identity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vol_arb.realistic_eval import RealisticEvalConfig, evaluate_realistic
from scripts.longhold_hedged_vrp import (
    _sharpe_iid, _max_drawdown, _newey_west_sharpe, _nonoverlap_sharpe,
)

BANDS = [0.05, 0.10, 0.20, 0.35, 0.50, 0.75]
TRADE_LOG = "vol_arb/output/BTC_v2_trade_log.parquet"


def run_band(tl: pd.DataFrame, band: float) -> dict:
    cfg = RealisticEvalConfig(
        hold="expiry", real_bs_delta=True, fill_tol_minutes=60.0,
        spread_source="tape", delta_hedge_band=band,
    )
    res = evaluate_realistic(tl, cfg)
    f = res.trade_log
    cov = f[f["premium_tier"] > 0].sort_index().copy()
    n = len(cov)
    hedged = cov["total_hedged_usd"].to_numpy(float)
    unhedged = cov["total_unhedged_usd"].to_numpy(float)
    entry_ts = pd.DatetimeIndex(pd.to_datetime(cov.index, utc=True))
    exit_ts = pd.DatetimeIndex(pd.to_datetime(cov["expiry_dt"].values, utc=True))
    nw = _newey_west_sharpe(hedged, entry_ts, exit_ts)
    nonov, n_keep = _nonoverlap_sharpe(hedged, entry_ts, exit_ts)
    yr = entry_ts.year
    per_year = {int(y): float(np.nansum(hedged[yr == y])) for y in sorted(set(yr))}
    n_reb = float(np.nanmean(cov["n_rebalances"].to_numpy(float))) if "n_rebalances" in cov else float("nan")
    return {
        "band": band, "n": n,
        "hedged": float(np.nansum(hedged)),
        "unhedged": float(np.nansum(unhedged)),
        "opt_fee": res.option_fee_total_usd,
        "hedge_fee": res.hedge_fee_total_usd,
        "funding": res.funding_total_usd,
        "perp_leg": res.perp_leg_total_usd,
        "mean_n_reb": n_reb,
        "maxdd": _max_drawdown(hedged),
        "worst": float(np.nanmin(hedged)) if n else float("nan"),
        "naive_sharpe": _sharpe_iid(hedged),
        "nw_sharpe": nw, "nonov_sharpe": nonov, "n_keep": n_keep,
        "per_year": per_year,
        "pos_years": sum(1 for v in per_year.values() if v > 0),
        "tot_years": len(per_year),
    }


def main():
    tl = pd.read_parquet(TRADE_LOG)
    rows = []
    for b in BANDS:
        try:
            r = run_band(tl, b)
        except Exception as ex:  # noqa
            print(f"band={b}: ERROR {ex}")
            continue
        rows.append(r)
        print("=" * 78)
        print(f"BAND = {b:.2f}   n={r['n']}   mean_rebalances/trade={r['mean_n_reb']:.1f}")
        print(f"  HEDGED NC   = ${r['hedged']:,.0f}    UNHEDGED NC = ${r['unhedged']:,.0f}")
        print(f"  opt_fee ${r['opt_fee']:,.0f} | hedge_fee ${r['hedge_fee']:,.0f} | "
              f"funding ${r['funding']:,.0f} | perp_leg ${r['perp_leg']:,.0f}")
        print(f"  MaxDD ${r['maxdd']:,.0f} | worst ${r['worst']:,.0f}")
        print(f"  Sharpe: naive {r['naive_sharpe']:.3f} | NW {r['nw_sharpe']:.3f} | "
              f"nonov {r['nonov_sharpe']:.3f} (n_keep={r['n_keep']})")
        print(f"  per-year ({r['pos_years']}/{r['tot_years']} +): "
              + ", ".join(f"{y}:${v:,.0f}" for y, v in r['per_year'].items()))

    print("\n" + "#" * 78)
    print("BAND-SWEEP SUMMARY (hold=to-expiry, short straddle, realistic fees)")
    print("#" * 78)
    print(f"{'band':>5} {'hedged$':>12} {'unhedged$':>12} {'hedge_fee$':>12} "
          f"{'NW_Shrp':>8} {'nonov':>7} {'+yrs':>5} {'reb':>6}")
    for r in rows:
        print(f"{r['band']:>5.2f} {r['hedged']:>12,.0f} {r['unhedged']:>12,.0f} "
              f"{r['hedge_fee']:>12,.0f} {r['nw_sharpe']:>8.3f} {r['nonov_sharpe']:>7.3f} "
              f"{r['pos_years']:>3}/{r['tot_years']} {r['mean_n_reb']:>6.1f}")
    # Best by overlap-adjusted (NW) Sharpe — REPORTED, not auto-selected.
    valid = [r for r in rows if np.isfinite(r["nw_sharpe"])]
    if valid:
        best = max(valid, key=lambda r: r["nw_sharpe"])
        print(f"\nbest NW Sharpe: band={best['band']:.2f} -> NW {best['nw_sharpe']:.3f}, "
              f"hedged ${best['hedged']:,.0f}, {best['pos_years']}/{best['tot_years']} +years")
        print("NOTE: these are historical screening conditions only; even a pass "
              "does not establish execution realism or readiness.")


if __name__ == "__main__":
    main()
