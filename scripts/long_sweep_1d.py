"""LONG straddle sweep at hold=1d on the FULL fired set.

Forces signal=+1 on every fire-able bar (signal!=0) of the V2 trade log, then
runs the realistic engine at hold=1d. Reports BOTH un-hedged (directional,
total_unhedged_usd) and delta-hedged/gamma-scalp (total_hedged_usd) variants;
for hedged, sweeps the delta-hedge band {0.05,0.10,0.20,0.35}.

ADD-ONLY post-processor; never edits backtester/honest_pnl. Per-variant:
n_trades, total USD, non-overlapping Sharpe (headline) + Newey-West, MaxDD,
per-calendar-year signs, gate_no_fake_profit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vol_arb.realistic_eval import RealisticEvalConfig, evaluate_realistic
from vol_arb.honest_pnl import gate_no_fake_profit
from scripts.longhold_hedged_vrp import (
    _sharpe_iid, _newey_west_sharpe, _nonoverlap_sharpe, _max_drawdown,
)

HOLD = "1d"
BANDS = [0.05, 0.10, 0.20, 0.35]
FILL_TOL = 60.0


def per_year(vals, years):
    out = {}
    for y in sorted(set(years)):
        out[int(y)] = float(np.nansum(vals[years == y]))
    return out


def fmt_year(d):
    return " ".join(f"{y}:{'+' if v > 0 else '-'}" for y, v in d.items())


def main():
    tl = pd.read_parquet("vol_arb/output/BTC_v2_trade_log.parquet")
    fired = tl[tl["signal"] != 0].copy()
    fired["signal"] = 1.0  # force LONG on the full fired set
    print(f"Full fired set forced LONG: n={len(fired)}")

    # ---- Un-hedged is band-independent; compute once (band=0.05 placeholder) ----
    results = {}

    for band in BANDS:
        cfg = RealisticEvalConfig(
            hold=HOLD,
            real_bs_delta=True,
            fill_tol_minutes=FILL_TOL,
            spread_source="tape",
            delta_hedge_band=band,
        )
        res = evaluate_realistic(fired, cfg)
        f = res.trade_log
        cov = f[f["premium_tier"] > 0].copy().sort_index()
        n = len(cov)
        entry_ts = pd.DatetimeIndex(pd.to_datetime(cov.index, utc=True))
        # hold=1d exit
        exit_ts = entry_ts + pd.Timedelta(days=1)
        years = entry_ts.year.values

        hedged = cov["total_hedged_usd"].to_numpy(float)
        unhedged = cov["total_unhedged_usd"].to_numpy(float)

        # gate
        gate_passed = None
        gmsg = ""
        if "real_option_pnl_nc" in f.columns:
            try:
                gr = gate_no_fake_profit(f, model_col="total_hedged_usd",
                                         real_col="real_option_pnl_nc")
                gate_passed = bool(gr.passed)
                gmsg = str(gr.detail)
            except Exception as ex:
                gmsg = f"gate err {ex}"

        results[band] = dict(
            n=n,
            hedged_total=float(np.nansum(hedged)),
            unhedged_total=float(np.nansum(unhedged)),
            maxdd_hedged=_max_drawdown(hedged),
            maxdd_unhedged=_max_drawdown(unhedged),
            worst=float(np.nanmin(hedged)) if n else float("nan"),
            nonov_h=_nonoverlap_sharpe(hedged, entry_ts, exit_ts),
            nw_h=_newey_west_sharpe(hedged, entry_ts, exit_ts),
            nonov_u=_nonoverlap_sharpe(unhedged, entry_ts, exit_ts),
            nw_u=_newey_west_sharpe(unhedged, entry_ts, exit_ts),
            py_h=per_year(hedged, years),
            py_u=per_year(unhedged, years),
            gate=gate_passed,
            gmsg=gmsg,
            cov=res.coverage_frac,
        )

    # ---- Report ----
    print("=" * 78)
    print(f"LONG STRADDLE @ hold={HOLD}  (full fired set forced signal=+1)")
    print("=" * 78)
    b0 = results[BANDS[0]]
    print(f"coverage = {b0['cov']:.2%}   n_trades(covered) = {b0['n']}")
    print("-" * 78)
    print("UN-HEDGED (directional, band-independent):")
    print(f"  total_unhedged_usd = ${b0['unhedged_total']:,.0f}")
    print(f"  MaxDD = ${b0['maxdd_unhedged']:,.0f}")
    print(f"  nonoverlap Sharpe = {b0['nonov_u'][0]:.3f} (n_kept={b0['nonov_u'][1]})   NW = {b0['nw_u']:.3f}")
    print(f"  per-year: {fmt_year(b0['py_u'])}")
    print("-" * 78)
    print("HEDGED (gamma scalp) by band:")
    for b in BANDS:
        r = results[b]
        print(f"  band={b:.2f}: total=${r['hedged_total']:,.0f}  MaxDD=${r['maxdd_hedged']:,.0f}  "
              f"worst=${r['worst']:,.0f}")
        print(f"            nonov Sharpe={r['nonov_h'][0]:.3f}(n={r['nonov_h'][1]})  NW={r['nw_h']:.3f}  gate={r['gate']}")
        print(f"            per-year: {fmt_year(r['py_h'])}")
    print("=" * 78)

    # Best variant selection
    best_band = max(BANDS, key=lambda b: results[b]["hedged_total"])
    best_h = results[best_band]["hedged_total"]
    best_u = b0["unhedged_total"]
    if best_u > best_h:
        print(f"BEST: unhedged  total=${best_u:,.0f}")
    else:
        print(f"BEST: hedged band={best_band}  total=${best_h:,.0f}")

    # gate across all positives
    any_pos = (best_u > 0) or any(results[b]["hedged_total"] > 0 for b in BANDS)
    print(f"ANY long variant positive: {any_pos}")
    print(f"gate detail (best hedged band {best_band}): {results[best_band]['gmsg'][:200]}")


if __name__ == "__main__":
    main()
