"""LONG straddle (signal>0) sweep at hold=15m on the realistic engine.

Evaluates BOTH the un-hedged directional long (total_unhedged_usd) and the
delta-hedged gamma-scalp long (total_hedged_usd), sweeping the delta-hedge band
(longs favour a TIGHTER band to scalp realized variance). Full Deribit fee stack,
real BTC path, honest fills. Reports per variant: n_trades, total USD,
non-overlapping Sharpe (honest headline) + Newey-West, MaxDD, per-year signs,
gate_no_fake_profit.

ADD-ONLY. Never mutates the V2 path.
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


def _stats(col_vals, entry_ts, exit_ts):
    p = np.asarray(col_vals, float)
    m = np.isfinite(p)
    p = p[m]
    e = entry_ts[m]
    x = exit_ts[m]
    total = float(np.nansum(p))
    maxdd = _max_drawdown(p)
    nonov, nkept = _nonoverlap_sharpe(p, e, x)
    nw = _newey_west_sharpe(p, e, x)
    naive = _sharpe_iid(p)
    yr = e.year
    per_year = {int(y): float(np.nansum(p[yr == y])) for y in sorted(set(yr))}
    return dict(n=len(p), total=total, maxdd=maxdd, nonov=nonov, nkept=nkept,
                nw=nw, naive=naive, per_year=per_year)


def run_band(band):
    cfg = RealisticEvalConfig(
        hold="15m",
        real_bs_delta=True,
        fill_tol_minutes=60.0,
        spread_source="tape",
        delta_hedge_band=band,
    )
    res = evaluate_realistic(tl, cfg)
    f = res.trade_log
    # LONG subset: signal>0 AND covered
    long_mask = (f["signal"].to_numpy(float) > 0) & (f["premium_tier"].to_numpy() > 0)
    fl = f[long_mask].copy().sort_index()
    entry_ts = pd.DatetimeIndex(pd.to_datetime(fl.index, utc=True))
    # 15m hold: exit = exit_target
    if "exit_target" in fl.columns:
        exit_ts = pd.DatetimeIndex(pd.to_datetime(fl["exit_target"].values, utc=True))
    else:
        exit_ts = entry_ts + pd.Timedelta(minutes=15)

    hed = _stats(fl["total_hedged_usd"].to_numpy(float), entry_ts, exit_ts)
    unh = _stats(fl["total_unhedged_usd"].to_numpy(float), entry_ts, exit_ts)

    # gate on hedged long subset
    gate = None
    gmsg = ""
    if "real_option_pnl_nc" in fl.columns:
        try:
            gr = gate_no_fake_profit(fl, model_col="total_hedged_usd",
                                     real_col="real_option_pnl_nc")
            gate = bool(gr.passed)
            gmsg = str(gr.detail)
        except Exception as ex:
            gmsg = f"gate error: {ex}"
    return hed, unh, gate, gmsg


tl = pd.read_parquet(TRADE_LOG)
print("=" * 78)
print("LONG STRADDLE (signal>0) — hold=15m — realistic engine band sweep")
print("=" * 78)

# Un-hedged is band-independent (no hedge leg) -> report once from first band.
unh_reported = False
for band in BANDS:
    hed, unh, gate, gmsg = run_band(band)
    if not unh_reported:
        print("\n### UN-HEDGED (directional, total_unhedged_usd) [band-independent]")
        print(f"  n_trades={unh['n']}  total=${unh['total']:,.0f}  MaxDD=${unh['maxdd']:,.0f}")
        print(f"  nonoverlap_sharpe={unh['nonov']:.3f} (n_kept={unh['nkept']})  NW={unh['nw']:.3f}  naive={unh['naive']:.3f}")
        print(f"  per_year: " + "  ".join(f"{y}:${v:,.0f}[{'+' if v>0 else '-'}]" for y, v in unh['per_year'].items()))
        unh_reported = True
    print(f"\n### HEDGED band={band} (gamma-scalp, total_hedged_usd)")
    print(f"  n_trades={hed['n']}  total=${hed['total']:,.0f}  MaxDD=${hed['maxdd']:,.0f}")
    print(f"  nonoverlap_sharpe={hed['nonov']:.3f} (n_kept={hed['nkept']})  NW={hed['nw']:.3f}  naive={hed['naive']:.3f}")
    print(f"  per_year: " + "  ".join(f"{y}:${v:,.0f}[{'+' if v>0 else '-'}]" for y, v in hed['per_year'].items()))
    print(f"  gate_no_fake_profit={gate}  detail={gmsg}")
print("=" * 78)
