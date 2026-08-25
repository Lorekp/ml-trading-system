"""LONG straddle sweep at hold=1h — un-hedged (directional) + delta-hedged
(gamma scalp) on the FULL fired set, realistic fee stack.

The unconditional LONG null = take EVERY fired bar and force signal=+1 (long
straddle). Evaluate un-hedged (total_unhedged_usd) and delta-hedged across a
band sweep (longs favour a TIGHTER band to scalp realized variance).

ADD-ONLY: never mutates the V2 byte-identity. Reuses realistic_eval +
longhold_hedged_vrp overlap-adjusted Sharpe / DD helpers verbatim.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vol_arb.realistic_eval import RealisticEvalConfig, evaluate_realistic
from vol_arb.honest_pnl import gate_no_fake_profit
from scripts.longhold_hedged_vrp import (
    _sharpe_iid, _newey_west_sharpe, _nonoverlap_sharpe, _max_drawdown,
)

TRADE_LOG = "vol_arb/output/verify_frame_band35.parquet"
HOLD = "1h"
BANDS = [0.05, 0.10, 0.20, 0.35]


def _build_long_fired(path: str) -> pd.DataFrame:
    """Full fired set with EVERY bar forced LONG (signal=+1) — the unconditional
    long null at this horizon. Greeks are magnitudes; signal drives direction."""
    vf = pd.read_parquet(path)
    fired = vf[vf["signal"] != 0].copy()
    # entry timestamp lives in the 'ts' column (RangeIndex on disk); the engine
    # uses fired.index as entry_ts -> restore the datetime index.
    fired = fired.set_index(pd.DatetimeIndex(pd.to_datetime(fired["ts"], utc=True)))
    fired.index.name = None
    fired["signal"] = 1.0  # force LONG on the whole fired book
    return fired


def _per_year_signs(pnl: np.ndarray, entry_ts: pd.DatetimeIndex) -> dict:
    yr = entry_ts.year
    out = {}
    for y in sorted(set(yr)):
        out[int(y)] = float(np.nansum(pnl[yr == y]))
    return out


def _eval_variant(tl: pd.DataFrame, band: float, hedged: bool):
    cfg = RealisticEvalConfig(
        hold=HOLD, real_bs_delta=True, fill_tol_minutes=60.0,
        spread_source="tape", delta_hedge_band=band,
    )
    res = evaluate_realistic(tl, cfg)
    f = res.trade_log
    cov = f[f["premium_tier"] > 0].sort_index()
    n = len(cov)
    col = "total_hedged_usd" if hedged else "total_unhedged_usd"
    pnl = cov[col].to_numpy(float)
    entry_ts = pd.DatetimeIndex(pd.to_datetime(cov.index, utc=True))
    raw_exit = entry_ts + pd.Timedelta(minutes=60)
    expiry = pd.DatetimeIndex(pd.to_datetime(cov["expiry_dt"].values, utc=True))
    # clip hold to expiry, keeping tz (engine does the same clip internally)
    exit_ts = raw_exit.where(raw_exit < expiry, expiry)
    exit_ts = pd.DatetimeIndex(exit_ts)

    total = float(np.nansum(pnl))
    maxdd = _max_drawdown(pnl)
    naive = _sharpe_iid(pnl)
    nw = _newey_west_sharpe(pnl, entry_ts, exit_ts)
    nonov, n_keep = _nonoverlap_sharpe(pnl, entry_ts, exit_ts)
    per_year = _per_year_signs(pnl, entry_ts)

    # gate
    gate_passed = None
    gate_msg = ""
    if "real_option_pnl_nc" in f.columns:
        try:
            gr = gate_no_fake_profit(f, model_col=col, real_col="real_option_pnl_nc")
            gate_passed = bool(gr.passed)
            gate_msg = gr.detail
        except Exception as ex:  # noqa
            gate_msg = f"gate error: {ex}"
    else:
        unh = float(np.nansum(cov["total_unhedged_usd"].to_numpy(float)))
        gate_passed = not (total > 0 and unh <= 0)
        gate_msg = "real_option_pnl_nc absent; sign-primary fallback"

    return dict(n=n, total=total, maxdd=maxdd, naive=naive, nw=nw,
                nonov=nonov, n_keep=n_keep, per_year=per_year,
                gate=gate_passed, gate_msg=gate_msg, res=res)


def main():
    tl = _build_long_fired(TRADE_LOG)
    print("=" * 78)
    print(f"LONG STRADDLE SWEEP  hold={HOLD}  source={TRADE_LOG}")
    print(f"n_fired forced-long = {len(tl)}")
    print("=" * 78)

    # ---- UN-HEDGED (directional) — band irrelevant; run once at band 0.35 ----
    uh = _eval_variant(tl, 0.35, hedged=False)
    print("\n--- UN-HEDGED (directional / convexity) ---")
    print(f"  n_trades(cov) = {uh['n']}  coverage = {uh['res'].coverage_frac:.2%}")
    print(f"  TOTAL UNHEDGED = ${uh['total']:,.0f}")
    print(f"  MaxDD = ${uh['maxdd']:,.0f}")
    print(f"  Sharpe: naive={uh['naive']:.3f}  NW={uh['nw']:.3f}  "
          f"nonov={uh['nonov']:.3f} (n_keep={uh['n_keep']})")
    print(f"  per-year: " + "  ".join(
        f"{y}:${v:,.0f}[{'+' if v > 0 else '-'}]" for y, v in uh['per_year'].items()))
    print(f"  gate_no_fake_profit = {uh['gate']}")

    # ---- DELTA-HEDGED gamma scalp — sweep band ----
    print("\n--- DELTA-HEDGED (gamma scalp) band sweep ---")
    rows = []
    for b in BANDS:
        h = _eval_variant(tl, b, hedged=True)
        rows.append((b, h))
        py = "  ".join(f"{y}:{'+' if v > 0 else '-'}" for y, v in h['per_year'].items())
        print(f"\n  band={b:.2f}: n={h['n']}  TOTAL=${h['total']:,.0f}  MaxDD=${h['maxdd']:,.0f}")
        print(f"    Sharpe: naive={h['naive']:.3f}  NW={h['nw']:.3f}  "
              f"nonov={h['nonov']:.3f} (n_keep={h['n_keep']})")
        print(f"    per-year-sign: {py}")
        print(f"    per-year-$: " + "  ".join(f"{y}:${v:,.0f}" for y, v in h['per_year'].items()))
        print(f"    gate = {h['gate']}")

    # ---- best variant ----
    print("\n" + "=" * 78)
    print("SUMMARY")
    print(f"  UN-HEDGED: ${uh['total']:,.0f}  nonov_sharpe={uh['nonov']:.3f}  gate={uh['gate']}")
    best_b, best_h = max(rows, key=lambda r: r[1]['total'])
    print(f"  BEST HEDGED band={best_b:.2f}: ${best_h['total']:,.0f}  "
          f"nonov_sharpe={best_h['nonov']:.3f}  gate={best_h['gate']}")
    # overall best variant
    cand = [("unhedged", None, uh['total'], uh['nonov'], uh['gate'])]
    for b, h in rows:
        cand.append((f"hedged", b, h['total'], h['nonov'], h['gate']))
    bv = max(cand, key=lambda c: c[2])
    print(f"  OVERALL BEST: variant={bv[0]} band={bv[1]} total=${bv[2]:,.0f} "
          f"nonov={bv[3]:.3f} gate={bv[4]}")
    any_pos = any(c[2] > 0 and c[4] for c in cand)
    print(f"  ANY long variant positive & gate-passing? {any_pos}")
    print("=" * 78)


if __name__ == "__main__":
    main()
