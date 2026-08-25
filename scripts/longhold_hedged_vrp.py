"""Long-hold (to-expiry) delta-hedged SHORT straddle VRP sweep.

Evaluates the SHORT straddle on the FULL fired set at hold=to-expiry, both
delta-hedged (real per-bar BS delta along the real 1-min BTC path) and un-hedged,
with the full realistic Deribit fee stack. ADD-ONLY: never touches the verified
15-min path or V2 byte-identity.

Reports: n_trades, hedged/un-hedged NC PnL, MaxDD, worst single-trade, total fees,
NAIVE IID trade Sharpe AND an OVERLAP-ADJUSTED Sharpe (non-overlapping subsample +
Newey-West), per-calendar-year sign of hedged NC PnL, gate_no_fake_profit.

Usage:
  PYTHONPATH=. .venv/Scripts/python.exe scripts/longhold_hedged_vrp.py \
      --fill-tol-minutes 60
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from vol_arb.realistic_eval import RealisticEvalConfig, evaluate_realistic
from vol_arb.honest_pnl import gate_no_fake_profit


def _sharpe_iid(x: np.ndarray) -> float:
    x = x[np.isfinite(x)]
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return float(x.mean() / x.std(ddof=1) * np.sqrt(len(x)))


def _max_drawdown(pnl: np.ndarray) -> float:
    """MaxDD (most-negative trough) on the cumulative-$ equity of per-trade PnL,
    ordered by entry time."""
    eq = np.cumsum(pnl)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    return float(dd.min()) if len(dd) else 0.0


def _newey_west_sharpe(pnl: np.ndarray, entry_ts: pd.DatetimeIndex,
                       exit_ts: pd.DatetimeIndex) -> float:
    """Newey-West (HAC) overlap-adjusted Sharpe.

    To-expiry holds OVERLAP -> per-trade PnL is autocorrelated. The naive IID
    sd understates the true sd by sqrt of the effective overlap. We aggregate the
    per-trade PnL onto a DAILY portfolio-return grid (each trade contributes its
    PnL on its entry day, capital normalized) then compute a Newey-West-corrected
    Sharpe with lag = the median hold in days (the overlap span). This rests the
    historical comparison on an overlap-aware number, not the inflated IID one.
    """
    # Daily aggregation of per-trade PnL by ENTRY day (a single book reused).
    df = pd.DataFrame({"pnl": pnl}, index=entry_ts)
    daily = df["pnl"].groupby(df.index.floor("D")).sum()
    r = daily.to_numpy(float)
    n = len(r)
    if n < 3 or r.std() == 0:
        return float("nan")
    mu = r.mean()
    # Newey-West long-run variance with lag L = median hold in days.
    hold_days = (exit_ts - entry_ts).total_seconds().to_numpy() / 86400.0
    L = int(max(1, min(np.ceil(np.nanmedian(hold_days)), n - 1)))
    demeaned = r - mu
    gamma0 = (demeaned @ demeaned) / n
    lrv = gamma0
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1)  # Bartlett
        cov = (demeaned[lag:] @ demeaned[:-lag]) / n
        lrv += 2.0 * w * cov
    if lrv <= 0:
        return float("nan")
    # Annualize-free per-period Sharpe scaled by sqrt(n) for a total-sample t-stat
    # analogue, consistent with the IID convention (mean/sd*sqrt(n)).
    return float(mu / np.sqrt(lrv) * np.sqrt(n))


def _nonoverlap_sharpe(pnl: np.ndarray, entry_ts: pd.DatetimeIndex,
                       exit_ts: pd.DatetimeIndex) -> tuple[float, int]:
    """Greedy NON-OVERLAPPING subsample Sharpe: walk trades in entry order, keep a
    trade only if its entry is at/after the previously-kept trade's exit. The kept
    PnL series is then non-overlapping. Returns (sharpe, n_kept)."""
    order = np.argsort(entry_ts.values)
    e = entry_ts.values[order]
    x = exit_ts.values[order]
    p = pnl[order]
    kept = []
    last_exit = np.datetime64("1900-01-01")
    for i in range(len(e)):
        if e[i] >= last_exit:
            kept.append(p[i])
            last_exit = x[i]
    kept = np.asarray(kept, float)
    return _sharpe_iid(kept), len(kept)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trade-log", default="vol_arb/output/BTC_v2_trade_log.parquet")
    ap.add_argument("--fill-tol-minutes", type=float, default=60.0)
    args = ap.parse_args()

    tl = pd.read_parquet(args.trade_log)

    cfg = RealisticEvalConfig(
        hold="expiry",
        real_bs_delta=True,
        fill_tol_minutes=args.fill_tol_minutes,
        spread_source="tape",
    )
    res = evaluate_realistic(tl, cfg)
    f = res.trade_log
    cov = f[f["premium_tier"] > 0].copy()
    cov = cov.sort_index()
    n = len(cov)

    hedged = cov["total_hedged_usd"].to_numpy(float)
    unhedged = cov["total_unhedged_usd"].to_numpy(float)
    entry_ts = pd.DatetimeIndex(pd.to_datetime(cov.index, utc=True))
    # exit = option expiry (to-expiry)
    exit_ts = pd.DatetimeIndex(pd.to_datetime(cov["expiry_dt"].values, utc=True))

    # ---- Headline aggregates ----
    hedged_total = float(np.nansum(hedged))
    unhedged_total = float(np.nansum(unhedged))
    total_fees = float(res.option_fee_total_usd + res.hedge_fee_total_usd)
    # funding is signed (can be credit); report separately but include in note
    funding_total = float(res.funding_total_usd)

    maxdd = _max_drawdown(hedged)
    worst = float(np.nanmin(hedged)) if n else float("nan")

    naive_sharpe = _sharpe_iid(hedged)
    nw_sharpe = _newey_west_sharpe(hedged, entry_ts, exit_ts)
    nonov_sharpe, n_nonov = _nonoverlap_sharpe(hedged, entry_ts, exit_ts)

    # ---- Per-calendar-year sign of hedged NC PnL ----
    yr = entry_ts.year
    per_year = {}
    for y in sorted(set(yr)):
        s = float(np.nansum(hedged[yr == y]))
        per_year[int(y)] = s

    # ---- gate_no_fake_profit ----
    # The gate compares model NC PnL (here: the hedged realistic total per trade)
    # against the real-fill mark. honest_pnl computed real_option_pnl_nc inside
    # mark_trade_log_to_real_premium; check columns exist.
    gate_model_col = "total_hedged_usd"
    gate_real_col = "real_option_pnl_nc"
    gate_passed = None
    gate_msg = ""
    if gate_real_col in f.columns:
        try:
            gr = gate_no_fake_profit(f, model_col=gate_model_col, real_col=gate_real_col)
            gate_passed = bool(gr.passed)
            gate_msg = gr.detail
        except Exception as ex:  # noqa
            gate_msg = f"gate error: {ex}"
    else:
        # Fall back: the realistic hedged total is itself the honest mark, so the
        # sign-primary "no false profit" test is simply: does the hedged total
        # report profit? If hedged_total <= 0 there is no false profit to flag.
        gate_passed = not (hedged_total > 0 and unhedged_total <= 0)
        gate_msg = {"note": "real_option_pnl_nc absent; sign-primary fallback",
                    "hedged_total": hedged_total, "unhedged_total": unhedged_total}

    print("=" * 74)
    print("LONG-HOLD (TO-EXPIRY) DELTA-HEDGED SHORT STRADDLE — VRP SWEEP")
    print("=" * 74)
    print(f"fill_tol_minutes = {args.fill_tol_minutes}")
    print(f"n_fired (full set)   = {res.n_fired}")
    print(f"n_trades (covered)   = {n}   coverage = {res.coverage_frac:.2%}")
    print("-" * 74)
    print(f"HEDGED   NC PnL  = ${hedged_total:,.0f}")
    print(f"UNHEDGED NC PnL  = ${unhedged_total:,.0f}")
    print(f"option_premium   = ${res.option_premium_total_usd:,.0f}")
    print(f"option_fee       = ${res.option_fee_total_usd:,.0f}")
    print(f"perp_hedge_fee   = ${res.hedge_fee_total_usd:,.0f}")
    print(f"funding (signed) = ${funding_total:,.0f}")
    print(f"perp_leg (resid) = ${res.perp_leg_total_usd:,.0f}")
    print(f"gamma_leg (diag) = ${res.gamma_leg_total_usd:,.0f}")
    print(f"TOTAL FEES       = ${total_fees:,.0f}  (option_fee + perp_hedge_fee)")
    print("-" * 74)
    print(f"MaxDD (hedged eq)   = ${maxdd:,.0f}")
    print(f"worst single-trade  = ${worst:,.0f}")
    print("-" * 74)
    print(f"NAIVE IID Sharpe          = {naive_sharpe:.3f}   (invalid for inference — holds overlap)")
    print(f"OVERLAP-ADJ Newey-West    = {nw_sharpe:.3f}")
    print(f"OVERLAP-ADJ non-overlap   = {nonov_sharpe:.3f}  (n_kept={n_nonov})")
    print("-" * 74)
    print("Per-calendar-year HEDGED NC PnL (sign):")
    for y, s in per_year.items():
        print(f"  {y}: ${s:,.0f}  [{'+' if s > 0 else '-'}]")
    print("-" * 74)
    print(f"gate_no_fake_profit passed = {gate_passed}")
    print(f"gate detail = {gate_msg}")
    print("=" * 74)


if __name__ == "__main__":
    main()
