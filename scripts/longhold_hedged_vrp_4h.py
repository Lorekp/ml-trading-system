"""hold=4h SHORT-straddle sweep on the full fired set — delta-hedged & un-hedged.

ADD-ONLY: uses the extended realistic_eval engine (real_bs_delta path) as a
post-processor. Never touches backtester.py / honest_pnl.py / the 15-min path.

Reports: n_trades, hedged/un-hedged NC PnL, MaxDD, worst single-trade, total fees,
naive IID trade Sharpe, OVERLAP-ADJUSTED Sharpe (non-overlapping subsample +
Newey-West), per-calendar-year sign of hedged NC PnL, gate_no_fake_profit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vol_arb.realistic_eval import RealisticEvalConfig, evaluate_realistic
from vol_arb.honest_pnl import gate_no_fake_profit

TRADE_LOG = "vol_arb/output/BTC_v2_trade_log.parquet"
HOLD = "4h"
HOLD_MIN = 240


def maxdd(equity: np.ndarray) -> float:
    """Most-negative drawdown of a cumulative-PnL curve (USD)."""
    peak = np.maximum.accumulate(equity)
    return float((equity - peak).min())


def newey_west_sharpe(x: np.ndarray, L: int) -> float:
    """Per-trade Sharpe with a Newey-West (Bartlett) long-run-variance correction
    for the overlap of holding windows. L = max lag (= overlap span in #trades).
    Returns mean/sqrt(LRV) * sqrt(n) consistent with a per-trade Sharpe scale."""
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return float("nan")
    mu = x.mean()
    d = x - mu
    gamma0 = (d @ d) / n
    lrv = gamma0
    for k in range(1, min(L, n - 1) + 1):
        w = 1.0 - k / (L + 1.0)
        gk = (d[k:] @ d[:-k]) / n
        lrv += 2.0 * w * gk
    if lrv <= 0:
        return float("nan")
    # per-trade Sharpe = mu / sqrt(per-trade variance); NW corrects the variance
    return float(mu / np.sqrt(lrv))


def main():
    full = pd.read_parquet(TRADE_LOG)

    cfg = RealisticEvalConfig(hold=HOLD, real_bs_delta=True)
    res = evaluate_realistic(full, cfg)
    f = res.trade_log

    covered = f["premium_tier"].to_numpy() > 0
    cov = f[covered].copy()
    cov = cov.sort_index()
    n = len(cov)

    hedged = cov["total_hedged_usd"].to_numpy(float)
    unhedged = cov["total_unhedged_usd"].to_numpy(float)
    ok = np.isfinite(hedged) & np.isfinite(unhedged)
    cov = cov[ok]
    hedged = hedged[ok]
    unhedged = unhedged[ok]
    n = len(cov)

    hedged_tot = float(hedged.sum())
    unhedged_tot = float(unhedged.sum())

    # Fees breakdown (covered subset)
    opt_fee = float(np.nansum(np.where(covered, f["option_fee_usd"].to_numpy(float), np.nan)))
    hedge_fee = float(np.nansum(np.where(covered, f["hedge_fee_usd"].to_numpy(float), np.nan)))
    funding = float(np.nansum(np.where(covered, f["funding_usd"].to_numpy(float), np.nan)))
    total_fees = opt_fee + hedge_fee  # funding is signed carry, not a fee

    # MaxDD & worst trade on the time-ordered HEDGED curve
    equity = np.cumsum(hedged)
    mdd = maxdd(equity)
    worst = float(hedged.min())
    best = float(hedged.max())

    # ── Naive IID trade Sharpe (NOT deployable; overlap) ──
    mu, sd = hedged.mean(), hedged.std(ddof=1)
    naive_iid = float(mu / sd) if sd > 0 else float("nan")

    # ── Overlap-adjusted Sharpe ──
    # (1) Non-overlapping subsample: 4h hold; bars are 15-min apart so consecutive
    # fires can overlap. Pick a maximal set of trades whose [entry,entry+4h) do
    # NOT overlap (greedy by entry time).
    idx_ts = pd.to_datetime(cov.index, utc=True)
    order = np.argsort(idx_ts.values)
    hold_td = np.timedelta64(HOLD_MIN, "m")
    last_end = np.datetime64("1970-01-01T00:00:00")
    keep = np.zeros(n, dtype=bool)
    ent_sorted = idx_ts.values[order]
    for pos in range(n):
        t = ent_sorted[pos]
        if t >= last_end:
            keep[order[pos]] = True
            last_end = t + hold_td
    nonoverlap = hedged[keep]
    m2, s2 = nonoverlap.mean(), nonoverlap.std(ddof=1)
    nonoverlap_sharpe = float(m2 / s2) if s2 > 0 else float("nan")
    n_nonoverlap = int(keep.sum())

    # (2) Newey-West on the time-ordered series. Overlap span L = #trades whose
    # window can overlap a given trade. With 15-min bars and a 4h hold, up to
    # 16 consecutive fires overlap -> L=16.
    hedged_time = hedged[order]
    nw_sharpe = newey_west_sharpe(hedged_time, L=16)

    # ── Per-calendar-year sign of hedged NC PnL ──
    yr = pd.to_datetime(cov.index, utc=True).year
    per_year = {}
    for y in sorted(set(yr)):
        s = float(hedged[yr == y].sum())
        per_year[int(y)] = s

    # ── gate_no_fake_profit ──
    # The realistic HEADLINE is itself on real fills. Gate: does the headline
    # (hedged) claim profit while the same-trade real un-hedged premium loses?
    # honest_pnl.gate_no_fake_profit needs model_col vs real_option_pnl_nc.
    gate_df = f.copy()
    gate_df["realistic_hedged_nc"] = f["total_hedged_usd"]
    # real_option_pnl_nc is produced by mark_trade_log_to_real_premium inside the
    # engine; reconstruct it as the un-hedged real-fill option PnL we computed.
    gate_df["real_option_pnl_nc"] = f["total_unhedged_usd"]
    try:
        gate = gate_no_fake_profit(gate_df, model_col="realistic_hedged_nc",
                                   real_col="real_option_pnl_nc")
        gate_passed = bool(gate.passed)
        gate_detail = gate.detail
    except Exception as e:
        gate_passed = None
        gate_detail = {"error": str(e)}

    print("=" * 64)
    print(f"hold={HOLD} SHORT straddle — full fired set")
    print("=" * 64)
    print(f"n_fired          = {res.n_fired}")
    print(f"n_covered        = {n}")
    print(f"coverage_frac    = {n/res.n_fired:.1%}")
    print(f"hedged   NC PnL  = ${hedged_tot:,.0f}")
    print(f"unhedged NC PnL  = ${unhedged_tot:,.0f}")
    print(f"option_fee       = ${opt_fee:,.0f}")
    print(f"hedge_fee        = ${hedge_fee:,.0f}")
    print(f"funding (signed) = ${funding:,.0f}")
    print(f"TOTAL FEES       = ${total_fees:,.0f}  (opt_fee+hedge_fee)")
    print(f"MaxDD (hedged)   = ${mdd:,.0f}")
    print(f"worst trade      = ${worst:,.0f}")
    print(f"best  trade      = ${best:,.0f}")
    print(f"mean/trade       = ${mu:,.2f}")
    print("-" * 64)
    print(f"naive IID Sharpe = {naive_iid:.4f}   (NOT deployable; overlap)")
    print(f"non-overlap Sharpe = {nonoverlap_sharpe:.4f}  (n={n_nonoverlap})")
    print(f"Newey-West Sharpe (L=16) = {nw_sharpe:.4f}")
    print("-" * 64)
    print("per-year hedged NC PnL:")
    for y, s in per_year.items():
        print(f"  {y}: ${s:,.0f}  ({'+' if s > 0 else '-'})")
    print("-" * 64)
    print(f"gate_no_fake_profit passed = {gate_passed}")
    print(f"gate detail = {gate_detail}")
    print("=" * 64)

    # machine-readable dump
    import json
    out = {
        "hold": HOLD, "n_trades": n, "hedged_nc_usd": hedged_tot,
        "unhedged_nc_usd": unhedged_tot, "option_fee": opt_fee,
        "hedge_fee": hedge_fee, "funding": funding, "total_fees": total_fees,
        "maxdd_usd": mdd, "worst_trade_usd": worst, "best_trade_usd": best,
        "naive_iid_sharpe": naive_iid, "nonoverlap_sharpe": nonoverlap_sharpe,
        "n_nonoverlap": n_nonoverlap, "newey_west_sharpe": nw_sharpe,
        "per_year": per_year, "gate_passed": gate_passed,
    }
    print("JSON " + json.dumps(out, default=float))


if __name__ == "__main__":
    main()
