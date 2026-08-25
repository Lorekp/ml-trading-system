"""hold=1d SHORT-straddle realistic eval — full fired set, hedged + un-hedged.

Add-only driver. Uses the verified extended realistic_eval engine (real_bs_delta
path, chunked memory-safe driver, full Deribit fee stack). Reports the per-trade
hedged/un-hedged NC PnL, MaxDD, worst single trade, total fees, NAIVE IID Sharpe,
an OVERLAP-ADJUSTED Sharpe (non-overlapping subsample + Newey-West), per-calendar-
year sign of hedged NC PnL, and gate_no_fake_profit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vol_arb.realistic_eval import RealisticEvalConfig, evaluate_realistic
from vol_arb.honest_pnl import gate_no_fake_profit

TRADE_LOG = "vol_arb/output/BTC_v2_trade_log.parquet"


def max_drawdown(pnl_series: np.ndarray) -> float:
    """Worst peak-to-trough of the cumulative PnL (USD, negative number)."""
    eq = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    return float(dd.min())


def newey_west_sharpe(x: np.ndarray, L: int) -> float:
    """Per-trade mean/se Sharpe with a Newey-West HAC variance (lag L) to deflate
    the overlap autocorrelation. Returns the HAC-adjusted t-stat-style Sharpe
    (mean / HAC-se) scaled to per-trade (NOT annualized)."""
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 5:
        return float("nan")
    mu = x.mean()
    xc = x - mu
    gamma0 = (xc @ xc) / n
    var = gamma0
    for l in range(1, min(L, n - 1) + 1):
        w = 1.0 - l / (L + 1.0)
        cov = (xc[l:] @ xc[:-l]) / n
        var += 2.0 * w * cov
    se = np.sqrt(max(var, 0.0) / n)
    if se <= 0:
        return float("nan")
    return float(mu / (se * np.sqrt(n)))  # mean / sd_hac == per-trade Sharpe


def main():
    tl = pd.read_parquet(TRADE_LOG)

    cfg = RealisticEvalConfig(hold="1d", real_bs_delta=True)
    res = evaluate_realistic(tl, cfg)

    f = res.trade_log
    covered = f["premium_tier"].to_numpy() > 0
    cov = f[covered].copy()

    hedged = cov["total_hedged_usd"].to_numpy(float)
    unhedged = cov["total_unhedged_usd"].to_numpy(float)
    ok = np.isfinite(hedged)
    cov = cov[ok].copy()
    hedged = hedged[ok]
    unhedged = unhedged[ok]
    entry_ts = pd.to_datetime(cov.index, utc=True)

    n_trades = len(hedged)
    hedged_total = float(hedged.sum())
    unhedged_total = float(unhedged.sum())

    # Fees
    opt_fee = float(np.nansum(cov["option_fee_usd"].to_numpy(float)))
    hedge_fee = float(np.nansum(cov["hedge_fee_usd"].to_numpy(float)))
    funding = float(np.nansum(cov["funding_usd"].to_numpy(float)))
    total_fees = opt_fee + hedge_fee  # funding is signed carry, report separately

    # Order by entry time for drawdown / overlap analysis
    order = np.argsort(entry_ts.values)
    hedged_o = hedged[order]
    ts_o = entry_ts.values[order]
    maxdd = max_drawdown(hedged_o)
    worst = float(hedged.min())

    # Naive IID per-trade Sharpe
    naive_sharpe = float(hedged.mean() / hedged.std(ddof=1)) if hedged.std(ddof=1) > 0 else float("nan")

    # ── Overlap-adjusted Sharpe ──
    # 1d holds overlap heavily (entries every ~bar). Two estimators:
    # (a) NON-OVERLAPPING subsample: greedily pick trades whose [entry,exit] do
    #     not overlap a previously-kept trade (exit = entry + 1 day).
    exit_o = ts_o + np.timedelta64(1, "D")
    keep = []
    last_exit = np.datetime64("1900-01-01")
    for i in range(len(ts_o)):
        if ts_o[i] >= last_exit:
            keep.append(i)
            last_exit = exit_o[i]
    keep = np.array(keep, dtype=int)
    nonoverlap = hedged_o[keep]
    if len(nonoverlap) >= 5 and nonoverlap.std(ddof=1) > 0:
        nonoverlap_sharpe = float(nonoverlap.mean() / nonoverlap.std(ddof=1))
    else:
        nonoverlap_sharpe = float("nan")

    # (b) Newey-West HAC with lag = typical # overlapping trades. Trades fire on a
    #     ~hourly cadence; a 1d hold overlaps ~ (1 day / median gap) trades.
    gaps = np.diff(ts_o).astype("timedelta64[s]").astype(float)
    med_gap_s = float(np.median(gaps[gaps > 0])) if np.any(gaps > 0) else 3600.0
    L = int(np.clip(round(86400.0 / max(med_gap_s, 1.0)), 1, len(hedged_o) - 2))
    nw_sharpe = newey_west_sharpe(hedged_o, L)

    overlap_adj = nonoverlap_sharpe  # rest the verdict on the non-overlapping one

    # Per-calendar-year sign of hedged NC PnL
    yr = pd.Series(hedged, index=entry_ts).groupby(entry_ts.year).sum()
    per_year = {int(y): float(v) for y, v in yr.items()}
    per_year_signs = {int(y): ("+" if v > 0 else "-" if v < 0 else "0") for y, v in yr.items()}

    # gate_no_fake_profit: model_col = the hedged realistic NC PnL; real_col = the
    # honest un-hedged real-fill premium leg (option_premium_usd, the ground truth
    # money leg with no synthetic gamma). The gate's question: does the headline
    # claim profit that the real-fill money leg contradicts?
    g = f.copy()
    g["net_pnl_nc"] = g["total_hedged_usd"]
    g["real_option_pnl_nc"] = g["option_premium_usd"]
    gate = gate_no_fake_profit(g, model_col="net_pnl_nc", real_col="real_option_pnl_nc")

    # Also: gate on the un-hedged real money leg directly (strictest reading).
    print("=" * 70)
    print(f"HOLD=1d SHORT STRADDLE — full fired set, real_bs_delta hedge")
    print("=" * 70)
    print(f"n_fired           = {res.n_fired}")
    print(f"n_covered         = {res.n_covered}")
    print(f"n_trades (finite) = {n_trades}")
    print(f"coverage_frac     = {res.coverage_frac:.3%}")
    print("-" * 70)
    print(f"HEDGED NC PnL     = ${hedged_total:,.0f}")
    print(f"UN-HEDGED NC PnL  = ${unhedged_total:,.0f}")
    print(f"  option_premium  = ${res.option_premium_total_usd:,.0f}")
    print(f"  perp_leg        = ${res.perp_leg_total_usd:,.0f}")
    print(f"  gamma_diag(Ito) = ${res.gamma_leg_total_usd:,.0f}  (diagnostic only)")
    print("-" * 70)
    print(f"option_fee        = ${opt_fee:,.0f}")
    print(f"hedge_fee (perp)  = ${hedge_fee:,.0f}")
    print(f"funding (signed)  = ${funding:,.0f}")
    print(f"TOTAL FEES (opt+hedge) = ${total_fees:,.0f}")
    print("-" * 70)
    print(f"MaxDD (hedged, time-ordered) = ${maxdd:,.0f}")
    print(f"worst single trade (hedged)  = ${worst:,.0f}")
    print(f"hedged mean/trade            = ${hedged.mean():,.1f}")
    print("-" * 70)
    print(f"NAIVE IID per-trade Sharpe   = {naive_sharpe:.4f}")
    print(f"  n_nonoverlap subsample     = {len(keep)}")
    print(f"OVERLAP-ADJ Sharpe (non-ovl) = {nonoverlap_sharpe:.4f}")
    print(f"OVERLAP-ADJ Sharpe (NW L={L})  = {nw_sharpe:.4f}")
    print("-" * 70)
    print(f"per-year hedged NC PnL: {per_year}")
    print(f"per-year signs        : {per_year_signs}")
    print("-" * 70)
    print(f"GATE no_fake_profit  : passed={gate.passed}")
    print(f"  {gate.message}")
    print("=" * 70)

    return dict(
        n_trades=n_trades, hedged=hedged_total, unhedged=unhedged_total,
        opt_fee=opt_fee, hedge_fee=hedge_fee, funding=funding, total_fees=total_fees,
        maxdd=maxdd, worst=worst, naive=naive_sharpe, nonoverlap=nonoverlap_sharpe,
        nw=nw_sharpe, per_year=per_year_signs, gate=gate.passed, n_nonoverlap=len(keep),
    )


if __name__ == "__main__":
    main()
