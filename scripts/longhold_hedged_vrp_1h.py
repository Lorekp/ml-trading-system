"""hold=1h SHORT-straddle sweep on the full V2 fired set (delta-hedged + un-hedged).

ADD-ONLY: reads vol_arb/output/BTC_v2_trade_log.parquet, runs the realistic-eval
engine at hold=1h with real per-bar BS straddle delta, and reports the full
risk panel + gate_no_fake_profit on the OVERLAP-ADJUSTED Sharpe.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vol_arb.realistic_eval import RealisticEvalConfig, evaluate_realistic
from vol_arb.honest_pnl import gate_no_fake_profit

TL = "vol_arb/output/BTC_v2_trade_log.parquet"


def newey_west_sharpe(x, ts, hold_minutes):
    """Overlap-adjusted trade Sharpe. Two estimators:

    (1) NON-OVERLAPPING subsample: keep trades whose entry is >= prev kept
        entry + hold (no temporal overlap), IID Sharpe on that subset.
    (2) NEWEY-WEST: HAC mean/variance with lag L = number of bars in the hold
        window measured in *trade* spacing units. We approximate the
        autocorrelation horizon by the count of trades that fall inside one
        hold window of the median trade — i.e. how many trades overlap.

    Returns dict with both, plus naive IID for reference.
    """
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    ts = pd.DatetimeIndex(ts)
    n = len(x)
    mu = x.mean()
    sd = x.std(ddof=1)
    naive = mu / sd * np.sqrt(n) if sd > 0 else 0.0

    # (1) non-overlapping subsample by entry time
    order = np.argsort(ts.values)
    ts_s = ts.values[order]
    x_s = x[order]
    hold_ns = np.timedelta64(int(hold_minutes), "m")
    keep = []
    last = None
    for i in range(len(ts_s)):
        if last is None or ts_s[i] >= last + hold_ns:
            keep.append(i)
            last = ts_s[i]
    xk = x_s[keep]
    m_no = len(xk)
    sd_no = xk.std(ddof=1) if m_no > 1 else 0.0
    sharpe_nonoverlap = (xk.mean() / sd_no * np.sqrt(m_no)) if sd_no > 0 else 0.0

    # (2) Newey-West HAC on the time-ordered per-trade PnL.
    # Lag = max overlapping trades in one hold window.
    # Count, for each trade, how many later trades enter within hold window.
    t_i8 = ts_s.astype("datetime64[ns]").astype(np.int64)
    win_ns = int(hold_minutes) * 60 * 1_000_000_000
    hi = np.searchsorted(t_i8, t_i8 + win_ns, side="right")
    overlap = hi - np.arange(len(t_i8))
    L = int(np.median(overlap))
    L = max(L, 0)
    # HAC variance of the sum / mean
    e = x_s - x_s.mean()
    gamma0 = (e @ e) / n
    var_hac = gamma0
    for lag in range(1, L + 1):
        w = 1.0 - lag / (L + 1.0)
        cov = (e[lag:] @ e[:-lag]) / n
        var_hac += 2.0 * w * cov
    se_mean = np.sqrt(max(var_hac, 0.0) / n)
    sharpe_nw = (mu / se_mean) / np.sqrt(n) * np.sqrt(n) if se_mean > 0 else 0.0
    # The per-trade Sharpe scaled to n trades with HAC SE:
    # t-stat = mu / se_mean; report overlap-aware inference, not readiness.
    # Report the HAC t-stat / sqrt(n) as the per-trade Sharpe equivalent:
    sharpe_nw = mu / np.sqrt(max(var_hac, 1e-30)) * np.sqrt(n) / np.sqrt(max(L + 1, 1))
    return {
        "naive_iid": float(naive),
        "nonoverlap": float(sharpe_nonoverlap),
        "n_nonoverlap": int(m_no),
        "newey_west": float(sharpe_nw),
        "nw_lag": L,
    }


def max_drawdown(pnl_series_sorted):
    eq = np.cumsum(pnl_series_sorted)
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    return float(dd.min())


def main():
    tl = pd.read_parquet(TL)
    cfg = RealisticEvalConfig(hold="1h", real_bs_delta=True)
    res = evaluate_realistic(tl, cfg)
    f = res.trade_log
    covered = f["premium_tier"].to_numpy() > 0
    cov = f[covered].copy()
    cov = cov.sort_index()  # chronological for DD

    hedged = cov["total_hedged_usd"].to_numpy(float)
    unhedged = cov["total_unhedged_usd"].to_numpy(float)
    ts = pd.DatetimeIndex(cov.index)

    n_trades = int(np.isfinite(hedged).sum())
    hedged_total = float(np.nansum(hedged))
    unhedged_total = float(np.nansum(unhedged))

    # Fees: option fee + hedge fee + funding (funding can be credit).
    total_fees = float(res.option_fee_total_usd + res.hedge_fee_total_usd + res.funding_total_usd)

    # MaxDD on chronological hedged cumPnL.
    h_chrono = hedged[np.isfinite(hedged)]
    maxdd = max_drawdown(h_chrono)
    worst_trade = float(np.nanmin(hedged))

    # Sharpe (overlap-adjusted).
    sh = newey_west_sharpe(hedged, ts, hold_minutes=60)

    # Per-calendar-year sign of hedged NC PnL.
    yr = pd.Series(hedged, index=ts).groupby(ts.year).sum()
    per_year = {int(y): float(v) for y, v in yr.items()}
    per_year_signs = {y: ("+" if v > 0 else "-" if v < 0 else "0") for y, v in per_year.items()}

    # gate_no_fake_profit: model = hedged total, ground truth = real un-hedged fills.
    gate_df = f.copy()
    gate_df["net_pnl_nc"] = gate_df["total_hedged_usd"]
    # real_option_pnl_nc is set by mark_trade_log_to_real_premium inside the engine.
    gate = gate_no_fake_profit(gate_df, model_col="net_pnl_nc", real_col="real_option_pnl_nc")

    print("=" * 70)
    print("HOLD=1h SHORT STRADDLE — full fired set, delta-hedged + un-hedged")
    print("=" * 70)
    print(f"n_fired           = {res.n_fired}")
    print(f"n_covered (trades)= {n_trades}")
    print(f"coverage_frac     = {res.coverage_frac:.3%}")
    print(f"HEDGED   NC PnL   = ${hedged_total:,.0f}")
    print(f"UNHEDGED NC PnL   = ${unhedged_total:,.0f}")
    print(f"option_premium    = ${res.option_premium_total_usd:,.0f}")
    print(f"perp_leg          = ${res.perp_leg_total_usd:,.0f}")
    print(f"option_fee        = ${res.option_fee_total_usd:,.0f}")
    print(f"hedge_fee         = ${res.hedge_fee_total_usd:,.0f}")
    print(f"funding           = ${res.funding_total_usd:,.0f}")
    print(f"TOTAL FEES        = ${total_fees:,.0f}")
    print(f"MaxDD (chrono)    = ${maxdd:,.0f}")
    print(f"worst trade       = ${worst_trade:,.0f}")
    print("-" * 70)
    print(f"naive IID Sharpe  = {sh['naive_iid']:.3f}")
    print(f"non-overlap Sharpe= {sh['nonoverlap']:.3f}  (n={sh['n_nonoverlap']})")
    print(f"Newey-West Sharpe = {sh['newey_west']:.3f}  (lag={sh['nw_lag']})")
    print("-" * 70)
    print(f"per-year hedged PnL: {per_year}")
    print(f"per-year signs     : {per_year_signs}")
    print("-" * 70)
    print(f"gate_no_fake_profit passed = {gate.passed}")
    print(f"  {gate.message}")
    print("=" * 70)


if __name__ == "__main__":
    main()
