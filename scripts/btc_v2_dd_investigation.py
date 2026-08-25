"""Historical drawdown hypothesis tests for invalidated BTC V2 research.

Loads the saved V2 trade log and executes five exploratory hypothesis tests.
Because that trade log was invalidated, the output is forensic only.

Decision rule per hypothesis (bucket-based, NOT DD-period-filtered so
this is not curve-fitting):
  SUPPORTED if the candidate bucket's NC Sharpe is >= 1.0 below the
  "safe" bucket AND the expectancy ratio < 0.5 (candidate vs best).

Outputs a plain-text report to stdout. The caller pipes it into
``vol_arb/output/BTC_dd_investigation.md`` or reads it manually.

This is an investigation artifact - not part of the shipped pipeline.
Run once, interpret, then move the conclusion into
``BTC_dd_investigation.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.metrics import (
    build_nc_view,
    drawdown_metrics,
    top_drawdown_episodes,
)

TRADE_LOG = Path("vol_arb/output/BTC_v2_trade_log.parquet")
INITIAL_EQUITY = 100_000.0


def _bucket_stats(df: pd.DataFrame, bucket_col: str, pnl_col: str = "net_pnl_nc") -> pd.DataFrame:
    """Per-bucket NC Sharpe + expectancy + count.

    NC trade-Sharpe = mean(pnl) / std(pnl) * sqrt(N / years) so bucket
    sizes don't distort the comparison (scale-invariant).
    """
    out_rows = []
    for b, grp in df.groupby(bucket_col, observed=True):
        r = grp[pnl_col].values
        if len(r) < 2:
            continue
        ts = grp.index
        years = max((ts.max() - ts.min()).days / 365.25, 1e-9)
        trade_sharpe = (
            r.mean() / r.std(ddof=1) * np.sqrt(len(r) / years)
            if r.std(ddof=1) > 0 else 0.0
        )
        out_rows.append({
            "bucket": str(b),
            "n": len(r),
            "expectancy": r.mean(),
            "total_pnl": r.sum(),
            "win_rate_pct": (r > 0).mean() * 100,
            "trade_sharpe": trade_sharpe,
        })
    return pd.DataFrame(out_rows)


def _decide(df: pd.DataFrame, candidate_bucket: str, expectancy_col: str = "expectancy") -> str:
    """SUPPORTED / REJECTED on the pre-registered decision rule.

    Pre-registered rule:
      - Sharpe gap >= 1.0 between candidate and best bucket
      - AND candidate expectancy < 0.5 x best expectancy
    """
    if candidate_bucket not in df["bucket"].values:
        return f"INCONCLUSIVE (candidate bucket '{candidate_bucket}' not present)"
    cand = df.loc[df["bucket"] == candidate_bucket].iloc[0]
    best = df.loc[df["trade_sharpe"].idxmax()]
    sharpe_gap = best["trade_sharpe"] - cand["trade_sharpe"]
    exp_ratio = cand[expectancy_col] / best[expectancy_col] if best[expectancy_col] != 0 else 0.0
    supported = (sharpe_gap >= 1.0) and (exp_ratio < 0.5)
    return (
        f"{'SUPPORTED' if supported else 'REJECTED'}  "
        f"(gap={sharpe_gap:.2f}, exp_ratio={exp_ratio:.2f})"
    )


def main() -> None:
    print("=" * 76)
    print("BTC v2 DRAWDOWN INVESTIGATION")
    print("=" * 76)

    tl = pd.read_parquet(TRADE_LOG)
    print(f"\nLoaded {len(tl):,} bars from {TRADE_LOG}")
    active = tl[tl["signal"] != 0].copy()
    print(f"Active trades: {len(active):,}")

    # -- Baseline stats --
    nc_view = build_nc_view(tl)
    nc_dd = drawdown_metrics(nc_view, INITIAL_EQUITY, is_nc=True)
    print(f"\n[ BASELINE NC DRAWDOWN ]")
    print(f"  Max DD: {nc_dd['max_drawdown_pct']:.2f}%")
    print(f"  # DD events: {nc_dd['n_drawdown_events']}")
    print(f"  % time in DD: {nc_dd['pct_time_in_drawdown']:.1f}%")

    # -- Top drawdown episodes (info only, not used to filter tests) --
    print("\n[ TOP 5 DRAWDOWN EPISODES (NC) ]")
    eps = top_drawdown_episodes(nc_view, INITIAL_EQUITY, is_nc=True, k=5)
    for i, ep in enumerate(eps, 1):
        rec = ep["recovery_ts"].strftime("%Y-%m-%d") if ep["recovery_ts"] else "OPEN"
        print(
            f"  #{i}: {ep['start_ts']:%Y-%m-%d} -> trough {ep['trough_ts']:%Y-%m-%d} "
            f"-> recovery {rec}   depth={ep['depth_pct']:6.2f}%  "
            f"duration={ep['duration_days']}d  trades={ep['trades_count']:,}"
        )

    # ===================================================================
    # H1 - Shorts with jump_fraction in [0.10, 0.15) are the tail
    # ===================================================================
    print("\n" + "-" * 76)
    print("H1: SHORTS NEAR jump_fraction CIRCUIT-BREAKER BOUNDARY")
    print("-" * 76)
    print("  Hypothesis: shorts in [0.10, 0.15) fire just below the binary")
    print("  kill-switch (0.15) and have no MTM stop. Test: bucket shorts by")
    print("  jump_fraction; compare NC trade-Sharpe + expectancy across buckets.")
    shorts = active[active["signal"] == -1].copy()
    shorts["jf_bucket"] = pd.cut(
        shorts["jump_fraction"],
        bins=[-0.01, 0.05, 0.10, 0.15, 1.0],
        labels=["[0,0.05)", "[0.05,0.10)", "[0.10,0.15)", "[0.15+)"],
    )
    h1 = _bucket_stats(shorts, "jf_bucket")
    print("\n", h1.to_string(index=False))
    print(f"\n  VERDICT: H1 {_decide(h1, '[0.10,0.15)')}")

    # ===================================================================
    # H2 - Concurrent open positions amplify stress losses
    # ===================================================================
    print("\n" + "-" * 76)
    print("H2: CONCURRENT OPEN POSITIONS")
    print("-" * 76)
    print("  Hypothesis: 15m signal cadence x 30m hold => up to 2 simultaneous")
    print("  open positions; sustained regimes => correlated bleed.")
    # Count open positions: for each active-trade timestamp, how many earlier
    # trades are still open (entry_ts + 30min > current entry_ts)?
    ts = active.index.sort_values()
    # For each trade, number of prior trades with entry ts within [t-30min, t)
    # Use a rolling count on a constant-1 series indexed by trade ts.
    ones = pd.Series(1, index=ts)
    # rolling('30min') is right-aligned: at ts, counts trades in (ts-30min, ts].
    # We want (ts-30min, ts) i.e. exclude self; subtract 1.
    concurrent = ones.rolling("30min").sum().astype(int) - 1
    active_sorted = active.sort_index().copy()
    active_sorted["concurrent"] = concurrent.values
    active_sorted["concurrent_bucket"] = pd.cut(
        active_sorted["concurrent"],
        bins=[-1, 0, 1, 2, 100],
        labels=["0 (alone)", "1", "2", "3+"],
    )
    h2 = _bucket_stats(active_sorted, "concurrent_bucket")
    print("\n", h2.to_string(index=False))
    print(f"\n  VERDICT: H2 {_decide(h2, '2')}")
    # Secondary check: does the 3+ bucket exist meaningfully?
    if "3+" in h2["bucket"].values:
        print(f"  (3+ bucket exists: N={h2[h2.bucket == '3+'].iloc[0]['n']:,})")

    # ===================================================================
    # H3 - Low-DTE (5-10 day) trades: unhedgeable gamma
    # ===================================================================
    print("\n" + "-" * 76)
    print("H3: LOW-DTE TRADES (GAMMA ~ 1/sqrt(T))")
    print("-" * 76)
    print("  Hypothesis: 5-10 DTE trades have ~3x gamma of 45 DTE; DRL hedges")
    print("  10x/bar but not tick - fast moves leave unhedgeable delta.")
    active["dte_bucket"] = pd.cut(
        active["dte_days"],
        bins=[4.99, 10, 20, 45, 365],
        labels=["[5,10)", "[10,20)", "[20,45)", "[45+)"],
    )
    h3 = _bucket_stats(active, "dte_bucket")
    print("\n", h3.to_string(index=False))
    # Additional: is delta_hedge_pnl_nc the DOMINANT neg component in [5,10)?
    low_dte = active[(active["dte_days"] >= 5) & (active["dte_days"] < 10)]
    if len(low_dte) > 0:
        components = {
            "vega_pnl_nc": low_dte["vega_pnl_nc"].sum(),
            "theta_pnl_nc": low_dte["theta_pnl_nc"].sum(),
            "delta_hedge_pnl_nc": low_dte["delta_hedge_pnl_nc"].sum(),
            "option_fees_nc": -low_dte["option_fees_nc"].sum(),
            "slippage_nc": -low_dte["slippage_nc"].sum(),
            "hedge_costs_nc": -low_dte["hedge_costs_nc"].sum(),
        }
        print(f"\n  PnL decomposition for DTE in [5, 10)  (N={len(low_dte):,}):")
        for k, v in sorted(components.items(), key=lambda kv: kv[1]):
            print(f"    {k:<24s}  ${v:>14,.0f}")
    print(f"\n  VERDICT: H3 {_decide(h3, '[5,10)')}")

    # ===================================================================
    # H4 - Regime concentration (one year drives most DD)
    # ===================================================================
    print("\n" + "-" * 76)
    print("H4: REGIME CONCENTRATION (year-level)")
    print("-" * 76)
    print("  Hypothesis: one calendar year dominates the DD story.")
    # Tally DD-episode start years
    dd_years = pd.Series([ep["start_ts"].year for ep in top_drawdown_episodes(
        nc_view, INITIAL_EQUITY, is_nc=True, k=10_000
    )])
    dd_depths = pd.Series([ep["depth_pct"] for ep in top_drawdown_episodes(
        nc_view, INITIAL_EQUITY, is_nc=True, k=10_000
    )])
    dd_by_year = pd.DataFrame({"year": dd_years, "depth": dd_depths})
    # Worst DD per year
    worst_by_year = dd_by_year.groupby("year")["depth"].min().sort_values()
    print("\n  Worst DD episode per year (deeper = more negative):")
    for y, d in worst_by_year.items():
        print(f"    {y}:  {d:6.2f}%")
    # Per-year NC PnL
    active["year"] = active.index.year
    year_pnl = active.groupby("year")["net_pnl_nc"].agg(["sum", "count", "mean"])
    print("\n  Per-year NC PnL:")
    for y, row in year_pnl.iterrows():
        print(f"    {y}:  N={row['count']:>5,.0f}  total=${row['sum']:>10,.0f}  "
              f"avg=${row['mean']:>7.2f}")
    neg_years = (year_pnl["sum"] < 0).sum()
    worst_year = year_pnl["sum"].idxmin()
    worst_year_pct = year_pnl.loc[worst_year, "sum"] / year_pnl["sum"].sum() * 100
    print(f"\n  # negative years: {neg_years} / {len(year_pnl)}")
    print(f"  Worst year: {worst_year}  ({worst_year_pct:.1f}% of 5y total)")
    h4_supported = (neg_years == 0) and (len(year_pnl) >= 4)
    # H4 is diagnostic - never fixed via per-year filter. We just note.
    print(f"\n  VERDICT: H4 "
          f"{'REJECTED (all years profitable - no regime-tail story)' if h4_supported else 'SUPPORTED (one or more neg years)'}")

    # ===================================================================
    # H5 - Costs dominate during thin-edge periods
    # ===================================================================
    print("\n" + "-" * 76)
    print("H5: COST DRAG ON THIN-EDGE TRADES")
    print("-" * 76)
    print("  Hypothesis: thinnest-edge trades lose because costs dominate the")
    print("  gross pnl. Test: bucket by edge quintile, check cost/gross ratio.")
    active["edge_q"] = pd.qcut(active["edge"], 5, labels=["Q1 (thin)", "Q2", "Q3", "Q4", "Q5 (fat)"])
    cost_rows = []
    for q, grp in active.groupby("edge_q", observed=True):
        gross = grp["gross_pnl_nc"].abs().sum()
        costs = (grp["option_fees_nc"] + grp["slippage_nc"] + grp["hedge_costs_nc"]).sum()
        net = grp["net_pnl_nc"].sum()
        cost_rows.append({
            "edge_bucket": str(q),
            "n": len(grp),
            "avg_edge": grp["edge"].mean(),
            "gross_abs": gross,
            "total_costs": costs,
            "cost_ratio": costs / gross if gross > 0 else 0.0,
            "net_pnl": net,
            "net_per_trade": net / len(grp) if len(grp) else 0.0,
        })
    h5 = pd.DataFrame(cost_rows)
    print("\n", h5.to_string(index=False))
    # SUPPORTED if thinnest-edge cost_ratio > 0.5 AND net_per_trade < 0.5x best bucket
    thin = h5.iloc[0]
    best_net = h5["net_per_trade"].max()
    cost_ratio_exceeded = thin["cost_ratio"] > 0.5
    net_ratio = thin["net_per_trade"] / best_net if best_net > 0 else 0.0
    supported = cost_ratio_exceeded and (net_ratio < 0.5)
    print(f"\n  Q1 cost_ratio: {thin['cost_ratio']:.2f}  (threshold 0.5)")
    print(f"  Q1 net_per_trade: ${thin['net_per_trade']:.2f}  vs best ${best_net:.2f}")
    print(f"\n  VERDICT: H5 {'SUPPORTED' if supported else 'REJECTED'}")

    # -- Summary --
    print("\n" + "=" * 76)
    print("HYPOTHESIS SUMMARY")
    print("=" * 76)


if __name__ == "__main__":
    main()
