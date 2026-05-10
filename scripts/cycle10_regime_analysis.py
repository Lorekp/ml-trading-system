"""Cycle-10 V2 regime stratification + concentration analysis (2026-05-10).

Investigates why cycle-10 V2 backtest underperforms cycle-9 by -$2,150 NC PnL
despite cycle-10's predictions being -22% better in QLIKE.

Read-only analysis; writes a markdown report.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

PRE = REPO_ROOT / "backups/2026-05-09_pre_transvol_optuna_retrain/BTC_v2_trade_log_pre_retrain.parquet"
POST = REPO_ROOT / "backups/2026-05-09_pre_transvol_optuna_retrain/BTC_v2_trade_log_post_retrain.parquet"

print(f"Loading PRE  {PRE}")
pre = pd.read_parquet(PRE)
print(f"Loading POST {POST}")
post = pd.read_parquet(POST)


def add_month(df):
    df = df.copy()
    df["month"] = df.index.to_period("M").astype(str)
    return df


pre = add_month(pre)
post = add_month(post)

# ---- Per-month aggregates -----------------------------------------------
def per_month(df):
    nc = df["net_pnl_nc"].astype(float)
    fired = nc != 0
    g = df.assign(_fired=fired, _nc=nc, _abs_edge=df["edge"].abs(),
                  _rv=df["rv_pct"], _is_long=df["signal"] == 1,
                  _is_short=df["signal"] == -1)
    rows = []
    for m, sub in g.groupby("month"):
        n_trades = int(sub["_fired"].sum())
        nc_sum = float(sub["_nc"].sum())
        n_long = int((sub["_fired"] & sub["_is_long"]).sum())
        n_short = int((sub["_fired"] & sub["_is_short"]).sum())
        pnl_long = float(sub["_nc"][sub["_fired"] & sub["_is_long"]].sum())
        pnl_short = float(sub["_nc"][sub["_fired"] & sub["_is_short"]].sum())
        # All-bar abs(edge) and realized vol — gives the universe shape, not
        # just fired bars.
        abs_edge_mean = float(sub["_abs_edge"].mean())
        abs_edge_p90 = float(sub["_abs_edge"].quantile(0.90))
        rv_mean = float(sub["_rv"].mean())
        rv_p90 = float(sub["_rv"].quantile(0.90))
        rows.append({
            "month": m, "n_trades": n_trades, "nc_pnl": nc_sum,
            "n_long": n_long, "n_short": n_short,
            "pnl_long": pnl_long, "pnl_short": pnl_short,
            "abs_edge_mean": abs_edge_mean, "abs_edge_p90": abs_edge_p90,
            "rv_mean": rv_mean, "rv_p90": rv_p90,
        })
    return pd.DataFrame(rows).set_index("month")


pre_m = per_month(pre)
post_m = per_month(post)

# Align indexes
months = sorted(set(pre_m.index) | set(post_m.index))
pre_m = pre_m.reindex(months).fillna(0)
post_m = post_m.reindex(months).fillna(0)

# Deltas
delta = post_m - pre_m
delta = delta.add_suffix("_delta")
combined = pd.concat([
    pre_m.add_suffix("_pre"),
    post_m.add_suffix("_post"),
    delta,
], axis=1)

# ---- Regime classification ------------------------------------------------
# External market knowledge + memory:
#   - 2022-05: LUNA/UST collapse (May 7-12)
#   - 2022-06: 3AC + Celsius unwind (mid-June)
#   - 2022-09 to 2022-10: post-Merge selloff, FOMC shocks
#   - 2022-11: FTX collapse (Nov 6-11), peak crisis vol
#   - 2022-12: post-FTX recovery, year-end + tax-loss
#   - 2023-01 to 2023-02: regional bank wobble starts
#   - 2023-03: SVB / Signature collapse, banking crisis (Mar 9-13)
#   - 2023-04 to 2023-08: low-vol consolidation summer
#   - 2023-10: Israel-Hamas Oct 7
#   - 2023-11 to 2023-12: BTC ETF speculation rally
#   - 2024-01: BTC spot ETF approval Jan 10
#   - 2024-03: ATH ~$73K Mar 14, top
#   - 2024-04: post-ATH selloff, halving Apr 19
#   - 2024-05 to 2024-07: range-bound consolidation
#   - 2024-08: Aug 5 yen carry unwind / vol spike
#   - 2024-09 to 2024-10: pre-election chop
#   - 2024-11: Trump win Nov 5, post-election rally
#   - 2024-12: rally continuation
#   - 2025-01 to 2025-02: post-rally consolidation
#   - 2025-03: tariff-induced selloff
#   - 2025-04: tariff peak fear (Apr 2 "Liberation Day")
#   - 2025-05 to 2025-08: recovery summer
#   - 2025-09 to 2025-10: chop
#   - 2025-11: rally
#   - 2025-12 to 2026-01: year-end consolidation
#   - 2026-02: vol spike (per +$74K month → genuine event-driven)
#   - 2026-03 to 2026-04: tail / quiet end of sample

REGIME = {
    # Crisis / event-driven months (vol spike, structural breaks)
    "2022-05": "crisis",  # LUNA
    "2022-06": "crisis",  # 3AC/Celsius unwind
    "2022-11": "crisis",  # FTX
    "2022-12": "post-crisis",  # FTX aftermath
    "2023-03": "crisis",  # SVB
    "2024-03": "trending-up",  # ATH rally
    "2024-04": "post-event",  # halving + post-ATH selloff
    "2024-08": "shock",  # Yen carry unwind
    "2024-11": "trending-up",  # Trump win
    "2025-03": "shock",  # Tariff
    "2025-04": "shock",  # Tariff peak
    "2026-02": "shock",  # Spike (per data)
    # Trending months (clear directional moves)
    "2022-02": "trending-down",
    "2022-04": "trending-down",
    "2022-09": "trending-down",
    "2022-10": "trending-down",
    "2022-07": "trending-down",
    "2022-08": "trending-down",
    "2023-11": "trending-up",
    "2023-12": "trending-up",
    "2024-01": "trending-up",
    "2024-02": "trending-up",
    "2024-12": "trending-up",
    "2025-01": "trending-up",
    "2025-02": "post-event",
    "2025-11": "trending-up",
    # Calm / consolidation months
    "2022-03": "calm",
    "2023-01": "calm",
    "2023-02": "calm",
    "2023-04": "calm",
    "2023-05": "calm",
    "2023-06": "calm",
    "2023-07": "calm",
    "2023-08": "calm",
    "2023-09": "calm",
    "2023-10": "calm",
    "2024-05": "calm",
    "2024-06": "calm",
    "2024-07": "calm",
    "2024-09": "calm",
    "2024-10": "calm",
    "2025-05": "calm",
    "2025-06": "calm",
    "2025-07": "calm",
    "2025-08": "calm",
    "2025-09": "calm",
    "2025-10": "calm",
    "2025-12": "calm",
    "2026-01": "calm",
    "2026-03": "calm",
    "2026-04": "calm",
}

# Sanity: every month maps
missing = [m for m in months if m not in REGIME]
if missing:
    print(f"WARNING: missing regime tags for {missing}")

combined["regime"] = combined.index.map(REGIME)

# ---- Question 1 / 2: Regime decomposition -------------------------------
print("\n" + "=" * 72)
print("REGIME DECOMPOSITION")
print("=" * 72)
reg_agg = (combined.groupby("regime")
           .agg(n_months=("nc_pnl_pre", "size"),
                pre_pnl=("nc_pnl_pre", "sum"),
                post_pnl=("nc_pnl_post", "sum"),
                delta=("nc_pnl_delta", "sum"),
                pre_long=("pnl_long_pre", "sum"),
                post_long=("pnl_long_post", "sum"),
                pre_short=("pnl_short_pre", "sum"),
                post_short=("pnl_short_post", "sum"),
                pre_trades=("n_trades_pre", "sum"),
                post_trades=("n_trades_post", "sum"),
                ))
reg_agg["delta_long"] = reg_agg["post_long"] - reg_agg["pre_long"]
reg_agg["delta_short"] = reg_agg["post_short"] - reg_agg["pre_short"]
reg_agg["pct_of_total"] = reg_agg["delta"] / reg_agg["delta"].sum() * 100
print(reg_agg.round(0))

# ---- Question 3: Volatility-vs-delta correlation ------------------------
print("\n" + "=" * 72)
print("DELTA vs RV CORRELATION (per-month)")
print("=" * 72)
# Use pre-retrain RV (universe-level, both cycles see same data)
corr_rv = combined["nc_pnl_delta"].corr(combined["rv_mean_pre"])
spear_rv = combined["nc_pnl_delta"].corr(combined["rv_mean_pre"], method="spearman")
print(f"Pearson(delta, rv_mean):  {corr_rv:+.3f}")
print(f"Spearman(delta, rv_mean): {spear_rv:+.3f}")
# Bin by RV quartile
combined["rv_bin"] = pd.qcut(combined["rv_mean_pre"], 4,
                              labels=["Q1-low", "Q2", "Q3", "Q4-high"])
rv_bin_agg = combined.groupby("rv_bin", observed=False).agg(
    n_months=("nc_pnl_delta", "size"),
    delta_sum=("nc_pnl_delta", "sum"),
    delta_mean=("nc_pnl_delta", "mean"),
    pre_pnl_mean=("nc_pnl_pre", "mean"),
)
print("\nDelta by RV quartile:")
print(rv_bin_agg.round(1))

# ---- Question 4: |edge| distribution shift per month --------------------
print("\n" + "=" * 72)
print("|EDGE| DISTRIBUTION SHIFT PER MONTH")
print("=" * 72)
edge_shift = combined[["abs_edge_mean_pre", "abs_edge_mean_post"]].copy()
edge_shift["delta_abs_edge"] = (edge_shift["abs_edge_mean_post"]
                                  - edge_shift["abs_edge_mean_pre"])
edge_shift["pct_change"] = (edge_shift["delta_abs_edge"]
                              / edge_shift["abs_edge_mean_pre"] * 100)
edge_shift["pnl_delta"] = combined["nc_pnl_delta"]
# Rank by edge softness
print("\nTop-10 months where |edge| softened MOST in cycle-10:")
print(edge_shift.sort_values("delta_abs_edge").head(10).round(3))
print("\nTop-10 months where |edge| sharpened MOST in cycle-10:")
print(edge_shift.sort_values("delta_abs_edge").tail(10).round(3))

corr_edge = edge_shift["delta_abs_edge"].corr(edge_shift["pnl_delta"])
spear_edge = edge_shift["delta_abs_edge"].corr(edge_shift["pnl_delta"],
                                                  method="spearman")
print(f"\nPearson(delta_abs_edge, delta_pnl):  {corr_edge:+.3f}")
print(f"Spearman(delta_abs_edge, delta_pnl): {spear_edge:+.3f}")

# ---- Question 5: Long vs short imbalance per month ----------------------
print("\n" + "=" * 72)
print("LONG VS SHORT PER-MONTH IMBALANCE")
print("=" * 72)
ls = combined[["pnl_long_pre", "pnl_long_post", "pnl_short_pre",
                "pnl_short_post", "n_long_pre", "n_long_post",
                "n_short_pre", "n_short_post"]].copy()
ls["delta_long"] = ls["pnl_long_post"] - ls["pnl_long_pre"]
ls["delta_short"] = ls["pnl_short_post"] - ls["pnl_short_pre"]
ls["delta_n_long"] = ls["n_long_post"] - ls["n_long_pre"]
ls["delta_n_short"] = ls["n_short_post"] - ls["n_short_pre"]
ls["pnl_delta"] = combined["nc_pnl_delta"]
print("\nLong-side: top-10 months where cycle-10 LOST PnL on longs:")
print(ls.sort_values("delta_long").head(10)[
    ["delta_long", "delta_n_long", "pnl_delta"]].round(0))
print("\nShort-side: top-10 months where cycle-10 LOST PnL on shorts:")
print(ls.sort_values("delta_short").head(10)[
    ["delta_short", "delta_n_short", "pnl_delta"]].round(0))
print("\nShort-side: top-10 months where cycle-10 GAINED PnL on shorts:")
print(ls.sort_values("delta_short").tail(10)[
    ["delta_short", "delta_n_short", "pnl_delta"]].round(0))

# Aggregate long/short delta
print(f"\nTotal long-side  PnL delta: ${ls['delta_long'].sum():+,.0f} "
       f"(cycle-10 vs cycle-9)")
print(f"Total short-side PnL delta: ${ls['delta_short'].sum():+,.0f}")
print(f"Total trade-count delta: long {ls['delta_n_long'].sum():+.0f}, "
       f"short {ls['delta_n_short'].sum():+.0f}")

# ---- Question 6: Variance decomposition ---------------------------------
print("\n" + "=" * 72)
print("VARIANCE DECOMPOSITION")
print("=" * 72)
total_delta = combined["nc_pnl_delta"].sum()
# Top-3 worst months
worst3 = combined["nc_pnl_delta"].nsmallest(3).sum()
worst5 = combined["nc_pnl_delta"].nsmallest(5).sum()
worst10 = combined["nc_pnl_delta"].nsmallest(10).sum()
best3 = combined["nc_pnl_delta"].nlargest(3).sum()
best5 = combined["nc_pnl_delta"].nlargest(5).sum()
best10 = combined["nc_pnl_delta"].nlargest(10).sum()
print(f"Total NC PnL delta: ${total_delta:+,.0f}")
print(f"Sum of 3 worst-delta months:  ${worst3:+,.0f}")
print(f"Sum of 5 worst-delta months:  ${worst5:+,.0f}")
print(f"Sum of 10 worst-delta months: ${worst10:+,.0f}")
print(f"Sum of 3 best-delta months:   ${best3:+,.0f}")
print(f"Sum of 5 best-delta months:   ${best5:+,.0f}")
print(f"Sum of 10 best-delta months:  ${best10:+,.0f}")

# ---- Save consolidated table --------------------------------------------
out_csv = REPO_ROOT / "vol_arb" / "output" / "cycle10_regime_analysis.csv"
combined.to_csv(out_csv)
print(f"\nWrote per-month table to {out_csv}")

# Save regime aggregate
out_csv2 = REPO_ROOT / "vol_arb" / "output" / "cycle10_regime_summary.csv"
reg_agg.to_csv(out_csv2)
print(f"Wrote regime aggregate to {out_csv2}")

print("\nDONE")
