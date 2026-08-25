#!/usr/bin/env python
"""Multi-dimensional analysis from sweep_results.csv."""

import numpy as np
import pandas as pd

df = pd.read_csv("experiments/.cache/sweep_results.csv")
# Normalize corr_throttle for display
df["ct"] = df["corr_throttle"].apply(lambda x: "CT" if x != "none" else "none")
print(f"{len(df)} configs loaded\n")

W = 105


def header(title):
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")


def dim_table(col, name):
    header(f"DIMENSION: {name}")
    g = df.groupby(col).agg(
        SR_avg=("sharpe", "mean"), SR_std=("sharpe", "std"),
        CAGR_avg=("cagr", "mean"), CAGR_max=("cagr", "max"),
        DD_avg=("max_drawdown", "mean"), DD_max=("max_drawdown", "max"),
        n=("sharpe", "count"),
    )
    print(f"  {'Value':>8} {'SR avg':>7} {'SR std':>7} {'CAGR avg':>9} {'CAGR max':>9} "
          f"{'DD avg':>7} {'DD max':>7} {'n':>4}")
    for val, row in g.iterrows():
        v = f"{val:.0%}" if isinstance(val, float) else str(val)
        print(f"  {v:>8} {row.SR_avg:>7.2f} {row.SR_std:>7.2f} "
              f"{row.CAGR_avg:>8.0f}% {row.CAGR_max:>8.0f}% "
              f"{row.DD_avg:>6.1f}% {row.DD_max:>6.1f}% {int(row.n):>4}")


# ── MARGINAL EFFECTS ──

dim_table("sizing", "BASE SIZING PCT")
dim_table("ct", "CORRELATION THROTTLE (CT=0.90 vs none)")
dim_table("max_per_asset", "PER-ASSET CONCENTRATION CAP")
dim_table("max_exposure", "TOTAL EXPOSURE CAP (90% vs 95%)")
dim_table("max_dd", "MAX DRAWDOWN TOLERANCE")
dim_table("delever_start", "DELEVERAGING START")


# ── 2-WAY INTERACTIONS ──

def pivot(val, idx, col, label, fmt=".2f"):
    header(f"{label}")
    piv = df.pivot_table(val, idx, col, "mean")
    if fmt == ".0f":
        print(piv.round(0).fillna("-").to_string())
    else:
        print(piv.round(2).fillna("-").to_string())


pivot("sharpe", "sizing", "ct", "SIZING x CT (mean Sharpe)")
pivot("cagr", "sizing", "ct", "SIZING x CT (mean CAGR)", ".0f")
pivot("cagr", "sizing", "max_per_asset", "SIZING x PER-ASSET (mean CAGR)", ".0f")
pivot("sharpe", "sizing", "max_per_asset", "SIZING x PER-ASSET (mean Sharpe)")
pivot("max_drawdown", "sizing", "max_per_asset", "SIZING x PER-ASSET (mean MaxDD)")
pivot("sharpe", "sizing", "max_dd", "SIZING x MAX_DD (mean Sharpe)")
pivot("cagr", "sizing", "max_dd", "SIZING x MAX_DD (mean CAGR)", ".0f")
pivot("sharpe", "sizing", "delever_start", "SIZING x DELEVER (mean Sharpe)")
pivot("cagr", "sizing", "delever_start", "SIZING x DELEVER (mean CAGR)", ".0f")
pivot("sharpe", "max_dd", "delever_start", "MAX_DD x DELEVER (mean Sharpe)")


# ── 3-WAY INTERACTIONS ──

header("3-WAY: SIZING x PER-ASSET x CT (mean Sharpe)")
g = df.groupby(["sizing", "max_per_asset", "ct"])["sharpe"].mean().round(2)
print(g.unstack(["max_per_asset", "ct"]).to_string())

header("3-WAY: SIZING x PER-ASSET x CT (mean CAGR)")
g = df.groupby(["sizing", "max_per_asset", "ct"])["cagr"].mean().round(0)
print(g.unstack(["max_per_asset", "ct"]).to_string())

header("3-WAY: SIZING x MAX_DD x CT (mean Sharpe)")
g = df.groupby(["sizing", "max_dd", "ct"])["sharpe"].mean().round(2)
print(g.unstack(["max_dd", "ct"]).to_string())


# ── EFFICIENT FRONTIERS ──

header("EFFICIENT FRONTIER: Best Sharpe per CAGR bucket")
for lo, hi in [(10, 20), (20, 50), (50, 100), (100, 200), (200, 300), (300, 400)]:
    b = df[(df["cagr"] >= lo) & (df["cagr"] < hi)]
    if len(b) == 0:
        print(f"  CAGR {lo:>3}-{hi:>3}%: ---")
        continue
    best = b.loc[b["sharpe"].idxmax()]
    print(f"  CAGR {lo:>3}-{hi:>3}%: sz={best.sizing:.0%} mpa={best.max_per_asset:.0%} "
          f"exp={best.max_exposure:.0%} dd={best.max_dd:.0%} dl={best.delever_start:.0%} "
          f"ct={best.ct}  =>  SR={best.sharpe:.2f} CAGR={best.cagr:.0f}% "
          f"DD={best.max_drawdown:.1f}%")

header("EFFICIENT FRONTIER: Best CAGR per MaxDD bucket")
for lo, hi in [(0, 1), (1, 2), (2, 5), (5, 8), (8, 10), (10, 15)]:
    b = df[(df["max_drawdown"] >= lo) & (df["max_drawdown"] < hi)]
    if len(b) == 0:
        print(f"  DD {lo:>2}-{hi:>2}%: ---")
        continue
    best = b.loc[b["cagr"].idxmax()]
    print(f"  DD {lo:>2}-{hi:>2}%: sz={best.sizing:.0%} mpa={best.max_per_asset:.0%} "
          f"exp={best.max_exposure:.0%} dd={best.max_dd:.0%} dl={best.delever_start:.0%} "
          f"ct={best.ct}  =>  SR={best.sharpe:.2f} CAGR={best.cagr:.0f}% "
          f"DD={best.max_drawdown:.1f}%  ${best.final_eq:,.0f}")

header("EFFICIENT FRONTIER: Best Sharpe per MaxDD bucket")
for lo, hi in [(0, 1), (1, 2), (2, 5), (5, 8), (8, 10), (10, 15)]:
    b = df[(df["max_drawdown"] >= lo) & (df["max_drawdown"] < hi)]
    if len(b) == 0:
        print(f"  DD {lo:>2}-{hi:>2}%: ---")
        continue
    best = b.loc[b["sharpe"].idxmax()]
    print(f"  DD {lo:>2}-{hi:>2}%: sz={best.sizing:.0%} mpa={best.max_per_asset:.0%} "
          f"exp={best.max_exposure:.0%} dd={best.max_dd:.0%} dl={best.delever_start:.0%} "
          f"ct={best.ct}  =>  SR={best.sharpe:.2f} CAGR={best.cagr:.0f}% "
          f"DD={best.max_drawdown:.1f}%  ${best.final_eq:,.0f}")
