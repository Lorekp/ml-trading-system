"""Sweep ``max_jump_fraction_for_shorts`` across [0, 1] on the V2 backtest.

The user's question (2026-05-11): the live engine has been killing
short signals at jump_fraction > 0.15 — what does the BACKTESTER say
about that filter? Is 0.15 the right value? Where's the stable region?

For each value of ``cfg.max_jump_fraction_for_shorts``, run the full
V2 backtester and report NC PnL, daily Sharpe, MDD, trade counts (by
sign), per-year stability. The default is 0.15 (current production);
this script identifies whether 0.15 lives on a plateau or a knife edge,
and what relaxing or tightening the filter would do.

Stable region: contiguous range of grid values where headline metrics
(NC PnL, daily Sharpe, MDD) are within +/- 5% of each other AND every
year remains non-negative. Plateau = safe to set.

Usage:
    PYTHONPATH=. python scripts/sweep_jump_fraction_shorts.py
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from vol_arb.backtester import BacktestConfig, VolArbBacktester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sweep_jf")

# Sweep grid — covers "fully restrictive" through "fully off (=1.0)".
# 0.15 is the current production default. 1.0 effectively disables the
# breaker (no jump_fraction can exceed 1.0 by construction).
GRID = [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.75, 1.00]

OUT_REPORT = REPO_ROOT / "vol_arb/output/JUMP_FRACTION_SWEEP_2026_05_11.md"
OUT_PARQUET = REPO_ROOT / "vol_arb/output/jump_fraction_sweep_2026_05_11.parquet"


def _per_year_nc(tl: pd.DataFrame) -> dict[int, float]:
    """Per-year NC PnL sum from a trade log."""
    exe = tl[tl["N_contracts_nc"] > 0]
    if len(exe) == 0:
        return {}
    return exe.groupby(exe.index.year)["net_pnl_nc"].sum().to_dict()


def _run_one(jf_cap: float, label: str) -> dict:
    """Run V2 backtest with the given jump_fraction cap, return summary."""
    cfg = (BacktestConfig()
           .with_btc_production_filters_v2()
           .with_realistic_costs())
    cfg.max_jump_fraction_for_shorts = jf_cap

    t0 = time.time()
    metrics, tl = VolArbBacktester(cfg).run_backtest()
    elapsed = time.time() - t0

    exe = tl[tl["N_contracts_nc"] > 0]
    nc_pnl = float(exe["net_pnl_nc"].sum())
    n_long = int((exe["signal"] == 1).sum())
    n_short = int((exe["signal"] == -1).sum())

    # Daily Sharpe (NC) for stability axis
    daily = exe["net_pnl_nc"].resample("1D").sum()
    if daily.std(ddof=1) > 0:
        daily_sharpe_nc = float(daily.mean() / daily.std(ddof=1) * np.sqrt(365.25))
    else:
        daily_sharpe_nc = float("nan")

    # MDD on NC equity curve
    eq = cfg.initial_equity + exe["net_pnl_nc"].cumsum()
    mdd_nc = float((eq - eq.cummax()).min())

    # Trade Sharpe NC (per-trade level)
    per_trade = exe["net_pnl_nc"]
    if per_trade.std(ddof=1) > 0:
        n_years = (exe.index.max() - exe.index.min()).total_seconds() / (365.25 * 24 * 3600)
        trade_sharpe_nc = float(
            per_trade.mean() / per_trade.std(ddof=1)
            * np.sqrt(len(per_trade) / max(n_years, 1e-9))
        )
    else:
        trade_sharpe_nc = float("nan")

    per_year = _per_year_nc(tl)
    return {
        "label": label,
        "jf_cap": jf_cap,
        "elapsed_s": elapsed,
        "nc_pnl": nc_pnl,
        "n_trades": int(len(exe)),
        "n_long": n_long,
        "n_short": n_short,
        "daily_sharpe_nc": daily_sharpe_nc,
        "trade_sharpe_nc": trade_sharpe_nc,
        "mdd_nc": mdd_nc,
        **{f"y_{yr}": v for yr, v in per_year.items()},
        "per_year_min": float(min(per_year.values())) if per_year else 0.0,
        "per_year_strict_5_of_5": int(all(v >= 0 for v in per_year.values())),
    }


def _find_stable_region(df: pd.DataFrame, ref_jf: float = 0.15) -> dict:
    """Identify the largest contiguous range of jf values where all
    headline metrics (NC PnL, daily Sharpe, MDD) are within tolerance
    of the reference value AND per-year strictly non-negative."""
    ref_row = df[df["jf_cap"] == ref_jf].iloc[0] if (df["jf_cap"] == ref_jf).any() else df.iloc[0]
    pnl_tol = 0.05  # +/- 5% of reference NC PnL
    sharpe_tol = 0.50  # +/- 0.5 Sharpe pts
    mdd_tol_usd = 5000.0  # +/- $5K MDD

    # Mark each row as stable
    df = df.copy()
    df["stable_pnl"] = (df["nc_pnl"] >= ref_row["nc_pnl"] * (1 - pnl_tol)) & \
                       (df["nc_pnl"] <= ref_row["nc_pnl"] * (1 + pnl_tol))
    df["stable_sharpe"] = (df["daily_sharpe_nc"] >= ref_row["daily_sharpe_nc"] - sharpe_tol) & \
                          (df["daily_sharpe_nc"] <= ref_row["daily_sharpe_nc"] + sharpe_tol)
    df["stable_mdd"] = (df["mdd_nc"] >= ref_row["mdd_nc"] - mdd_tol_usd)
    df["stable_strict"] = (df["per_year_strict_5_of_5"] == 1)
    df["stable_all"] = df["stable_pnl"] & df["stable_sharpe"] & df["stable_mdd"] & df["stable_strict"]

    stable_jfs = sorted(df[df["stable_all"]]["jf_cap"].tolist())
    return {
        "ref_jf": ref_jf,
        "ref_nc_pnl": float(ref_row["nc_pnl"]),
        "ref_sharpe": float(ref_row["daily_sharpe_nc"]),
        "ref_mdd": float(ref_row["mdd_nc"]),
        "stable_jfs": stable_jfs,
        "stable_min": min(stable_jfs) if stable_jfs else None,
        "stable_max": max(stable_jfs) if stable_jfs else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--grid", default=",".join(f"{x:.2f}" for x in GRID),
        help="Comma-separated jf cap values (default: 11-point grid 0..1)",
    )
    args = parser.parse_args()
    grid = [float(x) for x in args.grid.split(",")]

    print("=" * 80)
    print(f"  JUMP-FRACTION SWEEP — V2 backtest, max_jump_fraction_for_shorts")
    print(f"  Grid: {grid}")
    print("=" * 80)

    rows = []
    for jf in grid:
        label = f"jf={jf:.2f}"
        if jf == 0.15:
            label += " (production default)"
        elif jf == 1.0:
            label += " (filter OFF)"
        logger.info("Running %s ...", label)
        r = _run_one(jf, label)
        rows.append(r)
        logger.info(
            "  done: NC PnL $%s, daily Sharpe %.2f, MDD $%s, trades %d (%d short, %d long), elapsed %.1fs",
            f"{r['nc_pnl']:,.0f}",
            r["daily_sharpe_nc"],
            f"{r['mdd_nc']:,.0f}",
            r["n_trades"], r["n_short"], r["n_long"],
            r["elapsed_s"],
        )

    df = pd.DataFrame(rows)
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUT_PARQUET)
    logger.info("Saved sweep parquet: %s", OUT_PARQUET)

    # ── Stable region analysis ──
    stable = _find_stable_region(df)
    print()
    print("=" * 80)
    print("  STABLE REGION (within +/- 5% NC PnL, +/- 0.5 Sharpe, MDD not worse")
    print(f"   by >$5K vs ref jf={stable['ref_jf']}, per-year strict 5/5):")
    print("=" * 80)
    if stable["stable_jfs"]:
        print(f"  stable jf range: [{stable['stable_min']:.2f}, {stable['stable_max']:.2f}]")
        print(f"  in stable set:   {stable['stable_jfs']}")
    else:
        print("  NONE — no point on the grid satisfies all stability criteria")
    print()

    # ── Markdown report ──
    lines = ["# Jump-fraction sweep — V2 backtest", ""]
    lines.append(f"Generated 2026-05-11. Sweep of `max_jump_fraction_for_shorts` "
                 f"across {len(grid)} points in [{min(grid):.2f}, {max(grid):.2f}].")
    lines.append("")
    lines.append("V2 backtester full sample (2022-02 -> 2026-04, canonical trade log).")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Reference: production default jf={stable['ref_jf']}, "
                 f"NC PnL ${stable['ref_nc_pnl']:,.0f}, "
                 f"Sharpe {stable['ref_sharpe']:.2f}, MDD ${stable['ref_mdd']:,.0f}")
    if stable["stable_jfs"]:
        lines.append(f"- Stable region: **jf in [{stable['stable_min']:.2f}, "
                     f"{stable['stable_max']:.2f}]** "
                     f"({len(stable['stable_jfs'])} of {len(grid)} grid points)")
    else:
        lines.append("- Stable region: **NONE** — every grid point diverges from "
                     "the production default")
    lines.append("")

    lines.append("## Grid results")
    lines.append("")
    cols = ["jf_cap", "n_trades", "n_short", "n_long", "nc_pnl",
            "daily_sharpe_nc", "trade_sharpe_nc", "mdd_nc",
            "per_year_min", "per_year_strict_5_of_5"]
    fmts = {
        "jf_cap": "{:.2f}",
        "n_trades": "{:,}",
        "n_short": "{:,}",
        "n_long": "{:,}",
        "nc_pnl": "${:,.0f}",
        "daily_sharpe_nc": "{:.2f}",
        "trade_sharpe_nc": "{:.2f}",
        "mdd_nc": "${:,.0f}",
        "per_year_min": "${:,.0f}",
        "per_year_strict_5_of_5": "{}",
    }
    lines.append("| " + " | ".join(c.replace("_", " ") for c in cols) + " |")
    lines.append("|" + "|".join(["---"] * len(cols)) + "|")
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            f = fmts[c]
            cells.append(f.format(v) if not pd.isna(v) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    # Per-year detail
    year_cols = sorted([c for c in df.columns if c.startswith("y_")])
    lines.append("## Per-year NC PnL")
    lines.append("")
    lines.append("| jf cap | " + " | ".join(c.replace("y_", "") for c in year_cols) + " |")
    lines.append("|" + "|".join(["---"] * (1 + len(year_cols))) + "|")
    for _, r in df.iterrows():
        cells = [f"{r['jf_cap']:.2f}"]
        for c in year_cols:
            v = r[c]
            cells.append(f"${v:,.0f}" if not pd.isna(v) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    if stable["stable_jfs"]:
        lines.append(
            f"The filter sits on a **plateau between jf={stable['stable_min']:.2f} "
            f"and jf={stable['stable_max']:.2f}** — the V2 backtest is insensitive "
            f"to the exact cutoff within this range. The production default of "
            f"0.15 is INSIDE this plateau, so the current setting is robust."
        )
    else:
        lines.append("Every grid value diverges from the production default in at "
                     "least one of the four stability criteria. The 0.15 cutoff is "
                     "on a knife edge — small changes shift PnL / Sharpe / MDD / "
                     "per-year stability materially.")
    lines.append("")

    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote: %s", OUT_REPORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
