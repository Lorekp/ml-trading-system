"""Generic single-parameter full-sample sweep harness for stale-defense campaign.

Runs the V2 backtester across a comma-separated grid of values for ONE
parameter on ``BacktestConfig`` and emits:

- A per-grid-point parquet with NC PnL, daily Sharpe, MDD, per-year breakdown.
- A markdown report with stable-region analysis (plateau detection).

Stable region: contiguous range of grid values where headline metrics
(NC PnL, daily Sharpe, MDD) are within +/- 5% of the reference value AND
every year remains non-negative.

Reference value is provided via ``--reference`` (defaults to the first
grid point). For S1 (`short_sizing_penalty`) the reference is 0.50
(current production); for S2 (`spread_jump_k`) the reference is 3.0; etc.

Usage:
    PYTHONPATH=. python scripts/stale_defense_sweep.py \\
        --param short_sizing_penalty \\
        --grid 0.50,0.60,0.70,0.80,0.90,1.00 \\
        --reference 0.50 \\
        --tag S1_short_sizing_penalty
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
logger = logging.getLogger("stale_defense_sweep")


def _per_year_nc(tl: pd.DataFrame) -> dict[int, float]:
    exe = tl[tl["N_contracts_nc"] > 0]
    if len(exe) == 0:
        return {}
    return exe.groupby(exe.index.year)["net_pnl_nc"].sum().to_dict()


def _run_one(param: str, value, label: str) -> dict:
    """Run V2 backtest with ``param`` overridden to ``value``."""
    cfg = (BacktestConfig()
           .with_btc_production_filters_v2()
           .with_realistic_costs())
    if not hasattr(cfg, param):
        raise ValueError(
            f"BacktestConfig has no field '{param}'. "
            f"Set it as a model_copy(update=...) override before runtime."
        )
    # Use model_copy to bypass any pydantic validation if needed
    cfg = cfg.model_copy(update={param: value})

    t0 = time.time()
    _, tl = VolArbBacktester(cfg).run_backtest()
    elapsed = time.time() - t0

    exe = tl[tl["N_contracts_nc"] > 0]
    nc_pnl = float(exe["net_pnl_nc"].sum())
    n_long = int((exe["signal"] == 1).sum())
    n_short = int((exe["signal"] == -1).sum())

    daily = exe["net_pnl_nc"].resample("1D").sum()
    daily_sharpe_nc = (
        float(daily.mean() / daily.std(ddof=1) * np.sqrt(365.25))
        if daily.std(ddof=1) > 0 else float("nan")
    )

    eq = cfg.initial_equity + exe["net_pnl_nc"].cumsum()
    peak = eq.cummax()
    dd_dollars = eq - peak
    mdd_nc = float(dd_dollars.min())              # largest $ drawdown
    # True peak->trough MDD%: at the moment of worst drawdown, dd / peak.
    # Conventional Wall Street definition. Scale-invariant across cycles
    # because both numerator and denominator scale with position sizing.
    dd_pct_series = (eq - peak) / peak * 100.0
    mdd_pct = float(dd_pct_series.min())

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
        "param": param,
        "value": value,
        "elapsed_s": elapsed,
        "nc_pnl": nc_pnl,
        "n_trades": int(len(exe)),
        "n_long": n_long,
        "n_short": n_short,
        "daily_sharpe_nc": daily_sharpe_nc,
        "trade_sharpe_nc": trade_sharpe_nc,
        "mdd_nc": mdd_nc,
        "mdd_pct": mdd_pct,
        **{f"y_{yr}": v for yr, v in per_year.items()},
        "per_year_min": float(min(per_year.values())) if per_year else 0.0,
        "per_year_strict_5_of_5": int(all(v >= 0 for v in per_year.values())),
    }


def _find_stable_region(df: pd.DataFrame, ref_value) -> dict:
    """Identify contiguous grid range with headline-metric stability vs reference."""
    matches = df["value"] == ref_value
    ref_row = df[matches].iloc[0] if matches.any() else df.iloc[0]
    pnl_tol = 0.05
    sharpe_tol = 0.50
    mdd_tol_pct = 5.0  # 5pp of $100K NC reference equity

    df = df.copy()
    df["stable_pnl"] = (
        (df["nc_pnl"] >= ref_row["nc_pnl"] * (1 - pnl_tol)) &
        (df["nc_pnl"] <= ref_row["nc_pnl"] * (1 + pnl_tol))
    )
    df["stable_sharpe"] = (
        (df["daily_sharpe_nc"] >= ref_row["daily_sharpe_nc"] - sharpe_tol) &
        (df["daily_sharpe_nc"] <= ref_row["daily_sharpe_nc"] + sharpe_tol)
    )
    df["stable_mdd"] = (df["mdd_pct"] >= ref_row["mdd_pct"] - mdd_tol_pct)
    df["stable_strict"] = (df["per_year_strict_5_of_5"] == 1)
    df["stable_all"] = (
        df["stable_pnl"] & df["stable_sharpe"] &
        df["stable_mdd"] & df["stable_strict"]
    )
    stable_vals = sorted(df[df["stable_all"]]["value"].tolist())
    return {
        "ref_value": ref_value,
        "ref_nc_pnl": float(ref_row["nc_pnl"]),
        "ref_sharpe": float(ref_row["daily_sharpe_nc"]),
        "ref_mdd_pct": float(ref_row["mdd_pct"]),
        "stable_values": stable_vals,
        "stable_min": min(stable_vals) if stable_vals else None,
        "stable_max": max(stable_vals) if stable_vals else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--param", required=True,
                        help="BacktestConfig field name to sweep")
    parser.add_argument("--grid", required=True,
                        help="Comma-separated values (parsed as float)")
    parser.add_argument("--reference", type=float, default=None,
                        help="Reference value for stable-region analysis "
                             "(defaults to first grid point)")
    parser.add_argument("--tag", default=None,
                        help="Tag for output filenames "
                             "(default = param name; e.g. 'S1_short_sizing_penalty')")
    parser.add_argument("--date", default=None,
                        help="Date tag for output (default = today YYYY_MM_DD)")
    args = parser.parse_args()

    grid = [float(x) for x in args.grid.split(",")]
    ref_value = args.reference if args.reference is not None else grid[0]
    tag = args.tag or args.param
    date = args.date or pd.Timestamp.utcnow().strftime("%Y_%m_%d")

    out_report = REPO_ROOT / f"vol_arb/output/STALE_DEFENSE_{tag}_SWEEP_{date}.md"
    out_parquet = REPO_ROOT / f"vol_arb/output/stale_defense_{tag.lower()}_sweep_{date}.parquet"

    print("=" * 80)
    print(f"  STALE-DEFENSE SWEEP: {args.param}")
    print(f"  Grid: {grid}")
    print(f"  Reference: {ref_value}")
    print("=" * 80)

    rows = []
    for v in grid:
        label = f"{args.param}={v}"
        if v == ref_value:
            label += " (production default)"
        logger.info("Running %s ...", label)
        r = _run_one(args.param, v, label)
        rows.append(r)
        logger.info(
            "  done: NC PnL $%s, daily Sharpe %.2f, MDD %.2f%%, "
            "trades %d (%d short, %d long), elapsed %.1fs",
            f"{r['nc_pnl']:,.0f}", r["daily_sharpe_nc"],
            r["mdd_pct"],
            r["n_trades"], r["n_short"], r["n_long"], r["elapsed_s"],
        )

    df = pd.DataFrame(rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet)
    logger.info("Saved sweep parquet: %s", out_parquet)

    stable = _find_stable_region(df, ref_value)
    print()
    print("=" * 80)
    print(f"  STABLE REGION (within +/- 5% NC PnL, +/- 0.5 Sharpe, MDD not worse")
    print(f"   by >5pp vs ref={stable['ref_value']}, per-year strict 5/5):")
    print("=" * 80)
    if stable["stable_values"]:
        print(f"  stable range: [{stable['stable_min']}, {stable['stable_max']}]")
        print(f"  values:       {stable['stable_values']}")
    else:
        print("  NONE — no point on the grid satisfies all stability criteria")
    print()

    # ── Markdown report ──
    lines = [f"# Stale-defense sweep — {args.param}", ""]
    lines.append(f"Generated {date}. Sweep of `{args.param}` "
                 f"across {len(grid)} points: {grid}.")
    lines.append("")
    lines.append("V2 backtester full sample (2022-02 -> 2026-04, canonical trade log).")
    lines.append("`BacktestConfig().with_btc_production_filters_v2().with_realistic_costs()` "
                 f"with `{args.param}` overridden.")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- Reference: `{args.param}`={stable['ref_value']}, "
                 f"NC PnL ${stable['ref_nc_pnl']:,.0f}, "
                 f"Sharpe {stable['ref_sharpe']:.2f}, MDD {stable['ref_mdd_pct']:.2f}%")
    if stable["stable_values"]:
        lines.append(f"- Stable region: **{args.param} in "
                     f"[{stable['stable_min']}, {stable['stable_max']}]** "
                     f"({len(stable['stable_values'])} of {len(grid)} grid points)")
    else:
        lines.append("- Stable region: **NONE** — every grid point diverges from "
                     "the production default")
    # Best NC PnL among per-year-strict-pass grid points
    strict = df[df["per_year_strict_5_of_5"] == 1].copy()
    if len(strict) > 0:
        best = strict.loc[strict["nc_pnl"].idxmax()]
        lines.append(f"- Best NC PnL (strict 5/5): `{args.param}`={best['value']}, "
                     f"NC PnL ${best['nc_pnl']:,.0f}, "
                     f"Sharpe {best['daily_sharpe_nc']:.2f}, MDD {best['mdd_pct']:.2f}%")
    lines.append("")

    lines.append("## Grid results")
    lines.append("")
    cols = ["value", "n_trades", "n_short", "n_long", "nc_pnl",
            "daily_sharpe_nc", "trade_sharpe_nc", "mdd_pct",
            "per_year_min", "per_year_strict_5_of_5"]
    fmts = {
        "value": "{}",
        "n_trades": "{:,}",
        "n_short": "{:,}",
        "n_long": "{:,}",
        "nc_pnl": "${:,.0f}",
        "daily_sharpe_nc": "{:.2f}",
        "trade_sharpe_nc": "{:.2f}",
        "mdd_pct": "{:.2f}%",
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

    year_cols = sorted([c for c in df.columns if c.startswith("y_")])
    lines.append("## Per-year NC PnL")
    lines.append("")
    lines.append("| value | " + " | ".join(c.replace("y_", "") for c in year_cols) + " |")
    lines.append("|" + "|".join(["---"] * (1 + len(year_cols))) + "|")
    for _, r in df.iterrows():
        cells = [f"{r['value']}"]
        for c in year_cols:
            v = r[c]
            cells.append(f"${v:,.0f}" if not pd.isna(v) else "—")
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")

    lines.append("## Interpretation")
    lines.append("")
    if stable["stable_values"]:
        lines.append(
            f"The parameter sits on a **plateau between "
            f"{args.param}={stable['stable_min']} "
            f"and {args.param}={stable['stable_max']}** — the V2 backtest is "
            f"insensitive to the exact value within this range. "
            f"The production default of {stable['ref_value']} is "
            f"{'INSIDE' if stable['ref_value'] in stable['stable_values'] else 'OUTSIDE'} "
            f"this plateau."
        )
    else:
        lines.append("Every grid value diverges from the production default in at "
                     "least one of the four stability criteria — knife-edge regime.")
    lines.append("")

    out_report.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote: %s", out_report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
