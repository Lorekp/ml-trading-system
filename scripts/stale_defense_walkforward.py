"""Generic single-parameter walk-forward gate harness for stale-defense campaign.

Pre-committed test (cycle-12 style) — given a baseline value and a
treatment value for ONE BacktestConfig parameter, evaluate 5 strict gates
to decide whether to ship the treatment.

PRE-COMMITTED GATES (defaults match cycle-12; can be tightened via CLI):

  G1 — OOS dPnL > 0.
  G2 — OOS daily Sharpe drop <= 0.5  (override via --sharpe-drop-max)
  G3 — OOS MDD worsens <= $2,000     (override via --mdd-worsen-max-usd)
  G4 — OOS per-year strict 2/2 non-negative.
  G5 — TRAIN dPnL and OOS dPnL same sign (cycle-12: both positive).

Reject if any gate fails. Ship only if ALL five gates pass.

Usage:
    PYTHONPATH=. python scripts/stale_defense_walkforward.py \\
        --param short_sizing_penalty \\
        --baseline 0.50 \\
        --treatment 1.00 \\
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
logger = logging.getLogger("stale_defense_wf")

TRAIN_YEARS = {2022, 2023, 2024}
OOS_YEARS = {2025, 2026}


def _run(param: str, value) -> pd.DataFrame:
    cfg = (BacktestConfig()
           .with_btc_production_filters_v2()
           .with_realistic_costs())
    if not hasattr(cfg, param):
        raise ValueError(f"BacktestConfig has no field '{param}'.")
    cfg = cfg.model_copy(update={param: value})
    _, tl = VolArbBacktester(cfg).run_backtest()
    return tl


def _metrics_on_slice(tl: pd.DataFrame, years: set[int]) -> dict:
    mask = tl.index.year.isin(years)
    df = tl.loc[mask]
    exe = df[df["N_contracts_nc"] > 0]
    if len(exe) == 0:
        return {
            "n_trades": 0, "n_short": 0, "n_long": 0, "nc_pnl": 0.0,
            "daily_sharpe_nc": float("nan"),
            "trade_sharpe_nc": float("nan"),
            "mdd_nc": 0.0,
            "per_year": {},
            "per_year_min": 0.0,
        }
    nc_pnl = float(exe["net_pnl_nc"].sum())
    daily = exe["net_pnl_nc"].resample("1D").sum()
    daily_sharpe = (
        float(daily.mean() / daily.std(ddof=1) * np.sqrt(365.25))
        if daily.std(ddof=1) > 0 else float("nan")
    )
    per_trade = exe["net_pnl_nc"]
    if per_trade.std(ddof=1) > 0:
        n_years = (exe.index.max() - exe.index.min()).total_seconds() / (365.25 * 24 * 3600)
        trade_sharpe = float(
            per_trade.mean() / per_trade.std(ddof=1)
            * np.sqrt(len(per_trade) / max(n_years, 1e-9))
        )
    else:
        trade_sharpe = float("nan")
    eq = exe["net_pnl_nc"].cumsum()
    mdd = float((eq - eq.cummax()).min())
    # MDD% on $100K NC reference equity (scale-invariant).
    mdd_pct = mdd / 100_000.0 * 100.0
    per_year = exe.groupby(exe.index.year)["net_pnl_nc"].sum().to_dict()
    return {
        "n_trades": int(len(exe)),
        "n_short": int((exe["signal"] == -1).sum()),
        "n_long": int((exe["signal"] == 1).sum()),
        "nc_pnl": nc_pnl,
        "daily_sharpe_nc": daily_sharpe,
        "trade_sharpe_nc": trade_sharpe,
        "mdd_nc": mdd,
        "mdd_pct": mdd_pct,
        "per_year": {int(k): float(v) for k, v in per_year.items()},
        "per_year_min": float(min(per_year.values())) if per_year else 0.0,
    }


def _fmt_money(v: float) -> str:
    return f"${v:,.0f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--param", required=True)
    parser.add_argument("--baseline", required=True, type=float)
    parser.add_argument("--treatment", required=True, type=float)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--date", default=None)
    parser.add_argument("--sharpe-drop-max", type=float, default=0.50,
                        help="G2: max OOS daily Sharpe drop tolerated (default 0.50; "
                             "cycle-12 used 0.50, escalating sweeps should tighten)")
    parser.add_argument("--mdd-worsen-max-pct", type=float, default=2.0,
                        help="G3: max OOS MDD worsening in pp tolerated (default 2.0pp = $2K on $100K NC ref)")
    parser.add_argument("--require-same-sign-positive", action="store_true",
                        default=True,
                        help="G5 strict mode: require both TRAIN and OOS dPnL > 0 "
                             "(default; cycle-12 template)")
    args = parser.parse_args()

    tag = args.tag or args.param
    date = args.date or pd.Timestamp.utcnow().strftime("%Y_%m_%d")
    out_report = REPO_ROOT / f"vol_arb/output/STALE_DEFENSE_{tag}_WALKFORWARD_{date}.md"

    print("=" * 80)
    print(f"  WALK-FORWARD: {args.param} baseline={args.baseline} vs treatment={args.treatment}")
    print(f"  TRAIN years: {sorted(TRAIN_YEARS)}")
    print(f"  OOS   years: {sorted(OOS_YEARS)}")
    print(f"  Gates:")
    print(f"    G1: OOS dPnL > 0")
    print(f"    G2: OOS daily Sharpe drop <= {args.sharpe_drop_max}")
    print(f"    G3: OOS MDD worsen <= {args.mdd_worsen_max_pct:.2f}pp")
    print(f"    G4: OOS per-year strict 2/2 non-negative")
    print(f"    G5: TRAIN + OOS dPnL same sign (both positive)")
    print("=" * 80)

    t0 = time.time()
    logger.info("Running V2 with %s=%s (baseline) ...", args.param, args.baseline)
    tl_base = _run(args.param, args.baseline)
    logger.info("Running V2 with %s=%s (treatment) ...", args.param, args.treatment)
    tl_trt = _run(args.param, args.treatment)
    logger.info("Both runs in %.1fs", time.time() - t0)

    base_train = _metrics_on_slice(tl_base, TRAIN_YEARS)
    trt_train  = _metrics_on_slice(tl_trt,  TRAIN_YEARS)
    base_oos   = _metrics_on_slice(tl_base, OOS_YEARS)
    trt_oos    = _metrics_on_slice(tl_trt,  OOS_YEARS)

    def _delta(b: dict, t: dict) -> dict:
        return {
            "n_trades":        t["n_trades"] - b["n_trades"],
            "n_short":         t["n_short"]  - b["n_short"],
            "n_long":          t["n_long"]   - b["n_long"],
            "nc_pnl":          t["nc_pnl"]   - b["nc_pnl"],
            "daily_sharpe_nc": t["daily_sharpe_nc"] - b["daily_sharpe_nc"],
            "trade_sharpe_nc": t["trade_sharpe_nc"] - b["trade_sharpe_nc"],
            "mdd_nc":          t["mdd_nc"]   - b["mdd_nc"],
            "mdd_pct":         t["mdd_pct"]  - b["mdd_pct"],
            "per_year_min":    t["per_year_min"] - b["per_year_min"],
        }

    dtrain = _delta(base_train, trt_train)
    doos = _delta(base_oos, trt_oos)

    g1 = doos["nc_pnl"] > 0
    g2 = doos["daily_sharpe_nc"] >= -args.sharpe_drop_max
    g3 = doos["mdd_pct"] >= -args.mdd_worsen_max_pct
    g4 = (
        all(v >= 0 for v in trt_oos["per_year"].values())
        and len(trt_oos["per_year"]) == len(OOS_YEARS)
    )
    if args.require_same_sign_positive:
        g5 = (dtrain["nc_pnl"] > 0) and (doos["nc_pnl"] > 0)
    else:
        g5 = (dtrain["nc_pnl"] * doos["nc_pnl"]) > 0
    all_pass = g1 and g2 and g3 and g4 and g5

    extra_trades_train = dtrain["n_trades"]
    extra_trades_oos = doos["n_trades"]
    avg_per_extra_train = dtrain["nc_pnl"] / max(extra_trades_train, 1)
    avg_per_extra_oos = doos["nc_pnl"] / max(extra_trades_oos, 1)

    # ── Print human-readable summary ──
    def _print_slice(name: str, b: dict, t: dict, d: dict):
        print()
        print("=" * 80)
        print(f"  {name}")
        print("=" * 80)
        b_label = f"{args.param}={args.baseline}"
        t_label = f"{args.param}={args.treatment}"
        print(f"  {'metric':<26} {b_label:>16} {t_label:>16} {'delta':>16}")
        print(f"  {'-'*26} {'-'*16} {'-'*16} {'-'*16}")
        for k, label in [
            ("nc_pnl", "NC PnL"), ("daily_sharpe_nc", "Daily Sharpe NC"),
            ("trade_sharpe_nc", "Trade Sharpe NC"), ("mdd_pct", "Max DD %"),
            ("n_trades", "Trades"), ("n_short", "Shorts"), ("n_long", "Longs"),
            ("per_year_min", "Per-year min"),
        ]:
            bv, tv, dv = b.get(k, 0), t.get(k, 0), d.get(k, 0)
            if isinstance(bv, float) and k in ("daily_sharpe_nc", "trade_sharpe_nc"):
                print(f"  {label:<26} {bv:>16.2f} {tv:>16.2f} {dv:>+16.2f}")
            elif k == "mdd_pct":
                print(f"  {label:<26} {bv:>15.2f}% {tv:>15.2f}% {dv:>+15.2f}pp")
            elif isinstance(bv, float) and ("pnl" in k or "year" in k):
                print(f"  {label:<26} {_fmt_money(bv):>16} {_fmt_money(tv):>16} "
                      f"{('+' if dv>=0 else '')+_fmt_money(dv):>16}")
            else:
                print(f"  {label:<26} {bv:>16,} {tv:>16,} "
                      f"{('+' if dv>=0 else '')+f'{dv:,}':>16}")

    _print_slice("TRAIN slice (2022-2024)", base_train, trt_train, dtrain)
    _print_slice("OOS slice (2025-2026)", base_oos, trt_oos, doos)

    print()
    print("=" * 80)
    print("  Per-year NC PnL")
    print("=" * 80)
    all_years = sorted(
        set(base_train["per_year"]) | set(trt_train["per_year"]) |
        set(base_oos["per_year"]) | set(trt_oos["per_year"])
    )
    print(f"  {'year':<6} {'split':<6} "
          f"{f'{args.param}={args.baseline}':>18} "
          f"{f'{args.param}={args.treatment}':>18} {'delta':>16}")
    for y in all_years:
        split = "TRAIN" if y in TRAIN_YEARS else "OOS"
        b_y = (base_train["per_year"].get(y) or base_oos["per_year"].get(y) or 0.0)
        t_y = (trt_train["per_year"].get(y) or trt_oos["per_year"].get(y) or 0.0)
        d_y = t_y - b_y
        print(f"  {y:<6} {split:<6} {_fmt_money(b_y):>18} {_fmt_money(t_y):>18} "
              f"{('+' if d_y >= 0 else '') + _fmt_money(d_y):>16}")

    print()
    print("=" * 80)
    print("  Marginal-trade quality (treatment - baseline)")
    print("=" * 80)
    print(f"  TRAIN: {extra_trades_train:+,} extra trades, "
          f"{('+' if dtrain['nc_pnl']>=0 else '')+_fmt_money(dtrain['nc_pnl'])} PnL, "
          f"avg {_fmt_money(avg_per_extra_train)} / extra trade")
    print(f"  OOS:   {extra_trades_oos:+,} extra trades, "
          f"{('+' if doos['nc_pnl']>=0 else '')+_fmt_money(doos['nc_pnl'])} PnL, "
          f"avg {_fmt_money(avg_per_extra_oos)} / extra trade")

    print()
    print("=" * 80)
    print("  PRE-COMMITTED GATE EVALUATION")
    print("=" * 80)
    gates = [
        ("G1: OOS dPnL > 0",
         f"OOS dPnL = {('+' if doos['nc_pnl']>=0 else '')+_fmt_money(doos['nc_pnl'])}",
         g1),
        (f"G2: OOS daily Sharpe drop <= {args.sharpe_drop_max}",
         f"OOS dSharpe = {doos['daily_sharpe_nc']:+.2f}",
         g2),
        (f"G3: OOS MDD worsen <= {args.mdd_worsen_max_pct:.2f}pp",
         f"OOS dMDD = {doos['mdd_pct']:+.2f}pp (baseline {base_oos['mdd_pct']:.2f}%, "
         f"treatment {trt_oos['mdd_pct']:.2f}%)",
         g3),
        ("G4: OOS per-year strict 2/2 non-negative",
         f"OOS per-year: {trt_oos['per_year']}",
         g4),
        ("G5: TRAIN + OOS dPnL same sign (both positive)",
         f"TRAIN dPnL = {('+' if dtrain['nc_pnl']>=0 else '')+_fmt_money(dtrain['nc_pnl'])}, "
         f"OOS dPnL = {('+' if doos['nc_pnl']>=0 else '')+_fmt_money(doos['nc_pnl'])}",
         g5),
    ]
    for name, detail, status in gates:
        marker = "PASS" if status else "FAIL"
        print(f"  [{marker}] {name}")
        print(f"          {detail}")
    print()
    if all_pass:
        print(f"  VERDICT: ALL GATES PASS — {args.param}={args.treatment} is the new default.")
    else:
        print(f"  VERDICT: AT LEAST ONE GATE FAILED — keep {args.param}={args.baseline}.")

    # ── Write markdown report ──
    lines = [f"# Walk-forward: `{args.param}` {args.baseline} vs {args.treatment}", ""]
    lines.append(f"Pre-committed test ({date}).")
    lines.append("")
    lines.append(f"- Parameter:  `{args.param}`")
    lines.append(f"- Baseline:   {args.baseline}")
    lines.append(f"- Treatment:  {args.treatment}")
    lines.append(f"- TRAIN: {sorted(TRAIN_YEARS)}")
    lines.append(f"- OOS:   {sorted(OOS_YEARS)}")
    lines.append("")
    lines.append("## Pre-committed gates")
    lines.append("")
    for name, detail, status in gates:
        marker = "**PASS**" if status else "**FAIL**"
        lines.append(f"- {marker} {name}")
        lines.append(f"  - {detail}")
    lines.append("")
    lines.append(f"## Verdict: {'**SHIP** (all 5 gates pass)' if all_pass else '**REJECT** (at least one gate failed)'}")
    lines.append("")

    for name_label, b, t, d in (
        ("TRAIN slice (2022-2024)", base_train, trt_train, dtrain),
        ("OOS slice (2025-2026)",   base_oos,   trt_oos,   doos),
    ):
        lines.append(f"## {name_label}")
        lines.append("")
        lines.append(f"| metric | {args.param}={args.baseline} | "
                     f"{args.param}={args.treatment} | delta |")
        lines.append("|---|---:|---:|---:|")
        for k, label in [
            ("nc_pnl", "NC PnL"), ("daily_sharpe_nc", "Daily Sharpe NC"),
            ("trade_sharpe_nc", "Trade Sharpe NC"), ("mdd_pct", "Max DD %"),
            ("n_trades", "Trades"), ("n_short", "Shorts"), ("n_long", "Longs"),
            ("per_year_min", "Per-year min"),
        ]:
            bv, tv, dv = b.get(k, 0), t.get(k, 0), d.get(k, 0)
            if k in {"daily_sharpe_nc", "trade_sharpe_nc"}:
                lines.append(f"| {label} | {bv:.2f} | {tv:.2f} | {dv:+.2f} |")
            elif k == "mdd_pct":
                lines.append(f"| {label} | {bv:.2f}% | {tv:.2f}% | {dv:+.2f}pp |")
            elif "pnl" in k or "year" in k:
                lines.append(f"| {label} | {_fmt_money(bv)} | {_fmt_money(tv)} | "
                             f"{('+' if dv>=0 else '')+_fmt_money(dv)} |")
            else:
                lines.append(f"| {label} | {bv:,} | {tv:,} | "
                             f"{('+' if dv>=0 else '')+f'{dv:,}'} |")
        lines.append("")

    lines.append("## Per-year NC PnL")
    lines.append("")
    lines.append(f"| year | split | {args.param}={args.baseline} | "
                 f"{args.param}={args.treatment} | delta |")
    lines.append("|---|---|---:|---:|---:|")
    for y in all_years:
        split = "TRAIN" if y in TRAIN_YEARS else "OOS"
        b_y = (base_train["per_year"].get(y) or base_oos["per_year"].get(y) or 0.0)
        t_y = (trt_train["per_year"].get(y) or trt_oos["per_year"].get(y) or 0.0)
        d_y = t_y - b_y
        lines.append(f"| {y} | {split} | {_fmt_money(b_y)} | {_fmt_money(t_y)} | "
                     f"{('+' if d_y>=0 else '')+_fmt_money(d_y)} |")
    lines.append("")

    lines.append("## Marginal-trade quality (treatment - baseline)")
    lines.append("")
    lines.append("| split | extra trades | extra short | extra long | extra PnL | avg per extra trade |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    lines.append(f"| TRAIN | {extra_trades_train:+,} | {dtrain['n_short']:+,} | "
                 f"{dtrain['n_long']:+,} | "
                 f"{('+' if dtrain['nc_pnl']>=0 else '')+_fmt_money(dtrain['nc_pnl'])} | "
                 f"{_fmt_money(avg_per_extra_train)} |")
    lines.append(f"| OOS | {extra_trades_oos:+,} | {doos['n_short']:+,} | "
                 f"{doos['n_long']:+,} | "
                 f"{('+' if doos['nc_pnl']>=0 else '')+_fmt_money(doos['nc_pnl'])} | "
                 f"{_fmt_money(avg_per_extra_oos)} |")
    lines.append("")
    lines.append(f"Reference baseline avg PnL/trade: TRAIN "
                 f"{_fmt_money(base_train['nc_pnl']/max(base_train['n_trades'],1))}, "
                 f"OOS {_fmt_money(base_oos['nc_pnl']/max(base_oos['n_trades'],1))}.")
    lines.append("")

    out_report.parent.mkdir(parents=True, exist_ok=True)
    out_report.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote: %s", out_report)
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
