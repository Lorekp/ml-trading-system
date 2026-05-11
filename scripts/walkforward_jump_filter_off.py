"""Walk-forward test: does removing `max_jump_fraction_for_shorts` hold OOS?

Pre-committed test (2026-05-11) to avoid in-sample cherry-pick after the
full-sample sweep showed jf=1.00 dominates jf=0.15 unanimously on
every year. Per ``feedback_no_cherry_pick.md``, accept a structural
result only when:

  (a) TRAIN years (2022-2024) confirm the win.
  (b) OOS years (2025-2026) confirm the win.
  (c) Per-year stability is strict.
  (d) Train-to-OOS direction is consistent.

PRE-COMMITTED GATES (locked BEFORE running):

  G1 — OOS PnL increase: jf=1.00 OOS NC PnL > jf=0.15 OOS NC PnL.
  G2 — OOS daily Sharpe drop ≤ 0.5.
  G3 — OOS MDD worsens ≤ $2,000.
  G4 — OOS per-year strict 2/2 (both 2025 and 2026 non-negative).
  G5 — Train-to-OOS sign consistency: dPnL is positive in BOTH
       TRAIN and OOS (i.e., the win replicates).

Reject if any gate fails. Ship only if ALL five gates pass.

Plus a TRADE COUNT delta — how many extra shorts fire, and how
profitable are the marginal trades on average.

Usage:
    PYTHONPATH=. python scripts/walkforward_jump_filter_off.py
"""
from __future__ import annotations

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
logger = logging.getLogger("walkforward_jf")

OUT_REPORT = REPO_ROOT / "vol_arb/output/JUMP_FILTER_WALKFORWARD_2026_05_11.md"

# Pre-committed splits
TRAIN_YEARS = {2022, 2023, 2024}
OOS_YEARS = {2025, 2026}

# Pre-committed gates
GATE_OOS_SHARPE_DROP_MAX = 0.50    # daily Sharpe drop tolerance
GATE_OOS_MDD_WORSEN_MAX = 2_000.0  # MDD worsening tolerance ($)


def _run(jf_cap: float) -> pd.DataFrame:
    """Run the full V2 backtester with the given jump-filter cap."""
    cfg = (BacktestConfig()
           .with_btc_production_filters_v2()
           .with_realistic_costs())
    cfg.max_jump_fraction_for_shorts = jf_cap
    _, tl = VolArbBacktester(cfg).run_backtest()
    return tl


def _metrics_on_slice(tl: pd.DataFrame, years: set[int]) -> dict:
    """Compute headline metrics on a slice of the trade log."""
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
    if daily.std(ddof=1) > 0:
        daily_sharpe = float(daily.mean() / daily.std(ddof=1) * np.sqrt(365.25))
    else:
        daily_sharpe = float("nan")
    per_trade = exe["net_pnl_nc"]
    if per_trade.std(ddof=1) > 0:
        n_years = (exe.index.max() - exe.index.min()).total_seconds() / (365.25 * 24 * 3600)
        trade_sharpe = float(
            per_trade.mean() / per_trade.std(ddof=1)
            * np.sqrt(len(per_trade) / max(n_years, 1e-9))
        )
    else:
        trade_sharpe = float("nan")
    # MDD on the YEAR-SLICE only (running from initial_equity on the slice)
    eq = exe["net_pnl_nc"].cumsum()
    mdd = float((eq - eq.cummax()).min())
    per_year = exe.groupby(exe.index.year)["net_pnl_nc"].sum().to_dict()
    return {
        "n_trades": int(len(exe)),
        "n_short": int((exe["signal"] == -1).sum()),
        "n_long": int((exe["signal"] == 1).sum()),
        "nc_pnl": nc_pnl,
        "daily_sharpe_nc": daily_sharpe,
        "trade_sharpe_nc": trade_sharpe,
        "mdd_nc": mdd,
        "per_year": {int(k): float(v) for k, v in per_year.items()},
        "per_year_min": float(min(per_year.values())) if per_year else 0.0,
    }


def _format_money(v: float) -> str:
    return f"${v:,.0f}"


def main() -> int:
    print("=" * 80)
    print("  WALK-FORWARD TEST: max_jump_fraction_for_shorts 0.15 vs 1.00 (OFF)")
    print("=" * 80)
    print(f"  TRAIN years: {sorted(TRAIN_YEARS)}")
    print(f"  OOS   years: {sorted(OOS_YEARS)}")
    print(f"  Pre-committed gates:")
    print(f"    G1: OOS dPnL > 0")
    print(f"    G2: OOS daily Sharpe drop <= {GATE_OOS_SHARPE_DROP_MAX}")
    print(f"    G3: OOS MDD worsen <= ${GATE_OOS_MDD_WORSEN_MAX:,.0f}")
    print(f"    G4: OOS per-year strict 2/2 non-negative")
    print(f"    G5: Train + OOS dPnL same sign (both positive)")
    print("=" * 80)

    t0 = time.time()
    logger.info("Running V2 with jf=0.15 (baseline, current production) ...")
    tl_base = _run(0.15)
    logger.info("Running V2 with jf=1.00 (filter OFF, treatment) ...")
    tl_trt = _run(1.00)
    logger.info("Both runs completed in %.1fs", time.time() - t0)

    # ── Compute metrics on TRAIN + OOS slices ──
    base_train = _metrics_on_slice(tl_base, TRAIN_YEARS)
    trt_train  = _metrics_on_slice(tl_trt,  TRAIN_YEARS)
    base_oos   = _metrics_on_slice(tl_base, OOS_YEARS)
    trt_oos    = _metrics_on_slice(tl_trt,  OOS_YEARS)

    # ── Deltas ──
    def _delta(b: dict, t: dict) -> dict:
        return {
            "n_trades":        t["n_trades"] - b["n_trades"],
            "n_short":         t["n_short"]  - b["n_short"],
            "nc_pnl":          t["nc_pnl"]   - b["nc_pnl"],
            "daily_sharpe_nc": t["daily_sharpe_nc"] - b["daily_sharpe_nc"],
            "trade_sharpe_nc": t["trade_sharpe_nc"] - b["trade_sharpe_nc"],
            "mdd_nc":          t["mdd_nc"]   - b["mdd_nc"],
            "per_year_min":    t["per_year_min"] - b["per_year_min"],
        }

    dtrain = _delta(base_train, trt_train)
    doos   = _delta(base_oos,   trt_oos)

    # ── Pre-committed gate evaluation ──
    g1 = doos["nc_pnl"] > 0
    g2 = doos["daily_sharpe_nc"] >= -GATE_OOS_SHARPE_DROP_MAX
    g3 = doos["mdd_nc"] >= -GATE_OOS_MDD_WORSEN_MAX
    g4 = all(v >= 0 for v in trt_oos["per_year"].values()) and len(trt_oos["per_year"]) == len(OOS_YEARS)
    g5 = (dtrain["nc_pnl"] > 0) and (doos["nc_pnl"] > 0)
    all_pass = g1 and g2 and g3 and g4 and g5

    # Marginal-trade quality: the EXTRA shorts in trt vs base
    extra_shorts_train = dtrain["n_short"]
    extra_shorts_oos   = doos["n_short"]
    extra_pnl_train    = dtrain["nc_pnl"]
    extra_pnl_oos      = doos["nc_pnl"]
    avg_per_extra_train = extra_pnl_train / max(extra_shorts_train, 1)
    avg_per_extra_oos   = extra_pnl_oos   / max(extra_shorts_oos, 1)

    # ── Print summary ──
    print()
    print("=" * 80)
    print("  TRAIN slice (2022-2024)")
    print("=" * 80)
    print(f"  {'metric':<26} {'jf=0.15':>14} {'jf=1.00':>14} {'delta':>14}")
    print(f"  {'-'*26} {'-'*14} {'-'*14} {'-'*14}")
    for k, label in [
        ("nc_pnl", "NC PnL"),
        ("daily_sharpe_nc", "Daily Sharpe NC"),
        ("trade_sharpe_nc", "Trade Sharpe NC"),
        ("mdd_nc", "Max DD NC"),
        ("n_trades", "Trades"),
        ("n_short", "Shorts"),
        ("per_year_min", "Per-year min"),
    ]:
        b, t = base_train[k], trt_train[k]
        d = dtrain[k]
        if isinstance(b, float) and (k == "daily_sharpe_nc" or k == "trade_sharpe_nc"):
            print(f"  {label:<26} {b:>14.2f} {t:>14.2f} {d:>+14.2f}")
        elif isinstance(b, float) and ("pnl" in k or "mdd" in k or "year" in k):
            print(f"  {label:<26} {_format_money(b):>14} {_format_money(t):>14} {('+' if d>=0 else '')+_format_money(d):>14}")
        else:
            print(f"  {label:<26} {b:>14,} {t:>14,} {('+' if d>=0 else '')+f'{d:,}':>14}")

    print()
    print("=" * 80)
    print("  OOS slice (2025-2026)")
    print("=" * 80)
    print(f"  {'metric':<26} {'jf=0.15':>14} {'jf=1.00':>14} {'delta':>14}")
    print(f"  {'-'*26} {'-'*14} {'-'*14} {'-'*14}")
    for k, label in [
        ("nc_pnl", "NC PnL"),
        ("daily_sharpe_nc", "Daily Sharpe NC"),
        ("trade_sharpe_nc", "Trade Sharpe NC"),
        ("mdd_nc", "Max DD NC"),
        ("n_trades", "Trades"),
        ("n_short", "Shorts"),
        ("per_year_min", "Per-year min"),
    ]:
        b, t = base_oos[k], trt_oos[k]
        d = doos[k]
        if isinstance(b, float) and (k == "daily_sharpe_nc" or k == "trade_sharpe_nc"):
            print(f"  {label:<26} {b:>14.2f} {t:>14.2f} {d:>+14.2f}")
        elif isinstance(b, float) and ("pnl" in k or "mdd" in k or "year" in k):
            print(f"  {label:<26} {_format_money(b):>14} {_format_money(t):>14} {('+' if d>=0 else '')+_format_money(d):>14}")
        else:
            print(f"  {label:<26} {b:>14,} {t:>14,} {('+' if d>=0 else '')+f'{d:,}':>14}")

    print()
    print("=" * 80)
    print("  Per-year NC PnL")
    print("=" * 80)
    all_years = sorted(set(base_train["per_year"]) | set(trt_train["per_year"]) | set(base_oos["per_year"]) | set(trt_oos["per_year"]))
    print(f"  {'year':<6} {'split':<6} {'jf=0.15':>14} {'jf=1.00':>14} {'delta':>14}")
    print(f"  {'-'*6} {'-'*6} {'-'*14} {'-'*14} {'-'*14}")
    for y in all_years:
        split = "TRAIN" if y in TRAIN_YEARS else "OOS"
        b_y = (base_train["per_year"].get(y) or base_oos["per_year"].get(y) or 0.0)
        t_y = (trt_train["per_year"].get(y) or trt_oos["per_year"].get(y) or 0.0)
        d_y = t_y - b_y
        print(f"  {y:<6} {split:<6} {_format_money(b_y):>14} {_format_money(t_y):>14} "
              f"{('+' if d_y >= 0 else '') + _format_money(d_y):>14}")

    print()
    print("=" * 80)
    print("  Marginal-trade quality (extra shorts fired at jf=1.00 vs jf=0.15)")
    print("=" * 80)
    print(f"  TRAIN: +{extra_shorts_train:,} extra shorts, "
          f"+{_format_money(extra_pnl_train)} PnL, "
          f"avg {_format_money(avg_per_extra_train)} / extra trade")
    print(f"  OOS:   +{extra_shorts_oos:,} extra shorts, "
          f"+{_format_money(extra_pnl_oos)} PnL, "
          f"avg {_format_money(avg_per_extra_oos)} / extra trade")
    print()
    print(f"  (Reference: existing baseline avg PnL/trade TRAIN "
          f"{_format_money(base_train['nc_pnl']/max(base_train['n_trades'],1))}, "
          f"OOS {_format_money(base_oos['nc_pnl']/max(base_oos['n_trades'],1))})")

    print()
    print("=" * 80)
    print("  PRE-COMMITTED GATE EVALUATION")
    print("=" * 80)
    gates = [
        ("G1: OOS dPnL > 0",
         f"OOS dPnL = {('+' if doos['nc_pnl']>=0 else '')+_format_money(doos['nc_pnl'])}",
         g1),
        (f"G2: OOS daily Sharpe drop <= {GATE_OOS_SHARPE_DROP_MAX}",
         f"OOS dSharpe = {doos['daily_sharpe_nc']:+.2f}",
         g2),
        (f"G3: OOS MDD worsen <= ${GATE_OOS_MDD_WORSEN_MAX:,.0f}",
         f"OOS dMDD = {('+' if doos['mdd_nc']>=0 else '')+_format_money(doos['mdd_nc'])}",
         g3),
        ("G4: OOS per-year strict 2/2 non-negative",
         f"OOS per-year: {trt_oos['per_year']}",
         g4),
        ("G5: TRAIN + OOS dPnL same sign (both positive)",
         f"TRAIN dPnL = {('+' if dtrain['nc_pnl']>=0 else '')+_format_money(dtrain['nc_pnl'])}, "
         f"OOS dPnL = {('+' if doos['nc_pnl']>=0 else '')+_format_money(doos['nc_pnl'])}",
         g5),
    ]
    for name, detail, status in gates:
        marker = "PASS" if status else "FAIL"
        print(f"  [{marker}] {name}")
        print(f"          {detail}")

    print()
    if all_pass:
        print("  VERDICT: ALL GATES PASS — jf=1.00 (filter OFF) is the new default.")
    else:
        print("  VERDICT: AT LEAST ONE GATE FAILED — keep jf=0.15 (production default).")

    # ── Write markdown report ──
    lines = ["# Jump-filter walk-forward: jf=0.15 vs jf=1.00 (OFF)", ""]
    lines.append("Pre-committed test (2026-05-11). Sweep at "
                 "`vol_arb/output/JUMP_FRACTION_SWEEP_2026_05_11.md` "
                 "showed jf=1.00 dominates jf=0.15 unanimously on every year of "
                 "the full sample. This walk-forward verifies the result holds "
                 "OOS without in-sample contamination.")
    lines.append("")
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

    lines.append("## TRAIN slice (2022-2024)")
    lines.append("")
    lines.append("| metric | jf=0.15 | jf=1.00 | delta |")
    lines.append("|---|---:|---:|---:|")
    for k, label in [
        ("nc_pnl", "NC PnL"), ("daily_sharpe_nc", "Daily Sharpe NC"),
        ("trade_sharpe_nc", "Trade Sharpe NC"), ("mdd_nc", "Max DD NC"),
        ("n_trades", "Trades"), ("n_short", "Shorts"),
        ("per_year_min", "Per-year min"),
    ]:
        b, t = base_train[k], trt_train[k]
        d = dtrain[k]
        if k in {"daily_sharpe_nc", "trade_sharpe_nc"}:
            lines.append(f"| {label} | {b:.2f} | {t:.2f} | {d:+.2f} |")
        elif "pnl" in k or "mdd" in k or "year" in k:
            lines.append(f"| {label} | {_format_money(b)} | {_format_money(t)} | {('+' if d>=0 else '')+_format_money(d)} |")
        else:
            lines.append(f"| {label} | {b:,} | {t:,} | {('+' if d>=0 else '')+f'{d:,}'} |")
    lines.append("")

    lines.append("## OOS slice (2025-2026)")
    lines.append("")
    lines.append("| metric | jf=0.15 | jf=1.00 | delta |")
    lines.append("|---|---:|---:|---:|")
    for k, label in [
        ("nc_pnl", "NC PnL"), ("daily_sharpe_nc", "Daily Sharpe NC"),
        ("trade_sharpe_nc", "Trade Sharpe NC"), ("mdd_nc", "Max DD NC"),
        ("n_trades", "Trades"), ("n_short", "Shorts"),
        ("per_year_min", "Per-year min"),
    ]:
        b, t = base_oos[k], trt_oos[k]
        d = doos[k]
        if k in {"daily_sharpe_nc", "trade_sharpe_nc"}:
            lines.append(f"| {label} | {b:.2f} | {t:.2f} | {d:+.2f} |")
        elif "pnl" in k or "mdd" in k or "year" in k:
            lines.append(f"| {label} | {_format_money(b)} | {_format_money(t)} | {('+' if d>=0 else '')+_format_money(d)} |")
        else:
            lines.append(f"| {label} | {b:,} | {t:,} | {('+' if d>=0 else '')+f'{d:,}'} |")
    lines.append("")

    lines.append("## Per-year NC PnL")
    lines.append("")
    lines.append("| year | split | jf=0.15 | jf=1.00 | delta |")
    lines.append("|---|---|---:|---:|---:|")
    for y in all_years:
        split = "TRAIN" if y in TRAIN_YEARS else "OOS"
        b_y = (base_train["per_year"].get(y) or base_oos["per_year"].get(y) or 0.0)
        t_y = (trt_train["per_year"].get(y) or trt_oos["per_year"].get(y) or 0.0)
        d_y = t_y - b_y
        lines.append(f"| {y} | {split} | {_format_money(b_y)} | {_format_money(t_y)} | "
                     f"{('+' if d_y >= 0 else '') + _format_money(d_y)} |")
    lines.append("")

    lines.append("## Marginal-trade quality (extra shorts at jf=1.00 vs jf=0.15)")
    lines.append("")
    lines.append("| split | extra shorts | extra PnL | avg per extra trade |")
    lines.append("|---|---:|---:|---:|")
    lines.append(f"| TRAIN | +{extra_shorts_train:,} | "
                 f"+{_format_money(extra_pnl_train)} | "
                 f"{_format_money(avg_per_extra_train)} |")
    lines.append(f"| OOS   | +{extra_shorts_oos:,} | "
                 f"+{_format_money(extra_pnl_oos)} | "
                 f"{_format_money(avg_per_extra_oos)} |")
    lines.append("")
    lines.append(f"Reference: existing baseline avg PnL/trade TRAIN "
                 f"{_format_money(base_train['nc_pnl']/max(base_train['n_trades'],1))}, "
                 f"OOS {_format_money(base_oos['nc_pnl']/max(base_oos['n_trades'],1))}.")
    lines.append("")
    lines.append("If the marginal-trade avg PnL is **lower** than the baseline "
                 "per-trade avg, the extra trades are the noisier tail of the "
                 "edge distribution — still positive but adding variance per "
                 "$ of PnL. If **higher**, the filter was killing some of "
                 "the higher-conviction shorts.")
    lines.append("")
    OUT_REPORT.parent.mkdir(parents=True, exist_ok=True)
    OUT_REPORT.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote: %s", OUT_REPORT)

    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
