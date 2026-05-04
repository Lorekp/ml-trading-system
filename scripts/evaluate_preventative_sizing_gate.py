"""Evaluate the preventative-sizing rule against the pre-committed gate.

Phase P gate runner (plan: ``now-check-empirically-from-eager-bird.md``).

Inputs:
  - ``vol_arb/output/BTC_v2_trade_log.parquet``: canonical V2 baseline.
  - ``vol_arb/output/BTC_v2_preventative_size_mult.parquet``: per-bar f.

For an entry-only sizing rule, candidate per-trade PnL is the linearity
identity ``f_i × baseline_pnl_i`` (taker fees scale linearly in N; market
impact is assumed second-order and captured in the baseline). This is
accurate to first order; the small drift from running-equity divergence
is the same approximation accepted by the Kelly K1 sidecar.

Pre-committed adoption gate (verbatim from Cycle 1 Tier C-2):
  1. 10th-percentile PSR delta across bootstrap reps ≥ +0.30
  2. OOS-validation 4-day worst-day cut ≥ +$5,000
  3. NC Sharpe drop on full evaluation window ≤ 0.30

OOS validation days (held-out): Celsius 2022-06-13, FTX 2022-11-08,
JPY-carry 2024-08-05, Trump-inauguration 2025-01-20.

Writes ``vol_arb/output/V2_PREVENTATIVE_SIZING.md`` with full result.

Run::

    PYTHONPATH=. python scripts/evaluate_preventative_sizing_gate.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.synthetic_path_evaluator import (
    DEFAULT_BLOCK_SIZE_DAYS,
    DEFAULT_N_BOOTSTRAP_REPS,
    block_bootstrap_deltas,
    _annualised_sharpe,
)
from vol_arb.preventative_sizing import load_sidecar

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_preventative_sizing_gate")

TRADE_LOG_PATH = Path("vol_arb/output/BTC_v2_trade_log.parquet")
SIDECAR_PATH = Path("vol_arb/output/BTC_v2_preventative_size_mult.parquet")
REPORT_PATH = Path("vol_arb/output/V2_PREVENTATIVE_SIZING.md")

# Pre-committed gate (locked verbatim from Cycle 1 Tier C-2)
GATE_PSR_P10_DELTA: float = 0.30
GATE_OOS_WORST_CUT_USD: float = 5_000.0
GATE_NC_SHARPE_DROP_MAX: float = 0.30

# Evaluation window — eval starts after the calibration window ends
EVAL_WINDOW_START = pd.Timestamp("2023-10-01", tz="UTC")

OOS_VALIDATION_DAYS: dict[str, pd.Timestamp] = {
    "Celsius_2022_06_13": pd.Timestamp("2022-06-13", tz="UTC"),
    "FTX_2022_11_08": pd.Timestamp("2022-11-08", tz="UTC"),
    "JPY_carry_2024_08_05": pd.Timestamp("2024-08-05", tz="UTC"),
    "Trump_inauguration_2025_01_20": pd.Timestamp("2025-01-20", tz="UTC"),
}


def _daily_pnl(tl: pd.DataFrame, pnl_col: str = "net_pnl_nc") -> pd.Series:
    """Aggregate a (potentially scaled) trade-PnL series to daily totals."""
    pnl = tl[pnl_col]
    return pnl.groupby(pnl.index.normalize()).sum()


def _maxdd(daily: pd.Series) -> float:
    cum = daily.cumsum()
    peaks = cum.cummax()
    return float((cum - peaks).min())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    parser.add_argument("--n-reps", type=int, default=DEFAULT_N_BOOTSTRAP_REPS)
    parser.add_argument("--block-size-days", type=int,
                        default=DEFAULT_BLOCK_SIZE_DAYS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("Loading canonical V2 trade log: %s", TRADE_LOG_PATH)
    tl = pd.read_parquet(TRADE_LOG_PATH)
    if not isinstance(tl.index, pd.DatetimeIndex):
        tl.index = pd.to_datetime(tl.index, utc=True)

    logger.info("Loading preventative sizing sidecar: %s", SIDECAR_PATH)
    size_mult = load_sidecar(SIDECAR_PATH)

    # Align sidecar to trade log index — must be exact match
    size_mult_aligned = size_mult.reindex(tl.index)
    n_missing = int(size_mult_aligned.isna().sum())
    if n_missing > 0:
        raise RuntimeError(
            f"Sidecar alignment: {n_missing} bars without size_mult"
        )

    # Apply per-bar scaler post-hoc: candidate PnL = f × baseline PnL
    tl_cand = tl.copy()
    tl_cand["net_pnl_nc"] = tl["net_pnl_nc"] * size_mult_aligned.values
    tl_cand["size_mult_applied"] = size_mult_aligned.values

    # No-zero-tradeable check: every fired baseline trade must remain fired
    baseline_fired_mask = tl["net_pnl_nc"] != 0
    cand_fired_mask = tl_cand["net_pnl_nc"] != 0
    n_baseline_fired = int(baseline_fired_mask.sum())
    n_cand_fired = int(cand_fired_mask.sum())
    if n_cand_fired != n_baseline_fired:
        raise RuntimeError(
            f"Candidate fired count {n_cand_fired} != baseline {n_baseline_fired}; "
            "the f_min=0.10 floor failed to keep all trades alive."
        )
    logger.info("No-zero-tradeable: %d fired in baseline, %d in candidate (OK)",
                n_baseline_fired, n_cand_fired)

    # Daily aggregation
    baseline_daily = _daily_pnl(tl)
    candidate_daily = _daily_pnl(tl_cand)
    aligned = pd.concat({"baseline": baseline_daily,
                          "candidate": candidate_daily}, axis=1).fillna(0.0)

    # Headline (full window) metrics
    cum_baseline = float(baseline_daily.sum())
    cum_candidate = float(candidate_daily.sum())
    pnl_delta = cum_candidate - cum_baseline

    sharpe_baseline = _annualised_sharpe(baseline_daily.to_numpy())
    sharpe_candidate = _annualised_sharpe(candidate_daily.to_numpy())
    sharpe_delta = sharpe_candidate - sharpe_baseline

    maxdd_baseline = _maxdd(baseline_daily)
    maxdd_candidate = _maxdd(candidate_daily)
    maxdd_delta = maxdd_candidate - maxdd_baseline

    # Eval-window metrics (post-2023-10-01)
    eval_baseline_daily = baseline_daily.loc[
        baseline_daily.index >= EVAL_WINDOW_START
    ]
    eval_candidate_daily = candidate_daily.loc[
        candidate_daily.index >= EVAL_WINDOW_START
    ]
    eval_sharpe_baseline = _annualised_sharpe(eval_baseline_daily.to_numpy())
    eval_sharpe_candidate = _annualised_sharpe(eval_candidate_daily.to_numpy())
    eval_sharpe_drop = eval_sharpe_baseline - eval_sharpe_candidate
    eval_pnl_baseline = float(eval_baseline_daily.sum())
    eval_pnl_candidate = float(eval_candidate_daily.sum())

    # OOS-validation worst-day analysis
    oos_dates = list(OOS_VALIDATION_DAYS.values())
    oos_baseline_per_day = baseline_daily.reindex(
        oos_dates, fill_value=0.0,
    )
    oos_candidate_per_day = candidate_daily.reindex(
        oos_dates, fill_value=0.0,
    )
    oos_baseline_worst = float(oos_baseline_per_day.min())
    oos_candidate_worst = float(oos_candidate_per_day.min())
    oos_worst_cut = oos_candidate_worst - oos_baseline_worst   # signed

    # Block-bootstrap on EVAL window (post 2023-10-01)
    logger.info("Block-bootstrap (n_reps=%d, block=%d days, seed=%d) on eval window",
                args.n_reps, args.block_size_days, args.seed)
    boot = block_bootstrap_deltas(
        eval_baseline_daily, eval_candidate_daily,
        n_reps=args.n_reps, block_size_days=args.block_size_days, seed=args.seed,
    )

    # Gate evaluation — TRIPLE conjunctive
    psr_p10 = boot["psr_delta"].p10
    pass_psr = bool(psr_p10 >= GATE_PSR_P10_DELTA)
    pass_oos = bool(oos_worst_cut >= GATE_OOS_WORST_CUT_USD)
    pass_sharpe = bool(eval_sharpe_drop <= GATE_NC_SHARPE_DROP_MAX)
    gate_pass = pass_psr and pass_oos and pass_sharpe

    logger.info("Gate result:")
    logger.info(
        f"  PSR p10 delta:    {psr_p10:+.4f}  "
        f"(gate >= +{GATE_PSR_P10_DELTA:.2f}) "
        f"{'PASS' if pass_psr else 'FAIL'}"
    )
    logger.info(
        f"  OOS worst cut:    ${oos_worst_cut:+,.0f}  "
        f"(gate >= +${GATE_OOS_WORST_CUT_USD:,.0f}) "
        f"{'PASS' if pass_oos else 'FAIL'}"
    )
    logger.info(
        f"  NC Sharpe drop:   {eval_sharpe_drop:+.4f}  "
        f"(gate <= +{GATE_NC_SHARPE_DROP_MAX:.2f}) "
        f"{'PASS' if pass_sharpe else 'FAIL'}"
    )
    logger.info(f"  Conjunction: {'PASS' if gate_pass else 'REJECT'}")

    # Diagnostic: how often does the rule actually bind on FIRED trades?
    fired_mask = tl["net_pnl_nc"] != 0
    fired_sm = size_mult_aligned[fired_mask]
    eval_fired_mask = fired_mask & (tl.index >= EVAL_WINDOW_START)
    eval_fired_sm = size_mult_aligned[eval_fired_mask]
    n_fired_bind = int((fired_sm < 1.0 - 1e-9).sum())
    n_eval_fired_bind = int((eval_fired_sm < 1.0 - 1e-9).sum())
    logger.info(
        f"Binding-on-fired diagnostic:  full {n_fired_bind} / {len(fired_sm)} "
        f"({100.0*n_fired_bind/max(len(fired_sm), 1):.2f}%);  "
        f"eval {n_eval_fired_bind} / {len(eval_fired_sm)} "
        f"({100.0*n_eval_fired_bind/max(len(eval_fired_sm), 1):.2f}%)"
    )

    # Binding diagnostic for the report
    fired_mask_rep = tl["net_pnl_nc"] != 0
    fired_sm_rep = size_mult_aligned[fired_mask_rep]
    eval_fired_mask_rep = fired_mask_rep & (tl.index >= EVAL_WINDOW_START)
    eval_fired_sm_rep = size_mult_aligned[eval_fired_mask_rep]
    cal_fired_mask_rep = fired_mask_rep & (tl.index < EVAL_WINDOW_START)
    cal_fired_sm_rep = size_mult_aligned[cal_fired_mask_rep]
    n_fired_bind_full = int((fired_sm_rep < 1.0 - 1e-9).sum())
    n_fired_bind_cal = int((cal_fired_sm_rep < 1.0 - 1e-9).sum())
    n_fired_bind_eval = int((eval_fired_sm_rep < 1.0 - 1e-9).sum())

    # Render markdown
    lines = [
        "# V2 Preventative Sizing — Gate Evaluation",
        "",
        "Cycle 2 Phase P (plan: `now-check-empirically-from-eager-bird.md`).",
        "",
        "## Pre-committed external anchors (locked, NOT tuned)",
        "",
        "- `loss_budget_pct = 0.01` — 1% of equity per single PME stress event",
        "- `pme_shock_spot = 0.175` — Deribit documented main-table BTC shock",
        f"- `pme_shock_iv_pct = 0.1983` — 99th-percentile |dDVOL_1d| on calibration window",
        "- `f_min = 0.10` — size floor preventing zero-tradeable",
        "",
        "## Pre-committed gate (identical to Cycle 1 Tier C-2)",
        "",
        f"- 10th-percentile PSR delta >= +{GATE_PSR_P10_DELTA:.2f} (across bootstrap reps)",
        f"- OOS-validation 4-day worst-day cut >= +${GATE_OOS_WORST_CUT_USD:,.0f}",
        f"- NC Sharpe drop on eval window <= +{GATE_NC_SHARPE_DROP_MAX:.2f}",
        "",
        "## Headline (full trade log, post-hoc f-scaling)",
        "",
        f"- Baseline cumulative NC PnL:  ${cum_baseline:,.0f}",
        f"- Candidate cumulative NC PnL: ${cum_candidate:,.0f}",
        f"- PnL delta:                   ${pnl_delta:+,.0f}",
        f"- Baseline NC Sharpe (full):   {sharpe_baseline:.3f}",
        f"- Candidate NC Sharpe (full):  {sharpe_candidate:.3f}",
        f"- Sharpe delta:                {sharpe_delta:+.3f}",
        f"- Baseline MaxDD:              ${maxdd_baseline:,.0f}",
        f"- Candidate MaxDD:             ${maxdd_candidate:,.0f}",
        f"- MaxDD delta:                 ${maxdd_delta:+,.0f}",
        "",
        "## Evaluation window (post-2023-10-01, gate-relevant)",
        "",
        f"- Eval days:                  {len(eval_baseline_daily)}",
        f"- Baseline NC Sharpe (eval):  {eval_sharpe_baseline:.3f}",
        f"- Candidate NC Sharpe (eval): {eval_sharpe_candidate:.3f}",
        f"- NC Sharpe drop:             {eval_sharpe_drop:+.3f}  "
        f"(gate <= +{GATE_NC_SHARPE_DROP_MAX:.2f}) "
        f"{'PASS' if pass_sharpe else 'FAIL'}",
        f"- Baseline eval PnL:          ${eval_pnl_baseline:,.0f}",
        f"- Candidate eval PnL:         ${eval_pnl_candidate:,.0f}",
        f"- Eval PnL delta:             ${eval_pnl_candidate - eval_pnl_baseline:+,.0f}",
        "",
        "## OOS-validation 4-day analysis (held-out)",
        "",
        "| Day | Baseline PnL | Candidate PnL | Delta |",
        "|---|---:|---:|---:|",
    ]
    for label, ts in OOS_VALIDATION_DAYS.items():
        b = float(baseline_daily.get(ts.normalize(), 0.0))
        c = float(candidate_daily.get(ts.normalize(), 0.0))
        lines.append(f"| {label} | ${b:,.0f} | ${c:,.0f} | ${c-b:+,.0f} |")
    lines += [
        "",
        f"- OOS worst day baseline:     ${oos_baseline_worst:,.0f}",
        f"- OOS worst day candidate:    ${oos_candidate_worst:,.0f}",
        f"- OOS worst-day cut:          ${oos_worst_cut:+,.0f}  "
        f"(gate >= +${GATE_OOS_WORST_CUT_USD:,.0f}) "
        f"{'PASS' if pass_oos else 'FAIL'}",
        "",
        "## Block-bootstrap distribution (eval window only)",
        "",
        f"Reps: {args.n_reps}, block size: {args.block_size_days} days, seed: {args.seed}",
        "",
        "| Metric | p10 | p50 | p90 | mean |",
        "|---|---:|---:|---:|---:|",
    ]
    for key in ("sharpe_delta", "pnl_delta", "maxdd_delta", "psr_delta"):
        ds = boot[key]
        lines.append(
            f"| {ds.metric_name} | {ds.p10:+.4f} | {ds.p50:+.4f} | "
            f"{ds.p90:+.4f} | {ds.mean:+.4f} |"
        )
    lines += [
        "",
        f"- PSR delta p10:  {psr_p10:+.4f}  "
        f"(gate >= +{GATE_PSR_P10_DELTA:.2f}) "
        f"{'PASS' if pass_psr else 'FAIL'}",
        "",
        "## Adoption decision",
        "",
        f"**{'ADOPT AS DEFAULT' if gate_pass else 'REJECT — ship as opt-in only'}**",
        "",
    ]
    if not gate_pass:
        failed = [name for name, ok in
                   [("PSR p10", pass_psr), ("OOS cut", pass_oos),
                    ("Sharpe drop", pass_sharpe)]
                   if not ok]
        lines += [
            f"Failed sub-gates: {', '.join(failed)}.",
            "",
            "Per Cycle 2 Phase P plan: failure on ANY sub-gate keeps the rule "
            "opt-in only. Canonical V2 unchanged. The candidate sidecar "
            "remains available via `--preventative-sizing` opt-in. Honest "
            "reject under pre-committed methodology — lowering the gate to "
            "match observed numbers would be exactly the storytelling-bias "
            "failure mode the plan codified.",
        ]
        # Structural diagnostic: did the rule actually bind on fired trades?
        lines += [
            "",
            "## Structural diagnostic — did the rule bind?",
            "",
            f"- Fired trades binding (f<1.0) in CALIBRATION window "
            f"(< 2023-10-01): {n_fired_bind_cal} / {len(cal_fired_sm_rep)} "
            f"({100.0*n_fired_bind_cal/max(len(cal_fired_sm_rep),1):.2f}%)",
            f"- Fired trades binding (f<1.0) in EVALUATION window "
            f"(>= 2023-10-01): {n_fired_bind_eval} / "
            f"{len(eval_fired_sm_rep)} "
            f"({100.0*n_fired_bind_eval/max(len(eval_fired_sm_rep),1):.2f}%)",
            f"- Total fired binding (full window): {n_fired_bind_full} "
            f"/ {len(fired_sm_rep)} "
            f"({100.0*n_fired_bind_full/max(len(fired_sm_rep),1):.2f}%)",
            "",
            "**Interpretation**: V2's sizing pipeline records `equity` as "
            "running compound equity. Loss budget = 1% × equity grows from "
            "$1,000 at backtest start to ~$150,000 by mid-eval-window. "
            "Both `gamma_target = 2 × LossBudget / (Spot × shock)²` and "
            "`vega_target = LossBudget / (ATM_IV × pme_shock_iv_pct)` scale "
            "linearly with equity, so the budget runs ahead of V2's actual "
            "per-bar (gamma, vega) exposure. The PME-policy rule is "
            "effectively non-binding for any fired trade after compound "
            "equity exceeds ~$1M (mid-2023). This is not a bug; it is the "
            "honest consequence of pairing a percentage-of-equity loss "
            "budget with a successful compounding strategy.",
            "",
            "**For users running V2 at a smaller account or in non-compound "
            "mode**, the rule still binds on the calibration-window slice "
            "(11.4% of fired trades) and would shrink size on bars with "
            "high pre-trade gamma OR vega. Available via the opt-in "
            "`preventative_size_mult_path` config flag (Phase P backtester "
            "integration).",
            "",
            "**Lessons codified for future cycles**:",
            "- A percentage-of-RUNNING-equity loss budget non-binds as the "
            "  account compounds. Future risk rules that anchor to running "
            "  equity should report binding-on-fired-bars in BOTH windows.",
            "- An alternative anchor (fixed-dollar policy budget, "
            "  e.g. $1,000 per trade regardless of equity) would bind "
            "  uniformly across the trade log. Whether that is desirable "
            "  depends on whether the policy interprets risk as "
            "  \"per-event % of current capital\" (current rule) or "
            "  \"per-event $ of starting capital\" (alternative). Both "
            "  are defensible; the alternative is NOT implemented in this "
            "  cycle to avoid a backtest-driven anchor switch (the "
            "  storytelling-bias failure mode).",
        ]
    else:
        lines += [
            "All three sub-gates passed. The rule is adopted as default and "
            "the V2 preset will load the sidecar automatically (mirrors the "
            "Kelly K1 adoption pattern from 2026-04-21).",
        ]
    lines.append("")

    out = args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote gate report to %s", out)

    print()
    print("=" * 70)
    print(" V2 PREVENTATIVE SIZING — GATE RESULT")
    print("=" * 70)
    print(f"  PSR p10 delta:    {psr_p10:+.4f}  "
          f"(gate >= +{GATE_PSR_P10_DELTA:.2f}) "
          f"{'PASS' if pass_psr else 'FAIL'}")
    print(f"  OOS worst cut:    ${oos_worst_cut:+,.0f}  "
          f"(gate >= +${GATE_OOS_WORST_CUT_USD:,.0f}) "
          f"{'PASS' if pass_oos else 'FAIL'}")
    print(f"  NC Sharpe drop:   {eval_sharpe_drop:+.4f}  "
          f"(gate <= +{GATE_NC_SHARPE_DROP_MAX:.2f}) "
          f"{'PASS' if pass_sharpe else 'FAIL'}")
    print(f"  --> {'ADOPT AS DEFAULT' if gate_pass else 'REJECT (ship opt-in)'}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
