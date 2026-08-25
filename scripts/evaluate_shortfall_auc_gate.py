"""Evaluate the Cycle 4 Phase B AUC-delta gate and render V4_SHORTFALL_HEAD.md.

Frozen historical gate (conjunctive; all four required):

- G1 — pooled-OOS AUC(TREATMENT) − AUC(BASELINE) ≥ +0.02
- G2 — 95% lower bound of paired stationary block bootstrap
       (mean block 5 days, paired by day, n_reps=1000) > 0
- G3 — paired DeLong p < 0.01 (one-sided, treatment > baseline)
- G4 — |AUC(PLACEBO) − AUC(BASELINE)| < 0.005 (leakage placebo)

ALL FOUR required. Failure on any → halt; honest reject. PASS → Phase C.

Run::

    PYTHONPATH=. python scripts/evaluate_shortfall_auc_gate.py
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.shortfall_head import (
    SHORTFALL_DESCRIPTIVE_USD,
    SHORTFALL_TEST_YEARS,
    SHORTFALL_THRESHOLD_USD,
    auc_delta_gate,
    per_year_auc_summary,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_shortfall_auc_gate")


# ─── Pre-committed paths ────────────────────────────────────────────────

DEFAULT_PREDICTIONS_PARQUET = Path(
    "vol_arb/output/BTC_v2_shortfall_oos_predictions.parquet"
)
DEFAULT_REPORT_PATH = Path("vol_arb/output/V4_SHORTFALL_HEAD.md")
DEFAULT_TRADE_LOG = Path("vol_arb/output/BTC_v2_trade_log.parquet")


def _format_auc_summary_table(
    target: pd.Series,
    p_baseline: pd.Series,
    p_treatment: pd.Series,
    p_placebo: pd.Series,
) -> str:
    """Render per-year + pooled AUC table for the report."""
    summary_b = per_year_auc_summary(target, p_baseline)
    summary_t = per_year_auc_summary(target, p_treatment)
    summary_p = per_year_auc_summary(target, p_placebo)

    lines = [
        "| Test year | Baseline AUC | Treatment AUC | Placebo AUC | Delta (T - B) |",
        "|---|---:|---:|---:|---:|",
    ]
    for yr in SHORTFALL_TEST_YEARS:
        b = summary_b.get(f"year_{yr}", float("nan"))
        t = summary_t.get(f"year_{yr}", float("nan"))
        p = summary_p.get(f"year_{yr}", float("nan"))
        delta = t - b if not (np.isnan(t) or np.isnan(b)) else float("nan")
        lines.append(
            f"| {yr} | {b:.4f} | {t:.4f} | {p:.4f} | {delta:+.4f} |"
        )
    b = summary_b["pooled"]
    t = summary_t["pooled"]
    p = summary_p["pooled"]
    delta = t - b
    lines.append(f"| **Pooled** | **{b:.4f}** | **{t:.4f}** | **{p:.4f}** | **{delta:+.4f}** |")
    return "\n".join(lines)


def _compute_descriptive_auc(
    trade_log_path: Path,
    p_baseline: pd.Series,
    p_treatment: pd.Series,
    p_placebo: pd.Series,
    threshold_usd: float,
) -> dict[str, float]:
    """Compute descriptive AUC at a secondary threshold (no decision axis).

    Loads the trade log to recompute the binary target at the secondary
    threshold (e.g., X=$500); returns pooled-OOS AUC for each classifier.
    """
    from vol_arb.shortfall_head import compute_shortfall_target

    if not trade_log_path.exists():
        logger.warning(
            "Trade log %s not found; skipping descriptive AUC at $%.0f",
            trade_log_path, threshold_usd,
        )
        return {}

    trade_log = pd.read_parquet(trade_log_path)
    target = compute_shortfall_target(trade_log, threshold_usd=threshold_usd)
    out: dict[str, float] = {"threshold_usd": float(threshold_usd)}
    for tag, scores in [("baseline", p_baseline), ("treatment", p_treatment),
                          ("placebo", p_placebo)]:
        summary = per_year_auc_summary(target, scores)
        out[f"pooled_{tag}"] = float(summary["pooled"])
    n_pos = int(target.sum())
    n_total = int(len(target))
    out["n_positives"] = n_pos
    out["n_total"] = n_total
    out["positive_rate_pct"] = (
        100.0 * n_pos / n_total if n_total > 0 else float("nan")
    )
    return out


def _render_report(
    *,
    gate_result,
    auc_summary_table: str,
    n_total: int,
    n_positives: int,
    threshold_usd: float,
    descriptive_500: dict[str, float],
    test_years: tuple[int, ...],
    seed: int,
    n_reps: int,
    block_mean_days: int,
) -> str:
    """Render the V4_SHORTFALL_HEAD.md report content."""
    g1 = "PASS" if gate_result.g1_pass else "FAIL"
    g2 = "PASS" if gate_result.g2_pass else "FAIL"
    g3 = "PASS" if gate_result.g3_pass else "FAIL"
    g4 = "PASS" if gate_result.g4_pass else "FAIL"
    verdict = "PASS — proceed to Phase C" if gate_result.all_pass else "FAIL — honest reject"

    descr_block = ""
    if descriptive_500:
        descr_block = f"""
## Descriptive secondary — X = ${descriptive_500['threshold_usd']:.0f}

This threshold is reported for transparency only; **NO** decision axis.
Class balance at this threshold was insufficient pre-experiment for a
gate (1.35% positive globally). Reported here so future cycles see the
unblinded numbers if they want to revisit.

- N total = {descriptive_500['n_total']:,}, N positives = {descriptive_500['n_positives']:,} ({descriptive_500['positive_rate_pct']:.2f}%)
- Pooled-OOS AUC: baseline = {descriptive_500.get('pooled_baseline', float('nan')):.4f}, treatment = {descriptive_500.get('pooled_treatment', float('nan')):.4f}, placebo = {descriptive_500.get('pooled_placebo', float('nan')):.4f}
- Delta = {descriptive_500.get('pooled_treatment', 0.0) - descriptive_500.get('pooled_baseline', 0.0):+.4f}
"""

    return f"""# V4 Conditional-Shortfall Classifier — Gate Result

Historical Cycle 4 Phase B comparison.

Three LightGBM classifiers with identical hyperparameters
(``DFL_LGBM_PARAMS``), identical walk-forward year folds (expanding by
calendar year, train_y < test_y), only the feature set differs:

- **BASELINE**: 10 DFL-Head features (``DFL_HEAD_FEATURES``)
- **TREATMENT**: 10 DFL features + ``sigma_epi_squared``
- **PLACEBO**: 10 DFL features + within-year-shuffled ``sigma_epi_squared``

σ²_epi inputs come from the WALK-FORWARD ensemble parquet
(``BTC_v2_har_pd_ensemble_var_walkforward.parquet``, Phase A.0 output).
The Cycle 3 K-fold parquet was NOT used (it leaks across years).

## Sample size

- Pre-committed shortfall threshold: **X = ${threshold_usd:.0f}**
- Test years: {list(test_years)}
- Pooled-OOS fired trades = {n_total:,}, positives = {n_positives:,} ({100.0 * n_positives / max(n_total, 1):.2f}%)

## Per-year + pooled AUC

{auc_summary_table}

## Pre-committed gate (CONJUNCTIVE, all four required)

| Gate | Definition | Threshold | Observed | Verdict |
|---|---|---|---|---|
| G1 | pooled-OOS AUC delta | ≥ {gate_result.g1_threshold:+.3f} | {gate_result.delta_pooled:+.4f} | **{g1}** |
| G2 | 95% bootstrap CI lower bound | > 0 | LCB = {gate_result.delta_lcb_95:+.4f} (median {gate_result.delta_median_bootstrap:+.4f}, UCB {gate_result.delta_ucb_95:+.4f}) | **{g2}** |
| G3 | paired DeLong one-sided p | < {gate_result.g3_p_threshold:.3f} | p = {gate_result.delong_p_one_sided:.4g} (var = {gate_result.delong_var_delta:.2e}) | **{g3}** |
| G4 | placebo AUC delta magnitude | < {gate_result.g4_threshold:.3f} | placebo Δ = {gate_result.delta_placebo:+.4f} | **{g4}** |

Bootstrap config: stationary block bootstrap, n_reps={n_reps}, mean block
length = {block_mean_days} days, paired by DAY (Politis-Romano 1994), seed = {seed}.
DeLong: covariance via Sun-Xu 2014 closed-form, alternative = "greater".

## Verdict

**{verdict}**

{"All four gates passed. Phase C (sidecar filter under Cycle 1 Tier C-2 triple-gate, K=3 DSR haircut) is authorised." if gate_result.all_pass else "Phase B failed. Cycle 3 + Cycle 4 jointly closed on the σ²_epi axis; the deep-ensemble epistemic-uncertainty signal does not add information beyond what the DFL Head's 10-feature decision panel already encodes for downside-asymmetric trade prediction. Per the Stage-37 critique, future research-cycles should pivot to alternative decision-aligned signal definitions, NOT escalate ensemble strength or model class on this same target / feature panel."}
{descr_block}

## Reproduction

```bash
PYTHONPATH=. python scripts/regenerate_walkforward_ensemble_variance.py
PYTHONPATH=. python scripts/run_shortfall_classifier.py
PYTHONPATH=. python scripts/evaluate_shortfall_auc_gate.py
```

All three scripts are deterministic under fixed seeds. Pooled-OOS
predictions parquet:
``vol_arb/output/BTC_v2_shortfall_oos_predictions.parquet``.
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cycle 4 Phase B: evaluate the four-part AUC-delta gate "
                    "(G1 effect size, G2 bootstrap CI, G3 DeLong, G4 placebo) "
                    "and render V4_SHORTFALL_HEAD.md.",
    )
    parser.add_argument("--predictions", type=Path,
                        default=DEFAULT_PREDICTIONS_PARQUET)
    parser.add_argument("--report-path", type=Path, default=DEFAULT_REPORT_PATH)
    parser.add_argument("--trade-log", type=Path, default=DEFAULT_TRADE_LOG,
                        help="For computing descriptive X=$500 AUC")
    parser.add_argument("--threshold-usd", type=float,
                        default=SHORTFALL_THRESHOLD_USD)
    parser.add_argument("--n-reps", type=int, default=1_000)
    parser.add_argument("--block-mean-days", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing report")
    args = parser.parse_args()

    # Load pooled-OOS predictions (Phase A.1 output)
    df = pd.read_parquet(args.predictions)
    required_cols = {"p_baseline", "p_treatment", "p_placebo", "y_short"}
    missing = required_cols - set(df.columns)
    if missing:
        raise KeyError(f"missing columns in {args.predictions}: {sorted(missing)}")

    p_baseline = df["p_baseline"]
    p_treatment = df["p_treatment"]
    p_placebo = df["p_placebo"]
    target = df["y_short"].astype(int)

    # Evaluate gate
    logger.info("Computing AUC-delta gate (n_reps=%d, block=%d days)...",
                args.n_reps, args.block_mean_days)
    gate_result = auc_delta_gate(
        target, p_baseline, p_treatment, p_placebo,
        block_bootstrap_n_reps=args.n_reps,
        block_mean_days=args.block_mean_days,
        seed=args.seed,
    )

    # Per-year + pooled summary table
    table = _format_auc_summary_table(target, p_baseline, p_treatment, p_placebo)

    # Filter to test-year fired trades for headline counts
    in_test_years = pd.Series(target.index.year.isin(SHORTFALL_TEST_YEARS),
                                index=target.index)
    valid_mask = ~(p_baseline.isna() | p_treatment.isna() | p_placebo.isna())
    eval_mask = in_test_years & valid_mask
    n_total = int(eval_mask.sum())
    n_positives = int(target.loc[eval_mask].sum())

    # Descriptive X=$500 secondary
    descr = {}
    if args.trade_log.exists():
        for x_descr in SHORTFALL_DESCRIPTIVE_USD:
            descr = _compute_descriptive_auc(
                args.trade_log, p_baseline, p_treatment, p_placebo,
                threshold_usd=x_descr,
            )

    # Console summary
    print()
    print("=" * 70)
    for line in gate_result.summary_lines():
        print(f"  {line}")
    print("=" * 70)

    # Render and save report
    report = _render_report(
        gate_result=gate_result,
        auc_summary_table=table,
        n_total=n_total,
        n_positives=n_positives,
        threshold_usd=args.threshold_usd,
        descriptive_500=descr,
        test_years=SHORTFALL_TEST_YEARS,
        seed=args.seed,
        n_reps=args.n_reps,
        block_mean_days=args.block_mean_days,
    )
    args.report_path.parent.mkdir(parents=True, exist_ok=True)
    if args.report_path.exists() and not args.force:
        logger.warning(
            "Report %s already exists; appending '_new' suffix. "
            "Use --force to overwrite.", args.report_path,
        )
        out_path = args.report_path.with_name(args.report_path.stem + "_new.md")
    else:
        out_path = args.report_path
    out_path.write_text(report, encoding="utf-8")
    logger.info("Wrote report to %s", out_path)

    return 0 if gate_result.all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
