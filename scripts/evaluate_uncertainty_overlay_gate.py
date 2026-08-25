"""Run V2 backtest for each of the 5 Cycle-3 Phase P strategies + gate.

For each strategy (F1, F2, F3, S1, S2):
  1. Configure V2 with the corresponding sidecar opt-in.
  2. Run the canonical V2 backtest end-to-end.
  3. Compare against baseline (canonical V2 trade log) using
     ``synthetic_path_evaluator.block_bootstrap_deltas``.
  4. Compute (PSR_p10_delta, OOS_worst_day_cut, NC_Sharpe_drop).
  5. Apply DSR haircut for K=5 multiple-testing.

Pre-committed gate (verbatim from Cycle 1 Tier C-2 / Cycle 2):
  - 10th-pct PSR delta >= +0.30
  - OOS worst-day cut >= +$5,000
  - NC Sharpe drop on eval window <= +0.30
  - All three required (conjunctive).

Best-by-PSR-p10-(DSR-haircut) strategy that passes ALL three is the
best historical-threshold candidate. The result does not change a preset.

Output: ``vol_arb/output/V2_EPISTEMIC_OVERLAY.md``.

Run::

    PYTHONPATH=. python scripts/evaluate_uncertainty_overlay_gate.py
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.backtester import BacktestConfig, VolArbBacktester
from vol_arb.synthetic_path_evaluator import (
    DEFAULT_BLOCK_SIZE_DAYS,
    DEFAULT_N_BOOTSTRAP_REPS,
    _annualised_sharpe,
    block_bootstrap_deltas,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate_uncertainty_overlay_gate")

# Pre-committed (Cycle 1 Tier C-2 verbatim)
GATE_PSR_P10_DELTA: float = 0.30
GATE_OOS_WORST_CUT_USD: float = 5_000.0
GATE_NC_SHARPE_DROP_MAX: float = 0.30

# DSR haircut for K=5 hypotheses tested
DSR_K: int = 5
# Bailey & López de Prado 2014: SR* / SR ≈ sqrt((K-1)(1-γ_E + γ_E/K))/sqrt(K)
# For K=5 with γ_E = 0.577 (Euler-Mascheroni): ~0.86
import math
_GE = 0.5772156649
DSR_MULTIPLIER = math.sqrt((DSR_K - 1) * (1 - _GE + _GE / DSR_K)) / math.sqrt(DSR_K)

EVAL_WINDOW_START = pd.Timestamp("2023-10-01", tz="UTC")
TRADE_LOG_PATH = Path("vol_arb/output/BTC_v2_trade_log.parquet")
OUTPUT_DIR = Path("vol_arb/output")
REPORT_PATH = OUTPUT_DIR / "V2_EPISTEMIC_OVERLAY.md"

OOS_VALIDATION_DAYS: dict[str, pd.Timestamp] = {
    "Celsius_2022_06_13": pd.Timestamp("2022-06-13", tz="UTC"),
    "FTX_2022_11_08": pd.Timestamp("2022-11-08", tz="UTC"),
    "JPY_carry_2024_08_05": pd.Timestamp("2024-08-05", tz="UTC"),
    "Trump_inauguration_2025_01_20": pd.Timestamp("2025-01-20", tz="UTC"),
}


@dataclass
class StrategyResult:
    tag: str
    kind: str  # "filter" | "scaler"
    sidecar: Path
    candidate_pnl: float
    sharpe_full: float
    sharpe_eval: float
    sharpe_drop_eval: float
    pnl_full: float
    maxdd_full: float
    pnl_delta: float
    sharpe_delta_full: float
    psr_delta_p10: float
    psr_delta_p10_haircut: float
    oos_worst_baseline: float
    oos_worst_candidate: float
    oos_worst_cut: float
    n_baseline_fired: int
    n_candidate_fired: int
    pass_psr: bool
    pass_oos: bool
    pass_sharpe: bool
    gate_pass: bool


def _daily_pnl(tl: pd.DataFrame) -> pd.Series:
    return tl["net_pnl_nc"].groupby(tl.index.normalize()).sum()


def _maxdd(daily: pd.Series) -> float:
    cum = daily.cumsum()
    peaks = cum.cummax()
    return float((cum - peaks).min())


def _run_strategy(
    tag: str, kind: str, sidecar: Path,
    baseline_tl: pd.DataFrame, baseline_daily: pd.Series,
    *, n_reps: int, block_size_days: int, seed: int,
) -> StrategyResult:
    logger.info("=" * 50)
    logger.info("[%s] sidecar: %s", tag, sidecar.name)
    config = BacktestConfig().with_historical_btc_v2().with_realistic_costs()
    if kind == "filter":
        config.uncertainty_filter_path = str(sidecar)
    else:
        config.preventative_size_mult_path = str(sidecar)

    t0 = time.time()
    metrics, candidate_tl = VolArbBacktester(config).run_backtest()
    logger.info("[%s] backtest finished in %.1fs", tag, time.time() - t0)

    candidate_daily = _daily_pnl(candidate_tl)
    n_baseline_fired = int((baseline_tl["net_pnl_nc"] != 0).sum())
    n_candidate_fired = int((candidate_tl["net_pnl_nc"] != 0).sum())
    logger.info("[%s] fired baseline=%d, candidate=%d",
                tag, n_baseline_fired, n_candidate_fired)

    # Headline (full) metrics
    pnl_full = float(candidate_daily.sum())
    sharpe_full = _annualised_sharpe(candidate_daily.to_numpy())
    maxdd_full = _maxdd(candidate_daily)
    pnl_delta = pnl_full - float(baseline_daily.sum())
    sharpe_delta_full = sharpe_full - _annualised_sharpe(baseline_daily.to_numpy())

    # Eval-window metrics
    eval_baseline_daily = baseline_daily.loc[
        baseline_daily.index >= EVAL_WINDOW_START
    ]
    eval_candidate_daily = candidate_daily.loc[
        candidate_daily.index >= EVAL_WINDOW_START
    ]
    eval_sharpe_baseline = _annualised_sharpe(eval_baseline_daily.to_numpy())
    eval_sharpe_candidate = _annualised_sharpe(eval_candidate_daily.to_numpy())
    sharpe_drop_eval = eval_sharpe_baseline - eval_sharpe_candidate

    # OOS
    oos_dates = list(OOS_VALIDATION_DAYS.values())
    oos_baseline = baseline_daily.reindex(oos_dates, fill_value=0.0)
    oos_candidate = candidate_daily.reindex(oos_dates, fill_value=0.0)
    oos_worst_baseline = float(oos_baseline.min())
    oos_worst_candidate = float(oos_candidate.min())
    oos_worst_cut = oos_worst_candidate - oos_worst_baseline

    # Bootstrap (eval window)
    boot = block_bootstrap_deltas(
        eval_baseline_daily, eval_candidate_daily,
        n_reps=n_reps, block_size_days=block_size_days, seed=seed,
    )
    psr_delta_p10 = boot["psr_delta"].p10
    # DSR haircut: shrink observed Sharpe (and PSR delta) by multiplier
    psr_delta_p10_haircut = psr_delta_p10 * DSR_MULTIPLIER

    pass_psr = psr_delta_p10_haircut >= GATE_PSR_P10_DELTA
    pass_oos = oos_worst_cut >= GATE_OOS_WORST_CUT_USD
    pass_sharpe = sharpe_drop_eval <= GATE_NC_SHARPE_DROP_MAX
    gate_pass = pass_psr and pass_oos and pass_sharpe

    logger.info(
        f"[{tag}] PSR_p10={psr_delta_p10:+.4f} "
        f"(haircut {psr_delta_p10_haircut:+.4f}), "
        f"OOS_cut=${oos_worst_cut:+,.0f}, "
        f"dSharpe(eval)={-sharpe_drop_eval:+.4f}"
    )
    logger.info(
        f"[{tag}] gate: PSR={pass_psr}, OOS={pass_oos}, Sharpe={pass_sharpe} "
        f"-> {'PASS' if gate_pass else 'FAIL'}"
    )

    return StrategyResult(
        tag=tag, kind=kind, sidecar=sidecar,
        candidate_pnl=pnl_full,
        sharpe_full=sharpe_full,
        sharpe_eval=eval_sharpe_candidate,
        sharpe_drop_eval=sharpe_drop_eval,
        pnl_full=pnl_full,
        maxdd_full=maxdd_full,
        pnl_delta=pnl_delta,
        sharpe_delta_full=sharpe_delta_full,
        psr_delta_p10=psr_delta_p10,
        psr_delta_p10_haircut=psr_delta_p10_haircut,
        oos_worst_baseline=oos_worst_baseline,
        oos_worst_candidate=oos_worst_candidate,
        oos_worst_cut=oos_worst_cut,
        n_baseline_fired=n_baseline_fired,
        n_candidate_fired=n_candidate_fired,
        pass_psr=pass_psr, pass_oos=pass_oos, pass_sharpe=pass_sharpe,
        gate_pass=gate_pass,
    )


STRATEGIES = [
    ("F1", "filter", "BTC_v2_uncert_F1_filter01.parquet"),
    ("F2", "filter", "BTC_v2_uncert_F2_filter05.parquet"),
    ("F3", "filter", "BTC_v2_uncert_F3_filter10.parquet"),
    ("S1", "scaler", "BTC_v2_uncert_S1_size_mult.parquet"),
    ("S2", "scaler", "BTC_v2_uncert_S2_size_mult.parquet"),
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=REPORT_PATH)
    parser.add_argument("--n-reps", type=int, default=DEFAULT_N_BOOTSTRAP_REPS)
    parser.add_argument("--block-size-days", type=int,
                        default=DEFAULT_BLOCK_SIZE_DAYS)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("Loading baseline V2 trade log: %s", TRADE_LOG_PATH)
    baseline_tl = pd.read_parquet(TRADE_LOG_PATH)
    if not isinstance(baseline_tl.index, pd.DatetimeIndex):
        baseline_tl.index = pd.to_datetime(baseline_tl.index, utc=True)
    baseline_daily = _daily_pnl(baseline_tl)
    logger.info("Baseline NC PnL: $%,.0f, fired=%d, days=%d",
                float(baseline_daily.sum()),
                int((baseline_tl["net_pnl_nc"] != 0).sum()),
                len(baseline_daily))

    logger.info("DSR haircut (K=%d): multiplier = %.4f",
                DSR_K, DSR_MULTIPLIER)

    results: list[StrategyResult] = []
    for tag, kind, name in STRATEGIES:
        sidecar = OUTPUT_DIR / name
        if not sidecar.exists():
            logger.error("Sidecar missing: %s — skipping %s", sidecar, tag)
            continue
        result = _run_strategy(
            tag, kind, sidecar, baseline_tl, baseline_daily,
            n_reps=args.n_reps, block_size_days=args.block_size_days,
            seed=args.seed,
        )
        results.append(result)

    # Render markdown
    lines = [
        "# V2 Epistemic-Uncertainty Overlay — Gate Result (Cycle 3 Phase P)",
        "",
        "5 pre-committed strategies, ONE backtest per strategy, "
        f"DSR-haircut multiplier = {DSR_MULTIPLIER:.4f} (K=5 hypotheses).",
        "",
        f"Baseline V2: NC PnL ${float(baseline_daily.sum()):,.0f}, "
        f"{int((baseline_tl['net_pnl_nc'] != 0).sum())} fired trades.",
        "",
        "## Pre-committed gate (Cycle 1 Tier C-2 verbatim)",
        "",
        f"- 10th-pct PSR delta (DSR-haircut) ≥ +{GATE_PSR_P10_DELTA:.2f}",
        f"- OOS-validation 4-day worst-day cut ≥ +${GATE_OOS_WORST_CUT_USD:,.0f}",
        f"- NC Sharpe drop on eval window ≤ +{GATE_NC_SHARPE_DROP_MAX:.2f}",
        "- All three required (conjunctive)",
        "",
        "## Per-strategy results",
        "",
        "| Tag | Kind | NC PnL | ΔPnL | Fired | ΔSharpe(eval) | OOS cut | PSR p10 (haircut) | Threshold |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for r in results:
        adopt_tag = "PASS" if r.gate_pass else "FAIL"
        lines.append(
            f"| **{r.tag}** | {r.kind} | "
            f"${r.pnl_full:,.0f} | ${r.pnl_delta:+,.0f} | "
            f"{r.n_candidate_fired} | "
            f"{-r.sharpe_drop_eval:+.3f} | "
            f"${r.oos_worst_cut:+,.0f} | "
            f"{r.psr_delta_p10_haircut:+.4f} | "
            f"{adopt_tag} |"
        )

    # Historical threshold result
    lines += ["", "## Historical threshold result", ""]
    passing = [r for r in results if r.gate_pass]
    if passing:
        # Best-by-PSR-haircut, prefer filters on tie
        passing.sort(
            key=lambda r: (r.psr_delta_p10_haircut, r.kind == "filter"),
            reverse=True,
        )
        best = passing[0]
        lines += [
            f"**PASS [{best.tag}] HISTORICAL THRESHOLDS.** Best result by "
            f"DSR-haircut PSR p10 = {best.psr_delta_p10_haircut:+.4f}.",
            "",
            f"- Sharpe drop on eval: {best.sharpe_drop_eval:+.4f}",
            f"- OOS worst-day cut: ${best.oos_worst_cut:+,.0f}",
            f"- ΔPnL (full): ${best.pnl_delta:+,.0f}",
            "",
            "All three historical sub-gates passed. This report does not "
            "change a preset or validate the candidate for paper/testnet use.",
        ]
    else:
        lines += [
            "**FAIL — no strategy passed all historical thresholds.**",
            "",
            "No strategy passed the conjunctive triple-gate after DSR "
            "haircut. The historical V2 reference remains unchanged. "
            "Lowering the gate to "
            "match observed numbers would be the canonical storytelling-"
            "bias failure mode the plan codified. Honest reject.",
            "",
            "Sub-gate failures by strategy:",
        ]
        for r in results:
            failed = []
            if not r.pass_psr:
                failed.append("PSR_p10")
            if not r.pass_oos:
                failed.append("OOS_cut")
            if not r.pass_sharpe:
                failed.append("Sharpe_drop")
            lines.append(f"- **{r.tag}**: failed {', '.join(failed) if failed else '—'}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote gate report to %s", args.out)

    print()
    print("=" * 70)
    print(" V2 EPISTEMIC-UNCERTAINTY OVERLAY -- GATE RESULT")
    print("=" * 70)
    for r in results:
        flag = "PASS" if r.gate_pass else "FAIL"
        print(
            f"  [{r.tag}] {flag}  PSR={r.psr_delta_p10_haircut:+.3f}  "
            f"OOS=${r.oos_worst_cut:+,.0f}  dSh={-r.sharpe_drop_eval:+.3f}  "
            f"dPnL=${r.pnl_delta:+,.0f}"
        )
    print("=" * 70)
    if passing:
        print(f"  PASS [{passing[0].tag}] HISTORICAL THRESHOLDS")
    else:
        print("  FAIL -- no candidate passed all historical thresholds")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
