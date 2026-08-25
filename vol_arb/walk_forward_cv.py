"""Walk-forward cross-validation harness for Phase 2 (profit_filter_min_ratio).

Design
------
Expanding-window CV over the 8 folds already used by the dashboard's robustness
tab (each fold = 1/8 of all bars, indexed sequentially). For each test fold k
in {2..8}, the calibration data is strictly the union of folds 1..(k-1); the
candidate value of `profit_filter_min_ratio` that maximises calibration NC
Sharpe (concatenated across the calibration folds) is applied on fold k.

Fold 1 is pure calibration — it cannot be used for OOS evaluation because no
prior data exists to choose its parameter.

The final OOS Sharpe is the Sharpe of the concatenated NC returns across folds
2..8 using the chosen parameter for each fold.

Important implementation detail
-------------------------------
The backtester is a sequential simulation with running state (profit-PnL EMA,
hedge EMA, etc.). A candidate's state at the start of fold k depends on all
earlier bars. To produce faithful per-fold trades we run the FULL backtest
once per candidate and slice the resulting trade log by fold boundaries. This
is correct because the trade produced at bar t depends only on bars <= t; the
slice of the full trade log restricted to fold k is identical to what you
would get by running the backtester end-to-end with that candidate and then
picking off the fold-k rows.

This matters for the calibration step: we do NOT run the backtester restricted
to calibration bars — that would lose the full pre-fold-k state. Instead we
run the full backtest per candidate and compute per-fold NC Sharpe on the
resulting trade log.

Usage
-----
    python -m vol_arb.walk_forward_cv

Outputs (vol_arb/output/):
    - phase2_wf_ratio_{000,015,030,045}.parquet  (per-candidate trade logs)
    - phase2_wf_cv_results.csv                    (fold-by-fold CV table)
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.backtester import BacktestConfig, VolArbBacktester

# ── Candidates + CV config ────────────────────────────────────────

CANDIDATES: tuple[float, ...] = (0.0, 0.15, 0.30, 0.45)
N_FOLDS: int = 8
BARS_PER_YEAR: float = 365.25 * 24 * 4  # 15-min bars


def _sharpe(returns: np.ndarray) -> float:
    """Annualised Sharpe on a bar-returns array. Uses ddof=1."""
    if len(returns) < 2:
        return 0.0
    sd = float(np.std(returns, ddof=1))
    if sd <= 0:
        return 0.0
    return (float(np.mean(returns)) / sd) * math.sqrt(BARS_PER_YEAR)


def _fold_boundaries(n_bars: int, n_folds: int = N_FOLDS) -> list[tuple[int, int]]:
    """Return equal eval-time ``(start, end)`` bar-index folds."""
    return [(i * n_bars // n_folds, (i + 1) * n_bars // n_folds) for i in range(n_folds)]


def _per_fold_sharpe_from_nc_eq(eq_nc: np.ndarray, n_folds: int = N_FOLDS) -> list[float]:
    """Per-fold NC Sharpe using the equity_nc path."""
    bounds = _fold_boundaries(len(eq_nc), n_folds)
    sharpes = []
    for (a, b) in bounds:
        seg = eq_nc[a:b]
        if len(seg) < 2:
            sharpes.append(0.0)
            continue
        # Bar returns: fractional changes on the NC equity path.
        r = np.diff(seg) / seg[:-1]
        # Drop any NaNs (shouldn't occur but defensive).
        r = r[np.isfinite(r)]
        sharpes.append(_sharpe(r))
    return sharpes


def _calibration_sharpe(
    eq_nc: np.ndarray, cal_fold_end_idx: int, n_folds: int = N_FOLDS,
) -> float:
    """Concatenated-returns Sharpe across folds 1..cal_fold_end_idx
    (1-indexed inclusive). `cal_fold_end_idx` is the k in "fold k is last
    calibration fold"."""
    bounds = _fold_boundaries(len(eq_nc), n_folds)
    # Concatenate returns across folds 1..cal_fold_end_idx (0-indexed 0..cal_fold_end_idx-1).
    segs = []
    for i in range(cal_fold_end_idx):
        a, b = bounds[i]
        seg = eq_nc[a:b]
        if len(seg) < 2:
            continue
        r = np.diff(seg) / seg[:-1]
        r = r[np.isfinite(r)]
        segs.append(r)
    if not segs:
        return 0.0
    return _sharpe(np.concatenate(segs))


def _run_or_load_backtest(min_ratio: float, output_dir: Path) -> pd.DataFrame:
    """Run the full backtest for the given profit_filter_min_ratio. Cache the
    trade log to parquet so re-running the harness on a second invocation
    (e.g. after code tweaks) skips the ~3-minute backtest where possible."""
    pct = int(round(min_ratio * 100))
    path = output_dir / f"phase2_wf_ratio_{pct:03d}.parquet"
    if path.exists():
        print(f"  [cache hit] {path.name}")
        return pd.read_parquet(path)
    cfg = (
        BacktestConfig()
        .with_historical_btc_v2()
        .model_copy(update=dict(profit_filter_min_ratio=min_ratio))
    )
    bt = VolArbBacktester(cfg)
    print(f"  [running] profit_filter_min_ratio={min_ratio:.2f}")
    _metrics, df = bt.run_backtest()
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    print(f"  [saved] {path.name}")
    return df


# ── Fallback sweep: walk-forward over scalar threshold pairs ─────

THRESHOLD_PAIRS: tuple[tuple[float, float], ...] = (
    (12.0, 22.0),  # V2 default (baseline)
    (15.0, 25.0),  # modest raise
    (18.0, 30.0),  # moderate raise
    (22.0, 35.0),  # aggressive raise
)


def _run_or_load_threshold_backtest(
    thr_long: float, thr_short: float, output_dir: Path,
) -> pd.DataFrame:
    """Run the full backtest for a given (entry_threshold_long, entry_threshold)
    pair. Cached per pair."""
    path = output_dir / f"phase2_wf_thr_L{int(thr_long):02d}_S{int(thr_short):02d}.parquet"
    if path.exists():
        print(f"  [cache hit] {path.name}")
        return pd.read_parquet(path)
    cfg = (
        BacktestConfig()
        .with_historical_btc_v2()
        .model_copy(update=dict(
            entry_threshold_long=thr_long,
            entry_threshold=thr_short,
        ))
    )
    bt = VolArbBacktester(cfg)
    print(f"  [running] thr_L={thr_long:.1f} thr_S={thr_short:.1f}")
    _metrics, df = bt.run_backtest()
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    print(f"  [saved] {path.name}")
    return df


def run_walk_forward_thresholds(
    pairs: tuple[tuple[float, float], ...] = THRESHOLD_PAIRS,
    n_folds: int = N_FOLDS,
    output_dir: Path | None = None,
) -> dict:
    """Same walk-forward CV structure as run_walk_forward(), but over
    (entry_threshold_long, entry_threshold) pairs instead of a scalar ratio.
    Pair (12, 22) is the Phase-1 baseline; other pairs are uniform raises."""
    if output_dir is None:
        output_dir = Path("vol_arb/output")

    logs: dict[tuple[float, float], pd.DataFrame] = {}
    for p in pairs:
        logs[p] = _run_or_load_threshold_backtest(p[0], p[1], output_dir)

    per_candidate_fold_sharpes: dict[tuple[float, float], list[float]] = {}
    for p in pairs:
        eq_nc = logs[p]["equity_nc"].values
        per_candidate_fold_sharpes[p] = _per_fold_sharpe_from_nc_eq(eq_nc, n_folds)

    cv_rows: list[dict] = []
    for k in range(2, n_folds + 1):
        cal_scores = {
            p: _calibration_sharpe(logs[p]["equity_nc"].values, k - 1, n_folds)
            for p in pairs
        }
        chosen = max(cal_scores, key=cal_scores.get)
        test_sh = per_candidate_fold_sharpes[chosen][k - 1]
        cv_rows.append({
            "fold": k,
            "chosen_pair": chosen,
            "calibration_sharpe": cal_scores[chosen],
            "test_sharpe": test_sh,
            "test_sharpes_by_candidate": {
                p: per_candidate_fold_sharpes[p][k - 1] for p in pairs
            },
        })

    # Concatenated OOS with chosen params.
    concat_chosen: list[np.ndarray] = []
    for row in cv_rows:
        k = row["fold"]
        p = row["chosen_pair"]
        eq = logs[p]["equity_nc"].values
        a = (k - 1) * len(eq) // n_folds
        b = k * len(eq) // n_folds
        seg = eq[a:b]
        if len(seg) >= 2:
            r = np.diff(seg) / seg[:-1]
            r = r[np.isfinite(r)]
            concat_chosen.append(r)
    oos_sharpe_chosen = _sharpe(np.concatenate(concat_chosen)) if concat_chosen else 0.0

    # Baseline: Phase-1 default pair (12, 22).
    baseline = pairs[0]
    baseline_per_fold = per_candidate_fold_sharpes[baseline]
    eq_baseline = logs[baseline]["equity_nc"].values
    concat_baseline: list[np.ndarray] = []
    for k in range(2, n_folds + 1):
        a = (k - 1) * len(eq_baseline) // n_folds
        b = k * len(eq_baseline) // n_folds
        seg = eq_baseline[a:b]
        if len(seg) >= 2:
            r = np.diff(seg) / seg[:-1]
            r = r[np.isfinite(r)]
            concat_baseline.append(r)
    oos_sharpe_baseline = _sharpe(np.concatenate(concat_baseline)) if concat_baseline else 0.0

    return {
        "per_candidate_fold_sharpes": per_candidate_fold_sharpes,
        "cv_table": cv_rows,
        "concat_oos_sharpe_chosen": oos_sharpe_chosen,
        "concat_oos_sharpe_baseline": oos_sharpe_baseline,
        "baseline_per_fold": baseline_per_fold,
    }


def format_threshold_report(results: dict) -> str:
    """Threshold-pair variant of format_report()."""
    lines: list[str] = []
    lines.append("=" * 80)
    lines.append("Phase 2 Walk-Forward CV — (entry_threshold_long, entry_threshold)")
    lines.append("=" * 80)
    lines.append("")
    lines.append("Per-candidate per-fold NC Sharpe (annualised, bar-returns):")
    header = f"  {'pair':>10}"
    for f in range(1, N_FOLDS + 1):
        header += f"   f{f:1d}"
    lines.append(header)
    for p, arr in results["per_candidate_fold_sharpes"].items():
        pair_str = f"({int(p[0])},{int(p[1])})"
        row = f"  {pair_str:>10}"
        for s in arr:
            row += f" {s:>5.1f}"
        lines.append(row)
    lines.append("")
    lines.append("Walk-forward CV decisions:")
    lines.append(
        f"  {'fold':>4} {'chosen':>12} {'cal_sh':>8} {'test_sh':>8}  "
        f"| per-candidate test fold Sharpes"
    )
    for row in results["cv_table"]:
        pair_str = f"({int(row['chosen_pair'][0])},{int(row['chosen_pair'][1])})"
        cand_str = " ".join(
            f"({int(p[0])},{int(p[1])}):{s:+.1f}"
            for p, s in row["test_sharpes_by_candidate"].items()
        )
        lines.append(
            f"  {row['fold']:>4} {pair_str:>12} "
            f"{row['calibration_sharpe']:>8.2f} {row['test_sharpe']:>8.2f}  | {cand_str}"
        )
    lines.append("")
    lines.append("Summary (folds 2..8 concatenated, NC bar returns):")
    lines.append(f"  Chosen-param OOS Sharpe:  {results['concat_oos_sharpe_chosen']:>6.2f}")
    lines.append(f"  Baseline (12,22) OOS:     {results['concat_oos_sharpe_baseline']:>6.2f}")
    delta = results["concat_oos_sharpe_chosen"] - results["concat_oos_sharpe_baseline"]
    lines.append(f"  Δ:                        {delta:>+6.2f}")
    lines.append("")
    lines.append("Plan accept-gate evaluation:")
    gate1 = delta >= 0.30
    lines.append(
        f"  (1) OOS Sharpe ≥ baseline + 0.3: {'PASS' if gate1 else 'FAIL'} "
        f"(Δ = {delta:+.2f})"
    )
    max_drop = max(
        results["baseline_per_fold"][k - 1] - row["test_sharpe"]
        for k, row in enumerate(results["cv_table"], start=2)
    )
    gate2 = max_drop < 0.5
    lines.append(
        f"  (2) No fold drops >0.5 vs baseline: {'PASS' if gate2 else 'FAIL'} "
        f"(max drop = {max_drop:+.2f})"
    )
    return "\n".join(lines)


def run_walk_forward(
    candidates: tuple[float, ...] = CANDIDATES,
    n_folds: int = N_FOLDS,
    output_dir: Path | None = None,
) -> dict:
    """Run the walk-forward CV. Returns a dict with:
        'per_candidate_fold_sharpes': {ratio: [fold_1_sh, ..., fold_8_sh]}
        'cv_table':                   list of dicts (one per test fold)
        'concat_oos_sharpe_chosen':   Sharpe of concatenated chosen-param returns
        'concat_oos_sharpe_baseline': Sharpe if ratio=0.0 used on every fold
        'baseline_per_fold':          ratio=0.0 per-fold Sharpes (for comparison)
    """
    if output_dir is None:
        output_dir = Path("vol_arb/output")

    # 1. Run one backtest per candidate.
    logs: dict[float, pd.DataFrame] = {}
    for c in candidates:
        logs[c] = _run_or_load_backtest(c, output_dir)

    # 2. Per-fold NC Sharpe matrix: candidate → [fold_1, ..., fold_N].
    per_candidate_fold_sharpes: dict[float, list[float]] = {}
    for c in candidates:
        eq_nc = logs[c]["equity_nc"].values
        per_candidate_fold_sharpes[c] = _per_fold_sharpe_from_nc_eq(eq_nc, n_folds)

    # 3. Walk-forward: for each test fold k (k=2..n_folds), choose the candidate
    #    maximising CALIBRATION concatenated Sharpe over folds 1..k-1.
    cv_rows: list[dict] = []
    for k in range(2, n_folds + 1):
        # Calibration Sharpe per candidate (concat returns, folds 1..k-1).
        cal_scores = {
            c: _calibration_sharpe(logs[c]["equity_nc"].values, k - 1, n_folds)
            for c in candidates
        }
        chosen = max(cal_scores, key=cal_scores.get)
        test_sh = per_candidate_fold_sharpes[chosen][k - 1]  # fold k is idx k-1
        cv_rows.append({
            "fold": k,
            "chosen_ratio": chosen,
            "calibration_sharpe": cal_scores[chosen],
            "test_sharpe": test_sh,
            "test_sharpes_by_candidate": {
                c: per_candidate_fold_sharpes[c][k - 1] for c in candidates
            },
        })

    # 4. Concatenated OOS Sharpe (chosen params). For each k=2..n_folds, take
    #    the chosen-param NC returns inside fold k, concatenate.
    concat_chosen: list[np.ndarray] = []
    for row in cv_rows:
        k = row["fold"]
        c = row["chosen_ratio"]
        eq = logs[c]["equity_nc"].values
        a = (k - 1) * len(eq) // n_folds
        b = k * len(eq) // n_folds
        seg = eq[a:b]
        if len(seg) >= 2:
            r = np.diff(seg) / seg[:-1]
            r = r[np.isfinite(r)]
            concat_chosen.append(r)
    oos_sharpe_chosen = _sharpe(np.concatenate(concat_chosen)) if concat_chosen else 0.0

    # 5. Baseline OOS Sharpe: ratio=0.0 on every fold.
    baseline = candidates[0]  # 0.0
    baseline_per_fold = per_candidate_fold_sharpes[baseline]
    eq_baseline = logs[baseline]["equity_nc"].values
    concat_baseline: list[np.ndarray] = []
    for k in range(2, n_folds + 1):
        a = (k - 1) * len(eq_baseline) // n_folds
        b = k * len(eq_baseline) // n_folds
        seg = eq_baseline[a:b]
        if len(seg) >= 2:
            r = np.diff(seg) / seg[:-1]
            r = r[np.isfinite(r)]
            concat_baseline.append(r)
    oos_sharpe_baseline = _sharpe(np.concatenate(concat_baseline)) if concat_baseline else 0.0

    return {
        "per_candidate_fold_sharpes": per_candidate_fold_sharpes,
        "cv_table": cv_rows,
        "concat_oos_sharpe_chosen": oos_sharpe_chosen,
        "concat_oos_sharpe_baseline": oos_sharpe_baseline,
        "baseline_per_fold": baseline_per_fold,
    }


def format_report(results: dict) -> str:
    """Human-readable walk-forward report."""
    lines: list[str] = []
    lines.append("=" * 76)
    lines.append("Phase 2 Walk-Forward CV — profit_filter_min_ratio")
    lines.append("=" * 76)
    lines.append("")
    lines.append("Per-candidate per-fold NC Sharpe (annualised, bar-returns):")
    header = f"  {'ratio':>6}"
    for f in range(1, N_FOLDS + 1):
        header += f"   f{f:1d}"
    lines.append(header)
    for c, arr in results["per_candidate_fold_sharpes"].items():
        row = f"  {c:>6.2f}"
        for s in arr:
            row += f" {s:>5.1f}"
        lines.append(row)
    lines.append("")
    lines.append("Walk-forward CV decisions:")
    lines.append(
        f"  {'fold':>4} {'chosen':>8} {'cal_sh':>8} {'test_sh':>8}  "
        f"| per-candidate test fold Sharpes"
    )
    for row in results["cv_table"]:
        cand_str = " ".join(
            f"{c:.2f}:{s:+.1f}" for c, s in row["test_sharpes_by_candidate"].items()
        )
        lines.append(
            f"  {row['fold']:>4} {row['chosen_ratio']:>8.2f} "
            f"{row['calibration_sharpe']:>8.2f} {row['test_sharpe']:>8.2f}  | {cand_str}"
        )
    lines.append("")
    lines.append("Summary (folds 2..8 concatenated, NC bar returns):")
    lines.append(f"  Chosen-param OOS Sharpe:  {results['concat_oos_sharpe_chosen']:>6.2f}")
    lines.append(f"  Baseline (ratio=0) OOS:   {results['concat_oos_sharpe_baseline']:>6.2f}")
    delta = results["concat_oos_sharpe_chosen"] - results["concat_oos_sharpe_baseline"]
    lines.append(f"  Δ:                        {delta:>+6.2f}")
    lines.append("")
    # Accept-gate evaluation per plan.
    lines.append("Plan accept-gate evaluation:")
    gate = delta >= 0.30
    lines.append(
        f"  (1) OOS Sharpe ≥ baseline + 0.3: {'PASS' if gate else 'FAIL'} "
        f"(Δ = {delta:+.2f})"
    )
    # (2) Per-fold drop check.
    max_drop = max(
        results["baseline_per_fold"][k - 1] - row["test_sharpe"]
        for k, row in enumerate(results["cv_table"], start=2)
    )
    gate2 = max_drop < 0.5
    lines.append(
        f"  (2) No fold drops >0.5 vs baseline: {'PASS' if gate2 else 'FAIL'} "
        f"(max drop = {max_drop:+.2f})"
    )
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("ratio", "threshold", "both"), default="ratio",
        help="Which parameter to sweep: 'ratio' for profit_filter_min_ratio, "
             "'threshold' for (entry_threshold_long, entry_threshold), or both.",
    )
    args = parser.parse_args()

    output_dir = Path("vol_arb/output")

    if args.mode in ("ratio", "both"):
        print("\n## Sweep 1 — profit_filter_min_ratio\n")
        results = run_walk_forward(output_dir=output_dir)
        print(format_report(results))
        rows = []
        for row in results["cv_table"]:
            flat = {
                "fold": row["fold"],
                "chosen_ratio": row["chosen_ratio"],
                "calibration_sharpe": row["calibration_sharpe"],
                "test_sharpe": row["test_sharpe"],
            }
            for c, s in row["test_sharpes_by_candidate"].items():
                flat[f"sharpe_at_ratio_{int(round(c*100)):03d}"] = s
            rows.append(flat)
        out_csv = output_dir / "phase2_wf_cv_results.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False)
        print(f"\nSaved CV table: {out_csv}")

    if args.mode in ("threshold", "both"):
        print("\n## Sweep 2 — (entry_threshold_long, entry_threshold)\n")
        results_thr = run_walk_forward_thresholds(output_dir=output_dir)
        print(format_threshold_report(results_thr))
        rows_thr = []
        for row in results_thr["cv_table"]:
            flat = {
                "fold": row["fold"],
                "chosen_thr_long": row["chosen_pair"][0],
                "chosen_thr_short": row["chosen_pair"][1],
                "calibration_sharpe": row["calibration_sharpe"],
                "test_sharpe": row["test_sharpe"],
            }
            for p, s in row["test_sharpes_by_candidate"].items():
                flat[f"sharpe_L{int(p[0]):02d}_S{int(p[1]):02d}"] = s
            rows_thr.append(flat)
        out_csv_thr = output_dir / "phase2_wf_cv_threshold_results.csv"
        pd.DataFrame(rows_thr).to_csv(out_csv_thr, index=False)
        print(f"\nSaved threshold CV table: {out_csv_thr}")


if __name__ == "__main__":
    main()
