#!/usr/bin/env python
"""Compare pruned_svi_har_pd_v1 predictions vs pruned_svi_v1 baseline.

Evaluates the certification gate for the HAR-PD augmentation (Session C
conditional shortlist → full 50-month walk-forward):

  (1) avg ΔQLIKE ≤ -3% vs pruned_svi_v1 baseline
  (2) ≥ 25/50 months improved (Δ < 0)
  (3) ≥ 15/50 months DM-significant at p<0.01 (improvement side)

Per-month QLIKE is computed on the OOS test window (identical per-month
splits between the two models; rows aligned by timestamp intersection to
avoid population drift). Diebold-Mariano test uses Newey-West HAC variance
with lag = 4 (15m bars, ~1h autocorrelation decay).

Inputs:
  - vol_model/output/BTCUSDT_pruned_svi_predictions.parquet          (baseline)
  - vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions.parquet   (candidate)

Both parquets must have columns {y_pred, y_true} and a DatetimeIndex. y_true
values are compared for identity post-merge; any drift triggers a warning
because the two models must share the exact same test set for fair DM.

Usage:
    PYTHONPATH=. python vol_model/compare_pruned_svi_har_pd_vs_baseline.py
    PYTHONPATH=. python vol_model/compare_pruned_svi_har_pd_vs_baseline.py \\
        --baseline vol_model/output/BTCUSDT_pruned_svi_predictions.parquet \\
        --candidate vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("compare_har_pd")


def _qlike_per_row(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Per-row QLIKE loss (Patton 2011): QLIKE(σ², ŝ²) = σ²/ŝ² − log(σ²/ŝ²) − 1.

    Here y_true and y_pred are log-variance (the model target), so we
    exponentiate to get variance before computing the loss. Rows with
    non-finite or non-positive variance are dropped by the caller.
    """
    var_true = np.exp(y_true)
    var_pred = np.exp(y_pred)
    r = var_true / var_pred
    return r - np.log(r) - 1.0


def _diebold_mariano_pvalue(d: np.ndarray, lag: int = 4) -> tuple[float, float]:
    """One-sided Diebold-Mariano test statistic and p-value for H0: E[d]=0 vs
    H1: E[d]<0 (i.e., candidate's loss is less than baseline's, so d=L_c−L_b
    has negative mean → candidate wins).

    Uses Newey-West HAC variance with truncation lag ``lag``.
    Returns (dm_stat, pvalue_lower_tail).
    """
    from scipy.stats import norm

    n = len(d)
    if n < 30:
        return (np.nan, np.nan)
    d_mean = float(np.mean(d))
    # Newey-West HAC variance
    gamma_0 = float(np.var(d, ddof=1))
    s2 = gamma_0
    for h in range(1, lag + 1):
        gamma_h = float(np.mean((d[h:] - d_mean) * (d[:-h] - d_mean)))
        s2 += 2.0 * (1.0 - h / (lag + 1)) * gamma_h
    if s2 <= 0:
        return (np.nan, np.nan)
    dm = d_mean / np.sqrt(s2 / n)
    # Lower-tail p (candidate improves iff d_mean < 0)
    p = float(norm.cdf(dm))
    return (float(dm), p)


def compare_predictions(
    baseline_df: pd.DataFrame,
    candidate_df: pd.DataFrame,
) -> pd.DataFrame:
    """Return a per-month comparison table with QLIKE + DM p-value + Spearman.

    Rows with non-finite y_pred or y_true (in either model) are dropped.
    Months are derived from the DatetimeIndex; any month present in only one
    of the two input dataframes is skipped (with a warning)."""
    base = baseline_df.copy()
    cand = candidate_df.copy()

    # Align by timestamp intersection — both models must cover the same rows
    common = base.index.intersection(cand.index)
    n_drop_base = len(base) - len(common)
    n_drop_cand = len(cand) - len(common)
    if n_drop_base or n_drop_cand:
        logger.warning(
            "Index drift: dropped %d rows from baseline, %d from candidate. "
            "Aligned: %d rows.",
            n_drop_base, n_drop_cand, len(common),
        )
    base = base.loc[common].sort_index()
    cand = cand.loc[common].sort_index()

    # y_true must match (same underlying label source)
    y_true_diff = np.abs(base["y_true"].values - cand["y_true"].values).max()
    if y_true_diff > 1e-9:
        logger.warning(
            "y_true max-abs-diff across the two prediction files: %.3e. "
            "This is SUSPICIOUS — both models should see the same labels. "
            "Continuing with the baseline's y_true.",
            y_true_diff,
        )
    y_true = base["y_true"].values
    y_pred_b = base["y_pred"].values
    y_pred_c = cand["y_pred"].values

    # Per-month rollup
    months = pd.Series(base.index).dt.strftime("%Y-%m")
    months.index = base.index
    rows = []
    for mo, mo_idx in months.groupby(months):
        idx = mo_idx.index
        mask = np.isfinite(y_true) & np.isfinite(y_pred_b) & np.isfinite(y_pred_c)
        row_mask = base.index.isin(idx) & mask
        if row_mask.sum() < 100:
            rows.append({
                "month": mo,
                "n": int(row_mask.sum()),
                "qlike_baseline": np.nan,
                "qlike_candidate": np.nan,
                "delta_qlike_pct": np.nan,
                "dm_stat": np.nan,
                "dm_p": np.nan,
                "spearman_baseline": np.nan,
                "spearman_candidate": np.nan,
            })
            continue
        yt = y_true[row_mask]
        yp_b = y_pred_b[row_mask]
        yp_c = y_pred_c[row_mask]

        # Per-row QLIKE
        q_b = _qlike_per_row(yt, yp_b)
        q_c = _qlike_per_row(yt, yp_c)
        finite = np.isfinite(q_b) & np.isfinite(q_c)
        q_b = q_b[finite]
        q_c = q_c[finite]

        qm_b = float(np.mean(q_b))
        qm_c = float(np.mean(q_c))
        # Δ relative to baseline, expressed as percent (negative = improvement)
        delta_pct = 100.0 * (qm_c - qm_b) / qm_b if qm_b > 0 else np.nan

        # Diebold-Mariano on d = q_c − q_b
        d = q_c - q_b
        dm, dm_p = _diebold_mariano_pvalue(d, lag=4)

        # Spearman (NumPy; both models)
        from scipy.stats import spearmanr
        sp_b = float(spearmanr(yt, yp_b).statistic) if len(yt) > 30 else np.nan
        sp_c = float(spearmanr(yt, yp_c).statistic) if len(yt) > 30 else np.nan

        rows.append({
            "month": mo,
            "n": int(row_mask.sum()),
            "qlike_baseline": qm_b,
            "qlike_candidate": qm_c,
            "delta_qlike_pct": delta_pct,
            "dm_stat": dm,
            "dm_p": dm_p,
            "spearman_baseline": sp_b,
            "spearman_candidate": sp_c,
        })
    return pd.DataFrame(rows)


def certification_gate(table: pd.DataFrame) -> dict:
    """Evaluate the three Session-C certification gates against a per-month
    comparison table."""
    table = table.dropna(subset=["delta_qlike_pct"])
    n_months = len(table)

    avg_delta = float(table["delta_qlike_pct"].mean()) if n_months else np.nan
    n_improved = int((table["delta_qlike_pct"] < 0).sum())
    n_dm_sig_improved = int(((table["dm_p"] < 0.01) & (table["delta_qlike_pct"] < 0)).sum())

    gate1 = avg_delta <= -3.0
    gate2 = n_improved >= 25 if n_months >= 25 else n_improved >= n_months / 2
    gate3 = n_dm_sig_improved >= 15 if n_months >= 30 else n_dm_sig_improved >= 3

    return {
        "n_months": n_months,
        "avg_delta_qlike_pct": avg_delta,
        "n_improved": n_improved,
        "n_dm_sig_improved": n_dm_sig_improved,
        "gate1_avg_delta_le_-3pct": bool(gate1),
        "gate2_n_improved_ge_25": bool(gate2),
        "gate3_n_dm_sig_ge_15": bool(gate3),
        "all_gates_pass": bool(gate1 and gate2 and gate3),
    }


def format_report(table: pd.DataFrame, gates: dict) -> str:
    lines = []
    lines.append("=" * 86)
    lines.append("  Phase 5 HAR-PD certification — pruned_svi_har_pd_v1 vs pruned_svi_v1")
    lines.append("=" * 86)
    lines.append("")
    lines.append(
        f"  {'month':>8}  {'n':>6}  {'qlike_base':>11}  {'qlike_cand':>11}  "
        f"{'Δ%':>6}  {'DM_stat':>8}  {'DM_p':>8}  {'rank_b':>7}  {'rank_c':>7}"
    )
    for _, r in table.iterrows():
        delta_str = f"{r['delta_qlike_pct']:>+5.2f}%" if np.isfinite(r["delta_qlike_pct"]) else "  n/a "
        dm_str = f"{r['dm_stat']:>+7.3f}" if np.isfinite(r["dm_stat"]) else "  n/a "
        dmp_str = f"{r['dm_p']:>7.4f}" if np.isfinite(r["dm_p"]) else "  n/a"
        qb = f"{r['qlike_baseline']:>10.5f}" if np.isfinite(r["qlike_baseline"]) else " n/a"
        qc = f"{r['qlike_candidate']:>10.5f}" if np.isfinite(r["qlike_candidate"]) else " n/a"
        rb = f"{r['spearman_baseline']:>+6.3f}" if np.isfinite(r["spearman_baseline"]) else "  n/a "
        rc = f"{r['spearman_candidate']:>+6.3f}" if np.isfinite(r["spearman_candidate"]) else "  n/a "
        lines.append(
            f"  {r['month']:>8}  {r['n']:>6}  {qb:>11}  {qc:>11}  "
            f"{delta_str:>6}  {dm_str:>8}  {dmp_str:>8}  {rb:>7}  {rc:>7}"
        )
    lines.append("")
    lines.append("Certification gate:")
    lines.append(
        f"  n months:          {gates['n_months']}"
    )
    lines.append(
        f"  avg ΔQLIKE:        {gates['avg_delta_qlike_pct']:+.2f}%   "
        f"(gate1: ≤ -3% → {'PASS' if gates['gate1_avg_delta_le_-3pct'] else 'FAIL'})"
    )
    lines.append(
        f"  n improved:        {gates['n_improved']}/{gates['n_months']}   "
        f"(gate2: ≥ 25 → {'PASS' if gates['gate2_n_improved_ge_25'] else 'FAIL'})"
    )
    lines.append(
        f"  n DM-sig improved: {gates['n_dm_sig_improved']}/{gates['n_months']}   "
        f"(gate3: ≥ 15 at p<0.01 → {'PASS' if gates['gate3_n_dm_sig_ge_15'] else 'FAIL'})"
    )
    lines.append("")
    verdict = "SHIP — all gates pass" if gates["all_gates_pass"] else "DO NOT SHIP — gate(s) failed"
    lines.append(f"  Verdict: {verdict}")
    lines.append("=" * 86)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--baseline",
        default="vol_model/output/BTCUSDT_pruned_svi_predictions.parquet",
    )
    parser.add_argument(
        "--candidate",
        default="vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions.parquet",
    )
    parser.add_argument(
        "--output-csv",
        default="vol_model/output/phase5_har_pd_cert_table.csv",
        help="Per-month comparison CSV (absolute path ok).",
    )
    args = parser.parse_args()

    base_path = Path(args.baseline)
    cand_path = Path(args.candidate)
    if not base_path.exists():
        raise FileNotFoundError(f"Baseline predictions not found: {base_path}")
    if not cand_path.exists():
        raise FileNotFoundError(f"Candidate predictions not found: {cand_path}")

    logger.info("Loading baseline predictions from %s", base_path)
    baseline_df = pd.read_parquet(base_path)
    logger.info("Loading candidate predictions from %s", cand_path)
    candidate_df = pd.read_parquet(cand_path)

    required = {"y_pred", "y_true"}
    for label, df in [("baseline", baseline_df), ("candidate", candidate_df)]:
        missing = required - set(df.columns)
        if missing:
            raise KeyError(f"{label} missing columns: {missing}")

    table = compare_predictions(baseline_df, candidate_df)
    gates = certification_gate(table)

    print(format_report(table, gates))

    out_csv = Path(args.output_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(out_csv, index=False)
    print(f"\nPer-month CSV saved: {out_csv}")


if __name__ == "__main__":
    main()
