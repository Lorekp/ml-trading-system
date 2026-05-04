"""V2 epistemic-uncertainty signal-quality audit (Cycle 3 Phase R).

Five measurement-only probes on the frozen K=10 deep-ensemble's per-bar
σ²_epi (artifact: ``vol_model/output/BTC_v2_har_pd_ensemble_var.parquet``)
joined to the canonical V2 trade log. Pre-gate threshold:
``Spearman(σ²_epi, |y_true - y_pred|) >= 0.20`` on the eval window. If
the pre-gate fails, Phase P is HALTED — documenting the negative finding
is the deliverable.

All probes are leakage-safe by construction (σ²_epi is OOF from
PurgedKFold + embargo; no future data informs any per-bar value).

Output: ``vol_arb/output/V2_EPISTEMIC_UNCERTAINTY_AUDIT.md``.

Run::

    PYTHONPATH=. python scripts/forensic/audit_epistemic_uncertainty.py
"""
from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("audit_epistemic_uncertainty")


# ─── Pre-committed parameters (locked in plan) ─────────────────────────

SIGNAL_QUALITY_PREGATE: float = 0.20      # Spearman threshold on eval window
HQR_COVERAGE_LO: float = 0.85
HQR_COVERAGE_HI: float = 0.95

CAL_WINDOW: tuple[pd.Timestamp, pd.Timestamp] = (
    pd.Timestamp("2021-04-01", tz="UTC"),
    pd.Timestamp("2023-10-01", tz="UTC"),
)
EVAL_WINDOW_START = pd.Timestamp("2023-10-01", tz="UTC")

QUARANTINE_DAYS: tuple[pd.Timestamp, ...] = (
    pd.Timestamp("2022-06-13", tz="UTC"),
    pd.Timestamp("2022-11-08", tz="UTC"),
    pd.Timestamp("2024-08-05", tz="UTC"),
    pd.Timestamp("2025-01-20", tz="UTC"),
    pd.Timestamp("2024-07-26", tz="UTC"),
    pd.Timestamp("2024-11-05", tz="UTC"),
)

ENSEMBLE_PARQUET = Path("vol_model/output/BTC_v2_har_pd_ensemble_var.parquet")
TRADE_LOG_PATH = Path("vol_arb/output/BTC_v2_trade_log.parquet")
OUTPUT_PATH = Path("vol_arb/output/V2_EPISTEMIC_UNCERTAINTY_AUDIT.md")


@dataclass
class ProbeResult:
    name: str
    metric: float
    threshold: float | None
    pass_fail: str | None  # "PASS" / "FAIL" / None for no-gate probes
    detail: list[str]


def _drop_quarantine(df: pd.DataFrame | pd.Series) -> pd.DataFrame | pd.Series:
    """Drop quarantine UTC dates."""
    if isinstance(df, pd.Series):
        return df[~df.index.normalize().isin(QUARANTINE_DAYS)]
    return df.loc[~df.index.normalize().isin(QUARANTINE_DAYS)]


def probe_signal_quality_spearman(
    sigma_epi_sq: pd.Series, y_true: pd.Series, y_pred: pd.Series,
) -> dict[str, ProbeResult]:
    """Spearman(σ²_epi, |y_true - y_pred|) on cal AND eval windows.

    Pre-gate: eval-window Spearman >= 0.20.
    """
    abs_err = (y_true - y_pred).abs()

    cal_start, cal_end = CAL_WINDOW
    cal_mask = ((sigma_epi_sq.index >= cal_start)
                & (sigma_epi_sq.index < cal_end))
    eval_mask = sigma_epi_sq.index >= cal_end

    aligned = pd.concat({"epi": sigma_epi_sq, "abs_err": abs_err},
                         axis=1).dropna()

    cal_aligned = aligned.loc[(aligned.index >= cal_start)
                                & (aligned.index < cal_end)]
    eval_aligned = aligned.loc[aligned.index >= cal_end]

    cal_aligned = _drop_quarantine(cal_aligned)
    eval_aligned = _drop_quarantine(eval_aligned)

    cal_rho = float("nan")
    cal_p = float("nan")
    if len(cal_aligned) >= 100:
        cal_rho, cal_p = spearmanr(
            cal_aligned["epi"].values, cal_aligned["abs_err"].values,
        )
        cal_rho = float(cal_rho)
        cal_p = float(cal_p)

    eval_rho = float("nan")
    eval_p = float("nan")
    if len(eval_aligned) >= 100:
        eval_rho, eval_p = spearmanr(
            eval_aligned["epi"].values, eval_aligned["abs_err"].values,
        )
        eval_rho = float(eval_rho)
        eval_p = float(eval_p)

    pass_fail = "PASS" if eval_rho >= SIGNAL_QUALITY_PREGATE else "FAIL"
    return {
        "cal": ProbeResult(
            name="signal_quality_cal",
            metric=cal_rho, threshold=None, pass_fail=None,
            detail=[
                f"Spearman(σ²_epi, |y_true-y_pred|) = {cal_rho:+.4f} "
                f"(p={cal_p:.4g}, n={len(cal_aligned)})",
                "Calibration window (no gate; reported for context)",
            ],
        ),
        "eval": ProbeResult(
            name="signal_quality_eval",
            metric=eval_rho, threshold=SIGNAL_QUALITY_PREGATE,
            pass_fail=pass_fail,
            detail=[
                f"Spearman(σ²_epi, |y_true-y_pred|) = {eval_rho:+.4f} "
                f"(p={eval_p:.4g}, n={len(eval_aligned)})",
                f"Pre-gate: ≥ +{SIGNAL_QUALITY_PREGATE:.2f} → {pass_fail}",
            ],
        ),
    }


def probe_signal_quality_pnl(
    sigma_epi_sq: pd.Series, trade_log: pd.DataFrame,
) -> ProbeResult:
    """Spearman(σ²_epi, |trade NC PnL|) on FIRED-trade subset."""
    fired = trade_log[trade_log["net_pnl_nc"] != 0]
    fired_aligned = sigma_epi_sq.reindex(fired.index).dropna()
    pnl = fired["net_pnl_nc"].reindex(fired_aligned.index).abs()
    pnl_aligned = pnl.dropna()
    common = fired_aligned.index.intersection(pnl_aligned.index)
    if len(common) < 100:
        return ProbeResult(
            name="signal_quality_pnl",
            metric=float("nan"), threshold=None, pass_fail=None,
            detail=[f"too few aligned fired bars ({len(common)})"],
        )
    rho, p = spearmanr(
        fired_aligned.loc[common].values, pnl_aligned.loc[common].values,
    )
    return ProbeResult(
        name="signal_quality_pnl",
        metric=float(rho), threshold=None, pass_fail=None,
        detail=[
            f"Spearman(σ²_epi, |NC PnL|) = {float(rho):+.4f} "
            f"(p={float(p):.4g}, n={len(common)} fired trades)",
            "Reports correlation; no gate.",
        ],
    )


def probe_regime_uncertainty(
    sigma_epi_sq: pd.Series, trade_log: pd.DataFrame,
) -> ProbeResult:
    """Per-cell σ²_epi mean by frozen (DVOL × jump) tercile cuts.

    Reuses the Cycle 2 calibration-window terciles. No gate — reports
    structure for transparency.
    """
    if "dvol_close" not in trade_log.columns or "jump_fraction" not in trade_log.columns:
        return ProbeResult(
            name="regime_uncertainty",
            metric=float("nan"), threshold=None, pass_fail=None,
            detail=["dvol_close or jump_fraction missing from trade log"],
        )
    fired = trade_log[trade_log["net_pnl_nc"] != 0]
    cal_slice = fired.loc[
        (fired.index >= CAL_WINDOW[0]) & (fired.index < CAL_WINDOW[1])
    ]
    if len(cal_slice) < 100:
        return ProbeResult(
            name="regime_uncertainty",
            metric=float("nan"), threshold=None, pass_fail=None,
            detail=[f"calibration slice too small ({len(cal_slice)})"],
        )
    dvol_low = float(cal_slice["dvol_close"].quantile(0.33))
    dvol_high = float(cal_slice["dvol_close"].quantile(0.67))
    jump_low = float(cal_slice["jump_fraction"].quantile(0.33))
    jump_high = float(cal_slice["jump_fraction"].quantile(0.67))

    def _label(v, lo, hi):
        if pd.isna(v):
            return "na"
        if v <= lo:
            return "low"
        if v >= hi:
            return "high"
        return "mid"

    fired = fired.copy()
    fired["dvol_t"] = fired["dvol_close"].apply(lambda v: _label(v, dvol_low, dvol_high))
    fired["jump_t"] = fired["jump_fraction"].apply(lambda v: _label(v, jump_low, jump_high))

    sigma_at_fired = sigma_epi_sq.reindex(fired.index)

    detail_lines = [
        f"Calibration tercile cuts: DVOL [{dvol_low:.2f}, {dvol_high:.2f}], "
        f"Jump [{jump_low:.4f}, {jump_high:.4f}]",
        "Per-cell mean σ²_epi (n=count of fired trades):",
        "",
        "| DVOL\\Jump | low | mid | high |",
        "|---|---|---|---|",
    ]
    for dl in ("low", "mid", "high"):
        row = [f"**{dl}**"]
        for jl in ("low", "mid", "high"):
            cell_idx = (fired["dvol_t"] == dl) & (fired["jump_t"] == jl)
            cell_sigma = sigma_at_fired[cell_idx].dropna()
            if len(cell_sigma) == 0:
                row.append("— (n=0)")
            else:
                row.append(
                    f"{float(cell_sigma.mean()):.6f} (n={len(cell_sigma)})"
                )
        detail_lines.append("| " + " | ".join(row) + " |")

    return ProbeResult(
        name="regime_uncertainty",
        metric=float(sigma_at_fired.mean()),
        threshold=None, pass_fail=None,
        detail=detail_lines,
    )


def probe_hqr_coverage(uncertainty_df: pd.DataFrame) -> ProbeResult:
    """Empirical 90% PI coverage on eval window. Pre-gate: in [0.85, 0.95]."""
    eval_mask = uncertainty_df.index >= EVAL_WINDOW_START
    eval_slice = uncertainty_df[eval_mask].dropna(
        subset=["pi_lo", "pi_hi", "y_true"],
    )
    eval_slice = _drop_quarantine(eval_slice)
    if len(eval_slice) < 100:
        return ProbeResult(
            name="hqr_coverage",
            metric=float("nan"), threshold=None, pass_fail="FAIL",
            detail=[f"eval slice too small ({len(eval_slice)})"],
        )
    in_pi = ((eval_slice["y_true"] >= eval_slice["pi_lo"])
             & (eval_slice["y_true"] <= eval_slice["pi_hi"]))
    cov = float(in_pi.mean())
    pass_fail = "PASS" if HQR_COVERAGE_LO <= cov <= HQR_COVERAGE_HI else "FAIL"
    return ProbeResult(
        name="hqr_coverage",
        metric=cov, threshold=HQR_COVERAGE_LO,
        pass_fail=pass_fail,
        detail=[
            f"Empirical 90% PI coverage on eval window: {cov:.4f} "
            f"(n={len(eval_slice)})",
            f"Pre-gate: in [{HQR_COVERAGE_LO}, {HQR_COVERAGE_HI}] → {pass_fail}",
        ],
    )


def probe_orthogonality_to_y_pred(
    sigma_epi_sq: pd.Series, y_pred: pd.Series, y_true: pd.Series,
) -> ProbeResult:
    """Does σ²_epi carry information BEYOND |residual|?

    Spearman(σ²_epi, residual_signed) on eval window. If σ²_epi is
    perfectly redundant with the model's residual, this should be near
    zero (since the residual sign is independent of the magnitude of
    epistemic uncertainty under a properly calibrated estimator).
    No gate; reports orthogonality for diagnostic interpretation.
    """
    eval_mask = sigma_epi_sq.index >= EVAL_WINDOW_START
    aligned = pd.concat({
        "epi": sigma_epi_sq[eval_mask],
        "resid": (y_true - y_pred)[eval_mask],
    }, axis=1).dropna()
    aligned = _drop_quarantine(aligned)
    if len(aligned) < 100:
        return ProbeResult(
            name="orthogonality_to_y_pred",
            metric=float("nan"), threshold=None, pass_fail=None,
            detail=[f"eval slice too small ({len(aligned)})"],
        )
    rho, p = spearmanr(aligned["epi"].values, aligned["resid"].values)
    return ProbeResult(
        name="orthogonality_to_y_pred",
        metric=float(rho), threshold=None, pass_fail=None,
        detail=[
            f"Spearman(σ²_epi, signed residual) = {float(rho):+.4f} "
            f"(p={float(p):.4g}, n={len(aligned)})",
            "Near-zero indicates σ²_epi is orthogonal to residual sign.",
            "Large |ρ| would indicate σ²_epi is correlated with the "
            "direction of error, suggesting it's not purely epistemic.",
        ],
    )


def render_markdown(
    results: dict, n_bars: int, n_eval_bars: int,
) -> str:
    sq_eval = results["signal_quality"]["eval"]
    pre_gate = sq_eval.pass_fail
    lines = [
        "# V2 Epistemic Uncertainty Audit (Cycle 3 Phase R)",
        "",
        f"K=10 deep-ensemble σ²_epi joined to canonical V2 trade log "
        f"({n_bars:,} bars, {n_eval_bars:,} on eval window).",
        "",
        "## Signal-quality pre-gate",
        "",
        f"**{pre_gate}** — {sq_eval.detail[1]}",
        "",
        "## Probes",
        "",
        "### Signal quality (Spearman ρ between σ²_epi and absolute error)",
        "",
        *[f"- {d}" for d in results["signal_quality"]["cal"].detail],
        *[f"- {d}" for d in results["signal_quality"]["eval"].detail],
        "",
        "### PnL correlation (fired-trade subset, no gate)",
        "",
        *[f"- {d}" for d in results["pnl_corr"].detail],
        "",
        "### Regime-conditional σ²_epi (frozen calibration terciles, no gate)",
        "",
        *results["regime"].detail,
        "",
        "### HQR conformal coverage (90% PI empirical, pre-gate [0.85, 0.95])",
        "",
        *[f"- {d}" for d in results["hqr_coverage"].detail],
        "",
        "### Orthogonality to signed residual (no gate, diagnostic only)",
        "",
        *[f"- {d}" for d in results["ortho"].detail],
        "",
        "## Verdict",
        "",
    ]
    if pre_gate == "PASS":
        lines += [
            "**SIGNAL-QUALITY PRE-GATE PASS.** Phase P is authorised to "
            "proceed: σ²_epi shows ≥ +0.20 Spearman correlation with "
            "absolute error on the eval window — indicating epistemic "
            "uncertainty carries non-redundant economic information.",
            "",
            "Next: run `scripts/generate_uncertainty_overlay_sidecars.py` "
            "to materialise the 5 pre-committed strategies (F1, F2, F3, "
            "S1, S2) and `scripts/evaluate_uncertainty_overlay_gate.py` "
            "for the triple-conjunctive gate.",
        ]
    else:
        lines += [
            "**SIGNAL-QUALITY PRE-GATE FAIL.** Phase P is HALTED.",
            "",
            "The deep-ensemble's per-bar σ²_epi does not correlate with "
            "absolute error at the pre-committed +0.20 Spearman threshold "
            "on the evaluation window. Per the plan's anti-overfitting "
            "discipline, lowering the gate to match the observed value "
            "is forbidden — that would be the canonical storytelling-bias "
            "failure mode. Documenting the negative finding IS the "
            "deliverable; epistemic-uncertainty overlays are not built.",
            "",
            "Possible interpretations (for future reference, not action):",
            "1. Aleatoric noise dominates epistemic uncertainty for "
            "   intraday log-vol forecasting at this horizon.",
            "2. K=10 is insufficient for stable ensemble variance — "
            "   would require K=20+ or alternative epistemic estimator.",
            "3. The base model's residuals are NOT structurally tied to "
            "   ensemble disagreement — i.e., epistemic uncertainty as "
            "   measured here is qualitatively different from prediction "
            "   error in our setting.",
        ]
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ensemble", type=Path, default=ENSEMBLE_PARQUET)
    parser.add_argument("--trade-log", type=Path, default=TRADE_LOG_PATH)
    parser.add_argument("--out", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    if not args.ensemble.exists():
        raise FileNotFoundError(
            f"Ensemble artifact missing: {args.ensemble}. Run "
            "scripts/train_har_pd_ensemble.py first."
        )

    logger.info("Loading ensemble artifact %s", args.ensemble)
    uncert = pd.read_parquet(args.ensemble)
    if not isinstance(uncert.index, pd.DatetimeIndex):
        uncert.index = pd.to_datetime(uncert.index, utc=True)
    logger.info("Ensemble: %d bars, span %s -> %s",
                len(uncert), uncert.index.min(), uncert.index.max())

    logger.info("Loading trade log %s", args.trade_log)
    tl = pd.read_parquet(args.trade_log)
    if not isinstance(tl.index, pd.DatetimeIndex):
        tl.index = pd.to_datetime(tl.index, utc=True)

    sigma_epi_sq = (uncert["sigma_epi"] ** 2).rename("sigma_epi_sq")
    y_true = uncert["y_true"]
    y_pred = uncert["ensemble_mean"]

    logger.info("Running probes...")
    sq = probe_signal_quality_spearman(sigma_epi_sq, y_true, y_pred)
    pnl = probe_signal_quality_pnl(sigma_epi_sq, tl)
    regime = probe_regime_uncertainty(sigma_epi_sq, tl)
    coverage = probe_hqr_coverage(uncert)
    ortho = probe_orthogonality_to_y_pred(sigma_epi_sq, y_pred, y_true)

    results = {
        "signal_quality": sq,
        "pnl_corr": pnl,
        "regime": regime,
        "hqr_coverage": coverage,
        "ortho": ortho,
    }

    n_bars = len(uncert)
    n_eval_bars = int((uncert.index >= EVAL_WINDOW_START).sum())

    report = render_markdown(results, n_bars, n_eval_bars)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(report, encoding="utf-8")
    logger.info("Wrote audit report to %s", args.out)

    print()
    print("=" * 70)
    print(" V2 EPISTEMIC UNCERTAINTY AUDIT")
    print("=" * 70)
    print(f"  Signal quality cal:  rho={sq['cal'].metric:+.4f}")
    print(f"  Signal quality eval: rho={sq['eval'].metric:+.4f} "
          f"(gate >= +{SIGNAL_QUALITY_PREGATE}) {sq['eval'].pass_fail}")
    print(f"  PnL correlation:     rho={pnl.metric:+.4f}")
    print(f"  HQR coverage:        {coverage.metric:.4f} "
          f"(gate [{HQR_COVERAGE_LO}, {HQR_COVERAGE_HI}]) {coverage.pass_fail}")
    print(f"  Orthogonality:       rho={ortho.metric:+.4f}")
    print(f"  Pre-gate verdict:    {sq['eval'].pass_fail}")
    print("=" * 70)
    print(f"  Full report: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
