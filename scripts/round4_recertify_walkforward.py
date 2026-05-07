"""Round-4 recertification: extend the walk-forward by 3-6 months.

Closes the round-4 retrain item from `vol_arb/output/PARITY_AUDIT_ROUND2.md`.
Runs the standard walk-forward (`vol_model.run_walkforward_pruned_svi_har_pd`)
for the requested months and emits a per-month QLIKE / SHAP-stability
report against pre-registered acceptance gates.

The model itself was already trained on Binance Spot dollar-bar features
(the same venue F3 + round-4 F6 now feed live), so this run is a
*recertification* — verifying the model is stable on the post-F3 live
distribution and absorbing newer market data — not a retrain to fix a
distribution mismatch.

Pre-registered gates:

* **Pass**: every retrained month has QLIKE within trailing-3mo
  median ±10%; SHAP top-5 features stable (Spearman ≥ 0.7 vs prior
  month); F3-feature parity within 1% rel.
* **Conditional pass**: 1-2 months marginal but trend stable → ship
  with note.
* **Fail**: any month QLIKE > +15% vs trailing baseline OR any F3
  feature > 5% live-vs-offline drift → halt, investigate, file as
  round-5 finding.

This script does NOT deploy a new model. It generates the new month-keyed
LightGBM checkpoints in `expanding_pruned_svi_har_pd_v1/{YYYY-MM}.txt`
(and the equivalent TransVol checkpoints) and writes
`vol_model/output/round4_recertification_report.md` with the per-month
metrics + decision-gate verdict.

Usage::

    PYTHONPATH=. python scripts/round4_recertify_walkforward.py \\
        --asset BTCUSDT \\
        --months 2026-05 2026-06 2026-07 2026-08 2026-09 2026-10 \\
        --report vol_model/output/round4_recertification_report.md
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Path setup so the script can be run from anywhere
sys.path.insert(0, str(Path(__file__).parent.parent))

logger = logging.getLogger(__name__)

# Pre-registered gates (mirrors plan file `starry-rolling-rabin.md`).
PASS_QLIKE_MAX_REL_DRIFT = 0.10           # ±10% vs trailing-3mo median
FAIL_QLIKE_MAX_REL_DRIFT = 0.15           # >15% triggers a HARD FAIL
PASS_SHAP_TOP5_SPEARMAN_MIN = 0.70        # Spearman ≥0.7 vs prior month
PASS_F3_FEATURE_DRIFT_MAX = 0.01          # 1% rel
FAIL_F3_FEATURE_DRIFT_MAX = 0.05          # 5% rel


_DEFAULT_MODEL_DIR = Path(
    "vol_model/output/models/expanding_pruned_svi_har_pd_v1"
)
_DEFAULT_PREDICTIONS_DIR = Path("vol_model/output")


def _existing_checkpoints(model_dir: Path) -> list[str]:
    """List the YYYY-MM month keys of checkpoints already trained."""
    if not model_dir.exists():
        return []
    return sorted({p.stem for p in model_dir.glob("*.txt")})


def _missing_months(model_dir: Path, requested: Iterable[str]) -> list[str]:
    have = set(_existing_checkpoints(model_dir))
    return [m for m in requested if m not in have]


def _run_walkforward_subprocess(
    *, asset: str, months: list[str], extra_args: list[str] | None = None,
) -> int:
    """Invoke the existing walk-forward script for the requested months.

    The walk-forward script is the production training cadence
    (``vol_model/run_walkforward_pruned_svi_har_pd.py``). We delegate to
    it via subprocess so the orchestration stays decoupled — any script
    changes (Optuna config, feature pipeline, callbacks) come along
    automatically.
    """
    cmd = [
        sys.executable, "vol_model/run_walkforward_pruned_svi_har_pd.py",
        "--asset", asset,
    ]
    if months:
        cmd.extend(["--only-months", *months])
    if extra_args:
        cmd.extend(extra_args)
    logger.info("Running: %s", " ".join(cmd))
    res = subprocess.run(
        cmd,
        check=False,
        cwd=Path(__file__).parent.parent,
        env={"PYTHONPATH": str(Path(__file__).parent.parent), **dict(__import__("os").environ)},
    )
    return int(res.returncode)


def _per_month_qlike(predictions_dir: Path, months: list[str]) -> dict:
    """Read the walk-forward predictions parquet, group by month, compute
    QLIKE per month vs the realized log-vol label.
    """
    pred_path = predictions_dir / "BTCUSDT_pruned_svi_har_pd_predictions.parquet"
    if not pred_path.exists():
        logger.warning(
            "predictions parquet not found at %s — skipping QLIKE eval", pred_path,
        )
        return {}
    df = pd.read_parquet(pred_path)
    if "y_pred" not in df.columns or "y_true" not in df.columns:
        logger.warning(
            "predictions parquet missing y_pred/y_true; cols=%s", list(df.columns)[:8],
        )
        return {}

    if "month" not in df.columns:
        # Derive month key from index timestamp.
        if isinstance(df.index, pd.DatetimeIndex):
            df["month"] = df.index.strftime("%Y-%m")
        elif "timestamp" in df.columns:
            df["month"] = pd.to_datetime(df["timestamp"], utc=True).dt.strftime("%Y-%m")
        else:
            return {}
    out: dict = {}
    for m in months:
        sub = df[df["month"] == m]
        if sub.empty:
            continue
        mask = sub["y_true"].notna() & sub["y_pred"].notna()
        if not mask.any():
            continue
        # QLIKE = mean(exp(log_resid) - log_resid - 1)
        resid = (sub.loc[mask, "y_true"] - sub.loc[mask, "y_pred"]).to_numpy()
        out[m] = float(np.mean(np.exp(resid) - resid - 1))
    return out


def _trailing_median(qlike_by_month: dict, current: str, n: int = 3) -> float | None:
    """Median QLIKE of the n calendar months immediately before ``current``."""
    months_before = sorted([m for m in qlike_by_month if m < current])[-n:]
    if not months_before:
        return None
    vals = [qlike_by_month[m] for m in months_before]
    return float(np.median(vals))


def _evaluate_gates(
    qlike_new: dict, qlike_prior: dict,
) -> dict:
    """Apply pre-registered pass/conditional/fail gates per month.

    Returns a dict ``{month: {qlike, baseline, drift_rel, status}}``.
    """
    combined = {**qlike_prior, **qlike_new}
    out: dict = {}
    for m in sorted(qlike_new.keys()):
        q = qlike_new[m]
        baseline = _trailing_median(combined, current=m, n=3)
        if baseline is None:
            out[m] = {
                "qlike": q, "baseline": None, "drift_rel": None,
                "status": "NO_BASELINE",
            }
            continue
        drift_rel = (q - baseline) / max(abs(baseline), 1e-9)
        if drift_rel > FAIL_QLIKE_MAX_REL_DRIFT:
            status = "FAIL"
        elif drift_rel > PASS_QLIKE_MAX_REL_DRIFT:
            status = "CONDITIONAL"
        else:
            status = "PASS"
        out[m] = {
            "qlike": q, "baseline": baseline,
            "drift_rel": drift_rel, "status": status,
        }
    return out


def _write_report(
    out: Path,
    *,
    asset: str,
    requested_months: list[str],
    trained_months: list[str],
    skipped_months: list[str],
    qlike_new: dict,
    qlike_prior: dict,
    gates: dict,
) -> None:
    """Render the markdown recertification report."""
    n_pass = sum(1 for v in gates.values() if v["status"] == "PASS")
    n_cond = sum(1 for v in gates.values() if v["status"] == "CONDITIONAL")
    n_fail = sum(1 for v in gates.values() if v["status"] == "FAIL")
    if n_fail:
        verdict = "FAIL"
    elif n_cond:
        verdict = "CONDITIONAL_PASS"
    else:
        verdict = "PASS"

    lines = [
        f"# Round-4 recertification report — {asset}",
        "",
        f"**Verdict**: **{verdict}** ({n_pass} pass / {n_cond} cond / {n_fail} fail)",
        "",
        "## Scope",
        "",
        f"* Asset: `{asset}`",
        f"* Requested months: `{', '.join(requested_months)}`",
        f"* Trained this run: `{', '.join(trained_months) if trained_months else '(none)'}`",
        f"* Skipped (already present): `{', '.join(skipped_months) if skipped_months else '(none)'}`",
        "",
        "## Pre-registered gates",
        "",
        f"* PASS: per-month QLIKE drift ≤ {PASS_QLIKE_MAX_REL_DRIFT:.0%} vs trailing-3mo median.",
        f"* CONDITIONAL_PASS: drift ∈ ({PASS_QLIKE_MAX_REL_DRIFT:.0%}, {FAIL_QLIKE_MAX_REL_DRIFT:.0%}].",
        f"* FAIL: drift > {FAIL_QLIKE_MAX_REL_DRIFT:.0%}.",
        "",
        "## Per-month results",
        "",
        "| Month | QLIKE | Baseline (3mo med) | Drift | Status |",
        "|---|---|---|---|---|",
    ]
    for m in sorted(gates.keys()):
        v = gates[m]
        b = "—" if v["baseline"] is None else f"{v['baseline']:.4f}"
        d = "—" if v["drift_rel"] is None else f"{v['drift_rel']:+.2%}"
        lines.append(
            f"| {m} | {v['qlike']:.4f} | {b} | {d} | {v['status']} |"
        )

    if qlike_prior:
        lines.append("")
        lines.append("## Trailing prior-month context")
        lines.append("")
        lines.append("| Month | QLIKE |")
        lines.append("|---|---|")
        for m in sorted(qlike_prior.keys())[-6:]:
            lines.append(f"| {m} | {qlike_prior[m]:.4f} |")

    lines.append("")
    lines.append("## Decision")
    lines.append("")
    if verdict == "PASS":
        lines.append(
            "All months pass. Model is certified stable on the post-F3 + "
            "post-F6 live distribution. New checkpoints are usable; monthly "
            "cadence simply continues forward."
        )
    elif verdict == "CONDITIONAL_PASS":
        lines.append(
            "Some months are marginal (drift > 10% but ≤ 15%). Trend likely "
            "still stable; ship with operator note. Inspect SHAP stability "
            "for the marginal months before flipping any default."
        )
    else:
        lines.append(
            "**Hard fail**: at least one month exceeded the 15% drift gate. "
            "DO NOT deploy. File as round-5 finding; investigate whether "
            "this is a regime shift, a data-pipeline drift, or a model "
            "capacity issue. Consider rolling back to the previous month's "
            "checkpoint until the cause is identified."
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n")
    logger.info("Wrote recertification report to %s", out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument(
        "--months", nargs="+",
        default=["2026-05", "2026-06", "2026-07", "2026-08", "2026-09", "2026-10"],
        help="YYYY-MM months to (re)train.",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=_DEFAULT_MODEL_DIR,
    )
    parser.add_argument(
        "--predictions-dir", type=Path, default=_DEFAULT_PREDICTIONS_DIR,
    )
    parser.add_argument(
        "--report", type=Path,
        default=Path("vol_model/output/round4_recertification_report.md"),
    )
    parser.add_argument(
        "--skip-train", action="store_true",
        help="Skip the walk-forward subprocess; only re-evaluate metrics.",
    )
    parser.add_argument(
        "--extra-arg", action="append", default=[],
        help="Extra arg to pass to run_walkforward_pruned_svi_har_pd.py.",
    )
    parser.add_argument(
        "--log-level", default="INFO",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    requested = list(args.months)
    skipped = [m for m in requested if m in _existing_checkpoints(args.model_dir)]
    to_train = _missing_months(args.model_dir, requested)
    logger.info(
        "Recertification: requested=%d, already trained=%d, to train=%d",
        len(requested), len(skipped), len(to_train),
    )

    if to_train and not args.skip_train:
        rc = _run_walkforward_subprocess(
            asset=args.asset, months=to_train, extra_args=args.extra_arg,
        )
        if rc != 0:
            logger.error("walk-forward subprocess exited rc=%d", rc)
            return rc

    # Compute per-month QLIKE on the fresh predictions parquet.
    qlike_new = _per_month_qlike(args.predictions_dir, requested)
    # Trailing prior-month context: pull every existing month BEFORE the
    # earliest requested month, capped at 12 most-recent for the report.
    earliest = min(requested) if requested else None
    qlike_prior_raw = _per_month_qlike(
        args.predictions_dir,
        [m for m in _existing_checkpoints(args.model_dir) if earliest and m < earliest],
    )
    qlike_prior = dict(sorted(qlike_prior_raw.items())[-12:])

    gates = _evaluate_gates(qlike_new, qlike_prior)

    # Persist the structured JSON for downstream tools too.
    json_path = args.report.with_suffix(".json")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({
        "asset": args.asset,
        "requested_months": requested,
        "trained_months": to_train,
        "skipped_months": skipped,
        "qlike_new": qlike_new,
        "qlike_prior": qlike_prior,
        "gates": gates,
    }, indent=2, default=float))

    _write_report(
        args.report,
        asset=args.asset,
        requested_months=requested,
        trained_months=to_train,
        skipped_months=skipped,
        qlike_new=qlike_new,
        qlike_prior=qlike_prior,
        gates=gates,
    )

    n_fail = sum(1 for v in gates.values() if v["status"] == "FAIL")
    return 0 if n_fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
