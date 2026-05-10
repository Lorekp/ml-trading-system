"""Second-pass SHAP audit on the PRUNED 62-feature list.

After the first-pass audit (`shap_audit_optuna.py`) on all 126 base
features produced a 62-feature union KEEP list, this script verifies
that all 62 actually carry signal once collinear/redundant features
have been removed. Same Optuna 200/50 search space + same 8 audit
months as the first pass — but trained on the 62-feature subset of X.

Why this matters: a feature can have non-trivial SHAP in a 126-feature
model because it's marginally informative or because it's redundant
with another feature that LightGBM happens to skip. Once we narrow
the input to 62 features, the redundant features are forced into use
and their SHAP rises — OR they reveal as truly low-signal and drop
to DROP/REVIEW. Either outcome is informative.

Output:
    vol_model/output/shap_audit_optuna_pruned_pre_oi_decision.csv
    vol_model/output/shap_audit_optuna_pruned_post_oi_decision.csv
    vol_model/output/shap_audit_optuna_pruned_summary.md
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

# Reuse the helpers from the all-126 audit script.
from scripts.shap_audit_optuna import (  # noqa: E402
    PRE_OI_MONTHS, POST_OI_MONTHS, _train_one_month, _build_decision,
    logger,
)
from vol_model.config import VolModelConfig  # noqa: E402
from vol_model.pipeline import build_dataset  # noqa: E402


def main() -> int:
    config = VolModelConfig(asset="BTCUSDT")
    logger.info("Building dataset (BTCUSDT, 15m, clip=False)...")
    X_full, y, _ = build_dataset(config, clip=False)

    pruned_path = REPO_ROOT / "vol_model" / "output" / "pruned_features.csv"
    pruned_features = [
        line.strip() for line in pruned_path.read_text().splitlines()
        if line.strip()
    ]
    missing = [f for f in pruned_features if f not in X_full.columns]
    if missing:
        raise ValueError(f"Pruned features missing from X: {missing}")

    X = X_full[pruned_features]
    logger.info(
        "Pruned input: %d rows, %d features (subset of %d full)",
        len(X), len(X.columns), len(X_full.columns),
    )
    label_clip_quantile = config.label_clip_quantile

    out_dir = REPO_ROOT / "vol_model" / "output"

    # Same 8 unique audit months as the first-pass.
    seen: dict[str, tuple] = {}
    unique_months: list[tuple[str, str]] = []
    for m, r in PRE_OI_MONTHS + POST_OI_MONTHS:
        if m not in seen:
            seen[m] = (m, r)
            unique_months.append((m, r))
    logger.info(
        "Auditing %d unique months on the %d-feature pruned set",
        len(unique_months), len(pruned_features),
    )

    all_shap: dict[str, dict] = {}
    all_meta: dict[str, dict] = {}
    for month, regime in unique_months:
        try:
            shap_dict, n_tr, n_te, qlike, n_trials = _train_one_month(
                X, y, label_clip_quantile, month,
            )
        except ValueError as e:
            logger.error("[%s] SKIPPED: %s", month, e)
            continue
        all_shap[month] = shap_dict
        all_meta[month] = {
            "regime": regime, "n_train": n_tr, "n_test": n_te,
            "test_qlike": qlike, "n_optuna_trials": n_trials,
        }

    if not all_shap:
        raise RuntimeError("No months produced SHAP")

    pre_oi_shap = {m: all_shap[m] for m, _ in PRE_OI_MONTHS if m in all_shap}
    post_oi_shap = {m: all_shap[m] for m, _ in POST_OI_MONTHS if m in all_shap}

    pre_decision, pre_matrix = _build_decision(pre_oi_shap, list(X.columns))
    pre_decision.to_csv(
        out_dir / "shap_audit_optuna_pruned_pre_oi_decision.csv", index=False,
    )
    pre_matrix.to_csv(
        out_dir / "shap_audit_optuna_pruned_pre_oi_per_month.csv",
    )

    post_decision, post_matrix = _build_decision(post_oi_shap, list(X.columns))
    post_decision.to_csv(
        out_dir / "shap_audit_optuna_pruned_post_oi_decision.csv", index=False,
    )
    post_matrix.to_csv(
        out_dir / "shap_audit_optuna_pruned_post_oi_per_month.csv",
    )

    # Decision aggregation: any feature that is DROP in BOTH runs is
    # a candidate for further pruning. REVIEW in both is also worth
    # flagging.
    pre_idx = pre_decision.set_index("feature")
    post_idx = post_decision.set_index("feature")

    n_keep_both = sum(
        1 for f in pruned_features
        if pre_idx.loc[f, "decision"] == "KEEP"
        and post_idx.loc[f, "decision"] == "KEEP"
    )
    drop_both = [
        f for f in pruned_features
        if pre_idx.loc[f, "decision"] == "DROP"
        and post_idx.loc[f, "decision"] == "DROP"
    ]
    review_both = [
        f for f in pruned_features
        if pre_idx.loc[f, "decision"] == "REVIEW"
        and post_idx.loc[f, "decision"] == "REVIEW"
    ]
    drop_or_review = sorted(set(drop_both) | set(review_both))

    lines = [
        "# Optuna-tuned SHAP audit — second pass on the 62-feature "
        "pruned list",
        "",
        f"Total pruned features audited: {len(pruned_features)}",
        f"KEEP in BOTH runs: {n_keep_both}",
        f"DROP in BOTH runs: {len(drop_both)}",
        f"REVIEW in BOTH runs: {len(review_both)}",
        "",
        "## Per-month Optuna config + QLIKE",
        "",
        "| month | regime | n_train | n_test | test QLIKE | trials |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for m, meta in all_meta.items():
        lines.append(
            f"| {m} | {meta['regime']} | {meta['n_train']:,} | "
            f"{meta['n_test']:,} | {meta['test_qlike']:.4f} | "
            f"{meta['n_optuna_trials']} |"
        )

    if drop_or_review:
        lines += [
            "",
            "## Candidates for further pruning (DROP or REVIEW in both runs)",
            "",
            "| feature | pre-OI | post-OI | pre median rank | post median rank |",
            "|---|---|---|---:|---:|",
        ]
        for f in drop_or_review:
            lines.append(
                f"| `{f}` | {pre_idx.loc[f, 'decision']} | "
                f"{post_idx.loc[f, 'decision']} | "
                f"{int(pre_idx.loc[f, 'median_rank'])} | "
                f"{int(post_idx.loc[f, 'median_rank'])} |"
            )
    else:
        lines += [
            "",
            "## All 62 pruned features land in KEEP in at least one of the two runs",
            "",
            "No further pruning indicated. Prune list is stable.",
        ]

    summary = out_dir / "shap_audit_optuna_pruned_summary.md"
    summary.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", summary)
    logger.info(
        "Result: %d/%d KEEP-in-both, %d DROP-both, %d REVIEW-both",
        n_keep_both, len(pruned_features),
        len(drop_both), len(review_both),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
