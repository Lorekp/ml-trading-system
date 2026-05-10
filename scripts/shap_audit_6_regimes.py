"""Per-regime SHAP audit across 6 regime-diverse months.

For each test month we fit an LGBM on data strictly BEFORE the month
(walk-forward style) on the FULL 126-feature parquet — no prune list,
no SHAP-tail drop. Then SHAP is computed on the test month itself,
so importance reflects what the model uses IN that regime.

Aggregating across 6 regime-diverse months gives us an empirical
basis for which features must be kept (rank flips per regime,
non-trivial SHAP in any regime) versus which are universally dead
(low/zero SHAP in every regime).

Output:
    vol_model/output/shap_audit_6_regimes_per_month.csv
    vol_model/output/shap_audit_6_regimes_decision.csv
    vol_model/output/shap_audit_6_regimes_summary.md

Six months chosen for regime diversity, ALL POST-MAY-2023 so the 15
OI features see real (non-zero-filled) data — the prior six-month
pick included 2022-06 + 2022-11 where OI was 100% zero-filled,
which artificially zeroed OI SHAP and forced a "DROP" verdict.

    2023-08  mid-2023 range-bound calm (low realized vol)
    2023-12  late-2023 ETF speculation / GBTC discount narrowing
    2024-04  BTC halving cycle / transition
    2024-08  yen-carry unwind (vol spike + cross-asset shock)
    2025-04  recent vol shock (regime — to be characterised)
    2026-01  recent OOS (most recent regime baseline)

Usage::

    PYTHONPATH=. python scripts/shap_audit_6_regimes.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from vol_model.config import VolModelConfig  # noqa: E402
from vol_model.model import qlike_metric, qlike_objective  # noqa: E402
from vol_model.pipeline import build_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("shap_audit")


REGIME_MONTHS = [
    ("2023-08", "mid-2023 range-bound calm (low realized vol)"),
    ("2023-12", "late-2023 ETF-speculation / GBTC discount narrowing"),
    ("2024-04", "BTC halving cycle / transition"),
    ("2024-08", "yen-carry unwind (vol spike + cross-asset shock)"),
    ("2025-04", "recent vol shock (regime characterised post-hoc)"),
    ("2026-01", "recent OOS (most recent regime baseline)"),
]


LGBM_PARAMS = {
    "objective": qlike_objective,
    "num_leaves": 31,
    "max_depth": 6,
    "learning_rate": 0.03,
    "min_data_in_leaf": 20,
    "colsample_bytree": 0.8,
    "subsample": 0.8,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "device": "cpu",
    "num_threads": -1,
    "verbosity": -1,
    "seed": 42,
}


def _train_one_month(X: pd.DataFrame, y: pd.Series, month: str) -> tuple:
    """Fit LGBM on data strictly before `month`, compute SHAP on the
    test month (full month). Returns (mean_abs_shap_dict, n_train,
    n_test, qlike_test)."""
    month_ts = pd.Timestamp(month, tz="UTC")
    next_month = (month_ts + pd.offsets.MonthBegin(1))

    train_mask = X.index < month_ts
    test_mask = (X.index >= month_ts) & (X.index < next_month)

    X_train = X[train_mask]
    y_train = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    if len(X_test) == 0:
        raise ValueError(f"No test rows for month {month}")
    if len(X_train) < 1000:
        raise ValueError(f"Train set too small for {month}: {len(X_train)} rows")

    # Train winsorization (1%/99%) — fit on TRAIN only, apply to both.
    lo = X_train.quantile(0.01)
    hi = X_train.quantile(0.99)
    X_train_w = X_train.clip(lo, hi, axis=1)
    X_test_w = X_test.clip(lo, hi, axis=1)

    # Validation split on the last 15% of train.
    val_split = int(len(X_train_w) * 0.85)
    dtrain = lgb.Dataset(X_train_w.iloc[:val_split],
                         label=y_train.iloc[:val_split].values,
                         free_raw_data=False)
    dval = lgb.Dataset(X_train_w.iloc[val_split:],
                       label=y_train.iloc[val_split:].values,
                       reference=dtrain, free_raw_data=False)

    logger.info("[%s] training: %d train rows, %d test rows",
                month, len(X_train_w), len(X_test_w))
    t0 = time.time()
    model = lgb.train(
        LGBM_PARAMS, dtrain, num_boost_round=1000,
        valid_sets=[dval], feval=qlike_metric,
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(0)],
    )
    train_s = time.time() - t0

    # SHAP on test month — this is the regime-conditional importance.
    t0 = time.time()
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test_w)
    shap_s = time.time() - t0

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_dict = dict(zip(X.columns, mean_abs_shap))

    # Test-month QLIKE (sanity check that the model is functional).
    y_pred = model.predict(X_test_w)
    resid = y_test.values - y_pred
    qlike = float(np.mean(np.exp(2 * resid) - 2 * resid - 1))

    logger.info("[%s] done in %.1fs (train %.1fs, shap %.1fs); "
                "test QLIKE=%.4f", month, train_s + shap_s,
                train_s, shap_s, qlike)

    return shap_dict, len(X_train_w), len(X_test_w), qlike


def main() -> int:
    config = VolModelConfig(asset="BTCUSDT")
    logger.info("Building dataset (BTCUSDT, 15m)...")
    X, y, _ = build_dataset(config)
    logger.info("Dataset: %d rows, %d features, range %s -> %s",
                len(X), len(X.columns), X.index.min(), X.index.max())

    per_month_shap: dict[str, dict[str, float]] = {}
    per_month_meta: dict[str, dict] = {}

    for month, regime_name in REGIME_MONTHS:
        try:
            shap_dict, n_train, n_test, qlike = _train_one_month(X, y, month)
        except ValueError as e:
            logger.error("[%s] SKIPPED: %s", month, e)
            continue
        per_month_shap[month] = shap_dict
        per_month_meta[month] = {
            "regime": regime_name,
            "n_train": n_train,
            "n_test": n_test,
            "test_qlike": qlike,
        }

    if not per_month_shap:
        logger.error("No months produced SHAP — abort")
        return 1

    # Build per-month matrix (rows: features, cols: months).
    feat_idx = list(X.columns)
    matrix = pd.DataFrame(index=feat_idx)
    for month, shap_dict in per_month_shap.items():
        matrix[month] = [shap_dict.get(f, 0.0) for f in feat_idx]

    # Per-month rank (1 = highest SHAP).
    rank_matrix = matrix.rank(ascending=False, method="min").astype(int)
    rank_matrix.columns = [f"{c}_rank" for c in matrix.columns]

    # Decision metrics.
    mean_shap = matrix.mean(axis=1)
    median_shap = matrix.median(axis=1)
    max_shap = matrix.max(axis=1)
    min_shap = matrix.min(axis=1)
    shap_std = matrix.std(axis=1)
    # rank stats
    raw_ranks = matrix.rank(ascending=False, method="min").astype(int)
    median_rank = raw_ranks.median(axis=1)
    best_rank = raw_ranks.min(axis=1)
    worst_rank = raw_ranks.max(axis=1)
    rank_range = worst_rank - best_rank
    # months with non-trivial SHAP (above 1/100th of column max)
    epsilon_per_col = matrix.max(axis=0) * 0.01
    above_eps = (matrix > epsilon_per_col).sum(axis=1)

    decision = pd.DataFrame({
        "feature": feat_idx,
        "mean_shap": mean_shap.values,
        "median_shap": median_shap.values,
        "max_shap": max_shap.values,
        "min_shap": min_shap.values,
        "shap_std": shap_std.values,
        "median_rank": median_rank.values.astype(int),
        "best_rank": best_rank.values.astype(int),
        "worst_rank": worst_rank.values.astype(int),
        "rank_range": rank_range.values.astype(int),
        "n_months_above_1pct": above_eps.values.astype(int),
    })
    # Decision flag:
    #   KEEP if median_rank <= 35 OR best_rank <= 25 OR n_months_above_1pct >= 4
    #   DROP if max_rank > 60 across all months AND n_months_above_1pct == 0
    #   REVIEW otherwise
    def _flag(row):
        if row["median_rank"] <= 35 or row["best_rank"] <= 25 \
                or row["n_months_above_1pct"] >= 4:
            return "KEEP"
        if row["best_rank"] > 60 and row["n_months_above_1pct"] == 0:
            return "DROP"
        return "REVIEW"
    decision["decision"] = decision.apply(_flag, axis=1)
    decision = decision.sort_values("mean_shap", ascending=False).reset_index(drop=True)

    # Persist outputs.
    out_dir = REPO_ROOT / "vol_model" / "output"
    per_month_path = out_dir / "shap_audit_post_oi_per_month.csv"
    decision_path = out_dir / "shap_audit_post_oi_decision.csv"
    summary_path = out_dir / "shap_audit_post_oi_summary.md"

    matrix_with_rank = matrix.join(raw_ranks.add_suffix("_rank"))
    matrix_with_rank.index.name = "feature"
    matrix_with_rank.to_csv(per_month_path)
    decision.to_csv(decision_path, index=False)

    n_keep = (decision["decision"] == "KEEP").sum()
    n_drop = (decision["decision"] == "DROP").sum()
    n_review = (decision["decision"] == "REVIEW").sum()

    lines = [
        "# Multi-Regime SHAP Audit (6 months)",
        "",
        "## Months audited",
        "",
        "| month | regime | n_train | n_test | test QLIKE |",
        "|---|---|---:|---:|---:|",
    ]
    for month, meta in per_month_meta.items():
        lines.append(
            f"| {month} | {meta['regime']} | {meta['n_train']:,} | "
            f"{meta['n_test']:,} | {meta['test_qlike']:.4f} |"
        )
    lines += [
        "",
        f"## Decision summary",
        f"- KEEP: {n_keep} features (median rank <=35 OR best rank <=25 OR "
        f"non-trivial SHAP in >=4 of 6 months)",
        f"- DROP: {n_drop} features (best rank >60 AND non-trivial in zero months)",
        f"- REVIEW: {n_review} features (judgment call — see CSV)",
        "",
        "## Top 50 features by mean SHAP across regimes",
        "",
        "| rank | feature | mean SHAP | median rank | best rank | "
        "worst rank | months >1% | decision |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for i, row in decision.head(50).iterrows():
        lines.append(
            f"| {i+1} | `{row['feature']}` | {row['mean_shap']:.4f} | "
            f"{row['median_rank']} | {row['best_rank']} | "
            f"{row['worst_rank']} | {row['n_months_above_1pct']}/6 | "
            f"{row['decision']} |"
        )
    lines += [
        "",
        "## Bottom 30 features by mean SHAP",
        "",
        "| rank | feature | mean SHAP | median rank | best rank | "
        "worst rank | months >1% | decision |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for i, row in decision.tail(30).iterrows():
        rank = decision.index.get_loc(i) + 1
        lines.append(
            f"| {rank} | `{row['feature']}` | {row['mean_shap']:.6f} | "
            f"{row['median_rank']} | {row['best_rank']} | "
            f"{row['worst_rank']} | {row['n_months_above_1pct']}/6 | "
            f"{row['decision']} |"
        )

    summary_path.write_text("\n".join(lines), encoding="utf-8")

    logger.info("Wrote %s", per_month_path)
    logger.info("Wrote %s", decision_path)
    logger.info("Wrote %s", summary_path)
    logger.info("KEEP=%d / DROP=%d / REVIEW=%d", n_keep, n_drop, n_review)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
