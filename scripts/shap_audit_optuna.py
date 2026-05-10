"""Per-regime SHAP audit with Optuna-tuned LightGBM (production-matched).

Supersedes ``shap_audit_6_regimes.py`` (which used fixed hyperparameters).
For each test month we run **Optuna 200 trials / 50 patience** on the
full 126-feature parquet — same hyperparameter search as the production
walk-forward (`run_walkforward_pruned_svi_har_pd.py`). This means the
SHAP rankings reflect the model the production pipeline actually
trains, not a fixed-config approximation.

12 audit months total (pre-OI 6 + post-OI 6 — disjoint sets):

    Pre-OI run (full regime diversity, OI=0 in pre-2023 months):
        2022-06  LUNA collapse aftermath + 3AC unwind
        2022-11  FTX collapse + contagion
        2023-08  mid-2023 range-bound calm
        2024-04  BTC halving cycle / transition
        2024-08  yen-carry unwind (vol spike + cross-asset shock)
        2026-01  recent OOS

    Post-OI run (live OI data in every month):
        2023-08  mid-2023 range-bound calm
        2023-12  late-2023 ETF speculation
        2024-04  BTC halving cycle / transition
        2024-08  yen-carry unwind
        2025-04  recent vol shock
        2026-01  recent OOS

Output:
    vol_model/output/shap_audit_optuna_pre_oi_decision.csv
    vol_model/output/shap_audit_optuna_pre_oi_per_month.csv
    vol_model/output/shap_audit_optuna_post_oi_decision.csv
    vol_model/output/shap_audit_optuna_post_oi_per_month.csv
    vol_model/output/shap_audit_optuna_summary.md

Usage::

    PYTHONPATH=. python scripts/shap_audit_optuna.py
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
import optuna  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from vol_model.config import VolModelConfig  # noqa: E402
from vol_model.labels import clip_label_quantiles  # noqa: E402
from vol_model.model import qlike_metric, qlike_objective  # noqa: E402
from vol_model.pipeline import build_dataset  # noqa: E402

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("shap_audit_optuna")
optuna.logging.set_verbosity(optuna.logging.WARNING)


PRE_OI_MONTHS = [
    ("2022-06", "LUNA collapse aftermath + 3AC unwind (post-crash high vol)"),
    ("2022-11", "FTX collapse + contagion (acute crisis)"),
    ("2023-08", "mid-2023 range-bound calm (low realized vol)"),
    ("2024-04", "BTC halving cycle / transition"),
    ("2024-08", "yen-carry unwind (vol spike + cross-asset shock)"),
    ("2026-01", "recent OOS (most recent regime baseline)"),
]

POST_OI_MONTHS = [
    ("2023-08", "mid-2023 range-bound calm (low realized vol)"),
    ("2023-12", "late-2023 ETF-speculation / GBTC discount narrowing"),
    ("2024-04", "BTC halving cycle / transition"),
    ("2024-08", "yen-carry unwind (vol spike + cross-asset shock)"),
    ("2025-04", "recent vol shock"),
    ("2026-01", "recent OOS (most recent regime baseline)"),
]


# Production walk-forward Optuna config (matches
# run_walkforward_pruned_svi_har_pd.py defaults: 200 trials / 50 patience)
N_TRIALS = 200
PATIENCE = 50


def _suggest_params(trial: optuna.Trial) -> dict:
    """Match the production Optuna search space for HAR-PD walk-forward."""
    return {
        "objective": qlike_objective,
        "num_leaves": trial.suggest_int("num_leaves", 16, 128),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float(
            "learning_rate", 1e-3, 5e-2, log=True
        ),
        "min_data_in_leaf": trial.suggest_int(
            "min_data_in_leaf", 50, 500
        ),
        "lambda_l1": trial.suggest_float(
            "lambda_l1", 1e-3, 10.0, log=True
        ),
        "lambda_l2": trial.suggest_float(
            "lambda_l2", 1e-3, 10.0, log=True
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree", 0.4, 0.9
        ),
        "device": "cpu",
        "num_threads": -1,
        "verbosity": -1,
        "seed": 42,
    }


def _train_one_month(
    X: pd.DataFrame, y: pd.Series, label_clip_quantile: float, month: str,
) -> tuple:
    """For a given test month, run Optuna 200/50 on a strictly-prior
    expanding window, train final LGBM with best params, compute SHAP
    on the test month."""
    month_ts = pd.Timestamp(month, tz="UTC")
    next_month = month_ts + pd.offsets.MonthBegin(1)

    train_mask = X.index < month_ts
    test_mask = (X.index >= month_ts) & (X.index < next_month)

    X_train_full = X[train_mask]
    y_train_full = y[train_mask]
    X_test = X[test_mask]
    y_test = y[test_mask]

    if len(X_test) == 0:
        raise ValueError(f"No test rows for month {month}")
    if len(X_train_full) < 1000:
        raise ValueError(f"Train too small for {month}: {len(X_train_full)} rows")

    # Per-fold leak-safe label clip on TRAIN slice quantiles only.
    y_lo = float(y_train_full.quantile(1.0 - label_clip_quantile))
    y_hi = float(y_train_full.quantile(label_clip_quantile))
    y_train_full_clipped = y_train_full.clip(y_lo, y_hi)
    y_test_clipped = y_test.clip(y_lo, y_hi)

    # Train winsorization (1%/99%) — fit on TRAIN only, apply to both.
    lo = X_train_full.quantile(0.01)
    hi = X_train_full.quantile(0.99)
    X_train_w = X_train_full.clip(lo, hi, axis=1)
    X_test_w = X_test.clip(lo, hi, axis=1)

    # Train/val split (last 15% of train for early stopping + Optuna).
    val_split = int(len(X_train_w) * 0.85)
    X_tr_inner = X_train_w.iloc[:val_split]
    y_tr_inner = y_train_full_clipped.iloc[:val_split]
    X_val_inner = X_train_w.iloc[val_split:]
    y_val_inner = y_train_full_clipped.iloc[val_split:]

    dtrain = lgb.Dataset(X_tr_inner, label=y_tr_inner.values, free_raw_data=False)
    dval = lgb.Dataset(
        X_val_inner, label=y_val_inner.values,
        reference=dtrain, free_raw_data=False,
    )

    # ── Optuna search ───────────────────────────────────────────
    def _objective(trial: optuna.Trial) -> float:
        params = _suggest_params(trial)
        booster = lgb.train(
            params, dtrain, num_boost_round=1000,
            valid_sets=[dval], feval=qlike_metric,
            callbacks=[lgb.early_stopping(50, verbose=False),
                       lgb.log_evaluation(0)],
        )
        # Best validation QLIKE (the metric we minimize).
        val_pred = booster.predict(X_val_inner)
        diff = y_val_inner.values - val_pred
        clipped_diff = np.clip(2.0 * diff, -20.0, 20.0)
        qlike = float(np.mean(np.exp(clipped_diff) - 2.0 * diff - 1.0))
        return qlike

    sampler = optuna.samplers.TPESampler(seed=42)
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=20)
    study = optuna.create_study(
        sampler=sampler, pruner=pruner, direction="minimize",
    )

    t0 = time.time()
    # Patience-based early stopping: stop if no improvement for PATIENCE
    # consecutive trials.
    best_seen = [float("inf")]
    n_since_best = [0]

    def _patience_callback(study: optuna.Study, _trial) -> None:
        if study.best_value < best_seen[0]:
            best_seen[0] = study.best_value
            n_since_best[0] = 0
        else:
            n_since_best[0] += 1
        if n_since_best[0] >= PATIENCE:
            study.stop()

    study.optimize(
        _objective, n_trials=N_TRIALS, callbacks=[_patience_callback],
        show_progress_bar=False,
    )
    optuna_s = time.time() - t0
    n_completed = len(study.trials)

    # ── Final train on best params ───────────────────────────────
    best_params = {**_suggest_params(_DummyTrial(study.best_params))}
    best_params.update({
        "num_leaves": study.best_params["num_leaves"],
        "max_depth": study.best_params["max_depth"],
        "learning_rate": study.best_params["learning_rate"],
        "min_data_in_leaf": study.best_params["min_data_in_leaf"],
        "lambda_l1": study.best_params["lambda_l1"],
        "lambda_l2": study.best_params["lambda_l2"],
        "colsample_bytree": study.best_params["colsample_bytree"],
    })

    t0 = time.time()
    final_model = lgb.train(
        best_params, dtrain, num_boost_round=1000,
        valid_sets=[dval], feval=qlike_metric,
        callbacks=[lgb.early_stopping(50, verbose=False),
                   lgb.log_evaluation(0)],
    )
    final_train_s = time.time() - t0

    # ── SHAP on test month ────────────────────────────────────────
    t0 = time.time()
    explainer = shap.TreeExplainer(final_model)
    shap_values = explainer.shap_values(X_test_w)
    shap_s = time.time() - t0

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    shap_dict = dict(zip(X.columns, mean_abs_shap))

    # Test-month QLIKE (sanity check).
    y_pred = final_model.predict(X_test_w)
    resid = y_test_clipped.values - y_pred
    qlike = float(np.mean(np.exp(np.clip(2 * resid, -20, 20))
                          - 2 * resid - 1))

    logger.info(
        "[%s] optuna=%.0fs (n=%d), final_train=%.0fs, shap=%.0fs; "
        "test QLIKE=%.4f; best params: leaves=%d depth=%d lr=%.4f",
        month, optuna_s, n_completed, final_train_s, shap_s, qlike,
        study.best_params["num_leaves"],
        study.best_params["max_depth"],
        study.best_params["learning_rate"],
    )

    return shap_dict, len(X_train_w), len(X_test_w), qlike, n_completed


class _DummyTrial:
    """Dummy that satisfies the _suggest_params signature without
    actually invoking suggest_*; we pre-fill the params dict with the
    Optuna study's best values."""
    def __init__(self, best_params: dict):
        self._b = best_params

    def suggest_int(self, name, *_args, **_kwargs):
        return self._b[name]

    def suggest_float(self, name, *_args, **_kwargs):
        return self._b[name]


def _build_decision(
    per_month_shap: dict[str, dict], feat_idx: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregate per-month SHAP into a decision DataFrame."""
    matrix = pd.DataFrame(index=feat_idx)
    for month, shap_dict in per_month_shap.items():
        matrix[month] = [shap_dict.get(f, 0.0) for f in feat_idx]

    raw_ranks = matrix.rank(ascending=False, method="min").astype(int)
    mean_shap = matrix.mean(axis=1)
    median_shap = matrix.median(axis=1)
    max_shap = matrix.max(axis=1)
    min_shap = matrix.min(axis=1)
    shap_std = matrix.std(axis=1)
    median_rank = raw_ranks.median(axis=1)
    best_rank = raw_ranks.min(axis=1)
    worst_rank = raw_ranks.max(axis=1)
    rank_range = worst_rank - best_rank
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

    def _flag(row):
        if row["median_rank"] <= 35 or row["best_rank"] <= 25 \
                or row["n_months_above_1pct"] >= 4:
            return "KEEP"
        if row["best_rank"] > 60 and row["n_months_above_1pct"] == 0:
            return "DROP"
        return "REVIEW"
    decision["decision"] = decision.apply(_flag, axis=1)
    decision = decision.sort_values("mean_shap", ascending=False).reset_index(drop=True)

    matrix_with_rank = matrix.join(raw_ranks.add_suffix("_rank"))
    matrix_with_rank.index.name = "feature"
    return decision, matrix_with_rank


def _run(months: list[tuple[str, str]], tag: str,
         X: pd.DataFrame, y: pd.Series, label_clip_quantile: float,
         out_dir: Path) -> tuple[pd.DataFrame, dict]:
    per_month_shap: dict[str, dict] = {}
    per_month_meta: dict[str, dict] = {}
    for month, regime in months:
        try:
            shap_dict, n_tr, n_te, qlike, n_trials = _train_one_month(
                X, y, label_clip_quantile, month,
            )
        except ValueError as e:
            logger.error("[%s] SKIPPED: %s", month, e)
            continue
        per_month_shap[month] = shap_dict
        per_month_meta[month] = {
            "regime": regime,
            "n_train": n_tr,
            "n_test": n_te,
            "test_qlike": qlike,
            "n_optuna_trials": n_trials,
        }

    if not per_month_shap:
        raise RuntimeError(f"No months produced SHAP for tag={tag}")

    decision, matrix = _build_decision(per_month_shap, list(X.columns))

    decision.to_csv(out_dir / f"shap_audit_optuna_{tag}_decision.csv",
                    index=False)
    matrix.to_csv(out_dir / f"shap_audit_optuna_{tag}_per_month.csv")

    n_keep = (decision["decision"] == "KEEP").sum()
    n_drop = (decision["decision"] == "DROP").sum()
    n_review = (decision["decision"] == "REVIEW").sum()
    logger.info(
        "[%s] decision: KEEP=%d / REVIEW=%d / DROP=%d", tag,
        n_keep, n_review, n_drop,
    )
    return decision, per_month_meta


def main() -> int:
    config = VolModelConfig(asset="BTCUSDT")
    logger.info("Building dataset (BTCUSDT, 15m, clip=False)...")
    # clip=False so we can apply per-fold clipping per month (matches
    # production walk-forward post Cycle-9 fix).
    X, y, _ = build_dataset(config, clip=False)
    logger.info("Dataset: %d rows, %d features, range %s -> %s",
                len(X), len(X.columns), X.index.min(), X.index.max())
    label_clip_quantile = config.label_clip_quantile

    out_dir = REPO_ROOT / "vol_model" / "output"

    # Dedupe: pre_oi and post_oi share 4 months (2023-08, 2024-04,
    # 2024-08, 2026-01). Compute once per unique month, assign to
    # both sets.
    seen: dict[str, tuple] = {}
    unique_months: list[tuple[str, str]] = []
    for m, r in PRE_OI_MONTHS + POST_OI_MONTHS:
        if m not in seen:
            seen[m] = (m, r)
            unique_months.append((m, r))
    logger.info("Audit will run %d unique months (%d duplicates skipped)",
                len(unique_months),
                len(PRE_OI_MONTHS) + len(POST_OI_MONTHS) - len(unique_months))

    logger.info("=" * 60)
    logger.info("Computing SHAP for %d unique audit months...",
                len(unique_months))
    logger.info("=" * 60)

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
            "regime": regime,
            "n_train": n_tr,
            "n_test": n_te,
            "test_qlike": qlike,
            "n_optuna_trials": n_trials,
        }

    if not all_shap:
        raise RuntimeError("No months produced SHAP")

    # Build per-set decisions (same SHAP values, different month sets).
    pre_oi_shap = {m: all_shap[m] for m, _ in PRE_OI_MONTHS if m in all_shap}
    pre_oi_meta = {m: all_meta[m] for m in pre_oi_shap}
    post_oi_shap = {m: all_shap[m] for m, _ in POST_OI_MONTHS if m in all_shap}
    post_oi_meta = {m: all_meta[m] for m in post_oi_shap}

    pre_decision, pre_matrix = _build_decision(pre_oi_shap, list(X.columns))
    pre_decision.to_csv(out_dir / "shap_audit_optuna_pre_oi_decision.csv",
                        index=False)
    pre_matrix.to_csv(out_dir / "shap_audit_optuna_pre_oi_per_month.csv")

    post_decision, post_matrix = _build_decision(post_oi_shap, list(X.columns))
    post_decision.to_csv(out_dir / "shap_audit_optuna_post_oi_decision.csv",
                         index=False)
    post_matrix.to_csv(out_dir / "shap_audit_optuna_post_oi_per_month.csv")

    pre_meta = pre_oi_meta
    post_meta = post_oi_meta

    n_keep_pre = (pre_decision["decision"] == "KEEP").sum()
    n_keep_post = (post_decision["decision"] == "KEEP").sum()
    logger.info("pre_oi KEEP=%d, post_oi KEEP=%d", n_keep_pre, n_keep_post)

    # Union curation (matches the cycle-8 logic from shap_audit_6_regimes.py).
    pre_idx = pre_decision.set_index("feature")
    post_idx = post_decision.set_index("feature")
    merged = pre_idx[["decision", "mean_shap"]].rename(
        columns={"decision": "pre", "mean_shap": "pre_shap"}
    ).join(
        post_idx[["decision", "mean_shap"]].rename(
            columns={"decision": "post", "mean_shap": "post_shap"}
        ), how="outer"
    )

    def _curate(row):
        f = row.name
        if f.startswith("oi_"):
            return row["post"]
        if row["pre"] == "KEEP" or row["post"] == "KEEP":
            return "KEEP"
        if row["pre"] == "DROP" and row["post"] == "DROP":
            return "DROP"
        return "REVIEW"

    merged["final"] = merged.apply(_curate, axis=1)
    keep_features = merged[merged["final"] == "KEEP"].sort_values(
        "pre_shap", ascending=False
    ).index.tolist()

    out_path = out_dir / "pruned_features_optuna_unioncurated.csv"
    out_path.write_text("\n".join(keep_features) + "\n")
    logger.info("Wrote union KEEP list (%d features) to %s",
                len(keep_features), out_path)

    # Summary doc
    lines = [
        "# Optuna-tuned SHAP audit (cycle 9, 2026-05-08)",
        "",
        f"Production-matched Optuna {N_TRIALS} trials / {PATIENCE} patience "
        "per month. Supersedes the fixed-hyperparam audit at "
        "`shap_audit_6_regimes_*.csv`.",
        "",
        "## Pre-OI run", "",
        "| month | regime | n_train | n_test | test QLIKE | trials |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for m, meta in pre_meta.items():
        lines.append(
            f"| {m} | {meta['regime']} | {meta['n_train']:,} | "
            f"{meta['n_test']:,} | {meta['test_qlike']:.4f} | "
            f"{meta['n_optuna_trials']} |"
        )
    lines += ["", "## Post-OI run", "",
              "| month | regime | n_train | n_test | test QLIKE | trials |",
              "|---|---|---:|---:|---:|---:|"]
    for m, meta in post_meta.items():
        lines.append(
            f"| {m} | {meta['regime']} | {meta['n_train']:,} | "
            f"{meta['n_test']:,} | {meta['test_qlike']:.4f} | "
            f"{meta['n_optuna_trials']} |"
        )

    n_keep = len(keep_features)
    n_total = len(merged)
    lines += [
        "",
        f"## Union decision: KEEP={n_keep} / total={n_total}",
        "",
        "Features that needed UNION (one-audit-only KEEP):",
        "",
    ]
    for f in keep_features:
        row = merged.loc[f]
        if row["pre"] != "KEEP" or row["post"] != "KEEP":
            lines.append(
                f"- `{f}`: pre={row['pre']} (shap={row['pre_shap']:.4f}) "
                f"post={row['post']} (shap={row['post_shap']:.4f})"
            )

    summary = out_dir / "shap_audit_optuna_summary.md"
    summary.write_text("\n".join(lines), encoding="utf-8")
    logger.info("Wrote %s", summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
