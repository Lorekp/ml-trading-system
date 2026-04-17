#!/usr/bin/env python
"""Walk-forward directional classifier head (W4.1).

Sibling to T2.1's ``run_walkforward_pruned_svi.py``. Predicts the SIGN of
the residual ``y_true - y_pred_base`` from the pruned+SVI base regression,
so it acts as a confidence/direction filter that can be layered on top of
the existing signal in the backtester.

Architecture:

1. Same 44-feature matrix as T2.1: 16 TransVol embeddings + pruned base +
   ``svi_atm_1d``. TransVol checkpoints are reused unchanged from
   ``expanding_hybrid/``.
2. Same walk-forward structure (expanding train window, 4-month val, 1-month
   test). Same winsorization on train quantiles. Same Optuna TPE + MedianPruner
   + custom ``OptunaPatience`` callback.
3. Target is binary: ``direction[t] = 1 if y_true[t] > y_pred_base[t] else 0``,
   loaded from ``vol_model/output/BTCUSDT_pruned_svi_predictions.parquet``.
4. LightGBM objective = ``"binary"`` with ``is_unbalance=True``; inner Optuna
   loop minimises validation log loss. Per-month OOS metrics reported: ROC
   AUC, log loss, Brier score.
5. **Minimum training-history constraint** is measured from the base
   predictions parquet start (2022-02), not from the full feature data
   start. This is a hard dependency: the classifier can only train where
   its target exists. Default ``--start 2023-06`` gives the earliest month
   with ≥12mo train + 4mo val; full walk-forward is ~34 months.

Usage:
    # Spot-check (3 months, ~30 min)
    PYTHONPATH=. python vol_model/run_walkforward_classifier_head.py \\
        --start 2026-01 --end 2026-03 \\
        --max-trials 100 --patience 30

    # Full walk-forward (~4-6h wall clock)
    PYTHONPATH=. python vol_model/run_walkforward_classifier_head.py \\
        --start 2023-06 --end 2026-03 \\
        --max-trials 200 --patience 50
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score

from vol_model.config import VolModelConfig
from vol_model.pipeline import build_dataset
from vol_model.run_walkforward import OptunaPatience
from vol_model.run_walkforward_hybrid import _extract_compact_embeddings
from vol_model.run_walkforward_pruned_svi import (
    ADD_SVI_FEATURES,
    DROP_FEATURES,
    HYBRID_DIR,
    _load_svi_atm_1d,
)

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wf_classifier_head")

OUTPUT_ROOT = Path("vol_model/output/models")
PRUNED_PATH = Path("vol_model/output/pruned_features.csv")
BASE_PREDS_PATH = Path(
    "vol_model/output/BTCUSDT_pruned_svi_predictions.parquet"
)
VOL_ARB_PATH = Path("vol_arb/output/BTC_vol_arb_svi.parquet")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Minimum training-history measured from the base predictions parquet start.
# The classifier cannot train on timestamps where its target label does not
# exist, so this constant — not the feature data start — governs the
# earliest feasible walk-forward month.
MIN_TRAIN_MONTHS = 12

# Supported target variants. The CLI flag ``--target`` chooses between:
#   "edge":     direction = (y_true > log(iv_used/100)).int8
#               — directly predicts whether realized vol exceeds implied
#               — matches the Stage 17.1f oracle test's framing
#               — naturally imbalanced (~24% positive on BTC 2022-2026)
#   "residual": direction = (y_true > y_pred_base).int8
#               — predicts the sign of the base regression's residual
#               — near-noise on the same feature set; kept for research only
TARGET_CHOICES = ("edge", "residual")


def _load_edge_direction_target() -> pd.Series:
    """Load the EDGE target: did realized log-RV exceed log-IV at this bar?

    Aligned to the T2.1 base-predictions parquet's index so the walk-forward
    structure exactly matches extract_pruned_svi_predictions.py. IV is taken
    from vol_arb's ``svi_iv_matched`` (the SVI surface evaluated at the exact
    straddle (K, T)) with fallback to ``iv`` and ``dvol_close``, then
    converted from percent to the log-fraction units of ``y_true``.

    Returns
    -------
    pd.Series
        Named "direction", int8, indexed by the base predictions parquet
        index (after restriction to rows where log IV is finite).
    """
    if not BASE_PREDS_PATH.exists():
        raise FileNotFoundError(
            f"Base predictions parquet not found at {BASE_PREDS_PATH}. "
            "Run run_walkforward_pruned_svi.py and "
            "extract_pruned_svi_predictions.py first.",
        )
    if not VOL_ARB_PATH.exists():
        raise FileNotFoundError(
            f"vol_arb parquet not found at {VOL_ARB_PATH}.",
        )
    preds = pd.read_parquet(BASE_PREDS_PATH)
    va = pd.read_parquet(VOL_ARB_PATH)

    # IV fallback chain mirrors VolArbBacktester.load_data()
    iv_source = (
        va["svi_iv_matched"].fillna(va["iv"]).fillna(va["dvol_close"])
    )
    # Align on the base predictions index (145k rows, 2022-02..2026-03)
    iv_aligned = iv_source.reindex(preds.index)
    # Convert percent → log fraction (match ln-RV label scale)
    log_iv = np.log(iv_aligned / 100.0)

    y_true = preds["y_true"]
    valid = log_iv.notna() & y_true.notna()
    if not valid.all():
        n_drop = int((~valid).sum())
        logger.info(
            "Dropped %d rows with missing IV or label for edge target",
            n_drop,
        )

    direction = pd.Series(
        (y_true[valid] > log_iv[valid]).astype(np.int8),
        index=preds.index[valid],
        name="direction",
    )
    logger.info(
        "Loaded EDGE direction target: %d rows, positive rate %.3f "
        "(range %s..%s)",
        len(direction), float(direction.mean()),
        direction.index.min(), direction.index.max(),
    )
    return direction


def _load_residual_direction_target() -> pd.Series:
    """Load the RESIDUAL target: did realized log-RV exceed the base forecast?

    direction = 1 when y_true > y_pred_base, else 0. This is the sign of
    the base regression's residual on its OOS test window. Kept as a
    research variant — note that this target is close to noise on the
    same feature set that T2.1 already optimized against.
    """
    if not BASE_PREDS_PATH.exists():
        raise FileNotFoundError(
            f"Base predictions parquet not found at {BASE_PREDS_PATH}.",
        )
    df = pd.read_parquet(BASE_PREDS_PATH)
    required = {"y_pred", "y_true"}
    missing = required - set(df.columns)
    if missing:
        raise KeyError(
            f"Base predictions parquet missing columns: {missing}. "
            f"Available: {list(df.columns)}",
        )
    direction = (df["y_true"] > df["y_pred"]).astype(np.int8)
    direction.name = "direction"
    logger.info(
        "Loaded RESIDUAL direction target: %d rows, positive rate %.3f "
        "(range %s..%s)",
        len(direction), float(direction.mean()),
        direction.index.min(), direction.index.max(),
    )
    return direction


def _load_direction_target(target: str) -> pd.Series:
    """Dispatch between the supported target variants."""
    if target == "edge":
        return _load_edge_direction_target()
    if target == "residual":
        return _load_residual_direction_target()
    raise ValueError(
        f"Unknown target '{target}'; choose from {TARGET_CHOICES}",
    )


def _make_classifier_objective(
    dtrain: lgb.Dataset,
    dval: lgb.Dataset,
    X_val: np.ndarray,
    y_val: np.ndarray,
    device: str,
):
    """Optuna objective for binary-classification LightGBM.

    Mirrors ``vol_model.run_walkforward._make_objective`` but swaps the QLIKE
    objective for binary logloss with class imbalance compensation. Returns
    validation log loss so the outer Optuna study minimises.
    """

    def objective(trial: optuna.trial.Trial) -> float:
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "is_unbalance": True,
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float(
                "learning_rate", 0.001, 0.05, log=True,
            ),
            "min_data_in_leaf": trial.suggest_int(
                "min_data_in_leaf", 50, 500,
            ),
            "lambda_l1": trial.suggest_float(
                "lambda_l1", 1e-3, 10.0, log=True,
            ),
            "lambda_l2": trial.suggest_float(
                "lambda_l2", 1e-3, 10.0, log=True,
            ),
            "colsample_bytree": trial.suggest_float(
                "colsample_bytree", 0.4, 0.9,
            ),
            "subsample": 0.8,
            "device": device,
            "num_threads": -1 if device == "cpu" else 1,
            "verbosity": -1,
            "seed": 42,
        }

        model = lgb.train(
            params, dtrain,
            num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )
        preds = model.predict(X_val)
        preds_clipped = np.clip(preds, 1e-7, 1.0 - 1e-7)
        return float(log_loss(y_val, preds_clipped))

    return objective


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward directional classifier head (W4.1)",
    )
    parser.add_argument(
        "--start", default="2023-06",
        help=(
            "First month to retrain (inclusive), YYYY-MM. Default 2023-06 "
            f"is the earliest month with ≥{MIN_TRAIN_MONTHS}mo train + 4mo "
            "val given the base predictions parquet starts 2022-02."
        ),
    )
    parser.add_argument(
        "--end", default="2026-03",
        help="Last month to retrain (inclusive), YYYY-MM",
    )
    parser.add_argument(
        "--max-trials", type=int, default=200,
        help="Optuna trial budget per month (default 200; spot-check: 100)",
    )
    parser.add_argument(
        "--patience", type=int, default=50,
        help="Optuna patience (default 50; spot-check: 30)",
    )
    parser.add_argument(
        "--target", choices=TARGET_CHOICES, default="edge",
        help=(
            "Classification target. 'edge' (default) = sign(y_true - log(IV)) — "
            "the direct edge predictor matching the Stage 17.1f oracle test. "
            "'residual' = sign(y_true - y_pred_base) — research variant."
        ),
    )
    parser.add_argument(
        "--output-dir", type=str, default="",
        help=(
            "Where to save the per-month classifier models. Default: "
            "vol_model/output/models/classifier_head_{target}_v1/"
        ),
    )
    args = parser.parse_args()

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = OUTPUT_ROOT / f"classifier_head_{args.target}_v1"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  WALK-FORWARD DIRECTIONAL CLASSIFIER HEAD (W4.1)")
    print("=" * 70)
    print(f"  Target variant:  {args.target}")
    print(f"  Base preds:      {BASE_PREDS_PATH}")
    print(f"  Reuse TransVol:  {HYBRID_DIR}")
    print(f"  Save models to:  {output_dir}")
    print(f"  Optuna:          {args.max_trials} trials, patience {args.patience}")
    print(f"  Min train win:   {MIN_TRAIN_MONTHS} months from target start")
    print("=" * 70)

    logger.info("Loading base features...")
    vol_config = VolModelConfig()
    X, _, _ = build_dataset(vol_config)

    pruned = pd.read_csv(PRUNED_PATH, header=None)[0].tolist()
    pruned_cols_v2 = [
        c for c in pruned if c in X.columns and c not in DROP_FEATURES
    ]
    X_pruned = X[pruned_cols_v2]

    drop_prefixes = (
        "ofi_", "depth_mid", "oi_change", "oi_zscore", "oi_acceleration",
    )
    tv_cols = [c for c in X.columns if not c.startswith(drop_prefixes)]
    X_tv = X[tv_cols]

    from vol_model.labels import load_labels
    labels = load_labels(vol_config)

    svi_add = _load_svi_atm_1d()
    direction_target = _load_direction_target(args.target)

    target_start = direction_target.index.min()
    min_V = target_start + pd.DateOffset(months=MIN_TRAIN_MONTHS)
    logger.info(
        "Target coverage: %s to %s (%d rows)",
        direction_target.index.min(),
        direction_target.index.max(),
        len(direction_target),
    )
    logger.info("Effective min-V: %s", min_V)

    start = pd.Timestamp(args.start + "-01", tz="UTC")
    end = pd.Timestamp(args.end + "-01", tz="UTC")
    months = pd.date_range(start, end, freq="MS")

    results: list[dict] = []
    t_start = time.time()

    for i, M in enumerate(months):
        month_str = M.strftime("%Y-%m")
        lgb_path = output_dir / f"{month_str}.txt"

        if lgb_path.exists():
            logger.info(
                "[%d/%d] %s: cached, skipping",
                i + 1, len(months), month_str,
            )
            continue

        tv_path = HYBRID_DIR / f"{month_str}_transvol.pt"
        if not tv_path.exists():
            logger.warning(
                "[%d/%d] %s: no TransVol checkpoint at %s — skipping",
                i + 1, len(months), month_str, tv_path,
            )
            continue

        test_end = M + pd.DateOffset(months=1)
        # V constraint differs from T2.1: the classifier requires ≥
        # MIN_TRAIN_MONTHS of target coverage before the val slice begins.
        V = max(M - pd.DateOffset(months=4), min_V)
        if V >= M:
            logger.info(
                "[%d/%d] %s: insufficient target history "
                "(would leave no val slice); skipping",
                i + 1, len(months), month_str,
            )
            continue

        logger.info("=" * 60)
        logger.info(
            "[%d/%d] %s: train=[%s, %s), val=[%s, %s), test=%s",
            i + 1, len(months), month_str,
            target_start.strftime("%Y-%m"), V.strftime("%Y-%m"),
            V.strftime("%Y-%m"), M.strftime("%Y-%m"), month_str,
        )
        t0 = time.time()

        # Extract TransVol embeddings from the frozen checkpoint
        try:
            emb_df = _extract_compact_embeddings(
                tv_path, X_tv, labels, test_end,
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception(
                "%s: TransVol embedding extraction failed: %s",
                month_str, exc,
            )
            continue

        # Build the same 44-feature matrix as T2.1
        common = (
            X_pruned.index.intersection(emb_df.index)
            .intersection(svi_add.index)
            .intersection(direction_target.index)
        )
        X_month = pd.concat(
            [emb_df.loc[common], X_pruned.loc[common], svi_add.loc[common]],
            axis=1,
        )
        y_month = direction_target.loc[X_month.index]

        # Drop rows with NaN svi_atm_1d (5% coverage gap) — mirror T2.1
        before = len(X_month)
        X_month = X_month.dropna(subset=ADD_SVI_FEATURES)
        y_month = y_month.loc[X_month.index]
        dropped_nan = before - len(X_month)
        if dropped_nan > 0:
            logger.info(
                "  Dropped %d rows with NaN svi_atm_1d (%.1f%%)",
                dropped_nan, 100 * dropped_nan / max(before, 1),
            )

        # Temporal masks; train window is explicitly bounded by target_start
        train_mask = (X_month.index >= target_start) & (X_month.index < V)
        val_mask = (X_month.index >= V) & (X_month.index < M)
        test_mask = (X_month.index >= M) & (X_month.index < test_end)

        n_train = int(train_mask.sum())
        n_val = int(val_mask.sum())
        n_test = int(test_mask.sum())
        logger.info(
            "  Data: train=%d, val=%d, test=%d, features=%d",
            n_train, n_val, n_test, X_month.shape[1],
        )
        if n_train < 500 or n_val < 50 or n_test == 0:
            logger.warning(
                "%s: insufficient data (train=%d, val=%d, test=%d)",
                month_str, n_train, n_val, n_test,
            )
            continue

        X_tr, y_tr = X_month[train_mask], y_month[train_mask]
        X_val, y_val = X_month[val_mask], y_month[val_mask]
        X_test, y_test = X_month[test_mask], y_month[test_mask]

        direction_rate_train = float(y_tr.mean())
        logger.info("  Train positive rate: %.3f", direction_rate_train)

        # Winsorize on train only (same as T2.1)
        lo, hi = X_tr.quantile(0.01), X_tr.quantile(0.99)
        X_tr_w = X_tr.clip(lo, hi, axis=1)
        X_val_w = X_val.clip(lo, hi, axis=1)
        X_test_w = X_test.clip(lo, hi, axis=1)

        # Optuna search — binary classification
        dtrain = lgb.Dataset(
            X_tr_w.values, label=y_tr.values, free_raw_data=False,
            params={"feature_pre_filter": False},
        )
        dval = lgb.Dataset(
            X_val_w.values, label=y_val.values,
            reference=dtrain, free_raw_data=False,
        )
        dtrain.construct()
        dval.construct()

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5),
        )
        obj = _make_classifier_objective(
            dtrain, dval, X_val_w.values, y_val.values, "cpu",
        )
        study.optimize(
            obj,
            n_trials=args.max_trials,
            callbacks=[OptunaPatience(args.patience)],
            show_progress_bar=False,
        )
        n_trials = len(study.trials)
        best_logloss_val = float(study.best_value)

        # Final retrain on [target_start, M)
        full_mask = (X_month.index >= target_start) & (X_month.index < M)
        X_full = X_month[full_mask].clip(lo, hi, axis=1)
        y_full = y_month[full_mask]

        best = study.best_params.copy()
        best["objective"] = "binary"
        best["metric"] = "binary_logloss"
        best["is_unbalance"] = True
        best["subsample"] = 0.8
        best["device"] = "cpu"
        best["num_threads"] = -1
        best["verbosity"] = -1
        best["seed"] = 42

        dtrain_full = lgb.Dataset(
            X_full.values, label=y_full.values, free_raw_data=False,
        )
        model = lgb.train(
            best, dtrain_full, num_boost_round=2000,
            valid_sets=[dval],
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        # Predict + evaluate on test month
        test_probs = model.predict(X_test_w.values)
        test_probs_clip = np.clip(test_probs, 1e-7, 1.0 - 1e-7)
        test_logloss = float(log_loss(y_test.values, test_probs_clip))
        try:
            test_auc = float(roc_auc_score(y_test.values, test_probs))
        except ValueError:
            # Single-class test slice — very rare; report NaN
            test_auc = float("nan")
        test_brier = float(brier_score_loss(y_test.values, test_probs))

        # Save the model + feature manifest
        model.save_model(str(lgb_path))
        (output_dir / f"{month_str}_features.json").write_text(
            json.dumps(list(X_month.columns)),
        )

        elapsed = time.time() - t0
        logger.info(
            "  [%d/%d] %s: auc=%.4f  logloss=%.4f  brier=%.4f  "
            "trials=%d  (%.0fs)",
            i + 1, len(months), month_str, test_auc, test_logloss,
            test_brier, n_trials, elapsed,
        )

        results.append({
            "month": month_str,
            "auc_test": test_auc,
            "logloss_test": test_logloss,
            "brier_test": test_brier,
            "best_logloss_val": best_logloss_val,
            "direction_rate_train": direction_rate_train,
            "n_train": n_train,
            "n_val": n_val,
            "n_test": n_test,
            "n_features": X_month.shape[1],
            "n_trials": n_trials,
            "time_s": elapsed,
        })

    # Summary
    total_elapsed = time.time() - t_start
    print()
    print("=" * 70)
    print(f"  CLASSIFIER HEAD WALK-FORWARD — {len(results)} months trained")
    print(f"  Total elapsed: {total_elapsed / 3600:.2f}h")
    print("=" * 70)
    if results:
        print(
            f"  {'Month':>8s}  {'AUC':>8s}  {'Logloss':>10s}  "
            f"{'Brier':>8s}  {'PosRate':>8s}  {'Trials':>7s}"
        )
        print("-" * 70)
        for r in results:
            print(
                f"  {r['month']:>8s}  {r['auc_test']:>8.4f}  "
                f"{r['logloss_test']:>10.4f}  {r['brier_test']:>8.4f}  "
                f"{r['direction_rate_train']:>8.3f}  {r['n_trials']:>7d}"
            )
        aucs = [r["auc_test"] for r in results if not np.isnan(r["auc_test"])]
        loglosses = [r["logloss_test"] for r in results]
        briers = [r["brier_test"] for r in results]
        if aucs:
            print("-" * 70)
            print(
                f"  {'AVG':>8s}  {np.mean(aucs):>8.4f}  "
                f"{np.mean(loglosses):>10.4f}  {np.mean(briers):>8.4f}"
            )
        print("=" * 70)

        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps({
            "target": args.target,
            "total_hours": total_elapsed / 3600,
            "n_months": len(results),
            "mean_auc": float(np.mean(aucs)) if aucs else None,
            "mean_logloss": float(np.mean(loglosses)) if loglosses else None,
            "mean_brier": float(np.mean(briers)) if briers else None,
            "min_train_months": MIN_TRAIN_MONTHS,
            "base_preds_path": str(BASE_PREDS_PATH),
            "vol_arb_path": (
                str(VOL_ARB_PATH) if args.target == "edge" else None
            ),
            "per_month": results,
        }, indent=2, default=str))
        logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
