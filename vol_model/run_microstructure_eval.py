#!/usr/bin/env python
"""Evaluate microstructure features: prune 121 features, train 3 models, compare.

Trains expanding-window models with the new OFI + corrected VPIN features
and compares against existing models in models/expanding/.

Saves to models/expanding_microstructure/ (never touches expanding/).

Usage:
    PYTHONPATH=. python vol_model/run_microstructure_eval.py
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vol_model.config import VolModelConfig
from vol_model.model import qlike_objective, qlike_loss, qlike_metric, build_lgbm_params
from vol_model.pipeline import build_dataset
from vol_model.run_walkforward import prune_features_for_month

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("microstructure_eval")

# Months to train (within L2 coverage: Apr 2023 - Feb 2026)
# Pick 3 spread across the range
TEST_MONTHS = ["2023-10", "2024-06", "2025-06"]

# Output directory (separate from existing expanding models)
MODEL_DIR = Path("vol_model/output/models/expanding_microstructure")
EXISTING_DIR = Path("vol_model/output/models/expanding")


def train_month(
    M: pd.Timestamp,
    X: pd.DataFrame,
    y: pd.Series,
    config: VolModelConfig,
    model_dir: Path,
    features: list[str] | None = None,
) -> dict:
    """Train one expanding-window month. Returns metrics dict."""
    month_str = M.strftime("%Y-%m")
    X_used = X[features] if features else X

    train_mask = (X_used.index < M)
    test_start = M
    test_end = M + pd.DateOffset(months=1)
    test_mask = (X_used.index >= test_start) & (X_used.index < test_end)

    n_train = train_mask.sum()
    n_test = test_mask.sum()

    if n_train < 100 or n_test == 0:
        logger.warning("  %s: insufficient data (train=%d, test=%d)", month_str, n_train, n_test)
        return {}

    X_train = X_used[train_mask]
    y_train = y[train_mask]
    X_test = X_used[test_mask]
    y_test = y[test_mask]

    # Winsorize
    lo = X_train.quantile(config.winsorize_lo)
    hi = X_train.quantile(config.winsorize_hi)
    X_train_w = X_train.clip(lo, hi, axis=1)
    X_test_w = X_test.clip(lo, hi, axis=1)

    # Train/val split (85/15)
    val_split = int(len(X_train_w) * 0.85)
    dtrain = lgb.Dataset(
        X_train_w.iloc[:val_split], label=y_train.iloc[:val_split].values,
        free_raw_data=False,
    )
    dval = lgb.Dataset(
        X_train_w.iloc[val_split:], label=y_train.iloc[val_split:].values,
        reference=dtrain, free_raw_data=False,
    )

    params = build_lgbm_params(config)
    t0 = time.time()
    model = lgb.train(
        params, dtrain,
        num_boost_round=config.n_estimators,
        valid_sets=[dval],
        feval=qlike_metric,
        callbacks=[lgb.early_stopping(config.early_stopping_rounds), lgb.log_evaluation(0)],
    )
    train_time = time.time() - t0

    preds = model.predict(X_test_w.values)
    y_test_arr = y_test.values
    fold_qlike = qlike_loss(y_test_arr, preds)
    fold_rmse = float(np.sqrt(np.mean((y_test_arr - preds) ** 2)))
    rank_ic = float(spearmanr(y_test_arr, preds).statistic)

    # Save model
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_dir / f"{month_str}.txt"))
    if features:
        (model_dir / f"{month_str}_features.json").write_text(json.dumps(features))

    logger.info("  %s: QLIKE=%.4f RMSE=%.4f RankIC=%.3f (train=%d, test=%d, feats=%d, %.1fs)",
                month_str, fold_qlike, fold_rmse, rank_ic, n_train, n_test,
                len(X_used.columns), train_time)

    return {
        "month": month_str, "qlike": fold_qlike, "rmse": fold_rmse,
        "rank_ic": rank_ic, "n_train": n_train, "n_test": n_test,
        "n_features": len(X_used.columns), "train_time": train_time,
        "preds": preds, "y_test": y_test_arr, "test_index": X_test.index,
    }


def evaluate_existing_model(
    M: pd.Timestamp,
    X: pd.DataFrame,
    y: pd.Series,
    config: VolModelConfig,
    model_dir: Path,
) -> dict | None:
    """Load existing model and evaluate on the same test set."""
    month_str = M.strftime("%Y-%m")
    model_path = model_dir / f"{month_str}.txt"

    if not model_path.exists():
        logger.warning("  %s: no existing model at %s", month_str, model_path)
        return None

    # Load feature list if available
    feat_path = model_dir / f"{month_str}_features.json"
    if feat_path.exists():
        features = json.loads(feat_path.read_text())
    else:
        # Try pruned_features.csv
        pruned_path = Path("vol_model/output/pruned_features.csv")
        if pruned_path.exists():
            features = pd.read_csv(pruned_path, header=None)[0].tolist()
        else:
            features = None

    X_used = X[features] if features else X
    model = lgb.Booster(model_file=str(model_path))

    train_mask = (X_used.index < M)
    test_start = M
    test_end = M + pd.DateOffset(months=1)
    test_mask = (X_used.index >= test_start) & (X_used.index < test_end)

    n_test = test_mask.sum()
    if n_test == 0:
        return None

    X_train = X_used[train_mask]
    X_test = X_used[test_mask]
    y_test = y[test_mask]

    lo = X_train.quantile(config.winsorize_lo)
    hi = X_train.quantile(config.winsorize_hi)
    X_test_w = X_test.clip(lo, hi, axis=1)

    # Filter to features the model knows about
    model_features = model.feature_name()
    if model_features and model_features[0].startswith("Column_"):
        # Model was trained with positional column names — use column order
        X_test_w = X_test_w.iloc[:, :len(model_features)]
    else:
        missing = [f for f in model_features if f not in X_test_w.columns]
        if missing:
            logger.warning("  %s: model expects %d features, %d missing",
                           month_str, len(model_features), len(missing))
            return None

    preds = model.predict(X_test_w.values)
    y_test_arr = y_test.values
    fold_qlike = qlike_loss(y_test_arr, preds)
    fold_rmse = float(np.sqrt(np.mean((y_test_arr - preds) ** 2)))
    rank_ic = float(spearmanr(y_test_arr, preds).statistic)

    return {
        "month": month_str, "qlike": fold_qlike, "rmse": fold_rmse,
        "rank_ic": rank_ic, "n_test": n_test,
        "n_features": len(model_features) if model_features else X_test_w.shape[1],
    }


def main():
    config = VolModelConfig()

    print("=" * 70)
    print("  MICROSTRUCTURE FEATURE EVALUATION")
    print("=" * 70)

    # ── Step 1: Build dataset with new features ──
    logger.info("Step 1: Building dataset with microstructure features...")
    t0 = time.time()
    X, y, t1 = build_dataset(config)
    logger.info("  Dataset: %d rows, %d features (%.1fs)",
                len(X), len(X.columns), time.time() - t0)
    logger.info("  Features: %s", list(X.columns))

    # Check for new features
    ofi_cols = [c for c in X.columns if "ofi" in c.lower() or "depth" in c.lower()]
    vpin_cols = [c for c in X.columns if "vpin" in c.lower()]
    logger.info("  OFI features: %s", ofi_cols)
    logger.info("  VPIN features: %s", vpin_cols)

    # ── Step 2: Prune features ──
    logger.info("\nStep 2: Pruning %d features...", len(X.columns))

    # Use training data up to the first test month for pruning
    prune_cutoff = pd.Timestamp(TEST_MONTHS[0] + "-01", tz="UTC")
    prune_mask = X.index < prune_cutoff
    t0 = time.time()
    pruned_features = prune_features_for_month(X[prune_mask], y[prune_mask])
    logger.info("  Pruned: %d -> %d features (%.1fs)",
                len(X.columns), len(pruned_features), time.time() - t0)
    logger.info("  Surviving features: %s", pruned_features)

    # Check which new features survived
    new_surviving = [f for f in pruned_features if "ofi" in f or "depth" in f]
    logger.info("  New microstructure features surviving: %s", new_surviving)

    # Save pruned feature list
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    (MODEL_DIR / "pruned_features.json").write_text(json.dumps(pruned_features))

    # ── Step 3: Train 3 months ──
    print(f"\n{'='*70}")
    print(f"  TRAINING {len(TEST_MONTHS)} MONTHS (expanding window)")
    print(f"{'='*70}")

    new_results = []
    old_results = []

    for month_str in TEST_MONTHS:
        M = pd.Timestamp(month_str + "-01", tz="UTC")
        logger.info("\n--- %s ---", month_str)

        # Train new model with pruned microstructure features
        logger.info("  New (microstructure, %d features):", len(pruned_features))
        new_r = train_month(M, X, y, config, MODEL_DIR, pruned_features)
        if new_r:
            new_results.append(new_r)

        # Evaluate existing model
        logger.info("  Existing (expanding):")
        old_r = evaluate_existing_model(M, X, y, config, EXISTING_DIR)
        if old_r:
            old_results.append(old_r)
            logger.info("  Existing: QLIKE=%.4f RMSE=%.4f RankIC=%.3f (feats=%d)",
                        old_r["qlike"], old_r["rmse"], old_r["rank_ic"], old_r["n_features"])
        else:
            logger.info("  Existing: no model available")

    # ── Step 4: Comparison ──
    print(f"\n{'='*70}")
    print(f"  COMPARISON")
    print(f"{'='*70}")

    header = f"  {'Month':>8s}  {'Model':>16s}  {'QLIKE':>8s}  {'RMSE':>8s}  {'RankIC':>8s}  {'Feats':>6s}"
    print(header)
    print("-" * 70)

    for month_str in TEST_MONTHS:
        new_r = next((r for r in new_results if r["month"] == month_str), None)
        old_r = next((r for r in old_results if r["month"] == month_str), None)

        if new_r:
            print(f"  {month_str:>8s}  {'+ microstructure':>16s}"
                  f"  {new_r['qlike']:>8.4f}  {new_r['rmse']:>8.4f}"
                  f"  {new_r['rank_ic']:>8.3f}  {new_r['n_features']:>6d}")
        if old_r:
            print(f"  {'':>8s}  {'baseline':>16s}"
                  f"  {old_r['qlike']:>8.4f}  {old_r['rmse']:>8.4f}"
                  f"  {old_r['rank_ic']:>8.3f}  {old_r['n_features']:>6d}")
        if new_r and old_r:
            dq = new_r["qlike"] - old_r["qlike"]
            dr = new_r["rank_ic"] - old_r["rank_ic"]
            better = "BETTER" if dq < 0 else "WORSE"
            print(f"  {'':>8s}  {'delta':>16s}"
                  f"  {dq:>+8.4f}  {'':>8s}"
                  f"  {dr:>+8.3f}  {'':>6s}  {better}")
        print()

    # Summary
    if new_results and old_results:
        avg_new_qlike = np.mean([r["qlike"] for r in new_results])
        avg_old_qlike = np.mean([r["qlike"] for r in old_results])
        avg_new_ic = np.mean([r["rank_ic"] for r in new_results])
        avg_old_ic = np.mean([r["rank_ic"] for r in old_results])
        print(f"  Average QLIKE:  new={avg_new_qlike:.4f}  old={avg_old_qlike:.4f}  delta={avg_new_qlike-avg_old_qlike:+.4f}")
        print(f"  Average RankIC: new={avg_new_ic:.3f}  old={avg_old_ic:.3f}  delta={avg_new_ic-avg_old_ic:+.3f}")


if __name__ == "__main__":
    main()
