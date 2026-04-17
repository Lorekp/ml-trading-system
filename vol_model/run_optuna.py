#!/usr/bin/env python
"""Optuna hyperparameter optimization for the vol model.

Searches LightGBM hyperparameters using TPE (Tree-structured Parzen Estimator)
with purged K-fold CV and custom QLIKE loss. Two-layer efficiency:
  1. LightGBM early stopping (50 rounds) — kills bad boosting iterations
  2. Optuna fold-level pruning (MedianPruner) — kills bad hyperparams after 2 folds

Usage:
    PYTHONPATH=. python vol_model/run_optuna.py
    PYTHONPATH=. python vol_model/run_optuna.py --n-trials 200
    PYTHONPATH=. python vol_model/run_optuna.py --n-trials 200 --retrain
"""

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner

from vol_model.config import VolModelConfig
from vol_model.pipeline import build_dataset
from vol_model.model import qlike_objective, qlike_metric, qlike_loss
from src.afml.cross_val import PurgedKFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vol_model.optuna")
optuna.logging.set_verbosity(optuna.logging.WARNING)


def precompute_folds(X, y, t1, config):
    """Pre-compute CV splits and per-fold winsorization bounds.

    Returns list of dicts, one per fold:
        train_idx, test_idx, lo_vals, hi_vals, dtrain, dval, y_test
    """
    cv = PurgedKFold(
        n_splits=config.n_splits,
        t1=t1,
        pct_embargo=config.pct_embargo,
        label_span=1,
    )

    folds = []
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # Winsorization bounds (fit on train only)
        lo_vals = X_train.quantile(config.winsorize_lo)
        hi_vals = X_train.quantile(config.winsorize_hi)

        # Apply winsorization
        X_train_w = X_train.clip(lo_vals, hi_vals, axis=1)
        X_test_w = X_test.clip(lo_vals, hi_vals, axis=1)

        # Build LGB datasets (reused across all trials)
        dtrain = lgb.Dataset(X_train_w, label=y_train.values, free_raw_data=False)
        dval = lgb.Dataset(X_test_w, label=y_test.values, reference=dtrain, free_raw_data=False)

        folds.append({
            "fold_i": fold_i,
            "train_idx": train_idx,
            "test_idx": test_idx,
            "dtrain": dtrain,
            "dval": dval,
            "X_test": X_test_w,
            "y_test": y_test.values,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
        })

        logger.info("  Fold %d: train=%d, test=%d", fold_i + 1, len(train_idx), len(test_idx))

    return folds


def create_objective(folds):
    """Create the Optuna objective function with pre-computed folds."""

    def objective(trial):
        params = {
            "objective": qlike_objective,
            "num_leaves": trial.suggest_int("num_leaves", 16, 128),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.05, log=True),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 50, 500),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 10.0, log=True),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "subsample": 0.8,
            "feature_pre_filter": False,
            "num_threads": -1,
            "verbosity": -1,
            "seed": 42,
        }

        fold_qlikes = []
        fold_iters = []

        for fold in folds:
            model = lgb.train(
                params,
                fold["dtrain"],
                num_boost_round=2000,
                valid_sets=[fold["dval"]],
                feval=qlike_metric,
                callbacks=[
                    lgb.early_stopping(50, verbose=False),
                    lgb.log_evaluation(period=0),  # silent
                ],
            )

            preds = model.predict(fold["X_test"])
            fold_qlike = qlike_loss(fold["y_test"], preds)
            fold_qlikes.append(fold_qlike)
            fold_iters.append(model.best_iteration)

            # Fold-level Optuna pruning
            trial.report(np.mean(fold_qlikes), fold["fold_i"])
            if trial.should_prune():
                raise optuna.TrialPruned()

        mean_qlike = float(np.mean(fold_qlikes))

        # Store per-fold details for the best trial
        trial.set_user_attr("fold_qlikes", fold_qlikes)
        trial.set_user_attr("fold_iters", fold_iters)

        return mean_qlike

    return objective


def run_study(folds, n_trials, seed=42):
    """Run the Optuna study."""
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=seed),
        pruner=MedianPruner(n_startup_trials=10, n_warmup_steps=2),
        study_name="vol_qlike",
    )

    objective = create_objective(folds)

    t0 = time.time()
    n_pruned = 0
    best_so_far = float("inf")

    def callback(study, trial):
        nonlocal n_pruned, best_so_far
        if trial.state == optuna.trial.TrialState.PRUNED:
            n_pruned += 1
            status = "PRUNED"
            val = "---"
        else:
            val = f"{trial.value:.4f}"
            if trial.value < best_so_far:
                best_so_far = trial.value
                status = "* BEST *"
            else:
                status = ""

        elapsed = time.time() - t0
        print(
            f"  Trial {trial.number + 1:>3d}/{n_trials}  "
            f"QLIKE={val:>8s}  best={best_so_far:.4f}  "
            f"pruned={n_pruned}  [{elapsed:.0f}s]  {status}",
            flush=True,
        )

    study.optimize(objective, n_trials=n_trials, callbacks=[callback])

    return study


def save_results(study, asset, output_dir):
    """Save study results to JSON."""
    best = study.best_trial

    results = {
        "best_params": best.params,
        "best_qlike": best.value,
        "best_fold_qlikes": best.user_attrs.get("fold_qlikes", []),
        "best_fold_iters": best.user_attrs.get("fold_iters", []),
        "n_trials": len(study.trials),
        "n_pruned": sum(
            1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
        ),
        "trials": [
            {
                "number": t.number,
                "value": t.value if t.value is not None else None,
                "state": t.state.name,
                "params": t.params,
            }
            for t in study.trials
        ],
    }

    out_path = Path(output_dir) / f"{asset}_optuna_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)

    return out_path


def print_summary(study, current_qlike=0.546):
    """Print final summary."""
    best = study.best_trial
    n_complete = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE
    )
    n_pruned = sum(
        1 for t in study.trials if t.state == optuna.trial.TrialState.PRUNED
    )

    print("\n" + "=" * 60)
    print("  OPTUNA RESULTS")
    print("=" * 60)
    print(f"  Trials: {len(study.trials)} ({n_complete} complete, {n_pruned} pruned)")
    print(f"  Best QLIKE: {best.value:.4f}")
    if current_qlike:
        delta = (best.value - current_qlike) / current_qlike * 100
        print(f"  vs default:  {delta:+.1f}% ({'better' if delta < 0 else 'worse'})")

    print(f"\n  Best parameters:")
    for k, v in sorted(best.params.items()):
        if isinstance(v, float):
            print(f"    {k:>20s}: {v:.6f}")
        else:
            print(f"    {k:>20s}: {v}")

    fold_qlikes = best.user_attrs.get("fold_qlikes", [])
    fold_iters = best.user_attrs.get("fold_iters", [])
    if fold_qlikes:
        print(f"\n  Per-fold breakdown:")
        for i, (q, it) in enumerate(zip(fold_qlikes, fold_iters)):
            print(f"    Fold {i + 1}: QLIKE={q:.4f}  iters={it}")
        print(f"    Std:   {np.std(fold_qlikes):.4f}")

    # Config-ready output
    print(f"\n  For config.py:")
    param_map = {
        "num_leaves": "num_leaves",
        "max_depth": "max_depth",
        "learning_rate": "learning_rate",
        "min_data_in_leaf": "min_child_samples",
        "lambda_l1": "reg_alpha",
        "lambda_l2": "reg_lambda",
        "colsample_bytree": "colsample",
    }
    for optuna_name, config_name in param_map.items():
        v = best.params[optuna_name]
        if isinstance(v, float):
            print(f"    {config_name}={v:.6f},")
        else:
            print(f"    {config_name}={v},")


def retrain_best(X, y, t1, config, best_params):
    """Retrain with best params and run full evaluation."""
    from vol_model.model import train_purged_cv, compute_shap_importance
    from vol_model.evaluate import compute_baselines, evaluate_predictions, print_evaluation

    # Override config with best params
    param_map = {
        "num_leaves": "num_leaves",
        "max_depth": "max_depth",
        "learning_rate": "learning_rate",
        "min_data_in_leaf": "min_child_samples",
        "lambda_l1": "reg_alpha",
        "lambda_l2": "reg_lambda",
        "colsample_bytree": "colsample",
    }
    for optuna_name, config_name in param_map.items():
        if optuna_name in best_params:
            setattr(config, config_name, best_params[optuna_name])

    # Use max iterations from best trial (with headroom)
    config.n_estimators = 2000

    print("\n" + "=" * 60)
    print("  RETRAINING WITH BEST PARAMS")
    print("=" * 60)

    results = train_purged_cv(X, y, t1, config)

    oof = results["oof_preds"]
    valid_mask = ~np.isnan(oof)
    y_valid = y.values[valid_mask]
    oof_valid = oof[valid_mask]

    baselines = compute_baselines(y, tsrv_path=config.get_tsrv_path(), freq_s=config.freq_s)
    baselines_valid = {k: v[valid_mask] for k, v in baselines.items()}

    metrics = evaluate_predictions(y_valid, oof_valid, baselines_valid)
    print_evaluation(metrics, results["fold_metrics"])

    # Save
    out = Path(config.output_dir)
    oof_df = pd.DataFrame({"y_true": y.values, "y_pred": oof}, index=y.index)
    oof_df.to_parquet(out / f"{config.asset}_oof_predictions.parquet")

    for i, model in enumerate(results["models"]):
        model.save_model(str(out / f"{config.asset}_model_fold{i}.txt"))

    # SHAP
    importance = compute_shap_importance(results["models"], X)
    if importance is not None:
        print(f"\n  Top 15 features by SHAP:")
        for _, row in importance.head(15).iterrows():
            print(f"    {row['feature']:>40s}  {row['mean_abs_shap']:.4f}")
        importance.to_csv(out / f"{config.asset}_shap_importance.csv", index=False)

    return results, metrics


def main():
    parser = argparse.ArgumentParser(description="Optuna hyperparameter search for vol model")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--tf", type=int, default=900)
    parser.add_argument("--n-trials", type=int, default=200)
    parser.add_argument("--retrain", action="store_true", help="Retrain with best params after study")
    parser.add_argument("--features", type=str, default=None,
                        help="CSV file with pruned feature names (one per line)")
    args = parser.parse_args()

    config = VolModelConfig(asset=args.asset, freq_s=args.tf)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("=" * 60)
    print(f"  OPTUNA HYPERPARAMETER SEARCH -- {args.asset}")
    print(f"  {args.n_trials} trials, TPE sampler, MedianPruner")
    print("=" * 60)

    # Load dataset
    print("\n[1/3] Loading dataset...")
    X, y, t1 = build_dataset(config)

    # Optional feature pruning
    if args.features:
        feature_list = pd.read_csv(args.features, header=None)[0].tolist()
        X = X[[f for f in feature_list if f in X.columns]]
        print(f"  Pruned to {len(X.columns)} features from {args.features}")

    print(f"  {len(X):,} rows, {len(X.columns)} features")

    # Pre-compute folds
    print(f"\n[2/3] Pre-computing {config.n_splits}-fold CV splits + winsorization...")
    folds = precompute_folds(X, y, t1, config)

    # Run study
    print(f"\n[3/3] Running {args.n_trials} trials...")
    study = run_study(folds, args.n_trials)

    # Results
    print_summary(study)
    out_path = save_results(study, args.asset, config.output_dir)
    print(f"\n  Saved: {out_path}")

    # Optional retrain
    if args.retrain:
        retrain_best(X, y, t1, config, study.best_params)

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
