#!/usr/bin/env python
"""Multi-timeframe 3-month spot-check: LightGBM-only, no TransVol/SVI.

For each timeframe, trains a quick LightGBM on base features using an
expanding window through 3 test months (2023-10, 2023-11, 2023-12),
then reports OOS quality metrics for comparison.

This validates whether the TSRV-based vol prediction signal exists at
each bar frequency before investing in full walk-forward + TransVol.

Usage:
    PYTHONPATH=. python scripts/multi_tf_spotcheck.py
    PYTHONPATH=. python scripts/multi_tf_spotcheck.py --tfs 300 1800 3600
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vol_model.config import VolModelConfig
from vol_model.model import qlike_loss, qlike_metric
from vol_model.pipeline import build_dataset

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("multi_tf_spot")

DEFAULT_TFS = [60, 300, 900, 1800, 3600, 14400, 28800, 43200, 86400]
# 3-month test window — use 2025 H2 so all TFs (including 5m starting 2024-03) have data
TEST_MONTHS = ["2025-07", "2025-08", "2025-09"]


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    diff = y_true - y_pred
    return float(np.mean(np.exp(diff) - diff - 1.0))


def direction_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    med = np.median(y_true)
    return float(np.mean((y_pred > med) == (y_true > med)))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-20))


def run_spotcheck(tf_seconds: int, n_trials: int = 50, asset: str = "BTCUSDT") -> dict:
    """Train + evaluate LightGBM at one timeframe."""
    tf_min = tf_seconds // 60
    label = f"{tf_min}m" if tf_seconds < 86400 else f"{tf_seconds // 3600}h"
    logger.info("=" * 60)
    logger.info("  Timeframe: %s (%ds) [%s]", label, tf_seconds, asset)
    logger.info("=" * 60)

    t0 = time.time()

    # Load data
    config = VolModelConfig(freq_s=tf_seconds, asset=asset)
    try:
        X, y, t1 = build_dataset(config)
    except Exception as exc:
        logger.error("Failed to build dataset for %s: %s", label, exc)
        return {"timeframe": label, "tf_seconds": tf_seconds, "status": "ERROR",
                "error": str(exc), "n_bars": 0}

    logger.info("  Dataset: %d rows, %d features", len(X), X.shape[1])

    # Use pruned feature list if available (matches 15m production setup)
    pruned_path = Path("vol_model/output/pruned_features.csv")
    if pruned_path.exists():
        pruned = pd.read_csv(pruned_path, header=None)[0].tolist()
        available = [c for c in pruned if c in X.columns]
        if len(available) > 20:
            X = X[available]
            logger.info("  Using pruned features: %d of %d available", len(available), len(pruned))

    # Collect OOS predictions across 3 test months
    all_preds = []
    all_true = []
    data_start = X.index.min()

    for month_str in TEST_MONTHS:
        M = pd.Timestamp(month_str + "-01", tz="UTC")
        test_end = M + pd.DateOffset(months=1)

        # 4-month validation window
        V = max(M - pd.DateOffset(months=4), data_start + pd.DateOffset(months=6))

        train_mask = X.index < V
        val_mask = (X.index >= V) & (X.index < M)
        test_mask = (X.index >= M) & (X.index < test_end)

        n_train, n_val, n_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
        logger.info("  %s: train=%d, val=%d, test=%d", month_str, n_train, n_val, n_test)

        if n_train < 100 or n_test == 0:
            logger.warning("  %s: insufficient data, skipping", month_str)
            continue

        X_tr, y_tr = X[train_mask].values, y[train_mask].values
        X_val, y_val = X[val_mask].values, y[val_mask].values
        X_test, y_test = X[test_mask].values, y[test_mask].values

        # Winsorize on train
        lo = np.percentile(X_tr, 1, axis=0)
        hi = np.percentile(X_tr, 99, axis=0)
        X_tr = np.clip(X_tr, lo, hi)
        X_val = np.clip(X_val, lo, hi)
        X_test = np.clip(X_test, lo, hi)

        # Quick LightGBM: fixed good hyperparams instead of full Optuna sweep
        params = {
            "objective": "regression",
            "metric": "custom",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 50,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.1,
            "lambda_l2": 1.0,
            "verbose": -1,
            "n_jobs": 4,
        }

        ds_tr = lgb.Dataset(X_tr, y_tr)
        ds_val = lgb.Dataset(X_val, y_val, reference=ds_tr)

        model = lgb.train(
            params,
            ds_tr,
            num_boost_round=1000,
            valid_sets=[ds_val],
            valid_names=["val"],
            feval=qlike_metric,
            callbacks=[
                lgb.early_stopping(50, verbose=False),
                lgb.log_evaluation(0),
            ],
        )

        preds = model.predict(X_test)
        all_preds.append(preds)
        all_true.append(y_test)

        ql = qlike(y_test, preds)
        logger.info("  %s OOS QLIKE=%.4f n_trees=%d", month_str, ql, model.best_iteration)

    elapsed = time.time() - t0

    if not all_preds:
        return {"timeframe": label, "tf_seconds": tf_seconds, "status": "NO_DATA",
                "n_bars": 0, "elapsed_s": elapsed}

    # Aggregate OOS metrics
    yt = np.concatenate(all_true)
    yp = np.concatenate(all_preds)

    ql = qlike(yt, yp)
    r2 = r_squared(yt, yp)
    da = direction_accuracy(yt, yp)
    rho, _ = spearmanr(yt, yp)

    return {
        "timeframe": label,
        "tf_seconds": tf_seconds,
        "n_bars": len(yt),
        "qlike": ql,
        "r2": r2,
        "direction_acc": da,
        "spearman_rho": rho,
        "elapsed_s": elapsed,
        "status": "PASS" if da > 0.55 else "MARGINAL" if da > 0.52 else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-TF spot-check")
    parser.add_argument("--tfs", nargs="+", type=int, default=DEFAULT_TFS)
    parser.add_argument("--trials", type=int, default=50)
    parser.add_argument("--asset", type=str, default="BTCUSDT")
    args = parser.parse_args()

    results = []
    for tf in args.tfs:
        result = run_spotcheck(tf, n_trials=args.trials, asset=args.asset)
        results.append(result)

    # Print comparison table
    print()
    print("=" * 85)
    print(f"  MULTI-TIMEFRAME SPOT-CHECK (3-month OOS, LightGBM-only) [{args.asset}]")
    print("=" * 85)
    print(f"  {'TF':<8} {'N bars':>8} {'QLIKE':>10} {'R²':>10} {'Dir Acc':>10} {'Spearman':>10} {'Gate':>8}")
    print("-" * 85)

    for r in results:
        if r.get("n_bars", 0) == 0:
            print(f"  {r['timeframe']:<8} {'---':>8} {'---':>10} {'---':>10} {'---':>10} {'---':>10} {r['status']:>8}")
        else:
            print(
                f"  {r['timeframe']:<8} {r['n_bars']:>8} "
                f"{r['qlike']:>10.4f} {r['r2']:>10.4f} "
                f"{r['direction_acc']:>10.2%} {r['spearman_rho']:>10.3f} {r['status']:>8}"
            )
    print("=" * 85)

    # Save
    out = pd.DataFrame(results)
    out_path = Path(f"vol_model/output/multi_tf_spotcheck_{args.asset}.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
