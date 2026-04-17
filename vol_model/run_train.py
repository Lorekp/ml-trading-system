#!/usr/bin/env python
"""Train the volatility prediction model end-to-end.

Usage:
    PYTHONPATH="." python vol_model/run_train.py
    PYTHONPATH="." python vol_model/run_train.py --asset BTCUSDT --n-splits 5
"""

import argparse
import logging
import time
import json
from pathlib import Path

import numpy as np
import pandas as pd

from vol_model.config import VolModelConfig
from vol_model.pipeline import build_dataset
from vol_model.model import train_purged_cv, compute_shap_importance
from vol_model.evaluate import compute_baselines, evaluate_predictions, print_evaluation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vol_model")


def main():
    parser = argparse.ArgumentParser(description="Train vol prediction model")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--tf", type=int, default=900, help="Timeframe in seconds")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--no-shap", action="store_true")
    args = parser.parse_args()

    config = VolModelConfig(
        asset=args.asset,
        freq_s=args.tf,
        n_splits=args.n_splits,
    )
    out = Path(config.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("=" * 60)
    print(f"  VOL MODEL TRAINING — {args.asset}, {args.tf // 60}m bars")
    print("=" * 60)

    # ── 1. Build dataset ──
    print("\n[1/4] Building dataset...")
    X, y, t1 = build_dataset(config)
    print(f"  Dataset: {len(X):,} rows, {len(X.columns)} features")
    print(f"  Label range: [{y.min():.2f}, {y.max():.2f}], mean={y.mean():.2f}, std={y.std():.2f}")
    print(f"  Date range: {X.index[0]} to {X.index[-1]}")

    # ── 2. Train with purged K-fold ──
    print(f"\n[2/4] Training LightGBM ({config.n_splits}-fold purged CV)...")
    results = train_purged_cv(X, y, t1, config)

    # ── 3. Evaluate ──
    print("\n[3/4] Evaluating...")
    oof = results["oof_preds"]
    valid_mask = ~np.isnan(oof)
    y_valid = y.values[valid_mask]
    oof_valid = oof[valid_mask]

    baselines = compute_baselines(y, tsrv_path=config.get_tsrv_path(), freq_s=config.freq_s)
    baselines_valid = {k: v[valid_mask] for k, v in baselines.items()}

    metrics = evaluate_predictions(y_valid, oof_valid, baselines_valid)
    print_evaluation(metrics, results["fold_metrics"])

    # ── 4. SHAP ──
    if not args.no_shap:
        print("\n[4/4] Computing SHAP importance...")
        importance = compute_shap_importance(results["models"], X)
        if importance is not None:
            print(f"\n  Top 20 features by SHAP importance:")
            for _, row in importance.head(20).iterrows():
                print(f"    {row['feature']:>40s}  {row['mean_abs_shap']:.4f}")
            importance.to_csv(out / f"{args.asset}_shap_importance.csv", index=False)
    else:
        importance = None
        print("\n[4/4] SHAP skipped (--no-shap)")

    # ── Save results ──
    # OOF predictions
    oof_df = pd.DataFrame({
        "y_true": y.values,
        "y_pred": oof,
    }, index=y.index)
    oof_df.to_parquet(out / f"{args.asset}_oof_predictions.parquet")

    # Models (LightGBM native format)
    for i, model in enumerate(results["models"]):
        model.save_model(str(out / f"{args.asset}_model_fold{i}.txt"))

    # Metrics
    all_metrics = {
        "config": config.model_dump(),
        "metrics": metrics,
        "fold_metrics": results["fold_metrics"],
        "n_features": len(X.columns),
        "n_samples": len(X),
        "feature_names": results["feature_names"],
    }
    with open(out / f"{args.asset}_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Output: {out}")
    print(f"  Runtime: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
