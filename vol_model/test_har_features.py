#!/usr/bin/env python
"""Test: do HAR-equivalent features + HAR prediction improve LightGBM?

Missing from current feature set:
  1. Raw lagged RV levels (rv_zscore strips level info)
  2. Rolling mean RV at HAR horizons (1h, 1d, 1w)
  3. HAR prediction as meta-feature (stacking)

Usage:
    PYTHONPATH=. python vol_model/test_har_features.py
"""

import logging
import time

import numpy as np
import pandas as pd
import lightgbm as lgb

from vol_model.config import VolModelConfig
from vol_model.pipeline import build_dataset
from vol_model.model import qlike_objective, qlike_metric, qlike_loss
from vol_model.evaluate import fit_har
from src.afml.cross_val import PurgedKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")


def make_har_features(y: pd.Series) -> pd.DataFrame:
    """Build HAR-equivalent features from the label series.

    These are the raw inputs to HAR (Corsi 2009) that our feature set lacks:
    - rv_lag1: previous-window ln(vol_ann)
    - rv_mean_4: rolling mean over 4 bars (1h)
    - rv_mean_96: rolling mean over 96 bars (1d)
    - rv_mean_672: rolling mean over 672 bars (1w)
    """
    feats = pd.DataFrame(index=y.index)
    feats["rv_lag1"] = y.shift(1)
    feats["rv_mean_4"] = y.rolling(4, min_periods=1).mean()
    feats["rv_mean_96"] = y.rolling(96, min_periods=1).mean()
    feats["rv_mean_672"] = y.rolling(672, min_periods=1).mean()
    return feats


def make_har_prediction(y: pd.Series) -> pd.Series:
    """Compute HAR prediction as a meta-feature (expanding window, no leakage)."""
    har = fit_har(y)
    return pd.Series(har["predictions"], index=y.index, name="har_pred")


def train_cv(X, y, t1, config, label=""):
    """Train with purged K-fold, return mean QLIKE."""
    cv = PurgedKFold(n_splits=config.n_splits, t1=t1, pct_embargo=config.pct_embargo, label_span=1)

    # Best Optuna params
    params = {
        "objective": qlike_objective,
        "num_leaves": 121,
        "max_depth": 6,
        "learning_rate": 0.002544,
        "min_data_in_leaf": 119,
        "lambda_l1": 0.100568,
        "lambda_l2": 0.469728,
        "colsample_bytree": 0.443224,
        "subsample": 0.8,
        "feature_pre_filter": False,
        "num_threads": -1,
        "verbosity": -1,
        "seed": 42,
    }

    fold_qlikes = []
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X)):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        lo = X_train.quantile(config.winsorize_lo)
        hi = X_train.quantile(config.winsorize_hi)
        X_train = X_train.clip(lo, hi, axis=1)
        X_test = X_test.clip(lo, hi, axis=1)

        dtrain = lgb.Dataset(X_train, label=y_train.values, free_raw_data=False)
        dval = lgb.Dataset(X_test, label=y_test.values, reference=dtrain, free_raw_data=False)

        model = lgb.train(
            params, dtrain, num_boost_round=2000,
            valid_sets=[dval], feval=qlike_metric,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        preds = model.predict(X_test)
        fq = qlike_loss(y_test.values, preds)
        fold_qlikes.append(fq)

    mean_q = np.mean(fold_qlikes)
    print(f"  {label:40s}  QLIKE={mean_q:.4f}  folds={[f'{q:.4f}' for q in fold_qlikes]}")
    return mean_q


def main():
    config = VolModelConfig()
    print("=" * 70)
    print("  TEST: HAR-EQUIVALENT FEATURES FOR LIGHTGBM")
    print("=" * 70)

    # Load base dataset
    print("\nLoading dataset...")
    X_base, y, t1 = build_dataset(config)
    print(f"  Base: {len(X_base)} rows, {len(X_base.columns)} features")

    # Load raw labels (before one-step-ahead alignment) for HAR feature construction
    from vol_model.labels import load_labels
    labels_raw = load_labels(config)
    # Align to X_base index (which is already shifted)
    # X_base.index = features[:-1].index, y.index = labels[1:].index
    # For HAR features, we need rolling means of the label available at prediction time
    # At time t (predicting t+1), we have labels up to t
    # X_base is indexed at feature times, y at label times (shifted by 1)
    # The label at the SAME time as X_base[t] is labels_raw aligned to X_base.index

    # Build HAR features on the label series aligned to X index
    # labels_raw is a DataFrame with "ln_vol" column; X_base.index are the feature timestamps
    y_at_feature_time = labels_raw["ln_vol"].reindex(X_base.index)

    har_feats = make_har_features(y_at_feature_time)
    har_pred = make_har_prediction(y_at_feature_time)

    # Drop NaN from HAR warmup
    valid = ~har_feats.isna().any(axis=1) & ~har_pred.isna()
    X_base_v = X_base[valid]
    y_v = y[valid]
    t1_v = t1[valid]
    har_feats_v = har_feats[valid]
    har_pred_v = har_pred[valid]

    print(f"  After HAR warmup: {len(X_base_v)} rows")

    t0 = time.time()

    # Test 1: Baseline (current features only, on valid subset)
    print("\n--- Experiment ---")
    q_base = train_cv(X_base_v, y_v, t1_v, config, "Baseline (71 features)")

    # Test 2: + HAR input features (rv_lag1, rv_mean_4/96/672)
    X_har_inputs = pd.concat([X_base_v, har_feats_v], axis=1)
    q_har_inputs = train_cv(X_har_inputs, y_v, t1_v, config, "+ HAR inputs (75 features)")

    # Test 3: + HAR prediction only
    X_har_pred = X_base_v.copy()
    X_har_pred["har_pred"] = har_pred_v
    q_har_pred = train_cv(X_har_pred, y_v, t1_v, config, "+ HAR prediction (72 features)")

    # Test 4: + HAR inputs + HAR prediction
    X_all = pd.concat([X_base_v, har_feats_v], axis=1)
    X_all["har_pred"] = har_pred_v
    q_all = train_cv(X_all, y_v, t1_v, config, "+ HAR inputs + prediction (76 features)")

    # Test 5: + just rv_lag1 (simplest possible: the tree sees yesterday's vol)
    X_lag1 = X_base_v.copy()
    X_lag1["rv_lag1"] = har_feats_v["rv_lag1"]
    q_lag1 = train_cv(X_lag1, y_v, t1_v, config, "+ rv_lag1 only (72 features)")

    print(f"\n--- Summary ---")
    print(f"  {'Configuration':40s}  {'QLIKE':>8s}  {'vs Base':>10s}")
    print(f"  {'-'*62}")
    for name, q in [
        ("Baseline (71 features)", q_base),
        ("+ rv_lag1 only", q_lag1),
        ("+ HAR inputs (4 features)", q_har_inputs),
        ("+ HAR prediction (1 feature)", q_har_pred),
        ("+ HAR inputs + prediction (5 features)", q_all),
    ]:
        delta = (q - q_base) / q_base * 100
        print(f"  {name:40s}  {q:.4f}  {delta:+.2f}%")

    print(f"\n  Runtime: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
