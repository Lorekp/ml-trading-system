"""LightGBM volatility prediction model with QLIKE custom loss.

QLIKE (Quasi-Likelihood) is the proper loss function for volatility forecasting.
It penalizes under-prediction more than over-prediction (asymmetric), which
matches the economic reality that underestimating vol is more dangerous.

In log-vol space (y = ln(vol_ann)):
    QLIKE = exp(2(y_true - y_pred)) - 2(y_true - y_pred) - 1
    gradient = 2(1 - exp(2(y_true - y_pred)))
    hessian = 4 * exp(2(y_true - y_pred))
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.afml.cross_val import PurgedKFold
from vol_model.config import VolModelConfig

logger = logging.getLogger(__name__)


# ── QLIKE loss ──────────────────────────────────────────────────

def qlike_objective(y_pred: np.ndarray, dtrain: lgb.Dataset):
    """Custom QLIKE objective for LightGBM (gradient + hessian)."""
    y_true = dtrain.get_label()
    diff = y_true - y_pred
    exp_term = np.exp(np.clip(2.0 * diff, -20.0, 20.0))
    grad = 2.0 * (1.0 - exp_term)
    hess = 4.0 * exp_term
    return grad, hess


def qlike_metric(y_pred: np.ndarray, dtrain: lgb.Dataset):
    """Custom QLIKE evaluation metric for LightGBM."""
    y_true = dtrain.get_label()
    diff = y_true - y_pred
    qlike = np.exp(np.clip(2.0 * diff, -20.0, 20.0)) - 2.0 * diff - 1.0
    return "qlike", float(np.mean(qlike)), False  # lower is better


def qlike_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute QLIKE loss (standalone, for evaluation)."""
    diff = y_true - y_pred
    qlike = np.exp(np.clip(2.0 * diff, -20.0, 20.0)) - 2.0 * diff - 1.0
    return float(np.mean(qlike))


# ── Model building ───���──────────────────────────────────────────

def build_lgbm_params(config: VolModelConfig) -> dict:
    """Build LightGBM parameter dict for lgb.train() API."""
    return {
        "objective": qlike_objective,  # custom QLIKE loss
        "max_depth": config.max_depth,
        "num_leaves": config.num_leaves,
        "learning_rate": config.learning_rate,
        "subsample": config.subsample,
        "colsample_bytree": config.colsample,
        "min_child_samples": config.min_child_samples,
        "reg_alpha": config.reg_alpha,
        "reg_lambda": config.reg_lambda,
        "seed": config.seed,
        "num_threads": -1,
        "verbosity": -1,
    }


# ── Training ─────────────────────────────────────��──────────────

def train_purged_cv(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    config: VolModelConfig,
) -> dict:
    """Train LightGBM with purged K-fold cross-validation.

    Returns dict with:
        models: list of fitted LGBMRegressor (one per fold)
        oof_preds: out-of-fold predictions (full array)
        fold_metrics: per-fold QLIKE, RMSE, MAE
        feature_names: column names
    """
    cv = PurgedKFold(
        n_splits=config.n_splits,
        t1=t1,
        pct_embargo=config.pct_embargo,
        label_span=1,  # label spans 1 bar (15m)
    )

    params = build_lgbm_params(config)
    oof_preds = np.full(len(y), np.nan)
    models = []
    fold_metrics = []

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(X)):
        logger.info("  Fold %d: train=%d, test=%d", fold_i + 1, len(train_idx), len(test_idx))

        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        # Winsorize features (fit on train, apply to both)
        lo_vals = X_train.quantile(config.winsorize_lo)
        hi_vals = X_train.quantile(config.winsorize_hi)
        X_train = X_train.clip(lo_vals, hi_vals, axis=1)
        X_test = X_test.clip(lo_vals, hi_vals, axis=1)

        # Build LightGBM datasets
        dtrain = lgb.Dataset(X_train, label=y_train.values, free_raw_data=False)
        dval = lgb.Dataset(X_test, label=y_test.values, reference=dtrain, free_raw_data=False)

        # Train with custom QLIKE objective (passed via params["objective"])
        model = lgb.train(
            params,
            dtrain,
            num_boost_round=config.n_estimators,
            valid_sets=[dval],
            feval=qlike_metric,
            callbacks=[
                lgb.early_stopping(config.early_stopping_rounds),
                lgb.log_evaluation(200),
            ],
        )

        # Predict
        preds = model.predict(X_test)
        oof_preds[test_idx] = preds

        # Metrics
        y_test_arr = y_test.values
        fold_qlike = qlike_loss(y_test_arr, preds)
        fold_rmse = float(np.sqrt(np.mean((y_test_arr - preds) ** 2)))
        fold_mae = float(np.mean(np.abs(y_test_arr - preds)))

        fold_metrics.append({
            "fold": fold_i + 1,
            "qlike": fold_qlike,
            "rmse": fold_rmse,
            "mae": fold_mae,
            "n_train": len(train_idx),
            "n_test": len(test_idx),
            "best_iteration": model.best_iteration,
        })

        models.append(model)
        logger.info("    QLIKE=%.4f  RMSE=%.4f  MAE=%.4f  (iter=%d)",
                     fold_qlike, fold_rmse, fold_mae, model.best_iteration)

    return {
        "models": models,
        "oof_preds": oof_preds,
        "fold_metrics": fold_metrics,
        "feature_names": list(X.columns),
    }


# ── SHAP ────────────��───────────────────────────────────────────

def compute_shap_importance(
    models: list,
    X: pd.DataFrame,
    max_samples: int = 5000,
) -> pd.DataFrame | None:
    """Compute SHAP feature importance averaged across fold models.

    Returns DataFrame with columns: feature, mean_abs_shap, sorted descending.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed — skipping importance analysis")
        return None

    # Sample for speed
    if len(X) > max_samples:
        X_sample = X.sample(max_samples, random_state=42)
    else:
        X_sample = X

    all_shap = np.zeros((len(X_sample), len(X_sample.columns)))

    for model in models:
        explainer = shap.TreeExplainer(model)
        sv = explainer.shap_values(X_sample)
        all_shap += np.abs(sv)

    all_shap /= len(models)

    importance = pd.DataFrame({
        "feature": X_sample.columns,
        "mean_abs_shap": all_shap.mean(axis=0),
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

    return importance
