"""
Feature Importance — AFML Chapter 8.

Three methods for determining which features matter:
1. MDI (Mean Decrease Impurity): Fast, in-sample, tree-based only
2. MDA (Mean Decrease Accuracy): Slow, OOS, any classifier
3. SFI (Single Feature Importance): OOS, no substitution effects

Rule: Only keep features important in ALL THREE methods.

"Backtesting is not a research tool. Feature importance is."
— López de Prado

Reference: López de Prado (2018), Ch 8, Snippets 8.1-8.5.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, accuracy_score, mean_squared_error
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import BaseCrossValidator


def mean_decrease_impurity(
    model,
    feature_names: list[str],
) -> pd.DataFrame:
    """
    MDI — Mean Decrease Impurity (Gini importance).

    Fast (in-sample), works with tree-based models only.
    Biased toward high-cardinality and continuous features.
    Use as a quick screen, NOT as final feature selection.

    Args:
        model: Fitted tree-based model with `feature_importances_` attribute.
        feature_names: List of feature names.

    Returns:
        DataFrame with 'mean' and 'std' columns, indexed by feature name.
    """
    if hasattr(model, 'estimators_'):
        # Ensemble: compute per-tree importance
        importances = np.array([
            tree.feature_importances_ for tree in model.estimators_
        ])
        result = pd.DataFrame({
            'mean': importances.mean(axis=0),
            'std': importances.std(axis=0),
        }, index=feature_names)
    else:
        result = pd.DataFrame({
            'mean': model.feature_importances_,
            'std': np.zeros(len(feature_names)),
        }, index=feature_names)

    # Normalize to sum to 1
    result['mean'] /= result['mean'].sum()
    return result.sort_values('mean', ascending=False)


def mean_decrease_accuracy(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: BaseCrossValidator,
    sample_weight: pd.Series | None = None,
    scoring: str = 'neg_log_loss',
    n_repeats: int = 1,
) -> pd.DataFrame:
    """
    MDA — Mean Decrease Accuracy (permutation importance).

    Slow (requires re-evaluation per feature per fold), but OOS and
    works with any classifier OR regressor. The gold standard for feature importance.

    For each feature: permute it, measure score drop. Large drop = important.

    Args:
        model: Unfitted model (will be re-fitted per fold).
        X: Feature DataFrame.
        y: Label Series.
        cv: Cross-validation splitter (use PurgedKFold!).
        sample_weight: Optional sample weights for fitting.
        scoring: 'neg_log_loss', 'accuracy', or 'neg_mse' (for regression).
        n_repeats: Number of permutation repeats per feature.

    Returns:
        DataFrame with 'mean' and 'std' importance, indexed by feature name.
    """
    from sklearn.base import clone

    feature_names = X.columns.tolist()
    importances = {feat: [] for feat in feature_names}

    def _score(model, X_eval, y_eval):
        if scoring == 'neg_log_loss':
            prob = model.predict_proba(X_eval)
            return -log_loss(y_eval, prob, labels=model.classes_)
        elif scoring == 'neg_mse':
            pred = model.predict(X_eval)
            return -mean_squared_error(y_eval, pred)
        else:  # accuracy
            pred = model.predict(X_eval)
            return accuracy_score(y_eval, pred)

    for train_idx, test_idx in cv.split(X):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        fit_params = {}
        if sample_weight is not None:
            fit_params['sample_weight'] = sample_weight.iloc[train_idx].values

        clf = clone(model)
        clf.fit(X_train, y_train, **fit_params)

        baseline = _score(clf, X_test, y_test)

        for feat in feature_names:
            scores_permuted = []
            for _ in range(n_repeats):
                X_test_perm = X_test.copy()
                X_test_perm[feat] = np.random.permutation(X_test_perm[feat].values)
                score_perm = _score(clf, X_test_perm, y_test)
                scores_permuted.append(baseline - score_perm)

            importances[feat].extend(scores_permuted)

    result = pd.DataFrame({
        'mean': {f: np.mean(v) for f, v in importances.items()},
        'std': {f: np.std(v) for f, v in importances.items()},
    })
    return result.sort_values('mean', ascending=False)


def single_feature_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv: BaseCrossValidator,
    sample_weight: pd.Series | None = None,
    scoring: str = 'neg_log_loss',
) -> pd.DataFrame:
    """
    SFI — Single Feature Importance.

    Train a separate model on EACH feature alone. Measures the standalone
    predictive power of each feature without substitution effects.

    Complements MDI/MDA: a feature that is important in MDI/MDA but not SFI
    is likely redundant (another correlated feature carries the same info).

    Args:
        model: Unfitted model.
        X: Feature DataFrame.
        y: Label Series.
        cv: Cross-validation splitter (use PurgedKFold!).
        sample_weight: Optional sample weights.
        scoring: 'neg_log_loss', 'accuracy', or 'neg_mse'.

    Returns:
        DataFrame with 'mean' and 'std' importance, indexed by feature name.
    """
    from sklearn.base import clone

    feature_names = X.columns.tolist()
    results = {}

    def _score(model, X_eval, y_eval):
        if scoring == 'neg_log_loss':
            prob = model.predict_proba(X_eval)
            return -log_loss(y_eval, prob, labels=model.classes_)
        elif scoring == 'neg_mse':
            pred = model.predict(X_eval)
            return -mean_squared_error(y_eval, pred)
        else:
            pred = model.predict(X_eval)
            return accuracy_score(y_eval, pred)

    for feat in feature_names:
        scores = []
        X_single = X[[feat]]

        for train_idx, test_idx in cv.split(X):
            X_train = X_single.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X_single.iloc[test_idx]
            y_test = y.iloc[test_idx]

            fit_params = {}
            if sample_weight is not None:
                fit_params['sample_weight'] = sample_weight.iloc[train_idx].values

            clf = clone(model)
            try:
                clf.fit(X_train, y_train, **fit_params)
                scores.append(_score(clf, X_test, y_test))
            except Exception:
                scores.append(np.nan)

        results[feat] = {'mean': np.nanmean(scores), 'std': np.nanstd(scores)}

    return pd.DataFrame(results).T.sort_values('mean', ascending=False)
