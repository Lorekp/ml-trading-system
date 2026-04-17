"""
Purged K-Fold Cross-Validation — AFML Chapter 7.

Standard k-fold CV leaks information in financial data because:
1. Labels overlap in time (triple-barrier spans multiple bars)
2. Serial correlation means nearby bars are not independent
3. Train/test boundaries allow information to flow from test to train

PurgedKFold fixes this by:
1. PURGING: removing training samples whose labels overlap with any test sample
2. EMBARGOING: additionally excluding h bars after each test fold boundary

Reference: López de Prado (2018), Ch 7, Snippets 7.1-7.4.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import BaseCrossValidator


class PurgedKFold(BaseCrossValidator):
    """
    Purged K-Fold cross-validation for financial time series.

    Ensures no information leakage from test to train by removing
    training samples that temporally overlap with any test sample.

    Args:
        n_splits: Number of folds.
        t1: Series mapping observation start → observation end (barrier touch time).
            This is the 't1' column from get_events().
        pct_embargo: Fraction of total observations to exclude after each test fold.
                     Default 0.01 = 1% of data as buffer zone.
        label_span: Number of bars a label spans (for index-based purge when t1 is None).
                    Default 30. Set to fwd_bars for your specific setup.
    """

    def __init__(
        self,
        n_splits: int = 5,
        t1: pd.Series | None = None,
        pct_embargo: float = 0.01,
        label_span: int = 30,
    ):
        super().__init__()
        self.n_splits = n_splits
        self.t1 = t1
        self.pct_embargo = pct_embargo
        self.label_span = label_span

    def get_n_splits(self, X=None, y=None, groups=None) -> int:
        return self.n_splits

    def split(self, X, y=None, groups=None):
        """
        Generate purged train/test splits.

        Yields:
            (train_indices, test_indices) tuples.
        """
        if self.t1 is None:
            raise ValueError("t1 must be provided for PurgedKFold")

        # Ensure t1 is aligned with X
        if isinstance(X, pd.DataFrame):
            indices = X.index
        else:
            indices = np.arange(len(X))
            if isinstance(self.t1.index, pd.DatetimeIndex):
                indices = self.t1.index[:len(X)]

        n_obs = len(indices)
        embargo_size = int(n_obs * self.pct_embargo)

        # Create time-based folds (chronological splits)
        fold_boundaries = np.linspace(0, n_obs, self.n_splits + 1, dtype=int)

        for i in range(self.n_splits):
            test_start = fold_boundaries[i]
            test_end = fold_boundaries[i + 1]

            test_idx = np.arange(test_start, test_end)
            if len(test_idx) == 0:
                continue

            # Get test time range
            if isinstance(indices, pd.DatetimeIndex):
                test_times = indices[test_start:test_end]
                t_test_start = test_times.min()
                t_test_end = test_times.max()
            else:
                t_test_start = test_start
                t_test_end = test_end

            # Build train set: all indices NOT in test, NOT purged, NOT embargoed
            train_idx = []
            for j in range(n_obs):
                if j in test_idx:
                    continue

                # Embargo: skip indices right after test fold
                if test_end <= j < test_end + embargo_size:
                    continue

                # Purge: skip if this training sample's label overlaps with test period
                if isinstance(indices, pd.DatetimeIndex):
                    obs_start = indices[j]
                    obs_end = self.t1.iloc[j] if j < len(self.t1) else obs_start

                    if pd.isna(obs_end):
                        obs_end = obs_start

                    # Overlap conditions (any of these → purge):
                    # 1. Label starts during test period
                    # 2. Label ends during test period
                    # 3. Label spans the entire test period
                    overlaps = (
                        (t_test_start <= obs_start <= t_test_end)
                        or (t_test_start <= obs_end <= t_test_end)
                        or (obs_start <= t_test_start and obs_end >= t_test_end)
                    )
                    if overlaps:
                        continue
                else:
                    # Index-based purge for non-datetime data
                    obs_end_idx = min(j + self.label_span, n_obs - 1)
                    if test_start <= obs_end_idx and j <= test_end:
                        continue

                train_idx.append(j)

            train_idx = np.array(train_idx)
            yield train_idx, test_idx


def purged_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
    t1: pd.Series,
    test_size: float = 0.2,
    pct_embargo: float = 0.01,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Single purged train/test split (for quick validation).

    The last `test_size` fraction of data is the test set.
    Training samples overlapping with test are purged.
    Embargo bars after the split point are excluded.

    Args:
        X: Feature DataFrame.
        y: Label Series.
        t1: Label end times.
        test_size: Fraction of data for test set.
        pct_embargo: Fraction of data for embargo zone.

    Returns:
        (X_train, X_test, y_train, y_test)
    """
    n = len(X)
    split_point = int(n * (1 - test_size))
    embargo_size = int(n * pct_embargo)

    # Test set: last test_size fraction
    X_test = X.iloc[split_point:]
    y_test = y.iloc[split_point:]

    if isinstance(X.index, pd.DatetimeIndex):
        test_start_time = X.index[split_point]

        # Purge + embargo from training
        train_mask = np.ones(split_point, dtype=bool)
        for i in range(split_point):
            # Embargo zone
            if i >= split_point - embargo_size:
                train_mask[i] = False
                continue
            # Purge: check if label overlaps with test period
            if i < len(t1):
                label_end = t1.iloc[i]
                if pd.notna(label_end) and label_end >= test_start_time:
                    train_mask[i] = False

        X_train = X.iloc[:split_point][train_mask]
        y_train = y.iloc[:split_point][train_mask]
    else:
        X_train = X.iloc[: split_point - embargo_size]
        y_train = y.iloc[: split_point - embargo_size]

    return X_train, X_test, y_train, y_test
