"""
Tests for purged cross-validation.

Critical: standard CV leaks in finance. These tests verify that purging
and embargo actually prevent leakage.
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest
from afml.cross_val import PurgedKFold, purged_train_test_split


@pytest.fixture
def time_series_data():
    """Simulated labeled data with overlapping labels."""
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="4h")

    X = pd.DataFrame({"feat1": np.random.randn(n), "feat2": np.random.randn(n)}, index=idx)
    y = pd.Series(np.random.choice([-1, 0, 1], n), index=idx)

    # Labels span 20 bars each (overlapping)
    t1 = pd.Series(index=idx, dtype="datetime64[ns]")
    for i in range(n):
        end_idx = min(i + 20, n - 1)
        t1.iloc[i] = idx[end_idx]

    return X, y, t1


class TestPurgedKFold:
    def test_produces_correct_number_of_splits(self, time_series_data):
        X, y, t1 = time_series_data
        cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
        splits = list(cv.split(X))
        assert len(splits) == 5

    def test_no_overlap_between_train_and_test(self, time_series_data):
        X, y, t1 = time_series_data
        cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
        for train_idx, test_idx in cv.split(X):
            overlap = set(train_idx) & set(test_idx)
            assert len(overlap) == 0, f"Train/test overlap: {overlap}"

    def test_train_smaller_than_without_purge(self, time_series_data):
        """Purging should REMOVE training samples, making train set smaller."""
        X, y, t1 = time_series_data
        cv_purged = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
        splits = list(cv_purged.split(X))
        # With 200 samples, 5 folds → test ~40, train ~160 without purge
        # Purging should make train < 160
        for train_idx, test_idx in splits:
            assert len(train_idx) < 200 - len(test_idx), (
                f"Purging should reduce train size. Got train={len(train_idx)}, test={len(test_idx)}"
            )

    def test_embargo_removes_post_test_bars(self, time_series_data):
        X, y, t1 = time_series_data
        # Large embargo: 10% of data
        cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.10)
        splits = list(cv.split(X))
        for train_idx, test_idx in splits:
            if len(test_idx) == 0:
                continue
            test_end = test_idx[-1]
            embargo_end = test_end + int(len(X) * 0.10)
            # No training sample should be in the embargo zone
            embargo_zone = set(range(test_end + 1, min(embargo_end + 1, len(X))))
            embargo_in_train = embargo_zone & set(train_idx)
            assert len(embargo_in_train) == 0, (
                f"Found {len(embargo_in_train)} training samples in embargo zone"
            )

    def test_all_test_indices_covered(self, time_series_data):
        """Union of all test sets should cover all data points."""
        X, y, t1 = time_series_data
        cv = PurgedKFold(n_splits=5, t1=t1, pct_embargo=0.01)
        all_test = set()
        for _, test_idx in cv.split(X):
            all_test.update(test_idx)
        assert len(all_test) == len(X), "All indices should appear in test at least once"


class TestPurgedTrainTestSplit:
    def test_basic_split(self, time_series_data):
        X, y, t1 = time_series_data
        X_train, X_test, y_train, y_test = purged_train_test_split(
            X, y, t1, test_size=0.2, pct_embargo=0.01,
        )
        assert len(X_test) > 0
        assert len(X_train) > 0
        assert len(X_train) + len(X_test) <= len(X)  # <= because purge removes some

    def test_no_future_leakage(self, time_series_data):
        """No training sample should have a label that extends into the test period."""
        X, y, t1 = time_series_data
        X_train, X_test, y_train, y_test = purged_train_test_split(
            X, y, t1, test_size=0.2, pct_embargo=0.01,
        )
        test_start = X_test.index[0]
        # Check that no training label extends into test period
        for i, train_time in enumerate(X_train.index):
            pos = t1.index.get_loc(train_time)
            label_end = t1.iloc[pos]
            if pd.notna(label_end):
                assert label_end < test_start, (
                    f"Training label at {train_time} extends to {label_end}, "
                    f"which is in the test period starting {test_start}"
                )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
