"""
Tests for feature importance (MDI, MDA, SFI).
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import KFold
from afml.importance import mean_decrease_impurity, mean_decrease_accuracy, single_feature_importance


@pytest.fixture
def obvious_feature_data():
    """Dataset where feature 'signal' is predictive, 'noise' is not."""
    np.random.seed(42)
    n = 500
    signal = np.random.randn(n)
    noise = np.random.randn(n)
    # Label is determined by signal, not noise
    y = (signal > 0).astype(int)
    X = pd.DataFrame({"signal": signal, "noise": noise})
    y = pd.Series(y, name="label")
    return X, y


class TestMDI:
    def test_signal_more_important_than_noise(self, obvious_feature_data):
        X, y = obvious_feature_data
        model = RandomForestClassifier(n_estimators=50, random_state=42, max_depth=5)
        model.fit(X, y)
        mdi = mean_decrease_impurity(model, X.columns.tolist())
        assert mdi.loc["signal", "mean"] > mdi.loc["noise", "mean"], (
            f"Signal should be more important: {mdi}"
        )

    def test_sums_to_one(self, obvious_feature_data):
        X, y = obvious_feature_data
        model = RandomForestClassifier(n_estimators=50, random_state=42)
        model.fit(X, y)
        mdi = mean_decrease_impurity(model, X.columns.tolist())
        assert mdi["mean"].sum() == pytest.approx(1.0, abs=0.01)


class TestMDA:
    def test_signal_more_important_than_noise(self, obvious_feature_data):
        X, y = obvious_feature_data
        model = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=5)
        cv = KFold(n_splits=3, shuffle=False)  # Simple CV for test speed
        mda = mean_decrease_accuracy(model, X, y, cv, scoring='accuracy')
        assert mda.loc["signal", "mean"] > mda.loc["noise", "mean"], (
            f"Signal should be more important in MDA: {mda}"
        )

    def test_returns_all_features(self, obvious_feature_data):
        X, y = obvious_feature_data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        cv = KFold(n_splits=3, shuffle=False)
        mda = mean_decrease_accuracy(model, X, y, cv, scoring='accuracy')
        assert set(mda.index) == set(X.columns)


class TestSFI:
    def test_signal_better_standalone(self, obvious_feature_data):
        X, y = obvious_feature_data
        model = RandomForestClassifier(n_estimators=20, random_state=42, max_depth=5)
        cv = KFold(n_splits=3, shuffle=False)
        sfi = single_feature_importance(model, X, y, cv, scoring='accuracy')
        assert sfi.loc["signal", "mean"] > sfi.loc["noise", "mean"], (
            f"Signal should have higher standalone importance: {sfi}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
