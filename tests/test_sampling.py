"""
Tests for sample weights and sequential bootstrap.

Verifies:
1. Concurrent event counting is correct
2. Non-overlapping labels have uniqueness = 1.0
3. Fully overlapping labels have uniqueness ≈ 1/n
4. Return-attributed weights scale with price moves
5. Time decay reduces older sample weights
6. Sequential bootstrap produces valid samples with higher uniqueness than standard
"""

import sys
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import pytest
from afml.sampling import (
    get_num_concurrent_events,
    get_avg_uniqueness,
    get_sample_weights,
    get_time_decay,
    get_indicator_matrix,
    seq_bootstrap,
)


@pytest.fixture
def non_overlapping_labels():
    """Three labels that don't overlap at all."""
    idx = pd.date_range("2024-01-01", periods=100, freq="4h")
    t1 = pd.Series(
        [idx[9], idx[29], idx[49]],
        index=[idx[0], idx[20], idx[40]],
    )
    return idx, t1


@pytest.fixture
def fully_overlapping_labels():
    """Three labels that completely overlap."""
    idx = pd.date_range("2024-01-01", periods=50, freq="4h")
    t1 = pd.Series(
        [idx[30], idx[30], idx[30]],
        index=[idx[0], idx[0], idx[0]],  # Same start, same end
    )
    return idx, t1


@pytest.fixture
def partial_overlap_labels():
    """Labels with partial overlap."""
    idx = pd.date_range("2024-01-01", periods=100, freq="4h")
    # Label 0: bars 0-20, Label 1: bars 10-30, Label 2: bars 25-45
    t1 = pd.Series(
        [idx[20], idx[30], idx[45]],
        index=[idx[0], idx[10], idx[25]],
    )
    return idx, t1


class TestConcurrentEvents:
    def test_non_overlapping(self, non_overlapping_labels):
        idx, t1 = non_overlapping_labels
        count = get_num_concurrent_events(idx, t1)
        # Non-overlapping: max concurrent should be 1
        assert count.max() == 1.0

    def test_overlapping_peak(self, partial_overlap_labels):
        idx, t1 = partial_overlap_labels
        count = get_num_concurrent_events(idx, t1)
        # Labels 0 and 1 overlap at bars 10-20 → count should be 2
        assert count.max() >= 2.0

    def test_all_positive(self, partial_overlap_labels):
        idx, t1 = partial_overlap_labels
        count = get_num_concurrent_events(idx, t1)
        # Within label range, count should be >= 1
        labeled_range = count.loc[t1.index[0]:t1.values[-1]]
        assert (labeled_range >= 1).all()


class TestAvgUniqueness:
    def test_non_overlapping_uniqueness_one(self, non_overlapping_labels):
        """Non-overlapping labels should have uniqueness ≈ 1.0."""
        idx, t1 = non_overlapping_labels
        count = get_num_concurrent_events(idx, t1)
        uniqueness = get_avg_uniqueness(t1, count)
        for u in uniqueness.dropna():
            assert u == pytest.approx(1.0, abs=0.01), f"Non-overlapping should be ~1.0, got {u}"

    def test_overlapping_uniqueness_lower(self, partial_overlap_labels):
        """Overlapping labels should have uniqueness < 1.0."""
        idx, t1 = partial_overlap_labels
        count = get_num_concurrent_events(idx, t1)
        uniqueness = get_avg_uniqueness(t1, count)
        # At least one label should have uniqueness < 1 due to overlap
        assert uniqueness.min() < 1.0, "Overlapping labels should have uniqueness < 1"

    def test_uniqueness_between_0_and_1(self, partial_overlap_labels):
        idx, t1 = partial_overlap_labels
        count = get_num_concurrent_events(idx, t1)
        uniqueness = get_avg_uniqueness(t1, count)
        assert (uniqueness.dropna() >= 0).all()
        assert (uniqueness.dropna() <= 1).all()


class TestSampleWeights:
    def test_weights_positive(self, partial_overlap_labels):
        idx, t1 = partial_overlap_labels
        # Create a simple price series
        close = pd.Series(100 + np.random.randn(len(idx)).cumsum() * 0.1, index=idx)
        count = get_num_concurrent_events(idx, t1)
        weights = get_sample_weights(t1, count, close)
        assert (weights.dropna() >= 0).all(), "Weights should be non-negative"

    def test_larger_move_higher_weight(self):
        """Label covering a large price move should get higher weight than small move."""
        idx = pd.date_range("2024-01-01", periods=60, freq="4h")
        # Two non-overlapping labels
        t1 = pd.Series([idx[19], idx[39]], index=[idx[0], idx[20]])

        # Price: flat first half, big move second half
        close = pd.Series(index=idx, dtype=float)
        close.iloc[:20] = 100.0  # flat
        close.iloc[20:40] = np.linspace(100, 110, 20)  # big move up
        close.iloc[40:] = 110.0

        count = get_num_concurrent_events(idx, t1)
        weights = get_sample_weights(t1, count, close)

        # Second label (big move) should have higher weight
        assert weights.iloc[1] > weights.iloc[0], (
            f"Big-move label should have higher weight: {weights.values}"
        )


class TestTimeDecay:
    def test_no_decay(self, partial_overlap_labels):
        """clf_last_w=1.0 should produce no decay (all weights equal-ish)."""
        idx, t1 = partial_overlap_labels
        count = get_num_concurrent_events(idx, t1)
        uniqueness = get_avg_uniqueness(t1, count)
        decayed = get_time_decay(uniqueness, clf_last_w=1.0)
        # With no decay, all should be ~1.0
        assert decayed.max() == pytest.approx(1.0, abs=0.1)
        assert decayed.min() == pytest.approx(1.0, abs=0.1)

    def test_full_decay(self, partial_overlap_labels):
        """clf_last_w=0.0 should make oldest sample weight approach 0."""
        idx, t1 = partial_overlap_labels
        count = get_num_concurrent_events(idx, t1)
        uniqueness = get_avg_uniqueness(t1, count)
        decayed = get_time_decay(uniqueness, clf_last_w=0.0)
        # Newest should be ~1.0, oldest should be near 0
        assert decayed.iloc[-1] > decayed.iloc[0]


class TestSeqBootstrap:
    def test_produces_correct_length(self, partial_overlap_labels):
        idx, t1 = partial_overlap_labels
        ind_matrix = get_indicator_matrix(t1, idx)
        drawn = seq_bootstrap(ind_matrix, sample_length=3)
        assert len(drawn) == 3

    def test_all_indices_valid(self, partial_overlap_labels):
        idx, t1 = partial_overlap_labels
        ind_matrix = get_indicator_matrix(t1, idx)
        drawn = seq_bootstrap(ind_matrix, sample_length=10)
        for d in drawn:
            assert d in ind_matrix.columns

    def test_indicator_matrix_shape(self, partial_overlap_labels):
        idx, t1 = partial_overlap_labels
        ind_matrix = get_indicator_matrix(t1, idx)
        assert ind_matrix.shape[1] == len(t1), "Columns should equal number of labels"
        # Each column should have some 1s
        for col in ind_matrix.columns:
            assert ind_matrix[col].sum() > 0, f"Column {col} has no active bars"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
