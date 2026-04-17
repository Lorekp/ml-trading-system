"""Tests for signal_engine.weights."""

from __future__ import annotations

import numpy as np
import pytest

from signal_engine.weights import compute_sample_weights


def test_interior_weights() -> None:
    """Interior weights should be approximately 1 / (fwd_bars + 1)."""
    n = 100
    fwd_bars = 4
    ts = np.arange(n, dtype=np.int64)
    w = compute_sample_weights(ts, fwd_bars)

    # Interior: indices fwd_bars .. n-1-fwd_bars
    interior = w[fwd_bars : n - fwd_bars]
    expected = 1.0 / (fwd_bars + 1)
    # After normalisation the interior values should be close to each other
    # (all equal before normalisation).
    assert np.std(interior) < 0.01, "Interior weights should be nearly constant"


def test_edge_correction() -> None:
    """Edge bars have higher weight than interior bars."""
    n = 50
    fwd_bars = 4
    ts = np.arange(n, dtype=np.int64)
    w = compute_sample_weights(ts, fwd_bars)

    # First bar (concurrency = 1) should have the highest weight
    assert w[0] > w[fwd_bars + 1]
    # Last bar should also be high
    assert w[-1] > w[fwd_bars + 1]


def test_normalisation() -> None:
    """Mean of weights should be approximately 1."""
    for n in [10, 100, 1000]:
        ts = np.arange(n, dtype=np.int64)
        w = compute_sample_weights(ts, fwd_bars=4)
        assert pytest.approx(w.mean(), abs=1e-10) == 1.0


def test_empty_array() -> None:
    """Empty input returns empty output."""
    w = compute_sample_weights(np.array([], dtype=np.int64), fwd_bars=4)
    assert len(w) == 0
    assert w.dtype == np.float64


def test_single_element() -> None:
    """Single element gets weight 1."""
    w = compute_sample_weights(np.array([0], dtype=np.int64), fwd_bars=4)
    assert len(w) == 1
    assert pytest.approx(w[0]) == 1.0


def test_fwd_bars_larger_than_n() -> None:
    """When fwd_bars > n, no crash and weights are still normalised."""
    n = 3
    ts = np.arange(n, dtype=np.int64)
    w = compute_sample_weights(ts, fwd_bars=10)
    assert len(w) == n
    assert pytest.approx(w.mean(), abs=1e-10) == 1.0
