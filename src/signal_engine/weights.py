"""Sample weights via AFML Chapter 4 average-uniqueness approximation.

Extracted from experiments/multi_approach_comparison.py (lines 217-237).
O(n) edge-corrected algorithm for regularly-spaced bars.
"""

from __future__ import annotations

import numpy as np


def compute_sample_weights(timestamps: np.ndarray, fwd_bars: int) -> np.ndarray:
    """Compute AFML Ch 4 average-uniqueness sample weights.

    For regularly-spaced bars the interior concurrency is constant at
    ``fwd_bars + 1``.  Edge bars have lower concurrency and therefore
    higher weight.  The resulting weights are normalised so that their
    mean equals 1.

    This is the *O(n) edge-correction* algorithm, NOT the quadratic
    version in ``afml.sampling``.

    Args:
        timestamps: Array of bar timestamps (used only for length).
        fwd_bars: Number of forward bars in the label horizon.

    Returns:
        Weight array of the same length as *timestamps*, with
        ``mean ≈ 1.0``.
    """
    n = len(timestamps)
    if n == 0:
        return np.array([], dtype=np.float64)

    # Interior concurrency = min(fwd_bars + 1, n)
    c = min(fwd_bars + 1, n)
    weights = np.full(n, 1.0 / c, dtype=np.float64)

    # Edge correction: first/last fwd_bars samples have lower concurrency
    for i in range(min(fwd_bars, n)):
        weights[i] = 1.0 / (i + 1)
        if n - 1 - i != i:  # avoid double-counting centre element
            weights[n - 1 - i] = 1.0 / (i + 1)

    mean_w = weights.mean()
    if mean_w > 0:
        weights = weights / mean_w
    return weights
