"""Tests for vol_model.har -- causal HAR forecaster.

Critical: verifies that HAR_pred(t) uses ONLY y values with timestamp < t.
This is the leakage guarantee the residual-target thread relies on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_model.har import (
    HAR_DEFAULT_HORIZONS_BARS,
    build_har_residual_target,
    fit_causal_har,
    reconstruct_from_residual,
)


def _synthetic_logvol_series(n: int = 2000, seed: int = 42) -> pd.Series:
    """Generate a synthetic log-vol series with serial correlation."""
    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(n) * 0.1
    # AR(1) with mean -3 (typical log-vol magnitude)
    y = np.zeros(n)
    y[0] = -3.0
    for i in range(1, n):
        y[i] = 0.9 * y[i - 1] + 0.1 * -3.0 + noise[i]
    idx = pd.date_range("2024-01-01", periods=n, freq="15min", tz="UTC")
    return pd.Series(y, index=idx, name="log_vol")


def test_fit_har_shape_and_index() -> None:
    """Output predictions align to input index."""
    y = _synthetic_logvol_series()
    fit = fit_causal_har(y)
    assert len(fit.predictions) == len(y)
    assert (fit.predictions.index == y.index).all()


def test_fit_har_warmup_nan() -> None:
    """First warmup_bars predictions are NaN (expected)."""
    y = _synthetic_logvol_series()
    fit = fit_causal_har(y, horizons=(1, 4, 96, 672))
    # At least the first max(horizons) rows should be NaN (warmup).
    assert fit.predictions.iloc[: fit.warmup_bars].isna().all()


def test_har_causality_no_future_data() -> None:
    """CRITICAL leakage test: prediction at time t must be unchanged when
    we mutate y values at positions >= t.

    If HAR_pred(t) ever depended on y[t] or y[t+1:], the mutation would
    change the prediction. This is THE leakage guarantee the residual
    thread needs.
    """
    y = _synthetic_logvol_series()
    fit_original = fit_causal_har(y)

    # Pick a prediction row well past warmup
    probe_idx = 1500
    original_pred_at_probe = fit_original.predictions.iloc[probe_idx]
    assert not np.isnan(original_pred_at_probe), "Probe row should have a prediction"

    # Mutate y at positions >= probe_idx. If HAR is causal, this must
    # leave predictions at [0, probe_idx] byte-identical.
    y_mutated = y.copy()
    y_mutated.iloc[probe_idx:] = 999.0  # obviously absurd future values

    fit_mutated = fit_causal_har(y_mutated)

    # All predictions at positions <= probe_idx must be identical.
    # (Strict equality allowed because no new arithmetic involving the
    #  future values could have touched past predictions.)
    left_original = fit_original.predictions.iloc[: probe_idx + 1]
    left_mutated = fit_mutated.predictions.iloc[: probe_idx + 1]

    mask = left_original.notna() & left_mutated.notna()
    assert mask.any(), "Need at least some valid predictions to compare"
    np.testing.assert_array_equal(
        left_original[mask].to_numpy(),
        left_mutated[mask].to_numpy(),
    )


def test_har_horizon_shift_excludes_current_bar() -> None:
    """The rolling-means use shift(1) so the current bar itself is never
    part of the feature vector -- verify directly on a small fixture."""
    # Synthetic: y = [1, 2, 3, 4, 5, ...]
    idx = pd.date_range("2024-01-01", periods=200, freq="15min", tz="UTC")
    y = pd.Series(np.arange(200.0), index=idx, name="log_vol")
    fit = fit_causal_har(y, horizons=(1, 4), min_train_bars=10)

    # At row 50, feature horizon=1 should equal y[49], NOT y[50].
    # horizon=4 should equal mean(y[46:50]) = (46+47+48+49)/4 = 47.5.
    from vol_model.har import _rolling_mean_shifted  # internal helper
    h1 = _rolling_mean_shifted(y, 1)
    h4 = _rolling_mean_shifted(y, 4)
    assert h1.iloc[50] == 49.0
    assert h4.iloc[50] == (46 + 47 + 48 + 49) / 4.0


def test_build_and_reconstruct_roundtrip() -> None:
    """residual + har_pred == y (where both are defined)."""
    y = _synthetic_logvol_series()
    residual, har_pred = build_har_residual_target(y)
    mask = residual.notna() & har_pred.notna()
    reconstructed = reconstruct_from_residual(residual[mask], har_pred[mask])
    np.testing.assert_allclose(reconstructed, y[mask].to_numpy(), atol=1e-12)


def test_reconstruct_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape mismatch"):
        reconstruct_from_residual(np.zeros(10), np.zeros(11))


def test_empty_input_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        fit_causal_har(pd.Series([], dtype=float))


def test_default_horizons_sane() -> None:
    assert HAR_DEFAULT_HORIZONS_BARS == (1, 4, 96, 672)
