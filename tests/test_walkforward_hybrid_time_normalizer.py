"""Cycle-25 (2026-05-11) regression tests for TransVol time-feature normalizer persistence.

Closes A2 M1 from the 7-agent audit. Pre-fix
``_extract_compact_embeddings`` in ``vol_model/run_walkforward_hybrid.py``
re-fit the time-feature normalizer's mean/std from the EXTRACTION-window
slice ``time_norm_vals[:max(len-1000, 1)]``. At h=15 (96 bars/day) that
slice extended through ~day 21 of the test month — a leak of ~10 days
of test-month data into the normalizer used to produce predictions for
that same month. Time features are cyclical (sin/cos of clock) so the
economic impact was sub-bp, but architecturally a leak.

Cycle-25 fix:
1. Training-time: persist ``time_normalizer_mean`` / ``time_normalizer_std``
   to the checkpoint (computed on train-only slice, mirrors how the
   feature normalizer is persisted).
2. Extraction-time: prefer the persisted values via the new
   ``_resolve_time_normalizer`` helper; fall back to the legacy re-fit
   with a DeprecationWarning for pre-cycle-25 checkpoints.

These tests probe the helper directly so they don't need a real
TransVol training run.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from vol_model.run_walkforward_hybrid import _resolve_time_normalizer


# ── Persisted values take precedence ─────────────────────────────────


def test_resolve_uses_persisted_time_normalizer_when_present():
    """Cycle-25: when the checkpoint contains
    ``time_normalizer_{mean,std}``, those values must be returned
    verbatim — not the re-fit from ``time_norm_vals``."""
    persisted_mean = np.array([0.5, -0.5], dtype=np.float32)
    persisted_std = np.array([1.0, 2.0], dtype=np.float32)
    ckpt = {
        "time_normalizer_mean": persisted_mean,
        "time_normalizer_std": persisted_std,
    }
    # Wildly different values in the extraction window — the helper
    # must IGNORE these when the checkpoint persisted normalizer is
    # present, otherwise the leak isn't actually closed.
    bad_vals = np.zeros((10_000, 2), dtype=np.float32) + 999.0

    mean, std = _resolve_time_normalizer(ckpt, bad_vals)
    np.testing.assert_array_equal(mean, persisted_mean)
    np.testing.assert_array_equal(std, persisted_std)


def test_resolve_does_not_emit_deprecation_warning_for_new_checkpoint():
    """A modern checkpoint must NOT trigger the deprecation warning
    (otherwise CI noise would flood after retrain)."""
    ckpt = {
        "time_normalizer_mean": np.array([0.0], dtype=np.float32),
        "time_normalizer_std": np.array([1.0], dtype=np.float32),
    }
    vals = np.zeros((100, 1), dtype=np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("error", DeprecationWarning)
        # Would raise if a DeprecationWarning fires.
        _resolve_time_normalizer(ckpt, vals)


# ── Legacy fallback emits deprecation warning ─────────────────────────


def test_resolve_falls_back_to_refit_for_legacy_checkpoint():
    """Pre-cycle-25 checkpoints lack the ``time_normalizer_*`` keys.
    The helper must fall back to the legacy
    ``[:max(len-1000, 1)]`` re-fit and emit a DeprecationWarning so
    the operator knows the leak persists until next retrain."""
    ckpt = {}  # legacy checkpoint
    np.random.seed(0)
    vals = np.random.normal(size=(2000, 3)).astype(np.float32)

    with pytest.warns(DeprecationWarning, match="cycle-25"):
        mean, std = _resolve_time_normalizer(ckpt, vals)

    # Verify the legacy slice semantics: [:max(len-1000, 1)] = [:1000].
    expected_mean = vals[:1000].mean(axis=0)
    expected_std = vals[:1000].std(axis=0).clip(min=1e-8)
    np.testing.assert_allclose(mean, expected_mean, atol=1e-6)
    np.testing.assert_allclose(std, expected_std, atol=1e-6)


def test_resolve_legacy_fallback_handles_short_window():
    """The legacy slice was ``[:max(len-1000, 1)]``, which guards
    against negative slice ends when the extraction window is small.
    Verify with len < 1000."""
    ckpt = {}
    vals = np.arange(500, dtype=np.float32).reshape(-1, 1)
    with pytest.warns(DeprecationWarning):
        mean, std = _resolve_time_normalizer(ckpt, vals)
    # 500 - 1000 = -500, so max(_, 1) = 1 → slice [:1] = single value.
    assert mean.shape == (1,)
    assert std.shape == (1,)


# ── Leak signature: persisted normalizer ignores test-month tail ──────


def test_persisted_normalizer_is_independent_of_extraction_window():
    """Cycle-25's core safety property: when the persisted normalizer
    is present, ``time_mean`` does NOT change when the extraction
    window's tail (test-month data) changes. The pre-fix legacy code
    would have produced different ``mean`` values because it re-fit
    on the extraction window."""
    persisted_mean = np.array([0.42], dtype=np.float32)
    persisted_std = np.array([1.0], dtype=np.float32)
    ckpt = {
        "time_normalizer_mean": persisted_mean,
        "time_normalizer_std": persisted_std,
    }

    # Two different "test month" tails — same training window. Pre-fix
    # would have returned different means because it re-fit on the
    # extraction-window slice, which includes test-month rows.
    base_vals = np.random.normal(size=(2000, 1)).astype(np.float32)
    tail_a = np.random.normal(loc=0.0, scale=1.0, size=(500, 1)).astype(np.float32)
    tail_b = np.random.normal(loc=10.0, scale=5.0, size=(500, 1)).astype(np.float32)
    vals_a = np.concatenate([base_vals, tail_a], axis=0)
    vals_b = np.concatenate([base_vals, tail_b], axis=0)

    mean_a, std_a = _resolve_time_normalizer(ckpt, vals_a)
    mean_b, std_b = _resolve_time_normalizer(ckpt, vals_b)
    # Persisted normalizer is independent of the extraction tail.
    np.testing.assert_array_equal(mean_a, mean_b)
    np.testing.assert_array_equal(std_a, std_b)
    np.testing.assert_array_equal(mean_a, persisted_mean)


def test_legacy_fallback_IS_sensitive_to_extraction_window():
    """The negative control: pre-cycle-25 fallback DOES depend on the
    extraction-window tail (this is the leak signature). The test
    above demonstrates the FIX; this one demonstrates the BUG so the
    test pair forms a regression sentinel."""
    ckpt = {}  # legacy
    base_vals = np.random.normal(size=(2000, 1)).astype(np.float32)
    tail_a = np.random.normal(loc=0.0, scale=1.0, size=(500, 1)).astype(np.float32)
    tail_b = np.random.normal(loc=10.0, scale=5.0, size=(500, 1)).astype(np.float32)
    vals_a = np.concatenate([base_vals, tail_a], axis=0)
    vals_b = np.concatenate([base_vals, tail_b], axis=0)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        mean_a, _ = _resolve_time_normalizer(ckpt, vals_a)
        mean_b, _ = _resolve_time_normalizer(ckpt, vals_b)
    # Legacy re-fit on [:1500] — vals_a and vals_b share the first 2000
    # rows, so they pull from the same slice and means actually agree.
    # But the slice INCLUDES rows 2000:1500 only if there are fewer than
    # 1500 base rows ... wait, [:max(2500-1000, 1)] = [:1500]. base_vals
    # has 2000 rows, tail has 500. So the [:1500] slice is fully inside
    # base_vals → same for both. To make the tail leak visible, we need
    # the slice to extend INTO the tail. That requires len ≤ 1000 +
    # tail_size, so a smaller fixture:
    small_base = np.random.normal(size=(800, 1)).astype(np.float32)
    small_tail_a = np.zeros((400, 1), dtype=np.float32) + 1.0
    small_tail_b = np.zeros((400, 1), dtype=np.float32) + 100.0
    vals_a2 = np.concatenate([small_base, small_tail_a], axis=0)  # 1200 rows
    vals_b2 = np.concatenate([small_base, small_tail_b], axis=0)
    # [:max(1200-1000, 1)] = [:200] — only first 200 of base, no tail.
    # Still doesn't see the tail. Make the base even smaller:
    tiny_base = np.random.normal(size=(500, 1)).astype(np.float32)
    tiny_tail_a = np.zeros((400, 1), dtype=np.float32) + 1.0
    tiny_tail_b = np.zeros((400, 1), dtype=np.float32) + 100.0
    vals_a3 = np.concatenate([tiny_base, tiny_tail_a], axis=0)  # 900 rows
    vals_b3 = np.concatenate([tiny_base, tiny_tail_b], axis=0)
    # [:max(900-1000, 1)] = [:1] — degenerate single-row slice.
    # The legacy slice ONLY pulls test-month data when len > 1000 AND
    # the test-month fraction is large enough. At production scale
    # (years of training, one-month test), [:len-1000] picks up the
    # last ~10 days of train+test for h=15 — that's the actual leak.
    # We document this by showing the slice picks up DIFFERENT rows
    # when test_end shifts:
    long_base = np.random.normal(size=(2000, 1)).astype(np.float32)
    test_short_a = np.zeros((100, 1), dtype=np.float32) + 1.0   # tiny test
    test_long_b  = np.zeros((1100, 1), dtype=np.float32) + 100.0  # bigger test
    leak_a = np.concatenate([long_base, test_short_a], axis=0)  # 2100 rows
    leak_b = np.concatenate([long_base, test_long_b], axis=0)   # 3100 rows
    # [:max(2100-1000,1)] = [:1100] → all from long_base
    # [:max(3100-1000,1)] = [:2100] → all base + 100 test rows leaked in
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        lm_a, _ = _resolve_time_normalizer({}, leak_a)
        lm_b, _ = _resolve_time_normalizer({}, leak_b)
    # The means MUST differ because lm_b's slice includes 100 test-month
    # rows of value 100 (pulled in by the bigger test window).
    assert lm_a[0] != lm_b[0], (
        "Legacy fallback should be sensitive to extraction-window length "
        "(this is the A2 M1 leak signature). If this assertion passes, "
        "the pre-fix code was leaking test-month data into the time "
        "normalizer — the cycle-25 fix prevents this."
    )
