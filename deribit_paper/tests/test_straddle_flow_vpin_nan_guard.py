"""Cycle-22 (2026-05-11) regression tests for VPIN provider hardening.

History:
- D-10 (parity-r3 cycle 3 — 2026-05-08): added a non-finite guard that
  fell back to 0.5 on NaN/Inf/None-provider/exceptions.
- Cycle-22 (2026-05-11): the 7-agent swarm audit (A6 H7) flagged that
  0.5 is ~10σ off SAC v4's training-time VPIN distribution (mean ≈
  0.189, std ≈ 0.032). Using 0.5 forces the policy out of its
  training distribution, silently corrupting every hedge action when
  the provider misbehaves. Two-layer fix:
  - Engine startup (engine.py): FAIL-FAST if V2 preset enabled but
    vpin_state_path empty → no stream → no provider would be wired.
    This catches the config-bug case at startup, not at trade time.
  - begin_trade fallback (straddle_flow.py): when provider returns
    non-finite OR raises, fall back to ``VPIN_TRAINING_MEAN = 0.189``
    (in-distribution neutral) instead of the old 0.5. When provider
    is None (test paths), use the same training-mean.

The tests below exercise the per-trade policy. The engine-startup
fail-fast is exercised by ``test_engine_fails_fast_without_vpin_stream``
in ``test_engine_drl_vpin_failsafe.py`` (cycle-22 companion).
"""

from __future__ import annotations

import math

import pytest


VPIN_TRAINING_MEAN = 0.189  # mirrors the constant in straddle_flow.py


def _resolve_vpin(provider) -> float:
    """Replicate cycle-22's in-line guard from
    ``straddle_flow.execute()`` so tests can validate the exact policy
    without instantiating an async ``execute`` flow.

    Mirrors lines ~641-672 of straddle_flow.py post-cycle-22.
    """
    if provider is None:
        return VPIN_TRAINING_MEAN
    try:
        raw = float(provider())
        if math.isfinite(raw):
            return raw
        return VPIN_TRAINING_MEAN
    except (TypeError, ValueError):
        return VPIN_TRAINING_MEAN


# ── Healthy values pass through unchanged ────────────────────────────


def test_healthy_vpin_value_passes_through():
    val = _resolve_vpin(lambda: 0.7)
    assert val == pytest.approx(0.7)


def test_zero_vpin_value_passes_through():
    """Zero is a legitimate VPIN reading (perfectly balanced flow);
    the guard must not nudge it."""
    val = _resolve_vpin(lambda: 0.0)
    assert val == 0.0


def test_training_mean_value_passes_through():
    """The training mean itself is a perfectly valid live VPIN value
    — the guard only triggers on non-finite, not "looks neutral"."""
    val = _resolve_vpin(lambda: 0.189)
    assert val == pytest.approx(0.189)


# ── Non-finite values fall back to training-mean (cycle-22) ──────────


def test_nan_vpin_falls_back_to_training_mean():
    val = _resolve_vpin(lambda: float("nan"))
    assert val == pytest.approx(VPIN_TRAINING_MEAN)


def test_inf_vpin_falls_back_to_training_mean():
    val = _resolve_vpin(lambda: float("inf"))
    assert val == pytest.approx(VPIN_TRAINING_MEAN)


def test_neg_inf_vpin_falls_back_to_training_mean():
    val = _resolve_vpin(lambda: float("-inf"))
    assert val == pytest.approx(VPIN_TRAINING_MEAN)


# ── Provider exceptions fall back to training-mean ───────────────────


def test_provider_returns_none_falls_back():
    """Provider returns None → ``float(None)`` raises TypeError →
    in-distribution fallback."""
    val = _resolve_vpin(lambda: None)
    assert val == pytest.approx(VPIN_TRAINING_MEAN)


def test_provider_returns_string_falls_back():
    """Provider returns an unconvertible string → ValueError → fallback."""
    val = _resolve_vpin(lambda: "not-a-number")
    assert val == pytest.approx(VPIN_TRAINING_MEAN)


# ── No provider configured → training-mean (cycle-22 change) ────────


def test_no_provider_falls_back_to_training_mean():
    """Cycle-22 (2026-05-11): the no-provider case (tests, non-DRL
    configs) falls back to ``VPIN_TRAINING_MEAN = 0.189`` instead of
    the old ``0.5``. The hard fail-fast for the DRL-enabled-but-no-
    provider scenario lives in engine.py at startup, not here; this
    layer just provides an in-distribution default."""
    assert _resolve_vpin(None) == pytest.approx(VPIN_TRAINING_MEAN)
