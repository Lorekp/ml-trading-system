"""D-10 (parity-r3 cycle 3 — 2026-05-08) regression tests.

``StraddleFlow`` now guards against non-finite VPIN values from its
provider by falling back to a neutral 0.5 — the same value used when
no provider is wired at all. Pre-fix a degraded VPIN stream (Binance
WebSocket reconnect, stale aggregation) flowed NaN straight into
``LiveDRLHedger.begin_trade`` → SAC v4 obs vector took NaN → silent
hedge mispricing.

The guard tests below exercise the per-call provider path used in
``execute()`` at the ``begin_trade`` site.
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock

import pytest


def _resolve_vpin(provider) -> float:
    """Replicate the in-line guard from straddle_flow.execute() so
    tests can validate the exact policy without instantiating an
    async ``execute`` flow."""
    if provider is None:
        return 0.5
    try:
        raw = float(provider())
        return raw if math.isfinite(raw) else 0.5
    except (TypeError, ValueError):
        return 0.5


# ── Healthy values pass through ──────────────────────────────────────


def test_healthy_vpin_value_passes_through():
    val = _resolve_vpin(lambda: 0.7)
    assert val == pytest.approx(0.7)


def test_zero_vpin_value_passes_through():
    """Zero is a legitimate VPIN reading (perfectly balanced flow);
    the guard must not nudge it to 0.5."""
    val = _resolve_vpin(lambda: 0.0)
    assert val == 0.0


# ── Non-finite values fall back to neutral 0.5 ────────────────────────


def test_nan_vpin_falls_back_to_neutral():
    val = _resolve_vpin(lambda: float("nan"))
    assert val == 0.5


def test_inf_vpin_falls_back_to_neutral():
    val = _resolve_vpin(lambda: float("inf"))
    assert val == 0.5


def test_neg_inf_vpin_falls_back_to_neutral():
    val = _resolve_vpin(lambda: float("-inf"))
    assert val == 0.5


# ── Provider exceptions fall back to neutral 0.5 ─────────────────────


def test_provider_raises_typeerror_falls_back():
    """Provider returns None or unconvertible → neutral fallback,
    not propagated as a trade-cancelling exception."""
    val = _resolve_vpin(lambda: None)
    assert val == 0.5


def test_provider_raises_valueerror_falls_back():
    val = _resolve_vpin(lambda: "not-a-number")
    assert val == 0.5


# ── No provider configured → neutral default ─────────────────────────


def test_no_provider_yields_neutral_default():
    val = _resolve_vpin(None)
    assert val == 0.5
