"""Unit tests for the sigmoid-adjusted square-root impact model."""

from __future__ import annotations

import numpy as np
import pytest

from vol_arb.drl_hedging.impact import (
    calibrate_kappa,
    pure_sqrt_impact,
    sigmoid_sqrt_impact,
)


from vol_arb.drl_hedging.impact import (
    _DEFAULT_KAPPA, _DEFAULT_SIGMOID_ALPHA, _DEFAULT_SIGMOID_K0,
)

KAPPA = _DEFAULT_KAPPA           # ≈ 1.768e-3, the calibrated default
ALPHA = _DEFAULT_SIGMOID_ALPHA   # 500, α·k0 = 2.5, just below monotonicity bound
K0 = _DEFAULT_SIGMOID_K0         # 0.005 = 0.5% participation knee
SIGMA = 0.80                     # 80% annualised vol — typical crypto


def test_low_participation_correction_saturates():
    """At participation p << k0 the sigmoid correction should approach 2x.

    At p→0 with default α=400, k0=0.005:  α*(p-k0) = -2
    sigmoid(-2) ≈ 0.1192 → correction = 1 + (1 - 0.1192) ≈ 1.881

    Full 2x saturation would require α*k0 >> 1 but the monotonicity
    constraint (empirically α ≲ 430) caps how close we can get.
    """
    Q = 0.0001
    V = 1000.0  # p = 1e-7 << 0.005
    srl = pure_sqrt_impact(Q, V, SIGMA, kappa=KAPPA).item()
    sig = sigmoid_sqrt_impact(
        Q, V, SIGMA, kappa=KAPPA, sigmoid_alpha=ALPHA, sigmoid_k0=K0,
    ).item()
    ratio = sig / srl
    assert 1.82 < ratio < 1.92, f"expected ~1.88, got {ratio:.3f}"


def test_high_participation_matches_pure_srl():
    """At p >> k0 the sigmoid correction should be ~1 — sigmoid-SRL ≈ pure SRL.

    With alpha=200, k0=0.005, at p=0.20 the sigmoid argument is
    200*(0.20-0.005)=39 → sigmoid ≈ 1 → correction ≈ 1 (within 1e-17).
    """
    Q = 20.0
    V = 100.0  # p = 0.20, well above 0.005 knee
    srl = pure_sqrt_impact(Q, V, SIGMA, kappa=KAPPA).item()
    sig = sigmoid_sqrt_impact(
        Q, V, SIGMA, kappa=KAPPA, sigmoid_alpha=ALPHA, sigmoid_k0=K0,
    ).item()
    assert abs(sig - srl) / srl < 0.01, f"expected within 1% at high p, got {sig/srl:.4f}"


def test_monotonic_in_trade_size():
    """Impact must be non-decreasing in Q for fixed V, sigma."""
    V = 100.0
    Qs = np.linspace(0.0, 50.0, 200)
    impacts = sigmoid_sqrt_impact(
        Qs, V, SIGMA, kappa=KAPPA, sigmoid_alpha=ALPHA, sigmoid_k0=K0,
    )
    diffs = np.diff(impacts)
    assert (diffs >= -1e-12).all(), "impact should be monotonic in |Q|"


def test_zero_trade_size_zero_impact():
    """Q=0 → impact=0 regardless of the sigmoid correction."""
    impact = sigmoid_sqrt_impact(
        0.0, 100.0, SIGMA, kappa=KAPPA, sigmoid_alpha=ALPHA, sigmoid_k0=K0,
    ).item()
    assert impact == 0.0


def test_zero_volume_does_not_blow_up():
    """V=0 must hit the impact_floor, not NaN or inf."""
    impact = sigmoid_sqrt_impact(
        1.0, 0.0, SIGMA, kappa=KAPPA, sigmoid_alpha=ALPHA, sigmoid_k0=K0,
        impact_floor=1e-8,
    ).item()
    assert np.isfinite(impact), "impact should be finite at V=0"
    assert impact > 0.0


def test_vectorisation_matches_loop():
    """Batch call must equal per-element scalar calls."""
    rng = np.random.default_rng(42)
    n = 100
    Q = rng.uniform(0.0, 10.0, n)
    V = rng.uniform(50.0, 200.0, n)
    sig_batch = sigmoid_sqrt_impact(
        Q, V, SIGMA, kappa=KAPPA, sigmoid_alpha=ALPHA, sigmoid_k0=K0,
    )
    sig_loop = np.array([
        sigmoid_sqrt_impact(q, v, SIGMA, kappa=KAPPA, sigmoid_alpha=ALPHA, sigmoid_k0=K0).item()
        for q, v in zip(Q, V)
    ])
    assert np.allclose(sig_batch, sig_loop, atol=1e-12)


def test_negative_Q_treated_as_absolute():
    """Impact is a function of |Q|, not signed Q."""
    V = 100.0
    pos = sigmoid_sqrt_impact(5.0, V, SIGMA, kappa=KAPPA).item()
    neg = sigmoid_sqrt_impact(-5.0, V, SIGMA, kappa=KAPPA).item()
    assert abs(pos - neg) < 1e-12


def test_calibrate_kappa_reproduces_baseline():
    """calibrate_kappa() must hit exactly 2 bps at the stated operating point.

    The calibrator must account for the sigmoid correction factor at the
    operating point (otherwise it over-produces by ~0.06% with the default
    alpha=500, p=0.02).
    """
    k = calibrate_kappa(
        baseline_impact_bps=2.0,
        typical_participation=0.02,
        typical_sigma=0.80,
        sigmoid_alpha=ALPHA,
        sigmoid_k0=K0,
    )
    V = 100.0
    Q = 0.02 * V  # exactly the typical participation
    impact = sigmoid_sqrt_impact(
        Q, V, 0.80, kappa=k, sigmoid_alpha=ALPHA, sigmoid_k0=K0,
    ).item()
    # 2 bps = 2e-4 — should hit exactly after the correction-aware calibration
    assert abs(impact - 2e-4) < 1e-9, f"expected 2bps, got {impact*1e4:.6f}bps"


@pytest.mark.parametrize("participation", [0.001, 0.002, 0.005, 0.01, 0.05, 0.20])
def test_correction_factor_in_bounds(participation):
    """Correction factor must always be in [1, 2] — never less, never more."""
    V = 1000.0
    Q = participation * V
    srl = pure_sqrt_impact(Q, V, SIGMA, kappa=KAPPA).item()
    sig = sigmoid_sqrt_impact(
        Q, V, SIGMA, kappa=KAPPA, sigmoid_alpha=ALPHA, sigmoid_k0=K0,
    ).item()
    if srl > 0:
        ratio = sig / srl
        assert 0.99 <= ratio <= 2.01, (
            f"correction factor {ratio:.3f} out of [1, 2] at p={participation}"
        )
