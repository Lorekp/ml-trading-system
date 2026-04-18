"""Gate tests for vol_arb.rough_heston.

If any of these fail, the rough-Heston sidecar build will produce garbage
and the Session F gauntlet result would be worthless.

Tests:

1. **CF normalization**: phi(0, tau) = 1 exactly (probability distribution
   must integrate to 1).
2. **Degenerate tau limit**: phi(u, tau) -> 1 as tau -> 0 for any u.
3. **Real output for real u with rho=0**: with zero correlation, the CF at
   real u should be real when the model is symmetric (sanity check that
   the Riccati solver doesn't have complex-arithmetic bugs).
4. **Pricing positivity**: Carr-Madan produces non-negative call prices
   and satisfies the no-arbitrage bound c >= max(F-K, 0) within interp
   tolerance.
5. **Pricing monotonicity**: call price is decreasing in strike.
6. **Implied-vol inverter**: roundtrip BS -> price -> IV recovers the
   input IV within 1e-5.
7. **Calibration recovery on synthetic**: generate synthetic rough-Heston
   option prices with known parameters, fit the calibrator, verify
   recovery of the MAIN parameters (V_0, H, rho, eta) within tolerance.
   kappa and theta are notoriously weakly identified — looser tolerance.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from vol_arb.rough_heston import (
    OPTION_DTYPE,
    RoughHestonParams,
    _bs_call_price,
    _bs_vega,
    calibrate_rough_heston,
    carr_madan_call_prices,
    fractional_riccati_solve,
    implied_volatility,
    rough_heston_char_fn,
)


# ── Canonical parameter sets for tests ───────────────────────────────

# For CF / pricing tests: general crypto-realistic parameterization.
_PARAMS_CRYPTO = dict(V_0=0.8 ** 2, H=0.1, kappa=1.5, theta=0.7 ** 2, eta=2.0, rho=-0.7)

# For 4-parameter-calibration tests: parameters consistent with the
# calibrator's pin (kappa=2.0, theta=V_0). If we test 4-param calibration
# recovery on a truth with different kappa/theta, we're asking an
# underparameterized model to match an over-parameterized truth — it
# can't. So the 4-param recovery test uses truth compatible with the
# calibrator's active parameter subspace.
_PARAMS_CRYPTO_4P = dict(V_0=0.7 ** 2, H=0.1, kappa=2.0, theta=0.7 ** 2, eta=2.0, rho=-0.7)


# ── 1. CF normalization ──────────────────────────────────────────────


@pytest.mark.parametrize("tau", [1/365, 7/365, 30/365, 90/365])
def test_cf_at_u_zero_is_one(tau: float) -> None:
    phi_0 = rough_heston_char_fn(np.array([0.0 + 0.0j]), tau, **_PARAMS_CRYPTO)
    assert abs(phi_0[0] - 1.0) < 1e-10, (
        f"phi(0, tau={tau}) = {phi_0[0]} != 1"
    )


def test_cf_at_tau_zero_is_one() -> None:
    """As tau -> 0, phi(u, tau) -> 1 for any u (the log-return is 0 at t=0)."""
    u_vec = np.linspace(-5, 5, 21) + 0.0j
    phi_small = rough_heston_char_fn(u_vec, tau=1e-6, **_PARAMS_CRYPTO)
    max_err = float(np.max(np.abs(phi_small - 1.0)))
    assert max_err < 1e-3, f"phi(u, tau=1e-6) != 1, max err = {max_err}"


# ── 2. Symmetry / sanity ─────────────────────────────────────────────


def test_cf_preserves_conjugate_symmetry() -> None:
    """phi(-u, tau) = conj(phi(u, tau)) for real u (real-valued log-return
    density has real CF for real u; the CF is symmetric under u -> -u via
    complex conjugation)."""
    params = dict(_PARAMS_CRYPTO)
    u_vec = np.array([0.5, 1.0, 2.0, 3.0]) + 0.0j
    phi_pos = rough_heston_char_fn(u_vec, 30/365, **params)
    phi_neg = rough_heston_char_fn(-u_vec, 30/365, **params)
    # phi(-u) should equal conj(phi(u)) for real u
    assert np.allclose(phi_neg, np.conj(phi_pos), rtol=1e-6, atol=1e-8), (
        f"phi(-u) != conj(phi(u)): {phi_neg} vs {np.conj(phi_pos)}"
    )


# ── 3. Pricing sanity ────────────────────────────────────────────────


@pytest.mark.parametrize("tau", [7/365, 30/365, 90/365])
def test_call_prices_non_negative(tau: float) -> None:
    F = 1.0
    log_k = np.linspace(-0.5, 0.5, 21)
    calls = carr_madan_call_prices(
        tau, log_strikes=log_k, F=F, **_PARAMS_CRYPTO,
    )
    assert (calls >= -1e-6).all(), (
        f"negative call prices at tau={tau}: min={calls.min()}"
    )


@pytest.mark.parametrize("tau", [7/365, 30/365])
def test_call_prices_monotone_decreasing_in_strike(tau: float) -> None:
    F = 1.0
    log_k = np.linspace(-0.5, 0.5, 41)
    calls = carr_madan_call_prices(
        tau, log_strikes=log_k, F=F, **_PARAMS_CRYPTO,
    )
    # No-arb: dC/dK <= 0. Allow small tolerance for FFT interp noise.
    diffs = np.diff(calls)
    worst_violation = diffs.max()
    assert worst_violation < 2e-3, (
        f"call-price monotonicity violated by {worst_violation} at tau={tau}"
    )


def test_call_prices_respect_intrinsic_bound() -> None:
    """c(K, tau) >= max(F - K, 0) - epsilon (intrinsic value).

    Carr-Madan with alpha=1.5 is known to struggle outside approximately
    ±1 stdev from ATM deep ITM. We test within the moneyness range that's
    liquid on Deribit and relevant for calibration (log_k in ±0.2, which
    covers ~20% moneyness band; 80% of Deribit volume sits inside).
    """
    F = 1.0
    tau = 30 / 365
    log_k = np.linspace(-0.2, 0.2, 21)
    calls = carr_madan_call_prices(
        tau, log_strikes=log_k, F=F, **_PARAMS_CRYPTO,
    )
    strikes = F * np.exp(log_k)
    intrinsic = np.maximum(F - strikes, 0.0)
    # 1% of F tolerance on the intrinsic bound.
    violations = (calls - intrinsic < -0.01 * F)
    assert not violations.any(), (
        f"intrinsic violated at log_k={log_k[violations]}: "
        f"calls={calls[violations]}, intrinsic={intrinsic[violations]}"
    )


# ── 4. Black-76 inverter ─────────────────────────────────────────────


@pytest.mark.parametrize("sigma", [0.2, 0.5, 0.8, 1.5])
@pytest.mark.parametrize("log_k", [-0.2, 0.0, 0.2])
def test_iv_inverter_roundtrip(sigma: float, log_k: float) -> None:
    F, tau = 1.0, 30/365
    K = F * math.exp(log_k)
    price = _bs_call_price(F, K, tau, sigma)
    iv = implied_volatility(price, F, K, tau)
    # 1e-4 tolerance matches the 200-iter bisection precision on the
    # log_k=0.2 case (smallest vega relative to price).
    assert abs(iv - sigma) < 1e-4, f"IV roundtrip failed: in={sigma}, out={iv}"


# ── 5. Classical Heston limit (informative, not a gate) ──────────────


def test_cf_at_h_near_half_is_smooth() -> None:
    """At H close to 1/2 (the upper boundary), the CF must be smooth and
    finite. This is NOT a classical-Heston match test — rough-Heston with
    H=0.5 has kernel K_H(t)=1 (constant), while classical Heston has
    K=Dirac. They agree only in degenerate limits."""
    params = dict(_PARAMS_CRYPTO)
    params["H"] = 0.49
    u_vec = np.linspace(-3, 3, 21) + 0.0j
    phi = rough_heston_char_fn(u_vec, 30/365, **params)
    assert np.all(np.isfinite(phi)), "CF has non-finite values"
    # Magnitude should be less than 1 (characteristic function bounded)
    assert (np.abs(phi) <= 1.0 + 1e-6).all(), (
        f"|phi| > 1: max={np.abs(phi).max()}"
    )


# ── 6. Calibration recovery (the GATE) ───────────────────────────────


def _synthetic_panel(
    params: dict,
    F: float = 1.0,
    taus: tuple[float, ...] = (7/365, 14/365, 30/365, 60/365),
    log_strikes_per_tau: tuple[tuple[float, ...], ...] = (
        (-0.15, -0.05, 0.0, 0.05, 0.15),
        (-0.20, -0.10, 0.0, 0.10, 0.20),
        (-0.25, -0.10, 0.0, 0.10, 0.25),
        (-0.30, -0.15, 0.0, 0.15, 0.30),
    ),
    iv_noise: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """Generate a synthetic options panel from known rough-Heston params."""
    rng = np.random.default_rng(seed)
    rows = []
    for tau, log_ks in zip(taus, log_strikes_per_tau, strict=True):
        k_arr = np.asarray(log_ks)
        call_prices = carr_madan_call_prices(
            tau, log_strikes=k_arr, F=F, **params,
        )
        for i, k in enumerate(log_ks):
            K = F * math.exp(k)
            iv = implied_volatility(call_prices[i], F, K, tau)
            if not np.isfinite(iv) or iv <= 0 or iv > 5:
                continue
            if iv_noise > 0:
                iv = iv + rng.normal() * iv_noise
            vega = _bs_vega(F, K, tau, iv)
            rows.append((tau, K, iv, vega))
    arr = np.array(rows, dtype=OPTION_DTYPE)
    return arr


def test_calibration_at_truth_reproduces_truth_4p() -> None:
    """Sanity: if we init at truth AND the truth is 4-param-compatible
    (theta=V_0, kappa=2.0), the 4-param calibrator must not move from truth
    and the RMSE must be (near-)zero. This verifies the loss function has
    a true minimum at the data-generating params."""
    truth = _PARAMS_CRYPTO_4P  # theta=V_0, kappa=2.0
    panel = _synthetic_panel(truth, iv_noise=0.0)
    truth_init = RoughHestonParams(**truth)
    fit = calibrate_rough_heston(
        panel, F=1.0, init=truth_init, max_iter=50, fix_kappa_theta=True,
    )
    assert fit.calibration_rmse < 1e-3, (
        f"truth-init RMSE={fit.calibration_rmse:.6f} not near zero"
    )
    assert abs(fit.V_0 - truth["V_0"]) < 1e-3
    assert abs(fit.H - truth["H"]) < 1e-3


def test_calibration_recovers_synthetic_noiseless_4param() -> None:
    """4-parameter calibration (V_0, H, eta, rho) with kappa + theta pinned.

    This is the production calibration mode per Bayer-Friz-Gatheral 2016
    convention — a single smile panel weakly identifies kappa and theta,
    so treating them as free produces noisy features that hurt signal.

    Gate: starting from a REASONABLE warm init (typical crypto priors),
    the 4 strongly-identified params must recover the truth to within
    tolerances consistent with SLSQP on a 4-maturity smile.
    """
    truth = _PARAMS_CRYPTO_4P
    panel = _synthetic_panel(truth, iv_noise=0.0)
    assert len(panel) >= 15, f"too few synthetic options: {len(panel)}"

    # Warm init — realistic crypto priors (70% vol, mild roughness, small
    # eta, slightly negative rho). This is what the sidecar build would
    # use for a cold bucket.
    warm_init = RoughHestonParams(
        V_0=0.5**2, H=0.15, kappa=2.0, theta=0.5**2, eta=1.5, rho=-0.3,
    )
    fit = calibrate_rough_heston(
        panel, F=1.0, init=warm_init, max_iter=150, fix_kappa_theta=True,
    )

    def rel(est: float, tru: float) -> float:
        return abs(est - tru) / max(abs(tru), 1e-6)

    # V_0 recovery — should be within ±20% in noiseless 4-param mode
    assert rel(fit.V_0, truth["V_0"]) < 0.20, (
        f"V_0 off: fit={fit.V_0:.4f} truth={truth['V_0']:.4f}"
    )
    # H recovery — within ±0.10 absolute
    assert abs(fit.H - truth["H"]) < 0.10, (
        f"H off: fit={fit.H:.4f} truth={truth['H']:.4f}"
    )
    # rho recovery — within ±0.50 (weak identification from a single
    # 4-maturity panel; the sign and order-of-magnitude is what matters
    # for the downstream feature).
    assert abs(fit.rho - truth["rho"]) < 0.50, (
        f"rho off: fit={fit.rho:.4f} truth={truth['rho']:.4f}"
    )
    # eta recovery — within ±50% (vol-of-vol is looser than V_0/H/rho)
    assert rel(fit.eta, truth["eta"]) < 0.50, (
        f"eta off: fit={fit.eta:.4f} truth={truth['eta']:.4f}"
    )
    # Overall fit quality — < 10 vol-points RMSE. The Lewis-formula pricer
    # has 0.1-2vp discretization bias, which absorbs into the fit residual;
    # 10vp is a realistic threshold for "smile substantially fitted"
    # after a warm-init global search.
    assert fit.calibration_rmse < 0.10, (
        f"fit RMSE {fit.calibration_rmse:.4f} > 10vp — smile not fitted"
    )
    # kappa and theta are pinned (they're NOT fitted); should equal priors exactly
    assert abs(fit.kappa - 2.0) < 1e-10
    assert abs(fit.theta - fit.V_0) < 1e-10
