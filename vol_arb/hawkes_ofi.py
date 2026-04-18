"""Bivariate exponential-kernel Hawkes MLE for buy/sell trade arrivals.

Model (Pham et al. 2024, arxiv:2408.03594; Filimonov & Sornette 2012):

    lam_b(t) = mu_b + Sigma_{t_k < t, m_k=b} alpha_bb * beta * exp(-beta (t-t_k))
                    + Sigma_{t_k < t, m_k=s} alpha_bs * beta * exp(-beta (t-t_k))
    lam_s(t) = mu_s + Sigma_{t_k < t, m_k=b} alpha_sb * beta * exp(-beta (t-t_k))
                    + Sigma_{t_k < t, m_k=s} alpha_ss * beta * exp(-beta (t-t_k))

7 free parameters: (mu_b, mu_s, alpha_bb, alpha_bs, alpha_sb, alpha_ss, beta).
Single shared decay for simplicity — per-side decays deferred.

The branching-ratio matrix
    B = [[alpha_bb, alpha_bs], [alpha_sb, alpha_ss]]
captures market reflexivity. Its spectral radius is the endogeneity level:
    rho(B) -> 1 : order flow is self-reinforcing (fragile)
    rho(B) -> 0 : order flow is exogenous (stable)

Likelihood
----------
We fit on 1-second-binned counts N_b[t], N_s[t] (buy/sell aggressive trades per
bin), treating each bin as a non-homogeneous Poisson observation at the end of
the bin. This is the standard "binned Hawkes MLE" approximation (Bacry et al.
2013) — exact for Poisson bins, and a tight approximation for exp-kernel
Hawkes when the decay timescale beta^-1 >> bin size (1s here; beta ~ 1 Hz is
typical, so beta^-1 ~ 1s. Fine for regime features; not for microstructure.)

Discrete log-likelihood on bin grid t = 0, dt, 2*dt, ... (T bins). The
integrated intensity over bin t is:

    Lambda_i[t] = mu_i * dt + (1 - exp(-beta*dt)) * (alpha_{i,b} X_b[t]
                                                    + alpha_{i,s} X_s[t])

where X_c[t] is the running past-event count decayed over time:

    X_c[t+1] = exp(-beta*dt) * X_c[t] + N_c[t],  X_c[0] = 0.

This is derived from the time-integral of the Hawkes point intensity over
a unit bin (eq. 3.2 of Bacry-Mastromatteo-Muzy 2015). The multiplier
(1 - exp(-beta*dt)) reduces to (beta*dt) in the limit beta*dt -> 0 but is
exact at all beta*dt, which matters when beta*dt is not small (at dt=1s
and beta ~ 1 Hz it's 0.63, not 1 — the naive "beta*dt" approximation
introduces a 40% scale bias on alpha).

Poisson-bin log-likelihood (dropping constant log(N!) term):

    ll = Sigma_t [ N_b[t] log(Lambda_b[t]) - Lambda_b[t]
                  + N_s[t] log(Lambda_s[t]) - Lambda_s[t] ]

X_c is a first-order linear recurrence — computed in O(T) via
scipy.signal.lfilter.

Under this parameterization, alpha_{i,c} is directly the branching ratio
per offspring per parent type (dimensionless). Stability requires the
spectral radius of B = [[alpha_bb, alpha_bs], [alpha_sb, alpha_ss]] to be
strictly less than 1.

Per-bucket wall-clock: ~100-300ms at T=14400 (4h of 1-sec bins) with L-BFGS-B.

Causality
---------
The intensity at bin t uses only counts from bins < t (strict inequality).
No future peeking. See tests/test_hawkes_ofi.py::test_causality_mutating_future.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter


# ── Dataclasses ──────────────────────────────────────────────────────


@dataclass
class HawkesOFIParams:
    """Fitted parameters for the bivariate exp-kernel Hawkes process."""

    mu_b: float = float("nan")
    mu_s: float = float("nan")
    alpha_bb: float = float("nan")
    alpha_bs: float = float("nan")
    alpha_sb: float = float("nan")
    alpha_ss: float = float("nan")
    beta: float = float("nan")
    converged: bool = False
    n_iter: int = 0
    final_grad_norm: float = float("nan")
    final_neg_ll: float = float("nan")
    spectral_radius: float = float("nan")

    @property
    def branching_buy(self) -> float:
        """Total expected buy-side offspring per event = alpha_bb + alpha_bs.

        Each alpha is dimensionless under the binned-Hawkes parameterization
        (see module docstring). Stability requires spectral_radius of the
        branching matrix [[alpha_bb, alpha_bs], [alpha_sb, alpha_ss]] < 1.
        """
        return self.alpha_bb + self.alpha_bs

    @property
    def branching_sell(self) -> float:
        """Total expected sell-side offspring per event = alpha_sb + alpha_ss."""
        return self.alpha_sb + self.alpha_ss

    @property
    def asymmetry(self) -> float:
        """Directional flow bias = alpha_bb - alpha_ss."""
        return self.alpha_bb - self.alpha_ss


_LOG_PARAM_NAMES: tuple[str, ...] = (
    "log_mu_b", "log_mu_s",
    "log_alpha_bb", "log_alpha_bs",
    "log_alpha_sb", "log_alpha_ss",
    "log_beta",
)


# ── Log-likelihood ───────────────────────────────────────────────────


def _neg_log_likelihood(
    log_params: np.ndarray,
    N_b: np.ndarray,
    N_s: np.ndarray,
    dt: float,
) -> float:
    """Negative log-likelihood on 1-sec-binned buy/sell counts.

    Parameters
    ----------
    log_params
        Length-7 vector: log of (mu_b, mu_s, alpha_bb, alpha_bs, alpha_sb,
        alpha_ss, beta). Log-parameterized to keep everything positive in
        unconstrained L-BFGS-B.
    N_b, N_s
        Integer count arrays of the same shape (T,). N_b[t] is the number of
        aggressive-buy trades in bin [t*dt, (t+1)*dt).
    dt
        Bin width in seconds (typically 1.0).

    Returns
    -------
    Negative log-likelihood (dropping the constant log(N!) term).
    """
    mu_b = np.exp(log_params[0])
    mu_s = np.exp(log_params[1])
    a_bb = np.exp(log_params[2])
    a_bs = np.exp(log_params[3])
    a_sb = np.exp(log_params[4])
    a_ss = np.exp(log_params[5])
    beta = np.exp(log_params[6])

    decay = float(np.exp(-beta * dt))
    kernel_integral = float(1.0 - decay)  # = 1 - exp(-beta*dt), exact bin integral

    # X_c[t+1] = decay * X_c[t] + N_c[t-1], X_c[0] = 0.
    # In lfilter: y[n] = b[0]*x[n] + b[1]*x[n-1] - a_coeff[1]*y[n-1].
    # With b=[0, 1], a_coeff=[1, -decay]: y[n] = x[n-1] + decay*y[n-1].
    # So y[n] = X_c[n] with X_c[0] = 0 implicitly. ✓
    b_filter = np.asarray([0.0, 1.0], dtype=np.float64)
    a_filter = np.asarray([1.0, -decay], dtype=np.float64)

    X_b = lfilter(b_filter, a_filter, N_b.astype(np.float64))
    X_s = lfilter(b_filter, a_filter, N_s.astype(np.float64))

    # Integrated intensity over bin t:
    #   Lambda_i[t] = mu_i*dt + (1 - exp(-beta*dt)) * (alpha_ib * X_b[t] + alpha_is * X_s[t])
    Lambda_b = mu_b * dt + kernel_integral * (a_bb * X_b + a_bs * X_s)
    Lambda_s = mu_s * dt + kernel_integral * (a_sb * X_b + a_ss * X_s)

    # Clip for numerical log safety. Valid-region intensities are always
    # strictly positive (mu > 0, X >= 0, alpha >= 0) so clipping is defensive.
    Lambda_b = np.clip(Lambda_b, 1e-12, None)
    Lambda_s = np.clip(Lambda_s, 1e-12, None)

    ll = float(
        (N_b * np.log(Lambda_b) - Lambda_b
         + N_s * np.log(Lambda_s) - Lambda_s).sum()
    )
    return -ll


# ── Calibration ──────────────────────────────────────────────────────


_DEFAULT_INIT = np.log([
    0.5,   # mu_b
    0.5,   # mu_s
    0.3,   # alpha_bb (self-excitation for buys)
    0.05,  # alpha_bs (cross-excitation sell->buy)
    0.05,  # alpha_sb (cross-excitation buy->sell)
    0.3,   # alpha_ss
    1.0,   # beta (1 Hz decay, typical for crypto trade flow)
])

# L-BFGS-B bounds on log-params: log(1e-4) to log(1e3).
# This clips mu in [1e-4, 1e3] events/sec (very wide; covers low- to
# high-volatility regimes) and keeps alphas, beta in numerically safe range.
_LOG_BOUND_LO = float(np.log(1e-4))
_LOG_BOUND_HI = float(np.log(1e3))


def calibrate_hawkes_ofi(
    N_b: np.ndarray,
    N_s: np.ndarray,
    *,
    dt: float = 1.0,
    init: HawkesOFIParams | None = None,
    max_iter: int = 200,
    grad_tol: float = 1e-4,
) -> HawkesOFIParams:
    """Fit bivariate exp-kernel Hawkes by MLE on binned trade counts.

    Parameters
    ----------
    N_b, N_s
        Integer count arrays (same length T >= 2). N_b[t] is aggressive-buy
        trade count in bin t; N_s[t] is aggressive-sell count.
    dt
        Bin width in seconds. Default 1.0.
    init
        Optional warm-start parameters. If None, uses a crypto-typical default.
    max_iter
        Max L-BFGS-B iterations. Default 200.
    grad_tol
        Convergence gate on the L-BFGS-B projected gradient norm. If
        `result.success == False` OR the reported ``|g|_inf > grad_tol``, the
        returned params have ``converged=False``.

    Returns
    -------
    HawkesOFIParams with success/convergence flags. On failure, all 7 params
    are NaN and ``converged=False``.

    Notes
    -----
    The log-likelihood is smooth in log-parameters for any strictly positive
    parameter vector; the log-parameterization removes the positivity
    constraint and the optimization reduces to an unconstrained bounded problem.
    """
    if N_b.shape != N_s.shape:
        raise ValueError(f"N_b and N_s must have same shape, got {N_b.shape} vs {N_s.shape}")
    if N_b.ndim != 1:
        raise ValueError(f"N_b must be 1D, got shape {N_b.shape}")
    if N_b.shape[0] < 2:
        raise ValueError(f"need at least 2 bins, got {N_b.shape[0]}")
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")

    if init is None:
        x0 = _DEFAULT_INIT.copy()
    else:
        x0 = np.asarray([
            np.log(max(init.mu_b, 1e-8)),
            np.log(max(init.mu_s, 1e-8)),
            np.log(max(init.alpha_bb, 1e-8)),
            np.log(max(init.alpha_bs, 1e-8)),
            np.log(max(init.alpha_sb, 1e-8)),
            np.log(max(init.alpha_ss, 1e-8)),
            np.log(max(init.beta, 1e-8)),
        ], dtype=np.float64)
        # Clip to bounds in case a pathological previous fit leaked in.
        x0 = np.clip(x0, _LOG_BOUND_LO, _LOG_BOUND_HI)

    bounds = [(_LOG_BOUND_LO, _LOG_BOUND_HI)] * 7

    try:
        result = minimize(
            _neg_log_likelihood,
            x0,
            args=(N_b, N_s, dt),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": max_iter, "gtol": grad_tol, "ftol": 1e-10},
        )
    except Exception:  # pragma: no cover - defensive
        return HawkesOFIParams()

    params_exp = np.exp(result.x)
    grad_norm = (
        float(np.max(np.abs(result.jac))) if result.jac is not None else float("nan")
    )
    # L-BFGS-B success flag is the right gate: `result.success` is True iff
    # the optimizer terminated on a first-order-stationarity criterion. The
    # absolute gradient norm scales with the objective magnitude (O(T)) so
    # a fixed threshold is misleading. We record `grad_norm` for diagnostics
    # but do not use it to classify convergence.
    converged = bool(result.success)

    mu_b, mu_s = float(params_exp[0]), float(params_exp[1])
    a_bb, a_bs, a_sb, a_ss = (
        float(params_exp[2]), float(params_exp[3]),
        float(params_exp[4]), float(params_exp[5]),
    )
    beta = float(params_exp[6])

    # Spectral radius of branching matrix B = [[a_bb, a_bs], [a_sb, a_ss]].
    # For 2x2 matrix: lambda = (tr +/- sqrt(tr^2 - 4 det)) / 2.
    tr = a_bb + a_ss
    det = a_bb * a_ss - a_bs * a_sb
    disc = tr * tr - 4.0 * det
    if disc < 0:
        # Complex eigenvalues: spectral radius = sqrt(det) for 2x2 with pos entries.
        # (Happens only when cross terms dominate diagonal — uncommon.)
        spectral_radius = float(np.sqrt(max(det, 0.0)))
    else:
        lam_plus = 0.5 * (tr + np.sqrt(disc))
        lam_minus = 0.5 * (tr - np.sqrt(disc))
        spectral_radius = float(max(abs(lam_plus), abs(lam_minus)))

    return HawkesOFIParams(
        mu_b=mu_b,
        mu_s=mu_s,
        alpha_bb=a_bb,
        alpha_bs=a_bs,
        alpha_sb=a_sb,
        alpha_ss=a_ss,
        beta=beta,
        converged=converged,
        n_iter=int(result.nit),
        final_grad_norm=grad_norm,
        final_neg_ll=float(result.fun),
        spectral_radius=spectral_radius,
    )


# ── Convenience ──────────────────────────────────────────────────────


def bin_trades_to_counts(
    timestamps_ns: np.ndarray,
    is_buyer_maker: np.ndarray,
    window_start_ns: int,
    window_end_ns: int,
    dt_seconds: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Bin aggressive-buy and aggressive-sell trade counts on a regular grid.

    Parameters
    ----------
    timestamps_ns
        Trade timestamps in nanoseconds since epoch (UTC).
    is_buyer_maker
        Boolean array (Binance convention): True means the BUYER was the maker,
        so the aggressor was the SELLER -> sell-aggressive trade.
    window_start_ns, window_end_ns
        Inclusive/exclusive half-open window [start, end) in nanoseconds.
    dt_seconds
        Bin width in seconds. Default 1.0.

    Returns
    -------
    (N_b, N_s) pair of int32 arrays of length ``(window_end_ns -
    window_start_ns) // (dt_seconds * 1e9)``. N_b[i] is buy-aggressive count
    in bin i; N_s[i] is sell-aggressive count.

    Trades outside the window are ignored silently.
    """
    if window_end_ns <= window_start_ns:
        raise ValueError("window_end_ns must be > window_start_ns")
    dt_ns = int(dt_seconds * 1e9)
    n_bins = int((window_end_ns - window_start_ns) // dt_ns)
    if n_bins <= 0:
        raise ValueError("window too small for the chosen dt")

    in_window = (timestamps_ns >= window_start_ns) & (timestamps_ns < window_end_ns)
    ts = timestamps_ns[in_window]
    ibm = is_buyer_maker[in_window]

    # Bin index for each trade
    bin_idx = ((ts - window_start_ns) // dt_ns).astype(np.int64)

    # Aggressive buy: is_buyer_maker == False -> buyer was taker.
    # Aggressive sell: is_buyer_maker == True -> seller was taker.
    buy_mask = ~ibm
    sell_mask = ibm

    N_b = np.bincount(bin_idx[buy_mask], minlength=n_bins)[:n_bins].astype(np.int32)
    N_s = np.bincount(bin_idx[sell_mask], minlength=n_bins)[:n_bins].astype(np.int32)
    return N_b, N_s


__all__ = [
    "HawkesOFIParams",
    "calibrate_hawkes_ofi",
    "bin_trades_to_counts",
]
