"""Vectorized Black-Scholes Greeks engine (r=0 default).

All functions accept and return numpy arrays. No pandas dependency.

Conventions:
    sigma:  decimal (0.8337 not 83.37%)
    vega:   per 1 percentage point IV move (multiplied by 0.01)
    theta:  per calendar day (divided by 365.25)
    delta:  call [0,1], put [-1,0]
    r=0:    crypto — no risk-free rate
"""

from __future__ import annotations

import numpy as np
from scipy.stats import norm


def bs_d1d2(
    S: np.ndarray, K: np.ndarray, T: np.ndarray, sigma: np.ndarray, r: float = 0.0
) -> tuple[np.ndarray, np.ndarray]:
    """Compute d1 and d2. Returns (d1, d2) with NaN where inputs are invalid."""
    sqrt_T = np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return d1, d2


def compute_greeks(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
    is_call: np.ndarray,
    r: float = 0.0,
) -> dict[str, np.ndarray]:
    """Compute Delta, Gamma, Vega, Theta in one vectorized pass.

    Parameters
    ----------
    S : spot price array
    K : strike price array
    T : time to expiry in years
    sigma : implied vol as decimal (iv / 100)
    is_call : boolean array (True = call, False = put)
    r : risk-free rate (default 0)

    Returns
    -------
    dict with keys: delta, gamma, vega, theta
    Invalid rows (T<=0, sigma<=0, S<=0, K<=0) get NaN.
    """
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    is_call = np.asarray(is_call, dtype=bool)

    valid = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)

    # Safe arrays for computation (avoid log/sqrt of invalid values)
    S_s = np.where(valid, S, 1.0)
    K_s = np.where(valid, K, 1.0)
    T_s = np.where(valid, T, 1.0)
    sig_s = np.where(valid, sigma, 1.0)

    d1, d2 = bs_d1d2(S_s, K_s, T_s, sig_s, r)
    sqrt_T = np.sqrt(T_s)

    pdf_d1 = norm.pdf(d1)
    cdf_d1 = norm.cdf(d1)

    # Delta: call = N(d1), put = N(d1) - 1
    delta = np.where(is_call, cdf_d1, cdf_d1 - 1.0)

    # Gamma: same for calls and puts
    gamma = pdf_d1 / (S_s * sig_s * sqrt_T)

    # Vega: per 1% vol move
    vega = S_s * pdf_d1 * sqrt_T * 0.01

    # Theta: per calendar day (with r=0 simplification)
    # Full formula: -(S*pdf_d1*sigma)/(2*sqrt_T) - r*K*exp(-rT)*N(d2)  [call]
    # With r=0:     -(S*pdf_d1*sigma)/(2*sqrt_T)
    theta = -(S_s * pdf_d1 * sig_s) / (2.0 * sqrt_T) / 365.25

    # Mask invalid rows
    out = {"delta": delta, "gamma": gamma, "vega": vega, "theta": theta}
    for arr in out.values():
        arr[~valid] = np.nan

    return out


def bs_price(
    S: np.ndarray,
    K: np.ndarray,
    T: np.ndarray,
    sigma: np.ndarray,
    is_call: np.ndarray,
    r: float = 0.0,
) -> np.ndarray:
    """Black-Scholes option price. Vectorized, returns USD prices."""
    S = np.asarray(S, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    sigma = np.asarray(sigma, dtype=np.float64)
    is_call = np.asarray(is_call, dtype=bool)

    valid = (T > 0) & (sigma > 0) & (S > 0) & (K > 0)
    S_s = np.where(valid, S, 1.0)
    K_s = np.where(valid, K, 1.0)
    T_s = np.where(valid, T, 1.0)
    sig_s = np.where(valid, sigma, 1.0)

    d1, d2 = bs_d1d2(S_s, K_s, T_s, sig_s, r)
    call_price = S_s * norm.cdf(d1) - K_s * np.exp(-r * T_s) * norm.cdf(d2)
    put_price = K_s * np.exp(-r * T_s) * norm.cdf(-d2) - S_s * norm.cdf(-d1)

    price = np.where(is_call, call_price, put_price)
    price = np.where(valid, price, 0.0)
    return price
