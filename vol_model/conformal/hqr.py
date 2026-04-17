"""HQR (Heteroscedastic Quantile Regression) + split-conformal wrapper.

Per Oosterhoorn & van der Linden 2024 (arxiv:2406.14904, "Enhancing
Reliability in Prediction Intervals using Point Forecasters"):

    1. Main point forecast ŷ(x) — reused from the production LGBM.
    2. Width estimator σ̂(x) — quantile LGBM at τ=0.5 fit to |y - ŷ| on a
       calibration fold. Captures heteroscedasticity as a function of features.
    3. Conformal multiplier q_α — computed on a held-out fold inside the
       calibration window, ensures finite-sample marginal coverage regardless
       of σ̂'s fit quality.
    4. Prediction interval: PI(x) = [ŷ(x) − q_α·σ̂(x), ŷ(x) + q_α·σ̂(x)].
    5. Spread = 2·q_α·σ̂(x).

The key advantage over vanilla quantile LGBM: the normalized nonconformity
score width correlates with actual error *by construction* — you're
regressing directly on residuals, not optimizing pinball loss which
targets the conditional median on right-skewed log-vol.

Key success criterion for this session: Spearman(spread, |y − ŷ|) ≥ 0.30
on each test month. The prior session's vanilla quantile LGBM got 0.06.

Leakage guards:
- sigma_model trained on calibration fold cal1 (first half of val window).
- q_conformal computed on cal2 (second half of val window), DISJOINT from
  cal1's training data.
- Test month is LATER in time than both cal1 and cal2.

Usage:
    result = fit_hqr(X_cal, y_cal, y_hat_cal, alpha=0.1)
    sigma_hat = result["sigma_model"].predict(X_test.values)
    spread = 2 * result["q_conformal"] * np.maximum(sigma_hat, 1e-6)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

logger = logging.getLogger(__name__)


@dataclass
class HQRFit:
    """Fitted HQR wrapper artifacts."""
    sigma_model: lgb.Booster
    q_conformal: float
    alpha: float
    n_sigma_train: int
    n_conformal: int
    sigma_cal_mean: float  # for diagnostics
    cal1_end_ts: pd.Timestamp  # last cal1 timestamp (must precede test month)
    cal2_end_ts: pd.Timestamp  # last cal2 timestamp (must precede test month)


def _default_sigma_params() -> dict:
    """LightGBM params for the width estimator (quantile τ=0.5)."""
    return dict(
        objective="quantile",
        alpha=0.5,
        metric="quantile",
        num_leaves=31,
        min_data_in_leaf=20,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        lambda_l2=1.0,
        verbosity=-1,
        num_threads=-1,
        seed=42,
    )


def fit_hqr(
    X_cal: pd.DataFrame,
    y_cal: pd.Series,
    y_hat_cal: np.ndarray,
    *,
    alpha: float = 0.1,
    cal_split: float = 0.5,
    num_boost_round: int = 300,
    sigma_params: dict | None = None,
) -> HQRFit:
    """Fit the HQR width estimator + conformal multiplier on a calibration fold.

    The calibration fold (X_cal, y_cal, y_hat_cal) is a time-ordered DataFrame
    split 50/50 by ``cal_split``: the earlier half trains σ̂(x); the later
    half computes the conformal quantile of nonconformity scores.

    Parameters
    ----------
    X_cal : features on the calibration window
    y_cal : labels (pd.Series aligned to X_cal.index)
    y_hat_cal : point forecasts from the main model on X_cal (same length)
    alpha : conformal miscoverage level (default 0.1 → 90% PI)
    cal_split : fraction of X_cal assigned to sigma-model training
    num_boost_round : LGBM boosting rounds for the width estimator
    sigma_params : override default sigma-model hyperparameters
    """
    if not isinstance(X_cal.index, pd.DatetimeIndex):
        raise ValueError("X_cal must have a DatetimeIndex (time-ordered split)")
    if len(X_cal) != len(y_cal) or len(X_cal) != len(y_hat_cal):
        raise ValueError(
            f"shape mismatch: X_cal={len(X_cal)}, y_cal={len(y_cal)}, y_hat={len(y_hat_cal)}"
        )

    order = np.argsort(X_cal.index.values)
    X_sorted = X_cal.iloc[order].copy()
    y_sorted = y_cal.iloc[order].to_numpy()
    y_hat_sorted = np.asarray(y_hat_cal)[order]

    n = len(X_sorted)
    split = int(n * cal_split)
    if split < 100 or (n - split) < 100:
        raise ValueError(
            f"calibration fold too small: n={n}, split={split} "
            f"(need ≥100 rows each for sigma-fit and conformal-quantile)"
        )

    X_cal1 = X_sorted.iloc[:split]
    X_cal2 = X_sorted.iloc[split:]
    abs_err_cal1 = np.abs(y_sorted[:split] - y_hat_sorted[:split])
    abs_err_cal2 = np.abs(y_sorted[split:] - y_hat_sorted[split:])

    # Fit σ̂(x) = quantile-LGBM at τ=0.5 on |y - ŷ| over cal1.
    # Why τ=0.5 (median)? It's robust to heavy-tailed residual distributions
    # (crypto vol has jumps) vs τ=0.25/0.75 which would use more rank info
    # from extremes. The conformal q_α correction absorbs any scale error.
    params = sigma_params or _default_sigma_params()
    dtrain = lgb.Dataset(X_cal1.values, label=abs_err_cal1)
    sigma_model = lgb.train(
        params,
        dtrain,
        num_boost_round=num_boost_round,
        callbacks=[lgb.log_evaluation(0)],
    )

    # Conformal multiplier on cal2 — DISJOINT from sigma-training data
    sigma_hat_cal2 = np.maximum(sigma_model.predict(X_cal2.values), 1e-6)
    nonconformity = abs_err_cal2 / sigma_hat_cal2

    # Ceiling quantile (Vovk 2012): q_α = ceil((n_cal2 + 1)·(1-α))/n_cal2-th order stat
    n_cal2 = len(nonconformity)
    q_level = np.clip((np.ceil((n_cal2 + 1) * (1 - alpha)) / n_cal2), 0.0, 1.0)
    q_conformal = float(np.quantile(nonconformity, q_level, method="higher"))

    logger.info(
        "[HQR] sigma-fit on %d rows, conformal on %d rows, alpha=%.2f → q_conformal=%.3f",
        split, n_cal2, alpha, q_conformal,
    )

    return HQRFit(
        sigma_model=sigma_model,
        q_conformal=q_conformal,
        alpha=alpha,
        n_sigma_train=split,
        n_conformal=n_cal2,
        sigma_cal_mean=float(sigma_hat_cal2.mean()),
        cal1_end_ts=X_cal1.index[-1],
        cal2_end_ts=X_cal2.index[-1],
    )


def predict_hqr(
    X_test: pd.DataFrame,
    y_hat_test: np.ndarray,
    fit: HQRFit,
    *,
    sigma_floor: float = 1e-6,
) -> dict[str, np.ndarray]:
    """Apply the fitted HQR wrapper to test data.

    Returns dict with:
        sigma_hat: σ̂(x) for each test row
        spread: 2 · q_conformal · σ̂(x)
        pi_lo: ŷ(x) − q_conformal · σ̂(x)
        pi_hi: ŷ(x) + q_conformal · σ̂(x)
    """
    if len(X_test) != len(y_hat_test):
        raise ValueError(f"shape mismatch: X_test={len(X_test)}, y_hat={len(y_hat_test)}")

    # Causality guard: test data must be strictly AFTER the calibration window
    if isinstance(X_test.index, pd.DatetimeIndex) and len(X_test) > 0:
        first_test = X_test.index.min()
        if first_test <= fit.cal2_end_ts:
            raise ValueError(
                f"HQR causality violation: test starts {first_test} but calibration "
                f"ended {fit.cal2_end_ts}. Calibration must strictly precede test."
            )

    sigma_hat = np.maximum(fit.sigma_model.predict(X_test.values), sigma_floor)
    half_width = fit.q_conformal * sigma_hat
    return {
        "sigma_hat": sigma_hat,
        "spread": 2.0 * half_width,
        "pi_lo": np.asarray(y_hat_test) - half_width,
        "pi_hi": np.asarray(y_hat_test) + half_width,
    }


def evaluate_hqr(
    y_test: np.ndarray,
    y_hat_test: np.ndarray,
    hqr_result: dict[str, np.ndarray],
    *,
    alpha: float = 0.1,
) -> dict[str, float]:
    """Score the HQR prediction intervals on a test window.

    Returns dict with:
        spearman_spread_err: Spearman(spread, |y - ŷ|). MUST be ≥ 0.30 to
            be actionable for sizing.
        coverage: fraction of test rows with y ∈ [PI_lo, PI_hi]. Should be
            near 1 − alpha (e.g. 0.90 for alpha=0.1).
        mean_spread: E[spread] — sanity that the intervals aren't collapsing.
        mean_abs_err: E[|y - ŷ|] — for context with mean_spread.
        crps_approx: CRPS approximation via Gaussian-from-spread assumption.
    """
    y_true = np.asarray(y_test)
    y_hat = np.asarray(y_hat_test)
    spread = hqr_result["spread"]
    pi_lo = hqr_result["pi_lo"]
    pi_hi = hqr_result["pi_hi"]

    abs_err = np.abs(y_true - y_hat)
    try:
        rho = spearmanr(spread, abs_err).statistic
    except Exception:  # noqa: BLE001
        rho = float("nan")

    coverage = float(np.mean((y_true >= pi_lo) & (y_true <= pi_hi)))

    # CRPS under Gaussian assumption: σ ≈ spread / (2 · 1.6449) for 90% PI
    z_alpha = 1.6449  # for α=0.1 (90% PI under Gaussian)
    sigma_gauss = np.maximum(spread / (2.0 * z_alpha), 1e-6)
    from scipy.stats import norm
    z = (y_true - y_hat) / sigma_gauss
    crps = float(np.mean(
        sigma_gauss * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    ))

    return {
        "spearman_spread_err": float(rho),
        "coverage": coverage,
        "mean_spread": float(spread.mean()),
        "mean_abs_err": float(abs_err.mean()),
        "crps_approx": crps,
    }


def crps_gaussian_baseline(
    y_test: np.ndarray, y_hat_test: np.ndarray, sigma_const: float,
) -> float:
    """CRPS baseline under a Gaussian with a constant σ (no heteroscedasticity).

    Used to establish "did HQR's heteroscedasticity add value on top of a
    flat-variance Gaussian?". If HQR's CRPS isn't lower than this baseline,
    the width estimator is adding no value.
    """
    y_true = np.asarray(y_test)
    y_hat = np.asarray(y_hat_test)
    sigma = max(sigma_const, 1e-6)
    z = (y_true - y_hat) / sigma
    from scipy.stats import norm
    return float(np.mean(
        sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    ))
