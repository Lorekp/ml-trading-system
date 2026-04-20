"""Quantile Estimation via Residual Simulation (QRS) — Session I Candidate I-3.

Target engineering: **distributional** wrapper around an existing point-
forecast model. Does NOT change the point forecast; strictly ADDS a
predictive distribution for tail-risk gating, Kelly sizing, and option-
trade thresholding.

Methodology
-----------
Per Moutari et al. 2025 (arXiv:2508.15922, crypto-specific): QRS on a
linear HAR model strictly dominates ML-native distributional methods
(direct pinball-LightGBM, LightGBMLSS, NGBoost) OOS on crypto realized
variance. The >40% quantile-crossing rate of direct pinball-LightGBM
(Marchese et al. 2025, arXiv:2511.20105) is sidestepped by construction:
all quantiles are derived from a single empirical residual distribution,
so monotonic ordering is automatic.

Algorithm
---------
1. Fit: take OOS residuals ``e[T] = y_true[T] - y_pred[T]`` from a held-
   out CALIBRATION slice (last 20% of each training fold, strictly
   before the test month). Store as an empirical distribution.
2. Predict: at inference time T+, draw ``B=1000`` bootstrap samples from
   the empirical residual distribution, add to the point forecast →
   empirical predictive distribution ``{y_pred[T+] + e_b}_{b=1..B}``.
   Take quantiles at ``(0.01, 0.10, 0.25, 0.50, 0.75, 0.90, 0.99)``.
3. Score: CRPS via the existing ``crps_from_quantiles`` (pinball-sum
   approximation, Bracher et al. 2021); PIT uniformity via Kolmogorov-
   Smirnov p-value; coverage at 80/90/95% levels.

Leakage safety
--------------
- Residual distribution fit on STRICTLY HELD-OUT calibration rows. The
  caller passes ``calibration_idx`` — typically the last 20% of the
  training fold, BEFORE the test month begins. Test-set residuals never
  enter the distribution.
- Seeded bootstrap (``seed=42`` by default) → byte-identical quantiles
  across runs. Regression-locked.

See ``tests/test_qrs.py`` for pins on leakage safety, seed determinism,
monotonic quantiles, and CRPS approximation against the Gaussian
closed-form baseline.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


DEFAULT_QUANTILES: tuple[float, ...] = (
    0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99,
)
# 9 levels supporting the three symmetric intervals used in the Phase 1
# gate: 80% (q_10↔q_90), 90% (q_05↔q_95), 98% (q_01↔q_99). The 95%
# interval requires q_025/q_975 — deliberately omitted to keep the
# default tuple round-percentile-valued; pass custom ``quantiles`` if
# a 95% interval is required downstream.


@dataclass(frozen=True)
class ResidualDistribution:
    """Empirical residual distribution fit on a held-out calibration slice.

    Attributes
    ----------
    residuals
        1-D array of OOS residuals ``y_true - y_pred`` from the
        calibration slice. No smoothing, no parametric fit — raw
        empirical distribution (non-parametric by design).
    sigma
        Plug-in standard deviation of residuals, for the Gaussian baseline
        comparator used by ``crps_gaussian`` in the scoring step.
    n
        Number of calibration rows.
    """
    residuals: np.ndarray
    sigma: float
    n: int


def fit_residual_distribution(
    oof_preds: np.ndarray,
    y_true: np.ndarray,
    *,
    calibration_mask: np.ndarray | None = None,
) -> ResidualDistribution:
    """Build the empirical residual distribution on a held-out calibration slice.

    Parameters
    ----------
    oof_preds
        1-D array of out-of-fold point forecasts.
    y_true
        1-D array of realized values (same length as ``oof_preds``).
    calibration_mask
        Boolean mask over the arrays selecting calibration rows. If None,
        use ALL rows (caller's responsibility to ensure no test
        contamination). Typical pattern: pass ``last 20% of each training
        fold, strictly before the test month``.

    Returns
    -------
    ResidualDistribution.
    """
    oof_preds = np.asarray(oof_preds, dtype=float)
    y_true = np.asarray(y_true, dtype=float)
    if oof_preds.shape != y_true.shape:
        raise ValueError(
            f"shape mismatch: oof_preds={oof_preds.shape} y_true={y_true.shape}"
        )
    if calibration_mask is None:
        calibration_mask = np.ones_like(oof_preds, dtype=bool)
    else:
        calibration_mask = np.asarray(calibration_mask, dtype=bool)
        if calibration_mask.shape != oof_preds.shape:
            raise ValueError(
                f"calibration_mask shape {calibration_mask.shape} "
                f"does not match predictions {oof_preds.shape}"
            )

    valid = calibration_mask & np.isfinite(oof_preds) & np.isfinite(y_true)
    if valid.sum() < 30:
        raise ValueError(
            f"too few calibration rows ({valid.sum()} < 30); "
            f"residual distribution would be unreliable"
        )

    residuals = y_true[valid] - oof_preds[valid]
    sigma = float(np.std(residuals, ddof=1))
    return ResidualDistribution(
        residuals=residuals, sigma=sigma, n=int(valid.sum()),
    )


def predict_distribution(
    point_preds: np.ndarray,
    residual_dist: ResidualDistribution,
    *,
    n_bootstrap: int = 1000,
    quantiles: tuple[float, ...] = DEFAULT_QUANTILES,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a predictive quantile frame via bootstrap residual simulation.

    At each test row i, the predictive distribution is
    ``{point_preds[i] + e_b}_{b=1..n_bootstrap}`` where ``e_b`` are
    independent draws from ``residual_dist.residuals``. We report the
    empirical quantiles at ``quantiles``.

    Parameters
    ----------
    point_preds
        1-D array of baseline point forecasts at the test rows.
    residual_dist
        Empirical residual distribution from ``fit_residual_distribution``.
    n_bootstrap
        Number of residual draws per test row. 1000 is enough to resolve
        the 1st/99th quantiles to within ±1 rank (standard QRS choice).
    quantiles
        Tuple of quantile levels in (0, 1). Returned as columns in the
        output DataFrame.
    seed
        RNG seed for the bootstrap. Same seed → byte-identical output.

    Returns
    -------
    DataFrame with shape ``(len(point_preds), len(quantiles))``, columns
    ``q_01`` etc. (formatted as ``q_{int(100*q):02d}``). Quantiles are
    monotonically non-decreasing across columns by construction.
    """
    point_preds = np.asarray(point_preds, dtype=float)
    if point_preds.ndim != 1:
        raise ValueError(f"point_preds must be 1-D, got shape {point_preds.shape}")
    if not all(0.0 < q < 1.0 for q in quantiles):
        raise ValueError(f"all quantiles must be in (0, 1); got {quantiles}")
    if sorted(quantiles) != list(quantiles):
        raise ValueError(f"quantiles must be pre-sorted ascending; got {quantiles}")

    rng = np.random.default_rng(seed)
    n_rows = len(point_preds)
    e_src = residual_dist.residuals

    # Draw (n_rows, n_bootstrap) residuals; broadcast to predictive samples.
    # Vectorized: uses int indices into e_src for speed + determinism.
    idx = rng.integers(0, e_src.size, size=(n_rows, n_bootstrap))
    samples = point_preds[:, None] + e_src[idx]  # (n_rows, n_bootstrap)

    # np.quantile along axis=1 gives per-row quantile vector. Use "linear"
    # interpolation for standard empirical-CDF quantile semantics.
    q_arr = np.quantile(samples, quantiles, axis=1, method="linear").T
    # q_arr shape: (n_rows, n_quantiles) — already monotonic by construction
    # because np.quantile returns a monotonic vector per row when queried
    # at sorted quantile levels.

    cols = [f"q_{int(round(100 * q)):02d}" for q in quantiles]
    return pd.DataFrame(q_arr, columns=cols)


def score_distributional(
    y_true: np.ndarray,
    q_df: pd.DataFrame,
    *,
    residual_dist: ResidualDistribution | None = None,
    coverage_levels: tuple[float, ...] = (0.80, 0.90, 0.98),
) -> dict:
    """Evaluate a predictive quantile frame against realized values.

    Returns a dict of:
    - ``crps``: CRPS approximation via ``crps_from_quantiles`` using
      the (q10, q50, q90) triplet (Bracher et al. 2021, pinball-sum).
    - ``crps_gaussian``: CRPS under a Gaussian residual assumption
      (same sigma as ``residual_dist``, mu = q50). Baseline comparator
      per the pre-registered I-3 gate.
    - ``pit_ks_pvalue``: Kolmogorov-Smirnov p-value for uniformity of
      the PIT histogram (P(Y ≤ y_true) empirical). Values > 0.05 indicate
      calibration.
    - ``coverage_80 / _90 / _95``: empirical coverage at the specified
      nominal level, computed from the quantile frame.
    - ``median_spread``: median of (q90 - q10) — the sharpness metric.

    Parameters
    ----------
    y_true
        1-D realized values, length matches ``q_df``.
    q_df
        Output of ``predict_distribution``.
    residual_dist
        Optional — required for the Gaussian comparator. If None,
        ``crps_gaussian`` is NaN.
    coverage_levels
        Tuple of nominal interval levels; each α implies the (1-α)/2
        and (1+α)/2 quantiles. Only valid levels present in q_df are
        reported.
    """
    from scipy.stats import kstest, norm

    y_true = np.asarray(y_true, dtype=float)
    if len(y_true) != len(q_df):
        raise ValueError(
            f"length mismatch: y_true={len(y_true)} q_df={len(q_df)}"
        )

    # CRPS from quantile triplet (Bracher et al. 2021 approximation)
    # Matches crps_from_quantiles in vol_model/signal_research.py:1198-1213.
    # Inlined here to avoid a circular import (signal_research imports
    # from vol_model.features etc., and qrs is designed to be a low-
    # level utility). The arithmetic is bit-identical.
    def _crps_from_triplet(y: np.ndarray, q10: np.ndarray, q50: np.ndarray,
                            q90: np.ndarray) -> float:
        total = 0.0
        for alpha, q in [(0.10, q10), (0.50, q50), (0.90, q90)]:
            under = (y < q).astype(float)
            total += float(np.mean((alpha - under) * (y - q)))
        return 2.0 * total / 3.0

    q10 = q_df["q_10"].to_numpy()
    q50 = q_df["q_50"].to_numpy()
    q90 = q_df["q_90"].to_numpy()
    crps = _crps_from_triplet(y_true, q10, q50, q90)

    # Gaussian baseline — same closed-form as signal_research.crps_gaussian
    # but localized here for the same reason.
    if residual_dist is not None and residual_dist.sigma > 1e-12:
        sigma = np.full_like(y_true, residual_dist.sigma)
        z = (y_true - q50) / sigma
        crps_g = float(np.mean(
            sigma * (z * (2 * norm.cdf(z) - 1)
                     + 2 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
        ))
    else:
        crps_g = float("nan")

    # PIT uniformity via empirical-CDF construction.
    # PIT(y) = fraction of quantile columns whose value is <= y_true. With
    # 7 levels in DEFAULT_QUANTILES we get at most 8 distinct PIT values
    # — we use a more granular approach: linearly interpolate the CDF
    # implied by the quantile levels.
    q_levels = np.array([float(c[2:]) / 100.0 for c in q_df.columns])
    q_values = q_df.to_numpy()  # (n_rows, n_quantiles)
    pit = np.empty(len(y_true))
    for i in range(len(y_true)):
        y_i = y_true[i]
        row = q_values[i]
        if y_i <= row[0]:
            pit[i] = 0.0
        elif y_i >= row[-1]:
            pit[i] = 1.0
        else:
            # Linear interpolation between adjacent (quantile_level,
            # quantile_value) knots. np.interp handles this exactly.
            pit[i] = float(np.interp(y_i, row, q_levels))
    ks_stat, ks_p = kstest(pit, "uniform")

    # Coverage: for each level α, use the (1-α)/2 and (1+α)/2 quantile
    # columns if present; otherwise NaN.
    cov_out = {}
    avail = set(q_df.columns)
    for alpha in coverage_levels:
        lo_col = f"q_{int(round(100 * (1 - alpha) / 2)):02d}"
        hi_col = f"q_{int(round(100 * (1 + alpha) / 2)):02d}"
        if lo_col in avail and hi_col in avail:
            lo = q_df[lo_col].to_numpy()
            hi = q_df[hi_col].to_numpy()
            cov_out[f"coverage_{int(round(100*alpha)):02d}"] = float(
                np.mean((y_true >= lo) & (y_true <= hi))
            )
        else:
            cov_out[f"coverage_{int(round(100*alpha)):02d}"] = float("nan")

    median_spread = float(np.median(q90 - q10))

    return {
        "crps": crps,
        "crps_gaussian": crps_g,
        "pit_ks_stat": float(ks_stat),
        "pit_ks_pvalue": float(ks_p),
        "median_spread": median_spread,
        **cov_out,
    }
