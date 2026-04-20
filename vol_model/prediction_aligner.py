"""Walk-forward prediction aligner — post-training QLIKE-bias correction.

Motivation
----------
QLIKE loss ``exp(2·(y_true - y_pred)) - 2·(y_true - y_pred) - 1`` is
asymmetric: under-prediction of log-vol incurs an exponentially growing
penalty, while over-prediction is bounded. A QLIKE-trained LGBM therefore
**systematically over-predicts log-vol** (safety margin) — the empirical
mean residual ``mean(y_true - y_pred)`` is typically negative on held-out
OOS data.

Fixing this post-hoc is cheap: fit a per-month calibration on trailing
out-of-sample residuals, apply to the target month's predictions. Matches
standard practice in realized-variance forecasting (Bollerslev-Patton-
Quaedvlieg 2016 and earlier: "Mincer-Zarnowitz-style intercept correction
is essentially free for a QLIKE-trained model").

Two methods
-----------
1. **"bias"** (QLIKE-optimal scalar shift):
   ``β* = 0.5 · log(mean(exp(2·(y_true - y_pred))))`` on the calibration
   slice; ``y_pred_adj = y_pred + β*``. This is the CLOSED-FORM QLIKE
   minimizer (derived by setting ∂/∂β of ``mean(exp(2(y_true - y_pred -
   β)) - 2(y_true - y_pred - β) - 1)`` to zero).

   Key property: for Gaussian residuals ε ~ N(μ, σ²), β* = μ + σ² —
   it adds the VARIANCE to the mean-shift. The σ² term captures the
   "safety margin" QLIKE demands; an MSE-optimal β = μ would correct
   this margin away and INCREASE test QLIKE for a QLIKE-trained model.

2. **"mz"** (QLIKE-optimal Mincer-Zarnowitz):
   ``(α*, β*) = argmin_over_cal_slice QLIKE(y_true, α·y_pred + β)``
   via scipy.optimize.minimize (Nelder-Mead, initialized at OLS). Also
   corrects scale miscalibration. More flexible than "bias" but 2
   parameters fit on the same slice → slightly more multi-testing.

Both are fit STRICTLY on trailing OOS rows (``[month_start - cal_months,
month_start)``) — no leakage from the target month into its own
calibration. Both minimize the SAME loss (QLIKE) used in production
training and evaluation, so calibration-slice QLIKE is guaranteed
non-increasing by construction.

**Key caveat** (why this might NOT help in practice): if the upstream
LGBM is already well-calibrated to QLIKE on the training distribution,
then on held-out OOS the fitted β* ≈ 0 — no correction needed, no gain.
The aligner's value appears specifically when the OOS residual
distribution differs from the training distribution (regime shift,
feature distribution drift, etc.). Empirical measurement on 50-month
production predictions is in ``vol_model/run_walkforward_aligner.py``.

Honest OOS discipline
---------------------
The calibration fit is a MODEL SELECTION step, and in principle every
such step on held-out data adds a multi-testing penalty. The usage
pattern is still honest because:
- Calibration rows for month M are strictly BEFORE M (no test-set
  contamination).
- A scalar bias or 2-param MZ fit is a very low-dimensional decision
  (1 or 2 scalars per month) relative to the OOS sample size
  (~8,000 rows × 3 months = ~24,000 rows of calibration).
- The same discipline is used in production volatility literature
  (Bollerslev-Patton-Quaedvlieg 2016 HARQ intercept correction).

Usage
-----
::

    from vol_model.prediction_aligner import (
        fit_monthly_calibrations, apply_calibrations,
    )
    preds = pd.read_parquet(".../predictions.parquet")  # y_pred, y_true

    cals = fit_monthly_calibrations(preds, method="bias", cal_months=3)
    corrected = apply_calibrations(preds, cals)

    # corrected has y_pred (original), y_true, y_pred_calibrated.

See also
--------
- ``vol_model.qrs`` for DISTRIBUTIONAL bias-correction (Session I I-3).
  The aligner here is a POINT-forecast correction; QRS is a distributional
  add-on. They can be composed (apply aligner first, then fit QRS on the
  post-calibration residuals).
- ``tests/test_prediction_aligner.py`` for the leakage + arithmetic pins.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from vol_model.model import qlike_loss

logger = logging.getLogger(__name__)


CalibrationMethod = Literal["bias", "mz", "none"]


@dataclass(frozen=True)
class MonthlyCalibration:
    """Per-month calibration coefficients fit on a trailing OOS slice.

    The applied transformation is ``y_pred_adj = alpha * y_pred + beta``.
    For ``method="bias"``, ``alpha`` is always 1.0. For ``method="none"``
    (insufficient calibration data), both coefficients are identity
    (alpha=1, beta=0) — i.e., pass-through.

    Attributes
    ----------
    month
        Target month string ``"YYYY-MM"``.
    method
        Method used to fit this calibration.
    alpha, beta
        Coefficients for the affine correction.
    n_cal
        Number of rows in the calibration slice used.
    cal_start, cal_end
        Half-open calibration window ``[cal_start, cal_end)``.
    qlike_before, qlike_after
        In-calibration-slice QLIKE before and after applying the
        correction (diagnostic; the correction is fit on this slice so
        qlike_after ≤ qlike_before by construction).
    """
    month: str
    method: CalibrationMethod
    alpha: float
    beta: float
    n_cal: int
    cal_start: pd.Timestamp
    cal_end: pd.Timestamp
    qlike_before: float
    qlike_after: float


def _month_window(month: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the half-open [start, end) window for a YYYY-MM month."""
    start = pd.Timestamp(f"{month}-01", tz="UTC")
    end = start + pd.DateOffset(months=1)
    return start, end


def fit_monthly_calibrations(
    preds_df: pd.DataFrame,
    *,
    months: list[str] | None = None,
    cal_months: int = 3,
    method: CalibrationMethod = "bias",
    min_rows: int = 500,
) -> list[MonthlyCalibration]:
    """Fit per-month calibration on trailing OOS residuals.

    For each target month M, the calibration slice is
    ``[M - cal_months, M)`` — strictly BEFORE the target month. If
    the slice has fewer than ``min_rows`` valid rows, the calibration
    is a pass-through (``method="none"``, α=1, β=0) and a warning is
    logged.

    Parameters
    ----------
    preds_df
        DataFrame indexed by timestamp with columns ``y_pred`` and
        ``y_true``. Typically from a walk-forward OOS predictions parquet.
    months
        List of target months to calibrate, ``"YYYY-MM"`` format. If
        None, calibrates every month present in ``preds_df.index``.
    cal_months
        Length of the trailing calibration window, in months. Default
        3 — balances stability (more data = more stable coefficients)
        against recency (closer data = better match to target regime).
    method
        ``"bias"`` or ``"mz"``. See module docstring.
    min_rows
        Minimum valid calibration rows required; below this, the month
        gets a pass-through calibration.

    Returns
    -------
    List of MonthlyCalibration (one per target month, in chronological
    order).
    """
    if "y_pred" not in preds_df.columns or "y_true" not in preds_df.columns:
        raise KeyError(
            "preds_df must contain 'y_pred' and 'y_true' columns "
            f"(got {list(preds_df.columns)})"
        )
    if not isinstance(preds_df.index, pd.DatetimeIndex):
        raise TypeError("preds_df must have a DatetimeIndex")
    if method not in ("bias", "mz"):
        raise ValueError(f"method must be 'bias' or 'mz'; got {method!r}")

    # Default: every unique month in the index
    if months is None:
        months = sorted(
            {f"{ts.year:04d}-{ts.month:02d}" for ts in preds_df.index}
        )

    calibrations: list[MonthlyCalibration] = []
    for m in months:
        month_start, _ = _month_window(m)
        cal_start = month_start - pd.DateOffset(months=cal_months)
        cal_end = month_start  # strictly-before the target month

        cal_mask = (
            (preds_df.index >= cal_start)
            & (preds_df.index < cal_end)
        )
        cal_df = preds_df.loc[cal_mask]
        # Filter out NaN rows
        valid = cal_df[["y_pred", "y_true"]].dropna()

        if len(valid) < min_rows:
            logger.warning(
                "%s: only %d valid calibration rows (< %d); using pass-through.",
                m, len(valid), min_rows,
            )
            calibrations.append(MonthlyCalibration(
                month=m, method="none",
                alpha=1.0, beta=0.0, n_cal=len(valid),
                cal_start=cal_start, cal_end=cal_end,
                qlike_before=float("nan"), qlike_after=float("nan"),
            ))
            continue

        y_true = valid["y_true"].to_numpy()
        y_pred = valid["y_pred"].to_numpy()

        if method == "bias":
            # QLIKE-optimal scalar shift. Derivation: minimize
            #   L(β) = mean(exp(2(y_true - y_pred - β)) - 2(y_true - y_pred - β) - 1)
            # dL/dβ = mean(-2·exp(2·residual - 2β) + 2) = 0
            # => mean(exp(2·residual)) · exp(-2β) = 1
            # => β* = 0.5 · log(mean(exp(2·residual)))
            # Clip inside log for numerical stability (tail outliers).
            residual = y_true - y_pred
            # Clip each term to prevent exp overflow at extreme residuals.
            # log(mean(exp(2r))) is robust to outliers when residuals
            # are bounded; our training uses clip_quantile on labels, so
            # residual magnitudes are controlled, but belt-and-braces.
            alpha = 1.0
            beta = float(
                0.5 * np.log(np.mean(np.exp(np.clip(2.0 * residual, -20.0, 20.0))))
            )
        elif method == "mz":
            # QLIKE-optimal MZ: minimize QLIKE over (α, β) via scipy.
            # Initialize at OLS to get a good starting point.
            from scipy.optimize import minimize
            X = np.column_stack([y_pred, np.ones_like(y_pred)])
            ols_coef, *_ = np.linalg.lstsq(X, y_true, rcond=None)
            alpha0, beta0 = float(ols_coef[0]), float(ols_coef[1])

            def _qlike_loss(params: np.ndarray) -> float:
                a, b = params
                diff = y_true - (a * y_pred + b)
                return float(np.mean(
                    np.exp(np.clip(2.0 * diff, -20.0, 20.0)) - 2.0 * diff - 1.0
                ))

            res = minimize(
                _qlike_loss, x0=np.array([alpha0, beta0]),
                method="Nelder-Mead",
                options={"xatol": 1e-6, "fatol": 1e-9, "maxiter": 500},
            )
            alpha, beta = float(res.x[0]), float(res.x[1])
        else:  # pragma: no cover
            raise RuntimeError(f"unexpected method: {method}")

        q_before = float(qlike_loss(y_true, y_pred))
        q_after = float(qlike_loss(y_true, alpha * y_pred + beta))

        calibrations.append(MonthlyCalibration(
            month=m, method=method,
            alpha=alpha, beta=beta, n_cal=int(len(valid)),
            cal_start=cal_start, cal_end=cal_end,
            qlike_before=q_before, qlike_after=q_after,
        ))

    return calibrations


def apply_calibrations(
    preds_df: pd.DataFrame,
    calibrations: list[MonthlyCalibration],
    *,
    calibrated_col: str = "y_pred_calibrated",
) -> pd.DataFrame:
    """Apply per-month calibration to a predictions DataFrame.

    For each row at timestamp T in month M, the calibrated prediction is
    ``alpha_M * y_pred + beta_M``. Rows in months without a matching
    calibration are pass-through (identity transform).

    Returns a new DataFrame with the original columns PLUS the
    ``calibrated_col`` column.
    """
    if "y_pred" not in preds_df.columns:
        raise KeyError("preds_df must contain 'y_pred' column")
    if not isinstance(preds_df.index, pd.DatetimeIndex):
        raise TypeError("preds_df must have a DatetimeIndex")

    cal_by_month = {c.month: c for c in calibrations}
    y_pred = preds_df["y_pred"].to_numpy()
    y_pred_adj = y_pred.copy()

    # Vectorized monthly assignment: compute row month strings once.
    row_months = (
        preds_df.index.strftime("%Y-%m").to_numpy()
    )
    # Apply per-month correction. This loop is O(n_months), not O(n_rows).
    for m, c in cal_by_month.items():
        mask = row_months == m
        if not mask.any():
            continue
        y_pred_adj[mask] = c.alpha * y_pred[mask] + c.beta

    out = preds_df.copy()
    out[calibrated_col] = y_pred_adj
    return out


def summarize_calibrations(
    calibrations: list[MonthlyCalibration],
    preds_df: pd.DataFrame,
    *,
    calibrated_col: str = "y_pred_calibrated",
) -> pd.DataFrame:
    """Build a per-month summary DataFrame: α, β, n_cal, QLIKE before/after
    applied to the TARGET MONTH (not the calibration slice).

    For honest OOS evaluation: the calibration fit is on trailing months,
    but the "improvement" metric we care about is QLIKE ON THE TARGET
    MONTH (genuinely held out from calibration).
    """
    if "y_true" not in preds_df.columns:
        raise KeyError("preds_df must contain 'y_true' for target-month QLIKE")
    if calibrated_col not in preds_df.columns:
        raise KeyError(
            f"preds_df missing {calibrated_col!r}; call apply_calibrations first"
        )

    rows = []
    for c in calibrations:
        start, end = _month_window(c.month)
        mask = (preds_df.index >= start) & (preds_df.index < end)
        test_df = preds_df.loc[mask].dropna(
            subset=["y_pred", "y_true", calibrated_col],
        )
        if len(test_df) == 0:
            rows.append({
                "month": c.month, "method": c.method,
                "alpha": c.alpha, "beta": c.beta, "n_cal": c.n_cal,
                "n_test": 0,
                "qlike_before_test": float("nan"),
                "qlike_after_test": float("nan"),
                "qlike_improvement_pct": float("nan"),
            })
            continue
        y_true = test_df["y_true"].to_numpy()
        q_before = float(qlike_loss(y_true, test_df["y_pred"].to_numpy()))
        q_after = float(qlike_loss(y_true, test_df[calibrated_col].to_numpy()))
        improvement = (q_after / q_before - 1.0) * 100.0 if q_before > 0 else float("nan")
        rows.append({
            "month": c.month, "method": c.method,
            "alpha": c.alpha, "beta": c.beta, "n_cal": c.n_cal,
            "n_test": int(len(test_df)),
            "qlike_before_test": q_before,
            "qlike_after_test": q_after,
            "qlike_improvement_pct": improvement,
        })
    return pd.DataFrame(rows)
