"""Tests for vol_model.prediction_aligner (walk-forward QLIKE-bias correction).

Pins:
1. Scalar bias recovers constant residual exactly.
2. Mincer-Zarnowitz recovers known linear (α, β) on clean synthetic.
3. **LEAKAGE CRITICAL**: calibration for month M uses ONLY rows at
   timestamps strictly before M. Mutating rows at timestamps ≥ month_start
   must NOT change the calibration for that month.
4. Insufficient calibration data → pass-through (α=1, β=0, method="none").
5. Per-row monthly assignment: each row calibrated with ITS OWN month's
   coefficients, not a neighbor's.
6. apply_calibrations is pure (no side effects on input DataFrame).
7. summarize_calibrations evaluates on TARGET MONTH (not calibration slice)
   — pins that the "improvement" metric is genuinely OOS.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_model.prediction_aligner import (
    MonthlyCalibration,
    apply_calibrations,
    fit_monthly_calibrations,
    summarize_calibrations,
)


# ── Fixtures ─────────────────────────────────────────────────────────


def _synth_preds(
    n_months: int = 12, bars_per_month: int = 100,
    bias: float = 0.0, noise: float = 0.1, seed: int = 0,
    scale: float = 1.0,
) -> pd.DataFrame:
    """Build a synthetic predictions frame.

    y_pred = mu (Gaussian ~N(−0.5, 0.3))
    y_true = scale * (y_pred - bias) + noise ~ N(0, noise)

    So residuals (y_true - y_pred) have mean ≈ −bias + (scale−1)·y_pred.
    When ``scale=1.0``, the bias-method should recover ``β = -bias``.
    """
    rng = np.random.default_rng(seed)
    n = n_months * bars_per_month
    idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    y_pred = rng.normal(loc=-0.5, scale=0.3, size=n)
    y_true = scale * (y_pred - bias) + rng.normal(loc=0.0, scale=noise, size=n)
    return pd.DataFrame({"y_pred": y_pred, "y_true": y_true}, index=idx)


# ── 1. Scalar bias ──────────────────────────────────────────────────


def test_bias_recovers_constant_residual() -> None:
    """When residuals are a KNOWN CONSTANT (zero variance), QLIKE-optimal
    β equals the constant — because β* = 0.5·log(mean(exp(2·c))) =
    0.5·log(exp(2c)) = c when all residuals equal c."""
    rng = np.random.default_rng(0)
    n = 30_000  # ~10 months of 15-min bars
    idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    y_pred = rng.normal(loc=0.0, scale=0.3, size=n)
    y_true = y_pred + 0.07  # constant residual of +0.07
    df = pd.DataFrame({"y_pred": y_pred, "y_true": y_true}, index=idx)

    cals = fit_monthly_calibrations(df, cal_months=3, method="bias")
    # Month 4 (2023-04) is the first with a full 3-month cal slice.
    cal_april = next(c for c in cals if c.month == "2023-04")
    assert cal_april.method == "bias"
    assert cal_april.alpha == 1.0
    np.testing.assert_allclose(cal_april.beta, 0.07, atol=1e-6)


def test_bias_adds_variance_term_for_gaussian_residuals() -> None:
    """For Gaussian residuals ε ~ N(μ, σ²), QLIKE-optimal β* = μ + σ².
    This pins the "variance correction" property that distinguishes
    QLIKE-optimal from MSE-optimal (which would give just μ)."""
    rng = np.random.default_rng(1)
    n = 40_000
    idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    mu = -0.05  # residuals have mean -0.05
    sigma = 0.2  # and std 0.2
    y_pred = rng.normal(loc=0.0, scale=0.3, size=n)
    y_true = y_pred + rng.normal(loc=mu, scale=sigma, size=n)
    df = pd.DataFrame({"y_pred": y_pred, "y_true": y_true}, index=idx)

    cals = fit_monthly_calibrations(df, cal_months=3, method="bias")
    cal_april = next(c for c in cals if c.month == "2023-04")
    # QLIKE-optimal β* = μ + σ² = -0.05 + 0.04 = -0.01
    expected_beta = mu + sigma ** 2
    np.testing.assert_allclose(cal_april.beta, expected_beta, atol=0.01)


def test_bias_guarantees_qlike_improvement_on_cal_slice() -> None:
    """QLIKE-optimal β minimizes QLIKE on the calibration slice by
    construction. qlike_after ≤ qlike_before always holds."""
    rng = np.random.default_rng(5)
    n = 30_000
    idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    y_pred = rng.normal(loc=0.0, scale=0.3, size=n) + 0.05  # biased
    y_true = rng.normal(loc=0.0, scale=0.3, size=n)
    df = pd.DataFrame({"y_pred": y_pred, "y_true": y_true}, index=idx)

    cals = fit_monthly_calibrations(df, cal_months=3, method="bias")
    for c in cals:
        if c.method == "none":
            continue
        # QLIKE optimality: monotone non-increase on fit slice.
        assert c.qlike_after <= c.qlike_before + 1e-12, (
            f"{c.month}: qlike_after {c.qlike_after} > before {c.qlike_before}"
        )


# ── 2. Mincer-Zarnowitz ──────────────────────────────────────────────


def test_mz_recovers_linear_relation() -> None:
    """If y_true = 2 * y_pred - 0.1 with near-zero noise, QLIKE-optimal
    MZ converges to α≈2, β≈-0.1 (same as MSE-optimal in the zero-noise
    limit).
    """
    rng = np.random.default_rng(2)
    n = 30_000
    idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    y_pred = rng.normal(loc=-0.5, scale=0.3, size=n)
    y_true = 2.0 * y_pred - 0.1 + rng.normal(loc=0, scale=0.01, size=n)
    df = pd.DataFrame({"y_pred": y_pred, "y_true": y_true}, index=idx)

    cals = fit_monthly_calibrations(df, cal_months=3, method="mz")
    cal_april = next(c for c in cals if c.month == "2023-04")
    assert cal_april.method == "mz"
    # Wider tolerance because QLIKE asymmetry adds a noise-variance term;
    # with very small noise (0.01), the shift is ~1e-4 — negligible.
    np.testing.assert_allclose(cal_april.alpha, 2.0, atol=5e-3)
    np.testing.assert_allclose(cal_april.beta, -0.1, atol=1e-2)


def test_mz_guarantees_qlike_improvement_on_cal_slice() -> None:
    """QLIKE-optimal MZ is the global minimizer over (α, β); Nelder-Mead
    initialized at OLS may not find the exact global min but should
    produce QLIKE non-increase on the cal slice (at worst tied with OLS
    starting point, which is already a reasonable starting guess).
    """
    rng = np.random.default_rng(9)
    n = 30_000
    idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    y_pred = rng.normal(loc=-0.5, scale=0.3, size=n)
    y_true = 1.1 * y_pred + 0.03 + rng.normal(loc=0, scale=0.2, size=n)
    df = pd.DataFrame({"y_pred": y_pred, "y_true": y_true}, index=idx)

    cals = fit_monthly_calibrations(df, cal_months=3, method="mz")
    for c in cals:
        if c.method == "none":
            continue
        assert c.qlike_after <= c.qlike_before + 1e-9, (
            f"{c.month}: MZ qlike_after {c.qlike_after} > before {c.qlike_before}"
        )


# ── 3. LEAKAGE CRITICAL ──────────────────────────────────────────────


def test_no_test_leakage() -> None:
    """The CORE invariant: calibration for month M uses ONLY rows at
    timestamps strictly before M. Mutating test-month rows must NOT
    change the calibration for that month.

    Setup: fit calibrations on the original frame. Then clobber every
    row in month 2023-04 and beyond with absurd values. Re-fit. The
    calibration for 2023-04 (and every prior month) must be byte-
    identical.
    """
    df_orig = _synth_preds(n_months=10, bars_per_month=3000, bias=0.05, seed=7)
    cals_orig = fit_monthly_calibrations(df_orig, cal_months=3, method="bias")

    # Corrupt rows at timestamps ≥ 2023-04-01
    cutoff = pd.Timestamp("2023-04-01", tz="UTC")
    df_mut = df_orig.copy()
    mask_future = df_mut.index >= cutoff
    df_mut.loc[mask_future, "y_pred"] = 9999.0
    df_mut.loc[mask_future, "y_true"] = -9999.0

    cals_mut = fit_monthly_calibrations(df_mut, cal_months=3, method="bias")

    # For every month ≤ 2023-04 (April is the first month whose cal
    # slice is [2023-01, 2023-04), which is entirely BEFORE the
    # mutation cutoff), calibration must be byte-identical.
    for c_o, c_m in zip(cals_orig, cals_mut):
        if c_o.month > "2023-04":
            continue
        assert c_o.alpha == c_m.alpha, f"alpha drift on {c_o.month}"
        assert c_o.beta == c_m.beta, f"beta drift on {c_o.month}"


def test_leakage_mz() -> None:
    """Same leakage invariant for MZ method — mutating test rows
    cannot affect MZ coefficients fit on prior months."""
    df_orig = _synth_preds(n_months=10, bars_per_month=3000, bias=0.05, seed=8)
    cals_orig = fit_monthly_calibrations(df_orig, cal_months=3, method="mz")

    cutoff = pd.Timestamp("2023-04-01", tz="UTC")
    df_mut = df_orig.copy()
    mask_future = df_mut.index >= cutoff
    df_mut.loc[mask_future, "y_pred"] = 9999.0
    df_mut.loc[mask_future, "y_true"] = -9999.0

    cals_mut = fit_monthly_calibrations(df_mut, cal_months=3, method="mz")
    for c_o, c_m in zip(cals_orig, cals_mut):
        if c_o.month > "2023-04":
            continue
        assert c_o.alpha == c_m.alpha
        assert c_o.beta == c_m.beta


# ── 4. Insufficient data fallback ────────────────────────────────────


def test_insufficient_cal_data_passthrough() -> None:
    """Months whose cal slice has fewer than min_rows valid rows must
    get a pass-through calibration (α=1, β=0, method='none')."""
    df = _synth_preds(n_months=4, bars_per_month=50, seed=0)  # only 50 rows/month
    cals = fit_monthly_calibrations(df, cal_months=3, method="bias", min_rows=500)
    # First 3 months have partial cal; months 4+ still won't have 500 rows
    for c in cals[:3]:
        assert c.method == "none"
        assert c.alpha == 1.0
        assert c.beta == 0.0


def test_empty_cal_for_earliest_month() -> None:
    """The very first month of data has ZERO calibration rows. Must
    fall back to pass-through without error."""
    df = _synth_preds(n_months=1, bars_per_month=1000, seed=0)
    cals = fit_monthly_calibrations(df, cal_months=3, method="bias", min_rows=500)
    assert len(cals) == 1
    assert cals[0].method == "none"
    assert cals[0].n_cal == 0


# ── 5. Monthly row assignment ────────────────────────────────────────


def test_apply_assigns_per_row_by_month() -> None:
    """Rows in month M get calibrated by M's calibration — not a
    neighbor's. Pin this by constructing per-month calibrations with
    distinct (α, β) and verifying the applied y_pred_calibrated
    matches per-row."""
    n = 3000
    idx = pd.date_range("2023-01-01", periods=n, freq="15min", tz="UTC")
    # 3000 rows × 15min = 31.25 days → spans Jan + Feb
    y_pred = np.ones(n)
    y_true = np.ones(n)
    df = pd.DataFrame({"y_pred": y_pred, "y_true": y_true}, index=idx)

    # Hand-built calibrations with distinct coefficients per month.
    cals = [
        MonthlyCalibration(
            month="2023-01", method="bias",
            alpha=1.0, beta=0.5, n_cal=0,
            cal_start=pd.Timestamp("2022-10-01", tz="UTC"),
            cal_end=pd.Timestamp("2023-01-01", tz="UTC"),
            qlike_before=0.0, qlike_after=0.0,
        ),
        MonthlyCalibration(
            month="2023-02", method="bias",
            alpha=1.0, beta=-0.3, n_cal=0,
            cal_start=pd.Timestamp("2022-11-01", tz="UTC"),
            cal_end=pd.Timestamp("2023-02-01", tz="UTC"),
            qlike_before=0.0, qlike_after=0.0,
        ),
    ]

    out = apply_calibrations(df, cals)
    # Jan rows: y_pred=1 → y_calibrated = 1·1 + 0.5 = 1.5
    jan_mask = out.index < pd.Timestamp("2023-02-01", tz="UTC")
    np.testing.assert_allclose(
        out.loc[jan_mask, "y_pred_calibrated"], 1.5, atol=1e-12,
    )
    # Feb rows: 1 - 0.3 = 0.7
    feb_mask = out.index >= pd.Timestamp("2023-02-01", tz="UTC")
    np.testing.assert_allclose(
        out.loc[feb_mask, "y_pred_calibrated"], 0.7, atol=1e-12,
    )


def test_apply_passthrough_when_no_calibration() -> None:
    """Rows in a month with no matching calibration get pass-through
    (y_pred_calibrated == y_pred)."""
    df = _synth_preds(n_months=2, bars_per_month=1000, seed=0)
    # Empty calibrations list
    out = apply_calibrations(df, [])
    np.testing.assert_array_equal(out["y_pred_calibrated"], out["y_pred"])


def test_apply_is_pure() -> None:
    """apply_calibrations must not mutate the input DataFrame."""
    df = _synth_preds(n_months=2, bars_per_month=500, seed=0)
    snap_pred = df["y_pred"].to_numpy().copy()
    snap_true = df["y_true"].to_numpy().copy()

    cals = [
        MonthlyCalibration(
            month="2023-01", method="bias",
            alpha=1.0, beta=0.5, n_cal=0,
            cal_start=pd.Timestamp("2022-10-01", tz="UTC"),
            cal_end=pd.Timestamp("2023-01-01", tz="UTC"),
            qlike_before=0.0, qlike_after=0.0,
        ),
    ]
    _ = apply_calibrations(df, cals)

    np.testing.assert_array_equal(df["y_pred"].to_numpy(), snap_pred)
    np.testing.assert_array_equal(df["y_true"].to_numpy(), snap_true)
    assert "y_pred_calibrated" not in df.columns


# ── 6. summarize_calibrations ────────────────────────────────────────


def test_summarize_evaluates_on_target_month() -> None:
    """summarize_calibrations reports QLIKE on the TARGET MONTH (test
    data, not calibration data). This pins that the 'improvement'
    column is genuinely OOS."""
    df = _synth_preds(n_months=10, bars_per_month=3000, bias=0.05, seed=3)
    cals = fit_monthly_calibrations(df, cal_months=3, method="bias")
    corrected = apply_calibrations(df, cals)
    summary = summarize_calibrations(cals, corrected)

    # April is the first month with full cal slice — check that test-set
    # QLIKE is evaluated on April rows only.
    row = summary.loc[summary["month"] == "2023-04"].iloc[0]
    april_mask = (
        (df.index >= pd.Timestamp("2023-04-01", tz="UTC"))
        & (df.index < pd.Timestamp("2023-05-01", tz="UTC"))
    )
    assert row["n_test"] == int(april_mask.sum())
    # Monotone improvement NOT guaranteed on any individual month
    # (calibration can hurt on an unrepresentative month), but across
    # the full setup with a systematic bias AND QLIKE-optimal fit on
    # stationary synthetic data, improvement is directionally likely.
    # We only pin that the summary has finite values — not the sign —
    # to avoid flaky tests on low-signal synthetic setups.
    cal_ok = summary.loc[summary["method"] != "none"]
    assert (cal_ok["qlike_improvement_pct"].abs() < 100).all(), (
        "improvement_pct has sanity-implausible values"
    )
    # At least the summary has finite QLIKE columns for calibrated months
    assert cal_ok["qlike_before_test"].notna().all()
    assert cal_ok["qlike_after_test"].notna().all()


def test_summarize_zero_rows_when_no_test_data() -> None:
    """If a calibrated month has no test-window rows in preds_df,
    n_test is 0 and QLIKE columns are NaN."""
    df = _synth_preds(n_months=2, bars_per_month=1000, seed=0)
    corrected = apply_calibrations(df, [])
    # Manufacture a calibration for a month not in df
    cals = [
        MonthlyCalibration(
            month="2099-12", method="bias",
            alpha=1.0, beta=0.0, n_cal=0,
            cal_start=pd.Timestamp("2099-09-01", tz="UTC"),
            cal_end=pd.Timestamp("2099-12-01", tz="UTC"),
            qlike_before=0.0, qlike_after=0.0,
        ),
    ]
    summary = summarize_calibrations(cals, corrected)
    row = summary.iloc[0]
    assert row["n_test"] == 0
    assert np.isnan(row["qlike_before_test"])


# ── 7. Input validation ──────────────────────────────────────────────


def test_fit_missing_cols_raises() -> None:
    df = pd.DataFrame({"a": [1, 2, 3]})
    with pytest.raises(KeyError, match="y_pred"):
        fit_monthly_calibrations(df)


def test_fit_non_datetime_index_raises() -> None:
    df = pd.DataFrame({"y_pred": [1, 2], "y_true": [1, 2]})
    with pytest.raises(TypeError, match="DatetimeIndex"):
        fit_monthly_calibrations(df)


def test_fit_invalid_method_raises() -> None:
    df = _synth_preds()
    with pytest.raises(ValueError, match="method"):
        fit_monthly_calibrations(df, method="unknown")  # type: ignore[arg-type]
