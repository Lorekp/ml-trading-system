"""Tests for the historical Phase 5 HAR-PD comparison harness.

These are pure-logic tests — no real model predictions are loaded. They
pin the QLIKE arithmetic, Diebold-Mariano arithmetic, and gate decisions
against hand-crafted synthetic cases with known answers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_model.compare_pruned_svi_har_pd_vs_baseline import (
    _diebold_mariano_pvalue,
    _qlike_per_row,
    evaluate_historical_thresholds,
    compare_predictions,
)


# ── 1. QLIKE arithmetic ─────────────────────────────────────────────


def test_qlike_zero_at_perfect_prediction():
    """QLIKE(σ², σ²) = 1 − log(1) − 1 = 0."""
    y = np.array([-1.0, 0.5, 2.0, -3.0])
    q = _qlike_per_row(y, y)
    assert np.allclose(q, 0.0, atol=1e-12)


def test_qlike_is_non_negative():
    """QLIKE is a Bregman divergence and must be >= 0 for all y_true, y_pred."""
    rng = np.random.default_rng(42)
    y_true = rng.normal(0, 1, 1000)
    y_pred = rng.normal(0, 1, 1000)
    q = _qlike_per_row(y_true, y_pred)
    assert (q >= -1e-12).all(), f"min QLIKE = {q.min()}"


def test_qlike_penalizes_overprediction_more_than_underprediction():
    """The training QLIKE (code convention, vol_model.model.qlike_metric) has
    r = σ²_pred / σ²_true, so r > 1 (OVER-prediction of variance) is
    penalised more than r < 1. This is the opposite asymmetry of Patton
    2011 but matches the historical training implementation.

    If this test is flipped back to Patton convention, also update
    _qlike_per_row to use r = σ²_true / σ²_pred — but that would break
    apples-to-apples with Session C / summary.json.
    """
    # y_true = y_pred for σ = 1 (log(σ)=0, so log(var)=0 → var=1).
    y_true = np.array([0.0])  # log-vol of 0 → vol=1 → var=1
    # Over-predict vol: y_pred = log(2.0), so vol_pred=2 → var_pred=4
    # r = 4/1 = 4, training qlike: diff = log(2); 2*diff = log(4);
    # exp(2*diff) = 4 ; qlike = 4 - log(4) - 1 ≈ 1.614
    q_over = _qlike_per_row(y_true, np.array([np.log(2.0)]))
    # Under-predict vol: y_pred = log(0.5), so vol_pred=0.5 → var_pred=0.25
    # r = 0.25/1 = 0.25; qlike = 0.25 - log(0.25) - 1 = 0.25 + 1.386 - 1 = 0.636
    q_under = _qlike_per_row(y_true, np.array([np.log(0.5)]))
    assert q_over[0] > q_under[0], (
        f"Training QLIKE: over-predict {q_over[0]:.3f} should exceed "
        f"under-predict {q_under[0]:.3f} (opposite of Patton's asymmetry)."
    )
    # Quantitative sanity against analytical values
    assert abs(q_over[0] - (4.0 - np.log(4.0) - 1.0)) < 1e-10
    assert abs(q_under[0] - (0.25 - np.log(0.25) - 1.0)) < 1e-10


# ── 2. Diebold-Mariano ──────────────────────────────────────────────


def test_dm_finds_improvement_with_negative_mean_diff():
    """If candidate loss is consistently below baseline loss (d < 0), DM
    should give a significant negative stat and small p-value (lower tail)."""
    rng = np.random.default_rng(0)
    n = 5000
    d = -0.01 + 0.005 * rng.standard_normal(n)  # mean = -0.01
    dm, p = _diebold_mariano_pvalue(d, lag=4)
    assert dm < -2.0, f"DM stat {dm} should be strongly negative"
    assert p < 0.01, f"p-value {p} should be < 0.01"


def test_dm_fails_to_reject_with_zero_mean_diff():
    """If candidate and baseline losses are identically distributed, DM
    should NOT reject — p-value should be moderate."""
    rng = np.random.default_rng(1)
    n = 5000
    d = rng.standard_normal(n) * 0.01  # mean ~ 0
    dm, p = _diebold_mariano_pvalue(d, lag=4)
    assert abs(dm) < 3.0, f"DM stat |{dm}| should be moderate, got |{dm}|"
    assert 0.01 < p < 0.99, f"p-value {p} should NOT reject at 0.01"


def test_dm_short_series_returns_nan():
    dm, p = _diebold_mariano_pvalue(np.array([0.1, -0.2]), lag=4)
    assert np.isnan(dm) and np.isnan(p)


# ── 3. Historical threshold logic ───────────────────────────────────


def _build_table(
    n_total: int,
    improved_delta_pct: float,
    n_improved: int,
    n_dm_sig_of_improved: int,
) -> pd.DataFrame:
    """Synthesize a per-month comparison table with exact gate-input totals.

    Parameters
    ----------
    n_total : int
        Total months in the table.
    improved_delta_pct : float
        Δ% applied to every "improved" month (should be negative).
    n_improved : int
        Count of months flagged as improvements (Δ < 0).
    n_dm_sig_of_improved : int
        Count of improved months flagged as DM-significant at p<0.01.
        Must be <= n_improved.
    """
    assert n_dm_sig_of_improved <= n_improved
    assert improved_delta_pct < 0
    rows = []
    for i in range(n_total):
        is_improved = i < n_improved
        is_dm_sig = i < n_dm_sig_of_improved
        # Improved months carry the negative delta; non-improved are exactly zero
        # so they don't skew the mean in either direction beyond expectation.
        delta = improved_delta_pct if is_improved else 0.0
        rows.append({
            "month": f"2022-{(i % 12) + 1:02d}",
            "n": 1000,
            "qlike_baseline": 1.0,
            "qlike_candidate": 1.0 + delta / 100.0,
            "delta_qlike_pct": delta,
            "dm_stat": -3.0 if is_dm_sig else -0.5,
            "dm_p": 0.001 if is_dm_sig else 0.3,
            "spearman_baseline": 0.3,
            "spearman_candidate": 0.32,
        })
    return pd.DataFrame(rows)


def test_gate_passes_on_strong_candidate():
    # Mean = -6% × 30/50 = -3.6% → gate1 PASS
    # 30 improved ≥ 25 → gate2 PASS
    # 20 DM-sig (all also improved) ≥ 15 → gate3 PASS
    table = _build_table(
        n_total=50, improved_delta_pct=-6.0, n_improved=30, n_dm_sig_of_improved=20,
    )
    gates = evaluate_historical_thresholds(table)
    assert gates["all_thresholds_pass"] is True, gates


def test_gate_fails_on_avg_delta_above_threshold():
    # Mean = -3% × 30/50 = -1.8% → gate1 FAIL
    table = _build_table(
        n_total=50, improved_delta_pct=-3.0, n_improved=30, n_dm_sig_of_improved=20,
    )
    gates = evaluate_historical_thresholds(table)
    assert gates["gate1_avg_delta_le_-3pct"] is False
    assert gates["all_thresholds_pass"] is False


def test_gate_fails_on_insufficient_improvements():
    # 20 improved < 25 → gate2 FAIL
    table = _build_table(
        n_total=50, improved_delta_pct=-10.0, n_improved=20, n_dm_sig_of_improved=18,
    )
    gates = evaluate_historical_thresholds(table)
    assert gates["gate2_n_improved_ge_25"] is False


def test_gate_fails_on_insufficient_dm_sig():
    # 10 DM-sig < 15 → gate3 FAIL
    table = _build_table(
        n_total=50, improved_delta_pct=-6.0, n_improved=30, n_dm_sig_of_improved=10,
    )
    gates = evaluate_historical_thresholds(table)
    assert gates["gate3_n_dm_sig_ge_15"] is False


# ── 4. End-to-end compare_predictions ───────────────────────────────


def test_compare_predictions_end_to_end():
    """Synthesise two sets of predictions spanning 3 full months where
    candidate strictly improves, run the comparison, and verify."""
    rng = np.random.default_rng(123)
    # 96 bars/day × ~92 days = ~8800 bars spans Feb-Apr.
    idx = pd.date_range(
        "2022-02-01", "2022-04-30 23:45:00", freq="15min", tz="UTC",
    )
    y_true = rng.normal(0, 1, len(idx))
    y_pred_b = y_true + 0.5 * rng.standard_normal(len(idx))
    y_pred_c = y_true + 0.25 * rng.standard_normal(len(idx))
    base = pd.DataFrame({"y_true": y_true, "y_pred": y_pred_b}, index=idx)
    cand = pd.DataFrame({"y_true": y_true, "y_pred": y_pred_c}, index=idx)

    table = compare_predictions(base, cand)
    # Should see Feb, Mar, Apr (3 months).
    assert len(table) == 3, f"expected 3 months, got {len(table)}: {table['month'].tolist()}"
    assert (table["delta_qlike_pct"] < 0).all(), (
        f"candidate should strictly improve all months; got {table['delta_qlike_pct'].tolist()}"
    )


def test_compare_predictions_warns_on_y_true_mismatch(caplog):
    """If the two files have different y_true values for the same timestamp,
    emit a warning — the underlying label source must be shared."""
    idx = pd.date_range("2022-02-01", periods=1000, freq="15min", tz="UTC")
    base = pd.DataFrame(
        {"y_true": np.zeros(1000), "y_pred": np.zeros(1000)}, index=idx
    )
    cand = pd.DataFrame(
        {"y_true": np.ones(1000), "y_pred": np.zeros(1000)}, index=idx
    )
    with caplog.at_level("WARNING"):
        compare_predictions(base, cand)
    assert any("y_true max-abs-diff" in r.message for r in caplog.records)
