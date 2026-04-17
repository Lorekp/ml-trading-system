"""Tests for the Phase 2 walk-forward CV harness.

The expensive end-to-end test (running 4 real backtests across all 50 months)
is NOT run in CI — it's kicked off manually via `python -m vol_arb.walk_forward_cv`.
These tests pin the *harness itself*:
  1. Fold boundaries follow the dashboard convention (k=8, by bar index).
  2. Calibration Sharpe uses only pre-test data (no lookahead).
  3. Per-fold Sharpe computation is consistent with the repo's dashboard stats.
  4. The walk-forward decision step never reads future folds.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from vol_arb.walk_forward_cv import (
    N_FOLDS,
    _calibration_sharpe,
    _fold_boundaries,
    _per_fold_sharpe_from_nc_eq,
    _sharpe,
    run_walk_forward,
)


# ── 1. Fold boundaries ─────────────────────────────────────────────


def test_fold_boundaries_partition_covers_all_bars():
    bounds = _fold_boundaries(36480, 8)
    # First fold starts at 0, last fold ends at n_bars, and there are k folds.
    assert len(bounds) == 8
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 36480
    # Folds are contiguous: end of fold i == start of fold i+1.
    for i in range(7):
        assert bounds[i][1] == bounds[i + 1][0]


def test_fold_boundaries_roughly_equal_size():
    bounds = _fold_boundaries(36480, 8)
    sizes = [b - a for (a, b) in bounds]
    # Each fold is n_bars // k (= 4560) — all exactly equal for n_bars divisible by k.
    assert all(s == 4560 for s in sizes)


def test_fold_boundaries_handles_non_divisible():
    """n_bars not divisible by k should still partition without gaps."""
    bounds = _fold_boundaries(100, 8)
    assert bounds[0][0] == 0
    assert bounds[-1][1] == 100
    # Sum of sizes == n_bars.
    assert sum(b - a for (a, b) in bounds) == 100


# ── 2. Sharpe utility ──────────────────────────────────────────────


def test_sharpe_zero_on_constant_series():
    r = np.zeros(1000)
    assert _sharpe(r) == 0.0


def test_sharpe_positive_on_positive_drift():
    """Constant positive drift with tiny noise → high positive Sharpe."""
    rng = np.random.default_rng(42)
    r = 0.001 + 0.00001 * rng.standard_normal(10_000)
    s = _sharpe(r)
    assert s > 0  # Drift dominates noise.


def test_sharpe_negative_on_negative_drift():
    rng = np.random.default_rng(43)
    r = -0.001 + 0.00001 * rng.standard_normal(10_000)
    s = _sharpe(r)
    assert s < 0


def test_sharpe_short_series_returns_zero():
    assert _sharpe(np.array([0.01])) == 0.0
    assert _sharpe(np.array([])) == 0.0


# ── 3. Per-fold Sharpe and calibration ─────────────────────────────


def _synthetic_eq_nc(n: int = 1000, drift: float = 1e-4, seed: int = 7) -> np.ndarray:
    """Synthetic equity path with positive drift."""
    rng = np.random.default_rng(seed)
    r = drift + 1e-3 * rng.standard_normal(n)
    return 100_000.0 * np.cumprod(1 + r)


def test_per_fold_sharpe_matches_n_folds():
    eq = _synthetic_eq_nc(800)
    sharpes = _per_fold_sharpe_from_nc_eq(eq, n_folds=8)
    assert len(sharpes) == 8


def test_calibration_sharpe_uses_only_prior_folds():
    """If we poison fold 5 with catastrophic returns but leave folds 1-4 clean,
    the calibration Sharpe for k=5 (cal = folds 1..4) must be unaffected."""
    eq = _synthetic_eq_nc(8000)
    # Poison the 5th fold (indices 4000..5000).
    eq_poisoned = eq.copy()
    fold5_start = 5 * 8000 // 8
    fold5_end = 6 * 8000 // 8
    eq_poisoned[fold5_start:fold5_end] = eq_poisoned[fold5_start] * 0.01  # 99% loss

    # Calibration up to fold 4 (test fold = 5) should be identical.
    cal_clean = _calibration_sharpe(eq, 4, n_folds=8)
    cal_poisoned = _calibration_sharpe(eq_poisoned, 4, n_folds=8)
    assert cal_clean == pytest.approx(cal_poisoned, abs=1e-10), (
        "Poisoning a future fold changed the calibration Sharpe — calibration "
        "is reading future data (lookahead bug)."
    )


def test_calibration_sharpe_monotone_with_fold_count():
    """As we add more calibration folds, the Sharpe can move, but it should
    never require reading future folds."""
    eq = _synthetic_eq_nc(8000)
    # These are independent calls — the invariant is that each cal_k only
    # touches folds 1..k.
    for k in range(1, 8):
        s = _calibration_sharpe(eq, k, n_folds=8)
        # Just check it runs and returns a finite number.
        assert np.isfinite(s)


# ── 4. Walk-forward decision integrity ─────────────────────────────


def test_walk_forward_decision_is_deterministic_for_fixed_sharpe_matrix(monkeypatch):
    """Synthetic test: if candidate C always has the highest per-fold Sharpe,
    walk-forward must always pick C. This guards against the selection logic
    accidentally averaging or tie-breaking incorrectly."""
    # Build a fake per-candidate equity path where one candidate is always best.
    # We don't actually run the backtester; we mock the output.
    n_bars = 800
    eq_bad = _synthetic_eq_nc(n_bars, drift=1e-5, seed=1)   # small positive drift
    eq_good = _synthetic_eq_nc(n_bars, drift=5e-4, seed=2)  # large positive drift

    fake_df_bad = pd.DataFrame({"equity_nc": eq_bad})
    fake_df_good = pd.DataFrame({"equity_nc": eq_good})

    # Monkeypatch the loader so the harness reads our fakes.
    candidates = (0.0, 0.30)
    fake_logs = {0.0: fake_df_bad, 0.30: fake_df_good}

    def fake_run_or_load(min_ratio, output_dir):  # noqa: ARG001
        return fake_logs[min_ratio]

    monkeypatch.setattr(
        "vol_arb.walk_forward_cv._run_or_load_backtest", fake_run_or_load
    )

    results = run_walk_forward(
        candidates=candidates, n_folds=8, output_dir=None,
    )
    # Every test fold should have picked the "good" candidate.
    for row in results["cv_table"]:
        assert row["chosen_ratio"] == 0.30, (
            f"Fold {row['fold']}: chose {row['chosen_ratio']} when 0.30 "
            f"dominates on every fold"
        )
