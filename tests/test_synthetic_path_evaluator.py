"""Tests for vol_arb/synthetic_path_evaluator.py.

Categories:
  1. Block-bootstrap distributions on synthetic daily PnL.
  2. Determinism under fixed seeds.
  3. Sanity: bootstrapping baseline-against-itself produces near-zero deltas.
  4. Synthetic-path: alternative-PnL shape + signal-direction sanity.
  5. Daily-aggregation correctness.
  6. Distribution summary metrics.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from vol_arb.synthetic_path_evaluator import (
    DistributionSummary,
    aggregate_path_to_daily,
    block_bootstrap_deltas,
    simulate_alternative_trade_pnl,
    synthetic_path_metrics,
)
from vol_model.ou_logvol import OUParams2D


# ─── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def daily_pnl_pair():
    """Two aligned daily-PnL series: candidate adds a +$5 boost on every day."""
    idx = pd.date_range("2023-01-01", periods=200, freq="D", tz="UTC")
    rng = np.random.default_rng(seed=1)
    baseline = pd.Series(rng.normal(20.0, 100.0, 200), index=idx)
    candidate = baseline + 5.0  # uniform positive shift
    return baseline, candidate


@pytest.fixture
def ou_params() -> OUParams2D:
    return OUParams2D(
        kappa_iv=0.001, theta_iv=-0.5, sigma_iv=0.005,
        a_iv=-0.0005, b_iv=0.999, var_eps_iv=1e-5,
        kappa_rv=0.002, theta_rv=-0.3, sigma_rv=0.008,
        a_rv=-0.0006, b_rv=0.998, var_eps_rv=4e-5,
        rho=0.5,
        n_observations=100_000,
        calibration_window_start="2021-04-01T00:00:00+00:00",
        calibration_window_end="2023-10-01T00:00:00+00:00",
    )


@pytest.fixture
def synthetic_fired_trade_log():
    """5-trade fixture matching the canonical trade-log column contract."""
    idx = pd.DatetimeIndex(
        [
            "2023-06-01 00:00",
            "2023-06-01 00:15",
            "2023-06-02 00:00",
            "2023-06-02 00:15",
            "2023-06-03 00:00",
        ],
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "signal": [-1, -1, 1, -1, 1],
            "iv_used": [60.0, 62.0, 65.0, 64.0, 70.0],
            "vega": [40.0, 40.0, 40.0, 40.0, 50.0],
            "N_contracts": [1.5, 1.5, 1.0, 1.0, 1.2],
            "net_pnl_nc": [50.0, -100.0, 30.0, -20.0, 100.0],
        },
        index=idx,
    )


# ─── 1. Bootstrap distribution sanity ───────────────────────────────────


class TestBootstrap:
    def test_sharpe_delta_p50_positive_when_candidate_better(self, daily_pnl_pair):
        baseline, candidate = daily_pnl_pair
        out = block_bootstrap_deltas(
            baseline, candidate, n_reps=500, block_size_days=5, seed=42,
        )
        # Candidate is uniformly +$5/day → annualised Sharpe should improve
        assert out["sharpe_delta"].p50 > 0
        assert out["pnl_delta"].p50 > 0

    def test_baseline_against_self_returns_zero_deltas(self, daily_pnl_pair):
        baseline, _ = daily_pnl_pair
        out = block_bootstrap_deltas(
            baseline, baseline, n_reps=300, block_size_days=5, seed=7,
        )
        # All deltas should be exactly zero in this case (same path, same idx)
        assert abs(out["sharpe_delta"].mean) < 1e-9
        assert abs(out["pnl_delta"].mean) < 1e-9
        assert abs(out["maxdd_delta"].mean) < 1e-9
        assert abs(out["psr_delta"].mean) < 1e-9

    def test_deterministic_under_seed(self, daily_pnl_pair):
        baseline, candidate = daily_pnl_pair
        a = block_bootstrap_deltas(baseline, candidate, n_reps=100, seed=42)
        b = block_bootstrap_deltas(baseline, candidate, n_reps=100, seed=42)
        assert a["sharpe_delta"].mean == b["sharpe_delta"].mean
        assert a["pnl_delta"].mean == b["pnl_delta"].mean
        assert a["sharpe_delta"].p10 == b["sharpe_delta"].p10

    def test_too_few_days_raises(self):
        idx = pd.date_range("2023-01-01", periods=10, freq="D", tz="UTC")
        a = pd.Series(np.zeros(10), index=idx)
        b = pd.Series(np.zeros(10), index=idx)
        with pytest.raises(ValueError, match="at least 30 aligned"):
            block_bootstrap_deltas(a, b)

    def test_returns_distribution_summary(self, daily_pnl_pair):
        baseline, candidate = daily_pnl_pair
        out = block_bootstrap_deltas(baseline, candidate, n_reps=200, seed=1)
        for key in ("sharpe_delta", "pnl_delta", "maxdd_delta", "psr_delta"):
            ds = out[key]
            assert isinstance(ds, DistributionSummary)
            assert ds.p10 <= ds.p50 <= ds.p90
            assert ds.n_reps == 200

    def test_misaligned_indices_handled_via_intersection(self):
        """Series with overlapping but non-identical indices are intersected."""
        idx_a = pd.date_range("2023-01-01", periods=100, freq="D", tz="UTC")
        idx_b = pd.date_range("2023-02-01", periods=100, freq="D", tz="UTC")
        rng = np.random.default_rng(0)
        a = pd.Series(rng.normal(0, 1, 100), index=idx_a)
        b = pd.Series(rng.normal(0, 1, 100), index=idx_b)
        # Only ~70 days overlap
        out = block_bootstrap_deltas(a, b, n_reps=100, seed=0)
        assert out["sharpe_delta"].n_reps == 100


# ─── 4. Synthetic-path mode ──────────────────────────────────────────────


class TestSyntheticPath:
    def test_output_shape(self, synthetic_fired_trade_log, ou_params):
        out = simulate_alternative_trade_pnl(
            synthetic_fired_trade_log, ou_params,
            n_paths=10, holding_minutes=15, seed=42,
        )
        assert out.shape == (10, 5)

    def test_deterministic_under_seed(self, synthetic_fired_trade_log, ou_params):
        a = simulate_alternative_trade_pnl(
            synthetic_fired_trade_log, ou_params, n_paths=20, seed=99,
        )
        b = simulate_alternative_trade_pnl(
            synthetic_fired_trade_log, ou_params, n_paths=20, seed=99,
        )
        np.testing.assert_array_equal(a, b)

    def test_short_signal_paid_on_iv_drop(self, synthetic_fired_trade_log,
                                            ou_params):
        """A short straddle profits on IV drop. Setting θ_iv far below
        entry IV means the OU path drifts DOWN → short PnL > 0 on average.
        """
        # Override theta to force IV downward (entry IV ≈ 60-70 vol pts;
        # log(0.4) = -0.916 in log space — far below entry log_iv ≈ log(0.6))
        params_drift_down = OUParams2D(
            **{**ou_params.__dict__,
               "theta_iv": -1.5,  # log(0.22) — extreme drop target
               "kappa_iv": 0.1,   # fast pull (still small over 15min)
               "var_eps_iv": 1e-8,  # low noise so drift dominates
              }
        )
        # Re-derive a, b
        params_drift_down = OUParams2D(
            **{**params_drift_down.__dict__,
               "b_iv": float(np.exp(-params_drift_down.kappa_iv
                                     * params_drift_down.dt_minutes)),
               "a_iv": params_drift_down.theta_iv * (
                    1.0 - float(np.exp(
                        -params_drift_down.kappa_iv
                        * params_drift_down.dt_minutes
                    ))
               )}
        )
        # Subset to short trades only
        shorts = synthetic_fired_trade_log[
            synthetic_fired_trade_log["signal"] == -1
        ]
        out = simulate_alternative_trade_pnl(
            shorts, params_drift_down,
            n_paths=200, holding_minutes=15, seed=7,
        )
        # Mean PnL across paths, summed across the 3 short trades
        per_trade_mean = out.mean(axis=0)
        # All shorts should have positive expected PnL (IV drifted down)
        assert (per_trade_mean > 0).sum() >= 2, (
            f"expected ≥2 of 3 shorts to have +ve mean PnL, got {per_trade_mean}"
        )

    def test_n_contracts_override_scales_pnl(self, synthetic_fired_trade_log,
                                               ou_params):
        """Halving N_contracts halves alternative PnL (linearity check)."""
        baseline_out = simulate_alternative_trade_pnl(
            synthetic_fired_trade_log, ou_params,
            n_paths=50, holding_minutes=15, seed=42,
        )
        scaled = synthetic_fired_trade_log["N_contracts"] * 0.5
        scaled_out = simulate_alternative_trade_pnl(
            synthetic_fired_trade_log, ou_params,
            n_paths=50, holding_minutes=15, seed=42,
            n_contracts_override=scaled,
        )
        np.testing.assert_allclose(scaled_out, 0.5 * baseline_out, rtol=1e-12)

    def test_missing_required_column_raises(self, ou_params):
        idx = pd.date_range("2023-06-01", periods=3, freq="15min", tz="UTC")
        tl = pd.DataFrame(
            {"signal": [-1, -1, -1], "net_pnl_nc": [10.0, 20.0, 30.0]},
            index=idx,
        )
        with pytest.raises(KeyError, match="missing columns"):
            simulate_alternative_trade_pnl(tl, ou_params, n_paths=5)


class TestAggregation:
    def test_daily_aggregation_sum(self, synthetic_fired_trade_log):
        # Simulate a known per-trade PnL matrix
        n_paths = 3
        # 5 trades, two on 2023-06-01, two on 2023-06-02, one on 2023-06-03
        alt_pnl = np.array(
            [
                [10.0, 20.0, 100.0, 30.0, 50.0],   # path 0
                [-5.0, 15.0, 200.0, -10.0, 0.0],   # path 1
                [0.0, 0.0, 0.0, 0.0, 0.0],         # path 2
            ],
            dtype=np.float64,
        )
        daily = aggregate_path_to_daily(synthetic_fired_trade_log, alt_pnl)
        assert daily.shape == (3, 3)
        # Day 0 (2023-06-01): trades 0 + 1
        np.testing.assert_allclose(daily[0, 0], 30.0)   # 10 + 20
        np.testing.assert_allclose(daily[1, 0], 10.0)   # -5 + 15
        # Day 1 (2023-06-02): trades 2 + 3
        np.testing.assert_allclose(daily[0, 1], 130.0)  # 100 + 30
        np.testing.assert_allclose(daily[1, 1], 190.0)  # 200 + -10
        # Day 2 (2023-06-03): trade 4
        np.testing.assert_allclose(daily[0, 2], 50.0)
        np.testing.assert_allclose(daily[1, 2], 0.0)


class TestSyntheticMetrics:
    def test_sharpe_distribution_increasing(self):
        # 100 paths × 30 days, increasing-mean profile across paths
        rng = np.random.default_rng(42)
        n_paths, n_days = 100, 30
        means = np.linspace(0.0, 50.0, n_paths)
        daily = (means[:, None] + rng.normal(0, 100, size=(n_paths, n_days)))
        out = synthetic_path_metrics(daily, metric="sharpe")
        assert out.metric_name == "synthetic_sharpe"
        assert out.p10 < out.p50 < out.p90
        assert out.n_reps == n_paths

    def test_total_pnl_matches_sum(self):
        daily = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]],
                         dtype=np.float64)
        out = synthetic_path_metrics(daily, metric="total_pnl")
        # p10 = 10.5 (sum of path 0); p90 = 54.0 (sum of path 1)
        # but with only 2 reps, percentiles are interpolations
        assert out.mean == pytest.approx((6.0 + 60.0) / 2.0)

    def test_unknown_metric_raises(self):
        daily = np.zeros((2, 5))
        with pytest.raises(ValueError, match="unknown metric"):
            synthetic_path_metrics(daily, metric="bogus")
