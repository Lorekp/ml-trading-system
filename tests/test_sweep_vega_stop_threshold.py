"""Tests for scripts/sweep_vega_stop_threshold.py.

Three categories:

1. Simulator unit tests on synthetic fixtures.
2. Mutation-based leakage probe on the simulator output.
3. Gate logic + CPCV path generation correctness.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.sweep_vega_stop_threshold import (
    CandidateResult,
    annualised_sharpe,
    assign_groups,
    evaluate_gate,
    make_cpcv_paths,
    psr,
    quarantine_filter,
    simulate_vega_stop_per_trade,
    GATE_PSR_PCT10_DELTA,
    GATE_OOS_WORST_CUT_USD,
    GATE_NC_SHARPE_DROP_MAX,
    OOS_VALIDATION_DAYS,
    IS_INVESTIGATION_DAYS,
    N_CPCV_GROUPS,
    N_TEST_PER_PATH,
)
from vol_model.leakage_probes import probe_causality_by_mutation


# ─── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_trade_log():
    """3 short trades + 1 long trade, all on the same day."""
    idx = pd.DatetimeIndex(
        [
            "2023-06-01 00:00",
            "2023-06-01 00:15",
            "2023-06-01 00:30",
            "2023-06-01 00:45",
        ],
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "signal": [-1, -1, 1, -1],
            "iv_used": [60.0, 62.0, 65.0, 70.0],
            "iv_exit": [61.0, 63.0, 64.0, 75.0],   # canonical exits
            "vega": [40.0, 40.0, 40.0, 50.0],
            "N_contracts": [1.5, 1.5, 1.0, 1.2],
            "net_pnl_nc": [-60.0, -60.0, -40.0, -300.0],  # canonical NC
        },
        index=idx,
    )


@pytest.fixture
def synthetic_minute_iv():
    """Path with a 25 vol-pt up-spike during the LAST trade's hold."""
    idx = pd.date_range(
        "2023-05-31 23:55", "2023-06-01 01:30", freq="1min", tz="UTC",
    )
    iv = np.full(len(idx), 60.0)
    # Spike: minutes 00:46..00:59 → IV from 70 → 95 vol pts
    spike_mask = (idx >= "2023-06-01 00:46") & (idx <= "2023-06-01 00:59")
    iv[spike_mask] = np.linspace(70.0, 95.0, int(spike_mask.sum()))
    return pd.DataFrame({"median_iv": iv, "n_trades": 1}, index=idx)


# ─── 1. Simulator behaviour ────────────────────────────────────────────


class TestSimulator:
    def test_no_stop_returns_canonical(
        self, synthetic_trade_log, synthetic_minute_iv,
    ):
        """Threshold = infinity → canonical PnL unchanged."""
        out = simulate_vega_stop_per_trade(
            synthetic_trade_log, synthetic_minute_iv, threshold_usd=1e15,
        )
        np.testing.assert_array_almost_equal(
            out["net_pnl_nc_simulated"].values,
            synthetic_trade_log["net_pnl_nc"].values,
            decimal=6,
        )

    def test_low_stop_triggers_on_iv_spike(
        self, synthetic_trade_log, synthetic_minute_iv,
    ):
        """A small threshold should fire on the LAST short during the spike.

        At minute 00:46 IV = 70 → 70 (no jump yet at start of spike); by
        00:50 IV ≈ 78. Adverse: -(-1) × 50 × (78-70) × 1.2 = +480 USD.
        With threshold $400 → triggers ~minute 00:50.
        """
        threshold = 400.0
        out = simulate_vega_stop_per_trade(
            synthetic_trade_log, synthetic_minute_iv, threshold_usd=threshold,
        )
        # Last trade should have a different (less-negative or even
        # positive) PnL after the stop, since we cap the vega bleed.
        last_orig = synthetic_trade_log["net_pnl_nc"].iloc[-1]
        last_new = out["net_pnl_nc_simulated"].iloc[-1]
        assert last_new != pytest.approx(last_orig)
        # Stop captures the move to ~78 IV instead of 75 IV (canonical
        # exit), so candidate vega leg is even MORE negative for a short
        # — we lose more on stop. Conservative.
        # Just assert behaviour changed.

    def test_signal_direction_matters(
        self, synthetic_trade_log, synthetic_minute_iv,
    ):
        """The stop fires on adverse-direction moves only.

        Build a tiny test where the SAME minute-IV path is adverse for
        a short but friendly for a long. Below baseline 60 vp ⇒ adverse
        for long (signal=+1) and friendly for short (signal=-1).
        """
        # Single short and single long, same IV history
        idx = pd.DatetimeIndex(
            ["2023-06-01 00:00", "2023-06-01 00:00:15"], tz="UTC",
        )
        # Two trades sharing the bar (rare but valid for unit test)
        tl = pd.DataFrame(
            {
                "signal": [-1, 1],
                "iv_used": [70.0, 70.0],
                "iv_exit": [70.0, 70.0],
                "vega": [50.0, 50.0],
                "N_contracts": [1.0, 1.0],
                "net_pnl_nc": [-1.0, -1.0],
            },
            index=idx,
        )
        # IV path drops to 50 (long-adverse, short-friendly)
        miv_idx = pd.date_range(
            "2023-05-31 23:55", "2023-06-01 00:30", freq="1min", tz="UTC",
        )
        miv = pd.DataFrame(
            {"median_iv": np.full(len(miv_idx), 50.0), "n_trades": 1},
            index=miv_idx,
        )
        out = simulate_vega_stop_per_trade(tl, miv, threshold_usd=100.0)
        # Short trade: minute IV (50) is BELOW iv_entry (70). Adverse
        # for short = -(-1)*50*(50-70)*1 = -1000 (negative = friendly).
        # Stop should NOT fire → canonical PnL unchanged.
        assert out["net_pnl_nc_simulated"].iloc[0] == pytest.approx(-1.0)
        # Long trade: same path is adverse. Stop should fire.
        assert out["net_pnl_nc_simulated"].iloc[1] != pytest.approx(-1.0)


# ─── 2. Leakage probe ──────────────────────────────────────────────────


class TestLeakage:
    def test_no_future_minute_iv_leak(
        self, synthetic_trade_log, synthetic_minute_iv,
    ):
        """Mutating minute_iv after some T must not change earlier trades' sim PnL."""
        # T = first trade's exit time = 00:15. Mutate minute_iv at >= 00:15.
        T = pd.Timestamp("2023-06-01 00:15", tz="UTC")
        threshold_usd = 500.0

        def feature_fn(raw_minute_iv: pd.DataFrame) -> pd.DataFrame:
            tl_subset = synthetic_trade_log.loc[
                synthetic_trade_log.index < T
            ]
            out = simulate_vega_stop_per_trade(
                tl_subset, raw_minute_iv, threshold_usd=threshold_usd,
            )
            return out

        result = probe_causality_by_mutation(
            feature_fn=feature_fn,
            raw_df=synthetic_minute_iv,
            T=synthetic_trade_log.index[0],
            mutation_cols=["median_iv"],
            mutation_mask=lambda idx, t: np.asarray(idx >= T),
        )
        assert result.passed, result.detail


# ─── 3. CPCV / gate / quarantine logic ──────────────────────────────────


class TestCPCVPaths:
    def test_path_count(self):
        paths = make_cpcv_paths(N_CPCV_GROUPS, N_TEST_PER_PATH)
        # C(6, 2) = 15
        assert len(paths) == 15

    def test_each_path_disjoint(self):
        paths = make_cpcv_paths(N_CPCV_GROUPS, N_TEST_PER_PATH)
        for p in paths:
            assert set(p.test_groups).isdisjoint(set(p.fit_groups))
            assert len(p.test_groups) == N_TEST_PER_PATH
            assert len(p.fit_groups) == N_CPCV_GROUPS - N_TEST_PER_PATH


class TestQuarantine:
    def test_oos_days_dropped(self):
        # 4 OOS dates and 2 IS dates should be dropped
        all_quarantine = OOS_VALIDATION_DAYS | IS_INVESTIGATION_DAYS
        assert len(all_quarantine) == 6

        # Build an index with one row on each quarantine day plus a clean day
        ts = list(all_quarantine) + [pd.Timestamp("2024-03-15", tz="UTC")]
        idx = pd.DatetimeIndex(ts).normalize()
        mask = quarantine_filter(idx)
        # Only the clean day (2024-03-15) should be retained
        assert int(mask.sum()) == 1

    def test_groups_chronological(self):
        """assign_groups produces chronological buckets."""
        idx = pd.date_range("2022-01-01", "2024-01-01", freq="1D", tz="UTC")
        groups = assign_groups(idx, n_groups=4)
        # First quarter of dates → group 0; last quarter → group 3
        assert groups[0] == 0
        assert groups[-1] == 3


class TestGate:
    def test_all_pass(self):
        """A candidate that improves OOS, has small Sharpe drop, and PSR ≥ +0.3 passes."""
        baseline = CandidateResult(
            threshold_usd=0.0, overall_sharpe=5.0,
            oos_worst_day_usd=-10_000.0,
            psrs=[0.7, 0.7, 0.7],
        )
        candidate = CandidateResult(
            threshold_usd=10_000.0, overall_sharpe=4.8,
            oos_worst_day_usd=-2_000.0,  # cut $8K
            psrs=[0.95, 0.95, 0.95],
            psr_deltas=[0.35, 0.35, 0.35],
        )
        gate = evaluate_gate(baseline, candidate)
        assert gate["all_pass"]

    def test_oos_fail(self):
        """Insufficient OOS cut → reject."""
        baseline = CandidateResult(
            threshold_usd=0.0, overall_sharpe=5.0,
            oos_worst_day_usd=-10_000.0,
        )
        candidate = CandidateResult(
            threshold_usd=10_000.0, overall_sharpe=4.8,
            oos_worst_day_usd=-7_000.0,  # only $3K cut, < $5K floor
            psr_deltas=[0.35, 0.35, 0.35],
        )
        gate = evaluate_gate(baseline, candidate)
        assert not gate["oos_pass"]
        assert not gate["all_pass"]

    def test_sharpe_drop_fail(self):
        """Sharpe drop > 0.3 → reject."""
        baseline = CandidateResult(
            threshold_usd=0.0, overall_sharpe=5.0,
            oos_worst_day_usd=-10_000.0,
        )
        candidate = CandidateResult(
            threshold_usd=10_000.0, overall_sharpe=4.0,  # drop 1.0
            oos_worst_day_usd=-2_000.0,
            psr_deltas=[0.35, 0.35, 0.35],
        )
        gate = evaluate_gate(baseline, candidate)
        assert not gate["sharpe_drop_pass"]
        assert not gate["all_pass"]

    def test_psr_fail(self):
        """PSR p10 below +0.3 floor → reject."""
        baseline = CandidateResult(
            threshold_usd=0.0, overall_sharpe=5.0,
            oos_worst_day_usd=-10_000.0,
        )
        candidate = CandidateResult(
            threshold_usd=10_000.0, overall_sharpe=4.8,
            oos_worst_day_usd=-2_000.0,
            psr_deltas=[0.05, 0.05, 0.05],
        )
        gate = evaluate_gate(baseline, candidate)
        assert not gate["psr_p10_pass"]
        assert not gate["all_pass"]

    def test_locked_thresholds(self):
        """Pre-committed gate constants do not drift."""
        assert GATE_PSR_PCT10_DELTA == 0.30
        assert GATE_OOS_WORST_CUT_USD == 5_000.0
        assert GATE_NC_SHARPE_DROP_MAX == 0.30


# ─── PSR + Sharpe sanity ──────────────────────────────────────────────


class TestMetrics:
    def test_zero_variance_returns_nan(self):
        constant = pd.Series([100.0] * 100)
        constant.index = pd.date_range("2023-01-01", periods=100, tz="UTC")
        assert np.isnan(annualised_sharpe(constant))
        assert np.isnan(psr(constant, 0.0))

    def test_positive_returns_high_sharpe(self):
        rng = np.random.default_rng(42)
        good = pd.Series(rng.normal(loc=0.05, scale=0.01, size=200))
        good.index = pd.date_range("2023-01-01", periods=200, tz="UTC")
        sr = annualised_sharpe(good)
        assert sr > 5.0
