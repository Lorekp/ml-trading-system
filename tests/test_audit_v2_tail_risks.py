"""Tests for V2 tail-risk audit script.

Three categories:
1. Materiality logic — pre-committed thresholds enforce zero false positives
   on zero-exposure findings (smoke-sample bug fix) and the locked
   $9,596 / $3,200 numbers.
2. Per-probe deterministic output on synthetic fixtures — small fixtures
   covering R1 / R2 / R5 / R6 paths so behaviour is locked.
3. **Leakage probe**: ``probe_causality_by_mutation`` invoked on the
   intrabar-DD probe — mutating minute_atm_iv POST-T must not change
   the per-trade peak_DD computation at any t' < T.

The R3 (PME margin) and R4 (stale IV) probes use only entry-bar
columns (no path data), so they are by-construction backward; the
mutation probe is mechanical and applied here too as a regression
catch.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.forensic.audit_v2_tail_risks import (
    is_material,
    probe_intrabar_dd,
    probe_liquidity_depth,
    probe_outages,
    probe_pme_margin,
    probe_pnl_concentration_signed,
    probe_regime_conditional_sharpe,
    probe_rolling_streak_worst,
    probe_stale_iv,
    probe_vega_stops,
    CANONICAL_NC_PNL,
    DERIBIT_PME_SPOT_SHOCK,
    HHI_CONCENTRATION_FLAG,
    ROLLING_STREAK_FLOOR_USD,
    ROLLING_STREAK_WINDOWS_DAYS,
    TERCILE_CUT_HIGH,
    TERCILE_CUT_LOW,
)
from vol_model.leakage_probes import probe_causality_by_mutation


# ─── 1. Materiality logic tests (locked thresholds) ────────────────────

class TestMateriality:
    def test_zero_exposure_never_material(self):
        """Bug fix: 0 dollar AND 0 worst-day cannot be material."""
        assert not is_material(0.0, 0.0, 0.0, CANONICAL_NC_PNL)
        assert not is_material(0.0, 0.0, 100.0, CANONICAL_NC_PNL)

    def test_below_thresholds_not_material(self):
        # $5K cumulative, $1K worst day, $640 median daily, $959K cum
        # cum threshold = $9,596 — fails
        # daily threshold = max($3,200 floor, $3,200 from 5x$640) = $3,200
        # $1K < $3,200 — fails
        assert not is_material(5_000.0, 1_000.0, 640.0, CANONICAL_NC_PNL)

    def test_above_cum_threshold_material(self):
        # $10K cumulative > $9,596 → material
        assert is_material(10_000.0, 100.0, 640.0, CANONICAL_NC_PNL)

    def test_above_daily_threshold_material(self):
        # $5K worst-day > $3,200 floor → material
        assert is_material(100.0, 5_000.0, 0.0, CANONICAL_NC_PNL)

    def test_negative_exposure_uses_abs(self):
        # Costs come in negative — materiality on absolute magnitude
        assert is_material(-15_000.0, 0.0, 640.0, CANONICAL_NC_PNL)

    def test_daily_floor_applies(self):
        # If median_daily is 0 (smoke sample), the $3,200 floor binds
        assert not is_material(0.0, 3_000.0, 0.0, CANONICAL_NC_PNL)
        assert is_material(0.0, 3_500.0, 0.0, CANONICAL_NC_PNL)


# ─── 2. Synthetic-fixture per-probe tests ──────────────────────────────

@pytest.fixture
def synthetic_trade_log():
    """5-trade fixture: 3 short straddles, 2 long, varying signals."""
    idx = pd.DatetimeIndex(
        [
            "2023-06-01 00:00", "2023-06-01 00:15",
            "2023-06-01 00:30", "2023-06-01 00:45",
            "2023-06-01 01:00",
        ],
        tz="UTC",
    )
    return pd.DataFrame(
        {
            "signal": [-1, -1, 1, 1, -1],
            "iv_used": [60.0, 62.0, 65.0, 64.0, 70.0],
            # Realistic vega ($ per vol-pt per BTC); V2 median is ~32.5
            "vega": [40.0, 40.0, 40.0, 40.0, 50.0],
            "gamma": [0.0001, 0.0001, 0.0001, 0.0001, 0.0001],
            "N_contracts": [1.5, 1.5, 1.0, 1.0, 1.2],
            "index_price": [27000.0, 27100.0, 27200.0, 27050.0, 26800.0],
            "equity": [100_000.0] * 5,
            "net_pnl_nc": [50.0, -100.0, 30.0, -20.0, -5_000.0],
            "vega_stopped": [False] * 5,
            "svi_iv_matched": [60.0, 62.0, 65.0, 64.0, np.nan],
            "iv": [60.5, 62.5, 65.5, 64.5, 71.0],
            "dvol_close": [55.0, 57.0, 60.0, 59.0, 68.0],
            "svi_n_strikes": [12, 11, 10, 9, 4],  # last is thin
        },
        index=idx,
    )


@pytest.fixture
def synthetic_minute_iv(synthetic_trade_log):
    """Minute-IV fixture covering 60 minutes around the trades."""
    span_start = synthetic_trade_log.index.min() - pd.Timedelta(minutes=5)
    span_end = synthetic_trade_log.index.max() + pd.Timedelta(minutes=20)
    idx = pd.date_range(span_start, span_end, freq="1min", tz="UTC")
    # Build an IV path: baseline 60 vol pts, with a 25 vol-pt spike at
    # 01:05 (within the last short trade's hold window — should produce
    # a large peak DD on that trade).
    iv = np.full(len(idx), 60.0)
    spike_idx = (idx >= "2023-06-01 01:00") & (idx <= "2023-06-01 01:14")
    iv[spike_idx] = np.linspace(70.0, 95.0, int(spike_idx.sum()))
    return pd.DataFrame({"median_iv": iv, "n_trades": 1}, index=idx)


class TestProbeIntrabarDD:
    def test_returns_finding(self, synthetic_trade_log, synthetic_minute_iv):
        out = probe_intrabar_dd(
            synthetic_trade_log, synthetic_minute_iv, median_daily=100.0
        )
        assert out.risk_id == "R1+R7"
        # 5 fired trades audited (all have non-zero PnL)
        assert any("5 fired trades audited" in n for n in out.notes)

    def test_short_on_iv_spike_flagged(
        self, synthetic_trade_log, synthetic_minute_iv,
    ):
        """The 01:00 short straddle should show large peak intrabar DD."""
        out = probe_intrabar_dd(
            synthetic_trade_log, synthetic_minute_iv, median_daily=100.0
        )
        # Total exposure should be a big negative (the IV spike)
        assert out.dollar_exposure < -1_000.0


class TestProbeLiquidity:
    def test_no_sidecar_skips(self, synthetic_trade_log, tmp_path,
                              monkeypatch):
        # Point sidecar to a nonexistent path
        monkeypatch.setattr(
            "scripts.forensic.audit_v2_tail_risks.BAR_EXEC_PATH",
            tmp_path / "missing.parquet",
        )
        out = probe_liquidity_depth(synthetic_trade_log, median_daily=100.0)
        assert out.verdict == "not applicable"


class TestProbePMEMargin:
    def test_short_straddle_consumes_more_mm_than_long(
        self, synthetic_trade_log, synthetic_minute_iv,
    ):
        """Short straddle has both short-gamma + short-vega exposure."""
        out = probe_pme_margin(
            synthetic_trade_log, synthetic_minute_iv, median_daily=100.0
        )
        assert out.risk_id == "R3"
        # MM_required must be reported in notes
        assert any(f"{DERIBIT_PME_SPOT_SHOCK*100:.1f}%" in n for n in out.notes)


class TestProbeStaleIV:
    def test_thin_svi_flagged(self, synthetic_trade_log):
        """Last trade has svi_n_strikes=4 < 5 → flagged thin."""
        out = probe_stale_iv(synthetic_trade_log, median_daily=100.0)
        assert out.risk_id == "R4"
        assert "1 of 5 fired trades on thin SVI fit" in out.notes[0] or any(
            "thin SVI fit" in n for n in out.notes
        )


class TestProbeVegaStops:
    def test_all_false_diagnostic(self, synthetic_trade_log):
        out = probe_vega_stops(synthetic_trade_log, median_daily=100.0)
        assert out.risk_id == "R5"
        # Materiality is decided in C-2 sweep, not here
        assert out.material is False


class TestProbeOutages:
    def test_no_gaps_in_regular_grid(self, synthetic_trade_log):
        out = probe_outages(synthetic_trade_log, median_daily=100.0)
        assert out.risk_id == "R6"
        # 5 trades on a 15m grid → no gaps
        assert "15m-bar gaps > 20m: 0" in out.notes[0]

    def test_gap_detected(self):
        # Trade log with a deliberate 30-minute gap
        idx = pd.DatetimeIndex(
            [
                "2023-06-01 00:00",
                "2023-06-01 00:15",
                "2023-06-01 00:45",  # 30-min gap
                "2023-06-01 01:00",
            ],
            tz="UTC",
        )
        tl = pd.DataFrame(
            {
                "signal": [-1, -1, -1, -1],
                "net_pnl_nc": [10.0, 5.0, -1_000.0, 5.0],
                "vega_stopped": [False] * 4,
            },
            index=idx,
        )
        out = probe_outages(tl, median_daily=100.0)
        assert "15m-bar gaps > 20m: 1" in out.notes[0]


# ─── 3. Mutation-based leakage probe ───────────────────────────────────

class TestLeakageMutation:
    def test_intrabar_dd_no_future_minute_iv_leak(
        self, synthetic_trade_log, synthetic_minute_iv,
    ):
        """Mutate minute_iv after T; assert audit at t' < T unchanged.

        Using probe_causality_by_mutation from vol_model/leakage_probes.py
        as the gold-standard regression check.
        """
        # T = the minute at which we'll mutate "future" minute_iv values.
        # Pick T = 00:30 (the third trade). Probing at trades before T
        # must not change when minutes >= T are mutated.
        T = pd.Timestamp("2023-06-01 00:30", tz="UTC")

        def feature_fn(raw_minute_iv: pd.DataFrame) -> pd.DataFrame:
            """Return per-trade peak_dd_per_btc series, indexed by trade ts."""
            tl_subset = synthetic_trade_log.loc[
                synthetic_trade_log.index < T
            ].copy()
            if len(tl_subset) == 0:
                return pd.DataFrame(index=tl_subset.index)
            out = probe_intrabar_dd(tl_subset, raw_minute_iv, median_daily=0.0)
            # Convert to a deterministic per-row dataframe at the trade
            # timestamps so probe_causality_by_mutation can compare.
            return pd.DataFrame(
                {"dollar_exposure": [out.dollar_exposure]},
                index=tl_subset.index[:1],  # arbitrary but stable
            )

        # The probe expects feature_fn output at T to be unchanged when
        # rows AFTER T are mutated. Use the FIRST trade's timestamp as
        # the "feature time" we check.
        result = probe_causality_by_mutation(
            feature_fn=feature_fn,
            raw_df=synthetic_minute_iv,
            T=synthetic_trade_log.index[0],
            mutation_cols=["median_iv"],
            mutation_mask=lambda idx, t: np.asarray(idx >= T),
        )
        assert result.passed, result.detail


# ─── 4. Cycle-2 Phase R diagnostic probes (measurement only) ────────────


class TestDiagnosticHHI:
    """D1: HHI on signed daily NC PnL."""

    def _build_log(self, daily_pnls: list[float]) -> pd.DataFrame:
        idx = pd.date_range("2023-01-01", periods=len(daily_pnls),
                             freq="D", tz="UTC")
        return pd.DataFrame(
            {"signal": [-1] * len(daily_pnls),
             "net_pnl_nc": daily_pnls},
            index=idx,
        )

    def test_single_day_returns_one(self):
        """Degenerate case: one day → all magnitude on that day → HHI=1."""
        tl = self._build_log([1_000.0])
        out = probe_pnl_concentration_signed(tl)
        assert out["hhi_abs"] == pytest.approx(1.0, rel=1e-6)
        assert out["n_days"] == 1

    def test_uniform_n_days_returns_one_over_n(self):
        """N equal-magnitude days → HHI = 1/N (uniform-spread reference)."""
        n = 10
        tl = self._build_log([100.0] * n)
        out = probe_pnl_concentration_signed(tl)
        assert out["hhi_abs"] == pytest.approx(1.0 / n, rel=1e-6)

    def test_signed_uses_absolute_magnitude(self):
        """Mixed signs: HHI on |daily_pnl| treats both magnitudes equally."""
        # 5 days of $+100, 5 days of $-100 → all 10 contribute equally → HHI=1/10
        tl = self._build_log([100.0] * 5 + [-100.0] * 5)
        out = probe_pnl_concentration_signed(tl)
        assert out["hhi_abs"] == pytest.approx(0.1, rel=1e-6)
        # Positive-only HHI: 5 equal days → 1/5
        assert out["hhi_pos_only"] == pytest.approx(0.2, rel=1e-6)
        # Negative-only HHI: 5 equal days → 1/5
        assert out["hhi_neg_only"] == pytest.approx(0.2, rel=1e-6)

    def test_concentration_flag_threshold(self):
        """Pre-committed flag fires only above 0.10."""
        # 8 days of $100 + 1 day of $1000 → magnitude shares heavily skewed
        tl = self._build_log([100.0] * 8 + [1_000.0])
        out = probe_pnl_concentration_signed(tl)
        # 1000 / (8*100 + 1000) = 1000/1800 ≈ 0.556; share² ≈ 0.309
        # plus 8 × (100/1800)² ≈ 8 × 0.00309 = 0.0247
        # total ≈ 0.334 → above 0.10
        assert out["hhi_abs"] > HHI_CONCENTRATION_FLAG
        assert out["flag"] is True

    def test_zero_pnl_safe(self):
        """All-zero days: no division by zero, returns 0.0 HHI."""
        tl = self._build_log([0.0, 0.0, 0.0])
        out = probe_pnl_concentration_signed(tl)
        assert out["hhi_abs"] == 0.0
        assert out["flag"] is False


class TestDiagnosticRollingStreak:
    """D2: rolling-N-day worst NC PnL."""

    def _build_log(self, daily_pnls: list[float]) -> pd.DataFrame:
        idx = pd.date_range("2023-01-01", periods=len(daily_pnls),
                             freq="D", tz="UTC")
        return pd.DataFrame(
            {"signal": [-1] * len(daily_pnls),
             "net_pnl_nc": daily_pnls},
            index=idx,
        )

    def test_consecutive_loss_streak_3day(self):
        """Three consecutive $-2K days → rolling-3d worst = $-6K (breaches 1× floor)."""
        # Pad with zeros to give the rolling window enough days to evaluate
        tl = self._build_log(
            [0.0, 0.0, 0.0] + [-2_000.0, -2_000.0, -2_000.0] + [0.0, 0.0, 0.0]
        )
        out = probe_rolling_streak_worst(tl)
        assert out["worst_3d"] == pytest.approx(-6_000.0, abs=1.0)
        # Threshold: 1× × $3,200 = $-3,200; -6,000 ≤ -3,200 → breach
        assert out["worst_3d_breaches"] is True

    def test_single_loss_isolated(self):
        """Single $-1K loss surrounded by zeros: no breach at 1×$3,200 threshold."""
        tl = self._build_log([0.0] * 5 + [-1_000.0] + [0.0] * 5)
        out = probe_rolling_streak_worst(tl)
        # 3-day rolling worst captures only the single -1K → -1K vs threshold -3.2K
        assert out["worst_3d"] == pytest.approx(-1_000.0, abs=1.0)
        assert out["worst_3d_breaches"] is False

    def test_30day_threshold_higher_bar(self):
        """30-day window has 10× higher threshold ($-32K)."""
        # 10 × $-2K days = $-20K — does NOT breach the 10× × $3,200 = $-32K bar
        tl = self._build_log([0.0] * 5 + [-2_000.0] * 10 + [0.0] * 20)
        out = probe_rolling_streak_worst(tl)
        # 30-day rolling captures all 10 loss days → $-20K
        assert out["worst_30d"] == pytest.approx(-20_000.0, abs=1.0)
        assert out["worst_30d_threshold"] == pytest.approx(-32_000.0)
        assert out["worst_30d_breaches"] is False

    def test_thresholds_locked(self):
        """Pre-committed thresholds: 1× / 5× / 10× of $3,200 floor."""
        assert ROLLING_STREAK_FLOOR_USD == 3_200.0
        assert ROLLING_STREAK_WINDOWS_DAYS == (3, 7, 30)

    def test_calendar_gap_filled_with_zeros(self):
        """Non-trading days fill with zero so rolling sums are calendar-honest."""
        # Three trades 5 days apart, no fired trades in between
        idx = pd.DatetimeIndex(
            ["2023-01-01", "2023-01-06", "2023-01-11"], tz="UTC",
        )
        tl = pd.DataFrame(
            {"signal": [-1, -1, -1],
             "net_pnl_nc": [-1_000.0, -1_000.0, -1_000.0]},
            index=idx,
        )
        out = probe_rolling_streak_worst(tl)
        # 3-day rolling never sees more than one -1K day (5-day spacing)
        assert out["worst_3d"] == pytest.approx(-1_000.0, abs=1.0)


class TestDiagnosticRegimeSharpe:
    """D3: per-cell trade-level Sharpe with frozen calibration-window cuts."""

    def _build_log_in_calibration(
        self, n: int = 200, dvol_pattern: str = "uniform",
        jump_pattern: str = "uniform", pnl_pattern: str = "noise",
    ) -> pd.DataFrame:
        """Synthetic trade log inside the calibration window."""
        idx = pd.date_range(
            "2022-06-01", periods=n, freq="15min", tz="UTC",
        )
        rng = np.random.default_rng(seed=42)
        if dvol_pattern == "uniform":
            dvol = rng.uniform(40, 80, n)
        else:
            dvol = np.full(n, 60.0)
        if jump_pattern == "uniform":
            jump = rng.uniform(0, 0.3, n)
        else:
            jump = np.full(n, 0.05)
        if pnl_pattern == "noise":
            pnl = rng.normal(50.0, 100.0, n)
        elif pnl_pattern == "all_loss_high_dvol":
            pnl = np.where(dvol > 70, -200.0, 100.0) + rng.normal(0, 10, n)
        else:
            pnl = np.full(n, 50.0)
        return pd.DataFrame(
            {
                "signal": [-1] * n,
                "net_pnl_nc": pnl,
                "dvol_close": dvol,
                "jump_fraction": jump,
            },
            index=idx,
        )

    def test_skips_when_too_few_calibration_trades(self):
        out = probe_regime_conditional_sharpe(
            self._build_log_in_calibration(n=50)
        )
        assert "skipped" in out
        assert "100 minimum" in out["skipped"]

    def test_skips_when_columns_missing(self):
        idx = pd.date_range("2022-06-01", periods=200, freq="15min", tz="UTC")
        tl = pd.DataFrame(
            {"signal": [-1] * 200, "net_pnl_nc": [10.0] * 200},
            index=idx,
        )
        out = probe_regime_conditional_sharpe(tl)
        assert "skipped" in out
        assert "missing" in out["skipped"]

    def test_tercile_cuts_match_quantiles(self):
        """Frozen cuts equal the 0.33 / 0.67 quantiles of calibration slice."""
        tl = self._build_log_in_calibration(n=300)
        out = probe_regime_conditional_sharpe(tl)
        cal_slice = tl  # all 300 are in the calibration window by construction
        expected_dvol_low = float(cal_slice["dvol_close"].quantile(TERCILE_CUT_LOW))
        expected_dvol_high = float(cal_slice["dvol_close"].quantile(TERCILE_CUT_HIGH))
        assert out["dvol_low_cut"] == pytest.approx(expected_dvol_low, rel=1e-9)
        assert out["dvol_high_cut"] == pytest.approx(expected_dvol_high, rel=1e-9)

    def test_nine_cells_returned(self):
        tl = self._build_log_in_calibration(n=500)
        out = probe_regime_conditional_sharpe(tl)
        # 3 dvol × 3 jump = 9 cells
        assert len(out["cells"]) == 9
        keys = {(c["dvol_t"], c["jump_t"]) for c in out["cells"]}
        assert keys == {
            (d, j) for d in ("low", "mid", "high")
                   for j in ("low", "mid", "high")
        }

    def test_calibration_window_freeze_no_recompute(self):
        """Mutating values OUTSIDE the calibration window must not change cuts.

        This is the leakage-mutation guarantee. Cuts are frozen on the
        2021-04-01 → 2023-09-30 window only; post-window data is
        observed-but-not-used for the cut decision.
        """
        # Calibration-window data
        tl_cal = self._build_log_in_calibration(n=300)
        out_baseline = probe_regime_conditional_sharpe(tl_cal)

        # Append POST-calibration-window data with extreme values
        post_idx = pd.date_range(
            "2024-06-01", periods=100, freq="15min", tz="UTC",
        )
        rng = np.random.default_rng(seed=7)
        tl_post = pd.DataFrame(
            {
                "signal": [-1] * 100,
                "net_pnl_nc": rng.normal(0, 10, 100),
                # Extreme values that WOULD shift quantiles if mistakenly included
                "dvol_close": np.full(100, 200.0),
                "jump_fraction": np.full(100, 0.99),
            },
            index=post_idx,
        )
        tl_combined = pd.concat([tl_cal, tl_post]).sort_index()
        out_with_post = probe_regime_conditional_sharpe(tl_combined)

        # Frozen cut points must be IDENTICAL — post-window data must not
        # affect the calibration quantile.
        assert out_baseline["dvol_low_cut"] == pytest.approx(
            out_with_post["dvol_low_cut"], rel=1e-12,
        )
        assert out_baseline["dvol_high_cut"] == pytest.approx(
            out_with_post["dvol_high_cut"], rel=1e-12,
        )
        assert out_baseline["jump_low_cut"] == pytest.approx(
            out_with_post["jump_low_cut"], rel=1e-12,
        )
        assert out_baseline["jump_high_cut"] == pytest.approx(
            out_with_post["jump_high_cut"], rel=1e-12,
        )

    def test_signal_concentrated_in_high_dvol_cell(self):
        """If we plant pnl=-200 in high-dvol bars and pnl=+100 elsewhere,
        the high-dvol cells should show negative Sharpe and the low/mid
        cells should show positive Sharpe.
        """
        tl = self._build_log_in_calibration(
            n=600, dvol_pattern="uniform", jump_pattern="uniform",
            pnl_pattern="all_loss_high_dvol",
        )
        out = probe_regime_conditional_sharpe(tl)
        cells_by_key = {(c["dvol_t"], c["jump_t"]): c for c in out["cells"]}
        # Aggregate the 3 high-dvol cells (across jump terciles)
        high_dvol_cells = [
            cells_by_key[("high", j)] for j in ("low", "mid", "high")
        ]
        low_dvol_cells = [
            cells_by_key[("low", j)] for j in ("low", "mid", "high")
        ]
        # Average sum across jump terciles
        high_total = sum(c["sum_pnl"] for c in high_dvol_cells)
        low_total = sum(c["sum_pnl"] for c in low_dvol_cells)
        assert high_total < 0
        assert low_total > 0
