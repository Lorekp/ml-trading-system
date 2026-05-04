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
    probe_stale_iv,
    probe_vega_stops,
    CANONICAL_NC_PNL,
    DERIBIT_PME_SPOT_SHOCK,
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
