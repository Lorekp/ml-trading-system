"""Tests for the Phase 7 validation script."""

from __future__ import annotations

import pandas as pd
import pytest

from deribit_paper.scripts.validate_dry_run import (
    BarComparison,
    ValidationReport,
    _simulate_backtester_signal,
    _simulate_live_predictor_signal,
    validate_parity,
)


def test_backtester_signal_long_when_edge_above_threshold():
    row = pd.Series({"rv_pred_pct": 80.0, "iv_used": 75.0})
    signal, edge = _simulate_backtester_signal(row, threshold_vp=0.5)
    assert signal == 1
    assert edge == pytest.approx(5.0)


def test_backtester_signal_short_when_edge_below_neg_threshold():
    row = pd.Series({"rv_pred_pct": 70.0, "iv_used": 75.0})
    signal, edge = _simulate_backtester_signal(row, threshold_vp=0.5)
    assert signal == -1
    assert edge == pytest.approx(-5.0)


def test_backtester_signal_flat_when_within_threshold():
    row = pd.Series({"rv_pred_pct": 75.2, "iv_used": 75.0})
    signal, edge = _simulate_backtester_signal(row, threshold_vp=0.5)
    assert signal == 0
    assert edge == pytest.approx(0.2)


def test_backtester_signal_flat_when_missing_fields():
    row = pd.Series({"rv_pred_pct": 0.0, "iv_used": 75.0})
    signal, edge = _simulate_backtester_signal(row, threshold_vp=0.5)
    assert signal == 0


def test_live_and_backtester_signals_match_on_same_row():
    """Phase 7 MVP: the two paths must agree on any given row."""
    row = pd.Series({"rv_pred_pct": 80.0, "iv_used": 75.0})
    live_signal, live_edge = _simulate_live_predictor_signal(row, threshold_vp=0.5)
    bt_signal, bt_edge = _simulate_backtester_signal(row, threshold_vp=0.5)
    assert live_signal == bt_signal
    assert live_edge == bt_edge


def test_validation_report_summary_pass():
    report = ValidationReport(
        n_bars=100, n_matches=98, n_mismatches=2,
        parity_pct=98.0, first_mismatch_ts=None, per_bar=[],
    )
    summary = report.summary()
    assert "PASS" in summary
    assert "98.0%" in summary


def test_validation_report_summary_fail():
    report = ValidationReport(
        n_bars=100, n_matches=80, n_mismatches=20,
        parity_pct=80.0, first_mismatch_ts=pd.Timestamp("2026-03-05 12:00:00", tz="UTC"),
        per_bar=[],
    )
    summary = report.summary()
    assert "FAIL" in summary
    assert "80.0%" in summary


def test_validate_parity_on_historical_parquet_full_match():
    """Run the validation script on a real historical slice.

    Uses the on-disk BTC_vol_arb.parquet.  Since the live and
    backtester paths both use the same row-level edge computation,
    parity must be 100%.  If this ever drops below 100%, either:
    - the backtester's signal formula has changed
    - the live predictor's formula has diverged
    Either way, a real bug to investigate.
    """
    from pathlib import Path
    parquet = Path("vol_arb/output/BTC_vol_arb.parquet")
    if not parquet.exists():
        pytest.skip("BTC_vol_arb.parquet not present on this machine")

    report = validate_parity(
        vol_arb_parquet=parquet,
        start="2026-03-01",
        end="2026-03-02",  # 1 day = 96 bars
        threshold_vp=0.5,
    )
    assert report.n_bars > 0
    assert report.parity_pct == 100.0, (
        f"Expected 100% parity, got {report.parity_pct:.1f}%: "
        f"first mismatch at {report.first_mismatch_ts}"
    )
