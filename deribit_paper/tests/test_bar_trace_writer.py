"""Cycle-33 (2026-05-13): tests for the per-bar feature-trace writer.

Pins:
1. ``BarTraceWriter`` is a no-op when ``path=None`` (production
   default): zero file I/O.
2. Round-trip: write → re-read JSONL → field equality.
3. Source-regex sentinel: engine constructs a writer using
   ``cfg.feature_trace_path`` and the predictor accepts a ``trace=``
   kwarg. Any future refactor that drops the wire-up fails CI.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from deribit_paper.diagnostics.bar_trace import (
    BarTrace,
    BarTraceWriter,
    TOLERANCES,
    trace_to_jsonl_dict,
)


def test_disabled_writer_is_noop(tmp_path: Path) -> None:
    """``path=None`` MUST yield a writer that performs zero file I/O."""
    w = BarTraceWriter(path=None)
    assert not w.enabled
    trace = BarTrace(bar_close_ts=pd.Timestamp("2026-05-13T16:00:00Z"))
    w.record(trace)  # must not raise
    # No file was created (path is None, so nothing to check on disk).
    # Sanity: the tmp_path remains empty after the call.
    assert list(tmp_path.iterdir()) == []


def test_round_trip(tmp_path: Path) -> None:
    """A populated ``BarTrace`` round-trips through JSONL to identical values."""
    out = tmp_path / "bar_trace.jsonl"
    w = BarTraceWriter(path=str(out))
    assert w.enabled

    trace = BarTrace(
        bar_close_ts=pd.Timestamp("2026-05-13T16:45:00Z"),
        side="live",
        index_price_close=63450.5,
        n_perp_trades=812,
        n_binance_fut_trades=1234,
        tsrv=0.000123,
        rv_fast=0.000110,
        rv_slow=0.000130,
        jump_fraction=0.05,
        vol_ann_continuous=0.42,
        svi_a=0.0,
        svi_b=0.21,
        svi_rho=-0.30,
        svi_m=0.0,
        svi_sigma=0.5,
        svi_T_years=0.025,
        svi_F=63500.0,
        svi_n_strikes=11,
        atm_strike=63500.0,
        svi_iv_at_atm_pct=42.1,
        iv_used_pct=42.1,
        iv_source="svi",
        feature_vector={"tv_0": 0.1, "tv_1": 0.2, "jump_fraction": 0.05},
        transvol_embedding=[0.1] * 16,
        transvol_status="computed",
        y_pred_raw=-0.85,
        beta_star=0.037,
        y_pred_aligned=-0.813,
        rv_pred_pct=44.36,
        edge_vol_pts=2.26,
        base_spread_iv=1.5,
        spread_jump_k=3.0,
        half_spread=0.87,
        effective_long_thresh=30.87,
        effective_short_thresh=25.87,
        signal_raw=0,
        signal_final=0,
        filter_reason="FLAT — edge (+2.260vp) within +30.87/-25.87",
    )
    w.record(trace)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["side"] == "live"
    assert row["bar_close_ts"] == "2026-05-13T16:45:00+00:00"
    assert row["index_price_close"] == 63450.5
    assert row["svi_iv_at_atm_pct"] == 42.1
    assert row["iv_source"] == "svi"
    assert row["feature_vector"]["jump_fraction"] == 0.05
    assert row["transvol_embedding"] == [0.1] * 16
    assert row["signal_final"] == 0


def test_nan_fields_omitted(tmp_path: Path) -> None:
    """NaN/None fields are stripped from the serialised JSONL payload."""
    out = tmp_path / "bar_trace_nan.jsonl"
    w = BarTraceWriter(path=str(out))
    trace = BarTrace(
        bar_close_ts=pd.Timestamp("2026-05-13T17:00:00Z"),
        index_price_close=float("nan"),
        rv_pred_pct=44.0,
    )
    w.record(trace)
    row = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert "index_price_close" not in row
    assert row["rv_pred_pct"] == 44.0
    assert row["bar_close_ts"]  # always retained


def test_tolerances_dict_shape() -> None:
    """``TOLERANCES`` must have a ``default`` key plus one per checkpoint c0..c8."""
    assert "default" in TOLERANCES
    for c in ("c0", "c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8"):
        assert c in TOLERANCES, f"missing tolerance for {c}"
        assert "rtol" in TOLERANCES[c]
        assert "atol" in TOLERANCES[c]


def test_engine_wires_writer_from_config() -> None:
    """Source sentinel: engine constructs the writer using ``cfg.feature_trace_path``.

    Future refactor that drops the wire-up will fail this test in CI.
    """
    src = Path("deribit_paper/engine.py").read_text(encoding="utf-8")
    assert "BarTraceWriter(" in src, "engine must construct BarTraceWriter"
    assert "feature_trace_path" in src, (
        "engine must read cfg.feature_trace_path"
    )
    assert "bar_trace_writer" in src, (
        "engine must store the writer on self.bar_trace_writer"
    )
    # Pass-through into predictor.
    assert "trace=bar_trace" in src or "trace=trace" in src, (
        "engine.process_bar must pass the trace to predictor.predict"
    )


def test_predictor_accepts_trace_kwarg() -> None:
    """Source sentinel: ``Predictor.predict`` accepts a ``trace=`` keyword."""
    src = Path("deribit_paper/signal/predictor.py").read_text(encoding="utf-8")
    assert "trace: \"BarTrace | None\" = None" in src, (
        "Predictor.predict must accept ``trace`` kwarg with the BarTrace forward-ref type."
    )
    # Predictor populates each of C6, C7, C8 when trace is non-None.
    assert "trace.y_pred_raw" in src, "C6 hook missing"
    assert "trace.rv_pred_pct" in src, "C7 hook missing"
    assert "trace.edge_vol_pts" in src, "C8 hook missing"
