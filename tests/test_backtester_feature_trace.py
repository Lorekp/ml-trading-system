"""Cycle-34 (2026-05-13): tests for backtester's feature_trace_path emit.

Pins:
1. ``BacktestConfig.feature_trace_path`` defaults to ``None`` (existing
   byte-identical behaviour preserved).
2. When set, ``_prepare()`` writes one JSONL row per bar in the window
   with the expected checkpoint keys (C0/C1/C3/C7/C8).
3. ``_iv_source_offline`` resolves the 3-tier iv chain identically to
   the ``load_data`` cascade.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from vol_arb.backtester import (
    BacktestConfig,
    _iv_source_offline,
    _safe_float,
    _safe_int,
)


def test_default_feature_trace_path_is_none() -> None:
    """Cycle-34: default ``feature_trace_path`` MUST be ``None`` (parity preserved)."""
    cfg = BacktestConfig()
    assert cfg.feature_trace_path is None
    assert cfg.feature_trace_start is None
    assert cfg.feature_trace_end is None


def test_iv_source_offline_resolves_three_tier_chain() -> None:
    """Mirror of the offline ``svi_iv_matched.fillna(iv).fillna(dvol_close)`` cascade."""
    assert _iv_source_offline(42.1, 41.9, 38.0) == "svi"
    assert _iv_source_offline(float("nan"), 41.9, 38.0) == "iv"
    assert _iv_source_offline(float("nan"), float("nan"), 38.0) == "dvol_close"
    assert _iv_source_offline(None, None, None) is None


def test_safe_float_clean() -> None:
    assert _safe_float(1.5) == 1.5
    assert _safe_float(None) is None
    assert _safe_float(float("nan")) is None
    assert _safe_float(float("inf")) is None
    assert _safe_float("not a float") is None
    assert _safe_float(2) == 2.0


def test_safe_int_clean() -> None:
    assert _safe_int(1) == 1
    assert _safe_int(None) is None
    assert _safe_int("x") is None


def test_emit_feature_trace_writes_per_bar_jsonl(tmp_path: Path) -> None:
    """End-to-end: a 3-bar synthetic df produces 3 JSONL rows with expected keys.

    We test ``_emit_feature_trace`` directly rather than running the
    full backtester (which requires a real predictions parquet). The
    helper consumes a df and produces JSONL — minimal-fixture sufficient.
    """
    from vol_arb.backtester import VolArbBacktester as Backtester

    df = pd.DataFrame({
        "index_price": [63000.0, 63500.0, 64000.0],
        "vol_ann": [0.42, 0.43, 0.41],
        "jump_fraction": [0.01, 0.05, 0.10],
        "iv_used": [42.0, 43.0, 41.0],
        "iv": [42.1, 43.1, 41.1],
        "svi_iv_matched": [42.0, 43.0, 41.0],
        "dvol_close": [40.0, 41.0, 39.0],
        "rv_pred_pct": [44.0, 44.5, 43.0],
        "y_pred": [-0.83, -0.81, -0.85],
        "edge": [2.0, 1.5, 2.0],
        "spread_iv": [1.5, 1.55, 1.7],
        "signal": [0, 0, 0],
    }, index=pd.DatetimeIndex(
        ["2026-05-13T14:30:00Z", "2026-05-13T14:45:00Z", "2026-05-13T15:00:00Z"],
        name="timestamp", tz="UTC",
    ))

    out = tmp_path / "trace.jsonl"
    cfg = BacktestConfig(
        feature_trace_path=str(out),
        feature_trace_start=None,
        feature_trace_end=None,
    )
    # Stub-construct the backtester just enough that _emit_feature_trace works.
    bt = Backtester.__new__(Backtester)
    bt.cfg = cfg
    bt._emit_feature_trace(df)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    for i, line in enumerate(lines):
        row = json.loads(line)
        assert row["side"] == "offline"
        # C0
        assert "index_price_close" in row
        # C1 — cycle-38b shifts TSRV columns +1 bar, so the window's
        # FIRST row has NaN jump_fraction/vol_ann (dropped from JSONL).
        if i > 0:
            assert "jump_fraction" in row
        # C3
        assert "iv_used_pct" in row
        assert row["iv_source"] == "svi"
        # C7
        assert "rv_pred_pct" in row
        assert "aligner_month_key" in row
        # C8
        assert "edge_vol_pts" in row
        assert "half_spread" in row
        assert "effective_long_thresh" in row
        assert "signal_final" in row


def test_emit_feature_trace_respects_window(tmp_path: Path) -> None:
    """Window-bounds filter rows; 3-bar df with mid-bar bounds → 1 row."""
    from vol_arb.backtester import VolArbBacktester as Backtester

    df = pd.DataFrame({
        "index_price": [63000.0, 63500.0, 64000.0],
        "vol_ann": [0.42, 0.43, 0.41],
        "jump_fraction": [0.01, 0.05, 0.10],
        "iv_used": [42.0, 43.0, 41.0],
        "iv": [42.1, 43.1, 41.1],
        "svi_iv_matched": [42.0, 43.0, 41.0],
        "dvol_close": [40.0, 41.0, 39.0],
        "rv_pred_pct": [44.0, 44.5, 43.0],
        "y_pred": [-0.83, -0.81, -0.85],
        "edge": [2.0, 1.5, 2.0],
        "spread_iv": [1.5, 1.55, 1.7],
        "signal": [0, 0, 0],
    }, index=pd.DatetimeIndex(
        ["2026-05-13T14:30:00Z", "2026-05-13T14:45:00Z", "2026-05-13T15:00:00Z"],
        name="timestamp", tz="UTC",
    ))

    out = tmp_path / "trace.jsonl"
    cfg = BacktestConfig(
        feature_trace_path=str(out),
        feature_trace_start="2026-05-13T14:45:00Z",
        feature_trace_end="2026-05-13T14:45:00Z",
    )
    bt = Backtester.__new__(Backtester)
    bt.cfg = cfg
    bt._emit_feature_trace(df)

    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    row = json.loads(lines[0])
    assert row["bar_close_ts"].startswith("2026-05-13T14:45:00")


def test_extract_predictions_accepts_emit_trace_flag() -> None:
    """Source sentinel: the predictions extractor exposes --emit-trace-jsonl."""
    src = Path("vol_model/extract_pruned_svi_har_pd_predictions.py").read_text(
        encoding="utf-8",
    )
    assert "--emit-trace-jsonl" in src
    assert "_emit_trace_rows" in src
    assert "transvol_embedding" in src


def test_backtester_emit_method_present() -> None:
    """Source sentinel: backtester exposes ``_emit_feature_trace``."""
    src = Path("vol_arb/backtester.py").read_text(encoding="utf-8")
    assert "feature_trace_path" in src
    assert "_emit_feature_trace" in src
    # Trace emission is gated by the config field — preserves byte-identity.
    assert "if self.cfg.feature_trace_path:" in src
