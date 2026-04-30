"""Tests for ``scripts/h15_alignment_harness.py``.

I4 in plan ``C:\\Users\\laure\\.claude\\plans\\live-engine-h15-alignment-2026-04-26\\``.

Three contracts under test:

1. **Empty inputs**: when both live + V2 logs are empty, the script
   produces a "no trade overlap" report without raising.
2. **Synthetic match**: 1 live trade + 1 V2 trade at the SAME timestamp
   produce a single-row diff parquet with the expected per-column
   delta.
3. **Disjoint timestamps**: live and V2 logs that don't overlap on
   ``started_at`` produce a "no trade overlap" report.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "h15_alignment_harness.py"
spec = importlib.util.spec_from_file_location("_harness", SCRIPT_PATH)
_harness = importlib.util.module_from_spec(spec)
sys.modules["_harness"] = _harness
spec.loader.exec_module(_harness)


def test_stage_a_empty_inputs_returns_empty_diff():
    diff = _harness._stage_a_diff(pd.DataFrame(), pd.DataFrame())
    assert diff.empty


def test_summarise_empty_no_buckets():
    summary = _harness._summarise(pd.DataFrame())
    assert summary["n_trades_total"] == 0
    assert summary["buckets"] == {}


def test_format_report_empty(tmp_path):
    out = tmp_path / "report.md"
    _harness._format_report({"n_trades_total": 0, "buckets": {}}, out)
    assert out.exists()
    md = out.read_text(encoding="utf-8")
    assert "No trade overlap" in md or "no trade overlap" in md.lower()


def test_stage_a_single_match():
    """One live + one V2 trade at same timestamp → 1-row diff."""
    ts = pd.Timestamp("2026-04-15 12:00:00", tz="UTC")
    live = pd.DataFrame([{
        "started_at": ts.isoformat(),
        "signal": 1,
        "n_contracts_btc": 0.3,
        "y_pred_pct": 72.0,
        "iv_used_pct": 60.0,
        "edge_vol_pts": 12.0,
        "kelly_mult": 1.0,
        "entry_call_price": 0.05,
        "entry_put_price": 0.05,
        "exit_call_price": 0.06,
        "exit_put_price": 0.04,
        "realized_pnl_usd": 100.0,
    }])
    v2 = pd.DataFrame(
        [{
            "signal": 1,
            "size_btc": 0.4,
            "rv_pred_pct": 70.0,
            "iv_used_pct": 60.0,
            "edge_vp": 10.0,
            "size_mult": 1.0,
            "entry_call_price": 0.045,
            "entry_put_price": 0.05,
            "exit_call_price": 0.055,
            "exit_put_price": 0.04,
            "net_pnl_nc": 80.0,
        }],
        index=[ts],
    )

    diff = _harness._stage_a_diff(live, v2)
    assert len(diff) == 1
    assert diff.iloc[0]["d_signal"] == 0  # both fired LONG.
    # Live size 0.3 - V2 size 0.4 = -0.1.
    assert diff.iloc[0]["d_n_contracts_btc"] == pytest.approx(-0.1)
    # Live PnL 100 - V2 80 = +20.
    assert diff.iloc[0]["d_realized_pnl_usd"] == pytest.approx(20.0)


def test_stage_a_disjoint_timestamps_returns_empty():
    """Live trades all at one window, V2 trades at another → 0 overlap."""
    live = pd.DataFrame([{
        "started_at": "2026-04-15 12:00:00+00:00",
        "signal": 1,
        "n_contracts_btc": 0.3,
        "y_pred_pct": 72.0,
        "iv_used_pct": 60.0,
        "edge_vol_pts": 12.0,
        "kelly_mult": 1.0,
        "entry_call_price": 0.05,
        "entry_put_price": 0.05,
        "exit_call_price": 0.06,
        "exit_put_price": 0.04,
        "realized_pnl_usd": 100.0,
    }])
    v2 = pd.DataFrame(
        [{
            "signal": 1,
            "size_btc": 0.4,
            "rv_pred_pct": 70.0,
        }],
        index=[pd.Timestamp("2026-05-15 12:00:00", tz="UTC")],
    )

    diff = _harness._stage_a_diff(live, v2)
    assert diff.empty


def test_summarise_populates_buckets_on_real_diff():
    """Sanity check on the bucket structure."""
    ts = pd.Timestamp("2026-04-15 12:00:00", tz="UTC")
    diff = pd.DataFrame(
        [{
            "d_signal": 0.0,
            "d_n_contracts_btc": -0.1,
            "d_entry_call_price": 0.005,
            "d_entry_put_price": 0.0,
            "d_exit_call_price": 0.005,
            "d_exit_put_price": 0.0,
            "d_realized_pnl_usd": 20.0,
        }],
        index=[ts],
    )
    summary = _harness._summarise(diff)
    assert summary["n_trades_total"] == 1
    assert "config_drift_signal_mismatch" in summary["buckets"]
    assert "pnl_med_abs_diff_usd" in summary["buckets"]
