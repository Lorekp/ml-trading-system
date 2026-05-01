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
    """One live + one V2 trade at same timestamp → 1-row diff.

    Uses the REAL V2 schema (``signal``, ``N_contracts``, ``rv_pred_pct``,
    ``iv_used``, ``edge``, ``net_pnl_nc``) — no leg-price columns.
    """
    ts = pd.Timestamp("2026-04-15 12:00:00", tz="UTC")
    live = pd.DataFrame([{
        "started_at": ts.isoformat(),
        "signal": 1,
        "n_contracts_btc": 0.3,
        "y_pred_pct": 72.0,
        "iv_used_pct": 60.0,
        "edge_vol_pts": 12.0,
        "realized_pnl_usd": 100.0,
    }])
    v2 = pd.DataFrame(
        [{
            "signal": 1,
            "N_contracts": 0.4,
            "rv_pred_pct": 70.0,
            "iv_used": 60.0,
            "edge": 10.0,
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
        "realized_pnl_usd": 100.0,
    }])
    v2 = pd.DataFrame(
        [{
            "signal": 1,
            "N_contracts": 0.4,
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
            "d_y_pred_pct": 2.0,
            "d_iv_used_pct": 0.0,
            "d_edge_vol_pts": 2.0,
            "d_realized_pnl_usd": 20.0,
        }],
        index=[ts],
    )
    summary = _harness._summarise(diff)
    assert summary["n_trades_total"] == 1
    assert "config_drift_signal_mismatch" in summary["buckets"]
    assert "pnl_med_abs_diff_usd" in summary["buckets"]


# ─── E.1.2: real-schema sanity + live-fired-V2-flat bucket ──────────────


def test_v2_column_names_in_real_trade_log():
    """The right-hand side of ``_NUMERIC_COLS_LIVE_TO_V2`` must reference
    columns that actually exist in the V2 trade log. This is the test
    that would have caught B2 (the previous map referenced ``size_btc``,
    ``edge_vp``, ``size_mult``, etc. — none of which are in the parquet)."""
    v2_path = REPO_ROOT / "vol_arb" / "output" / "BTC_v2_trade_log.parquet"
    if not v2_path.exists():
        pytest.skip(f"{v2_path} not on disk — skipping schema check")
    cols = set(pd.read_parquet(v2_path, columns=None).columns)
    for live_col, v2_col in _harness._NUMERIC_COLS_LIVE_TO_V2:
        assert v2_col in cols, (
            f"V2 column {v2_col!r} (mapped from live {live_col!r}) not in trade log."
        )


def test_live_fired_v2_flat_bucket_counted():
    """Bar where live fired but V2 was flat → in the diagnostic bucket,
    NOT in the diff (diff only inner-joins fired-on-both)."""
    ts1 = pd.Timestamp("2026-04-15 12:00:00", tz="UTC")
    ts2 = pd.Timestamp("2026-04-15 12:15:00", tz="UTC")
    live = pd.DataFrame([
        {"started_at": ts1.isoformat(), "signal": 1, "n_contracts_btc": 0.3,
         "y_pred_pct": 72.0, "iv_used_pct": 60.0, "edge_vol_pts": 12.0,
         "realized_pnl_usd": 100.0},
        {"started_at": ts2.isoformat(), "signal": 1, "n_contracts_btc": 0.3,
         "y_pred_pct": 65.0, "iv_used_pct": 60.0, "edge_vol_pts": 5.0,
         "realized_pnl_usd": 0.0},
    ])
    v2 = pd.DataFrame(
        [
            {"signal": 1, "N_contracts": 0.3, "rv_pred_pct": 72.0,
             "iv_used": 60.0, "edge": 12.0, "net_pnl_nc": 100.0},
            {"signal": 0, "N_contracts": 0.0, "rv_pred_pct": 65.0,
             "iv_used": 60.0, "edge": 5.0, "net_pnl_nc": 0.0},
        ],
        index=[ts1, ts2],
    )

    diff = _harness._stage_a_diff(live, v2)
    # Only ts1 ends up in the diff (V2 flat at ts2 excludes it).
    assert len(diff) == 1
    assert diff.attrs["live_fired_v2_flat_count"] == 1
    summary = _harness._summarise(diff)
    assert summary["live_fired_v2_flat_count"] == 1
