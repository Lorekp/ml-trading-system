"""Cycle-35 (2026-05-13): tests for the feature-trace divergence comparator.

Pins:
1. Identical synthetic traces produce 0 flagged bars at any checkpoint.
2. A C4-feature mismatch in one bar causes that bar to be flagged at C4.
3. Tolerance ladder: a per-field deviation just inside rtol passes,
   just outside fails.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import pytest


def _hist_count(out: str, key: str) -> int:
    """Extract the count for histogram line ``key`` from the diff output."""
    m = re.search(rf"\s{key}\s+(\d+)\s*\(", out)
    return int(m.group(1)) if m else -1


SCRIPT = Path("scripts/feature_trace_diff.py")


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _identical_bar(ts: str) -> dict:
    return {
        "bar_close_ts": ts,
        "index_price_close": 63000.0,
        "tsrv": 0.000123,
        "jump_fraction": 0.05,
        "vol_ann": 0.42,
        "vol_ann_continuous": 0.41,
        "svi_a": 0.0, "svi_b": 0.21, "svi_rho": -0.3, "svi_m": 0.0, "svi_sigma": 0.5,
        "svi_T_years": 0.025, "svi_F": 63500.0,
        "atm_strike": 63500.0,
        "svi_iv_at_atm_pct": 42.1,
        "iv_used_pct": 42.1,
        "iv_source": "svi",
        "feature_vector": {"tv_0": 0.10, "tv_1": 0.20, "jump_fraction": 0.05},
        "transvol_embedding": [0.10, 0.20] + [0.0] * 14,
        "transvol_status": "computed",
        "y_pred_raw": -0.85,
        "aligner_month_key": "2026-05",
        "beta_star": 0.037,
        "y_pred_aligned": -0.813,
        "rv_pred_pct": 44.36,
        "edge_vol_pts": 2.26,
        "base_spread_iv": 1.5,
        "spread_jump_k": 3.0,
        "half_spread": 0.87,
        "effective_long_thresh": 30.87,
        "effective_short_thresh": 25.87,
        "signal_raw": 0,
        "signal_final": 0,
    }


def _run_diff(live: Path, offline: Path) -> tuple[int, str]:
    proc = subprocess.run(
        [sys.executable, str(SCRIPT), "--live", str(live), "--offline", str(offline)],
        capture_output=True, text=True, encoding="utf-8",
    )
    return proc.returncode, proc.stdout + proc.stderr


def test_identical_traces_are_clean(tmp_path: Path) -> None:
    """Identical live and offline rows → all bars 'clean'."""
    live = tmp_path / "live.jsonl"
    off = tmp_path / "offline.jsonl"
    rows = [_identical_bar("2026-05-13T14:30:00Z"),
            _identical_bar("2026-05-13T14:45:00Z")]
    _write_jsonl(live, rows)
    _write_jsonl(off, rows)
    rc, out = _run_diff(live, off)
    assert rc == 0, out
    assert _hist_count(out, "clean") == 2, out


def test_c4_feature_mismatch_flagged_at_c4(tmp_path: Path) -> None:
    """Perturbing one feature in the live row → bar flagged at C4."""
    live_row = _identical_bar("2026-05-13T14:30:00Z")
    live_row["feature_vector"] = dict(live_row["feature_vector"])
    live_row["feature_vector"]["jump_fraction"] = 0.99  # huge drift

    off_row = _identical_bar("2026-05-13T14:30:00Z")

    live = tmp_path / "live.jsonl"
    off = tmp_path / "offline.jsonl"
    _write_jsonl(live, [live_row])
    _write_jsonl(off, [off_row])
    rc, out = _run_diff(live, off)
    assert rc == 0, out
    assert _hist_count(out, "c4") == 1, out
    assert _hist_count(out, "clean") == 0, out


def test_within_tolerance_is_clean(tmp_path: Path) -> None:
    """A tiny deviation within rtol passes (still 'clean')."""
    live_row = _identical_bar("2026-05-13T14:30:00Z")
    off_row = _identical_bar("2026-05-13T14:30:00Z")
    # Bump rv_pred_pct by 1e-5 — well inside the c7 rtol=1e-4.
    live_row["rv_pred_pct"] = 44.36
    off_row["rv_pred_pct"] = 44.36 + 1e-5

    live = tmp_path / "live.jsonl"
    off = tmp_path / "offline.jsonl"
    _write_jsonl(live, [live_row])
    _write_jsonl(off, [off_row])
    rc, out = _run_diff(live, off)
    assert rc == 0, out
    assert _hist_count(out, "clean") == 1, out


def test_iv_source_mismatch_flagged_at_c3(tmp_path: Path) -> None:
    """``iv_source`` is a string field — exact match required at C3."""
    live_row = _identical_bar("2026-05-13T14:30:00Z")
    off_row = _identical_bar("2026-05-13T14:30:00Z")
    live_row["iv_source"] = "svi"
    off_row["iv_source"] = "dvol_close"

    live = tmp_path / "live.jsonl"
    off = tmp_path / "offline.jsonl"
    _write_jsonl(live, [live_row])
    _write_jsonl(off, [off_row])
    rc, out = _run_diff(live, off)
    assert rc == 0, out
    assert _hist_count(out, "c3") == 1, out
