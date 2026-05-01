"""Tests for ``scripts/refresh_aligner_monthly.py``.

The orchestrator is mostly subprocess plumbing (delegates to existing
runners). The critical-path logic that needs locked-in regression
coverage is:

1. **Phase 1 data-availability gates** — must refuse to proceed when
   any of the three data sources (Binance aggTrades, Deribit option
   trades, DVOL) is incomplete.
2. **Phase 6 sanity gate** — must refuse to ship a JSON whose target-
   month β* is out of the safety band, OR that drops historical months.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "refresh_aligner_monthly.py"

spec = importlib.util.spec_from_file_location("_refresh_monthly", SCRIPT)
module = importlib.util.module_from_spec(spec)
sys.modules["_refresh_monthly"] = module
spec.loader.exec_module(module)


# ── Phase 1 gates ─────────────────────────────────────────────────────────


def test_aggtrades_check_local_present(tmp_path, monkeypatch):
    """Local monthly parquet → OK without network probe."""
    fake_dir = tmp_path / "agg"
    fake_dir.mkdir()
    (fake_dir / "2026-04.parquet").write_bytes(b"x" * 1000)
    monkeypatch.setattr(module, "BTC_AGGTRADES_DIR", fake_dir)
    ok, msg = module._check_aggtrades("2026-04")
    assert ok and "local parquet present" in msg


def test_aggtrades_check_binance_404(tmp_path, monkeypatch):
    """No local + Binance 404 → fail with informative message."""
    monkeypatch.setattr(module, "BTC_AGGTRADES_DIR", tmp_path / "nonexistent")
    fake_resp = MagicMock(status_code=404)
    with patch("requests.head", return_value=fake_resp):
        ok, msg = module._check_aggtrades("2026-04")
    assert not ok and "404" in msg and "not yet published" in msg


def test_aggtrades_check_binance_published(tmp_path, monkeypatch):
    """No local + Binance 200 → OK, will download next phase."""
    monkeypatch.setattr(module, "BTC_AGGTRADES_DIR", tmp_path / "nonexistent")
    fake_resp = MagicMock(status_code=200)
    with patch("requests.head", return_value=fake_resp):
        ok, msg = module._check_aggtrades("2026-04")
    assert ok and "available" in msg


def test_option_trades_below_threshold(tmp_path, monkeypatch):
    """6/30 days = 20% coverage → fail (need ≥80%)."""
    fake_dir = tmp_path / "opt"
    fake_dir.mkdir()
    for d in range(1, 7):
        (fake_dir / f"2026-04-{d:02d}.parquet").write_bytes(b"x")
    monkeypatch.setattr(module, "BTC_OPTIONS_DIR", fake_dir)
    ok, msg = module._check_option_trades("2026-04")
    assert not ok and "20%" in msg


def test_option_trades_meets_threshold(tmp_path, monkeypatch):
    """24/30 days = 80% coverage → OK."""
    fake_dir = tmp_path / "opt"
    fake_dir.mkdir()
    for d in range(1, 25):
        (fake_dir / f"2026-04-{d:02d}.parquet").write_bytes(b"x")
    monkeypatch.setattr(module, "BTC_OPTIONS_DIR", fake_dir)
    ok, msg = module._check_option_trades("2026-04")
    assert ok and "80%" in msg


# ── Phase 6 sanity gate ───────────────────────────────────────────────────


def test_sanity_gate_target_in_band(tmp_path, monkeypatch):
    """Target month present + in band → OK."""
    new = tmp_path / "new.json"
    new.write_text(json.dumps({
        "2026-03": 0.05, "2026-04": 0.06,
    }))
    monkeypatch.setattr(module, "BETAS_JSON", tmp_path / "live.json")
    assert module.phase6_sanity_gate(new, "2026-04") is True


def test_sanity_gate_rejects_out_of_band(tmp_path, monkeypatch):
    """Target month β* outside [-0.5, +0.5] → reject."""
    new = tmp_path / "new.json"
    new.write_text(json.dumps({"2026-04": 0.99}))   # out of band
    monkeypatch.setattr(module, "BETAS_JSON", tmp_path / "live.json")
    assert module.phase6_sanity_gate(new, "2026-04") is False


def test_sanity_gate_rejects_target_missing(tmp_path, monkeypatch):
    """Target month not in JSON keys → reject."""
    new = tmp_path / "new.json"
    new.write_text(json.dumps({"2026-03": 0.05}))      # no 2026-04
    monkeypatch.setattr(module, "BETAS_JSON", tmp_path / "live.json")
    assert module.phase6_sanity_gate(new, "2026-04") is False


def test_sanity_gate_rejects_dropped_history(tmp_path, monkeypatch):
    """New JSON drops a month present in old → reject (no silent
    coverage regression)."""
    old = tmp_path / "live.json"
    old.write_text(json.dumps({
        "2026-01": 0.04, "2026-02": 0.03, "2026-03": 0.05,
    }))
    new = tmp_path / "new.json"
    new.write_text(json.dumps({
        "2026-02": 0.03, "2026-03": 0.05, "2026-04": 0.06,
        # 2026-01 dropped
    }))
    monkeypatch.setattr(module, "BETAS_JSON", old)
    assert module.phase6_sanity_gate(new, "2026-04") is False


# ── Auto-pick ─────────────────────────────────────────────────────────────


def test_autopick_returns_prior_month():
    """``_autopick_month`` returns the most recent complete month."""
    result = module._autopick_month()
    # Must be a valid YYYY-MM string strictly before the current month.
    import re
    assert re.match(r"^\d{4}-(0[1-9]|1[0-2])$", result)
    today = pd.Timestamp.now(tz="UTC")
    cur_period = pd.Period(today, freq="M")
    picked = pd.Period(result, freq="M")
    assert picked < cur_period
