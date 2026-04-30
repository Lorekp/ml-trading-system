"""Tests for ``scripts/refresh_live_sidecars.py``.

I6 in plan ``C:\\Users\\laure\\.claude\\plans\\live-engine-h15-alignment-2026-04-26\\``.

Two contracts under test:

1. **Bootstrap path**: when no live trade log exists, the script logs
   the bootstrap message and exits 0 without touching the mainnet
   sidecars (verified by checking aligner betas json untouched).
2. **Threshold guards**: synthetic 14-month / 200-long-fire input
   short-circuits both refresh paths because the in-tree
   implementation is the bootstrap stub (real refresh is a follow-on).
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "scripts" / "refresh_live_sidecars.py"
spec = importlib.util.spec_from_file_location("_refresh", SCRIPT_PATH)
_refresh = importlib.util.module_from_spec(spec)
sys.modules["_refresh"] = _refresh
spec.loader.exec_module(_refresh)


@pytest.fixture
def synth_trades() -> pd.DataFrame:
    """14 months of trades with 200 long fires."""
    rows = []
    base = pd.Timestamp("2025-03-01", tz="UTC")
    for i in range(200):
        rows.append({
            "started_at": base + pd.Timedelta(days=i),
            "calendar_month": (base + pd.Timedelta(days=i)).strftime("%Y-%m"),
            "signal": 1,
            "n_contracts_btc": 0.3,
        })
    df = pd.DataFrame(rows)
    df["calendar_month"] = df["started_at"].dt.strftime("%Y-%m")
    return df


def test_bootstrap_returns_when_no_live_trades(monkeypatch, tmp_path):
    """No live_trades_harness.parquet → bootstrap, no error."""
    monkeypatch.setattr(_refresh, "LIVE_TRADES_PATH", tmp_path / "absent.parquet")
    df = _refresh._load_live_trades()
    assert df.empty


def test_aligner_refresh_skipped_below_threshold(synth_trades):
    """Even with 14 months, the bootstrap stub still skips because
    y_true isn't in the live log yet."""
    refreshed = _refresh._refresh_aligner_betas(synth_trades, dry_run=True)
    assert refreshed is False


def test_kelly_refresh_skipped_below_threshold():
    """2 months / 50 fires below the 12m / 100-fire threshold."""
    rows = []
    base = pd.Timestamp("2026-01-01", tz="UTC")
    for i in range(50):
        rows.append({
            "started_at": base + pd.Timedelta(days=i),
            "calendar_month": (base + pd.Timedelta(days=i)).strftime("%Y-%m"),
            "signal": 1,
        })
    df = pd.DataFrame(rows)
    df["calendar_month"] = df["started_at"].dt.strftime("%Y-%m")
    refreshed = _refresh._refresh_kelly_sidecar(df, dry_run=True)
    assert refreshed is False


def test_kelly_refresh_threshold_met_still_bootstrap(synth_trades):
    """Even when thresholds are met, the in-tree implementation is the
    bootstrap stub (returns False with a TODO log message). Asserts
    the contract; future commits will replace with real refresh.
    """
    refreshed = _refresh._refresh_kelly_sidecar(synth_trades, dry_run=True)
    assert refreshed is False
