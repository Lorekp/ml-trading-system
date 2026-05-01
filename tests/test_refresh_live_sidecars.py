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


# ── E.2.9: slip refresh from observed live fills ─────────────────────────


def _make_slip_trades(
    *, n: int = 60, slip_pct: float = 0.05, spot: float = 60_000.0,
) -> pd.DataFrame:
    rows = []
    base = pd.Timestamp("2026-04-01", tz="UTC")
    for i in range(n):
        rows.append({
            "started_at": base + pd.Timedelta(minutes=15 * i),
            "signal": 1,
            "taker_slippage_pct": slip_pct,
            "index_price_at_entry": spot,
        })
    df = pd.DataFrame(rows)
    return df


def test_slip_refresh_below_threshold_skips():
    """< MIN_TRADES_FOR_SLIP (30) rows → skip without write."""
    trades = _make_slip_trades(n=10)
    refreshed = _refresh._refresh_slip_sidecar(trades, dry_run=True)
    assert refreshed is False


def test_slip_refresh_above_threshold_writes_dry_run():
    """≥ MIN_TRADES_FOR_SLIP rows + dry_run → True (would have written)."""
    trades = _make_slip_trades(n=60)
    refreshed = _refresh._refresh_slip_sidecar(trades, dry_run=True)
    assert refreshed is True


def test_slip_refresh_actual_write(tmp_path, monkeypatch):
    """Above-threshold input + dry_run=False writes the parquet."""
    out = tmp_path / "bar_execution_quality_live.parquet"
    monkeypatch.setattr(_refresh, "LIVE_SLIP_PARQUET", out)
    trades = _make_slip_trades(n=60, slip_pct=0.05, spot=60_000.0)

    refreshed = _refresh._refresh_slip_sidecar(trades, dry_run=False)
    assert refreshed is True
    assert out.exists()

    df = pd.read_parquet(out)
    # 60 trades at 15-min spacing → 60 unique bars; slip = 0.05% × 60K = $30/BTC.
    assert len(df) == 60
    assert "bar_taker_slip_usd" in df.columns
    assert df["bar_taker_slip_usd"].median() == pytest.approx(30.0, abs=0.01)


def test_load_live_trades_corrupt_parquet_returns_empty(tmp_path, monkeypatch):
    """E.2.13: a corrupt parquet → empty DF + log, not a crash."""
    bad = tmp_path / "trades.parquet"
    bad.write_bytes(b"not a valid parquet file")
    monkeypatch.setattr(_refresh, "LIVE_TRADES_PATH", bad)
    df = _refresh._load_live_trades()
    assert df.empty


def test_load_live_trades_missing_started_at_returns_empty(tmp_path, monkeypatch):
    """E.2.13: parquet exists but lacks the started_at column → empty DF."""
    bad = tmp_path / "trades.parquet"
    pd.DataFrame({"foo": [1, 2]}).to_parquet(bad)
    monkeypatch.setattr(_refresh, "LIVE_TRADES_PATH", bad)
    df = _refresh._load_live_trades()
    assert df.empty


def test_slip_refresh_filters_extreme_values(tmp_path, monkeypatch):
    """A row with >1% slip is filtered out; the remaining good rows must
    still meet the threshold for the write to proceed."""
    out = tmp_path / "bar_execution_quality_live.parquet"
    monkeypatch.setattr(_refresh, "LIVE_SLIP_PARQUET", out)

    trades = _make_slip_trades(n=60, slip_pct=0.05, spot=60_000.0)
    # Inject 2 outliers: 5% slip × 60K = $3000/BTC > 600 cap.
    trades.loc[trades.index[:2], "taker_slippage_pct"] = 5.0

    refreshed = _refresh._refresh_slip_sidecar(trades, dry_run=False)
    assert refreshed is True
    df = pd.read_parquet(out)
    assert len(df) == 58           # 60 - 2 outliers
    assert df["bar_taker_slip_usd"].max() < _refresh.SLIP_USD_BTC_MAX
