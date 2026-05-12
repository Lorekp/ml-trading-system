"""Cycle-28a (2026-05-13) regression tests for SVI recal JSONL audit log.

The rolling ``svi_cache.parquet`` evicts old rows as buckets get
overwritten by subsequent recals. For post-hoc parity audit we want
every SVI fit preserved. Cycle-28a appends every recal's per-expiry
param rows to ``deribit_paper/logs/svi_recals.jsonl``.

These tests pin:
- Each recal emits at least one JSONL line per ``params_df`` row.
- Each line is valid JSON with the documented schema.
- Multiple recals append (not overwrite).
- Audit emission is best-effort: a disk error must NOT raise into the
  recal hot path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from deribit_paper.features.svi_live import LiveSVICalibrator, SVICacheMeta


def _synthetic_trades_for_audit(n_per_expiry: int = 40, n_expiries: int = 2):
    """Build trades that successfully calibrate via calibrate_surface."""
    rng = np.random.default_rng(0)
    rows = []
    base_ts = pd.Timestamp("2026-05-12 11:00:00", tz="UTC")
    strikes = np.linspace(55_000, 65_000, 7)
    for exp_idx in range(n_expiries):
        expiry = base_ts + pd.Timedelta(days=3 + exp_idx * 4)
        for i in range(n_per_expiry):
            ts = base_ts + pd.Timedelta(minutes=i)
            strike = float(rng.choice(strikes))
            rows.append({
                "timestamp": ts,
                "instrument_name": f"BTC-EXP{exp_idx}-{int(strike)}-C",
                "strike": strike,
                "option_type": "C" if rng.random() > 0.5 else "P",
                "expiry_dt": expiry,
                "iv": 70.0 + rng.normal(0, 3.0),
                "price": 0.05,
                "amount": 1.0,
                "direction": 1 if rng.random() > 0.5 else -1,
                "index_price": 60_000.0,
                "mark_price": 0.05,
            })
    return pd.DataFrame(rows)


def test_audit_log_emitted_after_recalibrate(tmp_path):
    """Every successful recal must append at least one JSONL line for
    each row in ``_params_df``."""
    audit_path = tmp_path / "svi_recals.jsonl"
    cal = LiveSVICalibrator(cache_path=tmp_path / "svi_cache.parquet")
    cal.recalibrate(_synthetic_trades_for_audit(), now=pd.Timestamp("2026-05-12 12:00:00", tz="UTC"))
    if not cal.has_params:
        pytest.skip("Calibrate fell back; audit log requires non-empty params")
    cal._emit_recal_audit(audit_path=audit_path)
    assert audit_path.exists(), "audit JSONL must be created"
    lines = audit_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == len(cal._params_df), (
        f"audit lines={len(lines)} != params_df rows={len(cal._params_df)}"
    )


def test_audit_log_schema(tmp_path):
    """Each line must be valid JSON with the documented keys."""
    audit_path = tmp_path / "svi_recals.jsonl"
    cal = LiveSVICalibrator(cache_path=tmp_path / "svi_cache.parquet")
    cal.recalibrate(_synthetic_trades_for_audit(), now=pd.Timestamp("2026-05-12 12:00:00", tz="UTC"))
    if not cal.has_params:
        pytest.skip("Calibrate fell back; audit log requires non-empty params")
    cal._emit_recal_audit(audit_path=audit_path)
    for line in audit_path.read_text(encoding="utf-8").strip().split("\n"):
        rec = json.loads(line)
        # Mandatory keys
        assert "recalibrated_at" in rec
        assert "bucket_ts" in rec
        # SVI params (at least core 5)
        for param in ("a", "b", "rho", "m", "sigma"):
            assert param in rec, f"missing {param} in audit record"
            assert rec[param] is None or isinstance(rec[param], (int, float))


def test_audit_log_appends_across_recals(tmp_path):
    """A second recal must APPEND to the audit log, not overwrite."""
    audit_path = tmp_path / "svi_recals.jsonl"
    cal = LiveSVICalibrator(cache_path=tmp_path / "svi_cache.parquet")
    trades = _synthetic_trades_for_audit()

    cal.recalibrate(trades, now=pd.Timestamp("2026-05-12 12:00:00", tz="UTC"))
    if not cal.has_params:
        pytest.skip("Calibrate fell back")
    cal._emit_recal_audit(audit_path=audit_path)
    first_count = len(audit_path.read_text(encoding="utf-8").strip().split("\n"))

    cal.recalibrate(trades, now=pd.Timestamp("2026-05-12 13:00:00", tz="UTC"))
    cal._emit_recal_audit(audit_path=audit_path)
    second_count = len(audit_path.read_text(encoding="utf-8").strip().split("\n"))

    assert second_count == 2 * first_count, (
        f"second recal must append (expected {2 * first_count}, got {second_count})"
    )


def test_audit_log_is_best_effort_no_raise_on_disk_error(tmp_path, monkeypatch):
    """A disk error during audit emission MUST be logged but NOT
    propagate into the recal hot path. Pin this property explicitly:
    if the file open raises, the method returns silently."""
    cal = LiveSVICalibrator(cache_path=tmp_path / "svi_cache.parquet")
    cal.recalibrate(_synthetic_trades_for_audit(), now=pd.Timestamp("2026-05-12 12:00:00", tz="UTC"))
    if not cal.has_params:
        pytest.skip("Calibrate fell back")

    # Patch Path.open to always raise.
    import builtins
    real_open = builtins.open

    def boom(*args, **kwargs):
        if "svi_recals" in str(args[0]):
            raise OSError("disk full")
        return real_open(*args, **kwargs)

    monkeypatch.setattr(builtins, "open", boom)
    # Should NOT raise — the audit is best-effort.
    cal._emit_recal_audit(audit_path=tmp_path / "svi_recals.jsonl")


def test_audit_log_skipped_when_no_params(tmp_path):
    """An empty params_df must skip emission silently (no half-written
    file, no crash)."""
    audit_path = tmp_path / "svi_recals.jsonl"
    cal = LiveSVICalibrator(cache_path=tmp_path / "svi_cache.parquet")
    # _params_df is None at this point.
    cal._emit_recal_audit(audit_path=audit_path)
    assert not audit_path.exists()


def test_audit_log_called_from_recalibrate_path(tmp_path):
    """Integration: the production ``recalibrate`` method must invoke
    ``_emit_recal_audit`` so audit emission is automatic, not
    operator-managed. Pre-cycle-28 calling recal did NOT emit audit.
    """
    audit_path = tmp_path / "logs" / "svi_recals.jsonl"
    cal = LiveSVICalibrator(cache_path=tmp_path / "svi_cache.parquet")

    # Patch _emit_recal_audit to assert it's reached.
    state = {"called": False}
    real_emit = cal._emit_recal_audit

    def spy(*args, **kwargs):
        state["called"] = True
        return real_emit(*args, **kwargs)

    cal._emit_recal_audit = spy  # type: ignore[method-assign]
    cal.recalibrate(_synthetic_trades_for_audit(), now=pd.Timestamp("2026-05-12 12:00:00", tz="UTC"))
    assert state["called"] is True, (
        "recalibrate() must call _emit_recal_audit() so audit emission "
        "is automatic — otherwise cycle-28 forensics rely on operator discipline"
    )
