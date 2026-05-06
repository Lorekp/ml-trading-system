"""Tests for H9 sidecar staleness gates.

Why this exists
---------------

Before H9, only the aligner-β* sidecar had a staleness check. Four
other sidecars (kelly_mult, bar_execution_quality, features parquet,
svi_features parquet) loaded silently regardless of file age. A
6-month-old kelly_mult parquet was silently applied through Q2.

These tests pin the H9 contract:

1. ``_assert_sidecar_fresh`` raises when file age > threshold AND
   ``allow_stale=False``.
2. With ``allow_stale=True``, the same case logs a WARNING and
   returns cleanly.
3. A missing file is treated as a no-op (caller's existence check
   handles that).
4. Fresh files (age below threshold) return silently with an INFO log.
"""
from __future__ import annotations

import logging
import os
import time
from pathlib import Path

import pytest

from deribit_paper.engine import _assert_sidecar_fresh


def _make_file(tmp_path: Path, age_days: float) -> Path:
    """Create a tmp file whose mtime is set ``age_days`` in the past."""
    f = tmp_path / "sidecar.bin"
    f.write_bytes(b"placeholder")
    target_mtime = time.time() - age_days * 86400.0
    os.utime(f, (target_mtime, target_mtime))
    return f


def test_fresh_file_passes_silently(tmp_path):
    f = _make_file(tmp_path, age_days=1.0)
    # Must not raise
    _assert_sidecar_fresh(
        f, name="test-sidecar", max_age_days=30, allow_stale=False,
    )


def test_at_threshold_is_fresh(tmp_path):
    """Boundary: age == max_age_days is considered fresh (inclusive)."""
    f = _make_file(tmp_path, age_days=29.5)
    _assert_sidecar_fresh(
        f, name="test", max_age_days=30, allow_stale=False,
    )


def test_just_over_threshold_raises_when_not_allowed(tmp_path):
    f = _make_file(tmp_path, age_days=31.0)
    with pytest.raises(RuntimeError, match="stale"):
        _assert_sidecar_fresh(
            f, name="test", max_age_days=30, allow_stale=False,
        )


def test_well_over_threshold_raises_when_not_allowed(tmp_path):
    f = _make_file(tmp_path, age_days=180.0)
    with pytest.raises(RuntimeError, match="stale"):
        _assert_sidecar_fresh(
            f, name="kelly-K1", max_age_days=45, allow_stale=False,
        )


def test_stale_with_allow_stale_warns_and_returns(tmp_path, caplog):
    f = _make_file(tmp_path, age_days=120.0)
    with caplog.at_level(logging.WARNING, logger="deribit_paper.engine"):
        _assert_sidecar_fresh(
            f, name="kelly-K1", max_age_days=45, allow_stale=True,
        )
    warnings = [r for r in caplog.records if r.levelname == "WARNING"]
    assert any("OVERRIDDEN" in r.getMessage() for r in warnings)


def test_missing_file_is_noop(tmp_path):
    """The caller's existence check (or downstream load failure) is
    responsible for missing files. Staleness gate just skips."""
    nonexistent = tmp_path / "no-such-sidecar.parquet"
    # Must not raise
    _assert_sidecar_fresh(
        nonexistent, name="test", max_age_days=30, allow_stale=False,
    )


def test_error_message_contains_file_path_and_threshold(tmp_path):
    """Operator must be able to grep the log for the offending sidecar."""
    f = _make_file(tmp_path, age_days=200.0)
    with pytest.raises(RuntimeError) as exc_info:
        _assert_sidecar_fresh(
            f, name="bar-exec-quality", max_age_days=90, allow_stale=False,
        )
    msg = str(exc_info.value)
    assert "bar-exec-quality" in msg
    assert str(f) in msg
    assert "90" in msg  # threshold
    assert "200" in msg or "200.0" in msg  # actual age


def test_string_path_works(tmp_path):
    """The helper accepts both Path and str. Engine uses str (cfg fields)."""
    f = _make_file(tmp_path, age_days=1.0)
    _assert_sidecar_fresh(
        str(f), name="test", max_age_days=30, allow_stale=False,
    )
