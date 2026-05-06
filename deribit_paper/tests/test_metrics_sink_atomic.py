"""Tests for atomic parquet writes in MetricsSink (Tier 2 bonus, 2026-05-06).

Why this exists
---------------

Pre-Tier-2, ``MetricsSink.flush`` wrote directly to the target parquet
path. A crash mid-write would leave a corrupt file (or zero bytes on
some filesystems) — the next flush couldn't read it back, dropping all
prior rows.

Tier 2 bonus added ``_atomic_write_parquet`` which writes to ``.tmp``
and uses ``os.replace`` for the swap. ``HarnessWriter`` was already
doing this; ``MetricsSink`` was the gap.

Tests:
1. Normal flush still produces a readable parquet at the target path.
2. ``.tmp`` file does NOT linger after a successful write.
3. The atomic-write helper itself is correct (round-trip preserves data).
"""
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest

from deribit_paper.monitoring.metrics_sink import (
    MetricsSink,
    _atomic_write_parquet,
)


def test_atomic_write_helper_round_trips(tmp_path):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    path = tmp_path / "out.parquet"
    _atomic_write_parquet(df, path)

    assert path.exists()
    loaded = pd.read_parquet(path)
    pd.testing.assert_frame_equal(loaded, df)


def test_atomic_write_helper_does_not_leave_tmp(tmp_path):
    df = pd.DataFrame({"a": [1]})
    path = tmp_path / "out.parquet"
    _atomic_write_parquet(df, path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    assert not tmp.exists(), "tmp file should be renamed away on success"


def test_atomic_write_helper_overwrites_existing(tmp_path):
    path = tmp_path / "out.parquet"
    _atomic_write_parquet(pd.DataFrame({"a": [1]}), path)
    _atomic_write_parquet(pd.DataFrame({"a": [99]}), path)
    loaded = pd.read_parquet(path)
    assert loaded["a"].iloc[0] == 99


def test_metrics_sink_flush_uses_atomic_write(tmp_path, monkeypatch):
    """MetricsSink.flush must route through _atomic_write_parquet on
    both code paths (file exists / new file)."""
    path = tmp_path / "metrics.parquet"
    sink = MetricsSink(path, flush_every=100)

    calls = []
    real_atomic = _atomic_write_parquet

    def spy(df, p):
        calls.append((p,))
        real_atomic(df, p)

    # Patch the module-level helper that flush uses
    import deribit_paper.monitoring.metrics_sink as mod
    monkeypatch.setattr(mod, "_atomic_write_parquet", spy)

    # First flush: new file path
    sink.record({"a": 1})
    sink.flush()
    assert len(calls) == 1
    assert path.exists()

    # Second flush: existing file path
    sink.record({"a": 2})
    sink.flush()
    assert len(calls) == 2

    # Final state
    df = pd.read_parquet(path)
    assert len(df) == 2


def test_metrics_sink_flush_resilience_to_existing_corrupt_file(tmp_path):
    """If the existing parquet is corrupt, flush logs + overwrites
    (existing behaviour preserved). The overwrite is still atomic."""
    path = tmp_path / "corrupt.parquet"
    path.write_bytes(b"not a parquet file")

    sink = MetricsSink(path, flush_every=100)
    sink.record({"a": 1})
    sink.flush()

    # File now contains valid parquet (the fallback overwrite path)
    df = pd.read_parquet(path)
    assert len(df) == 1
    # No tmp file lingering
    assert not (tmp_path / "corrupt.parquet.tmp").exists()
