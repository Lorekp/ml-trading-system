"""Cycle-29 (2026-05-13) regression tests for the TransVol time-normalizer
key-name fix.

The bug: `deribit_paper/signal/model_loader.py` read `time_mean` /
`time_std` from the TransVol checkpoint. Cycle-25 (2026-05-11) had
already moved persistence to `time_normalizer_mean` /
`time_normalizer_std` (see `vol_model/run_walkforward_hybrid.py:277-278`),
so the live lookup always returned `np.zeros(0)`. The subsequent
`prewarm_from_parquet` then crashed at `Tn = (T - time_mean) / time_std`
with `operands could not be broadcast together with shapes (32,2) (0,)`,
and the engine fell back to FROZEN baseline `tv_*` features for the
entire session — a ~6vp `Δrv_pred` drift vs offline.

These tests pin:
1. The loader extracts the cycle-25 keys when present.
2. The loader returns `(0,)` arrays when keys are absent (legacy
   pre-cycle-25 checkpoint; intentional fallback path).
3. `prewarm_from_parquet` computes a legacy time normalizer from the
   prewarm parquet when given `(0,)` arrays — does NOT raise.
4. The source contains the cycle-25 key string (sentinel against a
   future refactor reverting to the broken names).
"""

from __future__ import annotations

import re
import warnings
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest


# ── Source-regex sentinel ─────────────────────────────────────────────


def test_model_loader_reads_cycle25_key_names():
    """Cycle-29 sentinel: model_loader.py must reference the cycle-25
    persistence keys (`time_normalizer_mean` / `time_normalizer_std`),
    NOT the broken pre-cycle-29 names (`time_mean` / `time_std`)."""
    src = Path(__file__).resolve().parents[1].joinpath(
        "signal", "model_loader.py",
    ).read_text()
    # Must reference the cycle-25 canonical keys at least once each
    # in the actual lookup (not just a comment).
    assert re.search(r'ckpt\.get\(["\']time_normalizer_mean["\']', src), (
        "model_loader.py must read 'time_normalizer_mean' from the checkpoint"
    )
    assert re.search(r'ckpt\.get\(["\']time_normalizer_std["\']', src), (
        "model_loader.py must read 'time_normalizer_std' from the checkpoint"
    )
    # Must NOT have an active lookup on the broken pre-cycle-29 names
    # (the names may appear in docstrings/comments — exclude those by
    # requiring the lookup form ckpt.get(...)).
    assert not re.search(r'ckpt\.get\(["\']time_mean["\']', src), (
        "model_loader.py must not call ckpt.get('time_mean'); "
        "use ckpt.get('time_normalizer_mean') instead"
    )
    assert not re.search(r'ckpt\.get\(["\']time_std["\']', src), (
        "model_loader.py must not call ckpt.get('time_std'); "
        "use ckpt.get('time_normalizer_std') instead"
    )


# ── prewarm_from_parquet legacy fallback ──────────────────────────────


def _build_synthetic_parquet(tmp_path: Path, n_rows: int = 50) -> Path:
    """Build a minimal parquet that prewarm_from_parquet can consume."""
    import pandas as pd

    rng = np.random.default_rng(0)
    idx = pd.date_range("2026-05-01", periods=n_rows, freq="15min", tz="UTC")
    df = pd.DataFrame({
        "duration_s": rng.uniform(60.0, 600.0, n_rows),
        "n_bars_in_window": rng.integers(100, 500, n_rows).astype(float),
    }, index=idx)
    # Add one feature column so the prewarm passes the feature_columns check.
    for i in range(3):
        df[f"feat_{i}"] = rng.normal(0.0, 1.0, n_rows)
    path = tmp_path / "synthetic_features.parquet"
    df.to_parquet(path)
    return path


def test_prewarm_legacy_fallback_with_empty_time_normalizer(tmp_path):
    """Cycle-29 core regression: prewarm_from_parquet MUST NOT raise
    when handed `(0,)` time_mean/time_std (legacy pre-cycle-25
    checkpoint). It must compute a legacy normalizer from the parquet
    in-place."""
    from deribit_paper.features.transvol_history import TransVolHistoryBuffer

    parquet_path = _build_synthetic_parquet(tmp_path, n_rows=64)
    buf = TransVolHistoryBuffer(max_size=32, n_features=3, seq_len=8)

    # Legacy: empty arrays. Pre-cycle-29 this raised broadcast error.
    feat_mean = np.zeros(3, dtype=np.float32)
    feat_std = np.ones(3, dtype=np.float32)
    time_mean = np.zeros(0, dtype=np.float32)
    time_std = np.zeros(0, dtype=np.float32)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        n_loaded = buf.prewarm_from_parquet(
            features_parquet=parquet_path,
            feature_columns=["feat_0", "feat_1", "feat_2"],
            time_columns=("duration_s", "n_bars_in_window"),
            feat_mean=feat_mean,
            feat_std=feat_std,
            time_mean=time_mean,
            time_std=time_std,
            n_rows=32,
        )

    assert n_loaded == 32, f"expected 32 rows loaded, got {n_loaded}"
    # The buffer should contain 32 rows of properly-normalized time
    # features. ``latest_window`` returns the last seq_len rows.
    feats, times = buf.latest_window()
    assert times.shape == (8, 2), f"unexpected time-feature shape: {times.shape}"
    # The legacy fallback should have emitted a DeprecationWarning.
    msgs = [str(w.message) for w in captured if issubclass(w.category, DeprecationWarning)]
    assert any("time_normalizer" in m for m in msgs), (
        f"expected DeprecationWarning mentioning time_normalizer; got: {msgs}"
    )


def test_prewarm_uses_supplied_normalizer_when_present(tmp_path):
    """Sanity: when time_mean/time_std are supplied (cycle-25+ ckpt),
    the prewarm uses them and does NOT trigger the legacy fallback."""
    from deribit_paper.features.transvol_history import TransVolHistoryBuffer

    parquet_path = _build_synthetic_parquet(tmp_path, n_rows=64)
    buf = TransVolHistoryBuffer(max_size=32, n_features=3, seq_len=8)

    feat_mean = np.zeros(3, dtype=np.float32)
    feat_std = np.ones(3, dtype=np.float32)
    # Supplied normalizer (typical cycle-25 case)
    time_mean = np.array([300.0, 250.0], dtype=np.float32)
    time_std = np.array([100.0, 75.0], dtype=np.float32)

    with warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        buf.prewarm_from_parquet(
            features_parquet=parquet_path,
            feature_columns=["feat_0", "feat_1", "feat_2"],
            time_columns=("duration_s", "n_bars_in_window"),
            feat_mean=feat_mean,
            feat_std=feat_std,
            time_mean=time_mean,
            time_std=time_std,
            n_rows=32,
        )

    # No DeprecationWarning about time_normalizer should fire.
    dep = [
        w for w in captured
        if issubclass(w.category, DeprecationWarning)
        and "time_normalizer" in str(w.message)
    ]
    assert not dep, (
        "Supplied normalizer should NOT trigger legacy fallback; got: "
        f"{[str(w.message) for w in dep]}"
    )


# ── End-to-end checkpoint round-trip ──────────────────────────────────


def test_loader_extracts_cycle25_keys_from_mock_checkpoint(tmp_path):
    """Mock a cycle-25-format checkpoint and load via _load_transvol's
    relevant lookup logic. The cycle-29 fix means time_mean/time_std
    have shape (2,), not (0,).

    We don't call the full ``load_bundle`` because that also loads
    LightGBM, DRL, etc. Instead, recreate the model_loader extraction
    snippet locally and assert the keys are read correctly.
    """
    # Verify the key lookup logic directly (no need for full ckpt load).
    cycle25_ckpt = {
        "config": object(),                       # placeholder
        "model_state_dict": {},
        "normalizer_mean": np.array([0.0] * 5),
        "normalizer_std": np.array([1.0] * 5),
        "time_normalizer_mean": np.array([100.0, 50.0], dtype=np.float32),
        "time_normalizer_std": np.array([10.0, 5.0], dtype=np.float32),
    }
    # Mirror the model_loader extraction:
    time_mean = (
        np.asarray(cycle25_ckpt["time_normalizer_mean"])
        if cycle25_ckpt.get("time_normalizer_mean") is not None else np.zeros(0)
    )
    time_std = (
        np.asarray(cycle25_ckpt["time_normalizer_std"])
        if cycle25_ckpt.get("time_normalizer_std") is not None else np.zeros(0)
    )
    assert time_mean.shape == (2,)
    assert time_std.shape == (2,)
    np.testing.assert_allclose(time_mean, [100.0, 50.0])
    np.testing.assert_allclose(time_std, [10.0, 5.0])


def test_loader_returns_empty_for_pre_cycle25_checkpoint():
    """A pre-cycle-25 checkpoint has neither time_normalizer_* nor the
    older time_* keys. The loader must return shape (0,) — which
    prewarm_from_parquet now handles via the legacy fallback above."""
    pre25_ckpt = {
        "config": object(),
        "model_state_dict": {},
        "normalizer_mean": np.array([0.0] * 5),
        "normalizer_std": np.array([1.0] * 5),
        # No time_normalizer_* keys.
    }
    time_mean = (
        np.asarray(pre25_ckpt["time_normalizer_mean"])
        if pre25_ckpt.get("time_normalizer_mean") is not None else np.zeros(0)
    )
    time_std = (
        np.asarray(pre25_ckpt["time_normalizer_std"])
        if pre25_ckpt.get("time_normalizer_std") is not None else np.zeros(0)
    )
    assert time_mean.shape == (0,)
    assert time_std.shape == (0,)
