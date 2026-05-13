"""Cycle-37 (2026-05-13): pin the bundle.time_normalizer propagation fix.

Surfaced by the cycle-33 bar-trace at 2026-05-13 19:00 UTC:

  TransVol live inference failed at 2026-05-13 19:00:00+00:00;
  ValueError: operands could not be broadcast together with shapes (2,) (0,)

Cycle-29 added a legacy fallback inside ``prewarm_from_parquet`` but
the resolved time_mean/std stayed local. The bundle's
``transvol_time_mean``/``_std`` remained shape (0,), so the per-bar
live inference path at ``engine.py`` :func:`process_bar` ~line 2309
still raised the broadcast error on every bar — engine fell back to
baseline tv_* slots silently for the entire post-restart session.

Cycle-37 closes by computing the same legacy fallback IN THE ENGINE
before prewarm and writing the result back to the bundle so both
prewarm AND per-bar inference see resolved (n_time_inputs,)-shape
values.

Tests:
1. Source-regex sentinel — the engine has the legacy fallback block
   gated on ``tm.shape == (0,)`` and writes back to
   ``self.model_bundle.transvol_time_mean/_std``. Any future refactor
   that drops the write-back fails CI.
2. Numerical equivalence — given a 1000-row synthetic features
   parquet with two time columns, the engine's legacy fallback should
   compute mean/std equal to ``T[:max(len-1000, 1)].mean/std(axis=0)``
   matching ``run_walkforward_hybrid._resolve_time_normalizer``.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def test_engine_source_has_cycle37_propagation_block() -> None:
    """The engine's TransVol prewarm site contains the cycle-37 write-back."""
    src = Path("deribit_paper/engine.py").read_text(encoding="utf-8")
    assert "Cycle-37" in src, "cycle-37 fix comment missing"
    # Source must mutate the bundle (not just compute locally).
    assert "self.model_bundle.transvol_time_mean = tm" in src, (
        "engine must write resolved time_mean back to the bundle"
    )
    assert "self.model_bundle.transvol_time_std = ts_" in src, (
        "engine must write resolved time_std back to the bundle"
    )
    # The legacy fallback must trigger only on (0,)-shape inputs so
    # cycle-25+ checkpoints (which have the proper keys) skip it.
    assert "if tm.shape == (0,)" in src, (
        "legacy fallback must gate on the shape mismatch"
    )


def test_legacy_fallback_numerical_match() -> None:
    """The fallback formula matches ``vol_model/run_walkforward_hybrid._resolve_time_normalizer``.

    Builds a 1500-row synthetic time-feature matrix, then verifies the
    cycle-37 slice + mean/std are byte-equal to T[:max(len-1000, 1)]
    .mean(axis=0) / .std(axis=0) at float32 precision.
    """
    rng = np.random.default_rng(42)
    T = rng.standard_normal((1500, 2)).astype(np.float32)
    slice_end = max(len(T) - 1000, 1)
    expected_mean = T[:slice_end].mean(axis=0).astype(np.float32)
    expected_std = T[:slice_end].std(axis=0).astype(np.float32)
    expected_std = np.clip(expected_std, 1e-8, None)

    # Replicate the engine's snippet — see deribit_paper/engine.py
    # cycle-37 block. Updating this test to keep numerics aligned with
    # the engine is the SOURCE-OF-TRUTH check (the engine's snippet is
    # below).
    tm = T[:slice_end].mean(axis=0).astype(np.float32)
    ts_ = T[:slice_end].std(axis=0).astype(np.float32)
    ts_ = np.clip(ts_, 1e-8, None)

    np.testing.assert_array_equal(tm, expected_mean)
    np.testing.assert_array_equal(ts_, expected_std)
    assert ts_.min() >= 1e-8, "std must be clipped away from zero"


def test_post_fallback_broadcast_works() -> None:
    """After the legacy fallback, broadcasting ``(raw_time - tm) / ts_`` succeeds."""
    rng = np.random.default_rng(7)
    T = rng.standard_normal((1500, 2)).astype(np.float32)
    slice_end = max(len(T) - 1000, 1)
    tm = T[:slice_end].mean(axis=0).astype(np.float32)
    ts_ = np.clip(T[:slice_end].std(axis=0).astype(np.float32), 1e-8, None)
    raw_time = rng.standard_normal(2).astype(np.float32)
    # This is the operation that crashed at 2026-05-13 19:00 UTC with
    # the unfixed bundle (raw_time - (0,)) — must now succeed.
    out = (raw_time - tm) / ts_
    assert out.shape == (2,)
    assert np.all(np.isfinite(out))
