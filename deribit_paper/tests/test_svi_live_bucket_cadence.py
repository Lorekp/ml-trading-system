"""Cycle-30 (2026-05-13) regression tests for SVI bucket cadence.

Pre-cycle-30 ``LiveSVICalibrator.__init__`` defaulted
``bucket_freq="1h"`` while the canonical offline pipeline
(``vol_arb/run_pipeline.py:121``) defaults to
``svi_bucket_freq = bar_freq_str = "{bar_minutes}min"`` = ``"15min"``
at h=15 production. Empirical: ``vol_arb/output/svi_features_15m.parquet``
has DISTINCT ``svi_atm_1d`` per 15-min bar. Live IV was sticky for
4 consecutive 15-min bars while offline IV updated per bar →
~3vp Δiv drift on the parity audit (2026-05-13).

Cycle-30 flips the live default to ``"15min"`` to match offline.

These tests pin:
1. The production constant is ``"15min"`` (not ``"1h"`` or anything else).
2. The matching cross-reference: ``vol_arb/run_pipeline.py``'s default
   resolves to the same string.
"""

from __future__ import annotations

import re
from pathlib import Path

from deribit_paper.features.svi_live import LiveSVICalibrator


def test_default_bucket_freq_is_15min(tmp_path):
    """Cycle-30 production constant pin: ``LiveSVICalibrator`` defaults
    to ``bucket_freq="15min"``."""
    cal = LiveSVICalibrator(cache_path=tmp_path / "svi.parquet")
    assert cal.svi_config.bucket_freq == "15min", (
        f"cycle-30: LiveSVICalibrator default bucket_freq must be '15min' "
        f"to match offline run_pipeline.py:121; got {cal.svi_config.bucket_freq!r}"
    )


def test_source_pins_15min_constant():
    """Source-regex sentinel: ``svi_live.py`` must hardcode
    ``bucket_freq="15min"`` as the production default. A future
    refactor that reverts to ``"1h"`` (or any other non-15min value)
    must fail CI here.
    """
    src = Path(__file__).resolve().parents[1].joinpath(
        "features", "svi_live.py",
    ).read_text()
    # The default-construct block of LiveSVICalibrator.__init__ must
    # contain bucket_freq="15min" (with double-quoted string literal).
    assert re.search(r'bucket_freq=["\']15min["\']', src), (
        "cycle-30: svi_live.py must hardcode bucket_freq=\"15min\" in the default "
        "SVIConfig construction"
    )


def test_offline_pipeline_default_matches_live():
    """Cross-reference: ``vol_arb/run_pipeline.py:121`` defaults
    ``svi_bucket_freq`` to ``bar_freq_str = f"{args.bar_minutes}min"``
    which is ``"15min"`` at h=15 production. The live default must
    match that string exactly (so the F-4 observable timestamps land
    on the same decision bars in both pipelines)."""
    src = Path(__file__).resolve().parents[2].joinpath(
        "vol_arb", "run_pipeline.py",
    ).read_text()
    # The offline pipeline computes bar_freq_str from args.bar_minutes.
    # At h=15 production (the live setting), that resolves to "15min".
    # This test pins the construction expression so a future refactor
    # that hardcodes "1h" or similar fails CI.
    assert re.search(r'bar_freq_str\s*=\s*f["\']\{args\.bar_minutes\}min["\']', src), (
        "cycle-30: vol_arb/run_pipeline.py must compute bar_freq_str from "
        "args.bar_minutes (offline-canon SVI cadence is bar-cadence-derived)"
    )
    assert re.search(
        r'cfg_kwargs\[["\']svi_bucket_freq["\']\]\s*=\s*bar_freq_str',
        src,
    ), (
        "cycle-30: vol_arb/run_pipeline.py must default svi_bucket_freq to "
        "bar_freq_str so live and offline are coherent at h=15"
    )
