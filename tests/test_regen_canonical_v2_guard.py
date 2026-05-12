"""Cycle-28b (2026-05-13) regression test for the regen_canonical_v2
``svi_bucket_freq`` parity guard.

The guard refuses to regen the canonical V2 trade log when live and
offline disagree on ``svi_bucket_freq``. Pre-cycle-28 a manual
``--svi-bucket-freq 15min`` override applied upstream would silently
produce a trade log that doesn't match what live can consume — the
F-4 contract applies at the live bucket cadence, mismatched offline
buckets land on the wrong observable timestamps.

This test pins the guard so a future refactor that drops it fails CI.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def _add_repo_root_to_path():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))


def _import_guard():
    """Import the guard fresh so monkeypatches re-apply."""
    spec = importlib.util.spec_from_file_location(
        "regen_canonical_v2",
        REPO_ROOT / "scripts" / "regen_canonical_v2.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._assert_svi_bucket_freq_parity


def test_guard_passes_when_live_and_offline_agree():
    """Both configs default to ``"1h"`` — the guard must pass without
    raising in production state."""
    guard = _import_guard()
    # No exception → guard accepted the default state.
    guard()


def test_guard_blocks_when_offline_diverges_to_15min(monkeypatch):
    """If a future change flips offline's default to anything other
    than ``"1h"``, the guard must abort. We simulate the divergence
    by monkeypatching ``VolArbConfig`` to return ``"15min"``."""
    guard = _import_guard()
    from vol_arb.config import VolArbConfig

    class _Stub:
        svi_bucket_freq = "15min"

    monkeypatch.setattr(
        "vol_arb.config.VolArbConfig",
        lambda *args, **kwargs: _Stub(),
    )
    with pytest.raises(SystemExit, match="parity guard FAILED"):
        guard()


def test_guard_blocks_when_live_diverges(monkeypatch):
    """Conversely, a flip in PaperConfig's default must also abort.
    Pin both directions of the parity check so neither side can drift
    without the guard yelling."""
    guard = _import_guard()
    from deribit_paper.config import PaperConfig

    class _Stub:
        svi_bucket_freq = "30min"

    def stub_with_v2_preset(cls=None, base=None):
        return _Stub()

    monkeypatch.setattr(
        PaperConfig, "with_v2_preset", classmethod(stub_with_v2_preset),
    )
    with pytest.raises(SystemExit, match="parity guard FAILED"):
        guard()
