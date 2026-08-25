"""Regression tests for the historical V2 regeneration
``svi_bucket_freq`` parity guard.

The guard refuses to regenerate the invalidated historical V2 artifact when
paper replay and offline configs disagree on ``svi_bucket_freq``. A manual
``--svi-bucket-freq 15min`` override applied upstream would silently
produce a trade log that doesn't match what replay can consume — the
F-4 contract applies at the replay cadence, mismatched offline
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
        "regenerate_historical_v2",
        REPO_ROOT / "scripts" / "regenerate_historical_v2.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module._assert_svi_bucket_freq_parity


def test_guard_passes_when_replay_and_offline_agree():
    """Cycle-30 (2026-05-13): both configs default to ``"15min"`` —
    the guard must pass for the archived replay state."""
    guard = _import_guard()
    # No exception → guard accepted the default state.
    guard()


def test_guard_blocks_when_offline_diverges_to_1h(monkeypatch):
    """If a future change flips offline's default to anything other
    than ``"15min"``, the cycle-28b guard must abort. We simulate the
    divergence by monkeypatching ``VolArbConfig`` to return ``"1h"``."""
    guard = _import_guard()
    from vol_arb.config import VolArbConfig

    class _Stub:
        svi_bucket_freq = "1h"

    monkeypatch.setattr(
        "vol_arb.config.VolArbConfig",
        lambda *args, **kwargs: _Stub(),
    )
    with pytest.raises(SystemExit, match="parity guard FAILED"):
        guard()


def test_guard_blocks_when_replay_diverges(monkeypatch):
    """Conversely, a flip in PaperConfig's default must also abort.
    Pin both directions of the cycle-28b config-parity check so neither
    side can drift without the guard yelling."""
    guard = _import_guard()
    from deribit_paper.config import PaperConfig

    class _Stub:
        svi_bucket_freq = "30min"

    def stub_historical_v2_preset(cls=None, base=None):
        return _Stub()

    monkeypatch.setattr(
        PaperConfig,
        "historical_v2_preset",
        classmethod(stub_historical_v2_preset),
    )
    with pytest.raises(SystemExit, match="parity guard FAILED"):
        guard()


# ── Cycle-32 (2026-05-13) — parquet-cadence layer ─────────────────────


def test_detect_parquet_cadence_15min_synthetic(tmp_path):
    """Cycle-32 cadence detector: a parquet with 15-min-spaced
    timestamps returns ``"15min"``."""
    import pandas as pd

    spec = __import__("importlib").util.spec_from_file_location(
        "regenerate_historical_v2",
        REPO_ROOT / "scripts" / "regenerate_historical_v2.py",
    )
    module = __import__("importlib").util.module_from_spec(spec)
    spec.loader.exec_module(module)
    detect = module._detect_parquet_cadence

    idx = pd.date_range("2026-05-01", periods=40, freq="15min", tz="UTC")
    df = pd.DataFrame({"timestamp": idx, "a": 0.0})
    path = tmp_path / "synth_15min.parquet"
    df.to_parquet(path)
    assert detect(path) == "15min"


def test_detect_parquet_cadence_1h_synthetic(tmp_path):
    """1h-spaced timestamps return ``"1h"``."""
    import pandas as pd

    spec = __import__("importlib").util.spec_from_file_location(
        "regenerate_historical_v2",
        REPO_ROOT / "scripts" / "regenerate_historical_v2.py",
    )
    module = __import__("importlib").util.module_from_spec(spec)
    spec.loader.exec_module(module)
    detect = module._detect_parquet_cadence

    idx = pd.date_range("2026-05-01", periods=24, freq="1h", tz="UTC")
    df = pd.DataFrame({"timestamp": idx, "a": 0.0})
    path = tmp_path / "synth_1h.parquet"
    df.to_parquet(path)
    assert detect(path) == "1h"


def test_cycle32_detector_round_trip_1h_15min(tmp_path):
    """Cycle-32 sanity: the cadence detector returns ``"1h"`` for
    1h-spaced and ``"15min"`` for 15min-spaced parquets. Together
    with the unit tests above this is sufficient to pin the cycle-32
    layer-2 behaviour: when paired with the guard's matching logic,
    a mis-built parquet is caught."""
    import pandas as pd

    spec = __import__("importlib").util.spec_from_file_location(
        "regenerate_historical_v2",
        REPO_ROOT / "scripts" / "regenerate_historical_v2.py",
    )
    module = __import__("importlib").util.module_from_spec(spec)
    spec.loader.exec_module(module)
    detect = module._detect_parquet_cadence

    # Build both cadences in the same test
    for freq, expected in [("15min", "15min"), ("1h", "1h")]:
        idx = pd.date_range("2026-05-01", periods=24, freq=freq, tz="UTC")
        df = pd.DataFrame({"timestamp": idx, "a": 0.0})
        path = tmp_path / f"synth_{expected}.parquet"
        df.to_parquet(path)
        actual = detect(path)
        assert actual == expected, (
            f"cycle-32 detector returned {actual!r} for {freq}-spaced parquet; "
            f"expected {expected!r}"
        )
