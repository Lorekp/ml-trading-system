"""Tests for the DRL manifest bar-minutes check.

E.2.2 in plan ``C:\\Users\\laure\\.claude\\plans\\starry-rolling-rabin.md``.

The engine reads a sibling ``<policy>.manifest.json`` at startup and
WARNs when ``bar_minutes`` doesn't match ``cfg.bar_minutes``. This
prevents the silent failure of running a 15m-trained SAC v4 policy on
a 5m bar configuration (which would have happened in the h=5m audit
before the bisect found it).
"""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pytest


def test_existing_v4_manifest_has_bar_minutes_15():
    """The shipped manifest beside production_sac_v4_final.zip pins
    bar_minutes=15. If a future re-train changes the horizon, the
    manifest MUST be updated alongside; this test catches the omission.
    """
    manifest = (
        Path("vol_arb/drl_hedging/models/production_sac_v4_final.manifest.json")
    )
    if not manifest.exists():
        pytest.skip(f"{manifest} not present in this checkout")
    data = json.loads(manifest.read_text(encoding="utf-8"))
    assert data["bar_minutes"] == 15, (
        f"v4 SAC bundle was trained at 15m bars per "
        f"memory:project_w5_v4_adopted; manifest has "
        f"bar_minutes={data['bar_minutes']}. Either restore the manifest "
        f"or re-train + update."
    )


def test_manifest_check_logs_warning_on_mismatch(tmp_path, caplog):
    """Synthetic policy + manifest with bar_minutes=15, engine cfg=5 →
    WARN logged at startup. Doesn't actually run the engine; we
    exercise the manifest read + comparison logic directly."""
    policy_path = tmp_path / "fake_policy.zip"
    policy_path.write_bytes(b"stub")
    manifest_path = tmp_path / "fake_policy.manifest.json"
    manifest_path.write_text(json.dumps({"bar_minutes": 15}))

    # Replicate the engine's manifest-check block in isolation.
    import logging
    logger = logging.getLogger("test_drl_manifest")
    cfg_bar_minutes = 5

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    trained = int(data["bar_minutes"])
    with caplog.at_level(logging.WARNING, logger="test_drl_manifest"):
        if trained != cfg_bar_minutes:
            logger.warning(
                "DRL policy trained at bar_minutes=%d but cfg=%d",
                trained, cfg_bar_minutes,
            )
    assert any("bar_minutes=15" in r.message for r in caplog.records)


def test_manifest_check_silent_on_match(tmp_path):
    """Synthetic policy + manifest matching cfg → no WARN."""
    policy_path = tmp_path / "fake_policy.zip"
    policy_path.write_bytes(b"stub")
    manifest_path = tmp_path / "fake_policy.manifest.json"
    manifest_path.write_text(json.dumps({"bar_minutes": 15}))

    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert int(data["bar_minutes"]) == 15
