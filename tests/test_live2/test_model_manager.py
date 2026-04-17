"""D3 ModelManager tests: queue persistence, subprocess timeout, vintage check,
and the require_metadata fail-safe in SklearnBackend."""

from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import joblib
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from live.model_manager import ModelManager
from live.signal_generator import SklearnBackend


# ── Queue persistence ────────────────────────────────────────────────────────


def test_queue_persists_across_restart(tmp_path: Path):
    state_path = tmp_path / "state.json"
    model_dir = tmp_path / "models"

    m1 = ModelManager(model_dir=str(model_dir), state_path=state_path)
    m1.queue_training(["BTC", "ETH"], retrain=True)

    # New instance must see the queue
    m2 = ModelManager(model_dir=str(model_dir), state_path=state_path)
    assert m2._pending == ["BTC", "ETH"]
    assert m2._retrain is True


def test_queue_dedupes_and_preserves_order(tmp_path: Path):
    m = ModelManager(model_dir=str(tmp_path / "models"),
                     state_path=tmp_path / "s.json")
    m.queue_training(["BTC", "ETH"])
    m.queue_training(["ETH", "SOL"])  # dupe "ETH", new "SOL"
    assert m._pending == ["BTC", "ETH", "SOL"]


def test_proc_assets_requeued_on_restart(tmp_path: Path):
    """Simulating a crash mid-training: the state file says a subprocess was
    running; on restart the assets must be pushed back onto the queue."""
    state_path = tmp_path / "state.json"
    with open(state_path, "w") as f:
        json.dump({
            "pending": ["XRP"],
            "retrain": False,
            "proc_assets": ["BTC", "ETH"],
            "proc_started_at": "2026-04-16T00:00:00+00:00",
        }, f)

    m = ModelManager(model_dir=str(tmp_path / "models"), state_path=state_path)
    # XRP (original pending) + BTC, ETH (in-flight before crash)
    assert set(m._pending) == {"XRP", "BTC", "ETH"}


# ── Subprocess lifecycle ─────────────────────────────────────────────────────


class _FakeProc:
    def __init__(self, poll_rc: int | None = None, pid: int = 4242):
        self._poll_rc = poll_rc
        self.pid = pid
        self.stderr = MagicMock()
        self.stderr.read.return_value = b""
        self.killed = False

    def poll(self):
        return self._poll_rc

    def kill(self):
        self.killed = True
        self._poll_rc = -9

    def wait(self, timeout=None):
        return self._poll_rc


def test_spawn_then_reap_success(tmp_path: Path, monkeypatch):
    """tick() starts a subprocess, subsequent tick() with a succeeded proc
    reaps and returns True (caller should reload)."""
    m = ModelManager(model_dir=str(tmp_path / "models"),
                     state_path=tmp_path / "s.json")
    m.queue_training(["BTC"], retrain=False)

    fake = _FakeProc(poll_rc=None)
    monkeypatch.setattr("live.model_manager.subprocess.Popen",
                         lambda *a, **kw: fake)

    # First tick spawns. Returns False (not finished).
    assert m.tick() is False
    assert m._proc is fake
    assert m._pending == []

    # Not finished yet
    assert m.tick() is False

    # Finished with rc=0
    fake._poll_rc = 0
    assert m.tick() is True
    assert m._proc is None


def test_reap_on_failure_logs_and_clears_state(tmp_path: Path, monkeypatch, caplog):
    m = ModelManager(model_dir=str(tmp_path / "models"),
                     state_path=tmp_path / "s.json")
    m.queue_training(["BTC"])
    fake = _FakeProc(poll_rc=None)
    monkeypatch.setattr("live.model_manager.subprocess.Popen",
                         lambda *a, **kw: fake)
    m.tick()  # spawn

    fake._poll_rc = 1
    fake.stderr.read.return_value = b"oops"
    caplog.clear()
    assert m.tick() is True
    assert any("exited with rc=1" in r.message for r in caplog.records)
    assert m._proc is None


def test_timeout_kills_and_requeues(tmp_path: Path, monkeypatch):
    """A subprocess that exceeds max_runtime_s must be killed and its assets
    requeued so the next tick retries them."""
    now = [1_000_000.0]  # mutable clock

    m = ModelManager(
        model_dir=str(tmp_path / "models"),
        state_path=tmp_path / "s.json",
        max_runtime_s=60,
        clock=lambda: now[0],
    )
    m.queue_training(["BTC", "ETH"])

    fake = _FakeProc(poll_rc=None)  # will never terminate on its own
    monkeypatch.setattr("live.model_manager.subprocess.Popen",
                         lambda *a, **kw: fake)
    m.tick()  # spawn
    assert m._proc is fake

    # Advance clock past the timeout
    now[0] += 120

    # Next tick must kill + requeue
    m.tick()
    assert fake.killed is True
    assert m._proc is None
    assert set(m._pending) == {"BTC", "ETH"}


def test_spawn_failure_restores_queue(tmp_path: Path, monkeypatch):
    """If Popen raises, the queued assets must NOT be lost."""
    m = ModelManager(model_dir=str(tmp_path / "models"),
                     state_path=tmp_path / "s.json")
    m.queue_training(["BTC", "ETH"], retrain=True)

    def boom(*a, **kw):
        raise OSError("fork failed")
    monkeypatch.setattr("live.model_manager.subprocess.Popen", boom)

    m.tick()
    assert m._proc is None
    # Queue should survive
    assert set(m._pending) == {"BTC", "ETH"}
    assert m._retrain is True


# ── Vintage ──────────────────────────────────────────────────────────────────


def test_stale_models_empty_when_model_dir_missing(tmp_path: Path):
    m = ModelManager(model_dir=str(tmp_path / "nonexistent"),
                     state_path=tmp_path / "s.json")
    assert m.stale_models() == []


def test_stale_models_flags_old_files(tmp_path: Path):
    model_dir = tmp_path / "models" / "15m"
    model_dir.mkdir(parents=True)
    p = model_dir / "BTC_model.joblib"
    p.touch()
    # Make it 60 days old
    old = time.time() - 60 * 86_400
    os.utime(p, (old, old))

    m = ModelManager(model_dir=str(tmp_path / "models"),
                     state_path=tmp_path / "s.json")
    stale = m.stale_models(max_age_days=30)
    assert len(stale) == 1
    asset, tf, age = stale[0]
    assert asset == "BTC"
    assert tf == "15m"
    assert age >= 59  # roughly 60 days


def test_stale_models_excludes_recent(tmp_path: Path):
    model_dir = tmp_path / "models" / "15m"
    model_dir.mkdir(parents=True)
    (model_dir / "BTC_model.joblib").touch()
    m = ModelManager(model_dir=str(tmp_path / "models"),
                     state_path=tmp_path / "s.json")
    assert m.stale_models(max_age_days=1) == []


# ── require_metadata fail-safe ──────────────────────────────────────────────


def _write_model(tf_dir: Path, asset: str, *, with_meta: bool):
    """Emit a trivially-fit joblib pair plus optional sidecar metadata."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(100, 4))
    y = (X[:, 0] > 0).astype(int)
    sc = StandardScaler().fit(X)
    m = LogisticRegression().fit(sc.transform(X), y)
    joblib.dump(m, tf_dir / f"{asset}_model.joblib")
    joblib.dump(sc, tf_dir / f"{asset}_scaler.joblib")
    if with_meta:
        with open(tf_dir / f"{asset}_meta.json", "w") as f:
            json.dump({"asset": asset, "tf": tf_dir.name,
                       "conf_threshold": 0.04, "vol_threshold": 0.002}, f)


def test_require_metadata_skips_legacy(tmp_path: Path):
    tf_dir = tmp_path / "15m"
    tf_dir.mkdir()
    _write_model(tf_dir, "BTC", with_meta=True)   # modern
    _write_model(tf_dir, "ETH", with_meta=False)  # legacy

    be = SklearnBackend(require_metadata=True)
    n = be.load_models(tmp_path)
    assert n == 1
    assert be.has_model("BTC", "15m") is True
    assert be.has_model("ETH", "15m") is False


def test_require_metadata_false_loads_everything(tmp_path: Path):
    tf_dir = tmp_path / "15m"
    tf_dir.mkdir()
    _write_model(tf_dir, "BTC", with_meta=True)
    _write_model(tf_dir, "ETH", with_meta=False)

    be = SklearnBackend(require_metadata=False)
    n = be.load_models(tmp_path)
    assert n == 2
    assert be.has_model("ETH", "15m") is True
