"""Background model training orchestrator.

Spawns download_and_train.py as a subprocess for assets that need models.
Handles both data download (Binance Vision) and model training in one pipeline.
Non-blocking — engine continues trading while training runs in background.

Hardening (D3 fix):
  - Queue + subprocess state persisted to `data/live/model_mgr_state.json` so a
    pending retrain survives a process restart.
  - Subprocess has a wall-clock timeout (default 1h); on timeout the process is
    killed and the asset list is requeued.
  - `stale_models(max_age_days)` surfaces models whose joblib mtime is older
    than the threshold, so the engine can alert and/or trigger a retrain.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)

_DEFAULT_STATE_FILENAME = "model_mgr_state.json"


class ModelManager:
    """Manages background training for assets missing or stale models.

    State is persisted to disk so a queued retrain is preserved across restarts.

    Typical usage:
        mgr = ModelManager(model_dir="data/live/models")
        mgr.queue_training(["SOL", "AVAX"])   # queue
        reloaded = mgr.tick()                 # call each bar
        stale = mgr.stale_models(max_age_days=35)
    """

    def __init__(
        self,
        model_dir: str = "data/live/models",
        train_script: str = "scripts/download_and_train.py",
        *,
        max_runtime_s: int = 3600,
        state_path: str | Path | None = None,
        clock: callable = time.time,  # injectable for tests
    ) -> None:
        self._model_dir = Path(model_dir)
        self._train_script = train_script
        self._max_runtime_s = int(max_runtime_s)
        self._clock = clock

        self._proc: subprocess.Popen | None = None
        self._proc_started_at: float | None = None
        self._proc_assets: list[str] = []

        self._pending: list[str] = []
        self._retrain: bool = False

        # State file sits alongside the model directory by default, so both
        # live data + manager state share the same lifecycle.
        if state_path is None:
            state_path = self._model_dir.parent / _DEFAULT_STATE_FILENAME
        self._state_path = Path(state_path)

        self._restore_state()

    # ── Queue ────────────────────────────────────────────────────────────────

    def queue_training(self, assets: list[str], retrain: bool = False) -> None:
        """Add assets to the training queue and persist."""
        new = [a for a in assets if a not in self._pending]
        if new:
            self._pending.extend(new)
            if retrain:
                self._retrain = True
            logger.info("Queued %d assets for %s: %s",
                        len(new), "monthly retrain" if retrain else "training", new)
            self._persist_state()

    # ── Tick ─────────────────────────────────────────────────────────────────

    def tick(self) -> bool:
        """Advance the state machine one step. Returns True iff training just
        finished and the caller should reload models from disk."""
        if self._proc is not None:
            # Timeout check — kill and requeue if exceeded
            if (self._proc_started_at is not None
                    and self._clock() - self._proc_started_at > self._max_runtime_s):
                self._timeout_current_proc()
                return False

            rc = self._proc.poll()
            if rc is None:
                return False  # still running
            return self._reap_proc(rc)

        # Start new training if pending
        if not self._pending:
            return False
        self._spawn_proc()
        return False

    # ── Vintage ──────────────────────────────────────────────────────────────

    def stale_models(self, max_age_days: int = 35) -> list[tuple[str, str, float]]:
        """Return (asset, tf_label, age_days) for each model whose joblib mtime
        is older than `max_age_days`. Missing model_dir → empty list."""
        if not self._model_dir.exists():
            return []
        now = self._clock()
        stale: list[tuple[str, str, float]] = []
        for tf_dir in self._model_dir.iterdir():
            if not tf_dir.is_dir():
                continue
            tf_label = tf_dir.name
            for mp in tf_dir.glob("*_model.joblib"):
                try:
                    mtime = mp.stat().st_mtime
                except OSError:
                    continue
                age_days = (now - mtime) / 86_400.0
                if age_days > max_age_days:
                    asset = mp.stem.replace("_model", "")
                    stale.append((asset, tf_label, age_days))
        return stale

    @property
    def is_training(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    # ── Internals ────────────────────────────────────────────────────────────

    def _spawn_proc(self) -> None:
        assets = self._pending
        self._pending = []

        cmd = [sys.executable, self._train_script, "--assets", ",".join(assets)]
        if self._retrain:
            cmd.append("--retrain")
        retrain_flag = self._retrain
        self._retrain = False  # consumed when we spawn

        try:
            self._proc = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception as e:
            # Spawn failed — put the assets back and log. State is still dirty,
            # so persist.
            logger.error("Failed to start training subprocess: %s", e)
            self._pending = assets + self._pending
            self._retrain = self._retrain or retrain_flag
            self._persist_state()
            return

        self._proc_started_at = self._clock()
        self._proc_assets = list(assets)
        logger.info("Started background training for %d assets: %s (pid=%d, retrain=%s)",
                    len(assets), assets, self._proc.pid, retrain_flag)
        self._persist_state()

    def _reap_proc(self, rc: int) -> bool:
        if rc == 0:
            logger.info("Background training finished successfully (assets=%s)",
                        self._proc_assets)
        else:
            stderr = ""
            try:
                stderr = self._proc.stderr.read().decode(errors="replace")[-500:]
            except Exception:
                pass
            logger.warning("Background training exited with rc=%d: %s", rc, stderr)
        self._proc = None
        self._proc_started_at = None
        self._proc_assets = []
        self._persist_state()
        return True  # signal caller to reload models

    def _timeout_current_proc(self) -> None:
        """Kill a hung subprocess, requeue its assets, persist."""
        logger.error("TRAIN_TIMEOUT: subprocess exceeded %ss (assets=%s) — killing",
                     self._max_runtime_s, self._proc_assets)
        try:
            self._proc.kill()
            self._proc.wait(timeout=5)
        except Exception as e:
            logger.error("Failed to kill timed-out subprocess: %s", e)
        # Requeue the assets so they get retried on next spawn
        requeued = [a for a in self._proc_assets if a not in self._pending]
        self._pending.extend(requeued)
        self._proc = None
        self._proc_started_at = None
        self._proc_assets = []
        self._persist_state()

    # ── Persistence ──────────────────────────────────────────────────────────

    def _persist_state(self) -> None:
        """Atomically write the manager's state to disk."""
        state = {
            "pending": list(self._pending),
            "retrain": bool(self._retrain),
            "proc_assets": list(self._proc_assets),
            "proc_started_at": (
                datetime.fromtimestamp(self._proc_started_at, tz=timezone.utc).isoformat()
                if self._proc_started_at is not None else None
            ),
        }
        try:
            self._state_path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self._state_path.with_suffix(self._state_path.suffix + ".tmp")
            with open(tmp, "w") as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, self._state_path)
        except Exception as e:
            logger.warning("ModelManager: failed to persist state to %s: %s",
                            self._state_path, e)

    def _restore_state(self) -> None:
        """Re-read persisted state. If a subprocess was marked running at the
        time of the last write but no process is alive now, we treat those
        assets as interrupted and push them back onto the pending queue.
        """
        if not self._state_path.exists():
            return
        try:
            with open(self._state_path) as f:
                state = json.load(f)
        except Exception as e:
            logger.warning("ModelManager: failed to load state from %s: %s",
                            self._state_path, e)
            return

        self._pending = list(state.get("pending", []))
        self._retrain = bool(state.get("retrain", False))
        proc_assets = list(state.get("proc_assets", []))
        if proc_assets:
            # We crashed or were restarted mid-training — requeue those assets.
            added = [a for a in proc_assets if a not in self._pending]
            if added:
                logger.warning(
                    "ModelManager: prior subprocess was in-flight for %d assets; "
                    "requeuing (they will be retried on next tick)", len(added))
                self._pending.extend(added)
                self._persist_state()
