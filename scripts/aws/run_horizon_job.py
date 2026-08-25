#!/usr/bin/env python
"""On-box per-horizon volatility-forecast driver (one box == one horizon H).

Ports the *job* half of the Pokerbot AWS-spot stack to the quant vol-model.
The cloud plumbing (spot launch, IMDSv2, InstanceInitiatedShutdownBehavior,
Model-A bootstrap, BLAS pinning, idempotent S3-skip-if-result-exists, the
babysitter wall-clock/launch caps and ``shutdown -h now`` teardown) lives in
the *launcher* + *babysitter* that the sibling build agents port from:

    Pokerbot/scripts/launch_sd_cfr_training_spot.py   (Model-A bootstrap, spot block,
                                                        IMDSv2 hop=2, terminate-on-shutdown)
    Pokerbot/scripts/lbr6_search_babysitter.ps1        (Gen-2 babysitter, 3 fixes)
    Pokerbot/scripts/setup_labelgen_iam.py / verify_labelgen_iam.py (IAM)
    Pokerbot/scripts/labelgen_ec2_diagnostics.py       (console-output failure report)

THIS file is the single command the launcher's user-data invokes once code +
this-horizon's data shard are staged into the repo layout:

    OMP_NUM_THREADS=1 ... python scripts/aws/run_horizon_job.py --horizon 240

It is the on-box orchestrator that the design doc calls "the job each box runs".
It is deliberately self-contained, idempotent and restart-proof so a spot
reclaim + relaunch resumes without redoing finished work, and it runs
*byte-identically* locally (for the smoke) and on the box — the only difference
is whether ``--bucket``/``QUANT_BUCKET`` is set (S3 upload is skipped when no
bucket is configured, e.g. the local smoke).

Pipeline executed per horizon H (minutes; see Investigate findings):

  0. SKIP if the result shard already exists in S3 (idempotent relaunch).
  1. ENSURE TSRV bars at H exist  (TSRV/output/{asset}_tsrv_{H}m.parquet).
     If absent, rebuild via TSRV/run_tsrv_series.py --tf H*60 (reads raw
     aggTrades — AVOID: for H in {5,15,30,60,240,480,1440} the bars are pre-staged).
  2. BUILD HAR-PD features at H   (vol_model/features/har_pd.py --horizon-minutes H).
  3. BUILD SVI features at H       (vol_arb/run_pipeline.py --svi --bar-minutes H),
     unless the per-horizon svi parquet is already staged.
  4. RUN the walk-forward forecast (run_walkforward_pruned_svi_har_pd.py
     --asset A --tf H*60 --no-transvol). This is the long Optuna job; it is
     itself idempotent (skips months whose .txt checkpoint exists).
  5. EXTRACT per-bar OOS predictions + SCORE forecast quality (QLIKE / OOS R^2 /
     RankIC, per-month AND per-year) and the downstream V2 short-vol backtest
     hook (NC Sharpe / NC PnL / NC MaxDD, per-year) — the anti-overfit scorecard.
  6. WRITE the result shard (shard.json + per_month.parquet + predictions.parquet
     + trade_log.parquet) and upload to s3://BUCKET/PREFIX/results/h{H}/.

Constants are env-overridable (QUANT_BUCKET / QUANT_PREFIX / QUANT_REGION /
QUANT_ASSET). No money is spent by importing or smoke-running this file.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# BLAS / OMP thread pinning — MUST run before numpy/pandas/scipy import.
# (Pokerbot user_data() :186-187.) numpy/pandas/scipy thrash on 64-vCPU spot
# boxes otherwise; LightGBM has its own threading and is left to use cores.
# Setting them here too (in addition to the launcher's exports) makes the
# local smoke and any direct invocation safe without the wrapper env.
# ---------------------------------------------------------------------------
import os as _os

for _v in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "NPY_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    _os.environ.setdefault(_v, "1")

import argparse
import json
import logging
import shutil
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Config (env-overridable; do NOT hardcode pokerbot values).
# Region default eu-central-1 matches the user's ~/.aws/config [default]
# (the pokerbot-port investigation corrected the prompt's us-east-1).
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]

ASSET = _os.environ.get("QUANT_ASSET", "BTCUSDT")
BUCKET = _os.environ.get("QUANT_BUCKET", "").strip()  # empty => local-only (smoke)
PREFIX = _os.environ.get("QUANT_PREFIX", "volmodel/v1").strip("/")
REGION = _os.environ.get("QUANT_REGION", "eu-central-1")

# Horizons whose TSRV bars are pre-staged (no raw-aggTrades rebuild needed).
PRESTAGED_TSRV_HORIZONS = {5, 15, 30, 60, 240, 480, 1440}

# Walk-forward window — keep --end at the TSRV coverage edge (2026-03) so all
# horizons are cross-comparable (per Investigate: TSRV-at-H ends 2026-03-31).
WF_START_DEFAULT = "2022-02"
WF_END_DEFAULT = "2026-03"
WF_TRIALS_DEFAULT = 200  # production cycle-8 config (feedback_optuna_prod_config)
WF_PATIENCE_DEFAULT = 50

# Background checkpoint-sync cadence (seconds). The walk-forward writes one
# {month}.txt + {month}_features.json per completed month; a daemon thread
# mirrors output_dir -> s3://BUCKET/PREFIX/checkpoints/h{H}/ every cadence so a
# spot reclaim + relaunch loses at most the in-progress month. The 2-minute
# interrupt notice triggers a FINAL flush before the box dies (see the
# launcher user_data interrupt poller). Env-overridable for tuning/tests.
CKPT_SYNC_SECONDS = int(_os.environ.get("QUANT_CKPT_SYNC_SECONDS", "300"))

logging.basicConfig(
    level=logging.INFO,
    stream=sys.stdout,
    format="%(asctime)s %(levelname)s [horizon-job] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("run_horizon_job")


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------
def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _git_sha() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT, capture_output=True, text=True, timeout=15,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return _os.environ.get("QUANT_GIT_SHA", "unknown")


def _run(cmd: list[str], *, log_path: Path | None = None) -> None:
    """Run a subprocess in the repo with PYTHONPATH set, streaming to stdout.

    Mirrors the Pokerbot ``... 2>&1 | tee /var/log/x.log`` rule — NO output
    redirect that hides the console (Pokerbot user_data :137-143 root-caused a
    13-attempt failure to an ``exec >logfile`` redirect). We stream to our own
    stdout (cloud-init captures it) AND optionally tee to a per-step log file.
    """
    env = dict(_os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + _os.pathsep + env.get("PYTHONPATH", "")
    pretty = " ".join(cmd)
    log.info("RUN: %s", pretty)
    t0 = time.time()
    proc = subprocess.Popen(
        cmd, cwd=REPO_ROOT, env=env,
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1,
    )
    tee = log_path.open("w", encoding="utf-8") if log_path else None
    try:
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            if tee:
                tee.write(line)
    finally:
        if tee:
            tee.close()
    rc = proc.wait()
    dt = time.time() - t0
    if rc != 0:
        raise RuntimeError(f"command failed (rc={rc}, {dt:.0f}s): {pretty}")
    log.info("OK (%.0fs): %s", dt, pretty)


def _py(*args: str, log_path: Path | None = None) -> None:
    _run([sys.executable, *args], log_path=log_path)


# --- S3 (boto3 only; the apt aws-cli v1 is broken post-pip per Pokerbot ----
#     ordering fix #3, so every transfer here goes through boto3). ----------
_S3 = None


def _s3():
    global _S3
    if _S3 is None:
        import boto3  # lazy; not needed for the local smoke
        _S3 = boto3.client("s3", region_name=REGION)
    return _S3


def _s3_key(*parts: str) -> str:
    return "/".join([PREFIX, *parts])


def _s3_exists(key: str) -> bool:
    if not BUCKET:
        return False
    from botocore.exceptions import ClientError
    try:
        _s3().head_object(Bucket=BUCKET, Key=key)
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey", "NotFound"):
            return False
        raise


def _s3_upload(local: Path, key: str) -> None:
    if not BUCKET:
        log.info("S3 disabled (no QUANT_BUCKET) — skip upload of %s", local.name)
        return
    if not local.exists():
        log.warning("upload skipped, missing local file: %s", local)
        return
    log.info("UPLOAD s3://%s/%s  (%.1f MB)", BUCKET, key, local.stat().st_size / 1e6)
    _s3().upload_file(str(local), BUCKET, key)


def _s3_download(key: str, local: Path) -> bool:
    if not BUCKET or not _s3_exists(key):
        return False
    local.parent.mkdir(parents=True, exist_ok=True)
    _s3().download_file(BUCKET, key, str(local))
    log.info("DOWNLOAD s3://%s/%s -> %s", BUCKET, key, local)
    return True


# ---------------------------------------------------------------------------
# Checkpoint / resume (spot-resilience). UNIT = one walk-forward month.
#
# The walk-forward driver (run_walkforward_pruned_svi_har_pd.py) writes one
# `{YYYY-MM}.txt` LightGBM booster + `{YYYY-MM}_features.json` per completed
# month into output_dir, and on (re)start it NATIVELY skips any month whose
# `.txt` already exists (lines 445-448 of that script: "cached, skipping").
# So per-month resume needs no surgery to the research driver — we only have to
# (a) restore the prior months' files onto disk before the driver runs, and
# (b) keep mirroring them up as each month finishes so a reclaim loses at most
# the in-progress month.
#
# These helpers use a DEDICATED `checkpoints/h{H}/` S3 prefix (separate from the
# final `results/h{H}/models/` artifacts) so the checkpoint is the live,
# during-run mirror and is cheap to overwrite. They are best-effort: a sync
# failure logs a warning and never aborts the (long, expensive) job.
# ---------------------------------------------------------------------------
def _ckpt_prefix(h: int) -> str:
    """S3 key prefix (no trailing slash) for this horizon's live checkpoint."""
    return _s3_key("checkpoints", f"h{h}")


# Files that constitute a per-month checkpoint. `.txt` (booster) +
# `_features.json` (manifest) are the resume-critical pair; summary.json and
# per-month shap csvs ride along so a resumed box has full context.
_CKPT_SUFFIXES = (".txt", ".json", ".csv")


def _ckpt_files(model_dir: Path) -> list[Path]:
    if not model_dir.exists():
        return []
    return sorted(
        p for p in model_dir.iterdir()
        if p.is_file() and p.suffix in _CKPT_SUFFIXES
    )


def checkpoint_sync_up(h: int, model_dir: Path) -> int:
    """Mirror completed-month checkpoint files output_dir -> S3 (idempotent).

    Uploads only files whose size differs from (or is absent in) S3, so a
    periodic call is cheap once the month set is stable. Returns the count
    uploaded. Best-effort: never raises (the job must not die on an S3 hiccup).
    """
    if not BUCKET:
        return 0
    prefix = _ckpt_prefix(h) + "/"
    # Build a {basename: size} map of what is already in S3 to skip no-op puts.
    remote: dict[str, int] = {}
    try:
        paginator = _s3().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                remote[obj["Key"][len(prefix):]] = obj["Size"]
    except Exception as e:  # noqa: BLE001
        log.warning("[ckpt] list failed (continuing): %s", e)
    n = 0
    for f in _ckpt_files(model_dir):
        try:
            if remote.get(f.name) == f.stat().st_size:
                continue  # already mirrored, unchanged
            _s3().upload_file(str(f), BUCKET, prefix + f.name)
            n += 1
        except Exception as e:  # noqa: BLE001
            log.warning("[ckpt] upload %s failed (continuing): %s", f.name, e)
    if n:
        log.info("[ckpt] synced %d checkpoint file(s) up -> s3://%s/%s", n, BUCKET, prefix)
    return n


def checkpoint_restore_down(h: int, model_dir: Path) -> int:
    """Restore prior-run checkpoint files S3 -> output_dir before the WF starts.

    After this, the walk-forward's native skip-if-exists guard reuses every
    already-completed month, so a relaunched box resumes mid-horizon. Returns
    the count restored. Best-effort.
    """
    if not BUCKET:
        return 0
    model_dir.mkdir(parents=True, exist_ok=True)
    prefix = _ckpt_prefix(h) + "/"
    n = 0
    try:
        paginator = _s3().get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
            for obj in page.get("Contents", []):
                rel = obj["Key"][len(prefix):]
                if not rel:
                    continue
                dst = model_dir / rel
                # Skip if a same-size local copy already exists (idempotent restart).
                if dst.exists() and dst.stat().st_size == obj["Size"]:
                    continue
                dst.parent.mkdir(parents=True, exist_ok=True)
                _s3().download_file(BUCKET, obj["Key"], str(dst))
                n += 1
    except Exception as e:  # noqa: BLE001
        log.warning("[ckpt] restore failed (starting fresh): %s", e)
    if n:
        log.info("[ckpt] restored %d checkpoint file(s) from s3://%s/%s "
                 "(completed months will be skipped)", n, BUCKET, prefix)
    else:
        log.info("[ckpt] no prior checkpoint at s3://%s/%s — fresh start", BUCKET, prefix)
    return n


class _CheckpointSyncer:
    """Daemon thread mirroring output_dir checkpoints to S3 every cadence.

    Also installs a SIGTERM/SIGINT handler that triggers a FINAL flush, so the
    box's `shutdown` (and the spot-interrupt poller's `kill`) drains the latest
    completed month before teardown. Idempotent: start()/stop() are safe to
    call even with no bucket (the smoke path) — it becomes a no-op.
    """

    def __init__(self, h: int, model_dir: Path, period_s: int = CKPT_SYNC_SECONDS):
        self.h = h
        self.model_dir = model_dir
        self.period_s = max(30, int(period_s))
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._prev_handlers: dict[int, object] = {}

    def _loop(self) -> None:
        while not self._stop.wait(self.period_s):
            try:
                checkpoint_sync_up(self.h, self.model_dir)
            except Exception as e:  # noqa: BLE001
                log.warning("[ckpt] background sync error (continuing): %s", e)

    def _on_signal(self, signum, _frame) -> None:  # noqa: ANN001
        log.warning("[ckpt] signal %s received — FINAL checkpoint flush before exit", signum)
        try:
            checkpoint_sync_up(self.h, self.model_dir)
        except Exception as e:  # noqa: BLE001
            log.warning("[ckpt] final flush error: %s", e)
        # Re-raise default behaviour so the process actually terminates.
        prev = self._prev_handlers.get(signum)
        if callable(prev):
            prev(signum, _frame)
        else:
            raise SystemExit(143 if signum == signal.SIGTERM else 130)

    def start(self) -> "_CheckpointSyncer":
        if not BUCKET:
            return self  # smoke / local-only: no S3, nothing to sync
        for sig in (signal.SIGTERM, signal.SIGINT):
            try:
                self._prev_handlers[sig] = signal.getsignal(sig)
                signal.signal(sig, self._on_signal)
            except (ValueError, OSError):
                # signal only works on the main thread; harmless if unavailable.
                pass
        self._thread = threading.Thread(target=self._loop, name="ckpt-sync", daemon=True)
        self._thread.start()
        log.info("[ckpt] background syncer started (every %ds -> s3://%s/%s)",
                 self.period_s, BUCKET, _ckpt_prefix(self.h) + "/")
        return self

    def stop(self, *, final_flush: bool = True) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=30)
        if final_flush and BUCKET:
            try:
                checkpoint_sync_up(self.h, self.model_dir)
            except Exception as e:  # noqa: BLE001
                log.warning("[ckpt] stop() final flush error: %s", e)


# ---------------------------------------------------------------------------
# Path resolution — reuse the repo's own per-horizon path helpers so the
# files this driver builds land exactly where the pipeline scripts read them.
# (Importing vol_arb.paths is cheap and pure; verified clean in Investigate.)
# ---------------------------------------------------------------------------
def _paths(asset: str, h: int) -> dict[str, Path]:
    from vol_arb.paths import (
        CURRENCY_FOR_ASSET,
        har_pd_model_dir,
        har_pd_predictions_path,
        svi_features_path,
        trade_log_path,
    )
    currency = CURRENCY_FOR_ASSET[asset]
    tsrv = REPO_ROOT / "TSRV" / "output" / f"{asset}_tsrv_{h}m.parquet"
    har_pd_feat = REPO_ROOT / "vol_model" / "output" / f"{asset}_features_har_pd_{h}m.parquet"
    svi = REPO_ROOT / svi_features_path(currency, horizon_minutes=h)
    # walk-forward --no-transvol output dir is the har_pd dir tagged 'no_tv'.
    model_dir = REPO_ROOT / har_pd_model_dir(asset, horizon_minutes=h)
    model_dir = model_dir.with_name(model_dir.name + "_no_tv")
    # extract writes the standard per-horizon predictions path; --no-transvol
    # carries no path suffix of its own, so we route extract to an explicit
    # output to avoid colliding with any TransVol-included artifact.
    preds = REPO_ROOT / har_pd_predictions_path(asset, horizon_minutes=h)
    preds = preds.with_name(preds.stem + "_no_tv" + preds.suffix)
    trade_log = REPO_ROOT / trade_log_path(currency, horizon_minutes=h, variant="v2",
                                           suffix=f"horizon_eval_h{h}m")
    return {
        "currency": Path(currency),  # wrapped only to keep dict[str,Path] type
        "tsrv": tsrv,
        "har_pd_feat": har_pd_feat,
        "svi": svi,
        "model_dir": model_dir,
        "summary_json": model_dir / "summary.json",
        "preds": preds,
        "trade_log": trade_log,
    }


# ---------------------------------------------------------------------------
# Step 0.5: stage a locally-downloaded data shard into the repo layout.
# The launcher's user_data does `aws s3 cp .../data/h{H}/ /tmp/shard/` BEFORE pip
# (the download-before-pip ordering fix), then passes --shard-dir /tmp/shard. This
# copies those files into the per-tool output dirs the path helpers read, so the
# pipeline finds them on disk and never re-downloads from S3 (faster, and works
# even if the box's boto3 creds lag). Missing files are fine — the ensure_* steps
# fall back to an S3 pull or a rebuild. Idempotent: skips files already present.
# ---------------------------------------------------------------------------
def stage_shard(h: int, paths: dict[str, Path], shard_dir: Path) -> None:
    if not shard_dir or not shard_dir.exists():
        log.info("[step0.5] no shard-dir to stage (%s) — relying on S3/rebuild", shard_dir)
        return
    # Map staged filenames -> their destination repo path. The staging uploader
    # (stage_data_to_s3.py) keys per-horizon feature files by basename under
    # data/h{H}/features/<name> but the launcher flattens them into /tmp/shard,
    # so match by basename against the 4 files this horizon needs.
    wanted = {p.name: p for p in (paths["tsrv"], paths["har_pd_feat"], paths["svi"])}
    # the engineered feature cache — assemble_features (vol_model/pipeline.py:52) names it by
    # TF-LABEL: '{min}m' below 1h but '{hours}h' at/above (e.g. h480 -> _8h, NOT _480m). This MUST
    # match the launcher download key (volmodel_env.horizon_feature_keys / tf_label) and the staged
    # S3 object, or the basename lookup below skips it -> "No cached features found".
    _asset = paths["tsrv"].name.split("_tsrv_")[0]
    _tf = f"{h}m" if h < 60 else f"{h // 60}h"
    feat_cache = REPO_ROOT / "vol_model" / "output" / f"{_asset}_features_{_tf}.parquet"
    wanted[feat_cache.name] = feat_cache
    n = 0
    for src in sorted(shard_dir.rglob("*")):
        if not src.is_file():
            continue
        dst = wanted.get(src.name)
        if dst is None:
            # shared csvs (pruned_features.csv etc.) -> vol_model/output by name
            if src.suffix == ".csv":
                dst = REPO_ROOT / "vol_model" / "output" / src.name
            else:
                continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists():
            continue
        shutil.copy2(src, dst)
        n += 1
        log.info("[step0.5] staged %s -> %s", src.name, dst)
    log.info("[step0.5] staged %d shard file(s) from %s", n, shard_dir)


# ---------------------------------------------------------------------------
# Step 1: ensure TSRV bars at H
# ---------------------------------------------------------------------------
def ensure_tsrv(asset: str, h: int, paths: dict[str, Path], *, allow_rebuild: bool) -> None:
    if paths["tsrv"].exists():
        log.info("[step1] TSRV bars present: %s", paths["tsrv"].name)
        return
    # Try to pull a pre-staged bar from S3 (the launcher stages per-horizon shards).
    if _s3_download(_s3_key("data", f"h{h}", paths["tsrv"].name), paths["tsrv"]):
        return
    if not allow_rebuild:
        raise FileNotFoundError(
            f"TSRV bars missing and rebuild disabled: {paths['tsrv']}. "
            f"For h in {sorted(PRESTAGED_TSRV_HORIZONS)} these are pre-staged; "
            f"stage the shard or pass --allow-tsrv-rebuild (reads 191GB aggTrades)."
        )
    log.warning("[step1] REBUILDING TSRV at h=%dm from raw aggTrades (slow!)", h)
    _py("TSRV/run_tsrv_series.py", "--asset", asset, "--tf", str(h * 60), "--no-plot",
        log_path=paths["model_dir"].parent / f"tsrv_build_h{h}m.log")


# ---------------------------------------------------------------------------
# Step 2: HAR-PD features at H
# ---------------------------------------------------------------------------
def ensure_har_pd(asset: str, h: int, paths: dict[str, Path]) -> None:
    if paths["har_pd_feat"].exists():
        log.info("[step2] HAR-PD features present: %s", paths["har_pd_feat"].name)
        return
    if _s3_download(_s3_key("data", f"h{h}", paths["har_pd_feat"].name), paths["har_pd_feat"]):
        return
    if 1440 % h != 0:
        raise ValueError(f"h={h} does not divide 1440; har_pd requires a divisor horizon.")
    _py("vol_model/features/har_pd.py", "--asset", asset,
        "--horizon-minutes", str(h), "--tsrv", str(paths["tsrv"]),
        log_path=paths["model_dir"].parent / f"har_pd_build_h{h}m.log")


# ---------------------------------------------------------------------------
# Step 3: SVI features at H
# ---------------------------------------------------------------------------
def ensure_svi(asset: str, h: int, paths: dict[str, Path]) -> None:
    if paths["svi"].exists():
        log.info("[step3] SVI features present: %s", paths["svi"].name)
        return
    if _s3_download(_s3_key("data", f"h{h}", paths["svi"].name), paths["svi"]):
        return
    currency = str(paths["currency"])
    log.warning("[step3] BUILDING SVI at h=%dm (reads data/deribit; minutes)", h)
    _py("vol_arb/run_pipeline.py", "--currency", currency, "--svi",
        "--bar-minutes", str(h), "--svi-bucket-freq", f"{h}min",
        log_path=paths["model_dir"].parent / f"svi_build_h{h}m.log")
    if not paths["svi"].exists():
        raise FileNotFoundError(
            f"SVI build finished but expected loader file is absent: {paths['svi']}. "
            f"(h=15 uses legacy svi_features_15m.parquet; others svi_features_h{h}m.parquet.)"
        )


# ---------------------------------------------------------------------------
# Step 4: walk-forward forecast (the long job). Idempotent at month grain.
# Restore any prior per-month .txt checkpoints from S3 first so a spot
# relaunch resumes instead of restarting (Investigate risk #7).
# ---------------------------------------------------------------------------
def run_walkforward(asset: str, h: int, paths: dict[str, Path],
                    start: str, end: str, trials: int, patience: int) -> None:
    model_dir = paths["model_dir"]
    model_dir.mkdir(parents=True, exist_ok=True)
    # RESUME (spot-resilience step a): restore prior per-month checkpoints from
    # the live checkpoint prefix so the walk-forward's native skip-if-exists
    # guard reuses every completed month. Then ALSO pull any final-results
    # models from a previously-completed run (back-compat with the old layout).
    if BUCKET:
        checkpoint_restore_down(h, model_dir)
        _sync_models_down(h, model_dir)
    # CHECKPOINT (spot-resilience step b): mirror output_dir -> checkpoint prefix
    # every CKPT_SYNC_SECONDS during the run, plus a SIGTERM/SIGINT-triggered
    # FINAL flush (the spot-interrupt poller `kill`s us; the box `shutdown` sends
    # SIGTERM). Each completed {month}.txt is captured within one cadence, so a
    # reclaim loses at most the in-progress month.
    syncer = _CheckpointSyncer(h, model_dir).start()
    # Pass --output-dir EXPLICITLY so the walk-forward writes exactly where this
    # driver's _paths() expects. Without it the script's own --tag branch
    # (run_walkforward...:339-341) resolves a HORIZON-AGNOSTIC dir
    # (expanding_pruned_svi_v1_har_pd_v1_no_tv) that (a) does not match the
    # per-horizon model_dir _paths() computes, and (b) would COLLIDE across
    # horizons. Forcing --output-dir removes both the path mismatch and the
    # cross-horizon clobber in one move.
    try:
        # Fleet flags: pass exactly `--no-transvol --drop-orphans`. Orphan-drop is the
        # SHAP-consistency tidy (apply-regardless); [1.2] --prune-collinear and [1.3]
        # --monotone-constraints were REJECTED (neutral-to-negative OOS QLIKE, per-year
        # stable in the wrong direction at h=60) and are NOT passed. The extract step needs
        # no matching flag: it reorders X to each month's saved `_features.json` manifest
        # (extract_pruned_svi_har_pd_predictions.py:461,471), so the 52-feature trained set
        # is honored implicitly from the model checkpoint.
        _py("vol_model/run_walkforward_pruned_svi_har_pd.py",
            "--asset", asset, "--tf", str(h * 60), "--no-transvol", "--drop-orphans",
            "--start", start, "--end", end,
            "--max-trials", str(trials), "--patience", str(patience),
            "--output-dir", str(model_dir),
            log_path=model_dir.parent / f"walkforward_h{h}m.log")
    finally:
        # Always drain the latest completed months up, even on failure/interrupt.
        syncer.stop(final_flush=True)
    if not paths["summary_json"].exists():
        raise FileNotFoundError(
            f"walk-forward finished but summary.json absent: {paths['summary_json']}"
        )


def _sync_models_down(h: int, model_dir: Path) -> None:
    """Best-effort resume: list+download prior result-shard model checkpoints."""
    if not BUCKET:
        return
    prefix = _s3_key("results", f"h{h}", "models") + "/"
    paginator = _s3().get_paginator("list_objects_v2")
    n = 0
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        for obj in page.get("Contents", []):
            rel = obj["Key"][len(prefix):]
            if not rel:
                continue
            dst = model_dir / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            if not dst.exists():
                _s3().download_file(BUCKET, obj["Key"], str(dst))
                n += 1
    if n:
        log.info("[step4] resumed %d prior checkpoint files from S3", n)


# ---------------------------------------------------------------------------
# Step 5a: extract per-bar OOS predictions
# ---------------------------------------------------------------------------
def extract_predictions(asset: str, h: int, paths: dict[str, Path],
                        start: str, end: str) -> None:
    if paths["preds"].exists():
        log.info("[step5a] predictions present: %s", paths["preds"].name)
        return
    # The extract resolves the --no-transvol model dir via tag 'no_tv'; we pass
    # both --model-dir and --output explicitly to remove all ambiguity. --no-transvol
    # mirrors the walk-forward so extract skips the .pt lookup (never staged on the
    # box) and runs torch-free.
    _py("vol_model/extract_pruned_svi_har_pd_predictions.py",
        "--asset", asset, "--tf", str(h * 60), "--no-transvol",
        "--start", start, "--end", end,
        "--model-dir", str(paths["model_dir"]),
        "--output", str(paths["preds"]),
        log_path=paths["model_dir"].parent / f"extract_h{h}m.log")


# ---------------------------------------------------------------------------
# Step 5b: forecast-quality scoring (QLIKE / OOS R^2 / RankIC, per-year)
# QLIKE on log-vol target == vol_model/model.py qlike_loss:
#   mean(exp(2*(y_true-y_pred)) - 2*(y_true-y_pred) - 1)
# ---------------------------------------------------------------------------
def _qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    d = y_true - y_pred
    return float(np.mean(np.exp(2.0 * d) - 2.0 * d - 1.0))


def _rank_ic(y_true: pd.Series, y_pred: pd.Series) -> float:
    if len(y_true) < 3:
        return float("nan")
    return float(y_true.corr(y_pred, method="spearman"))


def _oos_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")


def score_forecast(paths: dict[str, Path]) -> tuple[dict, pd.DataFrame]:
    df = pd.read_parquet(paths["preds"])
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    if df.index.tz is not None:  # to_period drops tz with a warning; normalize once
        df.index = df.index.tz_localize(None)
    df = df.dropna(subset=["y_pred", "y_true"]).sort_index()
    yt, yp = df["y_true"], df["y_pred"]

    # Per-month rollup (matches the walk-forward grain).
    rows = []
    for period, g in df.groupby(df.index.to_period("M")):
        rows.append({
            "month": str(period),
            "year": period.year,
            "n": len(g),
            "qlike": _qlike(g["y_true"].values, g["y_pred"].values),
            "rank_ic": _rank_ic(g["y_true"], g["y_pred"]),
            "oos_r2": _oos_r2(g["y_true"].values, g["y_pred"].values),
        })
    per_month = pd.DataFrame(rows)

    # Per-year rollup is the anti-overfit safeguard.
    per_year = []
    for year, g in df.groupby(df.index.year):
        per_year.append({
            "year": int(year),
            "n_months": int(df[df.index.year == year].index.to_period("M").nunique()),
            "qlike": _qlike(g["y_true"].values, g["y_pred"].values),
            "rank_ic": _rank_ic(g["y_true"], g["y_pred"]),
            "oos_r2": _oos_r2(g["y_true"].values, g["y_pred"].values),
        })

    forecast = {
        "n_bars": int(len(df)),
        "n_months": int(len(per_month)),
        "qlike_oos_mean": float(per_month["qlike"].mean()) if len(per_month) else float("nan"),
        "qlike_oos_median": float(per_month["qlike"].median()) if len(per_month) else float("nan"),
        "qlike_pooled": _qlike(yt.values, yp.values),
        "rank_ic_mean": float(per_month["rank_ic"].mean()) if len(per_month) else float("nan"),
        "oos_r2_pooled": _oos_r2(yt.values, yp.values),
        "per_year": per_year,
    }
    return forecast, per_month


# ---------------------------------------------------------------------------
# Step 5c: downstream V2 short-vol backtest hook (NC metrics, per-year).
# Replays the historical strategy via run_backtest.py --historical-btc-v2 with this
# horizon's predictions. honest_vega_iv stays DEFAULT-ON (Phase-6 fix).
# Backtest failure is NON-FATAL: a horizon's forecast scorecard still ships
# even if the h=15-frozen V2 thresholds produce no/empty trades at H.
# ---------------------------------------------------------------------------
def _assert_predictions_identity(trade_log_path: Path, preds_path: Path) -> None:
    """HARD MF-3 guard: prove the backtest actually consumed THESE predictions.

    A path-string compare is insufficient — at h=15 the production config
    legitimately points at the aligned_bias file, not the job's ``_no_tv``
    file, and the per-horizon factory used to silently substitute a stale
    aligned_bias sidecar at h!=15. The only honest check is value-level: join
    the intended predictions' ``y_pred`` onto the active trades and require
    exact-match coverage over every active bar. Any gap (path swapped, file
    missing-then-swallowed, horizon mismatch) makes coverage < 1.0 and we FAIL
    LOUD rather than ship a silently-wrong NC-Sharpe column.
    """
    tl = pd.read_parquet(trade_log_path)
    if not isinstance(tl.index, pd.DatetimeIndex):
        tl.index = pd.to_datetime(tl.index)
    active = tl[tl["signal"] != 0] if "signal" in tl.columns else tl
    if active.empty:
        # No trades fired (V2 thresholds h15-frozen). Nothing to verify — the
        # empty-trade case is handled informatively downstream, not a defect.
        return

    preds = pd.read_parquet(preds_path)
    if not isinstance(preds.index, pd.DatetimeIndex):
        preds.index = pd.to_datetime(preds.index)
    if "y_pred" not in preds.columns:
        raise RuntimeError(
            f"[step5c] predictions file {preds_path.name} has no 'y_pred' column "
            f"— cannot verify the backtest consumed the intended predictions."
        )

    # Normalise tz so the join is apples-to-apples (the backtester writes a
    # tz-aware index; predictions may be naive or tz-aware).
    a_idx = active.index
    p_idx = preds.index
    if a_idx.tz is not None:
        a_idx = a_idx.tz_localize(None)
    if p_idx.tz is not None:
        p_idx = p_idx.tz_localize(None)
    intended = pd.Series(preds["y_pred"].to_numpy(), index=p_idx)
    intended = intended[~intended.index.duplicated(keep="last")]

    joined = intended.reindex(a_idx)
    coverage = float(joined.notna().mean())
    if coverage < 0.999:
        n_miss = int(joined.isna().sum())
        raise RuntimeError(
            f"[step5c] PREDICTIONS-IDENTITY FAILURE: the V2 backtest at h={trade_log_path} "
            f"consumed predictions that do NOT match the intended file "
            f"{preds_path.name} — only {coverage:.4%} of {len(a_idx)} active-trade "
            f"bars have a matching y_pred timestamp ({n_miss} unmatched). This is "
            f"the MF-3 discard bug (stale aligned_bias / wrong-horizon sidecar). "
            f"Refusing to ship a silently-wrong NC-Sharpe."
        )


def score_downstream(asset: str, h: int, paths: dict[str, Path]) -> dict:
    from vol_arb.paths import CURRENCY_FOR_ASSET
    currency = CURRENCY_FOR_ASSET[asset]
    try:
        if not paths["trade_log"].exists():
            # BARE V2 COMPARISON STACK (MF-3): drive the V2 short-vol book with
            # the FRESHLY-TRAINED ``_no_tv`` predictions and turn OFF the
            # h15-tuned overlays the job does not regenerate per horizon —
            # Kelly (``--kelly-mult-path none``) and the aligner (the default
            # predictions_path is the aligned_bias file; passing --predictions-path
            # overrides it). Including those overlays would confound the
            # cross-horizon NC-Sharpe comparison; overlay-free is the honest
            # footing for deploy-horizon selection.
            _py("vol_arb/run_backtest.py", "--currency", currency,
                "--historical-btc-v2", "--horizon-minutes", str(h),
                "--predictions-path", str(paths["preds"]),
                "--kelly-mult-path", "none",
                "--output", str(paths["trade_log"]),
                log_path=paths["model_dir"].parent / f"backtest_h{h}m.log")
    except Exception as e:  # noqa: BLE001 — the *run* is a best-effort hook
        # A backtest that produces NO trades (V2 thresholds are h=15-frozen) is
        # informative, not a forecast defect — degrade gracefully here.
        log.warning("[step5c] downstream backtest failed (non-fatal): %s", e)
        return {
            "status": "failed",
            "error": str(e),
            "note": "V2 thresholds are h=15-frozen; absence of trades at H is "
                    "informative, not a forecast defect (feedback_thorough_experiments).",
        }
    # HARD identity guard — OUTSIDE the swallowing catch so it FAILS LOUD
    # (MF-3). A predictions mismatch is a correctness bug, not a benign
    # no-trades outcome; it must crash the shard, never become a quiet
    # ``status: failed``.
    _assert_predictions_identity(paths["trade_log"], paths["preds"])
    return _score_trade_log(paths["trade_log"])


def _score_trade_log(trade_log_path: Path) -> dict:
    """Score the V2 short-vol trade log in NON-COMPOUND terms.

    Contract fix: ``vol_arb.metrics.{risk_adjusted,drawdown,return}_metrics``
    each expect the trade-log FRAME (they read ``net_pnl``/``equity`` columns
    internally and do their own daily resampling) — NOT a pre-resampled daily
    Series. Passing a Series raised ``KeyError 'net_pnl'`` and silently blanked
    the downstream block. The honest non-compound view is produced by
    ``build_nc_view`` (renames ``net_pnl_nc -> net_pnl`` etc.); every metric is
    then called with ``is_nc=True`` so the NC formulas (additive equity, MDD as
    % of fixed notional) apply. The backtester
    writes a tz-aware ``DatetimeIndex``, so no ts-column reconstruction is
    needed.
    """
    from vol_arb.metrics import (
        build_nc_view,
        drawdown_metrics,
        return_metrics,
        risk_adjusted_metrics,
    )

    tl = pd.read_parquet(trade_log_path)
    if not isinstance(tl.index, pd.DatetimeIndex):
        tl.index = pd.to_datetime(tl.index)
    if tl.empty:
        return {"status": "empty", "n_trades": 0,
                "note": "trade log is empty (no bars produced)"}

    # Active trades only (signal != 0). If there's no signal column we can't
    # build the V2 NC view; degrade to a raw-PnL summary rather than crash.
    if "signal" not in tl.columns:
        pnl_col = next((c for c in ("net_pnl_nc", "net_pnl") if c in tl.columns), None)
        if pnl_col is None:
            return {"status": "empty", "n_trades": int(len(tl)),
                    "note": "trade log missing both 'signal' and a net_pnl column"}
        return {"status": "ok_raw", "n_trades": int(len(tl)),
                "nc_pnl_usd": float(tl[pnl_col].sum()),
                "note": "no 'signal' column — raw PnL sum only (NC view unavailable)"}

    active = tl[tl["signal"] != 0].copy()
    if active.empty:
        return {"status": "empty", "n_trades": 0,
                "note": "no active trades at this horizon (V2 thresholds h=15-frozen)"}
    if "net_pnl_nc" not in active.columns:
        # report_non_compounding was off; fall back to compound net_pnl as a frame.
        if "net_pnl" not in active.columns:
            return {"status": "empty", "n_trades": int(len(active)),
                    "note": "trade log has neither net_pnl_nc nor net_pnl"}
        nc = active
        is_nc = False
        pnl_col = "net_pnl"
    else:
        nc = build_nc_view(active)  # renames *_nc -> standard; pass is_nc=True
        is_nc = True
        pnl_col = "net_pnl"

    equity0 = 100_000.0
    ra = risk_adjusted_metrics(nc, initial_equity=equity0, is_nc=is_nc)
    dd = drawdown_metrics(nc, initial_equity=equity0, is_nc=is_nc)
    rm = return_metrics(nc, initial_equity=equity0, is_nc=is_nc)

    # short_frac on the V2 short-vol book (signal == -1 is a short straddle).
    short_frac = float((active["signal"] < 0).mean())

    per_year = []
    for year, g in nc.groupby(nc.index.year):
        gra = risk_adjusted_metrics(g, initial_equity=equity0, is_nc=is_nc)
        per_year.append({"year": int(year),
                         "nc_sharpe": float(gra.get("sharpe_daily", float("nan"))),
                         "nc_pnl_usd": float(g[pnl_col].sum()),
                         "n_trades": int(len(g))})

    max_dd_pct = float(dd.get("max_drawdown_pct", float("nan")))
    return {
        "status": "ok",
        "honest_vega_iv": True,
        "params_frozen_at": "h15_v2",
        "is_nc": is_nc,
        "n_trades": int(len(active)),
        "short_frac": short_frac,
        "nc_sharpe": float(ra.get("sharpe_daily", float("nan"))),
        "nc_pnl_usd": float(nc[pnl_col].sum()),
        # MDD as % of the fixed $100K reference -> USD (NC convention).
        "nc_max_dd_usd": float(max_dd_pct / 100.0 * equity0),
        "nc_max_dd_pct": max_dd_pct,
        "nc_total_return_pct": float(rm.get("total_return_pct", float("nan"))),
        "per_year_nc_sharpe": per_year,
    }


# ---------------------------------------------------------------------------
# Step 6: assemble + upload the result shard
# ---------------------------------------------------------------------------
def assemble_shard(asset: str, h: int, paths: dict[str, Path],
                   forecast: dict, downstream: dict) -> dict:
    keyp = f"results/h{h}"
    shard = {
        "horizon_minutes": h,
        "tf_seconds": h * 60,
        "asset": asset,
        "no_transvol": True,
        "git_sha": _git_sha(),
        "completed_utc": _utcnow_iso(),
        "forecast": forecast,
        "downstream": downstream,
        "downstream_hook_note": (
            "Finer (e.g. h=5) forecast updates edge=rv_pred-iv_used more often -> "
            "more short-vol VRP-carry entries / higher turnover for the directional "
            "book sizing (|pred_edge|). Coarser (h=240/1440) forecast -> fewer, "
            "higher-conviction entries feeding daily short-vol VRP carry conditioning."
        ),
        "artifacts": {
            "summary_json_key": _s3_key(keyp, "summary.json"),
            "per_month_parquet_key": _s3_key(keyp, "per_month.parquet"),
            "predictions_parquet_key": _s3_key(keyp, "predictions.parquet"),
            "trade_log_parquet_key": _s3_key(keyp, "trade_log.parquet"),
            "models_prefix": _s3_key(keyp, "models") + "/",
        },
    }
    return shard


def upload_results(h: int, paths: dict[str, Path], shard: dict,
                   per_month: pd.DataFrame) -> None:
    keyp = f"results/h{h}"
    out_dir = paths["model_dir"].parent / f"_shard_h{h}m"
    out_dir.mkdir(parents=True, exist_ok=True)

    shard_local = out_dir / "shard.json"
    shard_local.write_text(json.dumps(shard, indent=2, default=str))
    per_month_local = out_dir / "per_month.parquet"
    per_month.to_parquet(per_month_local)

    log.info("SHARD h=%dm:\n%s", h, json.dumps(shard, indent=2, default=str))

    # Upload comparable scorecard + artifacts (idempotent; overwrites OK).
    _s3_upload(shard_local, _s3_key(keyp, "shard.json"))
    _s3_upload(per_month_local, _s3_key(keyp, "per_month.parquet"))
    _s3_upload(paths["preds"], _s3_key(keyp, "predictions.parquet"))
    _s3_upload(paths["summary_json"], _s3_key(keyp, "summary.json"))
    if paths["trade_log"].exists():
        _s3_upload(paths["trade_log"], _s3_key(keyp, "trade_log.parquet"))
    # Per-month LGBM checkpoints (enables relaunch resume; Investigate risk #7).
    if paths["model_dir"].exists():
        for txt in sorted(paths["model_dir"].glob("*.txt")):
            _s3_upload(txt, _s3_key(keyp, "models", txt.name))
    # Final mirror to the live checkpoint prefix too, so a (rare) relaunch that
    # races the shard write still resumes from a complete month set rather than
    # the partial state of the previous background flush.
    checkpoint_sync_up(h, paths["model_dir"])


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, required=True,
                    help="Horizon H in MINUTES (5/15/30/60/240/1440; must divide 1440).")
    ap.add_argument("--asset", default=ASSET)
    ap.add_argument("--start", default=WF_START_DEFAULT)
    ap.add_argument("--end", default=WF_END_DEFAULT)
    ap.add_argument("--max-trials", type=int, default=WF_TRIALS_DEFAULT)
    ap.add_argument("--patience", type=int, default=WF_PATIENCE_DEFAULT)
    ap.add_argument("--allow-tsrv-rebuild", action="store_true",
                    help="Permit rebuilding TSRV bars from the 191GB raw aggTrades "
                         "(needed only for a non-pre-staged horizon).")
    ap.add_argument("--smoke", action="store_true",
                    help="3-fold smoke: forces start=2022-02 end=2022-04, "
                         "max-trials=10, patience=5 (fast end-to-end proof).")
    ap.add_argument("--skip-downstream", action="store_true",
                    help="Skip the V2 backtest hook (forecast-only shard).")
    ap.add_argument("--shard-dir", default=None,
                    help="Local dir holding this horizon's pre-downloaded data shard "
                         "(the launcher's `aws s3 cp .../data/h{H}/ /tmp/shard/`). "
                         "Files are copied into the repo layout before the pipeline runs.")
    # Optional config overrides. Primary config is via env vars (QUANT_BUCKET /
    # QUANT_PREFIX / QUANT_REGION / QUANT_ASSET), which the launcher EXPORTS in
    # user_data; these flags let the same command line be driven without env too.
    ap.add_argument("--bucket", default=None, help="S3 bucket (overrides QUANT_BUCKET).")
    ap.add_argument("--prefix", default=None, help="S3 prefix (overrides QUANT_PREFIX).")
    ap.add_argument("--region", default=None, help="AWS region (overrides QUANT_REGION).")
    ap.add_argument("--run-id", default=None, help="Run id tag (informational; for log correlation).")
    args = ap.parse_args()

    # Apply optional config overrides onto the module globals (env vars are the
    # primary path; flags win when both are given).
    global BUCKET, PREFIX, REGION
    if args.bucket is not None:
        BUCKET = args.bucket.strip()
    if args.prefix is not None:
        PREFIX = args.prefix.strip("/")
    if args.region is not None:
        REGION = args.region

    h = args.horizon
    if 1440 % h != 0:
        log.error("h=%d does not divide 1440 — har_pd would hard-error.", h)
        return 2

    start, end = args.start, args.end
    trials, patience = args.max_trials, args.patience
    if args.smoke:
        start, end, trials, patience = "2022-02", "2022-04", 10, 5
        log.info("SMOKE MODE: %s..%s trials=%d patience=%d", start, end, trials, patience)

    log.info("=== horizon job h=%dm (tf=%ds) asset=%s run=%s ===",
             h, h * 60, args.asset, args.run_id or "(none)")
    log.info("bucket=%s prefix=%s region=%s git=%s",
             BUCKET or "(local-only)", PREFIX, REGION, _git_sha())

    paths = _paths(args.asset, h)
    paths["model_dir"].parent.mkdir(parents=True, exist_ok=True)

    # --- Step 0: idempotent skip if the shard already exists in S3 ---------
    shard_key = _s3_key("results", f"h{h}", "shard.json")
    if not args.smoke and _s3_exists(shard_key):
        log.info("[step0] result shard already present (s3://%s/%s) — SKIP.",
                 BUCKET, shard_key)
        return 0

    t0 = time.time()
    if args.shard_dir:
        stage_shard(h, paths, Path(args.shard_dir))
    ensure_tsrv(args.asset, h, paths, allow_rebuild=args.allow_tsrv_rebuild)
    ensure_har_pd(args.asset, h, paths)
    ensure_svi(args.asset, h, paths)
    run_walkforward(args.asset, h, paths, start, end, trials, patience)
    extract_predictions(args.asset, h, paths, start, end)

    forecast, per_month = score_forecast(paths)
    downstream = ({"status": "skipped"} if args.skip_downstream
                  else score_downstream(args.asset, h, paths))

    shard = assemble_shard(args.asset, h, paths, forecast, downstream)
    upload_results(h, paths, shard, per_month)

    log.info("=== DONE h=%dm in %.1f min ===", h, (time.time() - t0) / 60.0)
    return 0


if __name__ == "__main__":
    sys.exit(main())
