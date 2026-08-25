#!/usr/bin/env python3
"""
Shared constants + S3 key conventions for the per-horizon vol-forecast AWS-spot stack.

Single source of truth for BUCKET / PREFIX / REGION / PROFILE / INSTANCE_PROFILE and the
result-shard / code / data key layout, so the launcher, the on-box driver (run_horizon_job.py),
the IAM setup/verify scripts, and the babysitter all agree.

The QUANT_* namespace is the SINGLE source of truth — the core constants are
re-exported from _config.py so the launcher, driver, stage/verify scripts, IAM setup, and the
babysitter/watch PowerShell scripts never drift. ``QUANT_BUCKET`` is required;
operational defaults remain deployment-neutral:
  * region eu-central-1.
  * worker instance-profile 'quant-worker' (created later by setup_quant_iam.py with admin creds).
  * only one [default] profile exists, so PROFILE defaults to "" (use default creds), not a name.

The 191GB raw aggTrades is NOT needed: TSRV-at-H bars already exist on disk, so each box ships only
its horizon's 4-file data shard (TSRV bar + engineered cache + har_pd sidecar + SVI features) plus
two tiny shared csvs. See the workflow investigation findings.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# --- core constants (env-overridable) ---
# SINGLE NAMESPACE: every script in this stack reads QUANT_* env vars (the launcher,
# driver, stage, verify, IAM setup, and the babysitter/watch PowerShell scripts). The
# old VOLMODEL_* namespace was removed to end the launcher<->driver inconsistency the
# build flagged. The authoritative source for BUCKET/PREFIX/REGION/
# INSTANCE_PROFILE is _config.py; this module re-exports them and adds the per-horizon
# S3 key helpers the launcher/babysitter need.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import _config as _cfg  # noqa: E402

BUCKET = _cfg.BUCKET
PREFIX = _cfg.PREFIX
REGION = _cfg.REGION
INSTANCE_PROFILE = _cfg.WORKER_PROFILE
# empty string => boto3 default creds (the single [default] profile = labelgen-worker)
PROFILE = os.environ.get("QUANT_PROFILE", "") or None

# Horizons (minutes). Constraint: har_pd requires 1440 % H == 0 (vol_model/features/har_pd.py:172),
# so only divisors of 1440 are valid. The forecast target is log(RV) over one H-bar, so QLIKE/R^2
# are cross-horizon comparable. tf passed to the walk-forward is SECONDS = H*60.
HORIZON_MINUTES = (5, 15, 30, 60, 240, 480, 1440)


def tf_seconds(horizon_minutes: int) -> int:
    """walk-forward --tf is in SECONDS; har_pd --horizon-minutes / run_pipeline --bar-minutes are MINUTES."""
    return horizon_minutes * 60


def tf_label(horizon_minutes: int) -> str:
    """Engineered-feature-cache timeframe label. MUST match vol_model/pipeline.py:52-53
    (assemble_features): sub-hour horizons are '{min}m', >=1h are '{hours}h'. So 60m->'1h',
    240m->'4h', 480m->'8h' (NOT '60m'/'240m'/'480m'). The har_pd/SVI/tsrv keys keep the '{H}m'
    convention; ONLY the engineered cache uses this label."""
    s = horizon_minutes * 60
    return f"{s // 60}m" if s < 3600 else f"{s // 3600}h"


# --- S3 key conventions (used by launcher + on-box driver + babysitter) ---
def code_key(prefix: str = PREFIX) -> str:
    return f"{prefix}/code/volcode.tar.gz"


def result_shard_key(prefix: str, horizon: int) -> str:
    """The authoritative completion signal for a horizon (idempotent skip checks this key)."""
    return f"{prefix}/results/h{horizon}/shard.json"


def results_prefix(prefix: str, horizon: int) -> str:
    return f"{prefix}/results/h{horizon}"


def shared_data_prefix(prefix: str = PREFIX) -> str:
    """Tiny shared csvs (pruned_features.csv, BTCUSDT_drop_features.csv) -> vol_model/output/."""
    return f"{prefix}/data/shared"


# ---------------------------------------------------------------------------
# S3 DATA LAYOUT — SINGLE SOURCE OF TRUTH (must match scripts/aws/stage_data_to_s3.py).
#
# stage_data_to_s3.py uploads a FLAT tree (NOT per-horizon shards):
#   {prefix}/data/tsrv_output/{asset}_tsrv_{H}m.parquet   <- DATA_DIRS ("TSRV/output","tsrv_output")
#   {prefix}/data/deribit/...                              <- DATA_DIRS ("data/deribit","deribit")
#   {prefix}/data/features/{asset}_features_{H}m.parquet   <- DATA_GLOBS (engineered caches)
#   {prefix}/data/features/{asset}_features_har_pd_{H}m.parquet  <- DATA_GLOBS (har_pd sidecars)
#   {prefix}/data/features/{svi_filename(H)}               <- DATA_GLOBS (SVI sidecars)
#   {prefix}/data/shared/{pruned_features.csv,BTCUSDT_drop_features.csv}  <- DATA_FILES
#
# The helpers below are the ONE place that names those flat keys; the launcher's
# user_data download block imports them so the launcher and the staging uploader
# can never drift again (the original bug: the launcher invented data/h{H}/ +
# data/shared/ shards that stage_data_to_s3.py never produced -> empty /tmp/shard
# -> "TSRV bars missing" crash). Keep this block and DATA_DIRS/DATA_GLOBS/DATA_FILES
# in stage_data_to_s3.py in lock-step.
# ---------------------------------------------------------------------------
def tsrv_output_prefix(prefix: str = PREFIX) -> str:
    return f"{prefix}/data/tsrv_output"


def deribit_prefix(prefix: str = PREFIX) -> str:
    return f"{prefix}/data/deribit"


def features_prefix(prefix: str = PREFIX) -> str:
    return f"{prefix}/data/features"


def svi_filename(horizon: int) -> str:
    """SVI naming asymmetry: h=15 uses the legacy unsuffixed name; all others use svi_features_h{H}m.

    Mirrors vol_arb.paths.svi_features_path(currency='BTC') exactly.
    """
    return "svi_features_15m.parquet" if horizon == 15 else f"svi_features_h{horizon}m.parquet"


def tsrv_bar_key(prefix: str, horizon: int, asset: str = "BTCUSDT") -> str:
    """This horizon's TSRV bar (the file whose absence crashed the smoke box)."""
    return f"{tsrv_output_prefix(prefix)}/{asset}_tsrv_{horizon}m.parquet"


def horizon_feature_keys(prefix: str, horizon: int, asset: str = "BTCUSDT") -> list[str]:
    """The 3 per-horizon files under data/features/ this horizon's job reads if pre-built:
    engineered cache, har_pd sidecar, SVI sidecar. Missing ones are rebuilt on-box."""
    fp = features_prefix(prefix)
    return [
        f"{fp}/{asset}_features_{tf_label(horizon)}.parquet",   # engineered cache: tf-label naming
        f"{fp}/{asset}_features_har_pd_{horizon}m.parquet",
        f"{fp}/{svi_filename(horizon)}",
    ]


def horizon_data_files(horizon: int, asset: str = "BTCUSDT") -> dict[str, str]:
    """
    The 4 per-horizon REPO-RELATIVE data files this horizon's forecast reads. These are the
    DESTINATIONS run_horizon_job.stage_shard() copies the staged basenames into; the matching
    FLAT S3 SOURCE keys are tsrv_bar_key() (tsrv) + horizon_feature_keys() (features/har_pd/svi).
    Both sides go through this module so the launcher download block and stage_data_to_s3.py
    cannot drift again (the original launcher invented data/h{H}/ shards that never existed).
    """
    return {
        "tsrv": f"TSRV/output/{asset}_tsrv_{horizon}m.parquet",
        "features": f"vol_model/output/{asset}_features_{tf_label(horizon)}.parquet",
        "har_pd": f"vol_model/output/{asset}_features_har_pd_{horizon}m.parquet",
        "svi": f"vol_arb/output/{svi_filename(horizon)}",
    }


# Shared tiny csvs needed by every horizon (pruned/drop feature lists; ~5KB total).
SHARED_FILES = (
    "vol_model/output/pruned_features.csv",
    "vol_model/output/BTCUSDT_drop_features.csv",
)
