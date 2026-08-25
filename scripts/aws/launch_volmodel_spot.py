#!/usr/bin/env python3
"""
Launch EC2 spot instance(s) to run the per-horizon volatility-forecast walk-forward.

ONE BOX PER HORIZON. Each box:
  * pulls the code tarball + that horizon's ~<=550MB data shard from S3,
  * pip-installs requirements-aws.txt (lean, torch-free; includes optuna/lightgbm/pydantic),
  * runs scripts/aws/run_horizon_job.py --horizon <H> (TSRV->har_pd->SVI prereqs then the
    run_walkforward_pruned_svi_har_pd.py walk-forward with --no-transvol),
  * uploads its result shard to s3://BUCKET/PREFIX/results/h{H}/shard.json,
  * self-terminates (shutdown -h now + InstanceInitiatedShutdownBehavior=terminate).

IDEMPOTENT: the launcher SKIPS a horizon whose result shard already exists in S3 (relaunch-on-reclaim
resumes the fleet without re-running finished horizons). Prints `LAUNCHED i-<id>` for the babysitter.

PORTED FROM the earlier Pokerbot spot-fleet implementation:
  scripts/launch_sd_cfr_training_spot.py  — the canonical single-box spot launcher. Reused VERBATIM
    (constants swapped): resolve_ubuntu_ami() (lines 66-79, Canonical Owners=099720109477 + jammy-22.04
    filter), the ec2.run_instances spot block (lines 323-345: IMDSv2 HttpTokens=required + hop=2,
    InstanceInitiatedShutdownBehavior=terminate, one-time spot, InstanceInterruptionBehavior=terminate,
    Run tag), the Model-A user_data() bootstrap structure (lines 144-234: download-before-pip ordering
    fix lines 161-167, NO `exec >logfile` redirect lines 137-143, BLAS/OMP thread pinning lines 186-187,
    boto3 log-upload block lines 222-232, final `shutdown -h now` line 136), build_and_upload_code()
    + _tar_filter (lines 46-63).
  scripts/launch_validate50_ec2.py        — grafted the --max-price -> SpotOptions["MaxPrice"] cap and
    the --wait poll-for-completion pattern (cited doc §2).

NEW vs Pokerbot (required by this workflow's design constraints; the Pokerbot launcher has neither):
  * BlockDeviceMappings gp3 30GB DeleteOnTermination=True (Pokerbot omits it; we need ~5GB data room).
  * Idempotent skip-if-result-shard-exists guard (launcher-side head_object + on-box `aws s3 ls` gate).
  * --horizon (repeatable) one-box-per-horizon fan-out with a per-horizon Horizon=<H> tag.

Constants come from ENV (see scripts/aws/volmodel_env.py): BUCKET / PREFIX / REGION / PROFILE /
INSTANCE_PROFILE. ``QUANT_BUCKET`` is required; deployment-neutral defaults cover the prefix,
region, and instance profile.

THIS SCRIPT SPENDS NO MONEY in --dry-run mode (ec2.run_instances DryRun=True -> DryRunOperation).
"""

from __future__ import annotations

import argparse
import base64
import io
import sys
import tarfile
import time
from pathlib import Path

import boto3
from botocore.config import Config
from botocore.exceptions import ClientError

# scripts/aws/launch_volmodel_spot.py -> repo root is parents[2]
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "aws"))

from volmodel_env import (  # noqa: E402
    BUCKET,
    PREFIX,
    REGION,
    PROFILE,
    INSTANCE_PROFILE,
    HORIZON_MINUTES,
    result_shard_key,
    code_key as default_code_key,
    # FLAT S3 data layout helpers (single source of truth shared with
    # stage_data_to_s3.py — see volmodel_env.py "S3 DATA LAYOUT" block).
    tsrv_bar_key,
    horizon_feature_keys,
    deribit_prefix,
    features_prefix,
    shared_data_prefix,
)

# ---------------------------------------------------------------------------
# Code tarball — Model-A allow-list (pure .py, EXCLUDE the multi-GB output/ trees).
# transvol/ stays in the tarball (it's tiny pure-python) but is NEVER imported on the
# --no-transvol path: the walk-forward + extract scripts now import torch + the transvol
# embedding extractor LAZILY (TransVol branch only), so the box runs torch-free.
# ---------------------------------------------------------------------------
_INCLUDE = [
    "vol_model",
    "vol_arb",
    "TSRV",
    "src",
    "scripts",
    "requirements.txt",
    "requirements-aws.txt",
]
_TAR_EXCLUDE_DIRS = {"__pycache__", "output", ".git", ".venv", "node_modules"}


def _tar_filter(ti: tarfile.TarInfo):
    parts = ti.name.split("/")
    if any(p in _TAR_EXCLUDE_DIRS for p in parts):
        return None
    if ti.name.endswith((".pyc", ".parquet", ".pt")):
        return None
    return ti


def build_and_upload_code(s3, bucket: str, prefix: str) -> str:
    """Tar the allow-listed pure-python tree (no output/, no parquet) and put_object to S3."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in _INCLUDE:
            p = ROOT / rel
            if p.exists():
                tar.add(p, arcname=rel, filter=_tar_filter)
            else:
                print(f"WARNING: {rel} missing from tarball (not on disk)", flush=True)
    size = buf.tell()
    buf.seek(0)
    key = default_code_key(prefix)
    s3.put_object(Bucket=bucket, Key=key, Body=buf.read())
    print(f"Uploaded code tarball -> s3://{bucket}/{key} ({size} bytes)", flush=True)
    return key


def resolve_ubuntu_ami(ec2, arch: str) -> str:
    """Latest Canonical Ubuntu 22.04 (Jammy) HVM-SSD server AMI. Ported verbatim (Pokerbot:66-79)."""
    name_arch = "amd64" if arch == "x86_64" else "arm64"
    resp = ec2.describe_images(
        Owners=["099720109477"],  # Canonical
        Filters=[
            {"Name": "name", "Values": [f"ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-{name_arch}-server-*"]},
            {"Name": "state", "Values": ["available"]},
            {"Name": "architecture", "Values": [arch]},
        ],
    )
    imgs = sorted(resp["Images"], key=lambda i: i["CreationDate"], reverse=True)
    if not imgs:
        raise SystemExit(f"no {arch} Ubuntu 22.04 AMI found in region")
    return imgs[0]["ImageId"]


def shard_exists(s3, bucket: str, prefix: str, horizon: int) -> bool:
    key = result_shard_key(prefix, horizon)
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        # 403 (no perm to head) — treat as "unknown, don't skip" so we don't wrongly suppress a launch.
        if code == "403":
            return False
        raise


# ---------------------------------------------------------------------------
# Model-A user-data bootstrap. Structure ported from Pokerbot user_data() (lines 144-234):
#   set -uxo pipefail; export region; apt python3-pip awscli; aws s3 cp ALL downloads BEFORE pip;
#   pip install; pin BLAS/OMP threads; run the job with `... 2>&1 | tee`; boto3 log upload; shutdown.
# Replaced the wheelhouse/JAX-cache machinery with a plain `pip install -r requirements.txt` + the
# 4 missing deps (optuna lightgbm pydantic torch-cpu). Replaced the training command with the
# per-horizon driver scripts/aws/run_horizon_job.py.
# ---------------------------------------------------------------------------
def _model_dir_name(horizon: int) -> str:
    """On-box walk-forward --no-transvol output dir name for this horizon.

    Mirrors run_horizon_job._paths() / vol_arb.paths.har_pd_model_dir exactly:
    the base dir is `expanding_pruned_svi_har_pd_v1` at h=15 (legacy unsuffixed)
    and `expanding_pruned_svi_har_pd_v1_h{H}m` elsewhere, then `_no_tv` appended.
    The spot-interrupt poller needs this to s3-cp the right per-horizon checkpoint
    dir directly from disk (no python process needed for the final flush).
    """
    base = "expanding_pruned_svi_har_pd_v1"
    if horizon != 15:
        base = f"{base}_h{horizon}m"
    return f"{base}_no_tv"


def user_data(*, bucket, prefix, region, code_key, horizon, asset, run_id,
              max_trials=None, patience=None, keep_alive=False,
              opt_num_threads=None) -> str:
    shard_key = result_shard_key(prefix, horizon)
    # FLAT S3 layout (matches stage_data_to_s3.py via volmodel_env helpers).
    # The launcher used to invent per-horizon data/h{H}/ + data/shared/ shards
    # that the staging uploader never produced -> empty /tmp/shard -> the
    # "TSRV bars missing" crash. We now pull from the EXACT flat keys that exist.
    tsrv_key = tsrv_bar_key(prefix, horizon, asset)        # THIS horizon's bar only
    feat_keys = horizon_feature_keys(prefix, horizon, asset)  # engineered/har_pd/svi sidecars
    deribit_pfx = deribit_prefix(prefix)                   # SVI-rebuild fallback inputs
    shared_pfx = shared_data_prefix(prefix)                # tiny shared csvs
    ckpt_prefix = f"{prefix}/checkpoints/h{horizon}"
    model_dir_rel = f"vol_model/output/models/{_model_dir_name(horizon)}"
    final = "echo BOOTSTRAP_DONE_KEEP_ALIVE" if keep_alive else "shutdown -h now"
    # Thread-packing environment. OPT_NUM_THREADS controls only the LightGBM fit
    # (the merged trainer reads it: run_walkforward.py:196 + run_walkforward_pruned_svi_har_pd.py:768).
    #   * --opt-num-threads T given  -> pin LightGBM AND BLAS/numpy to T (a PACKED horizon-process:
    #     several share the box; BLAS pinned to the same T so a co-tenant cannot oversubscribe vCPUs).
    #   * --opt-num-threads OMITTED  -> leave OPT_NUM_THREADS UNSET so the trainer uses native
    #     num_threads=-1 (all box cores). This is the h=15 byte-identity path AND the legacy
    #     one-box-per-horizon path; BLAS gets the full NPROC.
    if opt_num_threads is not None:
        thread_env_block = (
            f"export OPT_NUM_THREADS={opt_num_threads} "
            f"OMP_NUM_THREADS={opt_num_threads} OPENBLAS_NUM_THREADS={opt_num_threads} "
            f"MKL_NUM_THREADS={opt_num_threads} NPY_NUM_THREADS={opt_num_threads} "
            f"NUMEXPR_NUM_THREADS={opt_num_threads}\n"
            f'echo "thread util: PACKED, OPT_NUM_THREADS={opt_num_threads} '
            f'(LightGBM fit + BLAS pinned to {opt_num_threads})"'
        )
    else:
        thread_env_block = (
            "export OMP_NUM_THREADS=$NPROC OPENBLAS_NUM_THREADS=$NPROC "
            "MKL_NUM_THREADS=$NPROC NPY_NUM_THREADS=$NPROC NUMEXPR_NUM_THREADS=$NPROC\n"
            'echo "thread util: NPROC=$NPROC (OPT_NUM_THREADS unset -> native -1; h15 byte-identical)"'
        )
    # Optional reduced-Optuna passthrough -> the driver flags. Omitted entirely
    # when None so the driver keeps its production defaults (200 trials / 50
    # patience). Passing --max-trials 50 makes a CHEAP fast smoke / cost-tuned run.
    trials_flag = f" --max-trials {max_trials}" if max_trials is not None else ""
    patience_flag = f" --patience {patience}" if patience is not None else ""
    # Per-horizon pre-built sidecars (engineered cache / har_pd / SVI) -> /tmp/shard.
    # Each is downloaded SEPARATELY (not --recursive) and tolerant of absence (`|| true`):
    # a horizon whose sidecar was never pre-built rebuilds it on-box (har_pd from the TSRV
    # bar in Step 2; SVI from data/deribit in Step 3). stage_shard() then routes each present
    # basename into the repo layout the path-helpers read.
    feature_downloads = "\n".join(
        f"aws s3 cp s3://{bucket}/{k} /tmp/shard/ --region {region} --no-progress || true"
        for k in feat_keys
    )
    # NOTE: NO `exec > logfile 2>&1` redirect (Pokerbot:137-143 — that hid cloud-init console capture
    # and caused a 13-attempt blind failure). We `tee` the job line instead.
    return f"""#!/bin/bash
set -uxo pipefail
export DEBIAN_FRONTEND=noninteractive AWS_DEFAULT_REGION={region}

# --- IDEMPOTENT SKIP: if this horizon's result shard already exists, do nothing and self-terminate. ---
if aws s3 ls s3://{bucket}/{shard_key} --region {region}; then
  echo "RESULT SHARD ALREADY EXISTS s3://{bucket}/{shard_key} — skipping horizon {horizon}"
  {final}
  exit 0
fi

apt-get update -qq
apt-get install -y -qq python3-pip awscli gzip curl
aws --version
mkdir -p /opt/app
export PYTHONPATH=/opt/app

# CRITICAL ORDERING (Pokerbot:161-167): do ALL `aws s3 cp` DOWNLOADS *before* any pip install. The
# pip step upgrades botocore, which breaks the apt aws-cli v1; post-pip uploads use boto3 instead.
aws s3 cp s3://{bucket}/{code_key} /tmp/code.tar.gz --region {region}
tar -xzf /tmp/code.tar.gz -C /opt/app
ls -la /opt/app/scripts/aws/run_horizon_job.py

# Reconstruct the repo data layout the path-helpers expect, pulling from the FLAT
# S3 keys stage_data_to_s3.py actually produced (NOT the old per-horizon shard tree).
mkdir -p /opt/app/TSRV/output /opt/app/vol_model/output /opt/app/vol_arb/output /opt/app/data/deribit /tmp/shard
# shared tiny csvs (pruned_features.csv + BTCUSDT_drop_features.csv) -> vol_model/output:
aws s3 cp s3://{bucket}/{shared_pfx}/ /opt/app/vol_model/output/ --recursive --region {region} --no-progress
# deribit option trades + DVOL (+candles): the SVI-rebuild fallback inputs (Step 3) read
# data/deribit/{{trades/BTC_option,dvol/BTC_dvol_60m.parquet}}. Pre-built SVI sidecar below
# normally makes the rebuild a no-op, but stage these so an absent/renamed sidecar self-heals.
aws s3 cp s3://{bucket}/{deribit_pfx}/ /opt/app/data/deribit/ --recursive --region {region} --no-progress
# THIS horizon's data into /tmp/shard (flattened); run_horizon_job.stage_shard() routes each
# basename into the repo layout (tsrv->TSRV/output, har_pd/engineered cache->vol_model/output,
# svi->vol_arb/output). Download ONLY this horizon's TSRV bar (not all 8 horizons' bars):
aws s3 cp s3://{bucket}/{tsrv_key} /tmp/shard/ --region {region} --no-progress
{feature_downloads}
ls -la /tmp/shard/ || true

# deps: the dedicated LEAN AWS-box requirements (requirements-aws.txt). It already pins the
# three deps requirements.txt omits (optuna/lightgbm/pydantic) + boto3, and deliberately omits
# torch — the --no-transvol walk-forward + extract import torch LAZILY (TransVol branch only),
# so the box runs torch-free. No CPU torch wheel needed; keeps the image lean.
python3 -m pip install --upgrade pip
python3 -m pip install -r /opt/app/requirements-aws.txt
echo "=== deps install done (torch-free, --no-transvol) ==="

cd /opt/app
# THREAD UTILIZATION (CORRECTED): this box runs ONE horizon = a single process with SEQUENTIAL Optuna,
# so there is NO process-pool BLAS thrash to avoid (the Pokerbot pin=1 was for a multi-process pool).
# LightGBM reads OMP_NUM_THREADS — pinning it to 1 throttled training to a SINGLE thread (measured
# ~2.6x slower on a loaded 12-core box; ~one-thread-of-N on a clean N-vCPU box, i.e. the 16-vCPU heavy
# boxes would waste 15/16 cores). Give every math lib the box's full core count so LightGBM (which
# dominates runtime) AND numpy/pandas use all vCPUs. The pipeline's num_threads=-1 auto-detects -> uses
# exactly NPROC, no oversubscription.
NPROC=$(nproc)
{thread_env_block}

# CONFIG -> ENV (the driver reads QUANT_BUCKET/QUANT_PREFIX/QUANT_REGION/QUANT_ASSET from the
# environment; it does NOT take --bucket/--prefix/--region as positional config). Exporting here
# keeps the driver CLI minimal and is the single source of truth for the on-box S3 plumbing.
export QUANT_BUCKET={bucket} QUANT_PREFIX={prefix} QUANT_REGION={region} QUANT_ASSET={asset}

# --- SPOT-INTERRUPT POLLER (knowledge-transfer doc s4/s7 "fast path") -------------------------------
# On a spot reclaim AWS posts a 2-minute notice at the IMDSv2 metadata endpoint
# .../meta-data/spot/termination-time. This background poller watches it every 5s; on the notice it
# fires a FINAL `aws s3 cp` of THIS horizon's on-disk checkpoint dir (the per-month {{month}}.txt +
# {{month}}_features.json the walk-forward has already written) up to s3://{bucket}/{ckpt_prefix}/ so a
# relaunched box resumes from the last COMPLETED month -- losing at most the in-progress month. It
# uses the host aws-cli (still valid: the poller backgrounds BEFORE the pip botocore upgrade), then
# SIGTERMs the job as a belt-and-suspenders (the driver also flushes on SIGTERM). IMDSv2: fetch a
# token (hop limit 2 is set on the instance) and pass it on every metadata GET.
cat > /usr/local/bin/spot_interrupt_poller.sh <<'POLLEOF'
#!/bin/bash
CKPT_S3="s3://{bucket}/{ckpt_prefix}/"
MODEL_DIR="/opt/app/{model_dir_rel}"
flush() {{
  if [ -d "$MODEL_DIR" ]; then
    # only the resume-critical small files; --no-progress keeps the console clean.
    aws s3 cp "$MODEL_DIR/" "$CKPT_S3" --recursive --region {region} --no-progress \
      --exclude "*" --include "*.txt" --include "*.json" --include "*.csv" || true
    echo "SPOT-INTERRUPT: final checkpoint flush done -> $CKPT_S3"
  fi
}}
while true; do
  TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" \
    -H "X-aws-ec2-metadata-token-ttl-seconds: 60" 2>/dev/null || true)
  TT=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" \
    http://169.254.169.254/latest/meta-data/spot/termination-time 2>/dev/null || true)
  # The endpoint 404s (empty/'Not Found') until a reclaim is scheduled, then returns an ISO timestamp.
  if echo "$TT" | grep -Eq "^[0-9][0-9][0-9][0-9]-"; then
    echo "SPOT-INTERRUPT NOTICE at $TT -- draining checkpoint"
    flush
    # nudge the job to flush + exit cleanly; the box terminates on the 2-min deadline regardless.
    pkill -TERM -f run_horizon_job.py || true
    break
  fi
  sleep 5
done
POLLEOF
chmod +x /usr/local/bin/spot_interrupt_poller.sh
nohup /usr/local/bin/spot_interrupt_poller.sh >/var/log/spot_poller.log 2>&1 &
echo "spot-interrupt poller backgrounded (pid $!)"

# The driver's argparse accepts: --horizon --asset --shard-dir --max-trials --patience (+ optional
# --run-id and the rarely-used --bucket/--prefix/--region overrides which we satisfy via env above).
# --max-trials/--patience are threaded through ONLY when the launcher was given them (cheap smoke /
# cost-tuned fleet); otherwise the driver keeps its historical defaults (200 trials / 50 patience).
# The driver itself restores the checkpoint from {ckpt_prefix} on start and syncs it back every ~5 min.
python3 scripts/aws/run_horizon_job.py \
  --horizon {horizon} --asset {asset} --run-id {run_id} \
  --shard-dir /tmp/shard{trials_flag}{patience_flag} \
  2>&1 | tee /var/log/job.log
echo "job exit ${{PIPESTATUS[0]}}"

# Host aws-cli is now botocore-incompatible after the pip upgrade -> upload logs via boto3
# (Pokerbot:222-232). The result shard itself is uploaded by run_horizon_job.py.
python3 - <<'PYEOF'
import boto3
s3 = boto3.client("s3", region_name="{region}")
for src, key in [("/var/log/cloud-init-output.log", "{prefix}/results/h{horizon}/logs/cloud-init-output.log"),
                 ("/var/log/job.log", "{prefix}/results/h{horizon}/logs/job.log")]:
    try:
        s3.upload_file(src, "{bucket}", key); print("uploaded", key)
    except Exception as e:
        print("log upload fail", src, e)
PYEOF
{final}
"""


# AWS error codes meaning an ACCOUNT-WIDE quota/limit (NOT a per-type capacity issue): a different
# instance type will NOT help -> the launcher stops and reports LAUNCH_QUOTA so the babysitter backs
# off (waits for running boxes to free quota) instead of churning types.
_QUOTA_CODES = {"MaxSpotInstanceCountExceeded", "InstanceLimitExceeded", "VcpuLimitExceeded"}


def launch_one(ec2, s3, *, args, horizon, code_key, ami, run_id) -> str | None:
    """Build user-data + run_instances for a single horizon. Returns instance-id (or None on dry-run).

    Tries each type in the --instance-types chain in order: InsufficientInstanceCapacity -> next type
    INSTANTLY (fast-fail, the ec2 client has no retries); a quota error stops with LAUNCH_QUOTA.
    """
    if shard_exists(s3, args.bucket, args.prefix, horizon):
        print(f"SKIP h{horizon}: result shard already exists "
              f"(s3://{args.bucket}/{result_shard_key(args.prefix, horizon)})", flush=True)
        return None

    # h=15 NEVER gets a thread override (byte-identity): force OPT_NUM_THREADS unset even if
    # --opt-num-threads was passed globally. Every other horizon takes the packing value (or None).
    opt_threads = None if horizon == 15 else args.opt_num_threads
    ud = user_data(
        bucket=args.bucket, prefix=args.prefix, region=args.region, code_key=code_key,
        horizon=horizon, asset=args.asset, run_id=run_id,
        max_trials=args.max_trials, patience=args.patience, keep_alive=args.keep_alive,
        opt_num_threads=opt_threads,
    )
    # Instance-type fallback chain (try in order, fast-fail over on capacity in the launch loop below).
    itypes = ([t.strip() for t in args.instance_types.split(",") if t.strip()]
              if getattr(args, "instance_types", None) else [args.instance_type])
    run_kwargs = dict(
        ImageId=ami,
        InstanceType=itypes[0],
        MinCount=1,
        MaxCount=1,
        UserData=base64.b64encode(ud.encode()).decode(),
        IamInstanceProfile={"Name": args.instance_profile},
        MetadataOptions={"HttpTokens": "required", "HttpPutResponseHopLimit": 2},
        BlockDeviceMappings=[{
            "DeviceName": "/dev/sda1",
            "Ebs": {"VolumeSize": args.volume_gb, "VolumeType": "gp3", "DeleteOnTermination": True},
        }],
        TagSpecifications=[{
            "ResourceType": "instance",
            "Tags": [
                {"Key": "Name", "Value": f"volmodel-h{horizon}-{run_id}"},
                {"Key": "Project", "Value": "quant-volmodel"},
                {"Key": "Run", "Value": run_id},
                {"Key": "Horizon", "Value": str(horizon)},
            ],
        }],
        InstanceInitiatedShutdownBehavior="terminate",
    )
    if not args.on_demand:
        spot_opts = {
            "SpotInstanceType": "one-time",
            "InstanceInterruptionBehavior": "terminate",
        }
        if args.max_price:
            spot_opts["MaxPrice"] = str(args.max_price)
        run_kwargs["InstanceMarketOptions"] = {"MarketType": "spot", "SpotOptions": spot_opts}

    if args.dry_run:
        # Print the rendered user_data so the resilient-spot wiring (interrupt poller, checkpoint
        # sync, reduced-trials driver command, spot block) is verifiable without spending money.
        if not args.no_print_user_data:
            print(f"\n===== USER_DATA (h{horizon}, run={run_id}) BEGIN =====", flush=True)
            print(ud, flush=True)
            print(f"===== USER_DATA (h{horizon}) END =====\n", flush=True)
            spot = not args.on_demand
            print(f"DRYRUN h{horizon}: market={'SPOT (one-time, interrupt=terminate)' if spot else 'on-demand'}"
                  f"  max_price={args.max_price or '(on-demand cap)'}"
                  f"  max_trials={args.max_trials if args.max_trials is not None else '(driver default 200)'}"
                  f"  patience={args.patience if args.patience is not None else '(driver default 50)'}", flush=True)
        run_kwargs["DryRun"] = True
        try:
            ec2.run_instances(**run_kwargs)
            print(f"DRYRUN h{horizon}: run_instances unexpectedly succeeded (no DryRun?)", flush=True)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code == "DryRunOperation":
                print(f"DRYRUN h{horizon}: OK (DryRunOperation — permissions sufficient) "
                      f"ami={ami} type={itypes[0]} spot={not args.on_demand}", flush=True)
            elif "IAM Instance Profile" in str(e):
                # The 'quant-worker' instance profile does not exist yet (created later by admin via
                # setup_quant_iam.py). AWS validates the profile name BEFORE the DryRun permission
                # check, so a profile-less retry confirms the ec2:RunInstances permission itself now.
                run_kwargs.pop("IamInstanceProfile", None)
                try:
                    ec2.run_instances(**run_kwargs)
                except ClientError as e2:
                    code2 = e2.response.get("Error", {}).get("Code", "")
                    if code2 == "DryRunOperation":
                        print(f"DRYRUN h{horizon}: OK (DryRunOperation, profile-less) — "
                              f"ec2:RunInstances permitted. NOTE: instance-profile "
                              f"'{args.instance_profile}' does not exist yet; run setup_quant_iam.py "
                              f"(admin) before a real launch.", flush=True)
                    else:
                        print(f"DRYRUN h{horizon}: FAILED profile-less ({code2}: {e2})", flush=True)
                        raise
            else:
                print(f"DRYRUN h{horizon}: FAILED ({code}: {e})", flush=True)
                raise
        return None

    # FAILOVER over the type chain. The ec2 client (ec2_launch in main) has no retries, so a dry pool
    # raises InsufficientInstanceCapacity immediately -> we try the next type with NO backoff.
    last = None
    for itype in itypes:
        run_kwargs["InstanceType"] = itype
        try:
            resp = ec2.run_instances(**run_kwargs)
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "")
            if code == "InsufficientInstanceCapacity":
                print(f"  CAPACITY h{horizon}: no spot capacity for {itype} -> trying next type", flush=True)
                last = code
                continue
            if code in _QUOTA_CODES:
                print(f"  LAUNCH_QUOTA h{horizon}: {code} (on {itype}) -- account vCPU/spot LIMIT hit; "
                      f"other types won't help. Back off until running boxes free quota.", flush=True)
                return None
            print(f"  LAUNCH_ERROR h{horizon}: {code} (on {itype}): {e}", flush=True)
            return None
        iid = resp["Instances"][0]["InstanceId"]
        # Babysitter parses this EXACT token (regex 'LAUNCHED (i-[0-9a-f]+)'). Keep the format.
        print(f"LAUNCHED {iid} ({itype}, h{horizon}, {'spot' if not args.on_demand else 'on-demand'})", flush=True)
        print(f"  result shard: s3://{args.bucket}/{result_shard_key(args.prefix, horizon)}", flush=True)
        print(f"  console: aws ec2 get-console-output --instance-id {iid} --region {args.region} --latest", flush=True)
        return iid
    print(f"  CAPACITY_EXHAUSTED h{horizon}: no spot capacity for ANY of {itypes} (last={last})", flush=True)
    return None


def wait_for_shards(s3, *, bucket, prefix, horizons, region, poll_s=150, timeout_s=18 * 3600) -> bool:
    """Poll S3 until every horizon's result shard exists or timeout. Ported from validate50 --wait."""
    deadline = time.time() + timeout_s
    pending = set(horizons)
    while pending and time.time() < deadline:
        done = {h for h in pending if shard_exists(s3, bucket, prefix, h)}
        for h in sorted(done):
            print(f"DONE h{h}: shard present", flush=True)
        pending -= done
        if not pending:
            break
        print(f"waiting on horizons {sorted(pending)} "
              f"({int(deadline - time.time())}s budget left)", flush=True)
        time.sleep(poll_s)
    if pending:
        print(f"TIMEOUT: still waiting on {sorted(pending)}", flush=True)
        return False
    print("ALL HORIZON SHARDS PRESENT", flush=True)
    return True


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--horizon", type=int, action="append", default=None,
                   help="horizon in MINUTES; repeatable (one box per horizon). "
                        f"valid: {sorted(HORIZON_MINUTES)}. Default: all of them.")
    p.add_argument("--tf", type=int, action="append", default=None,
                   help="horizon in SECONDS (= H*60); repeatable. Convenience alias the babysitter "
                        "uses (it relaunches one reclaimed horizon at a time via --tf <H*60>). "
                        "Converted to --horizon H internally. Mutually combinable with --horizon.")
    p.add_argument("--no-transvol", action="store_true",
                   help="No-op accepted for babysitter compatibility: the launcher is ALWAYS "
                        "--no-transvol (the on-box driver hardcodes it). Accepting it here keeps "
                        "the babysitter's `--tf <s> --no-transvol` relaunch line valid.")
    p.add_argument("--asset", default="BTCUSDT")
    p.add_argument("--instance-type", default="c7i.4xlarge",
                   help="single instance type (used if --instance-types is not given).")
    p.add_argument("--instance-types", default=None,
                   help="COMMA-LIST of equivalent types tried in order for ONE launch. On "
                        "InsufficientInstanceCapacity the next type is tried INSTANTLY (no backoff); a "
                        "quota error stops (other types won't help). Overrides --instance-type. Build it "
                        "with `spotpick.py chain --vcpus N --arch arm64`.")
    p.add_argument("--arch", default="x86_64", choices=["x86_64", "arm64"])
    p.add_argument("--volume-gb", type=int, default=30, help="root gp3 EBS GB (DeleteOnTermination).")
    p.add_argument("--on-demand", action="store_true", help="use on-demand instead of spot")
    p.add_argument("--max-price", default=None,
                   help="optional spot MaxPrice cap (USD/hr, string). Omit = cap at on-demand price.")
    p.add_argument("--max-trials", type=int, default=None,
                   help="Optuna trial budget per walk-forward month, threaded into the on-box driver. "
                        "Omit = keep the driver's default (200). Pass a small value "
                        "(e.g. 50) for a lower-cost smoke run or to tune fleet cost.")
    p.add_argument("--patience", type=int, default=None,
                   help="Optuna patience per month, threaded into the on-box driver. "
                        "Omit = keep the driver's default (50).")
    p.add_argument("--opt-num-threads", type=int, default=None,
                   help="LightGBM fit thread count for THIS horizon-process (sets OPT_NUM_THREADS + "
                        "pins BLAS to the same value). Use for BOX-PACKING: several horizon-processes "
                        "share one box, each pinned to ~2-3 threads. Gate: "
                        "procs*threads <= box vCPUs. "
                        "OMIT for the legacy one-box-per-horizon run. h=15 ALWAYS ignores this "
                        "(forced native -1 for byte-identity).")
    p.add_argument("--keep-alive", action="store_true",
                   help="don't shutdown on finish (keep box for get-console-output inspection)")
    p.add_argument("--dry-run", action="store_true",
                   help="ec2.run_instances DryRun=True (no instance, no spend) — verifies permissions "
                        "AND prints the rendered user_data (interrupt poller + checkpoint sync + driver cmd).")
    p.add_argument("--no-print-user-data", action="store_true",
                   help="suppress the user_data dump under --dry-run (just do the permission probe).")
    p.add_argument("--wait", action="store_true", help="poll S3 until every result shard exists")
    p.add_argument("--run-id", default=None, help="S3/tag run id (default: run_<ts>)")
    p.add_argument("--bucket", default=BUCKET)
    p.add_argument("--prefix", default=PREFIX)
    p.add_argument("--region", default=REGION)
    p.add_argument("--profile", default=PROFILE, help="AWS profile (None/empty = default creds)")
    p.add_argument("--instance-profile", default=INSTANCE_PROFILE)
    p.add_argument("--skip-code-upload", action="store_true",
                   help="reuse the existing code tarball in S3 (don't rebuild+upload)")
    args = p.parse_args()

    # --tf is in SECONDS (babysitter alias) -> minutes; merge with --horizon (minutes), de-dup, order.
    tf_horizons = [int(s) // 60 for s in (args.tf or [])]
    merged = list(args.horizon or []) + tf_horizons
    horizons = sorted(set(merged)) if merged else sorted(HORIZON_MINUTES)
    bad = [h for h in horizons if h not in HORIZON_MINUTES]
    if bad:
        p.error(f"invalid horizon(s) {bad}; valid (1440%H==0): {sorted(HORIZON_MINUTES)}")

    session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    s3 = session.client("s3", region_name=args.region)
    ec2 = session.client("ec2", region_name=args.region)
    # run_instances client with NO retries so InsufficientInstanceCapacity fails FAST (the default
    # standard mode retries it ~4x -> slow type-failover). Launches are infrequent, so this is safe.
    ec2_launch = session.client("ec2", region_name=args.region,
                                config=Config(retries={"max_attempts": 1, "mode": "standard"}))

    run_id = args.run_id or f"run_{int(time.time())}"

    if args.skip_code_upload:
        code_key = default_code_key(args.prefix)
        print(f"Reusing code tarball s3://{args.bucket}/{code_key}", flush=True)
    elif args.dry_run:
        # Dry-run: don't mutate S3, just reference the expected key.
        code_key = default_code_key(args.prefix)
        print(f"DRYRUN: skipping code upload; would use s3://{args.bucket}/{code_key}", flush=True)
    else:
        code_key = build_and_upload_code(s3, args.bucket, args.prefix)

    ami = resolve_ubuntu_ami(ec2, args.arch)
    print(f"AMI: {ami}  types: {args.instance_types or args.instance_type}  arch: {args.arch}  "
          f"region: {args.region}  run: {run_id}  horizons: {horizons}", flush=True)

    launched = []
    for h in horizons:
        iid = launch_one(ec2_launch, s3, args=args, horizon=h, code_key=code_key, ami=ami, run_id=run_id)
        if iid:
            launched.append((h, iid))

    if launched:
        print(f"Launched {len(launched)} box(es): "
              + ", ".join(f"h{h}={iid}" for h, iid in launched), flush=True)
        print(f"Teardown: aws ec2 terminate-instances --region {args.region} --instance-ids "
              + " ".join(iid for _, iid in launched), flush=True)

    if args.wait and not args.dry_run:
        ok = wait_for_shards(s3, bucket=args.bucket, prefix=args.prefix,
                             horizons=horizons, region=args.region)
        sys.exit(0 if ok else 2)


if __name__ == "__main__":
    main()
