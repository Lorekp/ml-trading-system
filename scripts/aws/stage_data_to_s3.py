#!/usr/bin/env python3
"""Stage the minimal vol-forecast dataset + code tarball to S3 for the AWS spot fleet.

One-time, idempotent staging step for the per-horizon vol-forecast training run.
Uploads:
  1. The minimal DATA set (NOT the 191GB aggTrades) to s3://BUCKET/PREFIX/data/...
  2. A pure-python CODE tarball to s3://BUCKET/PREFIX/code/code.tar.gz

Both are idempotent: a data object is skipped if its key already exists (head_object),
and the code tarball is re-uploaded each run (it is small + content changes).

Ports the tarball build+upload + boto3-put_object pattern from the earlier
Pokerbot spot launcher:
    - build_and_upload_code / _tar_filter (lines 46-63): tarfile -> BytesIO -> s3.put_object,
      __pycache__/.pyc filter, "Uploaded ... (N bytes)" print.
    - _INCLUDE allow-list (lines 37-42): swapped to the quant minimal code set.
Region/bucket/prefix are NOT hardcoded to pokerbot values; they come from constants+env.
The real account default region is eu-central-1 (~/.aws/config [default]); the Pokerbot
files all hardcode eu-central-1 too, so we default to it (NOT us-east-1).

Run with the labelgen-worker default creds AFTER the bucket + worker S3 policy exist
(scripts/setup_quant_iam.py). In this build environment the bucket does not yet exist,
so a real upload will 404/403 until the user runs the admin IAM/bucket setup; a DryRun
mode (--dry-run) lists the exact manifest + byte totals without touching S3.

Usage:
  PYTHONPATH=. .venv/Scripts/python.exe scripts/aws/stage_data_to_s3.py --dry-run
  PYTHONPATH=. .venv/Scripts/python.exe scripts/aws/stage_data_to_s3.py            # real upload
  ... --bucket <bucket> --prefix volmodel/v1 --region eu-central-1
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import tarfile
from pathlib import Path

# --- repo root (this file is at <root>/scripts/aws/stage_data_to_s3.py) ---
ROOT = Path(__file__).resolve().parents[2]

# --- parametrized constants (env-overridable; QUANT_* namespace, single source of truth) ---
DEFAULT_BUCKET = os.environ.get("QUANT_BUCKET", "").strip() or None
DEFAULT_PREFIX = os.environ.get("QUANT_PREFIX", "volmodel/v1")
DEFAULT_REGION = os.environ.get("QUANT_REGION", os.environ.get("AWS_DEFAULT_REGION", "eu-central-1"))
DEFAULT_PROFILE = os.environ.get("QUANT_PROFILE") or os.environ.get("AWS_PROFILE")  # None => default creds

# ---------------------------------------------------------------------------
# DATA manifest. Each entry is (local relative path, S3 sub-key under PREFIX/data/).
# Per the Investigate findings the forecast reads ONLY these; the 191GB aggTrades,
# data/binance-1m, and cross-asset data/tsrv are NOT included.
#
# We are GENEROUS within the ~5GB budget to avoid an on-box rebuild / allow-list
# footgun: we ship the full TSRV/output bar tree, the full data/deribit option+DVOL
# tree (so any horizon's SVI can be (re)built on-box), the shared feature csvs, and
# the per-horizon engineered + har_pd + svi sidecars that already exist locally.
# ---------------------------------------------------------------------------

# Whole directories to mirror recursively (relative dir, S3 sub-prefix under data/).
#
# !!! LAYOUT CONTRACT — keep in lock-step with scripts/aws/volmodel_env.py "S3 DATA LAYOUT"
# block. The launcher (launch_volmodel_spot.py user_data) downloads from THESE EXACT flat
# sub-prefixes via volmodel_env.tsrv_bar_key / horizon_feature_keys / deribit_prefix /
# shared_data_prefix. Renaming a sub-prefix here REQUIRES updating those helpers, or the
# launcher will download nothing (the original "data/h{H}/ shard" bug -> empty /tmp/shard
# -> "TSRV bars missing" crash on the smoke box).  data/tsrv_output | data/deribit |
# data/features (DATA_GLOBS) | data/shared (DATA_FILES).
DATA_DIRS = [
    ("TSRV/output", "tsrv_output"),       # ~688M: all BTCUSDT_tsrv_{H}m.parquet bars (the forecast input)
    ("data/deribit", "deribit"),          # ~1.4G: option trades + DVOL (SVI build input, Step 2)
]

# data/tsrv (2.4G cross-asset) is INTENTIONALLY shipped too only if --include-data-tsrv
# is passed; by default it is excluded (the BTC forecast never reads it). Keep the flag
# so the user can opt in without editing code if a future horizon needs it.
OPTIONAL_DATA_DIRS = [
    ("data/tsrv", "data_tsrv"),           # 2.4G cross-asset — opt-in only (--include-data-tsrv)
]

# Individual shared files (small) every horizon needs.
DATA_FILES = [
    ("vol_model/output/pruned_features.csv", "shared/pruned_features.csv"),
    ("vol_model/output/BTCUSDT_drop_features.csv", "shared/BTCUSDT_drop_features.csv"),
]

# Pre-built per-horizon feature sidecars that already exist locally. We ship whichever
# are present (glob), so a box can skip Steps 1-2 entirely. Missing ones are silently
# skipped (the box rebuilds them, or that horizon is not yet launch-ready).
DATA_GLOBS = [
    ("vol_model/output", "BTCUSDT_features_*m.parquet", "features"),        # engineered caches (sub-hour: 5m/15m/30m)
    ("vol_model/output", "BTCUSDT_features_*h.parquet", "features"),        # engineered caches (>=1h: 1h/4h/8h/12h/24h) — assemble_features uses the {hours}h label
    ("vol_model/output", "BTCUSDT_features_har_pd_*m.parquet", "features"), # har_pd sidecars
    ("vol_arb/output", "svi_features_*.parquet", "features"),               # SVI sidecars (incl legacy 15m)
]

# Only ship engineered/feature parquets we actually want (avoid the 3.9G bar_series cache,
# model checkpoints, trade logs, etc.). The globs above are tight enough; this is a guard.
GLOB_MAX_BYTES = 1_500_000_000  # skip any single matched file >1.5G (e.g. bar_series) defensively

# ---------------------------------------------------------------------------
# CODE tarball allow-list (pure-python; EXCLUDE the multi-GB output/ trees).
# Generous to avoid the _INCLUDE footgun: ship the whole vol_model (incl transvol/,
# imported at module top even with --no-transvol), vol_arb, TSRV, src, scripts,
# requirements.txt, pyproject.toml. The tar filter drops output/, parquet, pyc, caches.
# ---------------------------------------------------------------------------
CODE_INCLUDE = [
    "vol_model",
    "vol_arb",
    "TSRV",
    "src",
    "scripts",
    "requirements.txt",
    "requirements-aws.txt",
    "pyproject.toml",
]

_EXCLUDE_DIR_PARTS = {
    "__pycache__", ".git", ".pytest_cache", ".mypy_cache", "output", "models",
    "dashboard", "node_modules", "tb_logs", "dist", "build", ".venv", ".ipynb_checkpoints",
}
_EXCLUDE_SUFFIX = (
    ".pyc", ".pyo", ".parquet", ".png", ".pt", ".bak", ".log",
    ".exe", ".node", ".tfevents", ".json.gz", ".so", ".dll", ".dylib", ".map",
    ".jpg", ".jpeg", ".gif", ".svg", ".ipynb", ".csv",
    ".db", ".db-wal", ".db-shm", ".sqlite", ".h5", ".npy", ".npz",
)
# .json is allowed (small config sidecars), but the big dashboard report.json lives under
# dashboard/ which is dir-excluded, so no extra json guard is needed.


def _tar_filter(ti: tarfile.TarInfo):
    """Drop caches, the multi-GB output/ trees, and large/binary artifacts from the code tarball.

    Ported from launch_sd_cfr_training_spot.py:_tar_filter (lines 53-56), extended to also
    strip output/ and parquet/binary so the tarball stays <5MB pure-python.
    """
    parts = ti.name.replace("\\", "/").split("/")
    if any(p in _EXCLUDE_DIR_PARTS for p in parts):
        return None
    if ti.name.endswith(_EXCLUDE_SUFFIX):
        return None
    if any(p.endswith(".pre_end_stamp_bak") or p.endswith(".pre_rq_bak") or p.endswith(".pre_recal_bak") for p in parts):
        return None
    return ti


def build_code_tarball() -> bytes:
    """Build the pure-python code tarball in memory. Ports build_and_upload_code (lines 56-63)."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for rel in CODE_INCLUDE:
            p = ROOT / rel
            if p.exists():
                tar.add(p, arcname=rel, filter=_tar_filter)
            else:
                print(f"  [warn] code include missing, skipped: {rel}", flush=True)
    return buf.getvalue()


def _iter_dir_files(root: Path):
    """Yield (abs_path, rel_to_root_posix) for every file under root, skipping caches/backups."""
    for p in sorted(root.rglob("*")):
        if not p.is_file():
            continue
        parts = p.relative_to(root).parts
        if any(part in _EXCLUDE_DIR_PARTS for part in parts):
            continue
        # skip backup trees (e.g. deribit *_20k_bak/, ETH cross-asset dups not needed)
        if any(part.endswith("_bak") or part.endswith("_20k_bak") for part in parts):
            continue
        name = p.name
        if name.endswith((".png", ".log", ".pre_end_stamp_bak", ".pre_rq_bak", ".pre_recal_bak")):
            continue
        yield p, p.relative_to(root).as_posix()


def collect_data_manifest(include_data_tsrv: bool):
    """Return a list of (local_abs_path, s3_key_suffix, size_bytes) for all data to stage.

    s3_key_suffix is relative to '{PREFIX}/data/'.
    """
    manifest = []

    dirs = list(DATA_DIRS)
    if include_data_tsrv:
        dirs += OPTIONAL_DATA_DIRS

    for rel_dir, sub in dirs:
        d = ROOT / rel_dir
        if not d.exists():
            print(f"  [warn] data dir missing, skipped: {rel_dir}", flush=True)
            continue
        for abs_path, rel in _iter_dir_files(d):
            manifest.append((abs_path, f"{sub}/{rel}", abs_path.stat().st_size))

    for rel_file, key_suffix in DATA_FILES:
        f = ROOT / rel_file
        if f.exists():
            manifest.append((f, key_suffix, f.stat().st_size))
        else:
            print(f"  [warn] data file missing, skipped: {rel_file}", flush=True)

    for rel_dir, pattern, sub in DATA_GLOBS:
        base = ROOT / rel_dir
        if not base.exists():
            continue
        for f in sorted(base.glob(pattern)):
            if not f.is_file():
                continue
            sz = f.stat().st_size
            if sz > GLOB_MAX_BYTES:
                print(f"  [skip] {f.name} too large ({sz/1e9:.2f}G > {GLOB_MAX_BYTES/1e9:.1f}G)", flush=True)
                continue
            manifest.append((f, f"{sub}/{f.name}", sz))

    # de-dup by s3 key (a feature file could match two globs); keep first.
    seen = set()
    deduped = []
    for abs_path, key_suffix, sz in manifest:
        if key_suffix in seen:
            continue
        seen.add(key_suffix)
        deduped.append((abs_path, key_suffix, sz))
    return deduped


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}B"
        n /= 1024
    return f"{n:.1f}TB"


def _key_exists(s3, bucket: str, key: str) -> bool:
    from botocore.exceptions import ClientError
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "")
        if code in ("404", "NoSuchKey", "NotFound"):
            return False
        # 403 on head usually means object absent + no s3:GetObject on the key — treat as absent
        # so we attempt the put (which needs s3:PutObject); a real perms problem surfaces there.
        if code in ("403", "Forbidden", "AccessDenied"):
            return False
        raise


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bucket", default=DEFAULT_BUCKET, required=DEFAULT_BUCKET is None,
                    help="S3 bucket (required unless QUANT_BUCKET is set)")
    ap.add_argument("--prefix", default=DEFAULT_PREFIX)
    ap.add_argument("--region", default=DEFAULT_REGION)
    ap.add_argument("--profile", default=DEFAULT_PROFILE, help="AWS profile; default uses default creds")
    ap.add_argument("--include-data-tsrv", action="store_true",
                    help="also stage data/tsrv (2.4G cross-asset); excluded by default")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the manifest + byte totals; do NOT touch S3 (no creds/bucket needed)")
    ap.add_argument("--skip-code", action="store_true", help="do not build/upload the code tarball")
    ap.add_argument("--skip-data", action="store_true", help="do not upload the data set")
    args = ap.parse_args(argv)

    data_prefix = f"{args.prefix}/data"
    code_key = f"{args.prefix}/code/code.tar.gz"

    print("=" * 78, flush=True)
    print(f"STAGE -> s3://{args.bucket}/{args.prefix}/  (region={args.region}, "
          f"profile={args.profile or 'default'}, dry_run={args.dry_run})", flush=True)
    print("=" * 78, flush=True)

    # --- build manifests first (works without creds; the dry-run path stops here) ---
    data_manifest = [] if args.skip_data else collect_data_manifest(args.include_data_tsrv)
    code_bytes = b"" if args.skip_code else build_code_tarball()

    data_total = sum(sz for _, _, sz in data_manifest)
    print(f"\nDATA manifest: {len(data_manifest)} objects, {_human(data_total)} total", flush=True)
    # show a compact head; full per-file list only if small
    for abs_path, key_suffix, sz in data_manifest[:40]:
        print(f"  data/{key_suffix:<48} {_human(sz):>10}", flush=True)
    if len(data_manifest) > 40:
        print(f"  ... +{len(data_manifest) - 40} more objects", flush=True)

    if not args.skip_code:
        print(f"\nCODE tarball: {code_key}  ({_human(len(code_bytes))})", flush=True)

    grand_total = data_total + len(code_bytes)
    print(f"\nGRAND TOTAL to stage: {_human(grand_total)} ({grand_total} bytes)", flush=True)

    if args.dry_run:
        print("\n[dry-run] No S3 calls made. Re-run without --dry-run (needs bucket + creds).", flush=True)
        return 0

    # --- real upload path: needs boto3 + creds + an existing bucket ---
    import boto3

    session = boto3.Session(profile_name=args.profile) if args.profile else boto3.Session()
    s3 = session.client("s3", region_name=args.region)

    uploaded_bytes = 0
    skipped = 0

    if not args.skip_data:
        print("\n--- uploading data (skip-if-exists) ---", flush=True)
        for abs_path, key_suffix, sz in data_manifest:
            key = f"{data_prefix}/{key_suffix}"
            if _key_exists(s3, args.bucket, key):
                skipped += 1
                continue
            s3.upload_file(str(abs_path), args.bucket, key)
            uploaded_bytes += sz
            print(f"  put s3://{args.bucket}/{key}  ({_human(sz)})", flush=True)

    if not args.skip_code:
        print("\n--- uploading code tarball (always overwrite) ---", flush=True)
        s3.put_object(Bucket=args.bucket, Key=code_key, Body=code_bytes)
        uploaded_bytes += len(code_bytes)
        print(f"  put s3://{args.bucket}/{code_key}  ({_human(len(code_bytes))})", flush=True)

    print("\n" + "=" * 78, flush=True)
    print(f"DONE. uploaded {_human(uploaded_bytes)} new; skipped {skipped} existing data objects.", flush=True)
    print(f"  code   -> s3://{args.bucket}/{code_key}", flush=True)
    print(f"  data   -> s3://{args.bucket}/{data_prefix}/", flush=True)
    print("=" * 78, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
