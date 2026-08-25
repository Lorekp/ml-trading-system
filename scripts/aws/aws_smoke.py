#!/usr/bin/env python3
"""AWS credential smoke test for the quant vol-model stack.

Ported/extended from Pokerbot scripts/aws_smoke.py (which only listed buckets).
Confirms (a) creds resolve via STS get-caller-identity and (b) the quant
result bucket is reachable via a HEAD. Needs NO admin — runs as the launcher
(default creds = IAM user 'labelgen-worker').

The bucket HEAD will 404/403 until setup_quant_iam.py has created the bucket;
that's reported, not fatal, so this doubles as a "has setup run yet?" probe.

Usage:
  python scripts/aws/aws_smoke.py
  QUANT_BUCKET=<bucket> python scripts/aws/aws_smoke.py --region eu-central-1
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _config as cfg  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="AWS creds + quant bucket smoke test")
    p.add_argument("--bucket", default=cfg.BUCKET)
    p.add_argument("--region", default=cfg.REGION)
    args = p.parse_args()

    out: dict[str, object] = {"bucket": args.bucket, "region": args.region}

    # 1. credentials resolve?
    try:
        ident = boto3.client("sts", region_name=args.region).get_caller_identity()
        out["caller_arn"] = ident.get("Arn")
        out["account"] = ident.get("Account")
        out["creds_ok"] = True
    except Exception as e:  # noqa: BLE001
        out["creds_ok"] = False
        out["creds_error"] = f"{type(e).__name__}: {e}"
        print(json.dumps(out, indent=2))
        sys.exit(1)

    # 2. quant bucket reachable? (HEAD — non-fatal if absent, just reported)
    s3 = boto3.client("s3", region_name=args.region)
    try:
        s3.head_bucket(Bucket=args.bucket)
        out["bucket_ok"] = True
        out["bucket_status"] = "reachable"
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "ClientError")
        out["bucket_ok"] = False
        out["bucket_status"] = code
        if code in ("404", "NoSuchBucket"):
            out["hint"] = "bucket missing — run scripts/aws/setup_quant_iam.py (admin) first"
        elif code == "403":
            out["hint"] = "bucket exists but launcher lacks access — check the launcher S3 policy"

    print(json.dumps(out, indent=2))
    # Smoke exits 0 if creds resolve; bucket absence is informational, not a
    # hard failure (the bucket is created by the separate admin setup step).
    sys.exit(0)


if __name__ == "__main__":
    main()
