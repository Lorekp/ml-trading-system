#!/usr/bin/env python3
"""One-time IAM + S3 bucket setup for the quant vol-model spot stack.

  >>> RUN WITH ADMIN-LEVEL CREDENTIALS. <<<

SELF-CONTAINED: this script has ZERO repo dependencies — only ``boto3`` + the
Python standard library. You can upload or paste it into **AWS CloudShell** and
run it directly:

    python3 setup_quant_iam.py --bucket <bucket>  # uses CloudShell's ambient console creds

CloudShell's console identity is admin-level, so no ``--profile`` is needed there.
``--profile`` stays available for local use against a named admin profile. The
script writes NO files (CloudShell home is ~1GB), it only mutates IAM + S3.

It mutates IAM (the launcher user's inline policy, the worker role, the worker
instance profile) and creates the S3 bucket. After it has run once, EVERYTHING
ELSE in the stack — verify_quant_iam.py, the launcher, the babysitter, and the
on-box worker — uses the ordinary launcher creds (the default profile = IAM user
'labelgen-worker'). The launcher user CANNOT and SHOULD NOT run this; it is
denied IAM/bucket-create by design.

Steps performed (all idempotent — EntityAlreadyExists / BucketAlreadyOwnedByYou
/ LimitExceeded are treated as OK):
  1. put_user_policy   — launcher inline policy on user labelgen-worker
                         (EC2 Run/Terminate/Describe*/GetConsoleOutput/CreateTags
                          + iam:PassRole->ec2.amazonaws.com + GetInstanceProfile
                          + S3 on the quant bucket)
  2. create_bucket     — the quant S3 bucket (region-aware LocationConstraint)
  3. create_role       — worker role with the EC2 trust policy
     put_role_policy   — worker S3 policy scoped to the bucket/prefix
  4. create_instance_profile + add_role_to_instance_profile  ('quant-worker')
  5. create_service_linked_role(spot.amazonaws.com)   — EntityAlreadyExists OK

CloudShell usage (admin, ambient creds):
  python3 setup_quant_iam.py --bucket <bucket> --region eu-central-1 --verify

Local usage (named admin profile):
  python3 scripts/aws/setup_quant_iam.py --profile admin --bucket <bucket> --region eu-central-1 --verify
"""

from __future__ import annotations

import argparse
import json
import sys

import boto3
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------------
# Inlined config (single source of truth duplicated here so the file is
# self-contained for CloudShell paste). These MUST match scripts/aws/_config.py.
# Every value is overridable on the CLI.
# ---------------------------------------------------------------------------
REGION = "eu-central-1"  # matches the existing pokerbot/account setup; NOT us-east-1
PREFIX = "volmodel/v1"
LAUNCHER_USER = "labelgen-worker"
LAUNCHER_POLICY_NAME = "quant-volmodel-launcher"
WORKER_ROLE = "quant-worker"
WORKER_PROFILE = "quant-worker"
WORKER_POLICY_NAME = "quant-volmodel-worker-s3"

# IAM status strings treated as success when summarising the run.
_OK_STATUSES = {"updated", "created", "exists", "ok", "already_attached"}


# ---------------------------------------------------------------------------
# Policy / trust documents (inlined; parametrised on bucket + prefix).
# ---------------------------------------------------------------------------
def _s3_resources(bucket: str, prefix: str) -> list[str]:
    """BOTH the bare bucket ARN (for s3:ListBucket) AND the object-glob ARN
    (Get/Put/Delete) must be present, or ListBucket 403s and the worker's
    skip-if-exists guard breaks (the Pokerbot ListBucket footgun)."""
    return [f"arn:aws:s3:::{bucket}", f"arn:aws:s3:::{bucket}/{prefix}/*"]


def _launcher_policy(bucket: str, prefix: str) -> dict:
    res = _s3_resources(bucket, prefix)
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "QuantLauncherS3",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
                "Resource": res,
            },
            {
                "Sid": "QuantLauncherEC2",
                "Effect": "Allow",
                "Action": [
                    "ec2:RunInstances",
                    "ec2:TerminateInstances",
                    "ec2:DescribeInstances",
                    "ec2:DescribeInstanceStatus",
                    "ec2:DescribeImages",
                    "ec2:DescribeSpotPriceHistory",
                    "ec2:GetConsoleOutput",
                    "ec2:CreateTags",
                    "iam:PassRole",
                ],
                "Resource": "*",
                "Condition": {"StringEquals": {"iam:PassedToService": "ec2.amazonaws.com"}},
            },
            {
                "Sid": "QuantLauncherIAMRead",
                "Effect": "Allow",
                "Action": ["iam:GetInstanceProfile"],
                "Resource": "*",
            },
        ],
    }


def _worker_policy(bucket: str, prefix: str) -> dict:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "QuantWorkerS3",
                "Effect": "Allow",
                "Action": ["s3:GetObject", "s3:PutObject", "s3:DeleteObject", "s3:ListBucket"],
                "Resource": _s3_resources(bucket, prefix),
            }
        ],
    }


def _ec2_assume_role_policy() -> dict:
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Effect": "Allow",
                "Principal": {"Service": "ec2.amazonaws.com"},
                "Action": "sts:AssumeRole",
            }
        ],
    }


def _session(*, profile: str | None) -> boto3.session.Session:
    # profile=None => CloudShell's ambient console creds (or local default profile).
    return boto3.Session(profile_name=profile) if profile else boto3.Session()


def _create_bucket(session: boto3.session.Session, *, bucket: str, region: str) -> str:
    """Idempotently create the result bucket. us-east-1 must NOT pass a
    LocationConstraint; every other region (incl. eu-central-1) must."""
    s3 = session.client("s3", region_name=region)
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=bucket)
        else:
            s3.create_bucket(
                Bucket=bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        return "created"
    except ClientError as e:
        code = e.response["Error"]["Code"]
        if code in ("BucketAlreadyOwnedByYou", "BucketAlreadyExists"):
            return "exists"
        return code


def setup(
    *,
    session: boto3.session.Session,
    region: str,
    bucket: str,
    prefix: str,
    launcher_user: str,
    worker_role: str,
    worker_profile: str,
) -> dict:
    iam = session.client("iam")
    results: dict[str, str] = {}

    launcher_pol = _launcher_policy(bucket, prefix)
    worker_pol = _worker_policy(bucket, prefix)
    assume = json.dumps(_ec2_assume_role_policy())

    # 1. launcher inline policy on the IAM user --------------------------------
    try:
        iam.put_user_policy(
            UserName=launcher_user,
            PolicyName=LAUNCHER_POLICY_NAME,
            PolicyDocument=json.dumps(launcher_pol),
        )
        results["launcher_policy"] = "updated"
    except ClientError as e:
        results["launcher_policy"] = e.response["Error"]["Code"]

    # 2. result bucket ---------------------------------------------------------
    results["create_bucket"] = _create_bucket(session, bucket=bucket, region=region)

    # 3. worker role + its S3 policy ------------------------------------------
    try:
        iam.create_role(
            RoleName=worker_role,
            AssumeRolePolicyDocument=assume,
            Description="quant vol-model EC2 spot worker (reads code+data, writes results)",
        )
        results["create_role"] = "created"
    except ClientError as e:
        results["create_role"] = (
            "exists" if e.response["Error"]["Code"] == "EntityAlreadyExists" else e.response["Error"]["Code"]
        )

    try:
        iam.put_role_policy(
            RoleName=worker_role,
            PolicyName=WORKER_POLICY_NAME,
            PolicyDocument=json.dumps(worker_pol),
        )
        results["worker_policy"] = "updated"
    except ClientError as e:
        results["worker_policy"] = e.response["Error"]["Code"]

    # 4. instance profile + attach role ---------------------------------------
    try:
        iam.create_instance_profile(InstanceProfileName=worker_profile)
        results["create_profile"] = "created"
    except ClientError as e:
        results["create_profile"] = (
            "exists" if e.response["Error"]["Code"] == "EntityAlreadyExists" else e.response["Error"]["Code"]
        )

    try:
        iam.add_role_to_instance_profile(InstanceProfileName=worker_profile, RoleName=worker_role)
        results["add_role_to_profile"] = "ok"
    except ClientError as e:
        code = e.response["Error"]["Code"]
        # LimitExceeded == the (only-allowed) role is already attached.
        results["add_role_to_profile"] = "already_attached" if code == "LimitExceeded" else code

    # 5. spot service-linked role ---------------------------------------------
    try:
        iam.create_service_linked_role(AWSServiceName="spot.amazonaws.com")
        results["spot_slr"] = "created"
    except ClientError as e:
        code = e.response["Error"]["Code"]
        results["spot_slr"] = "exists" if code in ("InvalidInput", "EntityAlreadyExists") else code

    return results


def _verify(session: boto3.session.Session, *, region: str, bucket: str, prefix: str,
            worker_profile: str) -> bool:
    """Inlined, self-contained post-setup verification (no repo dependency).

    Runs as the CURRENT identity (admin in CloudShell). Confirms the bucket
    exists + is writable, the worker instance profile exists, and the launcher
    user has its inline policy. This is a lighter check than the launcher-creds
    verify_quant_iam.py --strict gate (which the user runs locally afterward),
    but it proves the setup actually took effect."""
    ok = True
    s3 = session.client("s3", region_name=region)
    iam = session.client("iam")
    probe_key = f"{prefix}/_iam_probe/setup_probe.txt"

    print("\n--- post-setup verification (current identity) ---")
    # bucket round-trip
    try:
        s3.put_object(Bucket=bucket, Key=probe_key, Body=b"probe")
        s3.get_object(Bucket=bucket, Key=probe_key)
        s3.delete_object(Bucket=bucket, Key=probe_key)
        print(f"  [ok] S3 round-trip on s3://{bucket}/{prefix}/")
    except ClientError as e:
        ok = False
        print(f"  [FAIL] S3 round-trip: {e.response['Error']['Code']}")
    # instance profile
    try:
        iam.get_instance_profile(InstanceProfileName=worker_profile)
        print(f"  [ok] instance profile '{worker_profile}' exists")
    except ClientError as e:
        ok = False
        print(f"  [FAIL] get_instance_profile('{worker_profile}'): {e.response['Error']['Code']}")
    # launcher inline policy
    try:
        iam.get_user_policy(UserName=LAUNCHER_USER, PolicyName=LAUNCHER_POLICY_NAME)
        print(f"  [ok] launcher inline policy '{LAUNCHER_POLICY_NAME}' on user '{LAUNCHER_USER}'")
    except ClientError as e:
        ok = False
        print(f"  [FAIL] launcher policy: {e.response['Error']['Code']}")
    return ok


def _print_access_denied_help(*, caller_arn: str, region: str) -> None:
    print(
        f"""
ERROR: AccessDenied — caller is {caller_arn}

setup_quant_iam.py must run with ADMIN-level credentials (CloudShell console
identity, the root user, or an IAM admin), NOT the launcher user
'{LAUNCHER_USER}'. The launcher cannot modify IAM or create the bucket — that is
intentional (least privilege).

In AWS CloudShell (recommended — ambient admin creds, no profile needed):
  python3 setup_quant_iam.py --bucket <bucket> --region {region} --verify

Local, with a named admin profile in ~/.aws/credentials:
  python3 scripts/aws/setup_quant_iam.py --profile admin --bucket <bucket> --region {region} --verify

After admin setup, switch BACK to the launcher (default creds) and verify locally:
  python scripts/aws/verify_quant_iam.py --strict --instance-profile {WORKER_PROFILE}
""",
        file=sys.stderr,
    )


def main() -> None:
    p = argparse.ArgumentParser(
        description="One-time quant vol-model IAM + bucket setup (ADMIN credentials; CloudShell-ready)"
    )
    p.add_argument("--profile", default=None,
                   help="AWS profile with admin IAM/S3 perms. OMIT in CloudShell (uses ambient creds).")
    p.add_argument("--region", default=REGION)
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", default=PREFIX)
    p.add_argument("--launcher-user", default=LAUNCHER_USER)
    p.add_argument("--worker-role", default=WORKER_ROLE)
    p.add_argument("--worker-profile", default=WORKER_PROFILE)
    p.add_argument("--verify", action="store_true",
                   help="Run an inlined post-setup verification as the current identity.")
    args = p.parse_args()

    session = _session(profile=args.profile)
    caller = session.client("sts").get_caller_identity()
    caller_arn = caller.get("Arn", "")
    print(
        json.dumps(
            {
                "caller": caller_arn,
                "profile": args.profile or "(ambient/default creds)",
                "region": args.region,
                "bucket": args.bucket,
                "prefix": args.prefix,
            },
            indent=2,
        )
    )

    # Refuse to run as the launcher user — it lacks the perms.
    if f"user/{args.launcher_user}" in caller_arn:
        _print_access_denied_help(caller_arn=caller_arn, region=args.region)
        sys.exit(2)

    results = setup(
        session=session,
        region=args.region,
        bucket=args.bucket,
        prefix=args.prefix,
        launcher_user=args.launcher_user,
        worker_role=args.worker_role,
        worker_profile=args.worker_profile,
    )
    print(json.dumps(results, indent=2))

    failed = [k for k, v in results.items() if v not in _OK_STATUSES]
    if failed:
        if "AccessDenied" in results.values():
            _print_access_denied_help(caller_arn=caller_arn, region=args.region)
        else:
            print(f"Setup incomplete — failed: {failed}", file=sys.stderr)
        sys.exit(1)

    print(
        "\nSetup complete. Now verify locally as the launcher (default credentials):\n"
        f"  python scripts/aws/verify_quant_iam.py --strict "
        f"--region {args.region} --bucket {args.bucket} --prefix {args.prefix} "
        f"--instance-profile {args.worker_profile}",
        file=sys.stderr,
    )

    if args.verify:
        ok = _verify(
            session, region=args.region, bucket=args.bucket,
            prefix=args.prefix, worker_profile=args.worker_profile,
        )
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
