#!/usr/bin/env python3
"""Shared constants for the quant vol-model AWS spot stack.

Single source of truth for bucket / region / prefix / IAM names, so the
setup / verify / smoke / launcher / babysitter scripts never drift.

Ported from the Pokerbot stack (scripts/setup_labelgen_iam.py,
scripts/verify_labelgen_iam.py, scripts/aws_smoke.py, and
infra/labelgen/iam_policy_labelgen_{launcher,worker_role}.json), generalised
for the ml-trading-system vol-forecast horizon sweep.

Deployment identity is never hard-coded. ``QUANT_BUCKET`` is required; the
remaining operational names can be overridden through the ``QUANT_*``
environment variables below.

Admin is required ONLY to run setup_quant_iam.py. Once setup has run, the
launcher (labelgen-worker) does everything else with the default profile.
"""

from __future__ import annotations

import os


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


# --- AWS region -----------------------------------------------------------
REGION = os.environ.get("QUANT_REGION", "eu-central-1")

# --- S3 layout ------------------------------------------------------------
# The bucket is deployment-specific and must be supplied explicitly.
BUCKET = _required_env("QUANT_BUCKET")
PREFIX = os.environ.get("QUANT_PREFIX", "volmodel/v1").strip("/")

# --- IAM principals -------------------------------------------------------
# The launcher is the existing IAM user whose creds are configured locally.
LAUNCHER_USER = os.environ.get("QUANT_LAUNCHER_USER", "labelgen-worker")
LAUNCHER_POLICY_NAME = os.environ.get("QUANT_LAUNCHER_POLICY", "quant-volmodel-launcher")

# The EC2 worker role + instance profile the spot boxes assume.
WORKER_ROLE = os.environ.get("QUANT_WORKER_ROLE", "quant-worker")
WORKER_PROFILE = os.environ.get("QUANT_WORKER_PROFILE", "quant-worker")
WORKER_POLICY_NAME = os.environ.get("QUANT_WORKER_POLICY", "quant-volmodel-worker-s3")


def bucket_arn() -> str:
    return f"arn:aws:s3:::{BUCKET}"


def objects_arn() -> str:
    """ARN covering every object under this experiment's prefix."""
    return f"arn:aws:s3:::{BUCKET}/{PREFIX}/*"


def s3_policy_resources() -> list[str]:
    """Resource[] for an S3 policy.

    IMPORTANT (the Pokerbot ListBucket footgun): BOTH the bare bucket ARN
    (for s3:ListBucket) AND the object-glob ARN (for Get/Put/Delete) must be
    present, or ListBucket 403s and the worker's skip-if-exists guard breaks.
    """
    return [bucket_arn(), objects_arn()]


def launcher_policy(*, resources: list[str] | None = None) -> dict:
    """Inline policy attached to the launcher IAM user.

    Ports infra/labelgen/iam_policy_labelgen_launcher.json, parametrised on
    the quant bucket/prefix. Three statements:
      1. S3 on the quant bucket/prefix (ship code+data, read results).
      2. EC2 RunInstances/Terminate/Describe*/GetConsoleOutput/CreateTags
         + iam:PassRole conditioned to ec2.amazonaws.com (so the user can
         only pass roles to EC2, never escalate elsewhere).
      3. iam:GetInstanceProfile (the launcher's pre-flight existence check).
    """
    res = resources if resources is not None else s3_policy_resources()
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "QuantLauncherS3",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                ],
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
                "Condition": {
                    "StringEquals": {"iam:PassedToService": "ec2.amazonaws.com"}
                },
            },
            {
                "Sid": "QuantLauncherIAMRead",
                "Effect": "Allow",
                "Action": ["iam:GetInstanceProfile"],
                "Resource": "*",
            },
        ],
    }


def worker_policy(*, resources: list[str] | None = None) -> dict:
    """Inline policy attached to the EC2 worker ROLE (assumed by spot boxes).

    Ports infra/labelgen/iam_policy_labelgen_worker_role.json. Scoped to the
    quant bucket/prefix only — the box pulls code+data and pushes its result
    shard; nothing else.
    """
    res = resources if resources is not None else s3_policy_resources()
    return {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "QuantWorkerS3",
                "Effect": "Allow",
                "Action": [
                    "s3:GetObject",
                    "s3:PutObject",
                    "s3:DeleteObject",
                    "s3:ListBucket",
                ],
                "Resource": res,
            }
        ],
    }


def ec2_assume_role_policy() -> dict:
    """Trust policy: EC2 may assume the worker role."""
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
