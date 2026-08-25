#!/usr/bin/env python3
"""Pre-launch IAM gate for the quant vol-model spot stack.

Runs as the LAUNCHER (default creds = IAM user 'labelgen-worker'); needs NO
admin. Probes every permission the launcher + babysitter rely on and exits
non-zero (under --strict) unless all pass, so it can gate the launcher in CI:

    python scripts/aws/verify_quant_iam.py --strict && python scripts/aws/launch_*.py ...

Ported from Pokerbot scripts/verify_labelgen_iam.py. Differences from the
Pokerbot version:
  * EC2 RunInstances is probed with DryRun=True and DryRunOperation is treated
    as success (permission present, nothing launched) — same as Pokerbot.
  * S3 is a REAL put/get/delete/list ROUND-TRIP against the quant bucket under
    an _iam_probe/ key (Pokerbot did the same). This needs the bucket to exist,
    i.e. setup_quant_iam.py must have run first.
  * Constants come from scripts/aws/_config.py (bucket/prefix/region/profile).

This script CAN be run right now by the labelgen-worker creds present in this
environment IF the bucket already exists; the EC2 DryRun + GetInstanceProfile
checks pass without admin. The S3 round-trip will fail with NoSuchBucket until
setup_quant_iam.py has created the bucket — that's the expected pre-setup state.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

sys.path.insert(0, str(Path(__file__).resolve().parent))
import _config as cfg  # noqa: E402

# Errors that mean "permission granted, just a fake/absent instance id".
_CONSOLE_OUTPUT_OK_ERRORS = {
    "InvalidInstanceID.NotFound",
    "InvalidInstanceID.Malformed",
    "IncorrectInstanceState",
}
_DRYRUN_OK = {"DryRunOperation"}
# CreateTags / TerminateInstances validate the (fake) instance id BEFORE the
# DryRun short-circuit, so against a placeholder id they return an id-validity
# error rather than DryRunOperation. Reaching the id check means the IAM auth
# check already passed — treat these as "permission granted".
_FAKE_ID_OK = {
    "DryRunOperation",
    "InvalidID",
    "InvalidInstanceID.Malformed",
    "InvalidInstanceID.NotFound",
}

# Canonical (Canonical Ubuntu) is owner 099720109477; we only need ANY valid
# AMI id for the DryRun. Resolve the newest jammy x86_64 at runtime so we don't
# hard-code a stale id; fall back to a placeholder if DescribeImages is denied.
_FALLBACK_AMI = os.environ.get("QUANT_PROBE_AMI", "ami-0faab6bdbac9486fb")


def _check(name: str, fn) -> dict:
    try:
        fn()
        return {"name": name, "ok": True}
    except ClientError as e:
        code = e.response.get("Error", {}).get("Code", "ClientError")
        msg = e.response.get("Error", {}).get("Message", str(e))
        return {"name": name, "ok": False, "error": code, "message": msg}
    except Exception as e:  # noqa: BLE001
        return {"name": name, "ok": False, "error": type(e).__name__, "message": str(e)}


def _mark_dryrun_ok(check: dict, *, ok_codes: set[str]) -> dict:
    if not check["ok"] and check.get("error") in ok_codes:
        check = {**check, "ok": True, "note": f"{check['error']} — permission granted"}
    return check


def _resolve_probe_ami(ec2) -> str:
    """Newest Canonical jammy x86_64 AMI, or the fallback if Describe fails."""
    try:
        resp = ec2.describe_images(
            Owners=["099720109477"],
            Filters=[
                {"Name": "name", "Values": ["ubuntu/images/hvm-ssd/ubuntu-jammy-22.04-amd64-server-*"]},
                {"Name": "state", "Values": ["available"]},
                {"Name": "architecture", "Values": ["x86_64"]},
            ],
        )
        imgs = sorted(resp.get("Images", []), key=lambda i: i["CreationDate"], reverse=True)
        if imgs:
            return imgs[0]["ImageId"]
    except ClientError:
        pass
    return _FALLBACK_AMI


def main() -> None:
    p = argparse.ArgumentParser(description="Verify quant vol-model launcher/worker IAM (pre-launch gate)")
    p.add_argument("--bucket", default=cfg.BUCKET)
    p.add_argument("--prefix", default=cfg.PREFIX)
    p.add_argument("--region", default=boto3.session.Session().region_name or cfg.REGION)
    p.add_argument(
        "--instance-profile",
        default=cfg.WORKER_PROFILE,
        help="Worker instance profile to verify exists (env: QUANT_WORKER_PROFILE)",
    )
    p.add_argument("--instance-type", default="c6i.2xlarge", help="Instance type for the RunInstances DryRun")
    p.add_argument(
        "--strict",
        action="store_true",
        help="Require ALL checks (incl. diagnostics + instance profile) to pass — the EC2 launch gate",
    )
    args = p.parse_args()

    ec2 = boto3.client("ec2", region_name=args.region)
    s3 = boto3.client("s3", region_name=args.region)
    sts = boto3.client("sts")
    iam = boto3.client("iam")

    identity = sts.get_caller_identity()
    probe_prefix = f"{args.prefix}/_iam_probe"
    probe_key = f"{probe_prefix}/probe.txt"
    probe_ami = _resolve_probe_ami(ec2)

    checks = [
        _check(
            "ec2:DescribeSpotPriceHistory",
            lambda: ec2.describe_spot_price_history(
                InstanceTypes=[args.instance_type],
                ProductDescriptions=["Linux/UNIX"],
                MaxResults=5,
            ),
        ),
        _check("ec2:DescribeImages", lambda: ec2.describe_images(ImageIds=[probe_ami])),
        _check("ec2:DescribeInstances", lambda: ec2.describe_instances(MaxResults=5)),
        _check("ec2:DescribeInstanceStatus", lambda: ec2.describe_instance_status(MaxResults=5)),
        _check(
            "ec2:GetConsoleOutput",
            lambda: ec2.get_console_output(InstanceId="i-00000000000000000", Latest=True),
        ),
        _check(
            "ec2:CreateTags",
            lambda: ec2.create_tags(
                Resources=["i-00000000000000000"],
                Tags=[{"Key": "Project", "Value": "quant-volmodel"}],
                DryRun=True,
            ),
        ),
        _check(
            "ec2:TerminateInstances",
            lambda: ec2.terminate_instances(InstanceIds=["i-00000000000000000"], DryRun=True),
        ),
        _check(
            "ec2:RunInstances",
            lambda: ec2.run_instances(
                ImageId=probe_ami,
                InstanceType=args.instance_type,
                MinCount=1,
                MaxCount=1,
                DryRun=True,
            ),
        ),
        _check(
            "ec2:RunInstances(spot)",
            lambda: ec2.run_instances(
                ImageId=probe_ami,
                InstanceType=args.instance_type,
                MinCount=1,
                MaxCount=1,
                DryRun=True,
                InstanceMarketOptions={
                    "MarketType": "spot",
                    "SpotOptions": {
                        "SpotInstanceType": "one-time",
                        "InstanceInterruptionBehavior": "terminate",
                    },
                },
            ),
        ),
        # Real S3 round-trip against the quant bucket (needs the bucket to exist).
        _check("s3:PutObject", lambda: s3.put_object(Bucket=args.bucket, Key=probe_key, Body=b"probe")),
        _check("s3:GetObject", lambda: s3.get_object(Bucket=args.bucket, Key=probe_key)),
        _check(
            "s3:ListBucket",
            lambda: s3.list_objects_v2(Bucket=args.bucket, Prefix=probe_prefix, MaxKeys=1),
        ),
        _check("s3:DeleteObject", lambda: s3.delete_object(Bucket=args.bucket, Key=probe_key)),
    ]

    if args.instance_profile:
        checks.append(
            _check(
                "iam:GetInstanceProfile",
                lambda: iam.get_instance_profile(InstanceProfileName=args.instance_profile),
            )
        )
        checks.append(
            _check(
                "ec2:RunInstances(with_profile)",
                lambda: ec2.run_instances(
                    ImageId=probe_ami,
                    InstanceType=args.instance_type,
                    MinCount=1,
                    MaxCount=1,
                    DryRun=True,
                    IamInstanceProfile={"Name": args.instance_profile},
                ),
            )
        )
    elif args.strict:
        checks.append(
            {
                "name": "iam:GetInstanceProfile",
                "ok": False,
                "error": "MissingInstanceProfile",
                "message": "Pass --instance-profile or set QUANT_WORKER_PROFILE",
            }
        )

    # Normalise DryRun / fake-id errors to success.
    # RunInstances DryRun returns a clean DryRunOperation; CreateTags /
    # TerminateInstances return an id-validity error against the placeholder id
    # but only AFTER passing the IAM auth check, so both classes mean "granted".
    run_dryrun_names = {
        "ec2:RunInstances",
        "ec2:RunInstances(spot)",
        "ec2:RunInstances(with_profile)",
    }
    fake_id_names = {"ec2:CreateTags", "ec2:TerminateInstances"}
    for i, c in enumerate(checks):
        if c["name"] == "ec2:GetConsoleOutput" and not c["ok"]:
            if c.get("error") in _CONSOLE_OUTPUT_OK_ERRORS:
                checks[i] = {**c, "ok": True, "note": f"{c['error']} — permission granted"}
        elif c["name"] in run_dryrun_names:
            checks[i] = _mark_dryrun_ok(c, ok_codes=_DRYRUN_OK)
        elif c["name"] in fake_id_names:
            checks[i] = _mark_dryrun_ok(c, ok_codes=_FAKE_ID_OK)

    core_names = {"s3:PutObject", "s3:ListBucket", "ec2:RunInstances"}
    core_ok = all(c["ok"] for c in checks if c["name"] in core_names)
    all_ok = all(c["ok"] for c in checks)

    report = {
        "identity": identity.get("Arn"),
        "account": identity.get("Account"),
        "region": args.region,
        "bucket": args.bucket,
        "prefix": args.prefix,
        "instance_profile": args.instance_profile or None,
        "probe_ami": probe_ami,
        "principal": "launcher",
        "strict": args.strict,
        "worker_auth": "instance_role_via_IamInstanceProfile (the spot box assumes quant-worker)",
        "all_ok": all_ok,
        "core_ok": core_ok,
        "checks": checks,
    }
    print(json.dumps(report, indent=2))

    if args.strict:
        sys.exit(0 if all_ok else 1)
    sys.exit(0 if core_ok else 1)


if __name__ == "__main__":
    main()
