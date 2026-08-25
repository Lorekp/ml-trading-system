#!/usr/bin/env python3
"""OPTIONAL pre-baked EBS data-volume snapshot for the vol-forecast spot fleet.

================================================================================
RECOMMENDATION (measured 2026-06-21): **DO NOT BUILD THIS. Keep data in S3.**
================================================================================
This script exists so the option is documented and runnable IF the economics
ever change, but the measured numbers say the snapshot does not pay off. By
default it ONLY prints the cost/benefit analysis (``--analyze``, the default);
it builds a real volume/snapshot only under the explicit ``--build`` flag, and
even then it is gated behind ``--i-understand-this-costs-money`` so a stray run
spends nothing.

WHY S3 WINS (the math)
----------------------
The 191GB raw aggTrades stays LOCAL and is never shipped. The cloud dataset is
only the per-horizon forecast inputs:

  * Full staged dataset in S3 ............ ~2.6 GB (3762 objects)
  * What ONE box actually downloads = its OWN horizon's shard, not all 2.6 GB:
        h=5  shard ... 547 MB   (the largest; 549K 5-min bars)
        h=15 shard ... 207 MB
        h=30 shard ... 112 MB
        h>=60 shard .. <=10 MB  (coarse horizons are tiny)
    + 2 shared csvs (~1 KB) + the 1.4 MB code tarball.

In-region (eu-central-1) S3 -> EC2 transfer is fast and free of inter-region
egress charges. `aws s3 cp --recursive` parallelizes across objects (default 10
concurrent); a c7i.4xlarge has up to ~12.5 Gbps (~1.5 GB/s) of network, and S3
sustains several hundred MB/s in-region. Even at a PESSIMISTIC 100 MB/s effective:

        h=5  (547 MB) -> ~5.5 s
        h=15 (207 MB) -> ~2.1 s
        h=30 (112 MB) -> ~1.1 s
        h>=60         -> <0.1 s

At a realistic ~300 MB/s the worst box is <2 s. The walk-forward job itself runs
**4-12 HOURS**. So the S3 download is <0.05% of box runtime, paid once per launch.

WHAT A SNAPSHOT WOULD SAVE vs COST
----------------------------------
  SAVE:  ~2-6 s of boot download per (re)launch, and a re-download on a spot
         relaunch (which is again only seconds).
  COST:  (1) build + maintain a 2nd EBS volume + snapshot; (2) snapshot storage
         (~$0.05/GB-month -> ~$0.13/mo for 2.6 GB, trivial but non-zero);
         (3) extra launcher complexity (BlockDeviceMappings SnapshotId, a mount
         + symlink step in user_data, a stale-snapshot rebuild story whenever the
         sidecars change); (4) a snapshot is REGION-pinned and goes stale the
         moment any per-horizon sidecar is regenerated, whereas S3 staging is
         already idempotent skip-if-exists and always current.

CONCLUSION: the saving (seconds) is dwarfed by the job runtime (hours) and the
re-download on relaunch is itself only seconds, so the snapshot's marginal value
is ~nil while it adds real operational surface. **Keep data in S3.** The fleet's
spot-resilience comes from the per-MONTH walk-forward CHECKPOINT (see
run_horizon_job.py checkpoint_*), not from avoiding a 2-second data pull.

IF YOU STILL WANT IT (gated, optional)
--------------------------------------
`--build` would: create a small gp3 volume in an AZ, attach it to a helper
instance (or use a short-lived builder), `aws s3 sync` the dataset onto it,
snapshot it, and print the SnapshotId. The launcher would then accept
`--data-snapshot snap-xxxx` and add a BlockDeviceMappings entry mounting it at
/opt/app-data, with user_data symlinking the sidecars into the repo layout and
SKIPPING the S3 data `aws s3 cp`. That launcher wiring is intentionally NOT added
(it would complicate the verified user_data for a sub-2-second saving). This file
is the single, clearly-gated home for the capability if the trade-off ever flips
(e.g. a future 50 GB on-box dataset).

NO AWS RESOURCES ARE CREATED unless BOTH --build AND --i-understand-this-costs-money
are passed. The default (--analyze) is free and touches nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "aws"))

# Measured per-horizon shard sizes (MB), 2026-06-21 (see module docstring).
SHARD_MB = {5: 547.3, 15: 207.3, 30: 111.6, 60: 10.0, 120: 10.0, 240: 10.0, 480: 10.0, 1440: 10.0}
STAGED_TOTAL_GB = 2.6


def analyze() -> int:
    print("=" * 78)
    print("EBS DATA-VOLUME SNAPSHOT — COST/BENEFIT (in-region eu-central-1 S3 -> EC2)")
    print("=" * 78)
    print(f"Full staged dataset in S3 : {STAGED_TOTAL_GB:.1f} GB (a box pulls only its own shard)")
    print("\nPer-box download time @ a few candidate effective throughputs:")
    print(f"  {'horizon':>8}  {'shard MB':>9}  {'@100MB/s':>9}  {'@300MB/s':>9}  {'@600MB/s':>9}")
    for h in sorted(SHARD_MB):
        mb = SHARD_MB[h]
        print(f"  h{h:>6}  {mb:>9.1f}  {mb/100:>8.1f}s  {mb/300:>8.1f}s  {mb/600:>8.2f}s")
    print("\nWalk-forward job runtime per box : ~4-12 HOURS")
    print("Download as a fraction of runtime : <0.05%  (worst case, h=5)")
    print("\nVERDICT: snapshot saves seconds vs a multi-hour job; re-download on a spot")
    print("         relaunch is again only seconds. The marginal benefit does NOT justify")
    print("         the added launcher complexity + region-pinned staleness.")
    print("         >>> KEEP DATA IN S3. Do not build the snapshot. <<<")
    print("\nSpot-resilience is delivered by the per-MONTH walk-forward checkpoint")
    print("(run_horizon_job.py: checkpoint_restore_down / checkpoint_sync_up), not by")
    print("avoiding a sub-2-second in-region data pull.")
    print("=" * 78)
    return 0


def build(region: str, az: str, confirmed: bool) -> int:
    if not confirmed:
        print("REFUSING to build: pass --i-understand-this-costs-money to actually create a "
              "volume + snapshot. (Default --analyze is free; the recommendation is to KEEP S3.)")
        return 2
    # Intentionally NOT implemented end-to-end here: building real AWS resources is out of scope
    # for this task (SPEND NO MONEY) and the recommendation is not to build it. The steps a real
    # implementation would take are documented below so a future operator can complete it if the
    # economics ever flip — but the launcher --data-snapshot wiring is deliberately left unbuilt.
    print("[build] (stub — no AWS calls made) The end-to-end steps would be:")
    print("  1. ec2.create_volume(Size=8, VolumeType='gp3', AvailabilityZone=az)  # ~8GB room for 2.6GB")
    print("  2. attach to a short-lived builder instance; mkfs.ext4; mount /mnt/voldata")
    print("  3. aws s3 sync s3://BUCKET/PREFIX/data/  /mnt/voldata/  (in-region, seconds)")
    print("  4. unmount; ec2.create_snapshot(VolumeId=...); wait completed; print SnapshotId")
    print("  5. ec2.delete_volume(...) (snapshot retains the data)")
    print("  Then launch with --data-snapshot snap-xxxx (BlockDeviceMappings + mount + skip S3 cp).")
    print("\nNot executed (spend-no-money + recommended-against). Re-read the analysis: --analyze")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--analyze", action="store_true", default=True,
                    help="(default) print the cost/benefit analysis; touch nothing, spend nothing.")
    ap.add_argument("--build", action="store_true",
                    help="OPTIONAL: build the data volume + snapshot (gated; not recommended).")
    ap.add_argument("--i-understand-this-costs-money", dest="confirmed", action="store_true",
                    help="required alongside --build to actually create AWS resources.")
    ap.add_argument("--region", default="eu-central-1")
    ap.add_argument("--az", default="eu-central-1a")
    args = ap.parse_args()

    if args.build:
        return build(args.region, args.az, args.confirmed)
    return analyze()


if __name__ == "__main__":
    sys.exit(main())
