#!/usr/bin/env python3
"""Cross-horizon roll-up for the vol-forecast sweep.

Reads each horizon's self-describing result shard (``shard.json``) — either from
a local directory tree (after ``aws s3 cp .../results/ ./local/ --recursive``) or
straight from S3 — and prints a comparison table:

    horizon | NC-Sharpe | worst-yr NC-Sh | #yrs>0 | RankIC | worst-yr RankIC | NC-PnL | QLIKE(diag)

Historical HR-1 ranking discipline:
- DO NOT rank horizons by absolute QLIKE / OOS-R². Target variance spans ~1.9×
  across horizons (var(y_true) 0.32 @h30 vs 0.61 @h5), so "lowest QLIKE" merely
  flags the lowest-variance target, NOT the best forecast. QLIKE / OOS-R² are
  printed as NON-RANKING DIAGNOSTICS only.
- RANK on downstream NC-Sharpe (PRIMARY, scale-free decision metric) + RankIC
  (scale-invariant forecast quality).
- SELECT on WORST-YEAR NC-Sharpe, not the pooled mean — a horizon that wins on
  pooled NC-Sharpe but collapses in one walk-forward year is a cherry-pick. The
  shards already emit ``forecast.per_year`` and ``downstream.per_year_nc_sharpe``;
  this rollup reads them and surfaces worst-year NC-Sharpe, #years NC-Sharpe>0,
  and worst-year RankIC. This table is the deploy-selection input, not just a
  diagnostic.

Usage:
  # local (after pulling shards down):
  python scripts/aws/rollup_horizons.py --dir vol_model/output/sweep_results
  # straight from S3:
  python scripts/aws/rollup_horizons.py --bucket <bucket> \
      --prefix volmodel/v1 --region eu-central-1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def _load_local(root: Path) -> list[dict]:
    shards = []
    for p in sorted(root.rglob("shard.json")):
        try:
            shards.append(json.loads(p.read_text()))
        except Exception as e:  # noqa: BLE001
            print(f"  [warn] skip {p}: {e}", file=sys.stderr)
    return shards


def _load_s3(bucket: str, prefix: str, region: str) -> list[dict]:
    import boto3

    s3 = boto3.client("s3", region_name=region)
    paginator = s3.get_paginator("list_objects_v2")
    shards = []
    results_prefix = f"{prefix}/results/"
    for page in paginator.paginate(Bucket=bucket, Prefix=results_prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if not key.endswith("/shard.json"):
                continue
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            try:
                shards.append(json.loads(body))
            except Exception as e:  # noqa: BLE001
                print(f"  [warn] skip s3://{bucket}/{key}: {e}", file=sys.stderr)
    return shards


def _g(d: dict | None, *keys, default=float("nan")):
    """Nested get with a default; tolerant of missing blocks."""
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _fmt(v, prec=4, width=10) -> str:
    if v is None:
        return f"{'n/a':>{width}}"
    try:
        if isinstance(v, float) and (v != v):  # NaN
            return f"{'n/a':>{width}}"
        return f"{v:>{width},.{prec}f}"
    except (TypeError, ValueError):
        return f"{str(v):>{width}}"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", default=None,
                   help="local dir tree containing results/h{H}/shard.json files")
    p.add_argument("--bucket", default=os.environ.get("QUANT_BUCKET", "").strip() or None)
    p.add_argument("--prefix", default="volmodel/v1")
    p.add_argument("--region", default="eu-central-1")
    args = p.parse_args()

    if args.dir:
        shards = _load_local(Path(args.dir))
        src = f"dir:{args.dir}"
    else:
        if not args.bucket:
            p.error("--bucket is required unless QUANT_BUCKET is set")
        shards = _load_s3(args.bucket, args.prefix, args.region)
        src = f"s3://{args.bucket}/{args.prefix}/results/"

    if not shards:
        print(f"No shards found in {src}", file=sys.stderr)
        return 1

    shards.sort(key=lambda s: s.get("horizon_minutes", 0))

    print(f"\nCross-horizon roll-up  ({len(shards)} shard(s) from {src})")
    print("=" * 116)
    # PRIMARY ranking columns first (NC-Sharpe family + RankIC); QLIKE/OOS-R2
    # are pushed right and flagged (diag) — non-ranking diagnostics (HR-1).
    hdr = (f"{'horizon':>8} | {'NC_Sharpe':>10} | {'wyr_NCSh':>10} | {'#yr>0':>6} | "
           f"{'RankIC':>9} | {'wyr_RankIC':>10} | {'NC_PnL_USD':>13} | "
           f"{'n_trades':>9} | {'QLIKE(diag)':>11} | {'OOS_R2(diag)':>12}")
    print(hdr)
    print("-" * len(hdr))

    rows = []
    for s in shards:
        h = s.get("horizon_minutes", "?")
        fc = s.get("forecast", {})
        dn = s.get("downstream", {})

        # Per-year NC-Sharpe (downstream) → worst-year + #years>0.
        ds_py = dn.get("per_year_nc_sharpe", []) or []
        yr_sharpes = [_g(y, "nc_sharpe") for y in ds_py]
        yr_sharpes = [v for v in yr_sharpes if isinstance(v, (int, float)) and v == v]
        worst_yr_sharpe = min(yr_sharpes) if yr_sharpes else float("nan")
        n_yrs_pos = sum(1 for v in yr_sharpes if v > 0) if yr_sharpes else None

        # Per-year RankIC (forecast) → worst-year RankIC.
        fc_py = fc.get("per_year", []) or []
        yr_ics = [_g(y, "rank_ic") for y in fc_py]
        yr_ics = [v for v in yr_ics if isinstance(v, (int, float)) and v == v]
        worst_yr_ic = min(yr_ics) if yr_ics else float("nan")

        row = {
            "h": h,
            "nc_sharpe": _g(dn, "nc_sharpe"),
            "worst_yr_sharpe": worst_yr_sharpe,
            "n_yrs_pos": n_yrs_pos,
            "rank_ic": _g(fc, "rank_ic_mean"),
            "worst_yr_ic": worst_yr_ic,
            "nc_pnl": _g(dn, "nc_pnl_usd"),
            "n_trades": dn.get("n_trades", None),
            # Diagnostics ONLY — never used to rank/select (HR-1).
            "qlike": _g(fc, "qlike_oos_mean"),
            "oos_r2": _g(fc, "oos_r2_pooled"),
            "status": dn.get("status", "?"),
        }
        rows.append(row)
        print(f"{str(h)+'m':>8} | {_fmt(row['nc_sharpe'], prec=2)} | "
              f"{_fmt(row['worst_yr_sharpe'], prec=2)} | "
              f"{(str(row['n_yrs_pos']) if row['n_yrs_pos'] is not None else 'n/a'):>6} | "
              f"{_fmt(row['rank_ic'], prec=4, width=9)} | "
              f"{_fmt(row['worst_yr_ic'], prec=4)} | "
              f"{_fmt(row['nc_pnl'], prec=0, width=13)} | "
              f"{(str(row['n_trades']) if row['n_trades'] is not None else 'n/a'):>9} | "
              f"{_fmt(row['qlike'], prec=4, width=11)} | "
              f"{_fmt(row['oos_r2'], prec=4, width=12)}")
    print("=" * 116)

    def _best(key, *, maximize):
        valid = [r for r in rows if isinstance(r[key], (int, float)) and r[key] == r[key]]
        if not valid:
            return None
        return (max if maximize else min)(valid, key=lambda r: r[key])

    # SELECTION (HR-1): primary = WORST-YEAR NC-Sharpe (anti-cherry-pick), with
    # pooled NC-Sharpe + RankIC as secondary rank-on metrics. QLIKE/OOS-R2 are
    # NOT ranked — they are scale-dependent across horizons (target variance
    # ~1.9×) and only diagnostic.
    best_worst_sharpe = _best("worst_yr_sharpe", maximize=True)
    best_sharpe = _best("nc_sharpe", maximize=True)
    best_ic = _best("rank_ic", maximize=True)
    best_worst_ic = _best("worst_yr_ic", maximize=True)

    print("Deploy-selection (HR-1 — rank on NC-Sharpe + RankIC; SELECT on worst-year NC-Sharpe):")
    if best_worst_sharpe:
        print(f"  >> SELECT  best WORST-YEAR NC-Sharpe : h={best_worst_sharpe['h']}m   "
              f"({best_worst_sharpe['worst_yr_sharpe']:.2f})   "
              f"[#years NC-Sharpe>0 = {best_worst_sharpe['n_yrs_pos']}]")
    if best_sharpe:
        print(f"     pooled NC-Sharpe (secondary)      : h={best_sharpe['h']}m   "
              f"({best_sharpe['nc_sharpe']:.2f})")
    if best_ic:
        print(f"     highest RankIC (scale-invariant)  : h={best_ic['h']}m   "
              f"({best_ic['rank_ic']:.4f})")
    if best_worst_ic:
        print(f"     best worst-year RankIC            : h={best_worst_ic['h']}m   "
              f"({best_worst_ic['worst_yr_ic']:.4f})")
    print("  (QLIKE / OOS-R2 are NON-RANKING diagnostics — scale-dependent across "
          "horizons; do NOT select on them.)")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
