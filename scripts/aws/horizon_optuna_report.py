#!/usr/bin/env python3
"""Per-horizon Optuna-budget report: does a finished volmodel want MORE training/Optuna?

THE SIGNAL. Each walk-forward month runs Optuna with `n_trials=max_trials` (default 200) and an
`OptunaPatience(patience)` early-stop callback (vol_model/run_walkforward.py:61) that stops the study
after `patience` (default 50) trials with no improvement of the best value. Therefore, per month:
  * n_trials  < max_trials  -> patience tripped -> the search PLATEAUED (converged); more trials won't help.
  * n_trials == max_trials  -> never tripped patience -> the best was still improving within the last
                               `patience` trials when the budget ran out -> this month WANTS MORE trials.
The horizon-level "cap-hit rate" (fraction of months that ran to the cap) is the decision metric for
whether to raise --max-trials for that horizon.

SOURCES (per horizon), tried in order:
  1. s3://BUCKET/PREFIX/results/h{H}/logs/job.log   (authoritative: has the 'Optuna: N trials,
     patience P' line AND per-month 'trials=N'); uploaded by run_horizon_job at end-of-run.
  2. s3://BUCKET/PREFIX/checkpoints/h{H}/summary.json  (structured per_month[].n_trials; the cap is
     not stored there, so --max-trials / the 200 default is assumed).
  3. --local-log PATH or --local-summary PATH  (offline / testing).

USAGE (your AWS creds):
  PYTHONPATH=. .venv/Scripts/python.exe scripts/aws/horizon_optuna_report.py             # every finished horizon
  ... --horizon 480
  ... --local-log _opt_gate_ref/g3_t1.log         # test the parser on a local log
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "aws"))

DEFAULT_HORIZONS = (5, 15, 30, 60, 240, 480)
DEFAULT_MAX_TRIALS = 200  # production walk-forward budget (feedback_optuna_prod_config)

_CAP_RE = re.compile(r"Optuna:\s*(\d+)\s*trials,\s*patience\s*(\d+)")
# per-month COMPLETION line: "... [49/50] 2026-02: qlike=0.3406 ... trials=7 (6s)"
_MONTH_RE = re.compile(r"\[(\d+)/(\d+)\]\s*(\d{4}-\d{2}):\s*qlike=([0-9.]+).*?trials=(\d+)")


def parse_job_log(text: str):
    """Return (max_trials_or_None, patience_or_None, [ {month, qlike, trials} ])."""
    cap = pat = None
    m = _CAP_RE.search(text)
    if m:
        cap, pat = int(m.group(1)), int(m.group(2))
    months = [
        {"month": g.group(3), "qlike": float(g.group(4)), "trials": int(g.group(5))}
        for g in _MONTH_RE.finditer(text)
    ]
    return cap, pat, months


def parse_summary(text: str):
    d = json.loads(text)
    months = [
        {"month": r["month"], "qlike": r.get("pruned_svi_qlike"), "trials": int(r["n_trials"])}
        for r in d.get("per_month", []) if r.get("n_trials") is not None
    ]
    return None, None, months  # summary.json does not store the cap


def analyse(cap, months, assumed_cap: int):
    n = len(months)
    if n == 0:
        return None
    cap_known = cap is not None
    cap = cap if cap_known else assumed_cap
    at_cap = [m for m in months if m["trials"] >= cap]
    rate = len(at_cap) / n
    recent = months[-12:]
    rrate = sum(1 for m in recent if m["trials"] >= cap) / len(recent)
    by_year = defaultdict(lambda: [0, 0])
    for m in months:
        y = m["month"][:4]
        by_year[y][1] += 1
        if m["trials"] >= cap:
            by_year[y][0] += 1
    tr = sorted(m["trials"] for m in months)
    if rate == 0.0:
        verdict = "CONVERGED - budget ample; no month hit the cap, so more Optuna trials won't help."
    elif rate <= 0.15:
        verdict = "MOSTLY CONVERGED - a few months are budget-limited; marginal upside from more trials."
    elif rate <= 0.40:
        verdict = "PARTIALLY BUDGET-LIMITED - worth a higher --max-trials for this horizon."
    else:
        verdict = "BUDGET-LIMITED - raise --max-trials; many months were still improving at the cap."
    return {
        "n_months": n, "cap": cap, "cap_known": cap_known,
        "cap_hits": len(at_cap), "cap_hit_rate": rate, "recent12_rate": rrate,
        "trials_min": tr[0], "trials_median": tr[n // 2], "trials_max": tr[-1],
        "per_year": {y: (v[0], v[1]) for y, v in sorted(by_year.items())},
        "verdict": verdict,
    }


def _fmt(h, r):
    if r is None:
        return f"  h{h:<4} : no per-month trial data found (not finished, or log/summary absent)."
    capnote = "" if r["cap_known"] else " (cap ASSUMED)"
    head = (f"  h{h:<4} : {r['n_months']} months | cap={r['cap']}{capnote} | "
            f"hit cap {r['cap_hits']}/{r['n_months']} ({r['cap_hit_rate']*100:.0f}%) | "
            f"recent-12 {round(r['recent12_rate']*12)}/12 | "
            f"trials med {r['trials_median']} [{r['trials_min']}..{r['trials_max']}]")
    yr = "  ".join(f"{y}:{a}/{b}" for y, (a, b) in r["per_year"].items())
    caveat = "" if r["cap_known"] else "  [cap ASSUMED=200; if a reduced/smoke run pass --max-trials for the true cap-hit rate]"
    return f"{head}\n         per-year cap-hits: {yr}\n         => {r['verdict']}{caveat}"


def _get_s3_text(s3, bucket, key):
    try:
        return s3.get_object(Bucket=bucket, Key=key)["Body"].read().decode("utf-8", "replace")
    except Exception:
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--horizon", type=int, action="append", default=None,
                    help="horizon minutes; repeatable. Default: all finished horizons.")
    ap.add_argument("--max-trials", type=int, default=DEFAULT_MAX_TRIALS,
                    help="assumed cap if a source lacks the 'Optuna: N trials' line (default 200).")
    ap.add_argument("--local-log", default=None, help="parse a local job-log file instead of S3.")
    ap.add_argument("--local-summary", default=None, help="parse a local summary.json instead of S3.")
    ap.add_argument("--bucket", default=os.environ.get("QUANT_BUCKET", "").strip() or None)
    ap.add_argument("--prefix", default=os.environ.get("QUANT_PREFIX", "volmodel/v1"))
    ap.add_argument("--region", default=os.environ.get("QUANT_REGION", "eu-central-1"))
    args = ap.parse_args(argv)

    # offline single-file mode (testing)
    if args.local_log or args.local_summary:
        if args.local_log:
            cap, pat, months = parse_job_log(Path(args.local_log).read_text(encoding="utf-8", errors="replace"))
        else:
            cap, pat, months = parse_summary(Path(args.local_summary).read_text(encoding="utf-8"))
        r = analyse(cap, months, args.max_trials)
        src = args.local_log or args.local_summary
        print(f"=== Optuna-budget report (local: {src}) ===")
        print(_fmt(args.horizon[0] if args.horizon else 0, r))
        return 0

    if not args.bucket:
        ap.error("--bucket is required unless QUANT_BUCKET is set")

    import boto3
    s3 = boto3.client("s3", region_name=args.region)

    horizons = args.horizon if args.horizon else list(DEFAULT_HORIZONS)
    print(f"=== Optuna-budget report  s3://{args.bucket}/{args.prefix}  ===")
    print("    cap-hit = a month whose study ran the full trial budget => still improving => wants more.\n")
    any_done = False
    for h in horizons:
        cap = months = None
        # 1) authoritative job.log
        txt = _get_s3_text(s3, args.bucket, f"{args.prefix}/results/h{h}/logs/job.log")
        if txt:
            cap, _pat, months = parse_job_log(txt)
        # 2) fallback: checkpoints summary.json
        if not months:
            txt = _get_s3_text(s3, args.bucket, f"{args.prefix}/checkpoints/h{h}/summary.json")
            if txt:
                cap, _pat, months = parse_summary(txt)
        r = analyse(cap, months, args.max_trials) if months else None
        if r:
            any_done = True
        print(_fmt(h, r))
    if not any_done:
        print("\n(no finished horizons yet - run again once a shard appears in results/h*/)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
