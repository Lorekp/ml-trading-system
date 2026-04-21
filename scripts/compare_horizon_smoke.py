"""Compare per-horizon smoke results from Phase 2b.

Reads ``vol_model/output/horizon_smoke/h{N}m_no_tv/summary.json`` for
each horizon and prints a side-by-side table of per-month OOS QLIKE
and rank-IC.

Important caveat surfaced during the smoke run (2026-04-21):
- The SVI-features sidecar is fixed at 15-min resolution. After the
  feature-label inner join, the effective sampling at h=5m collapses
  to 15-min (every 3rd bar). The h=5m smoke is therefore degenerate —
  it tests "h=5 model on a sub-sampled 15-min grid", NOT a true
  5-min pipeline. A real h=5 production pipeline requires re-building
  SVI features at 5-min granularity (separate compute).
- h=30m and h=60m intersections are clean (15min divides 30min and
  60min cleanly).
- Cross-horizon ABSOLUTE QLIKE comparison is informative-only;
  prefer rank-IC (scale-invariant) for cross-horizon ranking.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


SMOKE_ROOT = Path("vol_model/output/horizon_smoke")
HORIZONS = [5, 15, 30, 60]


def main() -> int:
    summaries: dict[int, dict] = {}
    for h in HORIZONS:
        spath = SMOKE_ROOT / f"h{h}m_no_tv" / "summary.json"
        if not spath.exists():
            print(f"[h={h}m] no summary.json yet at {spath}")
            continue
        summaries[h] = json.loads(spath.read_text())

    if not summaries:
        print("No smoke summaries found — has any horizon completed yet?")
        return 1

    # Pull all distinct months across horizons
    all_months: list[str] = sorted({
        r["month"]
        for s in summaries.values()
        for r in s.get("per_month", [])
    })
    if not all_months:
        print("Summaries exist but no per-month entries — smoke did not produce results.")
        return 1

    print("=" * 100)
    print("HORIZON SMOKE — per-month OOS QLIKE  (lower = better forecast loss)")
    print("=" * 100)
    header = f"{'month':<10}" + "".join(f"  {f'h{h}m':>10}" for h in HORIZONS)
    print(header)
    print("-" * len(header))
    for m in all_months:
        cells = []
        for h in HORIZONS:
            s = summaries.get(h)
            v: str = "    --    "
            if s:
                for r in s.get("per_month", []):
                    if r["month"] == m and r.get("pruned_svi_qlike") is not None:
                        v = f"{r['pruned_svi_qlike']:>10.4f}"
                        break
            cells.append("  " + v)
        print(f"{m:<10}" + "".join(cells))

    # Avg row
    print("-" * len(header))
    avg_row = []
    for h in HORIZONS:
        s = summaries.get(h)
        if not s:
            avg_row.append("    --    ")
            continue
        vals = [
            r["pruned_svi_qlike"] for r in s.get("per_month", [])
            if r.get("pruned_svi_qlike") is not None
        ]
        avg_row.append(f"{np.mean(vals):>10.4f}" if vals else "    --    ")
    print(f"{'AVG':<10}" + "".join("  " + c for c in avg_row))

    # Rank-IC table (scale-invariant cross-horizon comparison)
    print()
    print("=" * 100)
    print("HORIZON SMOKE — per-month rank-IC  (Spearman rank correlation, higher = better)")
    print("=" * 100)
    print(header)
    print("-" * len(header))
    for m in all_months:
        cells = []
        for h in HORIZONS:
            s = summaries.get(h)
            v: str = "    --    "
            if s:
                for r in s.get("per_month", []):
                    if r["month"] == m and r.get("rank_ic") is not None:
                        v = f"{r['rank_ic']:>10.3f}"
                        break
            cells.append("  " + v)
        print(f"{m:<10}" + "".join(cells))
    print("-" * len(header))
    avg_row = []
    for h in HORIZONS:
        s = summaries.get(h)
        if not s:
            avg_row.append("    --    ")
            continue
        vals = [
            r["rank_ic"] for r in s.get("per_month", [])
            if r.get("rank_ic") is not None
        ]
        avg_row.append(f"{np.mean(vals):>10.3f}" if vals else "    --    ")
    print(f"{'AVG':<10}" + "".join("  " + c for c in avg_row))

    # Wall-clock per horizon
    print()
    print("Wall clock (h):")
    for h in HORIZONS:
        s = summaries.get(h)
        if s and "total_hours" in s:
            print(f"  h={h:>2}m  {s['total_hours']:.2f}h  ({s.get('n_months', 0)} months)")

    print()
    print("Caveats:")
    print("  - h=5m result is degenerate: SVI features are 15-min resolution,")
    print("    so the inner join collapses h=5 to effective h=15 sampling.")
    print("    A real h=5 pipeline needs SVI re-built at 5-min granularity.")
    print("  - QLIKE absolute values ARE NOT directly comparable across horizons")
    print("    (different target-distribution variances). rank-IC is the")
    print("    horizon-invariant metric — use it for cross-horizon ordering.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
