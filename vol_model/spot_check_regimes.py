"""Regime-hardness substrate selection for signal-improvement spot-checks.

Picks the set of months on which candidate signals must demonstrate
improvement over production pruned_svi_v1. Selection is deterministic and
reproducible from the existing walk-forward summary.json.

Methodology (from plan at ~/.claude/plans/starry-rolling-rabin.md):

  A. QLIKE-hardest: top-K months by absolute production QLIKE (harder = worse
     current performance, so a new signal has the most room to demonstrate
     real incremental skill).

  B. Macro hand-picks: months associated with the top DD episodes from the
     2026-04 drawdown investigation (BTC_dd_investigation.md) and Stage 18's
     largest QLIKE drops. Ensures the substrate includes named regime shifts.

Output is the UNION of A + B. Typical cardinality: 6-8 distinct months.
"""

from __future__ import annotations

import json
from pathlib import Path


# Macro-regime hand-picks keyed to DD investigation + pipeline history.
# Stored as a constant (not a runtime flag) so the substrate is reproducible.
MACRO_MONTHS: tuple[str, ...] = (
    "2022-11",   # FTX week -- top-1 DD -9.16%, credit-event shock
    "2024-11",   # Trump election -- top-2 DD -9.03%, scheduled regime shift
    "2023-07",   # Summer low-RV grind -- top-4 long-horizon DD
    "2024-12",   # Vol expansion -- Stage 18 biggest QLIKE drop (-41.7%)
)


def pick_regime_months(
    summary_json_path: str | Path,
    *,
    k_hardest: int = 4,
    include_easiest: int = 1,
    macro_months: tuple[str, ...] = MACRO_MONTHS,
) -> list[str]:
    """Return a sorted list of regime-substrate months (YYYY-MM).

    Parameters
    ----------
    summary_json_path
        Path to ``expanding_pruned_svi_v1/summary.json`` -- per-month QLIKE
        for the current production model.
    k_hardest
        Number of hardest months to pick by absolute production QLIKE
        (higher QLIKE = harder for production model).
    include_easiest
        Number of easiest months to include as a bracket sanity check.
    macro_months
        Explicit macro-regime months to union in.

    Returns
    -------
    list[str]
        Sorted unique list of YYYY-MM strings.

    Raises
    ------
    FileNotFoundError, ValueError if the summary.json cannot be read or
    is structurally invalid.
    """
    path = Path(summary_json_path)
    if not path.exists():
        raise FileNotFoundError(f"summary.json not found at {path}")

    data = json.loads(path.read_text())
    per_month = data.get("per_month")
    if not isinstance(per_month, list) or not per_month:
        raise ValueError(f"{path}: missing or empty 'per_month' key")

    # Sort by absolute production QLIKE (hardest = highest)
    sorted_desc = sorted(
        per_month,
        key=lambda r: r["pruned_svi_qlike"],
        reverse=True,
    )
    hardest = [r["month"] for r in sorted_desc[:k_hardest]]
    easiest = [r["month"] for r in sorted_desc[-include_easiest:]] if include_easiest else []

    union = set(hardest) | set(easiest) | set(macro_months)
    return sorted(union)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Print regime substrate months")
    parser.add_argument(
        "--summary",
        default="vol_model/output/models/expanding_pruned_svi_v1/summary.json",
    )
    args = parser.parse_args()

    months = pick_regime_months(args.summary)
    print(f"Regime substrate ({len(months)} months):")
    for m in months:
        print(f"  {m}")
