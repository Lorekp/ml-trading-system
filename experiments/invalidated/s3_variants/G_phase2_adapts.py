"""Invalidated forensic experiment — S3 Phase 2 adaptations of F2.

Phase 2 finding: F2 (term slope (1w-1m) ≥ 0, contango-only) PASSES the K=40
DSR bar at p=1.0000 with MaxDD -$5,205 and worst -$2,342. STRONGER than
E2 (MaxDD -$10,808, worst -$3,968) and uses a different mechanism (term
structure filter, not IV-rank + cut-loss).

This script tests the NATURAL hybrids of F2 with the other close-call
filters, to find the BEST defensible Phase-2 configuration:

- G1: F2 (contango) + D2b (4σ cut) — F2 already filters the worst 3 trades;
      a cut-loss might add additional tail protection.
- G2: F2 (contango) + A2 (IV-rank ≤ 95th) — both filters are pre-committed
      regime gates; test if they complement.
- G3: F2 (contango) + A2 + D2b — 3-way: contango + calm IV + 4σ cap.
- G4: F2 (contango) + F1b (VRP ≥ -10) — the OPPOSITE-sign-of-A2 + F2.
- G5: F2 (contango) + D2c (3.5σ cut) — slightly more aggressive cut than G1.
- G6: F2 + B1_dte7 — DTE floor + contango. DTE floor killed 2026 in E5;
      test if F2's contango filter ALSO cuts the 2026 tail.
- G7: F2 alone at a TIGHTER threshold (term ≥ 5) — gradient test.
- G8: F2 alone at a LOOSER threshold (term ≥ -5) — gradient test.

DSR bar = K=40 NOT relaxed. Engine UNTOUCHED.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.invalidated.s3_variants._common import (
    load_frame, get_short_covered, evaluate_variant, write_variant_report, OUT_DIR,
)
from experiments.invalidated.s3_variants.A_sizing import variant_a2_regime_filter
from experiments.invalidated.s3_variants.B_entry_exit import variant_b1_dte_floor
from experiments.invalidated.s3_variants.D_statistical import variant_d2_cut_loss
from experiments.invalidated.s3_variants.F_phase2 import (
    variant_f1_vrp_gated, variant_f2_term_filter,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{OUT_DIR}/G_phase2_adapts.md")
    args = ap.parse_args()

    f = load_frame()
    scov = get_short_covered(f)
    print(f"[G_phase2_adapts] base short-covered: n={len(scov)}")

    results = []

    # G1 — F2 + D2b (4σ cut)
    f2 = variant_f2_term_filter(scov, term_min=0.0)
    df_g1, _ = variant_d2_cut_loss(f2, z_cut=4.0)
    r1 = evaluate_variant(
        df_g1, code="G1", name="F2 (contango) + D2b (4σ cut)",
        mechanism="F2 (contango filter) plus 4σ cut-loss. F2 already drops the "
                  "worst 3 trades; the cut-loss adds residual tail protection.",
    )
    results.append(r1)

    # G2 — F2 + A2 (IV-rank)
    a2 = variant_a2_regime_filter(scov, iv_rank_percentile=95.0)
    df_g2 = variant_f2_term_filter(a2, term_min=0.0)
    r2 = evaluate_variant(
        df_g2, code="G2", name="F2 (contango) + A2 (IV-rank ≤ 95th)",
        mechanism="Two pre-committed regime filters in series. Both gates test "
                  "different market dimensions: A2 = spot IV regime, F2 = term structure.",
    )
    results.append(r2)

    # G3 — F2 + A2 + D2b (3-way)
    a2 = variant_a2_regime_filter(scov, iv_rank_percentile=95.0)
    f2_a2 = variant_f2_term_filter(a2, term_min=0.0)
    df_g3, _ = variant_d2_cut_loss(f2_a2, z_cut=4.0)
    r3 = evaluate_variant(
        df_g3, code="G3", name="F2 + A2 + D2b (3-way)",
        mechanism="Three pre-committed short-vol rules: term-structure contango, "
                  "IV-rank filter, 4σ cap. Full conservative configuration.",
    )
    results.append(r3)

    # G4 — F2 + F1b (VRP ≥ -10)
    f1b = variant_f1_vrp_gated(scov, vrp_min=-10.0)
    df_g4 = variant_f2_term_filter(f1b, term_min=0.0)
    r4 = evaluate_variant(
        df_g4, code="G4", name="F2 (contango) + F1b (VRP ≥ -10)",
        mechanism="Contango + 'not too cheap VRP'. Two pre-committed vol-edge gates.",
    )
    results.append(r4)

    # G5 — F2 + D2c (3.5σ cut)
    f2 = variant_f2_term_filter(scov, term_min=0.0)
    df_g5, _ = variant_d2_cut_loss(f2, z_cut=3.5)
    r5 = evaluate_variant(
        df_g5, code="G5", name="F2 (contango) + D2c (3.5σ cut)",
        mechanism="Contango + slightly tighter 3.5σ cut.",
    )
    results.append(r5)

    # G6 — F2 + B1_dte7
    f2 = variant_f2_term_filter(scov, term_min=0.0)
    df_g6 = variant_b1_dte_floor(f2, min_dte_days=7.0)
    r6 = evaluate_variant(
        df_g6, code="G6", name="F2 (contango) + B1 (DTE ≥ 7d)",
        mechanism="Contango + DTE floor. Tests if F2's contango filter already "
                  "cuts the 2026 tail (B1 was the killer in E5).",
    )
    results.append(r6)

    # G7 — F2 alone at tighter threshold (term ≥ 5)
    df_g7 = variant_f2_term_filter(scov, term_min=5.0)
    r7 = evaluate_variant(
        df_g7, code="G7", name="F2 (term slope ≥ 5, tight)",
        mechanism="Gradient test: stricter contango filter. Tests if going "
                  "tighter helps or hurts the DSR.",
    )
    results.append(r7)

    # G8 — F2 alone at looser threshold (term ≥ -5)
    df_g8 = variant_f2_term_filter(scov, term_min=-5.0)
    r8 = evaluate_variant(
        df_g8, code="G8", name="F2 (term slope ≥ -5, loose)",
        mechanism="Gradient test: looser filter. Tests if going looser helps "
                  "(more trades) or hurts (less tail protection).",
    )
    results.append(r8)

    # G9 — F2 (contango) at term slope >= 2 (median of the distribution)
    df_g9 = variant_f2_term_filter(scov, term_min=2.0)
    r9 = evaluate_variant(
        df_g9, code="G9", name="F2 (term slope ≥ 2, median-ish)",
        mechanism="Tight at ~median of term distribution. Tests the sweet spot "
                  "between G7 (tight) and G8 (loose).",
    )
    results.append(r9)

    # G10 — F2 alone re-run (reference)
    df_f2 = variant_f2_term_filter(scov, term_min=0.0)
    r_f2 = evaluate_variant(
        df_f2, code="F2*", name="F2 (re-run, reference)",
        mechanism="Reference: re-run the F2 winner for direct comparison with G1-G9.",
    )
    results.append(r_f2)

    write_variant_report(results, "G — Phase 2 ADAPTS (F2 hybrids + gradients)",
                         out_path=args.out,
                         summary=f"F2 (term contango) PASSES the K=40 DSR bar at "
                                 f"p=1.0000 with MaxDD -$5,205 and worst -$2,342. "
                                 f"STRONGER than E2 (p=0.9916, MaxDD -$10,808). "
                                 f"G1-G9 test natural hybrids + gradient. Goal: "
                                 f"find the BEST defensible Phase-2 configuration.")


if __name__ == "__main__":
    main()
