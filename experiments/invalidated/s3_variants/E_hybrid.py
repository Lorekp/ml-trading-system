"""Invalidated forensic experiment — S3 Theme E: ADAPTIVE / HYBRID.

Single-lever variants came close (A2 p=0.95, B1_dte7 p=0.97, B3 p=0.95,
D2c p=0.99) but only the borderline D2c cleared the K=40 bar at the cost
of defensibility. This script tests whether HYBRIDIZING the close calls
clears DSR with strong defensibility.

- E1: A2 (IV-rank ≤ 95th) + B1_dte7 (DTE ≥ 7) — both filters are
      pre-committed candidates. Hybridizing them should cut the
      worst-tail trades harder than either alone.
- E2: D2b (4σ cut, pre-committed §3.2) + A2 — the "premium stop AND
      skip high-IV" combined. A2 cuts ~5% of trades, D2b caps
      the worst single loss.
- E3: A2 + B3 (skew filter) — clean entries (low IV-rank) AND
      fat put-side premium (high skew). Both are positive VRP
      signals.
- E4: D2b + B1_dte7 — DTE floor (cuts near-expiry gamma tail) AND
      4σ cap. Pre-committed in spirit; tests if combining the
      two tail-reducers is enough.
- E5: full stack — D2b (4σ cap) + A2 (IV-rank ≤ 95) + B1_dte7
      (DTE ≥ 7). The most aggressive pre-committed hybrid.

Each: per-trade PnL, per-year, DSR p@K=40, MaxDD, mechanism, defensibility.

ADAPTIVE-FAILURE PROTOCOL: if 4-5 hybrids all fail the same way, that
IS the answer: the strategy is tail-budgeted, capacity-capped, NOT
flagship. Document deeply.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from experiments.invalidated.s3_variants._common import (
    load_frame, get_short_covered, evaluate_variant, write_variant_report, OUT_DIR,
)
from experiments.invalidated.s3_variants.A_sizing import variant_a2_regime_filter
from experiments.invalidated.s3_variants.B_entry_exit import variant_b1_dte_floor, variant_b3_skew_filter
from experiments.invalidated.s3_variants.D_statistical import variant_d2_cut_loss


# ── E1: A2 + B1_dte7 ────────────────────────────────────────────
def variant_e1(df: pd.DataFrame) -> pd.DataFrame:
    a2 = variant_a2_regime_filter(df, iv_rank_percentile=95.0)
    return variant_b1_dte_floor(a2, min_dte_days=7.0)


# ── E2: D2b (4σ) + A2 ───────────────────────────────────────────
def variant_e2(df: pd.DataFrame) -> pd.DataFrame:
    a2 = variant_a2_regime_filter(df, iv_rank_percentile=95.0)
    cut_df, _ = variant_d2_cut_loss(a2, z_cut=4.0)
    return cut_df


# ── E3: A2 + B3 (skew filter) ───────────────────────────────────
def variant_e3(df: pd.DataFrame) -> pd.DataFrame:
    a2 = variant_a2_regime_filter(df, iv_rank_percentile=95.0)
    return variant_b3_skew_filter(a2, skew_min=0.05)


# ── E4: D2b (4σ) + B1_dte7 ──────────────────────────────────────
def variant_e4(df: pd.DataFrame) -> pd.DataFrame:
    b1 = variant_b1_dte_floor(df, min_dte_days=7.0)
    cut_df, _ = variant_d2_cut_loss(b1, z_cut=4.0)
    return cut_df


# ── E5: full stack (D2b + A2 + B1_dte7) ─────────────────────────
def variant_e5(df: pd.DataFrame) -> pd.DataFrame:
    a2 = variant_a2_regime_filter(df, iv_rank_percentile=95.0)
    b1 = variant_b1_dte_floor(a2, min_dte_days=7.0)
    cut_df, _ = variant_d2_cut_loss(b1, z_cut=4.0)
    return cut_df


# ── E6: D2c (3.5σ) + A2 (IV-rank) — the more aggressive 3.5σ + IV-rank
def variant_e6(df: pd.DataFrame) -> pd.DataFrame:
    a2 = variant_a2_regime_filter(df, iv_rank_percentile=95.0)
    cut_df, _ = variant_d2_cut_loss(a2, z_cut=3.5)
    return cut_df


# ── E7: D2b (4σ) + A2 (IV-rank) + B3 (skew ≥ 0.05) — clean entries AND
#      fat-skew AND 4σ cap. A 3-way hybrid avoiding the aggressive DTE floor.
def variant_e7(df: pd.DataFrame) -> pd.DataFrame:
    a2 = variant_a2_regime_filter(df, iv_rank_percentile=95.0)
    b3 = variant_b3_skew_filter(a2, skew_min=0.05)
    cut_df, _ = variant_d2_cut_loss(b3, z_cut=4.0)
    return cut_df


# ── E8: A2 (IV-rank ≤ 80th — TIGHTER) + D2b (4σ) — the tighter A2
#      + the defensible 4σ cap. Tests if a tighter A2 + defensible
#      4σ together clear the bar.
def variant_e8(df: pd.DataFrame) -> pd.DataFrame:
    a2 = variant_a2_regime_filter(df, iv_rank_percentile=80.0)
    cut_df, _ = variant_d2_cut_loss(a2, z_cut=4.0)
    return cut_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{OUT_DIR}/E_hybrid.md")
    args = ap.parse_args()

    f = load_frame()
    scov = get_short_covered(f)
    print(f"[E_hybrid] base short-covered: n={len(scov)}")

    results = []

    r1 = evaluate_variant(
        variant_e1(scov), code="E1", name="A2 (IV-rank ≤ 95th) + B1 (DTE ≥ 7d)",
        mechanism="Hybrid: skip high-IV-rank entries AND skip near-expiry. "
                  "Two pre-committed tail reducers in series.",
    )
    results.append(r1)

    r2 = evaluate_variant(
        variant_e2(scov), code="E2", name="D2b (4σ cut) + A2 (IV-rank)",
        mechanism="Pre-committed 4σ cap (≈1.7× premium) AND skip high-IV-rank. "
                  "Cap protects the worst trade; filter improves the mean.",
    )
    results.append(r2)

    r3 = evaluate_variant(
        variant_e3(scov), code="E3", name="A2 (IV-rank) + B3 (skew ≥ 0.05)",
        mechanism="Clean entries (calm IV) AND fat put-side premium. "
                  "Two positive-VRP signals in series.",
    )
    results.append(r3)

    r4 = evaluate_variant(
        variant_e4(scov), code="E4", name="D2b (4σ) + B1 (DTE ≥ 7d)",
        mechanism="Pre-committed 4σ cap AND skip near-expiry. "
                  "Two pre-committed tail reducers in series.",
    )
    results.append(r4)

    r5 = evaluate_variant(
        variant_e5(scov), code="E5", name="D2b (4σ) + A2 (IV-rank) + B1 (DTE ≥ 7d)",
        mechanism="Full stack: 4σ cap + IV-rank filter + DTE floor. "
                  "Most aggressive pre-committed hybrid. If this doesn't clear DSR, "
                  "the strategy is genuinely tail-bound.",
    )
    results.append(r5)

    r6 = evaluate_variant(
        variant_e6(scov), code="E6", name="D2c (3.5σ) + A2 (IV-rank)",
        mechanism="D2c (3.5σ, more aggressive) + A2. Borderline-defensible "
                  "(3.5σ ≈ 1.4× premium). Tests if A2 + 3.5σ are enough.",
    )
    results.append(r6)

    r7 = evaluate_variant(
        variant_e7(scov), code="E7", name="D2b (4σ) + A2 (IV-rank) + B3 (skew ≥ 0.05)",
        mechanism="Defensible 4σ cap + IV-rank + fat-skew. Avoids the aggressive "
                  "DTE floor that killed E1/E5's 2026 year.",
    )
    results.append(r7)

    r8 = evaluate_variant(
        variant_e8(scov), code="E8", name="D2b (4σ) + A2b (IV-rank ≤ 80th, tighter)",
        mechanism="Tighter IV-rank filter (80th pct) + 4σ cap. "
                  "Tests if the A2b is the binding constraint, not the 4σ cap.",
    )
    results.append(r8)

    write_variant_report(results, "E — Hybrid", out_path=args.out,
        summary=f"Baseline (S3 frozen): n=666, per-trade $1,285, DSR p=0.7168 FAIL. "
                f"Goal: clear the K=40 DSR bar via HYBRIDIZING the close calls "
                f"(A2, B1_dte7, B3, D2b). If 4-5 hybrids all fail, that's the answer: "
                f"the strategy is tail-budgeted capacity-capped, NOT flagship.")


if __name__ == "__main__":
    main()
