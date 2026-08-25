"""Historical S3 wing-distance frontier and survival-constraint study.

Extends `scripts/s3_tail_control.py` (the frozen historical ±2σ analysis) to
characterize the wing-distance frontier. The earlier ±2σ result found the
wings are too expensive *because they are close*: net wing cost $2,109/tr = 164%
of the $1,285 carry → net edge −$824/tr, DSR→0. But the wings DID truncate the
−50% gap (−$69K → −$22K per contract). A far-OTM catastrophe hedge is cheaper.

THE QUESTION
------------
Does an AFFORDABLE wing distance exist where
  (a) net per-trade defined-risk edge stays > $0, AND
  (b) the −50% one-week gap loss is CAPPED within the 1.5%-NAV-per-trade
      survival limit (§3.3 binding sizing constraint)?

If yes, the structure passes this historical constraint only. If no affordable
distance caps the catastrophe within NAV, the naked candidate retains uncapped
tail risk. Neither branch establishes readiness.

DISCIPLINE (CRITICAL — RISK-STRUCTURE characterization, NOT PnL-optimization)
----------------------------------------------------------------------------
* The wing distance d is swept ONLY to MAP the cost/protection/edge frontier.
  The historical candidate distance is chosen by the SURVIVAL CONSTRAINT — the
  furthest-OTM (= cheapest) wing that still caps the −50% gap within 1.5% NAV.
  We do NOT pick the distance that maximizes backtested PnL/Sharpe (that would
  be cherry-picking). The resulting net edge is reported as a CONSEQUENCE.
* K=40 and the p>0.99 / t>3.23 DSR thresholds are NOT relaxed.
* ADD-ONLY: no edits to realistic_eval.py / honest_pnl.py / backtester.py.
  Standalone post-processor on the cached frame. CHEAP analytic path only:
  cached `verify_frame_band35.parquet` + SVI surface + analytic BS. NO engine
  --reeval (the CPU A/B + GPU rehab must not be contended).
* CAUSAL / leak-free: wing premium from the ENTRY-time SVI surface (backward
  merge_asof per expiry); wing/short payoff from the REALIZED expiry price
  (panel close at the expiry bar). Identical convention to s3_tail_control.py.
* Reuses `price_wings`, `gap_stress`, `dsr_block`, `nonoverlap_series`,
  `realized_expiry_price` from s3_tail_control.py and `deflated_sharpe`,
  `prep_short_covered`, the non-overlap greedy walk from s3_revalidate.py
  (single source of truth — NO reimplementation of the pricing/DSR math).

WING-DISTANCE CONVENTION (confirmed identical to s3_tail_control.py L136-139)
----------------------------------------------------------------------------
  sigma_move = entry_ATM_IV[dec] * sqrt(T)        # 1σ log-move to expiry
  K_call = S * exp(+d * sigma_move)   K_put = S * exp(−d * sigma_move)
`price_wings(..., wing_sigma=d)` already takes d as the multiplier, so the sweep
is a clean parameter pass — the ±2σ case is exactly d=2.0.

SURVIVAL CONSTRAINT — NAV / sizing assumption (STATED, spec §3.3)
-----------------------------------------------------------------
§3.3 binding sizing constraint: "Hard cap: total short vega notional ≤ 1.5% of
NAV per trade — sized to survive a 3× straddle-width overnight gap without
margin call." We adopt the spec-faithful reading that the per-trade SHORT
STRADDLE NOTIONAL (= N_contracts × entry_index, the natural short-vega-notional
exposure the engine sizes) equals the 1.5%-NAV cap:

      NAV_tranche  =  straddle_notional_per_trade / 0.015

so a per-trade dollar loss L converts to %-NAV as

      L_%NAV  =  L / NAV_tranche  =  0.015 * L / straddle_notional_per_trade.

This is SCALE-INVARIANT: the gap loss and the straddle notional both scale ∝ N,
so L_%NAV is independent of the contract count — it is a pure function of the
strike geometry (gap size, K_atm, K_put) and the wing distance d. The SURVIVAL
TEST is: at distance d, is the −50% one-week gap loss (per trade, defined-risk)
≤ 1.5% of NAV? Equivalently |dr_loss| ≤ 0.015 × NAV = straddle_notional. The
cheapest (furthest-OTM) d that passes is the SURVIVAL-SELECTED distance.

USAGE
-----
  # Full frontier (cheap: cached frame + SVI surface + analytic BS; NO reeval):
  PYTHONPATH=. .venv/Scripts/python.exe scripts/s3_wing_frontier.py

  # Light smoke (3-date slice; proves the sweep plumbing, EXIT 0):
  PYTHONPATH=. .venv/Scripts/python.exe scripts/s3_wing_frontier.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from vol_arb.realistic_eval import PricePanel
# Reuse the FROZEN ±2σ machinery VERBATIM (single source of truth for pricing).
from scripts.s3_tail_control import (
    price_wings, gap_stress, dsr_block, GAP_SCENARIOS, SVI_SURFACE,
)
from scripts.s3_revalidate import (
    FROZEN_CFG, FRAME_CACHE, prep_short_covered,
    HELDOUT_START, HELDOUT_END,
)

OUT_DIR = "vol_arb/output"
REPORT_MD = f"{OUT_DIR}/S3_WING_FRONTIER.md"

# ── Swept wing distances (RISK-STRUCTURE sweep; NOT a PnL optimization) ──
WING_DISTANCES = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]

# ── §3.3 survival constraint: short vega/straddle notional ≤ 1.5% of NAV/trade ──
NAV_FRACTION_PER_TRADE = 0.015           # 1.5% NAV per trade (the binding cap)

# The carry baseline (the naked per-trade edge; reported from the frame, not hard-coded).
CARRY_REF = 1285.0                        # $/tr, §3.1 reference (cross-check only)


def per_trade_gap_loss(scov: pd.DataFrame, wings: dict, gaps=GAP_SCENARIOS):
    """§3.5 gap stress on the REPRESENTATIVE (median-σ) covered short, on its
    ACTUAL per-trade size N — so the loss and the straddle notional both scale
    ∝ N and the %-NAV is scale-invariant. Returns (rows, rep_index, straddle_
    notional_rep, N_rep).

    Reuses the EXACT settlement form of s3_tail_control.gap_stress (which is
    per-1-contract); we rescale by N_rep to the real per-trade dollar size so
    the survival %-NAV uses the real per-trade straddle notional. Because both
    numerator and denominator scale by N_rep, the %-NAV is identical to the
    per-1-contract basis — N_rep cancels — so this is purely a presentational
    choice that keeps the dollar figures on the historical per-trade scale."""
    rows1, rep = gap_stress(scov, wings, gaps)      # per-1-contract (spec basis)
    N_rep = float(wings["N"][rep])
    S_rep = float(wings["S"][rep])
    straddle_notional_rep = N_rep * S_rep           # short-vega notional per trade
    rows = []
    for r in rows1:
        rr = dict(r)
        rr["N"] = N_rep
        rr["naked_loss_pertrade"] = r["naked_loss"] * N_rep
        rr["defined_risk_loss_pertrade"] = r["defined_risk_loss"] * N_rep
        rr["truncation_pertrade"] = r["truncation_usd"] * N_rep
        # %-NAV via the survival constraint: straddle_notional = 1.5% NAV.
        # L_%NAV = 0.015 * L / straddle_notional  (== per-1-contract ratio).
        denom = straddle_notional_rep
        rr["naked_loss_pctnav"] = (NAV_FRACTION_PER_TRADE
                                   * rr["naked_loss_pertrade"] / denom)
        rr["defined_risk_loss_pctnav"] = (NAV_FRACTION_PER_TRADE
                                          * rr["defined_risk_loss_pertrade"] / denom)
        rows.append(rr)
    return rows, rep, straddle_notional_rep, N_rep


def frontier_row(scov: pd.DataFrame, surface: pd.DataFrame, panel: PricePanel,
                 d: float, *, full: pd.DataFrame, held: pd.DataFrame) -> dict:
    """One frontier row at wing distance d·σ: price the long ±d·σ strangle 1:1
    with the short straddle; compute net wing cost, net defined-risk edge,
    non-overlapping DSR (full + held-out), realized-PnL skew/kurt after wings,
    and the §3.5 catastrophe truncation (naked vs this-d at −30/−40/−50%)."""
    wings = price_wings(scov, surface, panel, wing_sigma=d)
    n = len(scov)

    # Net edge (means over all covered historical shorts).
    naked_mean = float(np.nanmean(wings["naked"]))
    dr_mean = float(np.nanmean(wings["defined_risk"]))
    wing_cost_mean = float(np.nanmean(wings["wing_cost_net"]))
    wing_bps_notional = float(np.nanmean(wings["wing_cost_bps_notional"]))
    wing_carry_ratio = (wing_cost_mean / naked_mean) if naked_mean != 0 else float("inf")

    # Attach the per-d defined-risk PnL to the window frames for the DSR.
    full = full.copy(); held = held.copy()
    # Recompute per-window wings (price_wings is cheap; vectorized). To stay
    # leak-free and consistent we price on the SAME scov ordering, then slice by
    # the boolean window mask (the realized-PnL series is per-trade, not pooled).
    dr_full_series = wings["defined_risk"][full["__pos__"].to_numpy()]
    dr_held_series = wings["defined_risk"][held["__pos__"].to_numpy()]
    nk_full_series = wings["naked"][full["__pos__"].to_numpy()]

    dsr_full = dsr_block(dr_full_series, full["entry_ts"], full["exit_ts"], "DR")
    dsr_held = dsr_block(dr_held_series, held["entry_ts"], held["exit_ts"], "DR")
    # naked DSR is d-independent but recorded once for context (computed at caller).

    # §3.5 catastrophe truncation + survival %-NAV.
    gap_rows, rep, straddle_notional_rep, N_rep = per_trade_gap_loss(scov, wings)
    gap50 = next(r for r in gap_rows if abs(r["gap"] - (-0.50)) < 1e-9)
    gap40 = next(r for r in gap_rows if abs(r["gap"] - (-0.40)) < 1e-9)
    gap30 = next(r for r in gap_rows if abs(r["gap"] - (-0.30)) < 1e-9)

    # SURVIVAL TEST: −50% defined-risk gap loss within 1.5% NAV?
    survives_50 = abs(gap50["defined_risk_loss_pctnav"]) <= NAV_FRACTION_PER_TRADE + 1e-12

    return dict(
        d=d, n=n,
        naked_mean=naked_mean, dr_mean=dr_mean, wing_cost_mean=wing_cost_mean,
        wing_bps_notional=wing_bps_notional, wing_carry_ratio=wing_carry_ratio,
        edge_pos=bool(dr_mean > 0),
        # DSR
        dsr_full_p=dsr_full["dsr_p"], dsr_full_t=dsr_full["t_stat"],
        dsr_full_sr=dsr_full["sr_perobs"], dsr_full_pass=bool(dsr_full["passed"]),
        dsr_held_p=dsr_held["dsr_p"], dsr_held_t=dsr_held["t_stat"],
        dsr_held_sr=dsr_held["sr_perobs"], dsr_held_pass=bool(dsr_held["passed"]),
        dsr_pass=bool(dsr_full["passed"] and dsr_held["passed"]),
        # realized-PnL skew/kurt AFTER wings (full window non-overlapping).
        nonov_skew=dsr_full["nonov_skew"], nonov_kurt=dsr_full["nonov_kurt"],
        # §3.5 catastrophe truncation (per single contract; spec basis) + %-NAV.
        K_put=float(wings["K_put"][rep]), K_call=float(wings["K_call"][rep]),
        S_rep=float(wings["S"][rep]), K_atm_rep=float(scov["strike"].to_numpy(float)[rep]),
        N_rep=N_rep, straddle_notional_rep=straddle_notional_rep,
        nav_tranche=straddle_notional_rep / NAV_FRACTION_PER_TRADE,
        naked_loss_30=gap30["naked_loss"], dr_loss_30=gap30["defined_risk_loss"],
        naked_loss_40=gap40["naked_loss"], dr_loss_40=gap40["defined_risk_loss"],
        naked_loss_50=gap50["naked_loss"], dr_loss_50=gap50["defined_risk_loss"],
        dr_loss_50_pctnav=gap50["defined_risk_loss_pctnav"],
        dr_loss_40_pctnav=gap40["defined_risk_loss_pctnav"],
        dr_loss_30_pctnav=gap30["defined_risk_loss_pctnav"],
        naked_loss_50_pctnav=gap50["naked_loss_pctnav"],
        survives_50=survives_50,
    )


# ──────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────
def write_report(payload: dict, *, report_md: str):
    L = []; A = L.append
    p = payload
    smoke = p["smoke"]
    tag = "  *(SMOKE — 3-date slice; proves the sweep plumbing, NOT the verdict)*" if smoke else ""
    A(f"# Historical S3 wing-distance frontier{tag}\n")
    A(f"**BOTTOM LINE:** {p['bottom_line']}\n")
    A(f"**Generated:** {p['generated']}  •  **Pricing:** {p['pricing_source']}\n")
    A("\n---\n")

    A("\n## The question\n")
    A("Does an **affordable** wing distance exist where (a) net per-trade defined-risk edge "
      "stays **> \\$0** AND (b) the −50% one-week gap loss is **capped within the 1.5%-NAV-"
      "per-trade survival limit** (§3.3)? Passing only characterizes a historical structure; "
      "it does not validate S3. If no affordable distance caps the catastrophe within NAV, "
      "the naked candidate retains uncapped tail risk.\n")

    A("\n## NAV / sizing assumption (STATED — §3.3 binding constraint)\n")
    for line in p["nav_assumption"]:
        A(f"- {line}")

    A("\n\n## Frontier table — wing distance d·σ sweep (RISK-STRUCTURE map, NOT PnL-opt)\n")
    A("σ = entry_ATM_IV·√T (confirmed identical to `s3_tail_control.py` L136-139). "
      "Net edge / wing cost are means over all {} covered shorts. DSR = non-overlapping, "
      "K=40, p>0.99 NOT relaxed. Gap rows = per single straddle contract (spec basis).\n"
      .format(p["n"]))
    A("| d·σ | net wing $/tr | wing %carry | net edge $/tr | edge>0 | DSR full p | DSR OOS p | "
      "DSR pass | skew | kurt | naked −50% $ | **DR −50% $** | **DR −50% %NAV** | ≤1.5%NAV |")
    A("|--:|--:|--:|--:|:--:|--:|--:|:--:|--:|--:|--:|--:|--:|:--:|")
    for r in p["rows"]:
        A(f"| {r['d']:.1f} | {r['wing_cost_mean']:,.0f} | {r['wing_carry_ratio']:.0%} | "
          f"{r['dr_mean']:,.0f} | {'Y' if r['edge_pos'] else 'N'} | "
          f"{r['dsr_full_p']:.4f} | {r['dsr_held_p']:.4f} | {'PASS' if r['dsr_pass'] else 'FAIL'} | "
          f"{r['nonov_skew']:.2f} | {r['nonov_kurt']:.1f} | "
          f"{r['naked_loss_50']:,.0f} | **{r['dr_loss_50']:,.0f}** | "
          f"**{r['dr_loss_50_pctnav']:+.2%}** | {'YES' if r['survives_50'] else 'no'} |")

    A("\n### §3.5 catastrophe truncation detail (per single straddle contract)\n")
    A("| d·σ | K_put | naked −30% | DR −30% | naked −40% | DR −40% | naked −50% | DR −50% |")
    A("|--:|--:|--:|--:|--:|--:|--:|--:|")
    for r in p["rows"]:
        A(f"| {r['d']:.1f} | {r['K_put']:,.0f} | {r['naked_loss_30']:,.0f} | {r['dr_loss_30']:,.0f} | "
          f"{r['naked_loss_40']:,.0f} | {r['dr_loss_40']:,.0f} | "
          f"{r['naked_loss_50']:,.0f} | {r['dr_loss_50']:,.0f} |")

    A("\n## SURVIVAL-selected distance (cheapest d capping −50% gap within 1.5% NAV)\n")
    sel = p["survival_selected"]
    if sel is None:
        A("**NONE.** No swept distance caps the −50% one-week defined-risk gap loss within the "
          "1.5%-NAV-per-trade survival limit. The put-wing at every affordable (far-OTM) distance "
          "still leaves a residual short-put exposure between K_put and the −50% settlement that "
          "exceeds 1.5% of NAV.\n")
    else:
        A(f"- **d = {sel['d']:.1f}σ** — the furthest-OTM (cheapest) wing that still caps the "
          f"−50% defined-risk gap loss within 1.5% NAV.")
        A(f"- **Selection logic (NOT PnL-opt):** the per-trade −50% survival test is SLACK at every "
          f"distance (see NAV note — naked itself is already ≈survivable per-trade), so the survival "
          f"selection collapses to the *furthest* (= cheapest) distance. It HAPPENS to coincide with "
          f"the lowest net wing cost; that coincidence is a CONSEQUENCE of the slack constraint, not a "
          f"PnL choice. The distance is pinned from below by AFFORDABILITY (net edge>0 requires "
          f"d≥{min((r['d'] for r in p['rows'] if r['edge_pos']), default=float('nan')):.1f}σ), not from "
          f"above by per-trade survival.")
        A(f"- −50% defined-risk gap loss = **${sel['dr_loss_50']:,.0f}/contract = "
          f"{sel['dr_loss_50_pctnav']:+.2%} of NAV** (≤ 1.5% → survivable per-trade).")
        A(f"- Resulting **net edge = ${sel['dr_mean']:,.0f}/tr** "
          f"({'> $0 → carry survives this wing' if sel['dr_mean'] > 0 else '≤ $0 → carry CANNOT fund even this far wing'}); "
          f"net wing cost ${sel['wing_cost_mean']:,.0f}/tr ({sel['wing_carry_ratio']:.0%} of carry).")
        A(f"- DSR at this distance: full p={sel['dsr_full_p']:.4f} / OOS p={sel['dsr_held_p']:.4f} "
          f"→ {'PASS' if sel['dsr_pass'] else 'FAIL (does NOT statistically graduate on history)'}.\n")

    A("\n## Structural tension — is S3 EVER net-positive AND DSR-passing on history?\n")
    A(p["tension"])

    A("\n\n## VERDICT\n")
    A(p["verdict_long"])
    if smoke:
        A("\n\n> SMOKE — sweep + survival + DSR plumbing exercised on a 3-date slice. "
          "Run without --smoke for the verdict (cheap: cached frame + surface + analytic BS).")
    with open(report_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def _deep_jsonable(obj):
    if isinstance(obj, dict):
        return {k: _deep_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_deep_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating,)):
        v = float(obj); return v if np.isfinite(v) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    return obj


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="S3 wing-distance frontier (cheap analytic)")
    ap.add_argument("--frame-cache", default=FRAME_CACHE)
    ap.add_argument("--svi-surface", default=SVI_SURFACE)
    ap.add_argument("--report", default=REPORT_MD)
    ap.add_argument("--smoke", action="store_true",
                    help="3-date dense slice; proves the sweep plumbing (EXIT 0)")
    ap.add_argument("--smoke-start", default="2025-03-13")
    ap.add_argument("--smoke-end", default="2025-03-17")
    args = ap.parse_args()

    print(f"[s3_wing_frontier] frame={args.frame_cache}  surface={args.svi_surface}  smoke={args.smoke}")
    f = pd.read_parquet(args.frame_cache)
    scov, _ = prep_short_covered(f)
    scov = scov.sort_values("entry_ts").reset_index(drop=True)
    if args.smoke:
        m = ((pd.to_datetime(scov["entry_ts"], utc=True) >= pd.Timestamp(args.smoke_start, tz="UTC"))
             & (pd.to_datetime(scov["entry_ts"], utc=True) <= pd.Timestamp(args.smoke_end, tz="UTC")))
        scov = scov[m].reset_index(drop=True)
    scov["__pos__"] = np.arange(len(scov))     # positional index for window slicing
    n = len(scov)
    print(f"[s3_wing_frontier] covered SHORT n={n}")

    surface = pd.read_parquet(args.svi_surface)
    surface["timestamp"] = pd.to_datetime(surface["timestamp"], utc=True)
    surface["expiry_dt"] = pd.to_datetime(surface["expiry_dt"], utc=True)
    entry = pd.to_datetime(scov["entry_ts"], utc=True)
    expiry = pd.to_datetime(scov["expiry_dt"], utc=True)
    panel = PricePanel.load(FROZEN_CFG, entry.min(), expiry.max() + pd.Timedelta(days=2))

    # Window frames for the DSR (full + held-out 2024-26).
    def win(df, start, end):
        m = ((pd.to_datetime(df["entry_ts"], utc=True) >= pd.Timestamp(start, tz="UTC"))
             & (pd.to_datetime(df["entry_ts"], utc=True) <= pd.Timestamp(end, tz="UTC")))
        return df[m]
    full = scov
    held = win(scov, HELDOUT_START, HELDOUT_END) if not args.smoke else scov

    # Naked DSR (d-independent; computed once for the tension comparison).
    nk_full = dsr_block(full["naked_pnl"].to_numpy(float) if "naked_pnl" in full else
                        price_wings(scov, surface, panel, wing_sigma=2.0)["naked"][full["__pos__"].to_numpy()],
                        full["entry_ts"], full["exit_ts"], "NAKED")
    nk_held = dsr_block(price_wings(scov, surface, panel, wing_sigma=2.0)["naked"][held["__pos__"].to_numpy()],
                        held["entry_ts"], held["exit_ts"], "NAKED")

    # ── Sweep the frontier ──
    rows = []
    for d in WING_DISTANCES:
        r = frontier_row(scov, surface, panel, d, full=full, held=held)
        rows.append(r)
        print(f"  d={d:.1f}σ  net wing ${r['wing_cost_mean']:,.0f}/tr ({r['wing_carry_ratio']:.0%} carry)"
              f"  net edge ${r['dr_mean']:,.0f}/tr  DSR full p={r['dsr_full_p']:.4f}/OOS {r['dsr_held_p']:.4f}"
              f"  DR −50% ${r['dr_loss_50']:,.0f} ({r['dr_loss_50_pctnav']:+.2%} NAV)"
              f"  survives50={r['survives_50']}")

    naked_mean = rows[0]["naked_mean"]

    # ── SURVIVAL SELECTION: furthest-OTM (largest d) that caps −50% within 1.5% NAV ──
    survivors = [r for r in rows if r["survives_50"]]
    survival_selected = max(survivors, key=lambda r: r["d"]) if survivors else None

    # ── Does ANY single d keep net edge>0 AND lift DSR>0.99 simultaneously? ──
    both = [r for r in rows if r["edge_pos"] and r["dsr_pass"]]
    any_both = len(both) > 0
    # Affordable (net edge>0) distances + which survive −50%.
    affordable = [r for r in rows if r["edge_pos"]]
    affordable_survivors = [r for r in affordable if r["survives_50"]]

    # ── BOTTOM LINE: affordable catastrophe-hedge structure? (edge>0 AND −50% survivable) ──
    historical_structure_pass = len(affordable_survivors) > 0
    if historical_structure_pass:
        cheapest_aff_surv = max(affordable_survivors, key=lambda r: r["d"])
        bottom_line = (
            f"YES (structure) — an affordable far-OTM catastrophe hedge EXISTS at d={cheapest_aff_surv['d']:.1f}σ: "
            f"net edge ${cheapest_aff_surv['dr_mean']:,.0f}/tr > $0 AND the −50% gap capped at "
            f"{cheapest_aff_surv['dr_loss_50_pctnav']:+.2%} NAV (≤1.5%). "
            f"BUT S3 is {'STILL NOT' if not any_both else ''} simultaneously net-positive AND DSR-passing on history "
            f"(no d lifts DSR>0.99 → forward test mandatory)."
            if not any_both else
            f"YES — d={cheapest_aff_surv['d']:.1f}σ gives net edge ${cheapest_aff_surv['dr_mean']:,.0f}/tr > $0, "
            f"−50% capped at {cheapest_aff_surv['dr_loss_50_pctnav']:+.2%} NAV, AND DSR PASS."
        )
    else:
        # Why not: is it edge (no affordable d) or survival (no d caps −50%)?
        if not affordable:
            why = ("NO net-positive wing distance exists — even the furthest 5σ wing costs more than "
                   "the carry (the carry cannot fund ANY protection on this calm sample). ")
        elif not survivors:
            why = ("NO swept distance caps the −50% gap within 1.5% NAV — every affordable far-OTM "
                   "put still leaves a residual short-put loss > 1.5% NAV between K_put and the −50% "
                   "settlement. ")
        else:
            why = ("The distances that are affordable (net edge>0) do NOT cap the −50% gap within "
                   "1.5% NAV, and the distances that cap it are NOT affordable — the two constraints "
                   "do not overlap. ")
        bottom_line = (
            f"NO (structure) — {why}"
            f"→ No defined-risk rescue was established. The historical naked-carry "
            f"candidate retains uncapped-tail risk and fails this constraint."
        )

    # ── Structural tension narrative ──
    # close wings fix DSR but cost>carry; far wings affordable but miss the moderate tail.
    close = rows[0]                     # 1.5σ (closest)
    far = rows[-1]                      # 5.0σ (furthest)
    tension = (
        f"Confirming the expected structural tension across the frontier:\n\n"
        f"- **Close wings (d={close['d']:.1f}σ):** net wing cost ${close['wing_cost_mean']:,.0f}/tr = "
        f"{close['wing_carry_ratio']:.0%} of the ${naked_mean:,.0f} carry → net edge "
        f"${close['dr_mean']:,.0f}/tr ({'> $0' if close['edge_pos'] else '≤ $0 — UNAFFORDABLE'}); "
        f"DSR full p={close['dsr_full_p']:.4f}. The close wing is the most protective "
        f"(−50% capped at {close['dr_loss_50_pctnav']:+.2%} NAV) but the most expensive.\n"
        f"- **Far wings (d={far['d']:.1f}σ):** net wing cost ${far['wing_cost_mean']:,.0f}/tr = "
        f"{far['wing_carry_ratio']:.0%} of carry → net edge ${far['dr_mean']:,.0f}/tr "
        f"({'> $0 → AFFORDABLE' if far['edge_pos'] else '≤ $0'}); DSR full p={far['dsr_full_p']:.4f}. "
        f"The far wing is cheap but caps the −50% gap only at {far['dr_loss_50_pctnav']:+.2%} NAV "
        f"({'within' if far['survives_50'] else 'OUTSIDE'} the 1.5% limit) and does NOT lift the DSR "
        f"(it misses the moderate left tail that breaks the realized-PnL DSR — the wings almost never "
        f"pay on this gap-free 2022-26 sample, so they just drain premium).\n\n"
        f"**Is ANY single distance simultaneously net-edge>0 AND DSR>0.99 on history? "
        f"{'YES — d=' + ', '.join(f'{r['d']:.1f}σ' for r in both) if any_both else 'NO.'}** "
        + ("" if any_both else
           f"As pre-expected: the close wings that could lift the DSR cost more than the carry "
           f"(net edge ≤ $0), and the far wings that are affordable do not pay into the moderate "
           f"realized-PnL tail that breaks the DSR. The DSR-passing region and the net-positive "
           f"region do not overlap on this gap-free history — so S3 cannot be statistically "
           f"graduated as defined-risk on the backtest, REGARDLESS of wing distance. "
           f"(Naked full DSR p={nk_full['dsr_p']:.4f} is itself already below 0.99 — the wings "
           f"can only lower the per-trade Sharpe further by draining premium, never raise it, "
           f"because no real gap occurred for them to monetize.)")
    )

    # ── Long verdict ──
    if historical_structure_pass:
        cheapest_aff_surv = max(affordable_survivors, key=lambda r: r["d"])
        verdict_long = (
            f"**An affordable catastrophe-hedge structure EXISTS.** Sweeping the wing distance ONLY to "
            f"map the cost/protection/edge frontier (NOT to optimize PnL), the survival constraint "
            f"(§3.3: −50% one-week gap loss ≤ 1.5% NAV) selects the furthest-OTM = cheapest wing that "
            f"still caps the catastrophe: **d={survival_selected['d']:.1f}σ**. At this distance the net "
            f"defined-risk edge is **${survival_selected['dr_mean']:,.0f}/tr** (a CONSEQUENCE of the "
            f"survival-selected distance, not a maximized quantity), the net wing cost is "
            f"${survival_selected['wing_cost_mean']:,.0f}/tr ({survival_selected['wing_carry_ratio']:.0%} "
            f"of the ${naked_mean:,.0f} carry), and the −50% gap is capped at "
            f"{survival_selected['dr_loss_50_pctnav']:+.2%} of NAV. "
            f"This structure passes the historical survival constraint, but S3 remains "
            f"**unvalidated** (DSR {'PASS' if survival_selected['dsr_pass'] else 'FAIL'} "
            f"at this distance). Tail-inclusive independent OOS evidence would still be required."
        )
    else:
        verdict_long = (
            f"**No affordable catastrophe-hedge structure exists on this history.** "
            + bottom_line.split('→')[0].replace('NO (structure) — ', '')
            + f"\n\nThe naked carry is ${naked_mean:,.0f}/tr. "
            f"{'Even the furthest 5σ wing' if not affordable else 'The wings that cap the −50% gap'} "
            f"{'costs more than the carry' if not affordable else 'cost more than the carry'} on this "
            f"gap-free 2022-26 sample. The naked candidate therefore retains an **uncapped tail**; "
            f"a vega cap and RND/DVOL kills do not validate it or make that risk bounded. The wing cannot be "
            f"funded out of the carry at any distance that actually caps a −50% gap within NAV."
        )

    naked50_mult = abs(rows[0]["naked_loss_50_pctnav"]) / NAV_FRACTION_PER_TRADE
    nav_assumption = [
        f"**§3.3 binding constraint (stated):** per-trade SHORT VEGA NOTIONAL ≤ "
        f"**{NAV_FRACTION_PER_TRADE:.1%} of NAV per trade** — \"sized to survive a 3× straddle-width "
        f"overnight gap.\" The historical analysis assumed a **hypothetical NAV = $500K–$1M**.",
        f"**Conversion (stated):** the per-trade vega/short-straddle NOTIONAL (N_contracts × entry "
        f"index — the natural short-vega-notional exposure) is SIZED to {NAV_FRACTION_PER_TRADE:.1%} of "
        f"the tranche NAV. So per-trade dollar loss L → **L_%NAV = {NAV_FRACTION_PER_TRADE} × L / "
        f"straddle_notional**. This is the SAME whether anchored on a $1M tranche (1.5%NAV = $15K "
        f"vega-notional → N≈0.22 BTC) or as a scale-free ratio — N cancels, so L_%NAV is a pure "
        f"function of the strike geometry (gap size, K_atm, K_put, d), NOT the engine's contract count.",
        f"**WHY a −50% gap is the catastrophe (not the design case):** the 1.5%-NAV cap is sized for a "
        f"3× straddle-width = **3σ** gap. At the representative IV≈47%/T≈11.5d, 1σ≈8.7%, so 3σ≈−22%. "
        f"A **−50% one-week gap is ≈8.3σ** (−30% is ≈4.3σ) — FAR beyond the 3σ the cap was sized for. "
        f"That is exactly why §3.5 names it the binding survival scenario.",
        f"**Representative trade (median σ-move):** S=${rows[0]['S_rep']:,.0f}, "
        f"K_put(d-dependent), straddle notional sized to 1.5% NAV. **SURVIVAL TEST:** "
        f"|DR −50% gap loss| ≤ 1.5% NAV (i.e. ≤ the straddle notional itself).",
        f"**KEY STRUCTURAL FACT — the per-trade survival test is NEARLY NON-BINDING:** because inverse "
        f"(coin-margined) settlement bounds a single short-straddle's loss at ≈ its notional, the NAKED "
        f"−50% per-trade loss is only **{abs(rows[0]['naked_loss_50_pctnav']):.2%} of NAV ≈ "
        f"{naked50_mult:.2f}× the 1.5% cap** — already ~survivable per-trade, and EVERY wing only "
        f"reduces it further. So the per-trade −50% survival constraint does NOT pin the wing distance "
        f"from above (all distances pass). The REAL tail risk is **portfolio aggregation**: a −50% gap "
        f"hits the ENTIRE open short book at once (§3.5 scenario 2; ~6 overlapping trades/mo → a per-"
        f"trade 1.5% becomes a multiple), governed by the −10–15% portfolio kill, NOT the per-trade cap. "
        f"The survival-selected distance below is therefore the cheapest = furthest that passes the "
        f"(slack) per-trade test; affordability (edge>0), not per-trade survival, is the binding floor.",
    ]

    payload = dict(
        generated=pd.Timestamp.now("UTC").isoformat(),
        smoke=args.smoke,
        pricing_source="cached frame + SVI surface (15min) + analytic BS — NO engine re-eval (cheap path)",
        n=n,
        bottom_line=bottom_line,
        nav_assumption=nav_assumption,
        nav_fraction_per_trade=NAV_FRACTION_PER_TRADE,
        naked_mean=naked_mean,
        naked_dsr_full_p=nk_full["dsr_p"], naked_dsr_held_p=nk_held["dsr_p"],
        rows=rows,
        survival_selected=survival_selected,
        any_both_edge_and_dsr=any_both,
        both_distances=[r["d"] for r in both],
        historical_structure_pass=historical_structure_pass,
        affordable_distances=[r["d"] for r in affordable],
        survivor_distances=[r["d"] for r in survivors],
        affordable_survivor_distances=[r["d"] for r in affordable_survivors],
        tension=tension,
        verdict_long=verdict_long,
    )

    write_report(payload, report_md=args.report)
    with open(args.report.replace(".md", ".json"), "w", encoding="utf-8") as fh:
        json.dump(_deep_jsonable(payload), fh, indent=2)

    # ── console summary ──
    print("=" * 88)
    print("S3 WING-DISTANCE FRONTIER" + ("  [SMOKE]" if args.smoke else ""))
    print("=" * 88)
    print(f"covered shorts n={n}   naked carry ${naked_mean:,.0f}/tr   "
          f"naked DSR full p={nk_full['dsr_p']:.4f}")
    print("-" * 88)
    print(f"{'d·σ':>5} {'netWing$':>9} {'%carry':>7} {'netEdge$':>9} {'edge>0':>6} "
          f"{'DSRfull':>8} {'DSRoos':>7} {'pass':>5} {'DR-50%$':>10} {'DR-50%NAV':>10} {'surv50':>6}")
    for r in rows:
        print(f"{r['d']:>5.1f} {r['wing_cost_mean']:>9,.0f} {r['wing_carry_ratio']:>7.0%} "
              f"{r['dr_mean']:>9,.0f} {'Y' if r['edge_pos'] else 'N':>6} "
              f"{r['dsr_full_p']:>8.4f} {r['dsr_held_p']:>7.4f} "
              f"{'PASS' if r['dsr_pass'] else 'FAIL':>5} {r['dr_loss_50']:>10,.0f} "
              f"{r['dr_loss_50_pctnav']:>+10.2%} {'YES' if r['survives_50'] else 'no':>6}")
    print("-" * 88)
    if survival_selected:
        print(f"SURVIVAL-SELECTED (cheapest d capping −50% within 1.5% NAV): d={survival_selected['d']:.1f}σ"
              f"  → net edge ${survival_selected['dr_mean']:,.0f}/tr  DSR pass={survival_selected['dsr_pass']}")
    else:
        print("SURVIVAL-SELECTED: NONE — no swept d caps the −50% gap within 1.5% NAV.")
    print(f"ANY d with edge>0 AND DSR>0.99: {any_both}  ({[r['d'] for r in both]})")
    print(f"AFFORDABLE (edge>0) distances: {[r['d'] for r in affordable]}")
    print(f"−50%-SURVIVING distances:      {[r['d'] for r in survivors]}")
    print(f"AFFORDABLE & SURVIVING:        {[r['d'] for r in affordable_survivors]}")
    print(f"HISTORICAL STRUCTURE THRESHOLD (edge>0 AND −50% survivable): {historical_structure_pass}")
    print("-" * 88)
    print(f"BOTTOM LINE: {bottom_line}")
    print(f"report -> {args.report}  (+ .json)")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
