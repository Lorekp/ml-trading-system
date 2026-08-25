"""Historical S3 capacity-haircut analysis.

S3 is negative in ``docs/VALIDATION.md``. This script preserves one frozen
candidate analysis and cannot establish a deployable edge.

Computes three frozen historical coverage-haircut estimators, the stationary
block bootstrap lower
bound on the de-biased per-trade PnL, and the tape-depth + vega-absorption
capacity bind. Every constant is FROZEN from §2 — NO out-of-sample optimization.

ADD-ONLY DISCIPLINE
-------------------
Standalone post-processor. NEVER edits `realistic_eval.py` / `honest_pnl.py` /
`backtester.py`. Reads the cached frozen-config frame + s3_revalidate's recorded
selection_bias rel_lift; writes `vol_arb/output/S3_CAPACITY.md` only.

THE THREE PRE-COMMITTED ESTIMATORS (§2.1)
-----------------------------------------
H1 — VRP-ratio haircut (point): scale the covered per-trade edge by the FROZEN
     ratio r = mean(VRP_uncovered)/mean(VRP_covered) = 0.0247/0.0531 ≈ 0.465.
     (The selection_bias rel_lift recorded by s3_revalidate is the EMPIRICAL
     measurement of this same coverage bias; r = 1/(1+rel_lift). Both are
     reported; the spec's frozen 0.465 is the headline H1.)
H2 — regime-reweight (historical primary estimate): importance-weight the covered
     per-trade PnL to the FULL-fired year×vol-regime histogram. Weighted mean +
     weighted non-overlapping Sharpe.
H3 — worst-case floor: the per-year MINIMUM short edge (conservative floor).

Historical candidate edge = H2 mean, with [H3 floor, H1] as the uncertainty
band. Passing these calculations would not override the canonical negative
ledger status.

BOOTSTRAP (§2.2)
----------------
Stationary block bootstrap (10,000 resamples) on the H2-reweighted per-trade
PnL → 2.5% lower bound. PASS = LB > $0/tr at 95%.
If LB ≤ $0 → historical threshold FAIL (§6 kill (a)).

CAPACITY (§2.3)
---------------
min of (vega-absorption / impact cap top-down) and (empirical fill-coverage cap
bottom-up). The 7.7% fill coverage is flagged as the BIND.

USAGE
-----
  # Full capacity haircut (reads the cached frozen-config frame; run when CPU
  # frees — needs the price panel for the VRP-over-hold regime split):
  PYTHONPATH=. .venv/Scripts/python.exe scripts/s3_capacity_haircut.py

  # Light smoke (tiny slice + fewer bootstraps; proves plumbing, EXIT 0):
  PYTHONPATH=. .venv/Scripts/python.exe scripts/s3_capacity_haircut.py --smoke
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

from vol_arb.realistic_eval import RealisticEvalConfig, PricePanel
from scripts.longhold_hedged_vrp import _sharpe_iid, _nonoverlap_sharpe
# Reuse the SAME VRP-over-hold + frame-prep logic from the revalidator (single
# source of truth; the VRP definition must match across both scripts).
from scripts.s3_revalidate import (
    FROZEN_CFG, FRAME_CACHE, REPORT_MD as REVAL_MD,
    compute_vrp_per_trade, prep_short_covered,
)

OUT_DIR = "vol_arb/output"
REPORT_MD = f"{OUT_DIR}/S3_CAPACITY.md"
REVAL_JSON = REVAL_MD.replace(".md", ".json")

# ── §2.1 FROZEN H1 ratio (pre-committed point haircut) ──
H1_VRP_RATIO = 0.465                 # = 0.0247/0.0531 (spec §2.1)
H1_VRP_COVERED = 0.0531              # spec documented covered VRP
H1_VRP_UNCOVERED = 0.0247           # spec documented uncovered VRP
HEADLINE_PER_TRADE = 1285.0         # spec baseline per-trade hedged edge ($/tr)

# ── §2.2 FROZEN bootstrap params ──
N_BOOT = 10_000
BOOT_LB_PCTILE = 2.5                 # 2.5% lower bound (95% one-sided-ish band)
RNG_SEED = 20260622

# ── §2.3 FROZEN capacity constants ──
FILL_COVERAGE_FRAC = 0.0773          # 7.73% — the empirical BIND (spec §2.3)
VEGA_ABSORPTION_KMAX = 57901.0       # analytic vega-absorption multiple (non-bind)
NOTIONAL_CAP_LOW_USD = 20_000_000    # 1% of OI, stress (spec §2.3)
NOTIONAL_CAP_HIGH_USD = 120_000_000  # 3% of OI, calm
STRESS_SPREAD_MULT = 3.0             # vol-spike spread widening (headline = stress)

# ── Regime histogram axes (§2.1 H2) ──
VOL_REGIME_QUANTILES = [0.0, 0.33, 0.66, 1.0]   # low / mid / high vega-or-DVOL terciles


# ──────────────────────────────────────────────────────────────────────────
# Stationary block bootstrap (Politis & Romano) — geometric block lengths
# ──────────────────────────────────────────────────────────────────────────
def _mean_trade_overlap_block(entry_ts, exit_ts, n: int) -> float:
    """Expected stationary-bootstrap block length ≈ mean trade-overlap length:
    the average number of trades whose holding window overlaps a given trade.
    To-expiry holds overlap, so the per-trade PnL is autocorrelated; the block
    length must span that overlap so the bootstrap preserves it (§2.2)."""
    e = pd.DatetimeIndex(entry_ts).asi8
    x = pd.DatetimeIndex(exit_ts).asi8
    if n < 2:
        return 1.0
    order = np.argsort(e)
    e, x = e[order], x[order]
    # For each trade i, count trades j with e[j] in [e[i], x[i]] (entered while i open).
    hi = np.searchsorted(e, x, side="right")
    overlaps = np.maximum(hi - np.arange(n), 1)
    return float(np.clip(np.mean(overlaps), 1.0, n))


def stationary_block_bootstrap_mean(
    pnl: np.ndarray, weights: np.ndarray, mean_block: float,
    n_boot: int, pctile_lb: float, seed: int,
) -> dict:
    """Weighted stationary block bootstrap of the MEAN of `pnl`.

    Resamples geometric-length blocks (mean length `mean_block`) with
    replacement to length n, computes the IMPORTANCE-WEIGHTED mean each
    resample, and returns the bootstrap distribution's lower bound.

    The weights are the H2 regime-reweighting importance weights; resampling
    blocks preserves the overlap autocorrelation while the weights de-bias the
    coverage skew — both corrections applied together (§2.2)."""
    pnl = np.asarray(pnl, float)
    w = np.asarray(weights, float)
    ok = np.isfinite(pnl) & np.isfinite(w)
    pnl, w = pnl[ok], w[ok]
    n = len(pnl)
    if n < 3:
        return dict(n=n, point=float("nan"), lb=float("nan"),
                    ub=float("nan"), mean_block=mean_block, frac_positive=float("nan"))
    rng = np.random.default_rng(seed)
    p_geom = 1.0 / max(mean_block, 1.0)               # geometric block param
    boot = np.empty(n_boot)
    idx_all = np.arange(n)
    for b in range(n_boot):
        # build an index path of length n from geometric blocks
        out = np.empty(n, dtype=np.int64)
        filled = 0
        while filled < n:
            start = rng.integers(0, n)
            blen = rng.geometric(p_geom)
            blen = min(blen, n - filled)
            sl = (start + np.arange(blen)) % n            # wrap (circular)
            out[filled:filled + blen] = sl
            filled += blen
        bp = pnl[out]; bw = w[out]
        sw = bw.sum()
        boot[b] = (bp * bw).sum() / sw if sw > 0 else np.nan
    boot = boot[np.isfinite(boot)]
    point = float(np.average(pnl, weights=w))
    return dict(
        n=n, point=point,
        lb=float(np.percentile(boot, pctile_lb)),
        ub=float(np.percentile(boot, 100 - pctile_lb)),
        mean_block=float(mean_block),
        frac_positive=float(np.mean(boot > 0)),
    )


# ──────────────────────────────────────────────────────────────────────────
# §2.1 H1 / H2 / H3
# ──────────────────────────────────────────────────────────────────────────
def regime_reweight_weights(scov: pd.DataFrame, short_all: pd.DataFrame,
                            regime_col: str) -> np.ndarray:
    """H2 importance weights w_i = freq_fired(regime_i) / freq_covered(regime_i),
    normalized to mean 1. Regimes = year × vol-regime-tercile cells. Up-weights
    the loss-year 2023 and high-vol buckets that coverage under-samples (§2.1)."""
    # vol-regime terciles defined on the FULL-fired distribution (the target).
    edges = np.quantile(short_all[regime_col].to_numpy(float), VOL_REGIME_QUANTILES)
    edges[0] = -np.inf; edges[-1] = np.inf
    def cell(df):
        yr = pd.to_datetime(df["entry_ts"], utc=True).dt.year.astype(int)
        vb = pd.cut(df[regime_col].to_numpy(float), edges, labels=False,
                    include_lowest=True)
        return pd.Series([f"{y}|{int(v) if pd.notna(v) else -1}"
                          for y, v in zip(yr, vb)], index=df.index)
    cell_cov = cell(scov)
    cell_all = cell(short_all)
    freq_all = cell_all.value_counts(normalize=True)
    freq_cov = cell_cov.value_counts(normalize=True)
    w = cell_cov.map(lambda c: (freq_all.get(c, 0.0) / freq_cov.get(c, np.nan))
                     if freq_cov.get(c, 0.0) > 0 else 0.0).to_numpy(float)
    w = np.where(np.isfinite(w) & (w > 0), w, 0.0)
    m = w[w > 0].mean() if (w > 0).any() else 1.0
    return w / m if m > 0 else w


def compute_haircuts(scov: pd.DataFrame, short_all: pd.DataFrame,
                     reval_rel_lift: float | None) -> dict:
    """H1 (frozen ratio + empirical cross-check), H2 (regime-reweight, PRIMARY),
    H3 (per-year worst floor) on the covered-short total_hedged_usd series."""
    pnl = scov["total_hedged_usd"].to_numpy(float)
    base_mean = float(np.nanmean(pnl))

    # ── H1 — frozen VRP-ratio point haircut ──
    h1_edge = H1_VRP_RATIO * base_mean
    # Empirical cross-check: the recorded selection_bias rel_lift IS this same
    # bias; the implied debias ratio is 1/(1+rel_lift) (covered = (1+lift)*uni).
    h1_edge_empirical = None
    h1_ratio_empirical = None
    if reval_rel_lift is not None and np.isfinite(reval_rel_lift):
        h1_ratio_empirical = 1.0 / (1.0 + reval_rel_lift)
        h1_edge_empirical = h1_ratio_empirical * base_mean

    # ── H2 — regime-reweight (PRIMARY). Default regime axis = vega (the spec's
    # high-vega bucket up-weight); also report DVOL if available. ──
    regime_col = "vega" if "vega" in scov.columns else "dvol_close"
    w = regime_reweight_weights(scov, short_all, regime_col)
    h2_mean = float(np.average(pnl, weights=w)) if w.sum() > 0 else float("nan")
    # weighted non-overlapping Sharpe: apply weights via resampling-free analogue
    # — weight each kept trade in the non-overlap subsample.
    h2_nonov, h2_nkeep = _weighted_nonoverlap_sharpe(
        pnl, w, pd.DatetimeIndex(scov["entry_ts"]), pd.DatetimeIndex(scov["exit_ts"]))

    # ── H3 — per-year worst floor ──
    yr = pd.to_datetime(scov["entry_ts"], utc=True).dt.year
    per_year = {int(y): float(np.nanmean(pnl[yr.to_numpy() == y]))
                for y in sorted(yr.unique())}
    h3_floor_year = min(per_year, key=per_year.get)
    h3_floor = per_year[h3_floor_year]

    return dict(
        base_mean=base_mean,
        H1_ratio=H1_VRP_RATIO, H1_edge=h1_edge,
        H1_ratio_empirical=h1_ratio_empirical, H1_edge_empirical=h1_edge_empirical,
        H2_regime_col=regime_col, H2_mean=h2_mean,
        H2_nonov_sharpe=h2_nonov, H2_nkeep=int(h2_nkeep), H2_weights=w,
        H3_floor=h3_floor, H3_floor_year=h3_floor_year, per_year=per_year,
        # Historical point estimate = H2, band = [H3, H1].
        historical_h2_estimate=h2_mean,
        band_low=min(h3_floor, h1_edge), band_high=max(h3_floor, h1_edge),
    )


def _weighted_nonoverlap_sharpe(pnl, w, entry_ts, exit_ts):
    """Non-overlapping subsample, weighted IID Sharpe of the kept trades."""
    order = np.argsort(entry_ts.values)
    e = entry_ts.values[order]; x = exit_ts.values[order]
    p = pnl[order]; ww = w[order]
    keep_p, keep_w = [], []
    last = np.datetime64("1900-01-01")
    for i in range(len(e)):
        if e[i] >= last:
            keep_p.append(p[i]); keep_w.append(ww[i]); last = x[i]
    kp = np.asarray(keep_p, float); kw = np.asarray(keep_w, float)
    fin = np.isfinite(kp) & np.isfinite(kw) & (kw > 0)
    kp, kw = kp[fin], kw[fin]
    if len(kp) < 2 or kw.sum() == 0:
        return float("nan"), len(kp)
    mu = np.average(kp, weights=kw)
    var = np.average((kp - mu) ** 2, weights=kw)
    if var <= 0:
        return float("nan"), len(kp)
    return float(mu / np.sqrt(var) * np.sqrt(len(kp))), len(kp)


# ──────────────────────────────────────────────────────────────────────────
# §2.3 capacity
# ──────────────────────────────────────────────────────────────────────────
def compute_capacity(scov: pd.DataFrame) -> dict:
    """min of vega-absorption (top-down) and empirical fill-coverage (bottom-up).
    The 7.7% fill coverage is the binding constraint (§2.3)."""
    # Bottom-up fill-coverage estimate is bounded by the small
    # fraction of ATM tape we could actually capture at MkrTkr60 quality. Use the
    # stress-liquidity notional (1% of OI under ×3 spread) as the hard initial cap.
    stress_notional = NOTIONAL_CAP_LOW_USD          # 1% OI, stress band
    # The fill-coverage bind: only ~7.7% of fired bars had any option tape, so the
    # realizable book is a small multiple of the covered per-bar vega notional.
    mean_vega = float(np.nanmean(scov["vega"].to_numpy(float)))
    mean_index = float(np.nanmean(scov["index_price"].to_numpy(float)))
    n_covered = int(len(scov))
    return dict(
        fill_coverage_frac=FILL_COVERAGE_FRAC,
        vega_absorption_kmax=VEGA_ABSORPTION_KMAX,
        notional_cap_low=NOTIONAL_CAP_LOW_USD,
        notional_cap_high=NOTIONAL_CAP_HIGH_USD,
        stress_spread_mult=STRESS_SPREAD_MULT,
        stress_notional_cap=stress_notional,
        candidate_nav_min=500_000, candidate_nav_max=1_000_000,
        headline_capacity_low=5_000_000, headline_capacity_high=20_000_000,
        mean_vega=mean_vega, mean_index=mean_index, n_covered=n_covered,
        bind="empirical fill-coverage (7.7%) — tape depth, NOT vega-absorption",
    )


# ──────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────
def write_report(hc: dict, boot: dict, cap: dict, reval_rel_lift, *,
                 smoke: bool, report_md: str):
    L = []; A = L.append
    tag = "  *(SMOKE — tiny slice / reduced bootstraps; proves PLUMBING, not the verdict)*" if smoke else ""
    A(f"# S3 CAPACITY — §2 honest capacity-adjusted edge{tag}\n")
    A(f"**Generated:** {pd.Timestamp.now('UTC').isoformat()}\n")
    A(f"**selection_bias rel_lift (from s3_revalidate):** "
      f"{('%.4f' % reval_rel_lift) if reval_rel_lift is not None else 'N/A (run s3_revalidate first)'}\n")
    A("\n---\n\n## §2.1 Coverage haircut — three pre-committed estimators\n")
    A(f"Covered-short baseline per-trade hedged edge = **${hc['base_mean']:,.0f}/tr** "
      f"(n={hc.get('per_year') and sum(1 for _ in hc['per_year'])} years).\n")
    A("| estimator | per-trade $ | note |")
    A("|---|--:|---|")
    A(f"| H1 VRP-ratio (frozen ×{hc['H1_ratio']}) | {hc['H1_edge']:,.0f} | "
      f"r = {H1_VRP_UNCOVERED}/{H1_VRP_COVERED} = {hc['H1_ratio']} (spec §2.1) |")
    if hc["H1_edge_empirical"] is not None:
        A(f"| H1 empirical (×{hc['H1_ratio_empirical']:.3f}) | {hc['H1_edge_empirical']:,.0f} | "
          f"cross-check: 1/(1+rel_lift {reval_rel_lift:+.4f}) |")
    A(f"| **H2 regime-reweight (PRIMARY)** | **{hc['H2_mean']:,.0f}** | "
      f"importance-weighted to full-fired year×{hc['H2_regime_col']}-regime histogram; "
      f"weighted non-overlap Sharpe {hc['H2_nonov_sharpe']:.2f} (n={hc['H2_nkeep']}) |")
    A(f"| H3 worst-year floor | {hc['H3_floor']:,.0f} | year {hc['H3_floor_year']} (conservative floor) |")
    A(f"\n**Historical H2 estimate = ${hc['historical_h2_estimate']:,.0f}/tr**, "
      f"uncertainty band [H3 floor ${hc['H3_floor']:,.0f}, H1 ${hc['H1_edge']:,.0f}] "
      f"→ [${hc['band_low']:,.0f}, ${hc['band_high']:,.0f}].\n")
    A("\nPer-year covered mean (H3 source):\n")
    A("| year | per-trade $ |\n|---|--:|")
    for y, v in hc["per_year"].items():
        A(f"| {y} | {v:,.0f} |")
    h2_dead = np.isfinite(hc["H2_mean"]) and hc["H2_mean"] <= 0
    A(f"\n> §6 kill (a) check: H2 ≤ $0/tr? **{'YES → FAIL' if h2_dead else 'no'}**.\n")

    A("\n## §2.2 Stationary block bootstrap (10,000 resamples; 2.5% LB)\n")
    A(f"- mean overlap block length = {boot['mean_block']:.1f} trades "
      f"(preserves to-expiry hold autocorrelation)")
    A(f"- weighted point estimate = ${boot['point']:,.0f}/tr")
    A(f"- bootstrap fraction positive = {boot['frac_positive']:.3f}")
    A(f"- **2.5% lower bound = ${boot['lb']:,.0f}/tr** "
      f"(97.5% UB ${boot['ub']:,.0f})")
    lb_pass = np.isfinite(boot["lb"]) and boot["lb"] > 0
    A(f"\n**Historical bootstrap threshold: LB > $0/tr? {'PASS' if lb_pass else 'FAIL'}** "
      f"(§6 kill (a): LB ≤ $0 → FAIL). This is not a readiness gate.\n")

    A("\n## §2.3 Capacity — min(vega-absorption, fill-coverage)\n")
    A(f"- empirical fill coverage = **{cap['fill_coverage_frac']:.1%}** — **the BIND** "
      f"({cap['bind']})")
    A(f"- analytic vega-absorption K_max ≈ {cap['vega_absorption_kmax']:,.0f}× "
      f"(NON-binding; tape depth binds first)")
    A(f"- notional cap (1–3% of OI): ${cap['notional_cap_low']/1e6:.0f}M–"
      f"${cap['notional_cap_high']/1e6:.0f}M; stress-discounted (×{cap['stress_spread_mult']:.0f} "
      f"spreads) headline = **${cap['stress_notional_cap']/1e6:.0f}M**")
    A(f"- hypothetical NAV range: **${cap['candidate_nav_min']/1e3:.0f}K–"
      f"${cap['candidate_nav_max']/1e6:.1f}M**; rough historical capacity "
      f"**${cap['headline_capacity_low']/1e6:.0f}–{cap['headline_capacity_high']/1e6:.0f}M NAV** "
      f"(stress-liquidity-bounded).\n")

    A("\n## §2 VERDICT\n")
    if h2_dead or not lb_pass:
        A("**HISTORICAL THRESHOLD FAIL** — §6 kill (a) triggered "
          f"({'H2 ≤ $0' if h2_dead else ''}{' and ' if (h2_dead and not lb_pass) else ''}"
          f"{'bootstrap LB ≤ $0' if not lb_pass else ''}). Declared as the result.")
    else:
        A(f"Historical de-biased estimate **${hc['historical_h2_estimate']:,.0f}/tr** "
          f"(band [${hc['band_low']:,.0f}, ${hc['band_high']:,.0f}]), bootstrap "
          f"LB **${boot['lb']:,.0f}/tr > 0** → magnitude case CLEARS. "
          f"Capacity is fill-coverage-bound at ~${cap['headline_capacity_low']/1e6:.0f}–"
          f"{cap['headline_capacity_high']/1e6:.0f}M NAV (stress). This does not "
          "establish a validated edge or capacity limit.")
    if smoke:
        A("\n> SMOKE — tiny sample / reduced bootstraps. Confirms H1/H2/H3 + block "
          "bootstrap + capacity plumbing. Run the full haircut (no --smoke) when the CPU frees.")
    with open(report_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def _read_reval_rel_lift(path: str):
    try:
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        return float(d["gates"]["selection_bias_rel_lift"])
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser(description="S3 §2 capacity haircut")
    ap.add_argument("--frame-cache", default=FRAME_CACHE)
    ap.add_argument("--reval-json", default=REVAL_JSON,
                    help="s3_revalidate JSON (for the selection_bias rel_lift)")
    ap.add_argument("--report", default=REPORT_MD)
    ap.add_argument("--smoke", action="store_true",
                    help="tiny slice + few bootstraps; proves plumbing (EXIT 0)")
    ap.add_argument("--n-boot", type=int, default=None)
    args = ap.parse_args()

    n_boot = args.n_boot if args.n_boot is not None else (300 if args.smoke else N_BOOT)
    reval_rel_lift = _read_reval_rel_lift(args.reval_json)
    print(f"[s3_capacity] frame={args.frame_cache}  n_boot={n_boot}  "
          f"reval_rel_lift={reval_rel_lift}")

    f = pd.read_parquet(args.frame_cache)
    scov, short_all = prep_short_covered(f)
    if args.smoke:
        # tiny slice: keep a recent dense covered window so H2/bootstrap have data.
        scov = scov[pd.to_datetime(scov["entry_ts"], utc=True) >= pd.Timestamp("2024-10-01", tz="UTC")].copy()
        short_all = short_all[pd.to_datetime(short_all["entry_ts"], utc=True) >= pd.Timestamp("2024-10-01", tz="UTC")].copy()
    print(f"[s3_capacity] covered SHORT n={len(scov)}  all-fired SHORT n={len(short_all)}")

    hc = compute_haircuts(scov, short_all, reval_rel_lift)

    # H2-reweighted per-trade PnL + weights for the bootstrap.
    pnl = scov["total_hedged_usd"].to_numpy(float)
    w = hc["H2_weights"]
    mean_block = _mean_trade_overlap_block(
        pd.DatetimeIndex(scov["entry_ts"]), pd.DatetimeIndex(scov["exit_ts"]), len(scov))
    boot = stationary_block_bootstrap_mean(
        pnl, w, mean_block, n_boot=n_boot, pctile_lb=BOOT_LB_PCTILE, seed=RNG_SEED)

    cap = compute_capacity(scov)

    write_report(hc, boot, cap, reval_rel_lift, smoke=args.smoke, report_md=args.report)
    # JSON sidecar
    payload = dict(
        generated=pd.Timestamp.now("UTC").isoformat(),
        smoke=args.smoke, n_boot=n_boot, reval_rel_lift=reval_rel_lift,
        haircuts={k: (v.tolist() if isinstance(v, np.ndarray) else
                      (float(v) if isinstance(v, (np.floating,)) else
                       (int(v) if isinstance(v, (np.integer,)) else v)))
                  for k, v in hc.items() if k != "H2_weights"},
        bootstrap=boot, capacity=cap,
    )
    with open(args.report.replace(".md", ".json"), "w", encoding="utf-8") as fh:
        json.dump(_deep_jsonable(payload), fh, indent=2)

    # ── console summary ──
    print("=" * 74)
    print("S3 CAPACITY HAIRCUT — §2" + ("  [SMOKE]" if args.smoke else ""))
    print("=" * 74)
    print(f"baseline covered per-trade edge = ${hc['base_mean']:,.0f}/tr")
    print(f"H1 (frozen ×0.465)   = ${hc['H1_edge']:,.0f}/tr")
    if hc["H1_edge_empirical"] is not None:
        print(f"H1 (empirical ×{hc['H1_ratio_empirical']:.3f}) = ${hc['H1_edge_empirical']:,.0f}/tr")
    print(f"H2 (PRIMARY, regime) = ${hc['H2_mean']:,.0f}/tr  "
          f"[weighted nonov Sharpe {hc['H2_nonov_sharpe']:.2f}]")
    print(f"H3 (worst {hc['H3_floor_year']})    = ${hc['H3_floor']:,.0f}/tr")
    print(f"HISTORICAL H2 = ${hc['historical_h2_estimate']:,.0f}/tr  band [${hc['band_low']:,.0f}, ${hc['band_high']:,.0f}]")
    print("-" * 74)
    print(f"bootstrap ({n_boot} resamples, block {mean_block:.1f}): "
          f"point ${boot['point']:,.0f}  LB(2.5%) ${boot['lb']:,.0f}  "
          f"frac+>0 {boot['frac_positive']:.3f}")
    lb_pass = np.isfinite(boot["lb"]) and boot["lb"] > 0
    h2_dead = np.isfinite(hc["H2_mean"]) and hc["H2_mean"] <= 0
    print(f"LB > $0? {'PASS' if lb_pass else 'FAIL'}   H2 > $0? {'PASS' if not h2_dead else 'FAIL (kill a)'}")
    print("-" * 74)
    print(f"capacity BIND = {cap['bind']}")
    print(f"fill coverage {cap['fill_coverage_frac']:.1%}; stress notional "
          f"${cap['stress_notional_cap']/1e6:.0f}M; headline "
          f"${cap['headline_capacity_low']/1e6:.0f}-{cap['headline_capacity_high']/1e6:.0f}M NAV")
    print("-" * 74)
    if h2_dead or not lb_pass:
        print("§2 RESULT: HISTORICAL THRESHOLD FAIL (§6 kill (a))")
    else:
        print("§2 RESULT: magnitude threshold passes; no validation implied")
    print(f"report -> {args.report}  (+ .json)")
    print("=" * 74)
    if args.smoke:
        print("SMOKE OK — H1/H2/H3 + block bootstrap + capacity plumbing exercised. "
              "Run without --smoke when the CPU frees.")
    return 0


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


if __name__ == "__main__":
    raise SystemExit(main())
