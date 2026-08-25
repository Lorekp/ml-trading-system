"""Historical S3 frozen revalidation protocol.

S3 is negative in ``docs/VALIDATION.md``. Every parameter, window boundary,
gate threshold, and pass criterion here is frozen for forensic reproduction;
this script cannot promote the strategy. There is no OOS optimization:
the band (0.35), side (SHORT-only), hold (expiry) and all gate floors are inputs
taken from the existing plateau, not outputs of this run.

ADD-ONLY DISCIPLINE
-------------------
This is a standalone post-processor. It NEVER edits `realistic_eval.py`,
`honest_pnl.py` or `backtester.py` (V2 byte-identity preserved). It only calls
their public APIs and writes new outputs under `vol_arb/output/`.

WHAT IT DOES (§1)
-----------------
1.1 Slices the pre-committed windows (TRAIN 2022-2023 ref-only, HELD-OUT OOS
    2024-01..2026-05, FRESH OOS 2026-06+) on the FROZEN config output.
1.2 Runs the 4-fold anchored-expanding walk-forward (frozen band; tests
    STABILITY not selection). PASS = >=4/4 folds positive per-trade.
1.3 Calls the 4 gates with the pre-committed thresholds:
      - gate_no_fake_profit(model_col="total_hedged_usd", corr floor 0.85)
      - pnl_realism_gate(min_corr=0.85, max_total_rel_err=0.30)
      - sizing_sanity_gate(pnl_per_unit = total_hedged_usd/vega, exposure=vega)
      - selection_bias_gate(VRP_covered vs VRP_all_fired) — EXPECTED FAIL; its
        rel_lift is RECORDED (feeds the §2 haircut), NOT treated as a kill.
1.4 Computes the Deflated Sharpe Ratio on the NON-OVERLAPPING short series at
    K=40. PASS = DSR p < 0.01 (t > 3.23).
1.5 Emits the overall PASS (criteria 1-5; selection_bias recorded only).

THE FROZEN CONFIG (§0, pre-registered verbatim)
-----------------------------------------------
    RealisticEvalConfig(
        hold="expiry", real_bs_delta=True, fill_tol_minutes=60.0,
        spread_source="tape", delta_hedge_band=0.35, maker_wait_s=60.0,
    )

This is byte-identical to the config that produced
`vol_arb/output/verify_frame_band35.parquet` (the cached full-config output,
built by `scripts/verify_build_frames.py`). The full run can either re-run the
engine or reuse that cache via --frame-cache (default).

USAGE
-----
  # Full revalidation (CPU-heavy; run when the CPU frees) — reuse the cached
  # frozen-config frame (no engine re-run, instant):
  PYTHONPATH=. .venv/Scripts/python.exe scripts/s3_revalidate.py

  # Full revalidation, fresh engine eval (re-marks from the V2 trade log; slow):
  PYTHONPATH=. .venv/Scripts/python.exe scripts/s3_revalidate.py --reeval

  # Light smoke (3 dates, tiny slice) — proves the gate wiring, EXIT 0:
  PYTHONPATH=. .venv/Scripts/python.exe scripts/s3_revalidate.py --smoke
"""
from __future__ import annotations

import argparse
import json
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm

# Windows consoles default to cp1252 and choke on the report's unicode (→ σ ≥).
# The report file is written utf-8; make stdout utf-8 too so console prints match.
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

# Public APIs — engine + gates (NO edits to these modules).
from vol_arb.realistic_eval import RealisticEvalConfig, PricePanel, _to_i8_ns, evaluate_realistic
from vol_arb.honest_pnl import gate_no_fake_profit
from validation.gates import (
    GateResult, pnl_realism_gate, sizing_sanity_gate, selection_bias_gate,
)
# Reusable overfit-check helpers (the same ones the verifier used).
from scripts.longhold_hedged_vrp import (
    _sharpe_iid, _newey_west_sharpe, _nonoverlap_sharpe, _max_drawdown,
)

OUT_DIR = "vol_arb/output"
FRAME_CACHE = f"{OUT_DIR}/verify_frame_band35.parquet"
TRADE_LOG = f"{OUT_DIR}/BTC_v2_trade_log.parquet"
REPORT_MD = f"{OUT_DIR}/S3_REVALIDATION.md"

# ── §0 FROZEN config (pre-registered verbatim; do NOT change) ──
FROZEN_CFG = RealisticEvalConfig(
    hold="expiry", real_bs_delta=True, fill_tol_minutes=60.0,
    spread_source="tape", delta_hedge_band=0.35, maker_wait_s=60.0,
)

# ── §1.1 FROZEN time-split boundaries (UTC) ──
TRAIN_START, TRAIN_END = "2022-01-01", "2023-12-31"        # reference only
HELDOUT_START, HELDOUT_END = "2024-01-01", "2026-05-31"     # pure confirmation
FRESH_START = "2026-06-01"                                  # cleanest OOS

# ── §1.2 FROZEN walk-forward folds: (train_years, test_year_label, test_pred) ──
# anchored expanding; no re-fit (band frozen). PASS = >=4/4 test windows +ve.
WF_FOLDS = [
    ("2022",        2023, lambda y: y == 2023),
    ("2022-23",     2024, lambda y: y == 2024),
    ("2022-24",     2025, lambda y: y == 2025),
    ("2022-25", "2026H1", lambda y: y == 2026),
]

# ── §1.3 / §1.4 FROZEN gate thresholds ──
CORR_FLOOR = 0.85
MAX_REL_ERR = 0.30
SEL_BIAS_MAX_REL_LIFT = 0.50
DSR_K = 40
DSR_T_HURDLE = 3.23          # K=40 Bonferroni-style hurdle (spec §1.4)
NW_T_HURDLE = 3.23           # §1.5 criterion 1 hurdle
EULER_GAMMA = 0.5772156649

_YEAR_MIN = 365.25 * 24 * 60


# ──────────────────────────────────────────────────────────────────────────
# VRP per trade — the spec's definition (entry IV[dec] − realized vol over hold)
# EXACTLY as in scripts/verify_attack3_selection.py (reproduces +0.0531/+0.0247).
# ──────────────────────────────────────────────────────────────────────────
def realized_vol_over_hold(entry_ts, exit_ts, panel: PricePanel) -> np.ndarray:
    """Annualized realized vol of the real BTC path from entry to expiry per
    trade. Vectorized window bounds, per-trade RMS of 1-min log returns."""
    entry_i8 = _to_i8_ns(pd.DatetimeIndex(pd.to_datetime(entry_ts, utc=True)))
    exit_i8 = _to_i8_ns(pd.DatetimeIndex(pd.to_datetime(exit_ts, utc=True)))
    lo = np.searchsorted(panel.ts_i8, entry_i8, "left")
    hi = np.searchsorted(panel.ts_i8, exit_i8, "right")
    out = np.full(len(entry_i8), np.nan)
    for i in range(len(entry_i8)):
        a, b = lo[i], hi[i]
        if b - a < 3:
            continue
        r = np.diff(np.log(panel.close[a:b]))
        out[i] = np.sqrt(np.mean(r ** 2) * _YEAR_MIN)
    return out


def compute_vrp_per_trade(frame: pd.DataFrame, panel: PricePanel) -> np.ndarray:
    """VRP_i = entry IV[decimal] − realized vol over hold[annualized]. The single
    carry driver; covered>uncovered means coverage inflates the short carry."""
    iv = pd.to_numeric(frame["iv_used"], errors="coerce").to_numpy(float)
    iv_dec = iv / 100.0 if np.nanmedian(np.abs(iv)) > 5.0 else iv
    rv = realized_vol_over_hold(frame["entry_ts"], frame["expiry_dt"], panel)
    return iv_dec - rv


# ──────────────────────────────────────────────────────────────────────────
# Deflated Sharpe Ratio (Bailey & Lopez de Prado) — §1.4, K=40, non-overlapping
# ──────────────────────────────────────────────────────────────────────────
def expected_max_sharpe(K: int, var_sr: float) -> float:
    """Bailey-LdP expected max of K independent Sharpe estimates (the DSR SR0)."""
    z1 = norm.ppf(1 - 1.0 / K)
    z2 = norm.ppf(1 - 1.0 / (K * np.e))
    return np.sqrt(var_sr) * ((1 - EULER_GAMMA) * z1 + EULER_GAMMA * z2)


def deflated_sharpe(pnl_nonov: np.ndarray, K: int = DSR_K) -> dict:
    """DSR on the NON-OVERLAPPING per-trade PnL series. Returns the per-obs SR,
    the K-deflated benchmark SR0, the DSR probability of skill, the implied
    t-stat and t-hurdle, and a PASS flag (p < 0.01 == t > hurdle)."""
    x = np.asarray(pnl_nonov, float)
    x = x[np.isfinite(x)]
    n = len(x)
    if n < 3 or x.std(ddof=1) == 0:
        return dict(n=n, sr_perobs=float("nan"), sr0=float("nan"),
                    dsr_p=float("nan"), t_stat=float("nan"),
                    t_hurdle=float("nan"), passed=False)
    sr_perobs = float(x.mean() / x.std(ddof=1))           # per-observation SR
    # skew/kurtosis-aware variance of the SR estimate (Lo 2002 / Mertens form).
    g1 = float(pd.Series(x).skew())
    g2 = float(pd.Series(x).kurt())                        # excess kurtosis
    var_sr = (1.0 - g1 * sr_perobs + (g2) / 4.0 * sr_perobs ** 2) / (n - 1)
    if not np.isfinite(var_sr) or var_sr <= 0:             # Gaussian fallback
        var_sr = (1.0 + 0.5 * sr_perobs ** 2) / (n - 1)
    sd_sr = np.sqrt(var_sr)
    sr0 = expected_max_sharpe(K, var_sr)
    dsr_p = float(norm.cdf((sr_perobs - sr0) / sd_sr))     # P(skill)
    t_stat = sr_perobs / sd_sr
    t_hurdle = sr0 / sd_sr
    # PASS = DSR p<0.01 at K (== the deflated t clears the hurdle). The spec also
    # pins an absolute t>3.23 floor; require BOTH (the stricter binds).
    passed = bool(dsr_p > 0.99 and t_stat > DSR_T_HURDLE)
    return dict(n=n, sr_perobs=sr_perobs, sr0=float(sr0), dsr_p=dsr_p,
                t_stat=float(t_stat), t_hurdle=float(t_hurdle),
                skew=g1, kurt=g2, passed=passed)


# ──────────────────────────────────────────────────────────────────────────
# Frozen-config eval (reuse the cached frame, or re-run the engine)
# ──────────────────────────────────────────────────────────────────────────
def load_frozen_frame(args) -> pd.DataFrame:
    """Return the FROZEN-config per-trade frame.

    Default: reuse the cached `verify_frame_band35.parquet` (built by
    verify_build_frames.py at the identical frozen config — no engine re-run).
    --reeval / --smoke: call evaluate_realistic with FROZEN_CFG (optionally
    window-sliced for the smoke), so the engine + gate plumbing is exercised
    end-to-end (not a stub)."""
    if args.smoke or args.reeval:
        cfg = FROZEN_CFG
        if args.smoke:
            # tiny real slice — exercises the FULL engine path on 3 dense days.
            cfg = RealisticEvalConfig(**{**FROZEN_CFG.__dict__,
                                         "start": args.smoke_start,
                                         "end": args.smoke_end})
        tl = pd.read_parquet(args.trade_log)
        res = evaluate_realistic(tl, cfg)
        f = res.trade_log.copy()
        f.index.name = "entry_ts"
        f = f.reset_index()
        return f
    f = pd.read_parquet(args.frame_cache)
    return f


def prep_short_covered(f: pd.DataFrame):
    """Return (short_covered, all_fired_short) frames with parsed timestamps,
    sorted by entry. SHORT-only, covered = premium_tier>0."""
    f = f.copy()
    f["entry_ts"] = pd.to_datetime(f["entry_ts"], utc=True)
    f["exit_ts"] = pd.to_datetime(f["expiry_dt"], utc=True)   # to-expiry
    f["year"] = f["entry_ts"].dt.year
    short_all = f[f["signal"] < 0].copy().sort_values("entry_ts").reset_index(drop=True)
    short_cov = short_all[short_all["premium_tier"] > 0].copy().reset_index(drop=True)
    return short_cov, short_all


# ──────────────────────────────────────────────────────────────────────────
# §1.1 window slicing + §1.2 walk-forward
# ──────────────────────────────────────────────────────────────────────────
def slice_window(scov: pd.DataFrame, start: str | None, end: str | None) -> pd.DataFrame:
    m = pd.Series(True, index=scov.index)
    if start is not None:
        m &= scov["entry_ts"] >= pd.Timestamp(start, tz="UTC")
    if end is not None:
        m &= scov["entry_ts"] <= pd.Timestamp(end, tz="UTC")
    return scov[m]


def window_metrics(w: pd.DataFrame) -> dict:
    """Per-trade edge + Sharpe family on a covered-short window (total_hedged_usd)."""
    if len(w) == 0 or "total_hedged_usd" not in w.columns:
        return dict(n=0, total=0.0, per_trade=float("nan"), nw=float("nan"),
                    nonov=float("nan"), nkeep=0, maxdd=0.0, worst=float("nan"))
    pnl = w["total_hedged_usd"].to_numpy(float)
    entry = pd.DatetimeIndex(w["entry_ts"]); exit_ = pd.DatetimeIndex(w["exit_ts"])
    if len(pnl) == 0:
        return dict(n=0, total=0.0, per_trade=float("nan"), nw=float("nan"),
                    nonov=float("nan"), nkeep=0, maxdd=0.0,
                    worst=float("nan"))
    nonov, nk = _nonoverlap_sharpe(pnl, entry, exit_)
    return dict(
        n=len(pnl), total=float(np.nansum(pnl)),
        per_trade=float(np.nanmean(pnl)),
        nw=_newey_west_sharpe(pnl, entry, exit_),
        nonov=nonov, nkeep=int(nk),
        maxdd=_max_drawdown(pnl),
        worst=float(np.nanmin(pnl)),
    )


def run_walk_forward(scov: pd.DataFrame) -> tuple[list[dict], bool]:
    """§1.2 anchored-expanding WF: evaluate the FROZEN band on each next
    out-of-fold test year. PASS = >=4/4 test-window per-trade edge positive."""
    folds = []
    n_pos = 0
    for train_label, test_label, pred in WF_FOLDS:
        te = scov[scov["year"].map(pred)]
        m = window_metrics(te)
        positive = bool(np.isfinite(m["per_trade"]) and m["per_trade"] > 0)
        n_pos += int(positive)
        folds.append(dict(train=train_label, test=test_label, positive=positive, **m))
    passed = n_pos >= 4
    return folds, passed


# ──────────────────────────────────────────────────────────────────────────
# §1.3 gates
# ──────────────────────────────────────────────────────────────────────────
def run_gates(scov_heldout: pd.DataFrame, short_all: pd.DataFrame,
              vrp_cov: np.ndarray, vrp_all: np.ndarray, *, min_covered: int) -> dict:
    """Run the 4 pre-committed gates on the covered held-out subset.

    Returns a dict of GateResult-like records plus the selection_bias rel_lift
    (recorded; NOT a kill)."""
    # The covered held-out frame must carry premium_tier + real_option_pnl_nc for
    # gate_no_fake_profit. We override model_col to the full hedged edge (§1.3.1).
    g1 = gate_no_fake_profit(
        scov_heldout, model_col="total_hedged_usd",
        real_col="real_option_pnl_nc", min_covered=min_covered,
    )
    # §1.3.1 extra floor: per-trade corr >= 0.85 (beyond the sign-primary gate).
    corr1 = float(g1.detail.get("per_trade_corr", 0.0))
    g1_pass = bool(g1.passed and corr1 >= CORR_FLOOR)

    # Gate 2 — pnl_realism: model=total_hedged_usd, fill=real_option_pnl_nc.
    model_pnl = scov_heldout["total_hedged_usd"].to_numpy(float)
    fill_pnl = scov_heldout["real_option_pnl_nc"].to_numpy(float)
    g2 = pnl_realism_gate(model_pnl, fill_pnl, min_corr=CORR_FLOOR,
                          max_total_rel_err=MAX_REL_ERR)

    # Gate 3 — sizing_sanity: pnl_per_unit = total_hedged_usd/vega, exp = vega.
    vega = scov_heldout["vega"].to_numpy(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        pnl_per_unit = np.where(vega > 0, model_pnl / vega, np.nan)
    g3 = sizing_sanity_gate(pnl_per_unit, vega, max_sharpe_spread=1.0)

    # Gate 4 — selection_bias: VRP_covered vs VRP_all_fired (EXPECTED FAIL).
    g4 = selection_bias_gate(vrp_cov, vrp_all, max_rel_lift=SEL_BIAS_MAX_REL_LIFT)
    rel_lift = float(g4.detail.get("rel_lift", float("nan")))

    return {
        "gate_no_fake_profit": _gate_record(g1, override_pass=g1_pass,
                                            extra={"corr_floor": CORR_FLOOR,
                                                   "corr_pass": corr1 >= CORR_FLOOR}),
        "pnl_realism_gate": _gate_record(g2),
        "sizing_sanity_gate": _gate_record(g3),
        "selection_bias_gate": _gate_record(g4),  # recorded; NOT a kill
        "selection_bias_rel_lift": rel_lift,
        "selection_bias_vrp_covered": float(g4.detail.get("mean_traded", float("nan"))),
        "selection_bias_vrp_universe": float(g4.detail.get("mean_universe", float("nan"))),
    }


def _gate_record(g: GateResult, *, override_pass: bool | None = None, extra=None) -> dict:
    d = dict(name=g.name, passed=bool(g.passed if override_pass is None else override_pass),
             raw_passed=bool(g.passed), message=g.message, detail=_jsonable(g.detail))
    if extra:
        d.update(extra)
    return d


def _jsonable(d: dict) -> dict:
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, np.integer)):
            v = v.item()
        elif isinstance(v, (np.bool_,)):
            v = bool(v)
        elif isinstance(v, float) and not np.isfinite(v):
            v = None
        out[k] = v
    return out


# ──────────────────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────────────────
def write_report(payload: dict, folds: list[dict], gates: dict, dsr: dict,
                 heldout_m: dict, fresh_m: dict, train_m: dict, overall: dict,
                 *, smoke: bool, report_md: str):
    L = []
    A = L.append
    tag = "  *(SMOKE — 3-date tiny slice; gates may be degenerate; this proves PLUMBING, not the verdict)*" if smoke else ""
    A(f"# S3 REVALIDATION — §1 frozen protocol result{tag}\n")
    A(f"**Config (FROZEN §0):** `{payload['config']}`\n")
    A(f"**Generated:** {payload['generated']}  •  **Source:** {payload['source']}\n")
    A("\n---\n\n## §1.1 Time-split windows (covered SHORT subset, total_hedged_usd)\n")
    A("| window | n | total $ | per-trade $ | NW t | nonov t (n) | MaxDD $ | worst $ |")
    A("|---|--:|--:|--:|--:|--:|--:|--:|")
    for label, m in [("TRAIN 2022-23 (ref only)", train_m),
                     ("HELD-OUT OOS 2024-01..2026-05", heldout_m),
                     ("FRESH OOS 2026-06+", fresh_m)]:
        A(f"| {label} | {m['n']} | {m['total']:,.0f} | {m['per_trade']:,.0f} | "
          f"{m['nw']:.2f} | {m['nonov']:.2f} ({m['nkeep']}) | {m['maxdd']:,.0f} | {m['worst']:,.0f} |")
    A("\n## §1.2 Walk-forward (anchored-expanding; band FROZEN; stability test)\n")
    A("| fold | train | test | per-trade $ | n | positive |")
    A("|---|---|---|--:|--:|:--:|")
    for fo in folds:
        A(f"| {fo['train']}→{fo['test']} | {fo['train']} | {fo['test']} | "
          f"{fo['per_trade']:,.0f} | {fo['n']} | {'YES' if fo['positive'] else 'NO'} |")
    n_pos = sum(1 for fo in folds if fo["positive"])
    A(f"\n**WF result: {n_pos}/4 folds positive per-trade → "
      f"{'PASS' if overall['wf_pass'] else 'FAIL'}** (PASS requires ≥4/4).\n")
    A("\n## §1.3 Gates (covered held-out OOS; pre-committed thresholds)\n")
    for key in ["gate_no_fake_profit", "pnl_realism_gate", "sizing_sanity_gate", "selection_bias_gate"]:
        g = gates[key]
        note = "  *(EXPECTED FAIL — recorded, feeds §2 haircut; NOT a kill)*" if key == "selection_bias_gate" else ""
        A(f"- **{key}**: {'PASS' if g['passed'] else 'FAIL'}{note}  \n  {g['message']}")
    A(f"\n  - selection_bias **rel_lift = {gates['selection_bias_rel_lift']:+.4f}** "
      f"(VRP covered {gates['selection_bias_vrp_covered']:+.4f} vs all-fired "
      f"{gates['selection_bias_vrp_universe']:+.4f}) → §2 H1 de-biasing input.\n")
    A("\n## §1.4 Deflated Sharpe Ratio (non-overlapping SHORT series, K=40)\n")
    A(f"- non-overlapping n = {dsr['n']}, per-obs SR = {dsr['sr_perobs']:.4f}")
    A(f"- K=40 benchmark SR0 = {dsr['sr0']:.4f}, deflated t = {dsr['t_stat']:.3f} "
      f"(hurdle {dsr['t_hurdle']:.3f}; abs floor {DSR_T_HURDLE})")
    A(f"- **DSR p(skill) = {dsr['dsr_p']:.4f} → {'PASS' if dsr['passed'] else 'FAIL'}** "
      f"(PASS = p>0.99 / t>{DSR_T_HURDLE}).\n")
    A("\n## §1.5 OVERALL PASS (criteria 1-5; selection_bias recorded only)\n")
    crit = overall["criteria"]
    A("| # | criterion | result |")
    A("|---|---|:--:|")
    for i, (desc, ok) in enumerate(crit, 1):
        A(f"| {i} | {desc} | {'PASS' if ok else 'FAIL'} |")
    A(f"\n**S3 §1 REVALIDATION: {'PASS' if overall['overall_pass'] else 'FAIL'}**")
    if smoke:
        A("\n> SMOKE RUN — not the verdict. Confirms the engine + 4 gate APIs + WF + "
          "DSR are wired correctly end-to-end. Run the full protocol (no --smoke) "
          "when the CPU frees.")
    else:
        A("\n> If any of criteria 1-5 FAIL: **STOP and re-examine** (§1 binding clause — "
          "do not relax a threshold to pass). Then run `scripts/s3_capacity_haircut.py`.")
    with open(report_md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


# ──────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="S3 §1 frozen revalidation protocol")
    ap.add_argument("--frame-cache", default=FRAME_CACHE,
                    help="cached frozen-config per-trade frame (band 0.35)")
    ap.add_argument("--trade-log", default=TRADE_LOG)
    ap.add_argument("--reeval", action="store_true",
                    help="re-run evaluate_realistic with the FROZEN config (slow)")
    ap.add_argument("--smoke", action="store_true",
                    help="3-date tiny slice; proves gate wiring (EXIT 0)")
    # 2025-03-14..16 is a dense covered-short window (21 covered shorts on the
    # 15th alone) — gives the smoke a real covered subset so the gates are
    # exercised on live data, not just degenerate-empty stubs.
    ap.add_argument("--smoke-start", default="2025-03-14")
    ap.add_argument("--smoke-end", default="2025-03-16")
    ap.add_argument("--report", default=REPORT_MD)
    ap.add_argument("--per-fold-dir", default=OUT_DIR,
                    help="dir for per-fold parquet dumps")
    args = ap.parse_args()

    source = ("engine re-eval (FROZEN_CFG)" if args.reeval else
              ("SMOKE 3-date engine slice" if args.smoke else f"cached {args.frame_cache}"))
    print(f"[s3_revalidate] source = {source}")
    f = load_frozen_frame(args)
    scov, short_all = prep_short_covered(f)
    print(f"[s3_revalidate] covered SHORT n={len(scov)}  all-fired SHORT n={len(short_all)}")

    # Build the price panel ONCE for the VRP-over-hold computation (selection bias).
    cfg_panel = FROZEN_CFG
    pe = pd.to_datetime(short_all["entry_ts"], utc=True)
    px = pd.to_datetime(short_all["exit_ts"] if "exit_ts" in short_all else short_all["expiry_dt"], utc=True)
    panel = PricePanel.load(cfg_panel, pe.min(), px.max() + pd.Timedelta(days=1))
    vrp_cov = compute_vrp_per_trade(scov, panel)
    vrp_all = compute_vrp_per_trade(short_all, panel)

    # ── §1.1 windows ──
    train_m = window_metrics(slice_window(scov, TRAIN_START, TRAIN_END))
    heldout = slice_window(scov, HELDOUT_START, HELDOUT_END)
    heldout_m = window_metrics(heldout)
    fresh = slice_window(scov, FRESH_START, None)
    fresh_m = window_metrics(fresh)

    # ── §1.2 walk-forward ──
    folds, wf_pass = run_walk_forward(scov)
    # per-fold parquet dumps
    for fo, (_, test_label, pred) in zip(folds, WF_FOLDS):
        te = scov[scov["year"].map(pred)]
        path = f"{args.per_fold_dir}/S3_revalidation_fold_{str(test_label).replace(' ', '')}.parquet"
        te.to_parquet(path)
        fo["parquet"] = path

    # ── §1.3 gates (covered held-out OOS) ──
    # gate_no_fake_profit's default min_covered=30 is relaxed for the smoke only.
    min_covered = 5 if args.smoke else 30
    # Selection bias uses the WHOLE covered-short VRP vs ALL-fired-short VRP
    # (the universe), per §1.3.4. Run gates on the held-out covered subset for
    # the realism/sizing gates, but selection_bias is on the full coverage split.
    heldout_cov_idx = heldout.index
    gates = run_gates(
        heldout if not args.smoke else scov,   # smoke: use all covered (tiny held-out)
        short_all, vrp_cov, vrp_all, min_covered=min_covered,
    )

    # ── §1.4 DSR on the NON-OVERLAPPING held-out short series ──
    dsr_src = heldout if (not args.smoke and len(heldout) >= 3) else scov
    pnl = dsr_src["total_hedged_usd"].to_numpy(float)
    entry = pd.DatetimeIndex(dsr_src["entry_ts"]); exit_ = pd.DatetimeIndex(dsr_src["exit_ts"])
    # non-overlapping subsample (same greedy walk as _nonoverlap_sharpe), then DSR.
    order = np.argsort(entry.values)
    e = entry.values[order]; x = exit_.values[order]; p = pnl[order]
    kept = []
    last = np.datetime64("1900-01-01")
    for i in range(len(e)):
        if e[i] >= last:
            kept.append(p[i]); last = x[i]
    dsr = deflated_sharpe(np.asarray(kept, float), K=DSR_K)

    # ── §1.5 OVERALL PASS ──
    held_edge_pos = bool(np.isfinite(heldout_m["per_trade"]) and heldout_m["per_trade"] > 0)
    nw_t = heldout_m["nw"]
    c1 = held_edge_pos and np.isfinite(nw_t) and nw_t > NW_T_HURDLE
    c2 = wf_pass
    c3 = gates["gate_no_fake_profit"]["passed"] and gates["pnl_realism_gate"]["passed"]
    c4 = gates["sizing_sanity_gate"]["passed"]
    c5 = dsr["passed"]
    criteria = [
        (f"OOS held-out per-trade>0 AND non-overlap NW t>{NW_T_HURDLE}", bool(c1)),
        ("≥4/4 walk-forward folds positive per-trade", bool(c2)),
        ("gate_no_fake_profit AND pnl_realism_gate PASS (corr≥0.85, no sign flip)", bool(c3)),
        ("sizing_sanity_gate PASS (no sign flip)", bool(c4)),
        (f"DSR p<0.01 at K={DSR_K} on the non-overlapping series", bool(c5)),
    ]
    overall_pass = all(ok for _, ok in criteria)
    overall = dict(wf_pass=wf_pass, criteria=criteria, overall_pass=overall_pass)

    payload = dict(
        generated=pd.Timestamp.now("UTC").isoformat(),
        source=source,
        config=str(FROZEN_CFG),
        windows=dict(train=train_m, heldout=heldout_m, fresh=fresh_m),
        walk_forward=dict(folds=[{k: (v if not isinstance(v, (np.floating, np.integer)) else v.item())
                                  for k, v in fo.items() if k != "parquet"} for fo in folds],
                          n_positive=sum(1 for fo in folds if fo["positive"]),
                          passed=wf_pass),
        gates=gates,
        dsr=_jsonable(dsr),
        overall=dict(criteria=[[d, ok] for d, ok in criteria], overall_pass=overall_pass),
    )

    write_report(payload, folds, gates, dsr, heldout_m, fresh_m, train_m, overall,
                 smoke=args.smoke, report_md=args.report)
    with open(args.report.replace(".md", ".json"), "w", encoding="utf-8") as fh:
        json.dump(_deep_jsonable(payload), fh, indent=2)

    # ── console summary ──
    print("=" * 74)
    print("S3 REVALIDATION — §1 frozen protocol" + ("  [SMOKE]" if args.smoke else ""))
    print("=" * 74)
    print(f"HELD-OUT OOS: n={heldout_m['n']} per-trade ${heldout_m['per_trade']:,.0f} "
          f"NW t={heldout_m['nw']:.2f} nonov t={heldout_m['nonov']:.2f}")
    print(f"FRESH OOS   : n={fresh_m['n']} per-trade ${fresh_m['per_trade']:,.0f}")
    print(f"WALK-FWD    : {sum(1 for fo in folds if fo['positive'])}/4 positive → {'PASS' if wf_pass else 'FAIL'}")
    for key in ["gate_no_fake_profit", "pnl_realism_gate", "sizing_sanity_gate", "selection_bias_gate"]:
        g = gates[key]
        sfx = "  (EXPECTED FAIL → §2 haircut)" if key == "selection_bias_gate" else ""
        print(f"GATE {key:22s}: {'PASS' if g['passed'] else 'FAIL'}{sfx}")
    print(f"  selection_bias rel_lift = {gates['selection_bias_rel_lift']:+.4f}  (→ S3_CAPACITY H1 input)")
    print(f"DSR K=40    : n={dsr['n']} t={dsr['t_stat']:.3f} p={dsr['dsr_p']:.4f} → {'PASS' if dsr['passed'] else 'FAIL'}")
    print("-" * 74)
    print("§1.5 CRITERIA:")
    for i, (desc, ok) in enumerate(criteria, 1):
        print(f"  {i}. [{'PASS' if ok else 'FAIL'}] {desc}")
    print(f"OVERALL §1: {'PASS' if overall_pass else 'FAIL'}")
    print(f"report -> {args.report}  (+ .json)")
    print("=" * 74)
    if args.smoke:
        print("SMOKE OK — gate/WF/DSR plumbing exercised end-to-end. "
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
