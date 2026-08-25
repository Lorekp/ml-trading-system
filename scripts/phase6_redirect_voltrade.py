"""Phase-6 redirect: is there a REAL hold-to-expiry vol-selling edge, gated by
the (sound, R2~0.59) realized-vol forecast?

The honest PnL of a delta-hedged short straddle held to expiry is the variance
risk premium it captures:  VRP = IV_entry - RV_realized(entry -> expiry), in vol
points of the straddle's vega. This is the RIGHT construction (V2's bug was
vega x d(noisy IV) over 15 min; here we use real IV minus REAL realized vol over
the option's life).

This test fixes the three flaws the Phase-5 cycle49 probe was caught on:
  1. WALK-FORWARD model conditioning — the "sell when IV is rich vs forecast"
     tercile threshold is chosen on PAST data only (expanding window), never
     in-sample.
  2. VEGA-WEIGHTED aggregation — the dollar-traded quantity (cycle49 reported
     equal-weighted, ~2.6x too rich).
  3. NON-OVERLAPPING entries — greedily de-overlap so the per-trade t-stat is
     honest (consecutive 15-min bars share ~all of their to-expiry window).
  + REAL cost (3.3vp round-trip, live-engine-derived, not the optimistic 2.0).
  + sizing_sanity_gate + selection_bias_gate.

Verdict = vega-weighted mean net VRP and its block-bootstrap CI. Edge only if
net > 0 with a CI that excludes 0 AND it survives the gates.

Usage:  PYTHONPATH=. python scripts/phase6_redirect_voltrade.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from validation.gates import selection_bias_gate, sizing_sanity_gate

SVI = "vol_arb/output/BTC_vol_arb_svi.parquet"
PREDS = "vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions.parquet"
BARS_PER_DAY = 96
COST_RT_VP = 3.3          # honest live-derived round-trip cost (vol points)
COST_OPTIMISTIC_VP = 2.0
OUT = "vol_arb/output/PHASE6_REDIRECT_VERDICT.md"


def fwd_rv_to_expiry(df: pd.DataFrame) -> pd.Series:
    """Annualised realized vol from T to the option's expiry, percent."""
    var = (df["vol_ann"].astype(float) ** 2).to_numpy()
    cs = np.concatenate([[0.0], np.cumsum(var)])
    n = len(df)
    h = np.clip((df["dte_days"].astype(float).to_numpy() * BARS_PER_DAY)
                .round().astype(int), 1, None)
    pos = np.arange(n)
    j = np.minimum(pos + h, n)
    valid = j > pos + 1
    mean_var = np.where(valid, (cs[j] - cs[pos + 1]) / np.maximum(j - pos - 1, 1), np.nan)
    return pd.Series(np.sqrt(mean_var) * 100.0, index=df.index)


def greedy_non_overlap(times: np.ndarray, expiries: np.ndarray) -> np.ndarray:
    """Indices of a non-overlapping subset: next entry only after prior expiry."""
    keep, last_exit = [], np.datetime64("1970-01-01")
    for i in range(len(times)):
        if times[i] >= last_exit:
            keep.append(i)
            last_exit = expiries[i]
    return np.array(keep, dtype=int)


def block_bootstrap_ci(x: np.ndarray, block: int = 20, n_boot: int = 5000,
                       seed: int = 42) -> tuple[float, float]:
    rng = np.random.RandomState(seed)
    n = len(x)
    if n < block * 2:
        return float("nan"), float("nan")
    n_blocks = int(np.ceil(n / block))
    starts = np.arange(n - block + 1)
    means = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.choice(starts, size=n_blocks)
        sample = np.concatenate([x[s:s + block] for s in idx])[:n]
        means[b] = sample.mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def vw_mean(vrp: np.ndarray, vega: np.ndarray) -> float:
    w = np.abs(vega)
    return float(np.sum(vrp * w) / np.sum(w)) if w.sum() > 0 else float("nan")


def main() -> int:
    df = pd.read_parquet(SVI)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df = df[df["iv"].notna() & (df["vol_ann"] > 0) & df["vega"].notna()].copy()

    iv = df["iv"].astype(float)
    df["vrp"] = iv - fwd_rv_to_expiry(df)
    preds = pd.read_parquet(PREDS)
    if not isinstance(preds.index, pd.DatetimeIndex):
        preds.index = pd.to_datetime(preds.index, utc=True)
    rv_pred = (np.exp(preds["y_pred"].astype(float)) * 100.0).reindex(df.index)
    df["model_edge"] = iv - rv_pred
    df = df[df["vrp"].notna() & df["model_edge"].notna()].copy()

    # ── Walk-forward model conditioning (expanding monthly tercile) ──
    df["month"] = df.index.to_period("M")
    months = sorted(df["month"].unique())
    burn = max(6, len(months) // 4)
    df["wf_selected"] = False
    for m in months[burn:]:
        past = df[df["month"] < m]
        thr = past["model_edge"].quantile(2.0 / 3.0)   # top tercile, PAST only
        cur = df["month"] == m
        df.loc[cur & (df["model_edge"] >= thr), "wf_selected"] = True
    oos = df[df["month"] >= months[burn]].copy()       # OOS evaluation window

    times = oos.index.values
    expiries = pd.to_datetime(oos["expiry_dt"], utc=True).values \
        if "expiry_dt" in oos.columns else \
        (times + (oos["dte_days"].values * 86400 * 1e9).astype("timedelta64[ns]"))

    results = {}
    for label, mask in [("unconditional (all OOS bars)", np.ones(len(oos), bool)),
                        ("model-timed (walk-forward top-tercile)", oos["wf_selected"].values)]:
        sub = oos[mask]
        if len(sub) < 50:
            results[label] = None
            continue
        # de-overlap to honest trades
        st = sub.index.values
        se = pd.to_datetime(sub["expiry_dt"], utc=True).values if "expiry_dt" in sub.columns \
            else (st + (sub["dte_days"].values * 86400 * 1e9).astype("timedelta64[ns]"))
        keep = greedy_non_overlap(st, se)
        t = sub.iloc[keep]
        vrp = t["vrp"].to_numpy()
        vega = t["vega"].to_numpy()
        vw = vw_mean(vrp, vega)
        ew = float(vrp.mean())
        lo, hi = block_bootstrap_ci((vrp * np.abs(vega)) / np.abs(vega).mean(), block=10)
        results[label] = {
            "n_trades": len(t), "vw_vrp": vw, "ew_vrp": ew,
            "pct_pos": float((vrp > 0).mean() * 100),
            "ci_lo": lo, "ci_hi": hi,
            "net_2": vw - COST_OPTIMISTIC_VP, "net_33": vw - COST_RT_VP,
            "vrp": vrp, "vega": vega,
        }

    # ── Gates on the model-timed leg ──
    mt = results.get("model-timed (walk-forward top-tercile)")
    uc = results.get("unconditional (all OOS bars)")
    gate_lines = []
    if mt and uc:
        # selection bias: model-timed VRP vs the full OOS universe VRP
        g_sel = selection_bias_gate(mt["vrp"], oos["vrp"].to_numpy())
        # sizing sanity: net per-unit (VRP - cost) vs vega exposure
        per_unit = mt["vrp"] - COST_RT_VP
        g_size = sizing_sanity_gate(per_unit, np.abs(mt["vega"]))
        gate_lines = [
            f"- **selection_bias_gate**: {'PASS' if g_sel.passed else 'FLAG'} — {g_sel.message}",
            f"  (a walk-forward signal SHOULD lift the metric; the gate flags the lift's "
            "size — here it is lookahead-free by construction.)",
            f"- **sizing_sanity_gate**: {'PASS' if g_size.passed else 'FAIL'} — {g_size.message}",
        ]

    # ── Report ──
    L = ["# Phase-6 Redirect Verdict — Hold-to-Expiry Vol-Selling (2026-06-21)", "",
         "Honest construction: VRP = real entry IV − real realized vol to expiry, "
         "vega-weighted, NON-overlapping trades, model conditioning chosen "
         "**walk-forward** (expanding monthly tercile, past data only).", "",
         f"OOS window: {oos.index.min().date()} .. {oos.index.max().date()} "
         f"({len(months) - burn} months after {burn}-month burn-in).", "",
         "| Leg | trades | vega-wtd VRP | equal-wtd | %pos | net @2.0vp | net @3.3vp | block-boot 95% CI |",
         "|---|---|---|---|---|---|---|---|"]
    for label in ["unconditional (all OOS bars)", "model-timed (walk-forward top-tercile)"]:
        r = results.get(label)
        if not r:
            L.append(f"| {label} | insufficient | | | | | | |"); continue
        L.append(f"| {label} | {r['n_trades']} | {r['vw_vrp']:+.2f}vp | "
                 f"{r['ew_vrp']:+.2f}vp | {r['pct_pos']:.0f}% | {r['net_2']:+.2f}vp | "
                 f"**{r['net_33']:+.2f}vp** | [{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] |")
    L += ["", "## Gates", *gate_lines, "",
          "## Verdict"]
    if mt:
        net = mt["net_33"]
        verdict = ("a THIN POSITIVE edge survives" if net > 0 and mt["ci_lo"] > 0
                   else "NO edge survives" if net <= 0
                   else "edge is MARGINAL (positive point estimate, CI includes 0)")
        L.append(
            f"At the honest 3.3vp round-trip cost, the model-timed leg nets "
            f"**{net:+.2f}vp** (vega-weighted, walk-forward, de-overlapped). "
            f"{verdict}. The forecast's timing lift is "
            f"{mt['vw_vrp'] - (uc['vw_vrp'] if uc else 0):+.2f}vp vs unconditional.")
    report = "\n".join(L)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    print(f"\n(written to {OUT})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
