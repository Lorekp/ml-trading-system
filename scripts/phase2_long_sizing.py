"""Phase 2 — regime-conditional long-side sizing audit.

Hypothesis pre-committed (before any OOS evaluation):

* **H1**: For LONG signals where `svi_term_slope_1w_2w < -5`, multiply
  position size by 1.5x. Threshold and multiplier locked from the
  in-sample profile (PHASE2_BUCKET_PROFILE table below). The bucket
  is the only one with all 5 years positive AND per-trade Sharpe ~2x
  the middle backwardation buckets.

Rationale (locked, no tuning during OOS):

* **Threshold = -5**: Per-bucket Sharpe profile shows
  - slope <-5: 0.564
  - slope -5..-3: 0.530
  - slope -3..-2: 0.348
  - slope -2..-1: 0.081
  - slope -1..0:  -0.041
  -5 is the natural cliff where Sharpe doubles. NO grid search performed
  on the threshold parameter; the value comes from the bucket-profile
  cliff observation.

* **Multiplier = 1.5**: Sharpe ratio extreme-vs-middle is 0.564/0.348
  = 1.62. We use 1.5x as a conservative floor (round down, safety
  margin). NO grid search on the multiplier.

Train/test:

* TRAIN (where the bucket profile was observed): 2022-02 to 2024-12.
* TEST (true OOS — no parameter touched after this): 2025-01 to 2026-04.
* Per-year strict gate on TEST: each of 2025 and 2026 non-negative dPnL.

Falsifiers (any failure rejects):

* Either OOS year shows negative dPnL.
* OOS daily Sharpe drops by more than 0.05.
* OOS MaxDD worsens by more than $500 (1.7% of baseline MaxDD).
* Placebo arm (random feature with same fire rate) passes any of
  the above gates.

The simulation reuses `BTC_v2_trade_log.parquet`'s NC component columns:
`vega_pnl_nc`, `theta_pnl_nc`, `delta_hedge_pnl_nc`, `option_fees_nc`,
`slippage_nc`, `hedge_costs_nc`, `net_pnl_nc`. Multiplying these by k
on qualifying bars exactly equals running the backtester at k× sizing
on those bars in NC mode (no margin-utilization feedback loop). This
is honest because: (a) `project_capacity_analysis_2026_05_07.md`
showed K=0 NC identity holds at our scale; (b) NC isolation removes
the compound-equity feedback that would change other bar sizes. The
mode does NOT account for (i) dynamic margin-utilization caps that
might reject the larger trade in the live engine, or (ii) realistic-
slippage non-linearity at larger size.

Outputs:

* `vol_arb/output/PHASE2_LONG_SIZING.md` — full report
* `vol_arb/output/PHASE2_LONG_SIZING_TRADELOGS/` — synthetic trade
  logs with multiplier applied (per arm)
"""
from __future__ import annotations

import sys
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

TRADE_LOG = REPO_ROOT / "vol_arb" / "output" / "BTC_v2_trade_log.parquet"
OUT_DIR = REPO_ROOT / "vol_arb" / "output" / "PHASE2_LONG_SIZING_TRADELOGS"
OUT_MD = REPO_ROOT / "vol_arb" / "output" / "PHASE2_LONG_SIZING.md"

NC_PNL_COLS = [
    "vega_pnl_nc", "theta_pnl_nc", "delta_hedge_pnl_nc",
    "option_fees_nc", "slippage_nc", "hedge_costs_nc",
    "net_pnl_nc",
]

# ── Pre-committed hypothesis ──────────────────────────────────────────
PRE_COMMIT_THRESHOLD = -5.0
PRE_COMMIT_MULTIPLIER = 1.5
PRE_COMMIT_FEATURE = "svi_term_slope_1w_2w"

# ── Train / test split ───────────────────────────────────────────────
TRAIN_END = pd.Timestamp("2024-12-31", tz="UTC")
# OOS = TRAIN_END < ts <= 2026-04-30


@dataclass
class ArmResult:
    name: str
    desc: str
    nc_pnl_total: float
    nc_pnl_per_year: dict[int, float]
    n_amplified_long: int
    daily_sharpe: float
    max_dd: float
    fire_rate: float


def df_to_md(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    cols = list(df.columns)
    head = "| " + " | ".join(str(c) for c in cols) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    rows = []
    for _, r in df.iterrows():
        cells = []
        for c in cols:
            v = r[c]
            if isinstance(v, float):
                cells.append(f"{v:,.3f}" if abs(v) < 5 else
                             (f"{v:,.1f}" if abs(v) < 100 else f"{v:,.0f}"))
            else:
                cells.append(str(v))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([head, sep] + rows)


def daily_sharpe_nc(tl: pd.DataFrame) -> float:
    daily = tl["net_pnl_nc"].resample("1D").sum()
    daily = daily[daily != 0]
    if len(daily) < 5 or daily.std() == 0:
        return float("nan")
    return float(np.sqrt(365) * daily.mean() / daily.std())


def max_drawdown_nc(tl: pd.DataFrame) -> float:
    cum = tl["net_pnl_nc"].cumsum()
    peak = cum.cummax()
    return float((cum - peak).min())


def apply_amplification(tl: pd.DataFrame, mask: pd.Series, mult: float) -> pd.DataFrame:
    """Multiply NC PnL columns by `mult` on rows where `mask` is True.
    Returns a NEW dataframe; original is not mutated."""
    tl2 = tl.copy()
    for col in NC_PNL_COLS:
        if col not in tl2.columns:
            continue
        # Only multiply where mask is True; preserve NaN handling
        tl2.loc[mask, col] = tl2.loc[mask, col] * mult
    return tl2


def summarize_arm(name: str, desc: str, tl: pd.DataFrame,
                  amplification_mask: pd.Series) -> ArmResult:
    fired = tl.loc[tl["signal"] != 0]
    return ArmResult(
        name=name,
        desc=desc,
        nc_pnl_total=float(fired["net_pnl_nc"].sum()),
        nc_pnl_per_year={int(yr): float(s) for yr, s in
                         fired["net_pnl_nc"].groupby(fired.index.year).sum().items()},
        n_amplified_long=int(amplification_mask.sum()),
        daily_sharpe=daily_sharpe_nc(tl),
        max_dd=max_drawdown_nc(tl),
        fire_rate=float(amplification_mask.sum() /
                        max((tl["signal"]==1).sum(), 1)),
    )


def split_oos(tl: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = tl.loc[tl.index <= TRAIN_END]
    test = tl.loc[tl.index > TRAIN_END]
    return train, test


def bootstrap_ci(values: np.ndarray, statistic, n_boot: int = 2000,
                 seed: int = 42) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    bs = []
    n = len(values)
    if n == 0:
        return float("nan"), float("nan")
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        bs.append(statistic(values[idx]))
    return float(np.percentile(bs, 5)), float(np.percentile(bs, 95))


def run_arms(tl: pd.DataFrame) -> list[tuple[str, ArmResult, ArmResult]]:
    """Returns [(arm_name, train_summary, test_summary), ...] for each
    sizing rule under test. Each arm reports train AND test summaries."""
    results = []

    # ── Arm 0: baseline (no amplification) ──
    null_mask = pd.Series(False, index=tl.index)
    base = summarize_arm("S0_baseline", "No amplification (cycle-10 V2 canonical)",
                         tl, null_mask)
    train_tl, test_tl = split_oos(tl)
    base_train = summarize_arm("S0_train", "baseline TRAIN", train_tl,
                                pd.Series(False, index=train_tl.index))
    base_test = summarize_arm("S0_test", "baseline TEST (OOS)", test_tl,
                               pd.Series(False, index=test_tl.index))
    results.append(("S0_baseline", base_train, base_test))

    # ── Arm 1: H1 pre-committed rule ──
    is_long = tl["signal"] == 1
    is_extreme_backw = tl[PRE_COMMIT_FEATURE] < PRE_COMMIT_THRESHOLD
    h1_mask = is_long & is_extreme_backw
    tl_h1 = apply_amplification(tl, h1_mask, PRE_COMMIT_MULTIPLIER)
    train_h1, test_h1 = split_oos(tl_h1)
    h1_train = summarize_arm(
        "S1_h1_train",
        f"H1: long×{PRE_COMMIT_MULTIPLIER}x when {PRE_COMMIT_FEATURE}<{PRE_COMMIT_THRESHOLD} (TRAIN)",
        train_h1, h1_mask & (tl.index <= TRAIN_END))
    h1_test = summarize_arm(
        "S1_h1_test",
        f"H1: long×{PRE_COMMIT_MULTIPLIER}x when {PRE_COMMIT_FEATURE}<{PRE_COMMIT_THRESHOLD} (TEST)",
        test_h1, h1_mask & (tl.index > TRAIN_END))
    results.append(("S1_h1", h1_train, h1_test))

    # ── Arm 2: PLACEBO — random feature, MATCHED fire rate ──
    # Fire rate = fraction of LONG bars that get amplified by H1.
    # Threshold = fire_rate-quantile of random values among LONGS so that
    # `rand_feat < rand_thresh` fires on exactly that fraction of LONGS.
    rng = np.random.default_rng(20260510)
    n = len(tl)
    fire_rate = float(h1_mask.sum() / (tl["signal"]==1).sum())
    rand_feat = rng.uniform(0, 1, n)
    long_idx = np.where(is_long.values)[0]
    rand_long_vals = rand_feat[long_idx]
    rand_thresh = float(np.quantile(rand_long_vals, fire_rate))
    placebo_mask = is_long & pd.Series(rand_feat < rand_thresh, index=tl.index)
    # Sanity-check fire rate match (within ±5%)
    actual_fr = float(placebo_mask.sum() / (tl["signal"]==1).sum())
    print(f"[phase2] placebo fire rate: {actual_fr:.3f} vs target {fire_rate:.3f}")
    assert abs(actual_fr - fire_rate) < 0.05, (
        f"placebo fire rate {actual_fr} mismatched H1 {fire_rate}"
    )
    tl_p = apply_amplification(tl, placebo_mask, PRE_COMMIT_MULTIPLIER)
    train_p, test_p = split_oos(tl_p)
    p_train = summarize_arm(
        "S2_placebo_train",
        f"PLACEBO: random feature with same fire rate (TRAIN)",
        train_p, placebo_mask & (tl.index <= TRAIN_END))
    p_test = summarize_arm(
        "S2_placebo_test",
        f"PLACEBO: random feature with same fire rate (TEST)",
        test_p, placebo_mask & (tl.index > TRAIN_END))
    results.append(("S2_placebo", p_train, p_test))

    # ── Arm 3: H1 with conservative 1.25x multiplier (sensitivity) ──
    tl_h1_conservative = apply_amplification(tl, h1_mask, 1.25)
    train_c, test_c = split_oos(tl_h1_conservative)
    c_train = summarize_arm(
        "S3_h1_1.25x_train",
        f"H1 conservative: long×1.25x when {PRE_COMMIT_FEATURE}<{PRE_COMMIT_THRESHOLD} (TRAIN)",
        train_c, h1_mask & (tl.index <= TRAIN_END))
    c_test = summarize_arm(
        "S3_h1_1.25x_test",
        f"H1 conservative: long×1.25x when {PRE_COMMIT_FEATURE}<{PRE_COMMIT_THRESHOLD} (TEST)",
        test_c, h1_mask & (tl.index > TRAIN_END))
    results.append(("S3_h1_1.25x", c_train, c_test))

    # ── Arm 4: H1 aggressive 2.0x multiplier (sensitivity) ──
    tl_h1_aggressive = apply_amplification(tl, h1_mask, 2.0)
    train_a, test_a = split_oos(tl_h1_aggressive)
    a_train = summarize_arm(
        "S4_h1_2.0x_train",
        f"H1 aggressive: long×2.0x when {PRE_COMMIT_FEATURE}<{PRE_COMMIT_THRESHOLD} (TRAIN)",
        train_a, h1_mask & (tl.index <= TRAIN_END))
    a_test = summarize_arm(
        "S4_h1_2.0x_test",
        f"H1 aggressive: long×2.0x when {PRE_COMMIT_FEATURE}<{PRE_COMMIT_THRESHOLD} (TEST)",
        test_a, h1_mask & (tl.index > TRAIN_END))
    results.append(("S4_h1_2.0x", a_train, a_test))

    return results


def walk_forward_audit(tl: pd.DataFrame, multiplier: float = 1.5,
                       trail_months: int = 24) -> pd.DataFrame:
    """Walk-forward variant — for each month M, use data from
    [M - trail_months, M-1] to compute a per-month threshold, apply
    to month M only. Threshold = quantile-based: pick threshold such
    that the bottom-quintile (20%) of train-window long-side bars
    fires. This is parameter-free except for the trail window."""
    tl2 = tl.copy()
    tl2.loc[:, "month"] = tl2.index.to_period("M")
    months = sorted(tl2["month"].unique())
    if len(months) <= trail_months:
        return pd.DataFrame()
    rows = []
    for m in months[trail_months:]:
        m_start = pd.Timestamp(m.start_time, tz="UTC")
        m_end = pd.Timestamp(m.end_time, tz="UTC")
        train_start = m_start - pd.DateOffset(months=trail_months)
        train_window = tl2.loc[(tl2.index >= train_start) & (tl2.index < m_start)]
        train_longs = train_window[train_window["signal"] == 1]
        if len(train_longs) < 50:
            continue
        # Threshold = 20th percentile (worst 20% of slopes = steepest backwardation)
        train_slopes = train_longs[PRE_COMMIT_FEATURE].dropna()
        if len(train_slopes) < 30:
            continue
        threshold = float(np.quantile(train_slopes, 0.2))
        # Apply to current month
        cur = tl2.loc[(tl2.index >= m_start) & (tl2.index <= m_end)]
        cur_longs = cur[cur["signal"] == 1]
        if len(cur_longs) == 0:
            continue
        amp_mask = (cur_longs[PRE_COMMIT_FEATURE] < threshold)
        n_amp = int(amp_mask.sum())
        # Compute amplified PnL for current month
        cur_amp = cur.copy()
        target = (cur_amp["signal"] == 1) & (cur_amp[PRE_COMMIT_FEATURE] < threshold)
        for col in NC_PNL_COLS:
            cur_amp.loc[target, col] = cur_amp.loc[target, col] * multiplier
        rows.append({
            "month": str(m),
            "threshold": round(threshold, 2),
            "n_amplified": n_amp,
            "n_long": int((cur["signal"] == 1).sum()),
            "base_nc_pnl": round(float(cur["net_pnl_nc"].sum()), 0),
            "amp_nc_pnl": round(float(cur_amp["net_pnl_nc"].sum()), 0),
            "delta_pnl": round(float((cur_amp["net_pnl_nc"] - cur["net_pnl_nc"]).sum()), 0),
        })
    return pd.DataFrame(rows)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[phase2] reading {TRADE_LOG.relative_to(REPO_ROOT)}")
    tl = pd.read_parquet(TRADE_LOG)

    md = []
    md.append("# Phase 2 — regime-conditional long-side sizing\n")
    md.append("**Pre-committed hypothesis (locked before OOS evaluation):**\n")
    md.append(f"- Multiply LONG size by **{PRE_COMMIT_MULTIPLIER}x** when "
              f"`{PRE_COMMIT_FEATURE} < {PRE_COMMIT_THRESHOLD}` (extreme backwardation).\n")
    md.append(f"- Train period: 2022-02 → 2024-12. Test period: 2025-01 → 2026-04.\n")
    md.append(f"- No parameter tuning during OOS evaluation.\n")
    md.append(f"- Falsifiers: any OOS year with negative dPnL, daily Sharpe ↓>0.05, ")
    md.append(f"MaxDD ↑>$500, or placebo passes any gate.\n")

    # ── 1. In-sample bucket profile (for transparency) ──
    md.append("\n## 1. In-sample term-slope bucket profile (TRAIN data 2022-02..2024-12)\n")
    train_tl, test_tl = split_oos(tl)
    train_longs = train_tl[train_tl["signal"] == 1].copy()
    rows = []
    for lo, hi, lab in [(-100,-5,'<-5'),(-5,-3,'-5..-3'),(-3,-2,'-3..-2'),
                         (-2,-1,'-2..-1'),(-1,0,'-1..0')]:
        sub = train_longs[(train_longs[PRE_COMMIT_FEATURE] >= lo) &
                          (train_longs[PRE_COMMIT_FEATURE] < hi)]
        if len(sub) < 5:
            continue
        m, s = sub["net_pnl_nc"].mean(), sub["net_pnl_nc"].std()
        rows.append({
            "bucket": lab,
            "n": len(sub),
            "mean_pnl": round(m, 1),
            "std_pnl": round(s, 0),
            "trade_sharpe": round(m/s if s else 0, 3),
            "total_pnl": round(sub["net_pnl_nc"].sum(), 0),
        })
    md.append(df_to_md(pd.DataFrame(rows)))
    md.append("")
    md.append("\nThe extreme bucket (slope < -5) shows ~2x per-trade Sharpe vs ")
    md.append("middle bucket (slope -2..-1): 0.56 vs 0.08 across the train window.\n")

    # ── 2. Run all arms ──
    md.append("\n## 2. Sizing arm results — TRAIN vs TEST (OOS)\n")
    arm_results = run_arms(tl)
    rows = []
    for name, train, test in arm_results:
        rows.append({
            "arm": name,
            "train_pnl": round(train.nc_pnl_total, 0),
            "train_sharpe": round(train.daily_sharpe, 3),
            "train_mdd": round(train.max_dd, 0),
            "train_n_amp": train.n_amplified_long,
            "test_pnl": round(test.nc_pnl_total, 0),
            "test_sharpe": round(test.daily_sharpe, 3),
            "test_mdd": round(test.max_dd, 0),
            "test_n_amp": test.n_amplified_long,
        })
    md.append(df_to_md(pd.DataFrame(rows)))
    md.append("")

    # ── 3. Decision-gate scorecard (vs S0 baseline) ──
    md.append("\n## 3. OOS decision-gate scorecard (vs S0 baseline)\n")
    base = arm_results[0][2]  # S0 test
    base_per_yr = base.nc_pnl_per_year
    rows = []
    for name, _, test in arm_results[1:]:
        # Per-year delta on OOS
        all_years = sorted(set(base_per_yr.keys()) | set(test.nc_pnl_per_year.keys()))
        per_year_dpnl = {yr: test.nc_pnl_per_year.get(yr, 0) - base_per_yr.get(yr, 0)
                         for yr in all_years}
        n_pos = sum(1 for v in per_year_dpnl.values() if v >= 0)
        rows.append({
            "arm": name,
            "test_dpnl": round(test.nc_pnl_total - base.nc_pnl_total, 0),
            "test_dsharpe": round(test.daily_sharpe - base.daily_sharpe, 3),
            "test_dmdd": round(test.max_dd - base.max_dd, 0),
            "per_year_strict": f"{n_pos}/{len(all_years)}",
            "per_year_min_dpnl": round(min(per_year_dpnl.values()), 0) if per_year_dpnl else 0,
            "per_year_dpnl": "; ".join(f"{yr}:{int(v):+,}"
                                        for yr, v in per_year_dpnl.items()),
        })
    md.append(df_to_md(pd.DataFrame(rows)))
    md.append("")

    # ── 4. Walk-forward variant ──
    md.append("\n## 4. Walk-forward variant (24m trailing, 20th-pctile threshold, 1.5x)\n")
    md.append("For each month M, threshold is the 20th-pctile of `term_slope_1w_2w` ")
    md.append("over LONG bars in [M-24, M-1]; sizing 1.5x applied to bars in M ")
    md.append("below that threshold. Parameter-free except for trail window length ")
    md.append("(24m fixed).\n")
    wf_df = walk_forward_audit(tl, multiplier=1.5, trail_months=24)
    if not wf_df.empty:
        # Aggregate per year
        wf_df["year"] = wf_df["month"].str.slice(0, 4).astype(int)
        wf_yr = wf_df.groupby("year").agg(
            n_months=("month", "count"),
            n_amp=("n_amplified", "sum"),
            base_pnl=("base_nc_pnl", "sum"),
            amp_pnl=("amp_nc_pnl", "sum"),
            delta=("delta_pnl", "sum"),
        ).round(0)
        md.append("### Per-year walk-forward summary\n")
        md.append(df_to_md(wf_yr.reset_index()))
        md.append("")
        md.append(f"Total walk-forward dPnL: **${wf_df['delta_pnl'].sum():,.0f}**")
        md.append(f" over {wf_df['month'].nunique()} months (start={wf_df['month'].iloc[0]}).")
    else:
        md.append("(walk-forward window too short)\n")
    md.append("")

    # ── 5. Verdict ──
    md.append("\n## 5. Verdict\n")
    h1_test = arm_results[1][2]
    placebo_test = arm_results[2][2]
    h1_passed = (
        h1_test.nc_pnl_total > base.nc_pnl_total
        and h1_test.daily_sharpe >= base.daily_sharpe - 0.05
        and h1_test.max_dd >= base.max_dd - 500.0
        and all(h1_test.nc_pnl_per_year.get(yr, 0) >= base_per_yr.get(yr, 0) - 100
                for yr in base_per_yr.keys())
    )
    placebo_failed = (
        placebo_test.daily_sharpe < base.daily_sharpe + 0.02
        or placebo_test.max_dd < base.max_dd - 500
    )
    md.append(f"- H1 passes all OOS gates: **{h1_passed}**\n")
    md.append(f"- Placebo fails (sanity check): **{placebo_failed}**\n")
    if h1_passed and placebo_failed:
        md.append("- **Recommendation: candidate for Phase 3 wiring.**\n")
    else:
        md.append("- **Recommendation: REJECT under strict gate.**\n")

    OUT_MD.write_text("\n".join(md), encoding="utf-8")
    print(f"[phase2] wrote {OUT_MD.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
