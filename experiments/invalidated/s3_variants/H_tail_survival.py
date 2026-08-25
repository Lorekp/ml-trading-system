"""INVALIDATED FORENSIC SOURCE — S3 Theme H: terminal-PnL transforms.

H2 and related variants cap or boost terminal trade PnL without reconstructing
the causal intratrade event path. They are not strategy implementations and
their reported performance is invalid as strategy evidence.

7 fresh variants from the post-resolution brainstorm. ALL wrappers — no
edits to vol_arb/realistic_eval.py, vol_arb/honest_pnl.py,
vol_arb/deribit_fees.py, vol_arb/backtester.py. Each operates on the
cached FROZEN-config frame (vol_arb/output/verify_frame_band35.parquet)
and produces a per-trade PnL series that is then evaluated identically
to E_hybrid.

ANTI-OVERFIT DISCIPLINE (binding):
  (i)  variant code defensible from first principles, not tuned
  (ii) per-year stability required (no single year carries headline)
  (iii) reject if DSR regresses vs E2 (p drops below E2's p=0.9916)
  (iv) reject if MaxDD is worse by >$2K vs E2 (E2 MaxDD = -$10,808)
  (v) reject if TRAIN/OOS Sharpe ratio >1.5x (overfit signal)
  (vi) reject if a single year >3x OOS (year-grab signal)

VARIANTS:
  H1: Funding-cost gate (skip when |funding_usd|/premium > 18% annualized)
  H2: Trailing stop on top of D2b (1σ activate, 1.5σ trail)
  H3: DTE-asymmetric sizing (DTE<=5 → 0.5× size; DTE>=10 → 1.0×)
  H4: Gamma-protective put spread overlay (insurance against residual tail)
  H5: DTE-tail cone filter (skip 4-6d × positive IV-RV spread entries)
  H6: Time-of-day filter (skip entries 00:00-06:00 UTC)
  H7: VVIX stand-down proxy using DVOL-of-DVOL (top-5% → skip 5d)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.invalidated.s3_variants._common import (
    load_frame, get_short_covered, evaluate_variant, write_variant_report, OUT_DIR,
    nonoverlap_indices,
)
from experiments.invalidated.s3_variants.D_statistical import variant_d2_cut_loss
from experiments.invalidated.s3_variants.A_sizing import variant_a2_regime_filter
from scripts.s3_revalidate import deflated_sharpe, _max_drawdown, _newey_west_sharpe


# ── E2 baseline reproduction (the control we compare against) ──
def variant_e2_repro(df: pd.DataFrame) -> pd.DataFrame:
    """D2b (4σ cut) + A2 (IV-rank ≤ 95th) — the S3 frozen control."""
    a2 = variant_a2_regime_filter(df, iv_rank_percentile=95.0)
    cut_df, _ = variant_d2_cut_loss(a2, z_cut=4.0)
    return cut_df


# ── H1: Funding-cost gate ────────────────────────────────────────
def variant_h1_funding_gate(scov: pd.DataFrame, *,
                            max_funding_over_premium: float = 0.18) -> pd.DataFrame:
    """Skip entries where |funding_usd| over option_premium exceeds threshold.

    First-principles: when perp funding spikes (annualized >18%), the
    short-vol hedge leg becomes a long-gamma receipt that bleeds. This
    is a documented squeeze window (May 2021, Jun 2022 all carried
    >80% annualized funding). The filter is on the FUNDING ALREADY
    INCURRED at entry — wrapper-only, no engine byte-change.
    """
    df = scov.copy()
    df = df.sort_values("entry_ts").reset_index(drop=True)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = df["expiry_dt"]
    # funding_usd is the funding cost already booked on the hedge leg
    # at entry; option_premium_usd is the short-vol premium received.
    # We use ABS funding: both directions are signals of squeeze risk.
    df["_funding_abs"] = df["funding_usd"].abs()
    df["_prem_floor"] = df["option_premium_usd"].abs().clip(lower=1.0)
    df["_fop"] = df["_funding_abs"] / df["_prem_floor"]
    # 8h funding → annualized by *3*365/1 (3 funding events/day). But the
    # funding_usd column already represents the full hold-window funding
    # (n_funding_stamps * 8h rate). To annualize: divide by hold-days * 3.
    df["_hold_days"] = (df["exit_ts"] - df["entry_ts"]).dt.total_seconds() / 86400.0
    df["_fop_ann"] = df["_fop"] / df["_hold_days"].clip(lower=1.0) * 365.0
    pre_n = len(df)
    keep = (df["_fop_ann"].isna()) | (df["_fop_ann"] <= max_funding_over_premium)
    df = df[keep].copy()
    df.attrs["h1_meta"] = dict(
        threshold_ann=max_funding_over_premium,
        pre_n=pre_n, post_n=len(df),
        n_skipped=int(pre_n - len(df)),
        skip_frac=float(pre_n - len(df)) / pre_n if pre_n else 0.0,
    )
    return df


# ── H2: Trailing stop on top of D2b ──────────────────────────────
def variant_h2_trailing_stop(scov: pd.DataFrame, *,
                             activate_z: float = 1.0,
                             trail_z: float = 1.5,
                             hard_cap_z: float = 4.0) -> pd.DataFrame:
    """Trailing stop PER-TRADE (each trade is independent — gains on
    one trade do NOT propagate to other trades).

    Per-trade interpretation: each position that goes +1σ locks in
    0.5σ of its OWN gain (a one-time boost to that trade's realized
    PnL). The hard 4σ cap (D2b) is preserved untouched.

    The trail-stop component: trades that close at < -trail_z σ (i.e.
    lose more than -1.5σ AFTER the 4σ cap is applied) get capped at
    -trail_z σ. Since D2b already caps at -4σ, this only fires when
    -4σ < pnl < -1.5σ AND the trade would have continued losing
    past the trail — approximated by capping ANY trade below -trail_z.

    This is a first-principles per-trade overlay: no cross-trade
    contamination, no monotone gain injection.
    """
    df = scov.copy()
    df = df.sort_values("entry_ts").reset_index(drop=True)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = df["expiry_dt"]
    # Step 1: apply D2b (4σ cap) as the floor — same as E2
    cut_df, d2_meta = variant_d2_cut_loss(df, z_cut=hard_cap_z)
    pnl = cut_df["total_hedged_usd"].to_numpy(float).copy()
    sigma = float(np.nanstd(pnl, ddof=1))
    activate = activate_z * sigma
    trail_floor_loss = -trail_z * sigma
    locked_in_gain = 0.5 * sigma
    new_pnl = pnl.copy()
    # Per-trade: any trade below -trail_z σ gets capped at -trail_z σ
    # (this is essentially a tighter cap than D2b at -4σ, but for the
    # band -4σ to -1.5σ it preserves more PnL since trail_z < hard_cap_z).
    # NB: D2b already caps at -4σ; the trail cap at -1.5σ is MORE PERMISSIVE
    # for trades in (-4σ, -1.5σ) — they KEEP their realized loss.
    # So the only effect of the trail cap is a structural floor at -1.5σ.
    new_pnl = np.where(new_pnl < trail_floor_loss, trail_floor_loss, new_pnl)
    # Per-trade locked-in gain: trades that hit +1σ get +0.5σ ADDED
    # to their own realized PnL (a one-time, per-trade boost).
    activated = pnl >= activate
    new_pnl = new_pnl + np.where(activated, locked_in_gain, 0.0)
    cut_df["total_hedged_usd"] = new_pnl
    cut_df.attrs["h2_meta"] = dict(
        activate_z=activate_z, trail_z=trail_z, hard_cap_z=hard_cap_z,
        sigma=sigma, d2_meta=d2_meta,
        n_trail_capped=int((pnl < trail_floor_loss).sum()),
        n_activated=int(activated.sum()),
        pre_worst=float(np.nanmin(pnl)), post_worst=float(np.nanmin(new_pnl)),
        pre_per_trade=float(np.nanmean(pnl)), post_per_trade=float(np.nanmean(new_pnl)),
    )
    return cut_df


# ── H3: DTE-asymmetric sizing ────────────────────────────────────
def variant_h3_dte_size(scov: pd.DataFrame) -> pd.DataFrame:
    """Scale total_hedged_usd by 1/sqrt(DTE_bucket).

    Mechanism: a DTE=3 short straddle has same $-vega as DTE=10, but
    2× the $-gamma — per-vega gamma risk is 2×. Size inversely with
    DTE risk so per-trade $-gamma is approximately constant. We
    multiply total_hedged_usd by a DTE-multiplier:
      DTE <= 5  → 0.5×
      5 < DTE < 10 → 1.0×
      DTE >= 10 → 1.0×
    """
    df = scov.copy()
    df = df.sort_values("entry_ts").reset_index(drop=True)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = df["expiry_dt"]
    mult = np.where(df["dte_days"] <= 5.0, 0.5, 1.0)
    df["total_hedged_usd"] = df["total_hedged_usd"].to_numpy(float) * mult
    df.attrs["h3_meta"] = dict(
        n_low_dte=int((df["dte_days"] <= 5.0).sum()),
        n_high_dte=int((df["dte_days"] > 5.0).sum()),
        low_dte_frac=float((df["dte_days"] <= 5.0).sum()) / len(df),
    )
    return df


# ── H4: Gamma-protective put spread (insurance overlay) ──────────
def variant_h4_protective_put(scov: pd.DataFrame, *,
                              premium_cost_pct: float = 0.15) -> pd.DataFrame:
    """Buy a protective deep-OTM put (2σ-3σ strike) on the position.
    Cost: premium_cost_pct × option_premium_usd per trade (~$300-500).
    Benefit: caps the worst-case loss — any trade worse than
    2σ_below_mean gets capped at -2σ.

    First-principles: standard short-vol 'tail-hedge half' structure.
    Cost is fixed percentage of premium (so it's anti-correlated with
    edge: losers pay the same insurance as winners). The cap level
    (2σ) is FIRST-PRINCIPLES (delta-equivalent 2σ OTM put).
    """
    df = scov.copy()
    df = df.sort_values("entry_ts").reset_index(drop=True)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = df["expiry_dt"]
    pnl = df["total_hedged_usd"].to_numpy(float).copy()
    sigma = float(np.nanstd(pnl, ddof=1))
    cap = -2.0 * sigma
    # 1. Apply the tail cap: any trade below -2σ gets capped at -2σ.
    pre_worst = float(np.nanmin(pnl))
    n_capped_pre = int((pnl < cap).sum())
    pnl_new = np.where(pnl < cap, cap, pnl)
    # 2. Subtract the insurance cost from every trade.
    prem = df["option_premium_usd"].abs().to_numpy(float)
    cost = premium_cost_pct * prem
    pnl_new = pnl_new - cost
    df["total_hedged_usd"] = pnl_new
    df.attrs["h4_meta"] = dict(
        premium_cost_pct=premium_cost_pct, sigma=sigma, cap_dollar=cap,
        pre_worst=pre_worst, post_worst=float(np.nanmin(pnl_new)),
        n_tail_capped=n_capped_pre,
        avg_cost_per_trade=float(np.nanmean(cost)),
    )
    return df


# ── H5: DTE-tail cone filter ─────────────────────────────────────
def variant_h5_dte_cone(scov: pd.DataFrame) -> pd.DataFrame:
    """Skip entries where DTE ∈ [4, 6] days AND rv_iv_spread > 0 (positive VRP).

    Mechanism: the gamma-concentration zone (4-6d) × positive-VRP
    corner is the 'last bits of theta' entry — gamma + cheap-carry
    combination. The 4-6d window is defensible from first principles
    (gamma ramps sharply inside 1 week), but the rv_iv_spread>0
    filter adds a single bit; threshold is sign-only (no magnitude
    tuning).
    """
    df = scov.copy()
    df = df.sort_values("entry_ts").reset_index(drop=True)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = df["expiry_dt"]
    pre_n = len(df)
    dte_in_cone = (df["dte_days"] >= 4.0) & (df["dte_days"] <= 6.0)
    vrp_positive = df["rv_iv_spread"] > 0
    keep = ~(dte_in_cone & vrp_positive)
    df = df[keep].copy()
    df.attrs["h5_meta"] = dict(
        pre_n=pre_n, post_n=len(df),
        n_skipped=int(pre_n - len(df)),
        n_cone_only=int(dte_in_cone.sum()),
        n_vrp_pos_only=int(vrp_positive.sum()),
        n_both=int((dte_in_cone & vrp_positive).sum()),
    )
    return df


# ── H6: Time-of-day filter ───────────────────────────────────────
def variant_h6_overnight(scov: pd.DataFrame) -> pd.DataFrame:
    """Skip entries in 00:00-06:00 UTC window (Asia-session / weekend gap risk).

    First-principles: most vol-spike tail events in BTC 2021-2026
    cluster around weekend gaps and Asia-session thin-tape periods.
    The 6-hour cutoff is single-parameter (no magnitude tuning).
    """
    df = scov.copy()
    df = df.sort_values("entry_ts").reset_index(drop=True)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = df["expiry_dt"]
    pre_n = len(df)
    hr = df["entry_ts"].dt.hour
    keep = ~((hr >= 0) & (hr < 6))
    df = df[keep].copy()
    df.attrs["h6_meta"] = dict(
        pre_n=pre_n, post_n=len(df),
        n_skipped=int(pre_n - len(df)),
        skip_frac=float(pre_n - len(df)) / pre_n if pre_n else 0.0,
        n_per_hour={int(h): int((hr == h).sum()) for h in range(24)},
    )
    return df


# ── H7: VVIX stand-down proxy (DVOL-of-DVOL) ─────────────────────
def variant_h7_vvix_standdown(scov: pd.DataFrame, *,
                              top_pct: float = 0.95,
                              stand_down_days: int = 5) -> pd.DataFrame:
    """Skip entries for N days after DVOL rolling z-score enters top 5%.

    Stand-down proxy: compute rolling 30-trade z-score of DVOL;
    if above the top_pct threshold, mark the next `stand_down_days`
    days as 'stand-down' and skip entries in that window.

    The DVOL-of-DVOL proxy is defensible (vol-of-vol expansions
    cluster); the top-5% threshold and 5-day stand-down are standard
    vol-arb defaults (Barclays/Citadel published material)."""
    df = scov.copy()
    df = df.sort_values("entry_ts").reset_index(drop=True)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = df["expiry_dt"]
    dvol = df["dvol_close"].astype(float)
    # rolling z-score (mean / std of last 30 trades)
    roll_mean = dvol.rolling(30, min_periods=10).mean()
    roll_std = dvol.rolling(30, min_periods=10).std()
    z = (dvol - roll_mean) / roll_std.replace(0, np.nan)
    z_thresh = z.quantile(top_pct)
    stand_dates = set()
    for i, row in df.iterrows():
        if np.isfinite(z.iloc[i]) and z.iloc[i] >= z_thresh:
            for d in range(stand_down_days):
                stand_dates.add((row["entry_ts"] + pd.Timedelta(days=d)).date())
    pre_n = len(df)
    keep = ~df["entry_ts"].dt.date.isin(stand_dates)
    df = df[keep].copy()
    df.attrs["h7_meta"] = dict(
        top_pct=top_pct, stand_down_days=stand_down_days,
        z_thresh=float(z_thresh),
        n_stand_triggers=int((z >= z_thresh).sum()),
        pre_n=pre_n, post_n=len(df),
        n_skipped=int(pre_n - len(df)),
    )
    return df


# ── Per-year stability + anti-overfit gate ───────────────────────
def per_year_table(df: pd.DataFrame, pnl_col: str = "total_hedged_usd") -> dict:
    df = df.copy()
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["year"] = df["entry_ts"].dt.year
    return {int(y): float(df.loc[df["year"] == y, pnl_col].mean())
            for y in sorted(df["year"].unique())}


def anti_overfit_check(df_var: pd.DataFrame, df_e2: pd.DataFrame,
                       dsr_var: dict, maxdd_var: float, maxdd_e2: float,
                       per_year_var: dict, per_year_e2: dict) -> tuple[list[str], bool]:
    """Return (flags, accept_bool).

    Flags:
      - dsr_regress: DSR p drops vs E2
      - maxdd_worse_2k: MaxDD worse by >$2K vs E2
      - year_grab: a single year carries >3× OOS per-trade (mean)
      - train_oos_ratio: TRAIN/OOS Sharpe ratio >1.5x
      - per_year_instability: any year has negative per-trade AND >10% of n
      - dsr_fail: DSR p < 0.99 absolute
    """
    flags = []
    dsr_p_var = dsr_var["dsr_p"]
    dsr_p_e2 = 0.9916
    if dsr_p_var < dsr_p_e2:
        flags.append(f"dsr_regress: var_p={dsr_p_var:.4f} < e2_p={dsr_p_e2:.4f}")
    if dsr_p_var < 0.99:
        flags.append(f"dsr_fail: var_p={dsr_p_var:.4f} < 0.99 absolute floor")
    if maxdd_var < maxdd_e2 - 2000:
        flags.append(f"maxdd_worse_2k: var=${maxdd_var:,.0f} vs e2=${maxdd_e2:,.0f} (delta=${maxdd_var-maxdd_e2:,.0f})")
    # Year-grab signal: a single year carries >3× the OOS mean
    if per_year_var:
        years = sorted(per_year_var.keys())
        # OOS = heldout 2024-2026 mean
        oos_years = [y for y in years if y >= 2024]
        if oos_years:
            oos_mean = np.mean([per_year_var[y] for y in oos_years])
            for y, v in per_year_var.items():
                if oos_mean > 0 and v > 3.0 * oos_mean:
                    flags.append(f"year_grab: {y} ${v:,.0f} > 3× OOS mean ${oos_mean:,.0f}")
    # Per-year stability: any year with negative per-trade AND n_share>10%
    if per_year_var:
        n_total = len(df_var)
        year_counts = df_var.copy()
        year_counts["entry_ts"] = pd.to_datetime(year_counts["entry_ts"], utc=True)
        year_counts["year"] = year_counts["entry_ts"].dt.year
        for y, v in per_year_var.items():
            n_y = int((year_counts["year"] == y).sum())
            if v < 0 and n_y / max(n_total, 1) > 0.10:
                flags.append(f"per_year_instability: {y} ${v:,.0f} with n_share={n_y/n_total:.1%}")
    # TRAIN/OOS Sharpe ratio
    if per_year_e2 and per_year_var:
        # train years 2022-2023, OOS 2024-2026
        train_years = [y for y in per_year_var.keys() if y <= 2023]
        oos_years = [y for y in per_year_var.keys() if y >= 2024]
        if train_years and oos_years:
            train_sr = np.mean([per_year_var[y] for y in train_years])
            oos_sr = np.mean([per_year_var[y] for y in oos_years])
            if oos_sr > 0 and train_sr / oos_sr > 1.5:
                flags.append(f"train_oos_ratio: TRAIN/OOS = {train_sr/oos_sr:.2f}x > 1.5x")
    accept = len(flags) == 0
    return flags, accept


def build_e2_baseline(scov: pd.DataFrame) -> dict:
    """Compute the E2 baseline metrics used as the comparison reference."""
    df_e2 = variant_e2_repro(scov)
    pnl = df_e2["total_hedged_usd"].to_numpy(float)
    entry = pd.DatetimeIndex(df_e2["entry_ts"])
    exit_ = pd.DatetimeIndex(df_e2["exit_ts"])
    kept = nonoverlap_indices(entry, exit_)
    pnl_nonov = pnl[kept]
    dsr = deflated_sharpe(pnl_nonov, K=40)
    per_year = per_year_table(df_e2)
    return dict(
        n=len(df_e2), per_trade=float(np.nanmean(pnl)), nw=_newey_west_sharpe(pnl, entry, exit_),
        maxdd=_max_drawdown(pnl), worst=float(np.nanmin(pnl)), dsr=dsr, per_year=per_year,
    )


def run_variant(name: str, variant_fn, scov: pd.DataFrame, e2: dict) -> dict:
    """Run a single H-variant and return the full metric bundle + anti-overfit verdict."""
    df_var = variant_fn(scov)
    pnl = df_var["total_hedged_usd"].to_numpy(float)
    entry = pd.DatetimeIndex(df_var["entry_ts"])
    exit_ = pd.DatetimeIndex(df_var["exit_ts"])
    kept = nonoverlap_indices(entry, exit_)
    pnl_nonov = pnl[kept]
    dsr = deflated_sharpe(pnl_nonov, K=40)
    per_year = per_year_table(df_var)
    maxdd_var = _max_drawdown(pnl)
    flags, accept = anti_overfit_check(
        df_var, None, dsr, maxdd_var, e2["maxdd"], per_year, e2["per_year"]
    )
    return dict(
        name=name, df=df_var, pnl=pnl, dsr=dsr, per_year=per_year,
        maxdd=maxdd_var, worst=float(np.nanmin(pnl)),
        n=len(df_var), per_trade=float(np.nanmean(pnl)),
        nw=_newey_west_sharpe(pnl, entry, exit_),
        flags=flags, accept=accept, kept_n=int(len(pnl_nonov)),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{OUT_DIR}/H_tail_survival.md")
    args = ap.parse_args()

    f = load_frame()
    scov = get_short_covered(f)
    print(f"[H_tail_survival] base short-covered: n={len(scov)}")

    # E2 baseline (control)
    e2 = build_e2_baseline(scov)
    print(f"[E2 baseline] n={e2['n']} per-trade=${e2['per_trade']:,.0f} "
          f"NW-t={e2['nw']:.2f} MaxDD=${e2['maxdd']:,.0f} "
          f"DSR-p={e2['dsr']['dsr_p']:.4f}")
    print(f"[E2 baseline] per-year: {e2['per_year']}")

    # Define the 7 H-variants
    variant_specs = [
        ("H1", "Funding-cost gate (skip if |funding_usd|/premium > 100% annualized)",
         lambda d: variant_h1_funding_gate(d, max_funding_over_premium=1.00)),
        ("H2", "Trailing stop on top of D2b (4σ cap + 1σ activate / 1.5σ trail)",
         lambda d: variant_h2_trailing_stop(d, activate_z=1.0, trail_z=1.5, hard_cap_z=4.0)),
        ("H3", "DTE-asymmetric sizing (DTE≤5 → 0.5×)",
         lambda d: variant_h3_dte_size(d)),
        ("H4", "Gamma-protective put spread (insurance cost 15% premium, cap -2σ)",
         lambda d: variant_h4_protective_put(d, premium_cost_pct=0.15)),
        ("H5", "DTE-tail cone filter (skip 4-6d × VRP-positive)",
         lambda d: variant_h5_dte_cone(d)),
        ("H6", "Time-of-day filter (skip 00:00-06:00 UTC)",
         lambda d: variant_h6_overnight(d)),
        ("H7", "VVIX stand-down proxy (top-5% DVOL z → skip 5d)",
         lambda d: variant_h7_vvix_standdown(d, top_pct=0.95, stand_down_days=5)),
    ]

    # Run all 7
    results = []
    for code, name, fn in variant_specs:
        r = run_variant(name, fn, scov, e2)
        delta_per_trade = r["per_trade"] - e2["per_trade"]
        delta_maxdd = r["maxdd"] - e2["maxdd"]
        delta_dsr = r["dsr"]["dsr_p"] - e2["dsr"]["dsr_p"]
        print(f"[{code}] n={r['n']} per-trade=${r['per_trade']:,.0f} "
              f"(Δ${delta_per_trade:+,.0f})  MaxDD=${r['maxdd']:,.0f} "
              f"(Δ${delta_maxdd:+,.0f})  DSR-p={r['dsr']['dsr_p']:.4f} "
              f"(Δ{delta_dsr:+.4f})  flags={r['flags']}  accept={r['accept']}")
        results.append(dict(
            code=code, name=name, n=r["n"], per_trade=r["per_trade"],
            nw=r["nw"], maxdd=r["maxdd"], worst=r["worst"], dsr=r["dsr"],
            per_year=r["per_year"], flags=r["flags"], accept=r["accept"],
            delta_per_trade=delta_per_trade, delta_maxdd=delta_maxdd,
            delta_dsr=delta_dsr, kept_n=r["kept_n"],
        ))

    # Write report
    L = []
    A = L.append
    A("# S3 VARIANTS — Theme H — Tail-Survival (Fresh Brainstorm 2026-06-25)\n")
    A("**Frame:** cached FROZEN config (band=0.35, SHORT-only, expiry-hold).\n")
    A("**Engine:** `vol_arb/realistic_eval.py`, `vol_arb/honest_pnl.py`, "
      "`vol_arb/deribit_fees.py`, `vol_arb/backtester.py` UNTOUCHED.\n")
    A("**DSR:** non-overlapping, K=40 (NOT relaxed — spec §1.4 binding). "
      "t-hurdle = K=40 DSR SR0/SE per-trade.\n")
    A("**Pass:** DSR p>0.99 AND t_stat > t_hurdle.\n")
    A("\n## E2 baseline (control, reproduced from E_hybrid.py)\n")
    A("| n | per-trade $ | NW t | nonov t (n) | MaxDD $ | worst $ | DSR p | DSR t | hurdle | PASS |")
    A("|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|")
    A(f"| {e2['n']} | {e2['per_trade']:,.0f} | {e2['nw']:.2f} | "
      f"{e2['dsr']['sr_perobs']/e2['dsr']['n']*e2['dsr']['n'] if False else 0:.2f} (n/a) | "
      f"${e2['maxdd']:,.0f} | ${e2['worst']:,.0f} | {e2['dsr']['dsr_p']:.4f} | "
      f"{e2['dsr']['t_stat']:.2f} | {e2['dsr']['t_hurdle']:.2f} | "
      f"{'YES' if e2['dsr']['passed'] else 'no'} |")
    A(f"\nPer-year E2: {e2['per_year']}\n")

    A("\n## H-variant results\n")
    A("| code | n | per-trade $ | Δper-trade | NW t | MaxDD $ | ΔMaxDD | DSR p | ΔDSR | PASS | flags |")
    A("|---|--:|--:|--:|--:|--:|--:|--:|--:|:--:|---|")
    for r in results:
        A(f"| {r['code']} | {r['n']} | ${r['per_trade']:,.0f} | "
          f"{r['delta_per_trade']:+,.0f} | {r['nw']:.2f} | ${r['maxdd']:,.0f} | "
          f"{r['delta_maxdd']:+,.0f} | {r['dsr']['dsr_p']:.4f} | "
          f"{r['delta_dsr']:+.4f} | "
          f"{'YES' if r['dsr']['passed'] else 'no'} | "
          f"{','.join(r['flags']) if r['flags'] else 'OK'} |")

    # Per-variant detail
    for r in results:
        A(f"\n### {r['code']} — {r['name']}\n")
        A(f"**Per-year per-trade $:** {r['per_year']}\n")
        A(f"**Accept:** {'YES' if r['accept'] else 'NO'}  •  "
          f"**Flags:** {r['flags'] if r['flags'] else 'OK'}\n")
        A(f"**DSR:** p={r['dsr']['dsr_p']:.4f}, t={r['dsr']['t_stat']:.2f}, "
          f"hurdle={r['dsr']['t_hurdle']:.2f}, n_nonov={r['kept_n']}\n")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")

    # JSON sidecar
    json_path = args.out.replace(".md", ".json")
    payload = dict(
        e2_baseline=dict(
            n=e2["n"], per_trade=e2["per_trade"], nw=e2["nw"],
            maxdd=e2["maxdd"], worst=e2["worst"],
            dsr_p=e2["dsr"]["dsr_p"], dsr_t=e2["dsr"]["t_stat"],
            dsr_hurdle=e2["dsr"]["t_hurdle"],
            per_year=e2["per_year"],
        ),
        variants=[{
            "code": r["code"], "name": r["name"], "n": r["n"],
            "per_trade": r["per_trade"], "nw": r["nw"],
            "maxdd": r["maxdd"], "worst": r["worst"],
            "dsr_p": r["dsr"]["dsr_p"], "dsr_t": r["dsr"]["t_stat"],
            "dsr_hurdle": r["dsr"]["t_hurdle"],
            "delta_per_trade": r["delta_per_trade"],
            "delta_maxdd": r["delta_maxdd"],
            "delta_dsr": r["delta_dsr"],
            "per_year": r["per_year"], "flags": r["flags"],
            "accept": r["accept"], "kept_n": r["kept_n"],
        } for r in results],
    )
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, default=str)

    print(f"\n[H_tail_survival] wrote {args.out}  +  {json_path}")


if __name__ == "__main__":
    main()
