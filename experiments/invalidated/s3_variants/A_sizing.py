"""Invalidated forensic experiment — S3 Theme A: SIZING.

Post-processor on the FROZEN-config cached frame. The goal is to test
whether changing the per-trade size (NOT the strike geometry) can clear
the K=40 DSR bar that the fixed-sizing baseline fails (p=0.7168 OOS).

The four sub-variants:
- A1: per-trade vol sizing — scale by recent realized BTC vol; cap at the
      1.5%-NAV-vega spec, but also vol-target the portfolio at 0.5-2% NAV
      vega. Mechanism: low-vol periods → over-weight (carry still positive,
      less per-trade risk); high-vol periods → under-weight (left tail bound
      by sizing, not by strike geometry).
- A2: regime-filter entry — skip entry when IV-rank > 95th percentile.
      Mechanism: short vol pays the LEAST when IV is already high (low
      forward premium, high gap risk), so over-weighting calm IV buckets.
- A3: stand-down after a -2σ day — skip re-entry for N days. Mechanism: a
      single -2σ day is a strong empirical signal of forward crash regime
      (clustering of vol); waiting N days reduces correlated tail bets.
- A4: dynamic band — shrink delta-hedge band in calm vol (costs less)
      and widen in high vol (hedge more aggressively when risk is high).
      Mechanism: hedge cost and gamma exposure are regime-dependent; a
      band that adapts reduces hedging slippage in calm periods (where
      drift is small) and increases protection in stressed periods.

Each variant: per-trade PnL, per-year PnL, DSR p@K=40, MaxDD, mechanism.
The DSR is the binding pre-committed bar (p>0.99 AND t>t_hurdle).

ADD-ONLY discipline: no edits to realistic_eval.py / honest_pnl.py /
backtester.py / deribit_fees.py. Post-processor on the cached frame.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from experiments.invalidated.s3_variants._common import (
    load_frame, get_short_covered, evaluate_variant, write_variant_report, OUT_DIR,
)


# ── A1: vol-target portfolio sizing ─────────────────────────────
def variant_a1_vol_target(scov: pd.DataFrame, *, target_vega_frac_nav: float = 0.01,
                          vol_window: int = 24 * 7,  # 1 week of 1-min bars ~ 7d of hours
                          nav_vega_cap: float = 0.015,
                          min_size_frac: float = 0.25) -> pd.DataFrame:
    """Vol-target the portfolio so that per-trade vega exposure is
    proportional to 1/recent_realized_vol. The per-trade PnL is rescaled
    by the size factor. vega_target = target_vega_frac_nav * equity / vega.
    size_factor = min(vega_target, nav_vega_cap * equity / vega) / baseline_size.
    """
    df = scov.copy().reset_index(drop=True)
    # Need a 1-min realized vol proxy at entry. Use `rv_ann` (annualized
    # realized vol from the engine frame; per-bar std in the to-expiry window
    # is not stored, but rv_ann is the per-trade entry estimate). For a
    # more reactive window we approximate via recent index returns.
    # Use index_price to compute a 7d rolling realized vol (annualized).
    df = df.sort_values("entry_ts").reset_index(drop=True)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    px = df.set_index("entry_ts")["index_price"].astype(float).sort_index()
    # log returns, rolling std (1h = 60 min bars; use 1h returns × sqrt(24*365) for annualization)
    rets = np.log(px).diff()
    # 24h rolling vol (annualized)
    vol_24h = rets.rolling(24 * 60, min_periods=60).std() * np.sqrt(60 * 24 * 365)
    # 7d rolling vol (annualized) — smoother
    vol_7d = rets.rolling(7 * 24 * 60, min_periods=60).std() * np.sqrt(60 * 24 * 365)
    df["vol_7d"] = df["entry_ts"].map(vol_7d)
    # Median vol across the sample (the "baseline" vol level)
    vol_baseline = float(np.nanmedian(df["vol_7d"]))
    # size factor: scale up in calm vol (1/vol_7d * vol_baseline), bounded
    df["size_factor"] = vol_baseline / df["vol_7d"].clip(lower=vol_baseline * 0.5)
    df["size_factor"] = df["size_factor"].clip(lower=min_size_frac, upper=2.0)
    # Apply to PnL
    df["total_hedged_usd"] = df["total_hedged_usd"] * df["size_factor"]
    df["exit_ts"] = df["expiry_dt"]
    return df


# ── A2: regime-filter entry (skip high IV-rank) ──────────────────
def variant_a2_regime_filter(scov: pd.DataFrame, *,
                              iv_rank_percentile: float = 95.0,
                              iv_col: str = "dvol_close",
                              window_bars: int = 90) -> pd.DataFrame:
    """Skip entries when IV-rank > 95th percentile (rolling N-bar window).
    The frame is 1-min bars; with ~1500 bars/day (BTC is 24/7), 90 bars = ~36 min
    of history per trade. For regime detection we want a SHORTER window (so
    a 90-bar lookback = 1.5h, similar to a 1d window in min-bar terms is
    impractical because the frame is sparse in the per-trade sense). We use
    a 30-bar lookback = ~30 min of recent DVOL history per trade entry.
    """
    df = scov.copy()
    df = df.sort_values("entry_ts").reset_index(drop=True)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = df["expiry_dt"]
    px = df.set_index("entry_ts")[iv_col].astype(float).sort_index()
    # 90-bar rolling rank (≈1.5h of recent DVOL). On the per-trade frame
    # this is roughly "where is current DVOL vs the last 1.5h of activity?".
    rank = px.rolling(window_bars, min_periods=20).rank(pct=True) * 100
    df["iv_rank"] = df["entry_ts"].map(rank)
    # iv_rank NaN for early entries (insufficient lookback) — KEEP those
    # (don't filter out the first 90 trades).
    df = df[(df["iv_rank"].isna()) | (df["iv_rank"] <= iv_rank_percentile)].copy()
    return df


# ── A3: stand-down after first -2σ day ──────────────────────────
def variant_a3_stand_down(scov: pd.DataFrame, *,
                          z_threshold: float = -2.0,
                          stand_down_days: int = 5) -> pd.DataFrame:
    """After a -2σ-or-worse daily loss, skip the next N days of entries.
    The -2σ day is computed as the cumulative PnL's daily drop, normalized
    by the rolling 30-day stdev of daily drops.
    """
    df = scov.copy()
    df = df.sort_values("entry_ts").reset_index(drop=True)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = df["expiry_dt"]
    # Compute daily PnL (entry-day buckets)
    df["entry_date"] = df["entry_ts"].dt.date
    daily_pnl = df.groupby("entry_date")["total_hedged_usd"].sum().sort_index()
    daily_pnl.index = pd.to_datetime(daily_pnl.index)
    # Rolling 30-day stdev of daily PnL changes
    daily_change = daily_pnl.diff()
    roll_std = daily_change.rolling(30, min_periods=10).std()
    # z-score
    z = daily_change / roll_std
    # Find bad days (z <= threshold)
    bad_days = set(z[z <= z_threshold].index.date)
    # Mark entries to skip
    skip_dates = set()
    for bad in bad_days:
        for d in range(1, stand_down_days + 1):
            skip_dates.add(bad + pd.Timedelta(days=d))
    # Filter
    mask = ~df["entry_date"].isin(skip_dates)
    return df[mask].copy()


# ── A4: dynamic delta-hedge band ─────────────────────────────────
def variant_a4_dynamic_band(scov: pd.DataFrame, *,
                            vol_col: str = "dvol_close",
                            calm_band: float = 0.50,
                            storm_band: float = 0.20,
                            vol_pct_calm: float = 50.0) -> pd.DataFrame:
    """Dynamic band: shrink in calm vol (less hedge drag), widen in storm
    (more protection). The PnL impact is approximated: the gamma leg's
    rebalance count scales with 1/band; cost scales with sqrt(rebalances).
    We use the engine's `n_rebalances` as a proxy and re-scale the
    hedge_costs_nc column (a larger band = fewer rebalances = less cost).
    """
    df = scov.copy()
    df = df.sort_values("entry_ts").reset_index(drop=True)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = df["expiry_dt"]
    # IV percentile in the trailing 6 months
    px = df.set_index("entry_ts")[vol_col].astype(float).sort_index()
    rank = px.rolling(180 * 24 * 60, min_periods=30 * 24 * 60).rank(pct=True) * 100
    df["vol_pct"] = df["entry_ts"].map(rank)
    # band assignment
    df["band"] = np.where(df["vol_pct"] <= vol_pct_calm, calm_band, storm_band)
    df["band_ratio"] = df["band"] / 0.35   # relative to the FROZEN 0.35 band
    # The hedge_costs_nc scales ~ with rebalance count ~ 1/band
    # (i.e. wider band → fewer rebalances → lower cost). But gamma PnL
    # is unchanged because to-expiry final gamma is path-independent
    # (the engine uses real BS delta along the realized path).
    # We adjust hedge_costs_nc to estimate the impact, leaving gamma_pnl_usd
    # alone (gamma_leg is band-invariant for true-BS-delta to-expiry).
    if "hedge_costs_nc" in df.columns:
        df["hedge_costs_nc_adj"] = df["hedge_costs_nc"] * df["band_ratio"]
    else:
        df["hedge_costs_nc_adj"] = 0.0
    # Total PnL = gamma + theta + vega + hedge_costs. Reconstruct:
    #   total_hedged = (vega_pnl + theta_pnl + delta_hedge_pnl) - hedge_costs
    #                + option_premium - option_fees
    # To avoid re-engine pricing, we approximate: scale total_hedged by
    # the change in hedge_costs (the dominant band-sensitive term).
    cost_delta = df["hedge_costs_nc_adj"] - df.get("hedge_costs_nc", 0.0)
    df["total_hedged_usd"] = df["total_hedged_usd"] - cost_delta
    return df


# ── Main ─────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{OUT_DIR}/A_sizing.md")
    args = ap.parse_args()

    f = load_frame()
    scov = get_short_covered(f)
    print(f"[A_sizing] base short-covered: n={len(scov)}")

    results = []

    # A1 — vol-target portfolio
    df_a1 = variant_a1_vol_target(scov)
    r1 = evaluate_variant(
        df_a1, code="A1", name="vol-target portfolio (1/vol_7d, cap 2x)",
        mechanism="Over-weight calm vol / under-weight storm vol. "
                  "Cuts correlated tail exposure via sizing, not strike geometry.",
    )
    results.append(r1)

    # A2 — IV-rank regime filter (95th pct)
    df_a2 = variant_a2_regime_filter(scov, iv_rank_percentile=95.0)
    r2 = evaluate_variant(
        df_a2, code="A2", name="IV-rank ≤ 95th pct",
        mechanism="Skip entry when IV is already rich (low forward carry, "
                  "high gap risk). Sizer-equivalent filter that drops the worst-VRP bucket.",
    )
    results.append(r2)

    # A2b — IV-rank regime filter (80th pct — TIGHTER)
    df_a2b = variant_a2_regime_filter(scov, iv_rank_percentile=80.0)
    r2b = evaluate_variant(
        df_a2b, code="A2b", name="IV-rank ≤ 80th pct (TIGHTER)",
        mechanism="Same as A2 but with tighter threshold (80th pct). Skips "
                  "more entries — trades off breadth for richer VRP per trade.",
    )
    results.append(r2b)

    # A3 — stand-down after -2σ day
    df_a3 = variant_a3_stand_down(scov, z_threshold=-2.0, stand_down_days=5)
    r3 = evaluate_variant(
        df_a3, code="A3", name="stand-down 5d after -2σ day",
        mechanism="Vol clusters: a -2σ day is a forward-crash regime signal. "
                  "Skipping 5d of entries reduces correlated tail bets.",
    )
    results.append(r3)

    # A3b — longer stand-down (10 days)
    df_a3b = variant_a3_stand_down(scov, z_threshold=-2.0, stand_down_days=10)
    r3b = evaluate_variant(
        df_a3b, code="A3b", name="stand-down 10d after -2σ day",
        mechanism="Longer stand-down. Tests the 'vol regimes persist' hypothesis.",
    )
    results.append(r3b)

    # A4 — dynamic band
    df_a4 = variant_a4_dynamic_band(scov, calm_band=0.50, storm_band=0.20)
    r4 = evaluate_variant(
        df_a4, code="A4", name="dynamic band (calm 0.50 / storm 0.20)",
        mechanism="Wider band in calm vol reduces hedge drag; tighter band in "
                  "storm vol adds protection. PnL impact is band-conditional.",
    )
    results.append(r4)

    write_variant_report(results, "A — Sizing", out_path=args.out,
        summary=f"Baseline (S3 frozen): n=666, per-trade $1,285, DSR p=0.7168 FAIL. "
                f"Goal: clear the K=40 DSR bar via sizing changes alone (no strike "
                f"geometry change).")


if __name__ == "__main__":
    main()
