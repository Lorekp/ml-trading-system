#!/usr/bin/env python
"""Experiment B: Funding Regime as Signal FILTER (not feature).

KEY DISTINCTION: Previous tests added funding as a LogReg feature (killed:
Sharpe 2.573 → 2.281). This uses funding to SUPPRESS signals AFTER the
model makes its prediction. Different mechanism entirely.

Hypothesis: When funding is extremely positive for a sustained period
(crowded longs), the model's LONG signals are less reliable because
the crowded position creates reversal risk. Suppress signals that
agree with the crowded direction.

Variants:
  1. Baseline  — production config (no funding filter)
  2. Funding suppress — suppress crowded-direction signals when
     |funding_zscore| > threshold for > N consecutive settlements
  3. Funding sizing — reduce sizing_mult (not suppress) when crowded

Uses cached raw signals from adaptive_conf_cpcv.py (data/cache/).
Loads Binance funding rates from data/funding-rates/.

Success criteria: Sharpe >= baseline + 0.05, DD improvement
Kill criteria: Sharpe < baseline - 0.05 OR trade count drops > 30%

CPCV validation included (6-fold) to check robustness.

Usage:
    PYTHONPATH=src python experiments/funding_regime_filter_experiment.py [--smoke]
"""
import sys, os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import wilcoxon
from collections import Counter

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

SMOKE = "--smoke" in sys.argv

from adaptive_conf_vol_experiment import (
    ASSETS, SMOKE_ASSETS, ALL_ASSETS, CAP, VIP_TIERS, SPREAD_BPS,
    generate_all_raw_signals, compute_vol_all, apply_vol_scaling,
    filter_baseline, run_sim, get_vip_tier, HARD_CONF_FLOOR,
)

RAW_CACHE = Path("data/cache/adaptive_raw_signals.parquet")
FUNDING_DIR = Path("data/funding-rates")

# ============================================================
# Funding data loading
# ============================================================

def load_funding_rates():
    """Load 8-hourly funding rates for all assets. Returns dict[asset] -> Series."""
    funding = {}
    for name, sym in ASSETS.items():
        fp = FUNDING_DIR / f"{sym}_funding.parquet"
        if not fp.exists():
            continue
        df = pd.read_parquet(fp)
        if df.index.tz is not None:
            df.index = df.index.tz_localize(None)
        funding[name] = df["fundingRate"]
    return funding


def compute_funding_zscore(funding_rate, window=90):
    """Rolling z-score of funding rate. Window=90 settlements ≈ 30 days."""
    mean = funding_rate.rolling(window, min_periods=30).mean()
    std = funding_rate.rolling(window, min_periods=30).std()
    return ((funding_rate - mean) / std.replace(0, np.nan)).fillna(0)


def compute_funding_persistence(funding_zscore, threshold=2.0):
    """Count consecutive settlements where |zscore| > threshold.

    Returns Series of consecutive count (resets to 0 when below threshold).
    """
    extreme = (funding_zscore.abs() > threshold).astype(int)
    # Cumulative sum that resets on 0
    groups = (extreme != extreme.shift(1)).cumsum()
    persistence = extreme.groupby(groups).cumsum()
    return persistence


# ============================================================
# Funding filter variants
# ============================================================

def apply_funding_suppress(signals, funding_data, zscore_thresh=2.0, persist_thresh=4):
    """Suppress signals that agree with the crowded direction.

    When funding zscore > +thresh for > persist_thresh settlements:
      → suppress LONG signals (crowded longs → reversal risk)
    When funding zscore < -thresh for > persist_thresh settlements:
      → suppress SHORT signals (crowded shorts → squeeze risk)

    persist_thresh=4 settlements = 32 hours ≈ 1.3 days
    """
    out = signals.copy()
    suppress_mask = np.zeros(len(out), dtype=bool)

    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)

    for asset in out["asset"].unique():
        if asset not in funding_data:
            continue
        fr = funding_data[asset]
        zscore = compute_funding_zscore(fr)
        persistence = compute_funding_persistence(zscore, threshold=zscore_thresh)

        # Create funding regime at signal timestamps via ffill
        combined_idx = zscore.index.union(ts.unique())
        zscore_aligned = zscore.reindex(combined_idx).ffill()
        persist_aligned = persistence.reindex(combined_idx).ffill().fillna(0)

        asset_mask = out["asset"] == asset
        asset_ts = ts[asset_mask]

        zs_vals = zscore_aligned.reindex(asset_ts.values).values
        ps_vals = persist_aligned.reindex(asset_ts.values).values
        dirs = out.loc[asset_mask, "direction"].values

        # Suppress: crowded longs (zscore > thresh, persistent) → kill longs
        #           crowded shorts (zscore < -thresh, persistent) → kill shorts
        crowded_long = (zs_vals > zscore_thresh) & (ps_vals >= persist_thresh)
        crowded_short = (zs_vals < -zscore_thresh) & (ps_vals >= persist_thresh)

        suppress = (crowded_long & (dirs > 0)) | (crowded_short & (dirs < 0))
        suppress_mask[asset_mask.values] = suppress

    n_suppressed = suppress_mask.sum()
    result = out[~suppress_mask].reset_index(drop=True)
    return result, n_suppressed


def apply_funding_sizing(signals, funding_data, zscore_thresh=2.0, persist_thresh=4,
                         reduction=0.5):
    """Reduce sizing (not suppress) for crowded-direction signals.

    Same logic as suppress, but instead of removing, multiply sizing_mult
    by `reduction` (e.g., 0.5 = half size).
    """
    out = signals.copy()
    if "sizing_mult" not in out.columns:
        out["sizing_mult"] = 1.0

    ts = pd.to_datetime(out["timestamp"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)

    n_reduced = 0
    for asset in out["asset"].unique():
        if asset not in funding_data:
            continue
        fr = funding_data[asset]
        zscore = compute_funding_zscore(fr)
        persistence = compute_funding_persistence(zscore, threshold=zscore_thresh)

        combined_idx = zscore.index.union(ts.unique())
        zscore_aligned = zscore.reindex(combined_idx).ffill()
        persist_aligned = persistence.reindex(combined_idx).ffill().fillna(0)

        asset_mask = out["asset"] == asset
        asset_ts = ts[asset_mask]

        zs_vals = zscore_aligned.reindex(asset_ts.values).values
        ps_vals = persist_aligned.reindex(asset_ts.values).values
        dirs = out.loc[asset_mask, "direction"].values

        crowded_long = (zs_vals > zscore_thresh) & (ps_vals >= persist_thresh)
        crowded_short = (zs_vals < -zscore_thresh) & (ps_vals >= persist_thresh)
        reduce_mask = (crowded_long & (dirs > 0)) | (crowded_short & (dirs < 0))

        n_reduced += reduce_mask.sum()
        idx = out.index[asset_mask][reduce_mask]
        out.loc[idx, "sizing_mult"] = out.loc[idx, "sizing_mult"] * reduction

    return out, n_reduced


# ============================================================
# Main
# ============================================================

def main():
    T0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print("=" * 80)
    print(f"  FUNDING REGIME FILTER EXPERIMENT ({mode})")
    print(f"  {len(ASSETS)} assets, suppress vs sizing vs baseline")
    print("=" * 80)

    # Phase 1: Load raw signals
    print(f"\n  Phase 1: Loading raw signals...", flush=True)
    if RAW_CACHE.exists():
        raw = pd.read_parquet(RAW_CACHE)
        print(f"  Loaded {len(raw):,} cached signals")
    else:
        print(f"  Cache not found, generating...", flush=True)
        strat_sigs = generate_all_raw_signals()
        raw = pd.concat(list(strat_sigs.values()), ignore_index=True)
        raw = raw.sort_values("timestamp").reset_index(drop=True)
        RAW_CACHE.parent.mkdir(parents=True, exist_ok=True)
        raw.to_parquet(RAW_CACHE)
        print(f"  Generated and cached {len(raw):,} signals")

    # Phase 2: Load funding rates
    print(f"\n  Phase 2: Loading funding rates...", flush=True)
    funding_data = load_funding_rates()
    print(f"  Loaded funding for {len(funding_data)} assets")
    for name in sorted(funding_data.keys())[:5]:
        fr = funding_data[name]
        print(f"    {name}: {len(fr)} settlements, {fr.index[0].date()} to {fr.index[-1].date()}")
    if len(funding_data) > 5:
        print(f"    ... and {len(funding_data)-5} more")

    # Phase 3: Vol series
    print(f"\n  Phase 3: Computing vol series...", flush=True)
    vol_series = compute_vol_all()

    # Phase 4: Apply baseline filter + vol scaling
    print(f"\n  Phase 4: Building variants...", flush=True)
    base_filtered = filter_baseline(raw)

    # Variant 1: Baseline (no funding filter)
    v1 = apply_vol_scaling(base_filtered.copy(), vol_series)
    print(f"    1-Baseline:       {len(v1):>8,} signals")

    # Variant 2: Funding suppress (z>2, persist>4 settlements)
    v2_pre, n_supp = apply_funding_suppress(base_filtered, funding_data,
                                             zscore_thresh=2.0, persist_thresh=4)
    v2 = apply_vol_scaling(v2_pre, vol_series)
    print(f"    2-FundSuppress:   {len(v2):>8,} signals (suppressed {n_supp})")

    # Variant 3: Funding sizing (reduce to 50% instead of suppress)
    v3_pre, n_red = apply_funding_sizing(base_filtered.copy(), funding_data,
                                          zscore_thresh=2.0, persist_thresh=4,
                                          reduction=0.5)
    v3 = apply_vol_scaling(v3_pre, vol_series)
    print(f"    3-FundSizing:     {len(v3):>8,} signals (reduced {n_red})")

    # Variant 4: Aggressive suppress (z>1.5, persist>2)
    v4_pre, n_supp4 = apply_funding_suppress(base_filtered, funding_data,
                                              zscore_thresh=1.5, persist_thresh=2)
    v4 = apply_vol_scaling(v4_pre, vol_series)
    print(f"    4-AggrSuppress:   {len(v4):>8,} signals (suppressed {n_supp4})")

    variants = {"1-Baseline": v1, "2-FundSuppress": v2,
                "3-FundSizing": v3, "4-AggrSuppress": v4}

    # Phase 5: Portfolio sims
    print(f"\n{'='*80}")
    print("  PORTFOLIO RESULTS (MkrTkr60, VIP fees, $250K)")
    print(f"{'='*80}")

    results = {}
    for vname, vsigs in variants.items():
        r = run_sim(vsigs, CAP)
        results[vname] = r
        if r:
            print(f"\n  {vname}:")
            print(f"    Sharpe:    {r['sr']:.3f}")
            print(f"    Sortino:   {r['sortino']:.3f}")
            print(f"    PnL:       ${r['pnl']:,.0f}")
            print(f"    MaxDD:     {r['dd']:.2f}%")
            print(f"    Calmar:    {r['calmar']:.2f}")
            print(f"    PF:        {r['pf']:.2f}")
            print(f"    Trades:    {r['n_trades']:,}")
            print(f"    $/month:   ${r['pnl_mo']:,.0f}")
            print(f"    Pos mo:    {r['pos_months']}/{r['tot_months']}")
        else:
            print(f"\n  {vname}: FAILED")

    baseline_r = results.get("1-Baseline")
    if baseline_r is None:
        print("\n  Baseline failed.")
        return

    # Phase 6: Statistical comparison
    print(f"\n{'='*80}")
    print("  STATISTICAL COMPARISON vs BASELINE")
    print(f"{'='*80}")
    print(f"\n  {'Variant':<20} {'Sharpe':>8} {'Delta':>8} {'PnL':>12} {'DD':>8} {'Trades':>8} {'Wilcoxon':>12}")
    print("  " + "-" * 82)

    for vname, r in results.items():
        if r is None:
            continue
        delta = r["sr"] - baseline_r["sr"]
        if vname == "1-Baseline":
            p_str = "—"
        else:
            try:
                bm, vm = baseline_r["monthly"], r["monthly"]
                common = bm.index.intersection(vm.index)
                if len(common) > 10:
                    _, p = wilcoxon(vm.reindex(common).values - bm.reindex(common).values)
                    p_str = f"p={p:.4f}"
                else:
                    p_str = "n/a"
            except Exception:
                p_str = "n/a"
        print(f"  {vname:<20} {r['sr']:>8.3f} {delta:>+8.3f} ${r['pnl']:>10,.0f} {r['dd']:>7.2f}% {r['n_trades']:>8,} {p_str:>12}")

    # Phase 7: Year-by-year
    print(f"\n{'='*80}")
    print("  YEAR-BY-YEAR SHARPE")
    print(f"{'='*80}")
    years = set()
    for r in results.values():
        if r and r["monthly"] is not None:
            years.update(r["monthly"].index.year)
    years = sorted(years)
    header = f"  {'Year':<6}" + "".join(f"{v:>16}" for v in results.keys())
    print(header)
    print("  " + "-" * (6 + 16 * len(results)))
    for yr in years:
        row = f"  {yr:<6}"
        for r in results.values():
            if r is None:
                row += f"{'—':>16}"; continue
            m = r["monthly"]
            yr_m = m[m.index.year == yr]
            if len(yr_m) < 2:
                row += f"{'—':>16}"; continue
            dr = yr_m / CAP
            sr = float(dr.mean() / dr.std() * np.sqrt(12)) if dr.std() > 0 else 0
            row += f"{sr:>16.3f}"
        print(row)

    # Phase 8: CPCV validation for best non-baseline variant
    best_vname = max((v for v in results if v != "1-Baseline" and results[v]),
                     key=lambda v: results[v]["sr"], default=None)
    if best_vname and results[best_vname]["sr"] > baseline_r["sr"]:
        print(f"\n{'='*80}")
        print(f"  CPCV VALIDATION: {best_vname}")
        print(f"{'='*80}")

        months = pd.to_datetime(raw["timestamp"]).dt.to_period("M")
        unique_months = sorted(months.unique())
        n_folds = 6
        fold_size = len(unique_months) // n_folds
        folds = []
        for i in range(n_folds):
            start = i * fold_size
            end = start + fold_size if i < n_folds - 1 else len(unique_months)
            folds.append(set(unique_months[start:end]))

        cpcv_deltas = []
        for test_idx in range(n_folds):
            test_months = folds[test_idx]
            test_mask = months.isin(test_months).values
            test_raw = raw[test_mask].reset_index(drop=True)
            if len(test_raw) < 100:
                continue

            test_base = filter_baseline(test_raw)
            tb_scaled = apply_vol_scaling(test_base.copy(), vol_series)
            base_r = run_sim(tb_scaled, CAP)
            if base_r is None:
                continue

            # Apply same funding filter params as best variant
            if "Suppress" in best_vname:
                thresh = 2.0 if "Aggr" not in best_vname else 1.5
                persist = 4 if "Aggr" not in best_vname else 2
                tf, _ = apply_funding_suppress(test_base, funding_data,
                                                zscore_thresh=thresh,
                                                persist_thresh=persist)
            else:
                tf, _ = apply_funding_sizing(test_base.copy(), funding_data)
            tf_scaled = apply_vol_scaling(tf, vol_series)
            adapt_r = run_sim(tf_scaled, CAP)
            if adapt_r is None:
                continue

            delta = adapt_r["sr"] - base_r["sr"]
            cpcv_deltas.append(delta)
            s = sorted(test_months)
            print(f"    Path {test_idx} ({s[0]}-{s[-1]}): "
                  f"base={base_r['sr']:.3f} adapt={adapt_r['sr']:.3f} delta={delta:+.3f}")

        if cpcv_deltas:
            print(f"\n    CPCV mean delta:  {np.mean(cpcv_deltas):+.3f}")
            print(f"    CPCV wins:        {sum(1 for d in cpcv_deltas if d > 0)}/{len(cpcv_deltas)}")
            if len(cpcv_deltas) >= 5:
                try:
                    _, p = wilcoxon(cpcv_deltas)
                    print(f"    Wilcoxon p:       {p:.4f}")
                except Exception:
                    pass
            rng = np.random.RandomState(42)
            boot = [np.mean(rng.choice(cpcv_deltas, len(cpcv_deltas), replace=True))
                    for _ in range(5000)]
            print(f"    95% CI:           [{np.percentile(boot, 2.5):+.3f}, {np.percentile(boot, 97.5):+.3f}]")

    # Phase 9: Verdict
    print(f"\n{'='*80}")
    print("  EXPERIMENT VERDICT")
    print(f"{'='*80}")
    for vname, r in results.items():
        if vname == "1-Baseline" or r is None:
            continue
        delta = r["sr"] - baseline_r["sr"]
        trade_drop = 1.0 - r["n_trades"] / max(baseline_r["n_trades"], 1)
        dd_change = r["dd"] - baseline_r["dd"]
        verdict = "PASS" if delta >= 0.05 and trade_drop <= 0.30 else (
            "FAIL/KILL" if delta < -0.05 else "MARGINAL")
        print(f"\n  {vname}:")
        print(f"    Sharpe delta: {delta:+.3f} (need >= +0.05)")
        print(f"    Trade drop:   {trade_drop:.1%} (need <= 30%)")
        print(f"    DD change:    {dd_change:+.2f}pp")
        print(f"    VERDICT:      {verdict}")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
