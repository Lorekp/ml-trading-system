#!/usr/bin/env python
"""CPCV Validation: Adaptive Confidence Threshold Range.

Uses Combinatorially Purged Cross-Validation to tune the adaptive
confidence floor range without overfitting.

Approach:
  1. Load raw signals (from parquet cache or generate fresh)
  2. Split 68 months into 6 temporal folds (~11 months each)
  3. For each CPCV path (1 fold = test, 5 folds = train):
     - On training: sweep range candidates, pick best Sharpe
     - On test: evaluate that chosen range + baseline
  4. Compare adaptive vs baseline across all 6 paths
  5. Bootstrap CI

If adaptive beats baseline on most paths, the improvement is real.

Usage:
    PYTHONPATH=src python experiments/adaptive_conf_cpcv.py [--smoke] [--regen]
"""
import sys, os, time, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from scipy.stats import wilcoxon

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(0, "src")
warnings.filterwarnings("ignore")

SMOKE = "--smoke" in sys.argv
REGEN = "--regen" in sys.argv

# Import the signal generation from the main experiment
from adaptive_conf_vol_experiment import (
    ASSETS, SMOKE_ASSETS, ALL_ASSETS, CAP, VIP_TIERS, SPREAD_BPS,
    generate_all_raw_signals, compute_vol_all, apply_vol_scaling,
    run_sim, get_vip_tier, HARD_CONF_FLOOR,
)

CACHE_PATH = Path("data/cache/adaptive_raw_signals.parquet")
N_FOLDS = 6

# Range candidates to sweep
RANGE_CANDIDATES = [
    (0.060, 0.120),   # current (wide)
    (0.065, 0.105),   # balanced
    (0.070, 0.100),   # narrow
    (0.070, 0.110),   # medium
    (0.075, 0.095),   # tight
]


# ============================================================
# Signal loading / caching
# ============================================================

def load_or_generate_signals():
    """Load cached raw signals or generate fresh."""
    if CACHE_PATH.exists() and not REGEN:
        print(f"  Loading cached signals from {CACHE_PATH}...", flush=True)
        raw = pd.read_parquet(CACHE_PATH)
        print(f"  Loaded {len(raw):,} signals")
        return raw

    print(f"  Generating fresh raw signals...", flush=True)
    strat_sigs = generate_all_raw_signals()
    if not strat_sigs:
        raise RuntimeError("No signals generated")
    raw = pd.concat(list(strat_sigs.values()), ignore_index=True)
    raw = raw.sort_values("timestamp").reset_index(drop=True)

    # Cache to disk
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw.to_parquet(CACHE_PATH)
    print(f"  Cached {len(raw):,} signals to {CACHE_PATH}")
    return raw


# ============================================================
# Filtering functions
# ============================================================

def filter_baseline(signals):
    """Production: 70th pctl + 0.08 floor."""
    df = signals.copy()
    mask = (df["confidence"] >= df["window_conf_p70"]) & (df["confidence"] >= HARD_CONF_FLOOR)
    return df[mask].reset_index(drop=True)


def filter_adaptive(signals, lo=0.06, hi=0.12):
    """Adaptive floor with configurable range [lo, hi]."""
    df = signals.copy()

    # Expanding median IQR (walk-forward safe)
    window_iqrs = df.groupby("window_idx")["window_conf_iqr"].first().sort_index()
    expanding_median = window_iqrs.expanding().median()
    df["_exp_median_iqr"] = df["window_idx"].map(expanding_median)

    # Adaptive floor
    iqr_ratio = df["_exp_median_iqr"] / df["window_conf_iqr"].clip(lower=1e-8)
    adaptive_floor = (HARD_CONF_FLOOR * iqr_ratio).clip(lower=lo, upper=hi)
    effective_thresh = np.maximum(adaptive_floor, df["window_conf_p70"])

    mask = df["confidence"] >= effective_thresh
    df = df.drop(columns=["_exp_median_iqr"])
    return df[mask].reset_index(drop=True)


# ============================================================
# CPCV engine
# ============================================================

def split_by_month(signals):
    """Add month_period column for fold splitting."""
    ts = pd.to_datetime(signals["timestamp"])
    return ts.dt.to_period("M")


def run_fold_sim(signals_subset, vol_series):
    """Apply vol scaling and run portfolio sim on a signal subset."""
    if len(signals_subset) < 30:
        return None
    scaled = apply_vol_scaling(signals_subset, vol_series)
    return run_sim(scaled, CAP)


def cpcv_evaluate(raw, vol_series, folds, test_idx):
    """Run one CPCV path: select best range on train, evaluate on test.

    Returns: dict with test Sharpe for baseline and each candidate,
             plus which range was selected on training.
    """
    test_months = folds[test_idx]
    train_months = set()
    for i, fold in enumerate(folds):
        if i != test_idx:
            train_months |= fold

    months = split_by_month(raw)
    train_mask = months.isin(train_months)
    test_mask = months.isin(test_months)

    train_raw = raw[train_mask.values].reset_index(drop=True)
    test_raw = raw[test_mask.values].reset_index(drop=True)

    if len(train_raw) < 100 or len(test_raw) < 100:
        return None

    # Evaluate baseline on test
    test_base = filter_baseline(test_raw)
    base_result = run_fold_sim(test_base, vol_series)
    base_sr = base_result["sr"] if base_result else None

    # Sweep ranges on TRAINING data to select best
    best_train_sr = -999
    best_range = RANGE_CANDIDATES[0]
    train_range_results = {}

    for lo, hi in RANGE_CANDIDATES:
        train_filtered = filter_adaptive(train_raw, lo=lo, hi=hi)
        r = run_fold_sim(train_filtered, vol_series)
        sr = r["sr"] if r else -999
        train_range_results[(lo, hi)] = sr
        if sr > best_train_sr:
            best_train_sr = sr
            best_range = (lo, hi)

    # Evaluate SELECTED range on TEST data (the key OOS test)
    test_adaptive = filter_adaptive(test_raw, lo=best_range[0], hi=best_range[1])
    adapt_result = run_fold_sim(test_adaptive, vol_series)
    adapt_sr = adapt_result["sr"] if adapt_result else None

    # Also evaluate ALL ranges on test (for information, not for selection)
    test_all_ranges = {}
    for lo, hi in RANGE_CANDIDATES:
        test_filt = filter_adaptive(test_raw, lo=lo, hi=hi)
        r = run_fold_sim(test_filt, vol_series)
        test_all_ranges[(lo, hi)] = r["sr"] if r else None

    return {
        "test_baseline_sr": base_sr,
        "test_adaptive_sr": adapt_sr,
        "selected_range": best_range,
        "train_range_results": train_range_results,
        "test_all_ranges": test_all_ranges,
        "n_train": len(train_raw),
        "n_test_base": len(test_base),
        "n_test_adapt": len(test_adaptive) if adapt_result else 0,
    }


# ============================================================
# Main
# ============================================================

def main():
    T0 = time.time()
    mode = "SMOKE" if SMOKE else "FULL"
    print("=" * 80)
    print(f"  CPCV VALIDATION: ADAPTIVE CONFIDENCE RANGE ({mode})")
    print(f"  {N_FOLDS} folds, {len(RANGE_CANDIDATES)} range candidates")
    print("=" * 80)

    # Phase 1: Load signals
    print(f"\n  Phase 1: Loading raw signals...", flush=True)
    t0 = time.time()
    raw = load_or_generate_signals()
    print(f"  {len(raw):,} signals ({time.time()-t0:.0f}s)")

    # Phase 2: Vol series
    print(f"\n  Phase 2: Computing vol series...", flush=True)
    t0 = time.time()
    vol_series = compute_vol_all()
    print(f"  Done ({time.time()-t0:.0f}s)")

    # Phase 3: Build temporal folds
    months = split_by_month(raw)
    unique_months = sorted(months.unique())
    print(f"\n  Phase 3: Splitting {len(unique_months)} months into {N_FOLDS} folds")

    fold_size = len(unique_months) // N_FOLDS
    folds = []
    for i in range(N_FOLDS):
        start = i * fold_size
        end = start + fold_size if i < N_FOLDS - 1 else len(unique_months)
        fold_months = set(unique_months[start:end])
        folds.append(fold_months)
        s = sorted(fold_months)
        print(f"    Fold {i}: {s[0]} to {s[-1]} ({len(fold_months)} months)")

    # Phase 4: Run CPCV
    print(f"\n  Phase 4: Running {N_FOLDS} CPCV paths...", flush=True)
    results = []
    for test_idx in range(N_FOLDS):
        t0 = time.time()
        r = cpcv_evaluate(raw, vol_series, folds, test_idx)
        results.append(r)
        if r:
            delta = (r["test_adaptive_sr"] or 0) - (r["test_baseline_sr"] or 0)
            print(f"    Path {test_idx}: base={r['test_baseline_sr']:.3f}  "
                  f"adapt={r['test_adaptive_sr']:.3f}  "
                  f"delta={delta:+.3f}  "
                  f"range={r['selected_range']}  "
                  f"({time.time()-t0:.0f}s)", flush=True)
        else:
            print(f"    Path {test_idx}: FAILED", flush=True)

    valid = [r for r in results if r is not None]
    if not valid:
        print("\n  ERROR: No valid CPCV paths!")
        return

    # Phase 5: Results
    print(f"\n{'='*80}")
    print("  CPCV RESULTS")
    print(f"{'='*80}")

    base_sharpes = [r["test_baseline_sr"] for r in valid if r["test_baseline_sr"] is not None]
    adapt_sharpes = [r["test_adaptive_sr"] for r in valid if r["test_adaptive_sr"] is not None]
    deltas = [a - b for a, b in zip(adapt_sharpes, base_sharpes) if a is not None and b is not None]

    print(f"\n  {'Metric':<30} {'Baseline':>10} {'Adaptive':>10}")
    print("  " + "-" * 52)
    print(f"  {'Mean Sharpe':<30} {np.mean(base_sharpes):>10.3f} {np.mean(adapt_sharpes):>10.3f}")
    print(f"  {'Median Sharpe':<30} {np.median(base_sharpes):>10.3f} {np.median(adapt_sharpes):>10.3f}")
    print(f"  {'Std Sharpe':<30} {np.std(base_sharpes):>10.3f} {np.std(adapt_sharpes):>10.3f}")
    print(f"  {'Min Sharpe':<30} {np.min(base_sharpes):>10.3f} {np.min(adapt_sharpes):>10.3f}")
    print(f"  {'Max Sharpe':<30} {np.max(base_sharpes):>10.3f} {np.max(adapt_sharpes):>10.3f}")

    n_wins = sum(1 for d in deltas if d > 0)
    print(f"\n  Adaptive wins: {n_wins}/{len(deltas)} paths")
    print(f"  Mean delta:    {np.mean(deltas):+.3f}")
    print(f"  Median delta:  {np.median(deltas):+.3f}")

    # Paired Wilcoxon on CPCV Sharpes
    if len(deltas) >= 5:
        try:
            stat, p = wilcoxon(deltas)
            print(f"  Wilcoxon p:    {p:.4f} ({'significant' if p < 0.10 else 'not significant'})")
        except Exception:
            print(f"  Wilcoxon:      n/a")

    # Bootstrap CI on mean delta
    rng = np.random.RandomState(42)
    boot_deltas = [np.mean(rng.choice(deltas, size=len(deltas), replace=True)) for _ in range(5000)]
    ci_lo = np.percentile(boot_deltas, 2.5)
    ci_hi = np.percentile(boot_deltas, 97.5)
    print(f"  Delta 95% CI:  [{ci_lo:+.3f}, {ci_hi:+.3f}]")

    # Selected ranges across paths
    print(f"\n  Ranges selected on training data:")
    from collections import Counter
    range_counts = Counter(r["selected_range"] for r in valid)
    for rng_val, cnt in range_counts.most_common():
        print(f"    {rng_val}: {cnt}/{len(valid)} paths")

    # Per-range OOS performance (information only)
    print(f"\n  Per-range OOS Sharpe (all paths, for reference):")
    print(f"  {'Range':<18} {'Mean':>8} {'Median':>8} {'Wins':>6}")
    print("  " + "-" * 42)
    for lo, hi in RANGE_CANDIDATES:
        range_srs = [r["test_all_ranges"][(lo, hi)] for r in valid
                     if r["test_all_ranges"].get((lo, hi)) is not None]
        range_wins = sum(1 for s, b in zip(range_srs, base_sharpes) if s > b)
        if range_srs:
            print(f"  ({lo:.3f}, {hi:.3f})  {np.mean(range_srs):>8.3f} {np.median(range_srs):>8.3f} {range_wins:>4}/{len(range_srs)}")

    # Fixed 0.08 baseline for comparison
    print(f"  {'Baseline (0.08)':<18} {np.mean(base_sharpes):>8.3f} {np.median(base_sharpes):>8.3f} {'—':>6}")

    # Verdict
    print(f"\n{'='*80}")
    print("  VERDICT")
    print(f"{'='*80}")
    mean_delta = np.mean(deltas)
    pct_wins = n_wins / len(deltas) * 100
    robust = n_wins >= len(deltas) * 0.6 and ci_lo > 0
    print(f"\n  Mean Sharpe improvement:  {mean_delta:+.3f}")
    print(f"  Wins:                     {pct_wins:.0f}%")
    print(f"  CI excludes zero:         {'YES' if ci_lo > 0 else 'NO'}")
    print(f"  ROBUST:                   {'YES' if robust else 'NO'}")

    if robust:
        best_range = range_counts.most_common(1)[0][0]
        print(f"\n  Recommended range:        {best_range}")
        print(f"  (most frequently selected across CPCV training folds)")
    else:
        print(f"\n  Adaptive improvement does NOT survive CPCV validation.")
        print(f"  Stick with baseline 0.08 fixed floor.")

    print(f"\n  Runtime: {time.time()-T0:.0f}s")


if __name__ == "__main__":
    main()
