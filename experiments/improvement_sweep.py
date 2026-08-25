"""
Systematic Improvement Sweep — EXP-1 through EXP-17.

Baseline: Sharpe 2.076 (honest, regime shift(1), 16 assets).
Each experiment changes ONE thing and measures the delta.

Usage:
    python experiments/improvement_sweep.py --phase 1
    python experiments/improvement_sweep.py --exp 1
    python experiments/improvement_sweep.py --exp 4  # dollar bars re-test
"""

import sys
sys.path.insert(0, "src")

import argparse
import copy
import json
import logging
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("sweep")
logger.setLevel(logging.INFO)

from signal_engine.config import load_config, SystemConfig
from signal_engine.pipeline import run_backtest

RESULTS_DIR = Path("v4-results/improvement_sweep")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_CONFIG = Path("configs/default.yaml")

# Use all CPU cores for parallel backtesting
import os
MAX_WORKERS = max(1, os.cpu_count() - 1)  # leave 1 core free


def run_single(config: SystemConfig, label: str) -> dict[str, Any]:
    """Run a single backtest and return stats."""
    t0 = time.time()
    results = run_backtest(config, max_workers=MAX_WORKERS)
    elapsed = time.time() - t0
    stats = results.get("stats", {})
    stats["label"] = label
    stats["elapsed_s"] = round(elapsed, 1)
    stats["n_total_raw"] = results.get("n_total_raw", 0)
    stats["n_after_confidence"] = results.get("n_after_confidence", 0)
    stats["n_after_regime"] = results.get("n_after_regime", 0)
    return stats


def print_result(stats: dict, baseline_sharpe: float = 2.076) -> None:
    """Print formatted result with delta from baseline."""
    sharpe = stats.get("sharpe", 0)
    delta = sharpe - baseline_sharpe
    sign = "+" if delta >= 0 else ""
    logger.info(
        "  %-40s Sharpe=%6.3f (%s%.3f)  Return=%6.2f%%  MaxDD=%5.2f%%  Trades=%6d  Time=%.0fs",
        stats.get("label", "?"),
        sharpe,
        sign, delta,
        stats.get("total_return_pct", 0),
        stats.get("max_dd_pct", 0),
        stats.get("n_trades", 0),
        stats.get("elapsed_s", 0),
    )


def save_results(experiment_name: str, results: list[dict]) -> None:
    """Save experiment results to JSON."""
    path = RESULTS_DIR / f"{experiment_name}.json"
    with open(path, "w") as f:
        json.dump({
            "experiment": experiment_name,
            "timestamp": datetime.utcnow().isoformat(),
            "baseline_sharpe": 2.076,
            "results": results,
        }, f, indent=2, default=str)
    logger.info("  Saved to %s", path)


# =========================================================================
# PHASE 1: Filter Tuning
# =========================================================================

def exp1_confidence_sweep() -> list[dict]:
    """EXP-1: Absolute confidence threshold sweep."""
    logger.info("=" * 70)
    logger.info("EXP-1: Absolute confidence threshold sweep")
    logger.info("=" * 70)

    cfg_base = load_config(BASELINE_CONFIG)
    results = []

    # First run baseline (percentile 70)
    logger.info("Running baseline (percentile 70)...")
    baseline = run_single(cfg_base, "baseline_percentile70")
    print_result(baseline)
    results.append(baseline)

    # Sweep absolute thresholds
    for thresh in [0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.10]:
        cfg = copy.deepcopy(cfg_base)
        cfg.confidence_filter.mode = "absolute"
        cfg.confidence_filter.threshold = thresh
        label = f"absolute_{thresh:.2f}"
        logger.info("Running %s...", label)
        stats = run_single(cfg, label)
        print_result(stats)
        results.append(stats)

    save_results("EXP-1_confidence_sweep", results)
    return results


def exp2_vol_filter_sweep() -> list[dict]:
    """EXP-2: Volatility filter quantile sweep."""
    logger.info("=" * 70)
    logger.info("EXP-2: Volatility filter quantile sweep")
    logger.info("=" * 70)

    cfg_base = load_config(BASELINE_CONFIG)
    results = []

    for q in [0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        cfg = copy.deepcopy(cfg_base)
        cfg.features.vol_filter_quantile = q
        label = f"vol_q{q:.2f}"
        logger.info("Running %s...", label)
        stats = run_single(cfg, label)
        print_result(stats)
        results.append(stats)

    save_results("EXP-2_vol_filter_sweep", results)
    return results


def exp3_regime_ema_sweep() -> list[dict]:
    """EXP-3: Regime EMA span sweep."""
    logger.info("=" * 70)
    logger.info("EXP-3: Regime EMA span sweep")
    logger.info("=" * 70)

    cfg_base = load_config(BASELINE_CONFIG)
    results = []

    # Also test with regime disabled
    cfg_off = copy.deepcopy(cfg_base)
    cfg_off.regime_filter.enabled = False
    logger.info("Running regime_disabled...")
    stats = run_single(cfg_off, "regime_disabled")
    print_result(stats)
    results.append(stats)

    for span in [5, 10, 15, 20, 30, 50]:
        cfg = copy.deepcopy(cfg_base)
        cfg.regime_filter.ema_span = span
        label = f"regime_ema{span}"
        logger.info("Running %s...", label)
        stats = run_single(cfg, label)
        print_result(stats)
        results.append(stats)

    save_results("EXP-3_regime_ema_sweep", results)
    return results


# =========================================================================
# PHASE 3: New Features
# =========================================================================

def exp11_parkinson_vol() -> list[dict]:
    """EXP-11: Add Parkinson volatility feature."""
    logger.info("=" * 70)
    logger.info("EXP-11: Parkinson volatility feature")
    logger.info("=" * 70)

    cfg_base = load_config(BASELINE_CONFIG)
    results = []

    # Baseline
    logger.info("Running baseline (14 features)...")
    baseline = run_single(cfg_base, "baseline_14feat")
    print_result(baseline)
    results.append(baseline)

    # Add parkinson_vol
    cfg = copy.deepcopy(cfg_base)
    cfg.features.feature_list = cfg_base.features.feature_list + ["parkinson_vol_20"]
    label = "15feat_parkinson"
    logger.info("Running %s...", label)
    stats = run_single(cfg, label)
    print_result(stats)
    results.append(stats)

    save_results("EXP-11_parkinson_vol", results)
    return results


def exp12_corwin_schultz() -> list[dict]:
    """EXP-12: Add Corwin-Schultz spread estimator."""
    logger.info("=" * 70)
    logger.info("EXP-12: Corwin-Schultz spread estimator")
    logger.info("=" * 70)

    cfg_base = load_config(BASELINE_CONFIG)
    results = []

    # Baseline
    baseline = run_single(cfg_base, "baseline_14feat")
    print_result(baseline)
    results.append(baseline)

    # Add cs_spread
    cfg = copy.deepcopy(cfg_base)
    cfg.features.feature_list = cfg_base.features.feature_list + ["cs_spread"]
    label = "15feat_cs_spread"
    logger.info("Running %s...", label)
    stats = run_single(cfg, label)
    print_result(stats)
    results.append(stats)

    save_results("EXP-12_corwin_schultz", results)
    return results


def exp13_entropy() -> list[dict]:
    """EXP-13: Add rolling entropy feature."""
    logger.info("=" * 70)
    logger.info("EXP-13: Rolling entropy")
    logger.info("=" * 70)

    cfg_base = load_config(BASELINE_CONFIG)
    results = []

    baseline = run_single(cfg_base, "baseline_14feat")
    print_result(baseline)
    results.append(baseline)

    cfg = copy.deepcopy(cfg_base)
    cfg.features.feature_list = cfg_base.features.feature_list + ["entropy"]
    label = "15feat_entropy"
    logger.info("Running %s...", label)
    stats = run_single(cfg, label)
    print_result(stats)
    results.append(stats)

    save_results("EXP-13_entropy", results)
    return results


# =========================================================================
# Main
# =========================================================================

EXPERIMENTS = {
    1: ("Phase 1: Confidence sweep", exp1_confidence_sweep),
    2: ("Phase 1: Vol filter sweep", exp2_vol_filter_sweep),
    3: ("Phase 1: Regime EMA sweep", exp3_regime_ema_sweep),
    11: ("Phase 3: Parkinson vol", exp11_parkinson_vol),
    12: ("Phase 3: Corwin-Schultz", exp12_corwin_schultz),
    13: ("Phase 3: Entropy", exp13_entropy),
}


def run_phase(phase: int) -> None:
    """Run all experiments in a phase."""
    phase_map = {1: [1, 2, 3], 3: [11, 12, 13]}
    if phase not in phase_map:
        logger.error("Unknown phase: %d. Available: %s", phase, list(phase_map.keys()))
        return
    for exp_num in phase_map[phase]:
        name, func = EXPERIMENTS[exp_num]
        logger.info("\n>>> %s", name)
        func()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Improvement Sweep")
    parser.add_argument("--phase", type=int, help="Run all experiments in phase N")
    parser.add_argument("--exp", type=int, help="Run single experiment N")
    args = parser.parse_args()

    if args.exp:
        if args.exp in EXPERIMENTS:
            name, func = EXPERIMENTS[args.exp]
            logger.info(">>> %s", name)
            func()
        else:
            logger.error("Unknown experiment: %d. Available: %s", args.exp, sorted(EXPERIMENTS.keys()))
    elif args.phase:
        run_phase(args.phase)
    else:
        logger.info("Running Phase 1 (filter tuning)...")
        run_phase(1)
