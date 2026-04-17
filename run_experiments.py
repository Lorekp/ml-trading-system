#!/usr/bin/env python3
"""
Run all ML Trading System experiments.

Usage:
    python run_experiments.py --baselines              # Run baseline models
    python run_experiments.py --kronos                  # Run Kronos zero-shot
    python run_experiments.py --all                     # Run everything
    python run_experiments.py --baselines --timeframe 4H  # Specific timeframe

Requires data to be downloaded first (data/raw/binance-1m/ or homeserver).
"""

import argparse
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from evaluation.experiment import run_baseline_experiment, run_kronos_experiment

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("experiments")


def main():
    parser = argparse.ArgumentParser(description="ML Trading System Experiments")
    parser.add_argument("--baselines", action="store_true", help="Run baseline models")
    parser.add_argument("--kronos", action="store_true", help="Run Kronos zero-shot")
    parser.add_argument("--all", action="store_true", help="Run all experiments")
    parser.add_argument("--data-dir", type=str, default=None,
                        help="Data directory (default: auto-detect)")
    parser.add_argument("--timeframe", type=str, default=None,
                        help="Single timeframe to test (e.g., 4H)")
    parser.add_argument("--assets", type=str, nargs="+", default=None,
                        help="Specific assets to test")
    parser.add_argument("--n-eval", type=int, default=50,
                        help="Kronos evaluation points per asset")
    args = parser.parse_args()

    # Auto-detect data directory
    if args.data_dir:
        data_dir = args.data_dir
    else:
        # Check local first, then homeserver mount
        candidates = [
            Path("data/raw/binance-1m"),
            Path("/home/user/ml-data/binance-1m"),  # homeserver
        ]
        data_dir = None
        for c in candidates:
            if c.exists():
                data_dir = str(c)
                break
        if data_dir is None:
            logger.error("No data directory found. Download Binance data first.")
            sys.exit(1)

    logger.info(f"Data directory: {data_dir}")

    timeframes = [args.timeframe] if args.timeframe else None

    if args.baselines or args.all:
        logger.info("=" * 60)
        logger.info("RUNNING BASELINE EXPERIMENTS")
        logger.info("=" * 60)
        results = run_baseline_experiment(
            data_dir=data_dir,
            timeframes=timeframes,
            assets=args.assets,
        )
        logger.info(f"Baseline results: {len(results)} rows")
        if len(results) > 0:
            logger.info(f"\nBest baseline per timeframe:")
            for tf in results["timeframe"].unique():
                tf_results = results[results["timeframe"] == tf]
                best = tf_results.loc[tf_results["accuracy"].idxmax()]
                logger.info(f"  {tf}: {best['model']} accuracy={best['accuracy']:.4f}")

    if args.kronos or args.all:
        logger.info("=" * 60)
        logger.info("RUNNING KRONOS ZERO-SHOT EXPERIMENTS")
        logger.info("=" * 60)
        results = run_kronos_experiment(
            data_dir=data_dir,
            timeframes=timeframes,
            assets=args.assets,
            n_eval=args.n_eval,
        )
        logger.info(f"Kronos results: {len(results)} rows")

    if not (args.baselines or args.kronos or args.all):
        parser.print_help()


if __name__ == "__main__":
    main()
