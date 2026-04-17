#!/usr/bin/env python
"""Compute and cache all feature groups to parquet.

Usage:
    PYTHONPATH="." python vol_model/run_features.py
    PYTHONPATH="." python vol_model/run_features.py --asset ETHUSDT
    PYTHONPATH="." python vol_model/run_features.py --groups memory derivatives  # specific groups only
"""

import argparse
import logging
import time
from pathlib import Path

from vol_model.config import VolModelConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vol_model")


def compute_and_cache(group: str, config: VolModelConfig, out_dir: Path) -> None:
    """Compute a single feature group and save to parquet."""
    out_path = out_dir / f"{config.asset}_features_{group}.parquet"

    t0 = time.time()
    logger.info("[%s] Computing...", group)

    if group == "memory":
        from vol_model.features.memory import compute_memory_features
        df = compute_memory_features(config)

    elif group == "toxicity":
        from vol_model.features.toxicity import compute_toxicity_features
        df = compute_toxicity_features(config)

    elif group == "liquidity":
        from vol_model.features.liquidity import compute_liquidity_features
        df = compute_liquidity_features(config)

    elif group == "derivatives":
        from vol_model.features.derivatives import compute_derivatives_features
        df = compute_derivatives_features(config)

    else:
        raise ValueError(f"Unknown feature group: {group}")

    df.to_parquet(out_path)
    elapsed = time.time() - t0
    logger.info("[%s] Saved %d rows, %d cols to %s (%.0fs)",
                group, len(df), len(df.columns), out_path, elapsed)


def main():
    parser = argparse.ArgumentParser(description="Compute vol model features")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--tf", type=int, default=900, help="Timeframe in seconds")
    parser.add_argument("--groups", nargs="*",
                        default=["memory", "toxicity", "liquidity", "derivatives"],
                        help="Feature groups to compute")
    args = parser.parse_args()

    config = VolModelConfig(asset=args.asset, freq_s=args.tf)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()
    print("=" * 60)
    print(f"  VOL MODEL FEATURES — {args.asset}, {args.tf // 60}m bars")
    print(f"  Groups: {args.groups}")
    print("=" * 60)

    for group in args.groups:
        compute_and_cache(group, config, out_dir)

    elapsed = time.time() - t_start
    print(f"\n  Total runtime: {elapsed:.0f}s")


if __name__ == "__main__":
    main()
