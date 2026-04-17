#!/usr/bin/env python
"""Build features using the dollar-bar-native FeatureEngineer.

Usage:
    PYTHONPATH=. python vol_model/run_engineer.py
    PYTHONPATH=. python vol_model/run_engineer.py --asset ETHUSDT
"""

import argparse
import logging
import time
from pathlib import Path

from vol_model.config import VolModelConfig
from vol_model.features.engineer import FeatureEngineer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("vol_model")


def main():
    parser = argparse.ArgumentParser(description="Build dollar-bar features")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--tf", type=int, default=900)
    args = parser.parse_args()

    config = VolModelConfig(asset=args.asset, freq_s=args.tf)
    out_dir = Path(config.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print("=" * 60, flush=True)
    print(f"  DOLLAR-BAR FEATURE ENGINEER -- {args.asset}", flush=True)
    print("=" * 60, flush=True)

    eng = FeatureEngineer(config)
    features = eng.build()

    tf_label = f"{args.tf // 60}m" if args.tf < 3600 else f"{args.tf // 3600}h"
    out_path = out_dir / f"{args.asset}_features_{tf_label}.parquet"
    features.to_parquet(out_path)

    elapsed = time.time() - t0
    print(f"\n  {len(features)} windows, {len(features.columns)} features", flush=True)
    print(f"  Saved: {out_path}", flush=True)
    print(f"  Runtime: {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
