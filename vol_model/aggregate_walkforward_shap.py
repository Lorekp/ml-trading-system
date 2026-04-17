#!/usr/bin/env python
"""Aggregate per-month walk-forward SHAP into a Stage-15.9-equivalent drop list.

After ``run_walkforward_pruned_svi.py --save-shap`` has written
``shap_{YYYY-MM}.csv`` for every trained month under the per-asset model
directory, this script computes the mean |SHAP| per feature across all
months and emits:

  - ``{output_dir}/walkforward_shap_summary.csv`` — per-feature aggregated
    mean |SHAP| across months (with per-feature std + n_months).
  - ``{asset}_drop_features.csv`` — bottom-N of the top-N by aggregated
    mean |SHAP| (same schema as run_shap_analysis.py emits). Overwrites
    the single-train drop list with the walk-forward data-driven version.

The walk-forward aggregation is strictly stronger than the pre-cutoff
single-train SHAP that Stage 15 used originally: it sees every monthly
regime (bull / bear / sideways / high-vol / low-vol) and penalizes features
that are only helpful in one regime. This matches the Stage 15.9 rationale
for BTC's DROP_FEATURES — a feature has to be near-zero SHAP *consistently*
across months, not just on average.

Usage:
    PYTHONPATH=. python vol_model/aggregate_walkforward_shap.py --asset ETHUSDT
    PYTHONPATH=. python vol_model/aggregate_walkforward_shap.py --asset ETHUSDT \
        --prune-top-n 35 --drop-bottom-n 8
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from vol_arb.paths import (
    drop_features_path, pruned_features_path, pruned_svi_model_dir,
    CURRENCY_FOR_ASSET,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("agg_wf_shap")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate walk-forward SHAP into a drop list",
    )
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--model-dir", type=str, default="",
                        help="Override the per-month model dir that contains "
                             "shap_{YYYY-MM}.csv files (else resolver default)")
    parser.add_argument("--prune-top-n", type=int, default=35)
    parser.add_argument("--drop-bottom-n", type=int, default=8,
                        help="BTC baseline 8 — keep parity so no asset prunes deeper")
    parser.add_argument("--output-drop-path", type=str, default="",
                        help="Where to write the drop list (else per-asset default)")
    args = parser.parse_args()

    model_dir = Path(args.model_dir) if args.model_dir else pruned_svi_model_dir(args.asset)
    shap_files = sorted(model_dir.glob("shap_*.csv"))
    if not shap_files:
        raise FileNotFoundError(
            f"No shap_*.csv files under {model_dir}. Run "
            f"run_walkforward_pruned_svi.py --asset {args.asset} --save-shap first.",
        )

    logger.info("Aggregating %d per-month SHAP files from %s", len(shap_files), model_dir)

    frames = []
    for f in shap_files:
        df = pd.read_csv(f)
        month = f.stem.replace("shap_", "")
        df["month"] = month
        frames.append(df)
    wide = pd.concat(frames, ignore_index=True)

    summary = (
        wide.groupby("feature", as_index=False)
        .agg(
            mean_abs_shap=("mean_abs_shap", "mean"),
            std_abs_shap=("mean_abs_shap", "std"),
            n_months=("mean_abs_shap", "count"),
            median_abs_shap=("mean_abs_shap", "median"),
            p10_abs_shap=("mean_abs_shap", lambda s: s.quantile(0.10)),
            p90_abs_shap=("mean_abs_shap", lambda s: s.quantile(0.90)),
        )
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    summary_path = model_dir / "walkforward_shap_summary.csv"
    summary.to_csv(summary_path, index=False)
    logger.info("Saved per-feature aggregate: %s (%d features)", summary_path, len(summary))

    # Build drop list from the bottom-N of the top-N (mirrors run_shap_analysis.py).
    pruned_list = pd.read_csv(pruned_features_path(args.asset), header=None)[0].tolist()
    # Re-rank the pruned-35 by the walk-forward mean |SHAP| (not the single-train SHAP)
    pruned_summary = summary[summary["feature"].isin(pruned_list)].reset_index(drop=True)
    missing = set(pruned_list) - set(pruned_summary["feature"])
    if missing:
        logger.warning(
            "Features in pruned list but missing from walk-forward SHAP (%d): %s",
            len(missing), sorted(missing),
        )

    drop_features = (
        pruned_summary.head(args.prune_top_n)
        .tail(args.drop_bottom_n)["feature"]
        .tolist()
    )
    drop_path = Path(args.output_drop_path) if args.output_drop_path else drop_features_path(args.asset)
    drop_path.parent.mkdir(parents=True, exist_ok=True)
    with open(drop_path, "w", newline="") as f:
        for feat in drop_features:
            f.write(f"{feat}\n")

    total_shap = summary["mean_abs_shap"].sum()
    dropped_share = (
        pruned_summary.tail(args.drop_bottom_n)["mean_abs_shap"].sum() / total_shap * 100
    )
    logger.info(
        "Wrote walk-forward drop list (%d features, %.2f%% of total SHAP): %s",
        len(drop_features), dropped_share, drop_path,
    )
    print("\nWalk-forward drop list (bottom-%d of top-%d pruned, sorted by mean |SHAP|):" % (
        args.drop_bottom_n, args.prune_top_n,
    ))
    for f in drop_features:
        row = pruned_summary[pruned_summary["feature"] == f].iloc[0]
        print(f"  {f:40s}  mean={row['mean_abs_shap']:.6f}  "
              f"std={row['std_abs_shap']:.6f}  p10/p90=[{row['p10_abs_shap']:.6f}, {row['p90_abs_shap']:.6f}]")


if __name__ == "__main__":
    main()
