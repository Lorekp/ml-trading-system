"""Multi-timeframe signal quality comparison.

Loads OOS predictions from each timeframe's spot-check models and
computes standardized quality metrics for comparison:

- QLIKE loss (lower = better)
- R-squared (higher = better)
- Direction accuracy (higher = better; >55% is promising)

Usage:
    PYTHONPATH=. python scripts/multi_tf_comparison.py
    PYTHONPATH=. python scripts/multi_tf_comparison.py --tfs 60 300 900 3600
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("multi_tf")

DEFAULT_TFS = [60, 300, 900, 1800, 3600, 14400, 28800, 43200, 86400]


def qlike(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """QLIKE loss: E[exp(y_true - y_pred) - (y_true - y_pred) - 1]."""
    diff = y_true - y_pred
    return float(np.mean(np.exp(diff) - diff - 1.0))


def direction_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, iv: np.ndarray | None = None,
) -> float:
    """Fraction of bars where pred and true agree on direction vs IV.

    If IV is not available, measures sign(y_pred - median(y_pred))
    matching sign(y_true - median(y_true)).
    """
    if iv is not None:
        pred_dir = y_pred > iv
        true_dir = y_true > iv
    else:
        med = np.median(y_true)
        pred_dir = y_pred > med
        true_dir = y_true > med
    return float(np.mean(pred_dir == true_dir))


def r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return float(1.0 - ss_res / max(ss_tot, 1e-20))


def load_predictions(tf_seconds: int, tag_prefix: str = "spot") -> pd.DataFrame | None:
    """Load OOS predictions for a given TF from the spot-check output."""
    tf_min = tf_seconds // 60
    tag = f"{tag_prefix}_{tf_min}m"

    # Try the summary JSON first
    summary_path = Path(f"vol_model/output/models/{tag}/summary.json")
    if summary_path.exists():
        import json
        with open(summary_path) as f:
            summary = json.load(f)
        logger.info("Loaded summary for %dm from %s", tf_min, summary_path)

    # Look for prediction parquets
    pred_path = Path(f"vol_model/output/BTCUSDT_pruned_svi_predictions_{tf_min}m.parquet")
    if not pred_path.exists() and tf_seconds == 900:
        pred_path = Path("vol_model/output/BTCUSDT_pruned_svi_predictions.parquet")
    if not pred_path.exists():
        # Try inline predictions from the walk-forward summary
        model_dir = Path(f"vol_model/output/models/{tag}")
        if not model_dir.exists():
            logger.warning("No models found for %dm at %s", tf_min, model_dir)
            return None
        # Collect per-month test predictions from the model directory
        preds = []
        for txt_path in sorted(model_dir.glob("*.txt")):
            month = txt_path.stem  # e.g. "2026-01"
            pred_csv = model_dir / f"{month}_test_preds.csv"
            if pred_csv.exists():
                preds.append(pd.read_csv(pred_csv, index_col=0, parse_dates=True))
        if preds:
            return pd.concat(preds)
        logger.warning("No predictions found for %dm", tf_min)
        return None

    df = pd.read_parquet(pred_path)
    # Filter to non-NaN predictions (OOS only)
    if "y_pred" in df.columns:
        return df.dropna(subset=["y_pred"])
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Multi-TF signal quality comparison")
    parser.add_argument("--tfs", nargs="+", type=int, default=DEFAULT_TFS,
                        help="Timeframes in seconds to compare")
    parser.add_argument("--tag-prefix", default="spot",
                        help="Model directory prefix (default: spot)")
    args = parser.parse_args()

    results = []
    for tf in args.tfs:
        tf_min = tf // 60
        label = f"{tf_min}m" if tf < 86400 else f"{tf // 3600}h"

        df = load_predictions(tf, tag_prefix=args.tag_prefix)
        if df is None:
            results.append({
                "timeframe": label,
                "tf_seconds": tf,
                "n_bars": 0,
                "qlike": np.nan,
                "r2": np.nan,
                "direction_acc": np.nan,
                "status": "MISSING",
            })
            continue

        yt = df["y_true"].to_numpy()
        yp = df["y_pred"].to_numpy()
        n = len(yt)

        ql = qlike(yt, yp)
        r2 = r_squared(yt, yp)
        da = direction_accuracy(yt, yp)

        results.append({
            "timeframe": label,
            "tf_seconds": tf,
            "n_bars": n,
            "qlike": ql,
            "r2": r2,
            "direction_acc": da,
            "status": "PASS" if da > 0.55 else "FAIL",
        })

    # Print comparison table
    print()
    print("=" * 75)
    print("  MULTI-TIMEFRAME SIGNAL QUALITY COMPARISON")
    print("=" * 75)
    print(f"  {'TF':<8} {'N bars':>8} {'QLIKE':>10} {'R2':>10} {'Dir Acc':>10} {'Gate':>8}")
    print("-" * 75)

    for r in results:
        if r["n_bars"] == 0:
            print(f"  {r['timeframe']:<8} {'---':>8} {'---':>10} {'---':>10} {'---':>10} {'MISSING':>8}")
        else:
            gate = "PASS" if r["direction_acc"] > 0.55 else "FAIL"
            print(
                f"  {r['timeframe']:<8} {r['n_bars']:>8} "
                f"{r['qlike']:>10.4f} {r['r2']:>10.4f} "
                f"{r['direction_acc']:>10.2%} {gate:>8}"
            )
    print("=" * 75)

    # Save results
    out = pd.DataFrame(results)
    out_path = Path("vol_model/output/multi_tf_comparison.csv")
    out.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
