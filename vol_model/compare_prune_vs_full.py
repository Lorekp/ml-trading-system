#!/usr/bin/env python
"""A/B comparison: full-35 (no Stage-15.9 drop) vs drop-27 (with drop) predictions.

After running `run_walkforward_pruned_svi.py` twice — once with `--skip-drop
--tag full35` and once without `--skip-drop` (default drop-27) — extract
predictions for both and compute per-regime QLIKE + RankIC.

Regimes examined:
  - Per-year (2022..2026)
  - Per-vol quantile (low / mid / high based on realized TSRV vol)
  - Combined

Usage:
    PYTHONPATH=. python vol_model/compare_prune_vs_full.py --asset ETHUSDT \
        --full-predictions vol_model/output/ETHUSDT_pruned_svi_predictions_full35.parquet \
        --drop-predictions vol_model/output/ETHUSDT_pruned_svi_predictions.parquet

Writes:
  vol_arb/output/{asset}_prune_vs_full_comparison.md
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from vol_model.model import qlike_loss


def _regime_metrics(preds: pd.DataFrame, regime_mask: np.ndarray | None = None) -> dict:
    """Compute QLIKE + RankIC + n_samples on a slice of the predictions."""
    df = preds if regime_mask is None else preds.loc[regime_mask]
    if len(df) < 100:
        return {"qlike": np.nan, "rank_ic": np.nan, "rmse": np.nan, "n": len(df)}
    y_true = df["y_true"].values
    y_pred = df["y_pred"].values
    qlike = qlike_loss(y_true, y_pred)
    rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
    rank_ic = float(spearmanr(y_true, y_pred).statistic) if len(df) > 10 else np.nan
    return {"qlike": qlike, "rank_ic": rank_ic, "rmse": rmse, "n": len(df)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="ETHUSDT")
    parser.add_argument("--full-predictions", required=True,
                        help="Path to predictions with --skip-drop (full 35 features)")
    parser.add_argument("--drop-predictions", required=True,
                        help="Path to predictions with the Stage-15.9 drop applied (27 features)")
    parser.add_argument("--output", default="",
                        help="Markdown summary path (default vol_arb/output/{asset}_prune_vs_full_comparison.md)")
    args = parser.parse_args()

    full = pd.read_parquet(args.full_predictions)
    drop = pd.read_parquet(args.drop_predictions)
    common = full.index.intersection(drop.index)
    full = full.loc[common]
    drop = drop.loc[common]

    # Sanity — y_true must match
    if not np.allclose(full["y_true"].values, drop["y_true"].values, atol=1e-10):
        raise ValueError("y_true column differs between the two prediction files — "
                         "did you accidentally use different datasets?")

    # Per-year regime
    full["year"] = full.index.year
    drop["year"] = drop.index.year

    # Vol-quantile regime from y_true (log-vol)
    q_lo, q_hi = full["y_true"].quantile([0.33, 0.67])
    full["vol_regime"] = np.where(
        full["y_true"] <= q_lo, "low",
        np.where(full["y_true"] <= q_hi, "mid", "high"),
    )
    drop["vol_regime"] = full["vol_regime"].values

    rows = []
    # Overall
    rows.append({"regime": "ALL", "label": "all", **{
        f"{k}_full": v for k, v in _regime_metrics(full).items()
    }, **{f"{k}_drop": v for k, v in _regime_metrics(drop).items()}})

    # Per-year
    for year in sorted(full["year"].unique()):
        yr_mask = full["year"] == year
        rows.append({"regime": "year", "label": str(year),
                     **{f"{k}_full": v for k, v in _regime_metrics(full, yr_mask).items()},
                     **{f"{k}_drop": v for k, v in _regime_metrics(drop, yr_mask).items()}})

    # Per-vol-quantile
    for vr in ["low", "mid", "high"]:
        vr_mask = full["vol_regime"] == vr
        rows.append({"regime": "vol", "label": vr,
                     **{f"{k}_full": v for k, v in _regime_metrics(full, vr_mask).items()},
                     **{f"{k}_drop": v for k, v in _regime_metrics(drop, vr_mask).items()}})

    comp = pd.DataFrame(rows)
    comp["qlike_delta"] = comp["qlike_drop"] - comp["qlike_full"]
    comp["rank_ic_delta"] = comp["rank_ic_drop"] - comp["rank_ic_full"]

    out_path = Path(args.output) if args.output else Path(f"vol_arb/output/{args.asset}_prune_vs_full_comparison.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write(f"# {args.asset}: Full-35 vs Drop-27 SHAP Tail Prune\n\n")
        f.write(f"- full-35 predictions: `{args.full_predictions}` ({len(full):,} rows)\n")
        f.write(f"- drop-27 predictions: `{args.drop_predictions}` ({len(drop):,} rows)\n")
        f.write(f"- Shared timestamps: {len(common):,}\n\n")
        f.write("Lower QLIKE is better; higher RankIC is better. `Δ = drop − full` "
                "(negative QLIKE Δ = pruning helps; positive RankIC Δ = pruning helps).\n\n")
        f.write("| Regime | Label | N | QLIKE full | QLIKE drop | Δ QLIKE | RankIC full | RankIC drop | Δ RankIC |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for _, r in comp.iterrows():
            f.write(f"| {r['regime']} | {r['label']} | {int(r['n_full']):,} | "
                    f"{r['qlike_full']:.4f} | {r['qlike_drop']:.4f} | {r['qlike_delta']:+.4f} | "
                    f"{r['rank_ic_full']:.4f} | {r['rank_ic_drop']:.4f} | {r['rank_ic_delta']:+.4f} |\n")

        # Verdict
        overall_qlike_delta = comp.iloc[0]["qlike_delta"]
        overall_rankic_delta = comp.iloc[0]["rank_ic_delta"]
        verdict = (
            "✓ Prune helps" if overall_qlike_delta < 0 and overall_rankic_delta >= 0 else
            "✗ Prune hurts" if overall_qlike_delta > 0.005 else
            "~ Prune neutral (<0.5% QLIKE shift)"
        )
        f.write(f"\n**Verdict (overall)**: {verdict}\n")
        f.write(f"- QLIKE Δ: {overall_qlike_delta:+.4f}\n")
        f.write(f"- RankIC Δ: {overall_rankic_delta:+.4f}\n")

    print(f"\nWrote comparison: {out_path}")
    print(comp.to_string(index=False))


if __name__ == "__main__":
    main()
