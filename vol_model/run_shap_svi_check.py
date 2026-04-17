#!/usr/bin/env python
"""SHAP analysis on the hybrid+SVI spot-check models.

For each of the 3 spot-check months (2026-01, 2026-02, 2026-03):
  1. Load the trained hybrid+SVI model
  2. Reconstruct the test-month feature matrix the same way the spot-check did
  3. Compute SHAP values on a sample of the test set
  4. Rank features by mean |SHAP| importance
  5. Report where the 8 SVI features rank and what % of total importance they carry

Output:
  - Per-month ranking table (top 20 + all SVI features)
  - Aggregated ranking across all 3 months
  - Recommendation: keep all SVI features, keep top-N, or drop
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
import torch

from vol_model.config import VolModelConfig
from vol_model.pipeline import build_dataset
from vol_model.transvol.config import TransVolConfig
from vol_model.transvol.model import TransVolEncoder
from vol_model.transvol.dataset import TransVolDataset, FeatureNormalizer, TIME_COLS

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s", datefmt="%H:%M:%S")
logger = logging.getLogger("shap_svi")

SVI_DIR = Path("vol_model/output/models/expanding_hybrid_svi")
HYBRID_DIR = Path("vol_model/output/models/expanding_hybrid")
SVI_FEATURES_PATH = Path("vol_arb/output/svi_features_15m.parquet")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MONTHS = ["2026-01", "2026-02", "2026-03"]
SVI_COLS = [
    "svi_atm_1d", "svi_atm_1w", "svi_atm_2w", "svi_atm_1m",
    "svi_25d_skew_1w", "svi_25d_skew_1m",
    "svi_term_slope_1w_2w", "svi_term_slope_1w_1m",
]


def _extract_compact_embeddings(
    tv_path: Path, X_tv: pd.DataFrame, labels: pd.DataFrame, end_ts: pd.Timestamp,
) -> pd.DataFrame:
    """Replicate the embedding extraction used by the spot-check script."""
    from torch.utils.data import DataLoader

    common = X_tv.index.intersection(labels.index)
    feats = X_tv.loc[common].iloc[:-1]
    cont_var = labels["continuous_var"].loc[common].iloc[1:]
    jump_var = labels["jump_var"].loc[common].iloc[1:]
    feats.index = cont_var.index

    feats = feats.replace([np.inf, -np.inf], np.nan)
    valid = ~feats.isna().any(axis=1) & ~cont_var.isna() & ~jump_var.isna()
    feats = feats[valid]
    feats = feats[feats.index < end_ts]

    feat_vals = feats.values.astype(np.float32)
    time_vals = feats[TIME_COLS].values.astype(np.float32)

    ckpt = torch.load(str(tv_path), map_location="cpu", weights_only=False)
    config = TransVolConfig(n_features=feat_vals.shape[1])
    model = TransVolEncoder(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval().to(DEVICE)

    normalizer = FeatureNormalizer()
    normalizer.mean = ckpt.get("feat_mean", feat_vals.mean(axis=0))
    normalizer.std = ckpt.get("feat_std", feat_vals.std(axis=0).clip(min=1e-8))
    feat_norm = normalizer.transform(feat_vals)

    time_normalizer = FeatureNormalizer()
    time_normalizer.mean = ckpt.get("time_mean", time_vals.mean(axis=0))
    time_normalizer.std = ckpt.get("time_std", time_vals.std(axis=0).clip(min=1e-8))
    time_norm = time_normalizer.transform(time_vals)

    seq_len = config.seq_len
    ds = TransVolDataset(feat_norm, time_norm,
                         np.zeros(len(feat_norm), dtype=np.float32),
                         np.zeros(len(feat_norm), dtype=np.float32),
                         seq_len)
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)

    embeddings = []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(DEVICE)
            t = batch["time_features"].to(DEVICE)
            out = model(x, t)
            embeddings.append(out["embedding_compact"].cpu().numpy())

    emb_arr = np.concatenate(embeddings, axis=0)
    emb_index = feats.index[seq_len - 1:]
    return pd.DataFrame(
        emb_arr, index=emb_index[:len(emb_arr)],
        columns=[f"tv_{i}" for i in range(emb_arr.shape[1])],
    )


def main() -> None:
    # Load base data
    logger.info("Loading base data...")
    vol_config = VolModelConfig()
    X, y, _ = build_dataset(vol_config)
    pruned = pd.read_csv("vol_model/output/pruned_features.csv", header=None)[0].tolist()
    pruned_cols = [c for c in pruned if c in X.columns]
    X_pruned = X[pruned_cols]

    drop_prefixes = ("ofi_", "depth_mid", "oi_change", "oi_zscore", "oi_acceleration")
    tv_cols = [c for c in X.columns if not c.startswith(drop_prefixes)]
    X_tv = X[tv_cols]

    from vol_model.labels import load_labels
    labels = load_labels(vol_config)

    # Load SVI features with the +15min shift (matches spot-check alignment)
    svi_feats = pd.read_parquet(SVI_FEATURES_PATH)
    svi_feats = svi_feats[[c for c in SVI_COLS if c in svi_feats.columns]]
    svi_feats.index = svi_feats.index + pd.Timedelta(minutes=15)

    all_shap_df = []

    for month_str in MONTHS:
        model_path = SVI_DIR / f"{month_str}.txt"
        feat_path = SVI_DIR / f"{month_str}_features.json"
        if not model_path.exists() or not feat_path.exists():
            logger.warning("%s: model not found, skipping", month_str)
            continue

        logger.info("=" * 60)
        logger.info("%s: loading model", month_str)

        model = lgb.Booster(model_file=str(model_path))
        feat_cols = json.loads(feat_path.read_text())

        M = pd.Timestamp(month_str + "-01", tz="UTC")
        test_end = M + pd.DateOffset(months=1)

        # Reconstruct test features exactly as the spot-check did
        tv_path = HYBRID_DIR / f"{month_str}_transvol.pt"
        emb_df = _extract_compact_embeddings(tv_path, X_tv, labels, test_end)

        common = X_pruned.index.intersection(emb_df.index).intersection(svi_feats.index)
        X_month = pd.concat([
            emb_df.loc[common], X_pruned.loc[common], svi_feats.loc[common],
        ], axis=1)

        common_y = X_month.index.intersection(y.index)
        X_month = X_month.loc[common_y]

        test_mask = (X_month.index >= M) & (X_month.index < test_end)
        X_test = X_month[test_mask]
        if X_test.empty:
            logger.warning("%s: empty test set", month_str)
            continue

        # Reorder columns to match the model's training order
        X_test = X_test[feat_cols]

        # SHAP: sample to cap compute time
        n_sample = min(1500, len(X_test))
        X_sample = X_test.sample(n=n_sample, random_state=42)
        logger.info("%s: computing SHAP on %d samples (%d features)", month_str, n_sample, len(feat_cols))

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample.values)

        mean_abs = np.abs(shap_values).mean(axis=0)
        total = mean_abs.sum()

        df = pd.DataFrame({
            "feature": feat_cols,
            "mean_abs_shap": mean_abs,
            "pct_importance": 100 * mean_abs / total if total > 0 else 0,
            "month": month_str,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        df["rank"] = df.index + 1
        all_shap_df.append(df)

        # Per-month summary
        svi_rows = df[df["feature"].isin(SVI_COLS)].copy()
        print(f"\n{'=' * 70}")
        print(f"  {month_str} — Top 20 features + SVI rankings")
        print(f"{'=' * 70}")
        print(f"{'rank':>4s}  {'feature':<35s}  {'mean|SHAP|':>12s}  {'pct':>6s}")
        print("-" * 70)
        for _, row in df.head(20).iterrows():
            marker = " *" if row["feature"] in SVI_COLS else "  "
            print(f"{row['rank']:>4d}{marker}{row['feature']:<35s}  {row['mean_abs_shap']:>12.5f}  {row['pct_importance']:>5.2f}%")

        # Always show all SVI features
        print(f"\n  All SVI features (* = in top 20):")
        print("-" * 70)
        for _, row in svi_rows.iterrows():
            marker = " *" if row["rank"] <= 20 else "  "
            print(f"{row['rank']:>4d}{marker}{row['feature']:<35s}  {row['mean_abs_shap']:>12.5f}  {row['pct_importance']:>5.2f}%")

        svi_total_pct = svi_rows["pct_importance"].sum()
        print(f"\n  SVI features total importance: {svi_total_pct:.2f}% of model")

    # Aggregate across months
    if all_shap_df:
        combined = pd.concat(all_shap_df, ignore_index=True)
        agg = combined.groupby("feature").agg(
            mean_rank=("rank", "mean"),
            median_rank=("rank", "median"),
            mean_pct=("pct_importance", "mean"),
            min_rank=("rank", "min"),
            max_rank=("rank", "max"),
        ).reset_index()
        agg = agg.sort_values("mean_rank")

        print(f"\n\n{'=' * 70}")
        print(f"  AGGREGATED RANKING ACROSS 3 MONTHS")
        print(f"{'=' * 70}")
        print(f"{'rank':>5s}  {'feature':<35s}  {'avg rank':>8s}  {'avg %':>8s}  {'rng':>10s}")
        print("-" * 70)
        for i, (_, row) in enumerate(agg.iterrows()):
            marker = " *" if row["feature"] in SVI_COLS else "  "
            rng = f"{int(row['min_rank'])}-{int(row['max_rank'])}"
            print(f"{i + 1:>5d}{marker}{row['feature']:<35s}"
                  f"  {row['mean_rank']:>8.1f}  {row['mean_pct']:>7.2f}%  {rng:>10s}")

        # SVI summary
        svi_agg = agg[agg["feature"].isin(SVI_COLS)].copy()
        print(f"\n  SVI feature summary:")
        print(f"  - Total SVI features: {len(svi_agg)}")
        print(f"  - Avg rank range: {svi_agg['mean_rank'].min():.1f} - {svi_agg['mean_rank'].max():.1f}")
        print(f"  - Total mean importance: {svi_agg['mean_pct'].sum():.2f}%")
        print(f"  - SVI features in top 10: {(svi_agg['mean_rank'] <= 10).sum()}")
        print(f"  - SVI features in top 20: {(svi_agg['mean_rank'] <= 20).sum()}")
        print(f"  - SVI features in bottom third: {(svi_agg['mean_rank'] > 40).sum()}")

        # Save aggregated CSV
        out_path = Path("vol_model/output/svi_shap_spot_check.csv")
        agg.to_csv(out_path, index=False)
        combined.to_csv(out_path.with_name("svi_shap_per_month.csv"), index=False)
        print(f"\n  Saved: {out_path}")

        # Recommendation
        print(f"\n{'=' * 70}")
        print("  RECOMMENDATION")
        print(f"{'=' * 70}")
        top_svi = svi_agg[svi_agg["mean_rank"] <= 15]
        bot_svi = svi_agg[svi_agg["mean_rank"] > 40]
        if len(top_svi) >= 3:
            print(f"  -> KEEP top SVI features ({len(top_svi)}): " + ", ".join(top_svi["feature"].tolist()))
        if len(bot_svi) >= 1:
            print(f"  -> DROP bottom SVI features ({len(bot_svi)}): " + ", ".join(bot_svi["feature"].tolist()))
        if len(top_svi) < 3 and svi_agg["mean_pct"].sum() < 2.0:
            print(f"  -> CONSIDER DROPPING ALL SVI features (total importance {svi_agg['mean_pct'].sum():.2f}% is too low)")


if __name__ == "__main__":
    main()
