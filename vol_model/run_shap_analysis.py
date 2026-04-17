#!/usr/bin/env python
"""SHAP importance on full feature set for manual pruning decision.

Per-asset — emits BTC output to legacy `shap_126_features.csv` and ETH (or
any non-BTC asset) to `{asset}_shap_126_features.csv` via vol_arb.paths.

Usage:
    PYTHONPATH=. python vol_model/run_shap_analysis.py                 # BTC (default)
    PYTHONPATH=. python vol_model/run_shap_analysis.py --asset ETHUSDT
"""

import argparse
import logging
import time

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap

from vol_arb.paths import drop_features_path, pruned_features_path, shap_features_path
from vol_model.config import VolModelConfig
from vol_model.model import qlike_objective, qlike_loss, qlike_metric
from vol_model.pipeline import build_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("shap")

parser = argparse.ArgumentParser(description="SHAP importance analysis")
parser.add_argument("--asset", default="BTCUSDT",
                    help="Training asset (BTCUSDT or ETHUSDT)")
parser.add_argument("--cutoff", default="2023-10-01",
                    help="Training cutoff date (default matches BTC pruning baseline)")
parser.add_argument("--prune-top-n", type=int, default=35,
                    help="If >0, also write {asset}_pruned_features.csv with the "
                         "top-N features by mean |SHAP| (BTC baseline is 35)")
parser.add_argument("--drop-bottom-n", type=int, default=8,
                    help="If >0, also write {asset}_drop_features.csv with the "
                         "bottom-N of the top-N (by mean |SHAP|) so pruned+SVI "
                         "training can drop Stage-15.9-equivalent tail features. "
                         "BTC baseline is 8 (keeps 27-of-35 going into LGBM); "
                         "matched across assets to avoid over-pruning.")
parser.add_argument("--sample-size", type=int, default=10000,
                    help="Number of random samples drawn across the FULL dataset "
                         "for SHAP (not just pre-cutoff). Default 10K covers all "
                         "regimes across ~5 years; BTC Stage 15 used 5K pre-cutoff.")
parser.add_argument("--exclude-dead-shap", action="store_true",
                    help="Before ranking features into the top-N pruned list, "
                         "drop any feature whose mean |SHAP| is exactly zero "
                         "across the sample. On ETH the 10 OFI/Depth features "
                         "are 100%% NaN and zero-filled, so they register as "
                         "mean |SHAP| = 0; without this flag they occupy slots "
                         "in the top-35 pruned list that could host real "
                         "features. Not lossy: removing zero-signal features "
                         "is guaranteed-safe (no information content).")
args = parser.parse_args()

config = VolModelConfig(asset=args.asset)
X, y, t1 = build_dataset(config)
print(f"[{args.asset}] Dataset: {len(X)} rows, {len(X.columns)} features"
      f"  range={X.index.min()} to {X.index.max()}\n")

# Train on data up to cutoff (pre-cutoff avoids leakage for the *model*;
# SHAP itself is computed on the full dataset below so importance reflects
# behaviour across every month, not just the training regime).
cutoff = pd.Timestamp(args.cutoff, tz="UTC")
train_mask = X.index < cutoff
X_train = X[train_mask]
y_train = y[train_mask]

lo = X_train.quantile(0.01)
hi = X_train.quantile(0.99)
X_tr_w = X_train.clip(lo, hi, axis=1)

val_split = int(len(X_tr_w) * 0.85)
dtrain = lgb.Dataset(X_tr_w.iloc[:val_split], label=y_train.iloc[:val_split].values, free_raw_data=False)
dval = lgb.Dataset(X_tr_w.iloc[val_split:], label=y_train.iloc[val_split:].values, reference=dtrain, free_raw_data=False)

params = {
    "objective": qlike_objective,
    "num_leaves": 31, "max_depth": 6, "learning_rate": 0.03,
    "min_data_in_leaf": 20, "colsample_bytree": 0.8, "subsample": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 1.0,
    "device": "cpu", "num_threads": -1, "verbosity": -1, "seed": 42,
}

logger.info("Training model on %d pre-cutoff samples (cutoff=%s)...", len(X_tr_w), args.cutoff)
model = lgb.train(params, dtrain, num_boost_round=1000,
                  valid_sets=[dval], feval=qlike_metric,
                  callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

# Multi-regime SHAP: sample across the FULL dataset (all months, both
# pre-cutoff train and post-cutoff OOS). Same train-winsorised quantile
# bounds are applied so feature values stay in distribution.
X_full = X.clip(lo, hi, axis=1)
n_sample = min(args.sample_size, len(X_full))
sample_idx = np.random.RandomState(42).choice(len(X_full), n_sample, replace=False)
X_sample = X_full.iloc[sample_idx]
pre_n = int((X_sample.index < cutoff).sum())
post_n = len(X_sample) - pre_n
logger.info("SHAP sample: %d points (%d pre-cutoff, %d post-cutoff) across %d months",
            n_sample, pre_n, post_n,
            (X_sample.index.max() - X_sample.index.min()).days // 30)

t0 = time.time()
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_sample)
logger.info("SHAP done in %.1fs", time.time() - t0)

# Mean absolute SHAP per feature
mean_shap = np.abs(shap_values).mean(axis=0)
feat_shap = sorted(zip(X.columns, mean_shap), key=lambda x: -x[1])

# Also get gain importance for comparison
gain = model.feature_importance(importance_type="gain")
gain_dict = dict(zip(X.columns, gain))

# Print full ranking
print(f"\n{'=' * 90}")
print(f"  FEATURE IMPORTANCE (SHAP + Gain)")
print(f"{'=' * 90}")
print(f"  {'Rank':>4s}  {'Feature':35s}  {'SHAP':>10s}  {'Gain':>12s}  {'Tag':>10s}")
print("-" * 90)

for i, (f, s) in enumerate(feat_shap):
    tag = ""
    if "ofi" in f or "depth" in f:
        tag = "OFI/DEPTH"
    elif "vpin" in f:
        tag = "VPIN"
    elif "oi_" in f:
        tag = "OI"
    g = gain_dict.get(f, 0)
    print(f"  {i+1:4d}  {f:35s}  {s:10.6f}  {g:12.1f}  {tag:>10s}")

# Save to per-asset CSV via resolver
out = pd.DataFrame(feat_shap, columns=["feature", "mean_abs_shap"])
out["gain"] = out["feature"].map(gain_dict)
out_path = shap_features_path(args.asset)
out_path.parent.mkdir(parents=True, exist_ok=True)
out.to_csv(out_path, index=False)
print(f"\nSaved to {out_path}")

# Optionally remove zero-SHAP features (dead features from all-NaN zero-fill)
# from consideration for the pruned list. This is only applied to the ranked
# list used for emitting {asset}_pruned_features.csv — the full SHAP csv
# above is preserved unchanged for transparency / future reference.
if args.exclude_dead_shap:
    n_before = len(out)
    live_mask = out["mean_abs_shap"] > 0
    n_dead = (~live_mask).sum()
    dead_feats = out[~live_mask]["feature"].tolist()
    out_live = out[live_mask].reset_index(drop=True)
    print(
        f"\n[--exclude-dead-shap] Removed {n_dead} zero-SHAP features from "
        f"pruned-list candidate pool: {dead_feats}"
    )
    out_for_prune = out_live
else:
    out_for_prune = out

# Emit pruned feature list (top-N by SHAP) for downstream pruned+SVI training.
if args.prune_top_n > 0:
    top_features = out_for_prune.head(args.prune_top_n)["feature"].tolist()
    pruned_path = pruned_features_path(args.asset)
    pruned_path.parent.mkdir(parents=True, exist_ok=True)
    with open(pruned_path, "w", newline="") as f:
        for feat in top_features:
            f.write(f"{feat}\n")
    print(f"Saved top-{args.prune_top_n} pruned list to {pruned_path} ({len(top_features)} features)")

    # Stage 15.9 SHAP-tail drop list: bottom-N of the top-N by mean |SHAP|.
    # These are features that survived the top-N cut but add <1% SHAP each.
    # BTC drops 8 (hand-curated ffd_rv* + noise_level* cluster); ETH and
    # future assets use the data-driven bottom-N with the same N so pruning
    # depth stays consistent across assets — never more than BTC.
    if args.drop_bottom_n > 0:
        drop_features = out_for_prune.head(args.prune_top_n).tail(args.drop_bottom_n)["feature"].tolist()
        drop_path = drop_features_path(args.asset)
        drop_path.parent.mkdir(parents=True, exist_ok=True)
        with open(drop_path, "w", newline="") as f:
            for feat in drop_features:
                f.write(f"{feat}\n")
        print(f"Saved bottom-{args.drop_bottom_n} drop list to {drop_path}")
        top35 = out_for_prune.head(args.prune_top_n).copy()
        total_shap = out_for_prune["mean_abs_shap"].sum()
        top35["pct"] = 100 * top35["mean_abs_shap"] / total_shap
        dropped_shap_pct = top35.tail(args.drop_bottom_n)["pct"].sum()
        print(f"  Drop list aggregate SHAP: {dropped_shap_pct:.2f}% of total")

# Summary by feature group
print(f"\n{'=' * 70}")
print(f"  SUMMARY BY GROUP")
print(f"{'=' * 70}")
groups = {
    "VPIN (new)": [f for f, _ in feat_shap if "vpin" in f],
    "OFI (new)": [f for f, _ in feat_shap if "ofi" in f],
    "Depth (new)": [f for f, _ in feat_shap if "depth" in f],
    "OI": [f for f, _ in feat_shap if "oi_" in f],
    "Toxicity (other)": [f for f, _ in feat_shap if f in ("duration_s", "n_trades", "imbalance", "vov") or any(f.startswith(b) for b in ("duration_s_", "n_trades_", "imbalance_", "vov_"))],
    "TSRV/Vol": [f for f, _ in feat_shap if any(f.startswith(b) for b in ("tsrv_", "noise_", "bpv_", "parkinson", "rv_zscore", "ffd_rv", "hurst"))],
    "Jump": [f for f, _ in feat_shap if "jump_frac" in f],
    "Funding": [f for f, _ in feat_shap if "funding" in f or "time_to_funding" in f],
}

shap_dict = dict(feat_shap)
for gname, feats in groups.items():
    if not feats:
        continue
    total = sum(shap_dict.get(f, 0) for f in feats)
    avg = total / len(feats) if feats else 0
    best = max(feats, key=lambda f: shap_dict.get(f, 0))
    print(f"  {gname:20s}: {len(feats):3d} feats, total SHAP={total:.6f}, "
          f"avg={avg:.6f}, best={best} ({shap_dict[best]:.6f})")
