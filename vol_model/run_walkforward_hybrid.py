#!/usr/bin/env python
"""Walk-forward hybrid: per-month TransVol + LightGBM with strict no-leakage.

For each month M:
  1. TransVol: train on [start, V), val on [V, M) where V = M-4months (capped)
  2. Extract compact (16-dim) embeddings for [start, M+1month)
  3. LightGBM Optuna: train on [start, V), val on [V, M) with TransVol-OOS embeddings
  4. Final LightGBM: retrain on full [start, M) with best HPs
  5. Predict month M

Models saved to vol_model/output/models/expanding_hybrid/.
Never touches vol_model/output/models/expanding/.

Usage:
    PYTHONPATH=. python vol_model/run_walkforward_hybrid.py
    PYTHONPATH=. python vol_model/run_walkforward_hybrid.py --start 2023-01 --end 2026-03
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import torch
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from vol_arb.paths import pruned_features_path, transvol_dir
from vol_model.config import VolModelConfig
from vol_model.model import qlike_objective, qlike_loss, qlike_metric
from vol_model.pipeline import build_dataset
from vol_model.run_walkforward import OptunaPatience
from vol_model.transvol.config import TransVolConfig
from vol_model.transvol.model import TransVolEncoder
from vol_model.transvol.dataset import TransVolDataset, FeatureNormalizer, TIME_COLS
from vol_model.transvol.train import train_transvol, get_cosine_schedule_with_warmup

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wf_hybrid")

from vol_model.run_walkforward import _make_objective


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_all_data(asset: str):
    """Load features, labels, and pruned feature list once for the given asset."""
    vol_config = VolModelConfig(asset=asset)
    X, y, t1 = build_dataset(vol_config)

    # Pruned features (per-asset; BTC keeps legacy unsuffixed file)
    pruned_path = pruned_features_path(asset)
    pruned = pd.read_csv(pruned_path, header=None)[0].tolist()
    pruned_cols = [c for c in pruned if c in X.columns]

    # Also need full features for TransVol (101 after dropping OFI/OI)
    drop_prefixes = ("ofi_", "depth_mid", "oi_change", "oi_zscore", "oi_acceleration")
    tv_cols = [c for c in X.columns if not c.startswith(drop_prefixes)]

    # Load raw TSRV targets for TransVol
    from vol_model.labels import load_labels
    labels = load_labels(vol_config)

    return X, y, t1, pruned_cols, tv_cols, labels, vol_config


def _train_transvol_for_month(
    X_tv: pd.DataFrame,
    labels: pd.DataFrame,
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
    month_str: str,
    model_dir: Path,
) -> Path:
    """Train TransVol on [start, train_end), val on [train_end, val_end). Returns checkpoint path."""
    ckpt_dir = model_dir / f"_tv_tmp_{month_str}"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    best_path = ckpt_dir / "best_model.pt"

    # Align features and targets (one-step-ahead)
    common = X_tv.index.intersection(labels.index)
    feats = X_tv.loc[common].iloc[:-1]
    cont_var = labels["continuous_var"].loc[common].iloc[1:]
    jump_var = labels["jump_var"].loc[common].iloc[1:]
    feats.index = cont_var.index

    # Replace inf/nan
    feats = feats.replace([np.inf, -np.inf], np.nan)
    valid = ~feats.isna().any(axis=1) & ~cont_var.isna() & ~jump_var.isna()
    feats = feats[valid]
    cont_var = cont_var[valid]
    jump_var = jump_var[valid]

    # Convert to arrays
    feat_vals = feats.values.astype(np.float32)
    time_feat_vals = feats[TIME_COLS].values.astype(np.float32)
    cont_vals = cont_var.values.astype(np.float32)
    jump_vals = jump_var.values.astype(np.float32)

    # Temporal split
    train_pos = feats.index.searchsorted(train_end)
    val_pos = feats.index.searchsorted(val_end)

    if train_pos < 1000 or val_pos - train_pos < 100:
        logger.warning("  %s: insufficient data (train=%d, val=%d), skipping TransVol",
                        month_str, train_pos, val_pos - train_pos)
        return None

    # Z-normalize (fit on train only)
    normalizer = FeatureNormalizer().fit(feat_vals[:train_pos])
    feat_norm = normalizer.transform(feat_vals)
    time_norm = FeatureNormalizer().fit(time_feat_vals[:train_pos]).transform(time_feat_vals)

    # Normalize jump_var
    jump_mean = max(float(jump_vals[:train_pos].mean()), 1e-15)
    jump_norm = jump_vals / jump_mean

    config = TransVolConfig(n_features=feat_norm.shape[1])
    seq_len = config.seq_len

    # Build datasets
    train_ds = TransVolDataset(feat_norm[:train_pos], time_norm[:train_pos],
                                cont_vals[:train_pos], jump_norm[:train_pos], seq_len)
    val_ds = TransVolDataset(feat_norm[train_pos:val_pos], time_norm[train_pos:val_pos],
                              cont_vals[train_pos:val_pos], jump_norm[train_pos:val_pos], seq_len)

    if len(train_ds) < 100 or len(val_ds) < 50:
        logger.warning("  %s: too few sequences (train=%d, val=%d)", month_str, len(train_ds), len(val_ds))
        return None

    train_loader = DataLoader(train_ds, batch_size=512, shuffle=True,
                               num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1024, shuffle=False,
                             num_workers=0, pin_memory=True)

    # Train
    model = TransVolEncoder(config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5, weight_decay=0.01)
    steps_per_epoch = max(len(train_loader) // config.grad_accum_steps, 1)
    total_steps = steps_per_epoch * 100
    scheduler = get_cosine_schedule_with_warmup(optimizer, 500, total_steps)
    scaler = torch.amp.GradScaler()

    best_val = float("inf")
    no_improve = 0
    patience = 20

    for epoch in range(150):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader):
            x = batch["features"].to(DEVICE)
            t = batch["time_features"].to(DEVICE)
            targets = {"continuous_var": batch["continuous_var"].to(DEVICE),
                        "jump_var": batch["jump_var"].to(DEVICE)}
            with torch.amp.autocast("cuda"):
                out = model(x, t)
                losses = model.compute_loss(out, targets)
                loss = losses["total"] / config.grad_accum_steps
            scaler.scale(loss).backward()
            if (step + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                if step > 0:
                    scheduler.step()
                optimizer.zero_grad(set_to_none=True)

        # Validate
        model.eval()
        val_q = 0.0
        val_n = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["features"].to(DEVICE)
                t = batch["time_features"].to(DEVICE)
                targets = {"continuous_var": batch["continuous_var"].to(DEVICE),
                            "jump_var": batch["jump_var"].to(DEVICE)}
                with torch.amp.autocast("cuda"):
                    out = model(x, t)
                    losses = model.compute_loss(out, targets)
                val_q += losses["qlike"].item() * x.shape[0]
                val_n += x.shape[0]
        val_qlike = val_q / max(val_n, 1)

        if val_qlike < best_val:
            best_val = val_qlike
            torch.save({"model_state_dict": model.state_dict(), "config": config,
                         "normalizer_mean": normalizer.mean, "normalizer_std": normalizer.std,
                         "epoch": epoch + 1, "val_qlike": best_val}, best_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    logger.info("  %s TransVol: val_qlike=%.4f, epochs=%d", month_str, best_val, epoch + 1)
    normalizer.save(ckpt_dir / "normalizer.npz")
    return best_path


def _extract_compact_embeddings(
    ckpt_path: Path,
    X_tv: pd.DataFrame,
    labels: pd.DataFrame,
    end_ts: pd.Timestamp,
) -> pd.Series | pd.DataFrame:
    """Extract 16-dim compact embeddings from TransVol checkpoint."""
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location=DEVICE)
    config = ckpt["config"]
    model = TransVolEncoder(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    # Rebuild normalizer
    normalizer = FeatureNormalizer()
    normalizer.mean = ckpt["normalizer_mean"]
    normalizer.std = ckpt["normalizer_std"]

    # Align and prepare (same as training)
    common = X_tv.index.intersection(labels.index)
    feats = X_tv.loc[common].iloc[:-1]
    cont_var = labels["continuous_var"].loc[common].iloc[1:]
    feats.index = cont_var.index

    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mask = feats.index <= end_ts
    feats = feats[mask]

    feat_norm = normalizer.transform(feats.values.astype(np.float32))
    time_norm_vals = feats[TIME_COLS].values.astype(np.float32)
    # Normalize time features with same approach
    time_mean = time_norm_vals[:max(len(time_norm_vals) - 1000, 1)].mean(axis=0)
    time_std = time_norm_vals[:max(len(time_norm_vals) - 1000, 1)].std(axis=0).clip(min=1e-8)
    time_norm = ((time_norm_vals - time_mean) / time_std).astype(np.float32)

    # Dummy targets (not used for extraction)
    dummy = np.zeros(len(feats), dtype=np.float32)
    ds = TransVolDataset(feat_norm, time_norm, dummy, dummy, config.seq_len)
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0, pin_memory=True)

    embeddings = []
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(DEVICE)
            t = batch["time_features"].to(DEVICE)
            with torch.amp.autocast("cuda"):
                out = model(x, t)
            embeddings.append(out["embedding_compact"].cpu().numpy())

    emb = np.concatenate(embeddings)

    # Map back to timestamps (sequences start at index seq_len-1)
    # Dataset produces len(feats) - seq_len + 1 samples
    # First sample's label = index seq_len-1
    start_idx = config.seq_len - 1
    valid_index = feats.index[start_idx: start_idx + len(emb)]

    cols = [f"tv_{i}" for i in range(emb.shape[1])]
    return pd.DataFrame(emb, index=valid_index, columns=cols)


def main():
    parser = argparse.ArgumentParser(description="Walk-forward hybrid")
    parser.add_argument("--asset", default="BTCUSDT",
                        help="Training asset (BTCUSDT or ETHUSDT). Selects features, "
                             "pruned list, and output model dir via vol_arb.paths.")
    parser.add_argument("--start", default="2022-01")
    parser.add_argument("--end", default="2026-03")
    parser.add_argument("--max-trials", type=int, default=200)
    parser.add_argument("--patience", type=int, default=50)
    args = parser.parse_args()

    model_dir = transvol_dir(args.asset)
    model_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"  WALK-FORWARD HYBRID (strict no-leakage) — {args.asset}")
    print(f"  Checkpoints → {model_dir}")
    print("=" * 70)

    # Load all data once
    logger.info("Loading data...")
    X, y, t1, pruned_cols, tv_cols, labels, vol_config = _load_all_data(args.asset)
    X_pruned = X[pruned_cols]
    X_tv = X[tv_cols]
    data_start = X.index.min()

    logger.info("Data: %d rows, pruned=%d, tv=%d features", len(X), len(pruned_cols), len(tv_cols))

    # Generate month list
    start = pd.Timestamp(args.start + "-01", tz="UTC")
    end = pd.Timestamp(args.end + "-01", tz="UTC")
    months = pd.date_range(start, end, freq="MS")

    results = []

    for i, M in enumerate(months):
        month_str = M.strftime("%Y-%m")
        lgb_path = model_dir / f"{month_str}.txt"

        # Skip if cached
        if lgb_path.exists():
            logger.info("[%d/%d] %s: cached, skipping", i + 1, len(months), month_str)
            continue

        test_end = M + pd.DateOffset(months=1)
        test_mask = (X.index >= M) & (X.index < test_end)
        n_test = test_mask.sum()
        if n_test == 0:
            logger.info("[%d/%d] %s: no test data", i + 1, len(months), month_str)
            continue

        # Define V: OOS val boundary (last 4 months before M, min 12 months training)
        V = max(M - pd.DateOffset(months=4), data_start + pd.DateOffset(months=12))
        if V >= M:
            logger.info("[%d/%d] %s: insufficient history", i + 1, len(months), month_str)
            continue

        t0 = time.time()
        logger.info("[%d/%d] %s: V=%s, train=[%s, %s), val=[%s, %s), test=%s",
                     i + 1, len(months), month_str,
                     V.strftime("%Y-%m"), data_start.strftime("%Y-%m"),
                     V.strftime("%Y-%m"), V.strftime("%Y-%m"),
                     M.strftime("%Y-%m"), month_str)

        # ── Step 1: Train TransVol ──
        tv_path = _train_transvol_for_month(X_tv, labels, V, M, month_str, model_dir)

        if tv_path is None:
            logger.warning("[%d/%d] %s: TransVol failed, using pruned-only", i + 1, len(months), month_str)
            X_month = X_pruned
        else:
            # ── Step 2: Extract compact embeddings ──
            emb_df = _extract_compact_embeddings(tv_path, X_tv, labels, test_end)

            # ── Step 3: Build hybrid features ──
            common = X_pruned.index.intersection(emb_df.index)
            X_month = pd.concat([emb_df.loc[common], X_pruned.loc[common]], axis=1)

        # Align y
        common_y = X_month.index.intersection(y.index)
        X_month = X_month.loc[common_y]
        y_month = y.loc[common_y]

        # Masks
        train_mask = X_month.index < V
        val_mask = (X_month.index >= V) & (X_month.index < M)
        test_mask_m = (X_month.index >= M) & (X_month.index < test_end)

        n_train = train_mask.sum()
        n_val = val_mask.sum()
        n_test_m = test_mask_m.sum()

        if n_train < 100 or n_test_m == 0:
            logger.warning("[%d/%d] %s: insufficient aligned data", i + 1, len(months), month_str)
            continue

        # ── Step 3: Optuna LightGBM ──
        X_tr = X_month[train_mask]
        y_tr = y_month[train_mask]
        X_val = X_month[val_mask]
        y_val = y_month[val_mask]
        X_test = X_month[test_mask_m]
        y_test = y_month[test_mask_m]

        # Winsorize
        lo = X_tr.quantile(0.01)
        hi = X_tr.quantile(0.99)
        X_tr_w = X_tr.clip(lo, hi, axis=1)
        X_val_w = X_val.clip(lo, hi, axis=1)
        X_test_w = X_test.clip(lo, hi, axis=1)

        # Optuna
        dtrain = lgb.Dataset(X_tr_w.values, label=y_tr.values, free_raw_data=False,
                              params={"feature_pre_filter": False})
        dval = lgb.Dataset(X_val_w.values, label=y_val.values, reference=dtrain, free_raw_data=False)
        dtrain.construct()
        dval.construct()

        study = optuna.create_study(direction="minimize", sampler=TPESampler(seed=42),
                                     pruner=MedianPruner(n_startup_trials=5))
        obj = _make_objective(dtrain, dval, X_val_w.values, y_val.values, "cpu")
        study.optimize(obj, n_trials=args.max_trials,
                        callbacks=[OptunaPatience(args.patience)], show_progress_bar=False)

        n_trials = len(study.trials)

        # ── Step 4: Final retrain on full [start, M) ──
        full_train_mask = X_month.index < M
        X_full = X_month[full_train_mask].clip(lo, hi, axis=1)
        y_full = y_month[full_train_mask]

        best = study.best_params.copy()
        best["objective"] = qlike_objective
        best["subsample"] = 0.8
        best["device"] = "cpu"
        best["num_threads"] = -1
        best["verbosity"] = -1
        best["seed"] = 42

        dtrain_full = lgb.Dataset(X_full.values, label=y_full.values, free_raw_data=False)
        model = lgb.train(best, dtrain_full, num_boost_round=2000,
                           valid_sets=[dval], feval=qlike_metric,
                           callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)])

        # ── Step 5: Predict ──
        preds = model.predict(X_test_w.values)
        y_test_arr = y_test.values
        fold_qlike = qlike_loss(y_test_arr, preds)
        fold_rmse = float(np.sqrt(np.mean((y_test_arr - preds) ** 2)))
        rank_ic = float(spearmanr(y_test_arr, preds).statistic)

        # Save
        model.save_model(str(lgb_path))
        if tv_path:
            import shutil
            shutil.copy2(str(tv_path), str(model_dir / f"{month_str}_transvol.pt"))
        (model_dir / f"{month_str}_features.json").write_text(
            json.dumps(list(X_month.columns)))

        elapsed = time.time() - t0
        logger.info("[%d/%d] %s: QLIKE=%.4f RankIC=%.3f (train=%d, val=%d, test=%d, "
                     "trials=%d, feats=%d, %.0fs)",
                     i + 1, len(months), month_str, fold_qlike, rank_ic,
                     n_train, n_val, n_test_m, n_trials, len(X_month.columns), elapsed)

        results.append({
            "month": month_str, "qlike": fold_qlike, "rmse": fold_rmse,
            "rank_ic": rank_ic, "n_train": n_train, "n_val": n_val,
            "n_test": n_test_m, "n_trials": n_trials, "time_s": elapsed,
        })

    # Summary
    if results:
        print(f"\n{'=' * 70}")
        print(f"  RESULTS ({len(results)} months)")
        print(f"{'=' * 70}")
        print(f"  {'Month':>8s}  {'QLIKE':>8s}  {'RankIC':>8s}  {'Trials':>7s}  {'Time':>6s}")
        print("-" * 50)
        for r in results:
            print(f"  {r['month']:>8s}  {r['qlike']:>8.4f}  {r['rank_ic']:>8.3f}"
                  f"  {r['n_trials']:>7d}  {r['time_s']:>5.0f}s")
        avg_q = np.mean([r["qlike"] for r in results])
        avg_ic = np.mean([r["rank_ic"] for r in results])
        print(f"\n  Average: QLIKE={avg_q:.4f}  RankIC={avg_ic:.3f}")

    # Save results
    if results:
        pd.DataFrame(results).to_csv(model_dir / "walkforward_results.csv", index=False)


if __name__ == "__main__":
    main()
