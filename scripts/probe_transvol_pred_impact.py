"""End-to-end prediction-impact of the TransVol time-normalizer leak.

For one test month, build the production LGBM test matrix two ways:
  (LEAKY)  tv_* embeddings extracted with legacy window-refit time normalizer
           (= deployed production extraction path)
  (FIXED)  tv_* embeddings extracted with a frozen prior-only time normalizer
           (= cycle-25 leakage-safe normalizer)
Run the SAME deployed LGBM checkpoint on both and compare y_pred + QLIKE.

This quantifies whether the leak changes the DEPLOYED forecast materially.
"""
from __future__ import annotations

import json
import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vol_arb.paths import (
    drop_features_path, pruned_features_path, pruned_svi_model_dir,
    transvol_dir, CURRENCY_FOR_ASSET,
)
from vol_model.config import VolModelConfig
from vol_model.labels import load_labels
from vol_model.model import qlike_loss
from vol_model.pipeline import build_dataset
from vol_model.run_walkforward_pruned_svi_har_pd import (
    ADD_HAR_PD_FEATURES, ADD_SVI_FEATURES, _load_har_pd, _load_svi_atm_1d,
)
from vol_model.transvol.dataset import TIME_COLS, FeatureNormalizer, TransVolDataset
from vol_model.transvol.model import TransVolEncoder


def extract_emb(ckpt, feats_lbl, end_ts, time_override):
    config = ckpt["config"]
    model = TransVolEncoder(config); model.load_state_dict(ckpt["model_state_dict"]); model.eval()
    nrm = FeatureNormalizer(); nrm.mean = ckpt["normalizer_mean"]; nrm.std = ckpt["normalizer_std"]
    feats = feats_lbl[feats_lbl.index <= end_ts]
    fn = nrm.transform(feats.values.astype(np.float32))
    tv = feats[TIME_COLS].values.astype(np.float32)
    if time_override is None:
        se = max(len(tv) - 1000, 1)
        tmean = tv[:se].mean(axis=0); tstd = tv[:se].std(axis=0).clip(min=1e-8)
    else:
        tmean, tstd = time_override
    tn = ((tv - tmean) / tstd).astype(np.float32)
    dummy = np.zeros(len(feats), dtype=np.float32)
    ds = TransVolDataset(fn, tn, dummy, dummy, config.seq_len)
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
    embs = []
    with torch.no_grad():
        for b in loader:
            embs.append(model(b["features"], b["time_features"])["embedding_compact"].numpy())
    emb = np.concatenate(embs); si = config.seq_len - 1
    idx = feats.index[si: si + len(emb)]
    return pd.DataFrame(emb, index=idx, columns=[f"tv_{i}" for i in range(emb.shape[1])])


def main():
    asset = "BTCUSDT"; currency = CURRENCY_FOR_ASSET[asset]
    month = "2024-06"
    M = pd.Timestamp(month + "-01", tz="UTC"); test_end = M + pd.DateOffset(months=1)
    V = max(M - pd.DateOffset(months=4), pd.Timestamp("2021-01-04 21:45", tz="UTC") + pd.DateOffset(months=12))

    cfg = VolModelConfig(asset=asset)
    X, y, _ = build_dataset(cfg, clip=False)
    labels = load_labels(cfg, clip=False)
    dp = ("ofi_", "depth_mid", "oi_change", "oi_zscore", "oi_acceleration")
    tv_cols = [c for c in X.columns if not c.startswith(dp)]
    X_tv = X[tv_cols]

    # pruned base (v2) — mirror extract_pruned_svi_har_pd_predictions
    drop_path = drop_features_path(asset)
    if drop_path.exists() and drop_path.stat().st_size > 0:
        drop_set = set(pd.read_csv(drop_path, header=None)[0].tolist())
    else:
        from vol_model.run_walkforward_pruned_svi_har_pd import DROP_FEATURES
        drop_set = set(DROP_FEATURES)
    pruned = pd.read_csv(pruned_features_path(asset), header=None)[0].tolist()
    pruned_cols = [c for c in pruned if c in X.columns and c not in drop_set]
    X_pruned = X[pruned_cols]
    svi_add = _load_svi_atm_1d(currency, horizon_minutes=15)
    har_pd_add = _load_har_pd(asset, horizon_minutes=15)

    # label-indexed feats for emb extraction
    common = X_tv.index.intersection(labels.index)
    feats = X_tv.loc[common].iloc[:-1]
    cont = labels["continuous_var"].loc[common].iloc[1:]
    feats.index = cont.index
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    ckpt = torch.load(str(transvol_dir(asset, 15) / f"{month}_transvol.pt"),
                      weights_only=False, map_location="cpu")
    # frozen prior-only time normalizer
    prior_t = X_tv[X_tv.index < M][TIME_COLS].values.astype(np.float32)
    to_fixed = (prior_t.mean(axis=0), prior_t.std(axis=0).clip(min=1e-8))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb_leaky = extract_emb(ckpt, feats, test_end, None)        # deployed
        emb_fixed = extract_emb(ckpt, feats, test_end, to_fixed)    # cycle-25 fix

    model = lgb.Booster(model_file=str(pruned_svi_model_dir(asset).with_name(
        "expanding_pruned_svi_har_pd_v1") / f"{month}.txt"))
    feat_cols = json.loads((pruned_svi_model_dir(asset).with_name(
        "expanding_pruned_svi_har_pd_v1") / f"{month}_features.json").read_text())

    def build_and_predict(emb_df):
        com = (X_pruned.index.intersection(svi_add.index)
               .intersection(har_pd_add.index).intersection(emb_df.index))
        Xm = pd.concat([emb_df.loc[com], X_pruned.loc[com], svi_add.loc[com], har_pd_add.loc[com]], axis=1)
        cy = Xm.index.intersection(y.index)
        Xm = Xm.loc[cy]; ym = y.loc[cy]
        Xm = Xm.dropna(subset=ADD_SVI_FEATURES + ADD_HAR_PD_FEATURES)
        ym = ym.loc[Xm.index]
        train_mask = Xm.index < V
        test_mask = (Xm.index >= M) & (Xm.index < test_end)
        Xtr = Xm[train_mask]; lo, hi = Xtr.quantile(0.01), Xtr.quantile(0.99)
        Xtest = Xm[test_mask].clip(lo, hi, axis=1); ytest = ym[test_mask]
        for c in feat_cols:
            if c not in Xtest.columns: Xtest[c] = 0.0
        preds = model.predict(Xtest[feat_cols].values)
        return pd.Series(preds, index=Xtest.index), ytest

    p_leaky, ytest = build_and_predict(emb_leaky)
    p_fixed, _ = build_and_predict(emb_fixed)
    com = p_leaky.index.intersection(p_fixed.index)
    pl, pf, yt = p_leaky.loc[com].values, p_fixed.loc[com].values, ytest.loc[com].values

    dpred = np.abs(pl - pf)
    print(f"\n=== Prediction-impact of time-normalizer leak ({month}, n={len(com)}) ===")
    print(f"  max |y_pred_leaky - y_pred_fixed|   : {dpred.max():.3e}")
    print(f"  mean|y_pred_leaky - y_pred_fixed|   : {dpred.mean():.3e}")
    print(f"  std(y_pred)                         : {pl.std():.3e}")
    print(f"  mean|dpred| / std(y_pred)           : {dpred.mean()/max(pl.std(),1e-12):.3%}")
    print(f"  QLIKE leaky : {qlike_loss(yt, pl):.6f}")
    print(f"  QLIKE fixed : {qlike_loss(yt, pf):.6f}")
    print(f"  QLIKE delta (leaky-fixed): {qlike_loss(yt, pl)-qlike_loss(yt, pf):+.6f}")
    corr = np.corrcoef(pl, pf)[0, 1]
    print(f"  corr(y_pred leaky, fixed): {corr:.6f}")


if __name__ == "__main__":
    main()
