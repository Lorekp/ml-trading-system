"""Fast, surgical isolation of the TransVol embedding leak mechanism.

Restricts extraction to a small window [M - 5 days, test_end] so 4 conditions
run in seconds on CPU. Replicates _extract_compact_embeddings's relevant logic.

Conditions (mutate ONLY future bars index>T inside test window, measure
max|d_emb| at rows <= T):
  A) mutate ALL cols, deployed ckpt (legacy time-norm refit on window)
  B) mutate NON-TIME cols only, deployed ckpt
  C) mutate ALL cols, persisted (frozen prior) time normalizer
  D) mutate NON-TIME cols only, persisted time normalizer

Plus a realistic window-shift test (no mutation): does emb at fixed Tq change
when extraction END moves from mid-month to end-of-month? (the true A2-M1 leak
magnitude, non-adversarial).
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vol_arb.paths import transvol_dir
from vol_model.config import VolModelConfig
from vol_model.labels import load_labels
from vol_model.pipeline import build_dataset
from vol_model.transvol.dataset import TIME_COLS, FeatureNormalizer, TransVolDataset
from vol_model.transvol.model import TransVolEncoder

P = print


def extract(ckpt, feats_all, end_ts, time_override=None):
    """feats_all: DataFrame already indexed at LABEL timestamps (post one-step
    shift), NaN-filled. Returns emb DataFrame for index <= end_ts."""
    config = ckpt["config"]
    model = TransVolEncoder(config)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    nrm = FeatureNormalizer()
    nrm.mean = ckpt["normalizer_mean"]; nrm.std = ckpt["normalizer_std"]
    feats = feats_all[feats_all.index <= end_ts]
    fn = nrm.transform(feats.values.astype(np.float32))
    tv = feats[TIME_COLS].values.astype(np.float32)
    if time_override is not None:
        tmean, tstd = time_override
    else:
        # legacy fallback (no persisted time norm in deployed ckpt)
        se = max(len(tv) - 1000, 1)
        tmean = tv[:se].mean(axis=0); tstd = tv[:se].std(axis=0).clip(min=1e-8)
    tn = ((tv - tmean) / tstd).astype(np.float32)
    dummy = np.zeros(len(feats), dtype=np.float32)
    ds = TransVolDataset(fn, tn, dummy, dummy, config.seq_len)
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
    embs = []
    with torch.no_grad():
        for b in loader:
            embs.append(model(b["features"], b["time_features"])["embedding_compact"].numpy())
    emb = np.concatenate(embs)
    si = config.seq_len - 1
    idx = feats.index[si: si + len(emb)]
    return pd.DataFrame(emb, index=idx, columns=[f"tv_{i}" for i in range(emb.shape[1])])


def main():
    cfg = VolModelConfig(asset="BTCUSDT")
    X, y, _ = build_dataset(cfg, clip=False)
    labels = load_labels(cfg, clip=False)
    dp = ("ofi_", "depth_mid", "oi_change", "oi_zscore", "oi_acceleration")
    tv_cols = [c for c in X.columns if not c.startswith(dp)]
    X_tv = X[tv_cols]
    month = "2024-06"
    M = pd.Timestamp(month + "-01", tz="UTC")
    test_end = M + pd.DateOffset(months=1)
    ckpt = torch.load(str(transvol_dir("BTCUSDT", 15) / f"{month}_transvol.pt"),
                      weights_only=False, map_location="cpu")

    # Build label-indexed feats (mirror _extract_compact_embeddings alignment)
    common = X_tv.index.intersection(labels.index)
    feats = X_tv.loc[common].iloc[:-1]
    cont = labels["continuous_var"].loc[common].iloc[1:]
    feats.index = cont.index
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    # restrict to a small window for speed
    win_start = M - pd.Timedelta(days=5)
    feats = feats[(feats.index >= win_start) & (feats.index < test_end)]
    P(f"window rows: {len(feats)}  ({win_start} .. {test_end})")

    base = extract(ckpt, feats, test_end)
    base_m = base[(base.index >= M) & (base.index < test_end)]
    T = base_m.index[len(base_m) // 2]
    P(f"T = {T}; test rows in window = {len(base_m)}")

    # frozen prior time normalizer (leakage-safe, simulates cycle-25 fix)
    prior = X_tv[X_tv.index < M][TIME_COLS].values.astype(np.float32)
    to = (prior.mean(axis=0), prior.std(axis=0).clip(min=1e-8))

    nontime = [c for c in feats.columns if c not in TIME_COLS]
    allc = list(feats.columns)

    def cond(name, cols, override):
        fm = feats.copy()
        fut = (fm.index > T) & (fm.index < test_end)
        fm.loc[fut, cols] = fm.loc[fut, cols] * 1000.0 + 1e6
        b = extract(ckpt, feats, test_end, override)
        m = extract(ckpt, fm, test_end, override)
        b = b[(b.index >= M) & (b.index < test_end)]
        m = m[(m.index >= M) & (m.index < test_end)]
        c = b.index.intersection(m.index)
        past = c[c <= T]; futc = c[c > T]
        pd_ = float(np.nanmax(np.abs(b.loc[past].values - m.loc[past].values))) if len(past) else float("nan")
        fd_ = float(np.nanmax(np.abs(b.loc[futc].values - m.loc[futc].values))) if len(futc) else float("nan")
        P(f"[{name}] past(<=T)={pd_:.3e} future(>T)={fd_:.3e} -> past {'LEAK' if pd_>1e-5 else 'clean'}")

    P("\nMechanism isolation (future-bar mutation -> past embedding):")
    cond("A all-cols, legacy time-norm", allc, None)
    cond("B nontime-cols, legacy time-norm", nontime, None)
    cond("C all-cols, frozen time-norm (fix)", allc, to)
    cond("D nontime-cols, frozen time-norm (fix)", nontime, to)

    # Realistic non-adversarial window-shift leak magnitude
    P("\nRealistic A2-M1 leak (window-shift, NO mutation):")
    Tq = base_m.index[len(base_m)//4]
    end_mid = base_m.index[len(base_m)//2]
    b_full = extract(ckpt, feats, test_end)            # extraction end = test_end
    b_mid = extract(ckpt, feats, end_mid)              # extraction end = mid-month
    if Tq in b_full.index and Tq in b_mid.index:
        d = float(np.nanmax(np.abs(b_full.loc[Tq].values - b_mid.loc[Tq].values)))
        # relative to embedding scale
        scale = float(np.nanmean(np.abs(b_full.loc[Tq].values)))
        P(f"  legacy: max|d_emb| at Tq when end moves mid->test_end: {d:.3e} "
          f"(emb scale ~{scale:.3e}, rel {d/max(scale,1e-12):.2%})")
    b_full_fx = extract(ckpt, feats, test_end, to)
    b_mid_fx = extract(ckpt, feats, end_mid, to)
    if Tq in b_full_fx.index and Tq in b_mid_fx.index:
        d2 = float(np.nanmax(np.abs(b_full_fx.loc[Tq].values - b_mid_fx.loc[Tq].values)))
        P(f"  fixed : max|d_emb| at Tq when end moves mid->test_end: {d2:.3e} (should be 0)")


if __name__ == "__main__":
    main()
