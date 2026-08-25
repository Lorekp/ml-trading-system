"""Isolate the mechanism behind the P1 past-embedding leak.

Three conditions, each mutating ONLY future bars (index > T) inside the test
window, then measuring max|d_emb| at rows <= T (past) and > T (future):

  A) mutate ALL feature cols, deployed ckpt (legacy time-normalizer fallback)
  B) mutate NON-TIME cols only (leave duration_s, n_bars_in_window intact),
     deployed ckpt
  C) mutate ALL feature cols, but inject a PERSISTED time normalizer into the
     in-memory ckpt (simulates the cycle-25 fix) so extraction does NOT re-fit
     time mean/std on the window.

If past-leak vanishes in B and C but is present in A, the mechanism is the
time-feature-normalizer re-fit on the extraction window (architectural, time
features only). If past-leak persists in B and C, the transformer/window itself
leaks future->past (a real causal-attention bug).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import torch

from vol_arb.paths import transvol_dir
from vol_model.config import VolModelConfig
from vol_model.labels import load_labels
from vol_model.pipeline import build_dataset
from vol_model.transvol.dataset import TIME_COLS
import vol_model.run_walkforward_hybrid as wh


def extract(ckpt_path, X_tv, labels, end_ts, time_norm_override=None):
    """Replicate _extract_compact_embeddings but allow injecting a persisted
    time normalizer to simulate the cycle-25 fix."""
    ck = torch.load(str(ckpt_path), weights_only=False, map_location="cpu")
    if time_norm_override is not None:
        ck = dict(ck)
        ck["time_normalizer_mean"] = time_norm_override[0]
        ck["time_normalizer_std"] = time_norm_override[1]
    from vol_model.transvol.model import TransVolEncoder
    from vol_model.transvol.dataset import FeatureNormalizer, TransVolDataset
    from torch.utils.data import DataLoader
    config = ck["config"]
    model = TransVolEncoder(config)
    model.load_state_dict(ck["model_state_dict"])
    model.eval()
    normalizer = FeatureNormalizer()
    normalizer.mean = ck["normalizer_mean"]
    normalizer.std = ck["normalizer_std"]
    common = X_tv.index.intersection(labels.index)
    feats = X_tv.loc[common].iloc[:-1]
    cont_var = labels["continuous_var"].loc[common].iloc[1:]
    feats.index = cont_var.index
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mask = feats.index <= end_ts
    feats = feats[mask]
    feat_norm = normalizer.transform(feats.values.astype(np.float32))
    time_norm_vals = feats[TIME_COLS].values.astype(np.float32)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tmean, tstd = wh._resolve_time_normalizer(ck, time_norm_vals)
    time_norm = ((time_norm_vals - tmean) / tstd).astype(np.float32)
    dummy = np.zeros(len(feats), dtype=np.float32)
    ds = TransVolDataset(feat_norm, time_norm, dummy, dummy, config.seq_len)
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0)
    embs = []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["features"], batch["time_features"])
            embs.append(out["embedding_compact"].cpu().numpy())
    emb = np.concatenate(embs)
    start_idx = config.seq_len - 1
    valid_index = feats.index[start_idx: start_idx + len(emb)]
    cols = [f"tv_{i}" for i in range(emb.shape[1])]
    return pd.DataFrame(emb, index=valid_index, columns=cols)


def run_condition(name, X_tv, labels, ckpt, M, test_end, T, mutate_cols,
                  time_norm_override=None):
    Xm = X_tv.copy()
    fut = (Xm.index > T) & (Xm.index < test_end)
    Xm.loc[fut, mutate_cols] = Xm.loc[fut, mutate_cols] * 1000.0 + 1e6
    base = extract(ckpt, X_tv, labels, test_end, time_norm_override)
    mut = extract(ckpt, Xm, labels, test_end, time_norm_override)
    base = base[(base.index >= M) & (base.index < test_end)]
    mut = mut[(mut.index >= M) & (mut.index < test_end)]
    common = base.index.intersection(mut.index)
    past = common[common <= T]
    futc = common[common > T]
    dp = float(np.nanmax(np.abs(base.loc[past].values - mut.loc[past].values))) if len(past) else float("nan")
    df = float(np.nanmax(np.abs(base.loc[futc].values - mut.loc[futc].values))) if len(futc) else float("nan")
    print(f"[{name}] max|d_emb| past(<=T)={dp:.3e}  future(>T)={df:.3e}  "
          f"-> past {'LEAK' if dp > 1e-5 else 'clean'}")
    return dp, df


def main() -> None:
    cfg = VolModelConfig(asset="BTCUSDT")
    X, y, _ = build_dataset(cfg, clip=False)
    labels = load_labels(cfg, clip=False)
    drop_prefixes = ("ofi_", "depth_mid", "oi_change", "oi_zscore", "oi_acceleration")
    tv_cols = [c for c in X.columns if not c.startswith(drop_prefixes)]
    X_tv = X[tv_cols]
    month = "2024-06"
    M = pd.Timestamp(month + "-01", tz="UTC")
    test_end = M + pd.DateOffset(months=1)
    ckpt = transvol_dir("BTCUSDT", 15) / f"{month}_transvol.pt"

    base = extract(ckpt, X_tv, labels, test_end)
    base = base[(base.index >= M) & (base.index < test_end)]
    T = base.index[len(base.index) // 2]
    print(f"T = {T}; test-month rows = {len(base)}")

    nontime_cols = [c for c in X_tv.columns if c not in TIME_COLS]
    all_cols = list(X_tv.columns)

    # Fixed (clean) time normalizer to simulate cycle-25 fix: fit on a prior slice.
    prior = X_tv[X_tv.index < M]
    tn_mean = prior[TIME_COLS].values.astype(np.float32).mean(axis=0)
    tn_std = prior[TIME_COLS].values.astype(np.float32).std(axis=0).clip(min=1e-8)

    print("\nCondition A: mutate ALL cols, deployed ckpt (legacy time-norm fallback)")
    run_condition("A", X_tv, labels, ckpt, M, test_end, T, all_cols)
    print("\nCondition B: mutate NON-TIME cols only, deployed ckpt")
    run_condition("B", X_tv, labels, ckpt, M, test_end, T, nontime_cols)
    print("\nCondition C: mutate ALL cols, with PERSISTED time normalizer (cycle-25 fix simulated)")
    run_condition("C", X_tv, labels, ckpt, M, test_end, T, all_cols,
                  time_norm_override=(tn_mean, tn_std))
    print("\nCondition D: mutate NON-TIME cols only, with persisted time norm (cycle-25 fix)")
    run_condition("D", X_tv, labels, ckpt, M, test_end, T, nontime_cols,
                  time_norm_override=(tn_mean, tn_std))


if __name__ == "__main__":
    main()
