"""Embedding extraction from a trained PatchTST checkpoint.

Mirrors ``run_walkforward_hybrid._extract_compact_embeddings`` exactly in
interface: given a checkpoint path + feature matrix + labels, returns a
DataFrame of 16-dim embeddings indexed to the LABEL timestamp (t+1 of
each sequence's last input window).

Column naming: ``pt_0..pt_15`` (distinct from TransVol's ``tv_0..tv_15``
so the two embedding sets can coexist in a downstream LGBM feature
matrix -- handy for ensemble experiments).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vol_model.patchtst.encoder import PatchTSTEncoder
from vol_model.transvol.dataset import FeatureNormalizer, TIME_COLS, TransVolDataset

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_compact_embeddings(
    ckpt_path: Path,
    X_tv: pd.DataFrame,
    labels: pd.DataFrame,
    end_ts: pd.Timestamp,
    *,
    column_prefix: str = "pt",
) -> pd.DataFrame:
    """Load a PatchTST checkpoint and return 16-dim embeddings on index
    aligned to label timestamps.

    The feature pipeline is IDENTICAL to TransVol's extractor:
    one-step-ahead alignment (feats[:-1] predict labels[1:]), replace
    inf->NaN->0, normalize with the checkpoint's saved mean/std.
    """
    ckpt = torch.load(str(ckpt_path), weights_only=False, map_location=DEVICE)
    config = ckpt["config"]
    model = PatchTSTEncoder(config).to(DEVICE)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    normalizer = FeatureNormalizer()
    normalizer.mean = ckpt["normalizer_mean"]
    normalizer.std = ckpt["normalizer_std"]

    common = X_tv.index.intersection(labels.index)
    feats = X_tv.loc[common].iloc[:-1]
    cont_var = labels["continuous_var"].loc[common].iloc[1:]
    feats.index = cont_var.index

    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    mask = feats.index <= end_ts
    feats = feats[mask]

    feat_norm = normalizer.transform(feats.values.astype(np.float32))
    time_vals = feats[TIME_COLS].values.astype(np.float32)
    # Self-normalize time features from early history (same as TransVol extractor)
    time_mean = time_vals[:max(len(time_vals) - 1000, 1)].mean(axis=0)
    time_std = time_vals[:max(len(time_vals) - 1000, 1)].std(axis=0).clip(min=1e-8)
    time_norm = ((time_vals - time_mean) / time_std).astype(np.float32)

    dummy = np.zeros(len(feats), dtype=np.float32)
    ds = TransVolDataset(feat_norm, time_norm, dummy, dummy, config.seq_len)
    loader = DataLoader(ds, batch_size=1024, shuffle=False, num_workers=0,
                        pin_memory=(DEVICE.type == "cuda"))

    embeddings: list[np.ndarray] = []
    use_amp = DEVICE.type == "cuda"
    with torch.no_grad():
        for batch in loader:
            x = batch["features"].to(DEVICE)
            t = batch["time_features"].to(DEVICE)
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(x, t)
            embeddings.append(out["embedding_compact"].cpu().numpy())

    emb = np.concatenate(embeddings)
    start_idx = config.seq_len - 1
    valid_index = feats.index[start_idx: start_idx + len(emb)]
    cols = [f"{column_prefix}_{i}" for i in range(emb.shape[1])]
    return pd.DataFrame(emb, index=valid_index, columns=cols)
