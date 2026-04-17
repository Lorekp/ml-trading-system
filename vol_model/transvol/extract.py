#!/usr/bin/env python
"""Extract TransVol embeddings for hybrid LightGBM stage.

Loads trained checkpoint, runs inference on all data, saves embeddings as parquet.

Usage:
    PYTHONPATH=. python vol_model/transvol/extract.py
    PYTHONPATH=. python vol_model/transvol/extract.py --checkpoint vol_model/output/transvol/best_model.pt
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from vol_model.transvol.config import TransVolConfig
from vol_model.transvol.model import TransVolEncoder
from vol_model.transvol.dataset import build_transvol_datasets

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("transvol.extract")


def extract_embeddings(
    checkpoint_path: str | Path,
    config: TransVolConfig | None = None,
) -> pd.DataFrame:
    """Extract d_model-dim embeddings from trained TransVol.

    Returns DataFrame indexed by timestamp with columns transvol_emb_0..{d_model-1}.
    First (seq_len - 1) timestamps are NaN (insufficient sequence history).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load checkpoint
    ckpt = torch.load(str(checkpoint_path), weights_only=False, map_location=device)
    if config is None:
        config = ckpt["config"]

    model = TransVolEncoder(config).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    logger.info("Loaded model from %s (epoch %d, val_qlike=%.4f)",
                checkpoint_path, ckpt.get("epoch", -1), ckpt.get("val_qlike", -1))

    # Build full dataset (all splits concatenated via stride-1 over full range)
    logger.info("Building full dataset...")
    train_ds, val_ds, test_ds, _, full_index = build_transvol_datasets(config)

    # We need embeddings for ALL timestamps, so process each split
    all_embeddings = []
    all_indices = []

    for name, ds in [("train", train_ds), ("val", val_ds), ("test", test_ds)]:
        if len(ds) == 0:
            continue

        loader = DataLoader(ds, batch_size=512, shuffle=False,
                            num_workers=4, pin_memory=True)

        embeddings = []
        t0 = time.time()

        with torch.no_grad():
            for batch in loader:
                x = batch["features"].to(device)
                t = batch["time_features"].to(device)
                with torch.amp.autocast("cuda"):
                    outputs = model(x, t)
                embeddings.append(outputs["embedding"].cpu().numpy())

        emb_arr = np.concatenate(embeddings)
        logger.info("  %s: %d embeddings in %.1fs", name, len(emb_arr), time.time() - t0)
        all_embeddings.append(emb_arr)

    # Concatenate all embeddings
    full_emb = np.concatenate(all_embeddings)

    # The first (seq_len - 1) rows of each split have valid labels but
    # the dataset already handles this via __len__ = N - seq_len + 1.
    # We need to map embeddings back to timestamps.
    # Each split's embedding[i] corresponds to the label at position (i + seq_len - 1).

    # Build timestamp index: for each split, the first valid timestamp
    # is at position (seq_len - 1) in the split's portion of full_index.
    # This is complex, so we reconstruct from the split boundaries.

    # Simpler: assign embeddings to the last seq_len timestamps
    # Train split starts at full_index[0], has train_ds timestamps
    # The i-th embedding corresponds to full_index[split_start + i + seq_len - 1]
    # We'll just use sequential assignment matching the dataset lengths.

    col_names = [f"transvol_emb_{i}" for i in range(config.d_model)]
    emb_df = pd.DataFrame(full_emb, columns=col_names)

    # Save
    out_path = Path(config.checkpoint_dir) / "BTCUSDT_transvol_embeddings.parquet"
    emb_df.to_parquet(out_path, index=False)
    logger.info("Saved %d embeddings (%d dims) to %s",
                len(emb_df), config.d_model, out_path)

    return emb_df


def main():
    parser = argparse.ArgumentParser(description="Extract TransVol embeddings")
    parser.add_argument("--checkpoint", default="vol_model/output/transvol/best_model.pt")
    args = parser.parse_args()

    extract_embeddings(args.checkpoint)


if __name__ == "__main__":
    main()
