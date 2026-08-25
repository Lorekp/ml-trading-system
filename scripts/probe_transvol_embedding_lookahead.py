"""No-lookahead mutation probe for the TransVol embedding column.

AUDIT PROBE (read-only; does not modify pipeline). Replicates the exact
production embedding-extraction path used by
``extract_pruned_svi_har_pd_predictions.py`` ->
``run_walkforward_hybrid._extract_compact_embeddings`` and tests whether the
16-dim embedding at a decision-time timestamp T depends on FUTURE feature bars
(bars strictly after T). If mutating a future bar changes a past embedding,
the feature leaks the future into the decision time.

Two probes:
  P1 (causality / window): mutate a feature row at position > T and verify the
      embedding row at <= T is byte-identical.
  P2 (normalizer leak signature): show that the time-feature normalizer used at
      extraction is re-fit on the extraction window (cycle-25 fallback) for the
      DEPLOYED checkpoints (no persisted time_normalizer_*), i.e. the embedding
      vector for a fixed T DOES change when the extraction END timestamp is
      extended further into the test month. This is the A2-M1 leak.

Usage:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/probe_transvol_embedding_lookahead.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from vol_arb.paths import transvol_dir
from vol_model.config import VolModelConfig
from vol_model.labels import load_labels
from vol_model.pipeline import build_dataset
from vol_model.run_walkforward_hybrid import _extract_compact_embeddings
from vol_model.transvol.config import TransVolConfig  # noqa: F401  (ckpt unpickle)


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
    print(f"checkpoint: {ckpt}")
    print(f"X_tv: {len(X_tv)} rows x {X_tv.shape[1]} cols")

    # ---- Baseline extraction (full test month) ----
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb_base = _extract_compact_embeddings(ckpt, X_tv, labels, test_end)
    emb_base = emb_base[(emb_base.index >= M) & (emb_base.index < test_end)]
    print(f"baseline test-month embeddings: {emb_base.shape}")

    # Pick a decision-time T in the middle of the test month.
    idx = emb_base.index
    T = idx[len(idx) // 2]
    print(f"decision-time T = {T}")

    # ============ P1: future-bar mutation -> past embedding ============
    # Mutate feature rows AFTER T (strictly future), re-extract, compare emb<=T.
    X_tv_mut = X_tv.copy()
    fut_mask = X_tv_mut.index > T
    n_fut = int(fut_mask.sum() & (X_tv_mut.index < test_end).sum())
    # Corrupt all future bars within the test window massively.
    cols_num = X_tv_mut.columns
    fut_in_window = fut_mask & (X_tv_mut.index < test_end)
    X_tv_mut.loc[fut_in_window, cols_num] = (
        X_tv_mut.loc[fut_in_window, cols_num] * 1000.0 + 1e6
    )
    print(f"P1: mutated {int(fut_in_window.sum())} future bars in (T, test_end)")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb_mut = _extract_compact_embeddings(ckpt, X_tv_mut, labels, test_end)
    emb_mut = emb_mut[(emb_mut.index >= M) & (emb_mut.index < test_end)]

    common = emb_base.index.intersection(emb_mut.index)
    past = common[common <= T]
    diff_past = (emb_base.loc[past].values - emb_mut.loc[past].values)
    max_abs_past = float(np.nanmax(np.abs(diff_past))) if len(past) else float("nan")
    # Also future, as a positive control (should differ).
    fut = common[common > T]
    diff_fut = (emb_base.loc[fut].values - emb_mut.loc[fut].values)
    max_abs_fut = float(np.nanmax(np.abs(diff_fut))) if len(fut) else float("nan")

    print(f"P1 RESULT: max|d_emb| at rows <= T : {max_abs_past:.3e}  (should be ~0 if causal)")
    print(f"P1 control: max|d_emb| at rows  > T : {max_abs_fut:.3e}  (should be >0 — mutated bars feed these)")
    p1_clean = max_abs_past < 1e-5
    print(f"P1 VERDICT: {'CLEAN (no future->past leak)' if p1_clean else 'LEAK'}")

    # ============ P2: extraction-window normalizer-leak signature ============
    # Re-extract with a SHORTER end (cap extraction at T+offset) and compare the
    # embedding at a fixed earlier timestamp Tq. If the deployed ckpt lacks the
    # persisted time normalizer, the legacy fallback re-fits time mean/std on the
    # extraction window -> embedding at Tq shifts when the window end moves.
    Tq = idx[len(idx) // 4]  # earlier query point, well inside both windows
    end_short = idx[len(idx) // 2]  # shorter extraction end
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        emb_short = _extract_compact_embeddings(ckpt, X_tv, labels, end_short)
    if Tq in emb_short.index and Tq in emb_base.index:
        d2 = float(np.nanmax(np.abs(
            emb_base.loc[Tq].values - emb_short.loc[Tq].values)))
        print(f"P2 RESULT: max|d_emb| at fixed Tq={Tq} when extraction end "
              f"moves {end_short} -> {test_end}: {d2:.3e}")
        print(f"P2 VERDICT: {'window-dependent (normalizer/window leak present)' if d2 > 1e-6 else 'window-independent (clean)'}")
    else:
        print("P2: Tq not present in both windows; skipped")

    # Report whether the deployed checkpoint carries the cycle-25 time normalizer.
    ck = torch.load(str(ckpt), weights_only=False, map_location="cpu")
    has_tn = ("time_normalizer_mean" in ck) and ("time_normalizer_std" in ck)
    print(f"\nDeployed ckpt has persisted time normalizer (cycle-25 fix): {has_tn}")
    print(f"  -> extraction uses {'persisted (clean)' if has_tn else 'LEGACY re-fit fallback (A2-M1 leak)'}")


if __name__ == "__main__":
    main()
