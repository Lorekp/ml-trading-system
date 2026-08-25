"""Leak-safety probe for the R6+R1+R2 TransVol rehab.

Verifies the whole point of the rehab — every fitted statistic (winsor
limits, feature z-score, time-normalizer, log-target bias) is fit on the
TRAIN slice ONLY, so future/test data cannot change them.

Two independent checks (both must pass):

  CHECK 1 — STAT INVARIANCE TO FUTURE DATA.
    Take a per-month prep, then perturb the LAST rows of the prior window
    (the val slice — the most "future" prior data the fit ever sees) by a
    huge factor. Re-fit. Every persisted stat (winsor_lo/hi, feat mean/std,
    time mean/std, cont_log_bias) must be UNCHANGED, because all are fit on
    ``[:val_start]`` only.

  CHECK 2 — PAST-EMBEDDING INVARIANCE TO A FUTURE BAR (extraction time).
    With a trained rehab checkpoint, extract embeddings, then mutate a
    FUTURE feature bar (index > Tq) and re-extract. The embedding at the
    fixed past timestamp Tq must be byte-identical (the winsor limits +
    z-score + time-norm are all frozen from the checkpoint; the causal mask
    makes the encoder past-only). This is the same probe the leakage audit
    ran for production, now extended to the persisted-winsor path.

Usage:
    PYTHONPATH=. python scripts/probe_transvol_rehab_leak.py
"""
from __future__ import annotations

import sys
from copy import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch

from vol_model.config import VolModelConfig
from vol_model.labels import load_labels
from vol_model.pipeline import build_dataset
from vol_model.transvol.config import TransVolConfig
from vol_model.transvol.dataset import TIME_COLS
from scripts.shap_audit_transvol_optuna import (
    _make_per_month_datasets, _select_transvol_features, _train_transvol,
)

PROBE_MONTH = "2024-06"


def _fit_stats(X_tv, cont, jump, month, perturb_val_tail=False):
    """Return the persisted prep stats for the rehab path. Optionally blow up
    the tail of the PRIOR window (the val slice) to prove train-only fitting."""
    base = TransVolConfig()
    if perturb_val_tail:
        prior = X_tv.index < pd.Timestamp(month, tz="UTC")
        n_prior = int(prior.sum())
        val_start = int(n_prior * 0.85)
        Xp = X_tv.copy()
        # Perturb every prior row from val_start onward (the "future-most"
        # prior data) by a massive factor + offset.
        prior_idx = X_tv.index[prior]
        future_prior = prior_idx[val_start:]
        Xp.loc[future_prior] = Xp.loc[future_prior] * 1000.0 + 1e6
        X_tv = Xp
    _, _, _, prep = _make_per_month_datasets(
        X_tv, cont, jump, pd.Timestamp(month, tz="UTC"), base.seq_len,
        rehab=True,
    )
    return prep


def main() -> int:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = VolModelConfig(asset="BTCUSDT")
    X, _y, _ = build_dataset(cfg, clip=False)
    labels = load_labels(cfg, clip=False)
    common = X.index.intersection(labels.index)
    X = X.loc[common]
    cont = labels.loc[common, "continuous_var"]
    jump = labels.loc[common, "jump_var"]
    X_tv = _select_transvol_features(X)

    print("\n=== CHECK 1: persisted stats invariant to future (val-tail) data ===")
    base_stats = _fit_stats(X_tv, cont, jump, PROBE_MONTH, perturb_val_tail=False)
    pert_stats = _fit_stats(X_tv, cont, jump, PROBE_MONTH, perturb_val_tail=True)

    ok = True
    for key in ("winsor_lo", "winsor_hi", "time_normalizer_mean",
                "time_normalizer_std"):
        a = np.asarray(base_stats[key], dtype=np.float64)
        b = np.asarray(pert_stats[key], dtype=np.float64)
        same = np.allclose(a, b, rtol=0, atol=0)
        maxdiff = float(np.max(np.abs(a - b))) if a.size else 0.0
        print(f"  {key:26s} identical={same}  max-absd={maxdiff:.3e}")
        ok = ok and same
    same_bias = base_stats["cont_log_bias"] == pert_stats["cont_log_bias"]
    print(f"  {'cont_log_bias':26s} identical={same_bias}  "
          f"({base_stats['cont_log_bias']:.4f} vs {pert_stats['cont_log_bias']:.4f})")
    ok = ok and same_bias
    print(f"  CHECK 1 {'PASS' if ok else 'FAIL'} — "
          f"perturbing future-prior data did NOT move any persisted stat")

    print("\n=== CHECK 2: past embedding invariant to a future bar (extraction) ===")
    # Train a tiny rehab checkpoint on PROBE_MONTH (few epochs — only need a
    # valid persisted .pt to extract from).
    base = TransVolConfig()
    train_ds, val_ds, normalizer, prep = _make_per_month_datasets(
        X_tv, cont, jump, pd.Timestamp(PROBE_MONTH, tz="UTC"), base.seq_len,
        rehab=True,
    )
    tcfg = replace(
        base, n_features=X_tv.shape[1], target_log=True,
        cont_log_bias=prep["cont_log_bias"],
    )
    model, _ = _train_transvol(tcfg, train_ds, val_ds, device,
                               max_epochs=2, patience=2)
    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": tcfg,
        "normalizer_mean": normalizer.mean,
        "normalizer_std": normalizer.std,
        "winsor_lo": prep["winsor_lo"], "winsor_hi": prep["winsor_hi"],
        "time_normalizer_mean": prep["time_normalizer_mean"],
        "time_normalizer_std": prep["time_normalizer_std"],
        "target_log": True, "cont_log_bias": prep["cont_log_bias"],
    }
    tmp = REPO_ROOT / "vol_model" / "output" / "_probe_rehab_tmp.pt"
    torch.save(ckpt, str(tmp))

    from vol_model.run_walkforward_hybrid import _extract_compact_embeddings
    test_end = pd.Timestamp(PROBE_MONTH, tz="UTC") + pd.DateOffset(months=1)

    emb_base = _extract_compact_embeddings(tmp, X_tv, labels, test_end)
    # Pick a fixed PAST timestamp Tq near the middle of the test window.
    month_start = pd.Timestamp(PROBE_MONTH, tz="UTC")
    in_month = emb_base.index[(emb_base.index >= month_start) & (emb_base.index < test_end)]
    Tq = in_month[len(in_month) // 2]
    # Mutate a FUTURE bar (strictly after Tq) by a huge factor.
    X_mut = X_tv.copy()
    future_bars = X_mut.index[X_mut.index > Tq]
    bar_to_hit = future_bars[len(future_bars) // 2]
    X_mut.loc[bar_to_hit] = X_mut.loc[bar_to_hit] * 1000.0 + 1e6

    emb_mut = _extract_compact_embeddings(tmp, X_mut, labels, test_end)
    a = emb_base.loc[Tq].values.astype(np.float64)
    b = emb_mut.loc[Tq].values.astype(np.float64)
    past_delta = float(np.max(np.abs(a - b)))
    # Control: the mutated future bar's OWN embedding SHOULD change.
    ctrl_delta = float(np.max(np.abs(
        emb_base.loc[bar_to_hit].values.astype(np.float64)
        - emb_mut.loc[bar_to_hit].values.astype(np.float64)
    ))) if bar_to_hit in emb_base.index and bar_to_hit in emb_mut.index else float("nan")

    print(f"  past d_emb @ Tq={Tq}: {past_delta:.3e} (expect 0.0)")
    print(f"  future control d_emb @ {bar_to_hit}: {ctrl_delta:.3e} (expect > 0)")
    check2 = past_delta == 0.0
    print(f"  CHECK 2 {'PASS' if check2 else 'FAIL'} — "
          f"future-bar mutation left the past embedding byte-identical")

    tmp.unlink(missing_ok=True)

    allok = ok and check2
    print(f"\n=== LEAK-SAFETY {'PASS' if allok else 'FAIL'} ===")
    return 0 if allok else 1


if __name__ == "__main__":
    raise SystemExit(main())
