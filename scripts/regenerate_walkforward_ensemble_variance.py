"""Regenerate the K=10 deep-ensemble σ²_epi parquet under STRICT walk-forward
year folds — Cycle 4 Phase A.0 artifact.

Cycle 3's existing ``vol_model/output/BTC_v2_har_pd_ensemble_var.parquet``
was produced with ``train_purged_cv`` and ``n_splits=5`` over the FULL span
2021-01 → 2026-03 (verified: ``meta.json`` ``"n_splits_purged_cv": 5``).
PurgedKFold prevents adjacent-bar leakage but does NOT enforce strict
time-causality across years: a 2024 OOF prediction was made by a model
whose training set in some folds spans 2025. For Cycle 4 we need σ²_epi
that mirrors the DFL Head's expanding-window-by-year contract — train_y < test_y.

Output: ``vol_model/output/BTC_v2_har_pd_ensemble_var_walkforward.parquet``
with the same column schema as the Cycle 3 artifact:
``(ensemble_mean, sigma_epi, sigma_alea, sigma_total, spread, pi_lo,
pi_hi, y_true)``. Refused-on-overwrite via ``freeze_uncertainty``.

For each test year T ∈ {2023, 2024, 2025}:
1. Build dataset via ``vol_model.pipeline.build_dataset`` (production
   feature config, freq_s=900, n_estimators=1000, n_splits=5).
2. Filter ``(X, y, t1)`` to ``index < T-01-01 UTC``.
3. Pre-winsorize the pre-T slice using GLOBAL pre-T quantiles
   (matches ``config.winsorize_lo``, ``winsorize_hi``); apply the
   SAME quantiles to year-T data before predicting (walk-forward
   causality of winsorize parameters).
4. Train K=10 ensemble via ``train_har_pd_ensemble`` (5 PurgedKFold
   folds × 10 members → 50 boosters). Inside each fold, internal
   winsorize on already-winsorized data is effectively a no-op clip.
5. Per member, predict on year-T data: average ``model.predict``
   across the 5 fold-models → 1 prediction per bar per member.
6. Stack K=10 per-member predictions → ``(K, n_T)`` matrix → variance
   across K = σ²_epi for year T.
7. Combine with HQR aleatoric using ``cal_window = (2021-04-01,
   min(T-01-01, 2023-10-01))`` (cal slice strictly inside the
   training data — no test-year contamination).
8. Concatenate per-year frames; year 2022 + year 2026 rows = NaN
   (no train history / partial year).

Tests: ``tests/test_walkforward_ensemble_regeneration.py``.

Run::

    # Smoke test (~10 min): K=2, n_estimators=100, 1 test year
    PYTHONPATH=. python scripts/regenerate_walkforward_ensemble_variance.py --smoke

    # Full run (~25-45 min): K=10, n_estimators=1000, 3 test years
    PYTHONPATH=. python scripts/regenerate_walkforward_ensemble_variance.py
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vol_model.config import VolModelConfig
from vol_model.pipeline import build_dataset
from vol_model.uncertainty import (
    CAL_WINDOW,
    EnsembleParams,
    combine_with_hqr,
    freeze_uncertainty,
    predict_ensemble_var,
    train_har_pd_ensemble,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("regenerate_walkforward_ensemble_variance")


# ─── Pre-committed parameters (locked in plan) ─────────────────────────

DEFAULT_OUT = Path("vol_model/output/BTC_v2_har_pd_ensemble_var_walkforward.parquet")
DEFAULT_TEST_YEARS: tuple[int, ...] = (2023, 2024, 2025)
DEFAULT_N_ESTIMATORS = 1000
DEFAULT_K = 10
SMOKE_TEST_YEARS: tuple[int, ...] = (2024,)
SMOKE_K = 2
SMOKE_N_ESTIMATORS = 100


# ─── Per-test-year fold ─────────────────────────────────────────────────


def _train_fold_and_predict_year(
    X_full: pd.DataFrame,
    y_full: pd.Series,
    t1_full: pd.Series,
    test_year: int,
    *,
    base_config: VolModelConfig,
    K: int,
    min_train_bars: int = 5_000,
) -> dict[str, np.ndarray]:
    """Train walk-forward K-member ensemble for ``test_year`` and predict.

    Returns dict with arrays of length ``len(X_test_year)``:
      ensemble_mean, sigma_epi, sigma_alea, sigma_total, spread,
      pi_lo, pi_hi, y_true_year, timestamps (DatetimeIndex)

    Anti-leakage:
    - Training data strictly precedes ``test_year`` start.
    - Pre-winsorize uses ONLY pre-test-year quantiles.
    - HQR cal window is inside training data (does not overlap test_year).
    """
    test_start = pd.Timestamp(f"{test_year}-01-01", tz="UTC")
    test_end = pd.Timestamp(f"{test_year + 1}-01-01", tz="UTC")

    # Slice training (pre-test-year) and test (within test-year) data
    train_mask = X_full.index < test_start
    test_mask = (X_full.index >= test_start) & (X_full.index < test_end)
    X_train = X_full.loc[train_mask].copy()
    y_train = y_full.loc[train_mask].copy()
    t1_train = t1_full.loc[train_mask].copy()
    X_test = X_full.loc[test_mask].copy()
    y_test = y_full.loc[test_mask].copy()

    if len(X_train) < min_train_bars:
        raise ValueError(
            f"Test year {test_year}: only {len(X_train)} pre-T training bars; "
            f"need ≥ {min_train_bars} for stable K-fold"
        )
    if len(X_test) < 100:
        logger.warning(
            "Test year %d has only %d bars in the dataset", test_year, len(X_test)
        )

    logger.info(
        "  Test year %d: %d train bars [%s → %s], %d test bars [%s → %s]",
        test_year, len(X_train), X_train.index.min(), X_train.index.max(),
        len(X_test), X_test.index.min(), X_test.index.max(),
    )

    # Pre-winsorize train AND test using TRAIN quantiles (walk-forward
    # causality of winsorize parameters).
    lo_vals = X_train.quantile(base_config.winsorize_lo)
    hi_vals = X_train.quantile(base_config.winsorize_hi)
    X_train_w = X_train.clip(lo_vals, hi_vals, axis=1)
    X_test_w = X_test.clip(lo_vals, hi_vals, axis=1)

    # Train K-member ensemble on pre-winsorized training data
    params = EnsembleParams(K=K)
    logger.info("  Training K=%d ensemble (n_estimators=%d, n_splits=%d)...",
                K, base_config.n_estimators, base_config.n_splits)
    t0 = time.time()
    ensemble = train_har_pd_ensemble(
        X_train_w, y_train, t1_train, base_config,
        params=params,
        max_feature_lookback_bars=base_config.bars_per_24h * 7,
    )
    train_secs = time.time() - t0
    logger.info("  Ensemble trained in %.1f min", train_secs / 60)

    # Predict on year-T data
    # Per member, average across the K-fold models to get a single prediction
    # per bar; then variance across K members = σ²_epi.
    n_test = len(X_test_w)
    member_preds = np.empty((K, n_test), dtype=np.float64)
    for member_i, member in enumerate(ensemble["members"]):
        fold_models = member["models"]
        per_fold = np.empty((len(fold_models), n_test), dtype=np.float64)
        for fold_j, model in enumerate(fold_models):
            per_fold[fold_j] = model.predict(X_test_w.values)
        member_preds[member_i] = per_fold.mean(axis=0)
    var_dict = predict_ensemble_var(member_preds)

    # HQR cal window inside training data
    cal_start = CAL_WINDOW[0]
    cal_end = min(CAL_WINDOW[1], test_start)
    if cal_end <= cal_start:
        raise ValueError(
            f"Test year {test_year}: cal window degenerate "
            f"(cal_end={cal_end} <= cal_start={cal_start})"
        )

    # Build full timestamp series for HQR (cal subset + test set).
    # We want sigma_alea evaluated on year-T bars; HQR fits on cal slice
    # then `sigma_model.predict` extrapolates to year-T. Use a single
    # combined frame so combine_with_hqr can run end-to-end.
    cal_mask_full = (X_full.index >= cal_start) & (X_full.index < cal_end)
    cal_idx = X_full.index[cal_mask_full]

    # For HQR's input we need ensemble_mean + ensemble_std on the cal slice.
    # Predict the ensemble's per-member-then-fold-averaged mean on cal data.
    X_cal = X_full.loc[cal_idx].copy()
    X_cal_w = X_cal.clip(lo_vals, hi_vals, axis=1)
    cal_member_preds = np.empty((K, len(X_cal_w)), dtype=np.float64)
    for member_i, member in enumerate(ensemble["members"]):
        fold_models = member["models"]
        per_fold = np.empty((len(fold_models), len(X_cal_w)), dtype=np.float64)
        for fold_j, model in enumerate(fold_models):
            per_fold[fold_j] = model.predict(X_cal_w.values)
        cal_member_preds[member_i] = per_fold.mean(axis=0)
    cal_var_dict = predict_ensemble_var(cal_member_preds)

    combined_idx = cal_idx.append(X_test_w.index)
    combined_mean = np.concatenate([cal_var_dict["mean"], var_dict["mean"]])
    combined_std = np.concatenate([cal_var_dict["std"], var_dict["std"]])
    combined_y_true = np.concatenate([
        y_full.loc[cal_idx].to_numpy(),
        y_test.to_numpy(),
    ])

    hqr_out = combine_with_hqr(
        timestamps=combined_idx,
        ensemble_mean=combined_mean,
        ensemble_std=combined_std,
        y_true=combined_y_true,
        cal_window=(cal_start, cal_end),
    )

    # Slice HQR output to the test-year portion only.
    test_slice = slice(len(cal_idx), None)
    return {
        "timestamps": X_test_w.index,
        "y_true": y_test.to_numpy(),
        "ensemble_mean": var_dict["mean"],
        "sigma_epi": var_dict["std"],
        "sigma_alea": hqr_out["sigma_alea"][test_slice],
        "sigma_total": hqr_out["sigma_total"][test_slice],
        "spread": hqr_out["spread"][test_slice],
        "pi_lo": hqr_out["pi_lo"][test_slice],
        "pi_hi": hqr_out["pi_hi"][test_slice],
        "q_conformal": float(hqr_out["q_conformal"]),
        "cal_n_obs": int(hqr_out["cal_n_obs"]),
        "n_train_bars": int(len(X_train)),
        "ensemble_seeds": ensemble["seeds_used"],
        "feature_names": ensemble["feature_names"],
    }


# ─── Driver ─────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Cycle 4 Phase A.0: regenerate K=10 deep-ensemble "
                    "epistemic-variance parquet under expanding-window-by-year "
                    "walk-forward training (mirrors DFL Head fold contract).",
    )
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing artifact (defeat safety)")
    parser.add_argument("--smoke", action="store_true",
                        help=f"Smoke mode: K={SMOKE_K}, "
                             f"n_estimators={SMOKE_N_ESTIMATORS}, "
                             f"test_years={SMOKE_TEST_YEARS}")
    parser.add_argument("--K", type=int, default=DEFAULT_K)
    parser.add_argument("--n-estimators", type=int, default=DEFAULT_N_ESTIMATORS)
    parser.add_argument("--test-years", type=int, nargs="+",
                        default=list(DEFAULT_TEST_YEARS))
    args = parser.parse_args()

    if args.smoke:
        logger.info("SMOKE MODE")
        K = SMOKE_K
        n_estimators = SMOKE_N_ESTIMATORS
        test_years = list(SMOKE_TEST_YEARS)
        out_path = args.out.with_name(args.out.stem + "_smoke" + args.out.suffix)
    else:
        K = args.K
        n_estimators = args.n_estimators
        test_years = args.test_years
        out_path = args.out

    if out_path.exists() and not args.force:
        raise FileExistsError(
            f"{out_path} already exists. Use --force to overwrite "
            "(only after a NEW plan file authorises regeneration)."
        )

    # Build full dataset once
    logger.info("Building HAR-PD feature dataset...")
    t0 = time.time()
    cfg = VolModelConfig(
        asset="BTCUSDT",
        freq_s=900,
        n_estimators=n_estimators,
        early_stopping_rounds=50,
        n_splits=5,
    )
    X_full, y_full, t1_full = build_dataset(cfg)
    logger.info(
        "Dataset built in %.1fs: %d rows, %d features, span %s -> %s",
        time.time() - t0, len(X_full), len(X_full.columns),
        X_full.index.min(), X_full.index.max(),
    )

    # Per-year ensemble + prediction
    per_year_frames: list[pd.DataFrame] = []
    per_year_metadata: dict[int, dict] = {}
    for test_year in test_years:
        logger.info("=" * 70)
        logger.info("WALK-FORWARD FIT for test_year = %d", test_year)
        logger.info("=" * 70)
        result = _train_fold_and_predict_year(
            X_full, y_full, t1_full, test_year,
            base_config=cfg, K=K,
        )
        df_year = pd.DataFrame({
            "ensemble_mean": result["ensemble_mean"],
            "sigma_epi": result["sigma_epi"],
            "sigma_alea": result["sigma_alea"],
            "sigma_total": result["sigma_total"],
            "spread": result["spread"],
            "pi_lo": result["pi_lo"],
            "pi_hi": result["pi_hi"],
            "y_true": result["y_true"],
        }, index=result["timestamps"])
        per_year_frames.append(df_year)
        per_year_metadata[test_year] = {
            "n_test_bars": int(len(df_year)),
            "n_train_bars": int(result["n_train_bars"]),
            "q_conformal": float(result["q_conformal"]),
            "cal_n_obs": int(result["cal_n_obs"]),
            "ensemble_seeds": [int(s) for s in result["ensemble_seeds"]],
        }
        # Empirical 90% PI coverage on this test year
        df_year_clean = df_year.dropna(subset=["pi_lo", "pi_hi", "y_true"])
        in_pi = (
            (df_year_clean["y_true"] >= df_year_clean["pi_lo"])
            & (df_year_clean["y_true"] <= df_year_clean["pi_hi"])
        )
        cov = float(in_pi.mean()) if len(df_year_clean) > 0 else float("nan")
        logger.info(
            "  Test year %d: 90%% PI empirical coverage = %.4f "
            "(n=%d, target [0.85, 0.95])",
            test_year, cov, len(df_year_clean),
        )
        per_year_metadata[test_year]["pi_coverage_90"] = cov

    # Concatenate: full-bar-index frame, NaN for non-test years
    out_df = pd.DataFrame(
        index=X_full.index,
        columns=["ensemble_mean", "sigma_epi", "sigma_alea",
                 "sigma_total", "spread", "pi_lo", "pi_hi", "y_true"],
        dtype=np.float64,
    )
    out_df.index.name = "timestamp"
    for df_year in per_year_frames:
        common_idx = out_df.index.intersection(df_year.index)
        out_df.loc[common_idx] = df_year.loc[common_idx].values
    # y_true is always available for ALL bars (label series), fill in for NaN years
    out_df["y_true"] = y_full.reindex(out_df.index).to_numpy()

    # Freeze
    metadata = {
        "K": K,
        "n_estimators_per_member": n_estimators,
        "n_splits_purged_cv": cfg.n_splits,
        "test_years": list(test_years),
        "per_year": per_year_metadata,
        "cal_window_start": CAL_WINDOW[0].isoformat(),
        "cal_window_end": CAL_WINDOW[1].isoformat(),
        "alpha": 0.10,
        "n_bars_total": int(len(out_df)),
        "fold_contract": (
            "expanding-window-by-year: train_y < test_y. "
            "PurgedKFold(n_splits=5) inside each fit. "
            "Pre-winsorize uses pre-test-year quantiles only."
        ),
        "cycle_3_artifact_distinct_path": (
            "vol_model/output/BTC_v2_har_pd_ensemble_var.parquet "
            "(K-fold; this artifact at "
            "BTC_v2_har_pd_ensemble_var_walkforward.parquet is walk-forward)"
        ),
    }
    freeze_uncertainty(out_df, out_path, metadata=metadata, force=args.force)

    print()
    print("=" * 70)
    print(" Walk-forward K=%d ensemble + HQR uncertainty frozen" % K)
    print("=" * 70)
    print(f"  Output: {out_path}")
    print(f"  Test years: {test_years}")
    print(f"  K members: {K}, n_estimators={n_estimators}")
    print(f"  Total bars: {len(out_df):,}")
    for yr, md in per_year_metadata.items():
        print(f"  Year {yr}: n_test={md['n_test_bars']:,}, "
              f"n_train={md['n_train_bars']:,}, "
              f"q_conformal={md['q_conformal']:.4f}, "
              f"PI coverage={md.get('pi_coverage_90', float('nan')):.4f}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
