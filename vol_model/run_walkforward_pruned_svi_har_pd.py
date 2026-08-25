#!/usr/bin/env python
"""Walk-forward retrain: pruned_svi_v1 + HAR-PD path-dependent features.

Extends run_walkforward_pruned_svi.py with 6 HAR-PD features (Xiao et al. 2025,
arxiv:2503.00851): R1/R2 with learned exponential kernels α∈{0.05,0.1,0.3} on
signed realized flow and realized variance. Session C spot-check on 5 of 8
regime months recorded -2.86% avg ΔQLIKE and 3 DM-significant improvements at
p<0.01. This script evaluates that historical shortlist over a full
walk-forward; it does not certify a current result.

Changes vs run_walkforward_pruned_svi.py:
  - Loads the HAR-PD sidecar from
    ``vol_model/output/{asset}_features_har_pd_15m.parquet`` (built via
    ``python -m vol_model.features.har_pd --asset {asset}``).
  - Concatenates HAR-PD into the training matrix alongside pruned-base and
    svi_atm_1d. Reindex-only — HAR-PD features are already leakage-safe per
    their ``shift(1).rolling(K)`` construction (see the sidecar module's
    docstring and tests/test_har_pd.py::test_causality_* tests).
  - Output dir: ``vol_model/output/models/expanding_pruned_svi_har_pd_v1/``;
    it does not overwrite the historical ``expanding_pruned_svi_v1/`` artifacts.

Target: 50 months × ~10-15 min LightGBM-only per month ≈ 8-12h wall clock,
fits an overnight run (identical compute profile to pruned_svi_v1 — 6 extra
features is negligible).

Historical research thresholds (not current result certification):
  - avg ΔQLIKE ≤ -3% vs pruned_svi_v1 baseline
  - ≥25/50 months improved
  - ≥15/50 months DM-significant at p<0.01
Passing all three records a historical research candidate only. Promotion still
requires the evidence in ``docs/VALIDATION.md``.

Usage:
    PYTHONPATH=. python vol_model/run_walkforward_pruned_svi_har_pd.py
    # Faster smoke run with a shorter Optuna budget:
    PYTHONPATH=. python vol_model/run_walkforward_pruned_svi_har_pd.py \\
        --max-trials 100 --patience 30
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from scipy.stats import spearmanr

from vol_arb.paths import (
    drop_features_path, har_pd_features_path, pruned_features_path,
    pruned_svi_model_dir, svi_features_path, transvol_dir,
    CURRENCY_FOR_ASSET,
)
from vol_model.config import VolModelConfig
from vol_model.labels import detect_stamping
from vol_model.model import qlike_loss, qlike_metric, qlike_objective
from vol_model.pipeline import build_dataset
from vol_model.run_walkforward import OptunaPatience, _make_objective
# NOTE: ``torch`` and the TransVol embedding extractor
# (``run_walkforward_hybrid._extract_compact_embeddings``, which imports torch +
# the transvol package at module top) are imported LAZILY inside the TransVol
# branch only. This keeps the ``--no-transvol`` path (the AWS-spot horizon sweep)
# completely torch-free, so the worker box can ``pip install`` without the heavy
# torch wheel. See _transvol_device() and the emb_df extraction below.

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("wf_pruned_svi_har_pd")

def _transvol_device():
    """Lazily resolve the torch device. Only called on the TransVol path so the
    ``--no-transvol`` AWS-spot sweep never imports torch."""
    import torch  # lazy: keep --no-transvol torch-free
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dead-weight features to drop per Stage 15.9 SHAP analysis. The BTC
# list is kept as a fallback constant for code that imports it, but the
# per-asset DROP list now lives at
# ``vol_model/output/{asset}_drop_features.csv`` (produced by
# run_shap_analysis.py --drop-bottom-n). Loaded lazily in main() so each
# asset gets its own data-driven Stage 15.9 tail — no hand-curation.
DROP_FEATURES = [
    "ffd_rv", "ffd_rv_lag3", "ffd_rv_ema10", "ffd_rv_ema50",
    "noise_level", "noise_level_lag1", "noise_level_ema10", "noise_level_ema50",
]


def _load_drop_features(asset: str) -> list[str]:
    """Read the per-asset SHAP-tail drop list. Falls back to the BTC
    hand-curated constant if the asset hasn't been calibrated yet — so a
    fresh repo still runs BTC byte-identical."""
    path = drop_features_path(asset)
    if path.exists():
        with open(path) as f:
            drops = [line.strip() for line in f if line.strip()]
        logger.info("Loaded %d drop features for %s from %s", len(drops), asset, path)
        return drops
    logger.warning("No drop-features csv at %s — falling back to BTC DROP_FEATURES", path)
    return list(DROP_FEATURES)

# SVI features to add (only svi_atm_1d per Stage 15.10 recommendation)
ADD_SVI_FEATURES = ["svi_atm_1d"]

# HAR-PD features (Xiao et al. 2025, arxiv:2503.00851) — 6 columns, built
# ahead of time into the sidecar parquet. Feature list mirrors
# vol_model.features.har_pd.HAR_PD_RAW_FEATURES (kept duplicated here to keep
# the walk-forward script self-contained and avoid a compile-time dependency
# on the feature-build module).
ADD_HAR_PD_FEATURES = [
    "har_pd_R1_a005", "har_pd_R1_a01", "har_pd_R1_a03",
    "har_pd_R2_b005", "har_pd_R2_b01", "har_pd_R2_b03",
]


# ── Historical LGBM cleanup levers (2026-06-22) ─────────────────────
# All three are DEFAULT-OFF additive toggles. With every toggle off the
# training matrix and params dict match the historical reference behavior.
# The AWS fleet launcher can enable a selected lever per horizon via
# the corresponding CLI flag noted below.

# [1.1] orphan-drop (--drop-orphans): two SHAP-prune bookkeeping survivors.
# `noise_level_lag3` is statistically identical to the already-dropped
# `noise_level` (std 9.16e-9, skew 125.9, excess-kurt 22,797 — the most
# pathological tail in the active set); `ffd_rv_lag1` is the sole survivor of
# the otherwise-dropped ffd_rv family. Apply-regardless / zero-risk per audit.
AUDIT_ORPHAN_DROP = ["noise_level_lag3", "ffd_rv_lag1"]

# [1.2] collinearity prune (--prune-collinear): drop the >0.999-collinear
# ema10/ema50/lag1 duplicate copies of the five strong slow-feature families
# (tsrv_mean_1d, tsrv_mean_1w, parkinson_1d, time_to_funding, tsrv_ffd),
# keeping base + lag3 per family (the audit's exact keep-rule: "retain the
# most-differentiated lag per family, e.g. tsrv_mean_1d + tsrv_mean_1d_lag3;
# do not blindly keep only the base"). Derived by union-find on
# collinear_pairs_BTCUSDT_h15.csv at the >0.999 threshold. Restores
# colsample_bytree subsampling regularization (a 5-member near-identical
# family has ~97% chance >=1 copy is sampled per tree). 52 post-[1.1] base
# features -> 38. Expected ΔQLIKE ≈ 0 (27/54 active features already carry
# zero SHAP importance — the model self-selects one variant per family).
AUDIT_COLLINEAR_DROP = [
    "parkinson_1d_ema10", "parkinson_1d_ema50", "parkinson_1d_lag1",
    "time_to_funding_ema10", "time_to_funding_ema50", "time_to_funding_lag1",
    "tsrv_ffd_ema10", "tsrv_ffd_lag1",
    "tsrv_mean_1d_ema10", "tsrv_mean_1d_ema50", "tsrv_mean_1d_lag1",
    "tsrv_mean_1w_ema10", "tsrv_mean_1w_ema50", "tsrv_mean_1w_lag1",
]

# [1.3] monotone +1 constraints (--monotone-constraints): sign-certain
# past-RV / realized-variance features where the economic sign is
# domain-certain (higher past RV -> higher future vol). The constraint vector
# is built per-month aligned to the final matrix column order (so it is robust
# to whichever other toggles are active). Ambiguous-sign features
# (VPIN, imbalance, jump fractions, time_to_funding, the signed-flow HAR-PD
# R1 kernels) are LEFT UNCONSTRAINED (0). A genuine constraint can only forbid
# noise-fit non-monotone splits -> pure anti-overfit regularizer, expected to
# help most in the coarse small-sample regime (h=1440).
AUDIT_MONOTONE_POS_FEATURES = [
    "tsrv_mean_1d", "tsrv_mean_1w", "parkinson_1d", "bpv_tsrv_ratio",
    "har_pd_R2_b005", "har_pd_R2_b01", "har_pd_R2_b03",
]


def _build_monotone_vector(columns: list[str]) -> list[int]:
    """Build a LightGBM monotone_constraints vector aligned to ``columns``.

    +1 for each sign-certain past-RV feature present in the matrix, 0 for
    every other column. Returns a plain python list[int] of len(columns).
    Only the base-level feature name is constrained — collinear lag/ema
    siblings (if still present) are left unconstrained because the audit's
    sign-certainty argument is cleanest for the contemporaneous level.
    """
    pos = set(AUDIT_MONOTONE_POS_FEATURES)
    return [1 if c in pos else 0 for c in columns]


def _load_har_pd(asset: str, horizon_minutes: int = 15) -> pd.DataFrame:
    """Load HAR-PD sidecar with a −1-bar alignment shift.

    Alignment rationale (maximise usable information without leakage)
    ----------------------------------------------------------------
    Ideal: at decision time wall-clock T, predict the label for bar
    [T, T+h) using ALL data known at or before wall-clock T. The most
    recent complete TSRV bar is [T-h, T) — end-stamped at T, realised
    at wall-clock T, therefore safe to use. (h = horizon_minutes.)

    Raw HAR-PD sidecar (end-stamped, inherited from TSRV): the sidecar's
    construction uses ``shift(1).rolling(K)``, so HAR-PD at sidecar
    index T uses TSRV values at end-stamps T-1, T-2, ..., T-K (i.e.,
    bars ending h..K*h before wall-clock T). Latest info in those
    values is known at wall-clock T-h.

    Joined WITHOUT shift to the start-stamped label at index T (vol of
    [T, T+h)): feature-info cutoff = T-h, prediction window starts
    = T. That is an h-minute GAP of wasted data — the bar ending at T
    (known at T, strictly before the prediction window) is silently
    discarded. Non-leaky but sub-optimal.

    Fix: shift the sidecar index by −h. After the shift, "new sidecar
    index T" holds values that originally lived at T+h, i.e., a HAR-PD
    row computed from TSRV end-stamps T, T-1, ..., T-(K-1). Latest TSRV
    used = bar ending at wall-clock T, known at T. Joined to label at
    T (vol of [T, T+h)): feature-info cutoff = T, prediction start =
    T. Zero-minute gap, still non-leaky (feature is strictly known at
    decision time, prediction window is strictly in the future).

    This matches the historical SVI feature's 0-minute-gap alignment
    uses (``_load_svi_atm_1d`` applies +h to raw SVI whose natural
    stamp lives h AHEAD of the label). Both features therefore land
    with feature-info-cutoff = label-start = decision time.

    Leakage safety is empirically verified by
    tests/test_walkforward_pruned_svi_har_pd.py::test_no_lookahead_after_shift
    which mutates TSRV at bars starting at or after T and confirms that
    the post-shift HAR-PD value at new-index T is byte-identical.

    ``horizon_minutes`` defaults to 15 to preserve byte-identity with the
    pre-horizon-SSOT call site; callers should pass
    ``vol_config.horizon_minutes`` so the shift always equals one bar.
    """
    # Sidecar file naming via the horizon-aware path helper (SSOT in
    # vol_arb.paths.har_pd_features_path).
    path = har_pd_features_path(asset, horizon_minutes)
    if not path.exists():
        raise FileNotFoundError(
            f"HAR-PD sidecar not found at {path} — "
            f"build via `python -m vol_model.features.har_pd --asset {asset}"
            f" --horizon-minutes {horizon_minutes}`"
        )
    hpd = pd.read_parquet(path)
    if "timestamp" in hpd.columns and not isinstance(hpd.index, pd.DatetimeIndex):
        hpd = hpd.set_index("timestamp")
    if not isinstance(hpd.index, pd.DatetimeIndex):
        hpd.index = pd.to_datetime(hpd.index, utc=True)
    missing = [c for c in ADD_HAR_PD_FEATURES if c not in hpd.columns]
    if missing:
        raise KeyError(
            f"HAR-PD sidecar is missing columns: {missing}. "
            f"Available: {list(hpd.columns)}"
        )
    hpd_add = hpd[ADD_HAR_PD_FEATURES].copy()

    # MF-2 STAMPING-AWARE ALIGNMENT (2026-06-21)
    # ------------------------------------------
    # The HAR-PD sidecar inherits the stamping of the TSRV parquet it was
    # built from. The −h shift documented above is ONLY non-leaky when the
    # sidecar is END-stamped (the case the docstring describes): then "new
    # index T" holds the HAR-PD row whose latest TSRV input ends at wall-clock
    # T, joining to the label for [T, T+h) with a zero-minute gap.
    #
    # For a START-stamped sidecar (every non-15m horizon) the HAR-PD row at
    # index T already uses, via its shift(1).rolling(K) construction, only
    # TSRV bars whose realised windows end at or before T (the most recent is
    # [T-h, T), known at T). That row is ALREADY the correct decision-time
    # feature for predicting the label at T. Applying the −h shift would pull
    # in the row that consumes bar [T, T+h) — a ONE-BAR LOOKAHEAD LEAK. So we
    # shift ONLY when the sidecar is genuinely END-stamped; START-stamped
    # sidecars are joined as-is (zero shift, already aligned, non-leaky).
    stamping = detect_stamping(hpd_add.index, horizon_minutes * 60)
    if stamping == "END":
        hpd_add.index = hpd_add.index - pd.Timedelta(minutes=horizon_minutes)
        shift_desc = f"−{horizon_minutes}m (END-stamped)"
    else:
        shift_desc = f"0m ({stamping}-stamped, already aligned)"

    coverage = hpd_add.notna().all(axis=1).mean()
    logger.info(
        "Loaded HAR-PD sidecar (%s): %d rows, cols=%s, %.1f%% full-coverage",
        shift_desc, len(hpd_add), ADD_HAR_PD_FEATURES, 100 * coverage,
    )
    return hpd_add


def _load_svi_atm_1d(currency: str, horizon_minutes: int = 15) -> pd.DataFrame:
    """Load svi_atm_1d from the per-currency SVI features parquet with leakage shift.

    The SVI features parquet at h=15 is the legacy unsuffixed file
    (15-min resolution, built at 1h SVI calibration bucketed into 15m
    bars). For h ∈ {30m, 60m} the 15m SVI is on-grid and gets subsampled
    (every 2nd / 4th row). For h=5m a separate regen at
    ``--svi-bucket-freq 5min`` produces ``svi_features_h5m.parquet``
    which this loader picks up via the horizon-aware path helper. The
    alignment shift always equals one configured bar (horizon_minutes).
    """
    svi_path = svi_features_path(currency, horizon_minutes=horizon_minutes)
    if not svi_path.exists():
        raise FileNotFoundError(
            f"SVI features not found at {svi_path} — "
            f"rebuild via `python -m vol_arb.run_pipeline --currency {currency} "
            f"--svi --svi-bucket-freq {horizon_minutes}min`"
        )
    svi = pd.read_parquet(svi_path)
    # Ensure DatetimeIndex (ETH parquet may have timestamp as a column, not the index)
    if "timestamp" in svi.columns and not isinstance(svi.index, pd.DatetimeIndex):
        svi = svi.set_index("timestamp")
    if not isinstance(svi.index, pd.DatetimeIndex):
        svi.index = pd.to_datetime(svi.index, utc=True)
    missing = [c for c in ADD_SVI_FEATURES if c not in svi.columns]
    if missing:
        raise KeyError(
            f"SVI features parquet is missing required columns: {missing}. "
            f"Available SVI columns: {[c for c in svi.columns if 'svi' in c.lower()]}",
        )
    svi_add = svi[ADD_SVI_FEATURES].copy()

    # LEAKAGE FIX (Stage 15.1): X.index is the LABEL timestamp (T+1) but X's
    # data comes from the feature time T. SVI features indexed at T+1 would
    # be contemporaneous with the label. Shift SVI index forward by ONE
    # CONFIGURED BAR (= horizon_minutes) so SVI[T] aligns with X[T+1] —
    # i.e., SVI from the PREVIOUS bar is associated with the CURRENT bar's
    # label. Default 15m for byte-identity with the pre-horizon-SSOT
    # historical h=15 path.
    svi_add.index = svi_add.index + pd.Timedelta(minutes=horizon_minutes)

    coverage = svi_add[ADD_SVI_FEATURES[0]].notna().mean()
    logger.info(
        "Loaded SVI features (+%dm aligned): %d rows, cols=%s, %.1f%% coverage",
        horizon_minutes, len(svi_add), ADD_SVI_FEATURES, 100 * coverage,
    )
    return svi_add


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward pruned + svi_atm_1d training (T2.1)",
    )
    parser.add_argument("--asset", default="BTCUSDT",
                        help="Training asset (BTCUSDT or ETHUSDT). Resolves "
                             "TransVol checkpoint dir, SVI features, pruned "
                             "feature list, and output model dir per asset.")
    parser.add_argument("--start", default="2022-02",
                        help="First month to retrain (inclusive), YYYY-MM")
    parser.add_argument("--end", default="2026-03",
                        help="Last month to retrain (inclusive), YYYY-MM")
    parser.add_argument("--max-trials", type=int, default=200,
                        help="Optuna trial budget per month. Default 200 retains "
                             "the historical hybrid reference budget.")
    parser.add_argument("--patience", type=int, default=50,
                        help="Optuna patience. Default 50 retains the historical "
                             "hybrid reference budget.")
    parser.add_argument("--output-dir", type=str, default="",
                        help="Override save dir (else per-asset via resolver)")
    parser.add_argument("--tf", type=int, default=900,
                        help="Timeframe in seconds (default 900 = 15m)")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional suffix tag (appended after asset-aware default)")
    parser.add_argument("--transvol-dir", type=str, default="",
                        help="Override the TransVol checkpoint dir (else the "
                             "per-asset/per-horizon resolver). Used to point at "
                             "the rehab checkpoints (e.g. "
                             "vol_model/output/models/expanding_hybrid_clean_rehab) "
                             "so the rehab LGBM consumes the clean embeddings "
                             "without touching historical reference artifacts.")
    parser.add_argument("--skip-drop", action="store_true",
                        help="Train WITHOUT the Stage-15.9 SHAP-tail DROP list — "
                             "keeps all 35 pruned features (no drop, pure top-N) "
                             "so the A/B comparison full-35 vs drop-27 is clean.")
    parser.add_argument("--force-retrain", action="store_true",
                        help="C2 (parity-r3 cycle 8 — 2026-05-08): retrain even "
                             "when the per-month LightGBM checkpoint already "
                             "exists; overwrites in-place. Default behaviour "
                             "(no flag) skips months whose `.txt` checkpoint "
                             "is already on disk. Used by the orchestrator's "
                             "--force flag for an explicit overnight rerun.")
    parser.add_argument("--save-shap", action="store_true",
                        help="After each month's LGBM trains, compute TreeExplainer "
                             "SHAP on the test window and save per-month mean |SHAP| "
                             "to {output_dir}/shap_{month}.csv. Enables the "
                             "aggregate_walkforward_shap.py rollup — data-driven "
                             "drop list from walk-forward (not single pre-cutoff).")
    parser.add_argument("--prune-source-asset", default="",
                        help="Override the asset used to resolve pruned_features.csv "
                             "and drop_features.csv (but NOT the training data asset). "
                             "Pass BTCUSDT to train ETH with BTC's 27-feature pruned "
                             "list — useful for diagnosing ETH-specific outlier months "
                             "where the 12 ETH-only features may destabilise LGBM.")
    parser.add_argument("--predict-excess-logvol", action="store_true",
                        help="Reparameterize the target as y = log(RV_future) - "
                             "log(IV_at_decision_time). Auto-applies tag 'excess' "
                             "to the output model directory so existing predictions "
                             "are not overwritten. See VolModelConfig.predict_excess_logvol "
                             "docstring for rationale.")
    parser.add_argument("--drop-orphans", action="store_true",
                        help="LGBM-data-audit lever [1.1]: additionally drop the "
                             "two SHAP-prune orphan survivors (noise_level_lag3, "
                             "ffd_rv_lag1) from the pruned base set. Apply-regardless "
                             "under the historical audit. DEFAULT OFF so the "
                             "reference path is unchanged. The A/B sweep baseline "
                             "turns this ON.")
    parser.add_argument("--prune-collinear", action="store_true",
                        help="LGBM-data-audit lever [1.2]: additionally drop the 14 "
                             ">0.999-collinear ema/lag duplicate copies of the five "
                             "strong slow-feature families, keeping base+lag3 per "
                             "family under the historical audit's keep rule. "
                             "Restores colsample_bytree regularization. DEFAULT OFF.")
    parser.add_argument("--monotone-constraints", action="store_true",
                        help="LGBM-data-audit lever [1.3]: apply monotone +1 "
                             "constraints to the sign-certain past-RV / realized-"
                             "variance features (tsrv_mean_1d/1w, parkinson_1d, "
                             "bpv_tsrv_ratio, har_pd_R2_b*) via LightGBM "
                             "monotone_constraints. DEFAULT OFF.")
    parser.add_argument("--no-transvol", action="store_true",
                        help="Train WITHOUT TransVol embedding features "
                             "(pruned + SVI + HAR-PD only). Required at "
                             "horizons other than 15m unless TransVol "
                             "checkpoints have been pre-trained at the "
                             "active horizon (the historical h=15 run has 50 monthly "
                             "checkpoints; new horizons would need ~24h+ "
                             "of TransVol retraining each). Auto-applies "
                             "tag 'no_tv' to the output model directory "
                             "and predictions parquet so the no-TransVol "
                             "variant cannot collide with the historical "
                             "TransVol-included artifacts.")
    args = parser.parse_args()
    # Auto-tag excess runs so we never overwrite the default-target predictions.
    if args.predict_excess_logvol and not args.tag and not args.output_dir:
        args.tag = "excess"
    # Auto-tag no-TransVol runs so the artifact tree never collides with
    # the historical TransVol-included artifact tree.
    if args.no_transvol and not args.tag and not args.output_dir:
        args.tag = "no_tv"

    currency = CURRENCY_FOR_ASSET[args.asset]
    # Resolve TransVol checkpoint dir from the ACTIVE horizon (set via
    # --tf below before this line could have taken effect — re-read it
    # after vol_config is finalized). Falls back to legacy h=15
    # (unsuffixed) when args.tf is the default 900s.
    _hybrid_h = args.tf // 60
    hybrid_dir = (
        Path(args.transvol_dir) if args.transvol_dir
        else transvol_dir(args.asset, horizon_minutes=_hybrid_h)
    )

    # Resolve output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif args.tag:
        base = pruned_svi_model_dir(args.asset)
        output_dir = base.with_name(f"{base.name}_har_pd_v1_{args.tag}")
    else:
        # Default: expanding_pruned_svi_har_pd_v1 (sibling of pruned_svi_v1)
        base = pruned_svi_model_dir(args.asset)
        output_dir = base.with_name(f"{base.name.replace('_v1', '')}_har_pd_v1")
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    tf_label = f"{args.tf // 60}m" if args.tf < 86400 else f"{args.tf // 3600}h"
    print(f"  WALK-FORWARD PRUNED + svi_atm_1d + HAR-PD @ {tf_label} — {args.asset}")
    print("=" * 70)
    print(f"  Drop features: {DROP_FEATURES}")
    print(f"  Add SVI: {ADD_SVI_FEATURES}")
    print(f"  Add HAR-PD: {ADD_HAR_PD_FEATURES}")
    print(f"  Reuse TransVol from: {hybrid_dir}")
    print(f"  Save models to: {output_dir}")
    print(f"  Optuna: {args.max_trials} trials, patience {args.patience}")
    print("=" * 70)

    # Load all data once
    logger.info("Loading base features + labels...")
    vol_config = VolModelConfig(
        asset=args.asset,
        predict_excess_logvol=args.predict_excess_logvol,
    )
    if args.tf != 900:
        vol_config.freq_s = args.tf
        logger.info("Timeframe override: freq_s = %d (%dm bars)", args.tf, args.tf // 60)
    if args.predict_excess_logvol:
        logger.info("Target reparameterization ACTIVE: y = log(RV) - log(IV_at_T)")
    # clip=False: opt out of the global label clip in load_labels
    # (data-snoopy — full-series quantiles). Walk-forward applies
    # leakage-safe per-fold clipping inside the training loop instead.
    X, y, t1 = build_dataset(vol_config, clip=False)

    # Per-asset SHAP-tail DROP list: bottom-N of the top-N pruned (Stage
    # 15.9 methodology applied per asset). For BTC this reproduces the
    # original 8-feature drop exactly; for ETH/other assets it's the
    # data-driven bottom-N of the ETH SHAP ranking with the same N so
    # pruning depth stays consistent (never deeper than BTC).
    #
    # --prune-source-asset overrides the feature-list source (for diagnosing
    # outlier months by retraining ETH with BTC's 27-feature list).
    prune_asset = args.prune_source_asset or args.asset
    pruned_path = pruned_features_path(prune_asset)
    pruned = pd.read_csv(pruned_path, header=None)[0].tolist()
    pruned_cols_v1 = [c for c in pruned if c in X.columns]
    missing_pruned = [c for c in pruned if c not in X.columns]
    if missing_pruned:
        logger.warning(
            "%d features from %s's pruned list not in %s features: %s",
            len(missing_pruned), prune_asset, args.asset, missing_pruned,
        )
    if args.skip_drop:
        drop_set: set[str] = set()
        pruned_cols_v2 = list(pruned_cols_v1)
        logger.info(
            "Pruned base (%s, --skip-drop): %d features (no Stage-15.9 drop applied)",
            args.asset, len(pruned_cols_v2),
        )
    else:
        drop_set = set(_load_drop_features(prune_asset))
        pruned_cols_v2 = [c for c in pruned_cols_v1 if c not in drop_set]
        dropped = [c for c in pruned_cols_v1 if c in drop_set]
        logger.info(
            "Pruned base (%s): %d -> %d features (dropped %d: %s)",
            args.asset, len(pruned_cols_v1), len(pruned_cols_v2), len(dropped), dropped,
        )

    # ── LGBM-data-audit additive drops (DEFAULT OFF) ───────────────────────
    # [1.1] orphan-drop and [1.2] collinearity-prune both narrow the base
    # feature set. They are applied AFTER the Stage-15.9 drop so they compose
    # cleanly; with both flags off pruned_cols_v2 is unchanged (byte-identical).
    if args.drop_orphans:
        before_n = len(pruned_cols_v2)
        pruned_cols_v2 = [c for c in pruned_cols_v2 if c not in set(AUDIT_ORPHAN_DROP)]
        removed = [c for c in AUDIT_ORPHAN_DROP if c not in pruned_cols_v2]
        logger.info(
            "[1.1] --drop-orphans: %d -> %d base features (dropped %s)",
            before_n, len(pruned_cols_v2), removed,
        )
    if args.prune_collinear:
        before_n = len(pruned_cols_v2)
        coll = set(AUDIT_COLLINEAR_DROP)
        present = [c for c in pruned_cols_v2 if c in coll]
        pruned_cols_v2 = [c for c in pruned_cols_v2 if c not in coll]
        logger.info(
            "[1.2] --prune-collinear: %d -> %d base features (dropped %d: %s)",
            before_n, len(pruned_cols_v2), len(present), sorted(present),
        )
    X_pruned = X[pruned_cols_v2]

    # TransVol feature set (unchanged; TransVol uses the full 101 features
    # regardless of the LightGBM-level prune)
    drop_prefixes = ("ofi_", "depth_mid", "oi_change", "oi_zscore", "oi_acceleration")
    tv_cols = [c for c in X.columns if not c.startswith(drop_prefixes)]
    X_tv = X[tv_cols]

    from vol_model.labels import load_labels
    labels = load_labels(vol_config)

    # Load the SVI feature(s) — alignment shift = one configured bar.
    svi_add = _load_svi_atm_1d(currency, horizon_minutes=vol_config.horizon_minutes)

    # Load HAR-PD sidecar (6 features). No LEAKAGE shift — sidecar index is
    # already strictly-pre-label by construction (shift(1).rolling(K)). The
    # `-horizon_minutes` shift inside `_load_har_pd` is an information-recovery
    # shift, not a leakage fix (see docstring).
    har_pd_add = _load_har_pd(args.asset, horizon_minutes=vol_config.horizon_minutes)

    data_start = X.index.min()
    start = pd.Timestamp(args.start + "-01", tz="UTC")
    end = pd.Timestamp(args.end + "-01", tz="UTC")
    months = pd.date_range(start, end, freq="MS")

    results = []
    t_start = time.time()

    for i, M in enumerate(months):
        month_str = M.strftime("%Y-%m")
        lgb_path = output_dir / f"{month_str}.txt"

        if lgb_path.exists() and not args.force_retrain:
            logger.info("[%d/%d] %s: cached, skipping", i + 1, len(months), month_str)
            continue
        if lgb_path.exists() and args.force_retrain:
            logger.warning(
                "[%d/%d] %s: --force-retrain in effect; overwriting existing %s",
                i + 1, len(months), month_str, lgb_path.name,
            )

        if not args.no_transvol:
            tv_path = hybrid_dir / f"{month_str}_transvol.pt"
            if not tv_path.exists():
                logger.warning(
                    "[%d/%d] %s: no TransVol checkpoint at %s — skipping",
                    i + 1, len(months), month_str, tv_path,
                )
                continue
        else:
            tv_path = None  # not used in --no-transvol mode

        test_end = M + pd.DateOffset(months=1)
        V = max(M - pd.DateOffset(months=4), data_start + pd.DateOffset(months=12))

        logger.info("=" * 60)
        logger.info(
            "[%d/%d] %s: train=[%s, %s), val=[%s, %s), test=%s",
            i + 1, len(months), month_str,
            data_start.strftime("%Y-%m"), V.strftime("%Y-%m"),
            V.strftime("%Y-%m"), M.strftime("%Y-%m"), month_str,
        )
        t0 = time.time()

        # Extract TransVol embeddings from the frozen historical checkpoint,
        # unless the caller explicitly opted out via --no-transvol —
        # used at non-15m horizons where TransVol was never retrained.
        emb_df: pd.DataFrame | None
        if args.no_transvol:
            emb_df = None
        else:
            # Lazy import: pulls torch + the transvol package. Only on the
            # TransVol path, so --no-transvol stays torch-free (AWS-spot box).
            from vol_model.run_walkforward_hybrid import _extract_compact_embeddings
            try:
                emb_df = _extract_compact_embeddings(tv_path, X_tv, labels, test_end)
            except Exception as exc:  # noqa: BLE001
                logger.exception("%s: TransVol embedding extraction failed: %s", month_str, exc)
                continue

        # Build the training matrix: pruned base + svi_atm_1d + HAR-PD,
        # plus TransVol embeddings when available. Intersect across all
        # included feature groups so every training row has full coverage.
        if emb_df is not None:
            common = (
                X_pruned.index
                .intersection(emb_df.index)
                .intersection(svi_add.index)
                .intersection(har_pd_add.index)
            )
            X_month = pd.concat(
                [
                    emb_df.loc[common],
                    X_pruned.loc[common],
                    svi_add.loc[common],
                    har_pd_add.loc[common],
                ],
                axis=1,
            )
        else:
            common = (
                X_pruned.index
                .intersection(svi_add.index)
                .intersection(har_pd_add.index)
            )
            X_month = pd.concat(
                [
                    X_pruned.loc[common],
                    svi_add.loc[common],
                    har_pd_add.loc[common],
                ],
                axis=1,
            )

        common_y = X_month.index.intersection(y.index)
        X_month = X_month.loc[common_y]
        y_month = y.loc[common_y]

        # Drop rows with NaN in svi_atm_1d (5% coverage gap) or HAR-PD
        # (0.1% gap from the K_BARS burn-in at series start).
        before = len(X_month)
        X_month = X_month.dropna(subset=ADD_SVI_FEATURES + ADD_HAR_PD_FEATURES)
        y_month = y_month.loc[X_month.index]
        dropped_nan = before - len(X_month)
        if dropped_nan > 0:
            logger.info("  Dropped %d rows with NaN in svi/har_pd (%.1f%%)",
                        dropped_nan, 100 * dropped_nan / max(before, 1))

        # Temporal masks
        train_mask = X_month.index < V
        val_mask = (X_month.index >= V) & (X_month.index < M)
        test_mask = (X_month.index >= M) & (X_month.index < test_end)

        n_train, n_val, n_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
        logger.info(
            "  Data: train=%d, val=%d, test=%d, features=%d",
            n_train, n_val, n_test, X_month.shape[1],
        )
        if n_train < 100 or n_val == 0 or n_test == 0:
            logger.warning(
                "%s: insufficient data (train=%d, val=%d, test=%d) — "
                "walk-forward needs all three windows populated; typically "
                "happens when M < data_start + 12 months (not enough history "
                "for the 12-month training lead-in).",
                month_str, n_train, n_val, n_test,
            )
            continue

        X_tr, y_tr = X_month[train_mask], y_month[train_mask]
        X_val, y_val = X_month[val_mask], y_month[val_mask]
        X_test, y_test = X_month[test_mask], y_month[test_mask]

        # Cycle 9 leakage fix (2026-05-08): per-fold label clip using
        # quantiles from the TRAIN slice only. Replaces the global
        # clip in load_labels (which fitted bounds on the full series,
        # mildly leaking test-set tails into training labels). Magnitude
        # was small (0.1% tails at default label_clip_quantile=0.999)
        # but structurally a train-test contamination.
        clip_q = vol_config.label_clip_quantile
        y_lo = float(y_tr.quantile(1.0 - clip_q))
        y_hi = float(y_tr.quantile(clip_q))
        y_tr = y_tr.clip(y_lo, y_hi)
        y_val = y_val.clip(y_lo, y_hi)
        y_test = y_test.clip(y_lo, y_hi)

        # Winsorize on train only
        lo, hi = X_tr.quantile(0.01), X_tr.quantile(0.99)
        X_tr_w = X_tr.clip(lo, hi, axis=1)
        X_val_w = X_val.clip(lo, hi, axis=1)
        X_test_w = X_test.clip(lo, hi, axis=1)

        # Optuna search
        dtrain = lgb.Dataset(
            X_tr_w.values, label=y_tr.values, free_raw_data=False,
            params={"feature_pre_filter": False},
        )
        dval = lgb.Dataset(
            X_val_w.values, label=y_val.values, reference=dtrain, free_raw_data=False,
        )
        dtrain.construct()
        dval.construct()

        # [1.3] monotone constraint vector, aligned to the final matrix
        # column order. None when the toggle is off -> _make_objective adds
        # no monotone_constraints key (historical reference path unchanged).
        mono_vec = (
            _build_monotone_vector(list(X_month.columns))
            if args.monotone_constraints else None
        )
        if mono_vec is not None and i == 0:
            constrained = [c for c, m in zip(X_month.columns, mono_vec) if m == 1]
            logger.info(
                "[1.3] --monotone-constraints: +1 on %d/%d features: %s",
                len(constrained), len(mono_vec), constrained,
            )

        study = optuna.create_study(
            direction="minimize",
            sampler=TPESampler(seed=42),
            pruner=MedianPruner(n_startup_trials=5),
        )
        obj = _make_objective(
            dtrain, dval, X_val_w.values, y_val.values, "cpu",
            horizon_minutes=vol_config.horizon_minutes,
            monotone_constraints=mono_vec,
        )
        study.optimize(
            obj,
            n_trials=args.max_trials,
            callbacks=[OptunaPatience(args.patience)],
            show_progress_bar=False,
        )
        n_trials = len(study.trials)

        # Final retrain on [start, M)
        full_mask = X_month.index < M
        X_full = X_month[full_mask].clip(lo, hi, axis=1)
        y_full = y_month[full_mask]

        best = study.best_params.copy()
        best["objective"] = qlike_objective
        best["subsample"] = 0.8
        best["device"] = "cpu"
        best["num_threads"] = int(os.environ.get("OPT_NUM_THREADS", "-1"))
        best["verbosity"] = -1
        best["seed"] = 42
        # [1.3] carry the monotone constraint into the final retrain so the
        # saved model honours it. Omitted entirely when the toggle is off.
        if mono_vec is not None:
            best["monotone_constraints"] = mono_vec

        dtrain_full = lgb.Dataset(
            X_full.values, label=y_full.values, free_raw_data=False,
        )
        model = lgb.train(
            best, dtrain_full, num_boost_round=2000,
            valid_sets=[dval], feval=qlike_metric,
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
        )

        # Predict + evaluate on test month
        preds = model.predict(X_test_w.values)
        test_qlike = qlike_loss(y_test.values, preds)
        test_rmse = float(np.sqrt(np.mean((y_test.values - preds) ** 2)))
        test_rank_ic = float(spearmanr(y_test.values, preds).statistic)
        # OOS R^2 in log-vol space (additive metric for the A/B; only ever
        # written to summary.json, never to a byte-identity-checked artifact).
        _yt = y_test.values
        _ss_res = float(np.sum((_yt - preds) ** 2))
        _ss_tot = float(np.sum((_yt - _yt.mean()) ** 2))
        test_r2 = float(1.0 - _ss_res / _ss_tot) if _ss_tot > 0 else float("nan")

        # Save the model + feature manifest
        model.save_model(str(lgb_path))
        (output_dir / f"{month_str}_features.json").write_text(
            json.dumps(list(X_month.columns)),
        )

        # Per-month SHAP on the test window — for walk-forward Stage 15.9
        # tail analysis. Saving is opt-in (heavy); aggregate later with
        # vol_model/aggregate_walkforward_shap.py.
        if args.save_shap:
            try:
                import shap as _shap  # noqa: WPS433 — lazy import, not every run needs it
                explainer = _shap.TreeExplainer(model)
                shap_vals = explainer.shap_values(X_test_w.values)
                mean_abs = np.abs(shap_vals).mean(axis=0)
                shap_df = pd.DataFrame({
                    "feature": list(X_month.columns),
                    "mean_abs_shap": mean_abs,
                })
                shap_df.to_csv(output_dir / f"shap_{month_str}.csv", index=False)
            except Exception as shap_err:  # noqa: BLE001
                logger.warning("SHAP failed for %s: %s", month_str, shap_err)

        # Compare to the cached baseline from expanding_hybrid/ for this month.
        # Skipped in --no-transvol mode because the baseline model itself
        # uses TransVol embeddings (it would also be horizon-mismatched at
        # h!=15 — the baseline checkpoints live only at h=15).
        baseline_path = hybrid_dir / f"{month_str}.txt"
        baseline_qlike = None
        if baseline_path.exists() and emb_df is not None:
            base_feat_path = hybrid_dir / f"{month_str}_features.json"
            if base_feat_path.exists():
                base_cols = json.loads(base_feat_path.read_text())
                bmodel = lgb.Booster(model_file=str(baseline_path))

                # Build baseline test features (no SVI, full pruned_v1)
                X_pruned_v1 = X[pruned_cols_v1]
                common_base = X_pruned_v1.index.intersection(emb_df.index)
                X_base = pd.concat(
                    [emb_df.loc[common_base], X_pruned_v1.loc[common_base]],
                    axis=1,
                )
                X_base = X_base.loc[X_base.index.intersection(y.index)]
                X_base_test = X_base[
                    (X_base.index >= M) & (X_base.index < test_end)
                ]
                y_base_test = y.loc[X_base_test.index]
                X_base_test_w = pd.DataFrame(index=X_base_test.index)
                for c in base_cols:
                    X_base_test_w[c] = (
                        X_base_test[c].values if c in X_base_test.columns else 0.0
                    )
                X_base_test_w = X_base_test_w.clip(lo, hi, axis=1)
                base_preds = bmodel.predict(X_base_test_w.values)
                baseline_qlike = qlike_loss(y_base_test.values, base_preds)

        elapsed = time.time() - t0
        delta_pct = (
            ((test_qlike / baseline_qlike - 1) * 100)
            if baseline_qlike else float("nan")
        )

        logger.info(
            "  [%d/%d] %s: qlike=%.4f  base=%.4f  delta=%+.1f%%  "
            "rank_ic=%.3f  trials=%d  (%.0fs)",
            i + 1, len(months), month_str, test_qlike,
            baseline_qlike or 0, delta_pct, test_rank_ic, n_trials, elapsed,
        )

        results.append({
            "month": month_str,
            "baseline_qlike": baseline_qlike,
            "pruned_svi_qlike": test_qlike,
            "delta_pct": delta_pct,
            "rank_ic": test_rank_ic,
            "oos_r2": test_r2,
            "n_features": X_month.shape[1],
            "n_trials": n_trials,
            "time_s": elapsed,
        })

    # Summary
    total_elapsed = time.time() - t_start
    print()
    print("=" * 70)
    print(f"  PRUNED+SVI WALK-FORWARD — {len(results)} months retrained")
    print(f"  Total elapsed: {total_elapsed / 3600:.2f}h")
    print("=" * 70)
    if results:
        print(
            f"  {'Month':>8s}  {'Baseline':>10s}  {'Pruned+SVI':>10s}  "
            f"{'Delta':>8s}  {'RankIC':>8s}  {'Feats':>6s}  {'Trials':>7s}"
        )
        print("-" * 70)
        for r in results:
            base = r["baseline_qlike"] or 0
            print(
                f"  {r['month']:>8s}  {base:>10.4f}  {r['pruned_svi_qlike']:>10.4f}  "
                f"{r['delta_pct']:>+7.1f}%  {r['rank_ic']:>8.3f}  "
                f"{r['n_features']:>6d}  {r['n_trials']:>7d}"
            )
        valid = [r for r in results if r["baseline_qlike"]]
        if valid:
            avg_base = float(np.mean([r["baseline_qlike"] for r in valid]))
            avg_new = float(np.mean([r["pruned_svi_qlike"] for r in valid]))
            avg_delta = (avg_new / avg_base - 1) * 100
            print("-" * 70)
            print(
                f"  {'AVG':>8s}  {avg_base:>10.4f}  {avg_new:>10.4f}  "
                f"{avg_delta:>+7.1f}%"
            )
        print("=" * 70)

        # Persist the summary
        summary_path = output_dir / "summary.json"
        summary_path.write_text(json.dumps({
            "total_hours": total_elapsed / 3600,
            "n_months": len(results),
            "avg_delta_pct": avg_delta if valid else None,
            "per_month": results,
        }, indent=2, default=str))
        logger.info("Summary saved to %s", summary_path)


if __name__ == "__main__":
    main()
