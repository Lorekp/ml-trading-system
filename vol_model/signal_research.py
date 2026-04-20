"""Shared training infrastructure for signal-improvement research threads.

Each research thread (HAR-residual, quantile, PatchTST) produces test-window
predictions on the SAME per-month data split. This module factors out:

  1. `SpotCheckContext`: one-shot loader of the full feature matrix,
     labels, pruned-feature list, TransVol checkpoints, SVI features,
     and production baseline predictions.

  2. `build_month_data(ctx, month_str)`: returns a `MonthData` object
     with train/val/test splits, winsorization bounds, TransVol
     embeddings, and baseline predictions -- matches the production
     `run_walkforward_pruned_svi.py` wiring byte-for-byte.

  3. Thread-specific trainers:
       * ``train_pruned_svi_lgbm`` (control/baseline -- identical to
         production recipe, used to sanity-check our harness against
         the saved production predictions)
       * ``train_quantile_lgbm`` (Thread 2)
       * ``train_har_residual_lgbm`` (Thread 3)
       * PatchTST trainer lives in a separate module (heavier imports).

  4. `evaluate_candidate`: computes QLIKE / RankIC / RMSE / MZ / DM
     vs baseline, returns a `ThreadResult` row for the results CSV.

The design mirrors the existing production script but is split into
reusable pieces so all three threads share identical per-month wiring.
Anything that looks like a subtle deviation from production is called
out in comments.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
from scipy.stats import spearmanr

from vol_arb.paths import (
    CURRENCY_FOR_ASSET,
    drop_features_path,
    pruned_features_path,
    pruned_svi_model_dir,
    svi_features_path,
    transvol_dir,
)
from vol_model.config import VolModelConfig
from vol_model.evaluate import diebold_mariano_test, mincer_zarnowitz
from vol_model.har import build_har_residual_target, reconstruct_from_residual
from vol_model.labels import load_labels
from vol_model.model import qlike_loss, qlike_metric, qlike_objective
from vol_model.pipeline import build_dataset
from vol_model.run_walkforward import OptunaPatience, _make_objective
from vol_model.run_walkforward_hybrid import _extract_compact_embeddings

logger = logging.getLogger("signal_research")


# SHAP-tail drop features (Stage 15.9). Production constant.
DROP_FEATURES = (
    "ffd_rv", "ffd_rv_lag3", "ffd_rv_ema10", "ffd_rv_ema50",
    "noise_level", "noise_level_lag1", "noise_level_ema10", "noise_level_ema50",
)
ADD_SVI_FEATURES = ("svi_atm_1d",)

# Tier 1/2 feature pools — augmented onto the production base when a thread
# requests them. Populated in load_spot_check_context when include_extras=True.
ETH_TSRV_FEATURES = ("eth_tsrv_mean_1d", "eth_tsrv_mean_1w", "eth_tsrv_ema50")
RQ_FEATURES = ("rq_1d", "rq_1w")
SVI_SPOT_FEATURES = ("svi_atm_spot",)
HAR_PD_FEATURES = (
    "har_pd_R1_a005", "har_pd_R1_a01", "har_pd_R1_a03",
    "har_pd_R2_b005", "har_pd_R2_b01", "har_pd_R2_b03",
)
# Session E Candidate A: full SVI surface (7 levels + 4 deltas). Levels live in
# the existing SVI sidecar but are unused by production; deltas are 1h .diff.
# See vol_model/features/svi_surface.py for the compute function.
from vol_model.features.svi_surface import (  # noqa: E402
    SVI_SURFACE_FEATURES,
    compute_svi_surface_features,
)

# Session F Candidate B: Hawkes-OFI bivariate exp-kernel MLE features
# (5 levels + 2 deltas). Sidecar at vol_arb/output/hawkes_ofi_features_15m.parquet
# from `python -m vol_arb.build_hawkes_ofi_sidecar`.
from vol_model.features.hawkes_ofi import (  # noqa: E402
    HAWKES_OFI_FEATURES,
    compute_hawkes_ofi_features,
)

# Session F Candidate C: rough-Heston SDE-calibrated latent-parameter features
# (4 levels + 4 deltas). Sidecar at vol_arb/output/rough_heston_features_15m.parquet
# from `python -m vol_arb.build_rough_heston_sidecar`.
from vol_model.features.rough_heston import (  # noqa: E402
    ROUGH_HESTON_FEATURES,
    compute_rough_heston_features,
)

# Session H Candidate B: perp funding-rate state vector (5 features).
# Sidecar at vol_arb/output/funding_features_15m.parquet from
# `python -m vol_arb.build_funding_sidecar`.
from vol_model.features.funding import (  # noqa: E402
    FUNDING_FEATURES,
    compute_funding_features,
)

# Session H Candidate C: Variance Risk Premium (IV² − RV²) state vector
# (5 features). Sidecar at vol_arb/output/vrp_features_15m.parquet from
# `python -m vol_arb.build_vrp_sidecar`.
from vol_model.features.vrp import (  # noqa: E402
    VRP_FEATURES,
    compute_vrp_features,
)

# Tier-to-feature-pool dispatch. Consumed by build_month_data_augmented.
# Tuple layout (9 slots):
#   (eth_cols, rq_cols, svi_extra_cols, har_pd_cols,
#    svi_surface_cols, hawkes_ofi_cols, rough_heston_cols,
#    funding_cols, vrp_cols).
#
# NOTE: As more candidates land, this tuple is getting long; refactoring
# to a dict structure {tier: {attr: cols}} is an ongoing follow-up — kept
# as a 9-tuple for now so Session H doesn't land alongside a refactor.
TIER_FEATURE_MAP: dict[
    str,
    tuple[
        tuple[str, ...], tuple[str, ...], tuple[str, ...],
        tuple[str, ...], tuple[str, ...], tuple[str, ...],
        tuple[str, ...], tuple[str, ...], tuple[str, ...],
    ],
] = {
    "control":                   ((), (), (), (), (), (), (), (), ()),
    "eth":                       (ETH_TSRV_FEATURES, (), (), (), (), (), (), (), ()),
    "rq":                        ((), RQ_FEATURES, (), (), (), (), (), (), ()),
    "eth_and_rq":                (ETH_TSRV_FEATURES, RQ_FEATURES, (), (), (), (), (), (), ()),
    "svi_spot":                  ((), (), SVI_SPOT_FEATURES, (), (), (), (), (), ()),
    "har_pd":                    ((), (), (), HAR_PD_FEATURES, (), (), (), (), ()),
    # Session E Candidate A (REJECTED at pre-screen):
    "svi_surface":               ((), (), (), (), SVI_SURFACE_FEATURES, (), (), (), ()),
    "har_pd_and_svi_surface":    ((), (), (), HAR_PD_FEATURES, SVI_SURFACE_FEATURES, (), (), (), ()),
    # Session F Candidate B (REJECTED): Hawkes-OFI.
    "hawkes_ofi":                ((), (), (), (), (), HAWKES_OFI_FEATURES, (), (), ()),
    "har_pd_and_hawkes_ofi":     ((), (), (), HAR_PD_FEATURES, (), HAWKES_OFI_FEATURES, (), (), ()),
    # Session F/G Candidate C (REJECTED): rough-Heston SDE calibration.
    "rough_heston":              ((), (), (), (), (), (), ROUGH_HESTON_FEATURES, (), ()),
    "har_pd_and_rough_heston":   ((), (), (), HAR_PD_FEATURES, (), (), ROUGH_HESTON_FEATURES, (), ()),
    # Session H Candidate B: perp funding-rate state vector.
    # The standalone tier measures raw information; the stacked tier is
    # the decision metric (adds on top of current production).
    "funding":                   ((), (), (), (), (), (), (), FUNDING_FEATURES, ()),
    "har_pd_and_funding":        ((), (), (), HAR_PD_FEATURES, (), (), (), FUNDING_FEATURES, ()),
    # Session H Candidate C: Variance Risk Premium.
    "vrp":                       ((), (), (), (), (), (), (), (), VRP_FEATURES),
    "har_pd_and_vrp":            ((), (), (), HAR_PD_FEATURES, (), (), (), (), VRP_FEATURES),
}


# ---------- Contexts ----------


@dataclass
class SpotCheckContext:
    """Everything shared across spot-check months for one asset."""
    asset: str
    vol_config: VolModelConfig
    X_full: pd.DataFrame
    y_full: pd.Series
    pruned_base_cols: list[str]
    tv_cols: list[str]
    labels: pd.DataFrame
    svi_add: pd.DataFrame
    transvol_dir_path: Path
    baseline_predictions: pd.Series  # production pruned_svi_v1 predictions
    data_start: pd.Timestamp
    # Tier 1/2 optional feature pools (populated when include_extras=True).
    # Empty DataFrames when disabled so downstream augmenters are no-ops.
    eth_tsrv_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    rq_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    svi_spot_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    har_pd_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Session E Candidate A: SVI surface (already +15m-aligned at load time).
    svi_surface_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Session F Candidate B: Hawkes-OFI (already +15m-aligned at load time).
    hawkes_ofi_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Session F Candidate C: rough-Heston (already +15m-aligned at load time).
    rough_heston_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Session H Candidate B: perp funding rate (already +15m-aligned at load time).
    funding_features: pd.DataFrame = field(default_factory=pd.DataFrame)
    # Session H Candidate C: VRP = IV² − RV² (already +15m-aligned at load time).
    vrp_features: pd.DataFrame = field(default_factory=pd.DataFrame)


@dataclass
class MonthData:
    """Training data + metadata for a single spot-check month."""
    month_str: str
    M: pd.Timestamp            # test start
    V: pd.Timestamp            # val start
    test_end: pd.Timestamp
    X_month: pd.DataFrame      # full aligned feature matrix (before winsorization)
    y_month: pd.Series
    train_mask: pd.Series
    val_mask: pd.Series
    test_mask: pd.Series
    lo: pd.Series              # winsorization lower (train 1% quantile)
    hi: pd.Series              # winsorization upper (train 99% quantile)
    baseline_preds_test: np.ndarray  # production predictions aligned to test window

    @property
    def X_tr_w(self) -> pd.DataFrame:
        return self.X_month[self.train_mask].clip(self.lo, self.hi, axis=1)

    @property
    def y_tr(self) -> pd.Series:
        return self.y_month[self.train_mask]

    @property
    def X_val_w(self) -> pd.DataFrame:
        return self.X_month[self.val_mask].clip(self.lo, self.hi, axis=1)

    @property
    def y_val(self) -> pd.Series:
        return self.y_month[self.val_mask]

    @property
    def X_test_w(self) -> pd.DataFrame:
        return self.X_month[self.test_mask].clip(self.lo, self.hi, axis=1)

    @property
    def y_test(self) -> pd.Series:
        return self.y_month[self.test_mask]


@dataclass
class ThreadResult:
    method: str
    month: str
    qlike: float
    baseline_qlike: float
    delta_pct: float
    rank_ic: float
    rmse: float
    mz_alpha: float
    mz_beta: float
    mz_well_calibrated: bool
    dm_tstat: float
    dm_pval: float
    dm_a_better: bool
    n_train: int
    n_test: int
    n_trials: int
    time_s: float
    extra: dict = field(default_factory=dict)

    def as_row(self) -> dict:
        row = {
            "method": self.method,
            "month": self.month,
            "qlike": self.qlike,
            "baseline_qlike": self.baseline_qlike,
            "delta_pct": self.delta_pct,
            "rank_ic": self.rank_ic,
            "rmse": self.rmse,
            "mz_alpha": self.mz_alpha,
            "mz_beta": self.mz_beta,
            "mz_well_calibrated": self.mz_well_calibrated,
            "dm_tstat": self.dm_tstat,
            "dm_pval": self.dm_pval,
            "dm_a_better": self.dm_a_better,
            "n_train": self.n_train,
            "n_test": self.n_test,
            "n_trials": self.n_trials,
            "time_s": self.time_s,
        }
        for k, v in self.extra.items():
            row[f"extra_{k}"] = v
        return row


# ---------- Loaders ----------


def _load_drop_features(asset: str) -> list[str]:
    path = drop_features_path(asset)
    if path.exists():
        with open(path) as f:
            return [line.strip() for line in f if line.strip()]
    return list(DROP_FEATURES)


def _load_svi_atm_1d(currency: str, *, extra_cols: tuple[str, ...] = ()) -> pd.DataFrame:
    """Load svi_atm_1d (+ optional extras) with the mandatory +15 min leakage shift.

    Mirrors ``run_walkforward_pruned_svi._load_svi_atm_1d`` byte-for-byte
    for the control case. When ``extra_cols`` is non-empty, additional SVI
    columns (e.g. ``svi_atm_spot``) are appended with the same +15min shift.
    """
    svi_path = svi_features_path(currency)
    svi = pd.read_parquet(svi_path)
    if "timestamp" in svi.columns and not isinstance(svi.index, pd.DatetimeIndex):
        svi = svi.set_index("timestamp")
    if not isinstance(svi.index, pd.DatetimeIndex):
        svi.index = pd.to_datetime(svi.index, utc=True)
    requested = list(ADD_SVI_FEATURES) + [c for c in extra_cols if c not in ADD_SVI_FEATURES]
    missing = [c for c in requested if c not in svi.columns]
    if missing:
        logger.warning(
            "SVI features missing cols %s; loaded available subset. "
            "Regenerate svi_features_15m.parquet via "
            "`python -m vol_arb.run_pipeline --currency %s --svi`.",
            missing, currency,
        )
        requested = [c for c in requested if c in svi.columns]
    svi_add = svi[requested].copy()
    svi_add.index = svi_add.index + pd.Timedelta(minutes=15)
    return svi_add


def _load_baseline_predictions(asset: str) -> pd.Series:
    """Load production pruned_svi_v1 predictions for DM comparison."""
    path = Path("vol_model/output") / f"{asset}_pruned_svi_predictions.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"baseline predictions not found at {path}; "
            f"production pruned_svi_v1 must be built first."
        )
    df = pd.read_parquet(path)
    if "y_pred" not in df.columns:
        # May be a single-column parquet
        df = df.rename(columns={df.columns[0]: "y_pred"})
    series = df["y_pred"]
    if not isinstance(series.index, pd.DatetimeIndex):
        series.index = pd.to_datetime(series.index, utc=True)
    return series


def load_spot_check_context(
    asset: str = "BTCUSDT",
    *,
    include_extras: bool = False,
) -> SpotCheckContext:
    """Load all data needed for spot-checks (called once, reused across months).

    When ``include_extras=True``, the Tier 1/2 feature pools (ETH cross-asset
    TSRV, Realized Quarticity, SVI-ATM spot-vol) are pre-loaded into separate
    ctx slots so ``build_month_data_augmented`` can splice them into the
    feature matrix per thread without mutating the control baseline.

    Control (``include_extras=False``) is byte-identical to the prior session's
    context: pruned_base + TransVol + svi_atm_1d (44 features).
    """
    logger.info("Loading spot-check context for %s (extras=%s)...", asset, include_extras)
    currency = CURRENCY_FOR_ASSET[asset]

    # Build dataset with extras enabled only when requested. The extra
    # columns are dropped from pruned_base_cols regardless — they're spliced
    # in per-thread via build_month_data_augmented, not as part of the
    # baseline control feature set.
    vol_cfg_kwargs = dict(asset=asset)
    if include_extras:
        vol_cfg_kwargs.update(
            include_eth_tsrv=True,
            include_rq=True,
            # svi_spot_vol lives in the SVI features parquet (loaded
            # separately via _load_svi_atm_1d with extra_cols).
        )
    vol_config = VolModelConfig(**vol_cfg_kwargs)
    X, y, _t1 = build_dataset(vol_config)

    # Compute tv_cols and pruned_base_cols on the BASE feature matrix (pre-
    # extras join). The TransVol normalizer was fit on this schema; adding
    # extras would blow up the column count and break normalizer.transform.
    pruned_path = pruned_features_path(asset)
    pruned = pd.read_csv(pruned_path, header=None)[0].tolist()
    pruned_cols_v1 = [c for c in pruned if c in X.columns]
    drop_set = set(_load_drop_features(asset))
    pruned_base_cols = [c for c in pruned_cols_v1 if c not in drop_set]

    drop_prefixes = ("ofi_", "depth_mid", "oi_change", "oi_zscore", "oi_acceleration")
    tv_cols = [c for c in X.columns if not c.startswith(drop_prefixes)]

    # Now join the Tier 1 extras sidecar (ETH-TSRV + RQ at 15m, built via
    # `python -m vol_model.run_build_extra_features`). Joined AFTER tv_cols so
    # the TransVol normalizer (trained on the base 101-col schema) is not
    # affected. Extras are consumed via ctx.eth_tsrv_features / ctx.rq_features
    # by build_month_data_augmented.
    if include_extras:
        extras_path = Path("vol_model/output") / f"{asset}_features_extras_15m.parquet"
        if extras_path.exists():
            extras = pd.read_parquet(extras_path)
            common = X.index.intersection(extras.index)
            if len(common) >= 0.99 * len(X):
                X = X.join(extras, how="left")
                logger.info(
                    "Joined extras sidecar: +%d cols (ETH-TSRV %d + RQ %d × 5 windowing variants)",
                    extras.shape[1],
                    sum(c.startswith("eth_") for c in extras.columns),
                    sum(c.startswith("rq_") for c in extras.columns),
                )
            else:
                logger.warning(
                    "extras index coverage %.2f%% vs X; skipping join",
                    100.0 * len(common) / max(1, len(X)),
                )
        else:
            logger.warning(
                "Extras sidecar missing (%s); Tier 1 threads will have empty pools. "
                "Run `python -m vol_model.run_build_extra_features --asset %s` first.",
                extras_path, asset,
            )

    labels = load_labels(vol_config)

    # Extract extras from X_full into their own ctx slots. This keeps the
    # control feature set clean and lets build_month_data_augmented pull the
    # right subset per thread without scattering column-selection logic.
    eth_pool_cols = [c for c in ETH_TSRV_FEATURES if c in X.columns]
    rq_pool_cols = [c for c in RQ_FEATURES if c in X.columns]
    eth_df = X[eth_pool_cols].copy() if eth_pool_cols else pd.DataFrame()
    rq_df = X[rq_pool_cols].copy() if rq_pool_cols else pd.DataFrame()

    if include_extras:
        missing_eth = [c for c in ETH_TSRV_FEATURES if c not in eth_pool_cols]
        missing_rq = [c for c in RQ_FEATURES if c not in rq_pool_cols]
        if missing_eth:
            logger.warning("ETH pool missing columns: %s (ETH thread will be partial)", missing_eth)
        if missing_rq:
            logger.warning("RQ pool missing columns: %s (RQ thread will be partial)", missing_rq)

    # SVI-spot (Tier 2): load from SVI parquet IF the column exists there.
    # If not, svi_spot_features is empty and any thread requesting svi_spot
    # will log a warning — callers can defer the thread.
    svi_extras: tuple[str, ...] = ()
    svi_add = _load_svi_atm_1d(currency)
    if include_extras:
        try:
            svi_with_spot = _load_svi_atm_1d(currency, extra_cols=SVI_SPOT_FEATURES)
            svi_spot_only = svi_with_spot[list(SVI_SPOT_FEATURES)] if all(
                c in svi_with_spot.columns for c in SVI_SPOT_FEATURES
            ) else pd.DataFrame()
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load SVI-spot extras: %s", exc)
            svi_spot_only = pd.DataFrame()
    else:
        svi_spot_only = pd.DataFrame()

    baseline_predictions = _load_baseline_predictions(asset)

    # HAR-PD sidecar (Tier 3): separate parquet built via
    # `python -m vol_model.features.har_pd --asset {asset}`. Features:
    # R1 (signed-flow exponential) + R2 (realized-variance exponential) at
    # 3 kernel decays each, 6 raw features total. Strictly past-only per the
    # causality unit test (test_har_pd.py::test_causality_mutating_future_*).
    har_pd_df = pd.DataFrame()
    if include_extras:
        har_pd_path = Path("vol_model/output") / f"{asset}_features_har_pd_15m.parquet"
        if har_pd_path.exists():
            har_pd_df = pd.read_parquet(har_pd_path)
            # Session E correctness fix: apply the same -15m shift that
            # production's _load_har_pd (run_walkforward_pruned_svi_har_pd.py)
            # applies. Without this, the gauntlet uses HAR-PD data with one
            # fewer bar of context than production would see (the bar ending
            # at T is discarded). Non-leaky either way — but the gauntlet
            # pessimistically underestimates HAR-PD's information content.
            # Aligns the gauntlet's HAR-PD evaluation with production.
            har_pd_df.index = har_pd_df.index - pd.Timedelta(minutes=15)
            logger.info(
                "Loaded HAR-PD sidecar: %d rows x %d cols (-15m aligned) from %s",
                len(har_pd_df), har_pd_df.shape[1], har_pd_path,
            )
        else:
            logger.warning(
                "HAR-PD sidecar missing (%s); har_pd thread will be empty. "
                "Run `python -m vol_model.features.har_pd --asset %s` first.",
                har_pd_path, asset,
            )

    # Session E Candidate A: SVI surface features (7 levels + 4 .diff(4) deltas).
    # Computed directly from the existing SVI sidecar — no new parquet.
    # Applies the SAME +15m alignment shift as svi_atm_1d so the gauntlet
    # measures what production would see (not the 15-min-pessimistic alignment
    # the HAR-PD gauntlet uses, which is a known limitation from Session C).
    svi_surface_df = pd.DataFrame()
    if include_extras:
        try:
            svi_raw_path = Path("vol_arb/output/svi_features_15m.parquet")
            if svi_raw_path.exists():
                svi_raw = pd.read_parquet(svi_raw_path)
                if "timestamp" in svi_raw.columns and not isinstance(
                    svi_raw.index, pd.DatetimeIndex,
                ):
                    svi_raw = svi_raw.set_index("timestamp")
                svi_surface_df = compute_svi_surface_features(svi_raw)
                # Apply the same +15m shift as _load_svi_atm_1d — aligns
                # feature index with LABEL timestamp so the gauntlet
                # measures production-equivalent feature availability.
                svi_surface_df.index = svi_surface_df.index + pd.Timedelta(minutes=15)
                logger.info(
                    "Loaded SVI surface features: %d rows x %d cols (+15m aligned)",
                    len(svi_surface_df), svi_surface_df.shape[1],
                )
            else:
                logger.warning(
                    "SVI raw sidecar missing (%s); svi_surface thread will be empty.",
                    svi_raw_path,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not compute SVI surface features: %s", exc)
            svi_surface_df = pd.DataFrame()

    # Session F Candidate B: Hawkes-OFI sidecar (built by
    # `python -m vol_arb.build_hawkes_ofi_sidecar`). Already 15m-cadence;
    # apply +15m shift (same as SVI-surface) to align bucket-end stamp with
    # label decision time.
    hawkes_ofi_df = pd.DataFrame()
    if include_extras:
        try:
            hawkes_path = Path("vol_arb/output/hawkes_ofi_features_15m.parquet")
            if hawkes_path.exists():
                hawkes_raw = pd.read_parquet(hawkes_path)
                hawkes_ofi_df = compute_hawkes_ofi_features(hawkes_raw)
                hawkes_ofi_df.index = hawkes_ofi_df.index + pd.Timedelta(minutes=15)
                logger.info(
                    "Loaded Hawkes-OFI features: %d rows x %d cols (+15m aligned)",
                    len(hawkes_ofi_df), hawkes_ofi_df.shape[1],
                )
            else:
                logger.warning(
                    "Hawkes-OFI sidecar missing (%s); hawkes_ofi thread will be empty.",
                    hawkes_path,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load Hawkes-OFI features: %s", exc)
            hawkes_ofi_df = pd.DataFrame()

    # Session F Candidate C: rough-Heston sidecar (built by
    # `python -m vol_arb.build_rough_heston_sidecar`). Same alignment convention.
    rough_heston_df = pd.DataFrame()
    if include_extras:
        try:
            rh_path = Path("vol_arb/output/rough_heston_features_15m.parquet")
            if rh_path.exists():
                rh_raw = pd.read_parquet(rh_path)
                rough_heston_df = compute_rough_heston_features(rh_raw)
                rough_heston_df.index = rough_heston_df.index + pd.Timedelta(minutes=15)
                logger.info(
                    "Loaded rough-Heston features: %d rows x %d cols (+15m aligned)",
                    len(rough_heston_df), rough_heston_df.shape[1],
                )
            else:
                logger.warning(
                    "Rough-Heston sidecar missing (%s); rough_heston thread will be empty.",
                    rh_path,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load rough-Heston features: %s", exc)
            rough_heston_df = pd.DataFrame()

    # Session H Candidate B: perp funding-rate sidecar (built by
    # `python -m vol_arb.build_funding_sidecar`). Same +15m alignment
    # convention as the options sidecars above — at training decision
    # time T+15m the feature uses info ≤ T (conservatively causal).
    funding_df = pd.DataFrame()
    if include_extras:
        try:
            fund_path = Path("vol_arb/output/funding_features_15m.parquet")
            if fund_path.exists():
                fund_raw = pd.read_parquet(fund_path)
                funding_df = compute_funding_features(fund_raw)
                funding_df.index = funding_df.index + pd.Timedelta(minutes=15)
                logger.info(
                    "Loaded funding features: %d rows x %d cols (+15m aligned)",
                    len(funding_df), funding_df.shape[1],
                )
            else:
                logger.warning(
                    "Funding sidecar missing (%s); funding thread will be empty.",
                    fund_path,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load funding features: %s", exc)
            funding_df = pd.DataFrame()

    # Session H Candidate C: Variance Risk Premium sidecar (built by
    # `python -m vol_arb.build_vrp_sidecar`).
    vrp_df = pd.DataFrame()
    if include_extras:
        try:
            vrp_path = Path("vol_arb/output/vrp_features_15m.parquet")
            if vrp_path.exists():
                vrp_raw = pd.read_parquet(vrp_path)
                vrp_df = compute_vrp_features(vrp_raw)
                vrp_df.index = vrp_df.index + pd.Timedelta(minutes=15)
                logger.info(
                    "Loaded VRP features: %d rows x %d cols (+15m aligned)",
                    len(vrp_df), vrp_df.shape[1],
                )
            else:
                logger.warning(
                    "VRP sidecar missing (%s); vrp thread will be empty.",
                    vrp_path,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not load VRP features: %s", exc)
            vrp_df = pd.DataFrame()

    return SpotCheckContext(
        asset=asset,
        vol_config=vol_config,
        X_full=X,
        y_full=y,
        pruned_base_cols=pruned_base_cols,
        tv_cols=tv_cols,
        labels=labels,
        svi_add=svi_add,
        transvol_dir_path=transvol_dir(asset),
        baseline_predictions=baseline_predictions,
        data_start=X.index.min(),
        eth_tsrv_features=eth_df,
        rq_features=rq_df,
        svi_spot_features=svi_spot_only,
        har_pd_features=har_pd_df,
        svi_surface_features=svi_surface_df,
        hawkes_ofi_features=hawkes_ofi_df,
        rough_heston_features=rough_heston_df,
        funding_features=funding_df,
        vrp_features=vrp_df,
    )


# ---------- Per-month dataset builder ----------


def build_month_data(ctx: SpotCheckContext, month_str: str) -> MonthData | None:
    """Build a MonthData object for one spot-check month.

    Returns None and logs a warning if required artifacts (TransVol
    checkpoint, baseline predictions) are missing.
    """
    M = pd.Timestamp(month_str + "-01", tz="UTC")
    test_end = M + pd.DateOffset(months=1)
    V = max(M - pd.DateOffset(months=4), ctx.data_start + pd.DateOffset(months=12))

    tv_path = ctx.transvol_dir_path / f"{month_str}_transvol.pt"
    if not tv_path.exists():
        logger.warning("%s: no TransVol checkpoint at %s -- skipping month", month_str, tv_path)
        return None

    # Extract TransVol embeddings for the full window
    X_tv = ctx.X_full[ctx.tv_cols]
    try:
        emb_df = _extract_compact_embeddings(tv_path, X_tv, ctx.labels, test_end)
    except Exception:  # noqa: BLE001
        logger.exception("%s: TransVol embedding extraction failed -- skipping", month_str)
        return None

    X_pruned = ctx.X_full[ctx.pruned_base_cols]
    common = X_pruned.index.intersection(emb_df.index).intersection(ctx.svi_add.index)
    X_month = pd.concat(
        [emb_df.loc[common], X_pruned.loc[common], ctx.svi_add.loc[common]],
        axis=1,
    )
    common_y = X_month.index.intersection(ctx.y_full.index)
    X_month = X_month.loc[common_y]
    y_month = ctx.y_full.loc[common_y]

    # Drop NaN in SVI (same as production)
    X_month = X_month.dropna(subset=list(ADD_SVI_FEATURES))
    y_month = y_month.loc[X_month.index]

    train_mask = X_month.index < V
    val_mask = (X_month.index >= V) & (X_month.index < M)
    test_mask = (X_month.index >= M) & (X_month.index < test_end)

    n_train, n_val, n_test = train_mask.sum(), val_mask.sum(), test_mask.sum()
    if n_train < 100 or n_val == 0 or n_test == 0:
        logger.warning(
            "%s: insufficient data (train=%d, val=%d, test=%d) -- skipping",
            month_str, n_train, n_val, n_test,
        )
        return None

    X_tr = X_month[train_mask]
    lo = X_tr.quantile(0.01)
    hi = X_tr.quantile(0.99)

    # Align baseline predictions to test window
    test_idx = X_month.index[test_mask]
    baseline_slice = ctx.baseline_predictions.reindex(test_idx)
    if baseline_slice.isna().any():
        logger.warning(
            "%s: %d/%d test rows have no baseline prediction (will be dropped from DM test)",
            month_str, int(baseline_slice.isna().sum()), len(baseline_slice),
        )

    return MonthData(
        month_str=month_str,
        M=M,
        V=V,
        test_end=test_end,
        X_month=X_month,
        y_month=y_month,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        lo=lo,
        hi=hi,
        baseline_preds_test=baseline_slice.to_numpy(),
    )


# ---------- LGBM training helpers ----------


def _run_optuna_qlike(
    X_tr: pd.DataFrame, y_tr: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    *, n_trials: int, patience: int,
) -> optuna.Study:
    """Optuna HP search with QLIKE objective (production recipe)."""
    dtrain = lgb.Dataset(
        X_tr.values, label=y_tr.values, free_raw_data=False,
        params={"feature_pre_filter": False},
    )
    dval = lgb.Dataset(
        X_val.values, label=y_val.values, reference=dtrain, free_raw_data=False,
    )
    dtrain.construct()
    dval.construct()

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=TPESampler(seed=42),
        pruner=MedianPruner(n_startup_trials=5),
    )
    obj = _make_objective(dtrain, dval, X_val.values, y_val.values, "cpu")
    study.optimize(
        obj, n_trials=n_trials,
        callbacks=[OptunaPatience(patience)],
        show_progress_bar=False,
    )
    return study


def _retrain_best_lgbm(
    study: optuna.Study,
    X_train: pd.DataFrame, y_train: pd.Series,
    X_val: pd.DataFrame, y_val: pd.Series,
    *, objective_fn=None, alpha: float | None = None,
) -> lgb.Booster:
    """Retrain on full train window with best params (production recipe)."""
    best = study.best_params.copy()
    best["subsample"] = 0.8
    best["device"] = "cpu"
    best["num_threads"] = -1
    best["verbosity"] = -1
    best["seed"] = 42
    if alpha is not None:
        best["objective"] = "quantile"
        best["alpha"] = alpha
        feval = None
    else:
        best["objective"] = objective_fn or qlike_objective
        feval = qlike_metric

    dtrain_full = lgb.Dataset(X_train.values, label=y_train.values, free_raw_data=False)
    dval = lgb.Dataset(X_val.values, label=y_val.values, reference=dtrain_full, free_raw_data=False)

    model = lgb.train(
        best, dtrain_full, num_boost_round=2000,
        valid_sets=[dval], feval=feval,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    return model


# ---------- Thread trainers ----------


def train_pruned_svi_baseline(month: MonthData, *, n_trials: int, patience: int) -> tuple[np.ndarray, int]:
    """Baseline: reproduce production pruned_svi_v1 on this month's data.

    Used as a sanity-check that our harness matches production numbers.
    Returns test-window predictions + trial count.
    """
    preds, n_used, _ = train_pruned_svi_baseline_with_model(
        month, n_trials=n_trials, patience=patience,
    )
    return preds, n_used


def train_pruned_svi_baseline_with_model(
    month: MonthData, *, n_trials: int, patience: int,
) -> tuple[np.ndarray, int, lgb.Booster]:
    """Same as train_pruned_svi_baseline but also returns the trained Booster.

    Used by the HQR conformal wrapper which needs val-window predictions from
    the same model (for fitting the heteroscedastic width estimator + the
    conformal multiplier). Keeping the existing baseline function untouched
    preserves its byte-identical behaviour for the main gauntlet rows.
    """
    study = _run_optuna_qlike(month.X_tr_w, month.y_tr, month.X_val_w, month.y_val,
                              n_trials=n_trials, patience=patience)
    model = _retrain_best_lgbm(study, month.X_tr_w, month.y_tr, month.X_val_w, month.y_val)
    preds = model.predict(month.X_test_w.values)
    return preds, len(study.trials), model


# ---------- Tier 1/2 augmented dispatchers ----------


def build_month_data_augmented(
    ctx: SpotCheckContext,
    month_str: str,
    *,
    thread: str,
) -> MonthData | None:
    """Build MonthData for a Tier 1/2 thread by augmenting the control feature set.

    `thread` must be a key of ``TIER_FEATURE_MAP``. The returned MonthData has
    the control 44-feature set (TransVol + pruned_base + svi_atm_1d) plus the
    thread-specific extras (ETH-TSRV / RQ / SVI-spot). Winsorization bounds
    are recomputed on the augmented train window so the extra columns get
    proper 1%/99% clipping.

    Control thread (``thread="control"``) returns byte-identical output to
    ``build_month_data`` — the augmentation paths are all no-ops.
    """
    if thread not in TIER_FEATURE_MAP:
        raise ValueError(
            f"unknown thread '{thread}'; expected one of {sorted(TIER_FEATURE_MAP)}"
        )
    (
        eth_cols, rq_cols, svi_extra_cols, har_pd_cols, svi_surface_cols,
        hawkes_ofi_cols, rough_heston_cols, funding_cols, vrp_cols,
    ) = TIER_FEATURE_MAP[thread]

    md = build_month_data(ctx, month_str)
    if md is None:
        return None

    extras: list[pd.DataFrame] = []
    if eth_cols:
        if ctx.eth_tsrv_features.empty:
            logger.warning("%s: ETH pool unavailable (ctx built without include_extras); skipping ETH", month_str)
        else:
            present = [c for c in eth_cols if c in ctx.eth_tsrv_features.columns]
            if present:
                extras.append(ctx.eth_tsrv_features[present].reindex(md.X_month.index))
    if rq_cols:
        if ctx.rq_features.empty:
            logger.warning("%s: RQ pool unavailable (ctx built without include_extras); skipping RQ", month_str)
        else:
            present = [c for c in rq_cols if c in ctx.rq_features.columns]
            if present:
                extras.append(ctx.rq_features[present].reindex(md.X_month.index))
    if svi_extra_cols:
        if ctx.svi_spot_features.empty:
            logger.warning("%s: SVI-spot unavailable (column not in SVI parquet); skipping", month_str)
        else:
            present = [c for c in svi_extra_cols if c in ctx.svi_spot_features.columns]
            if present:
                extras.append(ctx.svi_spot_features[present].reindex(md.X_month.index))
    if har_pd_cols:
        if ctx.har_pd_features.empty:
            logger.warning("%s: HAR-PD pool unavailable (sidecar missing); skipping HAR-PD", month_str)
        else:
            present = [c for c in har_pd_cols if c in ctx.har_pd_features.columns]
            if present:
                extras.append(ctx.har_pd_features[present].reindex(md.X_month.index))
    if svi_surface_cols:
        if ctx.svi_surface_features.empty:
            logger.warning("%s: SVI-surface unavailable; skipping svi_surface", month_str)
        else:
            present = [c for c in svi_surface_cols if c in ctx.svi_surface_features.columns]
            if present:
                extras.append(ctx.svi_surface_features[present].reindex(md.X_month.index))
    if hawkes_ofi_cols:
        if ctx.hawkes_ofi_features.empty:
            logger.warning("%s: Hawkes-OFI unavailable; skipping hawkes_ofi", month_str)
        else:
            present = [c for c in hawkes_ofi_cols if c in ctx.hawkes_ofi_features.columns]
            if present:
                extras.append(ctx.hawkes_ofi_features[present].reindex(md.X_month.index))
    if rough_heston_cols:
        if ctx.rough_heston_features.empty:
            logger.warning("%s: rough-Heston unavailable; skipping rough_heston", month_str)
        else:
            present = [c for c in rough_heston_cols if c in ctx.rough_heston_features.columns]
            if present:
                extras.append(ctx.rough_heston_features[present].reindex(md.X_month.index))
    if funding_cols:
        if ctx.funding_features.empty:
            logger.warning("%s: funding unavailable; skipping funding", month_str)
        else:
            present = [c for c in funding_cols if c in ctx.funding_features.columns]
            if present:
                extras.append(ctx.funding_features[present].reindex(md.X_month.index))
    if vrp_cols:
        if ctx.vrp_features.empty:
            logger.warning("%s: VRP unavailable; skipping vrp", month_str)
        else:
            present = [c for c in vrp_cols if c in ctx.vrp_features.columns]
            if present:
                extras.append(ctx.vrp_features[present].reindex(md.X_month.index))

    if not extras:
        # Thread is effectively control — return as-is
        return md

    X_augmented = pd.concat([md.X_month] + extras, axis=1)
    # Drop rows where any extra column is entirely NaN (e.g. ETH coverage
    # starts later than BTC). Each extra will have independent NaN coverage;
    # we conservatively drop rows where ANY thread-extra column is NaN so
    # the LGBM sees a complete augmented vector. This mirrors the existing
    # svi_atm_1d dropna pattern in build_month_data.
    extra_col_names = [c for df in extras for c in df.columns]
    X_augmented = X_augmented.dropna(subset=extra_col_names)
    y_augmented = md.y_month.loc[X_augmented.index]

    # Recompute winsorization on the augmented train window
    train_mask = X_augmented.index.isin(md.X_month.index[md.train_mask])
    val_mask = X_augmented.index.isin(md.X_month.index[md.val_mask])
    test_mask = X_augmented.index.isin(md.X_month.index[md.test_mask])
    X_tr = X_augmented[train_mask]
    lo = X_tr.quantile(0.01)
    hi = X_tr.quantile(0.99)

    test_idx = X_augmented.index[test_mask]
    baseline_slice = ctx.baseline_predictions.reindex(test_idx)

    return MonthData(
        month_str=month_str,
        M=md.M,
        V=md.V,
        test_end=md.test_end,
        X_month=X_augmented,
        y_month=y_augmented,
        train_mask=pd.Series(train_mask, index=X_augmented.index),
        val_mask=pd.Series(val_mask, index=X_augmented.index),
        test_mask=pd.Series(test_mask, index=X_augmented.index),
        lo=lo,
        hi=hi,
        baseline_preds_test=baseline_slice.to_numpy(),
    )


def train_feature_augmented_lgbm(
    month: MonthData, *, n_trials: int, patience: int,
) -> tuple[np.ndarray, int]:
    """Tier 1/2 trainer: identical recipe to train_pruned_svi_baseline.

    The only difference vs baseline is the feature matrix dimensionality —
    handled upstream by build_month_data_augmented. Keeping a separate
    trainer name makes gauntlet logs (and the results CSV's ``method`` column)
    unambiguous about which candidate produced each row.
    """
    study = _run_optuna_qlike(
        month.X_tr_w, month.y_tr, month.X_val_w, month.y_val,
        n_trials=n_trials, patience=patience,
    )
    model = _retrain_best_lgbm(study, month.X_tr_w, month.y_tr, month.X_val_w, month.y_val)
    preds = model.predict(month.X_test_w.values)
    return preds, len(study.trials)


def train_har_residual_lgbm(
    month: MonthData, *, n_trials: int, patience: int,
) -> tuple[np.ndarray, int, dict]:
    """Thread 3: HAR-residual target.

    Steps:
      1. Compute HAR_pred causally on the full y_month series.
      2. Build residual target: y_residual = y - HAR_pred.
      3. Train LGBM on (X_train, y_residual_train) with QLIKE objective.
      4. Predict residuals on test window; reconstruct full pred.

    HAR causality is guaranteed by fit_causal_har (tested separately).
    """
    residuals, har_pred = build_har_residual_target(month.y_month)

    # The HAR warmup NaN rows must be dropped from the training window.
    # For test: we must have HAR_pred available on every test row.
    har_valid_mask = har_pred.notna()
    train_valid = month.train_mask & har_valid_mask
    val_valid = month.val_mask & har_valid_mask
    test_valid = month.test_mask & har_valid_mask

    if test_valid.sum() < 100:
        raise RuntimeError(
            f"HAR warmup leaves only {test_valid.sum()} valid test rows in {month.month_str}"
        )

    X_tr = month.X_month[train_valid].clip(month.lo, month.hi, axis=1)
    X_val = month.X_month[val_valid].clip(month.lo, month.hi, axis=1)
    X_test = month.X_month[test_valid].clip(month.lo, month.hi, axis=1)
    y_tr_res = residuals[train_valid]
    y_val_res = residuals[val_valid]

    study = _run_optuna_qlike(X_tr, y_tr_res, X_val, y_val_res,
                              n_trials=n_trials, patience=patience)
    model = _retrain_best_lgbm(study, X_tr, y_tr_res, X_val, y_val_res)
    res_preds = model.predict(X_test.values)
    full_preds = reconstruct_from_residual(res_preds, har_pred[test_valid].values)

    # Align to the full test window (fill HAR-warmup-dropped rows with HAR fallback)
    full_test_idx = month.X_test_w.index
    preds_series = pd.Series(np.nan, index=full_test_idx)
    preds_series.loc[X_test.index] = full_preds
    if preds_series.isna().any():
        # Fall back to raw HAR on warmup rows (rare once we're past 2022)
        fallback = har_pred.reindex(full_test_idx)
        preds_series = preds_series.fillna(fallback)
        # Final fallback: y_train mean
        preds_series = preds_series.fillna(float(month.y_tr.mean()))

    extra = {
        "har_warmup_dropped": int(month.test_mask.sum() - test_valid.sum()),
        "har_pred_mean": float(har_pred[test_valid].mean()),
    }
    return preds_series.values, len(study.trials), extra


def build_patchtst_month_data(
    ctx: SpotCheckContext,
    month_str: str,
    patchtst_embeddings: pd.DataFrame,
) -> MonthData | None:
    """Per-month dataset with PatchTST embeddings substituted for TransVol's.

    The PatchTST columns (pt_0..pt_15) fill the role of tv_0..tv_15 in the
    LGBM feature matrix; pruned_base + svi_atm_1d wiring is byte-identical
    to ``build_month_data``. Used by the PatchTST thread so downstream
    LGBM training matches the production recipe contract exactly (16-dim
    encoder embeddings + pruned base + 1 SVI).
    """
    M = pd.Timestamp(month_str + "-01", tz="UTC")
    test_end = M + pd.DateOffset(months=1)
    V = max(M - pd.DateOffset(months=4), ctx.data_start + pd.DateOffset(months=12))

    X_pruned = ctx.X_full[ctx.pruned_base_cols]
    common = X_pruned.index.intersection(patchtst_embeddings.index).intersection(ctx.svi_add.index)
    X_month = pd.concat(
        [patchtst_embeddings.loc[common], X_pruned.loc[common], ctx.svi_add.loc[common]],
        axis=1,
    )
    common_y = X_month.index.intersection(ctx.y_full.index)
    X_month = X_month.loc[common_y]
    y_month = ctx.y_full.loc[common_y]
    X_month = X_month.dropna(subset=list(ADD_SVI_FEATURES))
    y_month = y_month.loc[X_month.index]

    train_mask = X_month.index < V
    val_mask = (X_month.index >= V) & (X_month.index < M)
    test_mask = (X_month.index >= M) & (X_month.index < test_end)
    if train_mask.sum() < 100 or val_mask.sum() == 0 or test_mask.sum() == 0:
        return None

    X_tr = X_month[train_mask]
    lo = X_tr.quantile(0.01)
    hi = X_tr.quantile(0.99)

    test_idx = X_month.index[test_mask]
    baseline_slice = ctx.baseline_predictions.reindex(test_idx)

    return MonthData(
        month_str=month_str,
        M=M, V=V, test_end=test_end,
        X_month=X_month, y_month=y_month,
        train_mask=train_mask, val_mask=val_mask, test_mask=test_mask,
        lo=lo, hi=hi,
        baseline_preds_test=baseline_slice.to_numpy(),
    )


def train_quantile_lgbm(
    month: MonthData, *, alpha: float, n_trials: int, patience: int,
) -> tuple[np.ndarray, int]:
    """Thread 2: quantile LGBM head at specified alpha.

    Standard LGBM quantile loss for Optuna; same feature matrix as
    baseline. Returns test-window quantile predictions.

    NOTE: Optuna objective in _make_objective uses QLIKE for the
    SEARCH; here we refit the final model with quantile objective.
    This is fine because QLIKE and pinball-at-0.5 rank the same
    hyperparameters monotonically for well-behaved densities, and
    the q50 QLIKE is the thing we care about anyway. q10 and q90
    are refit with quantile objective at the chosen alpha.
    """
    # Optuna on QLIKE for speed -- quantile objective search is much
    # slower and the optimum HPs are nearly identical for a log-vol
    # target at these scales.
    study = _run_optuna_qlike(month.X_tr_w, month.y_tr, month.X_val_w, month.y_val,
                              n_trials=n_trials, patience=patience)
    model = _retrain_best_lgbm(
        study, month.X_tr_w, month.y_tr, month.X_val_w, month.y_val,
        alpha=alpha,
    )
    preds = model.predict(month.X_test_w.values)
    return preds, len(study.trials)


# ---------- Evaluation ----------


def evaluate_candidate(
    preds: np.ndarray,
    month: MonthData,
    *,
    method: str,
    time_s: float,
    n_trials: int,
    extra: dict | None = None,
) -> ThreadResult:
    """Compute the full ThreadResult row for one (method, month) combo."""
    y_true = month.y_test.to_numpy()
    baseline = month.baseline_preds_test

    qlike = float(qlike_loss(y_true, preds))

    # Baseline comparison on overlapping valid rows
    valid = ~np.isnan(baseline) & ~np.isnan(preds)
    if valid.sum() >= 30:
        baseline_qlike = float(qlike_loss(y_true[valid], baseline[valid]))
        dm = diebold_mariano_test(y_true[valid], preds[valid], baseline[valid])
        dm_tstat = dm["t_stat"]
        dm_pval = dm["p_value"]
        dm_a_better = bool(dm["a_better"])
    else:
        baseline_qlike = float("nan")
        dm_tstat = float("nan")
        dm_pval = float("nan")
        dm_a_better = False

    delta_pct = (qlike / baseline_qlike - 1) * 100 if baseline_qlike and np.isfinite(baseline_qlike) else float("nan")
    rmse = float(np.sqrt(np.mean((y_true - preds) ** 2)))
    try:
        rank_ic = float(spearmanr(y_true, preds).statistic)
    except Exception:  # noqa: BLE001
        rank_ic = float("nan")

    mz = mincer_zarnowitz(y_true, preds)

    return ThreadResult(
        method=method,
        month=month.month_str,
        qlike=qlike,
        baseline_qlike=baseline_qlike,
        delta_pct=delta_pct,
        rank_ic=rank_ic,
        rmse=rmse,
        mz_alpha=mz["alpha"],
        mz_beta=mz["beta"],
        mz_well_calibrated=bool(mz["well_calibrated"]),
        dm_tstat=dm_tstat,
        dm_pval=dm_pval,
        dm_a_better=dm_a_better,
        n_train=int(month.train_mask.sum()),
        n_test=int(month.test_mask.sum()),
        n_trials=n_trials,
        time_s=time_s,
        extra=extra or {},
    )


# ---------- CRPS for Thread 2 ----------


def crps_gaussian(y_true: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Closed-form CRPS of a Gaussian forecast.

    CRPS(N(mu, sigma), y) = sigma * [ (y-mu)/sigma * (2*Phi((y-mu)/sigma) - 1)
                                     + 2*phi((y-mu)/sigma) - 1/sqrt(pi) ]

    Standard reference (Gneiting & Raftery 2007). We use this as the
    Gaussian baseline against which the quantile-derived distribution
    will be compared.
    """
    from scipy.stats import norm
    sigma = np.maximum(sigma, 1e-8)
    z = (y_true - mu) / sigma
    return float(np.mean(
        sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
    ))


def crps_from_quantiles(
    y_true: np.ndarray, q_10: np.ndarray, q_50: np.ndarray, q_90: np.ndarray,
) -> float:
    """CRPS approximation from 3 quantiles via pinball-loss aggregation.

    Per Bracher et al. 2021, an N-quantile CRPS approximation is
    ``2 * mean(pinball_loss)`` where pinball_loss(alpha) = (alpha - 1_{y<q_alpha}) * (y - q_alpha).
    With 3 quantiles {0.1, 0.5, 0.9}, this is a low-resolution but
    honest, distribution-free score.
    """
    quantiles = [(0.10, q_10), (0.50, q_50), (0.90, q_90)]
    total = 0.0
    for alpha, q in quantiles:
        under = (y_true < q).astype(float)
        total += float(np.mean((alpha - under) * (y_true - q)))
    return 2.0 * total / len(quantiles)


def spread_abs_error_correlation(
    y_true: np.ndarray, q_10: np.ndarray, q_50: np.ndarray, q_90: np.ndarray,
) -> float:
    """Does the predicted spread (q90-q10) correlate with |y_true - q50|?

    Positive correlation = uncertainty signal is real (the model
    "knows when it doesn't know"). Used as a Thread-2 KPI.
    """
    spread = q_90 - q_10
    abs_err = np.abs(y_true - q_50)
    if len(spread) < 30:
        return float("nan")
    return float(spearmanr(spread, abs_err).statistic)
