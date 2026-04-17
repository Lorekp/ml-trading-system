"""Unified spot-check runner for signal-improvement research.

Orchestrates HAR-residual / quantile / PatchTST trainings across the
regime substrate picked by ``spot_check_regimes.pick_regime_months``.
Each thread is trained + evaluated on the same per-month data split;
results are appended to a single CSV and a per-thread writeup table.

Usage:
    PYTHONPATH=. python -m vol_model.run_spot_check_signal_research \\
        --thread har_residual --n-trials 50 --patience 20
    PYTHONPATH=. python -m vol_model.run_spot_check_signal_research \\
        --thread all --n-trials 50
    # just-the-leakage probes
    PYTHONPATH=. python -m vol_model.run_spot_check_signal_research --probes-only

Output artifact:
    vol_model/output/signal_research_spot_check_results.csv
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vol_model.leakage_probes import (
    probe_label_alignment,
    probe_parquet_end_stamped,
    run_probe_battery,
)
from vol_model.signal_research import (
    ADD_SVI_FEATURES,
    TIER_FEATURE_MAP,
    SpotCheckContext,
    MonthData,
    ThreadResult,
    build_month_data,
    build_month_data_augmented,
    build_patchtst_month_data,
    crps_from_quantiles,
    crps_gaussian,
    evaluate_candidate,
    load_spot_check_context,
    spread_abs_error_correlation,
    train_feature_augmented_lgbm,
    train_har_residual_lgbm,
    train_pruned_svi_baseline,
    train_quantile_lgbm,
)
from vol_model.spot_check_regimes import pick_regime_months

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("spot_check_runner")

RESULTS_CSV = Path("vol_model/output/signal_research_spot_check_results.csv")
SUMMARY_MD = Path("vol_model/output/SIGNAL_RESEARCH_NOTES.md")


# ---------- Leakage probes (run once per candidate, not per month) ----------


def run_leakage_probes(ctx: SpotCheckContext) -> dict[str, bool]:
    """Run the structural leakage probes that don't require training.

    Thread-specific probes (causal HAR, causal PatchTST attention) live
    in their respective unit tests -- already pytest-gated. Here we run
    the cross-cutting probes that touch the actual spot-check data.
    """
    results: dict[str, bool] = {}

    # Probe C: label alignment (X.index == label timestamp; feature time = X.index - freq)
    freq_td = pd.Timedelta(minutes=15)
    feat_idx = ctx.X_full.index - freq_td
    lab_idx = ctx.X_full.index
    r = probe_label_alignment(feat_idx, lab_idx, expected_shift=freq_td, n_samples=5000)
    logger.info("[probe] label_alignment: %s -- %s", "PASS" if r.passed else "FAIL", r.detail)
    results["label_alignment"] = r.passed

    # Probe D: SVI parquet END-stamped at 1h (svi_features_1h.parquet base freq)
    # The +15m shift is applied in our loader; the SOURCE parquet still needs
    # to be 1h-aligned.
    from vol_arb.paths import svi_features_path, CURRENCY_FOR_ASSET
    svi_path = svi_features_path(CURRENCY_FOR_ASSET[ctx.asset])
    # Note: svi_features_15m.parquet is the 15m-merged version; we check
    # the ACTUAL source file the loader reads.
    # The 15m-merged parquet should be END-stamped at 15min; check that.
    r = probe_parquet_end_stamped(svi_path, expected_freq=pd.Timedelta(minutes=15))
    logger.info("[probe] svi_parquet_end_stamp: %s -- %s", "PASS" if r.passed else "FAIL", r.detail)
    results["svi_parquet_end_stamp"] = r.passed

    # Probe: baseline predictions parquet is END-stamped
    base_path = Path("vol_model/output") / f"{ctx.asset}_pruned_svi_predictions.parquet"
    r = probe_parquet_end_stamped(base_path, expected_freq=pd.Timedelta(minutes=15))
    logger.info("[probe] baseline_preds_end_stamp: %s -- %s", "PASS" if r.passed else "FAIL", r.detail)
    results["baseline_preds_end_stamp"] = r.passed

    return results


# ---------- Thread dispatchers ----------


def run_har_residual_for_month(month: MonthData, n_trials: int, patience: int) -> ThreadResult:
    t0 = time.time()
    preds, n_used, extra = train_har_residual_lgbm(month, n_trials=n_trials, patience=patience)
    return evaluate_candidate(
        preds, month,
        method="har_residual",
        time_s=time.time() - t0,
        n_trials=n_used,
        extra=extra,
    )


def run_patchtst_for_month(
    ctx: SpotCheckContext,
    month_str: str,
    *,
    n_trials: int,
    patience: int,
    patchtst_epochs: int = 60,
    patchtst_patience: int = 10,
    patchtst_checkpoint_root: Path = Path("vol_model/output/patchtst_spot_check"),
    fresh_control_preds: np.ndarray | None = None,
    fresh_control_test_index: pd.DatetimeIndex | None = None,
) -> ThreadResult | None:
    """Full PatchTST thread for one month: train encoder + extract + LGBM eval."""
    from vol_model.patchtst.config import PatchTSTConfig
    from vol_model.patchtst.extract import extract_compact_embeddings
    from vol_model.patchtst.train import train_patchtst

    t_all = time.time()
    M = pd.Timestamp(month_str + "-01", tz="UTC")
    test_end = M + pd.DateOffset(months=1)
    V = max(M - pd.DateOffset(months=4), ctx.data_start + pd.DateOffset(months=12))

    # Configure and train PatchTST encoder on [data_start, V), val [V, M)
    ckpt_dir = patchtst_checkpoint_root / month_str
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    cfg = PatchTSTConfig(
        max_epochs=patchtst_epochs,
        patience=patchtst_patience,
        train_end=V.tz_convert("UTC").strftime("%Y-%m-%d"),
        val_end=M.tz_convert("UTC").strftime("%Y-%m-%d"),
        checkpoint_dir=str(ckpt_dir),
    )
    logger.info("  [patchtst] encoder training (train_end=%s val_end=%s)...",
                cfg.train_end, cfg.val_end)
    t0 = time.time()
    best_ckpt = train_patchtst(cfg)
    encoder_time = time.time() - t0
    logger.info("  [patchtst] encoder trained in %.0fs", encoder_time)

    # Extract embeddings for the full window (up to test_end)
    t0 = time.time()
    X_tv = ctx.X_full[ctx.tv_cols]
    emb_df = extract_compact_embeddings(best_ckpt, X_tv, ctx.labels, test_end, column_prefix="pt")
    extract_time = time.time() - t0
    logger.info("  [patchtst] extracted %d rows of 16-dim embeddings in %.0fs",
                len(emb_df), extract_time)

    # Build LGBM month data with PatchTST embeddings
    month = build_patchtst_month_data(ctx, month_str, emb_df)
    if month is None:
        logger.warning("  [patchtst] insufficient data after alignment -- skipping %s", month_str)
        return None

    # Override baseline with fresh control predictions if supplied (so DM and
    # delta_pct are candidate-vs-fresh-control, not vs the stale saved parquet).
    if fresh_control_preds is not None and fresh_control_test_index is not None:
        if len(fresh_control_preds) == len(month.baseline_preds_test):
            # Alignment check: patchtst MonthData must have identical test index
            pt_idx = month.X_month.index[month.test_mask]
            if pt_idx.equals(fresh_control_test_index):
                month.baseline_preds_test = fresh_control_preds
                logger.info("  [patchtst] baseline overridden to fresh control predictions")
            else:
                logger.warning(
                    "  [patchtst] test index mismatch vs control; baseline NOT overridden"
                )

    # LGBM training: same recipe as production control
    logger.info("  [patchtst] LGBM training on (pt_0..pt_15 + pruned + svi_atm_1d)...")
    t0 = time.time()
    preds, n_used = train_pruned_svi_baseline(month, n_trials=n_trials, patience=patience)
    lgbm_time = time.time() - t0

    extra = {
        "encoder_time_s": encoder_time,
        "extract_time_s": extract_time,
        "lgbm_time_s": lgbm_time,
        "encoder_ckpt": str(best_ckpt),
    }
    return evaluate_candidate(
        preds, month,
        method="patchtst",
        time_s=time.time() - t_all,
        n_trials=n_used,
        extra=extra,
    )


def run_hqr_for_month(
    ctx: SpotCheckContext,
    month_str: str,
    *,
    n_trials: int,
    patience: int,
    alpha: float = 0.1,
) -> ThreadResult | None:
    """HQR conformal wrapper evaluation on one month (Tier 3).

    Workflow:
      1. Build control-feature MonthData (44 features).
      2. Train fresh control (with model retained) — reuses the gauntlet's
         baseline trainer so this is apples-to-apples with the main rows.
      3. Predict on val window (calibration fold).
      4. Fit HQR on (X_val, y_val, y_hat_val). Splits val 50/50 for
         sigma-train / conformal-quantile.
      5. Predict HQR spread + PI on test window.
      6. Score: Spearman(spread, |err|), coverage, CRPS vs Gaussian baseline.

    Reports results with method='hqr_wrapper'. Distinct from the point-forecast
    rows — downstream summary analysis filters on method.
    """
    from vol_model.conformal.hqr import (
        fit_hqr, predict_hqr, evaluate_hqr, crps_gaussian_baseline,
    )
    from vol_model.signal_research import train_pruned_svi_baseline_with_model

    t0 = time.time()
    month = build_month_data(ctx, month_str)
    if month is None:
        return None

    preds_test, n_used, model = train_pruned_svi_baseline_with_model(
        month, n_trials=n_trials, patience=patience,
    )
    # Val predictions for calibration
    X_val = month.X_val_w
    y_val = month.y_val
    preds_val = model.predict(X_val.values)

    try:
        fit = fit_hqr(X_val, y_val, preds_val, alpha=alpha)
    except ValueError as exc:
        logger.warning("  [hqr_wrapper] fit_hqr failed: %s", exc)
        return None

    X_test = month.X_test_w
    y_test = month.y_test.to_numpy()
    hqr_out = predict_hqr(X_test, preds_test, fit)
    metrics = evaluate_hqr(y_test, preds_test, hqr_out, alpha=alpha)

    # CRPS baseline: Gaussian with val-window RMSE as σ (homoscedastic)
    val_rmse = float(np.sqrt(np.mean((y_val.to_numpy() - preds_val) ** 2)))
    crps_base = crps_gaussian_baseline(y_test, preds_test, sigma_const=val_rmse)
    crps_hqr = metrics["crps_approx"]
    crps_delta_pct = (crps_hqr / crps_base - 1) * 100 if crps_base > 0 else float("nan")

    # Package as a ThreadResult row (use evaluate_candidate for consistency)
    result = evaluate_candidate(
        preds_test, month,
        method="hqr_wrapper",
        time_s=time.time() - t0,
        n_trials=n_used,
        extra={
            "hqr_spearman_spread_err": metrics["spearman_spread_err"],
            "hqr_coverage": metrics["coverage"],
            "hqr_mean_spread": metrics["mean_spread"],
            "hqr_mean_abs_err": metrics["mean_abs_err"],
            "hqr_crps": crps_hqr,
            "hqr_crps_gaussian_baseline": crps_base,
            "hqr_crps_delta_pct": crps_delta_pct,
            "hqr_q_conformal": fit.q_conformal,
            "hqr_n_sigma_train": fit.n_sigma_train,
            "hqr_n_conformal": fit.n_conformal,
            "hqr_alpha": alpha,
        },
    )
    return result


def run_augmented_for_month(
    ctx: SpotCheckContext,
    month_str: str,
    thread: str,
    *,
    n_trials: int,
    patience: int,
    fresh_control_preds: np.ndarray | None = None,
    fresh_control_test_index: pd.DatetimeIndex | None = None,
) -> ThreadResult | None:
    """Tier 1/2 dispatcher: control + thread-specific feature augmentation.

    `thread` must be a key of TIER_FEATURE_MAP other than 'control'.
    Baseline predictions for DM comparison are overridden with the fresh-control
    predictions (computed earlier in the same month's gauntlet row).
    """
    month = build_month_data_augmented(ctx, month_str, thread=thread)
    if month is None:
        logger.warning("  [%s] build_month_data_augmented returned None; skipping", thread)
        return None

    # Override baseline with fresh control (fair DM comparison)
    if fresh_control_preds is not None and fresh_control_test_index is not None:
        test_idx = month.X_month.index[month.test_mask]
        try:
            ctrl_series = pd.Series(fresh_control_preds, index=fresh_control_test_index)
            aligned = ctrl_series.reindex(test_idx).to_numpy()
            month.baseline_preds_test = aligned
        except Exception:  # noqa: BLE001
            logger.exception("  [%s] failed to align fresh-control baseline; falling back to stale", thread)

    t0 = time.time()
    preds, n_used = train_feature_augmented_lgbm(month, n_trials=n_trials, patience=patience)
    n_features = month.X_month.shape[1]
    return evaluate_candidate(
        preds, month,
        method=thread,
        time_s=time.time() - t0,
        n_trials=n_used,
        extra={"n_features": n_features},
    )


def run_quantile_for_month(month: MonthData, n_trials: int, patience: int) -> list[ThreadResult]:
    """Returns 3 rows (q10, q50, q90) plus a composite row with CRPS.

    Optimization: share the QLIKE-Optuna search across the three alpha
    heads. HP optimum is near-identical for well-behaved log-vol targets;
    each alpha head only needs its own final refit with quantile
    objective. This cuts per-month quantile compute by ~3x.
    """
    from vol_model.signal_research import _run_optuna_qlike, _retrain_best_lgbm
    results: list[ThreadResult] = []
    alphas = {"q10": 0.10, "q50": 0.50, "q90": 0.90}
    preds_by_alpha: dict[str, np.ndarray] = {}

    # ONE Optuna search shared across the three alphas
    t_search = time.time()
    study = _run_optuna_qlike(
        month.X_tr_w, month.y_tr, month.X_val_w, month.y_val,
        n_trials=n_trials, patience=patience,
    )
    search_time = time.time() - t_search
    logger.info("  [quantile] shared Optuna search: %d trials (%.0fs)", len(study.trials), search_time)

    for name, alpha in alphas.items():
        t0 = time.time()
        model = _retrain_best_lgbm(
            study, month.X_tr_w, month.y_tr, month.X_val_w, month.y_val,
            alpha=alpha,
        )
        preds = model.predict(month.X_test_w.values)
        preds_by_alpha[name] = preds
        tr = evaluate_candidate(
            preds, month,
            method=f"quantile_{name}",
            time_s=time.time() - t0 + (search_time / 3),  # amortize search across alphas
            n_trials=len(study.trials),
        )
        results.append(tr)

    # Composite row: uncertainty signal quality
    y_true = month.y_test.to_numpy()
    q10, q50, q90 = preds_by_alpha["q10"], preds_by_alpha["q50"], preds_by_alpha["q90"]
    crps_q = crps_from_quantiles(y_true, q10, q50, q90)
    spread_corr = spread_abs_error_correlation(y_true, q10, q50, q90)
    # Approximate sigma from quantile spread: (q90 - q10) / (2 * 1.2816)
    sigma = np.maximum((q90 - q10) / (2 * 1.2816), 1e-6)
    crps_g = crps_gaussian(y_true, q50, sigma)
    monotone = bool(np.all(q10 <= q50) and np.all(q50 <= q90))

    # Use q50 for the composite row's core metrics (what matters for the backtester)
    t0 = time.time()
    tr_composite = evaluate_candidate(
        q50, month,
        method="quantile_composite",
        time_s=0.0,
        n_trials=0,
        extra={
            "crps_from_quantiles": crps_q,
            "crps_gaussian_from_q50_spread": crps_g,
            "spread_abs_err_corr": spread_corr,
            "q10_q50_q90_monotone": monotone,
            "mean_spread": float(np.mean(q90 - q10)),
        },
    )
    results.append(tr_composite)
    return results


# ---------- Orchestrator ----------


def run_gauntlet(
    ctx: SpotCheckContext,
    months: list[str],
    threads: list[str],
    *,
    n_trials: int,
    patience: int,
    scrap_threshold_pct: float = 5.0,
    prior_scraps: set[str] | None = None,
) -> list[ThreadResult]:
    """Run the gauntlet with asymmetric acceptance.

    If a candidate is > scrap_threshold_pct worse than fresh control on ANY
    month, it is SCRAPPED from all subsequent months. This matches the user's
    early-stop rule: don't burn compute on candidates that have catastrophic
    months. A "win" requires consistent no-worse-than-5% performance across
    the regime substrate; one bad month is enough to disqualify.

    Parameters
    ----------
    scrap_threshold_pct
        Delta (candidate vs fresh control, QLIKE) above which the candidate
        is scrapped. Default +5% = any month where candidate is 5% worse
        than control disqualifies the thread.
    prior_scraps
        Candidates already known to fail (e.g., from prior smoke runs).
        Skipped entirely.
    """
    all_results: list[ThreadResult] = []
    scrapped: set[str] = set(prior_scraps or set())
    if scrapped:
        logger.info("Asymmetric acceptance: PRE-SCRAPPED candidates = %s", sorted(scrapped))

    for i, month_str in enumerate(months, 1):
        logger.info("=" * 70)
        logger.info("[%d/%d] Building month data: %s", i, len(months), month_str)
        month = build_month_data(ctx, month_str)
        if month is None:
            logger.warning("Skipping %s (missing artifacts)", month_str)
            continue

        # Harness sanity: compare reported baseline_preds QLIKE to MonthData.
        # NOTE: month.baseline_preds_test comes from the SAVED predictions
        # parquet. Due to a label-pipeline shift since Stage 18, these
        # predictions were fit against slightly different labels than
        # ctx.y_full. The stale QLIKE is ~30% worse than a fresh control
        # on the same labels -- so any candidate's DM comparison against
        # the stale baseline is meaningless. If pruned_svi_control is in
        # the thread list, we OVERRIDE the baseline with the fresh control
        # predictions for all subsequent candidate DM tests.
        from vol_model.model import qlike_loss
        valid = ~np.isnan(month.baseline_preds_test)
        if valid.sum() >= 30:
            base_q = float(qlike_loss(month.y_test.to_numpy()[valid], month.baseline_preds_test[valid]))
            logger.info("  stale baseline QLIKE on test window: %.4f", base_q)

        # ----- Thread: pruned_svi_control (runs FIRST so its predictions
        # become the DM baseline for the candidates) -----
        fresh_control_preds: np.ndarray | None = None
        fresh_control_test_index: pd.DatetimeIndex | None = None
        if "pruned_svi_control" in threads:
            logger.info("  [pruned_svi_control] training (harness sanity + fresh baseline)...")
            try:
                t0 = time.time()
                preds, n_used = train_pruned_svi_baseline(month, n_trials=n_trials, patience=patience)
                fresh_control_preds = preds
                result = evaluate_candidate(
                    preds, month,
                    method="pruned_svi_control",
                    time_s=time.time() - t0,
                    n_trials=n_used,
                )
                logger.info(
                    "  [pruned_svi_control] QLIKE=%.4f  stale-base=%.4f  delta=%+.2f%%  (%.0fs)",
                    result.qlike, result.baseline_qlike, result.delta_pct, result.time_s,
                )
                all_results.append(result)
                # Override month.baseline_preds_test for subsequent threads
                # so DM is candidate-vs-fresh-control.
                month.baseline_preds_test = fresh_control_preds
                fresh_control_test_index = month.X_month.index[month.test_mask]
            except Exception:  # noqa: BLE001
                logger.exception("  [pruned_svi_control] FAILED on %s", month_str)

        if "har_residual" in threads and "har_residual" not in scrapped:
            logger.info("  [har_residual] training...")
            try:
                result = run_har_residual_for_month(month, n_trials, patience)
                logger.info(
                    "  [har_residual] QLIKE=%.4f  base=%.4f  delta=%+.2f%%  DM t=%+.2f (p=%.3g)  (%.0fs)",
                    result.qlike, result.baseline_qlike, result.delta_pct,
                    result.dm_tstat, result.dm_pval, result.time_s,
                )
                all_results.append(result)
                # Asymmetric acceptance: scrap if catastrophically worse
                if np.isfinite(result.delta_pct) and result.delta_pct > scrap_threshold_pct:
                    logger.warning(
                        "  *** SCRAPPED har_residual: delta=+%.2f%% on %s exceeds "
                        "scrap threshold +%.1f%%. Skipping remaining months.",
                        result.delta_pct, month_str, scrap_threshold_pct,
                    )
                    scrapped.add("har_residual")
            except Exception:  # noqa: BLE001
                logger.exception("  [har_residual] FAILED on %s", month_str)
        elif "har_residual" in scrapped and "har_residual" in threads:
            logger.info("  [har_residual] SKIPPED (scrapped on prior month)")

        if "patchtst" in threads and "patchtst" not in scrapped:
            logger.info("  [patchtst] training encoder + extract + LGBM...")
            try:
                result = run_patchtst_for_month(
                    ctx, month_str, n_trials=n_trials, patience=patience,
                    fresh_control_preds=fresh_control_preds,
                    fresh_control_test_index=fresh_control_test_index,
                )
                if result is not None:
                    logger.info(
                        "  [patchtst] QLIKE=%.4f  base=%.4f  delta=%+.2f%%  DM t=%+.2f (p=%.3g)  (%.0fs)",
                        result.qlike, result.baseline_qlike, result.delta_pct,
                        result.dm_tstat, result.dm_pval, result.time_s,
                    )
                    all_results.append(result)
                    if np.isfinite(result.delta_pct) and result.delta_pct > scrap_threshold_pct:
                        logger.warning(
                            "  *** SCRAPPED patchtst: delta=+%.2f%% on %s exceeds "
                            "scrap threshold +%.1f%%. Skipping remaining months.",
                            result.delta_pct, month_str, scrap_threshold_pct,
                        )
                        scrapped.add("patchtst")
            except Exception:  # noqa: BLE001
                logger.exception("  [patchtst] FAILED on %s", month_str)
        elif "patchtst" in scrapped and "patchtst" in threads:
            logger.info("  [patchtst] SKIPPED (scrapped on prior month)")

        # ----- Tier 1/2 augmented threads (eth, rq, eth_and_rq, svi_spot) -----
        # These reuse the fresh-control winsorization/label pipeline and splice
        # in additional feature columns per TIER_FEATURE_MAP. Asymmetric
        # acceptance gates are identical to other candidates.
        tier_threads = [t for t in threads if t in TIER_FEATURE_MAP and t != "control"]
        for tier_thread in tier_threads:
            if tier_thread in scrapped:
                logger.info("  [%s] SKIPPED (scrapped on prior month)", tier_thread)
                continue
            logger.info("  [%s] training (augmented feature set)...", tier_thread)
            try:
                result = run_augmented_for_month(
                    ctx, month_str, tier_thread,
                    n_trials=n_trials, patience=patience,
                    fresh_control_preds=fresh_control_preds,
                    fresh_control_test_index=fresh_control_test_index,
                )
                if result is not None:
                    logger.info(
                        "  [%s] QLIKE=%.4f  base=%.4f  delta=%+.2f%%  DM t=%+.2f (p=%.3g)  (%.0fs)",
                        result.method, result.qlike, result.baseline_qlike, result.delta_pct,
                        result.dm_tstat, result.dm_pval, result.time_s,
                    )
                    all_results.append(result)
                    if np.isfinite(result.delta_pct) and result.delta_pct > scrap_threshold_pct:
                        logger.warning(
                            "  *** SCRAPPED %s: delta=+%.2f%% on %s exceeds "
                            "scrap threshold +%.1f%%. Skipping remaining months.",
                            tier_thread, result.delta_pct, month_str, scrap_threshold_pct,
                        )
                        scrapped.add(tier_thread)
            except Exception:  # noqa: BLE001
                logger.exception("  [%s] FAILED on %s", tier_thread, month_str)

        # ----- HQR conformal wrapper (Tier 3) -----
        # Independent of point-forecast candidates: produces (spread, PI) on
        # fresh-control's test predictions. Scored by Spearman(spread, |err|)
        # and CRPS delta vs homoscedastic Gaussian baseline.
        if "hqr_wrapper" in threads and "hqr_wrapper" not in scrapped:
            logger.info("  [hqr_wrapper] training width estimator + conformal calibration...")
            try:
                result = run_hqr_for_month(ctx, month_str, n_trials=n_trials, patience=patience)
                if result is not None:
                    rho = result.extra.get("hqr_spearman_spread_err", float("nan"))
                    cov = result.extra.get("hqr_coverage", float("nan"))
                    crps_delta = result.extra.get("hqr_crps_delta_pct", float("nan"))
                    logger.info(
                        "  [hqr_wrapper] spread-err Spearman=%.3f  coverage=%.3f  "
                        "CRPS Δ vs Gaussian=%+.2f%%  (%.0fs)",
                        rho, cov, crps_delta, result.time_s,
                    )
                    all_results.append(result)
            except Exception:  # noqa: BLE001
                logger.exception("  [hqr_wrapper] FAILED on %s", month_str)
        elif "hqr_wrapper" in scrapped and "hqr_wrapper" in threads:
            logger.info("  [hqr_wrapper] SKIPPED (scrapped on prior month)")

        if "quantile" in threads and "quantile" not in scrapped:
            logger.info("  [quantile] training 3 alphas...")
            try:
                rs = run_quantile_for_month(month, n_trials, patience)
                for r in rs:
                    logger.info(
                        "  [%s] QLIKE=%.4f  base=%.4f  delta=%+.2f%%  (%.0fs)",
                        r.method, r.qlike, r.baseline_qlike, r.delta_pct, r.time_s,
                    )
                all_results.extend(rs)
                # Gate on quantile_composite (q50) -- that's the prediction
                # that replaces production's point estimate.
                composite = next((r for r in rs if r.method == "quantile_composite"), None)
                if composite and np.isfinite(composite.delta_pct) and composite.delta_pct > scrap_threshold_pct:
                    logger.warning(
                        "  *** SCRAPPED quantile: composite (q50) delta=+%.2f%% on %s "
                        "exceeds scrap threshold +%.1f%%. Skipping remaining months.",
                        composite.delta_pct, month_str, scrap_threshold_pct,
                    )
                    scrapped.add("quantile")
            except Exception:  # noqa: BLE001
                logger.exception("  [quantile] FAILED on %s", month_str)
        elif "quantile" in scrapped and "quantile" in threads:
            logger.info("  [quantile] SKIPPED (scrapped on prior month)")

        # Incremental save: write CSV after each month
        _save_results(all_results)

        # Early exit: if every candidate has been scrapped, stop the gauntlet.
        # Leave pruned_svi_control running alone is wasteful.
        candidate_threads = [t for t in threads if t != "pruned_svi_control"]
        if candidate_threads and all(t in scrapped for t in candidate_threads):
            logger.warning(
                "All candidate threads scrapped (%s). Early-exit the gauntlet after %s.",
                sorted(scrapped), month_str,
            )
            break

    return all_results


def _save_results(results: list[ThreadResult]) -> None:
    if not results:
        return
    rows = [r.as_row() for r in results]
    df = pd.DataFrame(rows)
    RESULTS_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_CSV, index=False)


# ---------- Main ----------


def main() -> None:
    parser = argparse.ArgumentParser(description="Signal-research spot-check runner")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument(
        "--thread",
        default="all",
        help="Comma-separated: har_residual,quantile,pruned_svi_control,all",
    )
    parser.add_argument(
        "--summary",
        default="vol_model/output/models/expanding_pruned_svi_v1/summary.json",
    )
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--probes-only", action="store_true",
                        help="Run leakage probes only; no training.")
    parser.add_argument("--months", type=str, default="",
                        help="Comma-separated YYYY-MM list (overrides regime picker)")
    parser.add_argument("--scrap-threshold-pct", type=float, default=5.0,
                        help="Asymmetric early-stop: candidate is scrapped from "
                             "subsequent months if any month's QLIKE is >= this%% "
                             "worse than fresh control (default 5.0).")
    parser.add_argument("--pre-scrap", type=str, default="",
                        help="Comma-separated candidates to treat as already "
                             "scrapped (skip entirely). Useful when a prior smoke "
                             "run has already disqualified a thread.")
    args = parser.parse_args()

    # Regime substrate
    if args.months:
        months = [m.strip() for m in args.months.split(",") if m.strip()]
    else:
        months = pick_regime_months(args.summary)
    logger.info("Regime substrate (%d months): %s", len(months), ", ".join(months))

    # Thread list.
    # Note: `pruned_svi_control` is included in "all" because (a) the BTC TSRV
    # END-stamp migration on 2026-04-11 silently changed the label target
    # from forward forecast to backward nowcast — fixed 2026-04-17 by adding
    # the -freq shift in labels.py when tsrv_end_stamped=True; and (b) any
    # candidate's DM comparison needs a fresh-control baseline trained on
    # CURRENT (forecast-restored) labels, not the stale saved predictions.
    # Tier 1 (2026-04-17): "eth", "rq", "eth_and_rq" — cross-asset + HARQ
    # features added via TIER_FEATURE_MAP. Tier 2 ("svi_spot") requires SVI
    # parquet regen; deferred if not available.
    if args.thread == "all":
        threads = ["pruned_svi_control", "har_residual", "quantile"]
    elif args.thread == "tier1":
        threads = ["pruned_svi_control", "eth", "rq", "eth_and_rq"]
    elif args.thread == "tier1_and_hqr":
        # This session's default — Tier 1 point-forecast candidates + Tier 3
        # HQR sizing wrapper. Tier 2 (svi_spot) requires SVI parquet regen and
        # is deferred per plan.
        threads = ["pruned_svi_control", "eth", "rq", "eth_and_rq", "hqr_wrapper"]
    elif args.thread == "all_tier123":
        threads = [
            "pruned_svi_control", "eth", "rq", "eth_and_rq", "svi_spot", "hqr_wrapper",
        ]
    else:
        threads = [t.strip() for t in args.thread.split(",") if t.strip()]

    # Auto-enable extras when any Tier 1/2 thread is requested
    needs_extras = any(t in TIER_FEATURE_MAP and t != "control" for t in threads)
    if needs_extras:
        logger.info("Tier 1/2 threads detected — loading context with include_extras=True")

    # Load context (shared across all threads + months)
    ctx = load_spot_check_context(args.asset, include_extras=needs_extras)

    # Leakage probes
    logger.info("=" * 70)
    logger.info("LEAKAGE PROBE BATTERY")
    logger.info("=" * 70)
    probe_results = run_leakage_probes(ctx)
    if not all(probe_results.values()):
        logger.error("LEAKAGE PROBES FAILED: %s", probe_results)
        if not args.probes_only:
            logger.error("Aborting before training to prevent poisoned results.")
            return
    if args.probes_only:
        logger.info("Probes-only run; exiting.")
        return

    # Pre-scrap (from prior smoke evidence)
    pre_scraps = {s.strip() for s in args.pre_scrap.split(",") if s.strip()}

    # Gauntlet
    logger.info("=" * 70)
    logger.info(
        "GAUNTLET: threads=%s, n_trials=%d, patience=%d, scrap-threshold=+%.1f%%, pre-scrap=%s",
        threads, args.n_trials, args.patience, args.scrap_threshold_pct, sorted(pre_scraps) or "none",
    )
    logger.info("=" * 70)
    results = run_gauntlet(
        ctx, months, threads,
        n_trials=args.n_trials, patience=args.patience,
        scrap_threshold_pct=args.scrap_threshold_pct,
        prior_scraps=pre_scraps,
    )

    # Summary
    _save_results(results)
    logger.info("Saved %d result rows to %s", len(results), RESULTS_CSV)

    # Compact per-method / per-month table in the log
    if results:
        df = pd.DataFrame([r.as_row() for r in results])
        for method in sorted(df["method"].unique()):
            sub = df[df["method"] == method]
            logger.info("")
            logger.info("--- %s ---", method)
            for _, row in sub.iterrows():
                logger.info(
                    "  %s: QLIKE=%.4f  base=%.4f  delta=%+.2f%%  DM t=%+.2f (p=%.3g)",
                    row["month"], row["qlike"], row["baseline_qlike"],
                    row["delta_pct"], row["dm_tstat"], row["dm_pval"],
                )
            avg_delta = sub["delta_pct"].mean()
            wins = int((sub["delta_pct"] < 0).sum())
            logger.info("  AVG delta: %+.2f%%, wins: %d / %d", avg_delta, wins, len(sub))


if __name__ == "__main__":
    main()
