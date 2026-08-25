#!/usr/bin/env python
"""Extract predictions from the pruned_svi_har_pd_v1 walk-forward models.

After ``run_walkforward_pruned_svi_har_pd.py`` finishes training 50 monthly
models, this script loads each one, rebuilds its exact test-window feature
matrix (TransVol embeddings + pruned base + svi_atm_1d + 6 HAR-PD features,
winsorized on the train slice), predicts, and concatenates all 50 months of
OOS predictions into a single parquet for the backtester to consume.

The output file is compatible with ``BacktestConfig.predictions_path`` —
same ``y_pred`` column name, same DatetimeIndex — so comparing pruned+SVI+HAR-PD
vs pruned+SVI is just two ``run_backtest`` invocations with different
``--predictions-path`` overrides, or (preferred for research comparison) a QLIKE/DM
comparison via ``vol_model/compare_pruned_svi_har_pd_vs_baseline.py``.

Usage:
    PYTHONPATH=. python vol_model/extract_pruned_svi_har_pd_predictions.py
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from vol_arb.paths import (
    drop_features_path, har_pd_predictions_path, pruned_features_path,
    pruned_svi_model_dir, pruned_svi_predictions_path, transvol_dir,
    CURRENCY_FOR_ASSET,
)
from vol_model.config import VolModelConfig
from vol_model.pipeline import build_dataset
# NOTE: _extract_compact_embeddings (imports torch + the transvol package at
# module top) is imported LAZILY inside the TransVol branch only, so the
# --no-transvol path used by the AWS-spot horizon sweep stays torch-free.
#
# ALL feature-list constants + loaders come from the HAR-PD walk-forward variant
# (run_walkforward_pruned_svi_har_pd), which is torch-free and carries the same
# ADD_SVI_FEATURES / DROP_FEATURES values as the pruned_svi-only variant. We
# deliberately do NOT import from run_walkforward_pruned_svi here — that module
# does `import torch` at top, which would re-introduce the torch dependency on
# the --no-transvol box. The har_pd variant also accepts ``horizon_minutes``
# (Phase-1 horizon-parametrization) so it supports h=5/30/60/240/1440 cleanly.
from vol_model.run_walkforward_pruned_svi_har_pd import (
    ADD_HAR_PD_FEATURES,
    ADD_SVI_FEATURES,
    DROP_FEATURES,
    _load_har_pd,
    _load_svi_atm_1d,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("extract_pruned_svi_har_pd")


def _emit_trace_rows(
    *,
    path: Path,
    X: pd.DataFrame,
    preds: np.ndarray,
    emb_df: pd.DataFrame,
    month_str: str,
    start: pd.Timestamp | None,
    end: pd.Timestamp | None,
) -> int:
    """Cycle-34 (2026-05-13): append per-row feature/tv/y_pred trace lines.

    Schema mirrors the live :class:`deribit_paper.diagnostics.bar_trace.BarTrace`
    JSONL with checkpoints C4 (``feature_vector``), C5 (``transvol_embedding``)
    and C6 (``y_pred_raw``). The remaining checkpoints (C0/C1/C2/C3/C7/C8) are
    populated by ``scripts/build_offline_bar_trace.py`` from the existing
    ``vol_arb/output/BTC_vol_arb_svi.parquet`` and ``svi_surface_params_15min.parquet``
    plus the offline backtester's per-bar materialised df.

    Append-only — caller pre-truncates the file once before the per-month
    loop, then this helper just appends per row.

    Returns the count of rows emitted (post window filter).
    """
    if start is not None or end is not None:
        idx = X.index
        if start is not None:
            idx = idx[idx >= start]
        if end is not None:
            idx = idx[idx <= end]
        if len(idx) == 0:
            return 0
        X_w = X.loc[idx]
        preds_w = preds[X.index.isin(idx)]
        emb_w = emb_df.loc[emb_df.index.isin(idx)] if emb_df is not None else None
    else:
        X_w = X
        preds_w = preds
        emb_w = emb_df

    # In --no-transvol mode there are no tv_* columns, so the embedding block
    # below is naturally skipped (emit-trace is a debug-only opt-in anyway).
    tv_cols = [c for c in X_w.columns if c.startswith("tv_")]

    n_written = 0
    with path.open("a", encoding="utf-8") as f:
        feat_dict_full = X_w.to_dict(orient="index")  # ts → {feat: val}
        for ts, y_raw in zip(X_w.index, preds_w):
            row_feats = feat_dict_full.get(ts, {})
            row: dict[str, object] = {
                "side": "offline",
                "bar_close_ts": pd.Timestamp(ts).isoformat(),
                "month_str": month_str,
                # C4: 77-named-feature vector (post-winsorize).
                "feature_vector": {
                    k: float(v) for k, v in row_feats.items()
                    if v is not None and pd.notna(v)
                },
                # C6: raw LGBM y_pred.
                "y_pred_raw": float(y_raw),
                # C7: aligner month key (β* lookup keyed by month).
                "aligner_month_key": pd.Timestamp(ts).strftime("%Y-%m"),
            }
            # C5: tv embedding (extracted from the same X row to guarantee
            # it's the exact vector LGBM saw, not the un-aligned emb_df).
            tv_vals = [row_feats.get(c) for c in tv_cols]
            if tv_cols and all(v is not None and pd.notna(v) for v in tv_vals):
                row["transvol_embedding"] = [float(v) for v in tv_vals]
                row["transvol_status"] = "computed"
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
            n_written += 1
    return n_written


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract pruned+SVI predictions")
    parser.add_argument("--asset", default="BTCUSDT",
                        help="Training asset (BTCUSDT or ETHUSDT). Resolves "
                             "model dir, SVI features, pruned feature list, "
                             "TransVol checkpoints, and output path per asset.")
    parser.add_argument("--start", default="2022-02")
    parser.add_argument("--end", default="2026-04")
    parser.add_argument(
        "--end-date", type=str, default="",
        help="Optional full ISO timestamp upper bound (e.g. 2026-05-11T12:15). "
             "When set, the loop also processes the partial month containing "
             "this timestamp, capping the test window at end_date and falling "
             "back to the most-recently-available model for any month whose "
             "walk-forward checkpoint hasn't been trained yet. Use for "
             "partial-month parity runs against the live engine.",
    )
    parser.add_argument("--model-dir", type=str, default="",
                        help="Override model dir (else per-asset resolver)")
    parser.add_argument("--output", type=str, default="",
                        help="Override output path (else per-asset resolver)")
    parser.add_argument("--tf", type=int, default=900,
                        help="Timeframe in seconds (default 900 = 15m)")
    parser.add_argument("--tag", type=str, default="",
                        help="Optional model dir suffix tag")
    parser.add_argument("--transvol-dir", type=str, default="",
                        help="Override the TransVol checkpoint dir (else the "
                             "per-asset/per-horizon resolver). Point at the rehab "
                             "checkpoints to extract the clean-rehab tv_* "
                             "predictions without touching production.")
    parser.add_argument("--no-transvol", action="store_true",
                        help="Mirror the walk-forward's --no-transvol training: skip the "
                             "TransVol .pt checkpoint lookup and embedding extraction "
                             "entirely (no torch import). REQUIRED on the AWS-spot box, "
                             "where .pt checkpoints are never staged. The LGBM feature "
                             "manifest produced by a --no-transvol walk-forward carries no "
                             "tv_* columns, so predictions reproduce exactly.")
    parser.add_argument("--predict-excess-logvol", action="store_true",
                        help="Match the excess-logvol training target: build "
                             "the same reparam'd label so test-time QLIKE/RankIC "
                             "are computed on the same target the model saw. "
                             "Auto-applies tag 'excess' when unset, and suffixes "
                             "the output predictions parquet with '_excess'.")
    parser.add_argument(
        "--emit-trace-jsonl", type=str, default="",
        help="Cycle-34 (2026-05-13): when set, append one JSONL line per "
             "predicted row containing the full 77-feature LGBM input "
             "vector (post-winsorize, in feat_cols order), tv_0..tv_15, "
             "and the raw y_pred (pre-β*). Used by "
             "``scripts/build_offline_bar_trace.py`` to construct the "
             "offline feature trace that diff'd against the live engine's "
             "``bar_trace_live.jsonl`` reveals which checkpoint diverges.",
    )
    parser.add_argument(
        "--emit-trace-start", type=str, default="",
        help="Optional ISO timestamp lower bound for --emit-trace-jsonl. "
             "When set, only rows with timestamp >= this are emitted. "
             "Defaults to the entire test window per month.",
    )
    parser.add_argument(
        "--emit-trace-end", type=str, default="",
        help="Optional ISO timestamp upper bound for --emit-trace-jsonl. "
             "When set, only rows with timestamp <= this are emitted.",
    )
    args = parser.parse_args()
    # Auto-tag & output-suffix so excess predictions never clobber defaults.
    if args.predict_excess_logvol and not args.tag and not args.model_dir:
        args.tag = "excess"

    currency = CURRENCY_FOR_ASSET[args.asset]
    # TransVol checkpoint dir must match the active horizon (see
    # foundation-fix invariant: model horizon == feature horizon ==
    # TransVol horizon). Legacy h=15 returns the unsuffixed BTC path.
    hybrid_dir = (
        Path(args.transvol_dir) if args.transvol_dir
        else transvol_dir(args.asset, horizon_minutes=args.tf // 60)
    )

    # Resolve model dir — default mirrors the HAR-PD walk-forward's output dir.
    if args.model_dir:
        model_dir = Path(args.model_dir)
    elif args.tag:
        base = pruned_svi_model_dir(args.asset)
        model_dir = base.with_name(f"{base.name}_har_pd_v1_{args.tag}")
    else:
        base = pruned_svi_model_dir(args.asset)
        model_dir = base.with_name(f"{base.name.replace('_v1', '')}_har_pd_v1")

    tf_min = args.tf // 60
    if args.output:
        output_path = Path(args.output)
    elif args.tf != 900:
        # Horizon-suffixed output via the Phase 1c path helper so this
        # matches what backtester presets + sidecar generators expect.
        # har_pd_predictions_path returns e.g. vol_model/output/
        # BTCUSDT_pruned_svi_har_pd_predictions_h60m.parquet at h=60.
        output_path = har_pd_predictions_path(
            args.asset, tf_min, aligned_bias=False,
        )
    elif args.predict_excess_logvol:
        base_path = pruned_svi_predictions_path(args.asset)
        output_path = base_path.with_name(
            base_path.stem.replace("_pruned_svi", "_pruned_svi_har_pd") + "_excess" + base_path.suffix
        )
    else:
        # h=15 default — legacy unsuffixed name via the path helper.
        output_path = har_pd_predictions_path(
            args.asset, tf_min, aligned_bias=False,
        )

    print("=" * 70)
    print(f"  EXTRACT PRUNED+SVI+HAR-PD PREDICTIONS — {args.asset}")
    print("=" * 70)
    print(f"  Models from:  {model_dir}")
    print(f"  TransVol:     {hybrid_dir}")
    print(f"  Output:       {output_path}")
    print("=" * 70)

    logger.info("Loading base data...")
    vol_config = VolModelConfig(
        asset=args.asset,
        predict_excess_logvol=args.predict_excess_logvol,
    )
    if args.tf != 900:
        vol_config.freq_s = args.tf
    X, y, _ = build_dataset(vol_config)

    # Per-asset SHAP-tail DROP list from vol_model/output/{asset}_drop_features.csv
    # (written by run_shap_analysis.py --drop-bottom-n). Falls back to the
    # BTC-hardcoded DROP_FEATURES constant if the file is missing.
    drop_path = drop_features_path(args.asset)
    if drop_path.exists() and drop_path.stat().st_size > 0:
        drop_set = set(pd.read_csv(drop_path, header=None)[0].tolist())
    elif drop_path.exists():
        drop_set = set()  # empty file = no drops (spot-check found none safe)
    else:
        drop_set = set(DROP_FEATURES)
    pruned = pd.read_csv(pruned_features_path(args.asset), header=None)[0].tolist()
    pruned_cols_v2 = [c for c in pruned if c in X.columns and c not in drop_set]
    X_pruned = X[pruned_cols_v2]

    drop_prefixes = (
        "ofi_", "depth_mid", "oi_change", "oi_zscore", "oi_acceleration",
    )
    tv_cols = [c for c in X.columns if not c.startswith(drop_prefixes)]
    X_tv = X[tv_cols]

    from vol_model.labels import load_labels
    labels = load_labels(vol_config)

    # Load horizon-matched sidecars — at h=5m the loaders MUST point at the
    # h=5m SVI/HAR-PD files (not the h=15 legacy unsuffixed ones), otherwise
    # extraction silently runs at the legacy 15-min cadence and produces
    # ~1/3 the expected h=5 prediction count. Prior bug: 144,955 predictions
    # at h=5 vs the expected ~432K, surfaced 2026-04-26 mid-Phase 2h5.5.
    svi_add = _load_svi_atm_1d(currency, horizon_minutes=tf_min)

    # HAR-PD sidecar with -h_min alignment shift (matches training).
    har_pd_add = _load_har_pd(args.asset, horizon_minutes=tf_min)

    data_start = X.index.min()
    start = pd.Timestamp(args.start + "-01", tz="UTC")
    end = pd.Timestamp(args.end + "-01", tz="UTC")

    # Partial-month parity support (2026-05-11): when --end-date is set,
    # extend the month loop to include the partial month containing it.
    # For any month whose checkpoint is missing (e.g. current in-progress
    # month) we fall back to the most-recent existing checkpoint — same
    # logic the live engine uses (latest walk-forward month), so backtest
    # predictions match live engine predictions on those bars.
    end_date_partial: pd.Timestamp | None = None
    if args.end_date:
        end_date_partial = pd.Timestamp(args.end_date)
        if end_date_partial.tz is None:
            end_date_partial = end_date_partial.tz_localize("UTC")
        # Extend the month grid to cover end_date_partial's month.
        end_month = end_date_partial.replace(
            day=1, hour=0, minute=0, second=0, microsecond=0,
        )
        if end_month > end:
            end = end_month
        logger.info("Partial-month extraction enabled: end_date=%s",
                    end_date_partial)
    months = pd.date_range(start, end, freq="MS")

    # Precompute the chronologically-sorted list of existing model
    # checkpoints — used as the fallback for any month whose own
    # checkpoint is missing (partial-month case for the in-progress
    # month). The fallback is the LATEST checkpoint with date <= M.
    existing_checkpoints = sorted(
        p.stem for p in model_dir.glob("*.txt")
        if (model_dir / f"{p.stem}_features.json").exists()
    )

    def _resolve_checkpoint(M: pd.Timestamp) -> tuple[Path, Path, str] | None:
        """Return (lgb_path, feat_manifest, used_month_str) for month M.

        Prefers M's own checkpoint; falls back to the most recent
        checkpoint strictly before M (for the partial-month current bar).
        """
        own_str = M.strftime("%Y-%m")
        own_lgb = model_dir / f"{own_str}.txt"
        own_feat = model_dir / f"{own_str}_features.json"
        if own_lgb.exists() and own_feat.exists():
            return own_lgb, own_feat, own_str
        # Fallback: most recent checkpoint strictly before M
        prior = [c for c in existing_checkpoints if c < own_str]
        if not prior:
            return None
        fb_str = prior[-1]
        return (
            model_dir / f"{fb_str}.txt",
            model_dir / f"{fb_str}_features.json",
            fb_str,
        )

    all_preds: list[pd.Series] = []
    all_true: list[pd.Series] = []
    n_ok = 0
    n_skip = 0

    # Cycle-34: prepare emit-trace window if requested.
    emit_trace_path = Path(args.emit_trace_jsonl) if args.emit_trace_jsonl else None
    def _utc_ts(s: str) -> pd.Timestamp:
        ts = pd.Timestamp(s)
        return ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")

    emit_trace_start = _utc_ts(args.emit_trace_start) if args.emit_trace_start else None
    emit_trace_end = _utc_ts(args.emit_trace_end) if args.emit_trace_end else None
    if emit_trace_path is not None:
        emit_trace_path.parent.mkdir(parents=True, exist_ok=True)
        # Truncate so re-runs overwrite (the offline-trace builder
        # invokes this script as a fresh pass — append-during-same-pass
        # is preserved because we open in append mode below).
        emit_trace_path.write_text("", encoding="utf-8")
        logger.info(
            "Cycle-34: --emit-trace-jsonl=%s (window=[%s, %s])",
            emit_trace_path, emit_trace_start, emit_trace_end,
        )

    for i, M in enumerate(months):
        month_str = M.strftime("%Y-%m")
        ckpt = _resolve_checkpoint(M)
        if ckpt is None:
            logger.warning("[%d/%d] %s: no checkpoint available (own or fallback), skipping",
                           i + 1, len(months), month_str)
            n_skip += 1
            continue
        lgb_path, feat_manifest, used_month = ckpt
        if used_month != month_str:
            logger.info("[%d/%d] %s: using fallback checkpoint %s",
                        i + 1, len(months), month_str, used_month)

        # TransVol checkpoint — also fall back to the most recent
        # existing one for the partial-month case. Skipped entirely in
        # --no-transvol mode (the AWS-spot box never stages .pt files).
        test_end = M + pd.DateOffset(months=1)
        # Cap test_end at end_date_partial if it's earlier (partial-month case)
        if end_date_partial is not None and test_end > end_date_partial:
            test_end = end_date_partial
        V = max(M - pd.DateOffset(months=4), data_start + pd.DateOffset(months=12))

        emb_df = None
        if not args.no_transvol:
            tv_path = hybrid_dir / f"{month_str}_transvol.pt"
            if not tv_path.exists():
                # Try the fallback used_month's TransVol
                fallback_tv = hybrid_dir / f"{used_month}_transvol.pt"
                if fallback_tv.exists():
                    tv_path = fallback_tv
                    logger.info("[%d/%d] %s: using fallback TransVol %s",
                                i + 1, len(months), month_str, used_month)
                else:
                    logger.warning("[%d/%d] %s: TransVol checkpoint missing", i + 1, len(months), month_str)
                    n_skip += 1
                    continue
            # Lazy import: pulls torch + transvol. Only on the TransVol path.
            from vol_model.run_walkforward_hybrid import _extract_compact_embeddings
            try:
                emb_df = _extract_compact_embeddings(tv_path, X_tv, labels, test_end)
            except Exception:  # noqa: BLE001
                logger.exception("[%d/%d] %s: TransVol extraction failed", i + 1, len(months), month_str)
                n_skip += 1
                continue

        # Build X_month — exactly the same concat as training. In --no-transvol
        # mode emb_df is None and is omitted (the manifest carries no tv_* cols).
        common = (
            X_pruned.index
            .intersection(svi_add.index)
            .intersection(har_pd_add.index)
        )
        if emb_df is not None:
            common = common.intersection(emb_df.index)
        frames = [X_pruned.loc[common], svi_add.loc[common], har_pd_add.loc[common]]
        if emb_df is not None:
            frames.insert(0, emb_df.loc[common])
        X_month = pd.concat(frames, axis=1)
        common_y = X_month.index.intersection(y.index)
        X_month = X_month.loc[common_y]
        y_month = y.loc[common_y]

        # Drop NaN rows in svi_atm_1d or HAR-PD (same as training)
        X_month = X_month.dropna(subset=ADD_SVI_FEATURES + ADD_HAR_PD_FEATURES)
        y_month = y_month.loc[X_month.index]

        # Temporal masks
        train_mask = X_month.index < V
        test_mask = (X_month.index >= M) & (X_month.index < test_end)
        if not test_mask.any():
            logger.warning("[%d/%d] %s: no test rows", i + 1, len(months), month_str)
            n_skip += 1
            continue

        # Winsorize on train (same as training loop) — this is the crucial
        # step to reproduce the exact same feature values the model saw
        X_tr = X_month[train_mask]
        lo, hi = X_tr.quantile(0.01), X_tr.quantile(0.99)
        X_test = X_month[test_mask].clip(lo, hi, axis=1)
        y_test = y_month[test_mask]

        # Load model + feature order
        feat_cols = json.loads(feat_manifest.read_text())
        # Reorder X_test to match the model's training feature order
        missing_cols = [c for c in feat_cols if c not in X_test.columns]
        if missing_cols:
            logger.warning(
                "[%d/%d] %s: %d feature(s) missing from X_test: %s",
                i + 1, len(months), month_str, len(missing_cols), missing_cols[:5],
            )
            for c in missing_cols:
                X_test[c] = 0.0
        X_test_ordered = X_test[feat_cols]

        model = lgb.Booster(model_file=str(lgb_path))
        preds = model.predict(X_test_ordered.values)

        preds_series = pd.Series(preds, index=X_test_ordered.index, name="y_pred")
        true_series = pd.Series(y_test.values, index=X_test_ordered.index, name="y_true")
        all_preds.append(preds_series)
        all_true.append(true_series)
        n_ok += 1
        logger.info(
            "[%d/%d] %s: %d predictions",
            i + 1, len(months), month_str, len(preds_series),
        )

        # Cycle-34: emit per-row feature/tv/y_pred trace if requested.
        if emit_trace_path is not None:
            _emit_trace_rows(
                path=emit_trace_path,
                X=X_test_ordered, preds=preds,
                emb_df=emb_df,
                month_str=month_str,
                start=emit_trace_start, end=emit_trace_end,
            )

    if not all_preds:
        logger.error("No predictions extracted — nothing to save")
        return

    # Concatenate and sort by timestamp
    y_pred_full = pd.concat(all_preds).sort_index()
    y_true_full = pd.concat(all_true).sort_index()

    # Drop any duplicate indices (shouldn't happen but defensive)
    if not y_pred_full.index.is_unique:
        dup_count = y_pred_full.index.duplicated().sum()
        logger.warning("Found %d duplicate indices, keeping first", dup_count)
        y_pred_full = y_pred_full[~y_pred_full.index.duplicated(keep="first")]
        y_true_full = y_true_full[~y_true_full.index.duplicated(keep="first")]

    out_df = pd.DataFrame({
        "y_pred": y_pred_full.values,
        "y_true": y_true_full.values,
    }, index=y_pred_full.index)
    out_df.index.name = "timestamp"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(output_path)

    print()
    print("=" * 70)
    print(f"  EXTRACTED {n_ok}/{len(months)} months ({n_skip} skipped)")
    print(f"  Total predictions: {len(out_df):,}")
    print(f"  Date range: {out_df.index.min()} to {out_df.index.max()}")
    print(f"  y_pred stats: mean={out_df['y_pred'].mean():.4f}, std={out_df['y_pred'].std():.4f}")
    print(f"  y_true stats: mean={out_df['y_true'].mean():.4f}, std={out_df['y_true'].std():.4f}")
    print(f"  Saved to: {output_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()
