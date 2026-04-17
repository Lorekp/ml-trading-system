#!/usr/bin/env python
"""Conservative spot-check pruning via SHAP + ablation retrains.

Methodology (user-directed):
  1. Pick N evenly-spaced months that have TransVol checkpoints.
  2. Phase A — train full-35 pruned+SVI model on each month with
     Optuna 200/50 (BTC benchmark). Record baseline QLIKE + RankIC +
     per-month SHAP + best LGBM hyperparameters.
  3. Phase B — feature-ablation loop: sort features by aggregated
     mean |SHAP| ascending. For each candidate:
       a. Drop all zero-SHAP features for free (they contribute nothing).
       b. For non-zero candidates, re-train each spot month WITH the
          reduced feature set + the month's pre-optimised HPs (no new
          Optuna — just one final LGBM retrain per month).
       c. Accept the drop iff every spot month stays within
          ``--delta-tolerance`` on BOTH QLIKE (lower-is-better) and
          RankIC (higher-is-better) vs baseline.
       d. Stop as soon as any month violates -> final drop list frozen.
  4. Write the drop list and a markdown summary table.

This is the professional "don't prune what the model actually uses"
test — no forced bottom-N cut, only removes features the model can do
without in every regime in the spot sample.

Usage:
    PYTHONPATH=. python vol_model/run_spot_check_ablation.py \\
        --asset ETHUSDT --n-months 6 --max-trials 200 --patience 50
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import shap as _shap
import torch
from scipy.stats import spearmanr

from vol_arb.paths import (
    drop_features_path, pruned_features_path, pruned_svi_model_dir,
    svi_features_path, transvol_dir, CURRENCY_FOR_ASSET,
)
from vol_model.config import VolModelConfig
from vol_model.model import qlike_loss, qlike_metric, qlike_objective
from vol_model.pipeline import build_dataset
from vol_model.run_walkforward import OptunaPatience, _make_objective
from vol_model.run_walkforward_hybrid import _extract_compact_embeddings
from vol_model.run_walkforward_pruned_svi import _load_svi_atm_1d, ADD_SVI_FEATURES

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("spot_ablation")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _build_month_matrices(
    M: pd.Timestamp,
    X_pruned: pd.DataFrame,
    svi_add: pd.DataFrame,
    labels: pd.DataFrame,
    y: pd.Series,
    X_tv: pd.DataFrame,
    tv_path: Path,
    data_start: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.Series, np.ndarray, np.ndarray, np.ndarray]:
    """Assemble the pruned+SVI+TransVol feature matrix for a given month M.

    Returns (X_month, y_month, train_mask, val_mask, test_mask) winsorised
    on train. Exactly mirrors run_walkforward_pruned_svi.py's assembly so
    metrics are directly comparable.
    """
    test_end = M + pd.DateOffset(months=1)
    V = max(M - pd.DateOffset(months=4), data_start + pd.DateOffset(months=12))

    emb_df = _extract_compact_embeddings(tv_path, X_tv, labels, test_end)
    common = X_pruned.index.intersection(emb_df.index).intersection(svi_add.index)
    X_month = pd.concat([emb_df.loc[common], X_pruned.loc[common], svi_add.loc[common]], axis=1)
    common_y = X_month.index.intersection(y.index)
    X_month = X_month.loc[common_y]
    y_month = y.loc[common_y]
    X_month = X_month.dropna(subset=ADD_SVI_FEATURES)
    y_month = y_month.loc[X_month.index]

    train_mask = np.asarray(X_month.index < V)
    val_mask = np.asarray((X_month.index >= V) & (X_month.index < M))
    test_mask = np.asarray((X_month.index >= M) & (X_month.index < test_end))
    return X_month, y_month, train_mask, val_mask, test_mask


def _winsorise_and_split(X_month, y_month, train_mask, val_mask, test_mask):
    X_tr = X_month[train_mask]
    lo, hi = X_tr.quantile(0.01), X_tr.quantile(0.99)
    X_w = X_month.clip(lo, hi, axis=1)
    return (X_w[train_mask], X_w[val_mask], X_w[test_mask],
            y_month[train_mask], y_month[val_mask], y_month[test_mask])


def _train_and_eval(
    X_train, y_train, X_val, y_val, X_test, y_test,
    params: dict, compute_shap: bool,
) -> dict:
    """Train LGBM with given params + early stopping, return metrics (+ SHAP)."""
    dtrain = lgb.Dataset(X_train.values, label=y_train.values, free_raw_data=False)
    dval = lgb.Dataset(X_val.values, label=y_val.values, reference=dtrain, free_raw_data=False)
    model = lgb.train(
        params, dtrain, num_boost_round=2000,
        valid_sets=[dval], feval=qlike_metric,
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(0)],
    )
    preds = model.predict(X_test.values)
    result = {
        "qlike": qlike_loss(y_test.values, preds),
        "rank_ic": float(spearmanr(y_test.values, preds).statistic),
        "n_test": len(y_test),
        "best_iter": int(model.best_iteration or model.num_trees()),
    }
    if compute_shap:
        try:
            explainer = _shap.TreeExplainer(model)
            shap_vals = explainer.shap_values(X_test.values)
            result["shap_mean_abs"] = np.abs(shap_vals).mean(axis=0)
            result["feature_order"] = list(X_test.columns)
        except Exception as e:  # noqa: BLE001
            logger.warning("SHAP failed: %s", e)
            result["shap_mean_abs"] = None
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Spot-check ablation pruning")
    parser.add_argument("--asset", default="BTCUSDT")
    parser.add_argument("--n-months", type=int, default=6,
                        help="Number of evenly-spaced spot months (default 6)")
    parser.add_argument("--max-trials", type=int, default=200,
                        help="Optuna trial budget per spot month (BTC benchmark)")
    parser.add_argument("--patience", type=int, default=50,
                        help="Optuna patience (BTC benchmark)")
    parser.add_argument("--delta-tolerance", type=float, default=0.001,
                        help="Max allowed absolute metric drift (QLIKE up or "
                             "RankIC down) on any spot month for a feature to "
                             "be droppable. Tighter = more conservative.")
    parser.add_argument("--start", default="2022-02")
    parser.add_argument("--end", default="2026-03")
    parser.add_argument("--out-dir", type=str, default="",
                        help="Where to write summary + drop list (default: model dir)")
    args = parser.parse_args()

    currency = CURRENCY_FOR_ASSET[args.asset]
    hybrid_dir = transvol_dir(args.asset)
    base_model_dir = pruned_svi_model_dir(args.asset)
    work_dir = Path(args.out_dir) if args.out_dir else base_model_dir.with_name(
        base_model_dir.name + "_spot",
    )
    work_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print(f"  SPOT-CHECK ABLATION — {args.asset}")
    print(f"  N spot months: {args.n_months}, Optuna {args.max_trials}/{args.patience}")
    print(f"  Delta tolerance: {args.delta_tolerance} (QLIKE up / RankIC down)")
    print(f"  Work dir: {work_dir}")
    print("=" * 72)

    # ── Load base data ───────────────────────────────────────────
    vol_config = VolModelConfig(asset=args.asset)
    X, y, _ = build_dataset(vol_config)

    pruned = pd.read_csv(pruned_features_path(args.asset), header=None)[0].tolist()
    pruned_cols = [c for c in pruned if c in X.columns]
    X_pruned = X[pruned_cols]

    drop_prefixes = ("ofi_", "depth_mid", "oi_change", "oi_zscore", "oi_acceleration")
    tv_cols = [c for c in X.columns if not c.startswith(drop_prefixes)]
    X_tv = X[tv_cols]

    from vol_model.labels import load_labels
    labels = load_labels(vol_config)
    svi_add = _load_svi_atm_1d(currency)

    data_start = X.index.min()
    start = pd.Timestamp(args.start + "-01", tz="UTC")
    end = pd.Timestamp(args.end + "-01", tz="UTC")
    all_months = pd.date_range(start, end, freq="MS")

    # Keep only months with TransVol checkpoints
    available = [M for M in all_months if (hybrid_dir / f"{M.strftime('%Y-%m')}_transvol.pt").exists()]
    if len(available) < args.n_months:
        raise RuntimeError(
            f"Only {len(available)} months have TransVol checkpoints in {hybrid_dir}; "
            f"need {args.n_months}. Let TransVol finish first.",
        )
    idx = np.linspace(0, len(available) - 1, args.n_months, dtype=int)
    spot_months = [available[i] for i in idx]
    logger.info("Spot months: %s", [M.strftime('%Y-%m') for M in spot_months])

    # ── Phase A — full-35 baseline per spot month ─────────────────
    baselines: dict[str, dict] = {}
    per_month_shap: dict[str, np.ndarray] = {}
    per_month_feature_order: dict[str, list[str]] = {}
    per_month_best_hps: dict[str, dict] = {}

    for M in spot_months:
        month_str = M.strftime("%Y-%m")
        tv_path = hybrid_dir / f"{month_str}_transvol.pt"
        print(f"\n[{month_str}] Phase A: full-35 train + SHAP...")

        X_month, y_month, tm, vm, em = _build_month_matrices(
            M, X_pruned, svi_add, labels, y, X_tv, tv_path, data_start,
        )
        X_tr, X_val, X_te, y_tr, y_val, y_te = _winsorise_and_split(
            X_month, y_month, tm, vm, em,
        )

        # Optuna HPO
        t0 = time.time()
        dtrain = lgb.Dataset(X_tr.values, label=y_tr.values, free_raw_data=False)
        dval_ds = lgb.Dataset(X_val.values, label=y_val.values, reference=dtrain, free_raw_data=False)
        study = optuna.create_study(direction="minimize")
        obj = _make_objective(dtrain, dval_ds, X_val.values, y_val.values, device="cpu")
        study.optimize(obj, n_trials=args.max_trials,
                       callbacks=[OptunaPatience(args.patience)], show_progress_bar=False)
        best_hps = study.best_params
        best_hps.update({
            "objective": qlike_objective, "metric": "custom",
            "device": "cpu", "num_threads": -1, "verbosity": -1, "seed": 42,
        })
        per_month_best_hps[month_str] = {k: v for k, v in best_hps.items() if k not in ("objective", "metric")}

        # Final train with best HPs on train+val, SHAP on test
        result = _train_and_eval(
            pd.concat([X_tr, X_val]), pd.concat([y_tr, y_val]),
            X_val, y_val,  # val reused for early stopping (small harmless leak — same as baseline run)
            X_te, y_te, best_hps, compute_shap=True,
        )
        baselines[month_str] = {"qlike": result["qlike"], "rank_ic": result["rank_ic"]}
        per_month_shap[month_str] = result["shap_mean_abs"]
        per_month_feature_order[month_str] = result["feature_order"]

        logger.info("[%s] baseline QLIKE=%.4f RankIC=%.4f (Optuna %.0fs, %d trials)",
                    month_str, result["qlike"], result["rank_ic"],
                    time.time() - t0, len(study.trials))

    # Persist baseline + HPs for resume / audit
    (work_dir / "baselines.json").write_text(json.dumps(baselines, indent=2))
    (work_dir / "best_hps.json").write_text(json.dumps(per_month_best_hps, indent=2))

    # ── Aggregate SHAP ────────────────────────────────────────────
    feat_order = per_month_feature_order[spot_months[0].strftime("%Y-%m")]
    # Only the pruned-35 feature subset is a drop candidate; TransVol
    # embeddings and SVI are kept regardless (they're not in the pruned list).
    pruned_in_features = [c for c in feat_order if c in set(pruned_cols)]
    shap_matrix = []
    for M in spot_months:
        vals = per_month_shap[M.strftime("%Y-%m")]
        feat_to_val = dict(zip(feat_order, vals))
        shap_matrix.append([feat_to_val.get(f, 0.0) for f in pruned_in_features])
    shap_matrix = np.array(shap_matrix)  # (N months, K features)

    mean_abs = shap_matrix.mean(axis=0)
    max_abs = shap_matrix.max(axis=0)
    feat_shap = pd.DataFrame({
        "feature": pruned_in_features,
        "mean_abs_shap": mean_abs,
        "max_abs_shap": max_abs,
        "zero_on_all": (shap_matrix == 0).all(axis=0),
    }).sort_values("mean_abs_shap", ascending=True).reset_index(drop=True)
    feat_shap.to_csv(work_dir / "spot_shap_aggregate.csv", index=False)
    logger.info("Saved aggregate SHAP over %d spot months to %s",
                len(spot_months), work_dir / "spot_shap_aggregate.csv")

    # ── Phase B — ablation loop ──────────────────────────────────
    zero_drops = feat_shap.loc[feat_shap["zero_on_all"], "feature"].tolist()
    logger.info("Zero-SHAP (free-drop) features: %d — %s", len(zero_drops), zero_drops)

    accepted_drops: list[str] = list(zero_drops)
    candidates = [f for f in feat_shap["feature"].tolist() if f not in set(zero_drops)]

    ablation_rows = []
    for cand in candidates:
        trial_drop = accepted_drops + [cand]
        logger.info("[ablation] trying drop of '%s' (total dropped = %d)", cand, len(trial_drop))
        per_month_deltas = []
        violated_any = False
        for M in spot_months:
            month_str = M.strftime("%Y-%m")
            tv_path = hybrid_dir / f"{month_str}_transvol.pt"
            X_month, y_month, tm, vm, em = _build_month_matrices(
                M, X_pruned, svi_add, labels, y, X_tv, tv_path, data_start,
            )
            keep_cols = [c for c in X_month.columns if c not in set(trial_drop)]
            X_month = X_month[keep_cols]
            X_tr, X_val, X_te, y_tr, y_val, y_te = _winsorise_and_split(
                X_month, y_month, tm, vm, em,
            )
            result = _train_and_eval(
                pd.concat([X_tr, X_val]), pd.concat([y_tr, y_val]),
                X_val, y_val, X_te, y_te,
                per_month_best_hps[month_str] | {
                    "objective": qlike_objective, "metric": "custom",
                    "device": "cpu", "num_threads": -1, "verbosity": -1, "seed": 42,
                },
                compute_shap=False,
            )
            b = baselines[month_str]
            dq = result["qlike"] - b["qlike"]  # positive = worse
            dr = b["rank_ic"] - result["rank_ic"]  # positive = worse
            per_month_deltas.append({"month": month_str, "dq": dq, "dr": dr,
                                     "qlike": result["qlike"], "rank_ic": result["rank_ic"]})
            if dq > args.delta_tolerance or dr > args.delta_tolerance:
                violated_any = True
                logger.info(
                    "  %s: Δ QLIKE=%+.4f Δ RankIC=%+.4f — REJECT (violates %s)",
                    month_str, dq, dr, args.delta_tolerance,
                )
                break
            logger.info("  %s: Δ QLIKE=%+.4f Δ RankIC=%+.4f — within tolerance",
                        month_str, dq, dr)

        ablation_rows.append({
            "candidate": cand, "accepted": not violated_any,
            "deltas": per_month_deltas,
            "accepted_after": list(accepted_drops) + ([cand] if not violated_any else []),
        })

        if not violated_any:
            accepted_drops.append(cand)
            logger.info("[ablation] ACCEPTED — drop list now %d features", len(accepted_drops))
        else:
            logger.info("[ablation] STOP — cannot remove '%s' without model degradation", cand)
            break

    # ── Save final drop list + markdown summary ───────────────────
    drop_path = drop_features_path(args.asset)
    drop_path.parent.mkdir(parents=True, exist_ok=True)
    with open(drop_path, "w", newline="") as f:
        for feat in accepted_drops:
            f.write(f"{feat}\n")
    logger.info("Wrote %d accepted drops to %s", len(accepted_drops), drop_path)

    (work_dir / "ablation_log.json").write_text(json.dumps(ablation_rows, indent=2, default=str))

    # Markdown summary
    md = work_dir / "spot_check_summary.md"
    with open(md, "w") as f:
        f.write(f"# {args.asset} — Spot-Check Ablation Pruning\n\n")
        f.write(f"- Spot months (N={len(spot_months)}): {[M.strftime('%Y-%m') for M in spot_months]}\n")
        f.write(f"- Optuna: {args.max_trials} trials / patience {args.patience}\n")
        f.write(f"- Delta tolerance: ±{args.delta_tolerance} on QLIKE / RankIC\n")
        f.write(f"- Zero-SHAP free drops (all months): {len(zero_drops)} — `{zero_drops}`\n")
        f.write(f"- Accepted ablation drops (non-zero SHAP but safe to remove): "
                f"{len(accepted_drops) - len(zero_drops)}\n")
        f.write(f"- **Total drops**: {len(accepted_drops)} / {len(pruned_in_features)} "
                f"-> {len(pruned_in_features) - len(accepted_drops)} features kept\n\n")
        f.write("## Baseline per month (full-35)\n\n")
        f.write("| Month | QLIKE | RankIC |\n|---|---:|---:|\n")
        for m, b in baselines.items():
            f.write(f"| {m} | {b['qlike']:.4f} | {b['rank_ic']:.4f} |\n")
        f.write(f"\n## Accepted drop list ({len(accepted_drops)} features)\n\n")
        for d in accepted_drops:
            label = "zero-SHAP" if d in zero_drops else "ablation-safe"
            f.write(f"- `{d}` *({label})*\n")
        if len(accepted_drops) < len(candidates) + len(zero_drops):
            f.write(f"\n## Stopped at (first feature that would have hurt)\n\n")
            rejected = next((r for r in ablation_rows if not r["accepted"]), None)
            if rejected:
                f.write(f"- `{rejected['candidate']}` — one or more months exceeded "
                        f"`{args.delta_tolerance}` tolerance.\n")

    print(f"\nSpot-check summary: {md}")
    print(f"Drop list: {drop_path}")


if __name__ == "__main__":
    main()
