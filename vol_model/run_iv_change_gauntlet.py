"""IV-change target gauntlet (Tier 3 orthogonal target, 2026-04-17).

A standalone gauntlet that tests whether a direct predictor of the
iv-change `ln(iv_exit / iv_entry)` beats the production ``edge``-derived
implicit predictor on signal quality.

**Target definition**:
    y_ivchg(t) = ln(iv_exit(t + 15m)) - ln(iv_used(t))

  - iv_used(t): the IV observed at decision time t (from the merged
    vol_arb dataset's iv_used column = svi_iv_matched → iv → dvol fallback).
  - iv_exit(t + 15m): the IV observed at exit time t+15m for the same
    strike + expiry contract (backtester's `iv_exit` column when a trade
    is active; reconstructed from the saved vol_arb data elsewhere).

**Metrics** (different from QLIKE — y_ivchg is signed, small, near zero):
  - Spearman(y_pred_ivchg, actual_ivchg) on active-trade subset.
  - Signed-accuracy: fraction of rows where sign(y_pred) = sign(actual).
  - Baseline to beat: production `edge` column's Spearman with iv_move,
    which is 0.425 on the post-foundation-fix trade log.

**Leakage guards**:
  1. iv_used and iv_exit are both AFTER feature timestamp — iv_used was
     already +15min-shifted in the SVI pipeline, so the features and
     iv_used are from disjoint information sets.
  2. iv_exit is at t+15m, strictly future of feature ts t. The backtester's
     merge_asof(forward, tolerance=5s) guarantees this.
  3. Train/test split is strictly time-ordered per spot_check_regimes.

**Scope**: same 8-month regime substrate as the RV-forecast gauntlet. Uses
the 44-feature `pruned_svi_v1` set + the same TransVol embeddings.

Usage::

    python -m vol_model.run_iv_change_gauntlet [--n-trials 80 --patience 25]

Results: ``vol_model/output/iv_change_gauntlet_results.csv``
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

import lightgbm as lgb

from vol_model.signal_research import (
    load_spot_check_context,
    build_month_data,
)
from vol_model.spot_check_regimes import pick_regime_months


logger = logging.getLogger("iv_change")


TRADE_LOG_PATH = Path("vol_arb/output/BTC_v2_trade_log.parquet")
OUTPUT_CSV = Path("vol_model/output/iv_change_gauntlet_results.csv")


def load_iv_change_targets(feature_index: pd.DatetimeIndex) -> pd.Series:
    """Load y_ivchg = ln(iv_exit/iv_used) aligned to feature timestamps.

    Reads from the freshly-regenerated V2 trade log (hold=15m, post-
    foundation-fix). For each feature timestamp t, picks up iv_used and
    iv_exit from the trade log's row at t. Only valid when iv_used > 0
    and iv_exit > 0 (which is essentially every row — the backtester
    always resolves iv_used via its 3-tier fallback).
    """
    if not TRADE_LOG_PATH.exists():
        raise FileNotFoundError(
            f"Need {TRADE_LOG_PATH} — regenerate via `python -m vol_arb.run_backtest "
            "--historical-btc-v2 --output {path}`"
        )
    tl = pd.read_parquet(TRADE_LOG_PATH)
    aligned = tl.reindex(feature_index)
    iv_used = aligned["iv_used"]
    iv_exit = aligned["iv_exit"]
    valid = (iv_used > 0) & (iv_exit > 0)
    y = pd.Series(np.nan, index=feature_index, dtype=np.float64)
    y[valid] = np.log(iv_exit[valid].values / iv_used[valid].values)
    return y


def _train_lgbm_mse(
    X_tr, y_tr, X_val, y_val, n_trials: int, patience: int, seed: int = 42,
) -> lgb.Booster:
    """Minimal LightGBM fit with MSE loss (the target is signed).

    Keeps Optuna search space narrow — this is a diagnostic thread, not
    a production candidate.
    """
    try:
        import optuna
    except ImportError:
        optuna = None

    fixed_params = dict(
        objective="regression",
        metric="l2",
        learning_rate=0.03,
        num_leaves=31,
        min_data_in_leaf=20,
        feature_fraction=0.9,
        bagging_fraction=0.9,
        bagging_freq=5,
        verbose=-1,
        seed=seed,
    )
    if optuna is None or n_trials <= 1:
        # Single-shot default train
        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_val, label=y_val, reference=dtr)
        model = lgb.train(
            fixed_params, dtr, num_boost_round=1000, valid_sets=[dva],
            callbacks=[lgb.early_stopping(patience, verbose=False)],
        )
        return model

    def objective(trial):
        params = dict(fixed_params)
        params["learning_rate"] = trial.suggest_float("learning_rate", 0.01, 0.1, log=True)
        params["num_leaves"] = trial.suggest_int("num_leaves", 16, 128)
        params["min_data_in_leaf"] = trial.suggest_int("min_data_in_leaf", 10, 100)
        params["feature_fraction"] = trial.suggest_float("feature_fraction", 0.6, 1.0)
        params["bagging_fraction"] = trial.suggest_float("bagging_fraction", 0.6, 1.0)

        dtr = lgb.Dataset(X_tr, label=y_tr)
        dva = lgb.Dataset(X_val, label=y_val, reference=dtr)
        model = lgb.train(
            params, dtr, num_boost_round=500, valid_sets=[dva],
            callbacks=[lgb.early_stopping(25, verbose=False)],
        )
        return model.best_score["valid_0"]["l2"]

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best_params = dict(fixed_params, **study.best_params)
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_val, label=y_val, reference=dtr)
    return lgb.train(
        best_params, dtr, num_boost_round=1000, valid_sets=[dva],
        callbacks=[lgb.early_stopping(patience, verbose=False)],
    )


def run_iv_change_month(
    ctx, month_str: str, y_ivchg: pd.Series,
    *, n_trials: int, patience: int,
) -> dict | None:
    """Run the IV-change thread for a single month, return metrics dict."""
    md = build_month_data(ctx, month_str)
    if md is None:
        return None

    # Subset y_ivchg to the month's feature index; drop NaN targets.
    y_chg = y_ivchg.reindex(md.X_month.index)
    mask_valid = y_chg.notna().values
    if mask_valid.sum() < 1000:
        logger.warning("%s: only %d valid iv_change rows; skipping",
                       month_str, mask_valid.sum())
        return None

    def _as_array(mask):
        return mask.values if hasattr(mask, "values") else np.asarray(mask)
    tr = _as_array(md.train_mask) & mask_valid
    va = _as_array(md.val_mask) & mask_valid
    te = _as_array(md.test_mask) & mask_valid

    X_tr = md.X_month.iloc[tr].clip(md.lo, md.hi, axis=1)
    X_va = md.X_month.iloc[va].clip(md.lo, md.hi, axis=1)
    X_te = md.X_month.iloc[te].clip(md.lo, md.hi, axis=1)
    y_tr = y_chg.iloc[tr].values
    y_va = y_chg.iloc[va].values
    y_te = y_chg.iloc[te].values

    logger.info("  %s: tr=%d val=%d test=%d (iv_change rows)",
                month_str, len(y_tr), len(y_va), len(y_te))

    t0 = time.perf_counter()
    model = _train_lgbm_mse(X_tr.values, y_tr, X_va.values, y_va, n_trials, patience)
    preds = model.predict(X_te.values)
    elapsed = time.perf_counter() - t0

    # Metrics
    mse = float(np.mean((preds - y_te) ** 2))
    rmse = float(np.sqrt(mse))
    spearman = float(stats.spearmanr(preds, y_te).correlation)
    # Signed accuracy: fraction of rows with matching sign
    # Use a small deadzone: if |y_te| < 0.01 (1% IV), skip (ambiguous)
    non_trivial = np.abs(y_te) > 0.01
    if non_trivial.sum() > 0:
        signed_acc = float(
            (np.sign(preds[non_trivial]) == np.sign(y_te[non_trivial])).mean()
        )
    else:
        signed_acc = float("nan")

    # Baseline: production edge on the same test rows
    # (edge column from the trade log is the pre-fix implicit predictor)
    tl = pd.read_parquet(TRADE_LOG_PATH)
    edge_aligned = tl["edge"].reindex(X_te.index).values
    mask_edge = np.isfinite(edge_aligned)
    if mask_edge.sum() > 100:
        baseline_spearman = float(
            stats.spearmanr(edge_aligned[mask_edge], y_te[mask_edge]).correlation
        )
        baseline_signed_acc = float(
            (np.sign(edge_aligned[mask_edge][non_trivial[mask_edge]]) ==
             np.sign(y_te[mask_edge][non_trivial[mask_edge]])).mean()
        ) if non_trivial[mask_edge].sum() > 0 else float("nan")
    else:
        baseline_spearman = float("nan")
        baseline_signed_acc = float("nan")

    return {
        "method": "iv_change",
        "month": month_str,
        "mse": mse,
        "rmse": rmse,
        "spearman": spearman,
        "baseline_spearman": baseline_spearman,
        "spearman_delta": spearman - baseline_spearman,
        "signed_accuracy": signed_acc,
        "baseline_signed_accuracy": baseline_signed_acc,
        "n_train": int(len(y_tr)),
        "n_test": int(len(y_te)),
        "time_s": elapsed,
    }


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", default="BTCUSDT")
    ap.add_argument("--n-trials", type=int, default=80)
    ap.add_argument("--patience", type=int, default=25)
    ap.add_argument("--months", default=None,
                    help="Comma-separated YYYY-MM override (default: regime picker)")
    args = ap.parse_args()

    # Regime months
    if args.months:
        months = [m.strip() for m in args.months.split(",") if m.strip()]
    else:
        months = pick_regime_months("vol_model/output/models/expanding_pruned_svi_v1/summary.json")
    logger.info("Regime substrate (%d months): %s", len(months), ", ".join(months))

    # Load context once
    logger.info("Loading spot-check context...")
    ctx = load_spot_check_context(args.asset, include_extras=False)

    # Build IV-change target on the full feature index
    logger.info("Building iv_change target from %s...", TRADE_LOG_PATH)
    y_ivchg = load_iv_change_targets(ctx.X_full.index)
    logger.info("  target non-null: %d / %d (%.1f%%)",
                y_ivchg.notna().sum(), len(y_ivchg),
                100 * y_ivchg.notna().mean())
    logger.info("  target stats: mean=%+.4f std=%.4f q05=%+.4f q95=%+.4f",
                y_ivchg.mean(), y_ivchg.std(),
                y_ivchg.quantile(0.05), y_ivchg.quantile(0.95))

    # Run per-month
    results = []
    for m in months:
        try:
            r = run_iv_change_month(
                ctx, m, y_ivchg,
                n_trials=args.n_trials, patience=args.patience,
            )
        except Exception:  # noqa: BLE001
            logger.exception("failed on %s", m)
            continue
        if r is None:
            continue
        logger.info(
            "  %s: Spearman=%.4f (baseline=%.4f, Δ=%+.4f) signed_acc=%.3f (base=%.3f) "
            "rmse=%.5f (%.0fs)",
            m, r["spearman"], r["baseline_spearman"], r["spearman_delta"],
            r["signed_accuracy"], r["baseline_signed_accuracy"],
            r["rmse"], r["time_s"],
        )
        results.append(r)

    if not results:
        logger.warning("no result rows produced")
        return 1

    df = pd.DataFrame(results)
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False)
    logger.info("wrote %s (%d rows)", OUTPUT_CSV, len(df))
    logger.info("Avg Spearman: model=%.4f baseline=%.4f  delta=%+.4f  (over %d months)",
                df["spearman"].mean(), df["baseline_spearman"].mean(),
                df["spearman_delta"].mean(), len(df))
    logger.info("Avg signed_accuracy: model=%.3f baseline=%.3f",
                df["signed_accuracy"].mean(), df["baseline_signed_accuracy"].mean())
    wins = (df["spearman_delta"] > 0).sum()
    logger.info("Model beats baseline Spearman on %d / %d months", wins, len(df))
    return 0


if __name__ == "__main__":
    sys.exit(main())
