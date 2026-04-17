#!/usr/bin/env python
"""Run vol-arb threshold sweep for all TFs that have trained models.

For each TF in TFS:
  1. Build vol_arb SVI parquet if missing (uses existing TSRV + Deribit trades).
  2. Extract predictions from existing expanding-window models if missing.
  3. Run threshold sweep and print results table.

Output files:
  vol_arb/output/BTC_vol_arb_svi_{tf}m.parquet
  vol_model/output/BTCUSDT_expanding_{tf}m_predictions.parquet
  vol_arb/output/{tf}m_threshold_sweep.csv
  vol_arb/output/backtest_tradelog_{tf}m_best.parquet
"""

from __future__ import annotations

import sys
sys.path.insert(0, ".")

import gc
import logging
import math
import os
from pathlib import Path

# Limit SVI parallel workers to avoid OOM — each fork copies the full dataset
os.environ["SVI_MAX_WORKERS"] = "2"

import lightgbm as lgb
import numpy as np
import pandas as pd

from vol_arb.backtester import BacktestConfig, VolArbBacktester, print_tearsheet
from vol_arb.config import VolArbConfig
from vol_arb.metrics import build_nc_view
from vol_arb.pipeline import build_vol_arb_dataset
from vol_model.config import VolModelConfig
from vol_model.pipeline import build_dataset
from vol_model.run_walkforward import train_walkforward

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("sweep_all_tfs")

# ── Config ────────────────────────────────────────────────────────────

# (tf_minutes, bucket_freq_str) — bucket_freq is the vol_arb bucket granularity
TFS = [
    (5,   "5min"),
    (60,  "60min"),
    (240, "4h"),
    (480, "8h"),
    (720, "12h"),
]

THRESHOLDS = [10, 15, 20, 25, 30, 35, 40]


def _minutes_to_bucket_label(tf_min: int) -> str:
    """60 -> '60min', 240 -> '4h', 15 -> '15min', etc."""
    if tf_min >= 60 and tf_min % 60 == 0:
        h = tf_min // 60
        return f"{h}h"
    return f"{tf_min}min"


# ── Step 1: build vol_arb SVI parquet ────────────────────────────────

def build_vol_arb_if_missing(tf_min: int, bucket_freq: str) -> str:
    out_path = f"vol_arb/output/BTC_vol_arb_svi_{tf_min}m.parquet"
    if Path(out_path).exists():
        logger.info("[%dm] vol_arb parquet exists, skipping build", tf_min)
        return out_path

    logger.info("[%dm] Building vol_arb SVI parquet -> %s", tf_min, out_path)
    tsrv_path = f"TSRV/output/BTCUSDT_tsrv_{tf_min}m.parquet"
    if not Path(tsrv_path).exists():
        raise FileNotFoundError(f"TSRV not found: {tsrv_path}")

    config = VolArbConfig(
        currency="BTC",
        option_trade_dir="data/deribit/trades/BTC_option",
        dvol_path="data/deribit/dvol/BTC_dvol_60m.parquet",
        tsrv_path=tsrv_path,
        bucket_freq=bucket_freq,
        moneyness_lo=0.95,
        moneyness_hi=1.05,
        dte_min_days=1,
        dte_max_days=14,
        svi_enabled=True,
        svi_bucket_freq="1h",
    )
    df = build_vol_arb_dataset(config)
    df.to_parquet(out_path)
    logger.info("[%dm] Saved %d rows -> %s", tf_min, len(df), out_path)
    del df
    gc.collect()
    return out_path


# ── Step 2: extract predictions from existing models ─────────────────

def extract_predictions_if_missing(tf_min: int) -> str:
    out_path = f"vol_model/output/BTCUSDT_expanding_{tf_min}m_predictions.parquet"
    if Path(out_path).exists():
        logger.info("[%dm] predictions parquet exists, skipping extraction", tf_min)
        return out_path

    freq_s = tf_min * 60
    model_dir = Path(f"vol_model/output/models/expanding_{tf_min}m")
    if not model_dir.exists() or not list(model_dir.glob("*.txt")):
        raise FileNotFoundError(
            f"No trained models found at {model_dir}. "
            f"Run vol_model/run_walkforward.py --tf {freq_s} --lookbacks exp --no-optuna first."
        )

    logger.info("[%dm] Extracting predictions from %s", tf_min, model_dir)
    config = VolModelConfig(asset="BTCUSDT", freq_s=freq_s)
    X, y, _ = build_dataset(config)

    # Determine test window (same logic as run_walkforward.py)
    max_lb = 12
    test_start = (X.index.min() + pd.DateOffset(months=max_lb)).replace(day=1)

    oof, _ = train_walkforward(
        X, y,
        lookback_months=-1,   # expanding window
        max_trials=1,
        patience=1,
        config=config,
        test_start=test_start,
        use_optuna=False,
        gpu_workers=0,
        save_models=False,    # don't overwrite; models already saved
        model_dir=model_dir,
    )

    oof_df = pd.DataFrame({"y_pred": oof, "y_true": y.values}, index=y.index)
    oof_df = oof_df[oof_df.index >= test_start]
    oof_df.to_parquet(out_path)
    logger.info("[%dm] Saved %d rows -> %s", tf_min, len(oof_df), out_path)

    # Free feature matrix + labels (can be large, especially for 5m)
    del X, y, oof, oof_df
    gc.collect()

    return out_path


# ── Step 3: threshold sweep ───────────────────────────────────────────

def run_sweep(tf_min: int, vol_arb_path: str, predictions_path: str) -> list[dict]:
    results = []
    best_nc_sharpe = -np.inf
    best_tl = best_metrics = best_cfg = None

    print(f"\n{'='*90}")
    print(f"  {tf_min}m THRESHOLD SWEEP")
    print(f"{'='*90}")
    print(f"  vol_arb:     {vol_arb_path}")
    print(f"  predictions: {predictions_path}")
    print(f"{'='*90}")
    print(f"  {'thresh':>8}  {'trades':>7}  {'nc_sh':>7}  {'nc_pnl':>14}  {'nc_dd':>7}  {'wr':>6}  {'pf':>6}")
    print(f"  {'-'*75}")

    for thresh in THRESHOLDS:
        cfg = BacktestConfig.for_timeframe(
            bar_minutes=tf_min,
            vol_arb_path=vol_arb_path,
            predictions_path=predictions_path,
            initial_equity=200_000.0,
            slippage_model="realistic_taker",
            sizing_model="composite_v2",
            ioc_cap_multiplier=2.0,
            use_profit_filter=True,
            profit_filter_min_ratio=0.0,
            execution_delay_s=1.0,
            entry_threshold=float(thresh),
            entry_threshold_long=float(thresh + 5),
        )
        bt = VolArbBacktester(cfg)
        metrics, tl = bt.run_backtest()
        del bt  # release backtester's internal DataFrame

        active = tl[(tl["signal"] != 0) & (tl["N_contracts"] > 0)]
        n_trades = len(active)
        del active

        has_nc = "net_pnl_nc" in tl.columns
        if has_nc:
            nc_view = build_nc_view(tl)
            nc_daily = nc_view["net_pnl"].resample("D").sum()
            nc_daily = nc_daily[nc_daily != 0]
            nc_sharpe = float(
                (nc_daily.mean() / nc_daily.std()) * np.sqrt(365)
            ) if len(nc_daily) > 1 and nc_daily.std() > 0 else 0.0
            nc_pnl = nc_view["net_pnl"].sum()
            nc_equity = cfg.initial_equity + nc_view["net_pnl"].cumsum()
            nc_peak = nc_equity.cummax()
            nc_dd = ((nc_equity - nc_peak) / nc_peak * 100).min()
            del nc_view, nc_daily, nc_equity, nc_peak
        else:
            nc_sharpe = nc_pnl = nc_dd = 0.0

        row = {
            "tf_min": tf_min,
            "threshold": thresh,
            "threshold_long": thresh + 5,
            "trades": n_trades,
            "nc_sharpe": nc_sharpe,
            "nc_pnl": nc_pnl,
            "nc_dd": nc_dd,
            "win_rate": metrics.get("win_rate", 0.0),
            "profit_factor": metrics.get("profit_factor", 0.0),
        }
        results.append(row)

        if nc_sharpe > best_nc_sharpe:
            # Release previous best trade log before replacing
            if best_tl is not None:
                del best_tl
            best_nc_sharpe = nc_sharpe
            best_tl = tl
            best_thresh = thresh
            best_metrics = metrics
            best_cfg = cfg
        else:
            del tl  # not the best — release immediately

        del metrics
        gc.collect()

        print(f"  {thresh:>2d}/{thresh+5:>2d}  {n_trades:>7,d}  {nc_sharpe:>7.2f}  "
              f"${nc_pnl:>12,.0f}  {nc_dd:>6.2f}%  {row['win_rate']:>5.1f}%  {row['profit_factor']:>5.2f}")

    print(f"\n  BEST: threshold={best_thresh}/{best_thresh+5} -> NC Sharpe {best_nc_sharpe:.2f}")

    # Save outputs
    df = pd.DataFrame(results)
    df.to_csv(f"vol_arb/output/{tf_min}m_threshold_sweep.csv", index=False)

    if best_tl is not None:
        best_tl.to_parquet(f"vol_arb/output/backtest_tradelog_{tf_min}m_best.parquet")
        print(f"\n  Tearsheet (best threshold={best_thresh}/{best_thresh+5}):")
        print_tearsheet(best_metrics, best_cfg)
        del best_tl, best_metrics, best_cfg
        gc.collect()

    return results


# ── Main ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    all_results = []

    for tf_min, bucket_freq in TFS:
        print(f"\n{'#'*90}")
        print(f"  TF = {tf_min}m")
        print(f"{'#'*90}")
        try:
            vol_arb_path = build_vol_arb_if_missing(tf_min, bucket_freq)
            pred_path = extract_predictions_if_missing(tf_min)
            sweep = run_sweep(tf_min, vol_arb_path, pred_path)
            all_results.extend(sweep)
        except Exception as e:
            logger.error("[%dm] FAILED: %s", tf_min, e)
            import traceback; traceback.print_exc()
        finally:
            gc.collect()

    # ── Summary table ────────────────────────────────────────────────
    if all_results:
        summary = pd.DataFrame(all_results)
        best = summary.loc[summary.groupby("tf_min")["nc_sharpe"].idxmax()]
        print(f"\n\n{'='*90}")
        print("  CROSS-TF SUMMARY (best threshold per TF)")
        print(f"{'='*90}")
        print(f"  {'TF':>6}  {'thresh':>8}  {'trades':>7}  {'nc_sh':>7}  {'nc_pnl':>14}  "
              f"{'nc_dd':>7}  {'wr':>6}  {'pf':>6}")
        print(f"  {'-'*80}")
        for _, row in best.iterrows():
            print(f"  {int(row['tf_min']):>4}m  "
                  f"{int(row['threshold']):>2d}/{int(row['threshold_long']):>2d}  "
                  f"{int(row['trades']):>7,d}  {row['nc_sharpe']:>7.2f}  "
                  f"${row['nc_pnl']:>12,.0f}  {row['nc_dd']:>6.2f}%  "
                  f"{row['win_rate']:>5.1f}%  {row['profit_factor']:>5.2f}")

        summary.to_csv("vol_arb/output/all_tfs_threshold_sweep.csv", index=False)
        print(f"\n  Full table saved to vol_arb/output/all_tfs_threshold_sweep.csv")
