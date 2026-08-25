#!/usr/bin/env python
"""Cycle-36 (2026-05-13): run the V2 backtester with feature-trace emission.

Purpose-built driver for the live↔offline parity trace. Builds the V2
production config, points it at the trace-window aligned predictions,
sets ``feature_trace_path`` + window bounds, and runs the backtest.

The backtester's ``_emit_feature_trace`` (cycle-34) appends one JSONL
line per bar to the trace path; ``scripts/build_offline_bar_trace.py``
then merges it with the predictions trace + SVI params for the
comparator.

Usage:
    PYTHONPATH=. python scripts/run_backtester_trace.py \
        --predictions vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias_tracewin.parquet \
        --trace-out vol_arb/output/bar_trace_backtester.jsonl \
        --start "2026-05-13 19:00:00+00:00" --end "2026-05-17 17:15:00+00:00"
"""
from __future__ import annotations

import argparse
import logging

from vol_arb.backtester import BacktestConfig, VolArbBacktester

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("run_backtester_trace")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", required=True)
    p.add_argument("--trace-out", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    args = p.parse_args()

    cfg = BacktestConfig().with_historical_btc_v2().with_realistic_costs()
    cfg.predictions_path = args.predictions
    cfg.feature_trace_path = args.trace_out
    cfg.feature_trace_start = args.start
    cfg.feature_trace_end = args.end
    # Window the backtest itself to the trace span so the per-bar df
    # the emitter walks is the parity window (predictions parquet only
    # covers 2026-04+ anyway).
    cfg.start_dt = args.start
    cfg.end_dt = args.end

    logger.info("V2 backtest (trace mode): predictions=%s window=[%s, %s]",
                args.predictions, args.start, args.end)
    bt = VolArbBacktester(cfg)
    metrics, tl = bt.run_backtest()
    logger.info("Backtest done: %d trade-log rows; trace -> %s",
                len(tl), args.trace_out)
    nc = metrics.get("non_compounding", {})
    logger.info("NC PnL=%.0f  trades=%d",
                nc.get("total_pnl", float("nan")),
                int((tl["net_pnl_nc"] != 0).sum()) if "net_pnl_nc" in tl else -1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
