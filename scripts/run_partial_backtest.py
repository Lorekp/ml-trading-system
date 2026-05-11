"""Thin wrapper: V2 backtest with end_dt slicing + suffixed output path.

Final pass of the partial-window orchestrator. Reads the (partial)
canonical inputs that the upstream pipeline produced + the partial
Kelly sidecar bootstrapped by the draft pass, runs the V2 backtester
sliced to [start_dt, end_dt], and writes the trade log to a suffixed
path so the canonical BTC_v2_trade_log.parquet is never overwritten.

Output naming:
  vol_arb/output/BTC_v2_trade_log_partial_<YYYYMMDDTHHMM>.parquet

Usage:
    PYTHONPATH=. python scripts/run_partial_backtest.py \
        --end-dt 2026-05-11T12:15:00 \
        --kelly-mult-path vol_arb/output/BTC_v2_kelly_mult_K1_slope_partial.parquet
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

from vol_arb.backtester import BacktestConfig, VolArbBacktester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("partial_backtest")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--end-dt", required=True,
                        help="ISO timestamp upper bound (inclusive).")
    parser.add_argument("--start-dt", default=None,
                        help="ISO timestamp lower bound (default None = full).")
    parser.add_argument("--kelly-mult-path", type=str, default=None,
                        help="Override Kelly sidecar. Default keeps V2 preset's "
                             "canonical K1+slope sidecar (which doesn't cover "
                             "partial-month bars beyond the canonical cutoff).")
    parser.add_argument("--predictions-path", type=str, default=None,
                        help="Override aligned predictions parquet.")
    parser.add_argument("--output", type=Path, default=None,
                        help="Override output path. Default uses "
                             "BTC_v2_trade_log_partial_<end_dt>.parquet "
                             "to avoid clobbering the canonical trade log.")
    args = parser.parse_args()

    cfg = (BacktestConfig()
           .with_btc_production_filters_v2()
           .with_realistic_costs())
    cfg.end_dt = args.end_dt
    if args.start_dt:
        cfg.start_dt = args.start_dt
    if args.kelly_mult_path:
        cfg.kelly_mult_path = args.kelly_mult_path
    if args.predictions_path:
        cfg.predictions_path = args.predictions_path

    # Default suffixed output path
    if args.output is None:
        suffix = pd.Timestamp(args.end_dt).strftime("%Y%m%dT%H%M")
        out_path = Path(f"vol_arb/output/BTC_v2_trade_log_partial_{suffix}.parquet")
    else:
        out_path = args.output

    logger.info("Partial backtest:")
    logger.info("  start_dt:    %s", cfg.start_dt)
    logger.info("  end_dt:      %s", cfg.end_dt)
    logger.info("  predictions: %s", cfg.predictions_path)
    logger.info("  kelly_mult:  %s", cfg.kelly_mult_path)
    logger.info("  output:      %s", out_path)

    metrics, tl = VolArbBacktester(cfg).run_backtest()
    logger.info("Partial NC PnL: $%s, trades: %d",
                f"{float(tl['net_pnl_nc'].sum()):,.0f}",
                int((tl['N_contracts_nc'] > 0).sum()))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tl.to_parquet(out_path)
    logger.info("Wrote: %s (%d rows)", out_path, len(tl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
