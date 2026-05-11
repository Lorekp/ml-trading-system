"""Thin wrapper: V2 backtest with kelly_mult_path=None for the two-pass
partial-window Kelly bootstrap.

Step 1 of the partial-window orchestrator: produces a draft trade log
WITHOUT loading any Kelly sidecar. The draft is then fed back to
``generate_kelly_sidecar.py`` (+ slope-amp generator) so the Kelly
parquet covers the partial window. A second invocation —
``run_partial_backtest.py`` with the new Kelly sidecar — produces the
final trade log.

Usage:
    PYTHONPATH=. python scripts/run_draft_backtest.py \
        --end-dt 2026-05-11T12:15:00 \
        --output vol_arb/output/BTC_v2_trade_log_draft.parquet
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
logger = logging.getLogger("draft_backtest")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--end-dt", required=True,
                        help="ISO timestamp upper bound (inclusive).")
    parser.add_argument("--start-dt", default=None,
                        help="ISO timestamp lower bound (default None = full).")
    parser.add_argument("--output", type=Path, required=True,
                        help="Output parquet path for the draft trade log.")
    parser.add_argument("--predictions-path", type=str, default=None,
                        help="Override aligned predictions parquet.")
    args = parser.parse_args()

    cfg = (BacktestConfig()
           .with_btc_production_filters_v2()
           .with_realistic_costs())
    cfg.kelly_mult_path = None        # draft pass: no Kelly amplification
    cfg.end_dt = args.end_dt
    if args.start_dt:
        cfg.start_dt = args.start_dt
    if args.predictions_path:
        cfg.predictions_path = args.predictions_path

    logger.info("Draft backtest: start_dt=%s, end_dt=%s, predictions=%s",
                cfg.start_dt, cfg.end_dt, cfg.predictions_path)
    metrics, tl = VolArbBacktester(cfg).run_backtest()
    logger.info("Draft NC PnL: $%s, trades: %d",
                f"{float(tl['net_pnl_nc'].sum()):,.0f}",
                int((tl['N_contracts_nc'] > 0).sum()))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tl.to_parquet(args.output)
    logger.info("Wrote: %s (%d rows)", args.output, len(tl))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
