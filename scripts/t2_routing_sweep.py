"""T2.2 routing sweep — SVI-skew-driven instrument routing vs baseline.

Tests whether gating instrument routing on ``svi_25d_skew_1w`` (the SVI
25-delta skew at 1 week) produces a better risk-adjusted outcome than the
current ATM-straddle-only baseline.

**Background**: Stage 8.4 of the pipeline doc shows the existing
``jump_skew``-based routing underperforms ATM straddles. Stage 15.5
proposes that SVI skew is a cleaner routing signal. SVI features proved
redundant as LightGBM inputs (Stage 15.6 spot-check) but the doc
specifically flagged ``svi_25d_skew_1w`` as a good candidate for
*instrument selection*, not prediction.

**The hypothesis**: when SVI prices puts richer than calls (positive
25d skew), the market expects more downside; trading a bear strangle
positions us asymmetrically for that regime. Symmetric for negative
skew → bull strangle. Near zero skew → ATM straddle (the default).

**The sweep**: fix the backtester on ``BTC_vol_arb_svi.parquet`` (which
contains the SVI features) and vary:

1. Baseline: no routing, ATM only
2. Jump-skew routing (legacy reference, threshold 0.05)
3. SVI-skew routing at multiple thresholds: 0 (always route), 3, 5, 7,
   10, 15 vol points

Captures non-comp Sharpe + MaxDD + trade counts + instrument mix for each.

**Decision rule**: keep the routing config if non-comp Sharpe >= baseline
AND non-comp MaxDD <= 1.10 * baseline.

Runs via::

    PYTHONPATH=. python scripts/t2_routing_sweep.py
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from vol_arb.backtester import BacktestConfig, VolArbBacktester

logger = logging.getLogger("t2_routing_sweep")


@dataclass
class SweepRow:
    """One config's summary for the sweep table."""
    label: str
    mode: str
    threshold: float
    n_trades: int
    total_pnl_nc: float
    sharpe_nc: float
    max_dd_nc_pct: float
    sortino_nc: float
    n_bear: int = 0
    n_bull: int = 0
    n_atm: int = 0
    routing_frac: float = 0.0


def _count_instruments(trade_log: pd.DataFrame) -> tuple[int, int, int]:
    """Extract instrument counts from trade log if the column exists."""
    if "instrument_type" not in trade_log.columns:
        return (0, 0, len(trade_log))
    # Only count active trades (signal != 0)
    active = trade_log[trade_log["signal"] != 0]
    n_bear = int((active["instrument_type"] == 1).sum())
    n_bull = int((active["instrument_type"] == -1).sum())
    n_atm = int((active["instrument_type"] == 0).sum())
    return n_bear, n_bull, n_atm


def run_one(
    cfg: BacktestConfig,
    *,
    label: str,
    mode: str,
    threshold: float,
) -> SweepRow:
    """Run the backtester with a specific routing config."""
    bt = VolArbBacktester(cfg)
    metrics, trade_log = bt.run_backtest()
    nc = metrics.get("non_compounding", {})

    n_bear, n_bull, n_atm = _count_instruments(trade_log)
    n_active = n_bear + n_bull + n_atm
    routing_frac = (n_bear + n_bull) / max(n_active, 1)

    return SweepRow(
        label=label,
        mode=mode,
        threshold=threshold,
        n_trades=int(metrics.get("n_trades", 0)),
        total_pnl_nc=float(nc.get("total_pnl", 0.0)),
        sharpe_nc=float(nc.get("sharpe_ratio", 0.0)),  # bar-level Sharpe
        max_dd_nc_pct=float(nc.get("max_drawdown_pct", 0.0)),
        sortino_nc=float(nc.get("sortino_ratio", 0.0)),
        n_bear=n_bear,
        n_bull=n_bull,
        n_atm=n_atm,
        routing_frac=routing_frac,
    )


def _build_config(
    vol_arb_path: str,
    predictions_path: str,
    threshold_entry: float,
    use_routing: bool,
    routing_mode: str,
    svi_skew_threshold: float,
    jump_skew_threshold: float,
) -> BacktestConfig:
    return BacktestConfig(
        vol_arb_path=vol_arb_path,
        predictions_path=predictions_path,
        entry_threshold=threshold_entry,
        use_sor=False,
        use_vrp=False,
        use_instrument_routing=use_routing,
        routing_mode=routing_mode,
        svi_skew_threshold=svi_skew_threshold,
        jump_skew_threshold=jump_skew_threshold,
    )


def _fmt_row(row: SweepRow) -> str:
    return (
        f"  {row.label:<28s}  "
        f"{row.n_trades:>7,d}  "
        f"${row.total_pnl_nc:>12,.0f}  "
        f"{row.sharpe_nc:>7.2f}  "
        f"{row.sortino_nc:>7.2f}  "
        f"{row.max_dd_nc_pct:>7.2f}  "
        f"{row.n_atm:>6d}  "
        f"{row.n_bear:>6d}  "
        f"{row.n_bull:>6d}  "
        f"{row.routing_frac * 100:>5.1f}%"
    )


def _print_table(rows: list[SweepRow], baseline: SweepRow) -> None:
    print()
    print("=" * 120)
    print(
        f"  {'CONFIG':<28s}  "
        f"{'trades':>7s}  "
        f"{'non-comp PnL':>13s}  "
        f"{'Sharpe':>7s}  "
        f"{'Sortino':>7s}  "
        f"{'MaxDD%':>7s}  "
        f"{'ATM':>6s}  "
        f"{'Bear':>6s}  "
        f"{'Bull':>6s}  "
        f"Route%"
    )
    print("-" * 120)
    for row in rows:
        marker = ""
        if row.label != baseline.label:
            dp = row.sharpe_nc - baseline.sharpe_nc
            dd_ratio = abs(row.max_dd_nc_pct) / max(abs(baseline.max_dd_nc_pct), 1e-9)
            passes_sharpe = row.sharpe_nc >= baseline.sharpe_nc
            passes_dd = dd_ratio <= 1.10
            if passes_sharpe and passes_dd:
                marker = "  ✓ KEEP"
            elif passes_sharpe and not passes_dd:
                marker = "  ~ Sharpe ok, DD worse"
            else:
                marker = ""
        print(_fmt_row(row) + marker)
    print("=" * 120)


def main() -> int:
    parser = argparse.ArgumentParser(description="T2.2 routing sweep")
    parser.add_argument(
        "--vol-arb-path", type=str,
        default="vol_arb/output/BTC_vol_arb_svi.parquet",
        help="Use the SVI variant by default so svi_25d_skew_1w is present",
    )
    parser.add_argument(
        "--predictions-path", type=str,
        default="vol_model/output/BTCUSDT_hybrid_predictions.parquet",
    )
    parser.add_argument(
        "--threshold", type=float, default=25.0,
        help="Entry threshold (vol pts), default 25",
    )
    parser.add_argument(
        "--svi-thresholds", type=str,
        default="0.0,3.0,5.0,7.0,10.0,15.0",
        help="Comma-separated SVI skew thresholds (vol pts) to sweep",
    )
    parser.add_argument(
        "--jump-thresholds", type=str,
        default="0.05,0.10",
        help="Comma-separated jump_skew thresholds (unitless) to sweep",
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--output", type=str, default="",
        help="Save sweep table to parquet",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    svi_thresholds = [float(x) for x in args.svi_thresholds.split(",") if x]
    jump_thresholds = [float(x) for x in args.jump_thresholds.split(",") if x]

    print()
    print("=" * 120)
    print("  T2.2 ROUTING SWEEP — SVI skew vs jump skew vs ATM baseline")
    print("=" * 120)
    print(f"  vol_arb_path:        {args.vol_arb_path}")
    print(f"  predictions_path:    {args.predictions_path}")
    print(f"  entry_threshold:     {args.threshold}")
    print("=" * 120)

    rows: list[SweepRow] = []

    # 1. Baseline — no routing
    print("\n[1/?] baseline (no routing)...")
    cfg = _build_config(
        args.vol_arb_path, args.predictions_path, args.threshold,
        use_routing=False, routing_mode="jump_skew",
        svi_skew_threshold=0.0, jump_skew_threshold=0.05,
    )
    baseline = run_one(cfg, label="BASELINE (ATM only)", mode="none", threshold=0.0)
    rows.append(baseline)
    print(
        f"    done: Sharpe={baseline.sharpe_nc:.2f}, "
        f"PnL=${baseline.total_pnl_nc:,.0f}, "
        f"MaxDD={baseline.max_dd_nc_pct:.2f}%",
    )

    # 2. Jump-skew routing (legacy) for reference
    for i, jt in enumerate(jump_thresholds, start=2):
        print(f"\n[{i}/?] jump-skew routing @ {jt}...")
        cfg = _build_config(
            args.vol_arb_path, args.predictions_path, args.threshold,
            use_routing=True, routing_mode="jump_skew",
            svi_skew_threshold=0.0, jump_skew_threshold=jt,
        )
        row = run_one(cfg, label=f"jump_skew t={jt:.3f}", mode="jump_skew", threshold=jt)
        rows.append(row)
        print(
            f"    done: Sharpe={row.sharpe_nc:.2f}, "
            f"PnL=${row.total_pnl_nc:,.0f}, "
            f"MaxDD={row.max_dd_nc_pct:.2f}%, "
            f"routed={row.routing_frac*100:.1f}%",
        )

    # 3. SVI-skew routing sweep
    offset = len(rows) + 1
    for i, st in enumerate(svi_thresholds, start=offset):
        print(f"\n[{i}/?] svi-skew routing @ {st} vol pts...")
        cfg = _build_config(
            args.vol_arb_path, args.predictions_path, args.threshold,
            use_routing=True, routing_mode="svi_skew",
            svi_skew_threshold=st, jump_skew_threshold=0.05,
        )
        row = run_one(cfg, label=f"svi_skew t={st:.1f}", mode="svi_skew", threshold=st)
        rows.append(row)
        print(
            f"    done: Sharpe={row.sharpe_nc:.2f}, "
            f"PnL=${row.total_pnl_nc:,.0f}, "
            f"MaxDD={row.max_dd_nc_pct:.2f}%, "
            f"routed={row.routing_frac*100:.1f}%",
        )

    _print_table(rows, baseline)

    # Save
    if args.output:
        df = pd.DataFrame([r.__dict__ for r in rows])
        df.to_parquet(args.output)
        print(f"\nSaved {len(df)} rows to {args.output}")

    # Return 0 if any config passes the decision rule
    any_passes = any(
        r.label != baseline.label
        and r.sharpe_nc >= baseline.sharpe_nc
        and abs(r.max_dd_nc_pct) <= 1.10 * abs(baseline.max_dd_nc_pct)
        for r in rows
    )
    return 0 if any_passes else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
