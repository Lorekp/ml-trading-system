"""Forensic test 6: PnL concentration analysis.

If the $11M Sharpe 12.79 is concentrated in a few days/months, it's a regime
play, not a robust strategy. If it's distributed, there's likely real edge
(but possibly still inflated by lookahead).
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np

from vol_arb.backtester import BacktestConfig, VolArbBacktester

logging.basicConfig(level=logging.WARNING)


def analyze(vol_arb_path: str, label: str, thresh: float = 20.0) -> None:
    cfg = BacktestConfig(
        vol_arb_path=vol_arb_path,
        predictions_path="vol_model/output/BTCUSDT_hybrid_predictions.parquet",
        entry_threshold=thresh,
    )
    bt = VolArbBacktester(cfg)
    m, trade_log = bt.run_backtest()

    active = trade_log[trade_log["signal"] != 0].copy()
    active["date"] = active.index.normalize()
    active["year_month"] = active.index.to_period("M")
    active["year"] = active.index.year

    total_pnl = active["net_pnl"].sum()
    print("=" * 70)
    print(f"{label} | threshold={thresh}")
    print("=" * 70)
    print(f"Total active trades: {len(active):,}")
    print(f"Total PnL: ${total_pnl:,.0f}")
    print(f"Sharpe: {m['sharpe_ratio']:.2f}")
    print()

    # Yearly PnL
    print("Yearly PnL:")
    yearly = active.groupby("year")["net_pnl"].agg(["sum", "count", "mean"])
    yearly["pct_of_total"] = 100 * yearly["sum"] / total_pnl
    print(yearly.to_string())
    print()

    # Top 10 days
    daily = active.groupby("date")["net_pnl"].sum().sort_values(ascending=False)
    print(f"Top 10 winning days (of {len(daily):,} total):")
    top10 = daily.head(10)
    print(top10.to_string())
    print(f"Top 10 days PnL: ${top10.sum():,.0f} ({100 * top10.sum() / total_pnl:.1f}% of total)")
    print()

    print("Top 10 losing days:")
    bottom10 = daily.tail(10)
    print(bottom10.to_string())
    print()

    # Concentration metric
    top_1pct = int(len(daily) * 0.01)
    top_5pct = int(len(daily) * 0.05)
    top1_pnl = daily.head(top_1pct).sum()
    top5_pnl = daily.head(top_5pct).sum()
    print(f"Top 1% of days ({top_1pct} days) contribute: ${top1_pnl:,.0f} ({100*top1_pnl/total_pnl:.1f}%)")
    print(f"Top 5% of days ({top_5pct} days) contribute: ${top5_pnl:,.0f} ({100*top5_pnl/total_pnl:.1f}%)")
    print()

    # PnL distribution
    print(f"Per-trade PnL distribution:")
    print(f"  Mean: ${active['net_pnl'].mean():.2f}")
    print(f"  Median: ${active['net_pnl'].median():.2f}")
    print(f"  Std: ${active['net_pnl'].std():.2f}")
    print(f"  Skew: {active['net_pnl'].skew():.2f}")
    print(f"  Max win: ${active['net_pnl'].max():,.0f}")
    print(f"  Max loss: ${active['net_pnl'].min():,.0f}")
    print()


def main() -> None:
    analyze(
        "vol_arb/output/BTC_vol_arb_svi_shifted.parquet",
        "SVI (shifted/fixed)",
        thresh=20.0,
    )
    print()
    analyze(
        "vol_arb/output/BTC_vol_arb_shifted.parquet",
        "Baseline raw IV (shifted/fixed)",
        thresh=20.0,
    )


if __name__ == "__main__":
    main()
