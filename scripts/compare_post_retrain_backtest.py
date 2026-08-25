"""Cycle 8 post-retrain V2 backtest impact comparison (2026-05-08).

After the 6-month --force retrain (C2 closure), this script runs the
invalidated historical V2 backtest against the fresh checkpoints and diffs key
metrics against the pre-retrain backup at
``backups/2026-05-08_pre_cycle8_retrain/BTC_v2_trade_log_pre_retrain.parquet``.

Reports:
  - Net non-compound PnL (USD)
  - Trade count
  - Trade-only Sharpe
  - Daily Sharpe
  - Max drawdown
  - Long / short trade split
  - Per-month PnL deltas

The backup is created by the operator before kicking off the retrain
so the baseline is the pre-retrain BTC_v2_trade_log.parquet (historical
cycle-7 trade log).

Usage::

    # After the retrain has finished overwriting checkpoints:
    python scripts/compare_post_retrain_backtest.py

    # Optionally generate a fresh predictions parquet first if the
    # walk-forward script didn't write one:
    python vol_model/extract_pruned_svi_har_pd_predictions.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

logger = logging.getLogger(__name__)


_BACKUP_DIR = REPO_ROOT / "backups" / "2026-05-08_pre_cycle8_retrain"
_PRE_RETRAIN_TRADE_LOG = _BACKUP_DIR / "BTC_v2_trade_log_pre_retrain.parquet"


def _run_canonical_v2_backtest() -> pd.DataFrame:
    """Run the invalidated historical V2 preset with realistic costs."""
    from vol_arb.backtester import BacktestConfig, VolArbBacktester
    cfg = (
        BacktestConfig()
        .with_historical_btc_v2()
        .with_realistic_costs()
    )
    _, trade_log = VolArbBacktester(cfg).run_backtest()
    return trade_log


def _per_trade_metrics(tl: pd.DataFrame) -> dict:
    """Extract the cycle-8 comparison metrics from a V2 trade log."""
    nc = tl["net_pnl_nc"].astype(float)
    fired = nc != 0
    n_trades = int(fired.sum())
    nc_pnl = float(nc.sum())
    # Trade-only Sharpe (annualised assuming N trades per year).
    fired_pnl = nc[fired]
    if len(fired_pnl) > 1 and fired_pnl.std(ddof=1) > 0:
        # Annualisation factor: sqrt(trades_per_year). For h=15m the
        # historical 15-minute heuristic is ~96 bars/day × 365 = ~35k bars/year;
        # trade rate is much lower. Use raw sharpe (PnL mean / std).
        sharpe_trade = float(fired_pnl.mean() / fired_pnl.std(ddof=1))
    else:
        sharpe_trade = 0.0
    # Long / short split. V2 trade log uses ``signal`` (1/-1/0) for the
    # direction; older variants used ``direction``. Handle both.
    dir_col = "signal" if "signal" in tl.columns else (
        "direction" if "direction" in tl.columns else None
    )
    if dir_col is not None:
        n_longs = int(((tl[dir_col] == 1) & fired).sum())
        n_shorts = int(((tl[dir_col] == -1) & fired).sum())
        pnl_longs = float(nc[(tl[dir_col] == 1) & fired].sum())
        pnl_shorts = float(nc[(tl[dir_col] == -1) & fired].sum())
    else:
        n_longs = n_shorts = 0
        pnl_longs = pnl_shorts = 0.0
    # Drawdown on cumulative NC PnL.
    cum = nc.cumsum()
    peak = cum.cummax()
    dd = cum - peak
    max_dd_usd = float(dd.min())
    # Daily Sharpe.
    if hasattr(tl.index, "date"):
        daily_pnl = nc.groupby(tl.index.date).sum()
    elif "timestamp" in tl.columns:
        daily_pnl = nc.groupby(pd.to_datetime(tl["timestamp"]).dt.date).sum()
    else:
        daily_pnl = pd.Series(dtype=float)
    if len(daily_pnl) > 1 and daily_pnl.std(ddof=1) > 0:
        sharpe_daily = float(daily_pnl.mean() / daily_pnl.std(ddof=1) * np.sqrt(365))
    else:
        sharpe_daily = 0.0
    return {
        "nc_pnl_usd": nc_pnl,
        "n_trades": n_trades,
        "sharpe_trade": sharpe_trade,
        "sharpe_daily": sharpe_daily,
        "max_dd_usd": max_dd_usd,
        "n_longs": n_longs,
        "n_shorts": n_shorts,
        "pnl_longs_usd": pnl_longs,
        "pnl_shorts_usd": pnl_shorts,
    }


def _per_month_pnl(tl: pd.DataFrame) -> dict[str, float]:
    """Sum NC PnL per calendar month."""
    nc = tl["net_pnl_nc"].astype(float)
    if hasattr(tl.index, "to_period"):
        try:
            keys = tl.index.to_period("M").astype(str)
        except (AttributeError, ValueError):
            return {}
    elif "timestamp" in tl.columns:
        keys = pd.to_datetime(tl["timestamp"]).dt.to_period("M").astype(str)
    else:
        return {}
    grouped = nc.groupby(keys).sum()
    return {str(k): float(v) for k, v in grouped.items()}


def _delta_table(post: dict, pre: dict) -> str:
    """Render the post-vs-pre delta table for the headline metrics."""
    keys = ["nc_pnl_usd", "n_trades", "sharpe_trade", "sharpe_daily",
            "max_dd_usd", "n_longs", "n_shorts", "pnl_longs_usd",
            "pnl_shorts_usd"]
    rows = ["| metric | pre-retrain | post-retrain | delta | delta% |",
            "|---|---:|---:|---:|---:|"]
    for k in keys:
        pre_val = pre[k]
        post_val = post[k]
        delta = post_val - pre_val
        pct = (delta / pre_val * 100) if pre_val else float("nan")
        rows.append(
            f"| `{k}` | {pre_val:,.4f} | {post_val:,.4f} | "
            f"{delta:+,.4f} | {pct:+.2f}% |"
        )
    return "\n".join(rows)


def main() -> int:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s: %(message)s")
    if not _PRE_RETRAIN_TRADE_LOG.exists():
        logger.error("Pre-retrain backup not found at %s", _PRE_RETRAIN_TRADE_LOG)
        return 1

    logger.info("Loading pre-retrain trade log: %s", _PRE_RETRAIN_TRADE_LOG)
    pre_tl = pd.read_parquet(_PRE_RETRAIN_TRADE_LOG)
    pre_metrics = _per_trade_metrics(pre_tl)
    pre_monthly = _per_month_pnl(pre_tl)
    logger.info("Pre-retrain: %d trades, NC PnL $%.0f",
                pre_metrics["n_trades"], pre_metrics["nc_pnl_usd"])

    logger.info("Running V2 backtest against fresh post-retrain checkpoints...")
    post_tl = _run_canonical_v2_backtest()
    post_metrics = _per_trade_metrics(post_tl)
    post_monthly = _per_month_pnl(post_tl)
    logger.info("Post-retrain: %d trades, NC PnL $%.0f",
                post_metrics["n_trades"], post_metrics["nc_pnl_usd"])

    # Persist the post-retrain trade log alongside the pre.
    post_path = _BACKUP_DIR / "BTC_v2_trade_log_post_retrain.parquet"
    post_tl.to_parquet(post_path, index=True)
    logger.info("Wrote post-retrain trade log to %s", post_path)

    # Build the impact report.
    report_lines = [
        "# Cycle 8 post-retrain V2 backtest impact (2026-05-08)",
        "",
        "## Headline metrics",
        "",
        _delta_table(post_metrics, pre_metrics),
        "",
        "## Per-month NC PnL (USD)",
        "",
        "| month | pre-retrain | post-retrain | delta |",
        "|---|---:|---:|---:|",
    ]
    all_months = sorted(set(pre_monthly) | set(post_monthly))
    for m in all_months:
        pre_m = pre_monthly.get(m, 0.0)
        post_m = post_monthly.get(m, 0.0)
        delta_m = post_m - pre_m
        report_lines.append(f"| {m} | {pre_m:,.0f} | {post_m:,.0f} | {delta_m:+,.0f} |")

    report_path = REPO_ROOT / "vol_arb" / "output" / "post_retrain_impact_cycle8.md"
    report_path.write_text("\n".join(report_lines))
    logger.info("Wrote impact report to %s", report_path)

    json_path = report_path.with_suffix(".json")
    json_path.write_text(json.dumps({
        "pre_retrain": pre_metrics,
        "post_retrain": post_metrics,
        "pre_monthly": pre_monthly,
        "post_monthly": post_monthly,
    }, indent=2, default=float))
    logger.info("Wrote impact JSON to %s", json_path)

    # Print the headline to stdout.
    print("=" * 72)
    print("CYCLE 8 POST-RETRAIN V2 IMPACT")
    print("=" * 72)
    print(_delta_table(post_metrics, pre_metrics))
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
