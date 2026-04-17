#!/usr/bin/env python
"""W4.1 gate 5: paired bootstrap CI on daily non-compound PnL improvement.

Runs two backtests (baseline + classifier-gated at a given threshold),
aggregates ``net_pnl_nc`` to daily buckets, pairs the two series on date,
and computes a nonparametric bootstrap 90% confidence interval on the
mean of the paired difference ``gated - baseline``.

The gate is satisfied if the 5th percentile of the bootstrap distribution
is strictly positive, i.e. the 90% CI lower bound is above zero. This is
the same decision rule used by the T3.2 RND validation for consistency.

Usage:
    PYTHONPATH=. python scripts/forensic/13_classifier_bootstrap.py \\
        --classifier-path vol_model/output/BTCUSDT_classifier_head_edge_predictions.parquet \\
        --classifier-threshold 0.55 \\
        --n-draws 10000

    # Sweep thresholds; write results table to a JSON summary:
    PYTHONPATH=. python scripts/forensic/13_classifier_bootstrap.py \\
        --classifier-path vol_model/output/BTCUSDT_classifier_head_edge_predictions.parquet \\
        --thresholds 0.55,0.60,0.65,0.70,0.75 \\
        --summary-path vol_arb/output/classifier_gate_sweep.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.backtester import BacktestConfig, VolArbBacktester

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("classifier_bootstrap")


@dataclass
class SweepRow:
    threshold: float
    n_trades_base: int
    n_trades_gated: int
    trade_retention_pct: float
    nc_sharpe_base: float
    nc_sharpe_gated: float
    nc_pnl_base: float
    nc_pnl_gated: float
    mean_daily_diff: float
    p5_diff: float
    p50_diff: float
    p95_diff: float
    gate_passed: bool
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        d = self.__dict__.copy()
        d["extras"] = dict(d["extras"])  # shallow copy
        return d


def _run_backtest(
    *, classifier_path: str | None, threshold: float,
    entry_threshold: float, initial_equity: float,
    nan_is_pass: bool,
) -> tuple[dict, pd.DataFrame]:
    """Run a single backtest with or without the classifier gate."""
    cfg_kwargs = dict(
        use_instrument_routing=False,
        report_non_compounding=True,
        entry_threshold=entry_threshold,
        initial_equity=initial_equity,
    )
    if classifier_path:
        cfg_kwargs["classifier_path"] = classifier_path
        cfg_kwargs["classifier_threshold"] = threshold
        cfg_kwargs["classifier_nan_is_pass"] = nan_is_pass
    cfg = BacktestConfig(**cfg_kwargs)
    bt = VolArbBacktester(cfg)
    metrics, trade_log = bt.run_backtest()
    return metrics, trade_log


def _restrict_window(
    trade_log: pd.DataFrame,
    window_start: pd.Timestamp | None,
    window_end: pd.Timestamp | None,
) -> pd.DataFrame:
    """Filter a trade log to a given UTC window (inclusive start, exclusive end)."""
    tl = trade_log
    if window_start is not None:
        tl = tl.loc[tl.index >= window_start]
    if window_end is not None:
        tl = tl.loc[tl.index < window_end]
    return tl


def _sharpe_from_trade_log(trade_log: pd.DataFrame) -> float:
    """Recompute bar-level NC Sharpe on a (possibly windowed) trade log."""
    r = trade_log["r_t_nc"].to_numpy()
    r_active = r[r != 0]
    if len(r_active) < 2:
        return 0.0
    mean = float(r_active.mean())
    std = float(r_active.std(ddof=1))
    if std <= 0:
        return 0.0
    bars_per_year = 365.25 * 24 * 4
    return mean / std * float(np.sqrt(bars_per_year))


def _daily_nc_pnl(trade_log: pd.DataFrame) -> pd.Series:
    """Aggregate net_pnl_nc to daily sums on the UTC date index."""
    if "net_pnl_nc" not in trade_log.columns:
        raise KeyError(
            "trade_log is missing net_pnl_nc — report_non_compounding must be on",
        )
    daily = (
        trade_log["net_pnl_nc"]
        .groupby(trade_log.index.normalize())
        .sum()
    )
    daily.index = pd.to_datetime(daily.index)
    daily.name = "daily_pnl_nc"
    return daily


def _paired_bootstrap(
    diff: np.ndarray, n_draws: int, seed: int,
) -> tuple[float, float, float, float]:
    """Bootstrap the mean of a paired difference series.

    Returns (mean, p5, p50, p95).
    """
    n = len(diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_draws, n))
    samples = diff[idx].mean(axis=1)
    return (
        float(diff.mean()),
        float(np.percentile(samples, 5)),
        float(np.percentile(samples, 50)),
        float(np.percentile(samples, 95)),
    )


def run_sweep(
    classifier_path: str,
    thresholds: list[float],
    n_draws: int,
    entry_threshold: float,
    initial_equity: float,
    min_trade_retention: float,
    seed: int,
    window_start: pd.Timestamp | None,
    window_end: pd.Timestamp | None,
    nan_is_pass: bool,
) -> list[SweepRow]:
    logger.info("Running baseline backtest (no classifier)...")
    t0 = time.time()
    base_metrics_full, base_tl_full = _run_backtest(
        classifier_path=None, threshold=0.55,
        entry_threshold=entry_threshold, initial_equity=initial_equity,
        nan_is_pass=nan_is_pass,
    )
    logger.info("Baseline full run done in %.1fs", time.time() - t0)
    base_tl = _restrict_window(base_tl_full, window_start, window_end)
    base_daily = _daily_nc_pnl(base_tl)
    n_base_trades = int((base_tl["signal"] != 0).sum())
    base_nc_sharpe = _sharpe_from_trade_log(base_tl)
    base_nc_pnl = float(base_tl["net_pnl_nc"].sum())
    if window_start or window_end:
        logger.info(
            "Baseline windowed to [%s, %s): %d rows (from %d full)",
            window_start, window_end, len(base_tl), len(base_tl_full),
        )
    logger.info(
        "Baseline: %d trades, windowed NC Sharpe=%.3f, NC PnL=$%.0f",
        n_base_trades, base_nc_sharpe, base_nc_pnl,
    )

    rows: list[SweepRow] = []
    for thr in thresholds:
        logger.info("--- Threshold %.2f ---", thr)
        t0 = time.time()
        g_metrics_full, g_tl_full = _run_backtest(
            classifier_path=classifier_path, threshold=thr,
            entry_threshold=entry_threshold, initial_equity=initial_equity,
            nan_is_pass=nan_is_pass,
        )
        logger.info("Gated done in %.1fs", time.time() - t0)
        g_tl = _restrict_window(g_tl_full, window_start, window_end)
        g_daily = _daily_nc_pnl(g_tl)
        n_gated_trades = int((g_tl["signal"] != 0).sum())
        retention_pct = (
            100.0 * n_gated_trades / n_base_trades if n_base_trades else 0.0
        )
        g_nc_sharpe = _sharpe_from_trade_log(g_tl)
        g_nc_pnl = float(g_tl["net_pnl_nc"].sum())

        # Pair on date index; outer join, fill missing days with 0
        paired = pd.concat(
            [base_daily.rename("base"), g_daily.rename("gated")],
            axis=1,
        ).fillna(0.0)
        diff = (paired["gated"] - paired["base"]).to_numpy()

        mean_diff, p5, p50, p95 = _paired_bootstrap(diff, n_draws, seed)
        gate_ok = (
            p5 > 0
            and retention_pct >= 100.0 * min_trade_retention
        )
        logger.info(
            "  trades: %d/%d (%.1f%%), NC Sharpe: %.3f -> %.3f, "
            "PnL: $%.0f -> $%.0f, mean daily diff: $%.0f, "
            "p5=$%.0f, p95=$%.0f, pass=%s",
            n_gated_trades, n_base_trades, retention_pct,
            base_nc_sharpe, g_nc_sharpe,
            base_nc_pnl, g_nc_pnl,
            mean_diff, p5, p95, gate_ok,
        )
        rows.append(SweepRow(
            threshold=thr,
            n_trades_base=n_base_trades,
            n_trades_gated=n_gated_trades,
            trade_retention_pct=retention_pct,
            nc_sharpe_base=base_nc_sharpe,
            nc_sharpe_gated=g_nc_sharpe,
            nc_pnl_base=base_nc_pnl,
            nc_pnl_gated=g_nc_pnl,
            mean_daily_diff=mean_diff,
            p5_diff=p5,
            p50_diff=p50,
            p95_diff=p95,
            gate_passed=gate_ok,
        ))

    return rows


def format_report(rows: list[SweepRow], min_retention: float) -> str:
    lines = []
    lines.append("=" * 84)
    lines.append("  W4.1 CLASSIFIER GATE — BOOTSTRAP CI SWEEP")
    lines.append("=" * 84)
    lines.append("")
    hdr = (
        f"  {'Thr':>5s}  {'Trades':>14s}  {'Retain':>7s}  "
        f"{'NC Sharpe':>15s}  {'Mean Diff':>11s}  "
        f"{'p5':>10s}  {'p95':>10s}  {'Pass':>5s}"
    )
    lines.append(hdr)
    lines.append("-" * 84)
    for r in rows:
        trades = f"{r.n_trades_gated:>6d}/{r.n_trades_base:<6d}"
        sharpe = f"{r.nc_sharpe_base:.2f}->{r.nc_sharpe_gated:.2f}"
        pass_str = "OK" if r.gate_passed else "NO"
        lines.append(
            f"  {r.threshold:>5.2f}  {trades:>14s}  "
            f"{r.trade_retention_pct:>6.1f}%  {sharpe:>15s}  "
            f"${r.mean_daily_diff:>10.0f}  ${r.p5_diff:>9.0f}  "
            f"${r.p95_diff:>9.0f}  {pass_str:>5s}"
        )
    lines.append("-" * 84)
    lines.append(
        f"  Gate criteria: p5_diff > 0 AND trade_retention >= {100*min_retention:.0f}%"
    )
    passing = [r for r in rows if r.gate_passed]
    if passing:
        winner = max(passing, key=lambda r: r.nc_sharpe_gated)
        lines.append("")
        lines.append(
            f"  WINNER: threshold {winner.threshold:.2f} — "
            f"NC Sharpe {winner.nc_sharpe_base:.3f} -> {winner.nc_sharpe_gated:.3f}, "
            f"retention {winner.trade_retention_pct:.1f}%, "
            f"90% CI [${winner.p5_diff:.0f}, ${winner.p95_diff:.0f}]/day"
        )
    else:
        lines.append("")
        lines.append("  NO THRESHOLD PASSED — classifier gate not adopted.")
    lines.append("=" * 84)
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--classifier-path", type=str, required=True,
        help="Path to the classifier predictions parquet",
    )
    parser.add_argument(
        "--thresholds", type=str, default="0.55,0.60,0.65,0.70,0.75",
        help="Comma-separated classifier thresholds to sweep",
    )
    parser.add_argument(
        "--classifier-threshold", type=float, default=None,
        help="Single threshold shortcut (equivalent to --thresholds <value>)",
    )
    parser.add_argument(
        "--n-draws", type=int, default=10000,
        help="Number of bootstrap draws (default 10000)",
    )
    parser.add_argument(
        "--entry-threshold", type=float, default=0.5,
        help="Primary edge threshold for the baseline backtest",
    )
    parser.add_argument(
        "--initial-equity", type=float, default=100_000.0,
        help="Initial equity for both runs",
    )
    parser.add_argument(
        "--min-trade-retention", type=float, default=0.50,
        help="Minimum fraction of baseline trades the gated run must keep",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Bootstrap RNG seed",
    )
    parser.add_argument(
        "--window-start", type=str, default="",
        help=(
            "Optional UTC start date (YYYY-MM-DD) to restrict both runs. "
            "Default: auto — uses the classifier parquet's min timestamp."
        ),
    )
    parser.add_argument(
        "--window-end", type=str, default="",
        help=(
            "Optional UTC end date (YYYY-MM-DD) to restrict both runs. "
            "Default: auto — uses the classifier parquet's max timestamp + 1d."
        ),
    )
    parser.add_argument(
        "--full-window", action="store_true",
        help=(
            "Skip auto-windowing and use the full backtest range — tests the "
            "'wire it into production' path with nan_is_pass=True."
        ),
    )
    parser.add_argument(
        "--nan-is-pass", action="store_true",
        help=(
            "Let bars with NaN classifier output pass through ungated "
            "(production default). Off = mute NaN bars (validation A/B)."
        ),
    )
    parser.add_argument(
        "--summary-path", type=str,
        default="vol_arb/output/classifier_gate_bootstrap.json",
        help="Where to write the per-threshold summary JSON",
    )
    parser.add_argument(
        "--report-path", type=str,
        default="vol_arb/output/classifier_gate_bootstrap_report.md",
        help="Where to write the human-readable report",
    )
    args = parser.parse_args()

    if args.classifier_threshold is not None:
        thresholds = [args.classifier_threshold]
    else:
        thresholds = [float(t.strip()) for t in args.thresholds.split(",")]

    # Window resolution
    if args.full_window:
        window_start = None
        window_end = None
        logger.info("Window: full backtest range")
    else:
        if args.window_start:
            window_start = pd.Timestamp(args.window_start, tz="UTC")
        else:
            clf_idx = pd.read_parquet(args.classifier_path).index
            window_start = pd.Timestamp(clf_idx.min())
        if args.window_end:
            window_end = pd.Timestamp(args.window_end, tz="UTC")
        else:
            clf_idx = pd.read_parquet(args.classifier_path).index
            window_end = pd.Timestamp(clf_idx.max()) + pd.Timedelta(days=1)
        logger.info("Window: [%s, %s)", window_start, window_end)

    rows = run_sweep(
        classifier_path=args.classifier_path,
        thresholds=thresholds,
        n_draws=args.n_draws,
        entry_threshold=args.entry_threshold,
        initial_equity=args.initial_equity,
        min_trade_retention=args.min_trade_retention,
        seed=args.seed,
        window_start=window_start,
        window_end=window_end,
        nan_is_pass=args.nan_is_pass,
    )

    report = format_report(rows, args.min_trade_retention)
    print(report)

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(
        {
            "classifier_path": args.classifier_path,
            "n_draws": args.n_draws,
            "min_trade_retention": args.min_trade_retention,
            "rows": [r.as_dict() for r in rows],
        },
        indent=2,
        default=str,
    ))
    logger.info("Summary JSON written to %s", summary_path)

    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    logger.info("Report written to %s", report_path)


if __name__ == "__main__":
    main()
