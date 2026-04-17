"""Forensic test 8: bootstrap confidence intervals for the Sharpe ratio.

The vol-arb strategy has a point estimate daily non-compound Sharpe of
3.01 (per the full backtester run at threshold=25, no routing, no SOR, no
VRP). That number is meaningless without knowing its confidence interval —
especially given the concentration analysis in Stage 14.4:

- Top 5% of days contribute 70.9% of non-comp PnL
- Median trade is NEGATIVE ($-13)
- Skewness 10.58 (vs S&P 500 ≈ -0.5)

With that much tail concentration, the empirical Sharpe could easily have
a CI like [1.2, 4.8] rather than [2.9, 3.1]. This script answers the
question: how fragile is the reported Sharpe?

## Methodology

1. **Aggregate to daily**: trades happen on 15-min bars, but intra-day
   trades are NOT independent (holding periods overlap, same regime). The
   unit of resampling is one calendar day.

2. **Block bootstrap on days** (with replacement):
   - Draw N days uniformly at random, with replacement, from the
     empirical distribution of daily PnL
   - Compute Sharpe on the resampled series
   - Repeat 10,000 times → empirical CI

3. **Per-regime bootstrap**: split daily PnL by regime and run the
   bootstrap within each:
   - Pre-2024 (the "quiet" regime — 2022-2023)
   - 2024-2025 (the "hot" regime where most PnL lives)
   - 2026-YTD (the forward window that the live strategy will face)

4. **Uniform-regime bootstrap** (the critical diagnostic): instead of
   sampling from the observed mix of regimes (which is heavily weighted
   toward 2024-2025 because those were the good years), sample an EQUAL
   number of days from each regime. This answers: *"if the strategy
   faced each regime equally, what's its expected Sharpe?"* This
   directly addresses the concentration risk flagged in Stage 14.4.

## Decision gate

The Tier 1 exit criterion (from the improvement plan) requires the **5th
percentile of the bootstrap non-comp Sharpe >= 1.5**. Below that, the
strategy is too fragile to risk capital.

Runs via::

    PYTHONPATH=. python scripts/forensic/08_bootstrap_sharpe.py \\
        --n-bootstrap 10000 --seed 42
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vol_arb.backtester import BacktestConfig, VolArbBacktester

logger = logging.getLogger("bootstrap_sharpe")

_DAYS_PER_YEAR = 365.25


def _daily_nc_pnl(trade_log: pd.DataFrame) -> pd.Series:
    """Aggregate non-compound PnL to daily buckets.

    Uses ``net_pnl_nc`` if present (fixed reference notional), otherwise
    raises — the bootstrap only makes sense on the non-compound view
    because compounding injects path dependence.
    """
    if "net_pnl_nc" not in trade_log.columns:
        raise ValueError(
            "trade_log has no net_pnl_nc column — "
            "run the backtester with report_non_compounding=True",
        )
    pnl = trade_log["net_pnl_nc"]
    daily = pnl.groupby(pnl.index.normalize()).sum()
    return daily


def _sharpe_from_pnl(daily_pnl: np.ndarray) -> float:
    """Annualized Sharpe from a daily PnL array.

    Sharpe = mean / std * sqrt(365.25). Zero std → 0.0 (avoid inf).
    No risk-free rate subtracted (crypto 24/7 convention).
    """
    if len(daily_pnl) < 2:
        return 0.0
    mu = float(np.mean(daily_pnl))
    sd = float(np.std(daily_pnl, ddof=1))
    if sd <= 0.0:
        return 0.0
    return mu / sd * np.sqrt(_DAYS_PER_YEAR)


def bootstrap_sharpe(
    daily_pnl: pd.Series,
    n_bootstrap: int,
    rng: np.random.Generator,
    *,
    sample_size: int | None = None,
) -> np.ndarray:
    """Bootstrap Sharpe distribution from a daily PnL series.

    Draws ``sample_size`` days (default: len of input) with replacement,
    ``n_bootstrap`` times, and returns the array of annualized Sharpes.
    """
    n = len(daily_pnl)
    if n < 2:
        return np.zeros(n_bootstrap)
    if sample_size is None:
        sample_size = n
    values = daily_pnl.values
    out = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=sample_size)
        out[b] = _sharpe_from_pnl(values[idx])
    return out


def bootstrap_uniform_regime(
    regimes: dict[str, pd.Series],
    n_bootstrap: int,
    rng: np.random.Generator,
    sample_size_per_regime: int,
) -> np.ndarray:
    """Bootstrap by sampling an equal number of days from each regime.

    Under the normal bootstrap, the regime mix matches the observed mix
    (heavy 2024-2025 weight). This version samples ``sample_size_per_regime``
    days from each regime, then concatenates and computes Sharpe. The
    resulting distribution is "what if each regime got equal representation?"
    — a direct test of regime-rotation fragility.
    """
    out = np.empty(n_bootstrap)
    for b in range(n_bootstrap):
        parts = []
        for name, series in regimes.items():
            if len(series) == 0:
                continue
            idx = rng.integers(0, len(series), size=sample_size_per_regime)
            parts.append(series.values[idx])
        if not parts:
            out[b] = 0.0
            continue
        combined = np.concatenate(parts)
        out[b] = _sharpe_from_pnl(combined)
    return out


@dataclass
class CIResult:
    """One CI row for the summary table."""
    label: str
    n_days: int
    point_estimate: float
    mean: float
    std: float
    p2_5: float
    p5: float
    p50: float
    p95: float
    p97_5: float


def ci_from_bootstrap(
    label: str,
    point_daily_pnl: pd.Series,
    bootstrap_samples: np.ndarray,
    sample_size_override: int | None = None,
) -> CIResult:
    """Wrap a bootstrap result into the summary struct."""
    n_days = len(point_daily_pnl) if sample_size_override is None else sample_size_override
    return CIResult(
        label=label,
        n_days=n_days,
        point_estimate=_sharpe_from_pnl(point_daily_pnl.values),
        mean=float(np.mean(bootstrap_samples)),
        std=float(np.std(bootstrap_samples)),
        p2_5=float(np.quantile(bootstrap_samples, 0.025)),
        p5=float(np.quantile(bootstrap_samples, 0.05)),
        p50=float(np.quantile(bootstrap_samples, 0.50)),
        p95=float(np.quantile(bootstrap_samples, 0.95)),
        p97_5=float(np.quantile(bootstrap_samples, 0.975)),
    )


def _print_row(ci: CIResult) -> None:
    print(
        f"  {ci.label:<30s}"
        f"  {ci.n_days:>6d}"
        f"  {ci.point_estimate:>7.3f}"
        f"  {ci.p2_5:>7.3f}"
        f"  {ci.p5:>7.3f}"
        f"  {ci.p50:>7.3f}"
        f"  {ci.p95:>7.3f}"
        f"  {ci.p97_5:>7.3f}",
    )


def _print_table(rows: list[CIResult]) -> None:
    print()
    print("=" * 96)
    print(
        f"  {'REGIME':<30s}"
        f"  {'N days':>6s}"
        f"  {'Sharpe':>7s}"
        f"  {'p2.5':>7s}"
        f"  {'p5':>7s}"
        f"  {'p50':>7s}"
        f"  {'p95':>7s}"
        f"  {'p97.5':>7s}",
    )
    print("-" * 96)
    for ci in rows:
        _print_row(ci)
    print("=" * 96)


def main() -> int:
    parser = argparse.ArgumentParser(description="Bootstrap Sharpe CI for vol-arb")
    parser.add_argument(
        "--n-bootstrap", type=int, default=10000,
        help="Number of bootstrap draws (default: 10,000)",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument(
        "--vol-arb-path", type=str,
        default="vol_arb/output/BTC_vol_arb.parquet",
    )
    parser.add_argument(
        "--predictions-path", type=str,
        default="vol_model/output/BTCUSDT_hybrid_predictions.parquet",
    )
    parser.add_argument(
        "--output", type=str, default="",
        help="Save bootstrap samples + CI table to parquet",
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    rng = np.random.default_rng(args.seed)

    cfg = BacktestConfig(
        vol_arb_path=args.vol_arb_path,
        predictions_path=args.predictions_path,
        entry_threshold=args.threshold,
        use_sor=False,
        use_instrument_routing=False,
        use_vrp=False,
    )

    print()
    print("=" * 96)
    print("  BOOTSTRAP SHARPE CI — non-compound vol-arb strategy")
    print("=" * 96)
    print(f"  vol_arb_path:        {cfg.vol_arb_path}")
    print(f"  entry_threshold:     {cfg.entry_threshold}")
    print(f"  n_bootstrap:         {args.n_bootstrap:,}")
    print(f"  seed:                {args.seed}")
    print("-" * 96)
    print("  Running backtester to produce trade log...")

    bt = VolArbBacktester(cfg)
    metrics, trade_log = bt.run_backtest()
    nc = metrics.get("non_compounding", {})
    print(
        f"  Backtest OK: {int(metrics.get('n_trades', 0)):,} trades, "
        f"non-comp Sharpe (backtester internal): "
        f"{nc.get('sharpe_ratio', 0):.3f} [bar-level]",
    )

    # Aggregate to daily non-comp PnL
    daily = _daily_nc_pnl(trade_log)
    print(
        f"  Daily PnL series: {len(daily):,} days "
        f"(from {daily.index.min().date()} to {daily.index.max().date()})",
    )
    print(
        f"  Daily mean:   ${daily.mean():>10,.2f}    "
        f"std: ${daily.std():>10,.2f}    "
        f"skew: {daily.skew():>6.2f}",
    )
    print("-" * 96)

    # Regime splits — by calendar year
    regimes: dict[str, pd.Series] = {
        "Pre-2024 (2022-2023)": daily[daily.index.year < 2024],
        "Hot 2024-2025":        daily[(daily.index.year >= 2024) & (daily.index.year <= 2025)],
        "Forward 2026-YTD":     daily[daily.index.year >= 2026],
    }

    for name, series in regimes.items():
        pt = _sharpe_from_pnl(series.values) if len(series) > 0 else 0.0
        print(
            f"  {name:<30s}  N={len(series):>5d}  "
            f"mean=${series.mean():>9,.2f}  "
            f"std=${series.std():>9,.2f}  "
            f"point Sharpe={pt:>6.3f}",
        )
    print("-" * 96)
    print(f"  Running {args.n_bootstrap:,} bootstrap draws...")

    # Overall bootstrap
    overall_samples = bootstrap_sharpe(daily, args.n_bootstrap, rng)
    overall_ci = ci_from_bootstrap("OVERALL", daily, overall_samples)

    # Per-regime bootstrap
    regime_cis: list[CIResult] = []
    for name, series in regimes.items():
        if len(series) < 2:
            continue
        samples = bootstrap_sharpe(series, args.n_bootstrap, rng)
        regime_cis.append(ci_from_bootstrap(name, series, samples))

    # Uniform-regime bootstrap — equal sampling from each
    min_regime_size = min(len(s) for s in regimes.values() if len(s) > 0)
    # Sample size per regime: use the smallest regime's size so all regimes
    # contribute equally
    uniform_samples = bootstrap_uniform_regime(
        regimes, args.n_bootstrap, rng,
        sample_size_per_regime=min_regime_size,
    )

    # The "point estimate" for uniform regime is the Sharpe if we took
    # min_regime_size days from each and concatenated — we approximate by
    # equally-weighting the per-regime point Sharpes via the combined empirical
    equal_pts = []
    for series in regimes.values():
        if len(series) > 0:
            equal_pts.append(series.values[:min_regime_size])
    equal_combined = pd.Series(
        np.concatenate(equal_pts) if equal_pts else np.array([]),
    )
    uniform_ci = ci_from_bootstrap(
        "UNIFORM-REGIME (equal mix)",
        equal_combined,
        uniform_samples,
        sample_size_override=min_regime_size * len(regimes),
    )

    # ── Summary table ──
    rows = [overall_ci] + regime_cis + [uniform_ci]
    _print_table(rows)

    # ── Decision gate ──
    print()
    print("  Tier 1 exit gate: 5th percentile non-comp Sharpe >= 1.5")
    p5 = overall_ci.p5
    if p5 >= 1.5:
        verdict = f"PASS — overall p5 Sharpe = {p5:.3f} >= 1.5"
        ok = True
    else:
        verdict = f"FAIL — overall p5 Sharpe = {p5:.3f} < 1.5"
        ok = False
    print(f"  VERDICT: {verdict}")

    uniform_p5 = uniform_ci.p5
    print()
    print("  Regime-rotation stress: uniform-regime p5 Sharpe:")
    if uniform_p5 >= 1.0:
        u_verdict = f"OK — uniform p5 = {uniform_p5:.3f} >= 1.0"
    elif uniform_p5 >= 0.5:
        u_verdict = f"WEAK — uniform p5 = {uniform_p5:.3f} in [0.5, 1.0)"
    else:
        u_verdict = f"FRAGILE — uniform p5 = {uniform_p5:.3f} < 0.5 (regime-dependent edge)"
    print(f"  {u_verdict}")
    print("=" * 96)
    print()

    # Save
    if args.output:
        df_samples = pd.DataFrame({
            "overall": overall_samples,
            "uniform_regime": uniform_samples,
        })
        for r in regime_cis:
            # keep series length equal to n_bootstrap
            idx = list(regimes.keys()).index(r.label)
            df_samples[f"regime_{idx}"] = bootstrap_sharpe(
                list(regimes.values())[idx], args.n_bootstrap,
                np.random.default_rng(args.seed + 1000 + idx),
            )
        df_samples.to_parquet(args.output)
        print(f"  Saved {len(df_samples):,} bootstrap rows to {args.output}")

    return 0 if ok else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
