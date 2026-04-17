"""Forensic test 5: permutation test for `iv_used` as a signal input.

**Null hypothesis (H0)**: the vol-arb strategy's edge does NOT come from the
true temporal relationship between the model's predicted realized vol
(`rv_pred_pct`) and the implied-vol of the traded instrument (`iv_used`). If
H0 is true, randomly permuting the `iv_used` values used for signal
generation should yield a strategy whose Sharpe is statistically
indistinguishable from the actual strategy.

**Alternative (H1)**: the edge DOES come from the true time-aligned
relationship. Under H1, shuffling destroys the signal and the shuffled
distribution of Sharpe should collapse to ~0.

**Decision**: if the actual non-compounding Sharpe is outside the 95th
percentile of the shuffled null distribution, we reject H0 — the edge is
real. If it's inside, the edge is an artifact of something else
(concentration, leakage, overfit on unrelated features).

## Implementation — "surgical shuffle"

A naive shuffle of the `iv_used` column has a fatal flaw. The backtester uses
`iv_used` in three places:

1. **Signal generation** — ``edge = rv_pred_pct - iv_used``, direction from
   the sign of this edge
2. **Entry pricing** — ``iv_entry = iv_used[T]``
3. **Exit pricing** — ``iv_exit = iv_used[T+15]``, used in vega P&L

A brute-force shuffle breaks all three simultaneously and introduces a
spurious "mean-reversion" edge: after shuffling, signal=+1 is conditionally
selected on LOW iv_entry draws (because ``edge = rv_pred - low = large``),
and the independently-drawn iv_exit reverts to the distribution mean →
positive vega P&L purely from selection bias, not from any real edge. We
observed this empirically on the first smoke test: shuffled runs produced
~$11.5M P&L vs $519K baseline — nonsense.

The clean test shuffles iv_used **only in the signal-generation pathway**,
by constructing a separate ``iv_used_signal`` column that feeds the edge
calculation while the true ``iv_used`` continues to drive entry/exit
pricing. This isolates the question "does the model's rv_pred contain
information time-aligned with the real iv_used?" without contaminating the
P&L mechanics.

Two modes are supported:

- ``--mode iv_signal`` (default): shuffle ``iv_used`` only for the edge
  calculation. Preserves all pricing. Directly tests backlog item #7.
- ``--mode rv_pred``: shuffle ``rv_pred_pct`` instead. Equivalent test
  (randomizes the signal direction), but from the rv side.

Both should produce the same conclusion if implemented correctly, since
``edge = rv_pred - iv_used`` and permuting either argument produces a
random sign.

Runs via::

    PYTHONPATH=. python scripts/forensic/05_shuffle_ivs.py \\
        --n-shuffles 20 --seed 42 --mode iv_signal
"""

from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from vol_arb.backtester import BacktestConfig, VolArbBacktester

logger = logging.getLogger("shuffle_ivs")


@dataclass
class ShuffleRun:
    """One permutation run's summary."""
    seed: int
    n_trades: int
    total_pnl_compound: float
    total_pnl_noncomp: float
    sharpe_compound: float
    sharpe_noncomp: float
    max_dd_compound: float
    max_dd_noncomp: float
    wall_seconds: float


class ShuffledBacktester(VolArbBacktester):
    """Backtester that permutes a single signal input without touching pricing.

    Subclasses :meth:`_prepare` so that the ``edge`` used for signal
    generation is computed from a shuffled column, but the ``iv_used`` /
    ``iv_exit`` columns that feed vega P&L remain unchanged.

    Modes:
    - ``iv_signal``: shuffle the iv_used values used for edge calc only
    - ``rv_pred``: shuffle the rv_pred_pct column before edge calc

    Either mode randomizes the signal direction; the P&L mechanics are
    identical to the real backtester in both cases.
    """

    def __init__(
        self,
        config: BacktestConfig,
        shuffle_seed: int,
        mode: str = "iv_signal",
    ):
        super().__init__(config)
        self._shuffle_seed = shuffle_seed
        if mode not in ("iv_signal", "rv_pred"):
            raise ValueError(f"unknown shuffle mode: {mode}")
        self._mode = mode

    def _prepare(self) -> pd.DataFrame:
        """Reimplementation of ``_prepare`` with a shuffled signal input.

        Mirrors the original ``_prepare`` in :mod:`vol_arb.backtester` but
        computes ``edge`` from a permuted input. Everything downstream
        (including ``iv_used``, ``iv_exit``, entry IV, vega) uses the real
        unshuffled values.
        """
        df = self.load_data()

        spread = self.compute_spread(df["jump_fraction"].values)
        df["spread_iv"] = spread

        # ── Surgical shuffle: permute ONE input to the edge calc ──
        rng = np.random.default_rng(self._shuffle_seed)
        perm = rng.permutation(len(df))

        if self._mode == "iv_signal":
            iv_for_signal = df["iv_used"].values[perm]
            rv_pred_for_signal = df["rv_pred_pct"].values
            logger.debug(
                "Shuffled iv_used for signal (seed=%d, n=%d)",
                self._shuffle_seed, len(df),
            )
        else:  # rv_pred
            iv_for_signal = df["iv_used"].values
            rv_pred_for_signal = df["rv_pred_pct"].values[perm]
            logger.debug(
                "Shuffled rv_pred_pct (seed=%d, n=%d)",
                self._shuffle_seed, len(df),
            )

        # Edge uses the shuffled input; adjusted_iv column stays real for
        # compatibility with downstream code that might log/inspect it.
        adjusted_iv_signal = iv_for_signal - df["vrp_baseline"].values
        edge_shuffled = rv_pred_for_signal - adjusted_iv_signal

        df["adjusted_iv"] = df["iv_used"].values - df["vrp_baseline"].values
        df["edge"] = edge_shuffled  # <-- permuted

        df["signal"] = self.generate_signals(
            edge_shuffled, spread / 2.0, df["jump_fraction"].values,
        )

        n_long = (df["signal"] == 1).sum()
        n_short = (df["signal"] == -1).sum()
        n_flat = (df["signal"] == 0).sum()
        logger.debug(
            "Shuffled signals: %d long, %d short, %d flat (%.1f%% active)",
            n_long, n_short, n_flat,
            100 * (n_long + n_short) / max(1, len(df)),
        )

        df = self.align_exits(df)
        return df


def run_one(
    config: BacktestConfig,
    *,
    seed: int | None,
    mode: str = "iv_signal",
) -> ShuffleRun:
    """Run the backtester once, either baseline (seed=None) or shuffled."""
    t0 = time.time()
    if seed is None:
        bt = VolArbBacktester(config)
    else:
        bt = ShuffledBacktester(config, shuffle_seed=seed, mode=mode)
    metrics, trade_log = bt.run_backtest()
    wall = time.time() - t0

    compound = metrics
    noncomp = metrics.get("non_compounding", {})

    return ShuffleRun(
        seed=seed if seed is not None else -1,
        n_trades=int(compound.get("n_trades", 0)),
        total_pnl_compound=float(compound.get("total_pnl", 0.0)),
        total_pnl_noncomp=float(noncomp.get("total_pnl", 0.0)),
        sharpe_compound=float(compound.get("sharpe_ratio", 0.0)),
        sharpe_noncomp=float(noncomp.get("sharpe_ratio", 0.0)),
        max_dd_compound=float(compound.get("max_drawdown_pct", 0.0)),
        max_dd_noncomp=float(noncomp.get("max_drawdown_pct", 0.0)),
        wall_seconds=wall,
    )


def summarize(
    baseline: ShuffleRun,
    shuffles: list[ShuffleRun],
    mode: str,
) -> None:
    """Print the null distribution and compare to actual."""
    if not shuffles:
        print("No shuffles produced — nothing to summarize.")
        return

    df_shuf = pd.DataFrame([s.__dict__ for s in shuffles])
    print()
    print("=" * 78)
    print(f"  SHUFFLE TEST ({mode}) — NULL DISTRIBUTION (non-compound view)")
    print("=" * 78)
    print(f"  N shuffles:          {len(shuffles)}")
    print(f"  Total wall (s):      {df_shuf['wall_seconds'].sum():.1f}")
    print("-" * 78)
    print(f"  Baseline (unshuffled):")
    print(f"    Non-comp Sharpe:   {baseline.sharpe_noncomp:>10.3f}")
    print(f"    Non-comp PnL:      ${baseline.total_pnl_noncomp:>14,.0f}")
    print(f"    N trades:          {baseline.n_trades:>10,d}")
    print("-" * 78)
    print(f"  Shuffled null (non-comp Sharpe):")
    print(f"    Mean:              {df_shuf['sharpe_noncomp'].mean():>10.3f}")
    print(f"    Std:               {df_shuf['sharpe_noncomp'].std():>10.3f}")
    print(f"    Min:               {df_shuf['sharpe_noncomp'].min():>10.3f}")
    print(f"    Max:               {df_shuf['sharpe_noncomp'].max():>10.3f}")
    print(f"    P5:                {df_shuf['sharpe_noncomp'].quantile(0.05):>10.3f}")
    print(f"    P50 (median):      {df_shuf['sharpe_noncomp'].quantile(0.50):>10.3f}")
    print(f"    P95:               {df_shuf['sharpe_noncomp'].quantile(0.95):>10.3f}")
    print(f"    P99:               {df_shuf['sharpe_noncomp'].quantile(0.99):>10.3f}")
    print("-" * 78)
    print(f"  Shuffled null (non-comp total PnL, USD):")
    print(f"    Mean:              ${df_shuf['total_pnl_noncomp'].mean():>14,.0f}")
    print(f"    Std:               ${df_shuf['total_pnl_noncomp'].std():>14,.0f}")
    print(f"    Min:               ${df_shuf['total_pnl_noncomp'].min():>14,.0f}")
    print(f"    Max:               ${df_shuf['total_pnl_noncomp'].max():>14,.0f}")
    print(f"    P5:                ${df_shuf['total_pnl_noncomp'].quantile(0.05):>14,.0f}")
    print(f"    P95:               ${df_shuf['total_pnl_noncomp'].quantile(0.95):>14,.0f}")
    print("-" * 78)

    # How extreme is the baseline?
    n_above = int((df_shuf["sharpe_noncomp"] >= baseline.sharpe_noncomp).sum())
    p_value = (n_above + 1) / (len(df_shuf) + 1)
    print(f"  Baseline beats {len(df_shuf) - n_above}/{len(df_shuf)} shuffles on Sharpe")
    print(f"  One-sided p-value:   {p_value:.4f} "
          f"(+1/+1 continuity correction)")
    print("-" * 78)

    # Verdict — pass if baseline is above p95 of null
    p95 = df_shuf["sharpe_noncomp"].quantile(0.95)
    if baseline.sharpe_noncomp > p95:
        verdict = (
            "PASS — baseline Sharpe is outside the 95th percentile of the "
            "shuffled null. Edge is NOT explained by random time-alignment."
        )
        ok = True
    else:
        verdict = (
            "FAIL — baseline Sharpe is inside the shuffled null distribution. "
            "Strategy has no detectable edge over a random signal on the "
            "same timestamps. Re-examine the signal generation path."
        )
        ok = False
    print(f"  VERDICT: {verdict}")
    print("=" * 78)
    print()
    return ok


def main() -> int:
    parser = argparse.ArgumentParser(description="Shuffle test for iv_used")
    parser.add_argument(
        "--n-shuffles", type=int, default=20,
        help="Number of permutation draws (default: 20)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Master RNG seed (default: 42)",
    )
    parser.add_argument(
        "--threshold", type=float, default=25.0,
        help="Backtester entry_threshold (default: 25)",
    )
    parser.add_argument(
        "--mode", type=str, default="iv_signal",
        choices=["iv_signal", "rv_pred"],
        help="Shuffle iv_used (default) or rv_pred_pct for the signal",
    )
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
        help="Save shuffle run table to parquet",
    )
    parser.add_argument(
        "--verbose", action="store_true",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    cfg = BacktestConfig(
        vol_arb_path=args.vol_arb_path,
        predictions_path=args.predictions_path,
        entry_threshold=args.threshold,
        use_sor=False,
        use_instrument_routing=False,
        use_vrp=False,
    )

    master_rng = np.random.default_rng(args.seed)
    child_seeds = master_rng.integers(0, 2**31 - 1, size=args.n_shuffles).tolist()

    print()
    print("=" * 78)
    print(f"  SHUFFLE TEST ({args.mode}) — baseline + {args.n_shuffles} permutations")
    print("=" * 78)
    print(f"  vol_arb_path:        {cfg.vol_arb_path}")
    print(f"  predictions_path:    {cfg.predictions_path}")
    print(f"  entry_threshold:     {cfg.entry_threshold}")
    print(f"  mode:                {args.mode}")
    print(f"  master seed:         {args.seed}")
    print("=" * 78)

    print("\n[baseline] running unshuffled backtest...")
    baseline = run_one(cfg, seed=None)
    print(
        f"[baseline] done in {baseline.wall_seconds:.1f}s — "
        f"non-comp Sharpe={baseline.sharpe_noncomp:.3f}, "
        f"PnL=${baseline.total_pnl_noncomp:,.0f}, "
        f"trades={baseline.n_trades}",
    )

    shuffles: list[ShuffleRun] = []
    for i, seed in enumerate(child_seeds, start=1):
        print(f"\n[shuffle {i}/{args.n_shuffles}] seed={seed} running...")
        s = run_one(cfg, seed=seed, mode=args.mode)
        shuffles.append(s)
        print(
            f"[shuffle {i}/{args.n_shuffles}] done in {s.wall_seconds:.1f}s — "
            f"non-comp Sharpe={s.sharpe_noncomp:.3f}, "
            f"PnL=${s.total_pnl_noncomp:,.0f}, "
            f"trades={s.n_trades}",
        )

    ok = summarize(baseline, shuffles, args.mode)

    if args.output:
        rows = [baseline.__dict__] + [s.__dict__ for s in shuffles]
        df_out = pd.DataFrame(rows)
        df_out["is_baseline"] = df_out["seed"] == -1
        df_out["mode"] = args.mode
        df_out.to_parquet(args.output)
        print(f"Saved {len(df_out)} rows to {args.output}")

    return 0 if ok else 2


if __name__ == "__main__":
    import sys
    sys.exit(main())
