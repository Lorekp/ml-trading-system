"""Forensic test 10: IV-movement oracle ceiling.

**Question**: "How much PnL is possible if our strategy knew the future
IV movement perfectly? And how close is the current model to that
ceiling?"

This is distinct from the existing `04_oracle_vs_model.py` which uses
a future-RV oracle (y_pred = actual future realized vol). The RV oracle
tells us "how much PnL could a perfect RV predictor earn" — a useful
bound on prediction quality. The IV oracle tells us something different
and more directly relevant to the strategy's P&L mechanics:

The vol-arb strategy's vega P&L is literally
``vega * (iv_exit - iv_entry) * sign``. If we had a perfect predictor of
the sign of ``(iv_exit - iv_entry)``, we would capture 100% of the
possible vega PnL. That's the IV oracle. Comparing the model's vega PnL
to the oracle's vega PnL gives the **capture ratio** — how much of the
theoretical ceiling the production model is achieving.

**Interpretation gate** (per the improvement plan):
- Capture ratio ≥ 60%: near ceiling; pivot effort to execution friction
  (Tier 3 RND circuit breaker, SOR realism, etc.)
- Capture ratio 30-60%: healthy room for signal improvement (Tier 2 SHAP
  prune + routing is worth the effort)
- Capture ratio < 30%: either big headroom OR a leakage red flag
  (a very low model capture on a finite-Sharpe backtest usually means
  the model is beating the oracle in some other way, e.g., via the
  delta hedge path or the sizing regime-filter)

Runs via::

    PYTHONPATH=. python scripts/forensic/10_iv_oracle.py
"""

from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from vol_arb.backtester import BacktestConfig, VolArbBacktester

logger = logging.getLogger("iv_oracle")


class IVOracleBacktester(VolArbBacktester):
    """Backtester whose signal is the perfect sign of future IV movement.

    Overrides ``_prepare`` to:
    1. Run the normal load + spread + align_exits flow (producing iv_exit)
    2. Replace ``signal`` with ``sign(iv_exit - iv_used)`` — the direction
       a perfect IV oracle would have picked

    All sizing, fees, slippage, and DRL hedging logic remain unchanged.
    """

    def _prepare(self) -> pd.DataFrame:
        df = self.load_data()

        spread = self.compute_spread(df["jump_fraction"].values)
        df["spread_iv"] = spread

        adjusted_iv = df["iv_used"].values - df["vrp_baseline"].values
        edge = df["rv_pred_pct"].values - adjusted_iv
        df["adjusted_iv"] = adjusted_iv
        df["edge"] = edge
        # Model signal (discarded but computed for consistency)
        _ = self.generate_signals(
            edge, spread / 2.0, df["jump_fraction"].values,
        )

        df = self.align_exits(df)

        # ── Oracle override: know future IV movement perfectly ──
        iv_delta = df["iv_exit"].values - df["iv_used"].values
        oracle_signal = np.sign(iv_delta).astype(np.int8)
        # Apply the same circuit breaker the normal strategy uses:
        # no shorts during high-jump regimes
        oracle_signal = np.where(
            (oracle_signal == -1)
            & (df["jump_fraction"].values > self.cfg.max_jump_fraction_for_shorts),
            0, oracle_signal,
        ).astype(np.int8)
        df["signal"] = oracle_signal

        n_long = (df["signal"] == 1).sum()
        n_short = (df["signal"] == -1).sum()
        n_flat = (df["signal"] == 0).sum()
        logger.info(
            "Oracle signals: %d long, %d short, %d flat (%.1f%% active)",
            n_long, n_short, n_flat,
            100 * (n_long + n_short) / max(1, len(df)),
        )

        return df


def run_one(
    config: BacktestConfig,
    *,
    oracle: bool,
) -> dict:
    bt = IVOracleBacktester(config) if oracle else VolArbBacktester(config)
    metrics, trade_log = bt.run_backtest()
    return {
        "metrics": metrics,
        "trade_log": trade_log,
    }


def _pnl_decomp(trade_log: pd.DataFrame, nc: bool = False) -> dict:
    """Breakdown of non-comp or compound P&L into vega/theta/hedge/slip."""
    suffix = "_nc" if nc else ""
    col = lambda c: f"{c}{suffix}"
    active = trade_log[trade_log["signal"] != 0]
    if len(active) == 0:
        return {k: 0.0 for k in ["vega", "theta", "hedge", "hedge_cost", "slippage", "net"]}
    # Component columns are PER-BTC in the trade log; multiply by N to get dollars
    N = active["N"].values if "N" in active.columns else 1.0
    out = {}
    for key in ["vega_pnl", "theta_pnl", "delta_hedge_pnl", "hedge_costs", "slippage"]:
        c = col(key)
        if c in active.columns:
            out[key] = float((active[c].values * N).sum())
        else:
            out[key] = 0.0
    out["net"] = float(active[col("net_pnl")].sum()) if col("net_pnl") in active.columns else 0.0
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="IV-movement oracle ceiling test")
    parser.add_argument("--threshold", type=float, default=25.0)
    parser.add_argument(
        "--vol-arb-path", type=str,
        default="vol_arb/output/BTC_vol_arb.parquet",
    )
    parser.add_argument(
        "--predictions-path", type=str,
        default="vol_model/output/BTCUSDT_oof_predictions.parquet",
    )
    parser.add_argument("--verbose", action="store_true")
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

    print()
    print("=" * 86)
    print("  IV-MOVEMENT ORACLE CEILING TEST")
    print("=" * 86)
    print(f"  vol_arb_path:        {cfg.vol_arb_path}")
    print(f"  predictions_path:    {cfg.predictions_path}")
    print(f"  entry_threshold:     {cfg.entry_threshold}")
    print("-" * 86)

    print("\n[model] running production backtest...")
    model = run_one(cfg, oracle=False)
    m_nc = model["metrics"].get("non_compounding", {})
    print(
        f"[model] trades={model['metrics']['n_trades']:,}, "
        f"non-comp PnL=${m_nc.get('total_pnl', 0):,.0f}, "
        f"non-comp Sharpe (backtester)={m_nc.get('sharpe_ratio', 0):.3f}",
    )

    # For the oracle, we want to use NO entry threshold — a perfect sign
    # predictor should trade whenever iv_exit != iv_entry. But keeping the
    # threshold gives a fair comparison at the same "edge filter level".
    # Run both: threshold=25 (apples-to-apples) AND threshold=0 (true ceiling).
    print("\n[oracle, threshold=25] running IV-movement oracle...")
    oracle25 = run_one(cfg, oracle=True)
    o25_nc = oracle25["metrics"].get("non_compounding", {})
    print(
        f"[oracle t=25] trades={oracle25['metrics']['n_trades']:,}, "
        f"non-comp PnL=${o25_nc.get('total_pnl', 0):,.0f}, "
        f"non-comp Sharpe (backtester)={o25_nc.get('sharpe_ratio', 0):.3f}",
    )

    print("\n[oracle, threshold=0] running IV-movement oracle (no edge filter)...")
    cfg0 = BacktestConfig(**{**cfg.__dict__, "entry_threshold": 0.0})
    oracle0 = run_one(cfg0, oracle=True)
    o0_nc = oracle0["metrics"].get("non_compounding", {})
    print(
        f"[oracle t=0]  trades={oracle0['metrics']['n_trades']:,}, "
        f"non-comp PnL=${o0_nc.get('total_pnl', 0):,.0f}, "
        f"non-comp Sharpe (backtester)={o0_nc.get('sharpe_ratio', 0):.3f}",
    )

    # ── P&L decomposition comparison (non-comp) ──
    m_dec = _pnl_decomp(model["trade_log"], nc=True)
    o25_dec = _pnl_decomp(oracle25["trade_log"], nc=True)
    o0_dec = _pnl_decomp(oracle0["trade_log"], nc=True)

    print()
    print("=" * 86)
    print("  P&L DECOMPOSITION (non-compound $)")
    print("=" * 86)
    print(f"  {'Component':<20s}  {'Model':>15s}  {'Oracle t=25':>15s}  {'Oracle t=0':>15s}")
    print("-" * 86)
    for key, label in [
        ("vega_pnl", "Vega PnL"),
        ("theta_pnl", "Theta PnL"),
        ("delta_hedge_pnl", "Delta Hedge PnL"),
        ("hedge_costs", "Hedge Costs"),
        ("slippage", "Slippage"),
        ("net", "Net PnL"),
    ]:
        print(
            f"  {label:<20s}  "
            f"${m_dec.get(key, 0):>14,.0f}  "
            f"${o25_dec.get(key, 0):>14,.0f}  "
            f"${o0_dec.get(key, 0):>14,.0f}"
        )
    print("=" * 86)

    # ── Capture ratio ──
    model_vega = m_dec.get("vega_pnl", 0.0)
    oracle25_vega = o25_dec.get("vega_pnl", 0.0)
    oracle0_vega = o0_dec.get("vega_pnl", 0.0)

    if oracle0_vega > 0:
        vega_capture_vs_0 = 100.0 * model_vega / oracle0_vega
    else:
        vega_capture_vs_0 = 0.0
    if oracle25_vega > 0:
        vega_capture_vs_25 = 100.0 * model_vega / oracle25_vega
    else:
        vega_capture_vs_25 = 0.0

    model_net = m_dec.get("net", 0.0)
    oracle0_net = o0_dec.get("net", 0.0)
    oracle25_net = o25_dec.get("net", 0.0)
    if oracle0_net > 0:
        net_capture_vs_0 = 100.0 * model_net / oracle0_net
    else:
        net_capture_vs_0 = 0.0
    if oracle25_net > 0:
        net_capture_vs_25 = 100.0 * model_net / oracle25_net
    else:
        net_capture_vs_25 = 0.0

    print()
    print("  CAPTURE RATIOS (model / oracle, non-compound $)")
    print("-" * 86)
    print(f"  Vega PnL vs oracle t=25:   {vega_capture_vs_25:>7.2f}%")
    print(f"  Vega PnL vs oracle t=0 :   {vega_capture_vs_0:>7.2f}%")
    print(f"  Net PnL  vs oracle t=25:   {net_capture_vs_25:>7.2f}%")
    print(f"  Net PnL  vs oracle t=0 :   {net_capture_vs_0:>7.2f}%")
    print("=" * 86)
    print()

    # Decision guidance
    capture = net_capture_vs_25
    if capture >= 60:
        verdict = "NEAR CEILING — pivot effort to execution friction (Tier 3)"
    elif capture >= 30:
        verdict = "HEALTHY HEADROOM — Tier 2 signal work is worth the effort"
    elif capture >= 0:
        verdict = (
            "LARGE HEADROOM — strong motivation for Tier 2 signal improvement; "
            "sanity-check by confirming T1.1 shuffle + T1.2 bootstrap are green"
        )
    else:
        verdict = (
            "ANOMALY — model beats oracle, indicates a leakage red flag or an "
            "unusual interaction (e.g., model makes money via delta hedge path "
            "while oracle focuses purely on vega direction)"
        )
    print(f"  VERDICT: {verdict}")
    print()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
