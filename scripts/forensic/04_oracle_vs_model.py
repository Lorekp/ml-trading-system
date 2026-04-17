"""Forensic test 4: Oracle vs model comparison.

Replace y_pred (the model's forecast) with y_true (the actual next-period
TSRV). If the model's PnL is close to or exceeds the oracle's PnL, there's
definitive leakage: the model is "better than perfect" which is impossible.

Uses the shifted (leakage-fixed) datasets so we're measuring pure prediction
quality, not timing leakage.
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np

from vol_arb.backtester import BacktestConfig, VolArbBacktester

logging.basicConfig(level=logging.WARNING)


def make_oracle_predictions(vol_arb_path: str) -> str:
    """Create an oracle predictions file: y_pred = y_true (perfect forecast)."""
    df = pd.read_parquet(vol_arb_path)
    # vol_ann in the merged file is TSRV vol_ann (annualized, decimal)
    # y_true in the OOF file is ln_vol = log(vol_ann)
    # For oracle, set y_pred = log(vol_ann)
    vol_ann = df["vol_ann"].clip(lower=1e-8)
    ln_vol = np.log(vol_ann)

    oracle = pd.DataFrame({
        "y_true": ln_vol.values,
        "y_pred": ln_vol.values,
    }, index=df.index)
    oracle_path = "vol_model/output/BTCUSDT_oracle_predictions.parquet"
    oracle.to_parquet(oracle_path)
    return oracle_path


def run_backtest(vol_arb_path: str, pred_path: str, label: str) -> None:
    print(f"\n{label}")
    print(f'{"thresh":>7s}  {"trades":>8s}  {"PnL":>14s}  {"Sharpe":>8s}')
    print("-" * 60)
    for thresh in [5.0, 10.0, 20.0, 50.0]:
        cfg = BacktestConfig(
            vol_arb_path=vol_arb_path,
            predictions_path=pred_path,
            entry_threshold=thresh,
        )
        bt = VolArbBacktester(cfg)
        m, _ = bt.run_backtest()
        print(
            f'{thresh:>7.1f}  {m["n_trades"]:>8,d}  '
            f'${m["total_pnl"]:>13,.0f}  {m["sharpe_ratio"]:>8.2f}'
        )


def main() -> None:
    hybrid_preds = "vol_model/output/BTCUSDT_hybrid_predictions.parquet"

    # Oracle uses the shifted vol_arb data (so it's measuring prediction value, not timing)
    oracle_path = make_oracle_predictions("vol_arb/output/BTC_vol_arb_svi_shifted.parquet")
    print(f"Oracle predictions saved to {oracle_path}")

    print("=" * 70)
    print("SVI IV (shifted/fixed) — Model vs Oracle")
    print("=" * 70)
    run_backtest(
        "vol_arb/output/BTC_vol_arb_svi_shifted.parquet",
        hybrid_preds,
        "Hybrid model (trained on 2021-2026)",
    )
    run_backtest(
        "vol_arb/output/BTC_vol_arb_svi_shifted.parquet",
        oracle_path,
        "Oracle (y_pred = y_true, perfect forecast)",
    )

    print()
    print("=" * 70)
    print("Baseline IV (shifted/fixed) — Model vs Oracle")
    print("=" * 70)
    run_backtest(
        "vol_arb/output/BTC_vol_arb_shifted.parquet",
        hybrid_preds,
        "Hybrid model",
    )
    run_backtest(
        "vol_arb/output/BTC_vol_arb_shifted.parquet",
        oracle_path,
        "Oracle",
    )


if __name__ == "__main__":
    main()
