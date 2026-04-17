"""Forensic test 2: shift straddle-derived columns by +15min and rerun backtest.

Straddle composite is START-stamped (bucket=12:00 uses trades from [12:00, 12:15)).
merge_asof(backward) at feature time 12:00 returns this bucket, leaking the full
15 minutes of future option data.

The fix: shift all straddle-derived columns forward by 15 minutes via index
arithmetic. After shifting, bar 12:15 holds what was previously labeled bar 12:00,
so the bar at 12:15 sees option data from trades in [12:00, 12:15) — all in the
past relative to 12:15.
"""

from __future__ import annotations

import logging
import pandas as pd
import numpy as np

from vol_arb.backtester import BacktestConfig, VolArbBacktester

logging.basicConfig(level=logging.WARNING)

# Columns that come from the straddle composite and need the shift
STRADDLE_COLS = [
    "iv", "strike", "index_price", "mark_price", "moneyness",
    "dte_days", "T_years", "delta", "gamma", "vega", "theta",
    "expiry_dt",
    # bear/bull strangle columns
    "iv_bear", "strike_bear", "mark_price_bear", "moneyness_bear",
    "delta_bear", "gamma_bear", "vega_bear", "theta_bear",
    "iv_bull", "strike_bull", "mark_price_bull", "moneyness_bull",
    "delta_bull", "gamma_bull", "vega_bull", "theta_bull",
    # svi_iv_matched depends on strike/expiry_dt from the straddle
    "svi_iv_matched",
]


def shift_straddle_cols(df: pd.DataFrame, periods: int = 1) -> pd.DataFrame:
    """Shift straddle-derived columns by `periods` bars (15min each)."""
    out = df.copy()
    cols = [c for c in STRADDLE_COLS if c in out.columns]
    out[cols] = out[cols].shift(periods)
    return out


def run_backtests(vol_arb_path: str, label: str, pred_path: str) -> None:
    print(f"\n{label}")
    print(f'{"thresh":>7s}  {"trades":>8s}  {"PnL":>13s}  {"Sharpe":>8s}  {"WinRate":>8s}  {"MaxDD":>8s}')
    print("-" * 70)
    for thresh in [0.5, 5.0, 10.0, 20.0]:
        cfg = BacktestConfig(
            vol_arb_path=vol_arb_path,
            predictions_path=pred_path,
            entry_threshold=thresh,
        )
        bt = VolArbBacktester(cfg)
        m, _ = bt.run_backtest()
        print(
            f'{thresh:>7.1f}  {m["n_trades"]:>8,d}  '
            f'${m["total_pnl"]:>12,.0f}  {m["sharpe_ratio"]:>8.2f}  '
            f'{m["win_rate_pct"]:>7.1f}%  {m["max_drawdown_pct"]:>7.1f}%'
        )


def main() -> None:
    pred_path = "vol_model/output/BTCUSDT_hybrid_predictions.parquet"

    # === Baseline (raw IV) ===
    print("=" * 70)
    print("BASELINE — Raw IV (original lookahead)")
    print("=" * 70)
    run_backtests(
        "vol_arb/output/BTC_vol_arb.parquet",
        "Original (leaky)",
        pred_path,
    )

    # Shift the baseline parquet
    df = pd.read_parquet("vol_arb/output/BTC_vol_arb.parquet")
    df_fixed = shift_straddle_cols(df, periods=1)
    # Drop the first bar (NaN after shift)
    df_fixed = df_fixed.dropna(subset=["iv"])
    out_path = "vol_arb/output/BTC_vol_arb_shifted.parquet"
    df_fixed.to_parquet(out_path)
    run_backtests(out_path, "Shifted +15min (fixed)", pred_path)

    # === SVI variant ===
    print()
    print("=" * 70)
    print("SVI — Maturity-Matched IV")
    print("=" * 70)
    run_backtests(
        "vol_arb/output/BTC_vol_arb_svi.parquet",
        "Original (leaky)",
        pred_path,
    )

    df_svi = pd.read_parquet("vol_arb/output/BTC_vol_arb_svi.parquet")
    df_svi_fixed = shift_straddle_cols(df_svi, periods=1)
    df_svi_fixed = df_svi_fixed.dropna(subset=["iv"])
    out_svi_path = "vol_arb/output/BTC_vol_arb_svi_shifted.parquet"
    df_svi_fixed.to_parquet(out_svi_path)
    run_backtests(out_svi_path, "Shifted +15min (fixed)", pred_path)


if __name__ == "__main__":
    main()
