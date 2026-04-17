"""Calibrate the continuous sqrt market impact model from Deribit ATM option trades.

Reads all daily option trade parquets, filters to ATM (moneyness 0.93-1.07),
computes per-trade absolute USD slippage, then fits a rolling 30-day
sqrt model:

    slip_usd = alpha + beta * sqrt(amount_btc)

Outputs a daily parameter series to vol_arb/output/impact_model_params.parquet.

Usage:
    PYTHONPATH=. python scripts/calibrate_impact.py
    PYTHONPATH=. python scripts/calibrate_impact.py --validate
    PYTHONPATH=. python scripts/calibrate_impact.py --quantile 0.75  # conservative
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
logger = logging.getLogger("calibrate_impact")

TRADE_DIR = Path("data/deribit/trades/BTC_option")
OUTPUT_PATH = Path("vol_arb/output/impact_model_params.parquet")
MONEYNESS_LO = 0.93
MONEYNESS_HI = 1.07


def load_all_atm_trades(quantile: float = 0.5) -> pd.DataFrame:
    """Load all ATM option trades with slippage computed."""
    files = sorted(TRADE_DIR.glob("*.parquet"))
    logger.info("Loading %d daily trade files...", len(files))

    all_rows = []
    for i, f in enumerate(files):
        df = pd.read_parquet(f)
        if len(df) < 5:
            continue

        # Filter to valid trades with mark price
        df = df[(df["mark_price"] > 0) & (df["index_price"] > 0) & (df["amount"] > 0)]
        if df.empty:
            continue

        # Parse strike from instrument name for moneyness filter
        parts = df["instrument_name"].str.split("-")
        strikes = parts.str[2].apply(pd.to_numeric, errors="coerce")
        moneyness = strikes / df["index_price"]
        mask = (moneyness >= MONEYNESS_LO) & (moneyness <= MONEYNESS_HI)
        df = df[mask]
        if df.empty:
            continue

        df = df[["timestamp", "price", "mark_price", "index_price", "amount"]].copy()
        df["slip_usd"] = (df["price"] - df["mark_price"]).abs() * df["index_price"]
        df["date"] = pd.to_datetime(df["timestamp"], unit="ms").dt.normalize()
        df["sqrt_amount"] = np.sqrt(df["amount"])
        all_rows.append(df[["date", "amount", "sqrt_amount", "slip_usd"]])

        if (i + 1) % 200 == 0:
            logger.info("  [%d/%d] loaded...", i + 1, len(files))

    result = pd.concat(all_rows, ignore_index=True)
    logger.info("Total ATM trades: %d", len(result))
    return result


def fit_rolling_sqrt_model(
    trades: pd.DataFrame, window_days: int = 30, quantile: float = 0.5,
) -> pd.DataFrame:
    """Fit alpha + beta * sqrt(amount) per rolling window.

    Uses quantile regression (percentile of slip at each size) rather
    than OLS to be robust to outliers. The quantile parameter controls
    conservatism: 0.5 = median, 0.75 = conservative.
    """
    dates = sorted(trades["date"].unique())
    logger.info("Fitting sqrt model over %d days (window=%d, quantile=%.2f)",
                len(dates), window_days, quantile)

    rows = []
    for i, d in enumerate(dates):
        # Trailing window
        window_start = d - pd.Timedelta(days=window_days)
        mask = (trades["date"] > window_start) & (trades["date"] <= d)
        window = trades[mask]

        if len(window) < 50:
            continue

        # Daily volume (trailing)
        daily_vol = window.groupby("date")["amount"].sum()
        trailing_vol_1d = float(daily_vol.mean()) if len(daily_vol) > 0 else 0.0

        # Two-stage calibration:
        # 1. alpha = base spread from small trades (< 0.5 BTC)
        # 2. beta = size-dependent impact from larger trades
        small = window[window["amount"] < 0.5]
        large = window[window["amount"] >= 0.5]

        if len(small) >= 10:
            alpha = float(small["slip_usd"].quantile(quantile))
        else:
            alpha = float(window["slip_usd"].quantile(quantile))

        if len(large) >= 20:
            # Fit beta on residual: (slip - alpha) vs sqrt(amount)
            residual = large["slip_usd"].values - alpha
            x = large["sqrt_amount"].values
            # OLS without intercept: residual = beta * sqrt(amount)
            beta = float(np.dot(x, residual) / np.dot(x, x))
            beta = max(beta, 0.0)
        else:
            beta = 0.0

        rows.append({
            "timestamp": d,
            "alpha": alpha,
            "beta": beta,
            "trailing_volume_1d": trailing_vol_1d,
            "n_trades_window": len(window),
        })

        if (i + 1) % 200 == 0:
            logger.info("  [%d/%d] days fitted", i + 1, len(dates))

    result = pd.DataFrame(rows)
    result["timestamp"] = pd.to_datetime(result["timestamp"], utc=True)
    logger.info("Fitted %d daily parameter rows", len(result))
    return result


def validate(params: pd.DataFrame, trades: pd.DataFrame) -> None:
    """Compare model predictions vs actual slippage at various sizes."""
    # Use last 30 days of params for current-state validation
    latest = params.iloc[-1]
    alpha, beta = latest["alpha"], latest["beta"]

    test_sizes = [0.1, 0.5, 1.0, 3.0, 5.0, 10.0, 20.0, 50.0]

    print()
    print("=" * 75)
    print("  IMPACT MODEL VALIDATION")
    print(f"  Model: slip = {alpha:.2f} + {beta:.2f} * sqrt(N)")
    print("=" * 75)
    print(f"{'Size (BTC)':<12} {'Predicted':>12} {'Actual Med':>12} {'Actual p75':>12} {'Error':>10}")
    print("-" * 75)

    # Use last 90 days of trades for validation
    recent_start = trades["date"].max() - pd.Timedelta(days=90)
    recent = trades[trades["date"] > recent_start]

    for size in test_sizes:
        predicted = alpha + beta * np.sqrt(size)

        lo = size * 0.7
        hi = size * 1.3
        bucket = recent[(recent["amount"] >= lo) & (recent["amount"] < hi)]

        if len(bucket) < 10:
            print(f"{size:<12.1f} ${predicted:>11.1f} {'N/A':>12} {'N/A':>12} {'N/A':>10}")
            continue

        actual_med = float(bucket["slip_usd"].median())
        actual_p75 = float(bucket["slip_usd"].quantile(0.75))
        error_pct = (predicted - actual_med) / max(actual_med, 0.01) * 100

        print(f"{size:<12.1f} ${predicted:>11.1f} ${actual_med:>11.1f} ${actual_p75:>11.1f} {error_pct:>9.1f}%")

    print("=" * 75)
    print(f"\nStraddle cost examples (2 legs, no jump multiplier):")
    for n in [0.5, 1, 3, 5, 10, 20]:
        per_btc = alpha + beta * np.sqrt(n)
        total = 2 * per_btc * n
        print(f"  {n:>5.1f} BTC straddle: ${per_btc:.0f}/BTC x {n} x 2 legs = ${total:,.0f} total")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate sqrt impact model")
    parser.add_argument("--quantile", type=float, default=0.5,
                        help="Slippage quantile (0.5=median, 0.75=conservative)")
    parser.add_argument("--window", type=int, default=30,
                        help="Rolling window in days (default 30)")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation after calibration")
    parser.add_argument("--output", type=str, default=str(OUTPUT_PATH))
    args = parser.parse_args()

    trades = load_all_atm_trades(quantile=args.quantile)
    params = fit_rolling_sqrt_model(trades, window_days=args.window, quantile=args.quantile)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    params.to_parquet(output_path)
    logger.info("Saved to %s (%d rows)", output_path, len(params))

    if args.validate:
        validate(params, trades)


if __name__ == "__main__":
    main()
