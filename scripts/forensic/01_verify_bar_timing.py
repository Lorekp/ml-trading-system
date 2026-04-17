"""Forensic test 1: prove the straddle composite lookahead.

Picks bar 2025-01-15 12:00:00 UTC and shows the actual trade timestamps
that built the straddle at that bar. If ANY trade timestamp is > 12:00,
the backtester is using future information.
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from vol_arb.options_loader import load_option_trades

TARGET = pd.Timestamp("2025-01-15 12:00:00", tz="UTC")
BUCKET_FREQ = "15min"


def main() -> None:
    # Load 1 day of trades
    trades = load_option_trades(
        "data/deribit/trades/BTC_option",
        "2025-01-15", "2025-01-15",
    )
    trades["bucket"] = trades["timestamp"].dt.floor(BUCKET_FREQ)

    # Find the bucket at 12:00
    bucket_trades = trades[trades["bucket"] == TARGET]
    print(f"Bar timestamp: {TARGET}")
    print(f"Bucket-matching trades: {len(bucket_trades):,}")
    print()
    print(f"Trade timestamp range within this bucket:")
    print(f"  Min: {bucket_trades['timestamp'].min()}")
    print(f"  Max: {bucket_trades['timestamp'].max()}")
    print()

    # How many of those trades happened AT or AFTER the bar timestamp?
    at_or_after = (bucket_trades["timestamp"] >= TARGET).sum()
    strictly_after = (bucket_trades["timestamp"] > TARGET).sum()
    print(f"Trades at timestamp >= bar time ({TARGET}): {at_or_after:,} / {len(bucket_trades):,}")
    print(f"Trades at timestamp  > bar time ({TARGET}): {strictly_after:,} / {len(bucket_trades):,}")

    # The lookahead diagnosis
    if strictly_after > 0:
        latest = bucket_trades["timestamp"].max()
        minutes_forward = (latest - TARGET).total_seconds() / 60.0
        print()
        print("=" * 70)
        print("LOOKAHEAD CONFIRMED")
        print("=" * 70)
        print(f"The bar at {TARGET} (START-stamped) is built from option trades")
        print(f"that extend {minutes_forward:.1f} minutes into the future.")
        print(f"The latest trade in this bucket happened at {latest}.")
        print()
        print("When merge_asof(backward) joins this bucket onto a feature at 12:00,")
        print(f"the feature sees option data constructed from trades up to {latest},")
        print(f"which is impossible at wall time 12:00.")
    else:
        print("No future trades in this bucket — no lookahead for THIS specific bar.")

    # Also check: what's the existing vol_arb parquet say for this bar?
    print()
    print("-" * 70)
    print("What the existing vol_arb parquet has at bar 12:00:")
    print("-" * 70)
    df = pd.read_parquet("vol_arb/output/BTC_vol_arb.parquet")
    row = df.loc[TARGET]
    for col in ["iv", "strike", "vega", "theta", "moneyness", "dte_days", "T_years"]:
        if col in row.index:
            val = row[col]
            print(f"  {col:15s} {val}")

    # And the SVI variant
    print()
    print("-" * 70)
    print("What BTC_vol_arb_svi.parquet has at bar 12:00:")
    print("-" * 70)
    df_svi = pd.read_parquet("vol_arb/output/BTC_vol_arb_svi.parquet")
    if TARGET in df_svi.index:
        row_svi = df_svi.loc[TARGET]
        for col in ["iv", "svi_iv_matched", "strike", "expiry_dt", "index_price"]:
            if col in row_svi.index:
                val = row_svi[col]
                print(f"  {col:20s} {val}")


if __name__ == "__main__":
    main()
