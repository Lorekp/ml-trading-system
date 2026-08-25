"""Realistic-eval CLI smoke + headline.

Usage:
  .venv/Scripts/python.exe scripts/run_realistic_eval.py \
    --trade-log vol_arb/output/BTC_v2_trade_log.parquet \
    --start 2024-01-01 --end 2024-01-08 --fill-tol-minutes 60 --report
"""
from __future__ import annotations

import argparse
import logging

import numpy as np
import pandas as pd

from vol_arb.realistic_eval import RealisticEvalConfig, evaluate_realistic, assert_brackets


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trade-log", default="vol_arb/output/BTC_v2_trade_log.parquet")
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--fill-tol-minutes", type=float, default=15.0)
    ap.add_argument("--spread-source", default="tape", choices=["tape", "synth"])
    ap.add_argument("--report", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.WARNING, format="%(message)s")

    tl = pd.read_parquet(args.trade_log)
    cfg = RealisticEvalConfig(
        start=args.start, end=args.end,
        fill_tol_minutes=args.fill_tol_minutes,
        spread_source=args.spread_source,
    )
    result = evaluate_realistic(tl, cfg)

    print("=" * 72)
    print(result.summary())
    print("=" * 72)

    if args.report:
        f = result.trade_log
        cov = f[f["premium_tier"] > 0]
        print(f"\nPER-TRADE LEG DECOMPOSITION (first {min(len(cov), 10)} covered):")
        show = cov[[
            "signal", "premium_in_btc", "premium_out_btc", "effective_half_spread",
            "option_premium_usd", "option_fee_usd", "gamma_pnl_usd", "hedge_fee_usd",
            "funding_usd", "total_unhedged_usd", "total_hedged_usd",
            "n_rebalances", "n_path_bars", "n_funding_stamps", "fill_tier",
        ]].head(10)
        with pd.option_context("display.width", 220, "display.max_columns", 30,
                               "display.float_format", lambda x: f"{x:,.4f}"):
            print(show.to_string())

        bracket = assert_brackets(tl, result, cfg)
        print("\nBRACKET SELF-CHECK")
        print(f"  optimistic (legacy net_pnl_nc, covered) = ${bracket.optimistic_usd:,.0f}")
        print(f"  pessimistic (un-hedged real fills)       = ${bracket.pessimistic_usd:,.0f}")
        print(f"  realistic (hedged, real path, full cost) = ${bracket.realistic_usd:,.0f}")
        print(f"  within_bracket                = {bracket.within_bracket}")
        print(f"  bracket_high_is_failure       = {bracket.bracket_high_is_failure}")
        print(f"  zero_fee_implied_gamma_usd    = ${bracket.zero_fee_implied_gamma_usd:,.2f}  (limit test, ~0)")
        print(f"  limit_test_passed             = {bracket.limit_test_passed}")
        print(f"  degenerate_equals_pessimistic = {bracket.degenerate_equals_pessimistic}")
        print(f"  closed_form_gamma_max_relerr  = {bracket.closed_form_gamma_max_relerr:.2e}")
        print(f"  PASSED                        = {bracket.passed}")


if __name__ == "__main__":
    main()
