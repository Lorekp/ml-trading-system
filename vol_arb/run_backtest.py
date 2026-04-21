"""CLI entry point for the vol-arb execution simulator.

Usage:
    PYTHONPATH=. python vol_arb/run_backtest.py
    PYTHONPATH=. python vol_arb/run_backtest.py --base-spread 2.0 --threshold 1.0
    PYTHONPATH=. python vol_arb/run_backtest.py --risk 0.02 --iv-shock 3.0
"""

from __future__ import annotations

import argparse
import logging

from vol_arb.backtester import BacktestConfig, VolArbBacktester, print_tearsheet
from vol_arb.metrics import build_nc_view, print_extended_tearsheet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="Vol-arb execution simulator")
    parser.add_argument("--currency", type=str, default="BTC",
                        help="Underlying currency (BTC|ETH). Selects per-currency "
                             "paths + vol-calibrated defaults via "
                             "BacktestConfig.for_currency().")
    parser.add_argument("--base-spread", type=float, default=1.5,
                        help="Base IV spread in vol points")
    parser.add_argument("--spread-k", type=float, default=3.0,
                        help="Jump sensitivity for spread widening")
    parser.add_argument("--threshold", type=float, default=25.0,
                        help="Entry threshold beyond half-spread (vol points)")
    parser.add_argument("--equity", type=float, default=100_000,
                        help="Initial equity in USD")
    parser.add_argument("--margin-util", type=float, default=0.05,
                        help="Target margin utilization (fraction of equity)")
    parser.add_argument("--max-participation", type=float, default=0.10,
                        help="Max participation rate of trailing 1h ATM volume")
    parser.add_argument("--short-penalty", type=float, default=0.50,
                        help="Short straddle sizing multiplier (tail risk)")
    parser.add_argument("--baseline-dvol", type=float, default=None,
                        help="DVOL at which full margin is deployed "
                             "(default: BTC 50.0, ETH 75.0 via --currency)")
    parser.add_argument("--max-jump-short", type=float, default=None,
                        help="Max jump_fraction for short signals "
                             "(default: BTC 0.15, ETH 0.20 via --currency)")
    parser.add_argument("--adverse-sel", type=float, default=0.5,
                        help="Maker adverse selection penalty (vol points)")
    parser.add_argument("--perp-slippage", type=float, default=2.0,
                        help="Perp hedge slippage (bps)")
    parser.add_argument("--option-maker-fee", type=float, default=0.0003,
                        help="Option fee (fraction of underlying; Deribit 0.03%%)")
    parser.add_argument("--perp-fee", type=float, default=0.0005,
                        help="Perp taker fee (fraction of underlying; Deribit 0.05%%)")
    parser.add_argument("--vrp", action="store_true",
                        help="Enable VRP baseline adjustment (off by default)")
    parser.add_argument("--no-sor", action="store_true",
                        help="Disable deterministic SOR (always use taker slippage)")
    parser.add_argument("--use-routing", action="store_true",
                        help="Enable dynamic instrument routing (ATM / bear strangle / bull strangle). Off by default — ATM straddle outperforms strangles in the current setup (see Stage 8.4).")
    parser.add_argument("--skew-threshold", type=float, default=0.05,
                        help="Jump skew threshold for instrument routing (default 0.05)")
    parser.add_argument("--sweep-bands", type=str, default="",
                        help="Comma-separated hedge band values to sweep (e.g. 0.02,0.05,0.10)")
    parser.add_argument("--output", type=str, default="",
                        help="Save trade log to parquet")
    parser.add_argument("--extended", action="store_true",
                        help="Print full pro-grade metrics tearsheet")
    parser.add_argument("--vol-arb-path", type=str, default="",
                        help="Override vol_arb_path (e.g., SVI-enabled dataset)")
    parser.add_argument("--predictions-path", type=str, default="",
                        help="Override predictions_path")
    parser.add_argument("--use-drl-hedging", action="store_true",
                        help="Use the DRL-trained SAC hedger instead of the rule-based band")
    parser.add_argument("--drl-policy", type=str, default="",
                        help="Override drl_policy_path (path to trained SAC .zip)")
    parser.add_argument("--classifier-path", type=str, default="",
                        help="Optional classifier head predictions parquet (W4.1)")
    parser.add_argument("--classifier-threshold", type=float, default=0.55,
                        help="Classifier confidence threshold, must be in (0.5, 1.0)")
    parser.add_argument("--realistic", action="store_true",
                        help="Apply realistic Deribit 2026 cost widening "
                             "(perp_slippage_bps 2->3, base_spread_iv 1.5->2.0, "
                             "maker_adverse_selection_iv 0.5->0.75). "
                             "Use for honest production Sharpe.")
    parser.add_argument("--use-pinned-exit-iv", action="store_true",
                        help="Switch from default rolling-ATM attribution "
                             "(documented in PIPELINE_DOCUMENTATION §17.3 as "
                             "'exact for ATM straddles') to fixed-contract "
                             "attribution where exit IV is reconstructed for "
                             "the entry's exact (K, D) via real same-contract "
                             "trade -> SVI surface -> legacy fallback. Use "
                             "only when explicitly modeling the alternative "
                             "fixed-contract interpretation.")
    parser.add_argument("--min-dte-days", type=float, default=None,
                        help="Mute signals on bars where option DTE < this "
                             "floor (0 = disabled, default). Motivated by "
                             "empirical 5/5-year bleed on DTE<5 trades "
                             "(flat fees but vega ∝ √DTE). Production "
                             "value: 5.0 for BTC.")
    parser.add_argument("--skip-long-hours-utc", type=str, default=None,
                        help="Comma-separated UTC hours at which LONG "
                             "signals are muted (shorts kept). Motivated "
                             "by US cash-open vol dislocation. Production "
                             "value: '14,15,16' for BTC. Empty = disabled.")
    parser.add_argument("--btc-production", action="store_true",
                        help="Apply the full BTC production filter/threshold "
                             "stack (with_btc_production_filters): "
                             "thr_S=22, thr_L=30, min_dte_days=5, "
                             "skip_long_hours_utc=(14,15,16), "
                             "target_margin_utilization=0.075, "
                             "use_sor=False. Normally combined with "
                             "--realistic.")
    parser.add_argument("--btc-production-v2", action="store_true",
                        help="Apply the v2 production stack "
                             "(with_btc_production_filters_v2): v1 "
                             "+ DRL hedging + profit_scaling(floor=0.3) "
                             "+ util=0.15 + holding=30m + kelly sizing "
                             "+ skip_long_if_term_slope_gt=0.0. "
                             "Normally combined with --realistic.")
    parser.add_argument("--dfl-scores-path", type=str, default="",
                        help="Enable DFL head filter. Path to walk-forward "
                             "P(win) scores parquet (produced by "
                             "vol_arb.dfl_head.compute_walkforward_scores). "
                             "Bars with P(win) < train-year quantile are muted.")
    parser.add_argument("--dfl-filter-quantile", type=float, default=0.25,
                        help="Quantile threshold for DFL filter (default 0.25 "
                             "= drop bottom 25%% of P(win) based on prior-year "
                             "empirical distribution). Only applied if "
                             "--dfl-scores-path is also set.")
    parser.add_argument("--kelly-mult-path", type=str, default="",
                        help="Enable long-side Kelly-shrinkage sizing boost. "
                             "Path to walk-forward per-bar size_mult parquet "
                             "(produced by scripts/generate_kelly_sidecar.py). "
                             "Multiplier >= 1.0 (upside only) applies AFTER "
                             "profit_scaling and BEFORE dynamic_btc_cap clip.")
    parser.add_argument("--horizon-minutes", type=int, default=15,
                        help="Prediction horizon in minutes (default 15). When "
                             "combined with --btc-production-v2 selects the "
                             "matching per-horizon V2 preset "
                             "(`with_btc_production_h{N}m()`). Sidecars at "
                             "h!=15 must already be regenerated; missing files "
                             "surface as a backtester load error rather than "
                             "silent fallback. Must divide 1440 (e.g., 5, 15, "
                             "30, 60). See plan starry-rolling-rabin.md.")
    args = parser.parse_args()
    if 1440 % args.horizon_minutes != 0:
        parser.error(
            f"--horizon-minutes={args.horizon_minutes} must divide 1440; "
            f"choose 5, 15, 30, 60, etc."
        )

    config_kwargs = dict(
        base_spread_iv=args.base_spread,
        spread_jump_k=args.spread_k,
        entry_threshold=args.threshold,
        initial_equity=args.equity,
        target_margin_utilization=args.margin_util,
        max_participation_rate=args.max_participation,
        short_sizing_penalty=args.short_penalty,
        maker_adverse_selection_iv=args.adverse_sel,
        perp_slippage_bps=args.perp_slippage,
        option_maker_fee=args.option_maker_fee,
        perp_taker_fee=args.perp_fee,
        use_vrp=args.vrp,
        use_sor=not args.no_sor,
        use_instrument_routing=args.use_routing,
        jump_skew_threshold=args.skew_threshold,
    )
    # Vol-baseline knobs: only apply when the user explicitly set them, so
    # for_currency() defaults (ETH vs BTC) flow through otherwise.
    if args.baseline_dvol is not None:
        config_kwargs["baseline_target_dvol"] = args.baseline_dvol
    if args.max_jump_short is not None:
        config_kwargs["max_jump_fraction_for_shorts"] = args.max_jump_short
    if args.vol_arb_path:
        config_kwargs["vol_arb_path"] = args.vol_arb_path
    if args.predictions_path:
        config_kwargs["predictions_path"] = args.predictions_path
    if args.dfl_scores_path:
        config_kwargs["dfl_scores_path"] = args.dfl_scores_path
        config_kwargs["dfl_filter_quantile"] = args.dfl_filter_quantile
    if args.kelly_mult_path:
        config_kwargs["kelly_mult_path"] = args.kelly_mult_path
    if args.use_drl_hedging:
        config_kwargs["use_drl_hedging"] = True
    if args.drl_policy:
        config_kwargs["drl_policy_path"] = args.drl_policy
    if args.classifier_path:
        config_kwargs["classifier_path"] = args.classifier_path
        config_kwargs["classifier_threshold"] = args.classifier_threshold

    config = BacktestConfig.for_currency(args.currency, **config_kwargs)
    if args.realistic:
        config = config.with_realistic_costs()
    if args.btc_production:
        config = config.with_btc_production_filters()
    if args.btc_production_v2:
        # Per-horizon dispatch: at h=15 use the named legacy preset
        # (preserves the test_production_v2_kelly_default_is_K1 invariant
        # and is byte-identical to current production). At h!=15 use the
        # generic for-horizon factory which composes for_timeframe(N) +
        # V1 filters + V2 horizon-invariants + horizon-aware paths.
        if args.horizon_minutes == 15:
            config = config.with_btc_production_filters_v2()
        else:
            config = BacktestConfig.with_btc_production_for_horizon(
                args.horizon_minutes,
            )
    # Explicit --min-dte-days / --skip-long-hours-utc override the preset
    # so users can sweep filter-level knobs while keeping realistic costs.
    if args.min_dte_days is not None:
        config = config.model_copy(update={"min_dte_days": float(args.min_dte_days)})
    if args.skip_long_hours_utc is not None:
        if args.skip_long_hours_utc.strip() == "":
            hrs = ()
        else:
            hrs = tuple(int(h.strip()) for h in args.skip_long_hours_utc.split(","))
            for h in hrs:
                if not (0 <= h <= 23):
                    parser.error(f"--skip-long-hours-utc: hour {h} out of range [0, 23]")
        config = config.model_copy(update={"skip_long_hours_utc": hrs})
    if args.use_pinned_exit_iv:
        config = config.model_copy(update={"use_pinned_exit_iv": True})
    bt = VolArbBacktester(config)

    # ── Hedge band sweep mode ──
    if args.sweep_bands:
        bands = [float(b.strip()) for b in args.sweep_bands.split(",")]
        comparison = bt.optimize_hedge_band(bands)
        print("\n" + "=" * 80)
        print("  DELTA HEDGE BAND SWEEP")
        print("=" * 80)
        print(f"  {'Band':>6s}  {'PnL':>12s}  {'Sharpe':>8s}  {'Sortino':>8s}"
              f"  {'MaxDD%':>8s}  {'HedgePnL':>12s}  {'HedgeCost':>12s}  {'MkrFill%':>9s}")
        print("-" * 80)
        for _, row in comparison.iterrows():
            print(f"  {row['band']:>6.3f}  ${row['total_pnl']:>11,.0f}"
                  f"  {row['sharpe']:>8.2f}  {row['sortino']:>8.2f}"
                  f"  {row['max_dd_pct']:>8.2f}  ${row['delta_hedge_pnl']:>11,.0f}"
                  f"  ${row['hedge_costs']:>11,.0f}  {row['maker_fill_pct']:>8.1f}%")
        print("=" * 80 + "\n")
        return

    # ── Standard backtest ──
    metrics, trade_log = bt.run_backtest()

    print_tearsheet(metrics, config)

    has_nc = "net_pnl_nc" in trade_log.columns

    if args.extended:
        print_extended_tearsheet(trade_log, initial_equity=config.initial_equity)
        if has_nc:
            print("\n" + "#" * 72)
            print("#" + "  NON-COMPOUNDING VIEW  (fixed reference notional)".center(70) + "#")
            print("#" * 72)
            nc_view = build_nc_view(trade_log)
            print_extended_tearsheet(
                nc_view, initial_equity=config.initial_equity, is_nc=True,
            )

    # P&L decomposition summary — run for compound and (optionally) non-compound
    _print_decomposition(trade_log, label="COMPOUND")
    if has_nc:
        _print_decomposition(trade_log, label="NON-COMPOUND", nc=True)

    if args.output:
        trade_log.to_parquet(args.output)
        logger.info("Trade log saved to %s", args.output)


def _print_decomposition(trade_log, *, label: str, nc: bool = False) -> None:
    """Print the P&L decomposition block for either the compound or
    non-compound columns of the trade log.  Kept as a free function so
    both views go through identical formatting logic.
    """
    active = trade_log[trade_log["signal"] != 0]
    if len(active) == 0:
        return

    suffix = "_nc" if nc else ""
    net_col = f"net_pnl{suffix}"
    maker_col = f"maker_filled{suffix}"

    print(f"P&L Decomposition — {label} (active trades):")
    for base in ["vega_pnl", "theta_pnl", "delta_hedge_pnl", "hedge_costs", "option_fees", "slippage"]:
        col = f"{base}{suffix}"
        if col in active.columns:
            print(f"  {base:15s}  ${active[col].sum():>14,.2f}")
    if net_col in active.columns:
        print(f"  {'net_pnl':15s}  ${active[net_col].sum():>14,.2f}")
    if maker_col in active.columns:
        fill_rate = active[maker_col].mean() * 100
        print(f"\n  Maker fill rate ({label.lower()}): {fill_rate:.1f}%"
              f" ({int(active[maker_col].sum()):,d} / {len(active):,d})")
    print()

    # Signal breakdown
    longs = active[active["signal"] == 1]
    shorts = active[active["signal"] == -1]
    if net_col in active.columns:
        print(f"Long straddles:  {len(longs):,d}  PnL: ${longs[net_col].sum():>14,.2f}")
        print(f"Short straddles: {len(shorts):,d}  PnL: ${shorts[net_col].sum():>14,.2f}")

    # Instrument routing (only for compound — instrument_type isn't _nc-suffixed)
    if not nc and "instrument_type" in active.columns:
        it = active["instrument_type"]
        n_atm = (it == 0).sum()
        n_bear = (it == 1).sum()
        n_bull = (it == -1).sum()
        if n_bear > 0 or n_bull > 0:
            print(f"\n  Instrument routing:")
            print(f"    ATM Straddle:   {n_atm:>7,d} ({100*n_atm/len(active):.1f}%)"
                  f"  PnL: ${active.loc[it == 0, net_col].sum():>14,.2f}")
            print(f"    Bear Strangle:  {n_bear:>7,d} ({100*n_bear/len(active):.1f}%)"
                  f"  PnL: ${active.loc[it == 1, net_col].sum():>14,.2f}")
            print(f"    Bull Strangle:  {n_bull:>7,d} ({100*n_bull/len(active):.1f}%)"
                  f"  PnL: ${active.loc[it == -1, net_col].sum():>14,.2f}")
    print()


if __name__ == "__main__":
    main()
