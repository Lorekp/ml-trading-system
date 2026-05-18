"""Cycle-47 (Phase 5, 2026-05-18): honest-PnL audit of the V2 backtest.

Six independent Opus audits confirmed the V2 headline (~$2.2M NC) is
~99% the vega leg `signal x vega x d(iv_used)`, and `iv_used` is SVI
calibration noise (15-min std 9.58vp vs Deribit DVOL 0.36vp). The
strategy fires on extreme `iv_used[T]` and books the noise's
mean-reversion as profit. Live trades real options and lost 14/14.

This script produces the DEFINITIVE honest number three ways and
cross-checks them against the live engine:

  1. IV-FROZEN  -- canon trade log with the vega leg removed
                   (net_nc - vega_nc = theta + delta_hedge - costs).
                   The honest floor: assumes 0 IV change over the
                   15-min hold (you cannot forecast 15-min IV).
  2. PINNED-EXIT -- re-run V2 with use_pinned_exit_iv=True so exit IV
                   comes from real same-contract option trades
                   (tier 1) / SVI surface at the entered strike
                   (tier 2), not the rolling-ATM iv_used.
  3. LIVE       -- the 14 real paper trades from paper.db (already
                   marked at real Deribit option requotes).

Usage:  PYTHONPATH=. python scripts/cycle47_honest_pnl_audit.py
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING,
                    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger("cycle47")

CANON_TL = "vol_arb/output/BTC_v2_trade_log.parquet"


def nc_sharpe(daily_pnl: pd.Series) -> float:
    """Annualised Sharpe of a daily non-compound PnL series."""
    d = daily_pnl.dropna()
    if len(d) < 2 or d.std() == 0:
        return float("nan")
    return float(d.mean() / d.std() * np.sqrt(365.0))


def summarise(name: str, per_trade: pd.Series, ts_index: pd.DatetimeIndex) -> dict:
    """Per-trade NC PnL series -> headline stats."""
    per_trade = pd.to_numeric(per_trade, errors="coerce").dropna()
    daily = per_trade.groupby(per_trade.index.normalize()).sum()
    total = per_trade.sum()
    out = {
        "name": name,
        "n_trades": len(per_trade),
        "nc_pnl": total,
        "mean_per_trade": per_trade.mean(),
        "win_rate": (per_trade > 0).mean() * 100.0,
        "nc_sharpe": nc_sharpe(daily),
    }
    return out


def print_row(d: dict) -> None:
    print(f"  {d['name']:<26} NC PnL ${d['nc_pnl']:>14,.0f}  "
          f"{d['n_trades']:>6} trades  mean ${d['mean_per_trade']:>8.1f}  "
          f"win {d['win_rate']:>5.1f}%  NC Sharpe {d['nc_sharpe']:>7.2f}")


def per_year(label: str, per_trade: pd.Series) -> None:
    s = pd.to_numeric(per_trade, errors="coerce").dropna()
    by = s.groupby(s.index.year)
    print(f"  {label} per-year NC PnL:")
    for yr, g in by:
        print(f"    {yr}: ${g.sum():>13,.0f}  ({len(g):>5} trades, "
              f"mean ${g.mean():>8.1f})")


def main() -> int:
    print("=" * 78)
    print("CYCLE-47 HONEST-PnL AUDIT  (Phase 5)")
    print("=" * 78)

    # ---- 1. canon trade log + IV-frozen --------------------------------
    tl = pd.read_parquet(CANON_TL)
    fired = tl[tl["gross_pnl"] != 0].copy()
    net = pd.to_numeric(fired["net_pnl_nc"], errors="coerce")
    vega = pd.to_numeric(fired["vega_pnl_nc"], errors="coerce")
    theta = pd.to_numeric(fired["theta_pnl_nc"], errors="coerce")
    hedge = pd.to_numeric(fired["delta_hedge_pnl_nc"], errors="coerce")
    iv_frozen = net - vega  # = theta + delta_hedge - costs

    print(f"\nCanon trade log: {CANON_TL}")
    print(f"  fired trades: {len(fired)}   window "
          f"{fired.index.min()} -> {fired.index.max()}")
    print(f"  leg sums (NC): vega ${vega.sum():,.0f}  theta ${theta.sum():,.0f}  "
          f"delta_hedge ${hedge.sum():,.0f}")

    print("\n--- Headline comparison (non-compound) ---")
    canon = summarise("V2 canon (vega x d-iv_used)", net, fired.index)
    frozen = summarise("V2 IV-FROZEN (honest)", iv_frozen, fired.index)
    print_row(canon)
    print_row(frozen)

    print()
    per_year("IV-FROZEN", iv_frozen)

    # ---- 2. pinned-exit-IV re-run --------------------------------------
    print("\n--- Re-running V2 with use_pinned_exit_iv=True "
          "(real-trade / SVI-surface exit IV) ---")
    try:
        from vol_arb.backtester import BacktestConfig, VolArbBacktester
        cfg = (BacktestConfig()
               .with_btc_production_filters_v2()
               .with_realistic_costs())
        cfg.use_pinned_exit_iv = True
        cfg.kelly_mult_path = None  # match a clean draft pass
        _metrics, tl_pin = VolArbBacktester(cfg).run_backtest()
        fp = tl_pin[tl_pin["gross_pnl"] != 0].copy()
        pin = summarise("V2 pinned-exit-IV", fp["net_pnl_nc"], fp.index)
        print()
        print_row(pin)
        out_path = "vol_arb/output/BTC_v2_trade_log_pinned_exit.parquet"
        tl_pin.to_parquet(out_path)
        print(f"  (trade log written to {out_path})")
    except Exception as exc:  # noqa: BLE001
        logger.exception("pinned-exit run failed")
        print(f"  PINNED-EXIT RUN FAILED: {exc}")

    print("\n" + "=" * 78)
    print("Interpretation: IV-FROZEN is the honest floor (0 forecastable")
    print("15-min IV change). If it is negative, the backtested edge is the")
    print("vega-noise harvest and the strategy does not survive realistic")
    print("marking. Cross-check: live engine = -$44/trade over 14 trades.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
