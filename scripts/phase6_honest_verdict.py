"""Historical Phase-6 diagnostic for the invalidated V2 15-minute straddle.

Runs the invalidated historical V2 preset two ways, full history:
  (1) honest_vega_iv=False  -> reproduces the legacy SVI-noise headline
  (2) honest_vega_iv=True   -> marks the SAME trades to the real DVOL IV move

then marks a recent 12-month sample to real Deribit fills and runs
``gate_no_fake_profit``. That function is a sign-only smoke check: it catches
one aggregate sign contradiction but does not validate magnitude, per-trade
correlation, out-of-sample performance, or deployability. Writes a short report.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
logging.basicConfig(level=logging.ERROR)

from vol_arb.backtester import BacktestConfig, VolArbBacktester
from vol_arb.honest_pnl import mark_trade_log_to_real_premium, gate_no_fake_profit
from vol_arb.options_loader import load_option_trades

OUT = "vol_arb/output/PHASE6_HONEST_VERDICT.md"
TRADES_DIR = "data/deribit/trades/BTC_option"


def nc_sharpe(per_trade: pd.Series) -> float:
    d = per_trade.groupby(per_trade.index.normalize()).sum().dropna()
    return float(d.mean() / d.std() * np.sqrt(365.0)) if d.std() > 0 else float("nan")


def run(honest: bool):
    cfg = BacktestConfig().with_historical_btc_v2().with_realistic_costs()
    cfg.honest_vega_iv = honest
    _, tl = VolArbBacktester(cfg).run_backtest()
    e = tl[tl["N_contracts_nc"] > 0].copy()
    net = pd.to_numeric(e["net_pnl_nc"], errors="coerce")
    return e, net


def gate_sample(e: pd.DataFrame):
    cutoff = pd.to_datetime(e.index.max(), utc=True) - pd.Timedelta(days=365)
    s = e[e.index >= cutoff]
    a = pd.to_datetime(s.index.min(), utc=True).strftime("%Y-%m-%d")
    b = (pd.to_datetime(s["exit_target"].max(), utc=True) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
    trades = load_option_trades(TRADES_DIR, start_date=a, end_date=b)
    marked = mark_trade_log_to_real_premium(s, trades, tol_minutes=60.0)
    return gate_no_fake_profit(marked.trade_log)


def main() -> int:
    el, netl = run(False)
    eh, neth = run(True)
    same = el.index.equals(eh.index)
    gl = gate_sample(el)
    gh = gate_sample(eh)
    lines = [
        "# Historical Phase-6 V2 PnL Diagnostic (2026-06-21)",
        "",
        f"Same trades fire in both runs: **{same}** (signal untouched; only the "
        "vega-leg IV marking differs).",
        "",
        "| Run | NC PnL | mean/trade | win% | NC Sharpe | sign-only smoke check (last-12m sample) |",
        "|---|---|---|---|---|---|",
        f"| Legacy (honest_vega=False) — the fiction | ${netl.sum():,.0f} | "
        f"${netl.mean():.1f} | {(netl>0).mean()*100:.1f}% | {nc_sharpe(netl):.2f} | "
        f"{'PASS' if gl.passed else 'FAIL'} |",
        f"| **Real-IV marking (honest_vega=True)** | ${neth.sum():,.0f} | "
        f"${neth.mean():.1f} | {(neth>0).mean()*100:.1f}% | {nc_sharpe(neth):.2f} | "
        f"{'PASS' if gh.passed else 'FAIL'} |",
        "",
        f"Legacy sign-only check: {gl.message}",
        "",
        f"Real-IV sign-only check: {gh.message}",
        "",
        "**Covered real-fill sample (last 12 months):** "
        f"${gh.detail.get('real_fill_mean', float('nan')):.1f}/trade — a loss "
        "consistent with the historical live-paper observation "
        "(−$44/trade, 14/14 losers).",
        "",
        "**Verdict:** marking the vega leg to the real DVOL IV move collapses the "
        f"headline from ${netl.sum():,.0f} to ${neth.sum():,.0f} on identical "
        "trades. The legacy headline fails the sign-only smoke check because its "
        "profit contradicts the covered real-fill loss. The real-IV variant "
        "passes only because both aggregates lose; that pass is not strategy "
        "validation. This diagnostic establishes no deployable edge for the "
        "15-minute straddle. Any salvage claim requires independent validation.",
    ]
    report = "\n".join(lines)
    with open(OUT, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
