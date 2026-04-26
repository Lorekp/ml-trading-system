"""DRL hedger horizon-mismatch ablation at h=5m (plan Phase 2h5.7.B).

The shipped SAC policy ``production_sac_v4_final`` was trained at the
DRLHedgingConfig default ``bar_minutes=15``. The V2 ``with_btc_production_h5m()``
preset still loads that policy and invokes it at ``holding_minutes=5``,
i.e. 3x more often per unit time. The audit (plan §Phase-0) flagged this
as a silent policy-cadence mismatch; this script measures the effect
WITHOUT retraining SAC.

Method
------
1. Build the V2 h=5 preset (``with_btc_production_h5m()``).
2. Run TWO backtests with that preset:
   - **DRL-on**: existing config (the canonical h=5m trade log).
   - **DRL-off**: `model_copy(use_drl_hedging=False)` — falls back to
     the rule-based ``simulate_discrete_hedging`` (a cadence-invariant
     Black-Scholes-bands hedger).
3. Compare headline NC PnL / NC Sharpe / NC MDD between the two runs.

Decision (pre-committed in plan §Phase 2h5.7.B):
- DRL-on better by >= +5% (NC PnL or +0.5 Sharpe): policy transfers
  despite mismatch; ship as-is.
- |d| < 5%: mismatch immaterial at h=5; ship with documented note.
- DRL-off better by >= +5%: mismatch is harmful; recommend
  ``with_btc_production_h5m_static_hedge()`` factory in follow-up.

Output: prints headline comparison; writes
``vol_arb/output/BTC_v2_trade_log_h5m_no_drl.parquet`` so downstream
analyses can reuse the DRL-off trade log.

Usage
-----
    cd C:/Claude/ml-trading-system
    python scripts/h5_drl_ablation.py
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.backtester import BacktestConfig, VolArbBacktester
from vol_arb.metrics import build_nc_view

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


CAPITAL = 100_000.0
H5_DRL_ON_PATH = Path("vol_arb/output/BTC_v2_trade_log_h5m.parquet")
H5_DRL_OFF_PATH = Path("vol_arb/output/BTC_v2_trade_log_h5m_no_drl.parquet")


def _headline_nc(tl: pd.DataFrame) -> dict:
    """Daily-NC NC PnL / NC Sharpe / NC MDD on a fired-bar trade log."""
    t = tl.loc[tl["signal"] != 0]
    if t.empty:
        return dict(nc_pnl=0.0, nc_sharpe=float("nan"), nc_mdd=0.0,
                    n_trades=0, worst_day=0.0, best_day=0.0)
    daily = t["net_pnl_nc"].resample("1D").sum()
    daily_active = daily.loc[daily != 0]
    sharpe = (
        float(daily_active.mean() / daily_active.std() * np.sqrt(365))
        if len(daily_active) >= 2 and daily_active.std() > 0
        else float("nan")
    )
    eq = daily.cumsum()
    mdd = float((eq - eq.cummax()).min())
    return dict(
        nc_pnl=float(t["net_pnl_nc"].sum()),
        nc_sharpe=sharpe,
        nc_mdd=mdd,
        n_trades=int((t["net_pnl_nc"] != 0).sum()),
        worst_day=float(daily.min()),
        best_day=float(daily.max()),
    )


def _run_h5_backtest(*, use_drl: bool) -> pd.DataFrame:
    cfg = BacktestConfig.with_btc_production_h5m()
    if not use_drl:
        cfg = cfg.model_copy(update=dict(use_drl_hedging=False))
    label = "DRL-on" if use_drl else "DRL-off"
    logger.info("Running h=5 V2 backtest (%s)...", label)
    t0 = time.time()
    bt = VolArbBacktester(cfg)
    # Use the public run_backtest() entry — it loads data, calls
    # compute_pnl, builds tearsheet metrics, and returns the trade log.
    _metrics, trade_log = bt.run_backtest()
    logger.info(
        "  %s: %.1f min wall-clock, %d fired bars",
        label, (time.time() - t0) / 60, int((trade_log["signal"] != 0).sum()),
    )
    return trade_log


def main() -> int:
    # ── DRL-off run ──
    if H5_DRL_OFF_PATH.exists():
        logger.info("Reusing existing DRL-off trade log at %s", H5_DRL_OFF_PATH)
        tl_off = pd.read_parquet(H5_DRL_OFF_PATH)
    else:
        tl_off = _run_h5_backtest(use_drl=False)
        tl_off.to_parquet(H5_DRL_OFF_PATH)
        logger.info("DRL-off trade log written to %s", H5_DRL_OFF_PATH)

    # ── DRL-on canonical (load existing) ──
    if not H5_DRL_ON_PATH.exists():
        logger.error("Canonical DRL-on h=5 trade log missing at %s — "
                     "run the V2 h=5 preset first.", H5_DRL_ON_PATH)
        return 1
    tl_on = pd.read_parquet(H5_DRL_ON_PATH)

    on = _headline_nc(tl_on)
    off = _headline_nc(tl_off)

    print("\n" + "=" * 80)
    print("  h=5m DRL-OFF ABLATION  —  plan Phase 2h5.7.B")
    print("=" * 80)
    print(f"  Capital reference: ${CAPITAL:,.0f}")
    print(f"\n  {'metric':<14}{'DRL-on':>16}{'DRL-off':>16}{'d (off-on)':>16}{'d %':>10}")
    print("  " + "-" * 74)
    for k in ("nc_pnl", "nc_sharpe", "nc_mdd", "n_trades", "worst_day", "best_day"):
        v_on, v_off = on[k], off[k]
        delta = v_off - v_on
        pct = (delta / abs(v_on) * 100) if v_on else float("nan")
        if k == "nc_sharpe":
            print(f"  {k:<14}{v_on:>16.3f}{v_off:>16.3f}{delta:>+16.3f}{pct:>+10.2f}")
        elif k == "n_trades":
            print(f"  {k:<14}{v_on:>16,}{v_off:>16,}{delta:>+16,}{pct:>+10.2f}")
        else:
            print(f"  {k:<14}${v_on:>14,.0f} ${v_off:>14,.0f} ${delta:>+14,.0f}{pct:>+10.2f}")

    # Pre-committed decision logic from plan §Phase 2h5.7.B
    pnl_pct = (off["nc_pnl"] - on["nc_pnl"]) / abs(on["nc_pnl"]) * 100 if on["nc_pnl"] else 0.0
    sharpe_delta = off["nc_sharpe"] - on["nc_sharpe"]
    print(f"\n  dNC PnL: {pnl_pct:+.2f}%   dNC Sharpe: {sharpe_delta:+.3f}")
    if pnl_pct >= 5 or sharpe_delta >= 0.5:
        print("  VERDICT: DRL-off WINS by >=5% PnL or +0.5 Sharpe — "
              "h=15 policy is harmful at h=5; recommend static-hedge variant.")
    elif pnl_pct <= -5 or sharpe_delta <= -0.5:
        print("  VERDICT: DRL-on WINS by >=5% PnL or +0.5 Sharpe — "
              "policy transfers despite cadence mismatch; ship as-is.")
    else:
        print("  VERDICT: DRL contribution at h=5 is within +/-5% / +/-0.5 Sharpe "
              "of the static-hedge baseline — mismatch is non-material.")
    print("=" * 80 + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
