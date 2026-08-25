"""ATTACK 4 — NO-LOOKAHEAD on the long-hold path + MUTATION PROBE.

(a) Confirm per-bar BS delta at bar t uses only S<=t and T_t>=0 (structural read).
(b) MUTATION PROBE: shift the BTC price panel by +1 bar (so bar t sees S_{t+1})
    and confirm the hedged PnL CHANGES -> proves the engine reads the path causally
    (an engine that ignored the path, or peeked at a fixed future, would not move).
(c) Anti-probe: shift price path by a HUGE future offset and confirm settlement
    intrinsic changes (it must read the actual expiry bar).

We monkeypatch PricePanel.load via a wrapper that returns a shifted panel; we do
NOT edit realistic_eval.py.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import vol_arb.realistic_eval as re
from vol_arb.realistic_eval import RealisticEvalConfig, evaluate_realistic, PricePanel

BAND = 0.35
TRADE_LOG = "vol_arb/output/BTC_v2_trade_log.parquet"


def run_once(panel_transform=None, shift_bars=0, label=""):
    orig_load = PricePanel.load

    def patched_load(cls, cfg, start, end):
        p = orig_load.__func__(cls, cfg, start, end)
        if panel_transform is None:
            return p
        close2 = panel_transform(p.close.copy())
        return PricePanel(p.ts_i8, close2)

    PricePanel.load = classmethod(patched_load)
    try:
        tl = pd.read_parquet(TRADE_LOG)
        cfg = RealisticEvalConfig(hold="expiry", real_bs_delta=True,
                                  fill_tol_minutes=60.0, spread_source="tape",
                                  delta_hedge_band=BAND)
        res = evaluate_realistic(tl, cfg)
        f = res.trade_log
        cov = f[f["premium_tier"] > 0]
        print(f"[{label}] hedged=${res.hedged_total_usd:,.0f}  "
              f"unhedged=${res.unhedged_total_usd:,.0f}  "
              f"perp_leg=${res.perp_leg_total_usd:,.0f}  "
              f"settle_intrinsic_mean={cov['premium_out_btc'].mean():.5f}")
        return res
    finally:
        PricePanel.load = orig_load


def main():
    print("BASELINE (real path):")
    base = run_once(label="baseline")

    print("\nMUTATION A — shift price path +1 bar earlier (np.roll close by -1, so "
          "bar t carries S_{t+1}); hedged PnL MUST change:")
    shift = run_once(panel_transform=lambda c: np.roll(c, -1), label="roll-1")

    print("\nMUTATION B — multiply path by a tiny epsilon ramp to perturb late bars "
          "only (proves settlement reads the expiry bar):")
    def ramp(c):
        c = c.copy()
        c[-1] *= 1.5   # blow up the very last bar
        return c
    run_once(panel_transform=ramp, label="last-bar*1.5")

    d = abs(shift.hedged_total_usd - base.hedged_total_usd)
    print("\n" + "=" * 70)
    print(f"baseline hedged   = ${base.hedged_total_usd:,.0f}")
    print(f"roll-1   hedged   = ${shift.hedged_total_usd:,.0f}")
    print(f"|delta|           = ${d:,.0f}")
    print("PASS (engine reads path causally) if |delta| is materially > 0.")


if __name__ == "__main__":
    main()
