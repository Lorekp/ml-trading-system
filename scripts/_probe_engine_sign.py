"""Confirm the engine is NOT suppressing longs via a sign/peek bug.

(A) use_implied_path self-check: GBM @ sigma=IV => RV==IV => gamma leg ~0
    (long & short both ~0). If longs were structurally broken this would not hold.
(B) Synthetic single big-up-move check on the Ito gamma form: a LONG straddle
    with realized var >> implied MUST produce POSITIVE gamma pnl; SHORT negative.
(C) Post-exit peek: searchsorted windows are bounded by exit_i8 (side='right');
    bars beyond exit are masked to 0. Verify numerically: extending the panel
    with post-exit bars must NOT change a trade's gamma pnl.
"""
import numpy as np, pandas as pd
from vol_arb.realistic_eval import (RealisticEvalConfig, evaluate_realistic,
    compute_gamma_leg_realpath, PricePanel)

tl = pd.read_parquet("vol_arb/output/BTC_v2_trade_log.parquet").copy()
tl["signal"] = np.where(tl["signal"].to_numpy() != 0, 1.0, 0.0)

# (A) implied-path self check at 4h long
cfg_imp = RealisticEvalConfig(hold="4h", real_bs_delta=True, fill_tol_minutes=60.0,
        spread_source="tape", delta_hedge_band=0.35, use_implied_path=True, zero_fees=True)
r = evaluate_realistic(tl, cfg_imp)
print(f"(A) implied-path zero-fee gamma_leg_total (long) = ${r.gamma_leg_total_usd:,.0f}  (should be ~0 vs real-path scale)")

# (B) synthetic big-up-move on the Ito form: 1 trade, S goes +10% over the window
S0=100000.0; K=100000.0
# build a tiny panel: entry t0, 16 1-min bars rising 1% each? use real func directly
ts = np.arange(0, 17).astype('int64')*60_000_000_000  # 17 ns-stamps 1min apart
close = S0*np.cumprod(np.r_[1.0, np.full(16, 1.005)])  # +0.5%/min ~ big realized move
panel = PricePanel(ts, close)
cfg = RealisticEvalConfig(hold="15m", delta_hedge_band=0.35)
entry = np.array([np.datetime64(0,'ns')]); exit_ = np.array([np.datetime64(16*60_000_000_000,'ns')])
for sgn,label in [(1.0,'LONG'),(-1.0,'SHORT')]:
    g,_,_,_,_,_ = compute_gamma_leg_realpath(entry, exit_, np.array([S0]),
        np.array([0.0]), np.array([1e-8]), np.array([sgn]), np.array([0.5]),
        np.array([7/365.25]), panel, cfg)
    print(f"(B) big-up-move {label} straddle gamma pnl/btc = {g[0]:+.6f}  (LONG must be >0, SHORT <0)")

# (C) post-exit peek: extend panel with a wild post-exit spike; gamma must be unchanged
close2 = close.copy()
ts2 = np.r_[ts, (np.arange(17,40)*60_000_000_000).astype('int64')]
close2 = np.r_[close2, close2[-1]*np.cumprod(np.full(23, 1.05))]  # huge post-exit moves
panel2 = PricePanel(ts2, close2)
gA,_,_,_,_,_ = compute_gamma_leg_realpath(entry, exit_, np.array([S0]),
    np.array([0.0]), np.array([1e-8]), np.array([1.0]), np.array([0.5]),
    np.array([7/365.25]), panel, cfg)
gB,_,_,_,_,_ = compute_gamma_leg_realpath(entry, exit_, np.array([S0]),
    np.array([0.0]), np.array([1e-8]), np.array([1.0]), np.array([0.5]),
    np.array([7/365.25]), panel2, cfg)
print(f"(C) gamma w/ short panel = {gA[0]:.8f} ; w/ post-exit spike added = {gB[0]:.8f} ; "
      f"identical={np.isclose(gA[0],gB[0])}  (must be True -> no post-exit peek)")
