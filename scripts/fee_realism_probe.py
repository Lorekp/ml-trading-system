"""Adversarial fee-realism probe for the LONG-straddle gamma scalp.

Confirms the full Deribit fee stack is actually charged at the working horizon
(4h, where long loss is smallest), that tighter bands charge MORE rebalance fees
(so longs are not flattered by under-charged gamma-scalp fees), and that the
negative result is honest (zero-fee longs still lose, so the loss is structural,
not fee-suppressed).
"""
import sys
import numpy as np
import pandas as pd

from vol_arb.realistic_eval import evaluate_realistic, RealisticEvalConfig
from vol_arb import deribit_fees as fees

tl = pd.read_parquet("vol_arb/output/BTC_v2_trade_log.parquet")
# Force the FULL fired set to LONG (signal=+1) -- the truly unconditional long.
tl = tl.copy()
tl["signal"] = 1.0

HOLD = "4h"

def run(band, zero_fees=False):
    cfg = RealisticEvalConfig(hold=HOLD, real_bs_delta=True, fill_tol_minutes=60.0,
                              spread_source="tape", delta_hedge_band=band,
                              zero_fees=zero_fees)
    res = evaluate_realistic(tl, cfg)
    return res

print("=== Deribit fee constants (deribit_fees.py) ===")
print(f"OPT_TRADE_FEE_BTC   = {fees.OPT_TRADE_FEE_BTC}  (0.03% underlying/fill, maker==taker)")
print(f"OPT_FEE_PREMIUM_CAP = {fees.OPT_FEE_PREMIUM_CAP}  (12.5% of fill premium)")
print(f"OPT_DELIVERY_FEE_BTC= {fees.OPT_DELIVERY_FEE_BTC}")
print(f"PERP_TAKER_FEE      = {fees.PERP_TAKER_FEE}  (0.05% coin-margined)")
print(f"PERP_SLIPPAGE_BPS   = {fees.PERP_SLIPPAGE_BPS}  -> one-way taker rate = {fees.PERP_TAKER_FEE + fees.PERP_SLIPPAGE_BPS/1e4:.6f}")
print()

print("=== Sanity: one-way perp rebalance fee on $100k notional ===")
print(f"  fee = ${fees.perp_rebalance_fee_usd(np.array([100000.0]))[0]:.2f}  (expect 100000*0.0008 = $80)")
print(f"  option fill fee on 0.05 BTC premium @ $60k = ${fees.option_fill_fee_btc(np.array([0.05]))[0]*60000:.2f}  "
      f"(min(0.0003, 0.125*0.05)=0.0003 BTC -> $18)")
print()

print("=== BAND SWEEP @ 4h LONG (real fees) ===")
rows = []
for band in [0.05, 0.10, 0.20, 0.35]:
    res = run(band)
    f = res.trade_log
    cov = f["premium_tier"].to_numpy() > 0
    n = int(cov.sum())
    opt_fee = res.option_fee_total_usd
    hedge_fee = res.hedge_fee_total_usd
    funding = res.funding_total_usd
    nreb = np.nansum(np.where(cov, f["n_rebalances"].to_numpy(float), np.nan))
    rows.append((band, res.hedged_total_usd, opt_fee, hedge_fee, funding, nreb, n))
    print(f"band={band:<4}: hedged_total=${res.hedged_total_usd:>12,.0f}  "
          f"opt_fee=${opt_fee:>10,.0f}  hedge_fee=${hedge_fee:>9,.0f}  "
          f"funding=${funding:>8,.0f}  tot_rebal={nreb:>9,.0f}  n={n}")
print()

print("=== FEE MONOTONICITY CHECK (tighter band must charge MORE hedge fee + more rebalances) ===")
bands = [r[0] for r in rows]
hfees = [r[3] for r in rows]
nrebs = [r[5] for r in rows]
mono_fee = all(hfees[i] >= hfees[i+1] for i in range(len(hfees)-1))
mono_reb = all(nrebs[i] >= nrebs[i+1] for i in range(len(nrebs)-1))
print(f"  hedge_fee by band (tight->wide): {[f'${x:,.0f}' for x in hfees]}  monotone_decreasing={mono_fee}")
print(f"  rebalances by band (tight->wide): {[f'{x:,.0f}' for x in nrebs]}  monotone_decreasing={mono_reb}")
print()

print("=== ZERO-FEE COUNTERFACTUAL @ band=0.05 (tightest, most fees) ===")
res_zf = run(0.05, zero_fees=True)
res_rf = run(0.05, zero_fees=False)
print(f"  zero-fee hedged_total = ${res_zf.hedged_total_usd:,.0f}")
print(f"  real-fee hedged_total = ${res_rf.hedged_total_usd:,.0f}")
print(f"  total fee drag        = ${res_zf.hedged_total_usd - res_rf.hedged_total_usd:,.0f}")
print(f"  => even with ZERO fees the long {'LOSES' if res_zf.hedged_total_usd < 0 else 'PROFITS'} "
      f"(${res_zf.hedged_total_usd:,.0f}): loss is {'STRUCTURAL' if res_zf.hedged_total_usd < 0 else 'fee-driven'}")
print()

print("=== PER-TRADE FEE REALISM (band=0.05, covered subset) ===")
f = res_rf.trade_log
cov = f["premium_tier"].to_numpy() > 0
fc = f[cov]
n = len(fc)
print(f"  n_covered = {n}")
print(f"  mean option_fee/trade = ${fc['option_fee_usd'].mean():.2f}")
print(f"  mean hedge_fee/trade  = ${fc['hedge_fee_usd'].mean():.2f}")
print(f"  mean funding/trade    = ${fc['funding_usd'].mean():.2f}")
print(f"  mean n_rebalances     = {fc['n_rebalances'].mean():.2f}")
print(f"  mean n_path_bars      = {fc['n_path_bars'].mean():.1f}  (4h=240min, so up to ~240 1-min bars)")
print(f"  mean N_contracts      = {fc['N_contracts_nc'].mean():.3f} BTC")
print(f"  mean index_price      = ${fc['index_price'].mean():,.0f}")
# expected option RT fee per trade: ~ entry 2 legs + exit 2 legs * 0.0003 BTC * N * index (uncapped)
exp_opt = 4 * fees.OPT_TRADE_FEE_BTC * fc['N_contracts_nc'].mean() * fc['index_price'].mean()
print(f"  expected uncapped 4-fill option fee/trade = ${exp_opt:.2f} (cap may reduce it)")
# expected per-rebalance perp fee: notional ~ |hedge_btc| * index ; check hedge_fee/n_rebal
hf_per_reb = fc['hedge_fee_usd'].sum() / max(fc['n_rebalances'].sum(), 1)
print(f"  realized hedge_fee per rebalance = ${hf_per_reb:.2f}")
print()
print("DONE")
