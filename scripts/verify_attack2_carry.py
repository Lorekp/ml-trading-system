"""ATTACK 2 — is the +$444,632 un-hedged to-expiry carry HONEST?

Decompose option_premium leg. Verify:
  - entry premium is the ADVERSE MkrTkr60 fill (short SELLS at mark*(1-h)).
  - exit = settlement intrinsic |S_T-K|/S_T, S_T = panel close at/after expiry bar.
  - no lookahead in S_T (end-stamped panel).
Hand-reproduce 3 trades from raw columns. Report per-trade $/trade carry and
whether it is economically sane for BTC straddles.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from vol_arb.realistic_eval import RealisticEvalConfig, PricePanel, _to_i8_ns

BAND = 0.35


def main():
    f = pd.read_parquet(f"vol_arb/output/verify_frame_band{int(BAND*100):02d}.parquet")
    cov = f[f["premium_tier"] > 0].copy().reset_index(drop=True)
    n = len(cov)
    print(f"covered n={n}")

    prem_in = cov["premium_in_btc"].to_numpy(float)        # adverse fill (received)
    prem_out = cov["premium_out_btc"].to_numpy(float)      # settlement intrinsic
    signal = cov["signal"].to_numpy(float)
    N = cov["N_contracts_nc"].to_numpy(float)
    index = cov["index_price"].to_numpy(float)
    opt_prem = cov["option_premium_usd"].to_numpy(float)
    opt_fee = cov["option_fee_usd"].to_numpy(float)
    unhedged = cov["total_unhedged_usd"].to_numpy(float)

    print(f"\nSIGNAL distribution: short(-1)={int((signal<0).sum())} long(+1)={int((signal>0).sum())}")
    print(f"premium_in_btc  mean={prem_in.mean():.5f}  (BTC, received by short)")
    print(f"premium_out_btc mean={prem_out.mean():.5f}  (BTC, settlement intrinsic paid)")
    print(f"option_premium_usd  sum=${opt_prem.sum():,.0f}  mean=${opt_prem.mean():.1f}/tr")
    print(f"option_fee_usd      sum=${opt_fee.sum():,.0f}  mean=${opt_fee.mean():.1f}/tr")
    print(f"total_unhedged_usd  sum=${np.nansum(unhedged):,.0f}  mean=${np.nanmean(unhedged):.1f}/tr")

    # reproduce option_premium_usd from raw
    pnl_btc = (prem_in - prem_out) * N
    pnl_btc = np.where(signal == 1, -pnl_btc, pnl_btc)
    repro = pnl_btc * index
    print(f"\nreproduce option_premium_usd from (prem_in-prem_out)*N*index: "
          f"max abs diff = {np.max(np.abs(repro - opt_prem)):.4f}")

    # entry premium adversity check: compare premium_in_btc to entry_call_real+entry_put_real (mid)
    if {"entry_call_real", "entry_put_real"}.issubset(cov.columns):
        mid_in = cov["entry_call_real"].to_numpy(float) + cov["entry_put_real"].to_numpy(float)
        # short: premium_in should be <= mid (sold below mark). long: >= mid.
        short = signal < 0
        frac_adverse_short = float(np.mean(prem_in[short] <= mid_in[short] + 1e-12))
        print(f"\nADVERSITY CHECK (short sells below mark):")
        print(f"  frac short trades with premium_in <= mid = {frac_adverse_short:.3f}")
        ratio = np.where(mid_in > 0, prem_in / mid_in, np.nan)
        print(f"  mean premium_in/mid (short) = {np.nanmean(ratio[short]):.4f} (should be <1)")
        print(f"  mean effective_half_spread = {cov['effective_half_spread'].mean():.4f}")
        print(f"  maker_entry frac = {cov['maker_entry'].mean():.3f}  "
              f"maker_exit frac = {cov['maker_exit'].mean():.3f}")

    # ── Hand-check 3 trades with NO-LOOKAHEAD settlement verification ──
    # Rebuild the price panel over the window and verify S_T = close at/after expiry bar.
    entry_ts = pd.to_datetime(cov["entry_ts"], utc=True)
    expiry = pd.to_datetime(cov["expiry_dt"], utc=True)
    win_start = entry_ts.min()
    win_end = expiry.max() + pd.Timedelta(days=1)
    cfg = RealisticEvalConfig(hold="expiry", real_bs_delta=True, fill_tol_minutes=60.0,
                              spread_source="tape", delta_hedge_band=BAND)
    panel = PricePanel.load(cfg, win_start, win_end)

    strike = cov["strike"].to_numpy(float)
    exit_i8 = _to_i8_ns(pd.DatetimeIndex(expiry))
    s1_idx = np.clip(np.searchsorted(panel.ts_i8, exit_i8, side="right") - 1, 0, len(panel.close) - 1)
    S1 = panel.close[s1_idx]
    intrinsic = (np.maximum(S1 - strike, 0.0) + np.maximum(strike - S1, 0.0)) / S1
    print(f"\nSETTLEMENT recompute vs stored prem_out: max abs diff = "
          f"{np.max(np.abs(intrinsic - prem_out)):.6f}")

    # verify the chosen panel bar is at/BEFORE expiry (end-stamped, no lookahead)
    panel_ts_at = panel.ts_i8[s1_idx]
    lookahead = panel_ts_at > exit_i8  # any chosen bar AFTER expiry = lookahead
    print(f"settlement bar AFTER expiry (lookahead) count = {int(lookahead.sum())} / {n}")
    gap_min = (exit_i8 - panel_ts_at) / 60e9
    print(f"median gap (expiry - settlement_bar) = {np.median(gap_min):.2f} min "
          f"(>=0 means causal; close to 0 is fine)")

    # pick 3 representative covered trades and print full reconciliation
    print("\n" + "#" * 70)
    print("HAND-CHECK 3 TRADES (entry premium, settlement, N, fee -> PnL)")
    print("#" * 70)
    idxs = [0, n // 2, n - 1]
    for i in idxs:
        carry_btc = (prem_in[i] - prem_out[i]) * N[i]
        if signal[i] == 1:
            carry_btc = -carry_btc
        carry_usd = carry_btc * index[i]
        print(f"\ntrade #{i}: entry={entry_ts.iloc[i]}  expiry={expiry.iloc[i]}")
        print(f"  signal={signal[i]:+.0f}  strike={strike[i]:.0f}  S_T={S1[i]:.1f}  "
              f"index={index[i]:.1f}  N={N[i]:.4f}")
        print(f"  premium_in(adverse)={prem_in[i]:.5f}  settle_intrinsic={prem_out[i]:.5f} BTC")
        print(f"  carry_btc=(in-out)*N={carry_btc:.5f}  -> carry_usd=${carry_usd:,.1f}")
        print(f"  option_fee_usd=${opt_fee[i]:,.1f}  total_unhedged=${unhedged[i]:,.1f}")
        print(f"  RECONCILE: opt_prem_usd stored=${opt_prem[i]:,.1f}  recomputed=${carry_usd:,.1f}  "
              f"unhedged check=${carry_usd - opt_fee[i]:,.1f}")

    # Economic sanity: $/trade carry, and per-trade carry as fraction of premium
    print("\n" + "#" * 70)
    print("ECONOMIC SANITY")
    print("#" * 70)
    print(f"mean carry $/trade (unhedged) = ${np.nanmean(unhedged):.1f}")
    notional_usd = N * index
    print(f"mean straddle notional $/trade = ${np.nanmean(notional_usd):,.0f}")
    print(f"mean premium received $/trade  = ${np.nanmean(prem_in * N * index):,.0f}")
    print(f"carry as %% of premium received = "
          f"{100*np.nanmean(unhedged)/np.nanmean(prem_in*N*index):.2f}%")
    print(f"carry as bps of notional        = "
          f"{1e4*np.nanmean(unhedged)/np.nanmean(notional_usd):.2f} bps")


if __name__ == "__main__":
    main()
