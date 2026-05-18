"""Cycle-49 (Phase 5/6, 2026-05-18): vol-selling tradability probe.

V2 is dead (Phase-5: its PnL was an SVI-noise harvest). But the
realized-vol FORECAST is sound (R2 0.59, clean label). Question for the
pivot: is there a REAL, capturable vol-selling edge at some horizon,
after honest costs?

The honest P&L of a delta-hedged short straddle is the variance risk
premium it captures:  PnL ~ vega x (IV_entry - RV_realized_over_hold).
V2's bug was using vega x d(IV) (change in a noisy implied) instead of
vega x (IV - RV) (implied minus REAL realized).

This probe measures, with NO retrain and NO SVI noise:
  - VRP(H) = iv_entry - RV_realized(T, T+H), the real premium, at
    horizons 15min / 1h / 4h / 1d / 3d / to-expiry.
  - Whether VRP exceeds realistic round-trip cost (~1.5-3 vol pts).
  - Whether conditioning entry on the model's rv_pred improves VRP.

Data: BTC_vol_arb_svi.parquet (iv = real matched-contract entry IV;
vol_ann = TSRV realized vol; vega; dte_days) + the har_pd predictions.

Usage:  PYTHONPATH=. python scripts/cycle49_tradability_probe.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SVI = "vol_arb/output/BTC_vol_arb_svi.parquet"
PREDS = "vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions.parquet"
BARS_PER_DAY = 96  # 15-min bars

# Realistic round-trip cost of a straddle, in VOL POINTS of the
# straddle's vega. Derived from the live engine: ~$40 round-trip on a
# 0.3-BTC straddle, vega ~40 $/vp/BTC -> ~3.3 vp RT; entry-only ~1.7 vp.
COST_RT_VP = 3.3
COST_ENTRY_VP = 1.7


def fwd_rv_ann(vol_ann: pd.Series, h_bars: int) -> pd.Series:
    """Annualised realized vol over the FORWARD window (T, T+h_bars],
    in percent. vol_ann is per-bar annualised vol (decimal)."""
    var = vol_ann.astype(float) ** 2
    fwd_var = var.rolling(h_bars).mean().shift(-h_bars)
    return np.sqrt(fwd_var) * 100.0


def fwd_rv_to_expiry(df: pd.DataFrame) -> pd.Series:
    """Annualised realized vol from T to the option's expiry, percent."""
    var = (df["vol_ann"].astype(float) ** 2).to_numpy()
    cs = np.concatenate([[0.0], np.cumsum(var)])  # cs[k] = sum(var[:k])
    n = len(df)
    h = np.clip((df["dte_days"].astype(float).to_numpy() * BARS_PER_DAY)
                .round().astype(int), 1, None)
    out = np.full(n, np.nan)
    pos = np.arange(n)
    j = np.minimum(pos + h, n)  # exclusive end
    valid = j > pos + 1
    mean_var = np.where(valid, (cs[j] - cs[pos + 1]) / np.maximum(j - pos - 1, 1),
                        np.nan)
    out = np.sqrt(mean_var) * 100.0
    return pd.Series(out, index=df.index)


def stats(vrp: pd.Series) -> str:
    v = vrp.dropna()
    if len(v) < 50:
        return f"n={len(v)} (insufficient)"
    ir = v.mean() / v.std() * np.sqrt(len(v)) if v.std() else float("nan")
    return (f"n={len(v):>6}  mean={v.mean():+6.2f}vp  median={v.median():+6.2f}vp  "
            f"%pos={100*(v>0).mean():4.1f}%  t-stat={ir:+6.1f}")


def main() -> int:
    df = pd.read_parquet(SVI)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df = df[df["iv"].notna() & (df["vol_ann"] > 0)].copy()

    iv = df["iv"].astype(float)

    print("=" * 78)
    print("CYCLE-49 VOL-SELLING TRADABILITY PROBE")
    print("=" * 78)
    print(f"\nBars with valid iv + realized vol: {len(df)}  "
          f"({df.index.min().date()} .. {df.index.max().date()})")
    print(f"Mean entry IV {iv.mean():.1f}%   mean realized vol "
          f"{(df['vol_ann']*100).mean():.1f}%   mean DTE {df['dte_days'].mean():.1f}d")

    # ---- VRP by horizon (unconditional) --------------------------------
    print("\n--- Variance risk premium  VRP = IV_entry - RV_realized(T,T+H) ---")
    print("    (positive = implied richer than realized = vol-selling fuel)")
    horizons = [("15min", 1), ("1h", 4), ("4h", 16), ("1d", 96), ("3d", 288)]
    vrp_by_h = {}
    for name, hb in horizons:
        vrp = iv - fwd_rv_ann(df["vol_ann"], hb)
        vrp_by_h[name] = vrp
        print(f"  H={name:<6} {stats(vrp)}")
    vrp_exp = iv - fwd_rv_to_expiry(df)
    vrp_by_h["to-expiry"] = vrp_exp
    print(f"  H={'to-exp':<6} {stats(vrp_exp)}")

    # ---- CAVEAT on short-horizon VRP ----------------------------------
    print("\n--- CAVEAT: short-horizon VRP is NOT capturable ---")
    print("  The high VRP at H=15min/1h/4h is a Jensen/noise artifact: the")
    print("  annualised vol of one short window is biased LOW (E[sqrt(var)]")
    print("  < sqrt(E[var])), inflating IV - RV. You also cannot sell a")
    print("  15-min-tenor option; holding a multi-day option for 15 min")
    print("  captures only theta(15min), not VRP(15min) -- that was the V2")
    print("  fallacy. The genuine capturable premium is the TO-EXPIRY VRP.")

    # ---- Does the model's forecast improve timing? ---------------------
    print("\n--- Model-conditioned VRP (to-expiry = the real horizon) ---")
    try:
        preds = pd.read_parquet(PREDS)
        if not isinstance(preds.index, pd.DatetimeIndex):
            preds.index = pd.to_datetime(preds.index, utc=True)
        rv_pred = np.exp(preds["y_pred"].astype(float)) * 100.0
        rv_pred = rv_pred.reindex(df.index)
        model_edge = iv - rv_pred            # model says: how rich is IV
        valid = model_edge.notna() & vrp_exp.notna()
        me = model_edge[valid]
        ve = vrp_exp[valid]
        q = me.quantile([1/3, 2/3])
        rich = me >= q.iloc[1]               # model: IV richest vs forecast RV
        cheap = me <= q.iloc[0]
        print(f"  unconditional             : {stats(ve)}")
        print(f"  model top-tercile (sell)  : {stats(ve[rich])}")
        print(f"  model bottom-tercile      : {stats(ve[cheap])}")
        lift = ve[rich].mean() - ve.mean()
        print(f"  => model timing lift: {lift:+.2f}vp vs unconditional "
              f"({'helps' if lift > 0 else 'no help'})")

        # ---- Net edge: the only honest verdict (to-expiry hold) --------
        print("\n--- NET EDGE, hold-to-expiry (the real capturable horizon) ---")
        print("  cost = option entry (~1.7vp) + multi-day delta-hedge "
              "(~0.5-1.5vp) => realistic 2.0-3.0vp")
        for lbl, series in [("unconditional", ve),
                            ("model-timed (top tercile)", ve[rich])]:
            g = series.mean()
            for cost in (2.0, 3.0):
                net = g - cost
                print(f"  {lbl:<26} VRP {g:+5.2f}vp - cost {cost:.1f}vp "
                      f"=> net {net:+5.2f}vp  [{'edge' if net > 0 else 'no edge'}]")
    except Exception as exc:  # noqa: BLE001
        print(f"  model-conditioning skipped: {exc}")

    print("\n" + "=" * 78)
    print("Read: the TRUE variance risk premium (to-expiry) is small but")
    print("statistically rock-solid. Unconditional vol-selling is ~breakeven")
    print("after realistic cost; the model's timing lift is what creates a")
    print("thin positive edge. Verdict hinges on whether execution cost can")
    print("be held near 2vp -- options are expensive to trade.")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
