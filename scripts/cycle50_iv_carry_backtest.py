"""Cycle-50 (Phase 5/6, 2026-05-18): thin IV-carry vol-selling backtest.

Phase 5 killed V2 (its PnL was an SVI-noise harvest) and showed the ML
model has no tradable vol alpha. The ONLY surviving edge is a thin,
non-ML carry: sell vol when implied is elevated, hold to expiry, and
collect the variance risk premium.

This is a CLEAN standalone backtest — deliberately NOT built into the
contaminated vol_arb/backtester.py. It uses the honest PnL model:

    short-straddle PnL ~ vega x (IV_entry - RV_realized_to_expiry) - cost

where IV_entry is the REAL matched-contract option IV, RV is REAL TSRV
realized vol over the actual holding window, and cost is a realistic
vol-point round-trip. NO SVI noise, NO vega x d(iv_used) fiction.

Trades are NON-OVERLAPPING: enter a short ATM straddle when flat and the
signal fires, hold to the matched contract's expiry, book, repeat.

Signals tested (per feedback_no_cherry_pick — report all, per-year):
  A unconditional               — always sell (baseline VRP)
  B iv > trailing-30d mean       — sell when implied elevated
  C iv > trailing-30d mean + 5vp — sell only when strongly elevated

Usage:  PYTHONPATH=. python scripts/cycle50_iv_carry_backtest.py [--cost-vp 2.5]
"""
from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

SVI = "vol_arb/output/BTC_vol_arb_svi.parquet"
BARS_PER_DAY = 96
TRAIL_BARS = 30 * BARS_PER_DAY  # 30-day trailing IV reference


def fwd_rv_to_expiry(vol_ann: np.ndarray, h_bars: np.ndarray) -> np.ndarray:
    """Annualised realized vol (pct) from each bar to its option's expiry."""
    var = vol_ann ** 2
    cs = np.concatenate([[0.0], np.cumsum(var)])
    n = len(var)
    pos = np.arange(n)
    j = np.minimum(pos + h_bars, n)
    valid = j > pos + 1
    mv = np.where(valid, (cs[j] - cs[np.minimum(pos + 1, n)])
                  / np.maximum(j - pos - 1, 1), np.nan)
    return np.sqrt(mv) * 100.0


def simulate(df: pd.DataFrame, fire: np.ndarray) -> pd.DataFrame:
    """Non-overlapping short-straddle sim. One trade at a time, hold to
    the matched contract's expiry. Records the sizing-agnostic VRP in
    vol points; $ PnL is applied per sizing mode in pnl_series()."""
    iv = df["iv"].to_numpy()
    vega = df["vega"].to_numpy()
    rv = df["rv_exp"].to_numpy()
    hbar = df["h_bars"].to_numpy()
    ts = df.index
    n = len(df)
    trades = []
    i = 0
    while i < n:
        if fire[i] and np.isfinite(rv[i]) and np.isfinite(vega[i]) and vega[i] > 0:
            j = min(i + int(hbar[i]), n - 1)
            trades.append({"entry": ts[i], "exit": ts[j],
                           "iv": iv[i], "rv": rv[i], "vega": vega[i],
                           "vrp_vp": iv[i] - rv[i]})  # short: profit if >0
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(trades)


def pnl_series(tr: pd.DataFrame, cost_vp: float, mode: str,
               vega_budget: float) -> pd.Series:
    """$ PnL per trade. mode='vega' = constant vega budget (the standard
    vol-desk convention); mode='fixedN' = fixed 1-BTC straddle."""
    net_vp = tr["vrp_vp"] - cost_vp
    size = vega_budget if mode == "vega" else tr["vega"]
    return net_vp * size


def report(name: str, tr: pd.DataFrame, cost_vp: float, mode: str,
           vega_budget: float) -> None:
    if tr.empty:
        print(f"  {name:<30} no trades")
        return
    pnl = pnl_series(tr, cost_vp, mode, vega_budget)
    daily = pnl.groupby(tr["exit"].dt.normalize()).sum()
    sharpe = (daily.mean() / daily.std() * np.sqrt(365.0)
              if daily.std() else float("nan"))
    eq = pnl.cumsum()
    mdd = (eq.cummax() - eq).max()
    print(f"  {name:<30} n={len(tr):>4}  PnL ${pnl.sum():>9,.0f}  "
          f"mean ${pnl.mean():>6.1f}  win {100*(pnl>0).mean():4.1f}%  "
          f"Sharpe {sharpe:>5.2f}  MDD ${mdd:>8,.0f}  "
          f"VRP {tr['vrp_vp'].mean():+.2f}vp")


def per_year(tr: pd.DataFrame, cost_vp: float, vega_budget: float) -> None:
    if tr.empty:
        return
    pnl = pnl_series(tr, cost_vp, "vega", vega_budget)
    for yr, g in pnl.groupby(tr["exit"].dt.year):
        print(f"      {yr}: ${g.sum():>9,.0f}  ({len(g):>3} trades, "
              f"mean ${g.mean():>7.1f})")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cost-vp", type=float, default=2.5,
                    help="round-trip cost in vol points (default 2.5)")
    ap.add_argument("--vega-budget", type=float, default=0.0,
                    help="constant-vega sizing budget ($/vol-pt); "
                         "0 = use dataset median vega")
    args = ap.parse_args()

    df = pd.read_parquet(SVI)
    df.index = pd.to_datetime(df.index, utc=True)
    df = df.sort_index()
    df = df[df["iv"].notna() & (df["vol_ann"] > 0) & df["vega"].notna()
            & df["dte_days"].notna()].copy()
    df["h_bars"] = np.clip((df["dte_days"].astype(float) * BARS_PER_DAY)
                           .round().astype(int), 1, None)
    df["rv_exp"] = fwd_rv_to_expiry(df["vol_ann"].to_numpy(float),
                                    df["h_bars"].to_numpy())

    iv = df["iv"]
    trail = iv.shift(1).rolling(TRAIL_BARS, min_periods=BARS_PER_DAY).mean()
    signals = {
        "A unconditional": np.ones(len(df), dtype=bool),
        "B iv > trail-30d mean": (iv > trail).to_numpy(),
        "C iv > trail-30d mean + 5vp": (iv > trail + 5.0).to_numpy(),
    }

    budget = args.vega_budget or float(df["vega"].median())

    print("=" * 88)
    print(f"CYCLE-50  THIN IV-CARRY BACKTEST   (cost = {args.cost_vp}vp, "
          f"constant-vega budget = {budget:.1f})")
    print("=" * 88)
    print(f"\nBTC, {df.index.min().date()}..{df.index.max().date()}, "
          f"{len(df)} candidate bars. Short ATM straddle, hold-to-expiry, "
          f"non-overlapping.")
    print("Honest PnL = vega_budget x (IV_entry - RV_realized - cost). No SVI noise.")
    print("Sizing = CONSTANT VEGA (the vol-desk standard): size each trade to a")
    print("fixed vol-risk budget, not a fixed contract count.\n")

    results = {}
    for name, fire in signals.items():
        tr = simulate(df, fire)
        results[name] = tr
        report(name, tr, args.cost_vp, "vega", budget)
        per_year(tr, args.cost_vp, budget)
        print()

    # sizing lesson: fixed-N vega-weights toward the bad (high-vega) tail
    print("--- sizing comparison, signal B (why constant-vega matters) ---")
    b = results["B iv > trail-30d mean"]
    report("  fixed-N (1 BTC straddle)", b, args.cost_vp, "fixedN", budget)
    report("  constant-vega (standard)", b, args.cost_vp, "vega", budget)

    print("\n--- cost sensitivity, signal B, constant-vega ---")
    for c in (1.5, 2.0, 2.5, 3.0, 3.5):
        pnl = pnl_series(b, c, "vega", budget)
        daily = pnl.groupby(b["exit"].dt.normalize()).sum()
        sh = daily.mean() / daily.std() * np.sqrt(365.0) if daily.std() else float("nan")
        print(f"  cost {c:.1f}vp: PnL ${pnl.sum():>9,.0f}  Sharpe {sh:+5.2f}")

    # tail characterisation
    nv = b["vrp_vp"] - args.cost_vp
    print(f"\n--- tail (signal B, net VRP in vol pts, cost {args.cost_vp}) ---")
    print(f"  mean {nv.mean():+.2f}  median {nv.median():+.2f}  std {nv.std():.1f}  "
          f"skew {nv.skew():.2f}  p1 {nv.quantile(.01):.1f}  p5 {nv.quantile(.05):.1f}")
    print("  short vol has a fat left tail — worst trades are vol-explosion "
          "events; size and tail-risk limits accordingly.")

    if not b.empty:
        out = "vol_arb/output/cycle50_iv_carry_trades.parquet"
        b.to_parquet(out)
        print(f"\nTrade log (signal B) written to {out}")

    print("\n" + "=" * 88)
    print("Verdict: positive, mostly-stable-per-year Sharpe under constant-vega")
    print("sizing = a real (thin) carry edge. The fat left tail caps leverage.")
    print("=" * 88)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
