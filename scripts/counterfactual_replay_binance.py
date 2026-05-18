"""Counterfactual replay — Binance 32-asset live paper trades under correct
inverse-vol (risk-parity) sizing.

Phase-5 sizing-fix verification (2026-05-18). The live paper engine lost
-$15,039.52 over 714 closed positions (2026-04-16..04-30); DENT + PIPPIN =
97.3% of it. Root cause: the live inverse-vol sizing overlay failed OPEN —
`compute_vol_mult` returned 1.0 (full notional) whenever an asset's vol
series was missing, and `_ensure_vol_series` only stored a series for assets
with >=30 daily bars and swallowed every fetch error. The highest-vol
microcaps were sized at full notional during warmup windows.

METHOD — risk-parity re-sizing (state-free)
-------------------------------------------
Per-trade PnL is exactly linear in notional, and each trade's realized
per-dollar return ``r = pnl / size_usd`` is a fact (the market outcome).
The question this answers: *if these identical 714 trades had been sized
by correct inverse-vol risk parity instead of however they were actually
sized, what would the PnL have been?*

  vol_mult_i   = clip(target_vol / v_i, 0.10, 2.0)     # fixed band, fail-safe
  C            = sum(size_usd) / sum(vol_mult)         # preserves TOTAL
                                                       # gross notional —
                                                       # a pure re-weighting,
                                                       # not a leverage change
  corrected_size_i = C * vol_mult_i
  corrected_pnl_i  = (pnl_i / size_usd_i) * corrected_size_i

``v_i`` is the point-in-time 20-day realized daily vol (last daily bar
strictly before entry) — the quantity ``compute_vol_mult`` uses. Daily
klines are pulled fresh from Binance USD-M futures.

This is deliberately independent of the engine's (unknowable) runtime
vol-cache state: it does not try to reconstruct what vol_mult was actually
applied. It states cleanly what *correct* risk-parity sizing produces on
the same trades. Caveat: it is a first-order re-weighting — it does not
re-solve the exposure caps or equity compounding (both second-order; both
would push the corrected loss slightly lower).

Usage (on the homeserver — has the DB and Binance access):
    python3 scripts/counterfactual_replay_binance.py \
        --db /hdd/binance-paper/data/live/paper_binance.db
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np
import pandas as pd

FAPI_KLINES = "https://fapi.binance.com/fapi/v1/klines"

TARGET_VOL = 0.025
FLOOR, CAP = 0.10, 2.0          # fixed clip band (the sizing fix)
VOL_WINDOW = 20                  # rolling daily-return std window


def fetch_daily_klines(asset: str, retries: int = 3) -> pd.Series | None:
    """Daily close series for {asset}USDT from Binance USD-M futures."""
    url = f"{FAPI_KLINES}?symbol={asset}USDT&interval=1d&limit=1500"
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=20) as resp:
                rows = json.loads(resp.read())
            if not rows:
                return None
            idx = pd.to_datetime([r[0] for r in rows], unit="ms").as_unit("ns")
            return pd.Series([float(r[4]) for r in rows], index=idx,
                             name=asset).sort_index()
        except Exception as e:  # noqa: BLE001
            if attempt == retries - 1:
                print(f"  WARN: kline fetch failed for {asset}: {e}")
                return None
            time.sleep(1.0 + attempt)
    return None


def point_in_time_vol(vol: pd.Series, ts: pd.Timestamp) -> float:
    """20-day realized daily vol on the last daily bar strictly before ts."""
    if vol is None or len(vol) == 0:
        return np.nan
    idx = vol.index.searchsorted(ts)
    if 0 < idx <= len(vol):
        v = vol.iloc[idx - 1]
        return float(v) if pd.notna(v) else np.nan
    return np.nan


def vol_mult(v: float) -> float:
    """Inverse-vol multiplier — fails safe to the floor (matches the fixed
    compute_vol_mult)."""
    if not np.isfinite(v) or v <= 0:
        return FLOOR
    return float(np.clip(TARGET_VOL / v, FLOOR, CAP))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="/hdd/binance-paper/data/live/paper_binance.db")
    ap.add_argument("--out", default="counterfactual_replay_binance.csv")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)
    trades = pd.read_sql_query(
        "SELECT asset, direction, size_usd, pnl, entry_time, timeframe "
        "FROM positions WHERE status='closed'", con)
    con.close()
    trades["entry_ts"] = pd.to_datetime(trades["entry_time"], utc=True
                                        ).dt.tz_localize(None)
    act = trades["pnl"].sum()
    print(f"Loaded {len(trades)} closed positions, total PnL ${act:,.2f}")

    vol_by_asset: dict[str, pd.Series] = {}
    for asset in sorted(trades["asset"].unique()):
        closes = fetch_daily_klines(asset)
        if closes is not None:
            vol_by_asset[asset] = closes.pct_change().rolling(VOL_WINDOW).std()
        time.sleep(0.15)
    print(f"Fetched daily vol for {len(vol_by_asset)}/"
          f"{trades['asset'].nunique()} assets")

    v = trades["asset"].map(lambda a: None).astype(float)
    daily_vol, vmult = [], []
    for t in trades.itertuples(index=False):
        vv = point_in_time_vol(vol_by_asset.get(t.asset), t.entry_ts)
        daily_vol.append(vv)
        vmult.append(vol_mult(vv))
    trades["daily_vol"] = daily_vol
    trades["vol_mult"] = vmult
    trades["r_per_dollar"] = trades["pnl"] / trades["size_usd"]

    # Risk-parity re-size: preserve total gross notional (pure re-weighting).
    C = trades["size_usd"].sum() / trades["vol_mult"].sum()
    trades["corrected_size"] = C * trades["vol_mult"]
    trades["corrected_pnl"] = trades["r_per_dollar"] * trades["corrected_size"]
    cor = trades["corrected_pnl"].sum()

    trades.drop(columns=["entry_ts"]).to_csv(args.out, index=False)

    print("\n" + "=" * 74)
    print("COUNTERFACTUAL REPLAY — correct inverse-vol risk-parity sizing")
    print("=" * 74)
    print(f"  gross notional preserved: ${trades['size_usd'].sum():,.0f} "
          f"-> ${trades['corrected_size'].sum():,.0f}  (C=${C:,.0f})")
    print(f"  actual    total PnL : ${act:>13,.2f}")
    print(f"  corrected total PnL : ${cor:>13,.2f}")
    print(f"  loss removed        : ${cor - act:>13,.2f}  "
          f"({(cor - act) / abs(act) * 100:.1f}% of the loss)")

    print("\n  per-asset (sorted by actual PnL):")
    g = trades.groupby("asset").agg(
        n=("pnl", "size"),
        actual=("pnl", "sum"),
        corrected=("corrected_pnl", "sum"),
        avg_size=("size_usd", "mean"),
        avg_corr_size=("corrected_size", "mean"),
        avg_volmult=("vol_mult", "mean"),
    ).sort_values("actual")
    for a, r in g.iterrows():
        print(f"    {a:9s} n={int(r['n']):4d}  actual ${r['actual']:>11,.2f}"
              f"  corrected ${r['corrected']:>10,.2f}"
              f"  size ${r['avg_size']:>7,.0f}->${r['avg_corr_size']:>7,.0f}"
              f"  vm={r['avg_volmult']:.2f}")
    print(f"\n  wrote per-trade detail -> {args.out}")
    print(f"  run at {datetime.now(timezone.utc).isoformat()}")


if __name__ == "__main__":
    main()
