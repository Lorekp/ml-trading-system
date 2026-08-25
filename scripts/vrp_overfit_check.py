"""Anti-overfit stress test for the SHORT-only delta-hedged to-expiry VRP carry.

Uses the verifier's cached per-band frames (short-only marks already computed at
bands 0.20/0.35/0.75) so NO engine re-run. Answers three overfit questions:
  A. PARAMETER robustness  — is short-only positive across the whole band plateau,
     per year, on overlap-adjusted Sharpe (not the in-sample argmax)?
  B. TRUE OOS time split    — choose the band on TRAIN years (2022-2023) only, then
     report a HELD-OUT test (2024-2026) never used to pick the band.
  C. GROUND-TRUTH cross-check — is the carry present in the RAW real-fill option leg
     (real_option_pnl_nc), independent of the engine's hedge math?

If the carry survives all three it is structurally robust, not a fitted artifact.
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from scripts.longhold_hedged_vrp import _sharpe_iid, _newey_west_sharpe, _nonoverlap_sharpe, _max_drawdown

BANDS = {0.20: "verify_frame_band20.parquet",
         0.35: "verify_frame_band35.parquet",
         0.75: "verify_frame_band75.parquet"}
DIR = "vol_arb/output/"


def load_short(band_file: str) -> pd.DataFrame:
    f = pd.read_parquet(DIR + band_file)
    f = f[(f["premium_tier"] > 0) & (f["signal"] < 0)].copy()   # SHORT-only, covered
    f["entry_ts"] = pd.to_datetime(f["entry_ts"], utc=True)
    f["exit_ts"] = pd.to_datetime(f["expiry_dt"], utc=True)
    f = f.sort_values("entry_ts").reset_index(drop=True)
    f["year"] = f["entry_ts"].dt.year
    return f


def metrics(pnl, entry, exit):
    entry = pd.DatetimeIndex(entry); exit = pd.DatetimeIndex(exit)
    nw = _newey_west_sharpe(pnl, entry, exit)
    nonov, nk = _nonoverlap_sharpe(pnl, entry, exit)
    return dict(total=float(np.nansum(pnl)), n=len(pnl),
                naive=_sharpe_iid(pnl), nw=nw, nonov=nonov, nkeep=nk,
                maxdd=_max_drawdown(pnl), worst=float(np.nanmin(pnl)) if len(pnl) else np.nan)


print("#" * 80)
print("A. PARAMETER ROBUSTNESS — SHORT-only across the band plateau (engine hedged PnL)")
print("#" * 80)
frames = {b: load_short(fn) for b, fn in BANDS.items()}
for b, f in frames.items():
    m = metrics(f["total_hedged_usd"].to_numpy(float), f["entry_ts"], f["exit_ts"])
    py = {int(y): float(f.loc[f.year == y, "total_hedged_usd"].sum()) for y in sorted(f.year.unique())}
    pos = sum(1 for v in py.values() if v > 0)
    print(f"\nband {b:.2f}: n={m['n']}  total ${m['total']:,.0f}  NW {m['nw']:.2f}  "
          f"nonov {m['nonov']:.2f}(n={m['nkeep']})  MaxDD ${m['maxdd']:,.0f}  worst ${m['worst']:,.0f}")
    print(f"   per-year ({pos}/{len(py)} +): " + ", ".join(f"{y}:${v:,.0f}" for y, v in py.items()))

print("\n" + "#" * 80)
print("B. TRUE OUT-OF-SAMPLE — pick band on TRAIN(2022-2023), report HELD-OUT(2024-2026)")
print("#" * 80)
TRAIN_YEARS = {2022, 2023}
# Choose band by best TRAIN Newey-West Sharpe (train years ONLY).
train_nw = {}
for b, f in frames.items():
    tr = f[f.year.isin(TRAIN_YEARS)]
    if len(tr) >= 3:
        train_nw[b] = _newey_west_sharpe(tr["total_hedged_usd"].to_numpy(float),
                                         pd.DatetimeIndex(tr["entry_ts"]), pd.DatetimeIndex(tr["exit_ts"]))
print("TRAIN(2022-2023) NW Sharpe by band:", {b: round(v, 3) for b, v in train_nw.items()})
best_band = max(train_nw, key=train_nw.get)
print(f"--> band chosen ON TRAIN ONLY = {best_band:.2f}")
fbest = frames[best_band]
te = fbest[~fbest.year.isin(TRAIN_YEARS)]
mt = metrics(te["total_hedged_usd"].to_numpy(float), te["entry_ts"], te["exit_ts"])
pyt = {int(y): float(te.loc[te.year == y, "total_hedged_usd"].sum()) for y in sorted(te.year.unique())}
print(f"HELD-OUT(2024-2026) @band {best_band:.2f}: n={mt['n']}  total ${mt['total']:,.0f}  "
      f"NW {mt['nw']:.2f}  nonov {mt['nonov']:.2f}(n={mt['nkeep']})  MaxDD ${mt['maxdd']:,.0f}")
print("   held-out per-year: " + ", ".join(f"{y}:${v:,.0f}" for y, v in pyt.items()))
# Robustness: ALL bands on held-out (does the choice even matter?)
print("   held-out total at EVERY band (choice-insensitivity):")
for b, f in frames.items():
    te2 = f[~f.year.isin(TRAIN_YEARS)]
    print(f"      band {b:.2f}: ${float(te2['total_hedged_usd'].sum()):,.0f}  "
          f"NW {_newey_west_sharpe(te2['total_hedged_usd'].to_numpy(float), pd.DatetimeIndex(te2['entry_ts']), pd.DatetimeIndex(te2['exit_ts'])):.2f}")

print("\n" + "#" * 80)
print("C. GROUND-TRUTH — raw real-fill option leg (real_option_pnl_nc), SHORT-only, hedge-math-independent")
print("#" * 80)
# real_option_pnl_nc is band-independent (the raw real-fill premium leg); take band20 frame.
f0 = frames[0.20]
# Convert BTC option pnl to USD-ish via index for comparability; if already USD leave it.
rp = f0["real_option_pnl_nc"].to_numpy(float)
# Sign note: real_option_pnl_nc is the honest_pnl ground-truth for the SHORT premium leg.
m = metrics(rp, f0["entry_ts"], f0["exit_ts"])
py = {int(y): float(f0.loc[f0.year == y, "real_option_pnl_nc"].sum()) for y in sorted(f0.year.unique())}
pos = sum(1 for v in py.values() if v > 0)
print(f"raw real-fill SHORT premium leg: n={m['n']}  total {m['total']:,.2f}  NW {m['nw']:.2f}  "
      f"nonov {m['nonov']:.2f}(n={m['nkeep']})")
print(f"   per-year ({pos}/{len(py)} +): " + ", ".join(f"{y}:{v:,.0f}" for y, v in py.items()))
print("   (positive & multi-year-stable here = the VRP carry is in the RAW real fills, "
      "not manufactured by the hedge model.)")

print("\n" + "=" * 80)
print("READ: plateau-positive (A) + held-out-positive with train-chosen band (B) + present in "
      "raw real fills (C) => structural VRP carry, not a fitted artifact. Any FAIL => overfit risk.")
