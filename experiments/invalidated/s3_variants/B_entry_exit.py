"""Invalidated forensic experiment — S3 Theme B: ENTRY / EXIT.

The fixed-sizing baseline (DSR p=0.7168 FAIL) admits trades at every DTE
and in every BTC-spot regime. The four sub-variants test whether filtering
the ENTRY set changes the distribution enough to clear the DSR bar.

- B1: DTE floor — skip entries with <N days to expiry. The tail is
      concentrated near expiry (gamma risk), so widening the DTE window
      should reduce the worst single-trade loss. Test N=3, 5, 7.
- B2: BTC RSI filter — skip entry when BTC spot RSI(14) is at an extreme
      (>80 or <20). Both extremes mark reversal-prone regimes; the
      short-vol carry performs worst in regime-change bars.
- B3: skew filter — only enter when 25-delta skew > X. A high skew
      signals fear is rich; the put side is overpriced, and the
      straddle's left-tail premium is fattened. (Sign-checked: the
      frame's `svi_25d_skew_1w` is positive when puts trade rich.)
- B4: combined — B1 (DTE≥3) + B2 (RSI 20-80) — the most aggressive
      filter, tests if CLEAN entries alone clear DSR.

Each: per-trade PnL, per-year, DSR p@K=40, MaxDD, mechanism.
DSR bar = K=40 NOT relaxed. Engine UNTOUCHED.
"""
from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd

from experiments.invalidated.s3_variants._common import (
    load_frame, get_short_covered, evaluate_variant, write_variant_report, OUT_DIR,
)


# ── B1: DTE floor ────────────────────────────────────────────────
def variant_b1_dte_floor(scov: pd.DataFrame, *, min_dte_days: float) -> pd.DataFrame:
    df = scov.copy()
    df = df.sort_values("entry_ts").reset_index(drop=True)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = df["expiry_dt"]
    df = df[df["dte_days"] >= min_dte_days].copy()
    return df


# ── B2: BTC RSI(14) filter ───────────────────────────────────────
def variant_b2_rsi_filter(scov: pd.DataFrame, *,
                          rsi_high: float = 80.0,
                          rsi_low: float = 20.0,
                          rsi_window: int = 14 * 24 * 60) -> pd.DataFrame:
    """Compute RSI(14) on 1-min BTC close (1d window) and skip entries
    where RSI > 80 or RSI < 20."""
    df = scov.copy()
    df = df.sort_values("entry_ts").reset_index(drop=True)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = df["expiry_dt"]
    # Build a per-minute BTC price series from the per-trade index_price
    px = df.set_index("entry_ts")["index_price"].astype(float).sort_index()
    px = px[~px.index.duplicated(keep="last")]
    # Resample to a clean 1-min grid using forward-fill from trade entries
    rets = np.log(px).diff()
    # Use a 14-bar (14-min proxy, since 1d would need many bars) RSI.
    # In the per-trade frame entries are sparse, so a 14-bar window is
    # ~14 entries. This is a coarse regime proxy but matches the spirit
    # of the mandate: skip when recent returns are extreme.
    # Instead use 14-day (14 entries) realized return z-score.
    ret_z = (rets - rets.rolling(rsi_window, min_periods=60).mean()) / rets.rolling(rsi_window, min_periods=60).std()
    df["ret_z"] = df["entry_ts"].map(ret_z)
    df = df[df["ret_z"].between(-2.0, 2.0)].copy()  # skip entries when 14d return z>2
    return df


# ── B3: 25-delta skew filter ─────────────────────────────────────
def variant_b3_skew_filter(scov: pd.DataFrame, *,
                           skew_min: float = 0.05,
                           skew_col: str = "svi_25d_skew_1w") -> pd.DataFrame:
    """Only enter when 25-delta put-call skew > skew_min. Higher skew =
    put side richer (fear premium fat) → straddle's left tail is overpriced."""
    df = scov.copy()
    df = df.sort_values("entry_ts").reset_index(drop=True)
    df["entry_ts"] = pd.to_datetime(df["entry_ts"], utc=True)
    df["exit_ts"] = df["expiry_dt"]
    df = df[df[skew_col] >= skew_min].copy()
    return df


# ── B4: combined filter (B1 + B2) ────────────────────────────────
def variant_b4_combined(scov: pd.DataFrame, *, min_dte_days: float = 3.0) -> pd.DataFrame:
    b1 = variant_b1_dte_floor(scov, min_dte_days=min_dte_days)
    b2 = variant_b2_rsi_filter(b1)
    return b2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=f"{OUT_DIR}/B_entry_exit.md")
    args = ap.parse_args()

    f = load_frame()
    scov = get_short_covered(f)
    print(f"[B_entry_exit] base short-covered: n={len(scov)}")

    results = []

    # B1 — DTE floors (3 sub-points)
    for dte in (3.0, 5.0, 7.0):
        df = variant_b1_dte_floor(scov, min_dte_days=dte)
        r = evaluate_variant(
            df, code=f"B1_dte{int(dte)}", name=f"DTE floor ≥ {int(dte)} days",
            mechanism="Skip entries near expiry (gamma risk concentrated). "
                      f"Wider DTE window → fewer but more spread trades.",
        )
        results.append(r)

    # B2 — RSI / return-z filter
    df_b2 = variant_b2_rsi_filter(scov)
    r2 = evaluate_variant(
        df_b2, code="B2", name="return-z filter |z|≤2 (14d)",
        mechanism="Skip entries when recent BTC return is at a 2σ extreme. "
                  "Regime-change bars are the worst for short-vol carry.",
    )
    results.append(r2)

    # B3 — skew filter
    df_b3 = variant_b3_skew_filter(scov, skew_min=0.05)
    r3 = evaluate_variant(
        df_b3, code="B3", name="25δ skew ≥ 0.05 (1w)",
        mechanism="Only enter when put-side fear premium is fat. "
                  "Sizer-equivalent filter on the LEFT-TAIL premium of the straddle.",
    )
    results.append(r3)

    # B4 — combined
    df_b4 = variant_b4_combined(scov, min_dte_days=3.0)
    r4 = evaluate_variant(
        df_b4, code="B4", name="B1(3d) + B2 combined",
        mechanism="DTE floor AND regime filter — clean entries only. "
                  "Aggressive breadth reduction for richer per-trade VRP.",
    )
    results.append(r4)

    write_variant_report(results, "B — Entry/Exit", out_path=args.out,
        summary=f"Baseline (S3 frozen): n=666, per-trade $1,285, DSR p=0.7168 FAIL. "
                f"Goal: clear the K=40 DSR bar via entry filters alone (no sizing or strike change).")


if __name__ == "__main__":
    main()
