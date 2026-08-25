"""Diagnose cache vs slow-path per-trade PnL drift.

Runs scripts/s3_revalidate.py --reeval TWICE on the FROZEN_CFG window:
  (a) cache DISABLED (rename _window_FROZEN + _tape_FROZEN to *.bak)
  (b) cache ENABLED (rename back)

For each run, captures the per-fold per-trade frame (S3_revalidation_fold_*.parquet)
for the held-out fold (2026H1 / label "2026H1"). The per-fold parquet contains
the trade_log with total_hedged_usd + legs for each covered short trade.

Diffs the per-trade PnL row-by-row. Finds the FIRST diverging trade and
extracts all contributing fields.

Usage:
    PYTHONPATH=. .venv/Scripts/python.exe scripts/_diagnose_cache_drift.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

REVAL = _REPO / "scripts" / "s3_revalidate.py"
KLINE_CACHE = _REPO / "data" / "binance-1m" / "BTCUSDT" / "_window_FROZEN.parquet"
TAPE_CACHE = _REPO / "data" / "deribit" / "_tape_FROZEN.parquet"
OUT_DIR = _REPO / "vol_arb" / "output"
DIAG_DIR = _REPO / "vol_arb" / "output" / "_cache_drift"

# Per-fold parquet s3_revalidate.py writes. We want the held-out OOS fold
# (the spec calls it "2026H1"; the parquet stem replaces space).
FOLD_PARQUET = OUT_DIR / "S3_revalidation_fold_2026H1.parquet"


def _run_with_cache(cache_enabled: bool, run_dir: Path) -> dict:
    """Run s3_revalidate.py --reeval with cache enabled or disabled.

    Restores renamed cache files in finally. Returns dict with the captured
    per-fold frame path + wall_seconds.
    """
    bak = []
    try:
        if not cache_enabled:
            for p in (KLINE_CACHE, TAPE_CACHE):
                if p.exists():
                    bp = p.with_suffix(p.suffix + ".bak")
                    shutil.move(str(p), str(bp))
                    bak.append((p, bp))
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, str(REVAL), "--reeval"],
            cwd=str(_REPO),
            env={**__import__("os").environ, "PYTHONPATH": str(_REPO)},
            capture_output=True, text=True, timeout=1800,
        )
        wall = time.time() - t0
        if proc.returncode != 0:
            print(f"[diag] run (cache={'on' if cache_enabled else 'off'}) FAILED rc={proc.returncode}")
            print(proc.stdout[-3000:])
            print(proc.stderr[-3000:])
            return dict(cache_enabled=cache_enabled, wall=wall, error=f"rc={proc.returncode}")
        # Copy the held-out fold parquet to the diag dir so the OFF run doesn't
        # overwrite the ON run's file (they share the same output path).
        if FOLD_PARQUET.exists():
            run_path = run_dir / FOLD_PARQUET.name
            shutil.copy(str(FOLD_PARQUET), str(run_path))
            return dict(cache_enabled=cache_enabled, wall=wall, fold_path=run_path)
        return dict(cache_enabled=cache_enabled, wall=wall, error="fold parquet missing")
    finally:
        for orig, backup in bak:
            if backup.exists() and not orig.exists():
                shutil.move(str(backup), str(orig))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--diag-dir", default=str(DIAG_DIR))
    args = ap.parse_args()

    diag_dir = Path(args.diag_dir)
    diag_dir.mkdir(parents=True, exist_ok=True)

    if not KLINE_CACHE.exists() or not TAPE_CACHE.exists():
        print(f"[diag] cache missing — run scripts/_build_window_cache.py first")
        return 2

    print("[diag] run cache ON (live path)...")
    on = _run_with_cache(cache_enabled=True, run_dir=diag_dir / "ON")
    print(f"[diag] ON wall={on.get('wall', 0):.1f}s fold={on.get('fold_path')}")
    print("[diag] run cache OFF (slow path)...")
    off = _run_with_cache(cache_enabled=False, run_dir=diag_dir / "OFF")
    print(f"[diag] OFF wall={off.get('wall', 0):.1f}s fold={off.get('fold_path')}")

    if "fold_path" not in on or "fold_path" not in off:
        print("[diag] one or both runs failed; aborting diff")
        return 1

    on_df = pd.read_parquet(on["fold_path"])
    off_df = pd.read_parquet(off["fold_path"])
    print(f"[diag] ON rows={len(on_df)}  OFF rows={len(off_df)}")

    # Same outer index? s3_revalidate sets index to original trade-log index
    # (entry_ts is a column, but the index is preserved). Compare on
    # (entry_ts, strike, expiry_dt) as a natural key.
    key_cols = ["entry_ts", "strike", "expiry_dt"]
    on_k = list(zip(on_df["entry_ts"].astype("int64").tolist(),
                    on_df["strike"].astype(str).tolist(),
                    on_df["expiry_dt"].astype("int64").tolist()))
    off_k = list(zip(off_df["entry_ts"].astype("int64").tolist(),
                     off_df["strike"].astype(str).tolist(),
                     off_df["expiry_dt"].astype("int64").tolist()))
    on_set, off_set = set(on_k), set(off_k)
    only_on = on_set - off_set
    only_off = off_set - on_set
    print(f"[diag] only in ON: {len(only_on)}, only in OFF: {len(only_off)}")

    # Align on the common key. Pick the FIRST index order from OFF (slow path
    # is the source of truth).
    common = on_set & off_set
    on_common = on_df[[tuple(r) in common for r in on_k]].copy()
    off_common = off_df[[tuple(r) in common for r in off_k]].copy()
    on_common["_k"] = list(zip(on_common["entry_ts"].astype("int64"),
                               on_common["strike"].astype(str),
                               on_common["expiry_dt"].astype("int64")))
    off_common["_k"] = list(zip(off_common["entry_ts"].astype("int64"),
                                off_common["strike"].astype(str),
                                off_common["expiry_dt"].astype("int64")))
    off_common = off_common.sort_values("_k").reset_index(drop=True)
    on_common = on_common.set_index("_k").loc[off_common["_k"].tolist()].reset_index()

    # Compare per-trade PnL and legs.
    pnl_cols = ["total_hedged_usd", "total_unhedged_usd", "gamma_pnl_usd",
                "option_premium_usd", "option_fee_usd", "hedge_fee_usd",
                "funding_usd", "perp_pnl_usd"]
    pnl_cols = [c for c in pnl_cols if c in on_common.columns and c in off_common.columns]

    diff_rows = []
    for c in pnl_cols:
        a = pd.to_numeric(on_common[c], errors="coerce").to_numpy()
        b = pd.to_numeric(off_common[c], errors="coerce").to_numpy()
        ne = ~np.isclose(a, b, rtol=1e-9, atol=1e-6, equal_nan=True)
        n_diff = int(ne.sum())
        max_abs = float(np.nanmax(np.abs(a[ne] - b[ne]))) if n_diff else 0.0
        print(f"[diag] {c:25s}: {n_diff:4d}/{len(a)} differ, max |Δ|={max_abs:.6f}")
        diff_rows.append(dict(col=c, n_diff=n_diff, max_abs_delta=max_abs))

    # Find FIRST diverging trade and dump full per-leg breakdown for it.
    # Sort by entry_ts to get a stable "first" definition.
    off_sorted = off_common.sort_values(["entry_ts", "strike", "expiry_dt"]).reset_index(drop=True)
    on_sorted = on_common.set_index(
        list(zip(off_common["entry_ts"].astype("int64"),
                 off_common["strike"].astype(str),
                 off_common["expiry_dt"].astype("int64")))
    ).loc[list(zip(off_sorted["entry_ts"].astype("int64"),
                   off_sorted["strike"].astype(str),
                   off_sorted["expiry_dt"].astype("int64")))].reset_index(drop=True)

    first_div_idx = None
    for c in pnl_cols:
        a = pd.to_numeric(on_sorted[c], errors="coerce").to_numpy()
        b = pd.to_numeric(off_sorted[c], errors="coerce").to_numpy()
        ne = ~np.isclose(a, b, rtol=1e-9, atol=1e-6, equal_nan=True)
        if ne.any():
            first_div_idx = int(np.where(ne)[0][0])
            first_div_col = c
            break

    summary = {
        "on_wall_seconds": on.get("wall"),
        "off_wall_seconds": off.get("wall"),
        "speedup": (off.get("wall") or 0) / max(on.get("wall") or 1, 1e-9),
        "n_on": len(on_df),
        "n_off": len(off_df),
        "n_common": len(common),
        "n_only_on": len(only_on),
        "n_only_off": len(only_off),
        "pnl_diff_rows": diff_rows,
        "first_div_idx": first_div_idx,
        "first_div_col": first_div_col if first_div_idx is not None else None,
    }
    if first_div_idx is not None:
        row_off = off_sorted.iloc[first_div_idx]
        row_on = on_sorted.iloc[first_div_idx]
        summary["first_div_trade"] = {
            "entry_ts": str(row_off["entry_ts"]),
            "strike": str(row_off["strike"]),
            "expiry_dt": str(row_off["expiry_dt"]),
            "exit_ts": str(row_off.get("exit_ts")) if "exit_ts" in row_off else None,
        }
        # Dump full per-leg for the first diverging trade from both paths.
        cols = [c for c in on_sorted.columns
                if c not in ("_k",)]
        keep = ["entry_ts", "strike", "expiry_dt", "exit_ts", "signal",
                "N_contracts_nc", "index_price",
                "entry_call_real", "entry_put_real", "exit_call_real", "exit_put_real",
                "premium_tier", "real_option_pnl_nc",
                "entry_call_fill", "entry_put_fill", "exit_call_fill", "exit_put_fill",
                "premium_in_btc", "premium_out_btc", "effective_half_spread",
                "maker_entry", "maker_exit", "fill_tier",
                "gamma_pnl_usd", "perp_pnl_usd", "hedge_fee_usd", "funding_usd",
                "n_rebalances", "n_funding_stamps", "n_path_bars",
                "option_premium_usd", "option_fee_usd",
                "total_unhedged_usd", "total_hedged_usd",
                "iv_used", "delta", "gamma", "vega", "T_years"]
        keep = [c for c in keep if c in on_sorted.columns]
        on_row = row_on[keep].to_dict()
        off_row = row_off[keep].to_dict()
        summary["first_div_trade"]["ON"] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else
                                                  (None if (isinstance(v, float) and not np.isfinite(v)) else
                                                   str(v) if hasattr(v, "isoformat") else v))
                                            for k, v in on_row.items()}
        summary["first_div_trade"]["OFF"] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else
                                                   (None if (isinstance(v, float) and not np.isfinite(v)) else
                                                    str(v) if hasattr(v, "isoformat") else v))
                                             for k, v in off_row.items()}

    out = diag_dir / "diagnostic.json"
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, default=str)
    print(f"[diag] summary -> {out}")
    if first_div_idx is not None:
        print(f"[diag] first divergence at idx {first_div_idx} col {first_div_col}")
        ft = summary["first_div_trade"]
        print(f"[diag] trade: entry_ts={ft['entry_ts']} strike={ft['strike']} expiry_dt={ft['expiry_dt']}")
        for leg in ("total_hedged_usd", "gamma_pnl_usd", "hedge_fee_usd",
                    "funding_usd", "option_premium_usd", "option_fee_usd",
                    "entry_call_real", "entry_put_real", "exit_call_real", "exit_put_real",
                    "entry_call_fill", "entry_put_fill", "exit_call_fill", "exit_put_fill",
                    "premium_in_btc", "premium_out_btc", "maker_entry", "maker_exit",
                    "real_option_pnl_nc", "iv_used"):
            ov = ft["OFF"].get(leg)
            nv = ft["ON"].get(leg)
            if ov != nv:
                print(f"  {leg:25s}  OFF={ov}  ON={nv}  Δ={((nv or 0) - (ov or 0)) if isinstance(ov, (int, float)) and isinstance(nv, (int, float)) else '?'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
