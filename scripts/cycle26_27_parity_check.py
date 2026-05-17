#!/usr/bin/env python
"""Post-cycle-26/27/28 live↔offline bar-by-bar parity check.

Compares live engine output (deribit_paper state/live_metrics.parquet)
against the offline V2 backtester / freshly-extended predictions on bars
where cycle-26 + cycle-27 + cycle-28 fixes were all active live.

Cycle-28 deploy + engine restart: 2026-05-12 23:52 UTC.
First bucket-boundary recal with full trade window: 01:00 UTC.
=> Clean comparison window: 2026-05-13 01:00 onwards.

Live data sources:
  - live_metrics.parquet (rv_pred_pct, iv_used_pct, signal) — up to 09:45.
  - JSONL signal log — fills the 10:00–11:45 gap (SHORT bars carry
    rv_pred / iv inline; FLAT bars only have edge).

Offline data sources:
  - vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions.parquet —
    raw LGBM y_pred (must be β*-adjusted to match live's aligned rv_pred).
  - vol_arb/output/BTC_vol_arb_svi.parquet — svi_iv_matched (offline iv).
  - vol_model/output/aligner_betas_by_month.json — β*_2026-05 = 0.0366.

Output: per-bar table + summary stats.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd


# ─────────────────────────────────────────────────────────────────────


def load_live_metrics(path: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.set_index("timestamp").sort_index()
    return df[["signal", "edge_vol_pts", "rv_pred_pct", "iv_used_pct"]].rename(
        columns={
            "edge_vol_pts": "edge_live",
            "rv_pred_pct": "rv_live",
            "iv_used_pct": "iv_live",
            "signal": "signal_live",
        }
    )


def load_jsonl_signals(path: str) -> pd.DataFrame:
    """Parse 'Signal @ ...' lines from the live JSONL log.

    Cycle-31+ era log formats:
      - SHORT: ``rv_pred (XX.XX%) < iv (YY.YY%) by -ZZ.ZZvp (eff_thresh=...)``
      - LONG (rare): same shape but ``>`` and positive edge.
      - FLAT short-side: ``edge (+/-X.XXXvp) within +13.00/-21.00``
      - FLAT contango-filter: ``LONG candidate edge=+X.XXXvp suppressed`` etc.
    All forms include the edge; rv/iv only on SHORT/LONG.
    """
    rv_re = re.compile(
        r"Signal @ (\S+ \S+\+00:00): (\w+).*?rv_pred \(([\d.]+)%\)"
        r" [<>] iv \(([\d.]+)%\) by (-?[\d.]+)vp",
        re.S,
    )
    edge_re = re.compile(
        r"Signal @ (\S+ \S+\+00:00): (\w+).*?edge[ =]\(?([+\-]?[\d.]+)vp",
        re.S,
    )
    rows: list[dict] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                obj = json.loads(line)
                msg = obj.get("msg", "")
                if "Signal @" not in msg:
                    continue
                m = rv_re.search(msg)
                if m:
                    ts, sig, rv, iv, edge = m.groups()
                    rows.append({
                        "timestamp": ts, "signal_str": sig,
                        "rv_live": float(rv), "iv_live": float(iv),
                        "edge_live": float(edge),
                    })
                    continue
                m = edge_re.search(msg)
                if m:
                    ts, sig, edge = m.groups()
                    rows.append({
                        "timestamp": ts, "signal_str": sig,
                        "rv_live": np.nan, "iv_live": np.nan,
                        "edge_live": float(edge),
                    })
            except Exception:
                continue
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df["signal_live"] = df["signal_str"].map({"SHORT": -1, "FLAT": 0, "LONG": 1})
    df = df.set_index("timestamp").sort_index()
    # Deduplicate: dual-log format (both loggers fire the same line) →
    # second copy. Keep first.
    df = df[~df.index.duplicated(keep="first")]
    return df[["signal_live", "edge_live", "rv_live", "iv_live"]]


def load_offline(
    preds_path: str, volarb_path: str, betas_path: str,
) -> pd.DataFrame:
    preds = pd.read_parquet(preds_path)
    preds.index = pd.to_datetime(preds.index, utc=True)
    # Apply per-month β* aligner correction.
    with open(betas_path, encoding="utf-8") as f:
        betas = json.load(f)
    months = preds.index.strftime("%Y-%m")
    beta_per_bar = months.map(betas).astype(float)
    preds["y_pred_aligned"] = preds["y_pred"] + beta_per_bar
    preds["rv_offline"] = np.exp(preds["y_pred_aligned"]) * 100.0

    va = pd.read_parquet(volarb_path)
    if not isinstance(va.index, pd.DatetimeIndex):
        va.index = pd.to_datetime(va.index, utc=True)
    # IV chain matches backtester.py:1146: svi_iv_matched → iv → dvol_close
    iv_cols = [c for c in ["svi_iv_matched", "iv", "dvol_close"] if c in va.columns]
    iv = va[iv_cols[0]].copy()
    for c in iv_cols[1:]:
        iv = iv.fillna(va[c])
    iv.name = "iv_offline"

    out = preds[["rv_offline"]].join(iv, how="left")
    out["edge_offline"] = out["rv_offline"] - out["iv_offline"]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", default="C:/tmp/live_metrics_latest.parquet")
    ap.add_argument("--jsonl-remote-host", default="monitor",
                    help="ssh host for JSONL extraction (uses /hdd/.../deribit_paper.jsonl)")
    ap.add_argument("--preds",
                    default="vol_model/output/BTCUSDT_pruned_svi_har_pd_predictions.parquet")
    ap.add_argument("--volarb",
                    default="vol_arb/output/BTC_vol_arb_svi.parquet")
    ap.add_argument("--betas",
                    default="vol_model/output/aligner_betas_by_month.json")
    ap.add_argument("--start", default="2026-05-13 01:00:00+00:00")
    ap.add_argument("--end", default="2026-05-13 12:00:00+00:00")
    args = ap.parse_args()

    start_ts = pd.Timestamp(args.start)
    end_ts = pd.Timestamp(args.end)

    # ── Live ──
    live_metrics = load_live_metrics(args.live)
    live_metrics_w = live_metrics[
        (live_metrics.index >= start_ts) & (live_metrics.index <= end_ts)
    ]
    print(f"Live metrics rows in window: {len(live_metrics_w)}")

    # JSONL — fetch a local snapshot via SSH
    jsonl_local = Path("C:/tmp/deribit_paper_jsonl_snapshot.jsonl")
    if not jsonl_local.exists() or jsonl_local.stat().st_size < 1024:
        print(f"(JSONL snapshot not present at {jsonl_local}; live data from "
              "live_metrics.parquet only)")
        live_jsonl_w = pd.DataFrame()
    else:
        live_jsonl = load_jsonl_signals(str(jsonl_local))
        live_jsonl_w = live_jsonl[
            (live_jsonl.index >= start_ts) & (live_jsonl.index <= end_ts)
        ]
        print(f"JSONL signal rows in window: {len(live_jsonl_w)}")

    # Merge: prefer live_metrics (has rv/iv for all bars), fall back to JSONL.
    if not live_jsonl_w.empty:
        merged_live = live_metrics_w.combine_first(live_jsonl_w)
    else:
        merged_live = live_metrics_w
    print(f"Live merged rows: {len(merged_live)}")

    # ── Offline ──
    offline = load_offline(args.preds, args.volarb, args.betas)
    offline_w = offline[(offline.index >= start_ts) & (offline.index <= end_ts)]
    print(f"Offline rows in window: {len(offline_w)}")

    # ── Join ──
    joined = merged_live.join(offline_w, how="inner")
    if joined.empty:
        print("\nNo overlap between live and offline windows.")
        return 1
    joined["d_rv"] = joined["rv_live"] - joined["rv_offline"]
    joined["d_iv"] = joined["iv_live"] - joined["iv_offline"]
    joined["d_edge"] = joined["edge_live"] - joined["edge_offline"]

    # ── Per-bar table ──
    show_cols = ["signal_live", "rv_live", "rv_offline", "d_rv",
                 "iv_live", "iv_offline", "d_iv",
                 "edge_live", "edge_offline", "d_edge"]
    cols_present = [c for c in show_cols if c in joined.columns]
    pd.set_option("display.precision", 3)
    pd.set_option("display.width", 200)
    print("\n=== Per-bar comparison ===")
    print(joined[cols_present].to_string())

    # ── Summary ──
    print("\n=== Summary ===")
    valid_rv = joined["d_rv"].dropna()
    valid_iv = joined["d_iv"].dropna()
    print(f"Bars: {len(joined)} (live & offline)")
    if len(valid_rv):
        print(f"drv_pred (vp): mean={valid_rv.mean():+.3f}  abs.mean={valid_rv.abs().mean():.3f}  "
              f"max_abs={valid_rv.abs().max():.3f}  std={valid_rv.std():.3f}")
        for target in (0.5, 1.0, 2.0):
            n = int((valid_rv.abs() < target).sum())
            print(f"  |drv| < {target}vp: {n}/{len(valid_rv)} ({100*n/len(valid_rv):.1f}%)")
    if len(valid_iv):
        print(f"div      (vp): mean={valid_iv.mean():+.3f}  abs.mean={valid_iv.abs().mean():.3f}  "
              f"max_abs={valid_iv.abs().max():.3f}  std={valid_iv.std():.3f}")
        for target in (0.5, 1.0, 2.0):
            n = int((valid_iv.abs() < target).sum())
            print(f"  |div| < {target}vp: {n}/{len(valid_iv)} ({100*n/len(valid_iv):.1f}%)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
