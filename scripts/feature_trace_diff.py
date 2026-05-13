#!/usr/bin/env python
"""Cycle-35 (2026-05-13): live↔offline feature-trace divergence comparator.

Inputs:
  - ``--live``    path to the live engine's ``bar_trace_live.jsonl``
                  (emitted by :class:`BarTraceWriter` when
                  ``cfg.feature_trace_path`` is set).
  - ``--offline`` path to the merged offline trace produced by
                  ``scripts/build_offline_bar_trace.py``.

Algorithm:
  1. Inner-join on ``bar_close_ts``.
  2. For each bar, walk checkpoints C0 → C8 in order, computing per-
     field |Δ|. Flag the FIRST checkpoint where any field exceeds
     :data:`~deribit_paper.diagnostics.bar_trace.TOLERANCES`.
  3. Emit
        (a) a per-bar table (timestamp, first_divergent_ckpt,
            divergent_fields, max_abs_delta, summary string),
        (b) an aggregate histogram (count of bars whose first divergence
            is at each checkpoint C0..C8 / 'clean'),
        (c) a worst-offender drilldown — for the bar with the largest
            ``|Δedge|``, print the full side-by-side at every checkpoint.

Usage:
    python scripts/feature_trace_diff.py \
        --live deribit_paper/logs/bar_trace_live.jsonl \
        --offline vol_arb/output/bar_trace_offline.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

# Import the tolerance dict so live/offline/comparator share one source of truth.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from deribit_paper.diagnostics.bar_trace import TOLERANCES  # noqa: E402


# ── Field → checkpoint grouping (drives the walk order) ─────────────


CHECKPOINTS: dict[str, list[str]] = {
    "c0": [
        "index_price_close", "bar_open_ts", "n_perp_trades",
        "n_binance_fut_trades", "bar_volume_usd",
        "dollar_bars_in_tsrv_window",
    ],
    "c1": [
        "tsrv", "rv_fast", "rv_slow", "bpv", "medrv",
        "continuous_var", "jump_var", "jump_fraction",
        "vol_ann_continuous", "vol_ann",
    ],
    "c2": [
        "svi_bucket_ts", "svi_expiry_dt",
        "svi_a", "svi_b", "svi_rho", "svi_m", "svi_sigma",
        "svi_T_years", "svi_F", "svi_rmse", "svi_level", "svi_n_strikes",
    ],
    "c3": [
        "atm_strike", "svi_iv_at_atm_pct", "fallback_iv_pct",
        "fallback_dvol_pct", "iv_used_pct", "iv_source",
    ],
    "c4": ["feature_vector"],  # special: dict diff
    "c5": ["transvol_embedding", "transvol_status"],
    "c6": ["y_pred_raw", "lgbm_model_sha256"],
    "c7": [
        "aligner_month_key", "beta_star", "y_pred_aligned", "rv_pred_pct",
    ],
    "c8": [
        "edge_vol_pts", "base_spread_iv", "spread_jump_k", "half_spread",
        "effective_long_thresh", "effective_short_thresh",
        "signal_raw", "signal_final",
    ],
}


def _load_jsonl_as_df(path: Path) -> pd.DataFrame:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["bar_close_ts"] = pd.to_datetime(df["bar_close_ts"], utc=True)
    return df.sort_values("bar_close_ts").reset_index(drop=True)


def _scalar_diff(
    live_val: object,
    off_val: object,
    rtol: float,
    atol: float,
) -> tuple[bool, float | None]:
    """Return (diverges, |delta|). String/None values compared as equality."""
    if live_val is None and off_val is None:
        return False, None
    # Strings (or one-side-None compared to a non-None string): exact match.
    if isinstance(live_val, str) or isinstance(off_val, str):
        return live_val != off_val, None
    try:
        lv = float(live_val) if live_val is not None else None
        ov = float(off_val) if off_val is not None else None
    except (TypeError, ValueError):
        return live_val != off_val, None
    if lv is None or ov is None:
        return True, None
    delta = abs(lv - ov)
    threshold = atol + rtol * max(abs(lv), abs(ov))
    return delta > threshold, delta


def _vec_diff(
    live_val: object,
    off_val: object,
    rtol: float,
    atol: float,
) -> tuple[bool, float | None, list[str]]:
    """Compare a list (TransVol embedding) or dict (feature vector).

    Returns (diverges, max_abs_delta, divergent_keys).
    """
    if live_val is None and off_val is None:
        return False, None, []
    if live_val is None or off_val is None:
        return True, None, ["entire vector"]

    # dict → element-wise diff
    if isinstance(live_val, dict) and isinstance(off_val, dict):
        keys = set(live_val.keys()) | set(off_val.keys())
        divergent: list[str] = []
        max_d = 0.0
        for k in keys:
            lv = live_val.get(k)
            ov = off_val.get(k)
            d, delta = _scalar_diff(lv, ov, rtol, atol)
            if d:
                divergent.append(k)
            if delta is not None and delta > max_d:
                max_d = delta
        return bool(divergent), max_d, divergent

    # list → index-wise diff
    if isinstance(live_val, list) and isinstance(off_val, list):
        if len(live_val) != len(off_val):
            return True, None, [f"len mismatch: {len(live_val)} vs {len(off_val)}"]
        divergent = []
        max_d = 0.0
        for i, (lv, ov) in enumerate(zip(live_val, off_val)):
            d, delta = _scalar_diff(lv, ov, rtol, atol)
            if d:
                divergent.append(f"[{i}]")
            if delta is not None and delta > max_d:
                max_d = delta
        return bool(divergent), max_d, divergent

    return live_val != off_val, None, ["type mismatch"]


def _diff_checkpoint(
    live_row: dict,
    off_row: dict,
    ckpt: str,
) -> tuple[bool, list[str], float | None]:
    """For checkpoint ``ckpt``, return (diverges, divergent_fields, max_abs_delta)."""
    tol = TOLERANCES.get(ckpt, TOLERANCES["default"])
    rtol, atol = float(tol["rtol"]), float(tol["atol"])
    divergent: list[str] = []
    max_abs = 0.0
    for field in CHECKPOINTS[ckpt]:
        lv = live_row.get(field)
        ov = off_row.get(field)
        if ckpt in {"c4", "c5"} and field in {"feature_vector", "transvol_embedding"}:
            d, delta, keys = _vec_diff(lv, ov, rtol, atol)
            if d:
                divergent.append(f"{field}={keys}")
                if delta is not None and delta > max_abs:
                    max_abs = delta
        else:
            d, delta = _scalar_diff(lv, ov, rtol, atol)
            if d:
                divergent.append(field)
                if delta is not None and delta > max_abs:
                    max_abs = delta
    return bool(divergent), divergent, max_abs if max_abs > 0 else None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--live", required=True)
    p.add_argument("--offline", required=True)
    p.add_argument("--max-table-rows", type=int, default=50,
                   help="Cap per-bar table size in the printout")
    args = p.parse_args()

    live_df = _load_jsonl_as_df(Path(args.live))
    off_df = _load_jsonl_as_df(Path(args.offline))
    if live_df.empty or off_df.empty:
        print(f"empty traces: live={len(live_df)} offline={len(off_df)}")
        return 1
    print(f"live rows:     {len(live_df)}  [{live_df['bar_close_ts'].min()}  ->  {live_df['bar_close_ts'].max()}]")
    print(f"offline rows:  {len(off_df)}   [{off_df['bar_close_ts'].min()} ->  {off_df['bar_close_ts'].max()}]")

    merged = live_df.merge(
        off_df, on="bar_close_ts", how="inner", suffixes=("_live", "_off"),
    )
    print(f"joined rows:   {len(merged)}\n")
    if merged.empty:
        print("no overlap between live and offline bar timestamps")
        return 1

    histogram: dict[str, int] = {f"c{i}": 0 for i in range(9)}
    histogram["clean"] = 0

    table_rows: list[dict] = []
    worst_edge_delta = (-1.0, None, None)  # (|d|, live_row, off_row)
    for _, m in merged.iterrows():
        live_row = {c[:-5]: m[c] for c in m.index if c.endswith("_live")}
        off_row = {c[:-4]: m[c] for c in m.index if c.endswith("_off")}
        # bar_close_ts wasn't suffixed since it was the merge key.
        live_row["bar_close_ts"] = m["bar_close_ts"]
        off_row["bar_close_ts"] = m["bar_close_ts"]

        first_div = None
        max_d = None
        div_fields: list[str] = []
        for ckpt in (f"c{i}" for i in range(9)):
            diverges, fields, delta = _diff_checkpoint(live_row, off_row, ckpt)
            if diverges:
                first_div = ckpt
                max_d = delta
                div_fields = fields
                break
        if first_div is None:
            histogram["clean"] += 1
        else:
            histogram[first_div] += 1

        # Track worst |Δedge|.
        l_edge = live_row.get("edge_vol_pts")
        o_edge = off_row.get("edge_vol_pts")
        if l_edge is not None and o_edge is not None:
            try:
                d_edge = abs(float(l_edge) - float(o_edge))
                if d_edge > worst_edge_delta[0]:
                    worst_edge_delta = (d_edge, live_row, off_row)
            except (TypeError, ValueError):
                pass

        table_rows.append({
            "ts": m["bar_close_ts"],
            "first_divergent_ckpt": first_div or "clean",
            "max_abs_delta": max_d,
            "divergent_fields": ",".join(div_fields[:5]),
        })

    print("=" * 70)
    print("  PER-BAR DIVERGENCE TABLE")
    print("=" * 70)
    tbl = pd.DataFrame(table_rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", None)
    print(tbl.head(args.max_table_rows).to_string(index=False))
    if len(tbl) > args.max_table_rows:
        print(f"... ({len(tbl) - args.max_table_rows} more rows truncated)")

    print()
    print("=" * 70)
    print("  CHECKPOINT HISTOGRAM (first divergence per bar)")
    print("=" * 70)
    total = sum(histogram.values())
    for k in (*[f"c{i}" for i in range(9)], "clean"):
        v = histogram[k]
        pct = 100.0 * v / max(total, 1)
        bar = "#" * int(round(pct / 2.0))
        print(f"  {k:<6} {v:>4} ({pct:5.1f}%) {bar}")

    if worst_edge_delta[1] is not None:
        print()
        print("=" * 70)
        print(f"  WORST |d_edge| BAR DRILLDOWN  |d_edge|={worst_edge_delta[0]:.4f}vp")
        print("=" * 70)
        live_w = worst_edge_delta[1]
        off_w = worst_edge_delta[2]
        print(f"  bar_close_ts: {live_w['bar_close_ts']}")
        for ckpt in (f"c{i}" for i in range(9)):
            print(f"\n  --- {ckpt.upper()} ---")
            for field in CHECKPOINTS[ckpt]:
                lv = live_w.get(field)
                ov = off_w.get(field)
                if lv is None and ov is None:
                    continue
                same_marker = "  " if lv == ov else "* "
                print(f"   {same_marker}{field:<30} live={_fmt(lv)!r}  off={_fmt(ov)!r}")

    return 0


def _fmt(v: object) -> object:
    if isinstance(v, float):
        if not math.isfinite(v):
            return None
        return round(v, 6)
    if isinstance(v, dict):
        return f"<dict with {len(v)} keys>"
    if isinstance(v, list):
        return f"<list len={len(v)}>"
    return v


if __name__ == "__main__":
    raise SystemExit(main())
