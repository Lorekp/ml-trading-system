#!/usr/bin/env python
"""Cycle-34 (2026-05-13): assemble the offline feature trace.

Stitches together the three offline trace sources into ONE JSONL file
keyed on ``bar_close_ts``, matching the live engine's
``bar_trace_live.jsonl`` schema produced by
:class:`deribit_paper.diagnostics.bar_trace.BarTraceWriter`:

  1. C0/C1/C3/C7/C8 — emitted by ``vol_arb/backtester.py::_emit_feature_trace``
     when run with ``feature_trace_path`` set. Captures raw inputs,
     TSRV-derived columns, the iv chain, rv_pred_pct, edge + signal.

  2. C2 — SVI surface state. Joined from
     ``vol_arb/output/svi_surface_params_15min.parquet`` using the
     F-4 + merge_asof contract (most recent ``bucket_ts`` with
     ``bucket_ts <= bar_close_ts`` for the same expiry key). Picks
     the expiry that minimises |T_years - target_T| where target_T
     is the live engine's straddle DTE — defaults to 1w if the live
     trace is unavailable to cross-reference.

  3. C4/C5/C6 — feature vector, tv embedding, raw y_pred. Produced
     by running ``vol_model/extract_pruned_svi_har_pd_predictions.py``
     with the ``--emit-trace-jsonl`` flag over the same window.

Output: one JSONL with one row per bar matching the live trace schema.

Usage:
    python scripts/build_offline_bar_trace.py \
        --backtester-trace vol_arb/output/bar_trace_backtester.jsonl \
        --predictions-trace vol_model/output/bar_trace_predictions.jsonl \
        --svi-params vol_arb/output/svi_surface_params_15min.parquet \
        --output vol_arb/output/bar_trace_offline.jsonl \
        --target-expiry-key 2026-05-22  # optional: pin the expiry

Or with the orchestrate flag (re-runs both upstream emitters):
    python scripts/build_offline_bar_trace.py \
        --orchestrate \
        --start "2026-05-13 14:30:00+00:00" \
        --end "2026-05-13 23:45:00+00:00" \
        --output vol_arb/output/bar_trace_offline.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger("build_offline_bar_trace")


def _load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        logger.warning("trace file missing: %s", path)
        return []
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _key(ts: str) -> str:
    """Normalise an ISO timestamp to ``YYYY-MM-DDTHH:MM:SS+00:00`` for joining."""
    return pd.Timestamp(ts).tz_convert("UTC").isoformat()


def _attach_svi_state(
    rows: list[dict],
    svi_params: pd.DataFrame,
    target_expiry_key: str | None,
) -> int:
    """Mutate ``rows`` in place to add C2 fields. Returns rows annotated."""
    if svi_params.empty:
        return 0
    if "timestamp" not in svi_params.columns:
        logger.warning("svi_params missing 'timestamp' column; skipping C2 join")
        return 0
    sp = svi_params.copy()
    sp["timestamp"] = pd.to_datetime(sp["timestamp"], utc=True)
    sp["_exp_key"] = pd.to_datetime(sp["expiry_dt"], utc=True).dt.strftime("%Y-%m-%d")
    sp = sp.sort_values("timestamp")

    n_annotated = 0
    for row in rows:
        ts = pd.Timestamp(row["bar_close_ts"])
        candidate = sp[sp["timestamp"] <= ts]
        if candidate.empty:
            continue
        if target_expiry_key is not None:
            candidate = candidate[candidate["_exp_key"] == target_expiry_key]
            if candidate.empty:
                continue
        latest_ts = candidate["timestamp"].max()
        bucket = candidate[candidate["timestamp"] == latest_ts]
        # If we don't have a target expiry, pick the closest-T row.
        if target_expiry_key is None and "T" in bucket.columns:
            target_T = 7.0 / 365.25  # 1-week default
            bucket = bucket.iloc[(bucket["T"] - target_T).abs().argsort()].head(1)
        sl = bucket.iloc[0]
        row["svi_bucket_ts"] = pd.Timestamp(sl["timestamp"]).isoformat()
        row["svi_expiry_dt"] = pd.Timestamp(sl["expiry_dt"]).isoformat()
        row["svi_age_seconds"] = float((ts - pd.Timestamp(sl["timestamp"])).total_seconds())
        for col in ("a", "b", "rho", "m", "sigma", "T", "F", "rmse", "level", "n_strikes"):
            if col in sl.index:
                v = sl[col]
                if pd.notna(v):
                    out_col = "svi_T_years" if col == "T" else f"svi_{col}"
                    row[out_col] = float(v) if col != "n_strikes" else int(v)
        n_annotated += 1
    return n_annotated


def _merge_predictions_trace(
    rows: list[dict],
    pred_rows: list[dict],
) -> int:
    """Mutate ``rows`` in place to add C4/C5/C6 fields from pred_rows.

    Joins on ``bar_close_ts``. Skips silently when no match (the
    backtester emits one row per bar; the predictions emitter only
    fires inside the test window of months with a model checkpoint).
    """
    pred_by_ts: dict[str, dict] = {}
    for r in pred_rows:
        ts = _key(r["bar_close_ts"])
        pred_by_ts[ts] = r
    n = 0
    for row in rows:
        ts = _key(row["bar_close_ts"])
        match = pred_by_ts.get(ts)
        if match is None:
            continue
        for k in ("feature_vector", "transvol_embedding", "transvol_status",
                  "y_pred_raw"):
            if k in match:
                row[k] = match[k]
        n += 1
    return n


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--backtester-trace", required=False,
                   default="vol_arb/output/bar_trace_backtester.jsonl")
    p.add_argument("--predictions-trace", required=False,
                   default="vol_model/output/bar_trace_predictions.jsonl")
    p.add_argument("--svi-params", required=False,
                   default="vol_arb/output/svi_surface_params_15min.parquet")
    p.add_argument("--output", required=True)
    p.add_argument("--target-expiry-key", default=None,
                   help="Optional YYYY-MM-DD expiry to pin C2 join. "
                        "When None, picks the closest-T-to-1w slice per bar.")
    p.add_argument("--orchestrate", action="store_true",
                   help="Re-run both upstream emitters (extract + backtester).")
    p.add_argument("--start", default=None, help="Window start (with --orchestrate)")
    p.add_argument("--end", default=None, help="Window end (with --orchestrate)")
    args = p.parse_args()

    if args.orchestrate:
        if not args.start or not args.end:
            logger.error("--orchestrate requires --start and --end")
            return 1
        logger.info("Running extract_pruned_svi_har_pd_predictions.py …")
        subprocess.check_call([
            sys.executable,
            "vol_model/extract_pruned_svi_har_pd_predictions.py",
            "--emit-trace-jsonl", args.predictions_trace,
            "--emit-trace-start", args.start,
            "--emit-trace-end", args.end,
            "--end-date", args.end,
        ])
        logger.info("Running backtester with feature_trace_path …")
        # We invoke the backtester via a minimal driver script (existing
        # vol_arb/run_v2_backtest.py forwards CLI args into BacktestConfig).
        subprocess.check_call([
            sys.executable, "-m", "vol_arb.run_v2_backtest",
            "--feature-trace-path", args.backtester_trace,
            "--feature-trace-start", args.start,
            "--feature-trace-end", args.end,
        ])

    bt_rows = _load_jsonl(Path(args.backtester_trace))
    logger.info("Loaded %d backtester trace rows", len(bt_rows))
    pred_rows = _load_jsonl(Path(args.predictions_trace))
    logger.info("Loaded %d predictions trace rows", len(pred_rows))

    if Path(args.svi_params).exists():
        svi_params = pd.read_parquet(args.svi_params)
    else:
        logger.warning("svi-params parquet missing: %s", args.svi_params)
        svi_params = pd.DataFrame()

    n_c2 = _attach_svi_state(bt_rows, svi_params, args.target_expiry_key)
    logger.info("C2 (SVI surface state) attached to %d rows", n_c2)
    n_pred = _merge_predictions_trace(bt_rows, pred_rows)
    logger.info("C4/C5/C6 (predictions trace) merged for %d rows", n_pred)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        for row in bt_rows:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
    logger.info("Wrote %d offline trace rows → %s", len(bt_rows), out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
