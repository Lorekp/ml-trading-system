"""Per-bar traceability viewer.

Joins the three per-bar data sources for full decision traceability:

1. ``deribit_paper/state/paper.db::bar_cursor`` — textual decision +
   reason for every bar (the "WHY").
2. ``deribit_paper/state/live_metrics.parquet`` — numeric inputs the
   predictor saw (edge, rv_pred, iv, jump_fraction, etc.).
3. ``vol_arb/output/BTC_v2_trade_log.parquet`` — fired-trade execution
   details (only rows where a trade actually executed).

Usage::

    # Last 20 bars on the homeserver:
    ssh monitor 'cd /hdd/ml-trading-system && PYTHONPATH=. python3 scripts/show_bar_timeline.py'

    # Last 20 bars locally (uses local paper.db if present, else fails):
    PYTHONPATH=. python scripts/show_bar_timeline.py

    # Specific window:
    python scripts/show_bar_timeline.py --since 2026-05-10T14:00 --until 2026-05-10T16:00

    # Filter to specific reason category:
    python scripts/show_bar_timeline.py --reason-contains "skew filter"

Output: one line per bar with columns: timestamp, sig, edge,
rv_pred%, iv%, jump_frac, reason, [trade_pnl if fired].
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

import pandas as pd  # noqa: E402

PAPER_DB = REPO_ROOT / "deribit_paper" / "state" / "paper.db"
LIVE_METRICS = REPO_ROOT / "deribit_paper" / "state" / "live_metrics.parquet"
TRADE_LOG = REPO_ROOT / "vol_arb" / "output" / "BTC_v2_trade_log.parquet"


def main() -> int:
    parser = argparse.ArgumentParser(description="Per-bar traceability viewer")
    parser.add_argument("--n", type=int, default=20,
                        help="Last N bars (default 20)")
    parser.add_argument("--since", type=str, default="",
                        help="ISO timestamp lower bound (UTC), e.g. 2026-05-10T14:00")
    parser.add_argument("--until", type=str, default="",
                        help="ISO timestamp upper bound (UTC)")
    parser.add_argument("--reason-contains", type=str, default="",
                        help="Filter to bars whose signal_reason contains this substring")
    parser.add_argument("--paper-db", type=str, default=str(PAPER_DB))
    parser.add_argument("--live-metrics", type=str, default=str(LIVE_METRICS))
    parser.add_argument("--trade-log", type=str, default=str(TRADE_LOG))
    args = parser.parse_args()

    db_path = Path(args.paper_db)
    if not db_path.exists():
        print(f"ERROR: paper.db not found at {db_path}", file=sys.stderr)
        return 1

    # Load bar_cursor (decision + reason).
    conn = sqlite3.connect(db_path)
    bc = pd.read_sql_query(
        "SELECT bar_timestamp, signal_direction, signal_reason, fired_at FROM bar_cursor",
        conn,
    )
    conn.close()
    bc["timestamp"] = pd.to_datetime(bc["bar_timestamp"], utc=True)
    bc = bc.set_index("timestamp").drop(columns=["bar_timestamp"]).sort_index()

    # Load live_metrics if present (numeric inputs).
    metrics_path = Path(args.live_metrics)
    if metrics_path.exists():
        lm = pd.read_parquet(metrics_path)
        if "timestamp" in lm.columns:
            lm = lm.set_index("timestamp")
        lm.index = pd.to_datetime(lm.index, utc=True)
        df = bc.join(lm, how="left")
    else:
        print(f"NOTE: {metrics_path} not present — numeric columns omitted")
        df = bc.copy()

    # Load trade_log to attach PnL where a trade fired.
    tl_path = Path(args.trade_log)
    if tl_path.exists():
        tl = pd.read_parquet(tl_path)
        tl.index = pd.to_datetime(tl.index, utc=True)
        if "net_pnl_nc" in tl.columns:
            df = df.join(tl[["net_pnl_nc"]].rename(columns={"net_pnl_nc": "trade_pnl"}), how="left")

    # Apply filters.
    if args.since:
        df = df[df.index >= pd.Timestamp(args.since, tz="UTC")]
    if args.until:
        df = df[df.index <= pd.Timestamp(args.until, tz="UTC")]
    if args.reason_contains:
        df = df[df["signal_reason"].str.contains(args.reason_contains, na=False, case=False)]

    if len(df) == 0:
        print("(no bars match the filters)")
        return 0

    df = df.tail(args.n)

    # Pretty-print.
    print(f"{'timestamp':>20} | {'sig':>3} | {'edge_vp':>8} | "
          f"{'rv%':>5} | {'iv%':>5} | {'jump_f':>6} | {'pnl':>8} | reason")
    print("-" * 130)
    for ts, row in df.iterrows():
        sig = int(row.get("signal_direction", 0) or 0)
        edge = row.get("edge_vol_pts", float("nan"))
        rv = row.get("rv_pred_pct", float("nan"))
        iv = row.get("iv_used_pct", float("nan"))
        jf = row.get("jump_fraction", float("nan"))
        pnl = row.get("trade_pnl", None)
        pnl_str = f"${pnl:>6.0f}" if pnl is not None and pd.notna(pnl) and pnl != 0 else "-"
        reason = str(row.get("signal_reason", ""))[:80]
        print(f"{ts.strftime('%Y-%m-%d %H:%M'):>20} | "
              f"{sig:>+3} | {edge:>+8.3f} | {rv:>5.2f} | {iv:>5.2f} | "
              f"{jf:>6.3f} | {pnl_str:>8} | {reason}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
