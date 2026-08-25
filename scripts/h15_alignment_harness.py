"""Historical replay harness: paper-engine versus V2 per-trade diff.

Inputs:
  ``vol_arb/output/live_trades_harness.parquet``  (live engine, C6).
  ``vol_arb/output/BTC_v2_trade_log.parquet``     (invalidated historical V2 artifact).

For each live trade, find the matching V2 trade-log row by
``started_at`` and compute per-column deltas. Aggregates into a
markdown summary attributing the residual PnL gap to one of four
buckets:

* **config drift**  — signal direction or size mismatch
  (entry_threshold_long, contango filter, target_margin_util drift).
* **price drift**   — entry/exit prices differ
  (slippage model, strike rounding, fill-time differences).
* **PnL residual**  — sign/magnitude of realized PnL difference.
* **strike rounding** — historical diagnostics observed roughly 2-4 vol
  points / 0.5-1% edge shrinkage on typical bars.

Output:
  ``vol_arb/output/h15_alignment/harness_stage_a.parquet``
  ``vol_arb/output/h15_alignment/H15_ALIGNMENT_REPORT.md``

Stage B (replay V2 backtester using LIVE features) is a follow-on; the
gap between Stage A and Stage B isolates feature-staleness drift. Until
the live engine emits ``y_true`` (realised RV at T+15m), Stage B has
limited information and is deliberately deferred.

Usage::

    # from the repository root
    PYTHONPATH=. python scripts/h15_alignment_harness.py
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

LIVE_TRADES_PATH = Path("vol_arb/output/live_trades_harness.parquet")
V2_TRADES_PATH = Path("vol_arb/output/BTC_v2_trade_log.parquet")
OUTPUT_DIR = Path("vol_arb/output/h15_alignment")
STAGE_A_PARQUET = OUTPUT_DIR / "harness_stage_a.parquet"


# ── Per-column diff contract ──────────────────────────────────────────────
#
# Verified against the on-disk schema of
# ``vol_arb/output/BTC_v2_trade_log.parquet`` (Apr 21 build, 145041 rows,
# 127 columns). The V2 trade log does NOT carry per-leg prices — option
# fills are aggregated into ``net_pnl_nc`` (net non-compound PnL) along
# with hedge & fees. The diff therefore runs at trade-aggregate level,
# not per-leg, and the live engine's leg prices are written to the
# harness parquet for diagnostics but NOT compared here.

_NUMERIC_COLS_LIVE_TO_V2: list[tuple[str, str]] = [
    # (live column, v2 column) — both in V2 trade log's actual schema.
    ("signal",                "signal"),
    ("n_contracts_btc",       "N_contracts"),
    ("y_pred_pct",            "rv_pred_pct"),
    ("iv_used_pct",           "iv_used"),       # V2 column lacks _pct suffix
    ("edge_vol_pts",          "edge"),
    ("realized_pnl_usd",      "net_pnl_nc"),
]


# ── Loading ──────────────────────────────────────────────────────────────

def _load_or_empty(path: Path, name: str) -> pd.DataFrame:
    if not path.exists():
        logger.info("%s not found at %s — treating as empty.", name, path)
        return pd.DataFrame()
    logger.info("Loading %s from %s", name, path)
    return pd.read_parquet(path)


# ── Stage A diff ─────────────────────────────────────────────────────────

def _stage_a_diff(live: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    """Per-trade diff between live and V2 trade logs.

    ``v2`` is the **full** V2 trade log including bars where the
    backtester chose flat (``signal == 0``). Only V2 rows where
    ``signal != 0`` are considered for the inner-join — those are the
    bars the backtester actually traded. Bars where live fired but V2
    was flat go into the ``live_fired_v2_flat`` bucket (see
    ``_summarise``) — those are the most diagnostic-rich misalignment
    cases (config drift on threshold or contango filter).

    ``live.started_at`` is parsed to UTC. Output is indexed on the
    matched timestamps with ``d_<col>`` delta columns.
    """
    if live.empty:
        return pd.DataFrame()

    live = live.copy()
    live["started_at"] = pd.to_datetime(live["started_at"], utc=True, errors="coerce")
    live = live.dropna(subset=["started_at"])
    live = live.set_index("started_at").sort_index()

    if v2.empty:
        logger.warning("V2 backtest trade log is empty — Stage A produces NaN columns.")
        diff = live.copy()
        for live_col, _v2_col in _NUMERIC_COLS_LIVE_TO_V2:
            diff[f"d_{live_col}"] = np.nan
        diff.attrs["live_fired_v2_flat_count"] = 0
        return diff

    # V2 trade log is already DatetimeIndex (per the backtester contract).
    v2 = v2.copy()
    if not isinstance(v2.index, pd.DatetimeIndex):
        v2.index = pd.to_datetime(v2.index, utc=True)

    # ── Live-fired-but-V2-flat bucket (config-drift diagnostic) ──
    # Compare against the full V2 (signal in {-1, 0, +1}) to detect bars
    # where live decided to trade but V2 didn't. These are typically
    # caused by (a) entry_threshold mismatch, (b) contango filter
    # drift, or (c) the V2 backtester's RND / DRL gating differing.
    live_idx = live.index
    v2_full_signal = (
        v2["signal"].reindex(live_idx, fill_value=0)
        if "signal" in v2.columns else pd.Series(0, index=live_idx)
    )
    live_fired = live["signal"].abs() > 0.5
    v2_flat_at_live_fire = v2_full_signal.abs() < 0.5
    live_fired_v2_flat = int((live_fired & v2_flat_at_live_fire).sum())

    # Inner-join on timestamps where BOTH sides fired (signal != 0).
    if "signal" in v2.columns:
        v2_fired = v2[v2["signal"] != 0]
    else:
        v2_fired = v2
    common = live.index.intersection(v2_fired.index)
    logger.info(
        "Stage A: live=%d (fired=%d), v2=%d (fired=%d), common=%d, "
        "live_fired_v2_flat=%d",
        len(live), int(live_fired.sum()),
        len(v2), len(v2_fired), len(common), live_fired_v2_flat,
    )
    if len(common) == 0:
        empty = pd.DataFrame()
        empty.attrs["live_fired_v2_flat_count"] = live_fired_v2_flat
        return empty

    out = live.loc[common].copy()
    for live_col, v2_col in _NUMERIC_COLS_LIVE_TO_V2:
        if live_col not in live.columns:
            continue
        live_vals = live.loc[common, live_col].to_numpy(dtype=float, na_value=np.nan)
        v2_vals = (
            v2_fired.loc[common, v2_col].to_numpy(dtype=float, na_value=np.nan)
            if v2_col in v2_fired.columns else np.full(len(common), np.nan)
        )
        out[f"d_{live_col}"] = live_vals - v2_vals
    out.attrs["live_fired_v2_flat_count"] = live_fired_v2_flat
    return out


def _summarise(diff: pd.DataFrame) -> dict:
    """Compute per-bucket summary for the markdown report."""
    summary: dict = {
        "n_trades_total": int(len(diff)),
        "buckets": {},
        "live_fired_v2_flat_count": int(
            diff.attrs.get("live_fired_v2_flat_count", 0)
        ),
    }
    if diff.empty:
        return summary

    # config drift: signal mismatch + size mismatch
    if "d_signal" in diff.columns:
        n_sig_diff = int((diff["d_signal"].abs() > 0.5).sum())
        summary["buckets"]["config_drift_signal_mismatch"] = n_sig_diff
    if "d_n_contracts_btc" in diff.columns:
        summary["buckets"]["config_drift_size_med_abs_delta_btc"] = float(
            np.nanmedian(diff["d_n_contracts_btc"].abs())
        )

    # PnL residual (the headline diagnostic — V2 has no leg prices, so
    # this is the only price-side comparison we can make)
    if "d_realized_pnl_usd" in diff.columns and diff["d_realized_pnl_usd"].notna().any():
        summary["buckets"]["pnl_med_abs_diff_usd"] = float(
            np.nanmedian(diff["d_realized_pnl_usd"].abs())
        )
        summary["buckets"]["pnl_p90_abs_diff_usd"] = float(
            np.nanpercentile(diff["d_realized_pnl_usd"].abs().dropna(), 90)
        )

    # Predictor-side drift (y_pred & iv): non-zero deltas indicate
    # feature-staleness drift, captured in Stage A as a leading indicator
    # for Stage B (which requires y_true backfill).
    for live_col in ("d_y_pred_pct", "d_iv_used_pct", "d_edge_vol_pts"):
        if live_col in diff.columns and diff[live_col].notna().any():
            summary["buckets"][f"feature_drift_{live_col}_p50_abs"] = float(
                np.nanmedian(diff[live_col].abs())
            )

    return summary


def _format_report(summary: dict, out_md: Path) -> None:
    lines: list[str] = []
    lines.append("# H=15m Live↔V2 Alignment Report (Stage A)\n")
    lines.append(
        f"**Trades compared (both fired):** {summary['n_trades_total']}\n"
    )
    lfvf = summary.get("live_fired_v2_flat_count", 0)
    lines.append(
        f"**Live-fired-but-V2-flat (config-drift bucket):** {lfvf}\n"
    )
    lines.append("")
    lines.append(
        "_The V2 trade log carries no per-leg prices; the diff therefore "
        "compares aggregate trade-level fields (signal, size, y_pred, "
        "iv_used, edge, net_pnl_nc). Live leg prices are written to the "
        "harness parquet for diagnostics but not differentiated here._\n"
    )
    lines.append("")
    if summary["n_trades_total"] == 0:
        lines.append(
            "_No trade overlap between live engine and V2 backtest. Run the_\n"
            "_paper engine in v2 preset for ≥1 week and re-run the harness._\n"
        )
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text("\n".join(lines), encoding="utf-8")
        return

    # Bucketed summary
    lines.append("## Drift attribution\n")
    lines.append("| bucket | metric | value |")
    lines.append("|---|---|---|")
    for bucket_key, value in summary["buckets"].items():
        if isinstance(value, float):
            lines.append(f"| {bucket_key} | metric | {value:.4f} |")
        else:
            lines.append(f"| {bucket_key} | metric | {value} |")
    lines.append("")

    lines.append("## Notes\n")
    lines.append(
        "* **live-fired-V2-flat**: live engine traded but V2 backtester "
        "stayed flat at the same bar — typically caused by "
        "`entry_threshold` mismatch, contango filter drift, or a V2 "
        "reject (RND tail probability, etc.) that the live engine does "
        "not enforce identically.\n"
    )
    lines.append(
        "* **Stage B (feature-drift attribution)**: deferred — requires "
        "y_true / TSRV at T+15m to be backfilled into the live trade "
        "log. Expected to land alongside the next-cycle SVI/TSRV "
        "live-feature audit.\n"
    )

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--live-trades", default=LIVE_TRADES_PATH, type=Path)
    parser.add_argument("--v2-trades",   default=V2_TRADES_PATH,   type=Path)
    parser.add_argument("--output-dir",  default=OUTPUT_DIR,       type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stage_a_path = args.output_dir / "harness_stage_a.parquet"
    report_path = args.output_dir / "H15_ALIGNMENT_REPORT.md"

    live = _load_or_empty(args.live_trades, "live trades")
    v2 = _load_or_empty(args.v2_trades, "V2 backtest")

    diff = _stage_a_diff(live, v2)
    if not diff.empty:
        diff.to_parquet(stage_a_path)
        logger.info("Wrote %s (%d rows)", stage_a_path, len(diff))
    else:
        logger.info("Stage A diff empty — skipping parquet output.")

    summary = _summarise(diff)
    _format_report(summary, report_path)
    logger.info("Wrote %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
