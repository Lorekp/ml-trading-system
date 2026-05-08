"""C3 (parity-r3 cycle 8 — 2026-05-08) — HAR-PD pre-`fd3e04b` drift audit.

Round-2 audit C3 carryover: from F3 Tier 5 ship through 2026-05-08
the live HAR-PD circular buffer never advanced past its pre-warm
seed because every per-bar ``append`` raised ``AttributeError:
'BarTSRV' object has no attribute 'rv_ann'`` and the warning was
swallowed. The fix (``fd3e04b``: ``BarTSRV.rv_ann`` derived
property) is correct going forward; the historical impact on the
6 ``har_pd_R*_*`` features over the affected window is unmeasured.

This script replays the pre-fix vs post-fix paths against a recent
slice of the offline TSRV parquet and reports per-feature drift:

* **Pre-fix path**: buffer pre-warmed to a fixed timestamp T0 and
  HELD STATIC for the entire window [T0+1, T1]. Mimics the
  silent-failure mode where ``append`` never advanced.
* **Post-fix path**: buffer advanced bar-by-bar from T0+1 to T1.
  Live engine's correct behavior post-``fd3e04b``.

Output:
    - Per-feature mean / median / max absolute delta over the window
    - Per-bar peak drift timestamp for context
    - JSON sidecar at ``vol_arb/output/har_pd_pre_fix_drift_audit.json``

Exits 0 always; this is an OBSERVATIONAL audit — no gate failure.

Usage::

    python scripts/audit_har_pd_pre_fix_drift.py [--n-bars 5000]
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from deribit_paper.features.har_pd_live import (  # noqa: E402
    HAR_PD_RAW_FEATURES,
    HARPDState,
    K_BARS_DEFAULT,
)

logger = logging.getLogger(__name__)


_TSRV_PATH = REPO_ROOT / "TSRV" / "output" / "BTCUSDT_tsrv_15m.parquet"


def _load_tsrv_window(n_bars: int) -> pd.DataFrame:
    """Load the last ``n_bars`` rows of the offline TSRV parquet."""
    if not _TSRV_PATH.exists():
        raise FileNotFoundError(f"TSRV parquet not found: {_TSRV_PATH}")
    df = pd.read_parquet(_TSRV_PATH)
    needed = {"rs_up", "rs_down", "rv_ann"}
    missing = needed - set(df.columns)
    if missing:
        raise KeyError(f"TSRV parquet missing columns: {missing}")
    return df.tail(n_bars).copy()


def _prewarm_state(window_df: pd.DataFrame, k_bars: int) -> HARPDState:
    """Build a HAR-PD state pre-warmed to the FIRST ``k_bars`` rows of
    the window (T0..T0+k_bars-1). The continuation rows after this
    represent the live-engine bars that should have advanced the
    buffer but didn't pre-fix.
    """
    state = HARPDState(k_bars=k_bars)
    seed_rows = window_df.iloc[:k_bars]
    rows = (
        (float(r.rs_up), float(r.rs_down), float(r.rv_ann))
        for r in seed_rows.itertuples()
    )
    n = state.prewarm(rows)
    if n < k_bars:
        raise RuntimeError(f"prewarm seeded only {n}/{k_bars} bars")
    return state


def _compute_drift(window_df: pd.DataFrame, k_bars: int = K_BARS_DEFAULT) -> dict:
    """Replay both paths and collect per-feature per-bar drift."""
    if len(window_df) <= k_bars:
        raise ValueError(
            f"window has {len(window_df)} bars, need > k_bars={k_bars} "
            f"for the continuation slice."
        )
    pre_fix = _prewarm_state(window_df, k_bars)
    post_fix = _prewarm_state(window_df, k_bars)
    continuation = window_df.iloc[k_bars:]

    # Per-bar feature snapshot for both paths.
    pre_records: list[dict] = []
    post_records: list[dict] = []
    timestamps: list[pd.Timestamp] = []

    for ts, row in continuation.iterrows():
        # Compute features BEFORE appending — strict-past contract.
        pre_feats = pre_fix.compute_features()
        post_feats = post_fix.compute_features()
        pre_records.append(pre_feats)
        post_records.append(post_feats)
        timestamps.append(pd.Timestamp(ts))
        # Pre-fix: silent-failure path → buffer stays frozen, do NOT append.
        # Post-fix: append normally to advance the buffer.
        post_fix.append(
            rs_up=float(row.rs_up),
            rs_down=float(row.rs_down),
            rv_ann=float(row.rv_ann),
        )

    pre_df = pd.DataFrame(pre_records, index=pd.DatetimeIndex(timestamps))
    post_df = pd.DataFrame(post_records, index=pd.DatetimeIndex(timestamps))
    delta = post_df - pre_df

    summary: dict = {
        "n_bars_continuation": int(len(continuation)),
        "k_bars": int(k_bars),
        "window_start": str(window_df.index[0]),
        "window_end": str(window_df.index[-1]),
        "per_feature": {},
    }
    for col in HAR_PD_RAW_FEATURES:
        d = delta[col].astype(float)
        peak_idx = d.abs().idxmax() if not d.isna().all() else None
        summary["per_feature"][col] = {
            "mean_abs_delta": float(d.abs().mean()),
            "median_abs_delta": float(d.abs().median()),
            "max_abs_delta": float(d.abs().max()),
            "peak_drift_ts": str(peak_idx) if peak_idx is not None else None,
            "post_fix_at_end": float(post_df[col].iloc[-1]),
            "pre_fix_at_end": float(pre_df[col].iloc[-1]),
            "delta_at_end": float(delta[col].iloc[-1]),
        }
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-bars", type=int, default=5000,
        help="trailing TSRV-bar slice to replay (default 5000 = ~52 days at 15m).",
    )
    parser.add_argument(
        "--k-bars", type=int, default=K_BARS_DEFAULT,
        help=f"HAR-PD kernel window (default {K_BARS_DEFAULT}).",
    )
    parser.add_argument(
        "--output",
        default=str(REPO_ROOT / "vol_arb" / "output" /
                    "har_pd_pre_fix_drift_audit.json"),
    )
    parser.add_argument(
        "--log-level", default="INFO",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=args.log_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    logger.info("Loading TSRV window: %d trailing bars", args.n_bars)
    window = _load_tsrv_window(args.n_bars)
    logger.info(
        "Loaded %d bars (%s -> %s)",
        len(window), window.index[0], window.index[-1],
    )

    summary = _compute_drift(window, k_bars=args.k_bars)

    # Print human-readable table.
    print("=" * 78)
    print(f"HAR-PD pre-`fd3e04b` drift audit (round-3 cycle 8)")
    print(f"  window: {summary['window_start']} -> {summary['window_end']}")
    print(f"  continuation bars: {summary['n_bars_continuation']}")
    print(f"  k_bars: {summary['k_bars']}")
    print("=" * 78)
    print(f"{'feature':<22} {'mean|d|':>12} {'med|d|':>12} {'max|d|':>12}")
    for name in HAR_PD_RAW_FEATURES:
        s = summary["per_feature"][name]
        print(f"{name:<22} {s['mean_abs_delta']:>12.4e} "
              f"{s['median_abs_delta']:>12.4e} {s['max_abs_delta']:>12.4e}")
    print()
    print("End-of-window snapshot (post-fix vs pre-fix):")
    print(f"{'feature':<22} {'post_fix':>14} {'pre_fix':>14} {'delta':>14}")
    for name in HAR_PD_RAW_FEATURES:
        s = summary["per_feature"][name]
        print(f"{name:<22} {s['post_fix_at_end']:>14.6f} "
              f"{s['pre_fix_at_end']:>14.6f} {s['delta_at_end']:>+14.6f}")
    print("=" * 78)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2, default=float))
    logger.info("Wrote audit summary to %s", out_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
