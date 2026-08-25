"""Shared utilities for the invalidated S3 forensic experiments.

All variants are post-processors on the cached FROZEN-config frame
(`vol_arb/output/verify_frame_band35.parquet`). They NEVER touch the
realistic_eval.py / honest_pnl.py / backtester.py / deribit_fees.py engine.

DSR is computed on the NON-OVERLAPPING per-trade PnL series at K=40
(skew/kurt-aware Lo-2002, the audited `deflated_sharpe` from s3_revalidate).
The K=40 bar is NOT relaxed — it is the spec's binding pre-committed hurdle.
These helpers support historical reconstruction only, not strategy execution.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

# Reuse the FROZEN config + frame + DSR plumbing from s3_revalidate
# (single source of truth — no reimplementation).
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

from scripts.s3_revalidate import (
    FROZEN_CFG, FRAME_CACHE, prep_short_covered,
    deflated_sharpe, _nonoverlap_sharpe, _newey_west_sharpe, _max_drawdown,
)

OUT_DIR = "vol_arb/output/S3_VARIANTS"


@dataclass
class VariantResult:
    code: str            # "A1", "A2", ...
    name: str            # "vol-target portfolio", ...
    mechanism: str       # WHY it should help
    n: int               # number of trades (post-filter)
    total_usd: float
    per_trade_usd: float
    nw_t: float
    nonov_t: float
    nonov_n: int
    maxdd_usd: float
    worst_usd: float
    dsr_p: float
    dsr_t: float
    dsr_hurdle: float
    passed: bool
    per_year: dict       # {year: per_trade_usd}
    extras: dict         # variant-specific metadata

    def to_row(self) -> dict:
        return dict(
            code=self.code, name=self.name, mechanism=self.mechanism,
            n=self.n, total_usd=round(self.total_usd, 0),
            per_trade_usd=round(self.per_trade_usd, 0),
            nw_t=round(self.nw_t, 3), nonov_t=round(self.nonov_t, 3),
            nonov_n=self.nonov_n, maxdd_usd=round(self.maxdd_usd, 0),
            worst_usd=round(self.worst_usd, 0),
            dsr_p=round(self.dsr_p, 4), dsr_t=round(self.dsr_t, 3),
            dsr_hurdle=round(self.dsr_hurdle, 3),
            passed=bool(self.passed),
        )


def load_frame() -> pd.DataFrame:
    """Load the FROZEN-config frame (no engine re-eval; reuse the cache)."""
    f = pd.read_parquet(FRAME_CACHE)
    return f


def get_short_covered(f: pd.DataFrame) -> pd.DataFrame:
    scov, _ = prep_short_covered(f)
    return scov


def compute_dsr(pnl: np.ndarray) -> dict:
    """Run the audited DSR on a per-trade PnL series. K=40, non-overlapping."""
    return deflated_sharpe(pnl, K=40)


def nonoverlap_indices(entry: pd.DatetimeIndex, exit_: pd.DatetimeIndex) -> np.ndarray:
    """Greedy walk (identical to s3_revalidate L481-487) → return kept mask."""
    order = np.argsort(entry.values)
    e = entry.values[order]; x = exit_.values[order]
    kept_orig = np.zeros(len(e), dtype=bool)
    last = np.datetime64("1900-01-01")
    next_keep = 0
    for i in range(len(e)):
        if e[i] >= last:
            kept_orig[order[i]] = True
            last = x[i]
    # Re-derive in original order
    kept = np.zeros(len(entry), dtype=bool)
    for orig_i in range(len(entry)):
        if (entry.values[orig_i] >= np.datetime64("1900-01-01")):
            pass
    # Re-run with original positions via a re-mapping (correct, preserves original order)
    e_orig = entry.values; x_orig = exit_.values
    last = np.datetime64("1900-01-01")
    for i in np.argsort(e_orig):
        if e_orig[i] >= last:
            kept[i] = True
            last = x_orig[i]
    return kept


def evaluate_variant(df: pd.DataFrame, *, code: str, name: str, mechanism: str,
                    pnl_col: str = "total_hedged_usd",
                    entry_col: str = "entry_ts",
                    exit_col: str = "exit_ts",
                    extras: dict | None = None) -> VariantResult:
    """Standardized metric bundle: per-trade edge, NW-t, non-ov DSR, MaxDD, per-year."""
    df = df.copy()
    df[entry_col] = pd.to_datetime(df[entry_col], utc=True)
    df[exit_col] = pd.to_datetime(df[exit_col], utc=True)
    pnl = df[pnl_col].to_numpy(float)
    entry = pd.DatetimeIndex(df[entry_col])
    exit_ = pd.DatetimeIndex(df[exit_col])
    # Non-overlapping walk
    kept_mask = nonoverlap_indices(entry, exit_)
    pnl_nonov = pnl[kept_mask]
    dsr = compute_dsr(pnl_nonov)
    # NW t
    nw_t = _newey_west_sharpe(pnl, entry, exit_)
    # Non-ov t
    if len(pnl_nonov) >= 3 and np.std(pnl_nonov, ddof=1) > 0:
        nonov_t = float(np.mean(pnl_nonov) / np.std(pnl_nonov, ddof=1))
    else:
        nonov_t = float("nan")
    # Per-year
    df["year"] = df[entry_col].dt.year
    per_year = {int(y): float(df.loc[df["year"] == y, pnl_col].mean())
                for y in sorted(df["year"].unique())}
    return VariantResult(
        code=code, name=name, mechanism=mechanism,
        n=len(df),
        total_usd=float(np.nansum(pnl)),
        per_trade_usd=float(np.nanmean(pnl)),
        nw_t=float(nw_t) if np.isfinite(nw_t) else float("nan"),
        nonov_t=float(nonov_t) if np.isfinite(nonov_t) else float("nan"),
        nonov_n=int(len(pnl_nonov)),
        maxdd_usd=_max_drawdown(pnl),
        worst_usd=float(np.nanmin(pnl)),
        dsr_p=dsr["dsr_p"],
        dsr_t=dsr["t_stat"],
        dsr_hurdle=dsr["t_hurdle"],
        passed=dsr["passed"],
        per_year=per_year,
        extras=extras or {},
    )


def write_variant_report(results: list[VariantResult], theme: str, *,
                        out_path: str, summary: str = "") -> None:
    """Write a per-theme markdown report + JSON sidecar."""
    L = []
    A = L.append
    A(f"# S3 VARIANTS — Theme {theme}\n")
    A(f"**Frame:** cached `{FRAME_CACHE}` (FROZEN band=0.35, SHORT-only, expiry-hold).\n")
    A(f"**Engine:** `vol_arb/realistic_eval.py` UNTOUCHED (V2 byte-identity preserved).\n")
    A(f"**DSR:** non-overlapping, K=40 (NOT relaxed — spec §1.4 binding). t-hurdle = K=40 DSR SR0/SE.\n")
    A(f"**Pass:** DSR p>0.99 AND t_stat > t_hurdle.\n")
    if summary:
        A(f"\n> {summary}\n")
    A("\n## Results table\n")
    A("| code | variant | n | per-trade $ | NW t | nonov t (n) | MaxDD $ | worst $ | DSR p | DSR t | hurdle | PASS |")
    A("|---|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|:--:|")
    for r in results:
        A(f"| {r.code} | {r.name} | {r.n} | {r.per_trade_usd:,.0f} | "
          f"{r.nw_t:.2f} | {r.nonov_t:.2f} ({r.nonov_n}) | {r.maxdd_usd:,.0f} | "
          f"{r.worst_usd:,.0f} | {r.dsr_p:.4f} | {r.dsr_t:.2f} | "
          f"{r.dsr_hurdle:.2f} | {'YES' if r.passed else 'no'} |")
    # Per-variant detail
    for r in results:
        A(f"\n### {r.code} — {r.name}\n")
        A(f"**Mechanism:** {r.mechanism}\n")
        A(f"**Per-year per-trade $:** {r.per_year}\n")
        if r.extras:
            A(f"**Extras:** {json.dumps(r.extras, indent=2, default=str)}\n")
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")
    # JSON sidecar
    json_path = out_path.replace(".md", ".json")
    with open(json_path, "w", encoding="utf-8") as fh:
        json.dump([{**r.to_row(), "per_year": r.per_year, "extras": r.extras}
                   for r in results], fh, indent=2, default=str)
    print(f"[{theme}] wrote {out_path}  +  {json_path}")
