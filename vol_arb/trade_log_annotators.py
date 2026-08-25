"""Post-hoc annotators for V2 trade log — Tier B of tail-risk plan.

Each annotator reads a sibling parquet (sidecar) of the historical V2
trade log and emits ONE additional column. Output goes to
``vol_arb/output/BTC_v2_trade_log_annotated.parquet`` — never touches
the canonical trade log, so V2 byte-identity is preserved.

**Leakage-safe contract** (mirrors ``vol_arb/dfl_head.py:18-26``):

* Pure post-hoc enrichment — every column at index t depends only on
  source data with timestamp ≤ t. Verified per-column by
  ``probe_causality_by_mutation``.
* No current-bar or future-bar reads. ``shift(1)`` / backward
  ``merge_asof`` only.
* Annotators are pure functions, idempotent, and depend only on
  read-only sources.
* Annotated output is DIAGNOSTIC. Feeding any annotator output back
  into a feature, sizer, or stop is OUT OF CONTRACT — that requires
  re-running the full leakage probe battery on the modified path.

Annotators (frozen list, pre-committed in plan):

* ``annotate_depth_ratio`` — R2: ``N_contracts / available_depth_btc``
  on the entry bar. Uses ``bar_execution_quality.parquet`` (END-stamped
  per pipeline convention).
* ``annotate_svi_n_strikes_at_entry`` — R4: ``svi_n_strikes`` field
  copied from the trade log itself; the annotator's job is to add a
  binary ``thin_svi`` flag (``svi_n_strikes < 5``) and the resulting
  PnL-at-risk column.

Usage::

    PYTHONPATH=. python -m vol_arb.run_backtest --historical-btc-v2 \\
        --realistic --annotate
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

# ─── Pre-committed constants (locked in plan) ────────────────────────────

THIN_SVI_N_STRIKES_FLOOR: int = 5  # min for a 5-parameter SVI fit

# Source paths (read-only)
DEFAULT_BAR_EXEC_PATH = Path("vol_arb/output/bar_execution_quality.parquet")
DEFAULT_OUTPUT_PATH = Path("vol_arb/output/BTC_v2_trade_log_annotated.parquet")


# ─── Annotators ──────────────────────────────────────────────────────────

def annotate_depth_ratio(
    trade_log: pd.DataFrame,
    bar_exec_path: str | Path = DEFAULT_BAR_EXEC_PATH,
) -> pd.DataFrame:
    """Append ``depth_ratio`` and ``slip_understatement_usd`` columns.

    For each entry bar, look up ``available_depth_btc`` from the
    bar-execution-quality sidecar (END-stamped) and compute:

    * ``depth_ratio = N_contracts / available_depth_btc`` (NaN where
      depth is missing).
    * ``slip_understatement_usd = max(0, depth_ratio - 1) ×
      bar_taker_slip_usd × N_contracts`` — a conservative lower-bound
      proxy for the additional slippage cost the SOR would have paid
      walking the book past available depth.

    Both columns are NaN on bars where depth coverage is missing
    (currently 42 of 7,417 fired trades). NaN-pass-through is the
    leakage-safe choice; a fallback constant would silently invent
    information.

    Parameters
    ----------
    trade_log
        V2 trade log indexed by timestamp. Required columns:
        ``signal``, ``net_pnl_nc``, ``N_contracts``.
    bar_exec_path
        Sidecar path. Default ``vol_arb/output/bar_execution_quality.parquet``.

    Returns
    -------
    DataFrame with the same index as ``trade_log`` and two new columns.
    """
    p = Path(bar_exec_path)
    if not p.exists():
        raise FileNotFoundError(f"bar_execution_quality sidecar not found at {p}")

    bex = pd.read_parquet(p)
    if not isinstance(bex.index, pd.DatetimeIndex):
        bex.index = pd.to_datetime(bex.index, utc=True)
    if "available_depth_btc" not in bex.columns:
        raise KeyError("bar_execution_quality.parquet missing 'available_depth_btc'")
    if "bar_taker_slip_usd" not in bex.columns:
        raise KeyError("bar_execution_quality.parquet missing 'bar_taker_slip_usd'")

    out = pd.DataFrame(index=trade_log.index)
    # Backward-asof not needed — bex is on the same 15m grid as the
    # trade log (END-stamped). Simple reindex (no method) keeps the
    # NaN-pass-through guarantee for missing rows.
    depth = bex["available_depth_btc"].reindex(trade_log.index)
    slip = bex["bar_taker_slip_usd"].reindex(trade_log.index)

    n_contracts = trade_log["N_contracts"].astype(float).values
    depth_arr = depth.values
    slip_arr = slip.values

    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = n_contracts / np.where(depth_arr > 0, depth_arr, np.nan)
    excess = np.clip(ratio - 1.0, a_min=0.0, a_max=None)

    understate = excess * slip_arr * n_contracts
    out["depth_ratio"] = ratio
    out["slip_understatement_usd"] = understate
    return out


def annotate_svi_n_strikes_at_entry(trade_log: pd.DataFrame) -> pd.DataFrame:
    """Append ``thin_svi`` flag and ``thin_svi_pnl_at_risk_usd``.

    A trade is flagged ``thin_svi`` when its ``svi_n_strikes`` is below
    the 5-strike floor for a 5-parameter SVI fit. The flag is purely
    diagnostic; the corresponding PnL is reported as "at risk" (= the
    NC PnL of those trades, which was fitted on a thin surface).

    No fitting, no threshold tuning — the floor is an SVI methodology
    constant.
    """
    if "svi_n_strikes" not in trade_log.columns:
        raise KeyError("trade_log missing 'svi_n_strikes' column")

    out = pd.DataFrame(index=trade_log.index)
    n_strikes = trade_log["svi_n_strikes"].astype(float).values
    is_thin = n_strikes < THIN_SVI_N_STRIKES_FLOOR
    pnl = trade_log["net_pnl_nc"].astype(float).values

    out["thin_svi"] = is_thin
    out["thin_svi_pnl_at_risk_usd"] = np.where(is_thin, pnl, 0.0)
    return out


# ─── Aggregator ──────────────────────────────────────────────────────────

ANNOTATORS = {
    "depth_ratio": annotate_depth_ratio,
    "svi_n_strikes_at_entry": annotate_svi_n_strikes_at_entry,
}


def annotate(
    trade_log: pd.DataFrame,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    *,
    bar_exec_path: str | Path = DEFAULT_BAR_EXEC_PATH,
    write_output: bool = True,
) -> pd.DataFrame:
    """Run all annotators and write the result to a sibling parquet.

    Returns the annotated DataFrame (canonical trade log + new columns)
    so callers can chain analysis. Never modifies the canonical
    parquet on disk; ``output_path`` is a separate artifact.
    """
    pieces: list[pd.DataFrame] = []
    pieces.append(annotate_depth_ratio(trade_log, bar_exec_path=bar_exec_path))
    pieces.append(annotate_svi_n_strikes_at_entry(trade_log))

    annotated = trade_log.join(pd.concat(pieces, axis=1))
    if write_output:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        annotated.to_parquet(out_p)
    return annotated
