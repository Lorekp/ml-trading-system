"""Shared sidecar loaders for backtest and live engines.

The V2 production stack consumes three walk-forward-fit sidecars at signal /
sizing / execution time:

* Kelly K1 multiplier (`BTC_v2_kelly_mult_K1.parquet`) — per-15min-bar
  upside-only sizing boost on long straddles, fit on trailing 12m of the
  V2 trade log per memory/project_kelly_sizing_2026_04_21.md.
* realistic_taker slippage (`bar_execution_quality.parquet`) — per-15min-bar
  empirical USD/BTC cost to cross market on one option leg, derived from
  the Deribit trade tape per memory/project_execution_model.md.
* Aligner β* (per-month scalar bias on log-vol predictions) — historically
  baked into ``BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet``
  via ``y_pred = y_pred_raw + β*_month``. The live engine cannot consume
  the parquet directly because it computes predictions in real time, so we
  expose a per-month dict that the predictor adds to fresh LGBM output.

This module centralises the leakage-safe walk-forward lookup logic in one
place so the backtester (which loads inline today) and the live engine
(which is wiring these in for the first time) read from a single contract.
Each loader returns ``None`` (or an empty dict for betas) when the source
file is missing, preserving the backtester's existing tolerant behaviour.

Design contract:

* Loaders are pure functions — no module-level state, no caching layer.
* Forward-fill semantics on time-aligned reindexes match the backtester
  (uses **past** value for bars without an exact match; never looks ahead).
* NaN / missing-month fallback values are explicit kwargs with V2 defaults
  matching ``vol_arb/backtester.py`` lines 1779 and 1869.
* Path arguments accept ``str`` or ``pathlib.Path`` interchangeably.

References to the in-tree call sites these helpers replace:

* ``vol_arb/backtester.py:1768-1790`` (bar_taker_slip ffill).
* ``vol_arb/backtester.py:1850-1878`` (kelly_mult reindex + clip).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ─── V2 defaults (single source of truth) ──────────────────────────────────
KELLY_FLOOR: float = 1.0
KELLY_CAP: float = 2.0
KELLY_FALLBACK: float = 1.0
SLIP_FALLBACK_USD: float = 12.0  # empirical median per backtester:1779


def load_kelly_mult(
    path: str | Path | None,
    target_index: pd.DatetimeIndex,
    *,
    floor: float = KELLY_FLOOR,
    cap: float = KELLY_CAP,
    fallback: float = KELLY_FALLBACK,
) -> np.ndarray | None:
    """Load per-bar Kelly K1 multiplier aligned to ``target_index``.

    Reindexes the sidecar's ``size_mult`` column onto the target bar index
    (no method= — exact match only; missing bars become NaN), replaces NaN
    with ``fallback``, and clips to ``[floor, cap]``. Mirrors the backtester
    contract at lines 1850-1878.

    Returns ``None`` (no-op) when ``path`` is ``None`` or the file does not
    exist — the live engine and backtester both interpret this as
    "Kelly disabled, multiplier ≡ 1.0 everywhere".
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning(
            "kelly_mult sidecar %s does not exist — skipping Kelly boost", p
        )
        return None

    df = pd.read_parquet(p)
    if "size_mult" in df.columns:
        ser = df["size_mult"]
    else:
        # Single-column sidecar: take the lone column. Defensive — current
        # production sidecar carries only ``size_mult``, but earlier
        # variants (K2, K3) may use other names.
        ser = df.iloc[:, 0]

    arr = ser.reindex(target_index).to_numpy(dtype=float)
    arr = np.where(np.isnan(arr), fallback, arr)
    arr = np.clip(arr, floor, cap)

    n_boost = int((arr > floor + 1e-9).sum())
    logger.info(
        "Kelly sidecar loaded (%s): %d bars with mult>%.2f, mean=%.4f, max=%.4f",
        p.name, n_boost, floor, float(arr.mean()), float(arr.max()),
    )
    return arr


def load_bar_exec_slip(
    path: str | Path | None,
    target_index: pd.DatetimeIndex,
    *,
    fallback_usd: float = SLIP_FALLBACK_USD,
    column: str = "bar_taker_slip_usd",
) -> np.ndarray | None:
    """Load per-bar empirical taker slip (USD per BTC, single leg).

    Forward-fills the sidecar onto ``target_index`` (last-known value for
    bars that don't appear in the sidecar — this is the leakage-safe
    walk-forward semantics) and replaces residual NaN with ``fallback_usd``.
    Mirrors the backtester contract at lines 1768-1790.

    Returns ``None`` when ``path`` is ``None`` or missing — caller falls
    back to its grid / flat slippage model.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning(
            "bar_exec_quality sidecar %s does not exist — falling back to grid", p
        )
        return None

    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index, utc=True)
    if column not in df.columns:
        logger.warning(
            "column %r not in bar_exec_quality sidecar %s — falling back", column, p
        )
        return None

    arr = (
        df[column]
        .reindex(target_index, method="ffill")
        .fillna(fallback_usd)
        .to_numpy(dtype=float)
    )
    logger.info(
        "Loaded realistic_taker slippage from %s: %d bars, median $%.2f/BTC",
        p.name, len(arr), float(np.nanmedian(arr)),
    )
    return arr


def load_aligner_betas(
    path: str | Path | None,
    *,
    pred_col: str = "y_pred",
    raw_col: str = "y_pred_raw",
) -> dict[str, float]:
    """Load per-calendar-month aligner β* as ``{"YYYY-MM": float}``.

    Auto-detects the sidecar format:

    * **JSON** (``*.json``): direct ``{"YYYY-MM": β*}`` mapping.
    * **Parquet** (everything else): compute ``mean(y_pred - y_pred_raw)``
      per calendar month, requiring both ``pred_col`` and ``raw_col`` to be
      present. This handles the production
      ``BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet`` shape.

    Returns ``{}`` when ``path`` is ``None`` or the file is missing — the
    live engine treats an empty dict as "no bias correction; predictions
    pass through raw".
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning("aligner sidecar %s does not exist — returning empty dict", p)
        return {}

    if p.suffix.lower() == ".json":
        with p.open("r", encoding="utf-8") as fh:
            raw = json.load(fh)
        return {str(k): float(v) for k, v in raw.items()}

    df = pd.read_parquet(p)
    if pred_col not in df.columns or raw_col not in df.columns:
        logger.warning(
            "aligner parquet %s missing %r and/or %r columns — returning empty",
            p, pred_col, raw_col,
        )
        return {}

    df.index = pd.to_datetime(df.index, utc=True)
    diff = df[pred_col] - df[raw_col]
    monthly = diff.groupby(diff.index.strftime("%Y-%m")).mean()
    out = {str(k): float(v) for k, v in monthly.items() if not pd.isna(v)}
    logger.info(
        "Loaded aligner betas from %s: %d months, median β=%+.4f, range [%+.4f, %+.4f]",
        p.name, len(out),
        float(np.median(list(out.values()))) if out else float("nan"),
        float(min(out.values())) if out else float("nan"),
        float(max(out.values())) if out else float("nan"),
    )
    return out


def lookup_beta_for_month(
    betas: dict[str, float],
    timestamp: pd.Timestamp,
    *,
    fallback_zero_warn: bool = True,
) -> float:
    """Pick the appropriate β* for ``timestamp``'s calendar month.

    Walk-forward fallback ladder:

    1. Exact match on ``timestamp.strftime("%Y-%m")``.
    2. Otherwise, the largest ``YYYY-MM`` in the dict that is ``<=`` the
       target month (most-recent-known β*). This is the live-trading case
       where the sidecar lags real time by a month or two.
    3. Otherwise, return ``0.0`` (no correction). Logs a warning unless
       ``fallback_zero_warn`` is False.
    """
    if not betas:
        if fallback_zero_warn:
            logger.warning("aligner betas dict empty — returning β*=0.0")
        return 0.0

    key = timestamp.strftime("%Y-%m")
    if key in betas:
        return float(betas[key])

    earlier = sorted(k for k in betas if k <= key)
    if earlier:
        chosen = earlier[-1]
        logger.info(
            "aligner β* fallback: %s not in sidecar, using nearest prior %s (β=%+.4f)",
            key, chosen, betas[chosen],
        )
        return float(betas[chosen])

    if fallback_zero_warn:
        logger.warning(
            "aligner β* fallback exhausted for %s (sidecar starts at %s) — β*=0.0",
            key, min(betas.keys()),
        )
    return 0.0


__all__ = [
    "KELLY_FLOOR",
    "KELLY_CAP",
    "KELLY_FALLBACK",
    "SLIP_FALLBACK_USD",
    "load_kelly_mult",
    "load_bar_exec_slip",
    "load_aligner_betas",
    "lookup_beta_for_month",
]
