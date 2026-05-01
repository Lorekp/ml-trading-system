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


def load_bar_exec_slip_series(
    path: str | Path | None,
    *,
    column: str = "bar_taker_slip_usd",
) -> pd.Series | None:
    """Load per-bar empirical taker slip (USD/BTC) as a tz-aware ``pd.Series``.

    Live-engine variant of ``load_bar_exec_slip``. Returns the raw series
    without target-index alignment; per-bar lookups go through
    ``lookup_slip_usd_at``. Returns ``None`` when ``path`` is missing or
    the column isn't present.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning(
            "bar_exec_quality sidecar %s does not exist — live slip disabled", p
        )
        return None
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index, utc=True)
    if column not in df.columns:
        logger.warning(
            "column %r not in bar_exec_quality sidecar %s — None", column, p
        )
        return None
    ser = df[column].astype(float)
    ser.name = column
    logger.info(
        "bar_exec_quality Series loaded (%s): %d bars, median $%.2f/BTC",
        p.name, len(ser), float(np.nanmedian(ser.to_numpy())),
    )
    return ser


def lookup_slip_usd_at(
    series: pd.Series | None,
    timestamp: pd.Timestamp,
    *,
    fallback_usd: float = SLIP_FALLBACK_USD,
) -> float:
    """Per-bar empirical slip USD/BTC. Forward-fill semantics via
    ``index.searchsorted``: use the nearest entry at-or-before
    ``timestamp`` (leakage-safe walk-forward). Falls back to
    ``fallback_usd`` when ``series`` is None or ``timestamp`` predates
    every entry in the sidecar.
    """
    if series is None or len(series) == 0:
        return fallback_usd
    # ``searchsorted(side="right")`` returns the insertion point AFTER
    # equal values, so the last index ≤ ts is searchsorted-1.
    idx = series.index.searchsorted(timestamp, side="right") - 1
    if idx < 0:
        return fallback_usd
    val = float(series.iloc[idx])
    if not np.isfinite(val):
        return fallback_usd
    return val


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


# E.1.11 (2026-05-01): hard bounds on aligner β*. Production betas
# observed in the bias-aligned sidecar fall in [-0.10, +0.20] across
# 50 months (median +0.052). [-0.5, +0.5] is a generous safety band
# that catches malformed / tampered JSON without rejecting realistic
# regime-shift values. Out-of-range entries log WARN and clip rather
# than crash — the live engine never wants a β* >> realistic range
# because (a) clip fails closed (bias=0) and (b) crash on every bar
# halts trading.
ALIGNER_BETA_MIN: float = -0.5
ALIGNER_BETA_MAX: float = +0.5


def _clip_beta(value: float, *, source: str) -> float:
    """Clip a single β* into the safety band, logging on out-of-range."""
    if not np.isfinite(value):
        logger.warning(
            "aligner β* non-finite at %s — substituting 0.0", source,
        )
        return 0.0
    if value < ALIGNER_BETA_MIN or value > ALIGNER_BETA_MAX:
        clipped = float(np.clip(value, ALIGNER_BETA_MIN, ALIGNER_BETA_MAX))
        logger.warning(
            "aligner β* %+.4f at %s out of [%+.2f, %+.2f] — clipped to %+.4f",
            value, source, ALIGNER_BETA_MIN, ALIGNER_BETA_MAX, clipped,
        )
        return clipped
    return float(value)


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

    Defensive contract (E.1.11 2026-05-01):

    * Malformed JSON (``json.JSONDecodeError``) logs ONLY the path and
      returns ``{}``. Never logs the bytes — guards against a tampered
      sidecar exfiltrating credentials by being echoed into log files.
    * Every β* is clipped to ``[ALIGNER_BETA_MIN, ALIGNER_BETA_MAX]``
      via :func:`_clip_beta`. Out-of-range values WARN and clip rather
      than crash so a bad sidecar can't halt trading.
    """
    if path is None:
        return {}
    p = Path(path)
    if not p.exists():
        logger.warning("aligner sidecar %s does not exist — returning empty dict", p)
        return {}

    if p.suffix.lower() == ".json":
        try:
            with p.open("r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except json.JSONDecodeError:
            logger.error(
                "aligner JSON %s malformed — returning empty dict (β* disabled)",
                p,                                                    # path only, NEVER bytes
            )
            return {}
        except OSError as exc:
            logger.error("aligner JSON %s unreadable (%s) — returning empty dict", p, exc)
            return {}
        if not isinstance(raw, dict):
            logger.error(
                "aligner JSON %s top-level is %s, expected dict — returning empty",
                p, type(raw).__name__,
            )
            return {}
        out: dict[str, float] = {}
        for k, v in raw.items():
            try:
                out[str(k)] = _clip_beta(float(v), source=f"{p.name}::{k}")
            except (TypeError, ValueError):
                logger.warning(
                    "aligner JSON %s key %r value %r non-numeric — skipped",
                    p.name, k, v,
                )
        return out

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
    out = {
        str(k): _clip_beta(float(v), source=f"{p.name}::{k}")
        for k, v in monthly.items() if not pd.isna(v)
    }
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


def load_kelly_mult_series(
    path: str | Path | None,
    *,
    column: str = "size_mult",
) -> pd.Series | None:
    """Load the per-bar Kelly multiplier as a tz-aware ``pd.Series``.

    Used by the live engine which needs scalar lookups by single
    timestamp rather than the array-aligned interface
    ``load_kelly_mult`` provides for the vectorised backtester.

    Returns ``None`` for missing path or missing column. The caller is
    expected to apply the floor / cap / NaN-fallback contract via
    ``lookup_kelly_mult_at`` for each lookup.
    """
    if path is None:
        return None
    p = Path(path)
    if not p.exists():
        logger.warning(
            "kelly_mult sidecar %s does not exist — Kelly disabled in live", p
        )
        return None
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index, utc=True)
    if column in df.columns:
        ser = df[column].astype(float)
    else:
        ser = df.iloc[:, 0].astype(float)
    ser.name = column
    logger.info(
        "Kelly sidecar loaded into Series (%s): %d bars, %d > %.2f, max=%.4f",
        p.name, len(ser),
        int((ser > KELLY_FLOOR + 1e-9).sum()), KELLY_FLOOR,
        float(ser.max()),
    )
    return ser


def lookup_kelly_mult_at(
    series: pd.Series | None,
    timestamp: pd.Timestamp,
    *,
    floor: float = KELLY_FLOOR,
    cap: float = KELLY_CAP,
    fallback: float = KELLY_FALLBACK,
    signal_direction: int | None = None,
) -> float:
    """Return the Kelly multiplier for a single ``timestamp``.

    Walk-forward fallback ladder (E.1.5 2026-05-01) — mirrors the
    leakage-safe semantics of :func:`lookup_slip_usd_at` exactly:

    1. **Walk-forward lookup**: ``index.searchsorted(side="right") - 1``
       returns the nearest sidecar entry at-or-before ``timestamp``.
       Pre-E.1.5 this used ``series.get(timestamp, fallback)`` (exact
       match) which silently disabled Kelly when live bar timestamps
       didn't equal a sidecar bucket key (clock drift, missed bar).
    2. NaN / missing / pre-history → ``fallback`` (1.0 default).
    3. Always clipped to ``[floor, cap]`` so a malformed sidecar can't
       produce risk-unbounded sizing.

    Short-side defence-in-depth (E.1.5):

    * When ``signal_direction == -1``, unconditionally returns
      ``floor`` (1.0 by default) regardless of sidecar contents.
      Defends against a future K2/K3 sidecar regen that accidentally
      writes ``mult > 1.0`` on shorts. The size-position layer
      already short-circuits at ``mult <= 1.0 + 1e-12``, but having
      the asymmetry enforced at the lookup keeps a single source of
      truth.

    ``series`` ``None`` → ``fallback`` (Kelly disabled globally).
    """
    if signal_direction == -1:
        return float(np.clip(floor, floor, cap))
    if series is None or len(series) == 0:
        return fallback
    idx = series.index.searchsorted(timestamp, side="right") - 1
    if idx < 0:
        return fallback
    val = float(series.iloc[idx])
    if not np.isfinite(val):
        return fallback
    return float(np.clip(val, floor, cap))


__all__ = [
    "KELLY_FLOOR",
    "KELLY_CAP",
    "KELLY_FALLBACK",
    "SLIP_FALLBACK_USD",
    "load_kelly_mult",
    "load_kelly_mult_series",
    "lookup_kelly_mult_at",
    "load_bar_exec_slip",
    "load_bar_exec_slip_series",
    "lookup_slip_usd_at",
    "load_aligner_betas",
    "lookup_beta_for_month",
]
