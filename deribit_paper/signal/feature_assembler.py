"""Build the LightGBM feature row at each bar close.

**W3.1 rewrite (2026-04-11)**: the assembler is now MANIFEST-DRIVEN and
fails FAST on unknown features, instead of the earlier hard-coded
override list that silently zeroed anything missing from the baseline.

The current production variant is the T2.1 pruned+SVI model from Stage
18: 44 features, of which 16 are TransVol embeddings, 27 are pruned
base features (TSRV-derived, VPIN, VOV, Parkinson, Hurst, imbalance,
microstructure), and 1 is ``svi_atm_1d`` (the constant-maturity 1-day
ATM implied vol).  The assembler handles all three sources:

1. **TransVol embeddings** (``tv_0..tv_15``) — computed fresh at each
   bar close from the live perp-trade window via the TransVol encoder.
   Passed into :meth:`build_row` as a 16-dim numpy array.
2. **Live-computable features** — registered in the
   ``_LIVE_HANDLERS`` dispatch table.  A handler takes an
   ``AssemblerContext`` and returns a float.  Current handlers:
   TSRV-derived fields (``tsrv``, ``rv_fast``, ``bpv``, ``medrv``,
   ``continuous_var``, ``jump_var``, ``jump_fraction``, ``rs_down``,
   ``rs_up``, ``down_jump_fraction``, ``up_jump_frac``, ``vol_ann``,
   ``n_trades``, ``duration_s``, ``n_bars_in_window``,
   ``tsrv_mean_1d``) plus ``svi_atm_1d`` (from the live SVI calibrator).
3. **Historical baseline** — the merged (base features + SVI features)
   parquet row loaded at startup.  Any feature in the manifest that is
   NOT a TransVol slot AND has no live handler falls back to the
   baseline row's value.  If it's not in the baseline either,
   :meth:`__init__` raises ``ValueError`` immediately — we never
   silently emit a zero for a feature the model was trained to use.

The earlier ``_LIVE_OVERRIDE_FIELDS`` tuple + imperative overwrite
loop is gone.  Adding a new live feature is now a one-liner: register
a handler in ``_LIVE_HANDLERS``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from deribit_paper.features.dollar_bar_buffer import DollarBarBuffer
from deribit_paper.features.dollar_bar_features import DollarBarFeatures
from deribit_paper.features.har_pd_live import (
    HAR_PD_RAW_FEATURES,
    K_BARS_DEFAULT as HAR_PD_K_BARS_DEFAULT,
    HARPDState,
)
from deribit_paper.features.tsrv_dollar_bar_grid import TSRVDollarBarFeatures
from deribit_paper.features.tsrv_features import TSRVFeatures
from deribit_paper.features.tsrv_history import TSRVHistoryBuffer
from deribit_paper.features.tsrv_live import BarTSRV
from deribit_paper.market_data.aggregator import Bar

logger = logging.getLogger(__name__)


# ────────────────────────────────────────────────────────────
# Dispatch table: live-computable features
# ────────────────────────────────────────────────────────────


@dataclass
class AssemblerContext:
    """Inputs available to a live-handler function for one bar.

    Kept minimal: only the fields a handler actually needs.  Pass-
    through objects (``svi_calibrator``, baseline row) are available
    for handlers that need them.

    ``har_pd_features`` is a per-bar precomputed dict of the 6 HAR-PD
    feature values populated by :meth:`FeatureAssembler.build_row`
    BEFORE the handler loop fires; the 6 HAR-PD live handlers just
    look up by name. Empty dict during HAR-PD warmup; in that case
    the handlers fall back to the historical baseline value, matching
    the rest of the assembler's fail-soft behaviour.
    """
    bar: Bar
    bar_tsrv: BarTSRV
    svi_calibrator: Any | None       # LiveSVICalibrator, late-bound
    baseline_row: pd.Series          # historical baseline for fallbacks
    har_pd_features: dict[str, float] = field(default_factory=dict)
    # F3 Tier 3 (parity-cycle round 3 — 2026-05-07): dollar-bar buffer +
    # cached-features view. Both ``None`` until Tier 6 wires the live
    # ``DollarBarSpotStream``-fed buffer into the engine. While ``None``,
    # the 16 dollar-bar-derived handlers fall back to the historical
    # baseline (same fail-soft pattern as HAR-PD warmup / SVI cold start).
    dollar_bar_buffer: Any | None = None        # DollarBarBuffer, late-bound
    dollar_bar_features: Any | None = None      # DollarBarFeatures, late-bound
    # F3 Tier 5 (parity-cycle round 3 — 2026-05-07): TSRV history
    # buffer + cached-features view for the 6 TSRV-derived handlers
    # (tsrv_mean_*, tsrv_ffd, up_jump_frac_ema50, bpv_tsrv_ratio_ema50).
    # ``None`` until Tier 6 wires the engine; while None those handlers
    # fall back to the historical baseline.
    tsrv_history_buffer: Any | None = None      # TSRVHistoryBuffer, late-bound
    tsrv_features: Any | None = None            # TSRVFeatures, late-bound
    # S-1d (parity-r3 cycle 7 — 2026-05-08): TSRV-on-dollar-bar-grid
    # wrapper. When wired (both DollarBarBuffer + TSRVHistoryBuffer
    # present), provides byte-equivalent-to-offline computations for
    # the 30 Tier-B columns. Falls back to ``tsrv_features``
    # (15-min-cadence) when None — preserves cycle-5/6 behaviour for
    # tests + cold start.
    tsrv_db_features: Any | None = None         # TSRVDollarBarFeatures, late-bound
    # S-1c (parity-r3 cycle 6 — 2026-05-08): live funding-rate provider
    # for the 5 ``funding_basis_*`` columns. ``None`` falls back to
    # baseline-fallback (matches HAR-PD warmup pattern).
    funding_provider: Any | None = None         # FundingRateProvider, late-bound


def _h_tsrv(ctx: AssemblerContext) -> float:
    return float(ctx.bar_tsrv.tsrv)


def _h_rv_fast(ctx: AssemblerContext) -> float:
    return float(ctx.bar_tsrv.rv_fast)


def _h_rv_slow(ctx: AssemblerContext) -> float:
    return float(ctx.bar_tsrv.rv_slow)


def _h_bpv(ctx: AssemblerContext) -> float:
    return float(ctx.bar_tsrv.bpv)


def _h_medrv(ctx: AssemblerContext) -> float:
    return float(ctx.bar_tsrv.medrv)


def _h_continuous_var(ctx: AssemblerContext) -> float:
    return float(ctx.bar_tsrv.continuous_var)


def _h_jump_var(ctx: AssemblerContext) -> float:
    return float(ctx.bar_tsrv.jump_var)


def _h_jump_fraction(ctx: AssemblerContext) -> float:
    return float(ctx.bar_tsrv.jump_fraction)


def _h_rs_down(ctx: AssemblerContext) -> float:
    return float(ctx.bar_tsrv.rs_down)


def _h_rs_up(ctx: AssemblerContext) -> float:
    return float(ctx.bar_tsrv.rs_up)


def _h_down_jump_fraction(ctx: AssemblerContext) -> float:
    return float(ctx.bar_tsrv.down_jump_fraction)


def _h_up_jump_frac(ctx: AssemblerContext) -> float:
    """S-1d (parity-r3 cycle 7 — 2026-05-08): prefer the dollar-bar-
    grid value (offline byte-equivalent via ``_ffill_to_bars``) over
    the current BarTSRV value. The current BarTSRV value is a
    1-bar look-ahead vs offline (offline uses shift(1) before ffill)."""
    db = ctx.tsrv_db_features
    if db is not None:
        v = db.up_jump_frac()
        if v is not None and np.isfinite(v):
            return float(v)
    feats = ctx.tsrv_features
    if feats is not None:
        v = feats.up_jump_frac()
        if v is not None and np.isfinite(v):
            return float(v)
    # Legacy fallback (test fixtures with only bar_tsrv, no buffers).
    if ctx.bar_tsrv is not None:
        return float(ctx.bar_tsrv.up_jump_fraction)
    return _baseline_or_nan(ctx, "up_jump_frac")


def _h_vol_ann(ctx: AssemblerContext) -> float:
    # F2 (2026-05-07): the offline TSRV parquet stores
    # ``vol_ann = sqrt(tsrv * periods_per_year)`` (TSRV-based, see
    # ``TSRV/tsrv_series.py:288-289``). Pre-fix the live handler
    # returned ``vol_ann_continuous`` (continuous-variance based),
    # producing a ~12% gap on jump bars (sqrt(tsrv) vs
    # sqrt(continuous_var) — they differ by the jump component).
    # The pre-fix comment was incorrect — the BACKTESTER ingests
    # the offline parquet's ``vol_ann`` field, which IS the
    # tsrv-based one. Use ``bar_tsrv.vol_ann`` to match.
    return float(ctx.bar_tsrv.vol_ann)


def _h_n_trades(ctx: AssemblerContext) -> float:
    """Last sub-bar's n_trades (F5, parity-cycle round 3 — 2026-05-07).

    Pre-F5: returned the WALL-CLOCK 15-min total perp trade count.
    Offline ``_window_and_aggregate`` takes ``series.groupby(window_labels).last()``
    which is the LAST sub-bar's n_trades within the 15-min window —
    a different quantity. With the dollar-bar buffer wired live, we
    return the latest dollar bar's n_trades to match offline.

    F-7 (parity-r3 cycle 3 — 2026-05-08): warmup fallback now reads
    ``baseline_row["n_trades"]`` rather than the WALL-CLOCK 15-min
    count from ``ctx.bar.n_perp_trades``. The wall-clock count is a
    sum across the full 15-min window — a different quantity from
    offline's last-sub-bar groupby. The baseline row preserves
    last-sub-bar semantics built from offline parquet at warmup, so
    using it keeps the warmup row monotonic with offline. Buffer
    fallback is fine for tests that explicitly avoid a baseline.
    """
    buf = ctx.dollar_bar_buffer
    if buf is not None and len(buf) > 0:
        bar = buf.latest_bar()
        if bar is not None:
            return float(bar.n_trades)
    # Warmup fallback (F-7): prefer the baseline_row's last-sub-bar
    # value over the 15-min wall-clock sum.
    if ctx.baseline_row is not None and "n_trades" in ctx.baseline_row.index:
        baseline_val = ctx.baseline_row.get("n_trades", np.nan)
        if pd.notna(baseline_val):
            return float(baseline_val)
    return float(ctx.bar.n_perp_trades)


def _h_duration_s(ctx: AssemblerContext) -> float:
    """Last sub-bar's duration_s (F4, parity-cycle round 3 — 2026-05-07).

    Pre-F4: returned the MEAN inter-trade gap within the 15-min window.
    Offline ``_window_and_aggregate`` uses ``series.groupby(window_labels).last()``
    on the dollar-bar duration_s series — i.e. the last sub-bar's full
    duration. With the dollar-bar buffer wired live, we mirror that.

    F-7 (parity-r3 cycle 3 — 2026-05-08): warmup fallback now reads
    ``baseline_row["duration_s"]`` instead of synthesising a mean
    inter-trade gap from ``ctx.bar.perp_trades``. The mean-gap
    calculation is wall-clock-window-derived and diverges from
    offline's last-sub-bar groupby; the baseline row holds the offline
    last-sub-bar value built at warmup. Falls back to the inter-trade
    gap only if no baseline column is available (legacy contract).
    """
    buf = ctx.dollar_bar_buffer
    if buf is not None and len(buf) > 0:
        bar = buf.latest_bar()
        if bar is not None and np.isfinite(bar.duration_s):
            return float(bar.duration_s)
    # Warmup fallback (F-7): prefer the baseline_row's last-sub-bar
    # value over the synthesised mean inter-trade gap.
    if ctx.baseline_row is not None and "duration_s" in ctx.baseline_row.index:
        baseline_val = ctx.baseline_row.get("duration_s", np.nan)
        if pd.notna(baseline_val) and float(baseline_val) > 0.0:
            return float(baseline_val)
    if not ctx.bar.perp_trades:
        return 1.0
    first = ctx.bar.perp_trades[0].timestamp
    last = ctx.bar.perp_trades[-1].timestamp
    n_pairs = max(len(ctx.bar.perp_trades) - 1, 1)
    return max((last - first).total_seconds() / n_pairs, 1e-6)


def _h_n_bars_in_window(ctx: AssemblerContext) -> float:
    return float(ctx.baseline_row.get("n_bars_in_window", 1) or 1)


def _baseline_or_nan(ctx: AssemblerContext, name: str) -> float:
    """Read ``name`` from the baseline row; return NaN if missing or NaN.

    F9 (parity-cycle round 2 — 2026-05-07): pre-fix the baseline-fallback
    handlers used ``ctx.baseline_row.get(name, 0.0) or 0.0``, which
    collapses NaN (and any falsy value) to 0.0 — defeating the
    line-519 ``~np.isfinite(values)`` fail-fast guard. Couples with F1
    (SVI parquet integer-position join → 100% NaN baseline) to silently
    feed ``svi_atm_1d=0%`` to LightGBM.

    Post-fix: NaN propagates so the guard catches a broken baseline
    and refuses to trade. Callers that NEED a sensible default (e.g.
    ``n_bars_in_window=1``) keep their inline ``or X`` after this
    helper if the value is non-NaN.
    """
    val = ctx.baseline_row.get(name, np.nan)
    if val is None:
        return float("nan")
    val = float(val)
    return val if np.isfinite(val) else float("nan")


def _h_tsrv_mean_1d(ctx: AssemblerContext) -> float:
    """tsrv_mean_1d: rolling mean of log(vol_ann) over the bpd × 1
    window. S-1d (cycle 7) prefers the dollar-bar-grid path (byte-
    equivalent to offline ``rolling(w).mean()`` on ffilled-to-db
    log_vol_ann); falls back to the cycle-5/6 15-min cadence path
    when no DollarBarBuffer is wired."""
    db = ctx.tsrv_db_features
    if db is not None:
        v = db.tsrv_mean_1d()
        if v is not None and np.isfinite(v):
            return float(v)
    feats = ctx.tsrv_features
    if feats is not None:
        v = feats.tsrv_mean_1d()
        if v is not None and np.isfinite(v):
            return float(v)
    return _baseline_or_nan(ctx, "tsrv_mean_1d")


def _make_har_pd_handler(name: str) -> Callable[[AssemblerContext], float]:
    """Return a live handler that reads the named HAR-PD feature from
    ``ctx.har_pd_features`` (precomputed once per bar by the assembler).

    During HAR-PD warmup (first 96 bars before pre-warm seed completes,
    or when no `bar_tsrv` was passed in), ``har_pd_features`` is empty
    and the handler falls back to the historical baseline value —
    matching the same fail-soft behaviour as ``svi_atm_1d`` when the
    SVI calibrator hasn't yet produced live params. F9 (2026-05-07):
    NaN baseline propagates to line-519 guard.
    """
    def _handler(ctx: AssemblerContext) -> float:
        live = ctx.har_pd_features.get(name)
        if live is None or not np.isfinite(live):
            return _baseline_or_nan(ctx, name)
        return float(live)

    _handler.__name__ = f"_h_{name}"
    return _handler


def _h_svi_atm_1d(ctx: AssemblerContext) -> float:
    """1-day constant-maturity ATM implied vol (percent units).

    Reconstructs the ArbitrageFreeSVI surface from the calibrator's
    most recent bucket of per-expiry slices and interpolates in total
    variance to T = 1 day.  Matches the offline scale used to train
    the T2.1 models (``vol_arb/output/svi_features_15m.parquet``).
    Falls back to the historical baseline value if the calibrator has
    no cached params yet. F9 (2026-05-07): NaN baseline propagates to
    line-519 guard rather than silently emitting 0% IV.

    F-4 fix (round-3 audit, 2026-05-08): pass ``ctx.bar.timestamp`` to
    ``atm_vol_at_dte`` so the calibrator only returns buckets whose
    ``bucket_ts + bucket_freq <= bar.timestamp`` — matches offline's
    `+ bucket_freq` reindex shift in ``rebuild_svi_features_with_spot.py``.
    Pre-fix live IV was up to ~1h fresher than the model trained on.
    """
    if ctx.svi_calibrator is None:
        return _baseline_or_nan(ctx, "svi_atm_1d")
    live = ctx.svi_calibrator.atm_vol_at_dte(1.0, now=ctx.bar.timestamp)
    if live is None or not np.isfinite(live):
        return _baseline_or_nan(ctx, "svi_atm_1d")
    return float(live)


# ── F3 Tier 3 (parity-cycle round 3 — 2026-05-07): 16 dollar-bar handlers ──
# Each handler reads from ``ctx.dollar_bar_features`` (a per-bar cached
# wrapper around the live ``DollarBarBuffer``). When the buffer is None
# or too short for the underlying operation, the wrapper returns
# ``None`` and the handler falls back to the historical baseline via
# ``_baseline_or_nan`` — same fail-soft pattern as HAR-PD warmup.


def _time_to_funding_seconds(ts: pd.Timestamp) -> float:
    """Seconds until the next 8/16/24h UTC funding boundary.

    Mirror of ``vol_model/features/engineer.py:417-419``::

        h = ts.hour + ts.minute/60 + ts.second/3600
        return (8 - h if h < 8 else
                16 - h if h < 16 else
                24 - h) * 3600
    """
    h = ts.hour + ts.minute / 60.0 + ts.second / 3600.0
    if h < 8.0:
        return (8.0 - h) * 3600.0
    if h < 16.0:
        return (16.0 - h) * 3600.0
    return (24.0 - h) * 3600.0


def _effective_timestamp_for_funding(ctx: AssemblerContext) -> pd.Timestamp:
    """Pick the timestamp the offline pipeline used for time_to_funding.

    S-1c (parity-r3 cycle 6 — 2026-05-08): offline computes
    ``time_to_funding`` on the dollar-bar grid, then aggregates to
    15-min via ``groupby(window_labels).last()`` — i.e. the value at
    the 15-min row is the LAST dollar bar's value in that window.
    Pre-cycle-6 the live handler used ``ctx.bar.timestamp`` (the
    15-min wall-clock boundary), which deviates by up to ~15 minutes
    from the offline last-dollar-bar timestamp. The fix: when a
    ``DollarBarBuffer`` is present, use its latest bar's timestamp
    instead. Falls back to the bar timestamp when no buffer is wired
    (test fixtures, cold start).
    """
    buf = getattr(ctx, "dollar_bar_buffer", None)
    if buf is not None and len(buf) > 0:
        last = buf.latest_bar()
        if last is not None and last.timestamp is not None:
            return pd.Timestamp(last.timestamp)
    return ctx.bar.timestamp


def _h_time_to_funding(ctx: AssemblerContext) -> float:
    """F3 Tier 6 — seconds until next 8h funding boundary.

    Cycle 6: timestamp is the latest dollar bar's microsecond-precision
    timestamp when a buffer is wired (matches offline ``groupby(...)
    .last()`` semantics); falls back to ``ctx.bar.timestamp`` (the
    15-min wall-clock boundary) when no buffer is available.
    """
    return _time_to_funding_seconds(_effective_timestamp_for_funding(ctx))


def _h_time_to_funding_lag1(ctx: AssemblerContext) -> float:
    """time_to_funding at the 1-bar-prior dollar bar.

    S-1d (parity-r3 cycle 7 — 2026-05-08): walks back through actual
    sub-15-min DollarBarBuffer timestamps. Offline computes
    ``series.shift(1)`` on the dollar-bar grid; the n-th-prior dollar
    bar's timestamp is what offline reads, NOT the 15-min wall-clock
    earlier value. Falls back to the cycle-6 ``eff_ts - 15min`` path
    when no DollarBarBuffer is wired.
    """
    buf = getattr(ctx, "dollar_bar_buffer", None)
    if buf is not None and len(buf) >= 2:
        prev_ts = buf.timestamp_at(-2)
        if prev_ts is not None:
            return _time_to_funding_seconds(prev_ts)
    eff_ts = _effective_timestamp_for_funding(ctx)
    return _time_to_funding_seconds(eff_ts - pd.Timedelta(minutes=15))


def _h_time_to_funding_lag3(ctx: AssemblerContext) -> float:
    """time_to_funding at the 3-bars-prior dollar bar."""
    buf = getattr(ctx, "dollar_bar_buffer", None)
    if buf is not None and len(buf) >= 4:
        prev_ts = buf.timestamp_at(-4)
        if prev_ts is not None:
            return _time_to_funding_seconds(prev_ts)
    eff_ts = _effective_timestamp_for_funding(ctx)
    return _time_to_funding_seconds(eff_ts - pd.Timedelta(minutes=15 * 3))


def _time_to_funding_ema_on_dollar_bars(
    ctx: AssemblerContext, *, span: int,
) -> float:
    """EWM(span) over the time_to_funding series at each dollar-bar
    timestamp in the buffer (S-1d cycle 7).

    Mirrors offline ``feats["time_to_funding"].ewm(span=span,
    min_periods=1).mean()``, which is computed on the dollar-bar grid
    and then aggregated to 15-min via ``.groupby(window_labels)
    .last()``. The latest dollar bar IS the last in the current
    15-min window so its EMA value matches offline byte-equal.

    Falls back to the cycle-6 15-min-cadence reconstruction when no
    DollarBarBuffer is wired.
    """
    buf = getattr(ctx, "dollar_bar_buffer", None)
    if buf is not None and len(buf) > 0:
        timestamps = list(buf.as_index())
        series = pd.Series(
            [_time_to_funding_seconds(t) for t in timestamps],
            dtype=np.float64,
        )
        ema_val = series.ewm(span=span, adjust=True, min_periods=1).mean().iloc[-1]
        return float(ema_val)
    return _time_to_funding_ema(
        _effective_timestamp_for_funding(ctx), span=span,
    )


def _h_time_to_funding_ema10(ctx: AssemblerContext) -> float:
    """EWM(span=10) over the time_to_funding series on the dollar-bar grid."""
    return _time_to_funding_ema_on_dollar_bars(ctx, span=10)


def _h_time_to_funding_ema50(ctx: AssemblerContext) -> float:
    """EWM(span=50) over the time_to_funding series on the dollar-bar grid."""
    return _time_to_funding_ema_on_dollar_bars(ctx, span=50)


def _time_to_funding_ema(ts: pd.Timestamp, *, span: int) -> float:
    """Helper: compute the EWM(span) of the time_to_funding series at
    ``ts`` with enough historical bars to converge.

    Mirrors offline ``feats["time_to_funding"].ewm(span=span,
    min_periods=1).mean().iloc[-1]``. We reconstruct the trailing
    ``5*span`` bars (more than enough for the EMA to converge given
    the smooth 8h-period sawtooth) and evaluate the EMA at the last
    point.
    """
    n = max(5 * span, 32)
    timestamps = [ts - pd.Timedelta(minutes=15 * (n - 1 - i)) for i in range(n)]
    series = pd.Series(
        [_time_to_funding_seconds(t) for t in timestamps],
        dtype=np.float64,
    )
    ema_val = series.ewm(span=span, adjust=True, min_periods=1).mean().iloc[-1]
    return float(ema_val)


def _make_dollar_bar_handler(
    name: str,
    method: str,
    *args: Any,
) -> Callable[[AssemblerContext], float]:
    """Build a handler that calls ``ctx.dollar_bar_features.<method>(*args)``
    and falls back to the baseline row's ``name`` value when the wrapper
    returns ``None`` or a non-finite result.
    """
    def _handler(ctx: AssemblerContext) -> float:
        feats = ctx.dollar_bar_features
        if feats is None:
            return _baseline_or_nan(ctx, name)
        try:
            v = getattr(feats, method)(*args)
        except (KeyError, ValueError, TypeError):
            return _baseline_or_nan(ctx, name)
        if v is None or not np.isfinite(v):
            return _baseline_or_nan(ctx, name)
        return float(v)
    _handler.__name__ = f"_h_{name}"
    return _handler


def _h_parkinson_1d(ctx: "AssemblerContext") -> float:
    """F-3 (parity-r3 cycle 2 — 2026-05-08): parkinson_1d with dynamic bpd.

    Pre-fix the window was hardcoded to 250. Offline derives ``bpd``
    per training run via ``avg_bars_per_day(bars.index, trailing_days=30)``
    in ``vol_model/features/engineer.py:485``; the rolling window then
    matches per-asset / per-regime activity. Live now reads bpd from
    the dollar-bar buffer at handler-call time so the rolling window
    matches the offline pipeline byte-for-byte (within the cold-start
    fallback at bpd=250).
    """
    feats = ctx.dollar_bar_features
    if feats is None:
        return _baseline_or_nan(ctx, "parkinson_1d")
    try:
        bpd = feats.bars_per_day()
        v = feats.parkinson_1d(bpd)
    except (KeyError, ValueError, TypeError):
        return _baseline_or_nan(ctx, "parkinson_1d")
    if v is None or not np.isfinite(v):
        return _baseline_or_nan(ctx, "parkinson_1d")
    return float(v)


# ── S-1b (parity-r3 cycle 5 — 2026-05-08): 3 TransVol-input bases ──
# The encoder consumes 101 features; pre-S-1b the live engine had no
# path to hurst_100 / ffd_rv / rv_zscore_*, so the corresponding
# `tv_X` slots silently fell back to baseline_row. These handlers now
# bind the live computation; lag/EMA variants follow in S-1c (require
# additional per-bar caching).


def _h_hurst_100(ctx: "AssemblerContext") -> float:
    """S-1b: rolling Hurst exponent over the trailing 100 dollar-bar
    log-returns. Mirrors offline ``_hurst_subsampled(returns, window=100)``
    evaluated at the latest bar (with the offline subsample-interpolate
    path collapsing to the bar's own R/S regression — they agree at every
    position p ∈ {window, window+step, ...})."""
    feats = ctx.dollar_bar_features
    if feats is None:
        return _baseline_or_nan(ctx, "hurst_100")
    try:
        v = feats.hurst_100(window=100)
    except (KeyError, ValueError, TypeError):
        return _baseline_or_nan(ctx, "hurst_100")
    if v is None or not np.isfinite(v):
        return _baseline_or_nan(ctx, "hurst_100")
    return float(v)


def _h_ffd_rv(ctx: "AssemblerContext") -> float:
    """S-1b: latest fractionally-differenced log-close (d=0.4, threshold=1e-5).
    Byte-equal to offline ``frac_diff_ffd(np.log(close))`` at the last
    index. Falls back to baseline when the buffer is shorter than the
    FFD weight window (~1458 bars at d=0.4)."""
    feats = ctx.dollar_bar_features
    if feats is None:
        return _baseline_or_nan(ctx, "ffd_rv")
    try:
        v = feats.ffd_rv(d=0.4, threshold=1e-5)
    except (KeyError, ValueError, TypeError):
        return _baseline_or_nan(ctx, "ffd_rv")
    if v is None or not np.isfinite(v):
        return _baseline_or_nan(ctx, "ffd_rv")
    return float(v)


def _make_rv_zscore_handler(
    horizon: str,
) -> "Callable[[AssemblerContext], float]":
    """S-1b: build a handler for ``rv_zscore_{horizon}``.

    Mirrors offline ``engineer.py:233-240``. ``horizon`` ∈ {1d, 1w, 1m};
    the underlying window is ``bpd × {1, 7, 30}`` clipped at offline's
    caps. ``DollarBarFeatures.rv_zscore`` resolves bpd live so the
    window adapts to per-asset bar density.
    """
    feature_name = f"rv_zscore_{horizon}"

    def _handler(ctx: "AssemblerContext") -> float:
        feats = ctx.dollar_bar_features
        if feats is None:
            return _baseline_or_nan(ctx, feature_name)
        try:
            v = feats.rv_zscore(horizon)
        except (KeyError, ValueError, TypeError):
            return _baseline_or_nan(ctx, feature_name)
        if v is None or not np.isfinite(v):
            return _baseline_or_nan(ctx, feature_name)
        return float(v)

    _handler.__name__ = f"_h_{feature_name}"
    return _handler


def _h_parkinson_1d_ema50(ctx: "AssemblerContext") -> float:
    """F-3 (parity-r3 cycle 2 — 2026-05-08): parkinson_1d_ema50 with
    dynamic ``rolling_window=bpd``.

    Pairs with :func:`_h_parkinson_1d` — both feed the same rolling-mean
    series internally (``DollarBarFeatures._parkinson_1d_rolling_series``
    caches per window), so calling both with the same ``bpd`` keeps the
    cache hit-rate high. The ``span`` stays 50 — that's the EMA span
    over the rolling-mean column, not a window count.
    """
    feats = ctx.dollar_bar_features
    if feats is None:
        return _baseline_or_nan(ctx, "parkinson_1d_ema50")
    try:
        bpd = feats.bars_per_day()
        v = feats.parkinson_1d_ema(50, rolling_window=bpd)
    except (KeyError, ValueError, TypeError):
        return _baseline_or_nan(ctx, "parkinson_1d_ema50")
    if v is None or not np.isfinite(v):
        return _baseline_or_nan(ctx, "parkinson_1d_ema50")
    return float(v)


# ── F3 Tier 5 (parity-cycle round 3 — 2026-05-07): 6 TSRV-grid handlers ──
# Each reads from ``ctx.tsrv_features`` (cached wrapper around the live
# ``TSRVHistoryBuffer``). Falls back to ``_baseline_or_nan`` when the
# buffer is None or underfull.


def _make_tsrv_handler(
    name: str,
    method: str,
    *args: Any,
) -> Callable[[AssemblerContext], float]:
    """Build a handler that calls ``ctx.tsrv_features.<method>(*args)``."""
    def _handler(ctx: AssemblerContext) -> float:
        feats = ctx.tsrv_features
        if feats is None:
            return _baseline_or_nan(ctx, name)
        try:
            v = getattr(feats, method)(*args)
        except (KeyError, ValueError, TypeError):
            return _baseline_or_nan(ctx, name)
        if v is None or not np.isfinite(v):
            return _baseline_or_nan(ctx, name)
        return float(v)
    _handler.__name__ = f"_h_{name}"
    return _handler


def _make_tsrv_db_handler(
    name: str,
    db_method: str,
    fallback_method: str,
    *args: Any,
) -> Callable[[AssemblerContext], float]:
    """S-1d (parity-r3 cycle 7 — 2026-05-08): handler that prefers
    the dollar-bar-grid computation (byte-equivalent to offline)
    over the 15-min-cadence fallback.

    Resolution order:
    1. ``ctx.tsrv_db_features.<db_method>(*args)`` — strict-parity
       path on the dollar-bar grid.
    2. ``ctx.tsrv_features.<fallback_method>(*args)`` — cycle-5/6
       15-min-cadence path; preserved for cold start / tests.
    3. ``_baseline_or_nan`` — historical baseline fallback (F9 NaN
       propagation contract).
    """
    def _handler(ctx: AssemblerContext) -> float:
        db_feats = ctx.tsrv_db_features
        if db_feats is not None:
            try:
                v = getattr(db_feats, db_method)(*args)
            except (KeyError, ValueError, TypeError):
                v = None
            if v is not None and np.isfinite(v):
                return float(v)
        feats = ctx.tsrv_features
        if feats is not None:
            try:
                v = getattr(feats, fallback_method)(*args)
            except (KeyError, ValueError, TypeError):
                v = None
            if v is not None and np.isfinite(v):
                return float(v)
        return _baseline_or_nan(ctx, name)
    _handler.__name__ = f"_h_{name}"
    return _handler


# ── S-1c (parity-r3 cycle 6 — 2026-05-08): variant-specific factories ──
# parkinson_1d_{lag1, lag3, ema10}: read the ROLLING-mean series (window
# = bpd) per the F-2/F-3 contract. The wrapper's ``parkinson_1d_lag``
# / ``parkinson_1d_ema`` accept a ``rolling_window`` kwarg that we
# bind to the live-resolved bpd value at handler-call time.

def _make_parkinson_1d_lag_handler(n: int) -> Callable[[AssemblerContext], float]:
    name = f"parkinson_1d_lag{n}"
    def _handler(ctx: AssemblerContext) -> float:
        feats = ctx.dollar_bar_features
        if feats is None:
            return _baseline_or_nan(ctx, name)
        try:
            bpd = feats.bars_per_day()
            v = feats.parkinson_1d_lag(n, rolling_window=bpd)
        except (KeyError, ValueError, TypeError):
            return _baseline_or_nan(ctx, name)
        if v is None or not np.isfinite(v):
            return _baseline_or_nan(ctx, name)
        return float(v)
    _handler.__name__ = f"_h_{name}"
    return _handler


def _make_parkinson_1d_ema_handler(span: int) -> Callable[[AssemblerContext], float]:
    name = f"parkinson_1d_ema{span}"
    def _handler(ctx: AssemblerContext) -> float:
        feats = ctx.dollar_bar_features
        if feats is None:
            return _baseline_or_nan(ctx, name)
        try:
            bpd = feats.bars_per_day()
            v = feats.parkinson_1d_ema(span, rolling_window=bpd)
        except (KeyError, ValueError, TypeError):
            return _baseline_or_nan(ctx, name)
        if v is None or not np.isfinite(v):
            return _baseline_or_nan(ctx, name)
        return float(v)
    _handler.__name__ = f"_h_{name}"
    return _handler


# ── S-1c: hurst_100 / ffd_rv / rv_zscore variant factories ──

def _make_hurst_lag_handler(n: int) -> Callable[[AssemblerContext], float]:
    name = f"hurst_100_lag{n}"
    def _handler(ctx: AssemblerContext) -> float:
        feats = ctx.dollar_bar_features
        if feats is None:
            return _baseline_or_nan(ctx, name)
        try:
            v = feats.hurst_100_lag(n)
        except (KeyError, ValueError, TypeError):
            return _baseline_or_nan(ctx, name)
        if v is None or not np.isfinite(v):
            return _baseline_or_nan(ctx, name)
        return float(v)
    _handler.__name__ = f"_h_{name}"
    return _handler


def _make_hurst_ema_handler(span: int) -> Callable[[AssemblerContext], float]:
    name = f"hurst_100_ema{span}"
    def _handler(ctx: AssemblerContext) -> float:
        feats = ctx.dollar_bar_features
        if feats is None:
            return _baseline_or_nan(ctx, name)
        try:
            v = feats.hurst_100_ema(span)
        except (KeyError, ValueError, TypeError):
            return _baseline_or_nan(ctx, name)
        if v is None or not np.isfinite(v):
            return _baseline_or_nan(ctx, name)
        return float(v)
    _handler.__name__ = f"_h_{name}"
    return _handler


def _make_ffd_rv_lag_handler(n: int) -> Callable[[AssemblerContext], float]:
    name = f"ffd_rv_lag{n}"
    def _handler(ctx: AssemblerContext) -> float:
        feats = ctx.dollar_bar_features
        if feats is None:
            return _baseline_or_nan(ctx, name)
        try:
            v = feats.ffd_rv_lag(n)
        except (KeyError, ValueError, TypeError):
            return _baseline_or_nan(ctx, name)
        if v is None or not np.isfinite(v):
            return _baseline_or_nan(ctx, name)
        return float(v)
    _handler.__name__ = f"_h_{name}"
    return _handler


def _make_ffd_rv_ema_handler(span: int) -> Callable[[AssemblerContext], float]:
    name = f"ffd_rv_ema{span}"
    def _handler(ctx: AssemblerContext) -> float:
        feats = ctx.dollar_bar_features
        if feats is None:
            return _baseline_or_nan(ctx, name)
        try:
            v = feats.ffd_rv_ema(span)
        except (KeyError, ValueError, TypeError):
            return _baseline_or_nan(ctx, name)
        if v is None or not np.isfinite(v):
            return _baseline_or_nan(ctx, name)
        return float(v)
    _handler.__name__ = f"_h_{name}"
    return _handler


def _make_rv_zscore_lag_handler(
    horizon: str, n: int,
) -> Callable[[AssemblerContext], float]:
    name = f"rv_zscore_{horizon}_lag{n}"
    def _handler(ctx: AssemblerContext) -> float:
        feats = ctx.dollar_bar_features
        if feats is None:
            return _baseline_or_nan(ctx, name)
        try:
            v = feats.rv_zscore_lag(horizon, n)
        except (KeyError, ValueError, TypeError):
            return _baseline_or_nan(ctx, name)
        if v is None or not np.isfinite(v):
            return _baseline_or_nan(ctx, name)
        return float(v)
    _handler.__name__ = f"_h_{name}"
    return _handler


def _make_rv_zscore_ema_handler(
    horizon: str, span: int,
) -> Callable[[AssemblerContext], float]:
    name = f"rv_zscore_{horizon}_ema{span}"
    def _handler(ctx: AssemblerContext) -> float:
        feats = ctx.dollar_bar_features
        if feats is None:
            return _baseline_or_nan(ctx, name)
        try:
            v = feats.rv_zscore_ema(horizon, span)
        except (KeyError, ValueError, TypeError):
            return _baseline_or_nan(ctx, name)
        if v is None or not np.isfinite(v):
            return _baseline_or_nan(ctx, name)
        return float(v)
    _handler.__name__ = f"_h_{name}"
    return _handler


# ── S-1c: down_jump_frac alias (column-name fix) ──
# The 101-feature TransVol input column is ``down_jump_frac`` (no
# "ion" suffix). Pre-cycle-6 the registry only had
# ``down_jump_fraction`` → ``_h_down_jump_fraction`` so the ``frac``
# column resolved to baseline-fallback. The alias closes that gap.
# Prefers TSRVFeatures.down_jump_frac() (last value in the buffer)
# over the current bar's BarTSRV — they are identical at decision time
# but the buffer-driven path ensures consistency with the bar-grid
# lag/EMA variants below.

def _h_down_jump_frac(ctx: AssemblerContext) -> float:
    """S-1d (parity-r3 cycle 7 — 2026-05-08): prefer the dollar-bar-
    grid value (offline byte-equivalent) over the BarTSRV direct path.
    The BarTSRV direct value is a 1-bar look-ahead vs offline (offline
    uses shift(1) before ffill onto bars)."""
    db = ctx.tsrv_db_features
    if db is not None:
        v = db.down_jump_frac()
        if v is not None and np.isfinite(v):
            return float(v)
    feats = ctx.tsrv_features
    if feats is not None:
        v = feats.down_jump_frac()
        if v is not None and np.isfinite(v):
            return float(v)
    # Buffer-less path (tests / cold start): use BarTSRV directly.
    if ctx.bar_tsrv is not None:
        return float(ctx.bar_tsrv.down_jump_fraction)
    return _baseline_or_nan(ctx, "down_jump_frac")


# ── S-1c: funding_basis live provider handlers ──
# These read from ``ctx.funding_provider`` (a ``FundingRateProvider``
# advance()'d once per bar by the engine BEFORE the handler loop).
# When the provider is None (legacy / test paths) the handlers fall
# back to the historical baseline value.

def _h_funding_basis(ctx: AssemblerContext) -> float:
    prov = ctx.funding_provider
    if prov is not None:
        v = prov.latest()
        if v is not None and np.isfinite(v):
            return float(v)
    return _baseline_or_nan(ctx, "funding_basis")


def _make_funding_basis_lag_handler(n: int) -> Callable[[AssemblerContext], float]:
    name = f"funding_basis_lag{n}"
    def _handler(ctx: AssemblerContext) -> float:
        prov = ctx.funding_provider
        if prov is not None:
            v = prov.latest_lag(n)
            if v is not None and np.isfinite(v):
                return float(v)
        return _baseline_or_nan(ctx, name)
    _handler.__name__ = f"_h_{name}"
    return _handler


def _make_funding_basis_ema_handler(span: int) -> Callable[[AssemblerContext], float]:
    name = f"funding_basis_ema{span}"
    def _handler(ctx: AssemblerContext) -> float:
        prov = ctx.funding_provider
        if prov is not None:
            v = prov.latest_ema(span)
            if v is not None and np.isfinite(v):
                return float(v)
        return _baseline_or_nan(ctx, name)
    _handler.__name__ = f"_h_{name}"
    return _handler


# Registry: feature name → handler.  Adding a new live-computable
# feature is one line here, not a tuple edit + imperative loop edit.
_LIVE_HANDLERS: dict[str, Callable[[AssemblerContext], float]] = {
    "tsrv": _h_tsrv,
    "rv_fast": _h_rv_fast,
    "rv_slow": _h_rv_slow,
    "bpv": _h_bpv,
    "medrv": _h_medrv,
    "continuous_var": _h_continuous_var,
    "jump_var": _h_jump_var,
    "jump_fraction": _h_jump_fraction,
    "rs_down": _h_rs_down,
    "rs_up": _h_rs_up,
    "down_jump_fraction": _h_down_jump_fraction,
    "up_jump_frac": _h_up_jump_frac,
    "vol_ann": _h_vol_ann,
    "n_trades": _h_n_trades,
    "duration_s": _h_duration_s,
    "n_bars_in_window": _h_n_bars_in_window,
    "tsrv_mean_1d": _h_tsrv_mean_1d,
    "svi_atm_1d": _h_svi_atm_1d,
    # C (parity 2026-05-07) — HAR-PD path-dependent live features.
    # Each handler is a closure that reads its slot from
    # ``ctx.har_pd_features`` (precomputed once per bar before the
    # handler loop fires; see ``FeatureAssembler.build_row``).
    **{name: _make_har_pd_handler(name) for name in HAR_PD_RAW_FEATURES},
    # F3 Tier 3 (parity-cycle round 3 — 2026-05-07): 16 dollar-bar
    # handlers. Each computes against ``ctx.dollar_bar_features``;
    # when the buffer is None / underfull the handler falls back to
    # the historical baseline via ``_baseline_or_nan``.
    "vov":                 _make_dollar_bar_handler("vov", "latest_raw", "vov"),
    "vov_ema10":           _make_dollar_bar_handler("vov_ema10", "ema", "vov", 10),
    "vov_ema50":           _make_dollar_bar_handler("vov_ema50", "ema", "vov", 50),
    "vov_lag3":            _make_dollar_bar_handler("vov_lag3", "lag", "vov", 3),
    "vpin":                _make_dollar_bar_handler("vpin", "latest_raw", "vpin"),
    "vpin_ema10":          _make_dollar_bar_handler("vpin_ema10", "ema", "vpin", 10),
    "vpin_ema50":          _make_dollar_bar_handler("vpin_ema50", "ema", "vpin", 50),
    "vpin_lag1":           _make_dollar_bar_handler("vpin_lag1", "lag", "vpin", 1),
    # imbalance: 5 variants (raw + 4 windows) — S-1b: ema50 already
    # present pre-cycle; raw + ema10 + lag1 + lag3 added cycle 5.
    "imbalance":           _make_dollar_bar_handler("imbalance", "latest_raw", "imbalance"),
    "imbalance_ema10":     _make_dollar_bar_handler("imbalance_ema10", "ema", "imbalance", 10),
    "imbalance_ema50":     _make_dollar_bar_handler("imbalance_ema50", "ema", "imbalance", 50),
    "imbalance_lag1":      _make_dollar_bar_handler("imbalance_lag1", "lag", "imbalance", 1),
    "imbalance_lag3":      _make_dollar_bar_handler("imbalance_lag3", "lag", "imbalance", 3),
    "duration_s_ema10":    _make_dollar_bar_handler("duration_s_ema10", "ema", "duration_s", 10),
    "duration_s_ema50":    _make_dollar_bar_handler("duration_s_ema50", "ema", "duration_s", 50),
    "duration_s_lag1":     _make_dollar_bar_handler("duration_s_lag1", "lag", "duration_s", 1),
    "duration_s_lag3":     _make_dollar_bar_handler("duration_s_lag3", "lag", "duration_s", 3),
    "n_trades_ema10":      _make_dollar_bar_handler("n_trades_ema10", "ema", "n_trades", 10),
    "n_trades_ema50":      _make_dollar_bar_handler("n_trades_ema50", "ema", "n_trades", 50),
    "n_trades_lag1":       _make_dollar_bar_handler("n_trades_lag1", "lag", "n_trades", 1),
    "n_trades_lag3":       _make_dollar_bar_handler("n_trades_lag3", "lag", "n_trades", 3),
    "vov_lag1":            _make_dollar_bar_handler("vov_lag1", "lag", "vov", 1),
    "vpin_lag3":           _make_dollar_bar_handler("vpin_lag3", "lag", "vpin", 3),
    # F-3 (parity-r3 cycle 2 — 2026-05-08): dynamic bpd-driven window.
    "parkinson_1d":        _h_parkinson_1d,
    "parkinson_1d_ema50":  _h_parkinson_1d_ema50,
    # F3 Tier 5 — TSRV-grid handlers, S-1d-rerouted to the dollar-
    # bar-grid path (offline byte-equivalent). The factory's first
    # arg is the DB-method (TSRVDollarBarFeatures); second is the
    # 15-min-cadence fallback method (TSRVFeatures). tsrv_mean_1d
    # already prefers the DB path above.
    "tsrv_mean_1d_lag3":    _make_tsrv_db_handler("tsrv_mean_1d_lag3", "tsrv_mean_1d_lag", "tsrv_mean_1d_lag", 3),
    "tsrv_mean_1d_ema50":   _make_tsrv_db_handler("tsrv_mean_1d_ema50", "tsrv_mean_1d_ema", "tsrv_mean_1d_ema", 50),
    "tsrv_mean_1w":         _make_tsrv_db_handler("tsrv_mean_1w", "tsrv_mean_1w", "tsrv_mean_1w"),
    "tsrv_ffd":             _make_tsrv_db_handler("tsrv_ffd", "tsrv_ffd", "tsrv_ffd"),
    "up_jump_frac_ema50":   _make_tsrv_db_handler("up_jump_frac_ema50", "up_jump_frac_ema", "up_jump_frac_ema", 50),
    "bpv_tsrv_ratio_ema50": _make_tsrv_db_handler("bpv_tsrv_ratio_ema50", "bpv_tsrv_ratio_ema", "bpv_tsrv_ratio_ema", 50),
    # F3 Tier 6 (parity-cycle round 3 — 2026-05-07): time-to-funding.
    # Pure datetime arithmetic — no buffer needed. Mirrors offline
    # ``vol_model/features/engineer.py:417-419``.
    "time_to_funding":       _h_time_to_funding,
    "time_to_funding_lag1":  _h_time_to_funding_lag1,
    "time_to_funding_lag3":  _h_time_to_funding_lag3,
    "time_to_funding_ema10": _h_time_to_funding_ema10,
    "time_to_funding_ema50": _h_time_to_funding_ema50,
    # S-1b (parity-r3 cycle 5 — 2026-05-08): TransVol-input bases that
    # were previously baseline-fallback. Lag/EMA variants are S-1c.
    "hurst_100":             _h_hurst_100,
    "ffd_rv":                _h_ffd_rv,
    "rv_zscore_1d":          _make_rv_zscore_handler("1d"),
    "rv_zscore_1w":          _make_rv_zscore_handler("1w"),
    "rv_zscore_1m":          _make_rv_zscore_handler("1m"),
    # ── S-1c (parity-r3 cycle 6 — 2026-05-08): full 101-feature live ──
    # 38 variant handlers + 3 new bases (down_jump_frac, noise_level
    # need handlers; funding_basis is wired in Phase 2). Closes the
    # baseline-fallback gap for every TransVol input column.
    # parkinson_1d variants (raw + ema50 already wired earlier in dict).
    "parkinson_1d_lag1":     _make_parkinson_1d_lag_handler(1),
    "parkinson_1d_lag3":     _make_parkinson_1d_lag_handler(3),
    "parkinson_1d_ema10":    _make_parkinson_1d_ema_handler(10),
    # hurst_100 variants.
    "hurst_100_lag1":        _make_hurst_lag_handler(1),
    "hurst_100_lag3":        _make_hurst_lag_handler(3),
    "hurst_100_ema10":       _make_hurst_ema_handler(10),
    "hurst_100_ema50":       _make_hurst_ema_handler(50),
    # ffd_rv variants.
    "ffd_rv_lag1":           _make_ffd_rv_lag_handler(1),
    "ffd_rv_lag3":           _make_ffd_rv_lag_handler(3),
    "ffd_rv_ema10":          _make_ffd_rv_ema_handler(10),
    "ffd_rv_ema50":          _make_ffd_rv_ema_handler(50),
    # rv_zscore variants (3 horizons × 4 windows = 12 entries).
    "rv_zscore_1d_lag1":     _make_rv_zscore_lag_handler("1d", 1),
    "rv_zscore_1d_lag3":     _make_rv_zscore_lag_handler("1d", 3),
    "rv_zscore_1d_ema10":    _make_rv_zscore_ema_handler("1d", 10),
    "rv_zscore_1d_ema50":    _make_rv_zscore_ema_handler("1d", 50),
    "rv_zscore_1w_lag1":     _make_rv_zscore_lag_handler("1w", 1),
    "rv_zscore_1w_lag3":     _make_rv_zscore_lag_handler("1w", 3),
    "rv_zscore_1w_ema10":    _make_rv_zscore_ema_handler("1w", 10),
    "rv_zscore_1w_ema50":    _make_rv_zscore_ema_handler("1w", 50),
    "rv_zscore_1m_lag1":     _make_rv_zscore_lag_handler("1m", 1),
    "rv_zscore_1m_lag3":     _make_rv_zscore_lag_handler("1m", 3),
    "rv_zscore_1m_ema10":    _make_rv_zscore_ema_handler("1m", 10),
    "rv_zscore_1m_ema50":    _make_rv_zscore_ema_handler("1m", 50),
    # ── S-1d (parity-r3 cycle 7 — 2026-05-08): all rerouted ──
    # tsrv_mean_1d/1w/tsrv_ffd lag/EMA variants, plus raw + 4 variants
    # for up_jump_frac, down_jump_frac, noise_level, bpv_tsrv_ratio.
    # First positional arg of _make_tsrv_db_handler is the DB-method
    # name on TSRVDollarBarFeatures; second is the 15-min-cadence
    # fallback method on TSRVFeatures (preserves cycle-5/6 behaviour
    # when no DollarBarBuffer is wired — tests, cold start).
    "tsrv_mean_1d_lag1":     _make_tsrv_db_handler("tsrv_mean_1d_lag1", "tsrv_mean_1d_lag", "tsrv_mean_1d_lag", 1),
    "tsrv_mean_1d_ema10":    _make_tsrv_db_handler("tsrv_mean_1d_ema10", "tsrv_mean_1d_ema", "tsrv_mean_1d_ema", 10),
    "tsrv_mean_1w_lag1":     _make_tsrv_db_handler("tsrv_mean_1w_lag1", "tsrv_mean_1w_lag", "tsrv_mean_1w_lag", 1),
    "tsrv_mean_1w_lag3":     _make_tsrv_db_handler("tsrv_mean_1w_lag3", "tsrv_mean_1w_lag", "tsrv_mean_1w_lag", 3),
    "tsrv_mean_1w_ema10":    _make_tsrv_db_handler("tsrv_mean_1w_ema10", "tsrv_mean_1w_ema", "tsrv_mean_1w_ema", 10),
    "tsrv_mean_1w_ema50":    _make_tsrv_db_handler("tsrv_mean_1w_ema50", "tsrv_mean_1w_ema", "tsrv_mean_1w_ema", 50),
    "tsrv_ffd_lag1":         _make_tsrv_db_handler("tsrv_ffd_lag1", "tsrv_ffd_lag", "tsrv_ffd_lag", 1),
    "tsrv_ffd_lag3":         _make_tsrv_db_handler("tsrv_ffd_lag3", "tsrv_ffd_lag", "tsrv_ffd_lag", 3),
    "tsrv_ffd_ema10":        _make_tsrv_db_handler("tsrv_ffd_ema10", "tsrv_ffd_ema", "tsrv_ffd_ema", 10),
    "tsrv_ffd_ema50":        _make_tsrv_db_handler("tsrv_ffd_ema50", "tsrv_ffd_ema", "tsrv_ffd_ema", 50),
    "up_jump_frac_lag1":     _make_tsrv_db_handler("up_jump_frac_lag1", "up_jump_frac_lag", "up_jump_frac_lag", 1),
    "up_jump_frac_lag3":     _make_tsrv_db_handler("up_jump_frac_lag3", "up_jump_frac_lag", "up_jump_frac_lag", 3),
    "up_jump_frac_ema10":    _make_tsrv_db_handler("up_jump_frac_ema10", "up_jump_frac_ema", "up_jump_frac_ema", 10),
    "bpv_tsrv_ratio":        _make_tsrv_db_handler("bpv_tsrv_ratio", "bpv_tsrv_ratio", "bpv_tsrv_ratio"),
    "bpv_tsrv_ratio_lag1":   _make_tsrv_db_handler("bpv_tsrv_ratio_lag1", "bpv_tsrv_ratio_lag", "bpv_tsrv_ratio_lag", 1),
    "bpv_tsrv_ratio_lag3":   _make_tsrv_db_handler("bpv_tsrv_ratio_lag3", "bpv_tsrv_ratio_lag", "bpv_tsrv_ratio_lag", 3),
    "bpv_tsrv_ratio_ema10":  _make_tsrv_db_handler("bpv_tsrv_ratio_ema10", "bpv_tsrv_ratio_ema", "bpv_tsrv_ratio_ema", 10),
    "down_jump_frac":        _h_down_jump_frac,
    "down_jump_frac_lag1":   _make_tsrv_db_handler("down_jump_frac_lag1", "down_jump_frac_lag", "down_jump_frac_lag", 1),
    "down_jump_frac_lag3":   _make_tsrv_db_handler("down_jump_frac_lag3", "down_jump_frac_lag", "down_jump_frac_lag", 3),
    "down_jump_frac_ema10":  _make_tsrv_db_handler("down_jump_frac_ema10", "down_jump_frac_ema", "down_jump_frac_ema", 10),
    "down_jump_frac_ema50":  _make_tsrv_db_handler("down_jump_frac_ema50", "down_jump_frac_ema", "down_jump_frac_ema", 50),
    "noise_level":           _make_tsrv_db_handler("noise_level", "noise_level", "noise_level"),
    "noise_level_lag1":      _make_tsrv_db_handler("noise_level_lag1", "noise_level_lag", "noise_level_lag", 1),
    "noise_level_lag3":      _make_tsrv_db_handler("noise_level_lag3", "noise_level_lag", "noise_level_lag", 3),
    "noise_level_ema10":     _make_tsrv_db_handler("noise_level_ema10", "noise_level_ema", "noise_level_ema", 10),
    "noise_level_ema50":     _make_tsrv_db_handler("noise_level_ema50", "noise_level_ema", "noise_level_ema", 50),
    # funding_basis: 5 columns. Live provider wires Binance perp REST
    # (Path A) + offline parquet fallback (Path B); see
    # ``deribit_paper/market_data/funding.py``.
    "funding_basis":         _h_funding_basis,
    "funding_basis_lag1":    _make_funding_basis_lag_handler(1),
    "funding_basis_lag3":    _make_funding_basis_lag_handler(3),
    "funding_basis_ema10":   _make_funding_basis_ema_handler(10),
    "funding_basis_ema50":   _make_funding_basis_ema_handler(50),
}


# TransVol embedding slots — filled from the encoder output, not from
# the dispatch table.
_TRANSVOL_SLOTS: frozenset[str] = frozenset(f"tv_{i}" for i in range(16))


# S-1c (parity-r3 cycle 6 — 2026-05-08): canonical 101-feature input
# column order for the TransVol encoder. Matches
# ``pd.read_parquet('vol_model/output/BTCUSDT_features_15m.parquet')``
# columns minus the 25 dropped OFI/depth/OI columns (see
# ``vol_model/transvol/dataset.py:142-144``). The encoder was trained
# on this exact ordering; the live row built by
# :meth:`FeatureAssembler.assemble_transvol_input_row` MUST honour it.
TRANSVOL_INPUT_COLUMNS: tuple[str, ...] = (
    "n_bars_in_window",
    "hurst_100", "hurst_100_lag1", "hurst_100_lag3",
    "hurst_100_ema10", "hurst_100_ema50",
    "ffd_rv", "ffd_rv_lag1", "ffd_rv_lag3",
    "ffd_rv_ema10", "ffd_rv_ema50",
    "rv_zscore_1d", "rv_zscore_1d_lag1", "rv_zscore_1d_lag3",
    "rv_zscore_1d_ema10", "rv_zscore_1d_ema50",
    "rv_zscore_1w", "rv_zscore_1w_lag1", "rv_zscore_1w_lag3",
    "rv_zscore_1w_ema10", "rv_zscore_1w_ema50",
    "rv_zscore_1m", "rv_zscore_1m_lag1", "rv_zscore_1m_lag3",
    "rv_zscore_1m_ema10", "rv_zscore_1m_ema50",
    "tsrv_ffd", "tsrv_ffd_lag1", "tsrv_ffd_lag3",
    "tsrv_ffd_ema10", "tsrv_ffd_ema50",
    "tsrv_mean_1d", "tsrv_mean_1d_lag1", "tsrv_mean_1d_lag3",
    "tsrv_mean_1d_ema10", "tsrv_mean_1d_ema50",
    "tsrv_mean_1w", "tsrv_mean_1w_lag1", "tsrv_mean_1w_lag3",
    "tsrv_mean_1w_ema10", "tsrv_mean_1w_ema50",
    "down_jump_frac", "down_jump_frac_lag1", "down_jump_frac_lag3",
    "down_jump_frac_ema10", "down_jump_frac_ema50",
    "up_jump_frac", "up_jump_frac_lag1", "up_jump_frac_lag3",
    "up_jump_frac_ema10", "up_jump_frac_ema50",
    "noise_level", "noise_level_lag1", "noise_level_lag3",
    "noise_level_ema10", "noise_level_ema50",
    "bpv_tsrv_ratio", "bpv_tsrv_ratio_lag1", "bpv_tsrv_ratio_lag3",
    "bpv_tsrv_ratio_ema10", "bpv_tsrv_ratio_ema50",
    "parkinson_1d", "parkinson_1d_lag1", "parkinson_1d_lag3",
    "parkinson_1d_ema10", "parkinson_1d_ema50",
    "duration_s", "duration_s_lag1", "duration_s_lag3",
    "duration_s_ema10", "duration_s_ema50",
    "n_trades", "n_trades_lag1", "n_trades_lag3",
    "n_trades_ema10", "n_trades_ema50",
    "imbalance", "imbalance_lag1", "imbalance_lag3",
    "imbalance_ema10", "imbalance_ema50",
    "vov", "vov_lag1", "vov_lag3", "vov_ema10", "vov_ema50",
    "vpin", "vpin_lag1", "vpin_lag3", "vpin_ema10", "vpin_ema50",
    "funding_basis", "funding_basis_lag1", "funding_basis_lag3",
    "funding_basis_ema10", "funding_basis_ema50",
    "time_to_funding", "time_to_funding_lag1", "time_to_funding_lag3",
    "time_to_funding_ema10", "time_to_funding_ema50",
)
TRANSVOL_TIME_COLUMNS: tuple[str, str] = ("duration_s", "n_bars_in_window")

assert len(TRANSVOL_INPUT_COLUMNS) == 101, (
    f"TRANSVOL_INPUT_COLUMNS must have 101 entries, got "
    f"{len(TRANSVOL_INPUT_COLUMNS)}"
)


# ────────────────────────────────────────────────────────────
# FeatureRow output + FeatureAssembler class
# ────────────────────────────────────────────────────────────


@dataclass
class FeatureRow:
    """One bar's feature row, ready to hand to LightGBM.

    ``values`` is the N-dim numpy array in the order LightGBM expects
    (N = 44 for the Stage 18 pruned+SVI variant, 51 for the older
    hybrid variant).  ``as_dict`` is a dict keyed by feature name for
    debugging/logging.
    """
    bar_timestamp: pd.Timestamp
    values: np.ndarray
    feature_names: list[str]
    as_dict: dict[str, float] = field(default_factory=dict)
    transvol_embedding: np.ndarray | None = None


class FeatureAssembler:
    """Manifest-driven feature assembler for the live paper engine.

    Usage (inside the engine)::

        assembler = FeatureAssembler(
            features_parquet="vol_model/output/BTCUSDT_features_15m.parquet",
            svi_features_parquet="vol_arb/output/svi_features_15m.parquet",
            bundle=model_bundle,
            svi_calibrator=live_svi_calibrator,
        )
        row = assembler.build_row(bar, bar_tsrv, transvol_embedding)
        rv_pred_ln = bundle.lgbm_model.predict(row.values.reshape(1, -1))[0]

    The constructor merges the two parquets to form a "baseline row"
    that covers every feature in the manifest.  It then asserts that
    every feature in ``bundle.lgbm_features`` is producible — either
    as a TransVol embedding slot, via a live handler, or from the
    merged baseline — and raises ``ValueError`` immediately if any
    feature cannot be produced.  The old behaviour of silently zeroing
    unknown features is gone.
    """

    def __init__(
        self,
        *,
        features_parquet: str | Path,
        svi_features_parquet: str | Path | None = None,
        bundle,  # ModelBundle — late binding to avoid circular import
        svi_calibrator: Any | None = None,
        dollar_bar_buffer: DollarBarBuffer | None = None,
        tsrv_history_buffer: TSRVHistoryBuffer | None = None,
        funding_provider: Any | None = None,
        rolling_history_bars: int = 672,  # 7 days at 15min
        har_pd_k_bars: int = HAR_PD_K_BARS_DEFAULT,
    ):
        self._features_parquet_path = Path(features_parquet)
        self._svi_features_parquet_path = (
            Path(svi_features_parquet) if svi_features_parquet else None
        )
        self._bundle = bundle
        self._svi_calibrator = svi_calibrator
        # F3 Tier 3 (parity-cycle round 3 — 2026-05-07): the live
        # DollarBarBuffer fed by ``DollarBarSpotStream``. ``None`` until
        # Tier 6 wires the engine; while None the 16 dollar-bar-derived
        # handlers fall back to the historical baseline.
        self._dollar_bar_buffer = dollar_bar_buffer
        # F3 Tier 5 (parity-cycle round 3 — 2026-05-07): live TSRV
        # history buffer (15-min cadence). ``None`` until Tier 6 wires
        # the engine to call ``buffer.append_bar`` per process_bar.
        # While None, the 6 TSRV-derived handlers fall back to baseline.
        self._tsrv_history_buffer = tsrv_history_buffer
        # S-1c (parity-r3 cycle 6 — 2026-05-08): live funding-rate
        # provider. ``None`` until the engine wires it; while None
        # the 5 ``funding_basis_*`` handlers fall back to baseline.
        self._funding_provider = funding_provider
        self._rolling_history_bars = rolling_history_bars
        # C (parity 2026-05-07): live HAR-PD state. Each call to
        # ``build_row`` reads strict-past values from this buffer
        # BEFORE appending the current bar. The engine pre-warms the
        # buffer from the historical TSRV parquet at startup so the
        # first live bar produces non-NaN HAR-PD features.
        self._har_pd_state = HARPDState(k_bars=har_pd_k_bars)

        if not self._features_parquet_path.exists():
            raise FileNotFoundError(
                f"Historical features parquet not found: "
                f"{self._features_parquet_path}",
            )

        logger.info(
            "FeatureAssembler loading baseline from %s",
            self._features_parquet_path,
        )
        base_df = pd.read_parquet(self._features_parquet_path)

        # Merge in SVI features if a second parquet was provided.  The
        # Stage 18 T2.1 models depend on ``svi_atm_1d`` which lives in
        # a separate parquet.  Right-wins on column collisions so SVI
        # values take precedence over any stale duplicates in the base
        # features parquet.
        if self._svi_features_parquet_path is not None:
            if not self._svi_features_parquet_path.exists():
                raise FileNotFoundError(
                    f"SVI features parquet not found: "
                    f"{self._svi_features_parquet_path}",
                )
            svi_df = pd.read_parquet(self._svi_features_parquet_path)
            logger.info(
                "  Merging SVI features from %s (%d cols, %d rows)",
                self._svi_features_parquet_path,
                len(svi_df.columns), len(svi_df),
            )
            # F1 (parity-cycle round 2 — 2026-05-07): the SVI-features
            # parquet ships with a ``RangeIndex`` and ``timestamp`` as a
            # column, while the base parquet uses a ``DatetimeIndex``.
            # The pre-fix ``base.join(svi, how="left")`` aligned by
            # INTEGER POSITION across mismatched indices, silently
            # producing NaN in 100% of merged rows (verified
            # empirically: 185,945/185,945 NaN). The ``_h_svi_atm_1d``
            # baseline-fallback then emitted ``svi_atm_1d=0.0`` to
            # LightGBM since the V2 stack was adopted, with the
            # ``or 0.0`` collapse (F9) defeating the line-519 NaN
            # guard. Set the SVI parquet's index to its ``timestamp``
            # column BEFORE the join so alignment is timestamp-based.
            if (
                "timestamp" in svi_df.columns
                and not isinstance(svi_df.index, pd.DatetimeIndex)
            ):
                svi_df = svi_df.set_index(
                    pd.DatetimeIndex(pd.to_datetime(svi_df["timestamp"], utc=True))
                ).drop(columns=["timestamp"])
                svi_df.index.name = base_df.index.name  # match base for clean join
            overlap = set(base_df.columns) & set(svi_df.columns)
            if overlap:
                # Drop from base so SVI wins on overlap
                base_df = base_df.drop(columns=list(overlap))
            full_df = base_df.join(svi_df, how="left")
            # Sanity check: at least one SVI column should land on a
            # non-NaN value in the recent baseline tail. Fail-fast if
            # the join still drops everything (e.g. timestamp parsing
            # mismatch on a future parquet variant).
            for svi_col in ("svi_atm_1d", "svi_atm_1w"):
                if svi_col in full_df.columns:
                    nan_rate_tail = (
                        full_df[svi_col].tail(672).isna().mean()
                    )
                    if nan_rate_tail > 0.5:
                        raise RuntimeError(
                            f"SVI features parquet join produced "
                            f"{nan_rate_tail:.1%} NaN in {svi_col} over "
                            f"the last 7 days — index alignment likely "
                            f"broken. Check that "
                            f"{self._svi_features_parquet_path} has a "
                            f"DatetimeIndex or a 'timestamp' column."
                        )
                    break
        else:
            full_df = base_df

        # F10 (parity-cycle round 2 — 2026-05-07): pre-F10 the assembler
        # kept ``self._historical_buffer = full_df.tail(672)`` as a
        # persistent field, but no handler ever read it after init —
        # only the LAST row was used to seed ``_baseline_row``. The
        # 672-bar buffer was dead memory that misleadingly suggested
        # live rolling-lookback computation existed when in fact every
        # ``*_lag`` / ``*_ema`` family member is baseline-frozen (see
        # F3 in the audit). Re-introduce a buffer with a real read path
        # only when F3 is picked up. For now, derive the baseline row
        # inline and let ``full_df`` go out of scope.
        baseline_window = full_df.tail(rolling_history_bars)
        logger.info(
            "  Loaded %d merged rows; baseline derived from last %d",
            len(full_df), len(baseline_window),
        )

        # Baseline = most recent NON-NaN-rich row we have.  Use the
        # final row of the recent slice as a pragmatic default —
        # downstream fail-fast will catch any feature that ends up NaN.
        self._baseline_row = baseline_window.iloc[-1].copy()
        self._baseline_ts = baseline_window.index[-1]
        logger.info(
            "  Baseline row timestamp: %s (%d columns)",
            self._baseline_ts, len(self._baseline_row),
        )

        # Fail-fast validation: every manifest feature must be
        # producible from at least one of the three sources.
        self._validate_manifest(list(bundle.lgbm_features))

    @property
    def baseline_timestamp(self) -> pd.Timestamp:
        return self._baseline_ts

    @property
    def har_pd_state(self) -> HARPDState:
        """Expose the HAR-PD circular buffer for testing / pre-warm."""
        return self._har_pd_state

    def prewarm_har_pd_from_tsrv(
        self,
        tsrv_parquet_path: str | Path,
    ) -> int:
        """Bulk-feed the last ``k_bars`` rows of an offline TSRV
        parquet into the HAR-PD buffer so the first live bar produces
        non-NaN HAR-PD features instead of waiting 24h.

        Returns the number of rows fed. Does NOT raise if the parquet
        is missing — engine's startup logger will warn and HAR-PD
        will warmup from live data over the first 96 bars.
        """
        p = Path(tsrv_parquet_path)
        if not p.exists():
            logger.warning(
                "HAR-PD pre-warm: TSRV parquet %s does not exist; "
                "buffer will warmup from live bars (96 bar / ~24h delay)",
                p,
            )
            return 0
        df = pd.read_parquet(p)
        required = {"rs_up", "rs_down", "rv_ann"}
        missing = required - set(df.columns)
        if missing:
            logger.warning(
                "HAR-PD pre-warm: TSRV parquet %s missing columns %s; "
                "buffer will warmup from live bars",
                p, missing,
            )
            return 0
        k = self._har_pd_state.k_bars
        tail = df[["rs_up", "rs_down", "rv_ann"]].tail(k)
        rows = (
            (float(r.rs_up), float(r.rs_down), float(r.rv_ann))
            for r in tail.itertuples()
        )
        n = self._har_pd_state.prewarm(rows)
        logger.info(
            "HAR-PD pre-warm: seeded buffer with %d rows from %s "
            "(window: %s → %s)",
            n, p.name, tail.index[0], tail.index[-1],
        )
        return n

    def _validate_manifest(self, feature_names: list[str]) -> None:
        """Raise ValueError listing any features we cannot produce."""
        missing: list[str] = []
        baseline_cols = set(self._baseline_row.index)
        for name in feature_names:
            if name in _TRANSVOL_SLOTS:
                continue
            if name in _LIVE_HANDLERS:
                continue
            if name in baseline_cols:
                continue
            missing.append(name)
        if missing:
            raise ValueError(
                f"FeatureAssembler cannot produce {len(missing)} manifest "
                f"feature(s): {missing}. Each feature must be one of: "
                "a TransVol slot (tv_0..tv_15), a registered live handler "
                "(_LIVE_HANDLERS), or present in the merged historical "
                "baseline parquet. Silently zeroing missing features "
                "(previous behaviour) was a foot-gun we removed in W3.1.",
            )
        logger.info(
            "  Manifest validated: %d features, all producible",
            len(feature_names),
        )

    def build_row(
        self,
        bar: Bar,
        bar_tsrv: BarTSRV,
        transvol_embedding: np.ndarray | None = None,
    ) -> FeatureRow | None:
        """Build the feature row for ``bar`` ready for LightGBM.

        The LightGBM model's feature order comes from the bundle's
        manifest — the assembler walks that list, dispatching each
        name to its source (TransVol slot / live handler / baseline).
        Unknown features cannot appear here because the constructor
        validates the manifest against all three sources.
        """
        feature_names = list(self._bundle.lgbm_features)
        n = len(feature_names)
        values = np.zeros(n, dtype=np.float64)
        as_dict: dict[str, float] = {}

        # C (parity 2026-05-07): compute HAR-PD features ONCE per bar
        # from the strict-past buffer BEFORE the handler loop. The
        # handlers just look up their slot. Must happen BEFORE we
        # append the current bar's TSRV — strict-past contract.
        har_pd_features = self._har_pd_state.compute_features()

        # F3 Tier 3 (parity-cycle round 3 — 2026-05-07): build a
        # ``DollarBarFeatures`` cached view over the live buffer. Each
        # of the 16 dollar-bar handlers reads from this view; the cache
        # ensures ewm/lag/parkinson series are computed once per bar
        # rather than per handler. ``None`` when no buffer is wired
        # (engine warmup / test fixtures); handlers detect that and
        # fall back to baseline.
        dollar_bar_features = (
            DollarBarFeatures(self._dollar_bar_buffer)
            if self._dollar_bar_buffer is not None else None
        )

        # F3 Tier 5: same pattern for the TSRV history buffer.
        tsrv_features = (
            TSRVFeatures(self._tsrv_history_buffer)
            if self._tsrv_history_buffer is not None else None
        )

        # S-1d (parity-r3 cycle 7 — 2026-05-08): TSRV-on-dollar-bar-
        # grid wrapper, byte-equivalent to offline. Constructed only
        # when both buffers are wired; falls through to ``tsrv_features``
        # otherwise.
        tsrv_db_features = (
            TSRVDollarBarFeatures(self._tsrv_history_buffer, self._dollar_bar_buffer)
            if (self._tsrv_history_buffer is not None
                and self._dollar_bar_buffer is not None)
            else None
        )

        ctx = AssemblerContext(
            bar=bar,
            bar_tsrv=bar_tsrv,
            svi_calibrator=self._svi_calibrator,
            baseline_row=self._baseline_row,
            har_pd_features=har_pd_features,
            dollar_bar_buffer=self._dollar_bar_buffer,
            dollar_bar_features=dollar_bar_features,
            tsrv_history_buffer=self._tsrv_history_buffer,
            tsrv_features=tsrv_features,
            tsrv_db_features=tsrv_db_features,
            funding_provider=self._funding_provider,
        )

        for i, name in enumerate(feature_names):
            if name in _TRANSVOL_SLOTS:
                if transvol_embedding is None:
                    # Fall back to baseline (stale) — smoke-only path
                    values[i] = float(self._baseline_row.get(name, 0.0) or 0.0)
                else:
                    idx = int(name.split("_", 1)[1])
                    values[i] = float(transvol_embedding[idx])
            elif name in _LIVE_HANDLERS:
                handler = _LIVE_HANDLERS[name]
                try:
                    values[i] = float(handler(ctx))
                except Exception as exc:  # noqa: BLE001 - defensive
                    logger.warning(
                        "Live handler failed for %s (%s); using baseline fallback",
                        name, exc,
                    )
                    # F9 (2026-05-07): no ``or 0.0`` collapse — let NaN
                    # propagate to the line-519 guard so a malformed
                    # baseline (e.g. F1 SVI-join NaN) refuses to trade
                    # rather than emitting a silent zero.
                    raw = self._baseline_row.get(name, np.nan)
                    values[i] = float(raw) if pd.notna(raw) else float("nan")
            else:
                # Baseline fallback (validated in __init__, so this is safe).
                # F9 (2026-05-07): NaN propagates instead of collapsing to 0.
                raw = self._baseline_row.get(name, np.nan)
                values[i] = float(raw) if pd.notna(raw) else float("nan")
            as_dict[name] = values[i]

        # C (parity 2026-05-07): append the current bar's TSRV to the
        # HAR-PD buffer AFTER the handler loop has consumed the
        # strict-past features. This makes the current bar's TSRV
        # available as strict-past for the NEXT bar's HAR-PD
        # computation. Before the buffer is warm we still append (the
        # warmup-fallback path returns NaN until 96 entries observed).
        try:
            self._har_pd_state.append(
                rs_up=float(bar_tsrv.rs_up),
                rs_down=float(bar_tsrv.rs_down),
                rv_ann=float(bar_tsrv.rv_ann),
            )
        except (AttributeError, TypeError, ValueError) as exc:
            # Defensive: a malformed BarTSRV (or a test stub missing a
            # field) shouldn't poison the buffer. Skip the append and
            # continue — handlers will fall back to baseline next bar.
            logger.warning(
                "HAR-PD append failed at %s (%s); buffer not advanced",
                bar.timestamp, exc,
            )

        # R1: Fail fast on NaN/inf features instead of silently zeroing.
        # TransVol slots (tv_*) are allowed to be zero-filled from baseline
        # when no live embedding is available. All other NaN = broken pipeline.
        bad_mask = ~np.isfinite(values)
        if bad_mask.any():
            bad_names = [
                feature_names[i] for i in range(n) if bad_mask[i]
            ]
            # TransVol slots are expected to fall back to baseline (may be 0)
            non_tv_bad = [nm for nm in bad_names if nm not in _TRANSVOL_SLOTS]
            if non_tv_bad:
                logger.error(
                    "Feature NaN/inf detected at %s: %s — refusing to trade",
                    bar.timestamp, non_tv_bad,
                )
                return None
            # Zero-fill only TransVol NaN/inf (expected when no live embedding)
            values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)

        return FeatureRow(
            bar_timestamp=bar.timestamp,
            values=values,
            feature_names=feature_names,
            as_dict=as_dict,
            transvol_embedding=transvol_embedding,
        )

    # ── S-1c (parity-r3 cycle 6 — 2026-05-08): TransVol input row ──

    def assemble_transvol_input_row(
        self,
        bar: Bar,
        bar_tsrv: BarTSRV,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Build the encoder's UNNORMALISED 101-feature + 2-time row.

        Walks ``TRANSVOL_INPUT_COLUMNS`` (the encoder's training-time
        column ordering) using the same ``_LIVE_HANDLERS`` dispatch as
        :meth:`build_row`, but produces the encoder-input layout rather
        than the LightGBM manifest layout.

        Returns
        -------
        (features_101, time_features_2)
            ``features_101`` is shape ``(101,) float32`` in
            ``TRANSVOL_INPUT_COLUMNS`` order.
            ``time_features_2`` is shape ``(2,) float32`` for
            ``(duration_s, n_bars_in_window)`` (matches
            ``vol_model/transvol/dataset.py:TIME_COLS``).

        Both arrays are UNNORMALISED — the caller (engine) applies
        the bundle's ``transvol_feat_mean/std`` before appending to
        the ``TransVolHistoryBuffer``. Decoupling normalisation from
        feature extraction keeps the buffer reusable across encoder
        checkpoints with different stats.

        NaN handling: any handler returning NaN propagates here. The
        caller (or :class:`TransVolHistoryBuffer.prewarm_from_parquet`)
        decides the policy (typically ``np.nan_to_num(0)`` for the
        warmup window, then strict NaN-fail in steady state).
        """
        # Build a context that mirrors build_row's setup so every
        # handler sees the same dependency graph (HAR-PD, dollar-bar
        # cached features, TSRV cached features, funding provider).
        har_pd_features = self._har_pd_state.compute_features()
        dollar_bar_features = (
            DollarBarFeatures(self._dollar_bar_buffer)
            if self._dollar_bar_buffer is not None else None
        )
        tsrv_features = (
            TSRVFeatures(self._tsrv_history_buffer)
            if self._tsrv_history_buffer is not None else None
        )
        # S-1d (parity-r3 cycle 7): TSRV-on-dollar-bar-grid wrapper.
        tsrv_db_features = (
            TSRVDollarBarFeatures(self._tsrv_history_buffer, self._dollar_bar_buffer)
            if (self._tsrv_history_buffer is not None
                and self._dollar_bar_buffer is not None)
            else None
        )
        ctx = AssemblerContext(
            bar=bar,
            bar_tsrv=bar_tsrv,
            svi_calibrator=self._svi_calibrator,
            baseline_row=self._baseline_row,
            har_pd_features=har_pd_features,
            dollar_bar_buffer=self._dollar_bar_buffer,
            dollar_bar_features=dollar_bar_features,
            tsrv_history_buffer=self._tsrv_history_buffer,
            tsrv_features=tsrv_features,
            tsrv_db_features=tsrv_db_features,
            funding_provider=self._funding_provider,
        )

        n = len(TRANSVOL_INPUT_COLUMNS)
        features = np.zeros(n, dtype=np.float32)
        for i, name in enumerate(TRANSVOL_INPUT_COLUMNS):
            handler = _LIVE_HANDLERS.get(name)
            if handler is not None:
                try:
                    features[i] = float(handler(ctx))
                except Exception as exc:  # noqa: BLE001 - defensive
                    logger.warning(
                        "TransVol handler failed for %s (%s); using baseline",
                        name, exc,
                    )
                    raw = self._baseline_row.get(name, np.nan)
                    features[i] = float(raw) if pd.notna(raw) else float("nan")
            else:
                # Pure baseline fallback — should be empty after S-1c
                # Phase 2 closes 101/101 coverage, but kept defensive.
                raw = self._baseline_row.get(name, np.nan)
                features[i] = float(raw) if pd.notna(raw) else float("nan")

        time_features = np.array([
            features[TRANSVOL_INPUT_COLUMNS.index(TRANSVOL_TIME_COLUMNS[0])],
            features[TRANSVOL_INPUT_COLUMNS.index(TRANSVOL_TIME_COLUMNS[1])],
        ], dtype=np.float32)

        return features, time_features
