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

from deribit_paper.features.har_pd_live import (
    HAR_PD_RAW_FEATURES,
    K_BARS_DEFAULT as HAR_PD_K_BARS_DEFAULT,
    HARPDState,
)
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
    return float(ctx.bar_tsrv.up_jump_fraction)


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
    return float(ctx.bar.n_perp_trades)


def _h_duration_s(ctx: AssemblerContext) -> float:
    if not ctx.bar.perp_trades:
        return float(ctx.baseline_row.get("duration_s", 1.0) or 1.0)
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
    # F9 (2026-05-07): NaN propagates to the line-519 NaN guard rather
    # than collapsing to 0.0. A bad baseline parquet → engine refuses
    # to trade rather than feeding LightGBM a zero-mean TSRV.
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
    """
    if ctx.svi_calibrator is None:
        return _baseline_or_nan(ctx, "svi_atm_1d")
    live = ctx.svi_calibrator.atm_vol_at_dte(1.0)
    if live is None or not np.isfinite(live):
        return _baseline_or_nan(ctx, "svi_atm_1d")
    return float(live)


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
}


# TransVol embedding slots — filled from the encoder output, not from
# the dispatch table.
_TRANSVOL_SLOTS: frozenset[str] = frozenset(f"tv_{i}" for i in range(16))


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
        rolling_history_bars: int = 672,  # 7 days at 15min
        har_pd_k_bars: int = HAR_PD_K_BARS_DEFAULT,
    ):
        self._features_parquet_path = Path(features_parquet)
        self._svi_features_parquet_path = (
            Path(svi_features_parquet) if svi_features_parquet else None
        )
        self._bundle = bundle
        self._svi_calibrator = svi_calibrator
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

        ctx = AssemblerContext(
            bar=bar,
            bar_tsrv=bar_tsrv,
            svi_calibrator=self._svi_calibrator,
            baseline_row=self._baseline_row,
            har_pd_features=har_pd_features,
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
