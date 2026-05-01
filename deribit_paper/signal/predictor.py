"""Per-bar signal generation — LightGBM + IV lookup + edge logic.

The predictor is the single component that turns a :class:`Bar` into
a trading :class:`Signal`.  It orchestrates:

1. :class:`FeatureAssembler.build_row` — 51-dim LightGBM input
2. ``lgbm_model.predict`` → ``rv_pred`` in ln(vol_ann) space
3. Edge calculation: ``edge = exp(rv_pred) - iv_used_decimal`` in vol points
4. Signal logic matching the backtester (asymmetric thresholds):
   - ``long`` if ``edge > +entry_threshold_long`` (default 30 vp)
   - ``short`` if ``edge < -entry_threshold`` (default 25 vp)
   - ``flat`` otherwise

The ``iv_used`` is obtained from the live SVI surface at the ATM
strike (via :class:`LiveSVICalibrator.evaluate`).  This is the same
``svi_iv_matched`` quantity the backtester uses for signal generation.

A :class:`Signal` carries enough context (strike, expiry, index_price,
vol_ann, jump_fraction, straddle selection) to hand directly to the
execution layer for position sizing + order placement.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd

from deribit_paper.features.straddle import StraddleSelection
from deribit_paper.features.svi_live import LiveSVICalibrator
from deribit_paper.features.tsrv_live import BarTSRV, compute_bar_tsrv
from deribit_paper.market_data.aggregator import Bar
from deribit_paper.signal.feature_assembler import FeatureAssembler

logger = logging.getLogger(__name__)


@dataclass
class Signal:
    """Per-bar trading signal with full context for execution."""
    timestamp: pd.Timestamp
    direction: int                    # +1 long, -1 short, 0 flat
    edge_vol_pts: float               # rv_pred - iv_used, in vol points (%)
    rv_pred_pct: float                # model's forecast (annualised vol, %)
    iv_used_pct: float                # SVI IV at the ATM strike (%)
    bar_tsrv: BarTSRV
    # E.1.8 (2026-05-01): SVI 1w-2w term slope at signal time, used by
    # the V2 contango filter and emitted into the harness row. Cached
    # here so the harness writer doesn't have to re-call a private
    # predictor method 15 minutes later (when the slope has changed).
    # ``None`` when SVI is uncalibrated.
    term_slope_1w_2w: float | None = None
    # Execution context (populated only when direction != 0 — else None)
    straddle: StraddleSelection | None = None
    reason: str = ""                   # human-readable explanation

    @property
    def is_actionable(self) -> bool:
        return self.direction != 0 and self.straddle is not None


class Predictor:
    """Converts a :class:`Bar` into a :class:`Signal`.

    Constructed once at engine startup with the loaded model bundle,
    live SVI calibrator, and configuration thresholds.  Called on
    every bar close with the new bar and the fresh TSRV stats.

    The predictor DOES NOT compute TSRV internally — the caller is
    expected to pass the already-computed :class:`BarTSRV` so that the
    engine can run TSRV computation once per bar and share it with
    both the feature assembler and any metrics logging.
    """

    def __init__(
        self,
        *,
        feature_assembler: FeatureAssembler,
        svi_calibrator: LiveSVICalibrator,
        entry_threshold_vol_pts: float = 25.0,
        entry_threshold_long_vol_pts: float = 30.0,
        base_spread_iv: float = 1.5,
        spread_jump_k: float = 3.0,
        max_jump_fraction_for_shorts: float = 0.15,
        skip_long_if_term_slope_gt: float | None = None,
        aligner_betas: dict[str, float] | None = None,
        require_aligner: bool = False,
        log_every_bar: bool = False,
    ):
        self._assembler = feature_assembler
        self._svi = svi_calibrator
        self._bundle = feature_assembler._bundle
        self._entry_threshold = entry_threshold_vol_pts
        self._entry_threshold_long = entry_threshold_long_vol_pts
        self._base_spread_iv = base_spread_iv
        self._spread_jump_k = spread_jump_k
        self._max_jump_fraction_for_shorts = max_jump_fraction_for_shorts
        self._skip_long_if_term_slope_gt = skip_long_if_term_slope_gt
        self._aligner_betas: dict[str, float] = dict(aligner_betas or {})
        self._log_every_bar = log_every_bar
        # E.1.14 (2026-05-01): under V2 preset, an empty aligner_betas
        # dict signals a configuration failure (sidecar JSON missing or
        # malformed), NOT a deliberate "β* disabled" choice. Fail fast
        # at startup rather than silently bypass β* every bar.
        if require_aligner and not self._aligner_betas:
            raise RuntimeError(
                "Predictor: ``require_aligner=True`` (V2 preset) but no "
                "β* sidecar loaded. Confirm aligner_betas_json exists, "
                "is valid JSON, and contains numeric per-month betas."
            )

    def predict(
        self,
        bar: Bar,
        bar_tsrv: BarTSRV,
        *,
        straddle: StraddleSelection | None = None,
        transvol_embedding: np.ndarray | None = None,
    ) -> Signal:
        """Generate the signal for one bar.

        Parameters
        ----------
        bar : Bar
            The just-closed bar from :class:`BarAggregator`.
        bar_tsrv : BarTSRV
            Fresh TSRV stats for this bar (from :func:`compute_bar_tsrv`).
        straddle : StraddleSelection, optional
            The ATM straddle pair picked for this bar.  If None, the
            signal can still be computed but ``is_actionable`` will be
            False because there's no instrument to trade.
        transvol_embedding : np.ndarray, optional
            The 16-dim TransVol embedding for this bar.  If None, the
            assembler will fall back to baseline (stale) values.
        """
        # 1) Build the 51-dim feature row for LightGBM
        row = self._assembler.build_row(bar, bar_tsrv, transvol_embedding)

        # R1: If the assembler detected NaN features, it returns None → flat signal
        if row is None:
            return self._flat_signal(
                bar, bar_tsrv, 0.0,
                reason="feature pipeline returned NaN — refusing to trade",
            )

        # 2) LightGBM prediction — outputs ln(vol_ann), convert to vol points.
        # Apply per-month aligner β* (walk-forward QLIKE-optimal scalar
        # bias) to mirror the offline contract baked into
        # BTCUSDT_pruned_svi_har_pd_predictions_aligned_bias.parquet
        # (y_pred = y_pred_raw + β*_month). Lookup falls back to the
        # most-recent-prior month when ``aligner_betas`` lags real time.
        pred_ln_raw = float(self._bundle.lgbm_model.predict(row.values.reshape(1, -1))[0])
        if self._aligner_betas:
            from vol_arb.sidecars import lookup_beta_for_month
            beta_star = lookup_beta_for_month(
                self._aligner_betas, bar.timestamp, fallback_zero_warn=False,
            )
            # E.1.11: defence-in-depth — even though load_aligner_betas
            # already clips to [-0.5, +0.5], a non-finite β* here would
            # produce inf or NaN through exp(), poisoning every signal
            # downstream. Bypass on non-finite + log ERROR so the
            # operator notices.
            if not math.isfinite(beta_star):
                logger.error(
                    "Predictor: non-finite β* at %s — bypassing β* (raw "
                    "prediction used). Investigate sidecar corruption.",
                    bar.timestamp,
                )
                pred_ln = pred_ln_raw
            else:
                pred_ln = pred_ln_raw + beta_star
        else:
            pred_ln = pred_ln_raw
        if not math.isfinite(pred_ln):
            logger.error(
                "Predictor: non-finite pred_ln at %s (raw=%s β=%s) — "
                "returning flat signal",
                bar.timestamp, pred_ln_raw,
                beta_star if self._aligner_betas else "n/a",
            )
            return self._flat_signal(
                bar, bar_tsrv, 0.0,
                reason="non-finite prediction (β* corruption?)",
            )
        rv_pred_decimal = math.exp(pred_ln)
        rv_pred_pct = rv_pred_decimal * 100.0

        # 3) SVI IV at the ATM strike — this is the 'iv_used' the
        #    backtester feeds into the signal computation.  If the
        #    straddle picker returned None or SVI isn't calibrated yet,
        #    fall back to the bar's ``vol_ann_continuous`` (same unit).
        iv_used_pct = self._lookup_iv(bar, straddle)
        if iv_used_pct is None:
            return self._flat_signal(
                bar, bar_tsrv, rv_pred_pct,
                reason="iv_used unavailable (no SVI slice or no straddle)",
            )

        # 4) Edge + signal direction — mirrors backtester's generate_signals() exactly:
        #    dynamic half-spread widens the effective threshold during stress
        #    (high jump_fraction), and asymmetric long/short thresholds account
        #    for higher entry slippage on long straddles.
        jf = float(bar_tsrv.jump_fraction) if bar_tsrv.jump_fraction is not None else 0.0
        jf = max(0.0, min(1.0, jf))
        half_spread = self._base_spread_iv * math.exp(self._spread_jump_k * jf) / 2.0

        # E.1.8: compute the 1w-2w term slope ONCE at predict time and
        # cache on the Signal so downstream code (harness, audits) sees
        # the slope at signal-fire time, NOT at trade-close time.
        term_slope = self._compute_term_slope_1w_2w()

        edge = rv_pred_pct - iv_used_pct
        direction = 0
        effective_long = half_spread + self._entry_threshold_long
        effective_short = half_spread + self._entry_threshold

        if edge > effective_long:
            direction = 1
            reason = (
                f"LONG — rv_pred ({rv_pred_pct:.2f}%) > iv ({iv_used_pct:.2f}%) "
                f"by {edge:+.3f}vp (eff_thresh={effective_long:.2f})"
            )
            # V2 contango filter (Phase C.5 2026-04-16): drop LONG signals
            # in IV contango regimes where the 1w-2w term slope is positive.
            if self._skip_long_if_term_slope_gt is not None:
                if term_slope is not None and term_slope > self._skip_long_if_term_slope_gt:
                    direction = 0
                    reason = (
                        f"FLAT (V2 contango filter) — term_slope_1w_2w "
                        f"{term_slope:+.3f} > {self._skip_long_if_term_slope_gt:.2f}; "
                        f"LONG candidate edge={edge:+.3f}vp suppressed"
                    )
        elif edge < -effective_short:
            direction = -1
            reason = (
                f"SHORT — rv_pred ({rv_pred_pct:.2f}%) < iv ({iv_used_pct:.2f}%) "
                f"by {edge:+.3f}vp (eff_thresh={effective_short:.2f})"
            )
            # Circuit breaker: kill short vol during liquidation cascades
            if jf > self._max_jump_fraction_for_shorts:
                direction = 0
                reason = (
                    f"FLAT (circuit breaker) — jump_fraction {jf:.3f} > "
                    f"{self._max_jump_fraction_for_shorts} (edge={edge:+.3f}vp)"
                )
        else:
            direction = 0
            reason = (
                f"FLAT — edge ({edge:+.3f}vp) within "
                f"+{effective_long:.2f}/-{effective_short:.2f}"
            )

        if self._log_every_bar:
            logger.info(
                "Signal @ %s: %s",
                bar.timestamp, reason,
            )

        return Signal(
            timestamp=bar.timestamp,
            direction=direction,
            edge_vol_pts=float(edge),
            rv_pred_pct=float(rv_pred_pct),
            iv_used_pct=float(iv_used_pct),
            bar_tsrv=bar_tsrv,
            term_slope_1w_2w=term_slope,
            straddle=straddle if direction != 0 else None,
            reason=reason,
        )

    def _compute_term_slope_1w_2w(self) -> float | None:
        """Per-bar IV term slope at 1w-2w (matches ``svi_term_slope_1w_2w``).

        Mirrors ``vol_arb/svi.py::_constant_maturity_features`` (line 1126):
        ``svi_term_slope_1w_2w = svi_atm_2w - svi_atm_1w``. Both legs in
        percent. Returns ``None`` when SVI hasn't calibrated yet so the
        caller can pass through (don't suppress signals during warmup).

        E.2.12 (2026-05-01): also returns ``None`` when either leg is
        exactly zero — a degenerate calibration that produces a
        spurious large-magnitude slope (e.g. zero 1w + 70 2w → +70vp
        slope incorrectly tripping the contango filter). Treat as
        warmup-equivalent and pass through.
        """
        if not self._svi.has_params:
            return None
        try:
            atm_1w = float(self._svi.atm_vol_at_dte(7.0))
            atm_2w = float(self._svi.atm_vol_at_dte(14.0))
        except (TypeError, ValueError):
            return None
        if not (math.isfinite(atm_1w) and math.isfinite(atm_2w)):
            return None
        if atm_1w <= 0.0 or atm_2w <= 0.0:
            logger.warning(
                "SVI atm_vol degenerate (1w=%s 2w=%s) — bypassing slope",
                atm_1w, atm_2w,
            )
            return None
        return float(atm_2w - atm_1w)

    def _lookup_iv(
        self, bar: Bar, straddle: StraddleSelection | None,
    ) -> float | None:
        """Look up the SVI IV at the ATM strike for this bar.

        Uses the frozen SVI params from :class:`LiveSVICalibrator` via
        ``evaluate_surface_batch``.  Returns None if:
        - No straddle was picked for this bar, OR
        - SVI has no params cached yet (before the first successful calibration), OR
        - The SVI slice for the straddle's expiry has no entries.
        """
        if straddle is None:
            return None
        if not self._svi.has_params:
            return None
        query = pd.DataFrame({
            "timestamp": [bar.timestamp],
            "strike": [float(straddle.strike)],
            "expiry_dt": [pd.Timestamp(straddle.expiry_dt)],
            "index_price": [float(bar.index_price) if bar.index_price is not None else float(straddle.index_price)],
        })
        result = self._svi.evaluate(query)
        if result is None or result.empty:
            return None
        iv = float(result.iloc[0])
        if not math.isfinite(iv) or iv <= 0.0:
            return None
        return iv

    def _flat_signal(
        self,
        bar: Bar,
        bar_tsrv: BarTSRV,
        rv_pred_pct: float,
        reason: str,
    ) -> Signal:
        return Signal(
            timestamp=bar.timestamp,
            direction=0,
            edge_vol_pts=float("nan"),
            rv_pred_pct=rv_pred_pct,
            iv_used_pct=float("nan"),
            bar_tsrv=bar_tsrv,
            straddle=None,
            reason=reason,
        )
