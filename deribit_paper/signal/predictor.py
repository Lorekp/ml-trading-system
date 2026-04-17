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
        self._log_every_bar = log_every_bar

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

        # 2) LightGBM prediction — outputs ln(vol_ann), convert to vol points
        pred_ln = float(self._bundle.lgbm_model.predict(row.values.reshape(1, -1))[0])
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
            straddle=straddle if direction != 0 else None,
            reason=reason,
        )

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
