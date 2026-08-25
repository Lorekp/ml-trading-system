"""Risk-neutral density (RND) tail-risk monitor for the paper engine.

**Purpose**: the vol-arb strategy is a lottery-ticket positive-skew trade
(70.9% of PnL from the top 5% of days per Stage 14.4). Most of that tail
P&L comes from vol EXPANSIONS — days when the market re-prices itself
with a sudden surge in implied vol. The opposite regime — vol
compressions into crash scenarios — is the risk that a short-vol
position can blow up in.

Deribit's live option market prices embed forward-looking
vol-of-vol and tail risk via the SVI surface. The **risk-neutral
density** (Breeden-Litzenberger) extracted from that surface tells us
the market's implied probability distribution over future spot prices.
When that distribution has abnormally fat tails, the market is pricing
in a higher chance of extreme moves — which is exactly when we want to
(a) stop opening new short-vol positions, and (b) shrink the size of
any new positions we do take.

This monitor:

1. At each bar close, reconstructs the front-expiry SVI slice from the
   ``LiveSVICalibrator``'s cached params.
2. Computes the RND on a log-moneyness grid and integrates for the
   probability of ±5% (configurable) moves at the front expiry.
3. Maintains a rolling history of tail probabilities to establish a
   baseline.
4. Fires a kill signal when the current tail prob > ``tail_halt_mult``
   × the rolling baseline.
5. Returns a size-scale factor that linearly scales position size
   downward as tail prob approaches the halt threshold.

The monitor is **defensive, not offensive**. It does NOT generate new
trades. It only suppresses or shrinks new trades when the market is
pricing in extreme moves that the strategy isn't designed to absorb.

Integration with ``TradingEngine``:

- Instantiated in :meth:`TradingEngine.initialize` alongside the other
  risk components.
- :meth:`update` is called at each bar close (synchronously, after the
  SVI surface is fresh).
- :meth:`should_halt` is checked before each signal decision, as part
  of the pre-signal risk gate (same place the kill switch is checked).
- :meth:`size_scale` multiplies the straddle flow's position sizing.

References:
- `vol_arb/svi.py::SVISlice.risk_neutral_density`
- `vol_arb/svi.py::slice_from_params_row`
- Breeden, D.T., and R.H. Litzenberger (1978). "Prices of State-
  contingent claims implicit in option prices." *J. Business* 51:621-651
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from vol_arb.svi import SVISlice, slice_from_params_row

logger = logging.getLogger(__name__)


@dataclass
class RNDSnapshot:
    """One bar's worth of tail-risk metrics extracted from the RND."""
    timestamp: pd.Timestamp
    front_expiry: pd.Timestamp
    front_T_years: float
    spot: float
    forward: float
    # Cumulative tail probabilities at ±5% moves (configurable)
    tail_prob_down: float
    tail_prob_up: float
    tail_prob_total: float
    # RND shape statistics
    rnd_integral: float       # should be ~1.0 for a well-calibrated surface
    rnd_skewness: float       # higher absolute skew → more asymmetric tail risk
    rnd_kurtosis: float       # higher → fatter tails


@dataclass
class RNDConfig:
    """All knobs for the RND monitor, grouped.

    **Tail-move threshold design.** The threshold is defined in
    *sigma units* of the ATM total variance: a log-moneyness move
    of magnitude ``tail_sigma_mult × sqrt(w_ATM)``, where ``w_ATM``
    is the SVI total variance at k=0 for the current slice.  This
    automatically adapts to both the term structure (w scales with
    T) AND the current vol regime, so the "tail" always means the
    same number of standard deviations no matter what BTC's implied
    vol is doing.  The baseline tail probability under normality is
    ~5% at 2σ; the RND's fat-tail shape makes it ~8-12% in practice.

    Under the earlier fixed-percentage design (``tail_move_pct = 5%``),
    the "tail" was 53% of the density at 1d and near-zero at 30d —
    the baseline saturated at one end and starved at the other.
    The sqrt(T)-scaled percent intermediate ``tail_move_pct_at_1d``
    was only marginally better (53% median baseline at 7d, because
    1.5% × sqrt(7) ≈ 4% is barely 1 daily-vol std).  Both designs
    produced a monitor that never fired on historical stress events.
    The sigma-multiple design fires correctly on historical data
    (see ``scripts/forensic/12_rnd_backtest.py``).

    **Baseline design.** The halt rule fires when the current tail
    prob exceeds ``tail_halt_mult`` × ``baseline``.  The baseline is
    the **max** of a short-window (30d default) and long-window (90d
    default) rolling median, so that a sustained elevated-vol regime
    cannot ratchet the short-window median up and silence the
    monitor.  The 90d window is an anchor.
    """
    # Tail definition: threshold at ``tail_sigma_mult × sqrt(w_ATM)``
    # log-moneyness, i.e., N standard deviations of the ATM-variance.
    # 2.0 = 2-sigma tail (~5% mass under normality, 8-12% under the
    # fat-tailed RND). Bigger = narrower tail = more sensitive halt.
    tail_sigma_mult: float = 2.0
    # Halt rule: if current tail_prob_total > halt_mult × baseline, halt.
    # W2.2 backtest showed that no halt_mult value delivers positive
    # Sharpe delta — the strategy's existing risk management already
    # handles the stress events this monitor is meant to catch.  We
    # At halt_mult=2.0 and with the sigma-mult design, the historical halt
    # rate is ~0%; retain the monitor only as historical paper/replay
    # telemetry.
    # It is not validated as a live tail-risk control.
    # Grid for numerical integration of the RND (log-moneyness).
    k_grid_min: float = -0.40    # ~ exp(-0.40) = 0.67, 33% down
    k_grid_max: float = 0.40     # ~ exp(0.40) = 1.49, 49% up
    k_grid_n: int = 801          # step size ~ 0.001
    # Front expiry filter: only use slices with DTE in this band.
    # Too close → the RND is noisy and the daily-expiry roll dominates;
    # too far → the tail signal is diluted by slow-moving vol.
    # The band is deliberately narrow; within the band, the slice
    # closest to ``target_dte_days`` is picked. This avoids the sawtooth
    # that a ``.iloc[0]`` pick on ``sort_values("_dte_days")`` produced
    # at each Friday weekly-expiry roll.
    min_dte_days: float = 3.0
    max_dte_days: float = 14.0
    target_dte_days: float = 7.0
    # Bar timeframe for deriving bar-count defaults. Set to match the
    # engine's bar_minutes so the rolling windows scale automatically.
    bar_minutes: int = 15
    # Rolling history windows for the baseline. Defaults (-1) resolve
    # from bar_minutes: 30 * bars_per_day, 90 * bars_per_day, bars_per_day.
    # Explicit overrides are preserved.
    baseline_window_bars: int = -1
    baseline_long_window_bars: int = -1
    baseline_warmup_bars: int = -1

    def __post_init__(self) -> None:
        bpd = 1440 // self.bar_minutes
        if self.baseline_window_bars < 0:
            self.baseline_window_bars = 30 * bpd
        if self.baseline_long_window_bars < 0:
            self.baseline_long_window_bars = 90 * bpd
        if self.baseline_warmup_bars < 0:
            self.baseline_warmup_bars = bpd
    # Halt-rule multiplier (see tail_sigma_mult docstring above).
    # Set to 2.0 after W2.2 backtest validation — higher than the
    # original 2.0 would lose coverage, lower would cut PnL.
    tail_halt_mult: float = 2.0
    # Size-scale rule: linear ramp from 1.0 at baseline to ``size_min``
    # at (halt_mult * baseline). Below baseline, no scaling.
    size_min: float = 0.25


class RNDMonitor:
    """Stateful monitor that tracks front-expiry tail-risk from the SVI surface.

    Usage::

        monitor = RNDMonitor(config=RNDConfig())

        # At each bar close (after SVI recalibration):
        snapshot = monitor.update(
            timestamp=bar.timestamp,
            spot=bar.index_price,
            params_df=svi_calibrator._params_df,
        )

        if monitor.should_halt():
            logger.warning("RND halted new signals: %s", monitor.halt_reason())
            return

        scale = monitor.size_scale()
        n_target *= scale  # shrink position by the RND-implied risk factor

    Thread safety: single-threaded asyncio. No locks.
    """

    def __init__(
        self,
        config: RNDConfig | None = None,
        *,
        started_at: pd.Timestamp | None = None,
    ):
        self.config = config or RNDConfig()
        # Deque sized to the LONGER of the two baseline windows so that
        # ``_baseline_tail_prob`` can compute both medians from a single
        # history buffer. Defaults to the 90-day long window.
        history_len = max(
            self.config.baseline_window_bars,
            self.config.baseline_long_window_bars,
        )
        self._history: deque[RNDSnapshot] = deque(maxlen=history_len)
        self._latest: RNDSnapshot | None = None
        self._halt_reason: str | None = None
        # Bug fix P1-3 (2026-05-07): track construction time so
        # ``size_scale`` can degrade to ``size_min`` when ``_latest``
        # stays None for too long. Pre-fix returned 1.0 (full size)
        # forever when ``update()`` was never called successfully —
        # e.g. SVI never calibrates (broken option_buffer), or every
        # update raises and the surrounding try/except in
        # engine.process_bar swallows it. The defensive direction is
        # ``size_min``, not ``1.0``: when the tail-risk monitor is
        # broken, we'd rather size down than full size blind. Same
        # silent-failure family as the VPIN bug.
        self._started_at: pd.Timestamp = (
            started_at if started_at is not None
            else pd.Timestamp.now(tz="UTC")
        )
        # Grace period before declaring "broken": same order as the
        # baseline_warmup_bars expressed in time. Default 1 day to
        # match the legitimate warmup window — a fresh engine has
        # time to receive the first SVI calibration before this trips.
        self._broken_grace_seconds: float = (
            self.config.baseline_warmup_bars * self.config.bar_minutes * 60.0
        )

    @property
    def latest(self) -> RNDSnapshot | None:
        return self._latest

    @property
    def n_samples(self) -> int:
        return len(self._history)

    def halt_reason(self) -> str | None:
        return self._halt_reason

    def update(
        self,
        *,
        timestamp: pd.Timestamp,
        spot: float,
        params_df: pd.DataFrame | None,
    ) -> RNDSnapshot | None:
        """Compute tail metrics from the current SVI surface and store.

        Returns the snapshot, or ``None`` if the surface is unavailable or
        no slice meets the DTE filter.
        """
        if params_df is None or params_df.empty:
            logger.debug("RND monitor: no SVI params available")
            return None

        slice_obj = self._select_front_slice(params_df, timestamp)
        if slice_obj is None:
            logger.debug("RND monitor: no front slice within DTE band")
            return None

        snapshot = self._compute_snapshot(
            timestamp=timestamp, spot=spot, svi_slice=slice_obj,
        )
        self._latest = snapshot
        self._history.append(snapshot)
        # Re-evaluate halt state
        self._halt_reason = self._check_halt(snapshot)
        return snapshot

    def should_halt(self) -> bool:
        """Return True if the RND tail probability exceeds the halt threshold."""
        return self._halt_reason is not None

    def is_warmup_failed(self, now: pd.Timestamp | None = None) -> bool:
        """Return True iff the monitor has been alive past its grace
        period AND has never received a successful ``update()``.

        Indicates a broken upstream (SVI never calibrated, option_buffer
        empty, engine not calling update()). Used by ``size_scale`` to
        degrade to ``size_min`` instead of returning 1.0 (full size).
        """
        if self._latest is not None:
            return False
        now = now or pd.Timestamp.now(tz="UTC")
        elapsed_seconds = (now - self._started_at).total_seconds()
        return elapsed_seconds > self._broken_grace_seconds

    def size_scale(self, now: pd.Timestamp | None = None) -> float:
        """Return a position-size multiplier in [size_min, 1.0].

        - 1.0 when current tail prob is at or below the rolling baseline
        - Linearly ramps down to ``size_min`` as tail prob approaches the
          halt threshold
        - ``size_min`` when at or above the halt threshold

        Returns 1.0 when we don't yet have enough history to form a
        baseline (legitimate warmup phase, ``_latest`` is set but
        ``baseline`` is None because ``baseline_warmup_bars`` haven't
        elapsed).

        P1-3 fix (2026-05-07): when the monitor has been alive past
        its warmup grace AND has never received an update (``_latest
        is None`` and elapsed > ``_broken_grace_seconds``), return
        ``size_min`` instead of 1.0. The defensive direction matters:
        a broken tail-risk monitor sizing trades at 100% is the worst
        of both worlds.
        """
        # P1-3: detect the broken-warmup case BEFORE the legitimate
        # warmup short-circuit. ``_latest is None`` after the grace
        # period == the upstream never delivered a single update.
        if self.is_warmup_failed(now=now):
            return self.config.size_min
        baseline = self._baseline_tail_prob()
        if baseline is None or self._latest is None:
            return 1.0
        current = self._latest.tail_prob_total
        if current <= baseline:
            return 1.0
        halt_level = baseline * self.config.tail_halt_mult
        if current >= halt_level:
            return self.config.size_min
        # Linear interpolation between baseline and halt_level
        frac = (current - baseline) / (halt_level - baseline)
        scale = 1.0 - frac * (1.0 - self.config.size_min)
        return float(scale)

    # ── Internals ────────────────────────────────────────────────────

    def _select_front_slice(
        self,
        params_df: pd.DataFrame,
        timestamp: pd.Timestamp,
    ) -> SVISlice | None:
        """Pick the front-expiry SVI slice that meets the DTE filter."""
        if "timestamp" in params_df.columns:
            # Use the latest bucket with timestamp <= now
            recent = params_df[params_df["timestamp"] <= timestamp]
            if recent.empty:
                return None
            latest_ts = recent["timestamp"].max()
            bucket = recent[recent["timestamp"] == latest_ts].copy()
        else:
            bucket = params_df.copy()

        if bucket.empty:
            return None

        # Compute DTE in days for each expiry
        if "expiry_dt" in bucket.columns:
            dte_days = (
                (bucket["expiry_dt"] - timestamp).dt.total_seconds() / 86_400.0
            )
            bucket = bucket.assign(_dte_days=dte_days)
        elif "T" in bucket.columns:
            bucket = bucket.assign(_dte_days=bucket["T"] * 365.25)
        else:
            return None

        # Apply DTE filter, then pick the slice CLOSEST to the target
        # DTE within the band. Picking the absolute smallest (old
        # behaviour) produced a sawtooth: every Friday the weekly
        # expired and the front slice jumped from ~1d to ~7d overnight.
        # Closest-to-target within a narrow band stabilises the RND
        # across the weekly roll.
        cfg = self.config
        eligible = bucket[
            (bucket["_dte_days"] >= cfg.min_dte_days)
            & (bucket["_dte_days"] <= cfg.max_dte_days)
        ]
        if eligible.empty:
            return None
        distance = (eligible["_dte_days"] - cfg.target_dte_days).abs()
        front = eligible.loc[distance.idxmin()]

        try:
            return slice_from_params_row(front)
        except Exception as exc:  # noqa: BLE001 - defensive
            logger.warning("RND monitor: failed to reconstruct slice: %s", exc)
            return None

    def _compute_snapshot(
        self,
        *,
        timestamp: pd.Timestamp,
        spot: float,
        svi_slice: SVISlice,
    ) -> RNDSnapshot:
        """Compute tail metrics + RND shape stats from an SVI slice."""
        cfg = self.config
        F = float(svi_slice.F)
        T = float(svi_slice.T)

        # Log-moneyness grid
        k_grid = np.linspace(cfg.k_grid_min, cfg.k_grid_max, cfg.k_grid_n)
        density = svi_slice.risk_neutral_density(k_grid)

        # Numerical integral (trapezoid).  For a well-calibrated SVI slice
        # this should be close to 1.0.
        rnd_integral = float(np.trapezoid(density, k_grid))
        if rnd_integral <= 0:
            rnd_integral = 1.0  # avoid div-by-zero; density is junk anyway
        density_norm = density / rnd_integral

        # Tail threshold in log-moneyness at ``sigma_mult × sqrt(w_ATM)``.
        # w_ATM is the SVI total variance at k=0 for the current slice.
        # This auto-scales with both term structure (w ∝ T) AND vol
        # regime (w ∝ IV²), so "tail" always means the same number of
        # standard deviations from the current forward.
        w_atm = float(svi_slice.total_variance(np.array([0.0]))[0])
        sigma_atm = float(np.sqrt(max(w_atm, 1e-12)))
        k_threshold = cfg.tail_sigma_mult * sigma_atm
        tail_down_k = -k_threshold
        tail_up_k = k_threshold

        tail_prob_down = float(
            np.trapezoid(
                density_norm[k_grid <= tail_down_k],
                k_grid[k_grid <= tail_down_k],
            ),
        )
        tail_prob_up = float(
            np.trapezoid(
                density_norm[k_grid >= tail_up_k],
                k_grid[k_grid >= tail_up_k],
            ),
        )

        # RND shape stats (moments computed in log-moneyness space)
        mean = float(np.trapezoid(k_grid * density_norm, k_grid))
        centered = k_grid - mean
        var = float(np.trapezoid(centered ** 2 * density_norm, k_grid))
        std = np.sqrt(max(var, 1e-12))
        skew_num = float(np.trapezoid(centered ** 3 * density_norm, k_grid))
        kurt_num = float(np.trapezoid(centered ** 4 * density_norm, k_grid))
        skewness = skew_num / (std ** 3) if std > 0 else 0.0
        # Excess kurtosis (normal = 0)
        kurtosis = (kurt_num / (std ** 4) - 3.0) if std > 0 else 0.0

        return RNDSnapshot(
            timestamp=timestamp,
            front_expiry=self._slice_expiry(svi_slice, timestamp),
            front_T_years=T,
            spot=spot,
            forward=F,
            tail_prob_down=tail_prob_down,
            tail_prob_up=tail_prob_up,
            tail_prob_total=tail_prob_down + tail_prob_up,
            rnd_integral=rnd_integral,
            rnd_skewness=skewness,
            rnd_kurtosis=kurtosis,
        )

    @staticmethod
    def _slice_expiry(svi_slice: SVISlice, now: pd.Timestamp) -> pd.Timestamp:
        """Infer the expiry timestamp from T (years) if not stored on the slice."""
        # SVISlice doesn't carry expiry_dt; reconstruct from T.
        delta = pd.Timedelta(seconds=float(svi_slice.T) * 365.25 * 24 * 3600)
        return now + delta

    def _baseline_tail_prob(self) -> float | None:
        """Rolling-window baseline tail probability.

        Returns ``max(median(last 30d), median(last 90d))``.  The short
        window tracks recent regime shifts; the long window is an
        anchor that prevents a sustained elevated-vol regime from
        ratcheting the short-window median upward and silencing the
        monitor.  The maximum of the two keeps the halt rule sensitive
        to spikes over BOTH the recent and the long-term distribution.

        Returns ``None`` until the warmup threshold is reached.
        """
        if len(self._history) < self.config.baseline_warmup_bars:
            return None
        tails = np.fromiter(
            (s.tail_prob_total for s in self._history),
            dtype=np.float64,
            count=len(self._history),
        )
        short_n = min(len(tails), self.config.baseline_window_bars)
        short_med = float(np.median(tails[-short_n:]))
        # Long window; falls back to whatever history we have so far.
        long_n = min(len(tails), self.config.baseline_long_window_bars)
        long_med = float(np.median(tails[-long_n:]))
        return max(short_med, long_med)

    def _check_halt(self, snapshot: RNDSnapshot) -> str | None:
        """Return a halt reason string if the halt rule fires, else None."""
        baseline = self._baseline_tail_prob()
        if baseline is None:
            return None  # warmup, rule not active
        halt_level = baseline * self.config.tail_halt_mult
        if snapshot.tail_prob_total > halt_level:
            return (
                f"RND tail prob {snapshot.tail_prob_total:.4f} > "
                f"{self.config.tail_halt_mult:.1f}× baseline "
                f"{baseline:.4f} (front-expiry DTE "
                f"{snapshot.front_T_years * 365.25:.1f}d, "
                f"skew={snapshot.rnd_skewness:.2f}, "
                f"kurt={snapshot.rnd_kurtosis:.2f})"
            )
        return None

    def to_dict(self) -> dict[str, Any]:
        """Snapshot of the monitor's current state for logging/telemetry."""
        return {
            "n_samples": self.n_samples,
            "is_halted": self.should_halt(),
            "halt_reason": self.halt_reason(),
            "size_scale": self.size_scale(),
            "latest_tail_prob": (
                self._latest.tail_prob_total if self._latest else None
            ),
            "latest_skew": self._latest.rnd_skewness if self._latest else None,
            "latest_kurt": self._latest.rnd_kurtosis if self._latest else None,
            "baseline_tail_prob": self._baseline_tail_prob(),
        }
