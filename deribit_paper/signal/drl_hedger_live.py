"""Live DRL delta-hedging inference wrapper (scalar per-trade per-tick).

The production :class:`vol_arb.drl_hedging.env.DRLHedger` exposes a
**vectorised batch API** (``simulate()``) optimised for backtest
rollouts — it takes arrays of N trades and internally loops over
10 ticks × N trades building batched observations for
``model.predict()``.

Live trading needs a different API:

- One straddle opens at a time (not N simultaneously)
- Each intra-bar hedge tick is a SEPARATE REST call to Deribit
- Per-trade state (inventory, entry spot, frozen SVI slice) must
  persist between ``step()`` calls, not be rebuilt from scratch
- The engine owns the wall clock — the hedger just computes target
  inventory given the current state

This module exposes a :class:`LiveDRLHedger` with three scalar methods:

- :meth:`begin_trade` — start a new hedged position.  Snapshots the
  SVI slice, records entry spot/delta/K/sigma, returns a trade_id.
- :meth:`step` — compute the target perp inventory for one intra-bar
  tick.  Pure scalar math + one model forward pass; no pandas.
- :meth:`close_trade` — tear down the per-trade state and return
  final hedge costs / residual summary for logging.

Implementation mirrors the ``env.step`` / ``DRLHedger.simulate`` math
exactly (sign-flipped virtual-short observation, VecNormalize,
sigmoid-SRL impact cost tracking).  The parity test
(``test_drl_hedger_live.py``) verifies this at 1e-6 tolerance.
"""

from __future__ import annotations

import logging
import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from deribit_paper.signal.model_loader import ModelBundle

logger = logging.getLogger(__name__)


# Matches vol_arb/drl_hedging/reward.py::straddle_delta_from_iv etc.
_INV_SQRT_2PI: float = 1.0 / math.sqrt(2.0 * math.pi)


def _norm_cdf(x: float) -> float:
    """Standard normal CDF via erf — matches vol_arb reward module."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _straddle_delta_from_iv(
    *, S: float, K: float, T: float, sigma: float,
) -> float:
    """Long straddle delta = 2*N(d1) - 1  (matches vol_arb.drl_hedging.reward)."""
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        if S > K:
            return 1.0
        if S < K:
            return -1.0
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    return 2.0 * _norm_cdf(d1) - 1.0


def _straddle_gamma_from_iv(
    *, S: float, K: float, T: float, sigma: float,
) -> float:
    """Long straddle gamma = 2 * call_gamma  (matches vol_arb.drl_hedging.reward)."""
    if T <= 0.0 or sigma <= 0.0 or S <= 0.0 or K <= 0.0:
        return 0.0
    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrt_T)
    pdf_d1 = math.exp(-0.5 * d1 * d1) * _INV_SQRT_2PI
    return 2.0 * pdf_d1 / (S * sigma * sqrt_T)


@dataclass
class _TradeSliceSVI:
    """Frozen SVI slice for one live trade (scalar per-field).

    Built once at :meth:`LiveDRLHedger.begin_trade` and reused for
    every :meth:`step` call during the trade's holding window.  Avoids
    DataFrame lookups in the inference hot path.
    """
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    T: float   # years — slice's own T (used as IV denominator)
    F: float   # forward at calibration

    def iv(self, strike: float) -> float:
        """SVI-implied vol (decimal) at a given strike."""
        if self.T <= 0.0 or self.F <= 0.0:
            return 0.0
        k = math.log(strike / self.F)
        km = k - self.m
        w = self.a + self.b * (self.rho * km + math.sqrt(km * km + self.sigma * self.sigma))
        w = max(w, 0.0)
        return math.sqrt(w / self.T)


@dataclass
class _LiveTradeState:
    """Per-trade mutable state held by :class:`LiveDRLHedger`."""
    trade_id: str
    signal: int                         # +1 long, -1 short
    K: float                            # straddle strike
    S_entry: float                      # index price at entry
    vol_ann_entry: float                # realised vol (decimal) for bar sim reference
    iv_entry: float                     # decimal IV at entry
    V_bg: float                         # background 1h volume (BTC) for impact calc
    jump_fraction: float                # entry-bar jump fraction
    vpin: float                         # 0-1 toxicity
    tau_entry: float                    # years
    delta_entry: float                  # portfolio delta (signed) at entry
    svi_slice: _TradeSliceSVI | None    # frozen slice (None → fall back to iv_entry)
    # Mutable across ticks
    hedge_inventory_btc: float = 0.0    # current perp position (signed, +long)
    tick: int = 0                       # which intra-bar step we're on
    delta_theo_entry_abs: float = 1.0   # denominator for obs normalisation
    # Accumulators for observability
    total_trade_size_btc: float = 0.0
    total_impact_cost_usd: float = 0.0
    step_log: list[dict[str, float]] = field(default_factory=list)


class LiveDRLHedger:
    """Scalar-inference wrapper around the production SAC v3 policy.

    Usage from the execution layer::

        hedger = LiveDRLHedger(bundle=model_bundle, n_steps_per_episode=10)

        trade_id = hedger.begin_trade(
            signal=-1, S_entry=60_000, K=60_000,
            vol_ann_entry=0.80, iv_entry=0.75,
            tau_entry=0.04, V_bg=100.0,
            jump_fraction=0.05, vpin=0.3,
            svi_slice_params={"a": ..., "b": ..., ...},
        )

        for tick in range(1, 11):
            # ... wait ~90s, fetch current perp spot ...
            target_inventory = hedger.step(trade_id, current_spot=60_100.0)
            # ... place a perp order to move to target_inventory ...

        summary = hedger.close_trade(trade_id)

    Model + VecNormalize stats come from :class:`ModelBundle`.
    """

    # Observation indices (MUST match vol_arb.drl_hedging.env.GymnasiumHedgingEnv)
    _OBS_DIM = 8
    _IDX_LOG_S = 0
    _IDX_TAU_REL = 1
    _IDX_DELTA = 2
    _IDX_INV = 3
    _IDX_VPIN = 4
    _IDX_GAMMA = 5
    _IDX_TICK = 6
    _IDX_JUMP = 7

    def __init__(
        self,
        *,
        bundle: ModelBundle,
        n_steps_per_episode: int = 10,
        bar_minutes: int = 15,
        dust_threshold_btc: float = 0.01,
    ):
        self._bundle = bundle
        self._n_steps = n_steps_per_episode
        self._bar_minutes = bar_minutes
        # D4 (parity-cycle round 2 — 2026-05-07): the dust threshold the
        # downstream PerpHedger uses to skip sub-dust orders. ``step()``
        # mirrors the skip on the inventory-update path so internal
        # state doesn't drift from actual perp position when the policy
        # emits a sub-dust action. Pre-D4 the inventory update fired
        # unconditionally; over many sub-dust ticks the drift compounds
        # and the obs vector goes out of training distribution.
        self._dust_threshold_btc = float(dust_threshold_btc)
        self._trades: dict[str, _LiveTradeState] = {}

    def active_trade_count(self) -> int:
        return len(self._trades)

    # ── begin_trade ────────────────────────────────────────────────

    def begin_trade(
        self,
        *,
        signal: int,
        S_entry: float,
        K: float,
        vol_ann_entry: float,
        iv_entry: float,
        tau_entry: float,
        V_bg: float,
        jump_fraction: float,
        vpin: float,
        svi_slice_params: dict[str, float] | None = None,
        trade_id: str | None = None,
    ) -> str:
        """Start a new hedged position.  Returns the trade_id."""
        if signal not in (1, -1):
            raise ValueError(f"signal must be ±1, got {signal}")
        if tau_entry <= 0.0:
            raise ValueError(f"tau_entry must be > 0, got {tau_entry}")

        tid = trade_id or f"live_{uuid.uuid4().hex[:12]}"

        # Portfolio delta at entry (signed: negative for short straddle)
        delta_long = _straddle_delta_from_iv(
            S=S_entry, K=K, T=tau_entry, sigma=iv_entry,
        )
        delta_entry = signal * delta_long

        slice_obj: _TradeSliceSVI | None = None
        if svi_slice_params is not None:
            slice_obj = _TradeSliceSVI(
                a=float(svi_slice_params["a"]),
                b=float(svi_slice_params["b"]),
                rho=float(svi_slice_params["rho"]),
                m=float(svi_slice_params["m"]),
                sigma=float(svi_slice_params["sigma"]),
                T=float(svi_slice_params["T"]),
                F=float(svi_slice_params["F"]),
            )

        state = _LiveTradeState(
            trade_id=tid,
            signal=signal,
            K=K,
            S_entry=S_entry,
            vol_ann_entry=vol_ann_entry,
            iv_entry=iv_entry,
            V_bg=V_bg,
            jump_fraction=jump_fraction,
            vpin=vpin,
            tau_entry=tau_entry,
            delta_entry=delta_entry,
            svi_slice=slice_obj,
            delta_theo_entry_abs=max(abs(delta_entry), 0.01),
        )
        self._trades[tid] = state
        logger.info(
            "LiveDRLHedger begin_trade %s signal=%+d K=%.0f S=%.0f iv=%.2f tau=%.4f",
            tid, signal, K, S_entry, iv_entry, tau_entry,
        )
        return tid

    # ── step ───────────────────────────────────────────────────────

    def step(
        self,
        trade_id: str,
        current_spot: float,
        *,
        dt_years: float | None = None,
    ) -> float:
        """Query the DRL policy for the target perp inventory at this tick.

        Parameters
        ----------
        trade_id : str
            Identifier returned by :meth:`begin_trade`.
        current_spot : float
            Live BTC index price (USD) observed right before this
            hedge tick.  Used to update the observation and compute
            the target inventory.
        dt_years : float, optional
            Time since the previous tick in years.  Defaults to
            ``1 / (10 * bar_per_year)`` which matches the backtester's
            ``hedge_sim_steps=10`` over a 15-min bar.

        Returns
        -------
        float
            The target perp inventory in BTC (signed).  The caller
            computes ``trade_size = target - current_inventory`` and
            submits a perp order for that delta.
        """
        if trade_id not in self._trades:
            raise KeyError(f"Unknown trade_id: {trade_id}")
        t = self._trades[trade_id]

        if current_spot <= 0.0:
            raise ValueError(f"current_spot must be positive, got {current_spot}")

        t.tick += 1

        # Compute the tau at this tick
        if dt_years is None:
            dt_years = (self._bar_minutes / 60.0 / 24.0 / 365.25) / self._n_steps
        tau_t = max(t.tau_entry - t.tick * dt_years, 1e-10)

        # IV from frozen SVI slice (or fall back to entry iv)
        iv_t = t.iv_entry
        if t.svi_slice is not None:
            svi_iv = t.svi_slice.iv(t.K)
            if math.isfinite(svi_iv) and 0.0 < svi_iv < 5.0:
                iv_t = svi_iv

        # Long-side delta + portfolio delta in ACTUAL (signed) coordinates
        delta_long_t = _straddle_delta_from_iv(
            S=current_spot, K=t.K, T=tau_t, sigma=iv_t,
        )
        delta_theo_t = t.signal * delta_long_t

        # Long gamma (portfolio gamma = signal * long_gamma for straddle)
        gamma_long_t = _straddle_gamma_from_iv(
            S=current_spot, K=t.K, T=tau_t, sigma=iv_t,
        )

        # ── Build the 8-dim observation in virtual-short convention ──
        # This MUST match vol_arb/drl_hedging/env.py::GymnasiumHedgingEnv._build_obs
        # and DRLHedger.simulate() for the policy to behave correctly.
        log_S = math.log(max(current_spot, 1e-12) / max(t.S_entry, 1e-12))
        tau_rel = tau_t / max(t.tau_entry, 1e-12)

        # Virtual-short observation flip:
        #   delta_obs = -delta_long_t  (always negative of long delta)
        #   inv_obs = -signal * H / |delta_entry_abs|
        #   gamma_obs = -gamma_long * S² / notional  (always negative)
        delta_obs = -delta_long_t
        inv_obs = (-t.signal) * t.hedge_inventory_btc / t.delta_theo_entry_abs

        notional = max(current_spot * abs(delta_theo_t) + 1.0, 1.0)
        gamma_obs = -gamma_long_t * current_spot * current_spot / notional

        tick_frac = t.tick / self._n_steps
        jump_flag = 1.0 if t.jump_fraction > 0.10 else 0.0

        obs = np.array([
            log_S,
            max(0.0, min(5.0, tau_rel)),
            max(-1.0, min(1.0, delta_obs)),
            max(-5.0, min(5.0, inv_obs)),
            max(0.0, min(1.0, t.vpin)),
            max(-5.0, min(5.0, gamma_obs)),
            tick_frac,
            jump_flag,
        ], dtype=np.float32)

        # Apply VecNormalize stats if present
        if self._bundle.drl_vec_normalize_mean is not None:
            mean = np.asarray(self._bundle.drl_vec_normalize_mean, dtype=np.float64)
            var = np.asarray(self._bundle.drl_vec_normalize_var, dtype=np.float64)
            clip = self._bundle.drl_vec_normalize_clip
            obs = ((obs - mean) / np.sqrt(var + 1e-8)).astype(np.float32)
            obs = np.clip(obs, -clip, clip)

        # Policy forward pass
        action, _ = self._bundle.drl_policy.predict(
            obs.reshape(1, -1), deterministic=True,
        )
        a = float(np.clip(action.reshape(-1)[0], -1.0, 1.0))

        # Decode in ACTUAL coordinates: target_H = -delta_theo_t, then
        # trade_size = a * |delta_required| * sign(delta_required)
        target_H_full = -delta_theo_t
        delta_required = target_H_full - t.hedge_inventory_btc
        sign = 1.0 if delta_required >= 0 else -1.0
        trade_size = a * abs(delta_required) * sign

        # D4 (parity-cycle round 2 — 2026-05-07): skip the inventory
        # update on sub-dust trades. The downstream ``PerpHedger`` will
        # also skip the order, so updating internal inventory here would
        # silently drift the policy's hedge_inventory_btc away from the
        # actual perp position by exactly the sub-dust amount per tick.
        # Over a 10-tick window (or many trades), the drift compounds.
        if abs(trade_size) < self._dust_threshold_btc:
            trade_size = 0.0

        # Update state
        t.hedge_inventory_btc += trade_size
        t.total_trade_size_btc += abs(trade_size)

        # Track for summary
        t.step_log.append({
            "tick": t.tick,
            "spot": current_spot,
            "delta_theo": delta_theo_t,
            "iv": iv_t,
            "tau": tau_t,
            "inventory_before": t.hedge_inventory_btc - trade_size,
            "trade_size": trade_size,
            "inventory_after": t.hedge_inventory_btc,
            "action": a,
        })

        return float(t.hedge_inventory_btc)

    # ── close_trade ────────────────────────────────────────────────

    def close_trade(self, trade_id: str) -> dict[str, Any]:
        """Tear down per-trade state and return a summary dict."""
        if trade_id not in self._trades:
            raise KeyError(f"Unknown trade_id: {trade_id}")
        t = self._trades.pop(trade_id)
        summary = {
            "trade_id": trade_id,
            "signal": t.signal,
            "K": t.K,
            "S_entry": t.S_entry,
            "final_inventory_btc": t.hedge_inventory_btc,
            "total_trade_size_btc": t.total_trade_size_btc,
            "n_ticks_executed": t.tick,
            "step_log": t.step_log,
        }
        logger.info(
            "LiveDRLHedger close_trade %s: final_inv=%.4f total_traded=%.4f ticks=%d",
            trade_id, t.hedge_inventory_btc, t.total_trade_size_btc, t.tick,
        )
        return summary

    def close_all(self) -> list[dict[str, Any]]:
        """Emergency close — used by the kill switch to flatten positions."""
        return [self.close_trade(tid) for tid in list(self._trades.keys())]
