"""Gymnasium hedging environment for the DRL delta-hedging agent.

Reuses data already produced by the vol-arb pipeline:

- ``BTC_vol_arb.parquet`` for the bar-level state (entry spot, vol_ann,
  straddle strike & expiry, trailing volume, jump fraction).
- ``svi_surface_params_1h.parquet`` for SVI-interpolated BSM repricing
  at every intra-bar tick.

Episode structure
-----------------

Each episode represents ONE short-straddle trade of 15 minutes.  At
``reset()``:

1. Sample a random active bar from the training window.  Optionally
   upsample high-vol bars so the agent sees jump regimes more often.
2. Build a ``_TradeState`` from the bar — strike, expiry, entry spot,
   entry IV, background volume, jump fraction, VPIN.
3. Look up the SVI slice matching the trade's expiry **once** and
   freeze (a, b, rho, m, sigma, T) scalars.  Intra-episode repricing
   then uses raw SVI math with zero DataFrame ops.
4. Generate a stochastic GBM price path for 10 intra-bar ticks.

Each ``step(action)`` call:

1. Interpret the action in [-1, 1] as a fraction of the required delta
   adjustment, and trade that many BTC on perps.
2. Compute cost via :func:`sigmoid_sqrt_impact` + perp taker fee.
3. Reprice the short-straddle position with the frozen SVI slice at
   the new spot / tau.
4. Compute the step reward via :func:`accounting_pnl_reward`.

Action space: ``Box(low=-1, high=1, shape=(1,))``

Observation space (8-dim, Box):

    [log(S_t/S_entry), tau/tau_entry, delta_t, H_{t-1}/|delta_theo_entry|,
     VPIN_t, gamma_t, tick/N, jump_indicator]

The ``DRLHedger`` adapter at the bottom of the file exposes the same
signature as ``vol_arb.backtester.simulate_discrete_hedging`` so the
backtester can switch between rule-based and DRL policies behind a
single config flag.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from vol_arb.drl_hedging.config import DRLHedgingConfig
from vol_arb.drl_hedging.impact import sigmoid_sqrt_impact
from vol_arb.drl_hedging.reward import (
    accounting_pnl_reward,
    residual_delta_reward,
    straddle_delta_from_iv,
    straddle_gamma_from_iv,
    straddle_price_from_iv,
)
from vol_arb.drl_hedging.vpin import compute_vpin_proxy_from_returns

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# Frozen per-slice SVI params (no DataFrames in the hot loop)
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class _SVISlice:
    """Frozen SVI params for a single (timestamp, expiry_dt) slice.

    Once computed at ``reset()``, the env evaluates IV at the current
    spot via raw numpy scalar math — no pandas ops in the hot loop.
    """
    a: float
    b: float
    rho: float
    m: float
    sigma: float
    T: float   # years — slice's own T, used as the denominator for IV
    F: float   # forward price at calibration time

    def iv(self, strike: float) -> float:
        """Return the SVI-implied vol (decimal) for a given strike."""
        if self.T <= 0.0:
            return 0.0
        k = float(np.log(strike / self.F)) if self.F > 0 else 0.0
        km = k - self.m
        w = self.a + self.b * (self.rho * km + float(np.sqrt(km * km + self.sigma * self.sigma)))
        w = max(w, 0.0)
        return float(np.sqrt(w / self.T))


# ─────────────────────────────────────────────────────────────────────────
# Trade state (resampled per episode)
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class _TradeState:
    """Snapshot of one straddle trade at entry.

    The ``signal`` field encodes the trade direction:
    - ``signal = -1`` — SHORT straddle (classic vol-selling)
    - ``signal = +1`` — LONG straddle (vol-buying)

    Portfolio delta at any tick is ``signal * long_straddle_delta``.
    For a short straddle the delta flips sign relative to the long
    calculation.  Gamma flips too: long is positive, short is negative.
    """
    entry_ts: pd.Timestamp
    expiry_ts: pd.Timestamp
    K: float                 # straddle strike
    S_entry: float           # index price at entry
    vol_ann_entry: float     # realised vol (annualised decimal) at entry
    iv_entry: float          # IV (decimal) at entry — from SVI slice
    V_bg: float              # background 1h volume (BTC)
    jump_fraction: float     # TSRV jump fraction at entry
    vpin: float              # toxicity signal in [0, 1]
    tau_entry: float         # T-to-expiry at entry (years)
    delta_entry: float       # theoretical portfolio delta at entry (signed)
    signal: int              # +1 long, -1 short

    def tau_at_tick(self, tick: int, dt_years: float) -> float:
        """Time-to-expiry at tick k (years)."""
        return max(self.tau_entry - tick * dt_years, 1e-10)


# ─────────────────────────────────────────────────────────────────────────
# Market data view — wraps the parquets and exposes sample access
# ─────────────────────────────────────────────────────────────────────────

class MarketDataView:
    """Lazy read-only view over the vol-arb parquet + SVI surface params.

    Filters to a training window and precomputes the active-bar index
    (bars with non-null straddle data) so ``sample_bar()`` is O(1).
    """

    def __init__(
        self,
        vol_arb_path: str,
        svi_params_path: str,
        period: tuple[str, str],
        high_vol_upsample_threshold: float = 0.80,
        high_vol_upsample_factor: float = 3.0,
        vpin_series: pd.Series | None = None,
    ):
        self.period = period
        df = pd.read_parquet(vol_arb_path)
        start, end = pd.Timestamp(period[0], tz="UTC"), pd.Timestamp(period[1], tz="UTC")
        df = df[(df.index >= start) & (df.index <= end)]

        required = ["iv", "strike", "expiry_dt", "index_price",
                    "vol_ann", "trailing_1h_volume", "jump_fraction"]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"vol_arb parquet missing required columns: {missing}")

        # Lag vol_ann by one bar: vol_ann[T] is the realized vol OF bar T
        # (open-stamped), so using it at decision time T is forward-looking.
        # The GBM simulator should use the PREVIOUS bar's realized vol as
        # its volatility estimate, which is what would be known at time T.
        df["vol_ann_lagged"] = df["vol_ann"].shift(1)

        # Active bars = bars with full state to open a trade
        active_mask = df[required].notna().all(axis=1) & df["vol_ann_lagged"].notna()
        self.df = df
        self.active_idx = df.index[active_mask]
        logger.info(
            "MarketDataView[%s..%s]: %d bars total, %d active",
            period[0], period[1], len(df), len(self.active_idx),
        )

        # SVI params — only the rows whose bucket timestamp falls in the period
        svi_params = pd.read_parquet(svi_params_path)
        svi_params = svi_params[(svi_params["timestamp"] >= start - pd.Timedelta(days=2))
                                & (svi_params["timestamp"] <= end + pd.Timedelta(days=2))]
        svi_params["_exp_key"] = pd.to_datetime(svi_params["expiry_dt"], utc=True).dt.strftime("%Y-%m-%d")
        svi_params = svi_params.sort_values(["_exp_key", "timestamp"]).reset_index(drop=True)
        self.svi_params = svi_params

        # Per-expiry arrays for O(log n) lookup inside reset()
        self._svi_by_expiry: dict[str, pd.DataFrame] = {
            k: g.reset_index(drop=True) for k, g in svi_params.groupby("_exp_key", sort=False)
        }

        # VPIN: either the true option-flow VPIN or the realised-vol proxy
        if vpin_series is not None:
            self.vpin = vpin_series.reindex(df.index).fillna(method="ffill").fillna(0.5)
        else:
            self.vpin = compute_vpin_proxy_from_returns(df).reindex(df.index).fillna(0.5)

        # Upsample weights for high-vol regimes
        if high_vol_upsample_threshold > 0.0 and high_vol_upsample_factor > 1.0:
            w = np.ones(len(self.active_idx), dtype=np.float64)
            high_vol_mask = (
                df.loc[self.active_idx, "vol_ann"].values > high_vol_upsample_threshold
            )
            w[high_vol_mask] = high_vol_upsample_factor
            self.sample_weights = w / w.sum()
        else:
            self.sample_weights = None

    def sample_bar(self, rng: np.random.Generator) -> pd.Timestamp:
        if self.sample_weights is None:
            i = rng.integers(0, len(self.active_idx))
        else:
            i = rng.choice(len(self.active_idx), p=self.sample_weights)
        return self.active_idx[i]

    def bar(self, ts: pd.Timestamp) -> pd.Series:
        return self.df.loc[ts]

    def svi_slice_for(
        self,
        ts: pd.Timestamp,
        expiry_dt: pd.Timestamp,
    ) -> _SVISlice | None:
        """Return the SVI slice calibrated at or before ``ts`` for the given expiry."""
        key = pd.Timestamp(expiry_dt).strftime("%Y-%m-%d")
        grp = self._svi_by_expiry.get(key)
        if grp is None or grp.empty:
            return None
        # Binary search over tz-naive numpy datetime array (pandas stores as
        # datetime64[ns, UTC] internally; we convert both sides to plain ns)
        ts_arr = grp["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None).values
        ts_ns = pd.Timestamp(ts).tz_convert("UTC").tz_localize(None).to_numpy()
        pos = np.searchsorted(ts_arr, ts_ns, side="right") - 1
        if pos < 0:
            return None
        row = grp.iloc[pos]
        return _SVISlice(
            a=float(row["a"]), b=float(row["b"]),
            rho=float(row["rho"]), m=float(row["m"]),
            sigma=float(row["sigma"]),
            T=float(row["T"]), F=float(row["F"]),
        )


# ─────────────────────────────────────────────────────────────────────────
# The Gym env
# ─────────────────────────────────────────────────────────────────────────

class GymnasiumHedgingEnv(gym.Env):
    """Delta-hedging environment for short-straddle positions.

    An episode = one straddle trade, held for 15 minutes, 10 intra-bar
    ticks.  The agent decides at each tick what fraction of the required
    delta adjustment to execute on the perp.  Reward is accounting P&L
    minus cost.
    """

    metadata = {"render_modes": []}

    # Observation indices — canonical order
    OBS_DIM = 8
    _IDX_LOG_S    = 0  # log(S_t / S_entry)
    _IDX_TAU_REL  = 1  # tau / tau_entry
    _IDX_DELTA    = 2  # portfolio delta
    _IDX_INV      = 3  # hedge inventory normalised
    _IDX_VPIN     = 4
    _IDX_GAMMA    = 5
    _IDX_TICK     = 6  # tick_index / N
    _IDX_JUMP     = 7  # jump indicator

    def __init__(
        self,
        config: DRLHedgingConfig,
        market_data: MarketDataView,
        *,
        worker_seed: int | None = None,
    ):
        super().__init__()
        self.cfg = config
        self.data = market_data

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(self.OBS_DIM,), dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32,
        )

        base_seed = int(config.seed) if worker_seed is None else int(worker_seed)
        self._rng = np.random.default_rng(base_seed)
        self._trade: _TradeState | None = None
        self._slice: _SVISlice | None = None
        self._path: np.ndarray | None = None
        self._tick: int = 0
        self._H: float = 0.0
        self._V_prev: float = 0.0
        self._delta_theo_entry_abs: float = 1.0  # denominator for inventory norm (safe default)

    # ── Reset ─────────────────────────────────────────────────────────
    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        # Sample an active bar and build the trade
        ts = self.data.sample_bar(self._rng)
        row = self.data.bar(ts)

        expiry = pd.Timestamp(row["expiry_dt"])
        self._slice = self.data.svi_slice_for(ts, expiry)

        # IV at entry: from the SVI slice (fallback to bar's own iv / vol_ann)
        K = float(row["strike"])
        S_entry = float(row["index_price"])
        if self._slice is not None:
            iv_entry = self._slice.iv(K)
            # Guard against pathological slices
            if not np.isfinite(iv_entry) or iv_entry <= 0.0 or iv_entry > 5.0:
                iv_entry = float(row["iv"]) / 100.0 if np.isfinite(row["iv"]) else float(row["vol_ann"])
        else:
            iv_entry = float(row["iv"]) / 100.0 if np.isfinite(row["iv"]) else float(row["vol_ann"])

        tau_entry = float(row["T_years"]) if "T_years" in row and np.isfinite(row["T_years"]) else (
            max((expiry - ts).total_seconds() / (365.25 * 24 * 3600), 1e-6)
        )

        # Sample trade direction: short (-1) by default, or both when
        # sample_signal_at_reset=True.  The SIGNED portfolio delta is
        # signal * delta_long (flips sign between long and short).
        if self.cfg.sample_signal_at_reset:
            signal = int(self._rng.choice([-1, 1]))
        else:
            signal = -1  # legacy: always short
        delta_long_entry = straddle_delta_from_iv(
            S=S_entry, K=K, T=tau_entry, sigma=iv_entry,
        )
        delta_entry = signal * delta_long_entry  # portfolio delta (signed)

        self._trade = _TradeState(
            entry_ts=ts,
            expiry_ts=expiry,
            K=K,
            S_entry=S_entry,
            vol_ann_entry=float(row["vol_ann_lagged"]),
            iv_entry=iv_entry,
            V_bg=float(row["trailing_1h_volume"]),
            jump_fraction=float(row.get("jump_fraction", 0.0) or 0.0),
            vpin=float(self.data.vpin.get(ts, 0.5) if hasattr(self.data.vpin, "get") else self.data.vpin.loc[ts]),
            tau_entry=tau_entry,
            delta_entry=delta_entry,
            signal=signal,
        )

        # Generate the intra-bar GBM price path
        self._path = self._simulate_gbm_path(
            S0=S_entry,
            sigma=self._trade.vol_ann_entry,
            n_steps=self.cfg.n_steps_per_episode,
            dt=self.cfg.dt_years,
        )

        self._tick = 0
        self._H = 0.0

        # Initial theoretical delta (denominator for inventory normalisation)
        # Use a floor so we don't divide by ~0 when the straddle starts perfectly ATM.
        self._delta_theo_entry_abs = max(abs(delta_entry), 0.01)

        # Mark-to-market of the straddle position at tick 0.
        # Sign convention: V = signal * long_mark.
        #   short (-1) → V negative (we owe the premium)
        #   long  (+1) → V positive (we hold the option)
        mark_long_0 = straddle_price_from_iv(
            S=self._path[0], K=K, T=tau_entry, sigma=iv_entry,
        )
        self._V_prev = signal * mark_long_0

        # Observation uses SIGN-FLIPPED components so the policy always
        # sees a "virtual short" view regardless of actual signal (matches
        # the DRLHedger inference-time normalisation).  The flip factor is
        # -signal: +1 for short (no flip), -1 for long (flip).
        # This is a no-op for signal=-1, so the env's legacy behaviour
        # (always-short training) is preserved when
        # ``sample_signal_at_reset=False``.
        flip = -signal
        obs = self._build_obs(
            S=self._path[0], tau=tau_entry,
            delta_theo=flip * delta_entry,   # = -delta_long_entry always
            iv=iv_entry,
            inv=0.0,                          # H=0 at tick 0
            gamma_sign=-1.0,                  # always "virtual short gamma"
        )
        return obs.astype(np.float32), {}

    # ── Step ──────────────────────────────────────────────────────────
    def step(self, action: np.ndarray):
        assert self._trade is not None and self._path is not None, "call reset() first"
        a = float(np.clip(action[0], -1.0, 1.0))

        self._tick += 1
        tick = self._tick
        t = self._trade

        P_t   = float(self._path[tick])
        P_tm1 = float(self._path[tick - 1])
        tau_t = t.tau_at_tick(tick, self.cfg.dt_years)

        # IV from frozen SVI slice at the new spot (k moves with S; slice stays fixed)
        iv_t = self._slice.iv(t.K) if self._slice is not None else t.iv_entry
        if not np.isfinite(iv_t) or iv_t <= 0.0:
            iv_t = t.iv_entry

        # Portfolio delta at tick t in ACTUAL (signed) coordinates
        delta_long_t = straddle_delta_from_iv(S=P_t, K=t.K, T=tau_t, sigma=iv_t)
        delta_theo_t = t.signal * delta_long_t  # +delta_long for long, -delta_long for short

        # Delta we need on the perp to flatten: target_H = -delta_portfolio
        target_H = -delta_theo_t
        delta_required = target_H - self._H

        # Action ∈ [-1, 1] is a fraction of the required adjustment, with
        # sign tracking the required-direction.  +1 = full hedge.
        trade_size = a * abs(delta_required)
        trade_size = trade_size * float(np.sign(delta_required) if delta_required != 0.0 else 1.0)

        # Sigmoid-SRL impact + fees
        impact = float(sigmoid_sqrt_impact(
            Q=abs(trade_size), V=t.V_bg,
            sigma=max(t.vol_ann_entry, 1e-4),
            kappa=self.cfg.kappa,
            sigmoid_alpha=self.cfg.sigmoid_alpha,
            sigmoid_k0=self.cfg.sigmoid_k0,
            impact_floor=self.cfg.impact_floor,
        ))
        cost = abs(trade_size) * P_t * (self.cfg.perp_taker_fee + impact)

        # Residual delta BEFORE the action settles — this is the unhedged
        # exposure that drives the reward's (controllable) P&L.
        H_tm1 = self._H
        residual_pre = delta_theo_t + H_tm1  # net exposure carried into the tick

        # Update inventory after the residual/reward is latched
        self._H = self._H + trade_size

        # ── Reward ────────────────────────────────────────────────────
        if self.cfg.reward_mode == "residual":
            reward = residual_delta_reward(
                delta_portfolio_t=delta_theo_t,
                H_t_minus_1=H_tm1,
                P_t=P_t,
                P_t_minus_1=P_tm1,
                cost=cost,
                reward_scale=self.cfg.reward_scale,
            )
        else:
            # Accounting PnL: reprice option via SVI-frozen slice
            mark_long_t = straddle_price_from_iv(S=P_t, K=t.K, T=tau_t, sigma=iv_t)
            V_t = t.signal * mark_long_t
            reward = accounting_pnl_reward(
                V_t=V_t,
                V_t_minus_1=self._V_prev,
                H_t_minus_1=H_tm1,
                P_t=P_t,
                P_t_minus_1=P_tm1,
                cost=cost,
                reward_scale=self.cfg.reward_scale,
            )
            self._V_prev = V_t

        # Reward clipping — the root-cause defence against pathological
        # states poisoning the dual-critic via outsized squared rewards.
        # Bounds |R| <= reward_clip_abs → bounds R^2 <= reward_clip_abs^2.
        # Set reward_clip_abs=0 to disable.
        reward_clip = getattr(self.cfg, "reward_clip_abs", 0.0)
        if reward_clip > 0.0:
            if reward > reward_clip:
                reward = reward_clip
            elif reward < -reward_clip:
                reward = -reward_clip

        terminated = tick >= self.cfg.n_steps_per_episode
        truncated = False

        # ── Observation in SIGN-FLIPPED (virtual short) convention ───
        # The policy always sees the same short-convention view regardless
        # of actual signal.  At inference, DRLHedger.simulate applies the
        # same flip to maintain the distribution match.
        flip = -t.signal  # +1 for short (no flip), -1 for long (flip)
        obs = self._build_obs(
            S=P_t, tau=tau_t,
            delta_theo=flip * delta_theo_t,       # = -delta_long_t always
            iv=iv_t,
            inv=flip * self._H,                    # signed inventory flipped
            gamma_sign=-1.0,                       # always "virtual short gamma"
        ).astype(np.float32)

        info = {
            "tick": tick,
            "iv": iv_t,
            "cost": cost,
            "H": self._H,
            "delta_theo": delta_theo_t,
            "residual_pre": residual_pre,
            "trade_size": trade_size,
            "impact": impact,
            "signal": t.signal,
        }
        return obs, float(reward), terminated, truncated, info

    # ── Helpers ───────────────────────────────────────────────────────
    def _build_obs(
        self,
        *,
        S: float,
        tau: float,
        delta_theo: float,
        iv: float,
        inv: float | None = None,
        gamma_sign: float = -1.0,
    ) -> np.ndarray:
        """Construct the 8-dim observation vector.

        Parameters
        ----------
        delta_theo : float
            Already-flipped portfolio delta (virtual short convention).
        inv : float, optional
            Already-flipped hedge inventory (virtual short convention).
            If ``None``, fall back to raw ``self._H`` (legacy behaviour
            for the always-short env).
        gamma_sign : float, default -1.0
            Sign to apply to |gamma_long|.  ``-1`` = virtual short
            (env default); ``+1`` only used in research/debugging.
        """
        t = self._trade
        assert t is not None
        log_S = float(np.log(max(S, 1e-12) / max(t.S_entry, 1e-12)))
        tau_rel = float(tau / max(t.tau_entry, 1e-12))

        inv_raw = self._H if inv is None else inv
        inv_norm = float(inv_raw / self._delta_theo_entry_abs)

        gamma_long = straddle_gamma_from_iv(S=S, K=t.K, T=tau, sigma=iv)
        notional = max(S * abs(delta_theo) + 1.0, 1.0)
        gamma_norm = float(gamma_sign * gamma_long * S * S / notional)

        tick_frac = float(self._tick / self.cfg.n_steps_per_episode)
        jump_flag = 1.0 if t.jump_fraction > 0.10 else 0.0

        obs = np.empty(self.OBS_DIM, dtype=np.float64)
        obs[self._IDX_LOG_S]   = log_S
        obs[self._IDX_TAU_REL] = tau_rel
        obs[self._IDX_DELTA]   = float(np.clip(delta_theo, -1.0, 1.0))
        obs[self._IDX_INV]     = float(np.clip(inv_norm, -5.0, 5.0))
        obs[self._IDX_VPIN]    = float(np.clip(t.vpin, 0.0, 1.0))
        obs[self._IDX_GAMMA]   = float(np.clip(gamma_norm, -5.0, 5.0))
        obs[self._IDX_TICK]    = tick_frac
        obs[self._IDX_JUMP]    = jump_flag
        return obs

    def _simulate_gbm_path(
        self, *, S0: float, sigma: float, n_steps: int, dt: float,
    ) -> np.ndarray:
        """Generate a GBM path of length ``n_steps + 1`` starting at S0."""
        drifts = (-0.5 * sigma * sigma) * dt
        vols = sigma * np.sqrt(dt)
        z = self._rng.standard_normal(n_steps)
        log_rets = drifts + vols * z
        log_S = np.concatenate([[np.log(S0)], np.log(S0) + np.cumsum(log_rets)])
        return np.exp(log_S)


# ─────────────────────────────────────────────────────────────────────────
# Backtester adapter — same signature as simulate_discrete_hedging
# ─────────────────────────────────────────────────────────────────────────

class DRLHedger:
    """Runtime adapter: loads a trained SAC policy and exposes it with
    the same signature as ``vol_arb.backtester.simulate_discrete_hedging``.

    The rollout is **fully vectorised across trades**: at each of the 10
    intra-bar ticks, a single batched observation tensor of shape
    ``(N, 8)`` is constructed, the policy is queried in one forward pass
    via ``model.predict(obs, deterministic=True)``, and the resulting
    ``(N, 1)`` action array is applied to all trades simultaneously.

    This gives O(n_steps) neural-net forward passes total — 10 per
    backtest, not 10*N — so training a larger policy later doesn't
    linearly blow up backtest time.

    The DRL hedger uses the sigmoid-adjusted square-root impact model
    (:func:`sigmoid_sqrt_impact`) instead of the rule-based linear bps
    slippage.  Hedge costs are therefore non-linear in trade size.
    """

    def __init__(
        self,
        policy_path: str,
        svi_params_path: str,
        *,
        config: DRLHedgingConfig | None = None,
        vec_normalize_path: str | None = None,
    ):
        """Load a trained SAC-family policy and the SVI surface.

        Parameters
        ----------
        policy_path : str
            Path to the saved .zip policy (SAC or TQC).
        svi_params_path : str
            Path to the SVI surface params parquet.
        config : DRLHedgingConfig, optional
            Override the default DRL config (used for impact kappa, reward
            scale, etc.).
        vec_normalize_path : str, optional
            Path to the saved VecNormalize stats pickle.  If not provided,
            the loader looks for ``best_model_vecnormalize.pkl`` alongside
            the policy file.  If no stats are found, the rollout uses raw
            (unnormalised) observations — this produces correct behaviour
            for policies trained WITHOUT VecNormalize, and garbage output
            for policies trained WITH it, so mismatches are loud.
        """
        path = Path(policy_path)
        if not path.exists():
            raise FileNotFoundError(f"DRL policy not found at {policy_path}")

        self._cfg = config if config is not None else DRLHedgingConfig()

        # Resolve the algorithm type from the config (default SAC).
        # Both SAC and TQC expose the same .predict() API, so the rest of
        # this class doesn't need to branch.
        algo = getattr(self._cfg, "algorithm", "risk_averse_sac")
        if algo == "tqc":
            try:
                from sb3_contrib import TQC  # type: ignore
                self._model = TQC.load(str(path), device="cpu")
            except ImportError:
                raise ImportError(
                    "DRLHedger was asked to load a TQC policy but sb3-contrib "
                    "is not installed.  pip install sb3-contrib"
                )
        else:
            from stable_baselines3 import SAC  # lazy import
            self._model = SAC.load(str(path), device="cpu")

        # ── VecNormalize stats ────────────────────────────────────
        self._vec_normalize = None
        self._obs_mean: np.ndarray | None = None
        self._obs_var: np.ndarray | None = None
        self._obs_clip: float = 10.0

        vn_candidates: list[Path] = []
        if vec_normalize_path is not None:
            vn_candidates.append(Path(vec_normalize_path))
        # Fall back to matching files in the same directory
        vn_candidates.extend([
            path.parent / "best_model_vecnormalize.pkl",
            path.parent / f"{path.stem}_vecnormalize.pkl",
        ])
        for candidate in vn_candidates:
            if candidate.exists():
                # Load only the running stats we need; we don't actually
                # want to construct a full VecNormalize wrapper because it
                # requires a vec_env to attach to.
                import pickle
                with open(candidate, "rb") as f:
                    payload = pickle.load(f)
                # VecNormalize.save() pickles the entire wrapper instance;
                # we extract the ``obs_rms`` (RunningMeanStd) and clip_obs.
                obs_rms = getattr(payload, "obs_rms", None)
                if obs_rms is not None:
                    self._obs_mean = np.asarray(obs_rms.mean, dtype=np.float64)
                    self._obs_var = np.asarray(obs_rms.var, dtype=np.float64)
                    self._obs_clip = float(getattr(payload, "clip_obs", 10.0))
                    logger.info(
                        "DRLHedger loaded VecNormalize stats from %s "
                        "(mean range [%.3f, %.3f], var range [%.3g, %.3g])",
                        candidate.name,
                        float(self._obs_mean.min()), float(self._obs_mean.max()),
                        float(self._obs_var.min()), float(self._obs_var.max()),
                    )
                    break

        # SVI params are pre-loaded once, not per-simulate() call
        svi_params = pd.read_parquet(svi_params_path)
        svi_params["_exp_key"] = pd.to_datetime(
            svi_params["expiry_dt"], utc=True
        ).dt.strftime("%Y-%m-%d")
        self._svi_params = svi_params.sort_values(["_exp_key", "timestamp"]).reset_index(drop=True)
        logger.info(
            "DRLHedger loaded policy %s with %d SVI slices (VecNormalize: %s)",
            path.name, len(self._svi_params),
            "yes" if self._obs_mean is not None else "no",
        )

    def _normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        """Apply VecNormalize's obs normalisation if stats were loaded.

        Matches the exact formula SB3's ``VecNormalize._normalize_obs``
        uses so the distribution at inference matches training.
        """
        if self._obs_mean is None or self._obs_var is None:
            return obs
        epsilon = 1e-8
        normalized = (obs - self._obs_mean) / np.sqrt(self._obs_var + epsilon)
        return np.clip(normalized, -self._obs_clip, self._obs_clip).astype(np.float32)

    def simulate(
        self,
        S_entry: np.ndarray,
        K: np.ndarray,
        sigma: np.ndarray,
        T_years: np.ndarray,
        vol_ann: np.ndarray,
        signal: np.ndarray,
        *,
        n_steps: int = 10,
        delta_hedge_band: float = 0.05,  # unused — kept for signature compat
        perp_fee: float = 0.0004,
        perp_slippage_bps: float = 2.0,  # unused — impact comes from sigmoid SRL
        seed: int = 42,
        vpin: np.ndarray | None = None,
        jump_fraction: np.ndarray | None = None,
        trailing_volume: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Vectorised rollout over ``N`` trades using the trained SAC policy.

        Parameters
        ----------
        S_entry, K, sigma, T_years, vol_ann, signal
            Per-trade arrays of shape ``(N,)`` — same as the rule-based
            ``simulate_discrete_hedging``.
        vpin : ndarray, optional
            Per-trade VPIN values in [0, 1].  Defaults to 0.5 (neutral).
        jump_fraction : ndarray, optional
            Per-trade TSRV jump fraction.  Defaults to 0 (no jump).
        trailing_volume : ndarray, optional
            Per-trade trailing 1h perp/option volume in BTC for the
            sigmoid-SRL impact denominator.  Defaults to 100 BTC.

        Returns
        -------
        (net_delta_pnl, hedge_costs) : tuple of ndarray of shape (N,)
            Per-BTC quantities, matching the rule-based simulator.
        """
        from scipy.stats import norm

        n = len(S_entry)
        rng = np.random.RandomState(seed)
        cfg = self._cfg

        # Active mask — skip trades with signal==0 (no position opened)
        active = signal != 0
        if not active.any():
            return np.zeros(n), np.zeros(n)

        # Optional inputs with safe defaults
        vpin_arr = np.full(n, 0.5) if vpin is None else np.asarray(vpin, dtype=np.float64)
        vpin_arr = np.nan_to_num(vpin_arr, nan=0.5)
        jump_arr = np.zeros(n) if jump_fraction is None else np.asarray(jump_fraction, dtype=np.float64)
        jump_arr = np.nan_to_num(jump_arr, nan=0.0)
        vol_bg = np.full(n, 100.0) if trailing_volume is None else np.asarray(trailing_volume, dtype=np.float64)
        vol_bg = np.nan_to_num(vol_bg, nan=100.0)
        vol_bg = np.maximum(vol_bg, 1e-3)  # floor

        # Time per step in years
        dt_step = (15.0 / n_steps) / (365.25 * 24 * 60)
        sqrt_dt = np.sqrt(dt_step)
        step_sigma = vol_ann * sqrt_dt  # (n,)

        # Random path
        dW = rng.standard_normal((n, n_steps))

        # Precompute BSM constants
        safe_sigma = np.where(sigma > 0, sigma, 0.01)
        safe_T = np.where(T_years > 0, T_years, 0.001)
        sigma_sqrt_T = safe_sigma * np.sqrt(safe_T)
        half_sigma2_T = 0.5 * safe_sigma * safe_sigma * safe_T

        # Initial straddle delta (short convention: delta_portfolio = signal * long_delta)
        d1_0 = (np.log(S_entry / K) + half_sigma2_T) / sigma_sqrt_T
        delta_theo_0 = signal * (2.0 * norm.cdf(d1_0) - 1.0)

        # Normalisation denominator for hedge inventory (matches env)
        delta_theo_entry_abs = np.maximum(np.abs(delta_theo_0), 0.01)

        # State
        S_current = S_entry.copy().astype(np.float64)
        hedge_position = np.zeros(n)
        net_delta_pnl = np.zeros(n)
        hedge_costs = np.zeros(n)

        # Indicator for "jump regime" — static per episode (matches env.reset logic)
        jump_flag = (jump_arr > 0.10).astype(np.float64)

        # Precompute the BSM gamma helper constants (d1 for gamma uses same formula)
        inv_sqrt_2pi = 1.0 / np.sqrt(2.0 * np.pi)

        for j in range(n_steps):
            # 1) Advance price (GBM)
            S_new = S_current * np.exp(step_sigma * dW[:, j] - 0.5 * step_sigma * step_sigma)
            dS = S_new - S_current

            # 2) Build observation for this tick (N x 8 batch)
            tau_t = np.maximum(safe_T - (j + 1) * dt_step, 1e-10)
            tau_rel = tau_t / safe_T

            # Straddle delta at new spot.  Portfolio delta in the TRUE signed
            # coordinates (positive when long, negative when short).
            d1 = (np.log(S_new / K) + half_sigma2_T) / sigma_sqrt_T
            delta_long_t = 2.0 * norm.cdf(d1) - 1.0
            delta_theo_t = signal * delta_long_t  # actual portfolio delta

            log_S = np.log(np.maximum(S_new, 1e-12) / np.maximum(S_entry, 1e-12))

            # Gamma of the actual portfolio (negated relative to long gamma for shorts)
            pdf_d1 = np.exp(-0.5 * d1 * d1) * inv_sqrt_2pi
            gamma_long = 2.0 * pdf_d1 / (S_new * safe_sigma * np.sqrt(np.maximum(tau_t, 1e-10)))
            notional = np.maximum(S_new * np.abs(delta_theo_t) + 1.0, 1.0)

            # ── Sign-flip normalisation: the env trains ONLY on short straddles,
            # so its observations use the "short-convention" encoding:
            #   delta_obs  = -delta_long_t   (always negative of long delta)
            #   inv_obs    = +H / |delta_entry|  (H goes positive while hedging short)
            #   gamma_obs  = -gamma_long * S²/notional  (always negative for short)
            #
            # To present long straddles to the same trained policy, we flip the
            # signed observation components so the policy always sees the same
            # "virtual short" view.  Hedging is sign-invariant under the
            # straddle ↔ -straddle reflection, so the action output stays valid.
            # The ACTUAL delta_required below is computed in the original
            # signed coordinates, so no flip is needed when decoding back to
            # trade_size.
            #
            # Derivation summary (both formulas hold for signal = +1 and -1):
            #   delta_obs = -delta_long_t
            #   inv_obs   = -signal * H / |delta_entry|
            #               (for short: +H/|..|; for long: -H/|..|, and H is
            #                negative for long hedging, so -H is positive)
            #   gamma_obs = -gamma_long * S² / notional  (always negative)
            delta_obs = -delta_long_t
            inv_obs = (-signal) * hedge_position / delta_theo_entry_abs
            gamma_obs = -gamma_long * S_new * S_new / notional

            tick_frac = np.full(n, (j + 1) / n_steps)

            obs_batch = np.stack([
                log_S.astype(np.float32),
                np.clip(tau_rel, 0.0, 5.0).astype(np.float32),
                np.clip(delta_obs, -1.0, 1.0).astype(np.float32),
                np.clip(inv_obs, -5.0, 5.0).astype(np.float32),
                np.clip(vpin_arr, 0.0, 1.0).astype(np.float32),
                np.clip(gamma_obs, -5.0, 5.0).astype(np.float32),
                tick_frac.astype(np.float32),
                jump_flag.astype(np.float32),
            ], axis=1)  # shape (n, 8)

            # Apply VecNormalize obs stats if the policy was trained with it
            obs_batch = self._normalize_obs(obs_batch)

            # 3) Policy forward pass — one batched call for all trades
            action_batch, _ = self._model.predict(obs_batch, deterministic=True)
            a = np.clip(action_batch.reshape(-1), -1.0, 1.0)

            # 4) Residual delta in the ACTUAL coordinate system
            target_H = -delta_theo_t
            delta_required = target_H - hedge_position

            # Agent decides fraction; sign tracks residual's sign.
            # Because obs_sign flipped the observation for long straddles, the
            # policy's output is in "virtual short" coordinates.  We flip back
            # so the trade size goes in the right direction in actual coords.
            trade_size = a * np.abs(delta_required) * np.where(delta_required >= 0, 1.0, -1.0)
            trade_size = np.where(active, trade_size, 0.0)

            # For the P&L accumulation we use the TRUE residual (portfolio
            # delta minus hedge inventory), exactly matching the rule-based
            # simulator's convention.
            residual = delta_theo_t - hedge_position

            # 5) Accumulate pre-trade residual P&L (matches rule-based convention)
            net_delta_pnl += np.where(active, residual * dS, 0.0)

            # 6) Cost: sigmoid-SRL impact + taker fee, per BTC of trade
            impact = sigmoid_sqrt_impact(
                Q=np.abs(trade_size),
                V=vol_bg,
                sigma=np.maximum(vol_ann, 1e-4),
                kappa=cfg.kappa,
                sigmoid_alpha=cfg.sigmoid_alpha,
                sigmoid_k0=cfg.sigmoid_k0,
                impact_floor=cfg.impact_floor,
            )
            cost_rate = perp_fee + impact  # decimal fraction of underlying
            hedge_costs += np.abs(trade_size) * S_new * cost_rate

            # 7) Update hedge position and spot
            hedge_position += trade_size
            S_current = S_new

        return net_delta_pnl, hedge_costs
