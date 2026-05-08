"""Configuration for the DRL delta-hedging environment and SAC training.

Single source of truth for all hyperparameters.  Everything the env and
trainer need is here — no magic numbers scattered across modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field


# ── Default reward-shaping: no per-step variance penalty; the variance term is
# injected via the RiskAverseSAC critic, not the reward itself.  Set
# LAMBDA_REWARD_VAR > 0 to use the reward-shaping fallback instead.
LAMBDA_REWARD_VAR: float = 0.0


def derive_dt_years(bar_minutes: float, n_steps: int) -> float:
    """D-13 (parity-r3 cycle 8 — 2026-05-08): single source of truth
    for ``dt_years = (bar_minutes / 60 / 24 / 365.25) / n_steps``.

    Pre-cycle-8 the formula was duplicated in:
    - ``DRLHedgingConfig.__post_init__`` (cfg-construction time)
    - ``LiveDRLHedger.step`` (intra-bar fallback when ``dt_years``
      isn't passed)

    The env reads from ``cfg.dt_years`` directly. The 3-way fanout
    risked silent drift if any caller forgot to update one of the
    sites. Centralising here so all three paths converge.
    """
    if bar_minutes <= 0:
        raise ValueError(f"bar_minutes must be positive, got {bar_minutes}")
    if n_steps <= 0:
        raise ValueError(f"n_steps must be positive, got {n_steps}")
    return (bar_minutes / 60.0 / 24.0 / 365.25) / n_steps


@dataclass
class DRLHedgingConfig:
    """All knobs for the DRL hedging env + SAC trainer.

    Grouped by concern — env physics, impact model, reward, SAC, data splits.
    """

    # ── Env physics ────────────────────────────────────────────────────
    # Bar timeframe in minutes. Used to derive dt_years.
    bar_minutes: int = 15
    # Number of intra-bar hedging ticks inside one bar window.  Matches
    # the existing ``simulate_discrete_hedging(n_steps=10)`` so we can A/B.
    n_steps_per_episode: int = 10
    # Delta-t between ticks, in years.  Derived from bar_minutes /
    # n_steps_per_episode in __post_init__ if left at sentinel (-1).
    dt_years: float = -1.0
    # GBM path generator seed (base — each reset() derives a worker-specific seed)
    seed: int = 42
    # Upsample high-vol regimes during episode sampling so the agent sees
    # jump conditions more often than their natural frequency.  Null disables.
    high_vol_upsample_threshold: float = 0.80  # vol_ann decimal
    high_vol_upsample_factor: float = 3.0

    # ── Impact model ───────────────────────────────────────────────────
    # Liquidity constant — calibrated via impact.calibrate_kappa() to
    # reproduce the existing 2-bps-at-2%-participation linear baseline
    # (evaluates to ~1.768e-3 with default alpha/k0).
    kappa: float = 1.768e-3
    # Sigmoid steepness — empirical monotonicity bound is ~430 at k0=0.005
    # (see impact.py comment).  Default 400 gives 1.88x saturation at p→0
    # and passes strict monotonicity tests on a 2000-point grid.
    sigmoid_alpha: float = 400.0
    # Sigmoid knee (participation rate).  Below this, the correction factor
    # boosts impact to capture the fixed-spread floor crypto perps exhibit.
    sigmoid_k0: float = 0.005
    # Numerical floor for background volume to avoid div-by-zero
    impact_floor: float = 1e-8
    # Fee charged by the venue on perp trades (Deribit/Bybit taker ~ 4 bps)
    perp_taker_fee: float = 0.0004

    # ── VPIN ───────────────────────────────────────────────────────────
    vpin_bucket_volume_btc: float = 50.0  # fixed volume per bucket (base units)
    vpin_rolling_window: int = 50          # n-bucket rolling window

    # ── Reward shaping ────────────────────────────────────────────────
    # Reward mode: "residual" (Deep Hedging objective — recommended)
    # or "accounting" (full mark-to-market — benchmark only).
    reward_mode: str = "residual"
    # Divide raw reward by this to bring scale to O(1).  Critical for
    # RiskAverseSAC: without scaling, the squared-reward critic chases
    # magnitudes in the 10^6-10^8 range and the variance penalty drowns
    # the actor loss.
    reward_scale: float = 100.0
    # Clip the SCALED reward to [-reward_clip_abs, +reward_clip_abs] as a
    # root-cause defence against pathological states generating outsized
    # squared-reward targets (which poisoned the 300k run).  5.0 = ~5 std
    # of the healthy reward distribution.  0 disables clipping.
    reward_clip_abs: float = 5.0
    # Std-penalty weight in RiskAverseSAC actor loss (mean-variance mix).
    # Used AFTER the warmup+rampup schedule below completes.
    lambda_risk: float = 0.3
    # Steps of pure mean-Q SAC (no variance penalty) before lambda starts
    # ramping up.  Lets the mean critic stabilise before the second
    # critic's noisy estimates enter the actor loss.
    lambda_risk_warmup_steps: int = 100_000
    # Steps over which lambda_risk ramps linearly from 0 → lambda_risk.
    lambda_risk_rampup_steps: int = 200_000

    # ── Signal sampling in env.reset ──────────────────────────────────
    # If True, sample signal ∈ {-1, +1} uniformly at each reset so the
    # policy trains on both long and short straddles.  If False, every
    # episode is a short straddle (legacy behaviour; kept for
    # reproducibility of early smoke runs).
    sample_signal_at_reset: bool = True

    # ── SAC hyperparameters ───────────────────────────────────────────
    learning_rate: float = 3e-4
    # Linear LR decay target.  Set equal to learning_rate to disable.
    learning_rate_final: float = 3e-5
    buffer_size: int = 1_000_000
    # Larger batch amortises GPU kernel-launch overhead.  512 is safe
    # with the default (256, 256) arch and fits easily in any modern GPU.
    batch_size: int = 512
    tau: float = 0.005             # target network soft-update rate
    gamma: float = 0.99
    train_freq: int = 1
    # 4 gradient updates per env step — the single biggest GPU-util lever.
    # Each transition gets replayed 4x more before eviction, improving
    # sample efficiency AND GPU utilisation simultaneously.
    gradient_steps: int = 4
    # FIXED entropy coefficient (no auto-tuning).  The previous auto-tuned
    # runs collapsed ent_coef to ~0.04 — exploration died → divergence.
    # Fixed 0.2 is the Deep Hedging / finance-RL community standard for
    # noisy-reward environments where the auto-tuner can't find a stable
    # equilibrium.  Can be overridden to "auto" for ablation studies.
    ent_coef: float | str = 0.2
    # Only used when ent_coef="auto".  Left at -1.0 = -dim(action) for SB3
    # compatibility if someone opts into auto-tuning.
    target_entropy: float = -1.0
    # Clip gradient L2 norm in actor + twin critic + critic_sq optimisers.
    # 0.5 is the literature-standard value for SAC/TD3.  0 disables.
    max_grad_norm: float = 0.5

    # Policy network architecture
    net_arch: tuple[int, ...] = (256, 256)

    # ── Vec-env wrapping ──────────────────────────────────────────────
    # VecNormalize applies running-mean/std normalisation to observations.
    # Required because the 8-dim obs has components on wildly different
    # scales (log_S ~ 1e-2 vs inv_norm ~ 5).  Running stats saved as
    # "<tag>_vecnormalize.pkl" alongside the policy for inference-time use.
    use_vec_normalize: bool = True
    # Use SubprocVecEnv (multiprocess) over DummyVecEnv (single-process)
    # when n_parallel_envs > 1.  ~linear speedup up to physical core count.
    use_subproc_env: bool = True
    # Keep the replay buffer in VRAM to eliminate CPU↔GPU transfer per batch.
    # ~1 GB VRAM cost for a 1M-buffer, 8-dim obs, 1-dim action.
    replay_buffer_on_gpu: bool = True
    # torch.compile(model.policy) — PyTorch 2+ JIT, ~20-30% fwd/bwd speedup.
    # Stretch goal, off by default because SB3 2.8 compatibility is unvalidated.
    torch_compile: bool = False

    # ── Training schedule ─────────────────────────────────────────────
    total_timesteps: int = 5_000_000
    n_parallel_envs: int = 8
    eval_freq: int = 50_000
    save_freq: int = 100_000
    log_interval: int = 100
    n_eval_episodes: int = 100

    # ── Algorithm selection ───────────────────────────────────────────
    # "risk_averse_sac" (default) or "tqc" (requires sb3-contrib).
    # TQC uses distributional quantile critics instead of dual Q, and is
    # the modern alternative to our dual-Q mean-variance approach if the
    # hardening fails to stabilise RiskAverseSAC.
    algorithm: str = "risk_averse_sac"
    tqc_top_quantiles_drop: int = 5  # 5 = moderate risk aversion; 10+ = strong

    # ── Data splits (inclusive start, inclusive end) ──────────────────
    train_period: tuple[str, str] = ("2021-01-01", "2023-12-31")
    eval_period:  tuple[str, str] = ("2024-01-01", "2024-12-31")
    test_period:  tuple[str, str] = ("2025-01-01", "2026-03-31")

    # ── Asset / currency ─────────────────────────────────────────────
    asset: str = "BTCUSDT"
    # `currency` is derived from asset if left empty; only set explicitly
    # for overrides. Used to resolve per-currency artifact paths below.
    currency: str = ""

    # ── Paths (auto-resolved from asset via vol_arb.paths in __post_init__
    # when left empty — so BTC produces legacy unsuffixed paths and ETH
    # gets the {currency}_ / {asset}_ prefixed variants automatically) ──
    vol_arb_path: str = ""
    svi_params_path: str = ""
    option_trades_dir: str = ""
    vpin_path: str = ""
    model_dir: str = "vol_arb/drl_hedging/models"
    tb_log_dir: str = "vol_arb/drl_hedging/tb_logs"
    eval_log_dir: str = "vol_arb/drl_hedging/eval"

    def __post_init__(self) -> None:
        if self.dt_years < 0:
            self.dt_years = derive_dt_years(
                self.bar_minutes, self.n_steps_per_episode,
            )
        # Resolve per-asset artifact paths. Import here to avoid a circular
        # dependency at module import time (vol_arb.paths → no heavy deps,
        # but keep it local for lazy cost).
        from vol_arb.paths import (
            dvol_path, option_trade_dir, svi_surface_params_path,
            vol_arb_path, vpin_path, CURRENCY_FOR_ASSET,
        )
        if not self.currency:
            self.currency = CURRENCY_FOR_ASSET[self.asset]
        # vol_arb_path — use the non-SVI variant here to match the historical
        # BTC default (``BTC_vol_arb.parquet``); override externally if the
        # DRL env should consume the SVI variant.
        if not self.vol_arb_path:
            self.vol_arb_path = str(vol_arb_path(self.currency, svi=False))
        if not self.svi_params_path:
            self.svi_params_path = str(svi_surface_params_path(self.currency))
        if not self.option_trades_dir:
            self.option_trades_dir = str(option_trade_dir(self.currency))
        if not self.vpin_path:
            self.vpin_path = str(vpin_path(self.currency))
