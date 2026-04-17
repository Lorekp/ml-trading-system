"""Training entry point for the risk-averse SAC hedging agent.

Usage::

    PYTHONPATH=. python vol_arb/drl_hedging/train_sac.py
    PYTHONPATH=. python vol_arb/drl_hedging/train_sac.py --total-timesteps 300000 --n-envs 8
    PYTHONPATH=. python vol_arb/drl_hedging/train_sac.py --algorithm tqc --tag tqc_300k
    PYTHONPATH=. python vol_arb/drl_hedging/train_sac.py --no-vec-normalize --no-grad-clip

The script mirrors the layout the rest of the vol-arb CLI entry points
use (argparse → build config → instantiate → run).  It writes:

- TensorBoard logs to ``vol_arb/drl_hedging/tb_logs/``
- Best / final policies to ``vol_arb/drl_hedging/models/``
- VecNormalize stats to ``<model_dir>/<tag>_vecnormalize.pkl``
- Per-eval metrics to ``vol_arb/drl_hedging/eval/``

All training stability, speed and quality features from the hardening
plan are wired here.  See ``DRLHedgingConfig`` for the knobs.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from vol_arb.drl_hedging.config import DRLHedgingConfig
from vol_arb.drl_hedging.env import GymnasiumHedgingEnv, MarketDataView

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_sac")


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────

def _make_env_factory(
    cfg: DRLHedgingConfig,
    period: tuple[str, str],
    worker_seed: int,
    vpin_series: pd.Series | None = None,
):
    """Return a callable that builds a fresh env — needed for SubprocVecEnv.

    If ``vpin_series`` is given, the MarketDataView uses it as the VPIN
    source (W5 true-VPIN path). Otherwise it falls back to the returns-
    based proxy (legacy v3 training path).
    """
    def _factory():
        data = MarketDataView(
            vol_arb_path=cfg.vol_arb_path,
            svi_params_path=cfg.svi_params_path,
            period=period,
            high_vol_upsample_threshold=cfg.high_vol_upsample_threshold,
            high_vol_upsample_factor=cfg.high_vol_upsample_factor,
            vpin_series=vpin_series,
        )
        return GymnasiumHedgingEnv(cfg, data, worker_seed=worker_seed)
    return _factory


def _load_vpin_series(
    parquet_path: str, column: str,
) -> pd.Series:
    """Load a VPIN column from a parquet into a DatetimeIndex-aligned Series."""
    df = pd.read_parquet(parquet_path)
    if column not in df.columns:
        raise KeyError(
            f"VPIN parquet {parquet_path} has no column '{column}'. "
            f"Available: {list(df.columns)}",
        )
    s = df[column].astype(float)
    s.name = "vpin"
    logger.info(
        "Loaded VPIN series from %s col=%s: %d rows, mean=%.3f, "
        "p5/p50/p95=%.3f/%.3f/%.3f",
        parquet_path, column, len(s), float(s.mean()),
        float(s.quantile(0.05)), float(s.quantile(0.50)),
        float(s.quantile(0.95)),
    )
    return s


def linear_schedule(initial: float, final: float) -> Callable[[float], float]:
    """SB3-compatible linear learning-rate schedule.

    SB3 passes ``progress_remaining`` ∈ [1.0, 0.0] (1 at start, 0 at end).
    """
    def _schedule(progress_remaining: float) -> float:
        return final + progress_remaining * (initial - final)
    return _schedule


# ─────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train the risk-averse SAC hedging agent")
    parser.add_argument("--asset", default="BTCUSDT",
                        help="Training asset (BTCUSDT or ETHUSDT). Flows through "
                             "DRLHedgingConfig to resolve per-asset data paths + "
                             "save dir via vol_arb.paths.")
    parser.add_argument("--total-timesteps", type=int, default=None,
                        help="Override DRLHedgingConfig.total_timesteps")
    parser.add_argument("--n-envs", type=int, default=None,
                        help="Override DRLHedgingConfig.n_parallel_envs")
    parser.add_argument("--lambda-risk", type=float, default=None,
                        help="Override DRLHedgingConfig.lambda_risk (target)")
    parser.add_argument("--vol-arb-path", type=str, default=None)
    parser.add_argument("--svi-params-path", type=str, default=None)
    parser.add_argument("--tag", type=str, default="",
                        help="Name suffix for the saved model/logs "
                             "(default: sac_hedger for BTC, sac_hedger_{ASSET} otherwise)")
    parser.add_argument("--algorithm", type=str, default=None,
                        choices=["risk_averse_sac", "tqc"],
                        help="Algorithm override (default from config: risk_averse_sac)")
    parser.add_argument("--target-entropy", type=float, default=None,
                        help="Override target_entropy (default -0.5)")
    parser.add_argument("--use-subproc", action="store_true",
                        help="Force SubprocVecEnv (default: from config)")
    parser.add_argument("--no-vec-normalize", action="store_true",
                        help="Disable VecNormalize obs wrapper")
    parser.add_argument("--no-grad-clip", action="store_true",
                        help="Disable gradient clipping (sets max_grad_norm=0)")
    parser.add_argument("--no-lambda-warmup", action="store_true",
                        help="Disable lambda_risk warmup+rampup schedule")
    parser.add_argument("--no-gpu-buffer", action="store_true",
                        help="Keep replay buffer on CPU instead of GPU")
    parser.add_argument("--torch-compile", action="store_true",
                        help="Enable torch.compile on the policy (experimental)")
    parser.add_argument("--vpin-parquet", type=str, default="",
                        help=(
                            "Optional parquet with a precomputed VPIN column "
                            "(W5 true-VPIN path). Default: empty → use the "
                            "legacy returns-based proxy from vpin.py."
                        ))
    parser.add_argument("--vpin-column", type=str, default="vpin_true",
                        help="Column to read from --vpin-parquet (default vpin_true)")
    args = parser.parse_args()

    cfg = DRLHedgingConfig(asset=args.asset)

    # Derive per-asset default tag so BTC keeps the legacy "sac_hedger"
    # directory and ETH/other assets land in "sac_hedger_ETHUSDT" etc.
    if not args.tag:
        args.tag = "sac_hedger" if args.asset == "BTCUSDT" else f"sac_hedger_{args.asset}"

    # Apply CLI overrides
    if args.total_timesteps is not None:
        cfg.total_timesteps = int(args.total_timesteps)
    if args.n_envs is not None:
        cfg.n_parallel_envs = int(args.n_envs)
    if args.lambda_risk is not None:
        cfg.lambda_risk = float(args.lambda_risk)
    if args.vol_arb_path is not None:
        cfg.vol_arb_path = args.vol_arb_path
    if args.svi_params_path is not None:
        cfg.svi_params_path = args.svi_params_path
    if args.algorithm is not None:
        cfg.algorithm = args.algorithm
    if args.target_entropy is not None:
        cfg.target_entropy = float(args.target_entropy)
    if args.use_subproc:
        cfg.use_subproc_env = True
    if args.no_vec_normalize:
        cfg.use_vec_normalize = False
    if args.no_grad_clip:
        cfg.max_grad_norm = 0.0
    if args.no_lambda_warmup:
        cfg.lambda_risk_warmup_steps = 0
        cfg.lambda_risk_rampup_steps = 0
    if args.no_gpu_buffer:
        cfg.replay_buffer_on_gpu = False
    if args.torch_compile:
        cfg.torch_compile = True

    # Defer heavy SB3 imports so `--help` doesn't pull torch
    from stable_baselines3.common.vec_env import (
        DummyVecEnv, SubprocVecEnv, VecNormalize,
    )
    from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback
    import torch

    Path(cfg.model_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.tb_log_dir).mkdir(parents=True, exist_ok=True)
    Path(cfg.eval_log_dir).mkdir(parents=True, exist_ok=True)

    # ── Load VPIN source once (if provided, or auto-discover per asset) ──
    vpin_series: pd.Series | None = None
    if not args.vpin_parquet:
        # Fall back to the resolver's canonical per-asset path, which the
        # recompute_vpin.py writer uses. BTC stays at vpin_btc_15m.parquet.
        default_vpin = cfg.vpin_path
        if Path(default_vpin).exists():
            args.vpin_parquet = default_vpin
    if args.vpin_parquet:
        vpin_series = _load_vpin_series(args.vpin_parquet, args.vpin_column)

    # ── Build training vec env ─────────────────────────────────────
    logger.info(
        "Building %s with %d parallel envs (vpin_source=%s)",
        "SubprocVecEnv" if cfg.use_subproc_env else "DummyVecEnv",
        cfg.n_parallel_envs,
        (f"parquet:{args.vpin_parquet}[{args.vpin_column}]"
         if vpin_series is not None else "proxy"),
    )
    factories = [
        _make_env_factory(
            cfg, cfg.train_period, worker_seed=cfg.seed + i,
            vpin_series=vpin_series,
        )
        for i in range(cfg.n_parallel_envs)
    ]
    VecEnvCls = SubprocVecEnv if cfg.use_subproc_env and cfg.n_parallel_envs > 1 else DummyVecEnv
    vec_env = VecEnvCls(factories)

    if cfg.use_vec_normalize:
        logger.info("Wrapping with VecNormalize (obs only — rewards already scaled in env)")
        vec_env = VecNormalize(
            vec_env, norm_obs=True, norm_reward=False, clip_obs=10.0,
        )

    # ── Build eval env (shares VecNormalize stats) ─────────────────
    logger.info("Building eval env on %s", cfg.eval_period)
    eval_env_raw = DummyVecEnv([
        _make_env_factory(
            cfg, cfg.eval_period, worker_seed=cfg.seed + 10_000,
            vpin_series=vpin_series,
        ),
    ])
    if cfg.use_vec_normalize:
        eval_env = VecNormalize(
            eval_env_raw, training=False, norm_obs=True, norm_reward=False,
            clip_obs=10.0,
        )
        # Share the running stats from the training env so eval uses the
        # same normalisation as training.  Must sync these at each eval
        # callback fire — SB3's EvalCallback does this automatically if
        # we set `eval_env.obs_rms = vec_env.obs_rms` before the callback.
        eval_env.obs_rms = vec_env.obs_rms
    else:
        eval_env = eval_env_raw

    # ── Learning rate schedule ────────────────────────────────────
    if cfg.learning_rate_final != cfg.learning_rate:
        lr_arg = linear_schedule(cfg.learning_rate, cfg.learning_rate_final)
        logger.info("LR schedule: linear %.1e → %.1e", cfg.learning_rate, cfg.learning_rate_final)
    else:
        lr_arg = cfg.learning_rate

    # ── Replay buffer location ────────────────────────────────────
    replay_buffer_kwargs: dict | None = None
    device = "auto"
    if cfg.replay_buffer_on_gpu and torch.cuda.is_available():
        logger.info("Replay buffer on GPU (VRAM-resident, no CPU↔GPU transfer per batch)")
        device = "cuda"
    elif cfg.replay_buffer_on_gpu:
        logger.warning("replay_buffer_on_gpu=True but CUDA unavailable; falling back to CPU")

    # ── Build model ────────────────────────────────────────────────
    logger.info(
        "Building %s (lambda_risk target=%.2f, warmup=%d, rampup=%d, grad_clip=%.2f)",
        cfg.algorithm, cfg.lambda_risk,
        cfg.lambda_risk_warmup_steps, cfg.lambda_risk_rampup_steps,
        cfg.max_grad_norm,
    )

    common_kwargs = dict(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=lr_arg,
        buffer_size=cfg.buffer_size,
        batch_size=cfg.batch_size,
        tau=cfg.tau,
        gamma=cfg.gamma,
        train_freq=cfg.train_freq,
        gradient_steps=cfg.gradient_steps,
        ent_coef=cfg.ent_coef,
        target_entropy=cfg.target_entropy,
        device=device,
        policy_kwargs=dict(
            net_arch=list(cfg.net_arch),
            activation_fn=torch.nn.ReLU,
        ),
        tensorboard_log=cfg.tb_log_dir,
        verbose=1,
    )

    if cfg.algorithm == "risk_averse_sac":
        from vol_arb.drl_hedging.sac_risk import RiskAverseSAC
        model = RiskAverseSAC(
            **common_kwargs,
            lambda_risk=cfg.lambda_risk,
            lambda_risk_warmup_steps=cfg.lambda_risk_warmup_steps,
            lambda_risk_rampup_steps=cfg.lambda_risk_rampup_steps,
            max_grad_norm=cfg.max_grad_norm,
        )
    elif cfg.algorithm == "tqc":
        try:
            from sb3_contrib import TQC  # lazy; only required for tqc path
        except ImportError as exc:
            raise ImportError(
                "algorithm='tqc' requires sb3-contrib.  Install with:\n"
                "    pip install sb3-contrib"
            ) from exc
        logger.info("TQC top-quantiles-to-drop: %d", cfg.tqc_top_quantiles_drop)
        model = TQC(
            **common_kwargs,
            top_quantiles_to_drop_per_net=cfg.tqc_top_quantiles_drop,
        )
    else:
        raise ValueError(f"Unknown algorithm: {cfg.algorithm}")

    # ── Optional torch.compile ─────────────────────────────────────
    if cfg.torch_compile:
        try:
            logger.info("Enabling torch.compile on model.policy (experimental)")
            model.policy = torch.compile(model.policy)
        except Exception as exc:
            logger.warning("torch.compile failed (%s) — continuing without", exc)

    # ── Callbacks ──────────────────────────────────────────────────
    # Tag-specific subdirectory so parallel runs don't overwrite each
    # other's best_model.zip / checkpoints.  Each training run writes to
    # ``<model_dir>/<tag>/`` and the caller (or the backtester) references
    # e.g. ``vol_arb/drl_hedging/models/sac_v3_300k/best_model.zip``.
    run_dir = Path(cfg.model_dir) / args.tag
    run_dir.mkdir(parents=True, exist_ok=True)
    run_eval_dir = Path(cfg.eval_log_dir) / args.tag
    run_eval_dir.mkdir(parents=True, exist_ok=True)

    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(run_dir),
        log_path=str(run_eval_dir),
        eval_freq=cfg.eval_freq,
        n_eval_episodes=cfg.n_eval_episodes,
        deterministic=True,
    )
    ckpt_cb = CheckpointCallback(
        save_freq=cfg.save_freq,
        save_path=str(run_dir),
        name_prefix=args.tag,
    )

    # ── Train ──────────────────────────────────────────────────────
    logger.info("Starting training for %d timesteps", cfg.total_timesteps)
    model.learn(
        total_timesteps=cfg.total_timesteps,
        callback=[eval_cb, ckpt_cb],
        log_interval=cfg.log_interval,
        tb_log_name=args.tag,
    )

    final_path = run_dir / f"{args.tag}_final.zip"
    model.save(str(final_path))
    logger.info("Saved final model to %s", final_path)

    # Save VecNormalize stats alongside the models so DRLHedger can load
    # them at inference.  Both ``<tag>_vecnormalize.pkl`` (matches the
    # final model) and ``best_model_vecnormalize.pkl`` (matches the
    # EvalCallback best_model.zip) live in the tag-specific directory.
    if cfg.use_vec_normalize:
        vec_path = run_dir / f"{args.tag}_vecnormalize.pkl"
        vec_env.save(str(vec_path))
        logger.info("Saved VecNormalize stats to %s", vec_path)
        best_vec_path = run_dir / "best_model_vecnormalize.pkl"
        vec_env.save(str(best_vec_path))
        logger.info("Saved VecNormalize stats to %s", best_vec_path)


if __name__ == "__main__":
    main()
