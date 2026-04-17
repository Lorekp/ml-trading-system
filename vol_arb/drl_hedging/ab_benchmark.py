"""Paired A/B benchmark: v3 (proxy VPIN) vs v4 (true VPIN) SAC hedger.

Runs each policy on its own environment (v3 with the returns-based VPIN
proxy, v4 with the true bucketed VPIN loaded from parquet) over the same
held-out period, with matched RNG seeds so episode i samples the same
underlying vol_arb bar in both environments. Then reports:

- Per-policy: mean reward, std reward, mean cost, std cost, Sortino,
  CVaR(5%), bars-hedged per episode.
- Paired diff (v4 - v3) for reward, cost, std of each, with a
  bootstrap 90% CI on the mean difference.

Adoption rule (from the W5 plan in
``C:\\Users\\laure\\.claude\\plans\\flickering-wandering-locket.md``):

    Adopt v4 only if Sortino improves AND hedge-cost std does not
    regress by more than 10%.

This script is deliberately simple — no CLI sweep over seeds, no
regime cells — because the primary question is "does swapping the
state feature give the policy a strictly better hedge trajectory?".
For multi-regime breakdown, run with multiple ``--period`` values.

Usage::

    PYTHONPATH=. python vol_arb/drl_hedging/ab_benchmark.py \\
        --v3-policy vol_arb/drl_hedging/models/production_sac_v3_final.zip \\
        --v3-vecnorm vol_arb/drl_hedging/models/production_sac_v3_final_vecnormalize.pkl \\
        --v4-policy vol_arb/drl_hedging/models/sac_v4_true_vpin/sac_v4_true_vpin_final.zip \\
        --v4-vecnorm vol_arb/drl_hedging/models/sac_v4_true_vpin/sac_v4_true_vpin_vecnormalize.pkl \\
        --vpin-parquet vol_arb/output/vpin_btc_15m.parquet \\
        --n-episodes 500
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from vol_arb.drl_hedging.config import DRLHedgingConfig
from vol_arb.drl_hedging.env import GymnasiumHedgingEnv, MarketDataView

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("ab_benchmark")


@dataclass
class PolicyStats:
    name: str
    n_episodes: int
    mean_reward: float
    std_reward: float
    median_reward: float
    mean_cost: float
    std_cost: float
    sortino: float
    cvar_5pct: float

    def as_dict(self) -> dict:
        return asdict(self)


def _compute_stats(
    name: str, rewards: np.ndarray, costs: np.ndarray,
) -> PolicyStats:
    rewards = np.asarray(rewards, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    mean_r = float(rewards.mean())
    std_r = float(rewards.std(ddof=1)) if len(rewards) > 1 else 0.0
    downside = rewards[rewards < 0]
    down_std = (
        float(np.sqrt(np.mean(downside * downside)))
        if len(downside) > 0 else 0.0
    )
    sortino = mean_r / down_std if down_std > 0 else float("nan")
    n_tail = max(1, int(0.05 * len(rewards)))
    cvar_5 = float(np.mean(np.sort(rewards)[:n_tail]))
    return PolicyStats(
        name=name,
        n_episodes=len(rewards),
        mean_reward=mean_r,
        std_reward=std_r,
        median_reward=float(np.median(rewards)),
        mean_cost=float(costs.mean()),
        std_cost=float(costs.std(ddof=1)) if len(costs) > 1 else 0.0,
        sortino=sortino,
        cvar_5pct=cvar_5,
    )


def _paired_bootstrap_ci(
    diff: np.ndarray,
    n_draws: int = 10_000,
    seed: int = 7,
) -> tuple[float, float, float]:
    """90% bootstrap CI on the mean of the paired difference series.

    Returns ``(mean, p5, p95)``.
    """
    n = len(diff)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_draws, n))
    samples = diff[idx].mean(axis=1)
    return (
        float(diff.mean()),
        float(np.percentile(samples, 5)),
        float(np.percentile(samples, 95)),
    )


def _load_vpin_series(parquet_path: str, column: str) -> pd.Series:
    df = pd.read_parquet(parquet_path)
    if column not in df.columns:
        raise KeyError(
            f"{parquet_path} has no column '{column}'; "
            f"available: {list(df.columns)}",
        )
    s = df[column].astype(float)
    s.name = "vpin"
    return s


def _build_env(
    cfg: DRLHedgingConfig,
    period: tuple[str, str],
    vpin_series: pd.Series | None,
    seed: int,
) -> GymnasiumHedgingEnv:
    data = MarketDataView(
        vol_arb_path=cfg.vol_arb_path,
        svi_params_path=cfg.svi_params_path,
        period=period,
        high_vol_upsample_threshold=0.0,  # no upsampling during eval
        high_vol_upsample_factor=1.0,
        vpin_series=vpin_series,
    )
    return GymnasiumHedgingEnv(cfg, data, worker_seed=seed)


def _rollout(env: GymnasiumHedgingEnv, predict_fn) -> tuple[float, float]:
    obs, _ = env.reset()
    total_reward = 0.0
    total_cost = 0.0
    done = False
    while not done:
        action = predict_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        total_cost += float(info.get("cost", 0.0))
        done = bool(terminated) or bool(truncated)
    return total_reward, total_cost


def _load_sac_policy(policy_path: str, vecnorm_path: str | None):
    """Load a SAC policy + optional VecNormalize stats.

    Returns a callable (obs) -> action that applies VecNormalize obs
    normalisation if vecnorm_path is provided.
    """
    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    import gymnasium as gym

    model = SAC.load(policy_path, device="cpu")

    vec_norm: VecNormalize | None = None
    if vecnorm_path:
        # VecNormalize needs a real vec env to load, but we only use its
        # obs_rms, so we build a one-shot dummy wrapper around a trivial
        # env with the same observation space as the policy.
        dummy_space_env = DummyVecEnv([
            lambda: _NullObsEnv(model.observation_space, model.action_space),
        ])
        vec_norm = VecNormalize.load(vecnorm_path, dummy_space_env)
        vec_norm.training = False
        vec_norm.norm_reward = False

    def predict(obs: np.ndarray) -> np.ndarray:
        if vec_norm is not None:
            obs_batched = obs.reshape(1, -1).astype(np.float32)
            obs_normed = vec_norm.normalize_obs(obs_batched)
            action, _ = model.predict(obs_normed[0], deterministic=True)
        else:
            action, _ = model.predict(obs, deterministic=True)
        return action

    return predict


import gymnasium as _gym


class _NullObsEnv(_gym.Env):
    """Trivial Gymnasium env used only to satisfy VecNormalize.load() API.

    SB3's ``VecNormalize.load`` requires a real vec env whose observation_space
    and action_space match the saved stats.  We never step it — it's only
    there so ``DummyVecEnv([lambda: _NullObsEnv(...)])`` constructs a wrapper
    with the correct spaces, and ``.obs_rms`` gets loaded from disk.
    """

    def __init__(self, obs_space, action_space):
        super().__init__()
        self.observation_space = obs_space
        self.action_space = action_space

    def reset(self, *, seed=None, options=None):  # type: ignore[override]
        super().reset(seed=seed)
        return self.observation_space.sample(), {}

    def step(self, action):
        return (
            self.observation_space.sample(), 0.0, False, False, {},
        )

    def render(self):  # pragma: no cover
        return None

    def close(self):
        return None


def run_ab(
    cfg: DRLHedgingConfig,
    period: tuple[str, str],
    v3_policy_path: str,
    v3_vecnorm_path: str | None,
    v4_policy_path: str,
    v4_vecnorm_path: str | None,
    vpin_parquet: str,
    vpin_column: str,
    n_episodes: int,
    seed: int,
) -> dict:
    logger.info("Loading v3 policy: %s", v3_policy_path)
    v3_predict = _load_sac_policy(v3_policy_path, v3_vecnorm_path)
    logger.info("Loading v4 policy: %s", v4_policy_path)
    v4_predict = _load_sac_policy(v4_policy_path, v4_vecnorm_path)

    logger.info("Loading true-VPIN series from %s[%s]",
                vpin_parquet, vpin_column)
    vpin_true = _load_vpin_series(vpin_parquet, vpin_column)

    # Build two envs that differ only in their VPIN source.
    # Matching seeds ⇒ matching bar samples ⇒ paired comparison.
    env_v3 = _build_env(cfg, period, vpin_series=None, seed=seed)
    env_v4 = _build_env(cfg, period, vpin_series=vpin_true, seed=seed)

    rewards_v3 = np.empty(n_episodes, dtype=np.float64)
    costs_v3 = np.empty(n_episodes, dtype=np.float64)
    rewards_v4 = np.empty(n_episodes, dtype=np.float64)
    costs_v4 = np.empty(n_episodes, dtype=np.float64)

    for i in range(n_episodes):
        r3, c3 = _rollout(env_v3, v3_predict)
        r4, c4 = _rollout(env_v4, v4_predict)
        rewards_v3[i] = r3
        costs_v3[i] = c3
        rewards_v4[i] = r4
        costs_v4[i] = c4
        if (i + 1) % 50 == 0:
            logger.info(
                "  [%d/%d] v3 mean_R=%.4f  v4 mean_R=%.4f  "
                "v3 mean_C=%.4f  v4 mean_C=%.4f",
                i + 1, n_episodes,
                rewards_v3[: i + 1].mean(), rewards_v4[: i + 1].mean(),
                costs_v3[: i + 1].mean(), costs_v4[: i + 1].mean(),
            )

    stats_v3 = _compute_stats("v3_proxy", rewards_v3, costs_v3)
    stats_v4 = _compute_stats("v4_true_vpin", rewards_v4, costs_v4)

    # Paired differences (v4 - v3). Positive reward diff is good;
    # negative cost diff is good.
    reward_diff = rewards_v4 - rewards_v3
    cost_diff = costs_v4 - costs_v3
    mean_r_diff, r_p5, r_p95 = _paired_bootstrap_ci(reward_diff, seed=seed)
    mean_c_diff, c_p5, c_p95 = _paired_bootstrap_ci(cost_diff, seed=seed)

    # Adoption rule: Sortino improves AND cost std <= 1.10 * baseline
    sortino_ok = (
        np.isfinite(stats_v4.sortino)
        and stats_v4.sortino > stats_v3.sortino
    )
    cost_std_ok = stats_v4.std_cost <= 1.10 * stats_v3.std_cost
    adopt = sortino_ok and cost_std_ok

    result = {
        "period": period,
        "n_episodes": n_episodes,
        "seed": seed,
        "v3": stats_v3.as_dict(),
        "v4": stats_v4.as_dict(),
        "paired_reward_diff": {
            "mean": mean_r_diff, "p5": r_p5, "p95": r_p95,
        },
        "paired_cost_diff": {
            "mean": mean_c_diff, "p5": c_p5, "p95": c_p95,
        },
        "adoption": {
            "sortino_ok": bool(sortino_ok),
            "cost_std_ok": bool(cost_std_ok),
            "adopt": bool(adopt),
        },
    }
    return result


def _print_report(result: dict) -> None:
    v3 = result["v3"]
    v4 = result["v4"]
    print("=" * 92)
    print(
        "  W5 A/B BENCHMARK — SAC v3 (proxy VPIN) vs v4 (true VPIN)"
    )
    print("=" * 92)
    print(f"  Period:    {result['period'][0]} -> {result['period'][1]}")
    print(f"  Episodes:  {result['n_episodes']}  (matched seed, paired)")
    print()
    print(
        f"  {'policy':>14s}  {'mean_R':>10s}  {'std_R':>10s}  "
        f"{'mean_C':>10s}  {'std_C':>10s}  {'Sortino':>9s}  "
        f"{'CVaR5%':>10s}"
    )
    print("-" * 92)
    for s in (v3, v4):
        print(
            f"  {s['name']:>14s}  {s['mean_reward']:>10.4f}  "
            f"{s['std_reward']:>10.4f}  {s['mean_cost']:>10.4f}  "
            f"{s['std_cost']:>10.4f}  {s['sortino']:>9.3f}  "
            f"{s['cvar_5pct']:>10.4f}"
        )
    print("-" * 92)
    print()
    r = result["paired_reward_diff"]
    c = result["paired_cost_diff"]
    print(
        f"  Reward diff (v4 - v3):  mean {r['mean']:+.4f}  "
        f"90% CI [{r['p5']:+.4f}, {r['p95']:+.4f}]"
    )
    print(
        f"  Cost   diff (v4 - v3):  mean {c['mean']:+.4f}  "
        f"90% CI [{c['p5']:+.4f}, {c['p95']:+.4f}]  (negative = better)"
    )
    print()
    ado = result["adoption"]
    print(
        f"  Adoption criteria:  Sortino_improves={ado['sortino_ok']}  "
        f"cost_std <= 1.10*baseline={ado['cost_std_ok']}"
    )
    verdict = "ADOPT v4" if ado["adopt"] else "REJECT v4 (fallback to v3)"
    print(f"  VERDICT: {verdict}")
    print("=" * 92)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="SAC v3 vs v4 paired A/B benchmark (W5.6)",
    )
    parser.add_argument("--asset", default="BTCUSDT",
                        help="Training asset (BTCUSDT or ETHUSDT). Routes VPIN "
                             "default and cfg.vol_arb_path/svi_params_path via "
                             "vol_arb.paths.")
    parser.add_argument("--v3-policy", type=str, required=True)
    parser.add_argument("--v3-vecnorm", type=str, default="")
    parser.add_argument("--v4-policy", type=str, required=True)
    parser.add_argument("--v4-vecnorm", type=str, default="")
    parser.add_argument(
        "--vpin-parquet", type=str, default="",
        help="Defaults to per-asset vpin_{currency_lower}_15m.parquet.",
    )
    parser.add_argument("--vpin-column", type=str, default="vpin_true")
    parser.add_argument("--period", type=str, default="",
                        help="YYYY-MM-DD:YYYY-MM-DD; default = cfg.test_period")
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260411)
    parser.add_argument(
        "--summary-path", type=str,
        default="vol_arb/output/sac_v3_v4_benchmark.json",
    )
    args = parser.parse_args()

    cfg = DRLHedgingConfig(asset=args.asset)
    if not args.vpin_parquet:
        args.vpin_parquet = cfg.vpin_path
    if args.period:
        a, b = args.period.split(":")
        period = (a.strip(), b.strip())
    else:
        period = cfg.test_period

    result = run_ab(
        cfg=cfg,
        period=period,
        v3_policy_path=args.v3_policy,
        v3_vecnorm_path=args.v3_vecnorm or None,
        v4_policy_path=args.v4_policy,
        v4_vecnorm_path=args.v4_vecnorm or None,
        vpin_parquet=args.vpin_parquet,
        vpin_column=args.vpin_column,
        n_episodes=args.n_episodes,
        seed=args.seed,
    )
    _print_report(result)

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(result, indent=2, default=str))
    logger.info("Summary JSON written to %s", summary_path)


if __name__ == "__main__":
    main()
