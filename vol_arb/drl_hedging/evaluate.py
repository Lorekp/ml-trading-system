"""Evaluation script: compare a trained DRL hedging policy against the
fixed-band rule-based baseline on a held-out window.

Runs ``n_episodes`` of the Gym env with:
  1. Random-action policy (sanity lower bound)
  2. Fixed-band rule (matches simulate_discrete_hedging at 0.05 BTC band)
  3. Trained SAC policy (deterministic=True)

And reports per-policy:
  - Mean / median / std of episode return
  - Mean hedge cost per episode
  - Stddev of hedge cost per episode  (← the key metric — lower is better)
  - Sortino, Ulcer Index, CVaR(5%) over the per-episode return distribution

These are diagnostic metrics; they do not by themselves validate a hedger or
trading result.

Usage::

    PYTHONPATH=. python vol_arb/drl_hedging/evaluate.py \\
        --policy vol_arb/drl_hedging/models/sac_hedger_final.zip
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np

from vol_arb.drl_hedging.config import DRLHedgingConfig
from vol_arb.drl_hedging.env import GymnasiumHedgingEnv, MarketDataView

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("evaluate")


def rollout_episode(env: GymnasiumHedgingEnv, policy_fn) -> dict:
    obs, _ = env.reset()
    total_reward = 0.0
    total_cost = 0.0
    done = False
    while not done:
        action = policy_fn(obs)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        total_cost += info.get("cost", 0.0)
        done = terminated or truncated
    return {"reward": total_reward, "cost": total_cost}


def random_policy(obs):
    return np.array([np.random.uniform(-1.0, 1.0)], dtype=np.float32)


def full_hedge_policy(obs):
    """Always trade +1 fraction — the rule-based behaviour when the
    required delta exceeds any band.  Matches the hedger's intent
    (fully eliminate the delta each tick) so is a meaningful baseline."""
    return np.array([1.0], dtype=np.float32)


def no_hedge_policy(obs):
    """Never trade — unhedged benchmark."""
    return np.array([0.0], dtype=np.float32)


def _metrics_from_returns(returns: np.ndarray, costs: np.ndarray) -> dict:
    returns = np.asarray(returns, dtype=np.float64)
    costs = np.asarray(costs, dtype=np.float64)
    mean_ret = float(returns.mean())
    std_ret = float(returns.std(ddof=1)) if len(returns) > 1 else 0.0
    downside = returns[returns < 0]
    down_std = float(np.sqrt(np.mean(downside * downside))) if len(downside) > 0 else 0.0
    sortino = mean_ret / down_std if down_std > 0 else float("nan")
    cvar_5 = float(np.mean(np.sort(returns)[: max(1, int(0.05 * len(returns)))]))
    return {
        "n": int(len(returns)),
        "mean_reward": mean_ret,
        "median_reward": float(np.median(returns)),
        "std_reward": std_ret,
        "mean_cost": float(costs.mean()),
        "std_cost": float(costs.std(ddof=1)) if len(costs) > 1 else 0.0,
        "sortino": sortino,
        "cvar_5pct": cvar_5,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset", default="BTCUSDT",
                        help="Training asset (BTCUSDT or ETHUSDT)")
    parser.add_argument("--policy", type=str, required=False,
                        help="Path to SAC .zip policy file (optional — skip to evaluate only baselines)")
    parser.add_argument("--period", type=str, default=None,
                        help="Held-out period, format YYYY-MM-DD:YYYY-MM-DD (defaults to cfg.test_period)")
    parser.add_argument("--n-episodes", type=int, default=500)
    parser.add_argument("--vol-arb-path", type=str, default=None)
    parser.add_argument("--svi-params-path", type=str, default=None)
    args = parser.parse_args()

    cfg = DRLHedgingConfig(asset=args.asset)
    if args.vol_arb_path is not None:
        cfg.vol_arb_path = args.vol_arb_path
    if args.svi_params_path is not None:
        cfg.svi_params_path = args.svi_params_path

    if args.period is None:
        period = cfg.test_period
    else:
        a, b = args.period.split(":")
        period = (a.strip(), b.strip())

    logger.info("Evaluating on %s .. %s over %d episodes", period[0], period[1], args.n_episodes)

    data = MarketDataView(
        vol_arb_path=cfg.vol_arb_path,
        svi_params_path=cfg.svi_params_path,
        period=period,
        high_vol_upsample_threshold=0.0,  # no upsampling during eval
        high_vol_upsample_factor=1.0,
    )
    env = GymnasiumHedgingEnv(cfg, data, worker_seed=cfg.seed + 99_999)

    policies: dict[str, callable] = {
        "no_hedge": no_hedge_policy,
        "full_hedge": full_hedge_policy,
        "random": random_policy,
    }

    if args.policy is not None:
        from stable_baselines3 import SAC  # lazy
        model = SAC.load(args.policy, device="cpu")

        def _drl_policy(obs):
            action, _ = model.predict(obs, deterministic=True)
            return action

        policies["drl_sac"] = _drl_policy

    summary: dict[str, dict] = {}
    for name, fn in policies.items():
        logger.info("Running policy %s", name)
        rewards = np.empty(args.n_episodes, dtype=np.float64)
        costs = np.empty(args.n_episodes, dtype=np.float64)
        for i in range(args.n_episodes):
            res = rollout_episode(env, fn)
            rewards[i] = res["reward"]
            costs[i] = res["cost"]
        summary[name] = _metrics_from_returns(rewards, costs)

    # Print comparison table
    print()
    print(f"{'policy':>12s}  {'mean_R':>10s}  {'median_R':>10s}  {'std_R':>10s}  "
          f"{'mean_cost':>10s}  {'std_cost':>10s}  {'sortino':>8s}  {'CVaR5%':>10s}")
    print("-" * 100)
    for name, m in summary.items():
        print(f"{name:>12s}  {m['mean_reward']:>10.4f}  {m['median_reward']:>10.4f}  "
              f"{m['std_reward']:>10.4f}  {m['mean_cost']:>10.4f}  {m['std_cost']:>10.4f}  "
              f"{m['sortino']:>8.3f}  {m['cvar_5pct']:>10.4f}")
    print()

    # ── Verdict: did the DRL policy beat the baselines? ───────────────
    if "drl_sac" in summary:
        drl = summary["drl_sac"]
        print("-" * 100)
        print("VERDICT")
        print("-" * 100)
        for ref_name in ("no_hedge", "full_hedge", "random"):
            if ref_name not in summary:
                continue
            ref = summary[ref_name]
            dR = drl["mean_reward"] - ref["mean_reward"]
            dStd = drl["std_reward"] - ref["std_reward"]
            dCost = drl["mean_cost"] - ref["mean_cost"]
            pct_std = 100.0 * dStd / ref["std_reward"] if ref["std_reward"] != 0 else 0.0
            verdict = (
                "BETTER" if (dR > 0 and dStd <= 0)
                else "WORSE" if (dR <= 0 and dStd >= 0)
                else "MIXED"
            )
            print(f"  vs {ref_name:>12s}: "
                  f"dR={dR:+.4f}  "
                  f"dStd={dStd:+.4f} ({pct_std:+.1f}%)  "
                  f"dCost={dCost:+.4f}   [{verdict}]")
        print()
        # Primary acceptance: beat no_hedge on BOTH mean and std
        if "no_hedge" in summary:
            nh = summary["no_hedge"]
            beats_mean = drl["mean_reward"] > nh["mean_reward"]
            beats_std = drl["std_reward"] < nh["std_reward"]
            if beats_mean and beats_std:
                print("  >>> PASS: DRL Pareto-dominates no_hedge (mean ^ + std v) <<<")
            else:
                print(f"  >>> PARTIAL: DRL vs no_hedge: "
                      f"mean {'win' if beats_mean else 'loss'}, "
                      f"std {'win' if beats_std else 'loss'}")
        print()


if __name__ == "__main__":
    main()
