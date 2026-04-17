"""Risk-averse Soft Actor-Critic with a second critic tracking E[R^2].

Standard SAC trains twin Q-networks to reduce overestimation bias, but
both estimate ``E[R]``.  For risk-averse hedging we also want to know
``E[R^2]`` so we can penalise ``sqrt(Var[R]) = sqrt(E[R^2] - E[R]^2)``
in the actor objective — the institutional mean-variance framing.

This module subclasses :class:`stable_baselines3.sac.SAC` and:

1. Builds a *third* ContinuousCritic (single-head, not twin) called
   ``critic_sq`` that estimates ``E[R^2]`` via standard TD on squared
   rewards.
2. Overrides ``train()`` to update ``critic_sq`` alongside the twin
   critics and to add the ``lambda_risk * std`` term to the actor loss.

The implementation deliberately mirrors the SB3 ``SAC.train()`` structure
so that downstream tools (TensorBoard logging, evaluation callbacks,
model.save/load) continue to work unmodified.  We import
``stable_baselines3`` lazily at class construction so the base env module
stays importable even on machines without sb3 installed.

References
----------
- Haarnoja et al. (2018), "Soft Actor-Critic: Off-Policy Maximum Entropy
  Deep Reinforcement Learning with a Stochastic Actor".
- Tamar et al. (2016), "Learning the Variance of the Reward-to-Go".
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


def _import_sb3():
    """Lazy import of stable_baselines3 pieces we need."""
    from stable_baselines3 import SAC  # noqa: WPS433
    from stable_baselines3.common.policies import ContinuousCritic  # noqa: WPS433
    from stable_baselines3.common.utils import polyak_update  # noqa: WPS433
    import torch as th  # noqa: WPS433
    import torch.nn.functional as F  # noqa: WPS433
    return SAC, ContinuousCritic, polyak_update, th, F


def _make_single_head_critic(th, ContinuousCritic, policy):
    """Build a single-headed ContinuousCritic mirroring the actor's feature
    extractor / net_arch, attached to ``policy``.

    SB3's default ``ContinuousCritic(n_critics=2)`` builds twin critics;
    for the squared-reward estimator we only need one head.
    """
    observation_space = policy.observation_space
    action_space = policy.action_space
    net_arch = policy.net_arch
    # SB3's net_arch can be a list or dict; for SAC it's usually a list shared
    if isinstance(net_arch, dict):
        qf_arch = net_arch.get("qf", net_arch.get("pi", [256, 256]))
    else:
        qf_arch = net_arch
    features_extractor = policy.make_features_extractor()
    critic = ContinuousCritic(
        observation_space=observation_space,
        action_space=action_space,
        net_arch=qf_arch,
        features_extractor=features_extractor,
        features_dim=features_extractor.features_dim,
        activation_fn=policy.activation_fn,
        normalize_images=False,
        n_critics=1,
        share_features_extractor=False,
    )
    return critic


class RiskAverseSAC:
    """Factory + subclass for risk-averse SAC with a squared-reward critic.

    Usage::

        model = RiskAverseSAC(
            "MlpPolicy", vec_env,
            lambda_risk=0.3,
            lambda_risk_warmup_steps=100_000,
            lambda_risk_rampup_steps=200_000,
            max_grad_norm=0.5,
            learning_rate=3e-4,
            ...,
        )
        model.learn(total_timesteps=1_000_000)

    The returned object is an instance of a dynamically-created subclass
    of ``stable_baselines3.SAC`` — anything that works on a plain SAC
    instance works here too (``save``, ``load``, ``predict``, callbacks).

    Stability features (added after the 300k run diverged):

    - **Gradient clipping** (``max_grad_norm``): caps the L2 norm of actor,
      critic, and critic_sq gradients before each optimiser step.  Literature-
      standard 0.5 default.
    - **Lambda warmup schedule** (``lambda_risk_warmup_steps`` +
      ``lambda_risk_rampup_steps``): the variance penalty's weight starts at
      0 and ramps linearly to ``lambda_risk`` over the warmup+rampup window.
      Lets the mean critic stabilise before the noisy second critic enters
      the actor loss.
    """

    def __new__(
        cls,
        *args,
        lambda_risk: float = 0.3,
        lambda_risk_warmup_steps: int = 0,
        lambda_risk_rampup_steps: int = 0,
        max_grad_norm: float = 0.0,
        **kwargs,
    ):
        SAC, ContinuousCritic, polyak_update, th, F = _import_sb3()

        # Build the subclass on first call; cache on cls.
        if not hasattr(cls, "_subclass"):
            cls._subclass = _build_subclass(
                SAC, ContinuousCritic, polyak_update, th, F,
            )

        instance = cls._subclass(*args, **kwargs)
        instance.lambda_risk_target = float(lambda_risk)
        instance.lambda_risk = float(lambda_risk)  # mutable current value, logged
        instance.lambda_risk_warmup_steps = int(lambda_risk_warmup_steps)
        instance.lambda_risk_rampup_steps = int(lambda_risk_rampup_steps)
        instance.max_grad_norm = float(max_grad_norm)
        instance._build_critic_sq()
        return instance


def _build_subclass(SAC, ContinuousCritic, polyak_update, th, F):
    """Create the actual SAC subclass at runtime (closure over sb3 symbols)."""

    class _RiskAverseSAC(SAC):
        lambda_risk: float = 0.3
        lambda_risk_target: float = 0.3
        lambda_risk_warmup_steps: int = 0
        lambda_risk_rampup_steps: int = 0
        max_grad_norm: float = 0.0

        def _build_critic_sq(self) -> None:
            """Create the squared-reward critic and its target."""
            self.critic_sq = _make_single_head_critic(th, ContinuousCritic, self.policy).to(self.device)
            self.critic_sq_target = _make_single_head_critic(th, ContinuousCritic, self.policy).to(self.device)
            self.critic_sq_target.load_state_dict(self.critic_sq.state_dict())
            self.critic_sq_target.set_training_mode(False)
            # Optimizer
            self.critic_sq_optimizer = th.optim.Adam(
                self.critic_sq.parameters(),
                lr=self.lr_schedule(1.0),
            )

        def _effective_lambda_risk(self) -> float:
            """Compute the current lambda_risk via the warmup+rampup schedule.

            - ``num_timesteps < warmup``              → 0  (pure mean-Q SAC)
            - ``warmup <= t < warmup + rampup``       → linear 0 → target
            - ``t >= warmup + rampup``                → target  (held constant)

            When ``warmup == 0 and rampup == 0`` the schedule is a no-op and
            this always returns ``lambda_risk_target`` (legacy behaviour).
            """
            t = self.num_timesteps
            warmup = self.lambda_risk_warmup_steps
            rampup = self.lambda_risk_rampup_steps
            if warmup <= 0 and rampup <= 0:
                return self.lambda_risk_target
            if t < warmup:
                return 0.0
            if rampup <= 0:
                return self.lambda_risk_target
            if t < warmup + rampup:
                progress = (t - warmup) / rampup
                return self.lambda_risk_target * progress
            return self.lambda_risk_target

        def train(self, gradient_steps: int, batch_size: int = 64) -> None:
            """Risk-averse SAC training step — mirrors SB3's SAC.train()
            with gradient clipping and lambda_risk warmup added.
            """
            # Switch to train mode
            self.policy.set_training_mode(True)
            self.critic_sq.set_training_mode(True)

            optimizers = [self.actor.optimizer, self.critic.optimizer, self.critic_sq_optimizer]
            if self.ent_coef_optimizer is not None:
                optimizers.append(self.ent_coef_optimizer)
            self._update_learning_rate(optimizers)

            # Snapshot the effective lambda_risk for this gradient-step batch
            # (the schedule only advances once per train() call, not per
            # inner gradient step — SB3 convention).
            self.lambda_risk = self._effective_lambda_risk()

            ent_coef_losses, ent_coefs = [], []
            actor_losses, critic_losses, critic_sq_losses = [], [], []
            grad_norms_actor, grad_norms_critic, grad_norms_critic_sq = [], [], []

            clip_norm = self.max_grad_norm
            do_clip = clip_norm is not None and clip_norm > 0

            for _ in range(gradient_steps):
                self._n_updates += 1
                replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

                # ── Entropy coefficient update ────────────────────────
                actions_pi, log_prob = self.actor.action_log_prob(replay_data.observations)
                log_prob = log_prob.reshape(-1, 1)

                ent_coef_loss = None
                if self.ent_coef_optimizer is not None and self.log_ent_coef is not None:
                    ent_coef = th.exp(self.log_ent_coef.detach())
                    ent_coef_loss = -(self.log_ent_coef * (log_prob + self.target_entropy).detach()).mean()
                    ent_coef_losses.append(ent_coef_loss.item())
                else:
                    ent_coef = self.ent_coef_tensor

                ent_coefs.append(ent_coef.item())

                if ent_coef_loss is not None and self.ent_coef_optimizer is not None:
                    self.ent_coef_optimizer.zero_grad()
                    ent_coef_loss.backward()
                    self.ent_coef_optimizer.step()

                # ── Standard twin-critic update (E[R]) ────────────────
                with th.no_grad():
                    next_actions, next_log_prob = self.actor.action_log_prob(replay_data.next_observations)
                    next_q_values = th.cat(self.critic_target(replay_data.next_observations, next_actions), dim=1)
                    next_q_values, _ = th.min(next_q_values, dim=1, keepdim=True)
                    next_q_values = next_q_values - ent_coef * next_log_prob.reshape(-1, 1)
                    target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values

                current_q_values = self.critic(replay_data.observations, replay_data.actions)
                critic_loss = 0.5 * sum(F.mse_loss(current_q, target_q_values) for current_q in current_q_values)
                assert isinstance(critic_loss, th.Tensor)
                critic_losses.append(critic_loss.item())

                self.critic.optimizer.zero_grad()
                critic_loss.backward()
                if do_clip:
                    g = th.nn.utils.clip_grad_norm_(
                        self.critic.parameters(), max_norm=clip_norm,
                    )
                    grad_norms_critic.append(float(g))
                self.critic.optimizer.step()

                # ── Squared-reward critic update (E[R^2]) ─────────────
                with th.no_grad():
                    # Use the actor's next action (deterministic piece): mirror E[R] TD
                    next_q_sq_tensors = self.critic_sq_target(replay_data.next_observations, next_actions)
                    next_q_sq = next_q_sq_tensors[0]
                    r_sq = replay_data.rewards * replay_data.rewards
                    target_q_sq = r_sq + (1 - replay_data.dones) * (self.gamma * self.gamma) * next_q_sq

                current_q_sq_tensors = self.critic_sq(replay_data.observations, replay_data.actions)
                current_q_sq = current_q_sq_tensors[0]
                critic_sq_loss = F.mse_loss(current_q_sq, target_q_sq)
                critic_sq_losses.append(critic_sq_loss.item())

                self.critic_sq_optimizer.zero_grad()
                critic_sq_loss.backward()
                if do_clip:
                    g = th.nn.utils.clip_grad_norm_(
                        self.critic_sq.parameters(), max_norm=clip_norm,
                    )
                    grad_norms_critic_sq.append(float(g))
                self.critic_sq_optimizer.step()

                # ── Actor update with risk-averse objective ───────────
                q_values_pi = th.cat(self.critic(replay_data.observations, actions_pi), dim=1)
                min_qf_pi, _ = th.min(q_values_pi, dim=1, keepdim=True)

                q_sq_pi_tensors = self.critic_sq(replay_data.observations, actions_pi)
                q_sq_pi = q_sq_pi_tensors[0]
                variance = th.clamp(q_sq_pi - min_qf_pi * min_qf_pi, min=1e-6)
                std = th.sqrt(variance)

                actor_loss = (ent_coef * log_prob - min_qf_pi + self.lambda_risk * std).mean()
                actor_losses.append(actor_loss.item())

                self.actor.optimizer.zero_grad()
                actor_loss.backward()
                if do_clip:
                    g = th.nn.utils.clip_grad_norm_(
                        self.actor.parameters(), max_norm=clip_norm,
                    )
                    grad_norms_actor.append(float(g))
                self.actor.optimizer.step()

                # ── Polyak soft updates ───────────────────────────────
                if self._n_updates % self.target_update_interval == 0:
                    polyak_update(self.critic.parameters(), self.critic_target.parameters(), self.tau)
                    polyak_update(self.critic_sq.parameters(), self.critic_sq_target.parameters(), self.tau)
                    # Match SB3: also copy BatchNorm running stats if present
                    polyak_update(self.batch_norm_stats, self.batch_norm_stats_target, 1.0)

            self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
            self.logger.record("train/ent_coef", float(np.mean(ent_coefs)))
            self.logger.record("train/actor_loss", float(np.mean(actor_losses)))
            self.logger.record("train/critic_loss", float(np.mean(critic_losses)))
            self.logger.record("train/critic_sq_loss", float(np.mean(critic_sq_losses)))
            self.logger.record("train/lambda_risk", float(self.lambda_risk))
            if len(ent_coef_losses) > 0:
                self.logger.record("train/ent_coef_loss", float(np.mean(ent_coef_losses)))
            if do_clip and grad_norms_actor:
                self.logger.record("train/grad_norm_actor", float(np.mean(grad_norms_actor)))
                self.logger.record("train/grad_norm_critic", float(np.mean(grad_norms_critic)))
                self.logger.record("train/grad_norm_critic_sq", float(np.mean(grad_norms_critic_sq)))

    return _RiskAverseSAC
