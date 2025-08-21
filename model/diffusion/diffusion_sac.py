"""
Soft Actor-Critic (SAC) with diffusion policy actor.

This module implements a SAC variant where the actor is a diffusion policy.
Since the exact log-likelihood of a diffusion policy is generally intractable,
we approximate log pi(a|s) by aggregating per-step Gaussian transition
likelihoods along the denoising chain, reusing utilities from VPGDiffusion.

Value-critic style (SAC-v1):
  - Q critics: Q1, Q2
  - V critic: V
  - Target network: V_target (Polyak averaging)

Actor loss:
  L_pi = E[ -Q_min(s,a) + alpha * log_pi(a|s) ]

Value loss:
  L_V = E[ ( V(s) - (Q_min(s,a') - alpha * log_pi(a'|s)) )^2 ]

Critic loss:
  L_Q = E[ ( Q(s,a) - (r + gamma * V_target(s')) )^2 ]

Note: This file implements the non-EBM version. For the EBM-augmented version
that replaces log pi with an energy surrogate and supports PBRS, see
`diffusion_sac_ebm.py`.
"""

from typing import Tuple
import copy
import torch
import torch.nn.functional as F
import logging

from model.diffusion.diffusion_vpg import VPGDiffusion

log = logging.getLogger(__name__)


class SACDiffusion(VPGDiffusion):
    def __init__(
        self,
        actor,
        critic_q1,
        critic_q2,
        critic_v,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        target_entropy: float = -1.0,
        **kwargs,
    ):
        """
        Args:
            actor: Diffusion actor network (e.g., `DiffusionMLP`)
            critic_q1: Q-function critic (state-action)
            critic_q2: Q-function critic (state-action)
            critic_v: Value-function critic (state-only)
            tau: Polyak averaging coefficient for V target
            alpha: Initial temperature for entropy
            automatic_entropy_tuning: If true, temperature is learned externally
            target_entropy: Target entropy used when tuning temperature
            kwargs: Passed to VPGDiffusion (e.g., denoising params)
        """
        super().__init__(actor=actor, critic=critic_v, **kwargs)

        # Critics
        self.critic_q1 = critic_q1.to(self.device)
        self.critic_q2 = critic_q2.to(self.device)
        self.critic_v = self.critic  # alias from parent
        self.target_v = copy.deepcopy(self.critic_v).to(self.device)

        self.tau = float(tau)
        self.alpha_init = float(alpha)
        self.automatic_entropy_tuning = bool(automatic_entropy_tuning)
        self.target_entropy = float(target_entropy)

    # ------------------------ Utilities ------------------------ #
    def _chain_logprob(self, cond: dict, chains: torch.Tensor) -> torch.Tensor:
        """
        Approximate log pi(a|s) by aggregating per-step Gaussian transition
        log-probabilities along the denoising chain. We average over
        denoising steps and sum over action dimensions and horizon.

        Args:
            cond: dict with "state" key (B, To, Do)
            chains: (B, K+1, H, A) denoised actions from earliest to final

        Returns:
            log_pi: (B,) approximate log-likelihood per sample
        """
        assert chains.dim() == 4, f"chains must be [B, K+1, H, A], got {chains.shape}"
        # Reuse VPGDiffusion.get_logprobs which returns per-step log-prob for chains
        logprobs = self.get_logprobs(cond, chains)  # [B*K, H, A]
        # Sum over horizon and action dims; average across K to stabilize scale
        logprobs = logprobs.sum(dim=(-2, -1))  # [B*K]
        B = chains.size(0)
        K = self.ft_denoising_steps
        logprobs = logprobs.view(B, K).mean(dim=1)  # [B]
        return logprobs

    def _sample_action_and_chain(self, obs: dict, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample an action and the full denoising chain from the actor.

        Returns:
            actions: (B, H, A)
            chains:  (B, K+1, H, A)
        """
        samples = super().forward(obs, deterministic=deterministic, return_chain=True)
        return samples.trajectories, samples.chains

    # ------------------------ Losses ------------------------ #
    def loss_critic(
        self,
        obs: dict,
        next_obs: dict,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        gamma: float,
    ) -> torch.Tensor:
        """
        Bellman residual for Q critics with value target:
            y = r + gamma * V_target(s')
        """
        with torch.no_grad():
            target_v = self.target_v(next_obs).view(-1)
            y = rewards.view(-1) + float(gamma) * target_v * (1.0 - terminated.view(-1))

        q1, q2 = self.critic_q1(obs, actions), self.critic_q2(obs, actions)
        q1 = q1.view(-1)
        q2 = q2.view(-1)
        loss_q = F.mse_loss(q1, y) + F.mse_loss(q2, y)
        return loss_q

    def loss_value(self, obs: dict, alpha: float) -> torch.Tensor:
        """
        Value regression towards soft state value target using sampled actions:
            target = min(Q1,Q2)(s,a') - alpha * log_pi(a'|s)
        """
        with torch.no_grad():
            actions, chains = self._sample_action_and_chain(obs, deterministic=False)
            log_pi = self._chain_logprob(obs, chains)  # [B]
        q1, q2 = self.critic_q1(obs, actions), self.critic_q2(obs, actions)
        q_min = torch.minimum(q1.view(-1), q2.view(-1))
        target = q_min - float(alpha) * log_pi
        v = self.critic_v(obs).view(-1)
        return F.mse_loss(v, target.detach())

    def loss_actor(self, obs: dict, alpha: float) -> torch.Tensor:
        """
        Policy loss using the same log pi approximation as in loss_value.
        """
        actions, chains = self._sample_action_and_chain(obs, deterministic=False)
        log_pi = self._chain_logprob(obs, chains)
        q1, q2 = self.critic_q1(obs, actions), self.critic_q2(obs, actions)
        q_min = torch.minimum(q1.view(-1), q2.view(-1))
        loss_pi = (-q_min + float(alpha) * log_pi).mean()
        return loss_pi

    def loss_temperature(self, obs: dict, alpha: torch.Tensor, target_entropy: float) -> torch.Tensor:
        """
        Temperature loss for automatic entropy tuning:
            L(alpha) = - E[ alpha * (log_pi + target_entropy) ]
        where log_pi is approximated by denoising-chain log-probability.
        """
        with torch.no_grad():
            _, chains = self._sample_action_and_chain(obs, deterministic=False)
            log_pi = self._chain_logprob(obs, chains)
        return -torch.mean(alpha * (log_pi + float(target_entropy)))

    # ------------------------ Target updates ------------------------ #
    def update_target_value(self, tau: float = None):
        tau = float(self.tau if tau is None else tau)
        for tgt, src in zip(self.target_v.parameters(), self.critic_v.parameters()):
            tgt.data.copy_(tgt.data * (1.0 - tau) + src.data * tau)


