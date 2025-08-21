"""
Preference-Guided SAC for diffusion policies with EBM integration.

This module extends SACDiffusion by:
  1) Using an InfoNCE-trained EBM as a surrogate for log pi in the actor loss:
       log pi(a|s) ≈ -E_psi(s,a) + C(s)
     Hence the actor loss becomes E[ -Q_min(s,a) - alpha * E(s,a) ].

  2) Optionally enabling potential-based reward shaping (PBRS) via KFreePotential,
     which is applied during rollout by the training agent (not inside losses).

The critic/value objectives remain standard SAC with a V-target. The PBRS logic
and energy scaling utilities live in the training agent; this module focuses on
the actor loss modification and EBM plumbing.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import logging

from model.diffusion.diffusion_sac import SACDiffusion

log = logging.getLogger(__name__)


class SACDiffusionEBM(SACDiffusion):
    def __init__(
        self,
        actor,
        critic_q1,
        critic_q2,
        critic_v,
        ebm: Optional[nn.Module] = None,
        ebm_ckpt_path: Optional[str] = None,
        use_ebm_reward_shaping: bool = False,  # handled in agent at rollout
        **kwargs,
    ):
        super().__init__(
            actor=actor,
            critic_q1=critic_q1,
            critic_q2=critic_q2,
            critic_v=critic_v,
            **kwargs,
        )

        # EBM for entropy surrogate in actor loss
        self.ebm_model = None
        if ebm is not None:
            self.set_ebm(ebm, ebm_ckpt_path)

        # Metadata toggle (PBRS applied externally by the training agent)
        self.use_ebm_reward_shaping = use_ebm_reward_shaping

        # Normalization stats for actions; provided by agent via setup
        self._stats: Optional[Dict[str, Any]] = None

    def set_ebm(self, ebm_model: nn.Module, ebm_ckpt_path: Optional[str] = None):
        self.ebm_model = ebm_model.to(self.device).eval()
        if ebm_ckpt_path is not None:
            try:
                checkpoint = torch.load(ebm_ckpt_path, map_location=self.device)
                state_dict = checkpoint.get("model", checkpoint)
                self.ebm_model.load_state_dict(state_dict, strict=False)
                log.info(f"Loaded EBM state from {ebm_ckpt_path}")
            except Exception as e:
                log.warning(f"Failed to load EBM checkpoint: {e}")

        # Freeze EBM
        for p in self.ebm_model.parameters():
            p.requires_grad = False

    def setup_action_norm_stats(self, stats: Dict[str, Any]):
        """Provide action normalization stats used by the EBM forward."""
        self._stats = stats

    # ------------------------ Actor loss override ------------------------ #
    def _compute_energy(self, obs: dict, actions: torch.Tensor) -> torch.Tensor:
        """Compute E(s,a) using EBM with action normalization to [-1, 1]."""
        assert self.ebm_model is not None, "EBM not set; call set_ebm first."
        poses = obs["state"][:, -1, :] if obs["state"].dim() == 3 else obs["state"]
        B = poses.size(0)

        # Ensure [B, H, A]
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)

        # Normalize actions to [-1, 1]
        if self._stats is not None:
            act_min = torch.tensor(self._stats["act_min"], dtype=torch.float32, device=actions.device)
            act_max = torch.tensor(self._stats["act_max"], dtype=torch.float32, device=actions.device)
            if act_min.ndim == 2:
                act_min = act_min.min(dim=0).values
            if act_max.ndim == 2:
                act_max = act_max.max(dim=0).values
            act_min = act_min.view(1, 1, -1)
            act_max = act_max.view(1, 1, -1)
            a_norm = torch.clamp((actions - act_min) / (act_max - act_min + 1e-6) * 2 - 1, -1.0, 1.0)
        else:
            a_norm = actions

        k_vec = torch.zeros(B, dtype=torch.long, device=poses.device)  # k=0 final action
        E_out = self.ebm_model(k_idx=k_vec, views=None, poses=poses, actions=a_norm)
        E = E_out[0] if isinstance(E_out, (tuple, list)) else E_out
        if E.dim() >= 2 and E.size(0) == B:  # average extra dims if any
            E = E.mean(dim=tuple(range(1, E.dim())))
        return E.view(-1)

    def loss_actor(self, obs: dict, alpha: float) -> torch.Tensor:
        """
        Replace log pi with -E(s,a) in the actor objective:
            L_pi ≈ E[ -Q_min(s,a) - alpha * E(s,a) ]
        """
        actions, _ = self._sample_action_and_chain(obs, deterministic=False)
        q1, q2 = self.critic_q1(obs, actions), self.critic_q2(obs, actions)
        q_min = torch.minimum(q1.view(-1), q2.view(-1))
        E = self._compute_energy(obs, actions)
        loss_pi = (-q_min - float(alpha) * E).mean()
        return loss_pi


