# agent/finetune/train_ppo_diffusion_ebm_agent.py
"""
Enhanced DPPO fine-tuning agent with Energy-Based Model integration.

This module extends the standard PPODiffusionAgent with energy-based model
capabilities and potential-based reward shaping.
"""

import os
import pickle
import einops
import numpy as np
import torch
import torch.nn as nn
import logging
import wandb
import math
import copy
from typing import Dict, Any, Optional, List

log = logging.getLogger(__name__)
from util.timer import Timer
from agent.finetune.train_ppo_diffusion_agent import TrainPPODiffusionAgent
from model.diffusion.diffusion_ppo_ebm import PPODiffusionEBM
from model.diffusion.energy_utils import build_k_use_indices, KFreePotential


def unravel_index(indices: torch.Tensor, shape: tuple):
    """
    Torch version of numpy.unravel_index (compatible with torch<2.1)
    Args:
        indices: Tensor of flat indices, shape (N,)
        shape: tuple of ints, the target shape
    Returns:
        tuple of Tensors, each (N,) giving coordinates along each dimension
    """
    coords = []
    for dim in reversed(shape):
        coord = indices % dim
        indices = indices // dim
        coords.append(coord)
    return tuple(reversed(coords))



class KFreePotentialLegacy:
    """Deprecated. Use KFreePotential from model.diffusion.energy_utils."""

    def __init__(
        self,
        ebm_model: nn.Module,
        pi0: nn.Module,
        K: int,
        stats: Dict[str, Any],
        k_use: List[int],
        beta_k: float = 1.0,
        alpha: float = 1.0,
        M: int = 1,
        use_mu_only: bool = True,
        device: Optional[torch.device] = None,
    ):
        self.ebm = ebm_model.eval()
        self.pi0 = pi0.eval()

        # Freeze the prior policy parameters
        for p in self.pi0.parameters():
            p.requires_grad = False

        self.K = K
        self.k_use = sorted(set(int(min(max(0, k), K)) for k in k_use))
        self.beta = beta_k
        self.alpha = alpha
        self.M = max(1, int(M))
        self.use_mu_only = use_mu_only
        self.device = device or next(ebm_model.parameters()).device

        # Legacy scaler reference removed; normalization is handled directly inside EBM
        self.stats = stats

    @torch.no_grad()
    def _phi_k_from_actions(
        self,
        poses: torch.Tensor,
        views,
        t_idx: torch.Tensor,
        k: int,
        A_k: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute energy term for a specific denoising step k.
        
        Returns: -β * E_θ(s, a_k, k) for use in log-sum-exp aggregation
        """
        B = poses.size(0)
        k_vec = torch.full((B,), int(k), dtype=torch.long, device=self.device)

        # Normalize actions using statistics
        act_min_t = torch.tensor(self.stats["act_min"], dtype=torch.float32, device=poses.device)
        act_max_t = torch.tensor(self.stats["act_max"], dtype=torch.float32, device=poses.device)
        if act_min_t.ndim == 2:
            act_min_t = act_min_t.min(dim=0).values
        if act_max_t.ndim == 2:
            act_max_t = act_max_t.max(dim=0).values
        act_min_t = act_min_t.view(1, 1, -1)
        act_max_t = act_max_t.view(1, 1, -1)

        # Normalize actions to [-1, 1] range
        a_norm = torch.clamp((A_k - act_min_t) / (act_max_t - act_min_t + 1e-6) * 2 - 1, -1.0, 1.0)

        # Compute energy using EBM (pass t_idx for temporal embedding consistency)
        E_out = self.ebm(k_idx=k_vec, t_idx=t_idx.to(self.device), views=None, poses=poses, actions=a_norm)
        E = E_out[0] if isinstance(E_out, (tuple, list)) else E_out

        # Average over additional dimensions if present
        if E.dim() >= 2 and E.size(0) == B:
            E = E.mean(dim=tuple(range(1, E.dim())))
        
        # Return -β * E for log-sum-exp aggregation
        # This corresponds to the exp(-β * E_θ(s, a_k, k)) term in the mathematical formula
        return -self.beta * E

    @torch.no_grad()
    def phi(self, obs_state: torch.Tensor, t_idx: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute the potential φ(s) for given observations.
        
        Implements the path free energy formula from the mathematical context:
        φ(s) = (α/β) * log[1/(M*|K|) * Σ_m Σ_k exp(-β * E_θ(s, a_k^(m), k))]
        
        0-based indexing: k = 0..K, where k=0 is the final (fully denoised) action.
        """
        # obs_state: [B, obs_dim] or [B, 1, obs_dim]
        if obs_state.dim() == 3 and obs_state.size(1) == 1:
            poses = obs_state[:, 0, :]
        else:
            poses = obs_state

        # Ensure poses is a proper tensor
        if not isinstance(poses, torch.Tensor):
            poses = torch.tensor(poses, device=obs_state.device, dtype=obs_state.dtype)
        
        # Ensure poses is contiguous
        if not poses.is_contiguous():
            poses = poses.contiguous()

        B = poses.size(0)
        device = poses.device
        # PPODiffusionEBM.forward expects cond["state"] shape [B, cond_steps, obs_dim]; current config cond_steps=1
        # Ensure poses is a proper tensor and reshape it correctly
        if not isinstance(poses, torch.Tensor):
            poses = torch.tensor(poses, device=device, dtype=torch.float32)
        
        cond_state = poses.view(B, 1, -1).contiguous()

        cond = {
            "state": cond_state,
        }

        # Collect all energy terms for proper log-sum-exp aggregation
        all_energy_terms = []

        if self.use_mu_only or self.M == 1:
            # Deterministic: single forward pass
            samples = self.pi0.forward(cond=cond, deterministic=True, return_chain=True)
            chain = samples.chains  # [B, K+1, H, A]

            for k in self.k_use:
                # 0-based: k=0 -> final action; k=K -> earliest action
                A_k = chain[:, k]  # [B, H, A]
                energy_term = self._phi_k_from_actions(
                    poses,
                    None,
                    t_idx if t_idx is not None else torch.zeros(B, dtype=torch.long, device=device),
                    k,
                    A_k,
                )
                all_energy_terms.append(energy_term)
        else:
            # Stochastic sampling: M forward passes
            for m in range(self.M):
                samples = self.pi0.forward(cond=cond, deterministic=False, return_chain=True)
                chain = samples.chains  # [B, K+1, H, A]
                
                for k in self.k_use:
                    A_k = chain[:, k]
                    energy_term = self._phi_k_from_actions(
                        poses,
                        None,
                        t_idx if t_idx is not None else torch.zeros(B, dtype=torch.long, device=device),
                        k,
                        A_k,
                    )
                    all_energy_terms.append(energy_term)

        # Stack all energy terms: [B, M*|K|]
        if all_energy_terms:
            energy_stack = torch.stack(all_energy_terms, dim=1)  # [B, M*|K|]
            
            # Apply proper log-sum-exp aggregation with normalization
            # φ(s) = (α/β) * log[1/(M*|K|) * Σ exp(-β * E)]
            #      = (α/β) * [log(Σ exp(-β * E)) - log(M*|K|)]
            log_sum_exp = self._logmeanexp(energy_stack, dim=1)  # [B]
            normalization_factor = torch.log(torch.tensor(self.M * len(self.k_use), dtype=torch.float32, device=device))
            
            phi = (self.alpha / self.beta) * (log_sum_exp - normalization_factor)
        else:
            phi = torch.zeros(B, device=device)

        return phi  # [B]

    @torch.no_grad()
    def shape_reward(
        self,
        s_t: torch.Tensor,
        s_tp1: torch.Tensor,
        gamma: float,
        t_t: Optional[torch.Tensor] = None,
        t_tp1: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute shaped reward using potential-based reward shaping.

        Args:
            s_t: Current states [B, obs_dim]
            s_tp1: Next states [B, obs_dim]
            gamma: Discount factor
            t_t: Current timesteps [B] (optional)
            t_tp1: Next timesteps [B] (optional)

        Returns:
            Shaped reward values [B]
        """
        phi_t = self.phi(s_t, t_t)
        phi_tp1 = self.phi(s_tp1, t_tp1)
        return gamma * phi_tp1 - phi_t

    def _make_zero_noise_list(self, B: int, H: int, A: int, K: int, device):
        """Create list of zero noise tensors for deterministic sampling."""
        zero = torch.zeros((B, H, A), device=device)
        return [zero.clone() for _ in range(K)]

    def _make_randn_noise_list(self, B: int, H: int, A: int, K: int, device):
        """Create list of random noise tensors for stochastic sampling."""
        return [torch.randn((B, H, A), device=device) for _ in range(K)]

    @torch.no_grad()
    def _logmeanexp(self, x: torch.Tensor, dim: int) -> torch.Tensor:
        """Compute log-mean-exp operation for numerical stability."""
        m = x.max(dim=dim, keepdim=True).values
        return (m + torch.log(torch.clamp(torch.mean(torch.exp(x - m), dim=dim, keepdim=True), min=1e-12))).squeeze(dim)


class TrainPPODiffusionEBMAgent(TrainPPODiffusionAgent):
    """
    Enhanced DPPO training agent with Energy-Based Model integration.

    This class extends TrainPPODiffusionAgent with:
    1. Potential-based reward shaping using EBMs
    2. Enhanced logging and monitoring of energy-based components
    """

    def __init__(self, cfg):
        """
        Construct the fine-tuning agent while avoiding passing nn.Module objects
        through Hydra configs. We temporarily disable PBRS in cfg to let Hydra
        instantiate the model, then inject the EBM and re-enable PBRS.
        """
        # Local import to avoid missing import if this file is used standalone
        from omegaconf import open_dict

        # ---------------------------------------------------------------------
        # 1) Read flags & prepare runtime holders
        # ---------------------------------------------------------------------
        self.use_energy_scaling = cfg.model.get("use_energy_scaling", False)
        self.use_ebm_reward_shaping = cfg.model.get("use_ebm_reward_shaping", False)

        # New: EBM scalar reward replacement per the Diffusion-MDP theory
        # If enabled, replace environment rewards with calibrated EBM utilities.
        self.use_ebm_reward = cfg.model.get("use_ebm_reward", False)
        self.ebm_reward_mode = cfg.model.get("ebm_reward_mode", "k0")  # "k0" or "dense"
        self.ebm_reward_clip_u_max = cfg.model.get("ebm_reward_clip_u_max", 5.0)
        self.ebm_reward_use_mad = cfg.model.get("ebm_reward_use_mad", True)
        # Global scale for utilities; optionally a per-k vector can be provided via cfg.model.ebm_reward_lambda_per_k
        self.ebm_reward_lambda = cfg.model.get("ebm_reward_lambda", 1.0)
        self.ebm_reward_lambda_per_k = cfg.model.get("ebm_reward_lambda_per_k", None)
        # Baseline sampling hyperparams
        self.ebm_reward_baseline_M = cfg.model.get("ebm_reward_baseline_M", 4)
        self.ebm_reward_baseline_use_mu_only = cfg.model.get("ebm_reward_baseline_use_mu_only", False)

        self.ebm_model = None            # runtime EBM (nn.Module)
        self.potential_function = None   # runtime potential φ(s)
        self.stats = None                # normalization stats used across methods

        # Load EBM once if EBM reward shaping is requested
        if self.use_ebm_reward_shaping:
            self._setup_ebm_model(cfg)

        # Cache extra configs
        if self.use_energy_scaling:
            log.info("Energy scaling enabled")
            self.energy_scaling_config = {
                "momentum": cfg.model.get("energy_scaling_momentum", 0.99),
                "eps": cfg.model.get("energy_scaling_eps", 1e-6),
                "use_mad": cfg.model.get("energy_scaling_use_mad", True),
            }

        if self.use_ebm_reward_shaping:
            log.info("EBM reward shaping enabled")
            self.ebm_config = {
                "lambda": cfg.model.get("pbrs_lambda", 1.0),
                "beta": cfg.model.get("pbrs_beta", 1.0),
                "alpha": cfg.model.get("pbrs_alpha", 1.0),
                "M": cfg.model.get("pbrs_M", 1),
                "use_mu_only": cfg.model.get("pbrs_use_mu_only", True),
                "k_use_mode": cfg.model.get("pbrs_k_use_mode", "tail:6"),
            }

        # ---------------------------------------------------------------------
        # 2) Temporarily disable EBM reward shaping in cfg before Hydra instantiation
        #    (PPODiffusionEBM.__init__ enforces ebm_model when use_ebm_reward_shaping=True)
        # ---------------------------------------------------------------------
        _orig_use_ebm_reward_shaping = bool(self.use_ebm_reward_shaping)
        if _orig_use_ebm_reward_shaping:
            with open_dict(cfg):
                cfg.model.use_ebm_reward_shaping = False

        # Parent ctor will instantiate self.model / envs / optimizers, etc.
        super().__init__(cfg)

        # ---------------------------------------------------------------------
        # 3) Inject EBM into the instantiated model and re-enable EBM reward shaping
        # ---------------------------------------------------------------------
        if _orig_use_ebm_reward_shaping and self.ebm_model is not None:
            # Prefer an explicit hook if the model provides one
            if hasattr(self.model, "set_ebm"):
                self.model.set_ebm(self.ebm_model, cfg.model.get("ebm_ckpt_path", None))
            else:
                # Fallback: attach as an attribute that the model reads
                self.model.ebm_model = self.ebm_model

            # Flip EBM reward shaping back on for the live model
            self.model.use_ebm_reward_shaping = True

            # (Optional) restore cfg flag for logging/saving consistency
            with open_dict(cfg):
                cfg.model.use_ebm_reward_shaping = True

        # ---------------------------------------------------------------------
        # 4) Build potential function (loads normalization stats into self.stats)
        # ---------------------------------------------------------------------
        if _orig_use_ebm_reward_shaping:
            self._setup_potential_function(cfg)

        # ---------------------------------------------------------------------
        # 5) Setup EBM scalar reward calibration utilities (baseline + scaler)
        # ---------------------------------------------------------------------
        self.energy_scaler_reward = None
        self.pi0_for_reward = None
        if self.use_ebm_reward:
            if self.ebm_model is None:
                # Ensure EBM is available when ebm reward is requested
                self._setup_ebm_model(cfg)
            # Build running scaler per denoising step (MAD or STD)
            from model.diffusion.energy_utils import EnergyScalerPerK
            self.energy_scaler_reward = EnergyScalerPerK(
                K=self.model.ft_denoising_steps,
                momentum=cfg.model.get("energy_scaling_momentum", 0.99),
                eps=cfg.model.get("energy_scaling_eps", 1e-6),
                use_mad=self.ebm_reward_use_mad,
            )
            # Frozen BC/base policy copy for baseline β_BC(s,k)
            self.pi0_for_reward = copy.deepcopy(self.model).eval()
            for p in self.pi0_for_reward.parameters():
                p.requires_grad = False
            log.info("Initialized EBM reward: scaler + frozen base policy baseline")

    def _setup_ebm_model(self, cfg):
        """Set up the energy-based model for PBRS."""
        # Use EBM configuration from config file if available, otherwise use defaults
        if hasattr(cfg.model, "ebm") and cfg.model.ebm is not None:
            ebm_config = cfg.model.ebm
        else:
            ebm_config = {
                "_target_": "model.ebm.EBMWrapper",
                "obs_dim": cfg.obs_dim,
                "action_dim": cfg.action_dim,
                "hidden_dim": 256,
                "num_layers": 3,
            }

        # Import and instantiate EBM model
        from hydra.utils import instantiate

        self.ebm_model = instantiate(ebm_config)

        # Load EBM checkpoint if specified
        ebm_ckpt_path = cfg.model.get("ebm_ckpt_path", None)
        assert ebm_ckpt_path is not None, "EBM checkpoint path is required"
        checkpoint = torch.load(ebm_ckpt_path, map_location=cfg.device)
        # Accept either {"model": state_dict} or a plain state_dict
        state_dict = checkpoint.get("model", checkpoint)
        self.ebm_model.load_state_dict(state_dict, strict=False)
        log.info(f"Loaded EBM from {ebm_ckpt_path}")

        self.ebm_model.to(cfg.device)
        self.ebm_model.eval()

        # Freeze EBM parameters
        for param in self.ebm_model.parameters():
            param.requires_grad = False

    def _replace_model_with_ebm(self, cfg):
        """
        (Optional) Rebuild model by passing ebm_model as ctor arg (no writing to cfg).
        Requires PPODiffusionEBM.__init__(..., ebm_model=None, ...)
        """
        from hydra.utils import instantiate

        new_model = instantiate(cfg.model, ebm_model=self.ebm_model)
        new_model.load_state_dict(self.model.state_dict(), strict=False)
        self.model = new_model
        log.info("Replaced model with EBM-enabled version")

    def _setup_potential_function(self, cfg):
        """Set up the potential function for reward shaping."""
        if not self.use_ebm_reward_shaping or self.ebm_model is None:
            return

        # Load normalization statistics
        normalization_path = self.cfg.env.wrappers.mujoco_locomotion_lowdim.normalization_path
        stats = self._load_normalization_stats(normalization_path)
        self.stats = stats  # save for _compute_energy_values

        # Get denoising steps configuration
        K = self.model.ft_denoising_steps
        k_use = build_k_use_indices(self.ebm_config["k_use_mode"], K)
        # Create frozen copy of diffusion model for prior policy
        pi0 = copy.deepcopy(self.model).eval()
        for p in pi0.parameters():
            p.requires_grad = False

        # Setup potential function
        self.potential_function = KFreePotential(
            ebm_model=self.ebm_model,
            pi0=pi0,
            K=K,
            stats=stats,
            k_use=k_use,
            beta_k=self.ebm_config["beta"],
            alpha=self.ebm_config["alpha"],
            M=self.ebm_config["M"],
            use_mu_only=self.ebm_config["use_mu_only"],
            device=self.device,
        )

        log.info(f"Setup potential function with k_use={k_use}")

    def _compute_energy_values_generic(self, obs: torch.Tensor, chains: torch.Tensor, t_idx_vec: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        """
        Compute EBM energies per denoising step for given obs and action chains.

        Args:
            obs:    [B, obs_dim]
            chains: [B, K+1, H, A]

        Returns:
            energies: [B, K+1] or None if EBM/stats unavailable
        """
        if self.ebm_model is None or self.stats is None:
            return None

        assert chains.dim() == 4, f"chains must be [B, K+1, H, A], got {chains.shape}"
        B, K_plus_1, H, A = chains.shape

        # Prepare normalization tensors once
        act_min = torch.tensor(self.stats["act_min"], dtype=torch.float32, device=self.device)
        act_max = torch.tensor(self.stats["act_max"], dtype=torch.float32, device=self.device)
        if act_min.ndim == 2:
            act_min = act_min.min(dim=0).values
        if act_max.ndim == 2:
            act_max = act_max.max(dim=0).values
        act_min = act_min.view(1, 1, -1)  # [1,1,A]
        act_max = act_max.view(1, 1, -1)  # [1,1,A]

        energy_values = torch.zeros(B, K_plus_1, device=self.device)
        with torch.no_grad():
            for k in range(0, K_plus_1):
                # Map k=0 to the final (fully denoised) action
                idx = K_plus_1 - 1 - k
                actions_k = chains[:, idx, :, :]  # [B, H, A]
                actions_norm = torch.clamp((actions_k - act_min) / (act_max - act_min + 1e-6) * 2 - 1, -1.0, 1.0)
                k_vec = torch.full((B,), k, dtype=torch.long, device=self.device)
                poses = obs[:, -1, :] if obs.dim() == 3 else obs
                if t_idx_vec is None:
                    t_vec = torch.zeros(B, dtype=torch.long, device=self.device)
                else:
                    t_vec = t_idx_vec.to(self.device).long().view(-1)
                E_out = self.ebm_model(k_idx=k_vec, t_idx=t_vec, views=None, poses=poses, actions=actions_norm)
                E = E_out[0] if isinstance(E_out, (tuple, list)) else E_out
                if E.dim() >= 2 and E.size(0) == B:
                    E = E.mean(dim=tuple(range(1, E.dim())))
                energy_values[:, k] = E
        return energy_values

    def _compute_ebm_reward_from_chains(self, obs_state_np, chains_np, t_idx_np: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Compute calibrated EBM reward R_t from energies, replacing env reward.

        Implements: u(t,k) = -(E(s,a_k,k) - beta_BC(s,k)) / tau_k,
        with tau_k from running MAD/STD over baseline energies per k.

        Modes:
          - k0: use only k=0 utility
          - dense: sum_k lambda_k * gamma_denoise^k * u(t,k)
        """
        if self.ebm_model is None or self.energy_scaler_reward is None or self.pi0_for_reward is None:
            return np.zeros((chains_np.shape[0],), dtype=np.float32)

        # Tensors
        obs_b = torch.from_numpy(obs_state_np).float().to(self.device)
        chains_curr = torch.from_numpy(chains_np).float().to(self.device)
        t_idx_vec = None
        if t_idx_np is not None:
            t_idx_vec = torch.from_numpy(t_idx_np).to(self.device).long()

        # Current energies per k
        E_curr = self._compute_energy_values_generic(obs_b, chains_curr, t_idx_vec)  # [B, K+1]
        if E_curr is None:
            return np.zeros((chains_np.shape[0],), dtype=np.float32)

        # Baseline chains from frozen base policy with robust aggregation over M samples
        with torch.no_grad():
            cond = {"state": obs_b}
            E_base_list = []
            M = max(1, int(self.ebm_reward_baseline_M))
            for m in range(M):
                det = bool(self.ebm_reward_baseline_use_mu_only)
                samples_bc = self.pi0_for_reward(cond=cond, deterministic=det, return_chain=True)
                chains_bc = samples_bc.chains  # [B, K+1, H, A]
                E_b = self._compute_energy_values_generic(obs_b, chains_bc, t_idx_vec)
                if E_b is None:
                    continue
                E_base_list.append(E_b.unsqueeze(0))
        if len(E_base_list) == 0:
            return np.zeros((chains_np.shape[0],), dtype=np.float32)
        E_base_stack = torch.cat(E_base_list, dim=0)  # [M,B,K+1]
        # median over samples for robustness
        E_base = E_base_stack.median(dim=0).values  # [B,K+1]
        if E_base is None:
            return np.zeros((chains_np.shape[0],), dtype=np.float32)

        B, K_plus_1 = E_curr.shape
        # Update scaler with baseline energies (per-k running scale)
        with torch.no_grad():
            for k in range(0, K_plus_1):
                k_vec = torch.full((B,), k, dtype=torch.long)
                self.energy_scaler_reward.update(k_vec, E_base[:, k].detach().cpu())

        # Compute utilities
        u = torch.zeros(B, K_plus_1, device=self.device)
        s_vec = self.energy_scaler_reward.s.to(self.device)  # length K+1
        eps = 1e-6
        for k in range(0, K_plus_1):
            tau_k = torch.clamp(s_vec[k], min=1e-3)
            u[:, k] = -(E_curr[:, k] - E_base[:, k]) / (tau_k + eps)
            u[:, k] = torch.clamp(u[:, k], -float(self.ebm_reward_clip_u_max), float(self.ebm_reward_clip_u_max))

        if self.ebm_reward_mode == "dense":
            # Build lambda_k
            if isinstance(self.ebm_reward_lambda_per_k, (list, tuple)):
                lam = torch.tensor(list(self.ebm_reward_lambda_per_k)[:K_plus_1], dtype=torch.float32, device=self.device)
                if lam.numel() < K_plus_1:
                    lam = torch.nn.functional.pad(lam, (0, K_plus_1 - lam.numel()), value=float(self.ebm_reward_lambda))
            else:
                lam = torch.full((K_plus_1,), float(self.ebm_reward_lambda), device=self.device)
            # gamma_denoise^k weights
            gamma_dn = float(self.model.gamma_denoising)
            w = torch.tensor([gamma_dn ** k for k in range(K_plus_1)], dtype=torch.float32, device=self.device)
            rew = (u * lam.view(1, -1) * w.view(1, -1)).sum(dim=1)
        else:
            # k0 only
            rew = u[:, 0]

        return rew.detach().cpu().numpy().astype(np.float32)

    def _load_normalization_stats(self, normalization_path):
        """Load normalization statistics from file."""
        if normalization_path.endswith(".npz"):
            data = np.load(normalization_path)
            return {
                "obs_min": data["obs_min"],
                "obs_max": data["obs_max"],
                "act_min": data["action_min"],
                "act_max": data["action_max"],
            }
        elif normalization_path.endswith(".pkl"):
            with open(normalization_path, "rb") as f:
                return pickle.load(f)
        else:
            raise ValueError(f"Unsupported normalization file format: {normalization_path}")

    def _compute_energy_values(self, obs, chains, denoising_inds):
        """
        Compute energy values for each denoising step (0-based indexing).

        Args:
            obs:     Tensor [B, obs_dim] on self.device.
            chains:  Tensor [B, K+1, H, A] with indices 0..K, where k=0 is the final (fully denoised) action.
            denoising_inds: (unused here; kept for signature compatibility)

        Returns:
            Tensor [B, K+1] where column k contains the energy for step k (k=0..K).
        """
        if not self.use_ebm_reward_shaping or self.ebm_model is None or self.stats is None:
            return None

        assert chains.dim() == 4, f"chains must be [B, K+1, H, A], got {chains.shape}"
        B, K_plus_1, H, A = chains.shape
        K = K_plus_1 - 1  # max k value

        # Prepare normalization tensors once
        act_min = torch.tensor(self.stats["act_min"], dtype=torch.float32, device=self.device)
        act_max = torch.tensor(self.stats["act_max"], dtype=torch.float32, device=self.device)
        if act_min.ndim == 2:
            act_min = act_min.min(dim=0).values
        if act_max.ndim == 2:
            act_max = act_max.max(dim=0).values
        act_min = act_min.view(1, 1, -1)  # [1,1,A]
        act_max = act_max.view(1, 1, -1)  # [1,1,A]

        energy_values = torch.zeros(B, K_plus_1, device=self.device)

        with torch.no_grad():
            for k in range(0, K + 1):
                # actions at denoising step k (0-based; k=0 is final action)
                actions_k = chains[:, k, :, :]  # [B, H, A]

                # Normalize actions to [-1, 1]
                actions_norm = torch.clamp((actions_k - act_min) / (act_max - act_min + 1e-6) * 2 - 1, -1.0, 1.0)

                # EBM energy (pass t_idx explicitly for consistency)
                k_vec = torch.full((B,), k, dtype=torch.long, device=self.device)
                t_vec = torch.zeros(B, dtype=torch.long, device=self.device)
                E_out = self.ebm_model(k_idx=k_vec, t_idx=t_vec, views=None, poses=obs, actions=actions_norm)
                E = E_out[0] if isinstance(E_out, (tuple, list)) else E_out

                # If E has extra dims (e.g., per-horizon), average them
                if E.dim() >= 2 and E.size(0) == B:
                    E = E.mean(dim=tuple(range(1, E.dim())))

                energy_values[:, k] = E

        return energy_values


    def _compute_pbrs_reward(self, obs_t, obs_tp1, gamma=0.99):
        """
        Compute potential-based reward shaping reward.

        Args:
            obs_t: Current observations [B, obs_dim]
            obs_tp1: Next observations [B, obs_dim]
            gamma: Discount factor

        Returns:
            Shaped reward values [B]
        """
        if not self.use_ebm_reward_shaping or self.potential_function is None:
            return torch.zeros(obs_t.size(0), device=self.device)

        shaped_reward = self.potential_function.shape_reward(obs_t, obs_tp1, gamma)
        return self.ebm_config["lambda"] * shaped_reward

    def _log_energy_scaling_stats(self, step):
        """No-op: current potential does not expose per-k scaler stats."""
        return

    def run(self):
        """Enhanced training loop with EBM integration (rollout-time PBRS)."""
        timer = Timer()
        run_results = []
        cnt_train_step = 0
        last_itr_eval = False
        done_venv = np.zeros((1, self.n_envs))
        while self.itr < self.n_train_itr:
            # Prepare video paths for each envs --- only applies for the first set of episodes if allowing reset within iteration and each iteration has multiple episodes from one env
            options_venv = [{} for _ in range(self.n_envs)]
            if self.itr % self.render_freq == 0 and self.render_video:
                for env_ind in range(self.n_render):
                    options_venv[env_ind]["video_path"] = os.path.join(
                        self.render_dir, f"itr-{self.itr}_trial-{env_ind}.mp4"
                    )

            # Define train or eval - all envs restart
            eval_mode = self.itr % self.val_freq == 0 and not self.force_train
            self.model.eval() if eval_mode else self.model.train()
            last_itr_eval = eval_mode

            # Reset env before iteration starts (1) if specified, (2) at eval mode, or (3) right after eval mode
            firsts_trajs = np.zeros((self.n_steps + 1, self.n_envs))
            if self.reset_at_iteration or eval_mode or last_itr_eval:
                prev_obs_venv = self.reset_env_all(options_venv=options_venv)
                firsts_trajs[0] = 1
            else:
                # if done at the end of last iteration, the envs are just reset
                firsts_trajs[0] = done_venv

            # Holder
            obs_trajs = {
                "state": np.zeros(
                    (self.n_steps, self.n_envs, self.n_cond_step, self.obs_dim)
                )
            }
            chains_trajs = np.zeros(
                (
                    self.n_steps,
                    self.n_envs,
                    self.model.ft_denoising_steps + 1,
                    self.horizon_steps,
                    self.action_dim,
                )
            )
            terminated_trajs = np.zeros((self.n_steps, self.n_envs))
            reward_trajs = np.zeros((self.n_steps, self.n_envs))
            # Track raw env reward separately for fair eval logging when EBM reward replaces env reward
            env_reward_trajs = np.zeros((self.n_steps, self.n_envs))
            pbrs_reward_trajs = np.zeros((self.n_steps, self.n_envs))
            if self.save_full_observations:  # state-only
                obs_full_trajs = np.empty((0, self.n_envs, self.obs_dim))
                obs_full_trajs = np.vstack(
                    (obs_full_trajs, prev_obs_venv["state"][:, -1][None])
                )

            # Collect a set of trajectories from env
            for step in range(self.n_steps):
                if step % 10 == 0:
                    print(f"Processed step {step} of {self.n_steps}")

                # Select action
                with torch.no_grad():
                    # Ensure state is a proper tensor with correct shape and type
                    state_tensor = torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)
                    if not state_tensor.is_contiguous():
                        state_tensor = state_tensor.contiguous()
                    
                    cond = {
                        "state": state_tensor
                    }
                    samples = self.model(
                        cond=cond,
                        deterministic=eval_mode,
                        return_chain=True,
                    )
                    output_venv = samples.trajectories.cpu().numpy()  # n_env x horizon x act
                    chains_venv = samples.chains.cpu().numpy()  # n_env x denoising x horizon x act
                action_venv = output_venv[:, : self.act_steps]

                # Apply multi-step action
                (
                    obs_venv,
                    reward_venv,
                    terminated_venv,
                    truncated_venv,
                    info_venv,
                ) = self.venv.step(action_venv)
                done_venv = terminated_venv | truncated_venv

                # Cache raw env reward before any replacement (for fair logging)
                env_reward_trajs[step] = reward_venv

                # Replace env reward with calibrated EBM utility if enabled
                if self.use_ebm_reward and self.ebm_model is not None:
                    # t_idx: use global env step index approximation: itr*n_steps + step
                    t_idx_np = np.full((self.n_envs,), int(self.itr * self.n_steps + step), dtype=np.int64)
                    ebm_rew = self._compute_ebm_reward_from_chains(
                        prev_obs_venv["state"], chains_venv, t_idx_np
                    )
                    reward_venv = ebm_rew

                # PBRS: r += lambda * (gamma * phi(s') - phi(s)) (optional)
                if self.use_ebm_reward_shaping and self.potential_function is not None:
                    with torch.no_grad():
                        # Ensure tensors are properly formatted
                        s_t = torch.from_numpy(prev_obs_venv["state"]).float().to(self.device)
                        s_tp1 = torch.from_numpy(obs_venv["state"]).float().to(self.device)
                        
                        # Ensure tensors are contiguous
                        if not s_t.is_contiguous():
                            s_t = s_t.contiguous()
                        if not s_tp1.is_contiguous():
                            s_tp1 = s_tp1.contiguous()
                        
                        r_shape = self.potential_function.shape_reward(s_t, s_tp1, self.gamma)
                        r_shape = r_shape.detach().cpu().numpy()
                    reward_venv = reward_venv + self.ebm_config["lambda"] * r_shape
                    pbrs_reward_trajs[step] = r_shape

                if self.save_full_observations:  # state-only
                    obs_full_venv = np.array(
                        [info["full_obs"]["state"] for info in info_venv]
                    )  # n_envs x act_steps x obs_dim
                    obs_full_trajs = np.vstack(
                        (obs_full_trajs, obs_full_venv.transpose(1, 0, 2))
                    )
                obs_trajs["state"][step] = prev_obs_venv["state"]
                chains_trajs[step] = chains_venv
                reward_trajs[step] = reward_venv
                terminated_trajs[step] = terminated_venv
                firsts_trajs[step + 1] = done_venv

                # update for next step
                prev_obs_venv = obs_venv

                # count steps --- not acounting for done within action chunk
                cnt_train_step += self.n_envs * self.act_steps if not eval_mode else 0

            # Summarize episode reward --- this needs to be handled differently depending on whether the environment is reset after each iteration. Only count episodes that finish within the iteration.
            episodes_start_end = []
            for env_ind in range(self.n_envs):
                env_steps = np.where(firsts_trajs[:, env_ind] == 1)[0]
                for i in range(len(env_steps) - 1):
                    start = env_steps[i]
                    end = env_steps[i + 1]
                    if end - start > 1:
                        episodes_start_end.append((env_ind, start, end - 1))
            if len(episodes_start_end) > 0:
                reward_trajs_split = [
                    reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                env_reward_trajs_split = [
                    env_reward_trajs[start : end + 1, env_ind]
                    for env_ind, start, end in episodes_start_end
                ]
                num_episode_finished = len(reward_trajs_split)
                episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in reward_trajs_split]
                )
                env_episode_reward = np.array(
                    [np.sum(reward_traj) for reward_traj in env_reward_trajs_split]
                )
                if (
                    self.furniture_sparse_reward
                ):  # only for furniture tasks, where reward only occurs in one env step
                    episode_best_reward = episode_reward
                else:
                    episode_best_reward = np.array(
                        [
                            np.max(reward_traj) / self.act_steps
                            for reward_traj in reward_trajs_split
                        ]
                    )
                avg_episode_reward = np.mean(episode_reward)
                avg_env_episode_reward = float(np.mean(env_episode_reward)) if num_episode_finished > 0 else 0.0
                avg_best_reward = np.mean(episode_best_reward)
                success_rate = np.mean(
                    episode_best_reward >= self.best_reward_threshold_for_success
                )
            else:
                episode_reward = np.array([])
                num_episode_finished = 0
                avg_episode_reward = 0
                avg_env_episode_reward = 0
                avg_best_reward = 0
                success_rate = 0
                log.info("[WARNING] No episode completed within the iteration!")

            # Update models
            if not eval_mode:
                with torch.no_grad():
                    obs_trajs["state"] = (
                        torch.from_numpy(obs_trajs["state"]).float().to(self.device)
                    )

                    # Calculate value and logprobs - split into batches to prevent out of memory
                    num_split = math.ceil(
                        self.n_envs * self.n_steps / self.logprob_batch_size
                    )
                    obs_ts = [{} for _ in range(num_split)]
                    obs_k = einops.rearrange(
                        obs_trajs["state"],
                        "s e ... -> (s e) ...",
                    )
                    obs_ts_k = torch.split(obs_k, self.logprob_batch_size, dim=0)
                    for i, obs_t in enumerate(obs_ts_k):
                        obs_ts[i]["state"] = obs_t
                    values_trajs = np.empty((0, self.n_envs))
                    for obs in obs_ts:
                        values = self.model.critic(obs).cpu().numpy().flatten()
                        values_trajs = np.vstack(
                            (values_trajs, values.reshape(-1, self.n_envs))
                        )
                    chains_t = einops.rearrange(
                        torch.from_numpy(chains_trajs).float().to(self.device),
                        "s e t h d -> (s e) t h d",
                    )
                    chains_ts = torch.split(chains_t, self.logprob_batch_size, dim=0)
                    logprobs_trajs = np.empty(
                        (
                            0,
                            self.model.ft_denoising_steps,
                            self.horizon_steps,
                            self.action_dim,
                        )
                    )
                    for obs, chains in zip(obs_ts, chains_ts):
                        logprobs = self.model.get_logprobs(obs, chains).cpu().numpy()
                        logprobs_trajs = np.vstack(
                            (
                                logprobs_trajs,
                                logprobs.reshape(-1, *logprobs_trajs.shape[1:]),
                            )
                        )

                    # normalize reward with running variance if specified
                    if self.reward_scale_running:
                        reward_trajs_transpose = self.running_reward_scaler(
                            reward=reward_trajs.T, first=firsts_trajs[:-1].T
                        )
                        reward_trajs = reward_trajs_transpose.T

                    # bootstrap value with GAE if not terminal - apply reward scaling with constant if specified
                    obs_venv_ts = {
                        "state": torch.from_numpy(obs_venv["state"]).float().to(self.device)
                    }
                    advantages_trajs = np.zeros_like(reward_trajs)
                    lastgaelam = 0
                    for t in reversed(range(self.n_steps)):
                        if t == self.n_steps - 1:
                            nextvalues = (
                                self.model.critic(obs_venv_ts)
                                .reshape(1, -1)
                                .cpu()
                                .numpy()
                            )
                        else:
                            nextvalues = values_trajs[t + 1]
                        nonterminal = 1.0 - terminated_trajs[t]
                        # delta = r + gamma*V(st+1) - V(st)
                        delta = (
                            reward_trajs[t] * self.reward_scale_const
                            + self.gamma * nextvalues * nonterminal
                            - values_trajs[t]
                        )
                        # A = delta_t + gamma*lamdba*delta_{t+1} + ...
                        advantages_trajs[t] = lastgaelam = (
                            delta
                            + self.gamma * self.gae_lambda * nonterminal * lastgaelam
                        )
                    returns_trajs = advantages_trajs + values_trajs

            if not eval_mode:
                # k for environment step
                obs_k = {
                    "state": einops.rearrange(
                        obs_trajs["state"],
                        "s e ... -> (s e) ...",
                    )
                }
                chains_k = einops.rearrange(
                    torch.tensor(chains_trajs, device=self.device).float(),
                    "s e t h d -> (s e) t h d",
                )
                returns_k = (
                    torch.tensor(returns_trajs, device=self.device).float().reshape(-1)
                )
                values_k = (
                    torch.tensor(values_trajs, device=self.device).float().reshape(-1)
                )
                advantages_k = (
                    torch.tensor(advantages_trajs, device=self.device)
                    .float()
                    .reshape(-1)
                )
                logprobs_k = torch.tensor(logprobs_trajs, device=self.device).float()

                # Update policy and critic
                total_steps = self.n_steps * self.n_envs * self.model.ft_denoising_steps
                clipfracs = []
                for update_epoch in range(self.update_epochs):
                    # for each epoch, go through all data in batches
                    flag_break = False
                    inds_k = torch.randperm(total_steps, device=self.device)
                    num_batch = max(1, total_steps // self.batch_size)  # skip last ones
                    for batch in range(num_batch):
                        start = batch * self.batch_size
                        end = start + self.batch_size
                        inds_b = inds_k[start:end]  # b for batch
                        batch_inds_b, denoising_inds_b = unravel_index(
                            inds_b,
                            (self.n_steps * self.n_envs, self.model.ft_denoising_steps),
                        )
                        obs_b = {"state": obs_k["state"][batch_inds_b]}
                        chains_prev_b = chains_k[batch_inds_b, denoising_inds_b]
                        chains_next_b = chains_k[batch_inds_b, denoising_inds_b + 1]
                        returns_b = returns_k[batch_inds_b]
                        values_b = values_k[batch_inds_b]
                        advantages_b = advantages_k[batch_inds_b]
                        logprobs_b = logprobs_k[batch_inds_b, denoising_inds_b]

                        # get loss
                        # Note: denoising-step advantage weighting is handled inside the model loss
                        (
                            pg_loss,
                            entropy_loss,
                            v_loss,
                            clipfrac,
                            approx_kl,
                            ratio,
                            bc_loss,
                            eta,
                        ) = self.model.loss(
                            obs_b,
                            chains_prev_b,
                            chains_next_b,
                            denoising_inds_b,
                            returns_b,
                            values_b,
                            advantages_b,
                            logprobs_b,
                            use_bc_loss=self.use_bc_loss,
                            reward_horizon=self.reward_horizon,
                        )
                        loss = (
                            pg_loss
                            + entropy_loss * self.ent_coef
                            + v_loss * self.vf_coef
                            + bc_loss * self.bc_loss_coeff
                        )
                        clipfracs += [clipfrac]

                        # update policy and critic
                        self.actor_optimizer.zero_grad()
                        self.critic_optimizer.zero_grad()
                        if self.learn_eta:
                            self.eta_optimizer.zero_grad()
                        loss.backward()
                        if self.itr >= self.n_critic_warmup_itr:
                            if self.max_grad_norm is not None:
                                torch.nn.utils.clip_grad_norm_(
                                    self.model.actor_ft.parameters(), self.max_grad_norm
                                )
                            self.actor_optimizer.step()
                            if self.learn_eta and batch % self.eta_update_interval == 0:
                                self.eta_optimizer.step()
                        self.critic_optimizer.step()
                        log.info(
                            f"approx_kl: {approx_kl}, update_epoch: {update_epoch}, num_batch: {num_batch}"
                        )

                        # Stop gradient update if KL difference reaches target
                        if self.target_kl is not None and approx_kl > self.target_kl:
                            flag_break = True
                            break
                    if flag_break:
                        break

                # Explained variation of future rewards using value function
                y_pred, y_true = values_k.cpu().numpy(), returns_k.cpu().numpy()
                var_y = np.var(y_true)
                explained_var = (
                    np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                )

            # Plot state trajectories (only in D3IL)
            if (
                self.itr % self.render_freq == 0
                and self.n_render > 0
                and self.traj_plotter is not None
            ):
                self.traj_plotter(
                    obs_full_trajs=obs_full_trajs,
                    n_render=self.n_render,
                    max_episode_steps=self.max_episode_steps,
                    render_dir=self.render_dir,
                    itr=self.itr,
                )

            # Update lr, min_sampling_std
            if self.itr >= self.n_critic_warmup_itr:
                self.actor_lr_scheduler.step()
                if self.learn_eta:
                    self.eta_lr_scheduler.step()
            self.critic_lr_scheduler.step()
            self.model.step()
            diffusion_min_sampling_std = self.model.get_min_sampling_denoising_std()

            # Save model
            if self.itr % self.save_model_freq == 0 or self.itr == self.n_train_itr - 1:
                self.save_model()

            # Log loss and save metrics
            run_results.append(
                {
                    "itr": self.itr,
                    "step": cnt_train_step,
                }
            )
            if self.save_trajs:
                run_results[-1]["obs_full_trajs"] = obs_full_trajs
                run_results[-1]["obs_trajs"] = obs_trajs
                run_results[-1]["chains_trajs"] = chains_trajs
                run_results[-1]["reward_trajs"] = reward_trajs
                run_results[-1]["pbrs_reward_trajs"] = pbrs_reward_trajs
            if self.itr % self.log_freq == 0:
                time = timer()
                run_results[-1]["time"] = time
                if eval_mode:
                    log.info(
                        f"eval: success rate {success_rate:8.4f} | avg episode reward {avg_episode_reward:8.4f} | avg env episode reward {avg_env_episode_reward:8.4f} | avg best reward {avg_best_reward:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "success rate - eval": success_rate,
                                "avg episode reward - eval": avg_episode_reward,
                                "avg episode env reward - eval": avg_env_episode_reward,
                                "avg best reward - eval": avg_best_reward,
                                "num episode - eval": num_episode_finished,
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["eval_success_rate"] = success_rate
                    run_results[-1]["eval_episode_reward"] = avg_episode_reward
                    run_results[-1]["eval_best_reward"] = avg_best_reward
                else:
                    log.info(
                        f"{self.itr}: step {cnt_train_step:8d} | loss {loss:8.4f} | pg loss {pg_loss:8.4f} | value loss {v_loss:8.4f} | bc loss {bc_loss:8.4f} | reward {avg_episode_reward:8.4f} | env reward {avg_env_episode_reward:8.4f} | eta {eta:8.4f} | t:{time:8.4f}"
                    )
                    if self.use_wandb:
                        wandb.log(
                            {
                                "total env step": cnt_train_step,
                                "loss": loss,
                                "pg loss": pg_loss,
                                "value loss": v_loss,
                                "bc loss": bc_loss,
                                "eta": eta,
                                "approx kl": approx_kl,
                                "ratio": ratio,
                                "clipfrac": np.mean(clipfracs),
                                "explained variance": explained_var,
                                "avg episode reward - train": avg_episode_reward,
                                "avg episode env reward - train": avg_env_episode_reward,
                                "num episode - train": num_episode_finished,
                                "diffusion - min sampling std": diffusion_min_sampling_std,
                                "actor lr": self.actor_optimizer.param_groups[0]["lr"],
                                "critic lr": self.critic_optimizer.param_groups[0]["lr"],
                            },
                            step=self.itr,
                            commit=True,
                        )
                    run_results[-1]["train_episode_reward"] = avg_episode_reward
                with open(self.result_path, "wb") as f:
                    pickle.dump(run_results, f)
            self.itr += 1
           