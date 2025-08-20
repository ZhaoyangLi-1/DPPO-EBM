"""
Soft Actor-Critic (SAC) with Energy-Based Model integration for diffusion policy.

This module extends the standard SAC diffusion implementation with energy-based model
capabilities, specifically for potential-based reward shaping.
"""

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import logging

log = logging.getLogger(__name__)
from model.diffusion.diffusion_sac import SACDiffusion
from model.diffusion.energy_utils import KFreePotential, build_k_use_indices


class SACDiffusionEBM(SACDiffusion):
    """
    Enhanced SACDiffusion with Energy-Based Model integration.
    
    This class extends SACDiffusion with potential-based reward shaping using EBM.
    """
    
    def __init__(
        self,
        # SAC parameters
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        target_entropy: Optional[float] = None,
        # EBM reward shaping parameters
        use_ebm_reward_shaping: bool = False,
        ebm_model: Optional[nn.Module] = None,
        ebm_ckpt_path: Optional[str] = None,
        pbrs_lambda: float = 1.0,
        pbrs_beta: float = 1.0,
        pbrs_alpha: float = 1.0,
        pbrs_M: int = 1,
        pbrs_use_mu_only: bool = True,
        pbrs_k_use_mode: str = "tail:6",
        # Energy scaling parameters
        use_energy_scaling: bool = True,
        energy_scaling_momentum: float = 0.99,
        energy_scaling_eps: float = 1e-6,
        energy_scaling_use_mad: bool = True,
        **kwargs,
    ):
        super().__init__(
            gamma=gamma,
            tau=tau,
            alpha=alpha,
            automatic_entropy_tuning=automatic_entropy_tuning,
            target_entropy=target_entropy,
            **kwargs,
        )
        
        # EBM reward shaping parameters
        self.use_ebm_reward_shaping = use_ebm_reward_shaping
        self.pbrs_lambda = pbrs_lambda
        self.pbrs_beta = pbrs_beta
        self.pbrs_alpha = pbrs_alpha
        self.pbrs_M = pbrs_M
        self.pbrs_use_mu_only = pbrs_use_mu_only
        self.pbrs_k_use_mode = pbrs_k_use_mode
        
        # Energy scaling parameters
        self.use_energy_scaling = use_energy_scaling
        self.energy_scaling_momentum = energy_scaling_momentum
        self.energy_scaling_eps = energy_scaling_eps
        self.energy_scaling_use_mad = energy_scaling_use_mad
        
        # Initialize EBM and potential function if enabled
        if self.use_ebm_reward_shaping:
            if ebm_model is None:
                raise ValueError("EBM model must be provided when use_ebm_reward_shaping=True")
            
            self.ebm_model = ebm_model
            if ebm_ckpt_path is not None:
                checkpoint = torch.load(ebm_ckpt_path, map_location=self.device)
                self.ebm_model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
                log.info(f"Loaded EBM from {ebm_ckpt_path}")
            
            # Build k_use indices for potential computation
            self.pbrs_k_use = build_k_use_indices(self.pbrs_k_use_mode, self.ft_denoising_steps)
            
            # Log k_use validation
            log.info(f"SACDiffusionEBM initialization: ft_denoising_steps={self.ft_denoising_steps}, pbrs_k_use={self.pbrs_k_use}, valid range: 0 to {self.ft_denoising_steps-1}")
        
        # Initialize potential function (will be set up in setup_potential_function)
        self.potential_function = None
        self.stats = None
        
        log.info(f"Initialized SAC EBM reward shaping with k_use_mode={self.pbrs_k_use_mode}")
    
    def setup_potential_function(self, stats: Dict[str, Any]):
        """
        Set up the potential function for reward shaping.
        
        Args:
            stats: Normalization statistics containing action bounds
        """
        if not self.use_ebm_reward_shaping:
            return
            
        self.stats = stats
        
        # Create a frozen copy of the current policy as prior
        pi0 = self.actor.eval()
        for p in pi0.parameters():
            p.requires_grad = False
        
        # Setup potential function
        self.potential_function = KFreePotential(
            ebm_model=self.ebm_model,
            pi0=pi0,
            K=self.ft_denoising_steps,
            stats=stats,
            k_use=self.pbrs_k_use,
            beta_k=self.pbrs_beta,
            alpha=self.pbrs_alpha,
            M=self.pbrs_M,
            use_mu_only=self.pbrs_use_mu_only,
            device=self.device,
        )
        
        log.info(f"SACDiffusionEBM setup_potential_function: ft_denoising_steps={self.ft_denoising_steps}, k_use={self.pbrs_k_use}, valid range: 0 to {self.ft_denoising_steps-1}")
    
    def compute_shaped_reward(self, obs_t: torch.Tensor, obs_tp1: torch.Tensor) -> torch.Tensor:
        """
        Compute shaped reward using EBM potential-based reward shaping.
        
        Args:
            obs_t: Current observations [B, obs_dim]
            obs_tp1: Next observations [B, obs_dim]
            
        Returns:
            Shaped reward values [B]
        """
        if not self.use_ebm_reward_shaping or self.potential_function is None:
            return torch.zeros(obs_t.size(0), device=obs_t.device)
        
        try:
            shaped_reward = self.potential_function.shape_reward(
                obs_t, obs_tp1, self.gamma
            )
            return self.pbrs_lambda * shaped_reward
        except Exception as e:
            log.warning(f"Error in EBM reward shaping: {e}")
            return torch.zeros(obs_t.size(0), device=obs_t.device)
    
    def compute_energy_values(self, obs: torch.Tensor, chains: torch.Tensor) -> Optional[torch.Tensor]:
        """
        Compute energy values for each denoising step.
        
        Args:
            obs: Observations [B, obs_dim]
            chains: Action chains [B, K+1, H, A]
            
        Returns:
            Energy values [B, K+1] or None if EBM not enabled
        """
        if not self.use_ebm_reward_shaping or self.ebm_model is None or self.stats is None:
            return None
        
        try:
            B, K_plus_1, H, A = chains.shape
            K = K_plus_1 - 1
            
            # Prepare normalization tensors
            act_min = torch.tensor(self.stats["act_min"], dtype=torch.float32, device=self.device)
            act_max = torch.tensor(self.stats["act_max"], dtype=torch.float32, device=self.device)
            if act_min.ndim == 2:
                act_min = act_min.min(dim=0).values
            if act_max.ndim == 2:
                act_max = act_max.max(dim=0).values
            act_min = act_min.view(1, 1, -1)
            act_max = act_max.view(1, 1, -1)
            
            energy_values = torch.zeros(B, K, device=self.device)
            
            with torch.no_grad():
                # Log k range validation (only once)
                if not hasattr(self, '_logged_k_range'):
                    log.info(f"SAC EBM k range validation: K={K}, valid range: 0 to {K-1}")
                    self._logged_k_range = True
                
                for k in range(K):
                    actions_k = chains[:, k, :, :]
                    
                    # Normalize actions to [-1, 1]
                    actions_norm = torch.clamp(
                        (actions_k - act_min) / (act_max - act_min + 1e-6) * 2 - 1, 
                        -1.0, 1.0
                    )
                    
                    # Compute EBM energy
                    k_vec = torch.full((B,), k, dtype=torch.long, device=self.device)
                    E_out = self.ebm_model(k_idx=k_vec, views=None, poses=obs, actions=actions_norm)
                    E = E_out[0] if isinstance(E_out, (tuple, list)) else E_out
                    
                    # Average over additional dimensions if present
                    if E.dim() >= 2 and E.size(0) == B:
                        E = E.mean(dim=tuple(range(1, E.dim())))
                    
                    energy_values[:, k] = E
            
            return energy_values
            
        except Exception as e:
            log.warning(f"Error in energy computation: {e}")
            return None
