"""
Enhanced DPPO with Energy-Based Model integration.

This module extends the standard PPODiffusion class with energy-based model
capabilities, including EnergyScalerPerK for per-step energy normalization
and potential-based reward shaping.
"""

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import logging
import math

log = logging.getLogger(__name__)
from model.diffusion.diffusion_ppo import PPODiffusion
from model.diffusion.energy_utils import EnergyScalerPerK, KFreePotential, build_k_use_indices


class PPODiffusionEBM(PPODiffusion):
    """
    Enhanced PPODiffusion with Energy-Based Model integration.
    
    This class extends PPODiffusion with:
    1. EnergyScalerPerK for per-step energy normalization
    2. KFreePotential for potential-based reward shaping
    3. Enhanced loss computation with energy-based components
    """
    
    def __init__(
        self,
        gamma_denoising: float,
        clip_ploss_coef: float,
        clip_ploss_coef_base: float = 1e-3,
        clip_ploss_coef_rate: float = 3,
        clip_vloss_coef: Optional[float] = None,
        clip_advantage_lower_quantile: float = 0,
        clip_advantage_upper_quantile: float = 1,
        norm_adv: bool = True,
        # EBM-specific parameters
        use_energy_scaling: bool = False,
        energy_scaling_momentum: float = 0.99,
        energy_scaling_eps: float = 1e-6,
        energy_scaling_use_mad: bool = True,
        # Potential-based reward shaping parameters
        use_pbrs: bool = False,
        pbrs_lambda: float = 1.0,
        pbrs_beta: float = 1.0,
        pbrs_alpha: float = 1.0,
        pbrs_M: int = 1,
        pbrs_eta: float = 0.3,
        pbrs_use_mu_only: bool = True,
        pbrs_k_use_mode: str = "tail:6",
        # EBM model (optional)
        ebm_model: Optional[nn.Module] = None,
        ebm_ckpt_path: Optional[str] = None,
        **kwargs,
    ):
        super().__init__(
            gamma_denoising=gamma_denoising,
            clip_ploss_coef=clip_ploss_coef,
            clip_ploss_coef_base=clip_ploss_coef_base,
            clip_ploss_coef_rate=clip_ploss_coef_rate,
            clip_vloss_coef=clip_vloss_coef,
            clip_advantage_lower_quantile=clip_advantage_lower_quantile,
            clip_advantage_upper_quantile=clip_advantage_upper_quantile,
            norm_adv=norm_adv,
            **kwargs,
        )
        
        # Energy scaling parameters
        self.use_energy_scaling = use_energy_scaling
        self.energy_scaling_momentum = energy_scaling_momentum
        self.energy_scaling_eps = energy_scaling_eps
        self.energy_scaling_use_mad = energy_scaling_use_mad
        
        # Initialize energy scaler if enabled
        if self.use_energy_scaling:
            self.energy_scaler = EnergyScalerPerK(
                K=self.denoising_steps,
                momentum=self.energy_scaling_momentum,
                eps=self.energy_scaling_eps,
                use_mad=self.energy_scaling_use_mad
            )
            log.info(f"Initialized EnergyScalerPerK with K={self.denoising_steps}")
        
        # Potential-based reward shaping parameters
        self.use_pbrs = use_pbrs
        self.pbrs_lambda = pbrs_lambda
        self.pbrs_beta = pbrs_beta
        self.pbrs_alpha = pbrs_alpha
        self.pbrs_M = pbrs_M
        self.pbrs_eta = pbrs_eta
        self.pbrs_use_mu_only = pbrs_use_mu_only
        self.pbrs_k_use_mode = pbrs_k_use_mode
        
        # Initialize EBM and potential function if PBRS is enabled
        if self.use_pbrs:
            if ebm_model is None:
                raise ValueError("EBM model must be provided when use_pbrs=True")
            
            self.ebm_model = ebm_model
            if ebm_ckpt_path is not None:
                checkpoint = torch.load(ebm_ckpt_path, map_location=self.device)
                self.ebm_model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
                log.info(f"Loaded EBM from {ebm_ckpt_path}")
            
            # Build k_use indices for potential computation
            self.pbrs_k_use = build_k_use_indices(self.pbrs_k_use_mode, self.denoising_steps)
            
            # Initialize potential function (will be set up in setup_potential_function)
            self.potential_function = None
            self.stats = None
            
            log.info(f"Initialized PBRS with k_use={self.pbrs_k_use}")
    
    def setup_potential_function(self, stats: Dict[str, Any]):
        """
        Set up the potential function for reward shaping.
        
        Args:
            stats: Normalization statistics containing action bounds
        """
        if not self.use_pbrs:
            return
            
        self.stats = stats
        
        # Create a frozen copy of the current policy as prior
        from copy import deepcopy
        pi0 = deepcopy(self)
        for param in pi0.parameters():
            param.requires_grad = False
        pi0.eval()
        
        # Initialize potential function
        self.potential_function = KFreePotential(
            ebm_model=self.ebm_model,
            pi0=pi0,
            K=self.denoising_steps,
            stats=self.stats,
            k_use=self.pbrs_k_use,
            beta_k=self.pbrs_beta,
            alpha=self.pbrs_alpha,
            M=self.pbrs_M,
            eta=self.pbrs_eta,
            use_mu_only=self.pbrs_use_mu_only,
            device=self.device
        )
        
        log.info("Set up potential function for reward shaping")
    
    def update_energy_scaler(self, k_vec: torch.Tensor, energy_vec: torch.Tensor):
        """
        Update the energy scaler with new energy values.
        
        Args:
            k_vec: Denoising step indices [B]
            energy_vec: Energy values corresponding to each step [B]
        """
        if self.use_energy_scaling and hasattr(self, 'energy_scaler'):
            self.energy_scaler.update(k_vec, energy_vec)
    
    def normalize_energy(self, k_vec: torch.Tensor, energy_vec: torch.Tensor) -> torch.Tensor:
        """
        Normalize energy values using the energy scaler.
        
        Args:
            k_vec: Denoising step indices [B]
            energy_vec: Energy values to normalize [B]
            
        Returns:
            Normalized energy values [B]
        """
        if self.use_energy_scaling and hasattr(self, 'energy_scaler'):
            return self.energy_scaler.normalize(k_vec, energy_vec)
        return energy_vec
    
    def compute_pbrs_reward(self, s_t: torch.Tensor, s_tp1: torch.Tensor, 
                          gamma: float = 0.99) -> torch.Tensor:
        """
        Compute potential-based reward shaping reward.
        
        Args:
            s_t: Current states [B, obs_dim]
            s_tp1: Next states [B, obs_dim]
            gamma: Discount factor
            
        Returns:
            Shaped reward values [B]
        """
        if not self.use_pbrs or self.potential_function is None:
            return torch.zeros(s_t.size(0), device=s_t.device)
        
        r_shape = self.potential_function.shape_reward(s_t, s_tp1, gamma)
        return self.pbrs_lambda * r_shape
    
    def get_energy_scaler_stats(self, k: int) -> Dict[str, float]:
        """
        Get current statistics for a specific denoising step k.
        
        Args:
            k: Denoising step index
            
        Returns:
            Dictionary containing mean, std, and count for step k
        """
        if self.use_energy_scaling and hasattr(self, 'energy_scaler'):
            return self.energy_scaler.get_stats(k)
        return {"mean": 0.0, "std": 1.0, "count": 0}
    
    def reset_energy_scaler(self):
        """Reset the energy scaler statistics."""
        if self.use_energy_scaling and hasattr(self, 'energy_scaler'):
            self.energy_scaler.reset()
            log.info("Reset energy scaler statistics")
    
    def loss(
        self,
        obs,
        chains_prev,
        chains_next,
        denoising_inds,
        returns,
        oldvalues,
        advantages,
        oldlogprobs,
        use_bc_loss=False,
        reward_horizon=4,
        # Additional parameters for EBM integration
        energy_values: Optional[torch.Tensor] = None,
        next_obs: Optional[Dict] = None,
        gamma: float = 0.99,
    ):
        """
        Enhanced PPO loss with energy-based model integration.
        
        Args:
            obs: Current observations
            chains_prev: Previous action chains
            chains_next: Next action chains
            denoising_inds: Denoising step indices
            returns: Return values
            oldvalues: Old value estimates
            advantages: Advantage estimates
            oldlogprobs: Old log probabilities
            use_bc_loss: Whether to use behavioral cloning loss
            reward_horizon: Action horizon for gradient backpropagation
            energy_values: Energy values for each denoising step [B, K] (optional)
            next_obs: Next observations for PBRS (optional)
            gamma: Discount factor for PBRS (optional)
        """
        # Update energy scaler if energy values are provided
        if energy_values is not None and self.use_energy_scaling:
            # energy_values should be [B, K] where K is the number of denoising steps
            B, K = energy_values.shape
            k_vec = denoising_inds.unsqueeze(0).expand(B, -1).flatten()  # [B*K]
            energy_vec = energy_values.flatten()  # [B*K]
            self.update_energy_scaler(k_vec, energy_vec)
        
        # Compute PBRS reward if enabled and next observations are provided
        pbrs_reward = None
        if self.use_pbrs and next_obs is not None:
            # Extract state observations
            s_t = obs.get("state", obs)
            s_tp1 = next_obs.get("state", next_obs)
            
            # Handle different observation formats
            if isinstance(s_t, dict):
                s_t = s_t.get("state", s_t)
            if isinstance(s_tp1, dict):
                s_tp1 = s_tp1.get("state", s_tp1)
            
            # Ensure proper shape
            if s_t.dim() == 3 and s_t.size(1) == 1:
                s_t = s_t[:, 0, :]
            if s_tp1.dim() == 3 and s_tp1.size(1) == 1:
                s_tp1 = s_tp1[:, 0, :]
            
            pbrs_reward = self.compute_pbrs_reward(s_t, s_tp1, gamma)
            
            # Add PBRS reward to environment returns
            if pbrs_reward is not None:
                returns = returns + pbrs_reward
        
        # Call parent loss function
        loss_tuple = super().loss(
            obs=obs,
            chains_prev=chains_prev,
            chains_next=chains_next,
            denoising_inds=denoising_inds,
            returns=returns,
            oldvalues=oldvalues,
            advantages=advantages,
            oldlogprobs=oldlogprobs,
            use_bc_loss=use_bc_loss,
            reward_horizon=reward_horizon,
        )
        
        # Add PBRS reward to loss tuple if computed
        if pbrs_reward is not None:
            loss_tuple = loss_tuple + (pbrs_reward.mean().item(),)
        else:
            loss_tuple = loss_tuple + (0.0,)
        
        return loss_tuple
    
    def get_energy_scaling_info(self) -> Dict[str, Any]:
        """
        Get information about energy scaling configuration.
        
        Returns:
            Dictionary containing energy scaling configuration
        """
        info = {
            "use_energy_scaling": self.use_energy_scaling,
            "use_pbrs": self.use_pbrs,
        }
        
        if self.use_energy_scaling:
            info.update({
                "energy_scaling_momentum": self.energy_scaling_momentum,
                "energy_scaling_eps": self.energy_scaling_eps,
                "energy_scaling_use_mad": self.energy_scaling_use_mad,
            })
        
        if self.use_pbrs:
            info.update({
                "pbrs_lambda": self.pbrs_lambda,
                "pbrs_beta": self.pbrs_beta,
                "pbrs_alpha": self.pbrs_alpha,
                "pbrs_M": self.pbrs_M,
                "pbrs_eta": self.pbrs_eta,
                "pbrs_use_mu_only": self.pbrs_use_mu_only,
                "pbrs_k_use_mode": self.pbrs_k_use_mode,
                "pbrs_k_use": getattr(self, 'pbrs_k_use', []),
            })
        
        return info
