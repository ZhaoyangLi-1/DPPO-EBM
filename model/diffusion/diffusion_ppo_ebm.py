"""
Enhanced DPPO with Energy-Based Model integration.

This module extends the standard PPODiffusion class with energy-based model
capabilities, specifically for potential-based reward shaping.
"""

from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
import logging

log = logging.getLogger(__name__)
from model.diffusion.diffusion_ppo import PPODiffusion
from model.diffusion.energy_utils import KFreePotential, build_k_use_indices


class PPODiffusionEBM(PPODiffusion):
    """
    Enhanced PPODiffusion with Energy-Based Model integration.
    
    This class extends PPODiffusion with potential-based reward shaping using EBM.
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
        # EBM reward shaping parameters (metadata only; shaping is done at rollout in the agent)
        use_ebm_reward_shaping: bool = False,
        ebm_model: Optional[nn.Module] = None,
        ebm_ckpt_path: Optional[str] = None,
        pbrs_lambda: float = 1.0,
        pbrs_beta: float = 1.0,
        pbrs_alpha: float = 1.0,
        pbrs_M: int = 1,
        pbrs_use_mu_only: bool = True,
        pbrs_k_use_mode: str = "tail:6",
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
        
        # EBM reward shaping metadata
        self.use_ebm_reward_shaping = use_ebm_reward_shaping
        self.pbrs_lambda = pbrs_lambda
        self.pbrs_beta = pbrs_beta
        self.pbrs_alpha = pbrs_alpha
        self.pbrs_M = pbrs_M
        self.pbrs_use_mu_only = pbrs_use_mu_only
        self.pbrs_k_use_mode = pbrs_k_use_mode

        # Optional EBM holder; may be injected later by the agent via set_ebm
        self.ebm_model = None
        self.potential_function = None
        self.stats = None

        if self.use_ebm_reward_shaping and ebm_model is not None:
            self.set_ebm(ebm_model, ebm_ckpt_path)
            log.info("PPODiffusionEBM: EBM attached at construction time")

    def set_ebm(self, ebm_model: nn.Module, ebm_ckpt_path: Optional[str] = None):
        """Attach an EBM model post-construction (used by the agent)."""
        self.ebm_model = ebm_model
        if ebm_ckpt_path is not None:
            try:
                checkpoint = torch.load(ebm_ckpt_path, map_location=self.device)
                self.ebm_model.load_state_dict(checkpoint.get("model", checkpoint), strict=False)
                log.info(f"Loaded EBM state from {ebm_ckpt_path}")
            except Exception as e:
                log.warning(f"Failed to load EBM checkpoint: {e}")
    
    def setup_potential_function(self, stats: Dict[str, Any]):
        """
        Set up the potential function for reward shaping.
        
        Args:
            stats: Normalization statistics containing action bounds
        """
        if not self.use_ebm_reward_shaping or self.ebm_model is None:
            return
            
        self.stats = stats
        
        # Create a frozen copy of the current policy as prior
        from copy import deepcopy
        pi0 = deepcopy(self)
        for param in pi0.parameters():
            param.requires_grad = False
        pi0.eval()
        
        # Build k_use indices with 0-based indexing (0..K-1)
        k_use = build_k_use_indices(self.pbrs_k_use_mode, self.ft_denoising_steps)
        
        # Log k_use validation
        log.info(f"PPODiffusionEBM setup_potential_function: ft_denoising_steps={self.ft_denoising_steps}, k_use={k_use}, valid range: 0 to {self.ft_denoising_steps-1}")

        # Initialize potential function (the agent will call this and then use it at rollout)
        self.potential_function = KFreePotential(
            ebm_model=self.ebm_model,
            pi0=pi0,
            K=self.ft_denoising_steps,
            stats=self.stats,
            k_use=k_use,
            beta_k=self.pbrs_beta,
            alpha=self.pbrs_alpha,
            M=self.pbrs_M,
            use_mu_only=self.pbrs_use_mu_only,
            device=self.device
        )
        
        log.info("Set up potential function for EBM reward shaping")
    # Intentionally do not override loss; PBRS is applied at rollout time in the agent.
    
    def get_ebm_info(self) -> Dict[str, Any]:
        """
        Get information about EBM configuration.
        
        Returns:
            Dictionary containing EBM configuration
        """
        info = {
            "use_ebm_reward_shaping": self.use_ebm_reward_shaping,
        }
        
        if self.use_ebm_reward_shaping:
            info.update({
                "pbrs_lambda": self.pbrs_lambda,
                "pbrs_beta": self.pbrs_beta,
                "pbrs_alpha": self.pbrs_alpha,
                "pbrs_M": self.pbrs_M,
                "pbrs_use_mu_only": self.pbrs_use_mu_only,
                "pbrs_k_use_mode": self.pbrs_k_use_mode,
                "pbrs_k_use": build_k_use_indices(self.pbrs_k_use_mode, self.ft_denoising_steps),
            })
        
        return info
    
