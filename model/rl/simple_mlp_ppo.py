"""
Simple MLP-based PPO implementation as baseline.

This module provides a straightforward PPO implementation using MLP actor and critic
networks, supporting both environment rewards and EBM rewards.
"""

from typing import Optional, Dict, Any
import torch
import torch.nn as nn
import logging

log = logging.getLogger(__name__)
from model.rl.gaussian_ppo import PPO_Gaussian


class SimpleMLP_PPO(PPO_Gaussian):
    """
    Simple MLP-based PPO implementation with support for both env and EBM rewards.
    
    This class extends PPO_Gaussian to provide a clean baseline implementation
    using simple MLP networks for the actor and critic.
    """
    
    def __init__(
        self,
        actor,
        critic,
        clip_ploss_coef: float,
        clip_vloss_coef: Optional[float] = None,
        norm_adv: Optional[bool] = True,
        # EBM reward parameters
        use_ebm_reward: bool = False,
        ebm_model: Optional[nn.Module] = None,
        ebm_reward_scale: float = 1.0,
        # Required parameters for parent class
        horizon_steps: int = 4,
        device: str = "cuda:0",
        **kwargs,
    ):
        """
        Initialize Simple MLP PPO.
        
        Args:
            actor: MLP actor network
            critic: MLP critic network
            clip_ploss_coef: Policy loss clipping coefficient
            clip_vloss_coef: Value loss clipping coefficient
            norm_adv: Whether to normalize advantages
            use_ebm_reward: Whether to use EBM rewards instead of env rewards
            ebm_model: EBM model for computing rewards
            ebm_reward_scale: Scaling factor for EBM rewards
            **kwargs: Additional arguments passed to parent class
        """
        # Extract EBM-specific parameters before calling parent
        ebm_params = {
            'use_ebm_reward': use_ebm_reward,
            'ebm_model': ebm_model,
            'ebm_reward_scale': ebm_reward_scale,
            'horizon_steps': horizon_steps,
            'device': device
        }
        
        # Remove EBM-specific params from kwargs to avoid passing to parent
        clean_kwargs = {k: v for k, v in kwargs.items() 
                       if k not in ['use_ebm_reward', 'ebm_model', 'ebm_reward_scale', 
                                   'ebm_reward_clip_max', 'obs_dim', 'action_dim', 'ebm', 'ebm_ckpt_path',
                                   'use_auto_scaling', 'use_dynamic_normalization', 'normalization_momentum',
                                   'temperature_scale', 'normalization_epsilon']}
        
        super().__init__(
            actor=actor,
            critic=critic,
            clip_ploss_coef=clip_ploss_coef,
            clip_vloss_coef=clip_vloss_coef,
            norm_adv=norm_adv,
            horizon_steps=horizon_steps,
            device=device,
            **clean_kwargs,
        )
        
        # EBM reward configuration
        self.use_ebm_reward = use_ebm_reward
        self.ebm_model = ebm_model
        self.ebm_reward_scale = ebm_reward_scale
        
        if self.use_ebm_reward and self.ebm_model is None:
            log.warning("use_ebm_reward is True but no EBM model provided - will be set later by agent")
        elif self.use_ebm_reward and self.ebm_model is not None:
            log.info(f"EBM reward enabled with scale={self.ebm_reward_scale}, model provided during init")
        
        if self.use_ebm_reward and self.ebm_model is not None:
            # Freeze EBM parameters
            for param in self.ebm_model.parameters():
                param.requires_grad = False
            self.ebm_model.eval()
            log.info("EBM model loaded and frozen for reward computation")
    
    def compute_ebm_reward(self, obs, actions, k_idx=None, t_idx=None):
        """
        Compute EBM-based rewards for given observations and actions.
        
        Args:
            obs: Observations dictionary with 'state' key
            actions: Action tensor [B, horizon_steps, action_dim]
            k_idx: Denoising step indices (optional)
            t_idx: Time step indices (optional)
            
        Returns:
            EBM rewards tensor [B]
        """
        if not self.use_ebm_reward or self.ebm_model is None:
            return torch.zeros(len(actions), device=actions.device)
        
        B = len(actions)
        device = actions.device
        
        # Default indices if not provided
        if k_idx is None:
            k_idx = torch.zeros(B, device=device, dtype=torch.long)
        if t_idx is None:
            t_idx = torch.zeros(B, device=device, dtype=torch.long)
        
        with torch.no_grad():
            # Extract state from observations
            if isinstance(obs, dict) and 'state' in obs:
                states = obs['state']
                if states.dim() == 3:  # [B, seq_len, state_dim]
                    states = states[:, -1]  # Take last state
            else:
                states = obs
            
            # Compute EBM energies
            try:
                energies, _ = self.ebm_model(
                    k_idx=k_idx,
                    t_idx=t_idx,
                    views=None,  # No vision input for simple baseline
                    poses=states,
                    actions=actions
                )
                
                # Convert energies to rewards (higher energy = higher reward)
                ebm_rewards = energies * self.ebm_reward_scale
                
                return ebm_rewards
                
            except Exception as e:
                log.warning(f"Failed to compute EBM reward: {e}")
                return torch.zeros(B, device=device)
    
    def get_reward_info(self) -> Dict[str, Any]:
        """
        Get information about reward configuration.
        
        Returns:
            Dictionary containing reward configuration
        """
        return {
            "use_ebm_reward": self.use_ebm_reward,
            "ebm_reward_scale": self.ebm_reward_scale,
            "has_ebm_model": self.ebm_model is not None,
        }
    
    def set_ebm_model(self, ebm_model: nn.Module, ebm_ckpt_path: Optional[str] = None):
        """
        Set EBM model for reward computation.
        
        Args:
            ebm_model: EBM model to use
            ebm_ckpt_path: Path to EBM checkpoint (optional)
        """
        self.ebm_model = ebm_model
        
        # Load checkpoint if provided
        if ebm_ckpt_path is not None:
            try:
                checkpoint = torch.load(ebm_ckpt_path, map_location=self.device)
                if "model" in checkpoint:
                    self.ebm_model.load_state_dict(checkpoint["model"])
                else:
                    self.ebm_model.load_state_dict(checkpoint)
                log.info(f"Loaded EBM checkpoint from {ebm_ckpt_path}")
            except Exception as e:
                log.warning(f"Failed to load EBM checkpoint: {e}")
        
        # Freeze EBM parameters
        for param in self.ebm_model.parameters():
            param.requires_grad = False
        self.ebm_model.eval()