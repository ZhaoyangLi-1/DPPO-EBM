"""
Soft Actor-Critic (SAC) for diffusion policy.

This module implements SAC algorithm for diffusion-based policies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Dict, Any

log = logging.getLogger(__name__)
from model.diffusion.diffusion import DiffusionModel


class SACDiffusion(DiffusionModel):
    """
    Soft Actor-Critic implementation for diffusion policies.
    
    SAC is an off-policy actor-critic algorithm that uses entropy regularization
    to encourage exploration and improve sample efficiency.
    """
    
    def __init__(
        self,
        actor: nn.Module,
        critic_q1: nn.Module,
        critic_q2: nn.Module,
        critic_v: nn.Module,
        # SAC parameters
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        automatic_entropy_tuning: bool = True,
        target_entropy: Optional[float] = None,
        # Diffusion parameters
        denoising_steps: int = 20,
        ft_denoising_steps: int = 20,
        horizon_steps: int = 4,
        obs_dim: int = 11,
        action_dim: int = 3,
        device: str = "cuda:0",
        **kwargs,
    ):
        super().__init__(**kwargs)
        
        # SAC parameters
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha
        self.automatic_entropy_tuning = automatic_entropy_tuning
        self.target_entropy = target_entropy
        
        # Diffusion parameters
        self.denoising_steps = denoising_steps
        self.ft_denoising_steps = ft_denoising_steps
        self.horizon_steps = horizon_steps
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.device = device
        
        # Networks
        self.actor = actor.to(device)
        self.critic_q1 = critic_q1.to(device)
        self.critic_q2 = critic_q2.to(device)
        self.critic_v = critic_v.to(device)
        
        # Target networks
        self.critic_q1_target = copy.deepcopy(critic_q1).to(device)
        self.critic_q2_target = copy.deepcopy(critic_q2).to(device)
        self.critic_v_target = copy.deepcopy(critic_v).to(device)
        
        # Freeze target networks
        for param in self.critic_q1_target.parameters():
            param.requires_grad = False
        for param in self.critic_q2_target.parameters():
            param.requires_grad = False
        for param in self.critic_v_target.parameters():
            param.requires_grad = False
        
        # Log alpha for entropy tuning
        if self.automatic_entropy_tuning:
            if self.target_entropy is None:
                self.target_entropy = -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
            self.alpha = self.log_alpha.exp()
        
        log.info(f"Initialized SAC with alpha={self.alpha}, target_entropy={self.target_entropy}")
    
    def forward(self, cond: Dict[str, torch.Tensor], deterministic: bool = False, return_chain: bool = False):
        """
        Forward pass through the actor network.
        
        Args:
            cond: Conditioning dictionary containing observations
            deterministic: Whether to use deterministic sampling
            return_chain: Whether to return the full denoising chain
            
        Returns:
            Action samples and optional denoising chain
        """
        return self.actor.forward(cond, deterministic=deterministic, return_chain=return_chain)
    
    def compute_q_values(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple:
        """
        Compute Q-values for given state-action pairs.
        
        Args:
            obs: Observations [B, obs_dim]
            actions: Actions [B, action_dim]
            
        Returns:
            Tuple of (q1_values, q2_values) [B]
        """
        q1 = self.critic_q1(obs, actions)
        q2 = self.critic_q2(obs, actions)
        return q1, q2
    
    def compute_v_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute V-value for given observations.
        
        Args:
            obs: Observations [B, obs_dim]
            
        Returns:
            V-values [B]
        """
        return self.critic_v(obs)
    
    def compute_target_q_values(self, obs: torch.Tensor, actions: torch.Tensor) -> tuple:
        """
        Compute target Q-values for given state-action pairs.
        
        Args:
            obs: Observations [B, obs_dim]
            actions: Actions [B, action_dim]
            
        Returns:
            Tuple of (target_q1_values, target_q2_values) [B]
        """
        q1_target = self.critic_q1_target(obs, actions)
        q2_target = self.critic_q2_target(obs, actions)
        return q1_target, q2_target
    
    def compute_target_v_value(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Compute target V-value for given observations.
        
        Args:
            obs: Observations [B, obs_dim]
            
        Returns:
            Target V-values [B]
        """
        return self.critic_v_target(obs)
    
    def update_target_networks(self):
        """Update target networks using soft update."""
        for target_param, param in zip(self.critic_q1_target.parameters(), self.critic_q1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_q2_target.parameters(), self.critic_q2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        for target_param, param in zip(self.critic_v_target.parameters(), self.critic_v.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def compute_actor_loss(self, obs: torch.Tensor, actions: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute actor loss for SAC.
        
        Args:
            obs: Observations [B, obs_dim]
            actions: Actions [B, action_dim]
            log_probs: Log probabilities of actions [B]
            
        Returns:
            Actor loss
        """
        q1, q2 = self.compute_q_values(obs, actions)
        q_min = torch.min(q1, q2)
        
        # SAC actor loss: maximize Q - α * log_prob
        actor_loss = (self.alpha * log_probs - q_min).mean()
        return actor_loss
    
    def compute_critic_loss(self, obs: torch.Tensor, actions: torch.Tensor, rewards: torch.Tensor, 
                           next_obs: torch.Tensor, dones: torch.Tensor) -> tuple:
        """
        Compute critic loss for SAC.
        
        Args:
            obs: Current observations [B, obs_dim]
            actions: Current actions [B, action_dim]
            rewards: Rewards [B]
            next_obs: Next observations [B, obs_dim]
            dones: Done flags [B]
            
        Returns:
            Tuple of (q1_loss, q2_loss, v_loss)
        """
        with torch.no_grad():
            # Sample next actions from current policy
            next_cond = {"state": next_obs}
            next_samples = self.actor.forward(next_cond, deterministic=False)
            next_actions = next_samples.trajectories[:, 0]  # Take first action
            
            # Compute next log probs (approximate)
            next_log_probs = torch.zeros(next_actions.size(0), device=self.device)
            
            # Compute target Q values
            next_q1_target, next_q2_target = self.compute_target_q_values(next_obs, next_actions)
            next_q_target = torch.min(next_q1_target, next_q2_target)
            
            # Compute target values
            target_q = rewards + self.gamma * (1 - dones) * (next_q_target - self.alpha * next_log_probs)
        
        # Current Q values
        current_q1, current_q2 = self.compute_q_values(obs, actions)
        
        # Critic losses
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        # V function loss (optional, for additional regularization)
        current_v = self.compute_v_value(obs)
        v_loss = F.mse_loss(current_v, target_q.detach())
        
        return q1_loss, q2_loss, v_loss
    
    def compute_alpha_loss(self, obs: torch.Tensor, actions: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
        """
        Compute alpha loss for automatic entropy tuning.
        
        Args:
            obs: Observations [B, obs_dim]
            actions: Actions [B, action_dim]
            log_probs: Log probabilities of actions [B]
            
        Returns:
            Alpha loss
        """
        if not self.automatic_entropy_tuning:
            return torch.tensor(0.0, device=self.device)
        
        alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
        return alpha_loss
    
    def get_alpha(self) -> torch.Tensor:
        """Get current alpha value."""
        if self.automatic_entropy_tuning:
            return self.log_alpha.exp()
        return torch.tensor(self.alpha, device=self.device)
    
    def update_alpha(self, alpha_optimizer: torch.optim.Optimizer):
        """Update alpha parameter."""
        if self.automatic_entropy_tuning:
            alpha_optimizer.step()
            self.alpha = self.log_alpha.exp()


# Import copy for deep copy operations
import copy
