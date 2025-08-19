"""
Example Energy-Based Model for DPPO integration.

This module provides a simple example EBM implementation that can be used
with the DPPO + EBM integration. This is meant as a demonstration and
should be replaced with your actual EBM model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from typing import Optional, Tuple

log = logging.getLogger(__name__)


class ExampleEBM(nn.Module):
    """
    Example Energy-Based Model for demonstration purposes.
    
    This is a simple MLP-based EBM that computes energy E(s,a,k) for
    state-action pairs at different denoising steps k.
    
    Args:
        obs_dim: Observation dimension
        action_dim: Action dimension
        hidden_dim: Hidden dimension for MLP layers
        num_layers: Number of MLP layers
        use_denoising_step: Whether to condition on denoising step k
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        use_denoising_step: bool = True,
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_denoising_step = use_denoising_step
        
        # Input dimension: obs + action + (optional) denoising step
        input_dim = obs_dim + action_dim
        if use_denoising_step:
            input_dim += 1  # Add denoising step embedding
        
        # Build MLP layers
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.append(nn.Linear(input_dim, hidden_dim))
            else:
                layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.ReLU())
        
        # Output layer (single energy value)
        layers.append(nn.Linear(hidden_dim, 1))
        
        self.mlp = nn.Sequential(*layers)
        
        # Denoising step embedding (if used)
        if use_denoising_step:
            self.step_embedding = nn.Embedding(100, 1)  # Support up to 100 denoising steps
        
        log.info(f"Initialized ExampleEBM with input_dim={input_dim}, hidden_dim={hidden_dim}, num_layers={num_layers}")
    
    def forward(
        self,
        poses: torch.Tensor,
        actions: torch.Tensor,
        k_idx: Optional[torch.Tensor] = None,
        views: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass to compute energy values.
        
        Args:
            poses: State observations [B, obs_dim] or [B, H, obs_dim]
            actions: Actions [B, action_dim] or [B, H, action_dim]
            k_idx: Denoising step indices [B] (optional)
            views: Additional view information (ignored in this example)
            
        Returns:
            Energy values [B] or [B, H]
        """
        # Handle different input shapes
        if poses.dim() == 3:
            B, H, obs_dim = poses.shape
            poses = poses.view(B * H, obs_dim)
            actions = actions.view(B * H, -1)
            if k_idx is not None:
                k_idx = k_idx.unsqueeze(1).expand(-1, H).reshape(-1)
            reshape_output = True
        else:
            B = poses.size(0)
            reshape_output = False
        
        # Ensure actions have correct shape
        if actions.dim() == 3:
            actions = actions.view(actions.size(0), -1)  # Flatten action dimensions
        
        # Concatenate observations and actions
        x = torch.cat([poses, actions], dim=-1)
        
        # Add denoising step information if provided
        if self.use_denoising_step and k_idx is not None:
            step_emb = self.step_embedding(k_idx).squeeze(-1)  # [B] or [B*H]
            x = torch.cat([x, step_emb.unsqueeze(-1)], dim=-1)
        
        # Compute energy through MLP
        energy = self.mlp(x).squeeze(-1)  # [B] or [B*H]
        
        # Reshape output if needed
        if reshape_output:
            energy = energy.view(B, H)
        
        return energy
    
    def get_energy_stats(self) -> dict:
        """Get statistics about the energy model."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "obs_dim": self.obs_dim,
            "action_dim": self.action_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "use_denoising_step": self.use_denoising_step,
        }


class SimpleEBM(nn.Module):
    """
    Very simple EBM for testing purposes.
    
    This is a minimal EBM that just computes a quadratic energy function.
    Useful for testing the integration without needing a complex model.
    """
    
    def __init__(self, obs_dim: int, action_dim: int):
        super().__init__()
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        
        # Simple quadratic energy: E(s,a) = ||s - s_target||^2 + ||a||^2
        self.obs_weight = nn.Parameter(torch.ones(obs_dim))
        self.action_weight = nn.Parameter(torch.ones(action_dim))
        
    def forward(
        self,
        poses: torch.Tensor,
        actions: torch.Tensor,
        k_idx: Optional[torch.Tensor] = None,
        views: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute simple quadratic energy.
        
        Args:
            poses: State observations [B, obs_dim] or [B, H, obs_dim]
            actions: Actions [B, action_dim] or [B, H, action_dim]
            k_idx: Denoising step indices (ignored in this simple model)
            views: Additional view information (ignored)
            
        Returns:
            Energy values [B] or [B, H]
        """
        # Handle different input shapes
        if poses.dim() == 3:
            B, H, obs_dim = poses.shape
            poses = poses.view(B * H, obs_dim)
            actions = actions.view(B * H, -1)
            reshape_output = True
        else:
            B = poses.size(0)
            reshape_output = False
        
        # Ensure actions have correct shape
        if actions.dim() == 3:
            actions = actions.view(actions.size(0), -1)
        
        # Compute quadratic energy
        obs_energy = torch.sum((poses * self.obs_weight.unsqueeze(0)) ** 2, dim=-1)
        action_energy = torch.sum((actions * self.action_weight.unsqueeze(0)) ** 2, dim=-1)
        energy = obs_energy + action_energy
        
        # Reshape output if needed
        if reshape_output:
            energy = energy.view(B, H)
        
        return energy


def create_ebm_model(
    model_type: str = "example",
    obs_dim: int = 11,
    action_dim: int = 3,
    **kwargs
) -> nn.Module:
    """
    Factory function to create EBM models.
    
    Args:
        model_type: Type of EBM to create ("example" or "simple")
        obs_dim: Observation dimension
        action_dim: Action dimension
        **kwargs: Additional arguments for the model
        
    Returns:
        EBM model instance
    """
    if model_type == "example":
        return ExampleEBM(obs_dim=obs_dim, action_dim=action_dim, **kwargs)
    elif model_type == "simple":
        return SimpleEBM(obs_dim=obs_dim, action_dim=action_dim)
    else:
        raise ValueError(f"Unknown EBM model type: {model_type}")


# Example usage and testing
if __name__ == "__main__":
    # Test the EBM models
    obs_dim, action_dim = 11, 3
    batch_size = 4
    horizon = 4
    
    # Test ExampleEBM
    ebm = ExampleEBM(obs_dim=obs_dim, action_dim=action_dim)
    poses = torch.randn(batch_size, horizon, obs_dim)
    actions = torch.randn(batch_size, horizon, action_dim)
    k_idx = torch.randint(0, 20, (batch_size,))
    
    energy = ebm(poses, actions, k_idx)
    print(f"ExampleEBM output shape: {energy.shape}")
    print(f"ExampleEBM energy range: [{energy.min():.3f}, {energy.max():.3f}]")
    
    # Test SimpleEBM
    simple_ebm = SimpleEBM(obs_dim=obs_dim, action_dim=action_dim)
    energy_simple = simple_ebm(poses, actions)
    print(f"SimpleEBM output shape: {energy_simple.shape}")
    print(f"SimpleEBM energy range: [{energy_simple.min():.3f}, {energy_simple.max():.3f}]")
