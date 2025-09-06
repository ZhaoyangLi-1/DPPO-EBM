"""
Preference-based reward model for FDPP.
This module implements the reward learning from human preferences using the Bradley-Terry model.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional


class PreferenceRewardModel(nn.Module):
    """
    Preference-based reward model that learns from human feedback.
    Uses Bradley-Terry model for preference learning.
    """
    
    def __init__(
        self,
        obs_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        output_activation: str = "tanh",
        device: str = "cuda:0",
    ):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.device = device
        
        # Build MLP network
        layers = []
        prev_dim = obs_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "tanh":
                layers.append(nn.Tanh())
            elif activation == "leaky_relu":
                layers.append(nn.LeakyReLU())
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, 1))
        
        if output_activation == "tanh":
            layers.append(nn.Tanh())
        elif output_activation == "sigmoid":
            layers.append(nn.Sigmoid())
        
        self.network = nn.Sequential(*layers)
        self.to(device)
    
    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute reward for given observation.
        
        Args:
            obs: Observation tensor of shape (batch_size, obs_dim)
        
        Returns:
            Reward tensor of shape (batch_size, 1)
        """
        return self.network(obs)
    
    def compute_preference_loss(
        self,
        obs_0: torch.Tensor,
        obs_1: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute preference learning loss using Bradley-Terry model.
        
        Args:
            obs_0: First observation in the pair
            obs_1: Second observation in the pair
            labels: Preference labels (-1: equal, 0: obs_0 preferred, 1: obs_1 preferred)
        
        Returns:
            Loss tensor
        """
        # Compute rewards
        r_0 = self.forward(obs_0)
        r_1 = self.forward(obs_1)
        
        # Bradley-Terry model probabilities
        logits = r_1 - r_0  # Shape: (batch_size, 1)
        
        # Create target probabilities based on labels
        # label=0 means obs_0 preferred (prob_1=0)
        # label=1 means obs_1 preferred (prob_1=1)
        # label=-1 means equal preference (prob_1=0.5)
        
        targets = torch.zeros_like(labels, dtype=torch.float32)
        targets[labels == 1] = 1.0
        targets[labels == -1] = 0.5
        
        # Binary cross-entropy loss
        loss = F.binary_cross_entropy_with_logits(
            logits.squeeze(-1), targets, reduction='mean'
        )
        
        return loss
    
    def predict_preference(
        self,
        obs_0: torch.Tensor,
        obs_1: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict preference probability between two observations.
        
        Args:
            obs_0: First observation
            obs_1: Second observation
        
        Returns:
            Probability that obs_1 is preferred over obs_0
        """
        r_0 = self.forward(obs_0)
        r_1 = self.forward(obs_1)
        
        logits = r_1 - r_0
        probs = torch.sigmoid(logits)
        
        return probs


class TrajectoryRewardModel(PreferenceRewardModel):
    """
    Extension of preference reward model for trajectory-level rewards.
    Computes rewards over state-action sequences.
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_dims: List[int] = [256, 256],
        activation: str = "relu",
        output_activation: str = "tanh",
        use_action: bool = False,
        device: str = "cuda:0",
    ):
        # If using actions, concatenate obs and action dimensions
        input_dim = obs_dim + action_dim if use_action else obs_dim
        
        super().__init__(
            obs_dim=input_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            output_activation=output_activation,
            device=device,
        )
        
        self.action_dim = action_dim
        self.use_action = use_action
        self.original_obs_dim = obs_dim
    
    def compute_trajectory_reward(
        self,
        states: torch.Tensor,
        actions: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute reward for a trajectory.
        
        Args:
            states: State sequence of shape (batch_size, seq_len, obs_dim)
            actions: Action sequence of shape (batch_size, seq_len, action_dim)
        
        Returns:
            Trajectory reward of shape (batch_size,)
        """
        batch_size, seq_len = states.shape[:2]
        
        if self.use_action and actions is not None:
            # Concatenate states and actions
            inputs = torch.cat([states, actions], dim=-1)
        else:
            inputs = states
        
        # Reshape for batch processing
        inputs_flat = inputs.reshape(-1, inputs.shape[-1])
        
        # Compute rewards for each timestep
        rewards_flat = self.forward(inputs_flat)
        rewards = rewards_flat.reshape(batch_size, seq_len)
        
        # Sum over trajectory
        trajectory_rewards = rewards.sum(dim=1)
        
        return trajectory_rewards


class PreferenceDataset:
    """
    Dataset for storing and sampling preference pairs.
    """
    
    def __init__(self, max_size: int = 10000):
        self.max_size = max_size
        self.buffer = []
        self.ptr = 0
        self.size = 0
    
    def add(
        self,
        obs_0: np.ndarray,
        obs_1: np.ndarray,
        label: int,
    ):
        """
        Add a preference pair to the dataset.
        
        Args:
            obs_0: First observation
            obs_1: Second observation
            label: Preference label
        """
        if self.size < self.max_size:
            self.buffer.append(None)
        
        self.buffer[self.ptr] = (obs_0, obs_1, label)
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)
    
    def sample(self, batch_size: int) -> Dict[str, np.ndarray]:
        """
        Sample a batch of preference pairs.
        
        Args:
            batch_size: Number of pairs to sample
        
        Returns:
            Dictionary with obs_0, obs_1, and labels
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        obs_0_list = []
        obs_1_list = []
        label_list = []
        
        for idx in indices:
            obs_0, obs_1, label = self.buffer[idx]
            obs_0_list.append(obs_0)
            obs_1_list.append(obs_1)
            label_list.append(label)
        
        return {
            'obs_0': np.array(obs_0_list),
            'obs_1': np.array(obs_1_list),
            'labels': np.array(label_list),
        }
    
    def __len__(self) -> int:
        return self.size