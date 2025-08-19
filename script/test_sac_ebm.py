#!/usr/bin/env python3
"""
Test script for SAC + EBM integration.

This script tests the basic functionality of the SAC + EBM implementation
without running full training.
"""

import torch
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.diffusion.diffusion_sac_ebm import SACDiffusionEBM
from model.diffusion.mlp_diffusion import DiffusionMLP
from model.common.critic import CriticObsAct, CriticObs
from model.ebm.ebm import EBMWrapper


def test_sac_ebm_initialization():
    """Test SAC + EBM model initialization."""
    print("Testing SAC + EBM initialization...")
    
    # Model parameters
    obs_dim = 11
    action_dim = 3
    horizon_steps = 4
    denoising_steps = 20
    ft_denoising_steps = 20
    device = "cpu"
    
    # Create networks
    actor = DiffusionMLP(
        time_dim=16,
        mlp_dims=[512, 512, 512],
        activation_type="ReLU",
        residual_style=True,
        cond_dim=obs_dim,
        horizon_steps=horizon_steps,
        action_dim=action_dim,
    )
    
    critic_q1 = CriticObsAct(
        cond_dim=obs_dim,
        action_dim=action_dim,
        mlp_dims=[256, 256, 256],
        activation_type="Mish",
        residual_style=True,
        double_q=False,
    )
    
    critic_q2 = CriticObsAct(
        cond_dim=obs_dim,
        action_dim=action_dim,
        mlp_dims=[256, 256, 256],
        activation_type="Mish",
        residual_style=True,
        double_q=False,
    )
    
    critic_v = CriticObs(
        cond_dim=obs_dim,
        mlp_dims=[256, 256, 256],
        activation_type="Mish",
        residual_style=True,
    )
    
    # Create EBM model
    ebm_model = EBMWrapper(
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dim=256,
        num_layers=4,
    )
    
    # Create SAC + EBM model
    model = SACDiffusionEBM(
        actor=actor,
        critic_q1=critic_q1,
        critic_q2=critic_q2,
        critic_v=critic_v,
        # SAC parameters
        gamma=0.99,
        tau=0.005,
        alpha=0.2,
        automatic_entropy_tuning=True,
        target_entropy=-action_dim,
        # EBM parameters
        use_ebm_reward_shaping=True,
        ebm_model=ebm_model,
        pbrs_lambda=0.5,
        pbrs_beta=1.0,
        pbrs_alpha=0.1,
        pbrs_M=4,
        pbrs_use_mu_only=True,
        pbrs_k_use_mode="tail:6",
        # Diffusion parameters
        denoising_steps=denoising_steps,
        ft_denoising_steps=ft_denoising_steps,
        horizon_steps=horizon_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )
    
    print("✓ SAC + EBM model initialized successfully")
    return model


def test_forward_pass(model):
    """Test forward pass through the model."""
    print("Testing forward pass...")
    
    batch_size = 4
    obs_dim = 11
    
    # Create dummy observations
    obs = torch.randn(batch_size, obs_dim)
    cond = {"state": obs.unsqueeze(1)}  # Add time dimension
    
    # Test forward pass
    with torch.no_grad():
        samples = model.forward(cond, deterministic=False, return_chain=True)
        
        # Check output shapes
        assert samples.trajectories.shape == (batch_size, 4, action_dim), f"Expected shape {(batch_size, 4, action_dim)}, got {samples.trajectories.shape}"
        assert samples.chains.shape == (batch_size, 21, 4, action_dim), f"Expected shape {(batch_size, 21, 4, action_dim)}, got {samples.chains.shape}"
    
    print("✓ Forward pass successful")


def test_critic_computation(model):
    """Test critic network computations."""
    print("Testing critic computations...")
    
    batch_size = 4
    obs_dim = 11
    action_dim = 3
    
    # Create dummy data
    obs = torch.randn(batch_size, obs_dim)
    actions = torch.randn(batch_size, action_dim)
    
    # Test Q-value computation
    q1, q2 = model.compute_q_values(obs, actions)
    assert q1.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {q1.shape}"
    assert q2.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {q2.shape}"
    
    # Test V-value computation
    v = model.compute_v_value(obs)
    assert v.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {v.shape}"
    
    print("✓ Critic computations successful")


def test_ebm_reward_shaping(model):
    """Test EBM reward shaping functionality."""
    print("Testing EBM reward shaping...")
    
    batch_size = 4
    obs_dim = 11
    
    # Create dummy observations
    obs_t = torch.randn(batch_size, obs_dim)
    obs_tp1 = torch.randn(batch_size, obs_dim)
    
    # Setup potential function with dummy stats
    stats = {
        "act_min": np.array([-1.0] * 3),
        "act_max": np.array([1.0] * 3),
    }
    model.setup_potential_function(stats)
    
    # Test reward shaping
    with torch.no_grad():
        shaped_reward = model.compute_shaped_reward(obs_t, obs_tp1)
        assert shaped_reward.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {shaped_reward.shape}"
    
    print("✓ EBM reward shaping successful")


def test_loss_computation(model):
    """Test loss computation."""
    print("Testing loss computations...")
    
    batch_size = 4
    obs_dim = 11
    action_dim = 3
    
    # Create dummy data
    obs = torch.randn(batch_size, obs_dim)
    actions = torch.randn(batch_size, action_dim)
    rewards = torch.randn(batch_size)
    next_obs = torch.randn(batch_size, obs_dim)
    dones = torch.zeros(batch_size)
    log_probs = torch.randn(batch_size)
    
    # Test critic loss
    q1_loss, q2_loss, v_loss = model.compute_critic_loss(obs, actions, rewards, next_obs, dones)
    assert isinstance(q1_loss, torch.Tensor), "Q1 loss should be a tensor"
    assert isinstance(q2_loss, torch.Tensor), "Q2 loss should be a tensor"
    assert isinstance(v_loss, torch.Tensor), "V loss should be a tensor"
    
    # Test actor loss
    actor_loss = model.compute_actor_loss(obs, actions, log_probs)
    assert isinstance(actor_loss, torch.Tensor), "Actor loss should be a tensor"
    
    # Test alpha loss
    alpha_loss = model.compute_alpha_loss(obs, actions, log_probs)
    assert isinstance(alpha_loss, torch.Tensor), "Alpha loss should be a tensor"
    
    print("✓ Loss computations successful")


def main():
    """Run all tests."""
    print("Running SAC + EBM tests...\n")
    
    try:
        # Test 1: Model initialization
        model = test_sac_ebm_initialization()
        print()
        
        # Test 2: Forward pass
        test_forward_pass(model)
        print()
        
        # Test 3: Critic computations
        test_critic_computation(model)
        print()
        
        # Test 4: EBM reward shaping
        test_ebm_reward_shaping(model)
        print()
        
        # Test 5: Loss computations
        test_loss_computation(model)
        print()
        
        print("🎉 All tests passed! SAC + EBM implementation is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
