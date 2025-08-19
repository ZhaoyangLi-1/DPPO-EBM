#!/usr/bin/env python3
"""
Test script for pure SAC implementation.

This script tests the basic functionality of the pure SAC implementation
without EBM integration.
"""

import torch
import numpy as np
import sys
import os

# Add the project root to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.diffusion.diffusion_sac import SACDiffusion
from model.diffusion.mlp_diffusion import DiffusionMLP
from model.common.critic import CriticObsAct, CriticObs


def test_sac_initialization():
    """Test SAC model initialization."""
    print("Testing SAC initialization...")
    
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
    
    # Create SAC model
    model = SACDiffusion(
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
        # Diffusion parameters
        denoising_steps=denoising_steps,
        ft_denoising_steps=ft_denoising_steps,
        horizon_steps=horizon_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=device,
    )
    
    print("✓ SAC model initialized successfully")
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
        assert samples.trajectories.shape == (batch_size, 4, 3), f"Expected shape {(batch_size, 4, 3)}, got {samples.trajectories.shape}"
        assert samples.chains.shape == (batch_size, 21, 4, 3), f"Expected shape {(batch_size, 21, 4, 3)}, got {samples.chains.shape}"
    
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
    
    # Test target network computations
    q1_target, q2_target = model.compute_target_q_values(obs, actions)
    assert q1_target.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {q1_target.shape}"
    assert q2_target.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {q2_target.shape}"
    
    v_target = model.compute_target_v_value(obs)
    assert v_target.shape == (batch_size,), f"Expected shape {(batch_size,)}, got {v_target.shape}"
    
    print("✓ Critic computations successful")


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


def test_target_network_update(model):
    """Test target network update functionality."""
    print("Testing target network updates...")
    
    # Store original parameters
    original_q1_params = {name: param.clone() for name, param in model.critic_q1.named_parameters()}
    original_q2_params = {name: param.clone() for name, param in model.critic_q2.named_parameters()}
    original_v_params = {name: param.clone() for name, param in model.critic_v.named_parameters()}
    
    # Update target networks
    model.update_target_networks()
    
    # Check that target networks have been updated (not identical to original)
    for name, param in model.critic_q1_target.named_parameters():
        if name in original_q1_params:
            assert not torch.allclose(param, original_q1_params[name]), f"Target Q1 {name} should be updated"
    
    for name, param in model.critic_q2_target.named_parameters():
        if name in original_q2_params:
            assert not torch.allclose(param, original_q2_params[name]), f"Target Q2 {name} should be updated"
    
    for name, param in model.critic_v_target.named_parameters():
        if name in original_v_params:
            assert not torch.allclose(param, original_v_params[name]), f"Target V {name} should be updated"
    
    print("✓ Target network updates successful")


def test_alpha_management(model):
    """Test alpha parameter management."""
    print("Testing alpha management...")
    
    # Test getting alpha
    alpha = model.get_alpha()
    assert isinstance(alpha, torch.Tensor), "Alpha should be a tensor"
    assert alpha.shape == (1,) or alpha.shape == (), "Alpha should be scalar"
    
    # Test alpha update (simulate optimizer step)
    original_alpha = alpha.clone()
    
    # Create a dummy optimizer for testing
    optimizer = torch.optim.Adam([model.log_alpha], lr=0.001)
    
    # Simulate a training step
    alpha_loss = model.compute_alpha_loss(
        torch.randn(4, 11), 
        torch.randn(4, 3), 
        torch.randn(4)
    )
    
    optimizer.zero_grad()
    alpha_loss.backward()
    model.update_alpha(optimizer)
    
    # Check that alpha has been updated
    new_alpha = model.get_alpha()
    assert not torch.allclose(new_alpha, original_alpha), "Alpha should be updated after optimization"
    
    print("✓ Alpha management successful")


def main():
    """Run all tests."""
    print("Running pure SAC tests...\n")
    
    try:
        # Test 1: Model initialization
        model = test_sac_initialization()
        print()
        
        # Test 2: Forward pass
        test_forward_pass(model)
        print()
        
        # Test 3: Critic computations
        test_critic_computation(model)
        print()
        
        # Test 4: Loss computations
        test_loss_computation(model)
        print()
        
        # Test 5: Target network updates
        test_target_network_update(model)
        print()
        
        # Test 6: Alpha management
        test_alpha_management(model)
        print()
        
        print("🎉 All tests passed! Pure SAC implementation is working correctly.")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
