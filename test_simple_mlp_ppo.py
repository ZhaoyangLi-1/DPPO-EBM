"""
Test script for Simple MLP PPO implementation.

This script tests the basic functionality of the Simple MLP PPO baseline
including model initialization, forward pass, and basic training setup.
"""

import os
import sys
import torch
import numpy as np
import logging
from omegaconf import OmegaConf

# Add project root to path
sys.path.append('/linting-slow-vol/DPPO-EBM')

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def test_model_initialization():
    """Test Simple MLP PPO model initialization."""
    log.info("Testing model initialization...")
    
    try:
        from model.rl.simple_mlp_ppo import SimpleMLP_PPO
        from model.common.mlp_gaussian import Gaussian_MLP
        from model.common.critic import CriticObs
        
        # Model parameters
        obs_dim = 11
        action_dim = 3
        horizon_steps = 4
        cond_steps = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create actor
        actor = Gaussian_MLP(
            action_dim=action_dim,
            horizon_steps=horizon_steps,
            cond_dim=obs_dim * cond_steps,
            mlp_dims=[256, 256, 256],
            activation_type="ReLU",
            tanh_output=True,
            residual_style=False,
            use_layernorm=False,
            dropout=0.1,
            fixed_std=None,
            learn_fixed_std=False,
            std_min=0.01,
            std_max=1.0,
        ).to(device)
        
        # Create critic
        critic = CriticObs(
            cond_dim=obs_dim * cond_steps,
            mlp_dims=[256, 256, 256],
            activation_type="ReLU",
            residual_style=False,
        ).to(device)
        
        # Create PPO model
        model = SimpleMLP_PPO(
            actor=actor,
            critic=critic,
            clip_ploss_coef=0.2,
            clip_vloss_coef=0.2,
            norm_adv=True,
            use_ebm_reward=False,
            ebm_model=None,
            ebm_reward_scale=1.0,
            horizon_steps=horizon_steps,
            device=device
        )
        
        log.info("✅ Model initialization successful")
        return model, obs_dim, action_dim, horizon_steps, cond_steps, device
        
    except Exception as e:
        log.error(f"❌ Model initialization failed: {e}")
        return None


def test_forward_pass(model, obs_dim, action_dim, horizon_steps, cond_steps, device):
    """Test forward pass through the model."""
    log.info("Testing forward pass...")
    
    try:
        batch_size = 4
        
        # Create dummy observations
        obs = {
            'state': torch.randn(batch_size, cond_steps, obs_dim, device=device)
        }
        
        # Forward pass
        with torch.no_grad():
            actions = model(cond=obs, deterministic=False)
        
        # Check output shape - model returns [B, horizon_steps, action_dim]
        expected_shape = (batch_size, horizon_steps, action_dim)
        if actions.shape == expected_shape:
            log.info(f"✅ Forward pass successful. Output shape: {actions.shape}")
            return True
        else:
            log.error(f"❌ Wrong output shape. Expected: {expected_shape}, Got: {actions.shape}")
            return False
            
    except Exception as e:
        log.error(f"❌ Forward pass failed: {e}")
        return False


def test_loss_computation(model, obs_dim, action_dim, horizon_steps, cond_steps, device):
    """Test loss computation."""
    log.info("Testing loss computation...")
    
    try:
        batch_size = 8
        
        # Create dummy data
        obs = {
            'state': torch.randn(batch_size, cond_steps, obs_dim, device=device)
        }
        actions = torch.randn(batch_size, horizon_steps, action_dim, device=device)
        returns = torch.randn(batch_size, device=device)
        oldvalues = torch.randn(batch_size, device=device)
        advantages = torch.randn(batch_size, device=device)
        oldlogprobs = torch.randn(batch_size, device=device)
        
        # Compute loss
        (
            pg_loss,
            entropy_loss,
            v_loss,
            clipfrac,
            approx_kl,
            ratio,
            bc_loss,
            std,
        ) = model.loss(
            obs=obs,
            actions=actions,
            returns=returns,
            oldvalues=oldvalues,
            advantages=advantages,
            oldlogprobs=oldlogprobs,
            use_bc_loss=False,
        )
        
        log.info(f"✅ Loss computation successful")
        log.info(f"    Policy loss: {pg_loss.item():.4f}")
        log.info(f"    Value loss: {v_loss.item():.4f}")
        log.info(f"    Entropy loss: {entropy_loss.item():.4f}")
        log.info(f"    Clip fraction: {clipfrac:.4f}")
        log.info(f"    Approx KL: {approx_kl:.4f}")
        return True
        
    except Exception as e:
        log.error(f"❌ Loss computation failed: {e}")
        return False


def test_ebm_integration():
    """Test EBM integration (without actual EBM model)."""
    log.info("Testing EBM integration...")
    
    try:
        from model.rl.simple_mlp_ppo import SimpleMLP_PPO
        from model.common.mlp_gaussian import Gaussian_MLP
        from model.common.critic import CriticObs
        
        obs_dim = 11
        action_dim = 3
        horizon_steps = 4
        cond_steps = 1
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create networks
        actor = Gaussian_MLP(
            action_dim=action_dim,
            horizon_steps=horizon_steps,
            cond_dim=obs_dim * cond_steps,
            mlp_dims=[128, 128],
            activation_type="ReLU",
        ).to(device)
        
        critic = CriticObs(
            cond_dim=obs_dim * cond_steps,
            mlp_dims=[128, 128],
            activation_type="ReLU",
        ).to(device)
        
        # Create model with EBM enabled (but no actual EBM model)
        model = SimpleMLP_PPO(
            actor=actor,
            critic=critic,
            clip_ploss_coef=0.2,
            use_ebm_reward=True,
            ebm_model=None,  # No actual EBM model
            ebm_reward_scale=2.0,
            horizon_steps=horizon_steps,
            device=device
        )
        
        # Test reward computation (should return zeros without EBM)
        batch_size = 4
        obs = {'state': torch.randn(batch_size, cond_steps, obs_dim, device=device)}
        actions = torch.randn(batch_size, horizon_steps, action_dim, device=device)
        
        ebm_rewards = model.compute_ebm_reward(obs, actions)
        
        if ebm_rewards.shape == (batch_size,) and torch.allclose(ebm_rewards, torch.zeros_like(ebm_rewards)):
            log.info("✅ EBM integration test successful (returns zeros without EBM model)")
            return True
        else:
            log.error(f"❌ EBM integration test failed. Shape: {ebm_rewards.shape}, Values: {ebm_rewards}")
            return False
            
    except Exception as e:
        log.error(f"❌ EBM integration test failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading."""
    log.info("Testing configuration loading...")
    
    try:
        # Test environment reward config
        env_config_path = "/linting-slow-vol/DPPO-EBM/cfg/gym/finetune/hopper-v2/ft_simple_mlp_ppo_env.yaml"
        if os.path.exists(env_config_path):
            cfg_env = OmegaConf.load(env_config_path)
            log.info(f"✅ Environment reward config loaded: use_ebm_reward = {cfg_env.model.use_ebm_reward}")
        else:
            log.error(f"❌ Environment reward config not found: {env_config_path}")
            return False
        
        # Test EBM reward config
        ebm_config_path = "/linting-slow-vol/DPPO-EBM/cfg/gym/finetune/hopper-v2/ft_simple_mlp_ppo_ebm.yaml"
        if os.path.exists(ebm_config_path):
            cfg_ebm = OmegaConf.load(ebm_config_path)
            log.info(f"✅ EBM reward config loaded: use_ebm_reward = {cfg_ebm.model.use_ebm_reward}")
        else:
            log.error(f"❌ EBM reward config not found: {ebm_config_path}")
            return False
        
        return True
        
    except Exception as e:
        log.error(f"❌ Configuration loading failed: {e}")
        return False


def main():
    """Run all tests."""
    log.info("=" * 60)
    log.info("Simple MLP PPO Implementation Test Suite")
    log.info("=" * 60)
    
    tests_passed = 0
    total_tests = 5
    
    # Test 1: Model initialization
    model_data = test_model_initialization()
    if model_data is not None:
        tests_passed += 1
        model, obs_dim, action_dim, horizon_steps, cond_steps, device = model_data
        
        # Test 2: Forward pass
        if test_forward_pass(model, obs_dim, action_dim, horizon_steps, cond_steps, device):
            tests_passed += 1
        
        # Test 3: Loss computation
        if test_loss_computation(model, obs_dim, action_dim, horizon_steps, cond_steps, device):
            tests_passed += 1
    
    # Test 4: EBM integration
    if test_ebm_integration():
        tests_passed += 1
    
    # Test 5: Configuration loading
    if test_config_loading():
        tests_passed += 1
    
    # Summary
    log.info("=" * 60)
    log.info(f"Test Results: {tests_passed}/{total_tests} tests passed")
    log.info("=" * 60)
    
    if tests_passed == total_tests:
        log.info("🎉 All tests passed! The Simple MLP PPO implementation is ready to use.")
        log.info("")
        log.info("Usage examples:")
        log.info("  # Environment rewards:")
        log.info("  ./run_simple_mlp_ppo.sh hopper-v2 42 env")
        log.info("")
        log.info("  # EBM rewards:")
        log.info("  ./run_simple_mlp_ppo.sh hopper-v2 42 ebm")
        log.info("")
        log.info("  # Both configurations:")
        log.info("  ./run_simple_mlp_ppo.sh hopper-v2 42 both")
        return 0
    else:
        log.error(f"❌ {total_tests - tests_passed} tests failed. Please check the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())