"""
Debug script for PPO reward issues.

This script helps diagnose common issues that cause decreasing rewards in PPO training.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import OmegaConf
import logging

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

def analyze_ppo_config(config_path):
    """Analyze PPO configuration for potential issues."""
    log.info(f"Analyzing config: {config_path}")
    
    cfg = OmegaConf.load(config_path)
    
    issues = []
    recommendations = []
    
    # Check learning rates
    actor_lr = cfg.train.actor_lr
    critic_lr = cfg.train.critic_lr
    
    if actor_lr >= 1e-3:
        issues.append(f"Actor learning rate too high: {actor_lr}")
        recommendations.append("Try actor_lr: 3e-4 or 1e-4")
    
    if critic_lr >= 5e-3:
        issues.append(f"Critic learning rate too high: {critic_lr}")
        recommendations.append("Try critic_lr: 1e-3 or 3e-4")
    
    # Check PPO clipping
    clip_coef = cfg.model.clip_ploss_coef
    if clip_coef >= 0.3:
        issues.append(f"PPO clipping coefficient too high: {clip_coef}")
        recommendations.append("Try clip_ploss_coef: 0.1 or 0.2")
    
    # Check batch size and update epochs
    batch_size = cfg.train.batch_size
    update_epochs = cfg.train.update_epochs
    n_steps = cfg.train.n_steps
    n_envs = cfg.env.n_envs
    
    total_samples = n_steps * n_envs
    if batch_size >= total_samples:
        issues.append(f"Batch size ({batch_size}) >= total samples ({total_samples})")
        recommendations.append(f"Try batch_size: {total_samples // 4}")
    
    if update_epochs >= 15:
        issues.append(f"Too many update epochs: {update_epochs}")
        recommendations.append("Try update_epochs: 5-10")
    
    # Check GAE lambda
    gae_lambda = cfg.train.gae_lambda
    if gae_lambda >= 0.98:
        issues.append(f"GAE lambda too high: {gae_lambda}")
        recommendations.append("Try gae_lambda: 0.95")
    
    # Check target KL
    target_kl = cfg.train.get('target_kl', None)
    if target_kl and target_kl <= 0.005:
        issues.append(f"Target KL too restrictive: {target_kl}")
        recommendations.append("Try target_kl: 0.01-0.02")
    
    # Check entropy coefficient
    ent_coef = cfg.train.get('ent_coef', 0.01)  # Default from base class
    if ent_coef <= 0.001:
        issues.append(f"Entropy coefficient too low: {ent_coef}")
        recommendations.append("Try ent_coef: 0.01-0.05")
    
    return issues, recommendations

def create_improved_config(base_config_path, output_path):
    """Create an improved configuration based on best practices."""
    cfg = OmegaConf.load(base_config_path)
    
    # Improve hyperparameters
    cfg.train.actor_lr = 3e-4
    cfg.train.critic_lr = 1e-3
    cfg.train.batch_size = 2000  # Smaller batches for more stable updates
    cfg.train.update_epochs = 5   # Fewer epochs to prevent overfitting
    cfg.train.target_kl = 0.015   # More reasonable KL target
    cfg.train.gae_lambda = 0.95   # Standard GAE lambda
    cfg.train.reward_scale_const = 1.0  # No reward scaling initially
    cfg.train.reward_scale_running = False  # Disable running reward scaling
    
    # Improve PPO clipping
    cfg.model.clip_ploss_coef = 0.1  # More conservative clipping
    cfg.model.clip_vloss_coef = 0.2
    
    # Add entropy regularization if missing
    if 'ent_coef' not in cfg.train:
        cfg.train.ent_coef = 0.01
    
    # Improve network architecture - smaller networks for more stable training
    cfg.model.actor.mlp_dims = [256, 256]  # Reduced from [256, 256, 256]
    cfg.model.actor.dropout = 0.0  # Remove dropout initially
    cfg.model.critic.mlp_dims = [256, 256]
    
    # Save improved config
    with open(output_path, 'w') as f:
        OmegaConf.save(cfg, f)
    
    log.info(f"Improved config saved to: {output_path}")
    return cfg

def diagnose_reward_scaling():
    """Provide guidance on reward scaling issues."""
    print("=== REWARD SCALING DIAGNOSIS ===")
    print("Common reward scaling issues in PPO:")
    print()
    print("1. RUNNING REWARD SCALING:")
    print("   - Can cause instability if environment rewards vary widely")
    print("   - Try setting: reward_scale_running: False")
    print()
    print("2. REWARD SCALING CONSTANT:")
    print("   - Should be 1.0 initially for debugging")
    print("   - Try setting: reward_scale_const: 1.0")
    print()
    print("3. ADVANTAGE NORMALIZATION:")
    print("   - Usually helps but can mask issues")
    print("   - Current setting: norm_adv: True (recommended)")
    print()

def diagnose_exploration():
    """Provide guidance on exploration issues."""
    print("=== EXPLORATION DIAGNOSIS ===")
    print("Exploration issues that cause reward collapse:")
    print()
    print("1. POLICY ENTROPY TOO LOW:")
    print("   - Add/increase entropy coefficient")
    print("   - Try: ent_coef: 0.01-0.05")
    print()
    print("2. ACTION STD TOO LOW:")
    print("   - Check if policy std is decreasing too fast")
    print("   - Consider using learn_fixed_std or fixed_std")
    print()
    print("3. EARLY STOPPING:")
    print("   - Target KL too restrictive")
    print("   - Try: target_kl: 0.015-0.02")
    print()

def recommend_debugging_steps():
    """Provide step-by-step debugging recommendations."""
    print("=== DEBUGGING RECOMMENDATIONS ===")
    print()
    print("Step 1: SIMPLIFY CONFIGURATION")
    print("- Use smaller networks: [256, 256] instead of [256, 256, 256]")
    print("- Lower learning rates: actor_lr=3e-4, critic_lr=1e-3")
    print("- Conservative clipping: clip_ploss_coef=0.1")
    print("- Disable reward scaling: reward_scale_running=False")
    print()
    print("Step 2: CHECK ENVIRONMENT")
    print("- Verify environment rewards are reasonable")
    print("- Check if environment is terminating too early")
    print("- Monitor episode lengths")
    print()
    print("Step 3: MONITOR KEY METRICS")
    print("- Policy loss (should be small and stable)")
    print("- Value loss (should decrease over time)")
    print("- Entropy (should decrease slowly)")
    print("- KL divergence (should be < target_kl)")
    print("- Explained variance (should be positive)")
    print()
    print("Step 4: GRADIENT ANALYSIS")
    print("- Check for exploding/vanishing gradients")
    print("- Monitor gradient norms")
    print("- Consider gradient clipping")
    print()

def main():
    """Main diagnosis function."""
    print("=" * 60)
    print("PPO REWARD DIAGNOSIS TOOL")
    print("=" * 60)
    
    # Analyze current config
    env_config = "/linting-slow-vol/DPPO-EBM/cfg/gym/finetune/hopper-v2/ft_simple_mlp_ppo_env.yaml"
    issues, recommendations = analyze_ppo_config(env_config)
    
    if issues:
        print("\n❌ POTENTIAL ISSUES FOUND:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
        
        print("\n✅ RECOMMENDATIONS:")
        for i, rec in enumerate(recommendations, 1):
            print(f"  {i}. {rec}")
    else:
        print("\n✅ No obvious configuration issues found.")
    
    print()
    diagnose_reward_scaling()
    print()
    diagnose_exploration()
    print()
    recommend_debugging_steps()
    
    # Create improved config
    improved_config_path = "/linting-slow-vol/DPPO-EBM/cfg/gym/finetune/hopper-v2/ft_simple_mlp_ppo_env_improved.yaml"
    create_improved_config(env_config, improved_config_path)
    
    print("\n" + "=" * 60)
    print("NEXT STEPS:")
    print("=" * 60)
    print("1. Try the improved configuration:")
    print(f"   ./run_simple_mlp_ppo.sh hopper-v2 42 env")
    print("   (Using the improved config)")
    print()
    print("2. Monitor these key metrics during training:")
    print("   - avg episode reward (should increase or stay stable)")
    print("   - policy loss (should be small, ~0.01-0.1)")
    print("   - entropy (should decrease slowly)")
    print("   - explained variance (should be positive, >0.5)")
    print()
    print("3. If rewards still decrease, try:")
    print("   - Even smaller learning rates (1e-4)")
    print("   - Smaller batch sizes (1000)")
    print("   - More conservative clipping (0.05)")

if __name__ == "__main__":
    main()