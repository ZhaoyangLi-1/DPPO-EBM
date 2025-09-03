#!/usr/bin/env python3

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import numpy as np
from omegaconf import OmegaConf
import hydra
from hydra import initialize, compose

# Test EBM reward computation
def test_ebm_reward():
    print("Testing EBM reward computation...")
    
    # Load config
    with initialize(config_path="cfg/gym/finetune/hopper-v2", version_base=None):
        cfg = compose(config_name="ft_simple_mlp_ppo_ebm")
    
    print(f"Config use_ebm_reward: {cfg.model.use_ebm_reward}")
    print(f"Config ebm_reward_scale: {cfg.model.ebm_reward_scale}")
    print(f"Config ebm_reward_clip_max: {cfg.model.ebm_reward_clip_max}")
    print(f"EBM checkpoint path: {cfg.model.ebm_ckpt_path}")
    
    # Check if EBM checkpoint exists
    if os.path.exists(cfg.model.ebm_ckpt_path):
        print("✓ EBM checkpoint exists")
    else:
        print("✗ EBM checkpoint not found")
        return
    
    try:
        # Import EBM model
        from model.ebm.ebm import EBM
        
        # Create EBM model
        ebm_cfg = cfg.model.ebm
        ebm_model = EBM(cfg=type('EBMConfig', (), {
            'ebm': type('EBMSubConfig', (), {
                'embed_dim': ebm_cfg.get('embed_dim', 256),
                'state_dim': cfg.obs_dim,
                'action_dim': cfg.action_dim,
                'nhead': ebm_cfg.get('nhead', 8),
                'depth': ebm_cfg.get('depth', 4),
                'dropout': ebm_cfg.get('dropout', 0.0),
                'use_cls_token': ebm_cfg.get('use_cls_token', False),
                'num_views': ebm_cfg.get('num_views', None),
            })()
        })()).to(cfg.device)
        
        # Load checkpoint
        checkpoint = torch.load(cfg.model.ebm_ckpt_path, map_location=cfg.device)
        if "model" in checkpoint:
            ebm_model.load_state_dict(checkpoint["model"])
        else:
            ebm_model.load_state_dict(checkpoint)
        
        ebm_model.eval()
        print("✓ EBM model loaded successfully")
        
        # Test with sample data
        batch_size = 5
        obs_dim = cfg.obs_dim
        action_dim = cfg.action_dim
        horizon_steps = cfg.horizon_steps
        
        # Create sample observations and actions
        states = torch.randn(batch_size, obs_dim).to(cfg.device)
        actions = torch.randn(batch_size, horizon_steps, action_dim).to(cfg.device)
        k_idx = torch.zeros(batch_size, device=cfg.device, dtype=torch.long)
        t_idx = torch.zeros(batch_size, device=cfg.device, dtype=torch.long)
        
        print(f"Sample states shape: {states.shape}")
        print(f"Sample actions shape: {actions.shape}")
        
        with torch.no_grad():
            # Compute EBM energies
            energies, _ = ebm_model(
                k_idx=k_idx,
                t_idx=t_idx,
                views=None,
                poses=states,
                actions=actions
            )
            
            print(f"Raw EBM energies: {energies.cpu().numpy()}")
            print(f"Energy stats: mean={energies.mean():.4f}, std={energies.std():.4f}, min={energies.min():.4f}, max={energies.max():.4f}")
            
            # Apply reward scaling
            scaled_rewards = energies * cfg.model.ebm_reward_scale
            print(f"Scaled rewards (scale={cfg.model.ebm_reward_scale}): {scaled_rewards.cpu().numpy()}")
            
            # Apply clipping
            clipped_rewards = torch.clamp(scaled_rewards, -cfg.model.ebm_reward_clip_max, cfg.model.ebm_reward_clip_max)
            print(f"Clipped rewards (max={cfg.model.ebm_reward_clip_max}): {clipped_rewards.cpu().numpy()}")
            
            print("\n=== Analysis ===")
            print(f"Hopper environment rewards are typically ~1000-3000")
            print(f"Current EBM reward range: [{clipped_rewards.min():.2f}, {clipped_rewards.max():.2f}]")
            print(f"Scale factor needed: ~{1500 / clipped_rewards.abs().mean():.1f}x")
            
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_ebm_reward()